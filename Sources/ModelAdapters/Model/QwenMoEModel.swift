import Foundation
import MLX
import MLXBridge
import StorageCore

/// Supplies routed-expert weights to the Qwen MoE block.
///
/// The interface is a *gather*, not a weight handle, for the reason M6A settled
/// on for K3: the two backends this milestone compares — a pager streaming
/// container tiles, and the same containers held resident — differ in where the
/// bytes come from and in nothing else. Making the gather the boundary is what
/// lets "streamed equals resident, bit for bit" be a statement about the byte
/// source rather than about two similar implementations.
public protocol QwenExpertBackend: AnyObject {
    /// `out[t, j, :] = W[expertIds[t][j]] · x[t, :]`
    ///
    /// - Parameters:
    ///   - x: `[tokens, inFeatures]`.
    ///   - expertIds: `[tokens][expertsPerToken]`, global expert indices.
    /// - Returns: `[tokens, expertsPerToken, outFeatures]`.
    func gather(
        _ x: MLXArray, layer: Int, projection: QwenExpertProjection, expertIds: [[Int]]
    ) throws -> MLXArray

    /// Called once a layer's experts are finished with, so a streaming backend
    /// can release its slots.
    func releaseLayer(_ index: Int)

    /// Called as soon as the router has chosen, before the first ``gather`` of
    /// the layer demands anything.
    ///
    /// A resident backend has nothing to do here. A streaming one does: this is
    /// the only moment at which all three projections' reads can be put in
    /// flight together, and the read-ahead window of a single reader is too
    /// small to reach the device's saturation point on its own at this expert
    /// size. Measured in M8.5 phase B; the numbers are in the experiment record.
    func prefetch(layer: Int, expertIds: [[Int]]) throws
}

extension QwenExpertBackend {
    public func releaseLayer(_ index: Int) {}
    public func prefetch(layer: Int, expertIds: [[Int]]) throws {}
}

// MARK: - Router

/// The Qwen3-MoE router.
///
/// Transcribed from `mlx_lm/models/qwen3_moe.py`; the three traps are stated in
/// ``QwenMoEConfig``. The one worth repeating where the code is:
///
/// **softmax first, top-k second.** The selected weights are a slice of the
/// distribution over all 128 experts, renormalised. Taking a softmax of the
/// selected 8 logits instead would produce weights that sum to 1 and look
/// entirely reasonable, and would be wrong everywhere.
public enum QwenRouter {

    public struct Selection {
        /// `[tokens][topK]`. Descending by gate value; the reference promises no
        /// order, so a comparison against it must be a set comparison.
        public var ids: [[Int]]
        /// `[tokens, topK]`, aligned with ``ids``, after renormalisation.
        public var weights: MLXArray
        public var logits: MLXArray
        /// Post-softmax, pre-selection. The space the top-k is decided in, and
        /// therefore the space the margin must be measured in.
        public var gates: MLXArray
        /// `(kth selected − first rejected) / max(|kth|, |first rejected|)` per
        /// token. Reported so a port can say whether a disagreement sits near a
        /// tie rather than merely that it disagreed (spec §17.1, v0.4.2).
        public var relativeMargins: [Double]
    }

    public static func select(
        _ x: MLXArray, router: AffineWeights, config: QwenMoEConfig
    ) -> Selection {
        select(logits: affineMM(x, router), config: config)
    }

    /// The same selection from logits produced elsewhere, so a streaming router
    /// that multiplies in row bands states the semantics in one place.
    public static func select(logits: MLXArray, config: QwenMoEConfig) -> Selection {
        let experts = config.numExperts
        let topK = config.numExpertsPerToken

        // `precise: true` in the reference means "accumulate in float32 even for
        // a lower-precision input". The port is already in float32, so the
        // faithful reading is a plain float32 softmax.
        let gates = softmax(logits.asType(.float32), axis: -1)
        gates.eval()

        let tokens = logits.shape[0]
        // The selection is made on the host: the chosen ids decide which
        // container tiles the pager must fetch, so they have to be concrete
        // values before the expert gather can be expressed at all.
        let values = gates.asArray(Float.self)

        var ids: [[Int]] = []
        var gathered = [Float](repeating: 0, count: tokens * topK)
        var margins: [Double] = []
        ids.reserveCapacity(tokens)

        for token in 0..<tokens {
            let row = Array(values[(token * experts)..<((token + 1) * experts)])
            // Descending by gate value, ties broken by ascending index.
            // `mx.argpartition` makes no order promise among the selected k, so
            // this picks one deterministic order and states it; the fixture
            // comparison is a set comparison for exactly that reason.
            let order = (0..<experts).sorted { (row[$0], -$1) > (row[$1], -$0) }
            let selected = Array(order.prefix(topK))
            ids.append(selected)
            for (slot, expert) in selected.enumerated() {
                gathered[token * topK + slot] = row[expert]
            }
            let kth = Double(row[order[topK - 1]])
            let rejected = Double(row[order[topK]])
            let scale = max(abs(kth), abs(rejected))
            margins.append(scale > 0 ? (kth - rejected) / scale : 0)
        }

        var weights = MLXArray(gathered, [tokens, topK])
        if config.normTopKProb {
            // No epsilon. The reference divides by the bare sum, and adding a
            // guard here would be a silent numerical difference from it; the sum
            // of eight softmax probabilities cannot be zero.
            weights = weights / sum(weights, axis: -1, keepDims: true)
        }

        return Selection(
            ids: ids, weights: weights, logits: logits, gates: gates, relativeMargins: margins)
    }
}

// MARK: - Blocks

/// One Qwen3 attention block: GQA with per-head q/k RMSNorm, then RoPE.
///
/// The ordering is the part a port gets wrong. `q_norm` and `k_norm` are
/// `RMSNorm(head_dim)` applied to the projection **reshaped into heads** and
/// **before** RoPE — normalising the flat `[L, heads*head_dim]` projection
/// instead would mix heads together and still produce finite, plausible output.
public enum QwenAttention {

    public struct Trace {
        public var queriesAfterNorm: MLXArray
        public var keysAfterNorm: MLXArray
        public var queriesAfterRoPE: MLXArray
        public var output: MLXArray
    }

    /// A single-sequence key/value cache.
    ///
    /// One sequence, no batching, no ring buffer, no quantized cache — spec
    /// §15.6 excludes batching and long context from the first target, and a
    /// cache with none of those is small enough to be obviously correct.
    public final class Cache {
        public private(set) var keys: MLXArray?
        public private(set) var values: MLXArray?
        public var offset: Int { keys?.shape[2] ?? 0 }

        public init() {}

        /// Bytes the cache holds. Grows by one token's keys and values per
        /// decode step, which is required growth: it is what lets the next step
        /// attend to everything before it.
        ///
        /// A block-allocating variant of this class was written and measured on
        /// device (M8.5 phase 2). It worked — the allocation went flat between
        /// block boundaries and the generated tokens were identical — and it is
        /// not here, because the growth it removed was never the problem. With
        /// KV allocation pinned flat the process footprint still climbed at
        /// ~194 KiB/token, the same residual this version shows, while costing
        /// 4.7% throughput and ~48 MB of slack. The experiment is in the record;
        /// the simpler code is what ships.
        public var allocatedBytes: Int {
            (keys?.nbytes ?? 0) + (values?.nbytes ?? 0)
        }

        public func append(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
            if let keys, let values {
                let mergedKeys = concatenated([keys, newKeys], axis: 2)
                let mergedValues = concatenated([values, newValues], axis: 2)
                self.keys = mergedKeys
                self.values = mergedValues
                return (mergedKeys, mergedValues)
            }
            keys = newKeys
            values = newValues
            return (newKeys, newValues)
        }
    }

    public static func forward(
        _ x: MLXArray,
        weights: QwenLayerWeights,
        config: QwenMoEConfig,
        cache: Cache,
        mask: MLXArray?
    ) -> (output: MLXArray, trace: Trace) {
        let tokens = x.shape[0]
        let heads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let headDim = config.headDim

        // [tokens, count * headDim] -> [1, count, tokens, headDim]
        func splitHeads(_ projected: MLXArray, count: Int, norm: MLXArray?) -> MLXArray {
            var value = projected.reshaped([1, tokens, count, headDim])
            if let norm {
                // RMSNorm over the last axis, which is now head_dim: the reshape
                // has to happen first for the normalisation to be per head.
                value = QwenNorm.rms(value, weight: norm, eps: config.rmsNormEps)
            }
            return value.transposed(0, 2, 1, 3)
        }

        let queries = splitHeads(
            affineMM(x, weights.queryProj), count: heads, norm: weights.queryNorm)
        let keys = splitHeads(affineMM(x, weights.keyProj), count: kvHeads, norm: weights.keyNorm)
        let values = splitHeads(
            affineMM(x, weights.valueProj), count: kvHeads, norm: MLXArray?.none)

        let offset = cache.offset
        let rotatedQueries = MLXFast.RoPE(
            queries, dimensions: headDim, traditional: false, base: config.ropeTheta,
            scale: 1.0, offset: offset)
        let rotatedKeys = MLXFast.RoPE(
            keys, dimensions: headDim, traditional: false, base: config.ropeTheta,
            scale: 1.0, offset: offset)

        let (allKeys, allValues) = cache.append(keys: rotatedKeys, values: values)

        // Grouped-query expansion, written out rather than left to
        // `scaledDotProductAttention`'s internal handling: the expansion is a
        // broadcast of the kv head across its query group, and doing it here
        // keeps the arithmetic below an ordinary dense attention that can be
        // read and compared against the reference.
        let factor = heads / kvHeads
        func expand(_ value: MLXArray) -> MLXArray {
            let length = value.shape[2]
            return broadcast(
                value.reshaped([1, kvHeads, 1, length, headDim]),
                to: [1, kvHeads, factor, length, headDim]
            ).reshaped([1, heads, length, headDim])
        }

        var scores = matmul(
            rotatedQueries * MLXArray(config.attentionScale),
            expand(allKeys).transposed(0, 1, 3, 2))
        if let mask { scores = scores + mask }
        // float32 throughout, matching MLX's own note that the softmax is done
        // in float32 regardless of input precision.
        let probabilities = softmax(scores.asType(.float32), axis: -1)
        let attended = matmul(probabilities, expand(allValues))

        let merged = attended.transposed(0, 2, 1, 3).reshaped([tokens, heads * headDim])
        let output = affineMM(merged, weights.outputProj)

        return (
            output,
            Trace(
                queriesAfterNorm: queries, keysAfterNorm: keys,
                queriesAfterRoPE: rotatedQueries, output: output)
        )
    }

    /// The additive causal mask for a prefill of `tokens` positions, or `nil`
    /// for a single decode step (which can attend to everything cached).
    public static func causalMask(tokens: Int, offset: Int) -> MLXArray? {
        guard tokens > 1 else { return nil }
        let total = tokens + offset
        var values = [Float](repeating: 0, count: tokens * total)
        for row in 0..<tokens {
            // Position `row` sits at absolute index `offset + row` and may not
            // see anything after it.
            for column in (offset + row + 1)..<total {
                values[row * total + column] = -Float.greatestFiniteMagnitude
            }
        }
        return MLXArray(values, [1, 1, tokens, total])
    }
}

/// RMSNorm, Qwen's.
///
/// Arithmetically identical to ``K3Norm/rms(_:weight:eps:)`` — epsilon added to
/// the mean of squares, inside the reciprocal square root, weight applied after
/// — and deliberately not shared with it. K3's is a transcription of
/// `KimiRMSNorm.forward` and is pinned by K3 fixtures; this one is a
/// transcription of `mlx.nn.RMSNorm` and is pinned by Qwen fixtures. If either
/// upstream changes, one of them must be able to move without the other.
public enum QwenNorm {
    public static func rms(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
        let x32 = x.asType(.float32)
        let normed = x32 * rsqrt(mean(x32 * x32, axis: -1, keepDims: true) + eps)
        return weight * normed
    }
}

/// The sparse MoE block. No shared expert, no latent projection, no norm on the
/// weighted sum — the routed experts are the entire feed-forward path.
public enum QwenMoEBlock {

    public struct Trace {
        public var selection: QwenRouter.Selection
        public var gateHalf: MLXArray
        public var upHalf: MLXArray
        public var activation: MLXArray
        public var expertOutputs: MLXArray
        public var output: MLXArray
    }

    public static func forward(
        _ x: MLXArray,
        weights: QwenLayerWeights,
        backend: QwenExpertBackend,
        layer: Int,
        config: QwenMoEConfig
    ) throws -> (output: MLXArray, trace: Trace) {
        let tokens = x.shape[0]
        let topK = config.numExpertsPerToken

        let selection = QwenRouter.select(x, router: weights.router, config: config)

        // The routing decision is the earliest moment the layer's byte demand
        // is known. Telling the backend now rather than at the first gather is
        // what lets a streaming one have more than one read in flight; it
        // cannot change which bytes are gathered or the order they are consumed
        // in, so it cannot move a number the bit-identity gate reads.
        try backend.prefetch(layer: layer, expertIds: selection.ids)

        // Both SwiGLU halves consume the same hidden state, gathered at
        // [tokens, topK, moeIntermediate].
        let gateHalf = try backend.gather(
            x, layer: layer, projection: .gateProj, expertIds: selection.ids)
        let upHalf = try backend.gather(
            x, layer: layer, projection: .upProj, expertIds: selection.ids)
        // `swiglu(gate, x) = silu(gate) * x`, and `SwitchGLU` calls it as
        // `activation(x_up, x_gate)` -> `silu(gate_proj) * up_proj`. The order
        // matters: silu is applied to the *gate* projection.
        let activation = K3Activation.silu(gateHalf) * upHalf

        // Each (token, slot) row now has its own activation, so the down
        // projection is a gather over `tokens * topK` rows with one expert each.
        let flatIds = selection.ids.flatMap { $0 }.map { [$0] }
        let expertOutputs = try backend.gather(
            activation.reshaped([tokens * topK, config.moeIntermediateSize]),
            layer: layer, projection: .downProj, expertIds: flatIds
        ).reshaped([tokens, topK, config.hiddenSize])

        let weighted = expertOutputs * expandedDimensions(selection.weights, axis: -1)
        let output = sum(weighted, axis: 1)

        backend.releaseLayer(layer)

        return (
            output,
            Trace(
                selection: selection, gateHalf: gateHalf, upHalf: upHalf,
                activation: activation, expertOutputs: expertOutputs, output: output)
        )
    }
}

// MARK: - The model

/// Qwen3-30B-A3B, config-driven, weight-source agnostic.
///
/// The decode loop is written against ``QwenWeightSource`` and
/// ``QwenExpertBackend`` so that "resident" and "streamed" are two arguments to
/// one implementation rather than two implementations. That is what makes the
/// bit-identity gate meaningful: the schedule, the gather geometry, the merge
/// order and the evaluation cadence cannot drift between the arms because there
/// is only one of each.
public final class QwenMoEModel {

    public let source: QwenWeightSource
    public let backend: QwenExpertBackend
    public var config: QwenMoEConfig { source.config }

    /// Per-layer capture points, populated when ``capturing`` is set.
    public private(set) var captures: [String: MLXArray] = [:]
    public var capturing = false
    /// Prefix applied to capture names, so a decode step's captures do not
    /// overwrite the prefill's. The fixtures use `""` and `"decode_"`.
    public var capturePrefix = ""

    private var caches: [QwenAttention.Cache]

    public init(source: QwenWeightSource, backend: QwenExpertBackend) {
        self.source = source
        self.backend = backend
        self.caches = (0..<source.config.numHiddenLayers).map { _ in QwenAttention.Cache() }
    }

    public func resetCache() {
        caches = (0..<config.numHiddenLayers).map { _ in QwenAttention.Cache() }
    }

    /// Bytes currently held by the key/value caches, summed over every layer.
    ///
    /// Exposed because the device phase found the process footprint growing per
    /// token and could not say how much of that was legitimate. The KV cache is
    /// the one component whose growth is *required* — it is what lets a decode
    /// step attend to everything before it — so a memory-growth gate that does
    /// not subtract it is measuring the wrong thing.
    ///
    /// Read from the arrays rather than computed from the config, deliberately:
    /// a computed figure would restate an assumption about dtype and shape,
    /// while `nbytes` reports what was actually allocated. Cheap — it touches
    /// two array headers per layer and evaluates nothing.
    public var kvCacheBytes: Int {
        caches.reduce(0) { $0 + $1.allocatedBytes }
    }

    private func capture(_ name: String, _ value: MLXArray) {
        guard capturing else { return }
        value.eval()
        captures["\(capturePrefix)\(name)"] = value
    }

    /// Hidden states for these tokens, appended to whatever the cache holds.
    ///
    /// - Returns: `[tokens, hiddenSize]` after the final norm.
    public func hidden(tokenIds: [Int]) throws -> MLXArray {
        guard !tokenIds.isEmpty else {
            throw TinyK3Error.configuration("hidden() needs at least one token")
        }
        var h = try source.embeddingRows(tokenIds)
        h.eval()
        capture("embedding", h)

        let mask = QwenAttention.causalMask(tokens: tokenIds.count, offset: caches[0].offset)

        for index in 0..<config.numHiddenLayers {
            let weights = try source.layerWeights(index)

            let normed = QwenNorm.rms(h, weight: weights.inputNorm, eps: config.rmsNormEps)
            let (attention, attentionTrace) = QwenAttention.forward(
                normed, weights: weights, config: config, cache: caches[index], mask: mask)
            if capturing, index == 0 || index == config.numHiddenLayers / 2 {
                capture("layer\(pad(index))_q_after_norm", attentionTrace.queriesAfterNorm)
                capture("layer\(pad(index))_k_after_norm", attentionTrace.keysAfterNorm)
                capture("layer\(pad(index))_q_after_rope", attentionTrace.queriesAfterRoPE)
                capture("layer\(pad(index))_attn_output", attentionTrace.output)
            }
            h = h + attention

            let postNormed = QwenNorm.rms(
                h, weight: weights.postAttentionNorm, eps: config.rmsNormEps)
            guard config.isSparse(layer: index) else {
                throw TinyK3Error.configuration(
                    "layer \(index) is dense in this config; this adapter implements the sparse "
                        + "path only and will not silently route a dense layer")
            }
            let (moe, moeTrace) = try QwenMoEBlock.forward(
                postNormed, weights: weights, backend: backend, layer: index, config: config)
            if capturing {
                capture("layer\(pad(index))_router_logits", moeTrace.selection.logits)
                capture("layer\(pad(index))_router_gates", moeTrace.selection.gates)
                capture("layer\(pad(index))_router_topk_weights", moeTrace.selection.weights)
                routerIds["\(capturePrefix)layer\(pad(index))"] = moeTrace.selection.ids
                routerMargins["\(capturePrefix)layer\(pad(index))"] =
                    moeTrace.selection.relativeMargins
            }
            h = h + moe
            h.eval()
            capture("layer\(pad(index))_output", h)

            source.releaseLayer(index)
        }

        let normed = QwenNorm.rms(h, weight: source.finalNormWeight, eps: config.rmsNormEps)
        normed.eval()
        capture("final_norm", normed)
        return normed
    }

    /// Router ids per capture point, which are integers and so do not belong in
    /// the `MLXArray` capture map.
    public private(set) var routerIds: [String: [[Int]]] = [:]
    public private(set) var routerMargins: [String: [Double]] = [:]

    /// Logits for the last position only.
    ///
    /// Only the last row is projected. At 151,936 columns the LM head is the
    /// largest tensor in the model, and a prefill that projected every position
    /// would spend most of its time producing rows nothing reads (spec §16.5).
    public func logits(tokenIds: [Int]) throws -> MLXArray {
        let hidden = try self.hidden(tokenIds: tokenIds)
        let last = hidden[(hidden.shape[0] - 1)..., 0...]
        let head = try source.lmHead()
        let logits = affineMM(last, head).reshaped([config.vocabSize])
        logits.eval()
        capture("logits_last", logits)
        return logits
    }

    /// One generated token, and what producing it cost.
    ///
    /// Wall clock only. The I/O and stall halves belong to whichever backend is
    /// installed and are read from *its* counters around the same interval —
    /// putting them here would make the model know which backend it has, which
    /// is the one thing the ``QwenExpertBackend`` boundary exists to prevent.
    public struct DecodeStep: Sendable {
        /// 0 is the token the prefill produced; 1 and up are single-token steps.
        public var index: Int
        public var tokenId: Int
        /// Positions fed to this forward pass: the whole prompt at step 0, one
        /// thereafter. It is what separates the prefill sample from the decode
        /// samples without the reader having to know the prompt length.
        public var inputTokens: Int
        /// The forward pass plus the argmax that reads it. The argmax is inside
        /// the measurement on purpose: it copies 151,936 floats back from the
        /// device, and a decode step really does pay for that.
        public var seconds: Double

        public var isPrefill: Bool { index == 0 }
    }

    /// Greedy decode, with a wall-clock reading per generated token.
    ///
    /// `onStep` is called after each token with that token's reading, so a
    /// caller can sample its backend's byte and stall counters on the same
    /// boundary rather than reconstructing the split afterwards.
    ///
    /// Generation runs to `maxTokens` and does not stop at an end-of-sequence
    /// id. That is a measurement decision: every prompt then contributes the
    /// same number of samples, and a run that stopped early would report a
    /// median over a different population. A caller that wants the text rather
    /// than the timing truncates at the first such id, which it can do because
    /// the ids are returned in order.
    /// - Parameter onLogits: Observes each step's logits **after** the step's
    ///   time has been taken, so an observer cannot inflate the number it is
    ///   watching. Optional and off by default because reading 151,936 floats
    ///   back per step is not free; a run that uses it is a correctness run,
    ///   not a timing one.
    ///
    ///   It exists so that a caller wanting the top-1/top-2 separation does not
    ///   have to write a second decode loop to get it. Two loops would be free
    ///   to drift, which is the exact failure the single-gather-loop discipline
    ///   elsewhere in this adapter exists to prevent.
    @discardableResult
    public func generateTimed(
        prompt: [Int], maxTokens: Int,
        onLogits: ((MLXArray) throws -> Void)? = nil,
        onStep: (DecodeStep) throws -> Void = { _ in }
    ) throws -> [DecodeStep] {
        var steps: [DecodeStep] = []
        var inputs = prompt
        for step in 0..<max(1, maxTokens) {
            if step > 0 { capturePrefix = step == 1 ? "decode_" : "decode\(step)_" }
            let started = MonotonicClock.now()
            let stepLogits = try logits(tokenIds: inputs)
            let next = try argmaxToken(stepLogits)
            let reading = DecodeStep(
                index: step, tokenId: next, inputTokens: inputs.count,
                seconds: MonotonicClock.seconds(since: started))
            steps.append(reading)
            try onLogits?(stepLogits)
            try onStep(reading)
            inputs = [next]
        }
        capturePrefix = ""
        return steps
    }

    /// Greedy decode. Returns the generated token ids, in order.
    ///
    /// One line, because there is one decode loop: two of them would be free to
    /// drift in exactly the way the single-gather-loop discipline elsewhere in
    /// this adapter exists to prevent.
    public func generate(prompt: [Int], maxTokens: Int) throws -> [Int] {
        try generateTimed(prompt: prompt, maxTokens: maxTokens).map(\.tokenId)
    }

    private func argmaxToken(_ logits: MLXArray) throws -> Int {
        let values = logits.asArray(Float.self)
        var best = 0
        for index in 1..<values.count where values[index] > values[best] { best = index }
        return best
    }

    private func pad(_ index: Int) -> String {
        index < 10 ? "0\(index)" : "\(index)"
    }
}
