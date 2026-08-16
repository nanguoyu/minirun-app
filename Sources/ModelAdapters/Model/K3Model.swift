import Foundation
import MLX
import MLXBridge

/// The Kimi K3 text model, at any scale its configuration describes.
///
/// This is M6's decode loop with one substitution: weights arrive through a
/// ``K3WeightSource`` instead of a resident checkpoint. Nothing else moved.
/// The layer schedule below is the part a port is most likely to get wrong, and
/// it is transcribed from `KimiDecoderLayer._forward_attn_residual`
/// (`modeling_kimi_k3_linear.py:1085-1159`):
///
/// ```text
/// 1. prefixSum = hidden
/// 2. if blockResidual is non-empty: hidden = attnRes(prefixSum, blockResidual, selfAttentionRes*)
/// 3. if layerIndex % blockSize == 0: append prefixSum to blockResidual; prefixSum = nil
/// 4. hidden = inputLayerNorm(hidden); hidden = selfAttention(hidden)
/// 5. prefixSum = (prefixSum + hidden) ?? hidden
/// 6. hidden = attnRes(prefixSum, blockResidual, mlpRes*)
/// 7. hidden = postAttentionLayerNorm(hidden); hidden = moeOrMlp(hidden)
/// 8. prefixSum = prefixSum + hidden; the layer returns prefixSum
/// ```
///
/// ## What "config-driven" buys at flagship scale
///
/// Everything that differs between the 8-layer tiny model and the 93-layer
/// flagship is read from `config.json`, not branched on:
///
/// | | tiny | flagship |
/// | --- | --- | --- |
/// | layers | 8 | 93 |
/// | KDA / MLA split | `kda_layers` / `full_attn_layers` | same fields, 69 / 24 |
/// | AttnRes block | 4 | 12 → checkpoints at layers 0, 12, …, 84 |
/// | dense prefix | `first_k_dense_replace` | same field, layer 0 alone |
/// | experts | 8 top-2 | 896 top-16, 2 shared |
///
/// The layer lists in `linear_attn_config` are **1-indexed** in the file;
/// ``TinyK3Config/isLinearAttention(layer:)`` is the only place that knows it.
public final class K3Model {

    public let config: TinyK3Config
    public let source: K3WeightSource
    public let expertBackend: RoutedExpertBackend

    /// Rows of `lm_head` materialised at a time. At flagship shape the whole
    /// projection is 163,840 × 7,168 — 2.35 GB in BF16 and 4.7 GB widened —
    /// so walking it in chunks is what keeps the process's peak a statement
    /// about the pager rather than about `lm_head` (spec 16.5).
    ///
    /// Row `r` of the output is a dot product over the full hidden width
    /// regardless of which chunk it lands in, so no partial sum crosses a chunk
    /// boundary and the result does not depend on the chunking.
    public var logitChunkRows = 16384

    /// Whether ``TinyK3Capture`` retains the MLX tensors used by the reference
    /// trace harness.
    ///
    /// Reference validation needs every layer output, AttnRes probability and
    /// router tensor. Product generation needs only the token, logits and the
    /// small CPU routing-margin record. Keeping the reference tensors during a
    /// multi-token product run can retain an earlier pass's evaluated graph
    /// while the next pass allocates its working layer. The default remains on
    /// so existing oracle callers keep their full capture; a product runner
    /// must opt out explicitly.
    public var captureDiagnosticTensors = true

    /// Test-only numerical gate for the pre-optimisation MLA expression.
    /// App targets cannot reach this internal seam and therefore cannot use it
    /// as an unbounded product fallback.
    var useExpandedMLAReferenceForTesting = false

    /// Per-layer wall time of the most recent `forward`, for the trace M9 must
    /// capture. Index is the layer number.
    public private(set) var lastLayerSeconds: [Double] = []

    /// Allocator-reclamation wall time at each evaluated layer boundary.
    ///
    /// This is kept separate from ``lastLayerSeconds``: returning unused pages
    /// is a memory/speed trade, and folding it into compute time would make the
    /// cost impossible to price in the full-artifact benchmark record.
    public private(set) var lastLayerReclaimSeconds: [Double] = []

    /// Wall time of the mid-layer evaluation barrier, per layer.
    ///
    /// Releasing the attention weights before the MLP half is widened requires
    /// evaluating everything computed from them first — an unevaluated graph
    /// node still holds its inputs, so without the barrier the release would
    /// free nothing. That barrier is a real GPU synchronisation added to every
    /// layer, so it is timed and reported next to the reclaim it resembles
    /// rather than being folded into ``lastLayerSeconds`` invisibly. It is
    /// *included* in the layer's own elapsed time; this array says how much of
    /// that time it was.
    public private(set) var lastLayerAttentionBarrierSeconds: [Double] = []

    /// Injectable only inside `ModelAdapters` tests. Product callers always use
    /// the allocator-backed default below.
    var layerBoundaryMemoryReclaimer: () -> Void = K3TransientMemory.reclaim

    /// Where per-layer activations go, for M9's criterion (b) oracle. `nil` —
    /// the default — is the uncaptured path, and the two must produce the same
    /// logits.
    ///
    /// The operators already build a full stage-by-stage `Trace` on every call
    /// and this loop already discards it; a sink keeps the selected layers'
    /// instead. Nothing is added to the graph. Values are held until after this
    /// layer's existing `hidden.eval()` and only then handed over, so every
    /// captured node is an already-materialised ancestor of `hidden` and the
    /// capture cannot change when anything is evaluated. See
    /// ``K3ActivationSink``.
    public var activationSink: K3ActivationSink?

    /// Called after every layer with `(index, seconds)`, before the next one
    /// starts. The completed layer has already been released, its autorelease
    /// pool has drained, and transient allocator pages have been reclaimed.
    ///
    /// A flagship token is 93 layers and minutes of wall time on a phone. With
    /// no per-layer signal the harness shows nothing between "started" and
    /// "finished", and an operator cannot tell a slow run from a hung one —
    /// which on a device that may be thermally throttling or about to be killed
    /// is the difference between a measurement and a mystery.
    ///
    /// It is allowed to `throw`, and that is the second reason it exists: a
    /// forward pass is otherwise a straight line with no place for a Stop button
    /// to interrupt it. Throwing here cannot leak a resident layer because the
    /// release and reclamation boundary precedes the callback.
    public var onLayerCompleted: ((Int, Double) throws -> Void)?

    public init(config: TinyK3Config, source: K3WeightSource, expertBackend: RoutedExpertBackend) {
        self.config = config
        self.source = source
        self.expertBackend = expertBackend
    }

    /// One forward pass over `tokenIds`, extending `state`.
    public func forward(tokenIds: [Int], state: inout TinyK3State) throws -> TinyK3Capture {
        let tokens = tokenIds.count
        // Row-addressable embedding: a handful of rows out of 163,840, read
        // from the source. Spec 16.4, and the reason the table never lands.
        let embedding = try source.embeddingRows(tokenIds)

        var hidden = embedding
        var blockResidual: MLXArray?
        var layerOutputs: [MLXArray] = []
        var attnResProbabilities: [MLXArray] = []
        var routerLogits: [MLXArray] = []
        var routerScores: [MLXArray] = []
        var routerIds: [[[Int]]] = []
        var routerWeights: [MLXArray] = []
        var routerMargins: [[Double]] = []
        var worstMargin = Double.infinity
        var layerSeconds: [Double] = []
        layerSeconds.reserveCapacity(config.numHiddenLayers)
        var layerReclaimSeconds: [Double] = []
        layerReclaimSeconds.reserveCapacity(config.numHiddenLayers)
        var layerBarrierSeconds: [Double] = []
        layerBarrierSeconds.reserveCapacity(config.numHiddenLayers)
        defer {
            lastLayerSeconds = layerSeconds
            lastLayerReclaimSeconds = layerReclaimSeconds
            lastLayerAttentionBarrierSeconds = layerBarrierSeconds
        }

        for index in 0..<config.numHiddenLayers {
            let elapsed: Double
            let barrierSeconds: Double
            do {
                (elapsed, barrierSeconds) = try autoreleasepool {
                    let started = Date()
                    // The attention half only. The MLP half of this layer is
                    // not widened until attention has finished with — and the
                    // source has released — the half below.
                    var attentionWeights: TinyK3AttentionWeights? =
                        try source.layerAttentionWeights(index)
                    defer { source.releaseLayer(index) }
                    var prefixSum: MLXArray? = hidden
                    // The rolling cache is a sibling output of the attention graph,
                    // not necessarily an ancestor that evaluating `hidden` alone can
                    // detach. Keep the current layer's cache outputs beside `hidden`
                    // and evaluate the whole boundary together before its weights are
                    // released. Otherwise the cache stored for the next token can
                    // retain the just-finished layer graph until that next token.
                    var rollingStateArrays: [MLXArray] = []

                    // Held, not written: the flush happens below, after `hidden.eval()`
                    // has already materialised every one of these as an ancestor.
                    let capturing = activationSink?.selects(layer: index) ?? false
                    var held: [(String, MLXArray)] = []
                    var heldIds: [(String, [[Int]])] = []
                    func hold(_ name: String, _ value: MLXArray?) {
                        if capturing, let value { held.append((name, value)) }
                    }
                    hold("input_hidden", hidden)
                    hold("block_residual_in", blockResidual)

                    if let residual = blockResidual, residual.shape[1] > 0 {
                        let applied = K3AttnRes.apply(
                            prefixSum: prefixSum!, blockResidual: residual,
                            scoreWeight: attentionWeights!.selfAttentionResScore,
                            eps: config.rmsNormEps)
                        hidden = applied.output
                        if captureDiagnosticTensors {
                            attnResProbabilities.append(applied.probabilities)
                        }
                        hold("attn_res_pre.output", applied.output)
                        hold("attn_res_pre.probs", applied.probabilities)
                    }

                    if index % config.attnResBlockSize == 0 {
                        let checkpoint = expandedDimensions(prefixSum!, axis: 1)
                        blockResidual =
                            blockResidual.map { concatenated([$0, checkpoint], axis: 1) } ?? checkpoint
                        prefixSum = nil
                    }
                    hold("block_residual_out", blockResidual)

                    let normed = K3Norm.rms(
                        hidden, weight: attentionWeights!.inputLayerNorm, eps: config.rmsNormEps)
                    hold("input_layernorm", normed)
                    let attention: MLXArray
                    if let kda = attentionWeights!.kda {
                        let result = K3KDA.forward(
                            normed, weights: kda, heads: config.kdaNumHeads, headDim: config.kdaHeadDim,
                            eps: config.rmsNormEps, gateLowerBound: config.kdaGateLowerBound,
                            state: state.kda[index], recordStates: capturing)
                        state.kda[index] = result.state
                        rollingStateArrays = [
                            result.state.convQ, result.state.convK, result.state.convV,
                            result.state.recurrent,
                        ]
                        attention = result.output
                        if capturing {
                            for (name, value) in K3Model.kdaCheckpoints(result.trace) { hold(name, value) }
                        }
                    } else if let mla = attentionWeights!.mla {
                        if capturing || useExpandedMLAReferenceForTesting {
                            let result = useExpandedMLAReferenceForTesting
                                ? K3MLA.forwardExpandedReferenceForTesting(
                                    normed, weights: mla, config: config,
                                    state: state.mla[index], recordTrace: capturing)
                                : {
                                    let value = K3MLA.forward(
                                        normed, weights: mla, config: config,
                                        state: state.mla[index])
                                    return (value.output, value.state, Optional(value.trace))
                                }()
                            state.mla[index] = result.state
                            rollingStateArrays = [
                                result.state.normalizedLatents, result.state.sharedKeys,
                            ]
                            attention = result.output
                            if capturing, let trace = result.trace {
                                for (name, value) in K3Model.mlaCheckpoints(trace) { hold(name, value) }
                            }
                        } else {
                            let result = K3MLA.forwardWithoutTrace(
                                normed, weights: mla, config: config,
                                state: state.mla[index])
                            state.mla[index] = result.state
                            rollingStateArrays = [
                                result.state.normalizedLatents, result.state.sharedKeys,
                            ]
                            attention = result.output
                        }
                    } else {
                        throw TinyK3Error.configuration("layer \(index) has neither KDA nor MLA weights")
                    }
                    hold("attention_output", attention)
                    prefixSum = prefixSum.map { $0 + attention } ?? attention
                    hold("prefix_sum_after_attention", prefixSum)

                    // The sub-layer boundary. Attention is finished with its
                    // weights, but MLX is lazy: every node between the input
                    // norm and `prefixSum` still holds the arrays it was built
                    // from, so dropping the references below would free nothing
                    // until something forced the graph. Evaluate first, then
                    // drop — in that order, or the release is a no-op.
                    //
                    // The rolling cache and the captured trace values are
                    // evaluated with it for the same reason the end-of-layer
                    // barrier evaluates the cache: they are siblings of the
                    // attention output rather than ancestors, and a sibling left
                    // lazy is a sibling still holding this layer's projections.
                    //
                    // This is a real synchronisation per layer. It is timed, and
                    // reported in `lastLayerAttentionBarrierSeconds`.
                    let barrierStarted = Date()
                    MLX.eval([prefixSum!] + rollingStateArrays + held.map(\.1))
                    let barrier = Date().timeIntervalSince(barrierStarted)
                    attentionWeights = nil
                    source.releaseLayerAttention(index)

                    // Only now is the MLP half read and widened. This is the
                    // whole point: the widest layer's two halves are never
                    // resident together.
                    let layer = try source.layerFeedForwardWeights(index)
                    let afterAttention = K3AttnRes.apply(
                        prefixSum: prefixSum!, blockResidual: blockResidual,
                        scoreWeight: layer.mlpResScore, eps: config.rmsNormEps)
                    hidden = afterAttention.output
                    if captureDiagnosticTensors {
                        attnResProbabilities.append(afterAttention.probabilities)
                    }
                    hold("attn_res_mlp.output", afterAttention.output)
                    hold("attn_res_mlp.probs", afterAttention.probabilities)

                    let feedInput = K3Norm.rms(
                        hidden, weight: layer.postAttentionLayerNorm, eps: config.rmsNormEps)
                    hold("post_attention_layernorm", feedInput)
                    let feedOutput: MLXArray
                    if let dense = layer.dense {
                        feedOutput = K3MLP.dense(feedInput, weights: dense, config: config)
                    } else if let moe = layer.moe {
                        let result = try K3MoE.forward(
                            feedInput, weights: moe, backend: expertBackend, layer: index, config: config)
                        feedOutput = result.output
                        if captureDiagnosticTensors {
                            routerLogits.append(result.trace.selection.logits)
                            routerScores.append(result.trace.selection.scoresForChoice)
                            routerIds.append(result.trace.selection.ids)
                            routerWeights.append(result.trace.selection.weights)
                        }
                        routerMargins.append(result.trace.selection.relativeMargins)
                        worstMargin = min(
                            worstMargin, result.trace.selection.relativeMargins.min() ?? .infinity)
                        if capturing {
                            for (name, value) in K3Model.moeCheckpoints(result.trace) { hold(name, value) }
                            heldIds.append(("moe.selected_expert_ids", result.trace.selection.ids))
                        }
                    } else {
                        throw TinyK3Error.configuration("layer \(index) has neither an MLP nor a MoE block")
                    }
                    hold("feed_output", feedOutput)

                    prefixSum = prefixSum! + feedOutput
                    hidden = prefixSum!
                    // Evaluating per layer keeps the graph — and therefore any paged
                    // tile a graph node still holds — from spanning the whole model.
                    // At 93 layers this is what makes `releaseLayer` safe to honour.
                    MLX.eval([hidden] + rollingStateArrays)
                    if captureDiagnosticTensors {
                        layerOutputs.append(hidden)
                    }
                    if capturing, let sink = activationSink {
                        // Everything in `held` is an ancestor of `hidden`, which the
                        // line above has just evaluated, so nothing here computes.
                        sink.record(layer: index, "layer_output", hidden)
                        for (name, value) in held { sink.record(layer: index, name, value) }
                        for (name, ids) in heldIds { sink.record(layer: index, name, ids: ids) }
                    }
                    return (Date().timeIntervalSince(started), barrier)
                }
            } catch {
                // The autorelease pool and `releaseLayer` defer have already
                // unwound. Reclaim their free pages on a failed layer too;
                // otherwise Stop or an I/O error would preserve the very
                // high-water allocation the success path now releases.
                layerBoundaryMemoryReclaimer()
                throw error
            }

            // `autoreleasepool` has returned, so the local `layer`, its raw
            // widening objects and command-side autoreleased objects are gone;
            // its `defer` has also told the source to drop the resident layer.
            // Only now is cache/allocator reclamation safe. Placing this in
            // either caller's progress callback would run too early, while the
            // layer was still live, and would leave the other caller unfixed.
            let reclaimStarted = Date()
            layerBoundaryMemoryReclaimer()
            layerReclaimSeconds.append(Date().timeIntervalSince(reclaimStarted))
            layerSeconds.append(elapsed)
            layerBarrierSeconds.append(barrierSeconds)
            // An observer reading the process footprint here sees the true
            // post-release boundary rather than pages that are merely eligible
            // to be returned later.
            try onLayerCompleted?(index, elapsed)
        }

        let outputStage = K3AttnRes.apply(
            prefixSum: hidden, blockResidual: blockResidual,
            scoreWeight: source.outputAttnResScore, eps: config.rmsNormEps)
        if captureDiagnosticTensors {
            attnResProbabilities.append(outputStage.probabilities)
        }
        let finalNorm = K3Norm.rms(
            outputStage.output, weight: source.finalNormWeight, eps: config.rmsNormEps)
        finalNorm.eval()

        let logits = try self.logits(for: finalNorm[(tokens - 1)..., 0...])
        var greedy = 0
        var best = -Float.infinity
        for (index, value) in logits.enumerated() where value > best {
            best = value
            greedy = index
        }

        return TinyK3Capture(
            embedding: embedding, layerOutputs: layerOutputs,
            outputAttnRes: outputStage.output, finalNorm: finalNorm,
            attnResProbabilities: attnResProbabilities,
            routerLogits: routerLogits, routerScoresForChoice: routerScores,
            routerTopKIds: routerIds, routerTopKWeights: routerWeights,
            minimumRelativeRoutingMargin: worstMargin.isFinite ? worstMargin : 0,
            routerRelativeMargins: routerMargins,
            logits: logits, greedyTokenId: greedy)
    }

    // MARK: - Capture checkpoints
    //
    // The operators' `Trace` structs already name every stage; these lists say
    // which stages a per-layer oracle can actually check and what the reference
    // calls them. `recurrentStatePerPosition` is deliberately absent: it is
    // `tokens × [96, 128, 128]` at flagship shape, 18.9 MB a layer, and the
    // final state below is the same information at the only point a spot check
    // can compare it.

    static func kdaCheckpoints(_ trace: K3KDA.Trace) -> [(String, MLXArray)] {
        [
            ("kda.q_proj", trace.qProj), ("kda.k_proj", trace.kProj),
            ("kda.v_proj", trace.vProj),
            ("kda.q_conv", trace.qConv), ("kda.k_conv", trace.kConv),
            ("kda.v_conv", trace.vConv),
            ("kda.gate_raw", trace.gateRaw), ("kda.beta_raw", trace.betaRaw),
            ("kda.q_l2norm", trace.qL2Norm), ("kda.k_l2norm", trace.kL2Norm),
            ("kda.beta_sigmoid", trace.betaSigmoid),
            ("kda.gate_log_decay", trace.gateLogDecay),
            ("kda.recurrent_state_final", trace.recurrentStateFinalVFirst),
            ("kda.recurrence_output", trace.recurrenceOutput),
            ("kda.output_gate_raw", trace.outputGateRaw), ("kda.o_norm", trace.oNorm),
            ("kda.output", trace.output),
        ]
    }

    static func mlaCheckpoints(_ trace: K3MLA.Trace) -> [(String, MLXArray)] {
        [
            ("mla.q_lora", trace.qLora), ("mla.q_lora_norm", trace.qLoraNorm),
            ("mla.q_states", trace.qStates),
            ("mla.kv_lora_with_rope", trace.kvLoraWithRope),
            ("mla.kv_lora_norm", trace.kvLoraNorm), ("mla.kv_states", trace.kvStates),
            ("mla.query_states", trace.queryStates), ("mla.key_states", trace.keyStates),
            ("mla.value_states", trace.valueStates),
            ("mla.attention_scores", trace.attentionScores),
            ("mla.attention_probs", trace.attentionProbs),
            ("mla.attention_output", trace.attentionOutput),
            ("mla.output_gate_sigmoid", trace.outputGateSigmoid),
            ("mla.output", trace.output),
        ]
    }

    static func moeCheckpoints(_ trace: K3MoE.Trace) -> [(String, MLXArray)] {
        [
            ("moe.router_logits", trace.selection.logits),
            ("moe.scores_for_choice", trace.selection.scoresForChoice),
            ("moe.routing_weights", trace.selection.weights),
            ("moe.routed_down_latent", trace.routedDownLatent),
            ("moe.expert_gate", trace.expertGate), ("moe.expert_up", trace.expertUp),
            ("moe.expert_activation", trace.expertActivation),
            ("moe.expert_outputs", trace.expertOutputs),
            ("moe.routed_weighted_sum", trace.routedWeightedSum),
            ("moe.routed_norm", trace.routedNorm), ("moe.routed_up", trace.routedUp),
            ("moe.shared_expert_output", trace.sharedExpertOutput),
            ("moe.output", trace.output),
        ]
    }

    /// The vocabulary projection, walked in row chunks.
    public func logits(for hidden: MLXArray) throws -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(config.vocabSize)
        var first = 0
        while first < config.vocabSize {
            let count = min(logitChunkRows, config.vocabSize - first)
            let rows = try source.lmHeadRows(from: first, count: count)
            let part = matmul(hidden, rows.transposed(1, 0)).asType(.float32)
            part.eval()
            values += part.asArray(Float.self)
            first += count
        }
        return values
    }

    /// Prefill followed by greedy decode steps, threading the state through.
    public func generate(
        tokenIds: [Int], decodeSteps: Int
    ) throws -> (captures: [TinyK3Capture], generated: [Int]) {
        var state = TinyK3State()
        var captures: [TinyK3Capture] = []
        var generated: [Int] = []

        let prefill = try forward(tokenIds: tokenIds, state: &state)
        captures.append(prefill)
        generated.append(prefill.greedyTokenId)

        for _ in 0..<decodeSteps {
            let step = try forward(tokenIds: [generated.last!], state: &state)
            captures.append(step)
            generated.append(step.greedyTokenId)
        }
        return (captures, generated)
    }
}
