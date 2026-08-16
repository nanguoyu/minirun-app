import Foundation
import MLX

/// Multi-head latent attention, NoPE, with an output gate.
///
/// Transcribed from `KimiMLAAttention.forward`,
/// `modeling_kimi_k3_linear.py:453-537`. Four things about this module are
/// counter-intuitive enough that the fixture manifest records them as traps:
///
/// - **No rotation at all.** `mla_use_nope` is true and `rotary_emb` is `None`,
///   so the `qk_rope_head_dim` slice is carried through the concatenation and
///   never rotated. There is no positional encoding anywhere in this model
///   beyond the causal mask and KDA's recurrence.
/// - **`k_rot` is shared across heads.** It is produced once per position,
///   `[1, T, 32]`, and broadcast — MQA-style, not per-head.
/// - **The scale is over the full query.** `1/sqrt(qkNope + qkRope)`, not over
///   the 64-wide nope part.
/// - **`config.head_dim` is 74 and is used by nothing.**
public enum K3MLA {

    /// The growing latent cache.
    ///
    /// K3's key/value decompression is linear. Retaining the normalized
    /// `kv_lora_rank` row plus the one shared `k_rot` slice therefore preserves all
    /// information needed by decode without materialising one key and value row
    /// per head. Decode absorbs the decompression weights into the query and
    /// output sides; prompt prefill keeps the eager reference expression.
    public struct State {
        public var normalizedLatents: MLXArray  // [positions, kvLoraRank]
        public var sharedKeys: MLXArray  // [positions, qkRopeHeadDim]

        public init(normalizedLatents: MLXArray, sharedKeys: MLXArray) {
            self.normalizedLatents = normalizedLatents
            self.sharedKeys = sharedKeys
        }
    }

    public struct Weights {
        public var qAProj: MLXArray
        public var qALayerNorm: MLXArray
        public var qBProj: MLXArray
        public var kvAProjWithMQA: MLXArray
        public var kvALayerNorm: MLXArray
        public var kvBProj: MLXArray
        public var gProj: MLXArray
        public var oProj: MLXArray
    }

    /// Every intermediate `subop-mla-layer03` records.
    public struct Trace {
        public var qLora: MLXArray
        public var qLoraNorm: MLXArray
        public var qStates: MLXArray
        public var kvLoraWithRope: MLXArray
        public var kvLoraNorm: MLXArray
        public var kvStates: MLXArray
        public var queryStates: MLXArray
        public var keyStates: MLXArray
        public var valueStates: MLXArray
        public var attentionScores: MLXArray
        public var attentionProbs: MLXArray
        public var attentionOutput: MLXArray
        public var outputGateSigmoid: MLXArray
        public var output: MLXArray
    }

    private enum DecodeExpression {
        case absorbed
        case expandedReference
    }

    /// - Parameters:
    ///   - x: `[tokens, hidden]`, already through `input_layernorm`.
    ///   - state: the KV cache from previous positions, or `nil`.
    /// - Returns: the layer output `[tokens, hidden]`, the extended cache, and
    ///   the stage-by-stage trace.
    public static func forward(
        _ x: MLXArray, weights: Weights, config: TinyK3Config, state: State?
    ) -> (output: MLXArray, state: State, trace: Trace) {
        let result = forward(
            x, weights: weights, config: config, state: state,
            decodeExpression: .expandedReference, recordTrace: true)
        return (result.output, result.state, result.trace!)
    }

    /// Product execution when no activation oracle selected this layer.
    ///
    /// Decode must not construct the expanded historical K/V graph merely to
    /// throw it away. Keeping this entry point internal makes the no-trace
    /// product path explicit without weakening the public reference API.
    static func forwardWithoutTrace(
        _ x: MLXArray, weights: Weights, config: TinyK3Config, state: State?
    ) -> (output: MLXArray, state: State) {
        let result = forward(
            x, weights: weights, config: config, state: state,
            decodeExpression: .absorbed, recordTrace: false)
        return (result.output, result.state)
    }

    /// Independent old-expression seam for numerical and performance gates.
    /// It is internal so an App target cannot silently select the expanded
    /// cache as a product fallback.
    static func forwardExpandedReferenceForTesting(
        _ x: MLXArray, weights: Weights, config: TinyK3Config, state: State?,
        recordTrace: Bool = true
    ) -> (output: MLXArray, state: State, trace: Trace?) {
        forward(
            x, weights: weights, config: config, state: state,
            decodeExpression: .expandedReference, recordTrace: recordTrace)
    }

    /// Trace-bearing absorbed expression for the numerical test gate. Product
    /// execution uses `forwardWithoutTrace`; the public trace API deliberately
    /// preserves the eager expression expected by activation oracles.
    static func forwardAbsorbedForTesting(
        _ x: MLXArray, weights: Weights, config: TinyK3Config, state: State?
    ) -> (output: MLXArray, state: State, trace: Trace) {
        let result = forward(
            x, weights: weights, config: config, state: state,
            decodeExpression: .absorbed, recordTrace: true)
        return (result.output, result.state, result.trace!)
    }

    private static func forward(
        _ x: MLXArray, weights: Weights, config: TinyK3Config, state: State?,
        decodeExpression: DecodeExpression, recordTrace: Bool
    ) -> (output: MLXArray, state: State, trace: Trace?) {
        let tokens = x.shape[0]
        let queryHeadDim = config.mlaQueryHeadDim

        let qLora = k3Linear(x, weights.qAProj)
        let qLoraNorm = K3Norm.rms(qLora, weight: weights.qALayerNorm, eps: config.rmsNormEps)
        let qStates = k3Linear(qLoraNorm, weights.qBProj)
        let heads = qStates.shape[1] / queryHeadDim

        // [tokens, heads, queryHeadDim] -> [heads, tokens, queryHeadDim]
        let query = qStates.reshaped([tokens, heads, queryHeadDim]).transposed(1, 0, 2)
        let queryParts = split(query, indices: [config.qkNopeHeadDim], axis: -1)
        let qPass = queryParts[0]
        let qRot = queryParts[1]

        let compressedKV = k3Linear(x, weights.kvAProjWithMQA)
        let kvParts = split(compressedKV, indices: [config.kvLoraRank], axis: -1)
        let kvLora = kvParts[0]
        let kRot = kvParts[1]

        let kvLoraNorm = K3Norm.rms(kvLora, weight: weights.kvALayerNorm, eps: config.rmsNormEps)
        let sharedCurrentKeys = kRot.reshaped([tokens, config.qkRopeHeadDim])
        let latents: MLXArray
        let sharedKeys: MLXArray
        if let state {
            latents = concatenated([state.normalizedLatents, kvLoraNorm], axis: 0)
            sharedKeys = concatenated([state.sharedKeys, sharedCurrentKeys], axis: 0)
        } else {
            latents = kvLoraNorm
            sharedKeys = sharedCurrentKeys
        }

        // Prefill needs the eager expression. Decode expands history only for
        // an activation oracle or the internal old-expression gate; normal
        // product decode never even constructs that graph.
        let needsExpanded = state == nil || recordTrace || decodeExpression == .expandedReference
        let expanded = needsExpanded
            ? expand(
                latents: latents, sharedKeys: sharedKeys,
                weights: weights, config: config, heads: heads)
            : nil
        let queryStates = concatenated([qPass, qRot], axis: -1)

        let scaling = MLXArray(Float(pow(Double(queryHeadDim), -0.5)))
        let attention: MLXArray
        let probs: MLXArray
        var scores: MLXArray
        if state != nil, decodeExpression == .absorbed {
            // Absorbed MLA decode:
            //   q_nope · (W_UK · c)^T == (q_nope · W_UK) · c^T
            //   (p · c) · W_UV^T == p · (W_UV · c)^T
            // This keeps history in its low-rank form and avoids reprojecting
            // every cached position on every token.
            let decompression = weights.kvBProj.reshaped([
                heads, config.qkNopeHeadDim + config.vHeadDim, config.kvLoraRank,
            ])
            let projections = split(
                decompression, indices: [config.qkNopeHeadDim], axis: 1)
            let keyProjection = projections[0]
            let valueProjection = projections[1]

            let latentQueries = matmul(qPass, keyProjection)
            let latentScores = matmul(latentQueries, latents.transposed(1, 0))
            let sharedScores = matmul(qRot, sharedKeys.transposed(1, 0))
            scores = (latentScores + sharedScores) * scaling
            if let mask = causalMask(queries: tokens, keys: latents.shape[0]) {
                scores = scores + mask
            }
            probs = softmax(scores.asType(.float32), axis: -1)
            let latentOutput = matmul(probs, latents)
            attention = matmul(latentOutput, valueProjection.transposed(0, 2, 1))
        } else {
            let expanded = expanded!
            scores = matmul(queryStates, expanded.keys.transposed(0, 2, 1)) * scaling
            if let mask = causalMask(queries: tokens, keys: expanded.keys.shape[1]) {
                scores = scores + mask
            }
            probs = softmax(scores.asType(.float32), axis: -1)
            attention = matmul(probs, expanded.values)
        }
        let merged = attention.transposed(1, 0, 2).reshaped([tokens, heads * config.vHeadDim])

        let gate = sigmoid(k3Linear(x, weights.gProj))
        let output = k3Linear(
            config.mlaUseOutputGate ? merged * gate : merged, weights.oProj)

        let trace = recordTrace
            ? Trace(
                qLora: qLora, qLoraNorm: qLoraNorm, qStates: qStates,
                kvLoraWithRope: compressedKV, kvLoraNorm: kvLoraNorm,
                kvStates: k3Linear(kvLoraNorm, weights.kvBProj),
                queryStates: queryStates, keyStates: expanded!.keys,
                valueStates: expanded!.values,
                attentionScores: scores,
                attentionProbs: probs,
                attentionOutput: merged,
                outputGateSigmoid: gate, output: output)
            : nil
        return (
            output,
            State(normalizedLatents: latents, sharedKeys: sharedKeys),
            trace)
    }

    private static func expand(
        latents: MLXArray, sharedKeys: MLXArray,
        weights: Weights, config: TinyK3Config, heads: Int
    ) -> (keys: MLXArray, values: MLXArray) {
        let positions = latents.shape[0]
        let decompressed = k3Linear(latents, weights.kvBProj)
            .reshaped([positions, heads, config.qkNopeHeadDim + config.vHeadDim])
            .transposed(1, 0, 2)
        let splitKV = split(
            decompressed, indices: [config.qkNopeHeadDim], axis: -1)
        let shared = broadcast(
            sharedKeys.reshaped([1, positions, config.qkRopeHeadDim]),
            to: [heads, positions, config.qkRopeHeadDim])
        return (concatenated([splitKV[0], shared], axis: -1), splitKV[1])
    }

    /// The masked-out sentinel: `torch.finfo(float32).min`, which is what
    /// `create_causal_mask` fills with — **not** `-infinity`.
    ///
    /// The softmax cannot tell the two apart (both underflow to a zero
    /// probability), so this looks like a free choice and is not: the
    /// reference's `attention_scores` capture records the summed value, and
    /// `-inf - (-inf)` is `NaN`, so a port using infinity cannot be compared
    /// against that capture at all. Found exactly that way.
    static let maskedScore = -Float.greatestFiniteMagnitude

    /// The additive causal mask: `0` where a query may attend, ``maskedScore``
    /// where it may not.
    ///
    /// Query `i` sits at absolute position `keys - queries + i`, so a decode
    /// step with one query and a full cache produces an all-zero mask — every
    /// key is in the past. Returns `nil` in that case rather than adding a
    /// matrix of zeros, which keeps the decode path's arithmetic identical to
    /// the reference's (`create_causal_mask` returns `None` there too).
    static func causalMask(queries: Int, keys: Int) -> MLXArray? {
        guard queries > 1 else { return nil }
        let offset = keys - queries
        var values = [Float](repeating: 0, count: queries * keys)
        for query in 0..<queries {
            for key in 0..<keys where key > query + offset {
                values[query * keys + key] = maskedScore
            }
        }
        return MLXArray(values, [1, queries, keys])
    }
}
