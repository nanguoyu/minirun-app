import MLX

/// DeepSeek V4.1's **single-pass** hyper-connections.
///
/// The arithmetic is V4's, unchanged: one float32 projection of the flattened
/// `hc_mult`-copy residual stream, RMS-scaled, split into three coefficient sets
/// with `hc_split_sinkhorn` — sigmoid + eps for `pre`, twice sigmoid for `post`,
/// a row softmax and twenty alternating normalizations for `comb`.
/// ``DeepSeekV4HyperConnections/splitSinkhorn(mixes:scale:base:multiplicity:iterations:epsilon:phaseAccounting:diagnostics:compiling:)``
/// *is* that kernel and is called here rather than copied.
///
/// What changed is which coefficients feed which sublayer.
///
/// ## The one-block shift
///
/// In V4 a sublayer contracted the stream with the `pre` it had just computed
/// from that same stream. In V4.1 it contracts with the `pre` the **previous**
/// sublayer produced, and hands its own on:
///
/// ```
/// residual = x
/// attn_pre, attn_post, attn_comb = hc_mixes(x, hc_attn_*)
/// x = attn(attn_norm(hc_pre(x, pre_mix)))      // the caller's pre_mix
/// x = hc_post(x, residual, attn_post, attn_comb)
/// residual = x
/// ffn_pre, ffn_post, ffn_comb = hc_mixes(x, hc_ffn_*)
/// x = ffn(ffn_norm(hc_pre(x, attn_pre)))       // this block's attention's
/// x = hc_post(x, residual, ffn_post, ffn_comb)
/// return x, ffn_pre                            // the next block's
/// ```
///
/// So a block's coefficients are computed from the stream *before* the sublayer
/// that will use them runs, and the mix threading is a value carried across the
/// whole stack: `make_identity_pre_mix` starts it at the first block and the
/// head ends it with `layer.hc_pre(h, pre_mix)`.
///
/// ## And `hc_head_*` is gone
///
/// V4 had a separate learned head contraction —
/// ``DeepSeekV4HyperConnections/collapseHead(residual:function:scale:base:multiplicity:epsilon:normEpsilon:phaseAccounting:diagnostics:)``,
/// with its own `hc_head_fn`/`base`/`scale`. V4.1 publishes no such tensors
/// (`docs/experiments/2026-09-10-v41-flash-container.md`: "`hc_head_*` is
/// gone"), and the reference's `Transformer.forward` ends with the last block's
/// own `hc_pre` under the mix that block returned. ``collapse(stream:preMix:)``
/// is that, and it is deliberately the same function as the sublayer
/// contraction rather than a second one that could disagree with it.
public enum DeepSeekV41HyperConnections {
    /// The three coefficient sets one sublayer's `hc_*` tensors produce.
    public struct Coefficients {
        /// `[tokens, multiplicity]` — what the *next* sublayer contracts with.
        public let pre: MLXArray
        /// `[tokens, multiplicity]` — this sublayer's expansion weights.
        public let post: MLXArray
        /// `[tokens, multiplicity, multiplicity]` — the doubly stochastic
        /// residual mixing matrix.
        public let combination: MLXArray
    }

    /// `make_identity_pre_mix`: a one-hot mix selecting the first copy.
    ///
    /// Float32, like the reference's, because every `pre_mix` downstream of it
    /// is a Sinkhorn output and those are float32.
    public static func identityPreMix(tokens: Int, multiplicity: Int) throws -> MLXArray {
        guard tokens > 0, multiplicity > 0 else {
            throw DeepSeekV41Error.configuration(
                "identity pre-mix needs positive tokens and multiplicity, got "
                    + "\(tokens) and \(multiplicity)")
        }
        var values = [Float](repeating: 0, count: tokens * multiplicity)
        for token in 0..<tokens { values[token * multiplicity] = 1 }
        return MLXArray(values, [tokens, multiplicity])
    }

    /// `Block.hc_mixes`: one float32 projection of the flattened stream,
    /// RMS-scaled, split and Sinkhorn-normalized.
    ///
    /// - Parameters:
    ///   - stream: `[tokens, multiplicity, hidden]`, the residual copies.
    ///   - function: `hc_attn_fn` or `hc_ffn_fn`,
    ///     `[(2 + multiplicity) * multiplicity, multiplicity * hidden]`.
    ///   - normEpsilon: `norm_eps`, which is `1e-20` in V4.1 and `1e-6` in V4 —
    ///     read from the configuration, never assumed.
    public static func mixes(
        stream: MLXArray,
        function: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        multiplicity: Int,
        iterations: Int,
        epsilon: Float,
        normEpsilon: Float,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> Coefficients {
        guard stream.ndim == 3, stream.shape[1] == multiplicity, stream.shape[0] > 0,
            stream.shape[2] > 0
        else {
            throw DeepSeekV41Error.configuration(
                "mHC stream must be [tokens, \(multiplicity), hidden], got \(stream.shape)")
        }
        guard normEpsilon.isFinite, normEpsilon > 0 else {
            throw DeepSeekV41Error.configuration(
                "mHC norm epsilon must be finite and positive, got \(normEpsilon)")
        }
        let flattenedWidth = multiplicity.multipliedReportingOverflow(by: stream.shape[2])
        guard !flattenedWidth.overflow else {
            throw DeepSeekV41Error.configuration("mHC flattened width overflows Int")
        }
        let mixedWidth = multiplicity.multipliedReportingOverflow(by: multiplicity + 2)
        guard !mixedWidth.overflow,
            function.ndim == 2,
            function.shape == [mixedWidth.partialValue, flattenedWidth.partialValue]
        else {
            throw DeepSeekV41Error.configuration(
                "hc_*_fn must be [\(mixedWidth.partialValue), \(flattenedWidth.partialValue)], "
                    + "got \(function.shape)")
        }

        // `x.flatten(2).float()`, then `F.linear(x, hc_fn) * rsqrt(mean(x^2) + eps)`.
        // The statistic is over the whole flattened `multiplicity * hidden`
        // stream, one per token, and the scale is applied *after* the
        // projection, not before it — which is not the same as normalizing the
        // input, and the reference is unambiguous about the order.
        let flattened = stream.asType(.float32).reshaped([-1, flattenedWidth.partialValue])
        let inverseRMS = rsqrt(
            mean(flattened * flattened, axis: -1, keepDims: true) + normEpsilon)
        let projected = k3Linear(flattened, function.asType(.float32)) * inverseRMS
        let split = try DeepSeekV4HyperConnections.splitSinkhorn(
            mixes: projected,
            scale: scale,
            base: base,
            multiplicity: multiplicity,
            iterations: iterations,
            epsilon: epsilon,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        return Coefficients(
            pre: split.pre, post: split.post, combination: split.combination)
    }

    /// `Block.hc_pre`: collapse the copies into one sublayer input.
    ///
    /// `[tokens, multiplicity, hidden] × [tokens, multiplicity] -> [tokens, hidden]`,
    /// summed in float32 and returned in the stream's own dtype — which is
    /// bfloat16 for the residual stream, so this is where that rounding happens.
    public static func contract(stream: MLXArray, preMix: MLXArray) throws -> MLXArray {
        guard stream.ndim == 3, preMix.ndim == 2,
            preMix.shape[0] == stream.shape[0], preMix.shape[1] == stream.shape[1]
        else {
            throw DeepSeekV41Error.configuration(
                "mHC contraction needs [tokens, multiplicity, hidden] and "
                    + "[tokens, multiplicity]; got \(stream.shape) and \(preMix.shape)")
        }
        let contracted = sum(
            expandedDimensions(preMix.asType(.float32), axis: -1) * stream.asType(.float32),
            axis: 1)
        return contracted.asType(stream.dtype)
    }

    /// The head's contraction, which in V4.1 is the same operation.
    ///
    /// `Transformer.forward` ends `h = layer.hc_pre(h, pre_mix)` — the *last
    /// block's* `hc_pre`, under the mix that block returned. There is no
    /// `hc_head_*` to read and no separate equation to get wrong, so this is an
    /// alias whose only job is to say at the call site that the stack is over.
    public static func collapse(stream: MLXArray, preMix: MLXArray) throws -> MLXArray {
        try contract(stream: stream, preMix: preMix)
    }

    /// `Block.hc_post`: expand a sublayer result back to `multiplicity` copies
    /// and mix the held residual in through `comb`.
    ///
    /// ```python
    /// y = post.unsqueeze(-1) * x.unsqueeze(-2) \
    ///     + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    /// ```
    ///
    /// ## `comb` is contracted down its columns, not across its rows
    ///
    /// This is the one line in the whole operator where a plausible reading is
    /// the wrong one, so it is worth spelling the broadcast out. `comb` is
    /// `[b, s, hc, hc]` and becomes `[b, s, hc, hc, 1]`; `residual` is
    /// `[b, s, hc, d]` and becomes `[b, s, hc, 1, d]`. The **third** axis is
    /// `hc` in both, so it aligns element-wise rather than broadcasting, and the
    /// product at `[b, s, i, j, e]` is `comb[i, j] * residual[i, e]`. `dim=2`
    /// then sums over `i`:
    ///
    /// ```text
    /// out[j, e] = Σ_i comb[i, j] · residual[i, e]        (V4.1)
    /// ```
    ///
    /// V4 contracts the other way — `Σ_j comb[i, j] · residual[j, e]`, which is
    /// what ``DeepSeekV4HyperConnections/combine(branch:residual:split:)``
    /// computes and what `Tools/v4_flash/hash_window_layer_reference.py` states
    /// for V4. Sinkhorn makes `comb` doubly stochastic, so both orientations
    /// produce a convex mixture and neither looks wrong from the outside; they
    /// are simply different models. V4.1's reference is unambiguous and this
    /// follows it.
    ///
    /// The whole thing is float32, rounded back to the sublayer output's dtype.
    public static func expand(
        branch: MLXArray,
        residual: MLXArray,
        coefficients: Coefficients
    ) throws -> MLXArray {
        guard branch.ndim == 2, residual.ndim == 3,
            branch.shape[0] == residual.shape[0],
            branch.shape[1] == residual.shape[2],
            coefficients.post.shape == [residual.shape[0], residual.shape[1]],
            coefficients.combination.shape == [
                residual.shape[0], residual.shape[1], residual.shape[1],
            ]
        else {
            throw DeepSeekV41Error.configuration(
                "mHC expansion shapes do not agree: branch \(branch.shape), residual "
                    + "\(residual.shape), post \(coefficients.post.shape), comb "
                    + "\(coefficients.combination.shape)")
        }
        let expanded = expandedDimensions(coefficients.post, axis: -1)
            * expandedDimensions(branch.asType(.float32), axis: 1)
        // [tokens, i, j, 1] × [tokens, i, 1, hidden] -> [tokens, i, j, hidden],
        // summed over i. See the note above on which axis `comb` is contracted
        // along.
        let mixed = sum(
            expandedDimensions(coefficients.combination, axis: -1)
                * expandedDimensions(residual.asType(.float32), axis: 2),
            axis: 1)
        return (expanded + mixed).asType(branch.dtype)
    }
}
