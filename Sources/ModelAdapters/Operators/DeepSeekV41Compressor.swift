import MLX

/// `Compressor`: pool `compress_ratio` consecutive tokens into one main-KV
/// latent.
///
/// ## What V4.1 dropped
///
/// V4's compressor overlapped its groups and added an absolute positional
/// encoding, and ``DeepSeekV4CompressorPool`` carries both. V4.1's does neither:
/// groups are disjoint and consecutive, and there is no positional term at all
/// inside the compressor — the latent is rotated *afterwards*, by
/// `Attention._compress_kv`, at the position of the first token of its group.
///
/// ## The two ratios are two different programs
///
/// - **ratio 1** (every decoder block that owns KV) is a plain projection:
///   `norm(wkv(x))`, no gate, in the checkpoint's bfloat16. There is nothing to
///   pool, so there is no state and every token yields a latent.
/// - **ratio 2** (every encoder block that owns KV) promotes `wkv` and `wgate`
///   to float32 — the checkpoint stores them bfloat16 and the reference's
///   `Linear(dtype=torch.float32)` widens them at load — pools `ratio` tokens
///   under a float32 softmax of the gate **per channel**, and only then rounds
///   back to the activation's dtype for the norm.
///
/// The gate's softmax is over the group axis, not the channel axis:
/// `score.softmax(dim=2)` on `[b, groups, ratio, head_dim]` normalizes the
/// `ratio` tokens against each other, separately for each of the `head_dim`
/// channels. A softmax over the wrong axis is a plausible-looking model that is
/// not this one.
///
/// ## Pre-RoPE is deliberate
///
/// The indexer needs the unrotated latent to derive its keys, so `Compressor`
/// returns it before RoPE and `Attention` rotates it after the indexer has run.
/// A caller that rotated first would silently change the index keys.
///
/// Block 20 of the published model has a `wkv` and **no** `wgate`: it is the
/// first decoder KV source, ratio 1, and the plain path has no gate to publish.
/// ``DeepSeekV41BlockArtifact/contains(_:)`` answers that question, and a
/// reader that treated the absence as damage would refuse a correct unit.
public enum DeepSeekV41Compressor {
    /// The tail of an incomplete group, carried across decode steps.
    ///
    /// `[ratio, headDimension]` float32 both. `scores` starts at `-inf`, which
    /// is what makes a slot that has never been written contribute nothing to
    /// the softmax; `values` starts at zero. Only a ratio above one has one.
    public struct State {
        public var values: MLXArray
        public var scores: MLXArray

        public init(ratio: Int, headDimension: Int) throws {
            guard ratio > 1, headDimension > 0 else {
                throw DeepSeekV41Error.configuration(
                    "a compressor state exists only for ratio > 1; got ratio \(ratio) and "
                        + "head dimension \(headDimension)")
            }
            self.values = MLXArray.zeros([ratio, headDimension], dtype: .float32)
            self.scores = MLXArray.full(
                [ratio, headDimension], values: MLXArray(-Float.infinity), dtype: .float32)
        }

        init(values: MLXArray, scores: MLXArray) {
            self.values = values
            self.scores = scores
        }
    }

    public struct Result {
        /// `[latents, headDimension]` before RoPE, or `nil` while the current
        /// group is still filling up — which is what a ratio-2 decode step
        /// returns on every other token.
        public let latent: MLXArray?
        public let state: State?
    }

    /// The whole `Compressor.forward`, both ratios and both phases.
    ///
    /// - Parameters:
    ///   - hidden: `[tokens, hidden]`, the block's `attn_norm`ed input.
    ///   - startPosition: 0 for a prefill chunk, the absolute position for a
    ///     one-token decode step.
    ///   - state: required for `ratio > 1`, ignored otherwise.
    public static func forward(
        hidden: MLXArray,
        ratio: Int,
        keyValueWeight: MLXArray,
        gateWeight: MLXArray?,
        normWeight: MLXArray,
        normEpsilon: Float,
        startPosition: Int,
        state: State?
    ) throws -> Result {
        guard hidden.ndim == 2, hidden.shape[0] > 0, ratio > 0, startPosition >= 0 else {
            throw DeepSeekV41Error.configuration(
                "compressor input must be [tokens, hidden] with a positive ratio and a "
                    + "nonnegative start position; got \(hidden.shape), \(ratio), "
                    + "\(startPosition)")
        }
        let dtype = hidden.dtype

        if ratio == 1 {
            guard gateWeight == nil else {
                throw DeepSeekV41Error.configuration(
                    "a ratio-1 compressor has no gate; block 20 publishes no wgate and the "
                        + "reference builds none")
            }
            let projected = k3Linear(
                hidden.asType(.float32), keyValueWeight.asType(.float32)).asType(dtype)
            return Result(
                latent: try DeepSeekV41RMSNorm.apply(
                    projected, weight: normWeight, epsilon: normEpsilon),
                state: nil)
        }

        guard let gateWeight, var state else {
            throw DeepSeekV41Error.configuration(
                "a ratio-\(ratio) compressor needs both a gate weight and a carried state")
        }
        // `x = x.float()` and both projections in float32: the checkpoint's
        // bfloat16 weights widened, then a float32 GEMM.
        let widened = hidden.asType(.float32)
        let values = k3Linear(widened, keyValueWeight.asType(.float32))
        let scores = k3Linear(widened, gateWeight.asType(.float32))

        let tokens = hidden.shape[0]
        var pooled: MLXArray?
        if startPosition == 0 {
            let remainder = tokens % ratio
            let cutoff = tokens - remainder
            if remainder > 0 {
                // The trailing partial group waits in the state, and only its
                // first `remainder` slots are written: the rest keep whatever
                // they held, which for a fresh state is `-inf` and zero.
                state.values = replacingLeadingRows(
                    state.values, with: values[cutoff..., 0...], count: remainder)
                state.scores = replacingLeadingRows(
                    state.scores, with: scores[cutoff..., 0...], count: remainder)
            }
            guard cutoff > 0 else { return Result(latent: nil, state: state) }
            let groups = cutoff / ratio
            let width = hidden.shape[1] == 0 ? 0 : values.shape[1]
            let groupedValues = values[0..<cutoff, 0...].reshaped([groups, ratio, width])
            let groupedScores = scores[0..<cutoff, 0...].reshaped([groups, ratio, width])
            pooled = sum(
                groupedValues * softmax(groupedScores, axis: 1, precise: true), axis: 1)
        } else {
            guard tokens == 1 else {
                throw DeepSeekV41Error.configuration(
                    "a decode step compresses exactly one token, got \(tokens)")
            }
            let slot = startPosition % ratio
            state.values = replacingRow(state.values, at: slot, with: values[0])
            state.scores = replacingRow(state.scores, at: slot, with: scores[0])
            if (startPosition + 1) % ratio == 0 {
                pooled = sum(
                    state.values * softmax(state.scores, axis: 0, precise: true),
                    axis: 0, keepDims: true)
            }
        }
        guard let pooled else { return Result(latent: nil, state: state) }
        return Result(
            latent: try DeepSeekV41RMSNorm.apply(
                pooled.asType(dtype), weight: normWeight, epsilon: normEpsilon),
            state: state)
    }

    private static func replacingRow(
        _ table: MLXArray, at row: Int, with values: MLXArray
    ) -> MLXArray {
        var rows = [MLXArray]()
        rows.reserveCapacity(table.shape[0])
        for index in 0..<table.shape[0] {
            rows.append(index == row ? values.asType(table.dtype) : table[index])
        }
        return stacked(rows, axis: 0)
    }

    private static func replacingLeadingRows(
        _ table: MLXArray, with values: MLXArray, count: Int
    ) -> MLXArray {
        guard count < table.shape[0] else { return values.asType(table.dtype) }
        return concatenated(
            [values.asType(table.dtype), table[count...]], axis: 0)
    }
}
