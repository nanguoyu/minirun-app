import MLX

/// `RMSNorm.forward` as V4.1 states it.
///
/// ``K3Norm/rms(_:weight:eps:)`` is the same three lines and is what this calls.
/// The one thing it does not do is the last one: the reference is
///
/// ```python
/// dtype = x.dtype; x = x.float()
/// x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
/// return (self.weight * x).to(dtype)
/// ```
///
/// so the result is rounded back to the **input's** dtype, not left in float32
/// and not taken from the weight's. Every norm in V4.1's text path is handed a
/// bfloat16 activation and a bfloat16 weight, so this rounding happens forty
/// times a block and is not optional.
///
/// `norm_eps` is `1e-20` in V4.1 against V4's `1e-6`. It is read from
/// `inference/config.json` through ``DeepSeekV41Config/rmsNormEpsilon``, never
/// written down here.
public enum DeepSeekV41RMSNorm {
    public static func apply(
        _ input: MLXArray, weight: MLXArray, epsilon: Float
    ) throws -> MLXArray {
        guard input.ndim >= 1, let width = input.shape.last,
            weight.ndim == 1, weight.shape[0] == width
        else {
            throw DeepSeekV41Error.configuration(
                "RMS norm weight \(weight.shape) does not match input \(input.shape)")
        }
        guard epsilon.isFinite, epsilon > 0 else {
            throw DeepSeekV41Error.configuration(
                "RMS norm epsilon must be finite and positive, got \(epsilon)")
        }
        return K3Norm.rms(input, weight: weight, eps: epsilon).asType(input.dtype)
    }
}

/// `ParallelEmbedding` and the expansion into `hc_mult` residual copies.
///
/// The published `embed.weight` is `BF16 [129280, 5120]` — 1.32 GB — so the
/// table is never resident: ``DeepSeekV41GlobalArtifact/embeddingRow(token:cancellationCheck:)``
/// reads one row per token id, which is the same window discipline V4's output
/// head uses and the reason the global unit reader refuses to load `embed` whole.
///
/// The expansion is `h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`: every copy starts
/// as the same embedding, and `make_identity_pre_mix` then makes the first
/// block read copy 0. Nothing is learned here.
public enum DeepSeekV41Embedding {
    /// `[tokens, hidden] -> [tokens, multiplicity, hidden]`.
    public static func expandToResidualStreams(
        _ embedded: MLXArray, multiplicity: Int
    ) throws -> MLXArray {
        guard embedded.ndim == 2, embedded.shape[0] > 0, embedded.shape[1] > 0,
            multiplicity > 0
        else {
            throw DeepSeekV41Error.configuration(
                "embedding must be [tokens, hidden] with a positive multiplicity; got "
                    + "\(embedded.shape) and \(multiplicity)")
        }
        return repeated(
            expandedDimensions(embedded, axis: 1), count: multiplicity, axis: 1)
    }

    /// The rows for a prompt, one storage read per distinct token id.
    ///
    /// Distinct, because a prompt repeats ids and the table is on the drive: a
    /// 512-token prompt of English text names far fewer than 512 rows, and each
    /// row read is 10,240 bytes at an arbitrary offset in a 1.32 GB file.
    public static func rows(
        for tokens: [Int],
        from artifact: DeepSeekV41GlobalArtifact,
        cancellationCheck: () throws -> Void = {}
    ) throws -> MLXArray {
        guard !tokens.isEmpty else {
            throw DeepSeekV41Error.configuration("cannot embed an empty token sequence")
        }
        var cache = [Int: MLXArray]()
        var stack = [MLXArray]()
        stack.reserveCapacity(tokens.count)
        for token in tokens {
            try cancellationCheck()
            if let held = cache[token] {
                stack.append(held)
                continue
            }
            let row = try artifact.embeddingRow(
                token: token, cancellationCheck: cancellationCheck)
            cache[token] = row
            stack.append(row)
        }
        return concatenated(stack, axis: 0)
    }
}
