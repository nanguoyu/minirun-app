import Foundation
import MLX

/// AttnRes: the residual stream is an attention over stored checkpoints.
///
/// This is the mechanism most likely to be ported as an ordinary additive
/// residual and to pass every small test while being wrong — M6A's follow-up
/// names it as the thing to implement first, for that reason.
///
/// Transcribed from `_apply_attn_res`, `modeling_kimi_k3_linear.py:1195-1208`:
///
/// ```text
/// v      = concat([blockResidual, prefixSum[:, None, :]], axis: 1)  // [T, B+1, H]
/// k      = v * rsqrt(mean(v^2, -1) + eps)                           // RMS normalise
/// scores = sum(k * (normWeight * projWeight), -1)                   // [T, B+1]
/// out    = softmax(scores, -1) @ v                                  // convex combination
/// ```
///
/// Two properties worth keeping in mind when reading a divergence report:
///
/// - The output is a **convex combination of the stored checkpoints**, never a
///   sum. A port that adds residuals the ordinary way diverges at layer 0 and
///   never recovers.
/// - The softmax runs close to saturation. On the committed prompts the largest
///   probability is 0.99999988; on two rejected prompts it reached exactly 1.0
///   in float32, at which point that layer's FFN input is bit-identically one
///   stored checkpoint and its routing is independent of all preceding context.
///   So a *layer* comparison can pass while the mechanism is subtly wrong —
///   which is why the trace bundles carry the per-call probability vectors and
///   why this port reports them.
public enum K3AttnRes {

    /// One AttnRes application.
    ///
    /// - Parameters:
    ///   - prefixSum: `[tokens, hidden]`.
    ///   - blockResidual: `[tokens, blocks, hidden]`. May have `blocks == 0`,
    ///     in which case the softmax is over the prefix sum alone and the
    ///     result is the prefix sum — the caller skips the call entirely in
    ///     that case, as the reference does, but the arithmetic agrees.
    ///   - scoreWeight: `normWeight * projWeight`, elementwise, `[hidden]`.
    ///     Precomputed once at load time; it is the only place the two weight
    ///     vectors are ever used, and always as a product.
    /// - Returns: `(output [tokens, hidden], probabilities [tokens, blocks + 1])`.
    public static func apply(
        prefixSum: MLXArray,
        blockResidual: MLXArray?,
        scoreWeight: MLXArray,
        eps: Float
    ) -> (output: MLXArray, probabilities: MLXArray) {
        let current = expandedDimensions(prefixSum.asType(.float32), axis: 1)
        let v: MLXArray
        if let blockResidual, blockResidual.shape[1] > 0 {
            v = concatenated([blockResidual.asType(.float32), current], axis: 1)
        } else {
            v = current
        }

        let variance = mean(v * v, axis: -1, keepDims: true)
        let k = v * rsqrt(variance + eps)
        let scores = sum(k * scoreWeight, axis: -1)
        let probabilities = softmax(scores, axis: -1)
        let output = matmul(expandedDimensions(probabilities, axis: 1), v).squeezed(axis: 1)
        return (output, probabilities)
    }
}
