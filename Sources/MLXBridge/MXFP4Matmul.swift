import Foundation
import MLX

/// Dense MXFP4 matmul: `y = x · W^T`, with `W` in checkpoint layout.
///
/// - Parameters:
///   - x: `[..., M, inFeatures]`. `float32` in gives `float32` out; `float16`
///     and `bfloat16` are accepted and produce their own dtype.
///   - weights: a non-stacked ``MXFP4Weights`` (`experts == nil`).
/// - Returns: `[..., M, outFeatures]`.
///
/// `transpose` is not a parameter. The checkpoint stores
/// `[outFeatures, inFeatures]`, so `transpose: true` is the only correct
/// reading, and `transpose: false` would additionally place the 32-element
/// scale groups along the wrong axis. Exposing it would only make a wrong call
/// expressible.
public func mxfp4MM(
    _ x: MLXArray,
    _ weights: MXFP4Weights,
    stream: StreamOrDevice = .default
) -> MLXArray {
    // `quantizedMM`, not `quantizedMatmul`: the latter is a deprecated alias in
    // mlx-swift 0.31.4 and warns.
    quantizedMM(
        x,
        weights.packed,
        scales: weights.scales,
        biases: nil,
        transpose: true,
        groupSize: MXFP4Weights.groupSize,
        bits: MXFP4Weights.bits,
        mode: .mxfp4,
        stream: stream)
}

/// Routed-expert MXFP4 matmul: `out[t, j, :] = W[rhsIndices[t, j]] · x[t, :]`.
///
/// This is the operator a MoE decode step needs: one activation per token, each
/// token routed to `expertsPerToken` of the stacked experts.
///
/// - Parameters:
///   - x: `[tokens, inFeatures]` activations, one row per token.
///   - weights: a stacked ``MXFP4Weights`` (`experts != nil`).
///   - rhsIndices: `[tokens, expertsPerToken]` expert ids. Need not be sorted,
///     and may repeat within a row.
///   - sortedIndices: passed through to MLX as a performance hint. Leave it
///     `false` unless the indices really are sorted — see below.
/// - Returns: `[tokens, expertsPerToken, outFeatures]`.
///
/// ## Why this wrapper reshapes for you
///
/// MLX's `gatherQuantizedMatmul` reads `x`'s last two axes as `(M, inFeatures)`
/// and *broadcasts everything before them* against `rhsIndices`' shape. So `x`
/// must be presented as `[tokens, 1, 1, inFeatures]` — batch shape
/// `(tokens, 1)`, which broadcasts against `(tokens, expertsPerToken)`.
///
/// Passing `[tokens, 1, inFeatures]` instead leaves batch shape `(tokens,)`.
/// When `tokens != expertsPerToken` that raises. When they are **equal** it is
/// a legal broadcast: no error, an identical output shape, and the operator
/// computes `W[rhsIndices[t, j]] · x[j]` — the slot index silently used as the
/// token index. Measured deviation from the reference was ~1.5 relative to the
/// output scale, against a 1e-5 tolerance.
///
/// That hazard is the reason this function takes plain `[tokens, inFeatures]`
/// and does the axis juggling itself. Callers cannot express the mistake.
///
/// ## `sortedIndices`
///
/// MLX documents it as "may allow a faster implementation if the passed indices
/// are sorted", which says nothing about what happens when they are not. On
/// mlx-swift 0.31.4 it was measured to have no effect on results even for
/// unsorted indices — but that is an implementation detail of one version, not
/// a promise, so this wrapper defaults it off and leaves sorting to the caller.
public func mxfp4GatherMM(
    _ x: MLXArray,
    _ weights: MXFP4Weights,
    rhsIndices: MLXArray,
    sortedIndices: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    precondition(
        weights.experts != nil,
        "mxfp4GatherMM needs a stacked MXFP4Weights; use mxfp4MM for a single matrix")
    precondition(
        x.ndim == 2,
        "x must be [tokens, inFeatures]; this function inserts the broadcast axes itself")
    precondition(
        rhsIndices.ndim == 2 && rhsIndices.shape[0] == x.shape[0],
        "rhsIndices must be [tokens, expertsPerToken] matching x's token count")

    let tokens = x.shape[0]
    let expertsPerToken = rhsIndices.shape[1]

    // `gatherQuantizedMM`, not `gatherQuantizedMatmul`: same deprecated-alias
    // situation as above. Same function, current name.
    let result = gatherQuantizedMM(
        x.reshaped([tokens, 1, 1, weights.inFeatures]),
        weights.packed,
        scales: weights.scales,
        biases: nil,
        lhsIndices: nil,
        rhsIndices: rhsIndices,
        transpose: true,
        groupSize: MXFP4Weights.groupSize,
        bits: MXFP4Weights.bits,
        mode: .mxfp4,
        sortedIndices: sortedIndices,
        stream: stream)

    // [tokens, expertsPerToken, 1, outFeatures] -> [tokens, expertsPerToken, outFeatures]
    return result.reshaped([tokens, expertsPerToken, weights.outFeatures])
}

/// Dequantize packed MXFP4 weights to a dense array. Validation and debugging
/// only — the whole point of the project is not to do this on the hot path.
///
/// MLX returns `bfloat16` for `mxfp4`; widening to `float32` is lossless
/// because an e2m1 value carries at most two significant bits and an e8m0 scale
/// is a power of two, so the product is exact in both. That is what lets the
/// tests demand bit-exact agreement with the CPU reference rather than a
/// tolerance (M2 record, §2).
public func mxfp4Dequantize(
    _ weights: MXFP4Weights,
    dtype: DType = .float32,
    stream: StreamOrDevice = .default
) -> MLXArray {
    dequantized(
        weights.packed,
        scales: weights.scales,
        biases: nil,
        groupSize: MXFP4Weights.groupSize,
        bits: MXFP4Weights.bits,
        mode: .mxfp4,
        dtype: dtype,
        stream: stream)
}
