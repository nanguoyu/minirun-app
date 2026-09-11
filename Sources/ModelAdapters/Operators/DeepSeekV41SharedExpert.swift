import Foundation
import MLX
import MLXBridge

/// The one shared expert every token goes through, over FP8 [32, 32] weights.
///
/// `MoE.forward` ends `y += self.shared_experts(x)`: one `Expert` with the same
/// SwiGLU and the same `swiglu_limit`, differing from a routed one only in that
/// it takes no routing weight and its `Linear`s are FP8 rather than FP4.
///
/// ## The [32, 32] block grid
///
/// V4-Flash-0731 used a 128×128 E4M3/E8M0 weight grid, and
/// ``DeepSeekV4BlockFP8LinearReference`` refuses anything else by name. This
/// checkpoint uses 32×32 — measured on every one of its 355 dense tensors — and
/// quantizes activations in blocks of 32 to match, not 128. So this is the same
/// bridge at this checkpoint's block size rather than a parameterisation of
/// V4's, which would have made a wrong block size expressible at both call
/// sites instead of neither.
///
/// It is a *reference*, not the product kernel: MLX exposes no FP8×FP8 Metal
/// operation, so the activation rounding is spelled out and the unchanged
/// weight bytes then go through ``blockFP8MM``. That is exactly the shape of
/// V4's bridge and it inherits V4's finding — the rounding is the reference's
/// arithmetic, not an approximation of it.
///
/// The dense path owns the block-FP8 projection for attention and the head.
/// This file carries only what the MoE needs, so Phase 2 does not have to wait
/// on it; if the two converge on one helper later, this is the one to delete.
public enum DeepSeekV41SharedExpert {

    /// One E8M0 exponent per 32×32 weight block, and per 32 activations.
    public static let blockSize = 32

    /// `Linear.forward` for an FP8 weight: round the activation, then multiply
    /// the unchanged checkpoint bytes.
    public static func project(
        _ input: MLXArray,
        weights: BlockFP8Weights,
        activation: DeepSeekV41ExpertActivation = .referenceFP8,
        diagnostics: DeepSeekV4Diagnostics = .validating,
        stream: StreamOrDevice = .default
    ) throws -> MLXArray {
        guard weights.scaleBlockRows == blockSize,
            weights.scaleBlockColumns == blockSize
        else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "shared expert weights must use a \(blockSize)x\(blockSize) scale grid; "
                    + "got \(weights.scaleBlockRows)x\(weights.scaleBlockColumns)")
        }
        let rounded = try activation.applied(input, diagnostics: diagnostics)
        return try blockFP8MM(rounded, weights, stream: stream)
    }

    /// `Expert.forward` with `weights=None` — the same clamps, no routing
    /// weight, and the same cast back to the input dtype before `w2`.
    public static func forward(
        _ input: MLXArray,
        gate gateWeights: BlockFP8Weights,
        down downWeights: BlockFP8Weights,
        up upWeights: BlockFP8Weights,
        swiGLULimit: Float,
        activation: DeepSeekV41ExpertActivation = .referenceFP8,
        diagnostics: DeepSeekV4Diagnostics = .validating,
        stream: StreamOrDevice = .default
    ) throws -> MLXArray {
        guard input.ndim == 2, input.shape[0] > 0, input.shape[1] > 0 else {
            throw DeepSeekV41Error.experts(
                "shared expert: input must be a non-empty [tokens, dim] matrix, got "
                    + "\(input.shape)")
        }
        guard gateWeights.inFeatures == input.shape[1],
            upWeights.inFeatures == input.shape[1],
            gateWeights.outFeatures == upWeights.outFeatures,
            downWeights.inFeatures == gateWeights.outFeatures,
            downWeights.outFeatures == input.shape[1]
        else {
            throw DeepSeekV41Error.experts(
                "shared expert: w1 \(gateWeights.outFeatures)x\(gateWeights.inFeatures), "
                    + "w2 \(downWeights.outFeatures)x\(downWeights.inFeatures) and w3 "
                    + "\(upWeights.outFeatures)x\(upWeights.inFeatures) do not compose "
                    + "over a \(input.shape[1])-wide hidden state")
        }
        guard swiGLULimit.isFinite, swiGLULimit >= 0 else {
            throw DeepSeekV41Error.experts(
                "shared expert: SwiGLU limit must be finite and nonnegative")
        }

        let dtype = input.dtype
        var gate = try project(
            input, weights: gateWeights, activation: activation,
            diagnostics: diagnostics, stream: stream).asType(.float32)
        var up = try project(
            input, weights: upWeights, activation: activation,
            diagnostics: diagnostics, stream: stream).asType(.float32)
        if swiGLULimit > 0 {
            let upper = MLXArray(swiGLULimit)
            gate = minimum(gate, upper, stream: stream)
            up = minimum(maximum(up, MLXArray(-swiGLULimit), stream: stream), upper,
                         stream: stream)
        }
        let hidden = (K3Activation.silu(gate) * up).asType(dtype)
        return try project(
            hidden, weights: downWeights, activation: activation,
            diagnostics: diagnostics, stream: stream).asType(.float32)
    }
}
