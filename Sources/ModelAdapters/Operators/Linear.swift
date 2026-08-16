import Foundation
import MLX
import MLXBridge

/// A linear projection whose *storage* differs between checkpoints while its
/// arithmetic does not.
///
/// M6 could type the MoE block's latent and shared projections as
/// ``MXFP4Weights`` because in the tiny checkpoint they are quantized. The
/// flagship checkpoint quantizes only the 896 routed experts: its
/// `quantization_config.ignore` list excludes `shared_experts` and its headers
/// store `routed_expert_down_proj`, `routed_expert_up_proj`, `gate.weight` and
/// every `shared_experts.*` as plain BF16 `.weight` with no
/// `weight_packed`/`weight_scale` sibling to read.
///
/// So the same block is mixed-precision at one scale and uniformly quantized at
/// another (spec 10.2). Making the *dtype* the variable — rather than adding a
/// second MoE implementation — keeps one transcription of the block for both.
public enum K3Projection {
    /// Output-major `[outFeatures, inFeatures]`, BF16 or F32.
    case dense(MLXArray)
    case mxfp4(MXFP4Weights)

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch self {
        case .dense(let weight): return k3Linear(x, weight)
        case .mxfp4(let weight): return mxfp4MM(x, weight)
        }
    }

    public var isQuantized: Bool {
        if case .mxfp4 = self { return true }
        return false
    }

    /// The arrays this projection keeps alive, in either representation. Used
    /// where a residency figure has to be the real one rather than a plan's
    /// idea of it.
    public var residentArrays: [MLXArray] {
        switch self {
        case .dense(let weight): return [weight]
        case .mxfp4(let weight): return [weight.packed, weight.scales]
        }
    }
}

/// `y = x · W^T` for a float32 weight stored `[outFeatures, inFeatures]`.
///
/// The checkpoint's orientation, and the only correct reading of it: every
/// `nn.Linear` in the modeling code is `bias=False` and stores its weight
/// output-major. There is no transpose flag here for the same reason
/// ``mxfp4MM`` has none — the other reading is not a variant worth being able
/// to express.
@inline(__always)
public func k3Linear(_ x: MLXArray, _ weight: MLXArray) -> MLXArray {
    matmul(x, weight.transposed(1, 0))
}
