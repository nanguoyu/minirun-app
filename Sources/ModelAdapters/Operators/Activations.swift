import Foundation
import MLX

/// The `situ` activation and the two smaller nonlinearities around it.
///
/// Everything here computes in float32 unconditionally. That is not a
/// convenience: `SituAndMul.forward` pins its arithmetic to float32 regardless
/// of parameter dtype (`modeling_kimi_k3_linear.py:88-95`), and it is one of the
/// five operations `Tools/k3_reference/precision_patch.py` had to transcribe in
/// order to measure the reference's own rounding. Mirroring the pin is what
/// makes the 1e-4 per-layer tolerance a statement about this port rather than
/// about a dtype choice.
public enum K3Activation {

    /// `situ(gate, up)`, where `x` is the *concatenated* `[w1(x), w3(x)]`.
    ///
    /// ```text
    /// situ(gate, up) = (beta * tanh(gate / beta) * sigmoid(gate))
    ///                * (linearBeta * tanh(up / linearBeta))
    /// ```
    ///
    /// This is **not** a SwiGLU. The up branch is squashed by
    /// `25 * tanh(up / 25)`, which is near-linear for `|up| << 25` and saturates
    /// beyond it — so an implementation that treats the up branch as the
    /// identity agrees on small activations and diverges on large ones. The
    /// committed `subop-situ` bundle carries a ±60 sweep for exactly that
    /// reason (M6A record, §6).
    ///
    /// - Parameter x: `[..., 2 * d]`, gate in the first half, up in the second.
    /// - Returns: `[..., d]`.
    public static func situ(_ x: MLXArray, beta: Float, linearBeta: Float?) -> MLXArray {
        let d = x.shape[x.ndim - 1] / 2
        let halves = split(x, indices: [d], axis: -1)
        let gate = halves[0].asType(.float32)
        var up = halves[1].asType(.float32)

        // The scalar factors are lifted to rank-0 arrays rather than left as
        // Swift `Float`s: mlx-swift's mixed scalar/array operators make a
        // three-term chain ambiguous to the type checker. It is the same
        // arithmetic — a rank-0 float32 array broadcasts identically.
        let betaScalar = MLXArray(beta)
        let situA = tanh(gate / beta) * betaScalar * sigmoid(gate)
        if let linearBeta {
            up = tanh(up / linearBeta) * MLXArray(linearBeta)
        }
        return situA * up
    }

    /// `x * sigmoid(x)`. The activation `ShortConvolution` applies after the
    /// depthwise convolution (`activation="silu"`).
    public static func silu(_ x: MLXArray) -> MLXArray {
        x * sigmoid(x)
    }

    /// `log(1 + exp(x))`, evaluated in the numerically stable form.
    ///
    /// `log1p(exp(x))` overflows to infinity for `x` above ~88 in float32,
    /// which the KDA gate reaches on nothing in these fixtures but would reach
    /// on a longer prompt — and an infinity there propagates through
    /// `-exp(A_log) * softplus(...)` into a state of zeros, silently. PyTorch
    /// avoids it with a `threshold=20` branch; this uses the branch-free
    /// identity `softplus(x) = max(x, 0) + log1p(exp(-|x|))`, which agrees with
    /// the threshold form to float32 rounding everywhere (above the threshold
    /// `log1p(exp(-x))` is already below the representable gap at `x`).
    public static func softplus(_ x: MLXArray) -> MLXArray {
        maximum(x, MLXArray(Float(0))) + log1p(exp(-abs(x)))
    }
}
