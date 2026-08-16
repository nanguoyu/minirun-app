import MLX

/// The exact power-of-two block scale both V4 activation references choose.
///
/// Given a block's ratio `r = max(|x|, floor) / dtypeMaximum`, the scale is
/// `2^ceil(log2 r)`. The published kernels derive it from the IEEE-754 exponent
/// field rather than from a logarithm, and so does this: for a positive normal
/// `r = 2^e * (1 + f)`, `ceil(log2 r)` is `e` when `f == 0` and `e + 1`
/// otherwise, so the scale's biased exponent is the ratio's biased exponent
/// plus one exactly when any mantissa bit is set. Every step is integer work on
/// the exponent field, so the result is exact by construction.
///
/// `exp2(ceil(log2 r))` would not be. A float32 `log2` rounds, and near a power
/// of two the rounding is decisive: one ulp of error on either side of an
/// integer either moves the ceiling by a whole factor of two or fails to move
/// it when it should, and the FP8 and FP4 midpoint policies are written against
/// the scale, so a doubled scale moves a tie and changes the model's output.
///
/// This runs where the maxima already are. Reading the small scale table back
/// to the host to do the same integer work in Swift cost the measured decode
/// pass 0.13 s over 471 syncs, and bought nothing: the arithmetic below is the
/// same arithmetic, and ``DeepSeekV4FP8ActivationReference/hostExactPowerOfTwoScales(_:)``
/// and its FP4 twin still state it in Swift so a test can hold the two against
/// each other, value by value and bit by bit.
enum DeepSeekV4PowerOfTwoScale {
    /// `2^ceil(log2(ratio))`, elementwise, for a positive normal float32
    /// `ratio`. A non-finite ratio yields itself, so a corrupted block keeps
    /// producing a non-finite value the run's output checks will refuse rather
    /// than a finite scale that would quietly launder it.
    static func ceiling(of ratio: MLXArray) -> MLXArray {
        let widened = ratio.asType(.float32)
        let bits = widened.view(dtype: .uint32)
        let biasedExponent = rightShift(bits, MLXArray(UInt32(23)))
            & MLXArray(UInt32(0xff))
        let mantissa = bits & MLXArray(UInt32(0x7f_ffff))
        let scaleExponent = biasedExponent + which(
            mantissa .> MLXArray(UInt32(0)),
            MLXArray(UInt32(1)),
            MLXArray(UInt32(0)))
        let scale = leftShift(scaleExponent, MLXArray(UInt32(23)))
            .view(dtype: .float32)
        return which(isFinite(widened), scale, widened)
    }
}
