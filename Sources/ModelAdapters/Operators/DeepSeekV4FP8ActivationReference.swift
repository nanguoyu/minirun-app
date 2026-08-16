import MLX

/// Equation-level reference for DeepSeek V4's in-place E4M3 activation pass.
///
/// The official sliding-window attention path applies this operation to the
/// non-rotary key/value channels in blocks of 64. Each row/block chooses a
/// power-of-two scale, rounds the normalized values to finite E4M3, and then
/// immediately dequantizes them. This implementation spells the rounding out in
/// MLX elementwise operations instead of a fused kernel: it is a correctness
/// oracle for the artifact runner, not the product hot path. Nothing it
/// computes crosses to the host — the scale table included, since `b7c2092`
/// measured that round trip at 471 syncs and 0.13 s of a decode pass.
public enum DeepSeekV4FP8ActivationReference {
    private static let fp8Maximum: Float = 448
    private static let minimumAbsoluteMaximum: Float = 1e-4
    private static let minimumNormal: Float = 1.0 / 64.0
    private static let subnormalStep: Float = 1.0 / 512.0

    /// `phaseAccounting` brackets the two host round trips this reference
    /// makes — the entry finiteness sweep and the scale search — and changes
    /// nothing else. It is nil for every caller that is not decoding.
    ///
    /// `diagnostics` decides whether the entry sweep runs at all; see
    /// ``DeepSeekV4Diagnostics``.
    public static func quantizeDequantize(
        _ input: MLXArray,
        blockSize: Int,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        guard input.ndim > 0, let width = input.shape.last, width > 0,
            blockSize > 0, width.isMultiple(of: blockSize),
            input.dtype.isFloatingPoint, !input.dtype.isComplex
        else {
            throw DeepSeekV4Error.attention(
                "FP8 activation input must be real floating point with a positive last "
                    + "dimension divisible by block size \(blockSize); got \(input.shape) "
                    + "\(input.dtype)")
        }

        let groups = width / blockSize
        var blockedShape = Array(input.shape.dropLast())
        blockedShape.append(groups)
        blockedShape.append(blockSize)
        let blocked = input.asType(.float32).reshaped(blockedShape)
        // Guard-only, and a full sync: it forces whatever graph produced the
        // activation before a single scale has been chosen. Diagnostic, so it
        // is skipped entirely — sync and check — unless the caller asked for it.
        if diagnostics.validateFiniteness {
            guard measuringPhase(phaseAccounting?.recordFinitenessSweep(nanoseconds:), {
                isFinite(blocked).all().item(Bool.self)
            }) else {
                throw DeepSeekV4Error.attention(
                    "FP8 activation input contains a non-finite value")
            }
        }
        let absolute = MLX.abs(blocked)
        // The bracket stays around what is left of the scale search: graph
        // construction, and no host round trip at all.
        let scale = measuringPhase(
            phaseAccounting?.recordActivationScaleSync(nanoseconds:)
        ) {
            exactPowerOfTwoScales(
                absolute.max(axis: -1), blockedShape: blockedShape)
        }
        let normalized = minimum(
            maximum(blocked / scale, MLXArray(-fp8Maximum)),
            MLXArray(fp8Maximum))
        let magnitude = MLX.abs(normalized)

        // E4M3FN has three mantissa bits, a minimum normal value of 2^-6,
        // subnormal spacing 2^-9, and the largest finite value 448. Construct
        // every possible normal step from exact host-side powers of two: GPU
        // log/exp can land one ulp below a power of two and change a tie. MLX's
        // public round operation also rounds halves away from zero, whereas an
        // FP8 cast uses round-to-nearest-even, so spell that rule out below.
        var normalStep = MLXArray(powerOfTwo(-9))
        for exponent in -5...8 {
            normalStep = which(
                magnitude .>= MLXArray(powerOfTwo(exponent)),
                MLXArray(powerOfTwo(exponent - 3)),
                normalStep)
        }
        let roundedNormal = roundToNearestEven(magnitude / normalStep) * normalStep
        let roundedSubnormal = roundToNearestEven(magnitude / subnormalStep) * subnormalStep
        let quantizedMagnitude = minimum(
            which(
                magnitude .< MLXArray(minimumNormal),
                roundedSubnormal,
                roundedNormal),
            MLXArray(fp8Maximum))
        let dequantized = sign(normalized) * quantizedMagnitude * scale
        return dequantized.reshaped(input.shape).asType(input.dtype)
    }

    private static func roundToNearestEven(_ input: MLXArray) -> MLXArray {
        let lower = floor(input)
        let fraction = input - lower
        let odd = lower - floor(lower / 2) * 2
        let upper = lower + 1
        return which(
            fraction .< MLXArray(0.5 as Float),
            lower,
            which(
                fraction .> MLXArray(0.5 as Float),
                upper,
                which(odd .> MLXArray(0.5 as Float), upper, lower)))
    }

    private static func powerOfTwo(_ exponent: Int) -> Float {
        Float(sign: .plus, exponent: exponent, significand: 1)
    }

    /// One exact power-of-two scale per block, chosen where the maxima are.
    ///
    /// The rule is ``hostExactPowerOfTwoScales(_:)`` below, expressed on the
    /// GPU by ``DeepSeekV4PowerOfTwoScale/ceiling(of:)``: the same IEEE-754
    /// exponent arithmetic on the same ratio, with no host round trip. The two
    /// are held to bit equality by test over a dense sweep of maxima.
    ///
    /// The host derivation's two throws have no expression here, and need
    /// none. Both are unreachable for a finite maximum — the ratio of a finite
    /// maximum to the dtype's maximum is always positive, finite and normal,
    /// because the floor below keeps it above `1e-4 / 448` — and for a
    /// non-finite one the scale is non-finite too, which the output head and
    /// `greedyToken` refuse.
    static func exactPowerOfTwoScales(
        _ maxima: MLXArray,
        blockedShape: [Int]
    ) -> MLXArray {
        var scaleShape = blockedShape
        scaleShape[scaleShape.count - 1] = 1
        let ratio = maximum(
            maxima.asType(.float32), MLXArray(minimumAbsoluteMaximum))
            / MLXArray(fp8Maximum)
        return DeepSeekV4PowerOfTwoScale.ceiling(of: ratio).reshaped(scaleShape)
    }

    /// The scale rule, one value at a time, in Swift.
    ///
    /// This is the derivation this reference shipped with and the operator
    /// above reproduces: it is kept so a test can state the rule independently
    /// of MLX and compare bit patterns, not because a pass runs it.
    static func hostExactPowerOfTwoScales(_ values: [Float]) throws -> [Float] {
        var scales = [Float]()
        scales.reserveCapacity(values.count)
        for value in values {
            let absoluteMaximum = Swift.max(value, minimumAbsoluteMaximum)
            let ratio = absoluteMaximum / fp8Maximum
            guard ratio.isFinite, ratio > 0 else {
                throw DeepSeekV4Error.attention(
                    "FP8 activation block contains a non-finite magnitude")
            }
            let bits = ratio.bitPattern
            let storedExponent = Int((bits >> 23) & 0xff)
            let mantissa = bits & 0x7f_ffff
            guard storedExponent > 0, storedExponent < 0xff else {
                throw DeepSeekV4Error.attention(
                    "FP8 activation scale is outside the finite normal Float range")
            }
            let exponent = storedExponent - 127 + (mantissa == 0 ? 0 : 1)
            let scaleExponent = exponent + 127
            guard scaleExponent > 0, scaleExponent < 0xff else {
                throw DeepSeekV4Error.attention(
                    "FP8 activation power-of-two scale is not representable")
            }
            scales.append(Float(bitPattern: UInt32(scaleExponent) << 23))
        }
        return scales
    }
}
