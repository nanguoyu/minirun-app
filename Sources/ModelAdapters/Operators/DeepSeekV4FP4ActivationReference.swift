import Foundation
import MLX

/// Equation-level reference for DeepSeek V4's indexer activation transform.
///
/// The published lightning indexer applies an orthonormal Walsh-Hadamard
/// transform, chooses one power-of-two scale for every group of 32 values,
/// rounds the scaled activation to finite E2M1, and immediately dequantizes it.
/// This implementation deliberately spells out every E2M1 midpoint rather than
/// fusing them. It is a correctness oracle for a later fused kernel, not the
/// product hot path — but it chooses its scales where the activations already
/// are, with no host round trip.
public enum DeepSeekV4FP4ActivationReference {
    public static let blockSize = 32

    private static let fp4Maximum: Float = 6
    private static let minimumAbsoluteMaximum = 6 * Float.leastNormalMagnitude

    /// Apply the official orthonormal Hadamard rotation and E2M1 round trip.
    public static func rotateQuantizeDequantize(
        _ input: MLXArray,
        stream: StreamOrDevice = .default,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        try validate(input, requireHadamardWidth: true)
        let scale = Float(1 / Foundation.sqrt(Double(input.shape[input.ndim - 1])))
        let rotated = hadamardTransform(input.asType(.float32), scale: scale, stream: stream)
        return try quantizeDequantize(
            rotated, phaseAccounting: phaseAccounting, diagnostics: diagnostics)
            .asType(input.dtype)
    }

    /// E2M1 round trip without the preceding rotation. Exposed separately so
    /// midpoint and scale selection can be tested independently.
    public static func quantizeDequantize(
        _ input: MLXArray,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        try validate(input, requireHadamardWidth: false)
        let width = input.shape[input.ndim - 1]
        let groups = width / blockSize
        var blockedShape = Array(input.shape.dropLast())
        blockedShape.append(groups)
        blockedShape.append(blockSize)
        let blocked = input.asType(.float32).reshaped(blockedShape)
        // Guard-only, and a full sync — the same shape as the FP8 reference's,
        // and skipped under the same switch.
        if diagnostics.validateFiniteness {
            guard measuringPhase(phaseAccounting?.recordFinitenessSweep(nanoseconds:), {
                isFinite(blocked).all().item(Bool.self)
            }) else {
                throw DeepSeekV4Error.attention(
                    "FP4 activation input contains a non-finite value")
            }
        }

        // What is left inside the bracket is graph construction: the scale is
        // chosen where the maxima are, with no host round trip.
        let scales = measuringPhase(
            phaseAccounting?.recordActivationScaleSync(nanoseconds:)
        ) {
            exactPowerOfTwoScales(
                MLX.abs(blocked).max(axis: -1), blockedShape: blockedShape)
        }
        let normalized = minimum(
            maximum(blocked / scales, MLXArray(-fp4Maximum)),
            MLXArray(fp4Maximum))
        let magnitude = MLX.abs(normalized)

        // These inequalities are the exact E2M1 midpoint policy in the
        // published kernel. Alternating strict and inclusive comparisons is
        // significant: midpoint ties select the representable value with an
        // even low mantissa bit.
        var rounded = MLXArray.zeros(magnitude.shape, dtype: .float32)
        rounded = which(magnitude .> MLXArray(0.25 as Float), MLXArray(0.5 as Float), rounded)
        rounded = which(magnitude .>= MLXArray(0.75 as Float), MLXArray(1 as Float), rounded)
        rounded = which(magnitude .> MLXArray(1.25 as Float), MLXArray(1.5 as Float), rounded)
        rounded = which(magnitude .>= MLXArray(1.75 as Float), MLXArray(2 as Float), rounded)
        rounded = which(magnitude .> MLXArray(2.5 as Float), MLXArray(3 as Float), rounded)
        rounded = which(magnitude .>= MLXArray(3.5 as Float), MLXArray(4 as Float), rounded)
        rounded = which(magnitude .> MLXArray(5 as Float), MLXArray(6 as Float), rounded)

        return (sign(normalized) * rounded * scales)
            .reshaped(input.shape)
            .asType(input.dtype)
    }

    private static func validate(
        _ input: MLXArray,
        requireHadamardWidth: Bool
    ) throws {
        guard input.ndim > 0, let width = input.shape.last, width > 0,
            width.isMultiple(of: blockSize),
            input.dtype.isFloatingPoint, !input.dtype.isComplex
        else {
            throw DeepSeekV4Error.attention(
                "FP4 activation input must be real floating point with a positive last "
                    + "dimension divisible by \(blockSize); got \(input.shape) \(input.dtype)")
        }
        if requireHadamardWidth, !supportsFloat32Hadamard(width) {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "FP4 activation width \(width) is not supported by MLX's float32 Hadamard transform")
        }
    }

    /// MLX supports `m * 2^k`, with m in {1, 12, 20, 28} and 2^k <= 8192
    /// for float32. Check that contract before entering the native operator.
    private static func supportsFloat32Hadamard(_ width: Int) -> Bool {
        for base in [1, 12, 20, 28] where width.isMultiple(of: base) {
            let quotient = width / base
            if quotient > 0, quotient <= 8192,
                quotient.nonzeroBitCount == 1
            {
                return true
            }
        }
        return false
    }

    /// One exact power-of-two scale per group of 32, chosen where the maxima
    /// are.
    ///
    /// The official kernel derives the block scale through IEEE-754 exponent
    /// bits, and ``DeepSeekV4PowerOfTwoScale/ceiling(of:)`` performs exactly
    /// that arithmetic on the GPU, so no log/exp approximation can move an
    /// E2M1 tie and no scale table crosses to the host.
    /// ``hostExactPowerOfTwoScales(_:)`` states the same rule in Swift and a
    /// test holds the two to bit equality.
    static func exactPowerOfTwoScales(
        _ maxima: MLXArray,
        blockedShape: [Int]
    ) -> MLXArray {
        var scaleShape = blockedShape
        scaleShape[scaleShape.count - 1] = 1
        let ratio = maximum(
            maxima.asType(.float32), MLXArray(minimumAbsoluteMaximum))
            / MLXArray(fp4Maximum)
        return DeepSeekV4PowerOfTwoScale.ceiling(of: ratio).reshaped(scaleShape)
    }

    /// The scale rule, one value at a time, in Swift: the derivation this
    /// reference shipped with, kept as the operator's test oracle.
    static func hostExactPowerOfTwoScales(_ values: [Float]) throws -> [Float] {
        var scales = [Float]()
        scales.reserveCapacity(values.count)
        for value in values {
            let absoluteMaximum = Swift.max(value, minimumAbsoluteMaximum)
            let ratio = absoluteMaximum / fp4Maximum
            guard ratio.isFinite, ratio > 0 else {
                throw DeepSeekV4Error.attention(
                    "FP4 activation block contains a non-finite magnitude")
            }
            let bits = ratio.bitPattern
            let storedExponent = Int((bits >> 23) & 0xff)
            let mantissa = bits & 0x7f_ffff
            guard storedExponent > 0, storedExponent < 0xff else {
                throw DeepSeekV4Error.attention(
                    "FP4 activation scale is outside the finite normal Float range")
            }
            let exponent = storedExponent - 127 + (mantissa == 0 ? 0 : 1)
            let scaleExponent = exponent + 127
            guard scaleExponent > 0, scaleExponent < 0xff else {
                throw DeepSeekV4Error.attention(
                    "FP4 activation power-of-two scale is not representable")
            }
            scales.append(Float(bitPattern: UInt32(scaleExponent) << 23))
        }
        return scales
    }
}
