import MLX

/// Equation-level reference for DeepSeek V4's in-place E4M3 activation pass.
///
/// The official sliding-window attention path applies this operation to the
/// non-rotary key/value channels in blocks of 64. Each row/block chooses a
/// power-of-two scale, rounds the normalized values to finite E4M3, and then
/// immediately dequantizes them. Nothing it computes crosses to the host — the
/// scale table included, since `b7c2092` measured that round trip at 471 syncs
/// and 0.13 s of a decode pass.
///
/// ## Why the rounding is written in exponent bits
///
/// The first version of this reference spelled every step out as its own
/// elementwise operation: a fourteen-step `which` ladder to find each value's
/// binade, then two independent round-to-nearest-even chains (one for the
/// normal grid, one for the subnormal one) selected between at the end. That
/// is 90 MLX primitives per call, and
/// `docs/experiments/2026-08-17-v4-cpu-profile.md` measured 439.5 calls per
/// decode pass — ~39,600 primitives, essentially the whole graph, against
/// 551 ms/pass inside `mlx::core::eval_impl` and `mlx::core::detail::compile`
/// entered exactly zero times.
///
/// ``roundToFiniteE4M3(_:)`` replaces the ladder and both chains with the
/// IEEE-754 arithmetic they were spelling out, in the same spirit as
/// ``DeepSeekV4PowerOfTwoScale``: rounding a float32 magnitude onto the E4M3
/// normal grid *is* round-to-nearest-even on its low 20 mantissa bits, and the
/// subnormal grid is a single exactly-rounded addition. The result is the same
/// value bit for bit — ``spelledOutQuantizeDequantize(_:blockSize:)`` keeps the
/// original chain so a test can hold the two against each other over a dense
/// sweep, exactly as ``hostExactPowerOfTwoScales(_:)`` does for the scale.
public enum DeepSeekV4FP8ActivationReference {
    private static let fp8Maximum: Float = 448
    private static let minimumAbsoluteMaximum: Float = 1e-4
    private static let minimumNormal: Float = 1.0 / 64.0
    private static let subnormalStep: Float = 1.0 / 512.0

    /// E4M3FN keeps three mantissa bits, so a float32 magnitude on the normal
    /// grid is one with its low 20 mantissa bits zero.
    private static let droppedMantissaBits: UInt32 = 20
    /// Round-to-nearest-even on those 20 bits: add `2^19 - 1` plus the lowest
    /// *kept* bit, then truncate. A dropped value below the halfway point
    /// cannot carry, one above it always carries, and one exactly at it carries
    /// only when the kept bit is already 1 — which is the tie-to-even rule.
    private static let roundingBias: UInt32 = (1 << (droppedMantissaBits - 1)) - 1
    private static let keptMantissaMask: UInt32 = ~((1 << droppedMantissaBits) - 1)
    /// `2^(23 - 9)`: the float32 binade whose ulp is exactly the E4M3 subnormal
    /// step, so `(m + magic) - magic` rounds `m` onto multiples of `2^-9` with
    /// the hardware's own round-to-nearest-even and no second rounding. Valid
    /// for `0 <= m < 2^-6`, which is the only range it is applied to.
    private static let subnormalRoundingMagic: Float = 16384

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
            phaseAccounting?.recordEval(.finitenessSweep)
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
        return roundAndDequantize(blocked: blocked, scale: scale)
            .reshaped(input.shape)
            .asType(input.dtype)
    }

    /// Everything after the block scale is chosen: clamp into range, round onto
    /// the E4M3FN grid, restore the sign, and undo the scale.
    ///
    /// Named rather than inlined at the call site because it is the exact
    /// subgraph an `MLX.compile` fusion would have to reproduce:
    /// `DeepSeekV4FP8RoundingTests` compiles *this function* and holds the
    /// resulting fused kernel to the uncompiled chain's bits.
    static func roundAndDequantize(blocked: MLXArray, scale: MLXArray) -> MLXArray {
        let normalized = minimum(
            maximum(blocked / scale, MLXArray(-fp8Maximum)),
            MLXArray(fp8Maximum))
        let magnitude = MLX.abs(normalized)
        let quantizedMagnitude = minimum(
            roundToFiniteE4M3(magnitude), MLXArray(fp8Maximum))
        return sign(normalized) * quantizedMagnitude * scale
    }

    /// One non-negative float32 magnitude per element, rounded to the nearest
    /// E4M3FN value, ties to even, in the same float32 dtype.
    ///
    /// Both grids are stated as exact integer or exactly-rounded float
    /// arithmetic, because neither may move a tie:
    ///
    /// - **Normal**, `m >= 2^-6`: E4M3 keeps three mantissa bits, so the grid
    ///   step inside a binade is `2^(e-3)` and rounding to it is
    ///   round-to-nearest-even on the low 20 bits of the float32 significand.
    ///   The bias-and-truncate below is that rule, and a carry out of the
    ///   significand walks into the exponent field on its own — which is
    ///   exactly what rounding `15.9` steps up to the next binade must do.
    /// - **Subnormal**, `m < 2^-6`: the grid is the multiples of `2^-9`, and
    ///   `(m + 2^14) - 2^14` lands on them. `2^14`'s binade has ulp `2^-9`, so
    ///   the addition is a single IEEE round-to-nearest-even onto the grid and
    ///   the subtraction is exact. The `which` discards whichever branch did
    ///   not apply, so the normal branch's carries into a subnormal input, and
    ///   the magic addition's behaviour above `2^-6`, are never read.
    ///
    /// A non-finite magnitude passes through: `+inf`'s bias-and-truncate is
    /// `+inf` again and a quiet NaN keeps its payload, so a corrupted block
    /// still reaches the output checks that refuse it rather than a finite
    /// value that would launder it.
    ///
    /// - Warning: the subnormal branch is `(m + c) - c`, which is exactly the
    ///   shape an optimiser permitted to assume real arithmetic may cancel to
    ///   `m`. Separate MLX primitives are separate Metal kernels and cannot be
    ///   cancelled across; a *fused* kernel — `MLX.compile`, or a hand-written
    ///   one — can be. Anything that fuses this subgraph must re-run
    ///   `DeepSeekV4FP8RoundingTests` before it is believed.
    static func roundToFiniteE4M3(_ magnitude: MLXArray) -> MLXArray {
        let bits = magnitude.view(dtype: .uint32)
        let keptLowBit =
            rightShift(bits, MLXArray(droppedMantissaBits)) & MLXArray(UInt32(1))
        let normal = ((bits + keptLowBit) + MLXArray(roundingBias))
            & MLXArray(keptMantissaMask)
        let magic = MLXArray(subnormalRoundingMagic)
        let subnormal = (magnitude + magic) - magic
        return which(
            magnitude .< MLXArray(minimumNormal),
            subnormal,
            normal.view(dtype: .float32))
    }

    /// The rounding above, spelled out one elementwise step at a time.
    ///
    /// This is the chain this reference shipped with and
    /// ``roundToFiniteE4M3(_:)`` reproduces: a `which` ladder over every
    /// possible binade, and an independent round-to-nearest-even chain per
    /// grid. It is kept so a test can state the rule without the bit
    /// arithmetic and compare bit patterns, not because a pass runs it — at 90
    /// MLX primitives per call against 41 it is the reason this file was
    /// rewritten.
    static func spelledOutQuantizeDequantize(
        _ input: MLXArray,
        blockSize: Int
    ) throws -> MLXArray {
        let groups = input.shape[input.ndim - 1] / blockSize
        var blockedShape = Array(input.shape.dropLast())
        blockedShape.append(groups)
        blockedShape.append(blockSize)
        let blocked = input.asType(.float32).reshaped(blockedShape)
        let scale = exactPowerOfTwoScales(
            MLX.abs(blocked).max(axis: -1), blockedShape: blockedShape)
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
