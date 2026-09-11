import MLX
import MLXBridge

/// DeepSeek V4.1 Flash's dense projections: block-FP8 at `[32, 32]`.
///
/// V4.1's `inference/model.py` sets `fp8_block_size = 32` and its
/// `quantization_config` states `weight_block_size [32, 32]`, against V4's 128.
/// That is the *only* difference in the linear path: the same E4M3FN codes, the
/// same `ue8m0` power-of-two activation scale, the same
/// `act_quant` → `fp8_gemm` order. So this type is a name for one set of
/// arguments to ``DeepSeekV4BlockFP8LinearReference`` and nothing else, and the
/// test beside it asserts that V4's own 128 fixtures still come out bit for bit
/// through the parameterised path.
///
/// ## Where this is not the published kernel, exactly
///
/// `fp8_gemm` accumulates `sum_k(a_code * w_code)` in float32 over one 32-wide
/// K block and multiplies that block's two scales in afterwards; the block
/// results are then summed. MLX's `.mxfp8` quantized matmul dequantizes the
/// weight and multiplies element by element instead. Same equation, different
/// place for the scale, and therefore a different float32 rounding — the same
/// class of difference ADR 0018 records for batched heads. The output is
/// bfloat16, which absorbs most of it.
/// `docs/experiments/2026-09-11-v41-phase1-dense.md` carries the measured gap
/// against the torch transcription.
public enum DeepSeekV41FP8Linear {
    /// `fp8_block_size` in V4.1's `inference/model.py`.
    public static let activationBlockSize = 32
    /// `weight_block_size` in its `quantization_config`, both dimensions.
    public static let weightBlockSize = 32

    /// `linear()` for an fp8 weight: quantize the activation, then the GEMM.
    public static func project(
        _ input: MLXArray,
        weights: BlockFP8Weights,
        stream: StreamOrDevice = .default,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        try DeepSeekV4BlockFP8LinearReference.project(
            input,
            weights: weights,
            activationBlockSize: activationBlockSize,
            weightBlockRows: weightBlockSize,
            weightBlockColumns: weightBlockSize,
            stream: stream,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
    }

    /// One FP8 matrix decoded to the bfloat16 the reference holds.
    ///
    /// `attn.wo_a` is the reason this exists. DeepSeek's `convert.py`
    /// dequantizes it, so `inference/model.py` declares it
    /// `Linear(..., dtype=torch.bfloat16)` and reads `.weight` directly through
    /// an `einsum` — there is no activation quantization on that path at all.
    /// Our container kept the published FP8 bytes, which is the more faithful
    /// thing to have carried; reaching the reference's tensor from them is this
    /// one call.
    public static func dequantizedToBFloat16(
        _ weights: BlockFP8Weights,
        stream: StreamOrDevice = .default
    ) throws -> MLXArray {
        guard weights.scaleBlockRows == weightBlockSize,
            weights.scaleBlockColumns == weightBlockSize
        else {
            throw DeepSeekV41Error.configuration(
                "V4.1 block-FP8 weights must use a 32x32 scale grid; got "
                    + "\(weights.scaleBlockRows)x\(weights.scaleBlockColumns)")
        }
        return blockFP8Dequantize(weights, dtype: .bfloat16, stream: stream)
    }

    /// `act_quant(x, 32, 'ue8m0', e8m0, inplace=True)`.
    ///
    /// The sliding-window K goes through this and stays in it: the ring buffer
    /// holds the *dequantized* values, over the whole post-RoPE vector,
    /// rotary tail included.
    public static func quantizeDequantizeActivation(
        _ input: MLXArray,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        try DeepSeekV4FP8ActivationReference.quantizeDequantize(
            input, blockSize: activationBlockSize,
            phaseAccounting: phaseAccounting, diagnostics: diagnostics)
    }
}

/// The two FP4 round trips V4.1 takes, which differ in more than their width.
///
/// `fp4_act_quant` in `inference/kernel.py` has two scale rules behind one
/// name, and the caller picks by passing a scale dtype:
///
/// - **E8M0, groups of 32** — the indexer's query and its index keys. The block
///   maximum is floored at `6 * 2^-126` and the scale rounded *up* to a power of
///   two, exactly like the FP8 activation scale. ``DeepSeekV4FP4ActivationReference``
///   already states this rule, and V4.1 uses it unrotated (there is no Walsh–
///   Hadamard transform in V4.1's indexer), so that operator's
///   `quantizeDequantize` is the whole of it.
/// - **E4M3, groups of 16** — the compressed KV latent. The block maximum is
///   floored at `6 * 2^-9`, so an all-zero group still gets a nonzero scale, and
///   the scale is `amax / 6` **rounded to the nearest E4M3 value**, not to a
///   power of two. That is a genuinely different rule and it is why this type
///   exists rather than another argument on the V4 one.
public enum DeepSeekV41FP4Activation {
    private static let fp4Maximum: Float = 6
    /// `6 * (2 ** -9)` in the kernel: the floor that keeps a zero group's scale
    /// nonzero, which the kernel attributes to training's compressed KV.
    private static let e4m3MinimumAbsoluteMaximum: Float = 6 * 0x1p-9

    /// The indexer's round trip: E8M0 scales over groups of 32, no rotation.
    public static func indexerQuantizeDequantize(
        _ input: MLXArray,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        try DeepSeekV4FP4ActivationReference.quantizeDequantize(
            input, phaseAccounting: phaseAccounting, diagnostics: diagnostics)
    }

    /// The compressed KV latent's round trip: E4M3 scales over groups of 16.
    public static func compressedKeyValueQuantizeDequantize(
        _ input: MLXArray,
        blockSize: Int = 16
    ) throws -> MLXArray {
        guard input.ndim > 0, let width = input.shape.last, width > 0,
            blockSize > 0, width.isMultiple(of: blockSize),
            input.dtype.isFloatingPoint, !input.dtype.isComplex
        else {
            throw DeepSeekV41Error.configuration(
                "compressed-KV FP4 input must be real floating point with a last dimension "
                    + "divisible by \(blockSize); got \(input.shape) \(input.dtype)")
        }
        let groups = width / blockSize
        var blockedShape = Array(input.shape.dropLast())
        blockedShape.append(groups)
        blockedShape.append(blockSize)
        let blocked = input.asType(.float32).reshaped(blockedShape)

        var scaleShape = blockedShape
        scaleShape[scaleShape.count - 1] = 1
        let maxima = maximum(
            MLX.abs(blocked).max(axis: -1),
            MLXArray(e4m3MinimumAbsoluteMaximum))
        // `T.Cast(FP8, amax / fp4_max)`, and no power-of-two rounding: the scale
        // is the E4M3 value nearest the ratio, so it needs the same
        // round-to-nearest-even the FP8 activation reference states.
        let scale = DeepSeekV4FP8ActivationReference
            .roundToFiniteE4M3(maxima / MLXArray(fp4Maximum))
            .reshaped(scaleShape)

        let normalized = minimum(
            maximum(blocked / scale, MLXArray(-fp4Maximum)),
            MLXArray(fp4Maximum))
        return (roundToE2M1(normalized) * scale)
            .reshaped(input.shape)
            .asType(input.dtype)
    }

    /// E2M1 with ties to even, as ``DeepSeekV4FP4ActivationReference`` states
    /// it: alternating strict and inclusive comparisons, so a midpoint selects
    /// the neighbour whose low mantissa bit is zero.
    static func roundToE2M1(_ normalized: MLXArray) -> MLXArray {
        let magnitude = MLX.abs(normalized)
        var rounded = MLXArray.zeros(magnitude.shape, dtype: .float32)
        rounded = which(magnitude .> MLXArray(0.25 as Float), MLXArray(0.5 as Float), rounded)
        rounded = which(magnitude .>= MLXArray(0.75 as Float), MLXArray(1 as Float), rounded)
        rounded = which(magnitude .> MLXArray(1.25 as Float), MLXArray(1.5 as Float), rounded)
        rounded = which(magnitude .>= MLXArray(1.75 as Float), MLXArray(2 as Float), rounded)
        rounded = which(magnitude .> MLXArray(2.5 as Float), MLXArray(3 as Float), rounded)
        rounded = which(magnitude .>= MLXArray(3.5 as Float), MLXArray(4 as Float), rounded)
        rounded = which(magnitude .> MLXArray(5 as Float), MLXArray(6 as Float), rounded)
        return sign(normalized) * rounded
    }
}
