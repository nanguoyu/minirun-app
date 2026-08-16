import Foundation
import MLX

/// Errors raised when packed affine operands do not describe a usable tensor.
///
/// Spec §5.8 ("fail explicitly"): an inconsistent block layout is reported with
/// the shapes involved, never silently padded, reshaped, or truncated.
public enum AffineBridgeError: Error, CustomStringConvertible {
    case wrongPackedDType(DType)
    case wrongScaleDType(DType)
    case scaleBiasDTypeMismatch(scales: DType, biases: DType)
    case wrongRank(packed: [Int], scales: [Int])
    case shapeMismatch(reason: String)

    public var description: String {
        switch self {
        case .wrongPackedDType(let dtype):
            return """
                packed affine weights must be .uint32 (the little-endian \
                reinterpretation of the checkpoint's stream), got \(dtype)
                """
        case .wrongScaleDType(let dtype):
            return """
                affine scales must be a floating-point type — .bfloat16 for this \
                checkpoint, .float16 or .float32 for others — got \(dtype)
                """
        case .scaleBiasDTypeMismatch(let scales, let biases):
            return "scales are \(scales) but biases are \(biases); MLX emits both in one dtype"
        case .wrongRank(let packed, let scales):
            return """
                packed \(packed) and scales \(scales) must both be rank 2 \
                (one matrix) or rank 3 (an expert stack), and must agree
                """
        case .shapeMismatch(let reason):
            return reason
        }
    }
}

/// Packed **MLX affine** weights, ready for MLX without repacking.
///
/// ## Affine is not MXFP4, in exactly two ways
///
/// `w = scale * q + bias`. So there is a bias per group as well as a scale, and
/// both are ordinary floating-point numbers rather than MX's e8m0 exponent byte.
/// Everything else — the packed `uint32` stream, the row-major
/// `[outFeatures, inFeatures]` orientation, the fact that no repacking happens —
/// is the same as ``MXFP4Weights``, and the two types stay separate rather than
/// merging behind an optional bias because a missing bias is not a variant of
/// affine, it is a different format (spec §10.2: dispatch on tensor metadata).
///
/// ## The scale dtype is read, never assumed
///
/// `mx.quantize(mode: .affine)` returns scales and biases in the *source
/// tensor's* dtype. `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` is a
/// bfloat16 model, so its scales are BF16 — an earlier draft of the tile
/// container assumed float16 and was corrected by measurement. This type
/// therefore accepts any floating-point scale dtype and requires only that
/// scales and biases agree with each other.
///
/// ## Group size 64, not 32
///
/// MX fixes the group at 32; affine does not, and this checkpoint uses 64. It is
/// an instance property rather than a static constant for that reason.
public struct AffineWeights {

    /// `uint32`, `[experts?, outFeatures, inFeatures * bits / 32]`.
    public let packed: MLXArray
    /// `[experts?, outFeatures, inFeatures / groupSize]`, floating point.
    public let scales: MLXArray
    /// Same shape and dtype as ``scales``.
    public let biases: MLXArray

    public let groupSize: Int
    public let bits: Int

    /// Number of routed experts, or `nil` for a single (non-stacked) matrix.
    public let experts: Int?

    public let outFeatures: Int
    public let inFeatures: Int

    /// Wrap arrays that are already in the layout above, validating that claim.
    public init(packed: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int)
        throws
    {
        guard packed.dtype == .uint32 else {
            throw AffineBridgeError.wrongPackedDType(packed.dtype)
        }
        guard scales.dtype == .bfloat16 || scales.dtype == .float16 || scales.dtype == .float32
        else {
            throw AffineBridgeError.wrongScaleDType(scales.dtype)
        }
        guard scales.dtype == biases.dtype else {
            throw AffineBridgeError.scaleBiasDTypeMismatch(
                scales: scales.dtype, biases: biases.dtype)
        }
        let packedShape = packed.shape
        let scalesShape = scales.shape
        guard scalesShape == biases.shape else {
            throw AffineBridgeError.shapeMismatch(
                reason: "scales \(scalesShape) and biases \(biases.shape) must have one shape")
        }
        guard packedShape.count == scalesShape.count,
            packedShape.count == 2 || packedShape.count == 3
        else {
            throw AffineBridgeError.wrongRank(packed: packedShape, scales: scalesShape)
        }
        guard bits > 0, 32 % bits == 0 else {
            throw AffineBridgeError.shapeMismatch(
                reason: "bits \(bits) must divide 32 for the uint32 packing")
        }
        guard groupSize > 0 else {
            throw AffineBridgeError.shapeMismatch(reason: "group size must be positive")
        }

        let experts = packedShape.count == 3 ? packedShape[0] : nil
        if let experts, scalesShape[0] != experts {
            throw AffineBridgeError.shapeMismatch(
                reason: "expert count differs: packed has \(experts), scales has \(scalesShape[0])")
        }
        let outFeatures = packedShape[packedShape.count - 2]
        guard scalesShape[scalesShape.count - 2] == outFeatures else {
            throw AffineBridgeError.shapeMismatch(
                reason: """
                    row count differs: packed has \(outFeatures), scales has \
                    \(scalesShape[scalesShape.count - 2])
                    """)
        }

        let inFeatures = packedShape[packedShape.count - 1] * (32 / bits)
        guard inFeatures % groupSize == 0 else {
            throw AffineBridgeError.shapeMismatch(
                reason: """
                    inFeatures \(inFeatures) is not a multiple of the group size \
                    \(groupSize); partial final groups are not supported
                    """)
        }
        let expectedGroups = inFeatures / groupSize
        guard scalesShape[scalesShape.count - 1] == expectedGroups else {
            throw AffineBridgeError.shapeMismatch(
                reason: """
                    group count differs: \(inFeatures) input features at group size \
                    \(groupSize) need \(expectedGroups) scales per row, got \
                    \(scalesShape[scalesShape.count - 1])
                    """)
        }

        self.packed = packed
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
        self.bits = bits
        self.experts = experts
        self.outFeatures = outFeatures
        self.inFeatures = inFeatures
    }

    /// Take ownership of page-aligned buffers without copying them.
    ///
    /// This is the pager-facing constructor and the reason the tile container
    /// aligns all three sub-regions independently: the packed, scale and bias
    /// arrays are three adoptions out of **one** slot holding one whole expert,
    /// and MLX wraps an adopted pointer for the Metal backend, which requires
    /// page-aligned memory.
    ///
    /// Ownership transfers on success. If validation throws, the `MLXArray`s
    /// built before the throw own the pointers, so a throw here is a programming
    /// error about shapes rather than a recoverable condition — validate shapes
    /// before allocating.
    public static func adopting(
        packedPointer: UnsafeMutableRawPointer,
        scalesPointer: UnsafeMutableRawPointer,
        biasesPointer: UnsafeMutableRawPointer,
        scaleDType: DType,
        experts: Int? = nil,
        outFeatures: Int,
        inFeatures: Int,
        groupSize: Int,
        bits: Int,
        packedFinalizer: @escaping () -> Void,
        scalesFinalizer: @escaping () -> Void,
        biasesFinalizer: @escaping () -> Void
    ) throws -> AffineWeights {
        let words = inFeatures * bits / 32
        let groups = inFeatures / groupSize
        let packedShape = experts.map { [$0, outFeatures, words] } ?? [outFeatures, words]
        let scalesShape = experts.map { [$0, outFeatures, groups] } ?? [outFeatures, groups]

        let packed = MLXArray(
            rawPointer: packedPointer, packedShape, dtype: .uint32, finalizer: packedFinalizer)
        let scales = MLXArray(
            rawPointer: scalesPointer, scalesShape, dtype: scaleDType, finalizer: scalesFinalizer)
        let biases = MLXArray(
            rawPointer: biasesPointer, scalesShape, dtype: scaleDType, finalizer: biasesFinalizer)
        return try AffineWeights(
            packed: packed, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
    }

    /// Expert `index` as a **stack of one**, keeping the leading axis.
    ///
    /// This is what a gather wants: `gatherQuantizedMM` indexes into an expert
    /// axis, so a single expert has to still have one. ``expert(_:)`` drops it
    /// and produces an operand for ``affineMM`` instead — the two are not
    /// interchangeable, and the precondition in ``affineGatherMM`` exists
    /// because the first version of the resident backend passed the wrong one.
    public func expertStack(_ index: Int) throws -> AffineWeights {
        guard let experts, index >= 0, index < experts else {
            throw AffineBridgeError.shapeMismatch(
                reason: "expert \(index) requested from a stack of \(experts.map(String.init) ?? "none")"
            )
        }
        let range = index..<(index + 1)
        return try AffineWeights(
            packed: packed[range], scales: scales[range], biases: biases[range],
            groupSize: groupSize, bits: bits)
    }

    /// Expert `index` of a stack, as a standalone matrix. No copy of the data.
    public func expert(_ index: Int) throws -> AffineWeights {
        guard let experts, index >= 0, index < experts else {
            throw AffineBridgeError.shapeMismatch(
                reason: "expert \(index) requested from a stack of \(experts.map(String.init) ?? "none")"
            )
        }
        return try AffineWeights(
            packed: packed[index], scales: scales[index], biases: biases[index],
            groupSize: groupSize, bits: bits)
    }
}

/// Dense affine matmul: `y = x · W^T`, with `W` in checkpoint layout.
///
/// `transpose` is not a parameter, for the same reason ``mxfp4MM`` has none: the
/// checkpoint stores `[outFeatures, inFeatures]`, so `transpose: true` is the
/// only correct reading, and `transpose: false` would additionally place the
/// group boundaries along the wrong axis. Exposing it would only make a wrong
/// call expressible.
public func affineMM(
    _ x: MLXArray,
    _ weights: AffineWeights,
    stream: StreamOrDevice = .default
) -> MLXArray {
    quantizedMM(
        x,
        weights.packed,
        scales: weights.scales,
        biases: weights.biases,
        transpose: true,
        groupSize: weights.groupSize,
        bits: weights.bits,
        mode: .affine,
        stream: stream)
}

/// Routed-expert affine matmul: `out[t, j, :] = W[rhsIndices[t, j]] · x[t, :]`.
///
/// - Parameters:
///   - x: `[tokens, inFeatures]` activations, one row per token.
///   - weights: a stacked ``AffineWeights`` (`experts != nil`).
///   - rhsIndices: `[tokens, expertsPerToken]` expert ids. Need not be sorted,
///     and may repeat within a row.
/// - Returns: `[tokens, expertsPerToken, outFeatures]`.
///
/// ## Why this wrapper reshapes for you
///
/// MLX's `gatherQuantizedMatmul` reads `x`'s last two axes as `(M, inFeatures)`
/// and *broadcasts everything before them* against `rhsIndices`' shape, so `x`
/// must be presented as `[tokens, 1, 1, inFeatures]`. Passing
/// `[tokens, 1, inFeatures]` instead is a legal broadcast whenever
/// `tokens == expertsPerToken`, produces an identical output shape, and silently
/// computes `W[rhsIndices[t, j]] · x[j]` — the slot index used as the token
/// index. M3 measured that mistake at ~1.5 relative to the output scale against
/// a 1e-5 tolerance. Callers here cannot express it.
///
/// ## `sortedIndices`
///
/// Left to the caller and defaulted off. Note that the mlx-lm reference *does*
/// sort: `SwitchGLU.__call__` sets `do_sort = indices.size >= 64`, so a prompt of
/// eight or more tokens takes a different path there than a single decode step
/// does. That is a property of the reference worth knowing when comparing
/// prefill against decode, not a correctness requirement here.
public func affineGatherMM(
    _ x: MLXArray,
    _ weights: AffineWeights,
    rhsIndices: MLXArray,
    sortedIndices: Bool = false,
    stream: StreamOrDevice = .default
) -> MLXArray {
    precondition(
        weights.experts != nil,
        "affineGatherMM needs a stacked AffineWeights; use affineMM for a single matrix")
    precondition(
        x.ndim == 2,
        "x must be [tokens, inFeatures]; this function inserts the broadcast axes itself")
    precondition(
        rhsIndices.ndim == 2 && rhsIndices.shape[0] == x.shape[0],
        "rhsIndices must be [tokens, expertsPerToken] matching x's token count")

    let tokens = x.shape[0]
    let expertsPerToken = rhsIndices.shape[1]

    let result = gatherQuantizedMM(
        x.reshaped([tokens, 1, 1, weights.inFeatures]),
        weights.packed,
        scales: weights.scales,
        biases: weights.biases,
        lhsIndices: nil,
        rhsIndices: rhsIndices,
        transpose: true,
        groupSize: weights.groupSize,
        bits: weights.bits,
        mode: .affine,
        sortedIndices: sortedIndices,
        stream: stream)

    return result.reshaped([tokens, expertsPerToken, weights.outFeatures])
}

/// Dequantize packed affine weights to a dense array. Validation and debugging
/// only — the whole point of the project is not to do this on the hot path.
///
/// Unlike MXFP4, this is **not** lossless into float32: an affine product is
/// `scale * q + bias` with a bfloat16 scale, so the result carries the source
/// dtype's rounding. Tests that compare against a dequantized reference must use
/// a tolerance, not bit-equality — which is the opposite of the M2 situation and
/// worth stating where someone will read it.
public func affineDequantize(
    _ weights: AffineWeights,
    dtype: DType = .float32,
    stream: StreamOrDevice = .default
) -> MLXArray {
    dequantized(
        weights.packed,
        scales: weights.scales,
        biases: weights.biases,
        groupSize: weights.groupSize,
        bits: weights.bits,
        mode: .affine,
        dtype: dtype,
        stream: stream)
}
