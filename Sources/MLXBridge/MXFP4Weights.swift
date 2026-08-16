import Foundation
import MLX

/// Errors raised when packed MXFP4 operands do not describe a usable tensor.
///
/// Spec §5.8 ("fail explicitly"): an inconsistent block layout is reported with
/// the shapes involved, never silently padded, reshaped, or truncated.
public enum MXFP4BridgeError: Error, CustomStringConvertible {
    case wrongPackedDType(DType)
    case wrongScalesDType(DType)
    case wrongRank(packed: [Int], scales: [Int])
    case shapeMismatch(reason: String)

    public var description: String {
        switch self {
        case .wrongPackedDType(let dtype):
            return """
                packed MXFP4 weights must be .uint32 (the little-endian \
                reinterpretation of the checkpoint's uint8 stream), got \(dtype)
                """
        case .wrongScalesDType(let dtype):
            return "MXFP4 e8m0 scales must be .uint8, got \(dtype)"
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

/// Packed MXFP4 weights in **Kimi-K3 checkpoint byte layout**, ready for MLX.
///
/// ## The byte-layout contract
///
/// This type performs no repacking, and that is its entire point. A
/// `compressed-tensors` `mxfp4-pack-quantized` checkpoint stores, per tensor:
///
/// - `weight_packed`: `uint8[outFeatures, inFeatures / 2]`, two 4-bit e2m1
///   elements per byte, the earlier element in the **low** nibble;
/// - `weight_scale`: `uint8[outFeatures, inFeatures / 32]`, one e8m0 exponent
///   per 32 consecutive input elements.
///
/// MLX wants the packed weights as `uint32[outFeatures, inFeatures / 8]`. That
/// is a *type* difference and not a *layout* difference: MLX's little-endian
/// `uint32` word holds elements 0–7 with element 0 in the low nibble, so its
/// byte image is exactly the checkpoint's `uint8` stream. Measured
/// byte-for-byte in M2; see
/// `docs/experiments/2026-08-08-mxfp4-reference-and-packing.md`.
///
/// The consequence is the one the project cares about: a pager slot holding raw
/// checkpoint bytes can be wrapped as an `MLXArray` and multiplied, with no
/// conversion pass and no extra copy. ``adopting(packedPointer:scalesPointer:experts:outFeatures:inFeatures:packedFinalizer:scalesFinalizer:)``
/// is that path; ``init(packedBytes:scaleBytes:experts:outFeatures:inFeatures:)``
/// is the copying one, for tests and for callers who own their buffers already.
///
/// ## Expert stacks
///
/// A routed-expert stack is the same bytes with a leading axis: expert *e*'s
/// slice is exactly its own `weight_packed` tensor, at `e * outFeatures *
/// inFeatures / 2`. Nothing is interleaved or reordered. See
/// `docs/experiments/2026-08-08-gather-qmm-validation.md`.
public struct MXFP4Weights {

    /// Elements sharing one e8m0 scale. Fixed at 32 by OCP MX for all MX formats.
    public static let groupSize = 32

    /// Bits per packed element.
    public static let bits = 4

    /// `uint32`, `[experts?, outFeatures, inFeatures / 8]`.
    public let packed: MLXArray

    /// `uint8` e8m0, `[experts?, outFeatures, inFeatures / 32]`.
    public let scales: MLXArray

    /// Number of routed experts, or `nil` for a single (non-stacked) matrix.
    public let experts: Int?

    public let outFeatures: Int
    public let inFeatures: Int

    /// Wrap arrays that are already in the layout above, validating that claim.
    public init(packed: MLXArray, scales: MLXArray) throws {
        guard packed.dtype == .uint32 else {
            throw MXFP4BridgeError.wrongPackedDType(packed.dtype)
        }
        guard scales.dtype == .uint8 else {
            throw MXFP4BridgeError.wrongScalesDType(scales.dtype)
        }
        let packedShape = packed.shape
        let scalesShape = scales.shape
        guard packedShape.count == scalesShape.count,
            packedShape.count == 2 || packedShape.count == 3
        else {
            throw MXFP4BridgeError.wrongRank(packed: packedShape, scales: scalesShape)
        }

        let experts = packedShape.count == 3 ? packedShape[0] : nil
        if let experts, scalesShape[0] != experts {
            throw MXFP4BridgeError.shapeMismatch(
                reason: "expert count differs: packed has \(experts), scales has \(scalesShape[0])"
            )
        }
        let outFeatures = packedShape[packedShape.count - 2]
        guard scalesShape[scalesShape.count - 2] == outFeatures else {
            throw MXFP4BridgeError.shapeMismatch(
                reason: """
                    row count differs: packed has \(outFeatures), scales has \
                    \(scalesShape[scalesShape.count - 2])
                    """
            )
        }

        // Eight 4-bit elements per uint32 word.
        let inFeatures = packedShape[packedShape.count - 1] * 8
        let expectedGroups = inFeatures / Self.groupSize
        guard inFeatures % Self.groupSize == 0 else {
            throw MXFP4BridgeError.shapeMismatch(
                reason: """
                    inFeatures \(inFeatures) is not a multiple of groupSize \
                    \(Self.groupSize); partial final groups are not supported
                    """
            )
        }
        guard scalesShape[scalesShape.count - 1] == expectedGroups else {
            throw MXFP4BridgeError.shapeMismatch(
                reason: """
                    group count differs: \(inFeatures) input features need \
                    \(expectedGroups) scales per row, got \
                    \(scalesShape[scalesShape.count - 1])
                    """
            )
        }

        self.packed = packed
        self.scales = scales
        self.experts = experts
        self.outFeatures = outFeatures
        self.inFeatures = inFeatures
    }

    /// Copy checkpoint bytes into MLX-owned arrays.
    ///
    /// `packedBytes` is the `weight_packed` byte stream and `scaleBytes` the
    /// `weight_scale` byte stream, exactly as they appear in the safetensors
    /// file, concatenated in expert order when `experts` is non-`nil`.
    public init(
        packedBytes: UnsafeRawBufferPointer,
        scaleBytes: UnsafeRawBufferPointer,
        experts: Int? = nil,
        outFeatures: Int,
        inFeatures: Int
    ) throws {
        let count = experts ?? 1
        let expectedPacked = count * outFeatures * inFeatures / 2
        let expectedScales = count * outFeatures * inFeatures / Self.groupSize
        guard packedBytes.count == expectedPacked else {
            throw MXFP4BridgeError.shapeMismatch(
                reason: """
                    packed byte count \(packedBytes.count) does not match \
                    \(expectedPacked) implied by the declared shape
                    """
            )
        }
        guard scaleBytes.count == expectedScales else {
            throw MXFP4BridgeError.shapeMismatch(
                reason: """
                    scale byte count \(scaleBytes.count) does not match \
                    \(expectedScales) implied by the declared shape
                    """
            )
        }

        let packedShape =
            experts.map { [$0, outFeatures, inFeatures / 8] } ?? [outFeatures, inFeatures / 8]
        let scalesShape =
            experts.map { [$0, outFeatures, inFeatures / Self.groupSize] }
            ?? [outFeatures, inFeatures / Self.groupSize]

        // The uint32 reinterpretation happens here and nowhere else. It is a
        // reinterpretation of the same little-endian byte image, not a
        // conversion: no arithmetic is performed on the packed bytes anywhere
        // in this file. If that ever stops being true, the no-repack result
        // this milestone rests on has been quietly given up.
        let packed = packedBytes.withMemoryRebound(to: UInt32.self) {
            MLXArray($0, packedShape)
        }
        let scales = MLXArray(scaleBytes.bindMemory(to: UInt8.self), scalesShape)
        try self.init(packed: packed, scales: scales)
    }

    /// Take ownership of page-aligned buffers without copying them.
    ///
    /// This is the pager-facing constructor: the caller allocates (for example
    /// with `posix_memalign`), reads checkpoint bytes into the allocation, and
    /// hands the pointer over. MLX calls the finalizer exactly once when the
    /// array's last reference goes away, which is where the buffer is released.
    ///
    /// Ownership transfers on success. If validation throws, the finalizers are
    /// **not** called by this function and the caller still owns the memory —
    /// but the `MLXArray`s built before the throw own the pointers, so in
    /// practice the caller must treat a throw here as a programming error about
    /// shapes, not a recoverable condition. Validate shapes before allocating.
    public static func adopting(
        packedPointer: UnsafeMutableRawPointer,
        scalesPointer: UnsafeMutableRawPointer,
        experts: Int? = nil,
        outFeatures: Int,
        inFeatures: Int,
        packedFinalizer: @escaping () -> Void,
        scalesFinalizer: @escaping () -> Void
    ) throws -> MXFP4Weights {
        let packedShape =
            experts.map { [$0, outFeatures, inFeatures / 8] } ?? [outFeatures, inFeatures / 8]
        let scalesShape =
            experts.map { [$0, outFeatures, inFeatures / Self.groupSize] }
            ?? [outFeatures, inFeatures / Self.groupSize]

        let packed = MLXArray(
            rawPointer: packedPointer, packedShape, dtype: .uint32, finalizer: packedFinalizer)
        let scales = MLXArray(
            rawPointer: scalesPointer, scalesShape, dtype: .uint8, finalizer: scalesFinalizer)
        return try MXFP4Weights(packed: packed, scales: scales)
    }

    /// Expert `index` of a stack, as a standalone matrix. No copy of the data.
    public func expert(_ index: Int) throws -> MXFP4Weights {
        guard let experts, index >= 0, index < experts else {
            throw MXFP4BridgeError.shapeMismatch(
                reason: "expert \(index) requested from a stack of \(experts.map(String.init) ?? "none")"
            )
        }
        return try MXFP4Weights(packed: packed[index], scales: scales[index])
    }
}
