import Foundation
import MLX

/// Named failures at the boundary between a V4 block-FP8 container and MLX.
///
/// DeepSeek V4 stores one E8M0 scale per 128×128 block. MLX 0.31.4's
/// `.mxfp8` operator consumes one E8M0 scale per 32 adjacent columns of each
/// row. ``BlockFP8Weights`` bridges those layouts by repeating each compact
/// block scale across the rows and four 32-column groups it governs. Weight
/// bytes are never reordered or converted.
public enum BlockFP8BridgeError: Error, CustomStringConvertible, Equatable {
    case invalidGeometry(String)
    case byteCount(name: String, actual: Int, expected: Int)
    case nonFiniteWeight(byteOffset: Int, byte: UInt8)
    case nonFiniteScale(byteOffset: Int)
    case activation(String)

    public var description: String {
        switch self {
        case .invalidGeometry(let detail):
            return "invalid block-FP8 geometry: \(detail)"
        case .byteCount(let name, let actual, let expected):
            return "\(name) has \(actual) bytes; geometry requires \(expected)"
        case .nonFiniteWeight(let offset, let byte):
            return String(
                format: "block-FP8 weight byte %d is non-finite E4M3FN (0x%02x)",
                offset, byte)
        case .nonFiniteScale(let offset):
            return "block-FP8 scale byte \(offset) is the non-finite E8M0 code 255"
        case .activation(let detail):
            return "invalid block-FP8 activation: \(detail)"
        }
    }
}

/// A finite DeepSeek-V4 block-FP8 matrix in the byte layout MLX consumes.
///
/// The checkpoint stores `uint8[outFeatures, inFeatures]` E4M3FN weights and
/// `uint8[outFeatures / scaleBlockRows, inFeatures / scaleBlockColumns]` E8M0
/// scales. MLX stores four 8-bit weights in each little-endian `uint32`, so the
/// packed weight byte image is already exact. Only the much smaller scale grid
/// is expanded, from one scale per 128×128 block to the equivalent one scale
/// per row and 32 columns accepted by MLX's `.mxfp8` kernel.
///
/// Non-finite encodings are rejected before an MLX graph is constructed.
/// MLX's current E4M3 decoder saturates `0x7f`/`0xff` instead of preserving the
/// checkpoint format's NaN meaning; accepting those bytes would therefore be a
/// silent numerical change.
public struct BlockFP8Weights {
    /// MLX's MX scale grouping. A 128-column V4 scale is repeated four times.
    public static let mlxGroupSize = 32
    public static let bits = 8

    /// Whether ``adopting(packedPointer:packedByteCount:compactScaleBytes:outFeatures:inFeatures:scaleBlockRows:scaleBlockColumns:packedFiniteness:finalizer:)``
    /// derives the packed matrix's finiteness itself, or takes it as a property
    /// the caller already established for these exact bytes.
    ///
    /// The scan is not an integrity check and never was — the container's
    /// recorded SHA-256 is that, under the reader's own tile digest policy. It
    /// is a *format* precondition: MLX's E4M3 decoder saturates `0x7f`/`0xff` rather
    /// than preserving the checkpoint's NaN/Inf meaning, so those codes must not
    /// reach it. Finiteness is therefore a pure function of the byte string, and
    /// a caller that knows it is handing over the same byte string it already
    /// scanned in this run is re-deriving an answer it holds.
    ///
    /// The default at every declaration site is ``scan``, for the reason
    /// `TileDigestPolicy.verifyEveryLoad` is the default in the layer above: a
    /// library cannot invent trust. Only a caller that both scanned these bytes during this run
    /// *and* holds verification authority binding the file to that scan may say
    /// otherwise, and a wrong statement here is unchecked.
    public enum PackedFinitenessCheck: Sendable, Equatable {
        /// Scan the packed bytes. The only value a caller may assume for itself.
        case scan
        /// The caller established these exact bytes are finite E4M3FN earlier in
        /// this run, and holds the authority that makes them the same bytes.
        case establishedForTheseBytes
    }

    /// `uint32[outFeatures, inFeatures / 4]`, with the checkpoint byte image.
    public let packed: MLXArray
    /// `uint8[outFeatures, inFeatures / 32]`, expanded from the compact grid.
    public let expandedScales: MLXArray

    public let outFeatures: Int
    public let inFeatures: Int
    public let scaleBlockRows: Int
    public let scaleBlockColumns: Int
    public let compactScaleBytes: Int
    public let expandedScaleBytes: Int

    /// Copy checked checkpoint bytes into MLX-owned arrays.
    public init(
        packedBytes: UnsafeRawBufferPointer,
        compactScaleBytes: UnsafeRawBufferPointer,
        outFeatures: Int,
        inFeatures: Int,
        scaleBlockRows: Int,
        scaleBlockColumns: Int
    ) throws {
        let geometry = try Self.validateGeometry(
            outFeatures: outFeatures,
            inFeatures: inFeatures,
            scaleBlockRows: scaleBlockRows,
            scaleBlockColumns: scaleBlockColumns)
        guard packedBytes.count == geometry.packedBytes else {
            throw BlockFP8BridgeError.byteCount(
                name: "packed weights", actual: packedBytes.count,
                expected: geometry.packedBytes)
        }
        guard compactScaleBytes.count == geometry.compactScaleBytes else {
            throw BlockFP8BridgeError.byteCount(
                name: "compact scales", actual: compactScaleBytes.count,
                expected: geometry.compactScaleBytes)
        }
        try Self.validateFiniteWeights(packedBytes)
        let expanded = try Self.expandScales(
            compactScaleBytes,
            scaleRows: geometry.scaleRows,
            scaleColumns: geometry.scaleColumns,
            scaleBlockRows: scaleBlockRows,
            repetitionsPerBlockColumn: geometry.repetitionsPerBlockColumn,
            outputColumns: geometry.expandedScaleColumns,
            outputByteCount: geometry.expandedScaleBytes)

        // Copy the little-endian byte image into aligned UInt32 storage. This
        // is a type reinterpretation, not a repack: byte 0 remains byte 0.
        var words = [UInt32](repeating: 0, count: geometry.packedBytes / 4)
        words.withUnsafeMutableBytes { destination in
            destination.copyMemory(from: packedBytes)
        }

        self.init(
            packed: MLXArray(words).reshaped([outFeatures, inFeatures / 4]),
            expandedScales: MLXArray(expanded).reshaped([
                outFeatures, inFeatures / Self.mlxGroupSize,
            ]),
            outFeatures: outFeatures,
            inFeatures: inFeatures,
            scaleBlockRows: scaleBlockRows,
            scaleBlockColumns: scaleBlockColumns,
            compactScaleBytes: geometry.compactScaleBytes,
            expandedScaleBytes: expanded.count)
    }

    /// Adopt the packed matrix without copying it; copy and expand only scales.
    ///
    /// Ownership of `packedPointer` transfers only after every validation has
    /// succeeded. On a thrown error the caller still owns it and `finalizer` is
    /// not invoked. A pager can therefore give this function the packed start
    /// of one held tile and release the whole tile from `finalizer` after MLX
    /// has completed the graph that uses it.
    ///
    /// `packedFiniteness` defaults to ``PackedFinitenessCheck/scan``; see that
    /// type for what a caller must hold before saying anything else.
    public static func adopting(
        packedPointer: UnsafeMutableRawPointer,
        packedByteCount: Int,
        compactScaleBytes: UnsafeRawBufferPointer,
        outFeatures: Int,
        inFeatures: Int,
        scaleBlockRows: Int,
        scaleBlockColumns: Int,
        packedFiniteness: PackedFinitenessCheck = .scan,
        finalizer: @escaping () -> Void
    ) throws -> BlockFP8Weights {
        let geometry = try validateGeometry(
            outFeatures: outFeatures,
            inFeatures: inFeatures,
            scaleBlockRows: scaleBlockRows,
            scaleBlockColumns: scaleBlockColumns)
        guard packedByteCount == geometry.packedBytes else {
            throw BlockFP8BridgeError.byteCount(
                name: "packed weights", actual: packedByteCount,
                expected: geometry.packedBytes)
        }
        guard compactScaleBytes.count == geometry.compactScaleBytes else {
            throw BlockFP8BridgeError.byteCount(
                name: "compact scales", actual: compactScaleBytes.count,
                expected: geometry.compactScaleBytes)
        }
        if packedFiniteness == .scan {
            try validateFiniteWeights(
                UnsafeRawBufferPointer(start: packedPointer, count: packedByteCount))
        }
        let expanded = try expandScales(
            compactScaleBytes,
            scaleRows: geometry.scaleRows,
            scaleColumns: geometry.scaleColumns,
            scaleBlockRows: scaleBlockRows,
            repetitionsPerBlockColumn: geometry.repetitionsPerBlockColumn,
            outputColumns: geometry.expandedScaleColumns,
            outputByteCount: geometry.expandedScaleBytes)

        // There are no throwing operations after ownership moves into MLX.
        let packed = MLXArray(
            rawPointer: packedPointer,
            [outFeatures, inFeatures / 4],
            dtype: .uint32,
            finalizer: finalizer)
        return BlockFP8Weights(
            packed: packed,
            expandedScales: MLXArray(expanded).reshaped([
                outFeatures, inFeatures / Self.mlxGroupSize,
            ]),
            outFeatures: outFeatures,
            inFeatures: inFeatures,
            scaleBlockRows: scaleBlockRows,
            scaleBlockColumns: scaleBlockColumns,
            compactScaleBytes: geometry.compactScaleBytes,
            expandedScaleBytes: expanded.count)
    }

    private init(
        packed: MLXArray,
        expandedScales: MLXArray,
        outFeatures: Int,
        inFeatures: Int,
        scaleBlockRows: Int,
        scaleBlockColumns: Int,
        compactScaleBytes: Int,
        expandedScaleBytes: Int
    ) {
        self.packed = packed
        self.expandedScales = expandedScales
        self.outFeatures = outFeatures
        self.inFeatures = inFeatures
        self.scaleBlockRows = scaleBlockRows
        self.scaleBlockColumns = scaleBlockColumns
        self.compactScaleBytes = compactScaleBytes
        self.expandedScaleBytes = expandedScaleBytes
    }

    /// A zero-copy logical view over a contiguous set of output rows.
    ///
    /// V4's grouped output projection stores all groups in one matrix but
    /// applies each consecutive row band to a different activation group. A
    /// row view lets that equation use the original packed bytes and expanded
    /// scales without dequantizing or copying the complete matrix. The parent
    /// matrix has already validated the compact 128-row scale grid, so the
    /// requested range does not itself need to end on a scale-row boundary.
    public func rowSlice(_ rows: Range<Int>) throws -> BlockFP8Weights {
        guard rows.lowerBound >= 0, !rows.isEmpty, rows.upperBound <= outFeatures else {
            throw BlockFP8BridgeError.invalidGeometry(
                "row slice \(rows) is outside 0..<\(outFeatures)")
        }
        let firstScaleRow = rows.lowerBound / scaleBlockRows
        let lastScaleRow = (rows.upperBound - 1) / scaleBlockRows
        let touchedScaleRows = lastScaleRow - firstScaleRow + 1
        let scaleColumns = inFeatures / scaleBlockColumns
        let compact = touchedScaleRows.multipliedReportingOverflow(by: scaleColumns)
        let expanded = rows.count.multipliedReportingOverflow(
            by: inFeatures / Self.mlxGroupSize)
        guard !compact.overflow, !expanded.overflow else {
            throw BlockFP8BridgeError.invalidGeometry(
                "row-slice scale accounting overflows Int")
        }
        return BlockFP8Weights(
            packed: packed[rows, 0...],
            expandedScales: expandedScales[rows, 0...],
            outFeatures: rows.count,
            inFeatures: inFeatures,
            scaleBlockRows: scaleBlockRows,
            scaleBlockColumns: scaleBlockColumns,
            compactScaleBytes: compact.partialValue,
            expandedScaleBytes: expanded.partialValue)
    }

    private struct Geometry {
        let packedBytes: Int
        let compactScaleBytes: Int
        let expandedScaleBytes: Int
        let scaleRows: Int
        let scaleColumns: Int
        let expandedScaleColumns: Int
        let repetitionsPerBlockColumn: Int
    }

    private static func validateGeometry(
        outFeatures: Int,
        inFeatures: Int,
        scaleBlockRows: Int,
        scaleBlockColumns: Int
    ) throws -> Geometry {
        guard outFeatures > 0, inFeatures > 0 else {
            throw BlockFP8BridgeError.invalidGeometry(
                "outFeatures and inFeatures must be positive")
        }
        guard scaleBlockRows > 0, scaleBlockColumns > 0 else {
            throw BlockFP8BridgeError.invalidGeometry(
                "scale block dimensions must be positive")
        }
        guard outFeatures % scaleBlockRows == 0 else {
            throw BlockFP8BridgeError.invalidGeometry(
                "\(outFeatures) rows do not divide into \(scaleBlockRows)-row blocks")
        }
        guard inFeatures % scaleBlockColumns == 0 else {
            throw BlockFP8BridgeError.invalidGeometry(
                "\(inFeatures) columns do not divide into \(scaleBlockColumns)-column blocks")
        }
        guard scaleBlockColumns % mlxGroupSize == 0 else {
            throw BlockFP8BridgeError.invalidGeometry(
                "scaleBlockColumns \(scaleBlockColumns) is not a multiple of MLX group size "
                    + "\(mlxGroupSize)")
        }
        guard inFeatures % 4 == 0 else {
            throw BlockFP8BridgeError.invalidGeometry(
                "inFeatures \(inFeatures) is not divisible by four FP8 bytes per uint32")
        }

        let packed = outFeatures.multipliedReportingOverflow(by: inFeatures)
        guard !packed.overflow else {
            throw BlockFP8BridgeError.invalidGeometry("packed byte count overflows Int")
        }
        let scaleRows = outFeatures / scaleBlockRows
        let scaleColumns = inFeatures / scaleBlockColumns
        let compact = scaleRows.multipliedReportingOverflow(by: scaleColumns)
        guard !compact.overflow else {
            throw BlockFP8BridgeError.invalidGeometry("compact scale byte count overflows Int")
        }
        let expandedRows = outFeatures
        let expandedColumns = inFeatures / mlxGroupSize
        let expanded = expandedRows.multipliedReportingOverflow(by: expandedColumns)
        guard !expanded.overflow else {
            throw BlockFP8BridgeError.invalidGeometry("expanded scale byte count overflows Int")
        }
        return Geometry(
            packedBytes: packed.partialValue,
            compactScaleBytes: compact.partialValue,
            expandedScaleBytes: expanded.partialValue,
            scaleRows: scaleRows,
            scaleColumns: scaleColumns,
            expandedScaleColumns: expandedColumns,
            repetitionsPerBlockColumn: scaleBlockColumns / mlxGroupSize)
    }

    /// Reject any non-finite E4M3FN code in the packed matrix, reporting the
    /// first one in ascending byte order.
    ///
    /// The two codes are `0x7f` and `0xff`, which are exactly the bytes whose
    /// seven low bits are all set — so one masked comparison per lane decides
    /// both, and the matrix is read once. A vector hit falls back to the element
    /// loop over that one chunk, which is what keeps the reported offset and
    /// byte identical to the element-at-a-time scan this replaces: a lane
    /// reduction says *that* a chunk contains a code, never which lane is first.
    ///
    /// `SIMD16` is the width deliberately: it is one NEON register, and it is
    /// the width that measured. `SIMD64<UInt8>` lowers through a stack temporary
    /// on this toolchain and scans at 0.8 GB/s — 2.5× *slower* than the element
    /// loop it was meant to replace — against 33 GB/s here. Anyone widening this
    /// should measure before believing it helps.
    static func validateFiniteWeights(_ bytes: UnsafeRawBufferPointer) throws {
        guard let base = bytes.baseAddress, bytes.count > 0 else { return }
        let count = bytes.count
        let lanes = MemoryLayout<SIMD16<UInt8>>.size
        let lowBits = SIMD16<UInt8>(repeating: 0x7f)
        var index = 0
        while index + lanes <= count {
            let chunk = (base + index).loadUnaligned(as: SIMD16<UInt8>.self)
            if any((chunk & lowBits) .== lowBits) {
                try reportFirstNonFiniteWeight(base, from: index, to: index + lanes)
            }
            index += lanes
        }
        try reportFirstNonFiniteWeight(base, from: index, to: count)
    }

    /// The element loop, over a bounded range: the tail too short for a vector,
    /// or the one chunk a vector comparison flagged.
    private static func reportFirstNonFiniteWeight(
        _ base: UnsafeRawPointer, from: Int, to: Int
    ) throws {
        for offset in from..<to {
            let byte = base.load(fromByteOffset: offset, as: UInt8.self)
            if byte == 0x7f || byte == 0xff {
                throw BlockFP8BridgeError.nonFiniteWeight(byteOffset: offset, byte: byte)
            }
        }
    }

    static func expandScales(
        _ compact: UnsafeRawBufferPointer,
        scaleRows: Int,
        scaleColumns: Int,
        scaleBlockRows: Int,
        repetitionsPerBlockColumn: Int,
        outputColumns: Int,
        outputByteCount: Int
    ) throws -> [UInt8] {
        let source = compact.bindMemory(to: UInt8.self)
        for (offset, byte) in source.enumerated() where byte == 0xff {
            throw BlockFP8BridgeError.nonFiniteScale(byteOffset: offset)
        }
        // Stated rather than assumed, because everything below writes through a
        // raw pointer. ``validateGeometry`` already establishes all four for the
        // initializers; this keeps the guarantee local to the code that relies
        // on it.
        guard scaleRows > 0, scaleColumns > 0, scaleBlockRows > 0,
            repetitionsPerBlockColumn > 0,
            scaleColumns * repetitionsPerBlockColumn == outputColumns,
            scaleRows * scaleBlockRows * outputColumns == outputByteCount,
            source.count == scaleRows * scaleColumns
        else {
            throw BlockFP8BridgeError.invalidGeometry(
                "scale grid expansion parameters do not describe one whole grid")
        }

        var expanded = [UInt8](repeating: 0, count: outputByteCount)
        expanded.withUnsafeMutableBufferPointer { destination in
            let output = destination.baseAddress!
            for blockRow in 0..<scaleRows {
                // Every row inside a V4 block carries the same scales, so the
                // block's first row is the whole pattern. Writing it costs
                // `outputColumns` stores rather than `scaleBlockRows ×
                // outputColumns`.
                let block = output + blockRow * scaleBlockRows * outputColumns
                for blockColumn in 0..<scaleColumns {
                    let value = source[blockRow * scaleColumns + blockColumn]
                    let start = blockColumn * repetitionsPerBlockColumn
                    for repetition in 0..<repetitionsPerBlockColumn {
                        block[start + repetition] = value
                    }
                }
                // The block's rows are contiguous, so the remaining ones are
                // filled by doubling what is already written: ⌈log2(128)⌉ = 7
                // copies per block instead of a store per element.
                var filled = 1
                while filled < scaleBlockRows {
                    let rows = min(filled, scaleBlockRows - filled)
                    (block + filled * outputColumns).update(
                        from: block, count: rows * outputColumns)
                    filled += rows
                }
            }
        }
        return expanded
    }
}

/// Dense `y = x · Wᵀ` over V4's finite block-FP8 checkpoint bytes.
public func blockFP8MM(
    _ x: MLXArray,
    _ weights: BlockFP8Weights,
    stream: StreamOrDevice = .default
) throws -> MLXArray {
    guard x.ndim >= 2 else {
        throw BlockFP8BridgeError.activation(
            "shape \(x.shape) must have a matrix dimension")
    }
    guard x.shape.last == weights.inFeatures else {
        throw BlockFP8BridgeError.activation(
            "last dimension \(x.shape.last ?? 0) does not match \(weights.inFeatures) input features")
    }
    guard x.dtype == .float32 || x.dtype == .float16 || x.dtype == .bfloat16 else {
        throw BlockFP8BridgeError.activation(
            "dtype must be float32, float16, or bfloat16; got \(x.dtype)")
    }
    return quantizedMM(
        x,
        weights.packed,
        scales: weights.expandedScales,
        biases: nil,
        transpose: true,
        groupSize: BlockFP8Weights.mlxGroupSize,
        bits: BlockFP8Weights.bits,
        mode: .mxfp8,
        stream: stream)
}

/// Dense decode for validation and debugging, never the product hot path.
public func blockFP8Dequantize(
    _ weights: BlockFP8Weights,
    dtype: DType = .float32,
    stream: StreamOrDevice = .default
) -> MLXArray {
    dequantized(
        weights.packed,
        scales: weights.expandedScales,
        biases: nil,
        groupSize: BlockFP8Weights.mlxGroupSize,
        bits: BlockFP8Weights.bits,
        mode: .mxfp8,
        dtype: dtype,
        stream: stream)
}
