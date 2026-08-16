import CoreFoundation
import Foundation
import MLX
import MLXBridge

/// A minimal safetensors reader: header JSON, byte ranges, bounds checks.
///
/// Deliberately not a general-purpose implementation. It reads exactly what
/// this checkpoint contains — `F32`, `BF16` and the `U8` streams that carry
/// MXFP4 `weight_packed`/`weight_scale` — and refuses anything else by name and
/// dtype rather than guessing (spec §5.8).
///
/// ## Why it memory-maps
///
/// The tiny checkpoint is 1.39 GiB, of which `embed_tokens` and `lm_head` are
/// 640 MiB each. Reading the file into a `Data` would put both in the process
/// before a single token is embedded, which is the opposite of what this
/// project is for. Mapping it means a tensor costs address space until it is
/// actually touched, and it makes ``rows(_:at:)`` — the row-addressable
/// embedding lookup spec §16.4 asks for — a genuine partial read rather than a
/// slice of something already resident.
public final class SafetensorsFile {

    /// One tensor's header entry, resolved to absolute file offsets.
    public struct TensorInfo: Sendable {
        public let name: String
        public let dtype: String
        public let shape: [Int]
        /// Absolute offset of the first byte in the file.
        public let offset: Int
        public let byteCount: Int

        public var elementCount: Int { shape.reduce(1, *) }
    }

    public let path: String
    public let tensors: [String: TensorInfo]
    /// Present exactly as the header spells them, for provenance reporting.
    public let metadata: [String: String]

    private let mapped: Data

    public init(path: String) throws {
        self.path = path
        let data = try Data(contentsOf: URL(fileURLWithPath: path), options: [.mappedIfSafe])
        self.mapped = data

        guard data.count >= 8 else {
            throw TinyK3Error.malformedSafetensors("file is \(data.count) bytes, shorter than the 8-byte header length")
        }
        let rawHeaderLength = data.withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        guard rawHeaderLength > 0, rawHeaderLength <= UInt64(Int.max) else {
            throw TinyK3Error.malformedSafetensors(
                "header length \(rawHeaderLength) cannot be represented by this process")
        }
        let headerLength = Int(rawHeaderLength)
        let dataStartResult = 8.addingReportingOverflow(headerLength)
        guard !dataStartResult.overflow, dataStartResult.partialValue <= data.count else {
            throw TinyK3Error.malformedSafetensors(
                "header claims \(headerLength) bytes, file has \(data.count - 8) after the length field")
        }
        let dataStart = dataStartResult.partialValue
        let payloadBytes = data.count - dataStart

        let headerData = data.subdata(in: 8..<dataStart)
        guard let root = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw TinyK3Error.malformedSafetensors("header is not a JSON object")
        }

        var metadata: [String: String] = [:]
        if let block = root["__metadata__"] as? [String: Any] {
            for (key, value) in block { metadata[key] = String(describing: value) }
        }
        self.metadata = metadata

        var tensors: [String: TensorInfo] = [:]
        var occupied: [(name: String, begin: Int, end: Int)] = []
        for (name, raw) in root where name != "__metadata__" {
            guard
                let entry = raw as? [String: Any],
                let dtype = entry["dtype"] as? String,
                let rawShape = entry["shape"] as? [Any],
                let rawOffsets = entry["data_offsets"] as? [Any], rawOffsets.count == 2
            else {
                throw TinyK3Error.malformedSafetensors("entry '\(name)' is not a well-formed tensor record")
            }
            let shape = try rawShape.enumerated().map { index, value in
                try Self.nonnegativeInteger(
                    value, field: "tensor '\(name)' shape[\(index)]")
            }
            let offsets = try rawOffsets.enumerated().map { index, value in
                try Self.nonnegativeInteger(
                    value, field: "tensor '\(name)' data_offsets[\(index)]")
            }
            let (begin, end) = (offsets[0], offsets[1])
            // Bounds are checked here, once, rather than at every read: a
            // truncated or hand-edited file must be rejected before any
            // pointer arithmetic is done with its numbers.
            guard end >= begin, end <= payloadBytes else {
                throw TinyK3Error.malformedSafetensors(
                    "tensor '\(name)' spans [\(begin), \(end)) but the payload is \(payloadBytes) bytes")
            }
            guard let itemSize = Self.itemSize(dtype) else {
                throw TinyK3Error.unsupportedDType(tensor: name, dtype: dtype)
            }
            var elementCount = 1
            for dimension in shape {
                let product = elementCount.multipliedReportingOverflow(by: dimension)
                guard !product.overflow else {
                    throw TinyK3Error.malformedSafetensors(
                        "tensor '\(name)' shape \(shape) exceeds Int.max elements")
                }
                elementCount = product.partialValue
            }
            let expectedResult = elementCount.multipliedReportingOverflow(by: itemSize)
            guard !expectedResult.overflow else {
                throw TinyK3Error.malformedSafetensors(
                    "tensor '\(name)' shape \(shape) at \(itemSize) bytes per element exceeds Int.max bytes")
            }
            let expected = expectedResult.partialValue
            guard end - begin == expected else {
                throw TinyK3Error.malformedSafetensors(
                    "tensor '\(name)' spans \(end - begin) bytes, but shape \(shape) at \(itemSize) "
                        + "bytes per element needs \(expected)")
            }
            tensors[name] = TensorInfo(
                name: name, dtype: dtype, shape: shape,
                offset: dataStart + begin, byteCount: end - begin)
            occupied.append((name, begin, end))
        }
        let ordered = occupied.sorted {
            $0.begin == $1.begin ? $0.end < $1.end : $0.begin < $1.begin
        }
        for (previous, current) in zip(ordered, ordered.dropFirst())
        where current.begin < previous.end {
            throw TinyK3Error.malformedSafetensors(
                "tensor '\(current.name)' range [\(current.begin), \(current.end)) overlaps "
                    + "tensor '\(previous.name)' range [\(previous.begin), \(previous.end))")
        }
        self.tensors = tensors
    }

    private static func nonnegativeInteger(_ raw: Any, field: String) throws -> Int {
        guard let number = raw as? NSNumber,
            CFGetTypeID(number) != CFBooleanGetTypeID(),
            let value = Int(number.stringValue), value >= 0
        else {
            throw TinyK3Error.malformedSafetensors(
                "\(field) is not a nonnegative integer representable by this process")
        }
        return value
    }

    static func itemSize(_ dtype: String) -> Int? {
        switch dtype {
        case "F32", "I32", "U32": return 4
        case "BF16", "F16", "I16", "U16": return 2
        case "U8", "I8", "BOOL": return 1
        case "F64", "I64", "U64": return 8
        default: return nil
        }
    }

    // MARK: - Access

    public func info(_ name: String) throws -> TensorInfo {
        guard let info = tensors[name] else { throw TinyK3Error.missingTensor(name) }
        return info
    }

    public func contains(_ name: String) -> Bool { tensors[name] != nil }

    /// A tensor's raw bytes, as a view into the mapping.
    public func bytes(_ name: String) throws -> Data {
        let info = try self.info(name)
        let end = info.offset.addingReportingOverflow(info.byteCount)
        guard !end.overflow, end.partialValue <= mapped.count else {
            throw TinyK3Error.malformedSafetensors(
                "tensor '\(name)' resolved range exceeds the mapped file")
        }
        return mapped.subdata(in: info.offset..<end.partialValue)
    }

    /// A tensor as a float32 `MLXArray`.
    ///
    /// `F32` is a reinterpretation — every Apple target is little-endian and
    /// the file declares little-endian — and `BF16` is widened by the exact
    /// shift, which is lossless in both directions of the comparison this
    /// milestone makes.
    ///
    /// The tiny K3 checkpoint stores **every non-MoE tensor as F32** despite
    /// `config.dtype` saying `bfloat16`; that trap is recorded in the M6A
    /// experiment record. BF16 is handled anyway because the flagship stores
    /// its non-expert weights that way and a reader that silently rejected it
    /// would be a surprise at M7 rather than here.
    public func float32(_ name: String) throws -> MLXArray {
        let info = try self.info(name)
        let raw = try bytes(name)
        switch info.dtype {
        case "F32":
            let values: [Float] = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            return MLXArray(values, info.shape)
        case "BF16":
            let words: [UInt16] = raw.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
            var values = [Float](repeating: 0, count: words.count)
            for index in 0..<words.count {
                values[index] = Float(bitPattern: UInt32(words[index]) << 16)
            }
            return MLXArray(values, info.shape)
        default:
            throw TinyK3Error.unsupportedDType(tensor: name, dtype: info.dtype)
        }
    }

    /// Selected rows of a 2-D tensor, read individually.
    ///
    /// This is the embedding lookup spec §16.4 asks for: a 163,840-row table is
    /// row-addressable, and a five-token prompt needs five of those rows. The
    /// mapping means the pages behind the other 163,835 are never faulted in.
    public func rows(_ name: String, at indices: [Int]) throws -> MLXArray {
        let info = try self.info(name)
        guard info.shape.count == 2 else {
            throw TinyK3Error.shapeMismatch(tensor: name, expected: [-1, -1], actual: info.shape)
        }
        guard let itemSize = Self.itemSize(info.dtype) else {
            throw TinyK3Error.unsupportedDType(tensor: name, dtype: info.dtype)
        }
        let width = info.shape[1]
        let rowBytesResult = width.multipliedReportingOverflow(by: itemSize)
        let outputCount = indices.count.multipliedReportingOverflow(by: width)
        guard !rowBytesResult.overflow, !outputCount.overflow else {
            throw TinyK3Error.configuration("row request for '\(name)' exceeds Int.max bytes")
        }
        let rowBytes = rowBytesResult.partialValue
        var values = [Float](repeating: 0, count: outputCount.partialValue)

        for (position, row) in indices.enumerated() {
            guard row >= 0, row < info.shape[0] else {
                throw TinyK3Error.configuration(
                    "row \(row) is out of range for '\(name)' with \(info.shape[0]) rows")
            }
            let relativeStart = row.multipliedReportingOverflow(by: rowBytes)
            let start = info.offset.addingReportingOverflow(relativeStart.partialValue)
            let end = start.partialValue.addingReportingOverflow(rowBytes)
            guard !relativeStart.overflow, !start.overflow, !end.overflow,
                end.partialValue <= mapped.count
            else {
                throw TinyK3Error.malformedSafetensors(
                    "row \(row) of '\(name)' resolves outside the mapped file")
            }
            let slice = mapped.subdata(in: start.partialValue..<end.partialValue)
            switch info.dtype {
            case "F32":
                slice.withUnsafeBytes { raw in
                    let source = raw.bindMemory(to: Float.self)
                    for column in 0..<width { values[position * width + column] = source[column] }
                }
            case "BF16":
                slice.withUnsafeBytes { raw in
                    let source = raw.bindMemory(to: UInt16.self)
                    for column in 0..<width {
                        values[position * width + column] = Float(bitPattern: UInt32(source[column]) << 16)
                    }
                }
            default:
                throw TinyK3Error.unsupportedDType(tensor: name, dtype: info.dtype)
            }
        }
        return MLXArray(values, [indices.count, width])
    }

    /// A contiguous row range of a 2-D tensor, as one float32 array.
    ///
    /// Used to walk `lm_head` in chunks: the vocabulary projection is 640 MiB
    /// in float32 and is the largest single tensor in the checkpoint, so
    /// materialising it whole would dominate the process's memory and make
    /// "memory remains bounded" a statement about nothing.
    public func rowRange(_ name: String, from first: Int, count: Int) throws -> MLXArray {
        let info = try self.info(name)
        guard info.shape.count == 2 else {
            throw TinyK3Error.shapeMismatch(tensor: name, expected: [-1, -1], actual: info.shape)
        }
        let rowEnd = first.addingReportingOverflow(count)
        guard first >= 0, count >= 0, !rowEnd.overflow, rowEnd.partialValue <= info.shape[0] else {
            throw TinyK3Error.configuration(
                "row range from \(first) with count \(count) is out of range for '\(name)' with \(info.shape[0]) rows")
        }
        guard let itemSize = Self.itemSize(info.dtype) else {
            throw TinyK3Error.unsupportedDType(tensor: name, dtype: info.dtype)
        }
        let width = info.shape[1]
        let rowElements = width.multipliedReportingOverflow(by: itemSize)
        let relativeStart = first.multipliedReportingOverflow(by: rowElements.partialValue)
        let byteCount = count.multipliedReportingOverflow(by: rowElements.partialValue)
        let start = info.offset.addingReportingOverflow(relativeStart.partialValue)
        let end = start.partialValue.addingReportingOverflow(byteCount.partialValue)
        guard !rowElements.overflow, !relativeStart.overflow, !byteCount.overflow,
            !start.overflow, !end.overflow, end.partialValue <= mapped.count
        else {
            throw TinyK3Error.configuration("row range for '\(name)' exceeds Int.max bytes")
        }
        let slice = mapped.subdata(in: start.partialValue..<end.partialValue)
        switch info.dtype {
        case "F32":
            let values: [Float] = slice.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            return MLXArray(values, [count, width])
        case "BF16":
            let words: [UInt16] = slice.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
            var values = [Float](repeating: 0, count: words.count)
            for index in 0..<words.count {
                values[index] = Float(bitPattern: UInt32(words[index]) << 16)
            }
            return MLXArray(values, [count, width])
        default:
            throw TinyK3Error.unsupportedDType(tensor: name, dtype: info.dtype)
        }
    }

    /// An MXFP4 linear's `<base>.weight_packed` / `<base>.weight_scale` pair,
    /// wrapped for MLX without repacking.
    ///
    /// The bytes go across as they lie in the file. M2 established that the
    /// checkpoint's `uint8[out, in/2]` stream and MLX's `uint32[out, in/8]`
    /// operand are the same byte image; `MXFP4Weights` is where that claim
    /// lives, and this function only supplies it with the right shapes.
    public func mxfp4(_ base: String, experts: Int? = nil) throws -> MXFP4Weights {
        let packedInfo = try info("\(base).weight_packed")
        let scaleInfo = try info("\(base).weight_scale")
        guard packedInfo.dtype == "U8", scaleInfo.dtype == "U8" else {
            throw TinyK3Error.unsupportedDType(
                tensor: "\(base).weight_packed", dtype: "\(packedInfo.dtype)/\(scaleInfo.dtype)")
        }
        guard packedInfo.shape.count == 2 else {
            throw TinyK3Error.shapeMismatch(
                tensor: "\(base).weight_packed", expected: [-1, -1], actual: packedInfo.shape)
        }
        let outFeatures = packedInfo.shape[0]
        let inFeaturesResult = packedInfo.shape[1].multipliedReportingOverflow(by: 2)
        guard !inFeaturesResult.overflow else {
            throw TinyK3Error.configuration("'\(base).weight_packed' input width exceeds Int.max")
        }
        if let experts {
            guard experts > 0, outFeatures % experts == 0 else {
                throw TinyK3Error.configuration(
                    "'\(base).weight_packed' has \(outFeatures) rows, not a positive whole stack of \(experts) experts")
            }
        }
        let inFeatures = inFeaturesResult.partialValue

        let packed = try bytes("\(base).weight_packed")
        let scales = try bytes("\(base).weight_scale")
        return try packed.withUnsafeBytes { packedBytes in
            try scales.withUnsafeBytes { scaleBytes in
                try MXFP4Weights(
                    packedBytes: packedBytes, scaleBytes: scaleBytes, experts: experts,
                    outFeatures: experts.map { outFeatures / $0 } ?? outFeatures,
                    inFeatures: inFeatures)
            }
        }
    }
}
