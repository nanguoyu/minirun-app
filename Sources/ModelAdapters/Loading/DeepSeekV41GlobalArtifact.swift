import Darwin
import Foundation
import MLX
import StorageCore

/// A tensor the V4.1 `global00` unit owns and the text path uses.
public enum DeepSeekV41GlobalTensor: String, Sendable, Equatable, CaseIterable {
    case embed = "embed.weight"
    case head = "head.weight"
    case finalNorm = "norm.weight"
    case imageStart = "image_start"
    case imageEnd = "image_end"
    case imageNewline = "image_newline"
}

/// A checked, row-addressed view of the published V4.1 `global00` unit.
///
/// The embedding and the output head are 1.32 GB each. This reader never
/// materializes either: rows are read by token id, or in a caller-declared
/// window. Everything else in the unit — the aligner, the 32-block vision tower
/// and the three image marker vectors — is **reconciled and named** but not
/// interpreted, because a text-only runner that pretended those tensors were
/// absent would be reading a manifest it had not checked.
public final class DeepSeekV41GlobalArtifact: @unchecked Sendable {
    /// One reconciled tensor of this unit: where it is and what shape it has.
    public struct TensorRecord: Sendable, Equatable {
        public let tensor: String
        public let dtype: DeepSeekV41PlainDType
        public let shape: [Int]
        public let bytes: UInt64
        let reference: String
        let fileBytes: UInt64
        let offset: UInt64
    }

    public let unitID: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let manifestFileCount: Int
    public let manifestBytes: UInt64

    private let fileAccess: ModelFileAccess
    private let records: [String: TensorRecord]
    /// The largest row window one read is allowed to materialize.
    private let maximumRowsPerRead: Int

    /// Filesystem convenience for tests and command-line probes.
    public convenience init(
        manifestURL: URL,
        expectedSourceRepository: String? = nil,
        expectedSourceRevision: String? = nil,
        maximumRowsPerRead: Int = 4_096
    ) throws {
        let directory = manifestURL.deletingLastPathComponent()
        try self.init(
            manifestData: try Data(contentsOf: manifestURL),
            unitReference: directory.lastPathComponent,
            filesystemDirectory: directory,
            fileAccess: .filesystem,
            expectedSourceRepository: expectedSourceRepository,
            expectedSourceRevision: expectedSourceRevision,
            maximumRowsPerRead: maximumRowsPerRead)
    }

    public convenience init(
        manifestData: Data,
        unitReference: String,
        fileAccess: ModelFileAccess,
        expectedSourceRepository: String? = nil,
        expectedSourceRevision: String? = nil,
        maximumRowsPerRead: Int = 4_096
    ) throws {
        try self.init(
            manifestData: manifestData,
            unitReference: unitReference,
            filesystemDirectory: nil,
            fileAccess: fileAccess,
            expectedSourceRepository: expectedSourceRepository,
            expectedSourceRevision: expectedSourceRevision,
            maximumRowsPerRead: maximumRowsPerRead)
    }

    private init(
        manifestData: Data,
        unitReference: String,
        filesystemDirectory: URL?,
        fileAccess: ModelFileAccess,
        expectedSourceRepository: String?,
        expectedSourceRevision: String?,
        maximumRowsPerRead: Int
    ) throws {
        let document: DeepSeekV41BlockArtifact.Document
        do {
            document = try JSONDecoder().decode(
                DeepSeekV41BlockArtifact.Document.self, from: manifestData)
        } catch {
            throw DeepSeekV41Error.artifact("manifest.json could not be decoded: \(error)")
        }
        guard DeepSeekV41UnitName.isSafeComponent(unitReference),
            document.unit == unitReference,
            unitReference.hasPrefix("global")
        else {
            throw DeepSeekV41Error.artifact(
                "manifest unit '\(document.unit)' does not name rooted global unit "
                    + "'\(unitReference)'")
        }
        guard document.expertOrder == nil else {
            throw DeepSeekV41Error.artifact(
                "\(unitReference) declares an expert_order; the global unit has no "
                    + "routed experts")
        }
        guard !document.sourceRepository.isEmpty, !document.sourceRevision.isEmpty else {
            throw DeepSeekV41Error.artifact(
                "manifest source repository and revision must be non-empty")
        }
        if let expectedSourceRepository,
            document.sourceRepository != expectedSourceRepository
        {
            throw DeepSeekV41Error.artifact(
                "source repository '\(document.sourceRepository)' does not match the "
                    + "artifact index")
        }
        if let expectedSourceRevision, document.sourceRevision != expectedSourceRevision {
            throw DeepSeekV41Error.artifact(
                "source revision '\(document.sourceRevision)' does not match the "
                    + "artifact index")
        }
        guard maximumRowsPerRead > 0 else {
            throw DeepSeekV41Error.artifact("the row read window must be positive")
        }

        var fileNames = Set<String>()
        var records = [String: TensorRecord]()
        var manifestBytes: UInt64 = 0

        for file in document.files {
            guard DeepSeekV41UnitName.isSafeComponent(file.name),
                fileNames.insert(file.name).inserted
            else {
                throw DeepSeekV41Error.artifact(
                    "manifest contains an unsafe or repeated file name '\(file.name)'")
            }
            guard file.kind == "blob" else {
                throw DeepSeekV41Error.artifact(
                    "\(file.name) uses payload kind '\(file.kind)'; the global unit is "
                        + "plain blobs only")
            }
            guard file.bytes > 0 else {
                throw DeepSeekV41Error.artifact("\(file.name) has a zero byte length")
            }
            let next = manifestBytes.addingReportingOverflow(file.bytes)
            guard !next.overflow else {
                throw DeepSeekV41Error.artifact(
                    "manifest payload byte total exceeds UInt64.max")
            }
            manifestBytes = next.partialValue
            let reference = filesystemDirectory?.appendingPathComponent(file.name).path
                ?? "\(unitReference)/\(file.name)"
            if let digest = file.sha256,
                !DeepSeekV41BlockArtifact.isLowercaseSHA256(digest)
            {
                throw DeepSeekV41Error.artifact("\(file.name) has a non-canonical SHA-256")
            }
            try fileAccess.withDescriptor(reference) { descriptor in
                let actual = try DeepSeekV41BlockArtifact.fileLength(
                    descriptor, reference: reference)
                guard actual == file.bytes else {
                    throw DeepSeekV41Error.artifact(
                        "\(file.name) is \(actual) bytes; manifest says \(file.bytes)")
                }
            }

            if let members = file.members2 {
                var expectedOffset: UInt64 = 0
                for member in members {
                    guard member.offset == expectedOffset else {
                        throw DeepSeekV41Error.artifact(
                            "\(file.name) members are not a gapless partition at byte "
                                + "\(expectedOffset)")
                    }
                    let record = try Self.record(
                        tensor: member.sourceTensor, dtypeName: member.dtype,
                        shape: member.shape, reference: reference,
                        fileBytes: file.bytes, offset: member.offset,
                        bytes: member.bytes)
                    guard records.updateValue(record, forKey: record.tensor) == nil else {
                        throw DeepSeekV41Error.artifact(
                            "tensor '\(record.tensor)' appears more than once")
                    }
                    let end = expectedOffset.addingReportingOverflow(member.bytes)
                    guard !end.overflow else {
                        throw DeepSeekV41Error.artifact(
                            "\(file.name) member byte partition overflows UInt64")
                    }
                    expectedOffset = end.partialValue
                }
                guard expectedOffset == file.bytes else {
                    throw DeepSeekV41Error.artifact(
                        "\(file.name) members describe \(expectedOffset) of "
                            + "\(file.bytes) bytes")
                }
            } else {
                guard let tensor = file.sourceTensor, let dtype = file.dtype,
                    let shape = file.shape
                else {
                    throw DeepSeekV41Error.artifact(
                        "\(file.name) has neither a member table nor standalone tensor "
                            + "metadata")
                }
                let record = try Self.record(
                    tensor: tensor, dtypeName: dtype, shape: shape,
                    reference: reference, fileBytes: file.bytes, offset: 0,
                    bytes: file.bytes)
                guard records.updateValue(record, forKey: record.tensor) == nil else {
                    throw DeepSeekV41Error.artifact(
                        "tensor '\(record.tensor)' appears more than once")
                }
            }
        }

        for tensor in DeepSeekV41GlobalTensor.allCases {
            guard records[tensor.rawValue] != nil else {
                throw DeepSeekV41Error.artifact(
                    "\(unitReference) has no required global tensor '\(tensor.rawValue)'")
            }
        }

        self.unitID = document.unit
        self.sourceRepository = document.sourceRepository
        self.sourceRevision = document.sourceRevision
        self.manifestFileCount = document.files.count
        self.manifestBytes = manifestBytes
        self.fileAccess = fileAccess
        self.records = records
        self.maximumRowsPerRead = maximumRowsPerRead
    }

    // MARK: - What this unit holds

    public func descriptor(
        of tensor: DeepSeekV41GlobalTensor
    ) throws -> (dtype: DeepSeekV41PlainDType, shape: [Int]) {
        let record = try require(tensor.rawValue)
        return (record.dtype, record.shape)
    }

    /// Every tensor name this unit publishes, sorted — including the vision
    /// tower and the aligner, which the text path does not execute.
    public var publishedTensors: [String] { records.keys.sorted() }

    /// The reconciled record for any published tensor, by full checkpoint name.
    public func record(of tensor: String) -> TensorRecord? { records[tensor] }

    /// Names this unit publishes that no text-path case claims: `aligner.*`,
    /// `vision.*`. Present so a text-only run can skip them *explicitly*.
    public var unusedTensors: [String] {
        let used = Set(DeepSeekV41GlobalTensor.allCases.map(\.rawValue))
        return records.keys.filter { !used.contains($0) }.sorted()
    }

    // MARK: - Loading

    /// Load one small BF16 or F32 tensor whole — a norm or an image marker.
    ///
    /// Refuses the embedding and the head by name: both are 1.32 GB, and a
    /// reader that let them through here would defeat the whole point of the
    /// row window below.
    public func loadVector(
        _ tensor: DeepSeekV41GlobalTensor,
        cancellationCheck: () throws -> Void = {}
    ) throws -> MLXArray {
        guard tensor != .embed, tensor != .head else {
            throw DeepSeekV41Error.artifact(
                "\(tensor.rawValue) is a row table; read it with loadRows(_:first:count:)")
        }
        let record = try require(tensor.rawValue)
        return try load(record, shape: record.shape, cancellationCheck: cancellationCheck)
    }

    /// Load `count` consecutive rows of the embedding or the output head.
    ///
    /// The result is `[count, hiddenSize]` in the table's own dtype. Nothing
    /// larger than ``maximumRowsPerRead`` is read at once, so a caller cannot
    /// accidentally ask for the whole 129,280-row table in one allocation.
    public func loadRows(
        _ tensor: DeepSeekV41GlobalTensor,
        first: Int,
        count: Int,
        cancellationCheck: () throws -> Void = {}
    ) throws -> MLXArray {
        guard tensor == .embed || tensor == .head else {
            throw DeepSeekV41Error.artifact(
                "\(tensor.rawValue) is not a row table; read it with loadVector(_:)")
        }
        let record = try require(tensor.rawValue)
        guard record.shape.count == 2 else {
            throw DeepSeekV41Error.artifact(
                "\(record.tensor) is \(record.shape); a row table is two-dimensional")
        }
        let rows = record.shape[0]
        let columns = record.shape[1]
        guard count > 0, count <= maximumRowsPerRead, first >= 0,
            first <= rows - count
        else {
            throw DeepSeekV41Error.artifact(
                "rows \(first)..<\(first + count) are outside \(record.tensor)'s "
                    + "\(rows) rows, or exceed the \(maximumRowsPerRead)-row read window")
        }
        let rowBytes = UInt64(columns) * UInt64(record.dtype.byteWidth)
        let windowed = TensorRecord(
            tensor: record.tensor, dtype: record.dtype, shape: [count, columns],
            bytes: UInt64(count) * rowBytes, reference: record.reference,
            fileBytes: record.fileBytes,
            offset: record.offset + UInt64(first) * rowBytes)
        return try load(
            windowed, shape: [count, columns], cancellationCheck: cancellationCheck)
    }

    /// One embedding row, by token id.
    public func embeddingRow(
        token: Int, cancellationCheck: () throws -> Void = {}
    ) throws -> MLXArray {
        try loadRows(.embed, first: token, count: 1, cancellationCheck: cancellationCheck)
    }

    // MARK: - Agreement with a configuration

    public func validateAgainst(config: DeepSeekV41Config) throws {
        func expect(
            _ tensor: DeepSeekV41GlobalTensor,
            _ dtype: DeepSeekV41PlainDType,
            _ shape: [Int]
        ) throws {
            let actual = try descriptor(of: tensor)
            guard actual.dtype == dtype, actual.shape == shape else {
                throw DeepSeekV41Error.artifact(
                    "\(tensor.rawValue) is \(actual.dtype.rawValue)\(actual.shape), "
                        + "expected \(dtype.rawValue)\(shape)")
            }
        }
        try expect(.embed, .bfloat16, [config.vocabularySize, config.hiddenSize])
        try expect(.head, .bfloat16, [config.vocabularySize, config.hiddenSize])
        try expect(.finalNorm, .bfloat16, [config.hiddenSize])
        for marker in [
            DeepSeekV41GlobalTensor.imageStart, .imageEnd, .imageNewline,
        ] {
            try expect(marker, .bfloat16, [config.hiddenSize])
        }
        // The vision tower is decoded, not executed — but its block count is
        // checked, because a manifest that carried 24 of them would be a
        // different publication than the configuration describes.
        let visionBlocks = Set(records.keys.compactMap { name -> Int? in
            guard name.hasPrefix("vision.blocks.") else { return nil }
            let rest = name.dropFirst("vision.blocks.".count)
            return Int(rest.prefix(while: \.isNumber))
        })
        guard visionBlocks == Set(0..<config.visionLayerCount) else {
            throw DeepSeekV41Error.artifact(
                "\(unitID) publishes \(visionBlocks.count) vision blocks; the "
                    + "configuration says \(config.visionLayerCount)")
        }
    }

    // MARK: - Private

    private func require(_ tensor: String) throws -> TensorRecord {
        guard let record = records[tensor] else {
            throw DeepSeekV41Error.artifact("\(unitID) has no tensor '\(tensor)'")
        }
        return record
    }

    private func load(
        _ record: TensorRecord, shape: [Int], cancellationCheck: () throws -> Void
    ) throws -> MLXArray {
        let data = try fileAccess.withDescriptor(record.reference) { descriptor -> Data in
            let actual = try DeepSeekV41BlockArtifact.fileLength(
                descriptor, reference: record.reference)
            guard actual == record.fileBytes else {
                throw DeepSeekV41Error.artifact(
                    "\(record.reference) changed length after manifest reconciliation")
            }
            guard let length = Int(exactly: record.bytes) else {
                throw DeepSeekV41Error.artifact(
                    "\(record.tensor) byte length is not representable by this process")
            }
            var data = Data(count: length)
            try data.withUnsafeMutableBytes { buffer in
                guard let base = buffer.baseAddress else {
                    throw DeepSeekV41Error.artifact(
                        "\(record.reference) could not allocate its read buffer")
                }
                var completed = 0
                let chunk = 4 * 1_024 * 1_024
                guard let start = off_t(exactly: record.offset) else {
                    throw DeepSeekV41Error.artifact(
                        "\(record.reference) has an unrepresentable read range")
                }
                while completed < length {
                    try cancellationCheck()
                    let take = min(chunk, length - completed)
                    let result = pread(
                        descriptor, base + completed, take, start + off_t(completed))
                    if result < 0 {
                        if errno == EINTR { continue }
                        throw StorageCoreError.posix(
                            operation: "pread", path: record.reference, code: errno)
                    }
                    guard result > 0 else {
                        throw StorageCoreError.shortTransfer(
                            operation: "pread", path: record.reference,
                            offset: record.offset + UInt64(completed),
                            expected: take, actual: 0)
                    }
                    completed += result
                }
            }
            return data
        }
        let pointer = UnsafeMutableRawPointer.allocate(
            byteCount: data.count, alignment: 16)
        data.copyBytes(to: pointer.assumingMemoryBound(to: UInt8.self), count: data.count)
        let array = MLXArray(
            rawPointer: pointer, shape, dtype: record.dtype.mlxDType,
            finalizer: { pointer.deallocate() })
        array.eval()
        return array
    }

    private static func record(
        tensor: String,
        dtypeName: String,
        shape: [Int],
        reference: String,
        fileBytes: UInt64,
        offset: UInt64,
        bytes: UInt64
    ) throws -> TensorRecord {
        guard !tensor.isEmpty, !tensor.contains("\0") else {
            throw DeepSeekV41Error.artifact("tensor '\(tensor)' has an unusable name")
        }
        guard let dtype = DeepSeekV41PlainDType(rawValue: dtypeName) else {
            throw DeepSeekV41Error.artifact(
                "\(tensor) has unsupported plain dtype '\(dtypeName)'")
        }
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw DeepSeekV41Error.artifact("\(tensor) has invalid shape \(shape)")
        }
        var elements: UInt64 = 1
        for dimension in shape {
            let product = elements.multipliedReportingOverflow(by: UInt64(dimension))
            guard !product.overflow else {
                throw DeepSeekV41Error.artifact("\(tensor) shape overflows UInt64")
            }
            elements = product.partialValue
        }
        let byteCount = elements.multipliedReportingOverflow(by: UInt64(dtype.byteWidth))
        guard !byteCount.overflow, byteCount.partialValue == bytes else {
            throw DeepSeekV41Error.artifact(
                "\(tensor) \(dtype.rawValue)\(shape) requires "
                    + "\(byteCount.overflow ? UInt64.max : byteCount.partialValue) bytes, "
                    + "manifest says \(bytes)")
        }
        let end = offset.addingReportingOverflow(bytes)
        guard !end.overflow, end.partialValue <= fileBytes else {
            throw DeepSeekV41Error.artifact(
                "\(tensor) range runs past its \(fileBytes)-byte blob")
        }
        return TensorRecord(
            tensor: tensor, dtype: dtype, shape: shape, bytes: bytes,
            reference: reference, fileBytes: fileBytes, offset: offset)
    }
}
