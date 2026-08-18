import Darwin
import Foundation
import MLX
import StorageCore

/// A checked, row-addressed view of the published V4 `global00` unit.
///
/// The embedding and output tables are each about one GiB in the current
/// artifact. This reader never materializes either table: embeddings are read
/// by token id and output weights are walked in a caller-declared row window.
/// The product supplies a rooted ``ModelFileAccess`` from the fully verified
/// artifact, so no repository revision is compiled into this adapter.
public final class DeepSeekV4GlobalArtifact: @unchecked Sendable {
    public struct Geometry: Sendable, Equatable {
        public let hiddenSize: Int
        public let vocabularySize: Int
        public let hyperConnectionMultiplicity: Int
        public let rmsNormEpsilon: Float
        public let hyperConnectionEpsilon: Float
        public let maximumHeadRowsPerRead: Int

        public init(
            hiddenSize: Int,
            vocabularySize: Int,
            hyperConnectionMultiplicity: Int,
            rmsNormEpsilon: Float,
            hyperConnectionEpsilon: Float,
            maximumHeadRowsPerRead: Int = 4_096
        ) throws {
            guard hiddenSize > 0, vocabularySize > 0,
                hyperConnectionMultiplicity > 0,
                maximumHeadRowsPerRead > 0,
                maximumHeadRowsPerRead <= vocabularySize,
                rmsNormEpsilon.isFinite, rmsNormEpsilon > 0,
                hyperConnectionEpsilon.isFinite, hyperConnectionEpsilon > 0
            else {
                throw DeepSeekV4Error.artifact(
                    "global geometry and read window must be finite and positive")
            }
            guard Self.checkedProduct([vocabularySize, hiddenSize, 2]) != nil,
                Self.checkedProduct([
                    hyperConnectionMultiplicity,
                    hyperConnectionMultiplicity,
                    hiddenSize,
                    4,
                ]) != nil
            else {
                throw DeepSeekV4Error.artifact("global tensor geometry overflows UInt64")
            }
            self.hiddenSize = hiddenSize
            self.vocabularySize = vocabularySize
            self.hyperConnectionMultiplicity = hyperConnectionMultiplicity
            self.rmsNormEpsilon = rmsNormEpsilon
            self.hyperConnectionEpsilon = hyperConnectionEpsilon
            self.maximumHeadRowsPerRead = maximumHeadRowsPerRead
        }

        public init(
            config: DeepSeekV4Config,
            maximumHeadRowsPerRead: Int = 4_096
        ) throws {
            try self.init(
                hiddenSize: config.hiddenSize,
                vocabularySize: config.vocabularySize,
                hyperConnectionMultiplicity: config.hyperConnectionMultiplicity,
                rmsNormEpsilon: config.rmsNormEpsilon,
                hyperConnectionEpsilon: config.hyperConnectionEpsilon,
                maximumHeadRowsPerRead: maximumHeadRowsPerRead)
        }

        private static func checkedProduct(_ values: [Int]) -> UInt64? {
            var product: UInt64 = 1
            for value in values {
                guard value > 0 else { return nil }
                let next = product.multipliedReportingOverflow(by: UInt64(value))
                guard !next.overflow else { return nil }
                product = next.partialValue
            }
            return product
        }
    }

    public struct HeadParameters {
        public let function: MLXArray
        public let scale: MLXArray
        public let base: MLXArray
        public let finalNorm: MLXArray

        fileprivate init(
            function: MLXArray,
            scale: MLXArray,
            base: MLXArray,
            finalNorm: MLXArray
        ) {
            self.function = function
            self.scale = scale
            self.base = base
            self.finalNorm = finalNorm
        }
    }

    public let unitID: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let geometry: Geometry
    public let manifestFileCount: Int
    public let manifestBytes: UInt64

    private enum TensorDType: String, Sendable {
        case bfloat16 = "BF16"
        case float32 = "F32"

        var byteWidth: UInt64 {
            switch self {
            case .bfloat16: return 2
            case .float32: return 4
            }
        }

        var mlxDType: DType {
            switch self {
            case .bfloat16: return .bfloat16
            case .float32: return .float32
            }
        }
    }

    private struct Document: Decodable {
        struct File: Decodable {
            struct Member: Decodable {
                let sourceTensor: String
                let dtype: String
                let shape: [Int]
                let offset: UInt64
                let bytes: UInt64

                enum CodingKeys: String, CodingKey {
                    case dtype, shape, offset, bytes
                    case sourceTensor = "source_tensor"
                }
            }

            let name: String
            let kind: String
            let bytes: UInt64
            let sha256: String?
            let sourceTensor: String?
            let dtype: String?
            let shape: [Int]?
            let members: [Member]?

            enum CodingKeys: String, CodingKey {
                case name, kind, bytes, sha256, dtype, shape, members
                case sourceTensor = "source_tensor"
            }
        }

        let unit: String
        let sourceRepository: String
        let sourceRevision: String
        let files: [File]

        enum CodingKeys: String, CodingKey {
            case unit, files
            case sourceRepository = "source_repo"
            case sourceRevision = "source_revision"
        }
    }

    private struct Record: Sendable {
        let tensor: String
        let reference: String
        let fileBytes: UInt64
        let dtype: TensorDType
        let shape: [Int]
        let offset: UInt64
        let bytes: UInt64
    }

    private let fileAccess: ModelFileAccess
    private let readAccounting: DeepSeekV4ReadAccounting?
    private let phaseAccounting: DeepSeekV4PhaseAccounting?
    private let diagnostics: DeepSeekV4Diagnostics
    private let records: [String: Record]
    private let headParameterLock = NSLock()
    private var admittedHeadParameters: HeadParameters?
    /// The dial's pinned tier, once a plan that pins the head has installed it.
    /// Nil for every run that does not, which is every run before this ladder
    /// had a globals rung.
    private var pinnedTier: DeepSeekV4PinnedWeightCache?
    /// The whole `[vocabulary, hidden]` BF16 head table, held in the **stored**
    /// dtype so ``headRows(from:count:)`` can widen per window exactly as the
    /// reading path does. See ``pinOutputHead(into:cancellationCheck:)``.
    private var residentHead: MLXArray?
    /// The table under construction, and how many of its rows are in it. The
    /// walk is sequential, so a contiguous watermark is the whole bookkeeping.
    private var headFillBuffer: UnsafeMutableRawPointer?
    private var headFilledRows = 0
    private let residentHeadLock = NSLock()

    // MARK: - The dial's census of this unit

    /// The two `global00` tables, in the two columns the memory dial ranks by,
    /// read from the published manifest alone.
    ///
    /// Metadata only: this opens no descriptor and reads no payload byte, so a
    /// screen can price a budget for an artifact that is on disk and not
    /// running — the same property `DeepSeekV4LayerArtifact.manifestCensus`
    /// gives the layer table.
    public struct ManifestCensus: Sendable, Equatable {
        /// `[vocabulary, hidden]` BF16. Read in full on every pass, and the
        /// residency pinning it costs — the two are equal, because the pinned
        /// form is the stored form and the widening stays per window.
        public let headTableBytes: UInt64
        /// One head row. The granularity the walk reads in.
        public let headRowBytes: UInt64
        /// `[vocabulary, hidden]` BF16 again — the same size as the head, which
        /// is exactly why it is not a rung.
        public let embeddingTableBytes: UInt64
        /// One embedding row: what a decode token actually reads of that table.
        public let embeddingRowBytes: UInt64
    }

    /// Read this unit's dial census from its manifest, without opening it.
    public static func manifestCensus(
        manifestData: Data, unitReference: String
    ) throws -> ManifestCensus {
        let document: Document
        do {
            document = try JSONDecoder().decode(Document.self, from: manifestData)
        } catch {
            throw DeepSeekV4Error.artifact("manifest.json could not be decoded: \(error)")
        }
        guard isSafeComponent(unitReference), document.unit == unitReference else {
            throw DeepSeekV4Error.artifact(
                "manifest unit '\(document.unit)' does not name rooted unit '\(unitReference)'")
        }

        var tables: [String: (bytes: UInt64, rowBytes: UInt64)] = [:]
        for file in document.files {
            var candidates: [(tensor: String, dtype: String, shape: [Int], bytes: UInt64)] = []
            if let members = file.members {
                candidates = members.map {
                    ($0.sourceTensor, $0.dtype, $0.shape, $0.bytes)
                }
            } else if let tensor = file.sourceTensor, let dtype = file.dtype,
                let shape = file.shape
            {
                candidates = [(tensor, dtype, shape, file.bytes)]
            }
            for candidate in candidates
            where candidate.tensor == "head.weight" || candidate.tensor == "embed.weight" {
                guard let dtype = TensorDType(rawValue: candidate.dtype), dtype == .bfloat16,
                    candidate.shape.count == 2, candidate.shape.allSatisfy({ $0 > 0 })
                else {
                    throw DeepSeekV4Error.artifact(
                        "\(candidate.tensor) is not a row-addressable BF16 table")
                }
                let rowBytes = UInt64(candidate.shape[1]).multipliedReportingOverflow(
                    by: dtype.byteWidth)
                let total = rowBytes.partialValue.multipliedReportingOverflow(
                    by: UInt64(candidate.shape[0]))
                guard !rowBytes.overflow, !total.overflow,
                    total.partialValue == candidate.bytes
                else {
                    throw DeepSeekV4Error.artifact(
                        "\(candidate.tensor) byte length does not match its declared shape")
                }
                tables[candidate.tensor] = (candidate.bytes, rowBytes.partialValue)
            }
        }
        guard let head = tables["head.weight"], let embedding = tables["embed.weight"] else {
            throw DeepSeekV4Error.artifact(
                "\(unitReference) does not declare both global tables")
        }
        return ManifestCensus(
            headTableBytes: head.bytes,
            headRowBytes: head.rowBytes,
            embeddingTableBytes: embedding.bytes,
            embeddingRowBytes: embedding.rowBytes)
    }

    /// The same census, off an artifact that is already open.
    public var census: ManifestCensus {
        let head = records["head.weight"]!
        let embedding = records["embed.weight"]!
        return ManifestCensus(
            headTableBytes: head.bytes,
            headRowBytes: (try? Self.rowBytes(head)).map(UInt64.init) ?? 0,
            embeddingTableBytes: embedding.bytes,
            embeddingRowBytes: (try? Self.rowBytes(embedding)).map(UInt64.init) ?? 0)
    }

    public convenience init(
        manifestURL: URL,
        geometry: Geometry,
        expectedSourceRepository: String? = nil,
        expectedSourceRevision: String? = nil
    ) throws {
        let data = try Data(contentsOf: manifestURL)
        let directory = manifestURL.deletingLastPathComponent()
        try self.init(
            manifestData: data,
            unitReference: directory.lastPathComponent,
            filesystemDirectory: directory,
            fileAccess: .filesystem,
            geometry: geometry,
            expectedSourceRepository: expectedSourceRepository,
            expectedSourceRevision: expectedSourceRevision)
    }

    public convenience init(
        manifestData: Data,
        unitReference: String,
        fileAccess: ModelFileAccess,
        geometry: Geometry,
        readAccounting: DeepSeekV4ReadAccounting? = nil,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating,
        expectedSourceRepository: String? = nil,
        expectedSourceRevision: String? = nil
    ) throws {
        try self.init(
            manifestData: manifestData,
            unitReference: unitReference,
            filesystemDirectory: nil,
            fileAccess: fileAccess,
            geometry: geometry,
            readAccounting: readAccounting,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics,
            expectedSourceRepository: expectedSourceRepository,
            expectedSourceRevision: expectedSourceRevision)
    }

    private init(
        manifestData: Data,
        unitReference: String,
        filesystemDirectory: URL?,
        fileAccess: ModelFileAccess,
        geometry: Geometry,
        readAccounting: DeepSeekV4ReadAccounting? = nil,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating,
        expectedSourceRepository: String?,
        expectedSourceRevision: String?
    ) throws {
        let document: Document
        do {
            document = try JSONDecoder().decode(Document.self, from: manifestData)
        } catch {
            throw DeepSeekV4Error.artifact(
                "global manifest.json could not be decoded: \(error)")
        }
        guard Self.isSafeComponent(unitReference), unitReference == "global00",
            document.unit == unitReference
        else {
            throw DeepSeekV4Error.artifact(
                "global manifest unit '\(document.unit)' does not name rooted unit '\(unitReference)'")
        }
        guard !document.sourceRepository.isEmpty, !document.sourceRevision.isEmpty else {
            throw DeepSeekV4Error.artifact(
                "global source repository and revision must be non-empty")
        }
        if let expectedSourceRepository,
            document.sourceRepository != expectedSourceRepository
        {
            throw DeepSeekV4Error.artifact(
                "global source repository does not match the artifact index")
        }
        if let expectedSourceRevision,
            document.sourceRevision != expectedSourceRevision
        {
            throw DeepSeekV4Error.artifact(
                "global source revision does not match the artifact index")
        }
        guard !document.files.isEmpty else {
            throw DeepSeekV4Error.artifact("global manifest contains no payload files")
        }

        var fileNames = Set<String>()
        var tensorNames = Set<String>()
        var records: [String: Record] = [:]
        var manifestBytes: UInt64 = 0
        for file in document.files {
            guard Self.isSafeComponent(file.name), fileNames.insert(file.name).inserted else {
                throw DeepSeekV4Error.artifact(
                    "global manifest contains an unsafe or repeated file name '\(file.name)'")
            }
            guard file.kind == "blob", file.bytes > 0,
                let digest = file.sha256, Self.isLowercaseSHA256(digest)
            else {
                throw DeepSeekV4Error.artifact(
                    "\(file.name) is not a non-empty blob with canonical SHA-256")
            }
            let nextManifestBytes = manifestBytes.addingReportingOverflow(file.bytes)
            guard !nextManifestBytes.overflow else {
                throw DeepSeekV4Error.artifact(
                    "global manifest payload byte total exceeds UInt64.max")
            }
            manifestBytes = nextManifestBytes.partialValue
            let reference = filesystemDirectory?.appendingPathComponent(file.name).path
                ?? "\(unitReference)/\(file.name)"
            try fileAccess.withDescriptor(reference) { descriptor in
                let actual = try Self.fileLength(descriptor, reference: reference)
                guard actual == file.bytes else {
                    throw DeepSeekV4Error.artifact(
                        "\(file.name) is \(actual) bytes; manifest says \(file.bytes)")
                }
            }

            if let members = file.members {
                guard file.sourceTensor == nil, file.dtype == nil, file.shape == nil,
                    !members.isEmpty
                else {
                    throw DeepSeekV4Error.artifact(
                        "\(file.name) mixes bundled and standalone tensor metadata")
                }
                var expectedOffset: UInt64 = 0
                for member in members {
                    guard member.offset == expectedOffset else {
                        throw DeepSeekV4Error.artifact(
                            "\(file.name) members are not a gapless partition at byte \(expectedOffset)")
                    }
                    let record = try Self.record(
                        tensor: member.sourceTensor,
                        dtypeName: member.dtype,
                        shape: member.shape,
                        reference: reference,
                        fileBytes: file.bytes,
                        offset: member.offset,
                        bytes: member.bytes,
                        seen: &tensorNames)
                    records[record.tensor] = record
                    let end = expectedOffset.addingReportingOverflow(member.bytes)
                    guard !end.overflow else {
                        throw DeepSeekV4Error.artifact(
                            "\(file.name) member partition overflows UInt64")
                    }
                    expectedOffset = end.partialValue
                }
                guard expectedOffset == file.bytes else {
                    throw DeepSeekV4Error.artifact(
                        "\(file.name) members describe \(expectedOffset) of \(file.bytes) bytes")
                }
            } else {
                guard let tensor = file.sourceTensor, let dtype = file.dtype,
                    let shape = file.shape
                else {
                    throw DeepSeekV4Error.artifact(
                        "\(file.name) has neither members nor standalone tensor metadata")
                }
                let record = try Self.record(
                    tensor: tensor,
                    dtypeName: dtype,
                    shape: shape,
                    reference: reference,
                    fileBytes: file.bytes,
                    offset: 0,
                    bytes: file.bytes,
                    seen: &tensorNames)
                records[record.tensor] = record
            }
        }

        let flattened = geometry.hyperConnectionMultiplicity.multipliedReportingOverflow(
            by: geometry.hiddenSize)
        guard !flattened.overflow else {
            throw DeepSeekV4Error.artifact("hyper-head width overflows Int")
        }
        try Self.require(records, "embed.weight", .bfloat16, [
            geometry.vocabularySize, geometry.hiddenSize,
        ])
        try Self.require(records, "head.weight", .bfloat16, [
            geometry.vocabularySize, geometry.hiddenSize,
        ])
        try Self.require(records, "hc_head_base", .float32, [
            geometry.hyperConnectionMultiplicity,
        ])
        try Self.require(records, "hc_head_fn", .float32, [
            geometry.hyperConnectionMultiplicity, flattened.partialValue,
        ])
        try Self.require(records, "hc_head_scale", .float32, [1])
        try Self.require(records, "norm.weight", .bfloat16, [geometry.hiddenSize])

        self.unitID = document.unit
        self.sourceRepository = document.sourceRepository
        self.sourceRevision = document.sourceRevision
        self.geometry = geometry
        self.manifestFileCount = document.files.count
        self.manifestBytes = manifestBytes
        self.fileAccess = fileAccess
        self.readAccounting = readAccounting
        self.phaseAccounting = phaseAccounting
        self.diagnostics = diagnostics
        self.records = records
    }

    /// Read only the named embedding rows, in request order.
    public func embeddingRows(
        _ tokenIDs: [Int],
        cancellationCheck: () throws -> Void = {}
    ) throws -> MLXArray {
        guard !tokenIDs.isEmpty else {
            throw DeepSeekV4Error.artifact("embedding request contains no token ids")
        }
        for token in tokenIDs where token < 0 || token >= geometry.vocabularySize {
            throw DeepSeekV4Error.artifact(
                "token id \(token) is outside 0..<\(geometry.vocabularySize)")
        }
        let record = records["embed.weight"]!
        return try readRows(
            record, indices: tokenIDs, cancellationCheck: cancellationCheck)
    }

    /// Read one bounded consecutive range of output-head rows — or serve it from
    /// the pinned table, when the dial pinned one.
    ///
    /// ## Why the pinned form is BF16 and the widening stays here
    ///
    /// The reading path allocates the window's stored BF16 bytes, wraps them in
    /// an `MLXArray`, and widens **that window** to float32 before returning it;
    /// `logits` then does one matmul per window against the widened rows. The
    /// pinned path slices the same rows out of the resident BF16 table and
    /// widens the slice, so every window sees the same values in the same dtype,
    /// each matmul consumes the same operand, and the reduction order over the
    /// vocabulary is unchanged. The logits are therefore bit-identical, not
    /// merely close — which is asserted on the synthetic fixture rather than
    /// argued here.
    ///
    /// Holding the widened float32 table instead would double the residency to
    /// 2.12 GB and drop the rung's ratio from 1.000 to 0.500, below every
    /// deterministic layer — and it would change nothing about the arithmetic
    /// except which allocation pays for the widening.
    public func headRows(
        from first: Int,
        count: Int,
        cancellationCheck: () throws -> Void = {}
    ) throws -> MLXArray {
        guard first >= 0, count > 0,
            count <= geometry.maximumHeadRowsPerRead
        else {
            throw DeepSeekV4Error.artifact(
                "head row request must be positive and no larger than "
                    + "\(geometry.maximumHeadRowsPerRead)")
        }
        let end = first.addingReportingOverflow(count)
        guard !end.overflow, end.partialValue <= geometry.vocabularySize else {
            throw DeepSeekV4Error.artifact(
                "head rows \(first)..<\(end.overflow ? Int.max : end.partialValue) exceed the vocabulary")
        }
        let record = records["head.weight"]!
        let windowBytes = UInt64(try Self.rowBytes(record)).multipliedReportingOverflow(
            by: UInt64(count))
        // Precedence: pinned, then read — the same order and the same shape as
        // the layer tier's two doors. The bounds above are rechecked whether or
        // not the tier serves, so a pinned table cannot outlive the geometry it
        // was admitted against.
        if let resident = residentHeadTable {
            if !windowBytes.overflow {
                pinnedTier?.noteOutputHeadServed(readBytes: windowBytes.partialValue)
                readAccounting?.recordPinnedServed(windowBytes.partialValue)
                phaseAccounting?.recordPinnedServed()
            }
            let widened = resident[first..<end.partialValue].asType(.float32)
            // A pinned head window still forces the graph: the slice is a fresh
            // graph node even though its bytes were never read. Deferred, and
            // the pin is exactly why it can be — the source is an MLX-owned
            // resident table, not a `malloc`'d read buffer with a deallocating
            // finalizer, so nothing is freed underneath the GPU. The matmul
            // that consumes this window ends in a blocking pull one statement
            // later, so submitting the widen early only lets the GPU start it
            // while the CPU builds that matmul.
            submittingToGPU(phaseAccounting, .weightMaterialisation) {
                MLX.asyncEval([widened])
            }
            return widened
        }
        return try readRowRange(
            record, first: first, count: count,
            cancellationCheck: cancellationCheck,
            // The table fills one window at a time, out of the windows the walk
            // already reads, so the filling pass reads exactly what an unpinned
            // pass reads and every later pass reads none of it. That is the
            // layer tier's fill rule applied to a table: bytes are counted as
            // loaded by the pass that read them and as served only afterwards.
            captureRawWindow: { raw in
                self.captureHeadWindow(raw, first: first, count: count, record: record)
            })
    }

    /// Hold the whole output-head table resident for the life of this artifact.
    ///
    /// Nothing is allocated here. Like the layer tier, the table fills lazily
    /// out of the reads the walk performs anyway; a run that never produces a
    /// logit never pays for a pin it did not use.
    func installPinnedTier(_ cache: DeepSeekV4PinnedWeightCache) {
        residentHeadLock.lock()
        if pinnedTier == nil, cache.pinsOutputHead { pinnedTier = cache }
        residentHeadLock.unlock()
    }

    /// Free the pinned table. Idempotent; the tier's counters survive it.
    func releasePinnedOutputHead() {
        residentHeadLock.lock()
        let held = residentHead != nil || headFillBuffer != nil
        residentHead = nil
        headFillBuffer?.deallocate()
        headFillBuffer = nil
        headFilledRows = 0
        let cache = pinnedTier
        residentHeadLock.unlock()
        if held { cache?.noteOutputHeadReleased() }
    }

    private var residentHeadTable: MLXArray? {
        residentHeadLock.lock()
        defer { residentHeadLock.unlock() }
        return residentHead
    }

    /// Copy one just-read window into the resident table, in walk order.
    ///
    /// The walk is strictly sequential from row 0, so a contiguous watermark is
    /// all the bookkeeping this needs: a window that does not continue the fill
    /// is ignored rather than remembered, and the table only becomes servable
    /// once every row is in it. Until then every window is read, which is what
    /// an unpinned run does.
    private func captureHeadWindow(
        _ raw: UnsafeRawBufferPointer, first: Int, count: Int, record: Record
    ) {
        guard let rowBytes = try? Self.rowBytes(record) else { return }
        residentHeadLock.lock()
        defer { residentHeadLock.unlock() }
        guard let cache = pinnedTier, residentHead == nil, headFilledRows == first,
            let tableBytes = Int(exactly: record.bytes),
            raw.count == rowBytes * count, let base = raw.baseAddress
        else { return }
        if headFillBuffer == nil {
            guard cache.admitOutputHead(residentBytes: record.bytes) else {
                // Refused and counted. This artifact does not try again: a
                // refusal is a statement about the plan's budget, not a
                // transient condition.
                pinnedTier = nil
                return
            }
            headFillBuffer = UnsafeMutableRawPointer.allocate(
                byteCount: tableBytes, alignment: 16)
        }
        guard let buffer = headFillBuffer else { return }
        buffer.advanced(by: first * rowBytes).copyMemory(from: base, byteCount: raw.count)
        headFilledRows = first + count
        cache.noteOutputHeadLoaded(readBytes: UInt64(raw.count))
        guard headFilledRows == geometry.vocabularySize else { return }
        // The whole table is in the buffer; hand it to MLX, which owns it from
        // here and frees it through the finalizer when the array dies.
        headFillBuffer = nil
        residentHead = MLXArray(
            rawPointer: buffer,
            [geometry.vocabularySize, geometry.hiddenSize],
            dtype: .bfloat16,
            finalizer: { buffer.deallocate() })
    }

    /// Load the small hyper-head and final-norm tensors. The two large tables
    /// remain row-addressed and are never retained by this value.
    ///
    /// These four members are a few hundred kilobytes of immutable checkpoint
    /// data that every token needs in full, so the first successful load and
    /// finiteness scan is retained for this artifact's lifetime — the run —
    /// instead of re-reading and re-scanning them per token. The row-addressed
    /// `head.weight` and `embed.weight` walks are unchanged and still retain
    /// nothing.
    public func loadHeadParameters(
        cancellationCheck: () throws -> Void = {}
    ) throws -> HeadParameters {
        headParameterLock.lock()
        let admitted = admittedHeadParameters
        headParameterLock.unlock()
        if let admitted { return admitted }
        let parameters = try readHeadParameters(cancellationCheck: cancellationCheck)
        headParameterLock.lock()
        admittedHeadParameters = parameters
        headParameterLock.unlock()
        return parameters
    }

    private func readHeadParameters(
        cancellationCheck: () throws -> Void
    ) throws -> HeadParameters {
        let function = try load(
            records["hc_head_fn"]!, cancellationCheck: cancellationCheck)
        let scale = try load(
            records["hc_head_scale"]!, cancellationCheck: cancellationCheck)
        let base = try load(
            records["hc_head_base"]!, cancellationCheck: cancellationCheck)
        let finalNorm = try load(
            records["norm.weight"]!, cancellationCheck: cancellationCheck)
        // Admission-time, and memoized by `loadHeadParameters`: this sweep runs
        // on the first pass of a run and never again, which is why the census
        // shows it on pass 1 and nowhere in the plateau.
        // No explicit `eval` here: the `asArray` below is the force, so the
        // whole loop is the wait and the bracket is the whole loop.
        try waitingForGPU(phaseAccounting, .finitenessSweep) {
            for (name, array) in [
                ("hc_head_fn", function), ("hc_head_scale", scale),
                ("hc_head_base", base), ("norm.weight", finalNorm),
            ] {
                guard array.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite) else {
                    throw DeepSeekV4Error.artifact("\(name) contains a non-finite value")
                }
            }
        }
        return HeadParameters(
            function: function, scale: scale, base: base, finalNorm: finalNorm)
    }

    /// Apply the learned final stream contraction and model RMS normalization.
    public func finalHidden(
        from residual: MLXArray,
        parameters: HeadParameters
    ) throws -> MLXArray {
        let collapsed = try DeepSeekV4HyperConnections.collapseHead(
            residual: residual,
            function: parameters.function,
            scale: parameters.scale,
            base: parameters.base,
            multiplicity: geometry.hyperConnectionMultiplicity,
            epsilon: geometry.hyperConnectionEpsilon,
            normEpsilon: geometry.rmsNormEpsilon,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        let normalized = K3Norm.rms(
            collapsed, weight: parameters.finalNorm,
            eps: geometry.rmsNormEpsilon).asType(.float32)
        // Deferred: the head walk's first window pull is the wait, and it is
        // one statement away. Nothing reads a host value from `normalized`.
        submittingToGPU(phaseAccounting, .passBoundary) {
            MLX.asyncEval([normalized])
        }
        return normalized
    }

    /// Project one final hidden row over the vocabulary in bounded row chunks.
    public func logits(
        for hidden: MLXArray,
        chunkRows: Int? = nil,
        cancellationCheck: () throws -> Void = {},
        boundaryReclaim: DeepSeekV4BoundaryReclaimPolicy = DeepSeekV4BoundaryReclaimPolicy()
    ) throws -> [Float] {
        guard hidden.ndim == 2, hidden.shape == [1, geometry.hiddenSize] else {
            throw DeepSeekV4Error.artifact(
                "final hidden must be [1, \(geometry.hiddenSize)], got \(hidden.shape)")
        }
        let window = chunkRows ?? geometry.maximumHeadRowsPerRead
        guard window > 0, window <= geometry.maximumHeadRowsPerRead else {
            throw DeepSeekV4Error.artifact(
                "logit row window must be inside 1...\(geometry.maximumHeadRowsPerRead)")
        }
        var values = [Float]()
        values.reserveCapacity(geometry.vocabularySize)
        var first = 0
        while first < geometry.vocabularySize {
            try cancellationCheck()
            let count = min(window, geometry.vocabularySize - first)
            let part: [Float] = try autoreleasepool {
                // Two brackets, disjoint by construction: the window's read
                // and widening, then the matmul that consumes it. This walk is
                // strictly serial — window n+1's read cannot start until
                // window n's matmul has been forced — so the split is what
                // says whether overlapping them is worth building.
                let rows = try measuringPhase(
                    phaseAccounting,
                    excludingGPUBoundaryFrom: phaseAccounting?.recordOutputHeadRead(nanoseconds:)
                ) {
                    try headRows(
                        from: first, count: count,
                        cancellationCheck: cancellationCheck)
                }
                return measuringPhase(
                    phaseAccounting,
                    excludingGPUBoundaryFrom: phaseAccounting?
                        .recordOutputHeadCompute(nanoseconds:)
                ) {
                    let projected = matmul(
                        hidden.asType(.float32), rows.transposed(1, 0)).asType(.float32)
                    // One sync per window, and the wait for it is
                    // `gpuWaitSeconds` rather than head compute: this walk is
                    // serial, so window n's wait is the head's own matmul
                    // draining and nothing else — but it is still a wait, and
                    // the term that names the matmul should name the matmul.
                    // `asArray` copies an array the `eval` already materialised.
                    waitingForGPU(phaseAccounting, .outputHead) {
                        projected.eval()
                    }
                    return projected.asArray(Float.self)
                }
            }
            guard part.count == count, part.allSatisfy(\.isFinite) else {
                throw DeepSeekV4Error.artifact(
                    "output head produced missing or non-finite logits")
            }
            values.append(contentsOf: part)
            // `part` is the only surviving value from the autorelease pool.
            // The tens-of-megabytes raw/widened row window has been evaluated
            // and released, so return its empty allocator pages before the
            // next window rather than carrying output-head high water into the
            // next token. A run with a non-zero MLX cache limit holds those
            // pages for the next window instead, until its stated margin.
            boundaryReclaim.reclaimAtBoundary(recordingTo: phaseAccounting)
            first += count
        }
        try cancellationCheck()
        return values
    }

    private func readRows(
        _ record: Record,
        indices: [Int],
        cancellationCheck: () throws -> Void
    ) throws -> MLXArray {
        let rowBytes = try Self.rowBytes(record)
        let allocationBytes = rowBytes.multipliedReportingOverflow(by: indices.count)
        guard !allocationBytes.overflow else {
            throw DeepSeekV4Error.artifact("embedding row buffer overflows Int")
        }
        let pointer = UnsafeMutableRawPointer.allocate(
            byteCount: allocationBytes.partialValue, alignment: 16)
        var transferred = false
        defer { if !transferred { pointer.deallocate() } }
        try fileAccess.withDescriptor(record.reference) { descriptor in
            try Self.requireUnchangedLength(
                descriptor, record: record)
            for (position, row) in indices.enumerated() {
                let source = try Self.rowOffset(record, row: row)
                let destination = rowBytes.multipliedReportingOverflow(by: position)
                guard !destination.overflow else {
                    throw DeepSeekV4Error.artifact("embedding destination offset overflows Int")
                }
                try Self.readExactly(
                    descriptor: descriptor,
                    reference: record.reference,
                    offset: source,
                    length: rowBytes,
                    destination: pointer + destination.partialValue,
                    cancellationCheck: cancellationCheck,
                    readAccounting: readAccounting)
            }
        }
        let raw = MLXArray(
            rawPointer: pointer,
            [indices.count, geometry.hiddenSize],
            dtype: .bfloat16,
            finalizer: { pointer.deallocate() })
        transferred = true
        let widened = raw.asType(.float32)
        waitingForGPU(phaseAccounting, .embedding) {
            widened.eval()
        }
        return widened
    }

    /// `captureRawWindow` sees the window's **stored** BF16 bytes, after the
    /// read and before the widening. It is how the pinned table is filled
    /// without a second read and without touching the value this returns.
    private func readRowRange(
        _ record: Record,
        first: Int,
        count: Int,
        cancellationCheck: () throws -> Void,
        captureRawWindow: ((UnsafeRawBufferPointer) -> Void)? = nil
    ) throws -> MLXArray {
        let rowBytes = try Self.rowBytes(record)
        let length = rowBytes.multipliedReportingOverflow(by: count)
        guard !length.overflow else {
            throw DeepSeekV4Error.artifact("head row buffer overflows Int")
        }
        let pointer = UnsafeMutableRawPointer.allocate(
            byteCount: length.partialValue, alignment: 16)
        var transferred = false
        defer { if !transferred { pointer.deallocate() } }
        try fileAccess.withDescriptor(record.reference) { descriptor in
            try Self.requireUnchangedLength(descriptor, record: record)
            try Self.readExactly(
                descriptor: descriptor,
                reference: record.reference,
                offset: try Self.rowOffset(record, row: first),
                length: length.partialValue,
                destination: pointer,
                cancellationCheck: cancellationCheck,
                readAccounting: readAccounting)
        }
        captureRawWindow?(
            UnsafeRawBufferPointer(start: pointer, count: length.partialValue))
        let raw = MLXArray(
            rawPointer: pointer,
            [count, geometry.hiddenSize],
            dtype: .bfloat16,
            finalizer: { pointer.deallocate() })
        transferred = true
        let widened = raw.asType(.float32)
        waitingForGPU(phaseAccounting, .weightMaterialisation) {
            widened.eval()
        }
        return widened
    }

    private func load(
        _ record: Record,
        cancellationCheck: () throws -> Void
    ) throws -> MLXArray {
        guard let length = Int(exactly: record.bytes) else {
            throw DeepSeekV4Error.artifact(
                "\(record.tensor) byte length is not representable by this process")
        }
        let pointer = UnsafeMutableRawPointer.allocate(byteCount: length, alignment: 16)
        var transferred = false
        defer { if !transferred { pointer.deallocate() } }
        try fileAccess.withDescriptor(record.reference) { descriptor in
            try Self.requireUnchangedLength(descriptor, record: record)
            try Self.readExactly(
                descriptor: descriptor,
                reference: record.reference,
                offset: record.offset,
                length: length,
                destination: pointer,
                cancellationCheck: cancellationCheck,
                readAccounting: readAccounting)
        }
        let raw = MLXArray(
            rawPointer: pointer, record.shape, dtype: record.dtype.mlxDType,
            finalizer: { pointer.deallocate() })
        transferred = true
        waitingForGPU(phaseAccounting, .weightMaterialisation) {
            raw.eval()
        }
        return raw
    }

    private static func record(
        tensor: String,
        dtypeName: String,
        shape: [Int],
        reference: String,
        fileBytes: UInt64,
        offset: UInt64,
        bytes: UInt64,
        seen: inout Set<String>
    ) throws -> Record {
        guard !tensor.isEmpty, !tensor.contains("\0"), seen.insert(tensor).inserted else {
            throw DeepSeekV4Error.artifact(
                "global tensor '\(tensor)' is empty or appears more than once")
        }
        guard let dtype = TensorDType(rawValue: dtypeName),
            !shape.isEmpty, shape.allSatisfy({ $0 > 0 })
        else {
            throw DeepSeekV4Error.artifact(
                "\(tensor) has unsupported dtype or invalid shape")
        }
        var elements: UInt64 = 1
        for dimension in shape {
            let product = elements.multipliedReportingOverflow(by: UInt64(dimension))
            guard !product.overflow else {
                throw DeepSeekV4Error.artifact("\(tensor) shape overflows UInt64")
            }
            elements = product.partialValue
        }
        let byteCount = elements.multipliedReportingOverflow(by: dtype.byteWidth)
        guard !byteCount.overflow, byteCount.partialValue == bytes else {
            throw DeepSeekV4Error.artifact(
                "\(tensor) byte length does not match \(dtype.rawValue)\(shape)")
        }
        let end = offset.addingReportingOverflow(bytes)
        guard !end.overflow, end.partialValue <= fileBytes else {
            throw DeepSeekV4Error.artifact(
                "\(tensor) range runs past its \(fileBytes)-byte blob")
        }
        return Record(
            tensor: tensor, reference: reference, fileBytes: fileBytes,
            dtype: dtype, shape: shape, offset: offset, bytes: bytes)
    }

    private static func require(
        _ records: [String: Record],
        _ tensor: String,
        _ dtype: TensorDType,
        _ shape: [Int]
    ) throws {
        guard let record = records[tensor],
            record.dtype == dtype, record.shape == shape
        else {
            throw DeepSeekV4Error.artifact(
                "global manifest has no required \(dtype.rawValue)\(shape) tensor '\(tensor)'")
        }
    }

    private static func rowBytes(_ record: Record) throws -> Int {
        guard record.dtype == .bfloat16, record.shape.count == 2 else {
            throw DeepSeekV4Error.artifact(
                "\(record.tensor) is not a row-addressable BF16 table")
        }
        let bytes = record.shape[1].multipliedReportingOverflow(by: 2)
        guard !bytes.overflow else {
            throw DeepSeekV4Error.artifact("\(record.tensor) row size overflows Int")
        }
        return bytes.partialValue
    }

    private static func rowOffset(_ record: Record, row: Int) throws -> UInt64 {
        guard row >= 0, row < record.shape[0] else {
            throw DeepSeekV4Error.artifact(
                "row \(row) is outside \(record.tensor)'s \(record.shape[0]) rows")
        }
        let rowBytes = try Self.rowBytes(record)
        let relative = UInt64(row).multipliedReportingOverflow(by: UInt64(rowBytes))
        let absolute = record.offset.addingReportingOverflow(relative.partialValue)
        guard !relative.overflow, !absolute.overflow else {
            throw DeepSeekV4Error.artifact("\(record.tensor) row offset overflows UInt64")
        }
        return absolute.partialValue
    }

    private static func requireUnchangedLength(
        _ descriptor: Int32,
        record: Record
    ) throws {
        let actual = try fileLength(descriptor, reference: record.reference)
        guard actual == record.fileBytes else {
            throw DeepSeekV4Error.artifact(
                "\(record.reference) changed length after manifest reconciliation")
        }
    }

    private static func readExactly(
        descriptor: Int32,
        reference: String,
        offset: UInt64,
        length: Int,
        destination: UnsafeMutableRawPointer,
        cancellationCheck: () throws -> Void,
        readAccounting: DeepSeekV4ReadAccounting?
    ) throws {
        guard length > 0, let start = off_t(exactly: offset) else {
            throw DeepSeekV4Error.artifact(
                "\(reference) has an unrepresentable read range")
        }
        var completed = 0
        let chunk = 4 * 1_024 * 1_024
        while completed < length {
            try cancellationCheck()
            let take = min(chunk, length - completed)
            let position = start.addingReportingOverflow(off_t(completed))
            guard !position.overflow else {
                throw DeepSeekV4Error.artifact("\(reference) read offset overflows off_t")
            }
            let result = pread(
                descriptor, destination + completed, take, position.partialValue)
            if result < 0 {
                if errno == EINTR { continue }
                throw StorageCoreError.posix(
                    operation: "pread", path: reference, code: errno)
            }
            guard result > 0 else {
                throw StorageCoreError.shortTransfer(
                    operation: "pread", path: reference,
                    offset: offset + UInt64(completed), expected: take, actual: 0)
            }
            readAccounting?.recordDeterministic(UInt64(result))
            completed += result
        }
        try cancellationCheck()
    }

    private static func fileLength(_ descriptor: Int32, reference: String) throws -> UInt64 {
        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            throw StorageCoreError.posix(
                operation: "fstat", path: reference, code: errno)
        }
        guard status.st_size >= 0 else {
            throw DeepSeekV4Error.artifact("\(reference) has a negative file length")
        }
        return UInt64(status.st_size)
    }

    private static func isSafeComponent(_ value: String) -> Bool {
        !value.isEmpty && value != "." && value != ".."
            && !value.contains("/") && !value.contains("\\") && !value.contains("\0")
    }

    private static func isLowercaseSHA256(_ value: String) -> Bool {
        value.count == 64 && value.utf8.allSatisfy { byte in
            (48...57).contains(byte) || (97...102).contains(byte)
        }
    }
}
