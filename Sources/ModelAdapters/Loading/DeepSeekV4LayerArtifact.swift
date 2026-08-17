import Darwin
import Foundation
import MLX
import MLXBridge
import StorageCore

/// A checked, lazy view of one published `layersNN` DeepSeek V4 unit.
///
/// Construction reconciles the public manifest with every container header and
/// file length, but deliberately does not make all matrices resident. Callers
/// load one non-expert matrix or one small plain member at a time. The product
/// supplies a ``ModelFileAccess`` backed by the fully verified artifact root;
/// no repository revision is compiled into this reader.
public final class DeepSeekV4LayerArtifact: @unchecked Sendable {
    public enum PlainDType: String, Sendable, Equatable {
        case bfloat16 = "BF16"
        case float32 = "F32"
        case int64 = "I64"

        fileprivate var byteWidth: Int {
            switch self {
            case .bfloat16: return 2
            case .float32: return 4
            case .int64: return 8
            }
        }

        fileprivate var mlxDType: DType? {
            switch self {
            case .bfloat16: return .bfloat16
            case .float32: return .float32
            case .int64: return nil
            }
        }
    }

    public let unitID: String
    public let layer: Int
    public let sourceRepository: String
    public let sourceRevision: String
    public let manifestFileCount: Int
    public let manifestBytes: UInt64

    /// One plain-tensor requirement used by a model block's metadata-only
    /// preflight. This stays internal to ModelAdapters: artifact manifests are
    /// public data, but which tensors make one architecture executable belongs
    /// to the model adapter rather than to the container API.
    struct PlainTensorExpectation: Sendable, Equatable {
        let dtype: PlainDType
        let shape: [Int]
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
            let rows: Int?
            let cols: Int?
            let count: Int?
            let elementBits: Int?
            let group: Int?
            let blockRows: Int?
            let layout: String?
            let sourceTensor: String?
            let dtype: String?
            let shape: [Int]?
            let sha256: String?
            let members: [Member]?

            enum CodingKeys: String, CodingKey {
                case name, kind, bytes, rows, cols, count, group, layout, dtype, shape, sha256
                case elementBits = "element_bits"
                case blockRows = "block_rows"
                case sourceTensor = "source_tensor"
                case members
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

    private struct MatrixRecord: Sendable {
        let tensor: String
        let reference: String
        let bytes: UInt64
        let layout: QuantizedTileContainer.Layout
    }

    private struct PlainRecord: Sendable {
        let tensor: String
        let reference: String
        let fileBytes: UInt64
        let dtype: PlainDType
        let shape: [Int]
        let offset: UInt64
        let bytes: UInt64
    }

    private struct TokenExpertMapKey: Hashable {
        let tensor: String
        let vocabularySize: Int
        let expertsPerToken: Int
        let expertCount: Int
    }

    private let fileAccess: ModelFileAccess
    private let tileDigestPolicy: TileDigestPolicy
    private let readAccounting: DeepSeekV4ReadAccounting?
    /// Not private: the layer blocks reach it to bracket the operator work
    /// they perform with this artifact's tensors, so the brackets around
    /// scale syncs, routing and attention land in the same pass decomposition
    /// as the reads. Nothing outside this module can see it.
    let phaseAccounting: DeepSeekV4PhaseAccounting?
    /// Not private for the same reason ``phaseAccounting`` is not: the layer
    /// blocks read it here and hand it to the operators they run with this
    /// artifact's tensors, so one statement by the caller that opened the
    /// artifact reaches every guard-only sweep inside the layer.
    let diagnostics: DeepSeekV4Diagnostics
    private let matrices: [String: MatrixRecord]
    private let plain: [String: PlainRecord]
    private let staticStateLock = NSLock()
    private var tokenExpertMaps = [TokenExpertMapKey: DeepSeekV4TokenExpertMap]()
    private var admittedContracts = [AnyHashable: [AnyObject]]()
    /// Tensors whose packed bytes this run has already scanned for non-finite
    /// E4M3 codes. Only consulted under
    /// ``TileDigestPolicy/trustHeldAuthority(_:)``; see ``loadBlockFP8``.
    private var scannedPackedTensors = Set<String>()
    /// The memory dial's pinned tier, or `nil` when nothing is pinned — the
    /// product scale, and every V4 run before the dial existed. Installed after
    /// construction because the plan is a function of the census, and the
    /// census is a function of this object.
    private var pinnedWeights: DeepSeekV4PinnedWeightCache?

    /// Filesystem convenience for tests and command-line tools.
    public convenience init(
        manifestURL: URL,
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
            expectedSourceRepository: expectedSourceRepository,
            expectedSourceRevision: expectedSourceRevision)
    }

    /// Rooted product entry point. `unitReference` is one repository component,
    /// for example `layers00`; manifest-supplied names never become paths.
    ///
    /// `tileDigestPolicy` defaults to ``TileDigestPolicy/verifyEveryLoad``: a
    /// reader cannot invent trust in bytes it did not verify. The product
    /// runtime states ``TileDigestPolicy/trustHeldAuthority(_:)`` where it
    /// already holds complete verification authority for this exact revision.
    public convenience init(
        manifestData: Data,
        unitReference: String,
        fileAccess: ModelFileAccess,
        tileDigestPolicy: TileDigestPolicy = .verifyEveryLoad,
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
            tileDigestPolicy: tileDigestPolicy,
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
        tileDigestPolicy: TileDigestPolicy = .verifyEveryLoad,
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
            throw DeepSeekV4Error.artifact("manifest.json could not be decoded: \(error)")
        }
        guard Self.isSafeComponent(unitReference), document.unit == unitReference,
            let layer = Self.layerNumber(unitReference)
        else {
            throw DeepSeekV4Error.artifact(
                "manifest unit '\(document.unit)' does not name rooted unit '\(unitReference)'")
        }
        guard !document.sourceRepository.isEmpty, !document.sourceRevision.isEmpty else {
            throw DeepSeekV4Error.artifact(
                "manifest source repository and revision must be non-empty")
        }
        if let expectedSourceRepository,
            document.sourceRepository != expectedSourceRepository
        {
            throw DeepSeekV4Error.artifact(
                "source repository '\(document.sourceRepository)' does not match the artifact index")
        }
        if let expectedSourceRevision,
            document.sourceRevision != expectedSourceRevision
        {
            throw DeepSeekV4Error.artifact(
                "source revision '\(document.sourceRevision)' does not match the artifact index")
        }
        guard !document.files.isEmpty else {
            throw DeepSeekV4Error.artifact("manifest contains no payload files")
        }
        // Trust is admitted once, here, against what this unit actually
        // declares — never assumed at a load. Authority that names a different
        // revision is authority over different bytes, and an opener that only
        // reopens a pathname cannot tell whether the object it hands back is
        // still the verified one, which is the guard the digest is traded for.
        if case .trustHeldAuthority(let held) = tileDigestPolicy {
            guard fileAccess.identityAssurance == .rootedDescriptorIdentity else {
                throw DeepSeekV4Error.artifact(
                    "held verification authority requires a rooted file access that "
                        + "rechecks descriptor identity on every open")
            }
            guard held.verifiedFileCount > 0, held.verifiedBytes > 0 else {
                throw DeepSeekV4Error.artifact(
                    "held verification authority states no verified files or bytes")
            }
            guard held.sourceRepository == document.sourceRepository,
                held.sourceRevision == document.sourceRevision
            else {
                throw DeepSeekV4Error.artifact(
                    "held verification authority covers "
                        + "\(held.sourceRepository)@\(held.sourceRevision), not "
                        + "\(document.sourceRepository)@\(document.sourceRevision)")
            }
        }

        let sourcePrefix = "layers.\(layer)."
        var fileNames = Set<String>()
        var tensors = Set<String>()
        var matrices: [String: MatrixRecord] = [:]
        var plain: [String: PlainRecord] = [:]
        var manifestBytes: UInt64 = 0

        for file in document.files {
            guard Self.isSafeComponent(file.name), fileNames.insert(file.name).inserted else {
                throw DeepSeekV4Error.artifact(
                    "manifest contains an unsafe or repeated file name '\(file.name)'")
            }
            guard file.bytes > 0 else {
                throw DeepSeekV4Error.artifact("\(file.name) has a zero byte length")
            }
            let nextManifestBytes = manifestBytes.addingReportingOverflow(file.bytes)
            guard !nextManifestBytes.overflow else {
                throw DeepSeekV4Error.artifact(
                    "manifest payload byte total exceeds UInt64.max")
            }
            manifestBytes = nextManifestBytes.partialValue
            let reference = filesystemDirectory?.appendingPathComponent(file.name).path
                ?? "\(unitReference)/\(file.name)"

            switch file.kind {
            case "tile-container":
                let layout = try fileAccess.withDescriptor(reference) { descriptor in
                    let layout = try QuantizedTileContainer.open(
                        fileDescriptor: descriptor, path: reference)
                    let actual = try Self.fileLength(descriptor, reference: reference)
                    guard actual == layout.totalBytes, actual == file.bytes else {
                        throw DeepSeekV4Error.artifact(
                            "\(file.name) byte length disagrees with its header or manifest")
                    }
                    return layout
                }
                guard file.rows == layout.geometry.rows,
                    file.cols == layout.geometry.cols,
                    file.count == layout.tileCount,
                    file.elementBits == layout.geometry.bits,
                    file.group == layout.geometry.groupSize,
                    file.layout == layout.geometry.quantMode.name
                else {
                    throw DeepSeekV4Error.artifact(
                        "\(file.name) manifest geometry disagrees with its container header")
                }
                if let blockRows = file.blockRows,
                    blockRows != layout.geometry.scaleBlockRows
                {
                    throw DeepSeekV4Error.artifact(
                        "\(file.name) block_rows disagrees with its container header")
                }
                if let tensor = file.sourceTensor {
                    try Self.validateTensorName(
                        tensor, prefix: sourcePrefix, seen: &tensors)
                    guard layout.geometry.quantMode == .fp8E4M3Block,
                        layout.geometry.bits == 8,
                        layout.geometry.scaleDType == .uint8E8M0,
                        layout.geometry.scaleBlockRows == 128,
                        layout.geometry.groupSize == 128,
                        layout.tileCount == 1,
                        layout.contentKind == .checkpointDerived,
                        layout.tileDigests?.count == 1,
                        file.blockRows == 128
                    else {
                        throw DeepSeekV4Error.artifact(
                            "\(file.name) is not a single checkpoint-derived FP8 [128, 128] matrix")
                    }
                    matrices[tensor] = MatrixRecord(
                        tensor: tensor, reference: reference, bytes: file.bytes, layout: layout)
                }

            case "blob":
                guard let digest = file.sha256, Self.isLowercaseSHA256(digest) else {
                    throw DeepSeekV4Error.artifact(
                        "\(file.name) has no canonical lowercase SHA-256")
                }
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
                        let record = try Self.plainRecord(
                            tensor: member.sourceTensor,
                            dtypeName: member.dtype,
                            shape: member.shape,
                            reference: reference,
                            fileBytes: file.bytes,
                            offset: member.offset,
                            bytes: member.bytes,
                            sourcePrefix: sourcePrefix,
                            seen: &tensors)
                        plain[record.tensor] = record
                        let end = expectedOffset.addingReportingOverflow(member.bytes)
                        guard !end.overflow else {
                            throw DeepSeekV4Error.artifact(
                                "\(file.name) member byte partition overflows UInt64")
                        }
                        expectedOffset = end.partialValue
                    }
                    guard expectedOffset == file.bytes else {
                        throw DeepSeekV4Error.artifact(
                            "\(file.name) members describe \(expectedOffset) of \(file.bytes) bytes")
                    }
                } else {
                    guard let tensor = file.sourceTensor,
                        let dtype = file.dtype,
                        let shape = file.shape
                    else {
                        throw DeepSeekV4Error.artifact(
                            "\(file.name) has neither a member table nor standalone tensor metadata")
                    }
                    let record = try Self.plainRecord(
                        tensor: tensor, dtypeName: dtype, shape: shape,
                        reference: reference, fileBytes: file.bytes,
                        offset: 0, bytes: file.bytes,
                        sourcePrefix: sourcePrefix, seen: &tensors)
                    plain[record.tensor] = record
                }

            default:
                throw DeepSeekV4Error.artifact(
                    "\(file.name) uses unsupported payload kind '\(file.kind)'")
            }
        }
        guard !matrices.isEmpty, !plain.isEmpty else {
            throw DeepSeekV4Error.artifact(
                "a model layer must contain both non-expert FP8 matrices and plain tensors")
        }

        self.unitID = document.unit
        self.layer = layer
        self.sourceRepository = document.sourceRepository
        self.sourceRevision = document.sourceRevision
        self.manifestFileCount = document.files.count
        self.manifestBytes = manifestBytes
        self.fileAccess = fileAccess
        self.tileDigestPolicy = tileDigestPolicy
        self.readAccounting = readAccounting
        self.phaseAccounting = phaseAccounting
        self.diagnostics = diagnostics
        self.matrices = matrices
        self.plain = plain
    }

    /// Run one metadata-only contract admission the first time this exact
    /// request is made against this artifact.
    ///
    /// A block's `validateArtifactContract` is a pure function of this
    /// artifact's reconciled manifest — fixed at construction — and of the
    /// caller's `key`. Full-model admission runs it for every layer before the
    /// first payload read, and every prefill and decode entry runs the same
    /// boundary again; from the second call onward that repetition cannot
    /// reach a different answer. `key` must therefore carry everything the
    /// admission inspected outside this artifact, and the objects it names by
    /// identity are retained in `participants` so no identifier can be
    /// recycled underneath a recorded admission. A failed admission is not
    /// recorded, so a genuinely different request still fails on every call.
    func admitContractOnce<Key: Hashable>(
        key: Key,
        participants: [AnyObject],
        validate: () throws -> Void
    ) throws {
        let erased = AnyHashable(key)
        staticStateLock.lock()
        let admitted = admittedContracts[erased] != nil
        staticStateLock.unlock()
        if admitted { return }
        try validate()
        staticStateLock.lock()
        admittedContracts[erased] = participants
        staticStateLock.unlock()
    }

    /// Reconcile every tensor a computation will consume before the first
    /// payload read or routed-expert request.
    ///
    /// `loadBlockFP8` and `loadFloatingTensor` repeat their local checks at the
    /// point of use. This whole-contract pass is intentionally separate: a
    /// missing late projection must not be discovered after earlier matrices
    /// have been read or expert work has begun.
    func validateTensorContract(
        blockFP8 expectedMatrices: [String: [Int]],
        plain expectedPlain: [String: PlainTensorExpectation]
    ) throws {
        guard !expectedMatrices.isEmpty, !expectedPlain.isEmpty else {
            throw DeepSeekV4Error.artifact(
                "an executable layer contract must name matrices and plain tensors")
        }
        for (tensor, expectedShape) in expectedMatrices {
            guard expectedShape.count == 2,
                expectedShape.allSatisfy({ $0 > 0 }),
                let record = matrices[tensor]
            else {
                throw DeepSeekV4Error.artifact(
                    "manifest has no required FP8 tensor '\(tensor)'")
            }
            let actual = [record.layout.geometry.rows, record.layout.geometry.cols]
            guard actual == expectedShape else {
                throw DeepSeekV4Error.artifact(
                    "\(tensor) is \(actual), expected \(expectedShape)")
            }
        }
        for (tensor, expected) in expectedPlain {
            guard !expected.shape.isEmpty,
                expected.shape.allSatisfy({ $0 > 0 }),
                let record = plain[tensor]
            else {
                throw DeepSeekV4Error.artifact(
                    "manifest has no required plain tensor '\(tensor)'")
            }
            guard record.dtype == expected.dtype, record.shape == expected.shape else {
                throw DeepSeekV4Error.artifact(
                    "\(tensor) is \(record.dtype.rawValue)\(record.shape), expected "
                        + "\(expected.dtype.rawValue)\(expected.shape)")
            }
        }
    }

    /// Whether this artifact recomputes a tile's recorded SHA-256 on every
    /// load, or reads under the held verification authority it was constructed
    /// with. See ``TileDigestPolicy``.
    public var recomputesTileDigestOnEveryLoad: Bool {
        tileDigestPolicy.recomputesDigestOnEveryLoad
    }

    // MARK: - The memory dial's census

    /// Deterministic bytes **one pass reads** from this layer.
    ///
    /// Block-FP8 matrices are counted at one tile stride each, which is what
    /// ``loadBlockFP8`` reads — the container header is opened and reconciled
    /// but never read as payload, so the file length would over-count it. Plain
    /// BF16/F32 members are counted at their member byte length.
    ///
    /// Two units are deliberately absent. **Routed-expert tiles** are not
    /// deterministic — the router picks them per token and they have their own
    /// pool budget — and they never enter `matrices`, because the manifest
    /// registers a tile container as a matrix only when it names a
    /// `source_tensor`. **I64 tables** (the token→expert map) are excluded
    /// because ``loadTokenExpertMap`` already retains them for the life of the
    /// run: a unit that is read once cannot be saved twice, and pinning it
    /// would book a saving that already exists.
    ///
    /// This is an **upper bound** on what a pass reads, and deliberately so. A
    /// layer's manifest may carry a plain tensor its own architecture family
    /// never loads (a hash-routing table on a learned-routing layer, say), and
    /// the authoritative list is the block's contract rather than the manifest.
    /// Over-counting reserves slightly more residency than the layer uses,
    /// which is the safe direction: the tier fills lazily on first touch and
    /// the run can only end up under its declared budget, never over it.
    public var deterministicReadBytesPerPass: UInt64 {
        var total = UInt64Accounting.SaturatingSum()
        for record in matrices.values { total.add(UInt64(record.layout.tileStride)) }
        for record in plain.values where record.dtype != .int64 { total.add(record.bytes) }
        return total.value
    }

    /// Bytes this layer occupies when the dial pins it.
    ///
    /// A pinned block-FP8 matrix holds the whole adopted tile — `adopting`
    /// takes ownership of the entire `tileStride` allocation, not only its
    /// packed region — **plus** the expanded scale grid MLX's kernel needs:
    /// `outFeatures × inFeatures / 32` bytes against a compact grid of
    /// `outFeatures × inFeatures / 16384`. That expansion is the whole
    /// difference between this and ``deterministicReadBytesPerPass``, and it is
    /// why a pinned V4 layer scores ~0.970 bytes saved per resident byte rather
    /// than K3's 1.000. See ``DeepSeekV4PinnedWeightCache`` for why the loaded
    /// form is held rather than the stored one.
    ///
    /// A pinned plain tensor costs exactly what it reads: the MLX array keeps
    /// the member's own dtype, so nothing is widened.
    public var pinnedResidentBytesPerLayer: UInt64 {
        var total = UInt64Accounting.SaturatingSum()
        for record in matrices.values {
            total.add(UInt64(record.layout.tileStride))
            total.add(Self.expandedScaleBytes(record.layout))
        }
        for record in plain.values where record.dtype != .int64 { total.add(record.bytes) }
        return total.value
    }

    /// `outFeatures × inFeatures / 32`, the MLX MX scale layout
    /// ``BlockFP8Weights`` expands the 128×128 block grid into.
    private static func expandedScaleBytes(_ layout: QuantizedTileContainer.Layout) -> UInt64 {
        expandedScaleBytes(rows: layout.geometry.rows, columns: layout.geometry.cols)
    }

    private static func expandedScaleBytes(rows: Int, columns: Int) -> UInt64 {
        let rows = UInt64(max(0, rows))
        let groups = UInt64(max(0, columns)) / UInt64(BlockFP8Weights.mlxGroupSize)
        return rows.multipliedReportingOverflow(by: groups).partialValue
    }

    // MARK: - The same census, before anything is opened

    /// One layer's dial census, read from the published manifest alone.
    ///
    /// Same three columns as the properties above, and the same rules — but no
    /// descriptor is opened, so a screen can price a budget for an artifact that
    /// is on disk and not running. That matters because the memory dial has to
    /// draw a ladder *before* the run that would consume it, and constructing a
    /// ``DeepSeekV4LayerArtifact`` per layer reconciles every container header
    /// in the unit.
    ///
    /// The identity that makes this exact rather than an estimate is the one the
    /// full constructor asserts: a manifest tile-container's `bytes` must equal
    /// its header's `totalBytes`, which is
    /// `QuantizedTileContainer.headerBytes + tileCount × tileStride`. So the
    /// stride is recovered from the manifest by the container's own arithmetic
    /// rather than by re-deriving a geometry, and a manifest that disagrees with
    /// its header is refused later by the constructor rather than smoothed over
    /// here.
    public struct ManifestCensus: Sendable, Equatable {
        public let layer: Int
        /// Deterministic bytes one pass reads — see
        /// ``deterministicReadBytesPerPass``.
        public let deterministicReadBytesPerPass: UInt64
        /// Bytes a pinned layer holds — see ``pinnedResidentBytesPerLayer``.
        public let pinnedResidentBytes: UInt64
        /// The widest routed-expert tile stride declared by this unit. Not part
        /// of either column above: the expert pool is its own budget term, and
        /// this is only the granularity that term is priced in.
        public let widestExpertTileStrideBytes: UInt64
    }

    public static func manifestCensus(
        manifestData: Data, unitReference: String
    ) throws -> ManifestCensus {
        let document: Document
        do {
            document = try JSONDecoder().decode(Document.self, from: manifestData)
        } catch {
            throw DeepSeekV4Error.artifact("manifest.json could not be decoded: \(error)")
        }
        guard isSafeComponent(unitReference), document.unit == unitReference,
            let layer = layerNumber(unitReference)
        else {
            throw DeepSeekV4Error.artifact(
                "manifest unit '\(document.unit)' does not name rooted unit '\(unitReference)'")
        }

        let headerBytes = UInt64(QuantizedTileContainer.headerBytes)
        var read = UInt64Accounting.SaturatingSum()
        var resident = UInt64Accounting.SaturatingSum()
        var widestExpertStride: UInt64 = 0

        for file in document.files {
            switch file.kind {
            case "tile-container":
                guard let tiles = file.count, tiles > 0, file.bytes > headerBytes else {
                    throw DeepSeekV4Error.artifact(
                        "\(file.name) declares no tile count or no payload beyond its header")
                }
                let stride = (file.bytes - headerBytes) / UInt64(tiles)
                guard let tensor = file.sourceTensor, !tensor.isEmpty else {
                    // No `source_tensor` is what makes a container routed
                    // experts rather than a deterministic matrix — the same
                    // test the constructor applies.
                    widestExpertStride = max(widestExpertStride, stride)
                    continue
                }
                guard let rows = file.rows, let columns = file.cols else {
                    throw DeepSeekV4Error.artifact(
                        "\(file.name) names a source tensor without a declared geometry")
                }
                read.add(stride)
                resident.add(stride)
                resident.add(expandedScaleBytes(rows: rows, columns: columns))

            case "blob":
                if let members = file.members {
                    for member in members where member.dtype != PlainDType.int64.rawValue {
                        read.add(member.bytes)
                        resident.add(member.bytes)
                    }
                } else if file.dtype != PlainDType.int64.rawValue {
                    read.add(file.bytes)
                    resident.add(file.bytes)
                }

            default:
                throw DeepSeekV4Error.artifact(
                    "\(file.name) uses unsupported payload kind '\(file.kind)'")
            }
        }
        guard read.value > 0 else {
            throw DeepSeekV4Error.artifact(
                "\(unitReference) declares no deterministic bytes for a pass to read")
        }
        return ManifestCensus(
            layer: layer,
            deterministicReadBytesPerPass: read.value,
            pinnedResidentBytes: resident.value,
            widestExpertTileStrideBytes: widestExpertStride)
    }

    /// Install the run's pinned tier. Called once, after the plan is made and
    /// before the first payload read; a second call is refused so a plan cannot
    /// be swapped underneath a partly filled tier.
    func installPinnedWeights(_ cache: DeepSeekV4PinnedWeightCache) {
        staticStateLock.lock()
        if pinnedWeights == nil { pinnedWeights = cache }
        staticStateLock.unlock()
    }

    private var installedPinnedWeights: DeepSeekV4PinnedWeightCache? {
        staticStateLock.lock()
        defer { staticStateLock.unlock() }
        return pinnedWeights
    }

    /// What the current policy lets this load say about the packed matrix's
    /// finiteness. `verifyEveryLoad` never memoizes — a reader that recomputes
    /// the digest on every load has not been told the bytes are stable, and it
    /// would be inventing exactly the trust ``TileDigestPolicy`` exists to make
    /// the caller state.
    private func packedFinitenessCheck(
        for tensor: String
    ) -> BlockFP8Weights.PackedFinitenessCheck {
        guard case .trustHeldAuthority = tileDigestPolicy else { return .scan }
        staticStateLock.lock()
        defer { staticStateLock.unlock() }
        return scannedPackedTensors.contains(tensor) ? .establishedForTheseBytes : .scan
    }

    private func notePackedTensorScanned(_ tensor: String) {
        guard case .trustHeldAuthority = tileDigestPolicy else { return }
        staticStateLock.lock()
        scannedPackedTensors.insert(tensor)
        staticStateLock.unlock()
    }

    /// Load one non-expert FP8 matrix. Peak transient storage is one container
    /// tile; no other layer matrix is retained by this object.
    ///
    /// The tile's recorded SHA-256 is recomputed here under
    /// ``TileDigestPolicy/verifyEveryLoad`` and skipped under
    /// ``TileDigestPolicy/trustHeldAuthority(_:)``. Nothing else about this
    /// path is conditional: the descriptor's identity is rechecked by the file
    /// access on every open, the file length is rechecked against both the
    /// manifest and the container header, and the requested geometry is
    /// rechecked against that header before a byte is read.
    ///
    /// The one other thing the same policy decides is whether adoption
    /// re-derives the packed matrix's E4M3 finiteness. Under
    /// `verifyEveryLoad` it is derived on every load, always. Under
    /// `trustHeldAuthority` it is derived on this run's **first** load of each
    /// tensor and taken as established afterwards, because it is a pure
    /// function of a byte string the same held authority already says is the
    /// same byte string — the identical premise the skipped digest rests on
    /// (ADR 0013). That is a narrower claim than the per-run digest memo ADR
    /// 0013 rejected: this scan is not an integrity check, so memoizing it
    /// replaces no guard. The bytes' integrity is still whatever the rooted
    /// opener's identity recheck makes it, on every open, memo or not.
    public func loadBlockFP8(
        tensor: String,
        outFeatures: Int,
        inFeatures: Int,
        cancellationCheck: () throws -> Void = {}
    ) throws -> BlockFP8Weights {
        try loadBlockFP8Reporting(
            tensor: tensor, outFeatures: outFeatures, inFeatures: inFeatures,
            cancellationCheck: cancellationCheck
        ).weights
    }

    /// One loaded block-FP8 matrix, and who owns the bytes behind it.
    struct LoadedBlockFP8 {
        let weights: BlockFP8Weights
        /// True when the resident tier holds these weights for the rest of the
        /// run — it served them, or this pass's fill took. False when this pass
        /// is the only owner, so the transient allocation behind `weights` lives
        /// exactly as long as the graph that references it.
        let residentTierOwns: Bool
    }

    /// ``loadBlockFP8(tensor:outFeatures:inFeatures:cancellationCheck:)``, and
    /// which tier ends up owning the result.
    ///
    /// The provenance is not a statistic. It decides whether the projection
    /// built on these weights has to be forced before the loader returns — see
    /// ``projectBlockFP8Reference(_:tensor:outFeatures:inFeatures:stream:cancellationCheck:)``.
    func loadBlockFP8Reporting(
        tensor: String,
        outFeatures: Int,
        inFeatures: Int,
        cancellationCheck: () throws -> Void = {}
    ) throws -> LoadedBlockFP8 {
        guard let record = matrices[tensor] else {
            throw DeepSeekV4Error.artifact("manifest has no FP8 tensor '\(tensor)'")
        }
        let layout = record.layout
        guard layout.geometry.rows == outFeatures, layout.geometry.cols == inFeatures else {
            throw DeepSeekV4Error.artifact(
                "\(tensor) is \(layout.geometry.rows)x\(layout.geometry.cols), expected "
                    + "\(outFeatures)x\(inFeatures)")
        }
        // Precedence: pinned, then read. The geometry above is rechecked on
        // every call whether or not the tier serves it, so a pinned tensor
        // cannot outlive a contract it no longer satisfies.
        //
        // A served tensor is the *same* `BlockFP8Weights` the filling pass
        // built, so no digest is recomputed and none is skipped: there is no
        // second read to have a policy about. The bytes it saves are counted
        // apart from bytes read, never inside them.
        let pinned = installedPinnedWeights
        if let pinned, let weights = pinned.blockFP8(layer: layer, tensor: tensor) {
            pinned.noteBlockFP8Served(readBytes: UInt64(layout.tileStride))
            readAccounting?.recordPinnedServed(UInt64(layout.tileStride))
            phaseAccounting?.recordPinnedServed()
            return LoadedBlockFP8(weights: weights, residentTierOwns: true)
        }
        return try fileAccess.withDescriptor(record.reference) { descriptor in
            let actual = try Self.fileLength(descriptor, reference: record.reference)
            guard actual == record.bytes, actual == layout.totalBytes else {
                throw DeepSeekV4Error.artifact(
                    "\(record.reference) changed length after manifest reconciliation")
            }
            let allocation = UnsafeMutableRawPointer.allocate(
                byteCount: layout.tileStride,
                alignment: max(16, Int(getpagesize())))
            var ownershipTransferred = false
            defer {
                if !ownershipTransferred { allocation.deallocate() }
            }
            try Self.readExactly(
                descriptor: descriptor,
                reference: record.reference,
                offset: layout.tileOffset(0),
                length: layout.tileStride,
                destination: allocation,
                cancellationCheck: cancellationCheck,
                readAccounting: readAccounting,
                phaseAccounting: phaseAccounting)
            let tile = UnsafeRawBufferPointer(
                start: allocation, count: layout.tileStride)
            // Stated where the digest used to imply it, so the length the
            // adoption below relies on is checked under either policy.
            guard tile.count == layout.tileStride,
                layout.tileStride >= layout.scaleOffsetInTile + layout.geometry.scaleBytes
            else {
                throw DeepSeekV4Error.artifact(
                    "\(record.reference) tile 0 is not one whole \(layout.tileStride)-byte tile")
            }
            switch tileDigestPolicy {
            case .verifyEveryLoad:
                // The whole single-tile matrix is re-hashed on every load. The
                // 2026-08-15 gate measured that at 45.1 s of a 58.2 s V4 decode
                // pass, which is why the other policy exists.
                try measuringPhase(phaseAccounting?.recordTileDigest(nanoseconds:)) {
                    try QuantizedTileContainer.verifyTileDigest(tile, layout: layout, tile: 0)
                }
            case .trustHeldAuthority:
                // No digest ran, so the digest brackets record nothing and the
                // metric shows the saving. This counter is what keeps the run
                // JSON honest about how many loads were served under trust.
                phaseAccounting?.recordTileDigestSkippedUnderAuthority()
            }
            // Adoption is neither the read nor the digest: it byte-scans the
            // whole packed matrix and expands the scale grid before the tile
            // becomes an operand (audit 2026-08-14, P2). It ran inside the
            // residual until it got this bracket, and the bracket stays where
            // it is so it keeps measuring whatever adoption still costs.
            let finiteness = self.packedFinitenessCheck(for: tensor)
            let weights = try measuringPhase(
                phaseAccounting?.recordTileAdoption(nanoseconds:)
            ) {
                try BlockFP8Weights.adopting(
                    packedPointer: allocation,
                    packedByteCount: layout.geometry.packedBytes,
                    compactScaleBytes: UnsafeRawBufferPointer(
                        start: allocation + layout.scaleOffsetInTile,
                        count: layout.geometry.scaleBytes),
                    outFeatures: layout.geometry.rows,
                    inFeatures: layout.geometry.cols,
                    scaleBlockRows: layout.geometry.scaleBlockRows,
                    scaleBlockColumns: layout.geometry.groupSize,
                    packedFiniteness: finiteness,
                    finalizer: { allocation.deallocate() })
            }
            ownershipTransferred = true
            if finiteness == .establishedForTheseBytes {
                phaseAccounting?.recordPackedFinitenessMemoHit()
            } else {
                // Recorded only after adoption returned, so a scan that threw
                // never establishes anything.
                self.notePackedTensorScanned(tensor)
            }
            // The fill happens on the pass that read the bytes, so they are
            // already inside `deterministicBytesRead` and are recorded here as
            // loaded rather than served.
            //
            // A fill that took makes the tier — not this graph — the owner of
            // the adopted allocation from here on, which is the same position a
            // served tensor is in. A refused fill leaves this pass the only
            // owner, so the projection must still be forced.
            let retained = pinned?.fill(
                layer: layer, tensor: tensor, weights: weights,
                readBytes: UInt64(layout.tileStride),
                residentBytes: UInt64(layout.tileStride)
                    + Self.expandedScaleBytes(layout)) ?? false
            return LoadedBlockFP8(weights: weights, residentTierOwns: retained)
        }
    }

    /// Read, digest-check, and execute one dynamic block-FP8 projection.
    ///
    /// ## Where this projection is evaluated, and why that is a decision
    ///
    /// This function used to end with `projected.eval()` unconditionally. The
    /// reason was real but narrow: a tile *this pass adopted* is owned by the
    /// `MLXArray` the pending graph holds, so the transient allocation behind it
    /// cannot be released until that graph runs. Forcing here bounded the
    /// loader's transient footprint at one matrix.
    ///
    /// It also made the model one graph per *matrix*. The 2026-08-16 census
    /// counted 510 of those drains in a single decode pass — half of all 1,056
    /// host syncs — and the seconds they cost land in the phase table's residual
    /// where they read as arithmetic.
    ///
    /// So the force is now conditional on the thing that actually motivated it.
    /// When the resident tier owns these weights — it served them, or this
    /// pass's fill took and the tier holds them from here on — **no allocation
    /// depends on this graph**, evaluating frees nothing, and the force is
    /// skipped. When this pass is the tile's only owner the force stays exactly
    /// where it was.
    ///
    /// That split is deliberate rather than conservative. Releasing a streamed
    /// layer's tiles at the layer boundary instead would work, but it raises the
    /// in-flight dense set from one matrix to a whole layer — ~140.8 MB at the
    /// published geometry against a `transientExecutionBytes` envelope with
    /// ~252 MB of margin, and at the 2 GB product scale *nothing* is pinned, so
    /// every layer would be streamed at exactly the budget where that margin is
    /// tightest. Spending it silently is what
    /// `DeepSeekV4ProductMemoryBudget`'s stated-terms discipline exists to
    /// prevent, so the wider release waits for the floor term that pays for it
    /// (ADR 0015).
    ///
    /// The arithmetic is untouched either way: the same ops in the same order,
    /// with only the point at which some of them are forced moved outward.
    ///
    /// This is still a bounded correctness path for layer bring-up; the product
    /// decoder must replace its unfused activation reference before V4 is
    /// advertised as a supported runtime.
    public func projectBlockFP8Reference(
        _ input: MLXArray,
        tensor: String,
        outFeatures: Int,
        inFeatures: Int,
        stream: StreamOrDevice = .default,
        cancellationCheck: () throws -> Void = {}
    ) throws -> MLXArray {
        try cancellationCheck()
        let output = try autoreleasepool { () throws -> MLXArray in
            let loaded = try loadBlockFP8Reporting(
                tensor: tensor,
                outFeatures: outFeatures,
                inFeatures: inFeatures,
                cancellationCheck: cancellationCheck)
            try cancellationCheck()
            let projected = try DeepSeekV4BlockFP8LinearReference.project(
                input, weights: loaded.weights, stream: stream,
                phaseAccounting: phaseAccounting,
                diagnostics: diagnostics)
            if !loaded.residentTierOwns {
                phaseAccounting?.recordEval(.projection)
                projected.eval()
            }
            return projected
        }
        try cancellationCheck()
        return output
    }

    /// Read and execute the grouped V4 attention output projection while
    /// retaining at most the one validated `wo_a` tile.
    ///
    /// Forced only when this pass owns the tile, for the reason
    /// ``projectBlockFP8Reference(_:tensor:outFeatures:inFeatures:stream:cancellationCheck:)``
    /// gives.
    public func projectGroupedBlockFP8Reference(
        _ input: MLXArray,
        tensor: String,
        outFeatures: Int,
        inFeatures: Int,
        groups: Int,
        stream: StreamOrDevice = .default,
        cancellationCheck: () throws -> Void = {}
    ) throws -> MLXArray {
        try cancellationCheck()
        let output = try autoreleasepool { () throws -> MLXArray in
            let loaded = try loadBlockFP8Reporting(
                tensor: tensor,
                outFeatures: outFeatures,
                inFeatures: inFeatures,
                cancellationCheck: cancellationCheck)
            try cancellationCheck()
            let projected = try DeepSeekV4BlockFP8LinearReference.projectGroupedOutput(
                input,
                weights: loaded.weights,
                groups: groups,
                stream: stream,
                cancellationCheck: cancellationCheck,
                phaseAccounting: phaseAccounting,
                diagnostics: diagnostics)
            if !loaded.residentTierOwns {
                phaseAccounting?.recordEval(.projection)
                projected.eval()
            }
            return projected
        }
        try cancellationCheck()
        return output
    }

    /// Load one BF16 or F32 member into an MLX-owned array. I64 members must be
    /// consumed through ``loadTokenExpertMap`` so integer range checks cannot
    /// be skipped accidentally.
    public func loadFloatingTensor(
        tensor: String,
        expectedDType: PlainDType,
        expectedShape: [Int],
        cancellationCheck: () throws -> Void = {}
    ) throws -> MLXArray {
        try loadFloatingTensorReporting(
            tensor: tensor, expectedDType: expectedDType,
            expectedShape: expectedShape, cancellationCheck: cancellationCheck
        ).array
    }

    /// One loaded plain tensor, and who owns the bytes behind it. The BF16
    /// projection closures need the same ownership answer the block-FP8 path
    /// does, and for the same reason.
    struct LoadedFloatingTensor {
        let array: MLXArray
        let residentTierOwns: Bool
    }

    func loadFloatingTensorReporting(
        tensor: String,
        expectedDType: PlainDType,
        expectedShape: [Int],
        cancellationCheck: () throws -> Void = {}
    ) throws -> LoadedFloatingTensor {
        guard let record = plain[tensor] else {
            throw DeepSeekV4Error.artifact("manifest has no plain tensor '\(tensor)'")
        }
        guard record.dtype == expectedDType, record.shape == expectedShape,
            let mlxDType = record.dtype.mlxDType
        else {
            throw DeepSeekV4Error.artifact(
                "\(tensor) is \(record.dtype.rawValue)\(record.shape), expected "
                    + "\(expectedDType.rawValue)\(expectedShape)")
        }
        // Precedence: pinned, then read — the same rule and the same order as
        // the block-FP8 path above, after the same dtype and shape recheck.
        let pinned = installedPinnedWeights
        if let pinned, let array = pinned.floatingTensor(layer: layer, tensor: tensor) {
            pinned.noteFloatingServed(readBytes: record.bytes)
            readAccounting?.recordPinnedServed(record.bytes)
            phaseAccounting?.recordPinnedServed()
            return LoadedFloatingTensor(array: array, residentTierOwns: true)
        }
        let data = try read(record, cancellationCheck: cancellationCheck)
        let pointer = UnsafeMutableRawPointer.allocate(
            byteCount: data.count, alignment: 16)
        data.copyBytes(to: pointer.assumingMemoryBound(to: UInt8.self), count: data.count)
        let array = MLXArray(
            rawPointer: pointer, record.shape, dtype: mlxDType,
            finalizer: { pointer.deallocate() })
        // Only the unpinned path reaches here: a pinned tensor returned above
        // without reading and without forcing anything.
        phaseAccounting?.recordEval(.weightMaterialisation)
        array.eval()
        // The array keeps the member's own dtype, so a pinned plain tensor
        // costs exactly the bytes it stops reading.
        let retained = pinned?.fill(
            layer: layer, tensor: tensor, array: array,
            readBytes: record.bytes, residentBytes: record.bytes) ?? false
        return LoadedFloatingTensor(array: array, residentTierOwns: retained)
    }

    /// Load a hash-routing table and narrow every I64 id explicitly. This is
    /// the only I64 interpretation exposed by the layer reader.
    ///
    /// The table is immutable checkpoint data addressed by this unit's
    /// reconciled manifest, so the first successful load for one request
    /// geometry is retained for this artifact's lifetime — which is the run —
    /// and every later token reuses it instead of re-reading and re-validating
    /// the same `[vocabulary, expertsPerToken]` table. A request that names a
    /// different geometry is a different key and is validated on its own.
    public func loadTokenExpertMap(
        tensor: String,
        vocabularySize: Int,
        expertsPerToken: Int,
        expertCount: Int,
        cancellationCheck: () throws -> Void = {}
    ) throws -> DeepSeekV4TokenExpertMap {
        let key = TokenExpertMapKey(
            tensor: tensor, vocabularySize: vocabularySize,
            expertsPerToken: expertsPerToken, expertCount: expertCount)
        staticStateLock.lock()
        let cached = tokenExpertMaps[key]
        staticStateLock.unlock()
        if let cached { return cached }
        let loaded = try readTokenExpertMap(
            tensor: tensor,
            vocabularySize: vocabularySize,
            expertsPerToken: expertsPerToken,
            expertCount: expertCount,
            cancellationCheck: cancellationCheck)
        staticStateLock.lock()
        tokenExpertMaps[key] = loaded
        staticStateLock.unlock()
        return loaded
    }

    private func readTokenExpertMap(
        tensor: String,
        vocabularySize: Int,
        expertsPerToken: Int,
        expertCount: Int,
        cancellationCheck: () throws -> Void
    ) throws -> DeepSeekV4TokenExpertMap {
        guard let record = plain[tensor] else {
            throw DeepSeekV4Error.artifact("manifest has no plain tensor '\(tensor)'")
        }
        guard record.dtype == .int64,
            record.shape == [vocabularySize, expertsPerToken]
        else {
            throw DeepSeekV4Error.artifact(
                "\(tensor) is \(record.dtype.rawValue)\(record.shape), expected "
                    + "I64[\(vocabularySize), \(expertsPerToken)]")
        }
        let data = try read(record, cancellationCheck: cancellationCheck)
        var ids = [Int32]()
        ids.reserveCapacity(data.count / MemoryLayout<Int64>.size)
        try data.withUnsafeBytes { bytes in
            for offset in stride(from: 0, to: bytes.count, by: MemoryLayout<Int64>.size) {
                let raw = Int64(littleEndian: bytes.loadUnaligned(
                    fromByteOffset: offset, as: Int64.self))
                guard let narrowed = Int32(exactly: raw) else {
                    throw DeepSeekV4Error.artifact(
                        "\(tensor) contains expert id \(raw), outside Int32")
                }
                ids.append(narrowed)
            }
        }
        return try DeepSeekV4TokenExpertMap(
            ids: ids, vocabularySize: vocabularySize,
            expertsPerToken: expertsPerToken, expertCount: expertCount)
    }

    private func read(
        _ record: PlainRecord,
        cancellationCheck: () throws -> Void
    ) throws -> Data {
        try fileAccess.withDescriptor(record.reference) { descriptor in
            let actual = try Self.fileLength(descriptor, reference: record.reference)
            guard actual == record.fileBytes else {
                throw DeepSeekV4Error.artifact(
                    "\(record.reference) changed length after manifest reconciliation")
            }
            guard let length = Int(exactly: record.bytes) else {
                throw DeepSeekV4Error.artifact(
                    "\(record.tensor) byte length is not representable by this process")
            }
            return try Self.readExactly(
                descriptor: descriptor, reference: record.reference,
                offset: record.offset, length: length,
                cancellationCheck: cancellationCheck,
                readAccounting: readAccounting,
                phaseAccounting: phaseAccounting)
        }
    }

    private static func plainRecord(
        tensor: String,
        dtypeName: String,
        shape: [Int],
        reference: String,
        fileBytes: UInt64,
        offset: UInt64,
        bytes: UInt64,
        sourcePrefix: String,
        seen: inout Set<String>
    ) throws -> PlainRecord {
        try validateTensorName(tensor, prefix: sourcePrefix, seen: &seen)
        guard let dtype = PlainDType(rawValue: dtypeName) else {
            throw DeepSeekV4Error.artifact(
                "\(tensor) has unsupported plain dtype '\(dtypeName)'")
        }
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw DeepSeekV4Error.artifact("\(tensor) has invalid shape \(shape)")
        }
        var elements: UInt64 = 1
        for dimension in shape {
            let product = elements.multipliedReportingOverflow(by: UInt64(dimension))
            guard !product.overflow else {
                throw DeepSeekV4Error.artifact("\(tensor) shape overflows UInt64")
            }
            elements = product.partialValue
        }
        let byteCount = elements.multipliedReportingOverflow(by: UInt64(dtype.byteWidth))
        guard !byteCount.overflow, byteCount.partialValue == bytes else {
            throw DeepSeekV4Error.artifact(
                "\(tensor) \(dtype.rawValue)\(shape) requires "
                    + "\(byteCount.overflow ? UInt64.max : byteCount.partialValue) bytes, "
                    + "manifest says \(bytes)")
        }
        let end = offset.addingReportingOverflow(bytes)
        guard !end.overflow, end.partialValue <= fileBytes else {
            throw DeepSeekV4Error.artifact(
                "\(tensor) range runs past its \(fileBytes)-byte blob")
        }
        return PlainRecord(
            tensor: tensor, reference: reference, fileBytes: fileBytes,
            dtype: dtype, shape: shape, offset: offset, bytes: bytes)
    }

    private static func validateTensorName(
        _ tensor: String, prefix: String, seen: inout Set<String>
    ) throws {
        guard tensor.hasPrefix(prefix), !tensor.contains("\0"),
            seen.insert(tensor).inserted
        else {
            throw DeepSeekV4Error.artifact(
                "tensor '\(tensor)' is outside this layer or appears more than once")
        }
    }

    private static func readExactly(
        descriptor: Int32,
        reference: String,
        offset: UInt64,
        length: Int,
        cancellationCheck: () throws -> Void,
        readAccounting: DeepSeekV4ReadAccounting?,
        phaseAccounting: DeepSeekV4PhaseAccounting?
    ) throws -> Data {
        var data = Data(count: length)
        try data.withUnsafeMutableBytes { buffer in
            guard let base = buffer.baseAddress else {
                throw DeepSeekV4Error.artifact("\(reference) could not allocate its read buffer")
            }
            try readExactly(
                descriptor: descriptor, reference: reference,
                offset: offset, length: length, destination: base,
                cancellationCheck: cancellationCheck,
                readAccounting: readAccounting,
                phaseAccounting: phaseAccounting)
        }
        return data
    }

    /// The one place the V4 decode thread blocks on a deterministic read.
    ///
    /// The phase bracket is per call, not per 4 MiB chunk: the loop below can
    /// run hundreds of iterations for one matrix, and a clock read inside it
    /// would measure the instrument as much as the read.
    private static func readExactly(
        descriptor: Int32,
        reference: String,
        offset: UInt64,
        length: Int,
        destination: UnsafeMutableRawPointer,
        cancellationCheck: () throws -> Void,
        readAccounting: DeepSeekV4ReadAccounting?,
        phaseAccounting: DeepSeekV4PhaseAccounting?
    ) throws {
        try measuringPhase(phaseAccounting?.recordDeterministicRead(nanoseconds:)) {
            try readChunks(
                descriptor: descriptor, reference: reference,
                offset: offset, length: length, destination: destination,
                cancellationCheck: cancellationCheck,
                readAccounting: readAccounting)
        }
    }

    private static func readChunks(
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
                    offset: offset + UInt64(completed),
                    expected: take, actual: 0)
            }
            readAccounting?.recordDeterministic(UInt64(result))
            completed += result
        }
        try cancellationCheck()
    }

    private static func fileLength(_ descriptor: Int32, reference: String) throws -> UInt64 {
        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            throw StorageCoreError.posix(operation: "fstat", path: reference, code: errno)
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

    private static func layerNumber(_ unit: String) -> Int? {
        guard unit.hasPrefix("layers") else { return nil }
        let suffix = unit.dropFirst("layers".count)
        guard !suffix.isEmpty, suffix.allSatisfy(\.isNumber) else { return nil }
        return Int(suffix)
    }

    private static func isLowercaseSHA256(_ value: String) -> Bool {
        value.count == 64 && value.utf8.allSatisfy { byte in
            (48...57).contains(byte) || (97...102).contains(byte)
        }
    }
}
