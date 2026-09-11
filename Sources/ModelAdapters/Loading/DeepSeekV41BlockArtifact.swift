import Darwin
import Foundation
import MLX
import MLXBridge
import StorageCore

/// Plain (unquantized) element types a V4.1 unit manifest declares.
public enum DeepSeekV41PlainDType: String, Sendable, Equatable, CaseIterable {
    case bfloat16 = "BF16"
    case float32 = "F32"

    public var byteWidth: Int {
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

/// A block-FP8 matrix a V4.1 block owns, named by its tensor suffix.
///
/// The raw value is the tensor name with the unit's own prefix removed —
/// `layers.7.attn.wq_a.weight` is `attn.wq_a.weight` in block 7 of the
/// backbone, and `mtp.0.attn.wq_a.weight` is the same case in DSpark block 0.
/// One enum covers both because the two are the same block with a different
/// expert count.
public enum DeepSeekV41BlockMatrix: String, Sendable, Equatable, CaseIterable {
    case attentionQueryDown = "attn.wq_a.weight"
    case attentionQueryUp = "attn.wq_b.weight"
    case attentionKeyValue = "attn.wkv.weight"
    case attentionOutputDown = "attn.wo_a.weight"
    case attentionOutputUp = "attn.wo_b.weight"
    /// Present only on `index_source_layers`.
    case indexerQueryUp = "attn.indexer.wq_b.weight"
    case sharedExpertGate = "ffn.shared_experts.w1.weight"
    case sharedExpertDown = "ffn.shared_experts.w2.weight"
    case sharedExpertUp = "ffn.shared_experts.w3.weight"
    /// Present only on `engram_layer_ids`.
    case engramKeyValue = "engram.wkv.weight"
    /// Present only on DSpark block 0.
    case draftMainProjection = "main_proj.weight"
}

/// A plain BF16 or F32 tensor a V4.1 block owns, named by its tensor suffix.
public enum DeepSeekV41BlockVector: String, Sendable, Equatable, CaseIterable {
    case attentionSink = "attn.attn_sink"
    case attentionQueryNorm = "attn.q_norm.weight"
    case attentionKeyValueNorm = "attn.kv_norm.weight"
    case attentionNorm = "attn_norm.weight"
    /// Compressor, present only on `kv_source_layers`.
    case compressorNorm = "attn.compressor.norm.weight"
    case compressorKeyValue = "attn.compressor.wkv.weight"
    /// The learned pooling gate — present only where the compressor actually
    /// pools, i.e. where the compression ratio is greater than 1.
    case compressorGate = "attn.compressor.wgate.weight"
    /// Indexer key path, present only on `kv_source_layers`.
    case indexerKey = "attn.indexer.wk.weight"
    case indexerKeyNorm = "attn.indexer.k_norm.weight"
    /// Indexer head weighting, present on every `index_source_layers` block.
    case indexerWeights = "attn.indexer.weights_proj.weight"
    case gateWeight = "ffn.gate.weight"
    case gateBias = "ffn.gate.bias"
    /// Used only when image tokens are present; the text path must load it or
    /// skip it explicitly rather than not know it is there.
    case gateBiasVisionLanguage = "ffn.gate.bias_vl"
    case feedForwardNorm = "ffn_norm.weight"
    case hyperAttentionFunction = "hc_attn_fn"
    case hyperAttentionBase = "hc_attn_base"
    case hyperAttentionScale = "hc_attn_scale"
    case hyperFeedForwardFunction = "hc_ffn_fn"
    case hyperFeedForwardBase = "hc_ffn_base"
    case hyperFeedForwardScale = "hc_ffn_scale"
    /// Engram, present only on `engram_layer_ids`.
    case engramQuery = "engram.q_weight"
    case engramKey = "engram.k_weight"
    /// DSpark block 0 normalizes the backbone's last hidden state.
    case draftMainNorm = "main_norm.weight"
    /// DSpark block 2 carries the draft stack's own final norm and heads.
    case draftNorm = "norm.weight"
    case draftMarkovEmbed = "markov_head.embed.weight"
    case draftMarkovHead = "markov_head.head.weight"
    case draftConfidence = "confidence_head.proj.weight"
}

/// Which of a routed expert's three matrices is meant.
public enum DeepSeekV41ExpertProjection: String, Sendable, Equatable, CaseIterable {
    case gate = "w1"
    case down = "w2"
    case up = "w3"
}

/// Where one routed expert's bytes are, given that a tile is two experts.
///
/// A V4.1 expert cannot be a tile: `w1` is `[2304, 5120]` FP4, whose scale
/// region is 22.5 alignment units, and the container pads only for the 2-D
/// block-scale mode. Two experts are 720 packed units and 45 scale units with
/// no padding at all, so the published tile is an expert *pair* and
/// `expert_order` maps an expert id to its tile and half.
///
/// Both sub-regions are row-major with one scale per 32 elements of one row, so
/// a half is a contiguous run inside each of them — which is what makes an
/// expert gather that addresses a half-tile possible without unpacking a pair.
public struct DeepSeekV41ExpertPlacement: Sendable, Equatable {
    public let expert: Int
    public let projection: DeepSeekV41ExpertProjection
    /// Repository-relative or rooted reference of the `.mxfp4tile` file.
    public let fileReference: String
    public let tile: Int
    /// 0 or 1 — which expert of the pair.
    public let half: Int
    /// Output rows of this one expert.
    public let rows: Int
    public let columns: Int
    /// File offset of the whole tile the expert shares.
    public let tileOffset: UInt64
    public let tileStride: Int
    /// Offset of this expert's packed run inside the tile.
    public let packedOffsetInTile: Int
    public let packedBytes: Int
    /// Offset of this expert's scale run inside the tile.
    public let scaleOffsetInTile: Int
    public let scaleBytes: Int
    /// Elements sharing one E8M0 exponent — 32, by OCP MX.
    public let scaleGroupSize: Int
}

/// The geometry of one Engram table, republished by a unit's manifest.
///
/// The row arithmetic a reader needs is entirely here: pick the part whose row
/// span contains the row, then ask ``EngramRowPageContainer/Layout/locate(row:)``.
public struct DeepSeekV41EngramGeometry: Sendable, Equatable {
    public struct Part: Sendable, Equatable {
        public let fileReference: String
        public let partIndex: Int
        public let rowBase: UInt64
        public let rows: UInt64
        public let bytes: UInt64
        public let layout: EngramRowPageContainer.Layout
    }

    public let valueTensor: String
    public let scaleTensor: String
    public let tableRows: UInt64
    public let rowValueBytes: Int
    public let rowScaleBytes: Int
    public let pageBytes: Int
    public let rowsPerPage: Int
    public let scaleOffsetInPage: Int
    public let firstPageOffset: UInt64
    public let scaleGroupSize: Int
    /// Parts in ascending row order, covering `[0, tableRows)` exactly once.
    public let parts: [Part]

    /// The part holding a table row, or `nil` when the row is outside the table.
    public func part(holding row: UInt64) -> Part? {
        parts.first { row >= $0.rowBase && row < $0.rowBase + $0.rows }
    }
}

/// A checked, lazy view of one published V4.1 block unit — `layersNN` or
/// `mtpNN`.
///
/// Construction reconciles the manifest with every container header and file
/// length in the unit and makes nothing resident. It deliberately does **not**
/// require a ``DeepSeekV41Config``: a unit is a self-describing object, and the
/// config's job is to say which units an execution needs, not to be the only
/// way to read one. ``validateAgainst(config:)`` is the separate, explicit
/// check that a unit is the block a given configuration describes.
///
/// The product supplies a ``ModelFileAccess`` rooted in the fully verified
/// artifact; no repository revision is compiled into this reader.
public final class DeepSeekV41BlockArtifact: @unchecked Sendable {
    /// Which half of the model a unit belongs to. The two differ only in tensor
    /// prefix and expert count, which is exactly why one reader covers both.
    public enum Family: String, Sendable, Equatable {
        /// `layersNN` — a backbone block.
        case backbone
        /// `mtpNN` — a DSpark draft block, parsed but not executed by the text
        /// path.
        case draft

        var tensorPrefix: String {
            switch self {
            case .backbone: return "layers."
            case .draft: return "mtp."
            }
        }

        var unitPrefix: String {
            switch self {
            case .backbone: return "layers"
            case .draft: return "mtp"
            }
        }
    }

    public let unitID: String
    public let family: Family
    /// The block number inside its family: 0..<40 for the backbone, 0..<3 for
    /// DSpark.
    public let block: Int
    public let sourceRepository: String
    public let sourceRevision: String
    public let manifestFileCount: Int
    public let manifestBytes: UInt64
    /// `expert_order[slot] == expert id`. The published order is the
    /// checkpoint's lexicographic tensor order, so slot 2 is expert 10.
    public let expertOrder: [Int]
    public var routedExpertCount: Int { expertOrder.count }
    /// The Engram table this block owns, or `nil` when it owns none.
    public let engram: DeepSeekV41EngramGeometry?

    private struct MatrixRecord: Sendable {
        let tensor: String
        let reference: String
        let bytes: UInt64
        let layout: QuantizedTileContainer.Layout
    }

    private struct ExpertRecord: Sendable {
        let reference: String
        let bytes: UInt64
        let layout: QuantizedTileContainer.Layout
        let rowsPerMember: Int
        let membersPerTile: Int
        let members: Int
    }

    private struct PlainRecord: Sendable {
        let tensor: String
        let reference: String
        let fileBytes: UInt64
        let dtype: DeepSeekV41PlainDType
        let shape: [Int]
        let offset: UInt64
        let bytes: UInt64
    }

    private let fileAccess: ModelFileAccess
    private let matrices: [DeepSeekV41BlockMatrix: MatrixRecord]
    private let experts: [DeepSeekV41ExpertProjection: ExpertRecord]
    private let plain: [DeepSeekV41BlockVector: PlainRecord]
    private let expertSlots: [Int: Int]

    /// Filesystem convenience for tests and command-line probes.
    public convenience init(
        manifestURL: URL,
        expectedSourceRepository: String? = nil,
        expectedSourceRevision: String? = nil
    ) throws {
        let directory = manifestURL.deletingLastPathComponent()
        try self.init(
            manifestData: try Data(contentsOf: manifestURL),
            unitReference: directory.lastPathComponent,
            filesystemDirectory: directory,
            fileAccess: .filesystem,
            expectedSourceRepository: expectedSourceRepository,
            expectedSourceRevision: expectedSourceRevision)
    }

    /// Rooted product entry point. `unitReference` is one repository component,
    /// for example `layers07`; a manifest-supplied file name never becomes a
    /// path of its own.
    public convenience init(
        manifestData: Data,
        unitReference: String,
        fileAccess: ModelFileAccess,
        expectedSourceRepository: String? = nil,
        expectedSourceRevision: String? = nil
    ) throws {
        try self.init(
            manifestData: manifestData,
            unitReference: unitReference,
            filesystemDirectory: nil,
            fileAccess: fileAccess,
            expectedSourceRepository: expectedSourceRepository,
            expectedSourceRevision: expectedSourceRevision)
    }

    private init(
        manifestData: Data,
        unitReference: String,
        filesystemDirectory: URL?,
        fileAccess: ModelFileAccess,
        expectedSourceRepository: String?,
        expectedSourceRevision: String?
    ) throws {
        let document: Document
        do {
            document = try JSONDecoder().decode(Document.self, from: manifestData)
        } catch {
            throw DeepSeekV41Error.artifact("manifest.json could not be decoded: \(error)")
        }
        guard DeepSeekV41UnitName.isSafeComponent(unitReference),
            document.unit == unitReference,
            let parsed = DeepSeekV41UnitName.block(unitReference)
        else {
            throw DeepSeekV41Error.artifact(
                "manifest unit '\(document.unit)' does not name rooted block unit "
                    + "'\(unitReference)'")
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
        guard !document.files.isEmpty else {
            throw DeepSeekV41Error.artifact("manifest contains no payload files")
        }

        let family = parsed.family
        let block = parsed.block
        let sourcePrefix = "\(family.tensorPrefix)\(block)."

        guard let order = document.expertOrder, !order.isEmpty else {
            throw DeepSeekV41Error.experts(
                "\(unitReference) declares no expert_order; a tile is an expert pair "
                    + "and nothing else says which expert is which half")
        }
        guard Set(order) == Set(0..<order.count) else {
            throw DeepSeekV41Error.experts(
                "expert_order is not a permutation of 0..<\(order.count)")
        }
        var expertSlots = [Int: Int]()
        expertSlots.reserveCapacity(order.count)
        for (slot, expert) in order.enumerated() { expertSlots[expert] = slot }

        var fileNames = Set<String>()
        var tensors = Set<String>()
        var matrices = [DeepSeekV41BlockMatrix: MatrixRecord]()
        var experts = [DeepSeekV41ExpertProjection: ExpertRecord]()
        var plain = [DeepSeekV41BlockVector: PlainRecord]()
        var engramParts = [Document.File]()
        var engramLayouts = [EngramRowPageContainer.Layout]()
        var engramReferences = [String]()
        var manifestBytes: UInt64 = 0

        for file in document.files {
            guard DeepSeekV41UnitName.isSafeComponent(file.name),
                fileNames.insert(file.name).inserted
            else {
                throw DeepSeekV41Error.artifact(
                    "manifest contains an unsafe or repeated file name '\(file.name)'")
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

            switch file.kind {
            case "tile-container":
                let layout = try fileAccess.withDescriptor(reference) { descriptor in
                    let layout = try QuantizedTileContainer.open(
                        fileDescriptor: descriptor, path: reference)
                    let actual = try Self.fileLength(descriptor, reference: reference)
                    guard actual == layout.totalBytes, actual == file.bytes else {
                        throw DeepSeekV41Error.artifact(
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
                    throw DeepSeekV41Error.artifact(
                        "\(file.name) manifest geometry disagrees with its container header")
                }
                if let tensor = file.sourceTensor {
                    // A named source tensor is what makes a container one dense
                    // matrix rather than a run of routed experts.
                    let matrix = try Self.matrixCase(
                        tensor: tensor, prefix: sourcePrefix, seen: &tensors,
                        file: file.name)
                    guard layout.geometry.quantMode == .fp8E4M3Block,
                        layout.geometry.bits == 8,
                        layout.geometry.scaleDType == .uint8E8M0,
                        layout.geometry.groupSize == 32,
                        layout.geometry.scaleBlockRows == 32,
                        file.blockRows == 32,
                        layout.tileCount == 1,
                        layout.contentKind == .checkpointDerived,
                        layout.tileDigests?.count == 1
                    else {
                        throw DeepSeekV41Error.artifact(
                            "\(file.name) is not a single checkpoint-derived FP8 [32, 32] "
                                + "matrix")
                    }
                    guard matrices[matrix] == nil else {
                        throw DeepSeekV41Error.artifact(
                            "\(file.name) repeats matrix '\(matrix.rawValue)'")
                    }
                    matrices[matrix] = MatrixRecord(
                        tensor: tensor, reference: reference, bytes: file.bytes,
                        layout: layout)
                } else {
                    let projection = try Self.expertProjection(fileName: file.name)
                    guard let rowsPerMember = file.rowsPerMember,
                        let membersPerTile = file.membersPerTile,
                        let members = file.members,
                        rowsPerMember > 0, membersPerTile > 0, members > 0
                    else {
                        throw DeepSeekV41Error.experts(
                            "\(file.name) is a routed-expert container without a member "
                                + "geometry")
                    }
                    guard layout.geometry.quantMode == .mxfp4,
                        layout.geometry.bits == 4,
                        layout.geometry.groupSize == 32,
                        layout.geometry.scaleDType == .uint8E8M0,
                        layout.geometry.scaleBlockRows == 1,
                        layout.contentKind == .checkpointDerived,
                        layout.tileDigests?.count == layout.tileCount
                    else {
                        throw DeepSeekV41Error.experts(
                            "\(file.name) is not a checkpoint-derived MXFP4 container with "
                                + "one digest per tile")
                    }
                    guard members == order.count,
                        members == layout.tileCount * membersPerTile,
                        layout.geometry.rows == rowsPerMember * membersPerTile,
                        layout.geometry.packedBytes % membersPerTile == 0,
                        layout.geometry.scaleBytes % membersPerTile == 0
                    else {
                        throw DeepSeekV41Error.experts(
                            "\(file.name) does not divide \(order.count) experts into "
                                + "\(file.count ?? -1) whole tiles of \(membersPerTile)")
                    }
                    guard experts[projection] == nil else {
                        throw DeepSeekV41Error.experts(
                            "\(file.name) repeats expert projection '\(projection.rawValue)'")
                    }
                    experts[projection] = ExpertRecord(
                        reference: reference, bytes: file.bytes, layout: layout,
                        rowsPerMember: rowsPerMember, membersPerTile: membersPerTile,
                        members: members)
                }

            case "engram-part":
                let layout = try fileAccess.withDescriptor(reference) { descriptor in
                    try EngramRowPageContainer.open(
                        fileDescriptor: descriptor, path: reference)
                }
                guard layout.totalBytes == file.bytes else {
                    throw DeepSeekV41Error.engram(
                        "\(file.name) is \(layout.totalBytes) bytes by its header; "
                            + "manifest says \(file.bytes)")
                }
                try Self.reconcileEngram(file: file, layout: layout)
                engramParts.append(file)
                engramLayouts.append(layout)
                engramReferences.append(reference)

            case "blob":
                if let digest = file.sha256, !Self.isLowercaseSHA256(digest) {
                    throw DeepSeekV41Error.artifact(
                        "\(file.name) has a non-canonical SHA-256")
                }
                try fileAccess.withDescriptor(reference) { descriptor in
                    let actual = try Self.fileLength(descriptor, reference: reference)
                    guard actual == file.bytes else {
                        throw DeepSeekV41Error.artifact(
                            "\(file.name) is \(actual) bytes; manifest says \(file.bytes)")
                    }
                }
                if let members = file.members2 {
                    guard file.sourceTensor == nil, file.dtype == nil, file.shape == nil,
                        !members.isEmpty
                    else {
                        throw DeepSeekV41Error.artifact(
                            "\(file.name) mixes bundled and standalone tensor metadata")
                    }
                    var expectedOffset: UInt64 = 0
                    for member in members {
                        guard member.offset == expectedOffset else {
                            throw DeepSeekV41Error.artifact(
                                "\(file.name) members are not a gapless partition at byte "
                                    + "\(expectedOffset)")
                        }
                        let (vector, record) = try Self.plainRecord(
                            tensor: member.sourceTensor, dtypeName: member.dtype,
                            shape: member.shape, reference: reference,
                            fileBytes: file.bytes, offset: member.offset,
                            bytes: member.bytes, sourcePrefix: sourcePrefix,
                            seen: &tensors, file: file.name)
                        guard plain[vector] == nil else {
                            throw DeepSeekV41Error.artifact(
                                "\(file.name) repeats plain tensor '\(vector.rawValue)'")
                        }
                        plain[vector] = record
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
                    let (vector, record) = try Self.plainRecord(
                        tensor: tensor, dtypeName: dtype, shape: shape,
                        reference: reference, fileBytes: file.bytes, offset: 0,
                        bytes: file.bytes, sourcePrefix: sourcePrefix, seen: &tensors,
                        file: file.name)
                    guard plain[vector] == nil else {
                        throw DeepSeekV41Error.artifact(
                            "\(file.name) repeats plain tensor '\(vector.rawValue)'")
                    }
                    plain[vector] = record
                }

            default:
                throw DeepSeekV41Error.artifact(
                    "\(file.name) uses unsupported payload kind '\(file.kind)'")
            }
        }

        guard Set(experts.keys) == Set(DeepSeekV41ExpertProjection.allCases) else {
            throw DeepSeekV41Error.experts(
                "\(unitReference) does not publish all three routed-expert projections")
        }
        // `expert(_:)` answers `(tile, half)` once for all three projections, so
        // the three have to pair experts the same way. They do in every
        // published unit — 384 experts, two to a tile, 192 tiles — and a unit
        // where they did not would make that one answer three different ones.
        guard Set(experts.values.map(\.membersPerTile)).count == 1 else {
            throw DeepSeekV41Error.experts(
                "\(unitReference) pairs experts differently in different projections")
        }
        guard !matrices.isEmpty, !plain.isEmpty else {
            throw DeepSeekV41Error.artifact(
                "a model block must contain both FP8 matrices and plain tensors")
        }

        self.engram = try Self.engramGeometry(
            parts: engramParts, layouts: engramLayouts, references: engramReferences,
            sourcePrefix: sourcePrefix, unitReference: unitReference)

        self.unitID = document.unit
        self.family = family
        self.block = block
        self.sourceRepository = document.sourceRepository
        self.sourceRevision = document.sourceRevision
        self.manifestFileCount = document.files.count
        self.manifestBytes = manifestBytes
        self.expertOrder = order
        self.expertSlots = expertSlots
        self.fileAccess = fileAccess
        self.matrices = matrices
        self.experts = experts
        self.plain = plain
    }

    // MARK: - What this block has

    public func contains(_ matrix: DeepSeekV41BlockMatrix) -> Bool {
        matrices[matrix] != nil
    }

    public func contains(_ vector: DeepSeekV41BlockVector) -> Bool {
        plain[vector] != nil
    }

    /// Every FP8 matrix this unit publishes, in a stable order.
    public var publishedMatrices: [DeepSeekV41BlockMatrix] {
        DeepSeekV41BlockMatrix.allCases.filter { matrices[$0] != nil }
    }

    /// Every plain tensor this unit publishes, in a stable order.
    public var publishedVectors: [DeepSeekV41BlockVector] {
        DeepSeekV41BlockVector.allCases.filter { plain[$0] != nil }
    }

    /// The full checkpoint tensor name behind a case — `layers.7.attn.wkv.weight`.
    public func tensorName(_ matrix: DeepSeekV41BlockMatrix) -> String {
        "\(family.tensorPrefix)\(block).\(matrix.rawValue)"
    }

    public func tensorName(_ vector: DeepSeekV41BlockVector) -> String {
        "\(family.tensorPrefix)\(block).\(vector.rawValue)"
    }

    /// `(rows, columns)` of one FP8 matrix, from the container header.
    public func geometry(
        of matrix: DeepSeekV41BlockMatrix
    ) throws -> (rows: Int, columns: Int) {
        let record = try require(matrix)
        return (record.layout.geometry.rows, record.layout.geometry.cols)
    }

    /// `(dtype, shape)` of one plain tensor, from the manifest.
    public func descriptor(
        of vector: DeepSeekV41BlockVector
    ) throws -> (dtype: DeepSeekV41PlainDType, shape: [Int]) {
        let record = try require(vector)
        return (record.dtype, record.shape)
    }

    /// Bytes of this unit a pass reads deterministically — every FP8 matrix and
    /// every plain tensor, and no routed-expert tile and no Engram page.
    ///
    /// This is the memory dial's read column for a V4.1 block: what a pass
    /// re-reads if the block is not pinned. The routed experts are excluded
    /// because they are chosen per token and are reported by the run's own
    /// expert accounting; the Engram parts are excluded because a table is
    /// ~100 GB and a token reads 24 pages of it, which is neither a per-block
    /// quantity nor a pinnable one.
    public var denseReadBytesPerPass: UInt64 {
        var total = UInt64Accounting.SaturatingSum()
        for record in matrices.values { total.add(record.bytes) }
        for record in plain.values { total.add(record.bytes) }
        return total.didOverflow ? 0 : total.value
    }

    /// What pinning this block's dense tensors costs.
    ///
    /// Larger than ``denseReadBytesPerPass`` by the FP8 scale expansion: a
    /// stored tile carries one E8M0 exponent per 32x32 block, and the resident
    /// ``BlockFP8Weights`` form carries one per row per 32 columns — 32 times
    /// as many scale bytes. V4's dial makes the same trade for the same reason
    /// (`DeepSeekV4MemoryDial`, ~1.031x stored at `[128, 128]`); at `[32, 32]`
    /// the stored grid is already dense enough that the expansion is 31/32 of
    /// the stored scale region, which is what the arithmetic below states.
    public var pinnedResidentBytes: UInt64 {
        var total = UInt64Accounting.SaturatingSum()
        for record in matrices.values {
            let geometry = record.layout.geometry
            total.add(UInt64(geometry.packedBytes))
            // The expanded grid: one exponent per row per `groupSize` columns.
            let rows = UInt64(geometry.rows)
            let groups = UInt64(geometry.cols / max(1, geometry.groupSize))
            total.add(rows.multipliedReportingOverflow(by: groups).partialValue)
        }
        for record in plain.values { total.add(record.bytes) }
        return total.didOverflow ? 0 : total.value
    }

    /// Bytes one whole tile of a routed-expert container occupies — the read
    /// granularity of an expert pair.
    public func expertTileStride(_ projection: DeepSeekV41ExpertProjection) throws -> Int {
        try require(projection).layout.tileStride
    }

    public func expertTileCount(_ projection: DeepSeekV41ExpertProjection) throws -> Int {
        try require(projection).layout.tileCount
    }

    // MARK: - Loading

    /// Load one FP8 [32, 32] matrix, recomputing its recorded SHA-256.
    ///
    /// Peak transient storage is one container tile. Every V4.1 dense matrix is
    /// a single-tile container, so that is also the matrix.
    /// - Parameter verifyDigest: recompute the tile's recorded SHA-256.
    ///   Default true, which is ADR 0020's accepted cost. A caller that holds a
    ///   *completed* verification authority over this exact revision may pass
    ///   false: the bytes have already been hashed once, the descriptor is
    ///   rooted and identity-checked on every open, and ADR 0013 measured the
    ///   same term at 45.1 s of a 58.2 s V4 decode pass. It is a parameter and
    ///   not a default because the premise — an authority held open for the
    ///   run — is a property of the caller and cannot be checked from here.
    public func loadBlockFP8(
        _ matrix: DeepSeekV41BlockMatrix,
        verifyDigest: Bool = true,
        cancellationCheck: () throws -> Void = {}
    ) throws -> BlockFP8Weights {
        let record = try require(matrix)
        let layout = record.layout
        return try fileAccess.withDescriptor(record.reference) { descriptor in
            let actual = try Self.fileLength(descriptor, reference: record.reference)
            guard actual == record.bytes, actual == layout.totalBytes else {
                throw DeepSeekV41Error.artifact(
                    "\(record.reference) changed length after manifest reconciliation")
            }
            let allocation = UnsafeMutableRawPointer.allocate(
                byteCount: layout.tileStride, alignment: max(16, Int(getpagesize())))
            var ownershipTransferred = false
            defer { if !ownershipTransferred { allocation.deallocate() } }
            try Self.readExactly(
                descriptor: descriptor, reference: record.reference,
                offset: layout.tileOffset(0), length: layout.tileStride,
                destination: allocation, cancellationCheck: cancellationCheck)
            let tile = UnsafeRawBufferPointer(
                start: allocation, count: layout.tileStride)
            guard layout.tileStride >= layout.scaleOffsetInTile + layout.geometry.scaleBytes
            else {
                throw DeepSeekV41Error.artifact(
                    "\(record.reference) tile 0 is not one whole "
                        + "\(layout.tileStride)-byte tile")
            }
            if verifyDigest {
                try QuantizedTileContainer.verifyTileDigest(tile, layout: layout, tile: 0)
            }
            let weights = try BlockFP8Weights.adopting(
                packedPointer: allocation,
                packedByteCount: layout.geometry.packedBytes,
                compactScaleBytes: UnsafeRawBufferPointer(
                    start: allocation + layout.scaleOffsetInTile,
                    count: layout.geometry.scaleBytes),
                outFeatures: layout.geometry.rows,
                inFeatures: layout.geometry.cols,
                scaleBlockRows: layout.geometry.scaleBlockRows,
                scaleBlockColumns: layout.geometry.groupSize,
                finalizer: { allocation.deallocate() })
            ownershipTransferred = true
            return weights
        }
    }

    /// Load one plain BF16 or F32 tensor into an MLX-owned array.
    public func loadVector(
        _ vector: DeepSeekV41BlockVector,
        cancellationCheck: () throws -> Void = {}
    ) throws -> MLXArray {
        let record = try require(vector)
        let data = try read(record, cancellationCheck: cancellationCheck)
        let pointer = UnsafeMutableRawPointer.allocate(
            byteCount: data.count, alignment: 16)
        data.copyBytes(to: pointer.assumingMemoryBound(to: UInt8.self), count: data.count)
        let array = MLXArray(
            rawPointer: pointer, record.shape, dtype: record.dtype.mlxDType,
            finalizer: { pointer.deallocate() })
        array.eval()
        return array
    }

    // MARK: - Routed experts

    /// Which tile and half hold one expert. The pair is the read granularity;
    /// the half is what the gather addresses.
    public func expert(_ id: Int) throws -> (tile: Int, half: Int) {
        guard let slot = expertSlots[id] else {
            throw DeepSeekV41Error.experts(
                "expert \(id) is outside 0..<\(routedExpertCount) in \(unitID)")
        }
        // Every projection divides the same expert count into the same pairs,
        // so the slot arithmetic is one answer rather than three.
        let members = try require(DeepSeekV41ExpertProjection.gate).membersPerTile
        return (slot / members, slot % members)
    }

    /// Everything a gather needs to read one expert's bytes out of the pair
    /// tile that holds it.
    public func expert(
        _ id: Int, projection: DeepSeekV41ExpertProjection
    ) throws -> DeepSeekV41ExpertPlacement {
        let record = try require(projection)
        guard let slot = expertSlots[id] else {
            throw DeepSeekV41Error.experts(
                "expert \(id) is outside 0..<\(routedExpertCount) in \(unitID)")
        }
        let tile = slot / record.membersPerTile
        let half = slot % record.membersPerTile
        let layout = record.layout
        let packedPerMember = layout.geometry.packedBytes / record.membersPerTile
        let scalePerMember = layout.geometry.scaleBytes / record.membersPerTile
        return DeepSeekV41ExpertPlacement(
            expert: id,
            projection: projection,
            fileReference: record.reference,
            tile: tile,
            half: half,
            rows: record.rowsPerMember,
            columns: layout.geometry.cols,
            tileOffset: layout.tileOffset(tile),
            tileStride: layout.tileStride,
            packedOffsetInTile: half * packedPerMember,
            packedBytes: packedPerMember,
            scaleOffsetInTile: layout.scaleOffsetInTile + half * scalePerMember,
            scaleBytes: scalePerMember,
            scaleGroupSize: layout.geometry.groupSize)
    }

    // MARK: - Engram rows

    /// One Engram row — 256 FP8 values and their 8 E8M0 exponents — in one
    /// aligned 4 KiB read.
    public func engramRow(_ row: UInt64) throws -> EngramRowPageContainer.Row {
        try engramRows([row])[0]
    }

    /// A batch of Engram rows, one aligned page read per distinct page.
    ///
    /// A module reads 24 of these per token and their addresses depend only on
    /// token ids, so the batch is issuable before the token's first block runs.
    public func engramRows(_ rows: [UInt64]) throws -> [EngramRowPageContainer.Row] {
        guard let engram else {
            throw DeepSeekV41Error.engram("\(unitID) publishes no engram table")
        }
        guard !rows.isEmpty else { return [] }
        // Grouped by part, so a batch spanning two parts opens two descriptors
        // rather than one per row.
        var byPart = [Int: [UInt64]]()
        for row in rows {
            guard let part = engram.part(holding: row) else {
                throw DeepSeekV41Error.engram(
                    "row \(row) is outside the \(engram.tableRows)-row table of \(unitID)")
            }
            byPart[part.partIndex, default: []].append(row)
        }
        var results = [UInt64: EngramRowPageContainer.Row]()
        results.reserveCapacity(rows.count)
        for (partIndex, partRows) in byPart {
            let part = engram.parts[partIndex]
            let read = try fileAccess.withDescriptor(part.fileReference) { descriptor in
                try EngramRowPageContainer.readRows(
                    fileDescriptor: descriptor, path: part.fileReference,
                    layout: part.layout, rows: partRows)
            }
            for row in read { results[row.row] = row }
        }
        return rows.map { results[$0]! }
    }

    // MARK: - Agreement with a configuration

    /// Check that this unit is the block a configuration describes.
    ///
    /// Separate from construction on purpose: a manifest is self-describing and
    /// can be opened without a config, but an *execution* must not discover a
    /// missing indexer after earlier payloads have been read.
    public func validateAgainst(config: DeepSeekV41Config) throws {
        let expectedExperts = family == .backbone
            ? config.routedExpertCount : config.draftRoutedExpertCount
        guard routedExpertCount == expectedExperts else {
            throw DeepSeekV41Error.experts(
                "\(unitID) publishes \(routedExpertCount) experts; the configuration "
                    + "says \(expectedExperts)")
        }
        let blockIndex = family == .backbone ? block : config.numberOfLayers + block
        guard blockIndex < config.publishedBlockCount else {
            throw DeepSeekV41Error.artifact(
                "\(unitID) is outside the \(config.publishedBlockCount) published blocks")
        }
        let mode = try config.attentionMode(block: blockIndex)

        // Attention, always.
        for matrix in [
            DeepSeekV41BlockMatrix.attentionQueryDown, .attentionQueryUp,
            .attentionKeyValue, .attentionOutputDown, .attentionOutputUp,
            .sharedExpertGate, .sharedExpertDown, .sharedExpertUp,
        ] {
            try expect(matrix)
        }
        for vector in [
            DeepSeekV41BlockVector.attentionSink, .attentionQueryNorm,
            .attentionKeyValueNorm, .attentionNorm, .gateWeight, .gateBias,
            .gateBiasVisionLanguage, .feedForwardNorm,
            .hyperAttentionFunction, .hyperAttentionBase, .hyperAttentionScale,
            .hyperFeedForwardFunction, .hyperFeedForwardBase, .hyperFeedForwardScale,
        ] {
            try expect(vector)
        }

        // The indexer and the compressor, exactly where CSA2 puts them.
        try expect(.indexerQueryUp, present: mode == .full || mode == .reindex)
        try expect(.indexerWeights, present: mode == .full || mode == .reindex)
        try expect(.indexerKey, present: mode == .full)
        try expect(.indexerKeyNorm, present: mode == .full)
        try expect(.compressorNorm, present: mode == .full)
        try expect(.compressorKeyValue, present: mode == .full)
        // The learned pooling gate exists only where the compressor pools more
        // than one token: block 20 compresses at ratio 1 and has no `wgate`.
        try expect(
            .compressorGate,
            present: mode == .full && (try config.compressionRatio(block: blockIndex)) > 1)

        // Engram, on its two blocks and nowhere else.
        let hasEngram = family == .backbone && config.hasEngram(block: block)
        try expect(.engramKeyValue, present: hasEngram)
        try expect(.engramQuery, present: hasEngram)
        try expect(.engramKey, present: hasEngram)
        guard (engram != nil) == hasEngram else {
            throw DeepSeekV41Error.engram(
                "\(unitID) \(engram == nil ? "lacks" : "publishes") an engram table; the "
                    + "configuration says otherwise")
        }
        if let engram, let expected = config.engramRowCount(block: block) {
            guard engram.tableRows == expected else {
                throw DeepSeekV41Error.engram(
                    "\(unitID) engram table has \(engram.tableRows) rows; the "
                        + "configuration says \(expected)")
            }
            guard engram.rowValueBytes == config.engramHeadDimension,
                engram.rowScaleBytes == config.engramRowScaleBytes,
                engram.scaleGroupSize == config.quantizedWeightBlock[1]
            else {
                throw DeepSeekV41Error.engram(
                    "\(unitID) engram rows are \(engram.rowValueBytes)+"
                        + "\(engram.rowScaleBytes) bytes; the configuration says "
                        + "\(config.engramHeadDimension)+\(config.engramRowScaleBytes)")
            }
        }

        // The DSpark heads, on the units that own them.
        try expect(.draftMainProjection, present: family == .draft && block == 0)
        try expect(.draftMainNorm, present: family == .draft && block == 0)
        let lastDraft = config.numberOfDraftLayers - 1
        try expect(.draftNorm, present: family == .draft && block == lastDraft)
        try expect(.draftMarkovEmbed, present: family == .draft && block == lastDraft)
        try expect(.draftMarkovHead, present: family == .draft && block == lastDraft)
        try expect(.draftConfidence, present: family == .draft && block == lastDraft)

        // Shapes the configuration fixes.
        let dim = config.hiddenSize
        try expectGeometry(.attentionQueryDown, config.queryLowRank, dim)
        try expectGeometry(
            .attentionQueryUp,
            config.numberOfAttentionHeads * config.attentionHeadDimension,
            config.queryLowRank)
        try expectGeometry(.attentionKeyValue, config.attentionHeadDimension, dim)
        // The output projection is grouped: `wo_a` takes one group's share of
        // the concatenated heads down to `o_lora_rank`, once per group.
        try expectGeometry(
            .attentionOutputDown,
            config.outputGroups * config.outputLowRank,
            config.numberOfAttentionHeads * config.attentionHeadDimension
                / config.outputGroups)
        try expectGeometry(.attentionOutputUp, dim, config.outputLowRank * config.outputGroups)
        try expectGeometry(.sharedExpertGate, config.expertIntermediateSize, dim)
        try expectGeometry(.sharedExpertDown, dim, config.expertIntermediateSize)
        try expectGeometry(.sharedExpertUp, config.expertIntermediateSize, dim)
        if contains(.indexerQueryUp) {
            try expectGeometry(
                .indexerQueryUp, config.indexHeadCount * config.indexHeadDimension,
                config.queryLowRank)
        }
        if contains(.engramKeyValue) {
            try expectGeometry(
                .engramKeyValue,
                dim * (config.hyperConnectionMultiplicity + 1),
                config.engramRowsPerToken * config.engramHeadDimension)
        }
        try expectShape(.gateWeight, .bfloat16, [expectedExperts, dim])
        try expectShape(.gateBias, .float32, [expectedExperts])
        try expectShape(.gateBiasVisionLanguage, .float32, [expectedExperts])
        try expectShape(.attentionSink, .float32, [config.numberOfAttentionHeads])
        try expectShape(.attentionNorm, .bfloat16, [dim])
        try expectShape(.feedForwardNorm, .bfloat16, [dim])
        try expectShape(.attentionQueryNorm, .bfloat16, [config.queryLowRank])
        try expectShape(
            .attentionKeyValueNorm, .bfloat16, [config.attentionHeadDimension])
        let hcMult = config.hyperConnectionMultiplicity
        for vector in [
            DeepSeekV41BlockVector.hyperAttentionFunction, .hyperFeedForwardFunction,
        ] {
            try expectShape(vector, .float32, [hcMult * (hcMult + 2), dim * hcMult])
        }
        for vector in [
            DeepSeekV41BlockVector.hyperAttentionBase, .hyperFeedForwardBase,
        ] {
            try expectShape(vector, .float32, [hcMult * (hcMult + 2)])
        }
        for vector in [
            DeepSeekV41BlockVector.hyperAttentionScale, .hyperFeedForwardScale,
        ] {
            try expectShape(vector, .float32, [3])
        }
        if contains(.engramQuery) {
            try expectShape(.engramQuery, .bfloat16, [hcMult, dim])
            try expectShape(.engramKey, .bfloat16, [hcMult, dim])
        }

        // Routed experts: three projections, the same pairing in each.
        for projection in DeepSeekV41ExpertProjection.allCases {
            let record = try require(projection)
            guard record.members == expectedExperts else {
                throw DeepSeekV41Error.experts(
                    "\(unitID) \(projection.rawValue) declares \(record.members) experts, "
                        + "not \(expectedExperts)")
            }
            let rows = projection == .down ? dim : config.expertIntermediateSize
            let columns = projection == .down ? config.expertIntermediateSize : dim
            guard record.rowsPerMember == rows, record.layout.geometry.cols == columns else {
                throw DeepSeekV41Error.experts(
                    "\(unitID) \(projection.rawValue) expert is "
                        + "\(record.rowsPerMember)x\(record.layout.geometry.cols), expected "
                        + "\(rows)x\(columns)")
            }
        }
    }

    private func expect(
        _ matrix: DeepSeekV41BlockMatrix, present: Bool = true
    ) throws {
        guard contains(matrix) == present else {
            throw DeepSeekV41Error.artifact(
                "\(unitID) \(present ? "has no" : "unexpectedly publishes") FP8 tensor "
                    + "'\(tensorName(matrix))'")
        }
    }

    private func expect(
        _ vector: DeepSeekV41BlockVector, present: Bool = true
    ) throws {
        guard contains(vector) == present else {
            throw DeepSeekV41Error.artifact(
                "\(unitID) \(present ? "has no" : "unexpectedly publishes") plain tensor "
                    + "'\(tensorName(vector))'")
        }
    }

    private func expectGeometry(
        _ matrix: DeepSeekV41BlockMatrix, _ rows: Int, _ columns: Int
    ) throws {
        let actual = try geometry(of: matrix)
        guard actual.rows == rows, actual.columns == columns else {
            throw DeepSeekV41Error.artifact(
                "\(tensorName(matrix)) is \(actual.rows)x\(actual.columns), expected "
                    + "\(rows)x\(columns)")
        }
    }

    private func expectShape(
        _ vector: DeepSeekV41BlockVector, _ dtype: DeepSeekV41PlainDType, _ shape: [Int]
    ) throws {
        let actual = try descriptor(of: vector)
        guard actual.dtype == dtype, actual.shape == shape else {
            throw DeepSeekV41Error.artifact(
                "\(tensorName(vector)) is \(actual.dtype.rawValue)\(actual.shape), "
                    + "expected \(dtype.rawValue)\(shape)")
        }
    }

    // MARK: - Private

    private func require(_ matrix: DeepSeekV41BlockMatrix) throws -> MatrixRecord {
        guard let record = matrices[matrix] else {
            throw DeepSeekV41Error.artifact(
                "\(unitID) has no FP8 tensor '\(tensorName(matrix))'")
        }
        return record
    }

    private func require(_ vector: DeepSeekV41BlockVector) throws -> PlainRecord {
        guard let record = plain[vector] else {
            throw DeepSeekV41Error.artifact(
                "\(unitID) has no plain tensor '\(tensorName(vector))'")
        }
        return record
    }

    private func require(_ projection: DeepSeekV41ExpertProjection) throws -> ExpertRecord {
        guard let record = experts[projection] else {
            throw DeepSeekV41Error.experts(
                "\(unitID) has no routed-expert container for '\(projection.rawValue)'")
        }
        return record
    }

    private func read(
        _ record: PlainRecord, cancellationCheck: () throws -> Void
    ) throws -> Data {
        try fileAccess.withDescriptor(record.reference) { descriptor in
            let actual = try Self.fileLength(descriptor, reference: record.reference)
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
                try Self.readExactly(
                    descriptor: descriptor, reference: record.reference,
                    offset: record.offset, length: length, destination: base,
                    cancellationCheck: cancellationCheck)
            }
            return data
        }
    }

    private static func matrixCase(
        tensor: String, prefix: String, seen: inout Set<String>, file: String
    ) throws -> DeepSeekV41BlockMatrix {
        try validateTensorName(tensor, prefix: prefix, seen: &seen)
        let suffix = String(tensor.dropFirst(prefix.count))
        guard let matrix = DeepSeekV41BlockMatrix(rawValue: suffix) else {
            throw DeepSeekV41Error.artifact(
                "\(file) names FP8 tensor '\(tensor)', which this adapter does not know")
        }
        return matrix
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
        seen: inout Set<String>,
        file: String
    ) throws -> (DeepSeekV41BlockVector, PlainRecord) {
        try validateTensorName(tensor, prefix: sourcePrefix, seen: &seen)
        let suffix = String(tensor.dropFirst(sourcePrefix.count))
        guard let vector = DeepSeekV41BlockVector(rawValue: suffix) else {
            throw DeepSeekV41Error.artifact(
                "\(file) names plain tensor '\(tensor)', which this adapter does not know")
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
        return (
            vector,
            PlainRecord(
                tensor: tensor, reference: reference, fileBytes: fileBytes,
                dtype: dtype, shape: shape, offset: offset, bytes: bytes)
        )
    }

    private static func expertProjection(
        fileName: String
    ) throws -> DeepSeekV41ExpertProjection {
        for projection in DeepSeekV41ExpertProjection.allCases
        where fileName.hasSuffix("-\(projection.rawValue).mxfp4tile") {
            return projection
        }
        throw DeepSeekV41Error.experts(
            "\(fileName) is a tile container with no source tensor and no routed-expert "
                + "projection in its name")
    }

    private static func reconcileEngram(
        file: Document.File, layout: EngramRowPageContainer.Layout
    ) throws {
        func agree<Value: Equatable>(_ declared: Value?, _ actual: Value, _ name: String) throws {
            guard let declared else {
                throw DeepSeekV41Error.engram("\(file.name) does not declare \(name)")
            }
            guard declared == actual else {
                throw DeepSeekV41Error.engram(
                    "\(file.name) declares \(name) \(declared); its header says \(actual)")
            }
        }
        try agree(file.tableRows, layout.tableRows, "table_rows")
        try agree(file.rowBase, layout.rowBase, "row_base")
        try agree(file.engramRows, layout.partRows, "rows")
        try agree(file.pages, layout.pageCount, "pages")
        try agree(file.partIndex, layout.partIndex, "part_index")
        try agree(file.pageBytes, layout.pageBytes, "page_bytes")
        try agree(file.rowsPerPage, layout.rowsPerPage, "rows_per_page")
        try agree(file.rowWeightBytes, layout.rowValueBytes, "row_weight_bytes")
        try agree(file.rowScaleBytes, layout.rowScaleBytes, "row_scale_bytes")
        try agree(file.scaleOffsetInPage, layout.scaleOffsetInPage, "scale_offset_in_page")
        try agree(file.padBytesPerPage, layout.padBytesPerPage, "pad_bytes_per_page")
        try agree(file.firstPageOffset, layout.firstPageOffset, "first_page_offset")
        try agree(file.group, layout.scaleGroupSize, "group")
    }

    private static func engramGeometry(
        parts: [Document.File],
        layouts: [EngramRowPageContainer.Layout],
        references: [String],
        sourcePrefix: String,
        unitReference: String
    ) throws -> DeepSeekV41EngramGeometry? {
        guard !parts.isEmpty else { return nil }
        let valueTensors = Set(parts.compactMap(\.sourceTensor))
        let scaleTensors = Set(parts.compactMap(\.sourceScaleTensor))
        guard valueTensors.count == 1, scaleTensors.count == 1,
            let valueTensor = valueTensors.first, let scaleTensor = scaleTensors.first,
            valueTensor.hasPrefix(sourcePrefix), scaleTensor.hasPrefix(sourcePrefix)
        else {
            throw DeepSeekV41Error.engram(
                "\(unitReference) engram parts do not name exactly one value tensor and "
                    + "one scale tensor of this block")
        }
        let ordered = zip(references, layouts)
            .sorted { $0.1.partIndex < $1.1.partIndex }
        guard ordered.map({ $0.1.partIndex }) == Array(0..<ordered.count) else {
            throw DeepSeekV41Error.engram(
                "\(unitReference) engram part indices are not 0..<\(ordered.count)")
        }
        let first = ordered[0].1
        var expectedBase: UInt64 = 0
        var built = [DeepSeekV41EngramGeometry.Part]()
        for (reference, layout) in ordered {
            guard layout.tableRows == first.tableRows,
                layout.rowValueBytes == first.rowValueBytes,
                layout.rowScaleBytes == first.rowScaleBytes,
                layout.pageBytes == first.pageBytes,
                layout.rowsPerPage == first.rowsPerPage,
                layout.scaleGroupSize == first.scaleGroupSize
            else {
                throw DeepSeekV41Error.engram(
                    "\(unitReference) engram parts do not share one table geometry")
            }
            guard layout.rowBase == expectedBase else {
                throw DeepSeekV41Error.engram(
                    "\(unitReference) engram part \(layout.partIndex) starts at row "
                        + "\(layout.rowBase); the previous part ends at \(expectedBase)")
            }
            expectedBase = layout.rowEnd
            built.append(DeepSeekV41EngramGeometry.Part(
                fileReference: reference,
                partIndex: layout.partIndex,
                rowBase: layout.rowBase,
                rows: layout.partRows,
                bytes: layout.totalBytes,
                layout: layout))
        }
        guard expectedBase == first.tableRows else {
            throw DeepSeekV41Error.engram(
                "\(unitReference) engram parts cover \(expectedBase) of "
                    + "\(first.tableRows) rows")
        }
        return DeepSeekV41EngramGeometry(
            valueTensor: valueTensor,
            scaleTensor: scaleTensor,
            tableRows: first.tableRows,
            rowValueBytes: first.rowValueBytes,
            rowScaleBytes: first.rowScaleBytes,
            pageBytes: first.pageBytes,
            rowsPerPage: first.rowsPerPage,
            scaleOffsetInPage: first.scaleOffsetInPage,
            firstPageOffset: first.firstPageOffset,
            scaleGroupSize: first.scaleGroupSize,
            parts: built)
    }

    private static func validateTensorName(
        _ tensor: String, prefix: String, seen: inout Set<String>
    ) throws {
        guard tensor.hasPrefix(prefix), !tensor.contains("\0"),
            seen.insert(tensor).inserted
        else {
            throw DeepSeekV41Error.artifact(
                "tensor '\(tensor)' is outside this unit or appears more than once")
        }
    }

    private static func readExactly(
        descriptor: Int32,
        reference: String,
        offset: UInt64,
        length: Int,
        destination: UnsafeMutableRawPointer,
        cancellationCheck: () throws -> Void
    ) throws {
        guard length > 0, let start = off_t(exactly: offset) else {
            throw DeepSeekV41Error.artifact("\(reference) has an unrepresentable read range")
        }
        var completed = 0
        let chunk = 4 * 1_024 * 1_024
        while completed < length {
            try cancellationCheck()
            let take = min(chunk, length - completed)
            let position = start.addingReportingOverflow(off_t(completed))
            guard !position.overflow else {
                throw DeepSeekV41Error.artifact("\(reference) read offset overflows off_t")
            }
            let result = pread(descriptor, destination + completed, take, position.partialValue)
            if result < 0 {
                if errno == EINTR { continue }
                throw StorageCoreError.posix(operation: "pread", path: reference, code: errno)
            }
            guard result > 0 else {
                throw StorageCoreError.shortTransfer(
                    operation: "pread", path: reference,
                    offset: offset + UInt64(completed), expected: take, actual: 0)
            }
            completed += result
        }
        try cancellationCheck()
    }

    static func fileLength(_ descriptor: Int32, reference: String) throws -> UInt64 {
        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            throw StorageCoreError.posix(operation: "fstat", path: reference, code: errno)
        }
        guard status.st_size >= 0 else {
            throw DeepSeekV41Error.artifact("\(reference) has a negative file length")
        }
        return UInt64(status.st_size)
    }

    static func isLowercaseSHA256(_ value: String) -> Bool {
        value.count == 64 && value.utf8.allSatisfy { byte in
            (48...57).contains(byte) || (97...102).contains(byte)
        }
    }

    // MARK: - Wire shape

    struct Document: Decodable {
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

        struct File: Decodable {
            let name: String
            let kind: String
            let bytes: UInt64
            // Tile containers.
            let rows: Int?
            let cols: Int?
            let count: Int?
            let elementBits: Int?
            let group: Int?
            let blockRows: Int?
            let layout: String?
            let rowsPerMember: Int?
            let membersPerTile: Int?
            let members: Int?
            // Blobs.
            let sourceTensor: String?
            let dtype: String?
            let shape: [Int]?
            let sha256: String?
            let members2: [Member]?
            // Engram parts.
            let sourceScaleTensor: String?
            let tableRows: UInt64?
            let rowBase: UInt64?
            let engramRows: UInt64?
            let pages: UInt64?
            let partIndex: Int?
            let pageBytes: Int?
            let rowsPerPage: Int?
            let rowWeightBytes: Int?
            let rowScaleBytes: Int?
            let scaleOffsetInPage: Int?
            let padBytesPerPage: Int?
            let firstPageOffset: UInt64?

            enum CodingKeys: String, CodingKey {
                case name, kind, bytes, rows, cols, count, group, layout, dtype, shape
                case sha256, pages
                case elementBits = "element_bits"
                case blockRows = "block_rows"
                case rowsPerMember = "rows_per_member"
                case membersPerTile = "members_per_tile"
                case sourceTensor = "source_tensor"
                case sourceScaleTensor = "source_scale_tensor"
                case tableRows = "table_rows"
                case rowBase = "row_base"
                case partIndex = "part_index"
                case pageBytes = "page_bytes"
                case rowsPerPage = "rows_per_page"
                case rowWeightBytes = "row_weight_bytes"
                case rowScaleBytes = "row_scale_bytes"
                case scaleOffsetInPage = "scale_offset_in_page"
                case padBytesPerPage = "pad_bytes_per_page"
                case firstPageOffset = "first_page_offset"
                // `members` is an integer on a routed-expert container and an
                // array on a bundled blob, so the two readings are separate
                // properties over one key.
                case members
            }

            init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                name = try container.decode(String.self, forKey: .name)
                kind = try container.decode(String.self, forKey: .kind)
                bytes = try container.decode(UInt64.self, forKey: .bytes)
                rows = try container.decodeIfPresent(Int.self, forKey: .rows)
                cols = try container.decodeIfPresent(Int.self, forKey: .cols)
                count = try container.decodeIfPresent(Int.self, forKey: .count)
                elementBits = try container.decodeIfPresent(Int.self, forKey: .elementBits)
                group = try container.decodeIfPresent(Int.self, forKey: .group)
                blockRows = try container.decodeIfPresent(Int.self, forKey: .blockRows)
                layout = try container.decodeIfPresent(String.self, forKey: .layout)
                rowsPerMember = try container.decodeIfPresent(Int.self, forKey: .rowsPerMember)
                membersPerTile = try container.decodeIfPresent(
                    Int.self, forKey: .membersPerTile)
                sourceTensor = try container.decodeIfPresent(String.self, forKey: .sourceTensor)
                dtype = try container.decodeIfPresent(String.self, forKey: .dtype)
                shape = try container.decodeIfPresent([Int].self, forKey: .shape)
                sha256 = try container.decodeIfPresent(String.self, forKey: .sha256)
                sourceScaleTensor = try container.decodeIfPresent(
                    String.self, forKey: .sourceScaleTensor)
                tableRows = try container.decodeIfPresent(UInt64.self, forKey: .tableRows)
                rowBase = try container.decodeIfPresent(UInt64.self, forKey: .rowBase)
                engramRows = try container.decodeIfPresent(UInt64.self, forKey: .rows)
                pages = try container.decodeIfPresent(UInt64.self, forKey: .pages)
                partIndex = try container.decodeIfPresent(Int.self, forKey: .partIndex)
                pageBytes = try container.decodeIfPresent(Int.self, forKey: .pageBytes)
                rowsPerPage = try container.decodeIfPresent(Int.self, forKey: .rowsPerPage)
                rowWeightBytes = try container.decodeIfPresent(
                    Int.self, forKey: .rowWeightBytes)
                rowScaleBytes = try container.decodeIfPresent(Int.self, forKey: .rowScaleBytes)
                scaleOffsetInPage = try container.decodeIfPresent(
                    Int.self, forKey: .scaleOffsetInPage)
                padBytesPerPage = try container.decodeIfPresent(
                    Int.self, forKey: .padBytesPerPage)
                firstPageOffset = try container.decodeIfPresent(
                    UInt64.self, forKey: .firstPageOffset)
                members = try? container.decodeIfPresent(Int.self, forKey: .members)
                members2 = try? container.decodeIfPresent([Member].self, forKey: .members)
            }
        }

        let unit: String
        let sourceRepository: String
        let sourceRevision: String
        let expertOrder: [Int]?
        let files: [File]

        enum CodingKeys: String, CodingKey {
            case unit, files
            case sourceRepository = "source_repo"
            case sourceRevision = "source_revision"
            case expertOrder = "expert_order"
        }
    }
}

/// Unit-name arithmetic shared by the block and global readers.
enum DeepSeekV41UnitName {
    static func isSafeComponent(_ value: String) -> Bool {
        !value.isEmpty && value != "." && value != ".."
            && !value.contains("/") && !value.contains("\\") && !value.contains("\0")
    }

    static func block(
        _ unit: String
    ) -> (family: DeepSeekV41BlockArtifact.Family, block: Int)? {
        for family in [
            DeepSeekV41BlockArtifact.Family.backbone, .draft,
        ] where unit.hasPrefix(family.unitPrefix) {
            let suffix = unit.dropFirst(family.unitPrefix.count)
            guard !suffix.isEmpty, suffix.allSatisfy(\.isNumber),
                let number = Int(suffix)
            else { return nil }
            return (family, number)
        }
        return nil
    }
}
