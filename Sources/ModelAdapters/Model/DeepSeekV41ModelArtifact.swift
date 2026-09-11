import Darwin
import Foundation
import MLX
import MLXBridge
import StorageCore

/// `index.json` of a published V4.1 container, as the runner reads it.
///
/// Narrower than ``DeepSeekV4ArtifactIndex`` on purpose: V4.1's units are named
/// by the checkpoint's own layer numbering (`layers07`, `mtp01`, `global00`),
/// there is exactly one source repository, and the three metadata identities —
/// `configuration`, `tokenizer`, `arguments` — are all required, because
/// ADR 0020 made `inference-config.json` the authority and `config.json` a
/// cross-check and a runner may not open a unit without both.
public struct DeepSeekV41ArtifactIndex: Sendable, Equatable {
    public struct Unit: Sendable, Equatable {
        public let id: String
        public let fileCount: Int
        public let bytes: UInt64
    }

    public let sourceRepository: String
    public let sourceRevision: String
    public let relationship: String
    /// `layers00 ... layersNN`, in block order.
    public let blockUnits: [Unit]
    /// `mtp00 ...`, present and named; the text path runs none of them.
    public let draftUnits: [Unit]
    public let globalUnit: Unit
    public let totalBytes: UInt64
    /// The repository-relative file names of the three metadata documents.
    public let configurationFile: String
    public let argumentsFile: String
    public let tokenizerFile: String

    public init(json data: Data) throws {
        let document: Document
        do {
            document = try JSONDecoder().decode(Document.self, from: data)
        } catch {
            throw DeepSeekV41Error.artifact("index.json could not be decoded: \(error)")
        }
        guard !document.sourceRepo.isEmpty, !document.sourceRevision.isEmpty else {
            throw DeepSeekV41Error.artifact("index.json names no source revision")
        }
        var blocks = [Int: Unit]()
        var drafts = [Int: Unit]()
        var global: Unit?
        for unit in document.units {
            guard Self.isSafeComponent(unit.unit) else {
                throw DeepSeekV41Error.artifact(
                    "index.json names a unit that is not one safe path component: "
                        + "\(unit.unit)")
            }
            let value = Unit(id: unit.unit, fileCount: unit.files, bytes: unit.bytes)
            if let block = Self.suffix(unit.unit, after: "layers") {
                guard blocks.updateValue(value, forKey: block) == nil else {
                    throw DeepSeekV41Error.artifact("index.json names \(unit.unit) twice")
                }
            } else if let draft = Self.suffix(unit.unit, after: "mtp") {
                guard drafts.updateValue(value, forKey: draft) == nil else {
                    throw DeepSeekV41Error.artifact("index.json names \(unit.unit) twice")
                }
            } else if unit.unit == "global00" {
                guard global == nil else {
                    throw DeepSeekV41Error.artifact("index.json names global00 twice")
                }
                global = value
            } else {
                throw DeepSeekV41Error.artifact(
                    "index.json names a unit this adapter does not know: \(unit.unit)")
            }
        }
        guard let global else {
            throw DeepSeekV41Error.artifact("index.json names no global00 unit")
        }
        guard blocks.keys.sorted() == Array(0..<blocks.count) else {
            throw DeepSeekV41Error.artifact(
                "index.json's layer units are not 0..<\(blocks.count) without gaps")
        }
        guard drafts.keys.sorted() == Array(0..<drafts.count) else {
            throw DeepSeekV41Error.artifact(
                "index.json's mtp units are not 0..<\(drafts.count) without gaps")
        }
        self.sourceRepository = document.sourceRepo
        self.sourceRevision = document.sourceRevision
        self.relationship = document.relationship
        self.blockUnits = blocks.keys.sorted().map { blocks[$0]! }
        self.draftUnits = drafts.keys.sorted().map { drafts[$0]! }
        self.globalUnit = global
        self.totalBytes = document.totalBytes
        self.configurationFile = document.configuration.file
        self.argumentsFile = document.arguments.file
        self.tokenizerFile = document.tokenizer.file
        for file in [configurationFile, argumentsFile, tokenizerFile] {
            guard Self.isSafeComponent(file) else {
                throw DeepSeekV41Error.artifact(
                    "index.json names a metadata file that is not one safe component: \(file)")
            }
        }
    }

    private struct Document: Decodable {
        struct Unit: Decodable {
            let unit: String
            let files: Int
            let bytes: UInt64
        }
        struct Identity: Decodable {
            let file: String
        }
        let sourceRepo: String
        let sourceRevision: String
        let relationship: String
        let units: [Unit]
        let totalBytes: UInt64
        let configuration: Identity
        let tokenizer: Identity
        let arguments: Identity

        enum CodingKeys: String, CodingKey {
            case sourceRepo = "source_repo"
            case sourceRevision = "source_revision"
            case relationship
            case units
            case totalBytes = "total_bytes"
            case configuration, tokenizer, arguments
        }
    }

    private static func suffix(_ unit: String, after prefix: String) -> Int? {
        guard unit.hasPrefix(prefix) else { return nil }
        let digits = unit.dropFirst(prefix.count)
        guard digits.count == 2, digits.allSatisfy(\.isNumber) else { return nil }
        return Int(digits)
    }

    private static func isSafeComponent(_ value: String) -> Bool {
        !value.isEmpty && value != "." && value != ".."
            && !value.contains("/") && !value.contains("\\")
            && value.unicodeScalars.allSatisfy {
                CharacterSet.alphanumerics.contains($0) || $0 == "." || $0 == "-"
                    || $0 == "_"
            }
    }
}

/// The published V4.1 container as a ``DeepSeekV41WeightSource``.
///
/// This is the storage-native half of phase 3, and everything in it is a
/// decision about *when bytes are read* rather than about what is computed —
/// the arithmetic is `DeepSeekV41Model`'s and is the same whether it runs over
/// this or over the fixture the end-to-end test uses.
///
/// ## What is resident, and why
///
/// - **Dense block tensors** (≈172 MB a block, 6.9 GB over forty) are pinned
///   when the memory dial's plan pays for them, and otherwise loaded as a unit
///   at the top of the block and released at the bottom. "As a unit" is the
///   point: a block's nine FP8 matrices and its plain tensors are one
///   residency decision, because a half-loaded block is a block that reads
///   twice.
/// - **The output head** (1.32 GB BF16) is resident like V4's, when the plan
///   reaches the globals rung. Unpinned it is walked in `logitChunkRows`
///   windows, which is the same discipline and the same arithmetic — phase 1
///   asserted the streamed and resident forms produce the identical tensor.
/// - **Embedding rows** are read on demand, one row per distinct token id. The
///   table is the same 1.32 GB as the head and a decode token reads 10,240
///   bytes of it, so pinning it would be ~130,000x worse per resident byte than
///   pinning a block (`DeepSeekV4MemoryDial.Globals` reached the same verdict
///   for V4 and K3 before it).
/// - **Routed experts** live in one run-scoped ``DeepSeekV41ExpertTilePool``
///   keyed on `(unit, projection, tile)`, per ADR 0016's rule that a run's
///   expert residency is one shared pool and not one per block.
/// - **Engram pages** are transient. 96 KiB a token, read at the *start* of the
///   token because the addresses depend only on token ids, and dropped when the
///   token finishes.
public final class DeepSeekV41ModelArtifact: DeepSeekV41WeightSource, @unchecked Sendable {
    public typealias ManifestLoader = (_ unitID: String) throws -> Data

    public let config: DeepSeekV41Config
    public let index: DeepSeekV41ArtifactIndex
    public let blocks: [DeepSeekV41BlockArtifact]
    public let global: DeepSeekV41GlobalArtifact
    public let readAccounting: DeepSeekV4ReadAccounting
    public let phaseAccounting: DeepSeekV4PhaseAccounting

    public var sourceRepository: String { index.sourceRepository }
    public var sourceRevision: String { index.sourceRevision }

    private let expertPool: DeepSeekV41ExpertTilePool
    private let diagnostics: DeepSeekV4Diagnostics
    private let verifiesTileDigests: Bool
    /// Whether a gather operand adopts its own page-aligned allocation rather
    /// than being copied into MLX-owned arrays — V4's `transferMode` for V4.1,
    /// in the shape a *stack* can take. See
    /// ``DeepSeekV41PooledExpertSource/StackLayout``.
    private let adoptsExpertOperands: Bool
    private let cancellationCheck: () throws -> Void
    private let headWindowRows: Int
    /// Whether the output head's row windows are evaluated as they are walked,
    /// and whether the pooled expert source calls ``cancellationCheck`` before
    /// each gather operand it builds.
    private let boundsLiveOperands: Bool

    private let lock = NSLock()
    private var pinnedBlocks: [Int: DeepSeekV41BlockWeights] = [:]
    private var pinnedHead: MLXArray?
    private var finalNorm: MLXArray?
    /// `[engram ordinal][row] -> row bytes`, filled at the start of a token and
    /// dropped at the start of the next one.
    private var engramRows: [[UInt64: EngramRowPageContainer.Row]] = []
    private var pinnedTierBytes: UInt64 = 0

    public init(
        config: DeepSeekV41Config,
        indexData: Data,
        fileAccess: ModelFileAccess,
        expertPoolSlots: Int,
        expertQueueDepth: Int,
        verifiesTileDigests: Bool = true,
        adoptsExpertOperands: Bool = true,
        headWindowRows: Int = 4_096,
        boundsLiveOperands: Bool = false,
        readAccounting: DeepSeekV4ReadAccounting = DeepSeekV4ReadAccounting(),
        phaseAccounting: DeepSeekV4PhaseAccounting = DeepSeekV4PhaseAccounting(),
        diagnostics: DeepSeekV4Diagnostics = .off,
        cancellationCheck: @escaping () throws -> Void = {},
        loadManifest: ManifestLoader
    ) throws {
        let index = try DeepSeekV41ArtifactIndex(json: indexData)
        guard index.blockUnits.count == config.numberOfLayers else {
            throw DeepSeekV41Error.artifact(
                "index.json publishes \(index.blockUnits.count) backbone units; the "
                    + "configuration says \(config.numberOfLayers)")
        }
        guard index.draftUnits.count == config.numberOfDraftLayers else {
            throw DeepSeekV41Error.artifact(
                "index.json publishes \(index.draftUnits.count) DSpark units; the "
                    + "configuration says \(config.numberOfDraftLayers)")
        }
        var blocks = [DeepSeekV41BlockArtifact]()
        blocks.reserveCapacity(index.blockUnits.count)
        for (block, unit) in index.blockUnits.enumerated() {
            try cancellationCheck()
            let artifact = try DeepSeekV41BlockArtifact(
                manifestData: try loadManifest(unit.id),
                unitReference: unit.id,
                fileAccess: fileAccess,
                expectedSourceRepository: index.sourceRepository,
                expectedSourceRevision: index.sourceRevision)
            guard artifact.family == .backbone, artifact.block == block else {
                throw DeepSeekV41Error.artifact(
                    "\(unit.id) is \(artifact.family.rawValue) block \(artifact.block); "
                        + "index.json places it at backbone block \(block)")
            }
            guard artifact.manifestFileCount == unit.fileCount,
                artifact.manifestBytes == unit.bytes
            else {
                throw DeepSeekV41Error.artifact(
                    "\(unit.id)'s manifest and index.json disagree about its files or bytes")
            }
            try artifact.validateAgainst(config: config)
            blocks.append(artifact)
        }
        try cancellationCheck()
        let global = try DeepSeekV41GlobalArtifact(
            manifestData: try loadManifest(index.globalUnit.id),
            unitReference: index.globalUnit.id,
            fileAccess: fileAccess,
            expectedSourceRepository: index.sourceRepository,
            expectedSourceRevision: index.sourceRevision,
            maximumRowsPerRead: headWindowRows)
        try global.validateAgainst(config: config)

        self.config = config
        self.index = index
        self.blocks = blocks
        self.global = global
        self.readAccounting = readAccounting
        self.phaseAccounting = phaseAccounting
        self.diagnostics = diagnostics
        self.verifiesTileDigests = verifiesTileDigests
        self.adoptsExpertOperands = adoptsExpertOperands
        self.cancellationCheck = cancellationCheck
        self.headWindowRows = headWindowRows
        self.boundsLiveOperands = boundsLiveOperands
        self.expertPool = try DeepSeekV41ExpertTilePool(
            slots: expertPoolSlots,
            queueDepth: expertQueueDepth,
            fileAccess: fileAccess,
            verifiesTileDigests: verifiesTileDigests,
            readAccounting: readAccounting,
            phaseAccounting: phaseAccounting)
        self.engramRows = Array(
            repeating: [:], count: config.engramLayerIDs.count)
    }

    // MARK: - Residency

    /// The dial's census of this artifact, from the manifests it already
    /// reconciled. No payload byte is read to build it.
    public var census: DeepSeekV4MemoryDial.Census? {
        guard let embed = global.record(of: DeepSeekV41GlobalTensor.embed.rawValue),
            let head = global.record(of: DeepSeekV41GlobalTensor.head.rawValue),
            embed.shape.count == 2
        else { return nil }
        let rowBytes = UInt64(embed.shape[1] * embed.dtype.byteWidth)
        return try? DeepSeekV4MemoryDial.Census(
            layerReadBytes: blocks.map(\.denseReadBytesPerPass),
            layerResidentBytes: blocks.map(\.pinnedResidentBytes),
            globals: DeepSeekV4MemoryDial.Globals(
                headResidentBytes: head.bytes,
                headReadBytesPerToken: head.bytes,
                embeddingResidentBytes: embed.bytes,
                embeddingReadBytesPerToken: rowBytes),
            expertBytesPerToken: 0)
    }

    /// What one pool slot holds, from the container's own header.
    ///
    /// A whole pair tile when this run verifies tile digests — the digest
    /// covers the pair, so a run that checks it must hold the pair — and one
    /// expert's half of one when it does not. The memory dial prices the pool
    /// as `slots x this`, so the two modes are two different reservations and
    /// the plan says which one it made.
    public func expertTileStrideBytes() throws -> Int {
        var widest = 0
        for block in blocks {
            for projection in DeepSeekV41ExpertProjection.allCases {
                let stride = try block.expertTileStride(projection)
                let members = max(1, verifiesTileDigests ? 1 : membersPerTile(block))
                widest = max(widest, stride / members)
            }
        }
        guard widest > 0 else {
            throw DeepSeekV41Error.experts(
                "this artifact publishes no routed-expert tile stride")
        }
        return widest
    }

    /// Experts to a published tile — two, everywhere this checkpoint publishes.
    private func membersPerTile(_ block: DeepSeekV41BlockArtifact) -> Int {
        guard block.routedExpertCount > 0,
            let count = try? block.expertTileCount(.gate), count > 0
        else { return 1 }
        return max(1, block.routedExpertCount / count)
    }

    public var expertPoolBudgetBytes: UInt64 {
        (try? expertTileStrideBytes()).map {
            UInt64($0).multipliedReportingOverflow(
                by: UInt64(expertPool.slots)).partialValue
        } ?? 0
    }

    /// Hold these blocks — and, when the plan reached the globals rung, the
    /// output head — resident for the run.
    ///
    /// Called once, after the plan is accepted and before the first payload
    /// read, exactly as V4's pinned tier is.
    public func installPinnedTier(blocks pinned: Set<Int>, outputHead: Bool) throws {
        for block in pinned.sorted() {
            try cancellationCheck()
            guard block >= 0, block < blocks.count else {
                throw DeepSeekV41Error.artifact(
                    "the pin plan names block \(block), which this artifact has not")
            }
            let weights = try loadBlock(block)
            lock.lock()
            pinnedBlocks[block] = weights
            pinnedTierBytes &+= blocks[block].pinnedResidentBytes
            lock.unlock()
        }
        if outputHead {
            try cancellationCheck()
            // In `headWindowRows` windows, and not in one call: the reader
            // refuses a read wider than its own declared window, which is the
            // rule that keeps an unpinned pass from materializing 1.32 GB by
            // accident. Pinning is the one place that legitimately wants the
            // whole table, and it says so by asking for it a window at a time
            // and concatenating -- so the peak while the tier fills is one
            // window above the table, not two tables.
            let table = try measuringPhase(
                phaseAccounting,
                excludingGPUBoundaryFrom: phaseAccounting.recordOutputHeadRead(nanoseconds:)
            ) { () throws -> MLXArray in
                var windows = [MLXArray]()
                var first = 0
                while first < config.vocabularySize {
                    try cancellationCheck()
                    let count = Swift.min(headWindowRows, config.vocabularySize - first)
                    windows.append(
                        try global.loadRows(
                            .head, first: first, count: count,
                            cancellationCheck: cancellationCheck))
                    first += count
                }
                let joined = concatenated(windows, axis: 0)
                joined.eval()
                return joined
            }
            lock.lock()
            pinnedHead = table
            pinnedTierBytes &+= UInt64(
                config.vocabularySize * config.hiddenSize * 2)
            lock.unlock()
        }
    }

    public var pinnedResidentBytes: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return pinnedTierBytes
    }

    public var pinnedBlockCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return pinnedBlocks.count
    }

    public var pinsOutputHead: Bool {
        lock.lock()
        defer { lock.unlock() }
        return pinnedHead != nil
    }

    /// Tiles the pool read, and tiles a caller found already resident.
    public var expertTileReads: Int { expertPool.tileReads }
    public var expertTileHits: Int { expertPool.tileHits }

    /// Drop every resident byte. A run that finished must not still hold them.
    public func release() {
        lock.lock()
        pinnedBlocks.removeAll(keepingCapacity: false)
        pinnedHead = nil
        finalNorm = nil
        engramRows = Array(repeating: [:], count: config.engramLayerIDs.count)
        pinnedTierBytes = 0
        lock.unlock()
        expertPool.shutdown()
    }

    // MARK: - DeepSeekV41WeightSource

    public func embeddingRows(_ tokens: [Int]) throws -> MLXArray {
        try measuringPhase(
            phaseAccounting,
            excludingGPUBoundaryFrom: phaseAccounting.recordDeterministicRead(nanoseconds:)
        ) {
            try DeepSeekV41Embedding.rows(
                for: tokens, from: global, cancellationCheck: cancellationCheck)
        }
    }

    public func finalNormWeight() throws -> MLXArray {
        lock.lock()
        if let finalNorm {
            lock.unlock()
            return finalNorm
        }
        lock.unlock()
        let weight = try global.loadVector(
            .finalNorm, cancellationCheck: cancellationCheck)
        lock.lock()
        finalNorm = weight
        lock.unlock()
        return weight
    }

    /// The logits, in row windows whether the head is resident or not.
    ///
    /// Pinning changes where a window comes *from* — a slice of the resident
    /// table instead of a read — and not how wide it is, and that is
    /// deliberate. `ParallelHead.forward` is `F.linear(x.float(),
    /// weight.float())`, so a pinned bfloat16 table handed to it whole is cast
    /// to float32 whole: **2.65 GB of transient on every pass**, which is a
    /// larger allocation than the residency it was supposed to be saving. A
    /// 4,096-row window casts 84 MB instead, and phase 1 asserted that the
    /// windowed and whole forms produce the identical tensor, so the choice
    /// costs no arithmetic.
    public func logits(normalized: MLXArray) throws -> MLXArray {
        lock.lock()
        let resident = pinnedHead
        lock.unlock()
        if let resident {
            return try DeepSeekV41Head.streamedLogits(
                normalized: normalized,
                vocabularySize: config.vocabularySize,
                windowRows: headWindowRows,
                boundsLiveWindows: boundsLiveOperands,
                cancellationCheck: cancellationCheck,
                phaseAccounting: phaseAccounting
            ) { first, count in
                self.readAccounting.recordPinnedServed(
                    UInt64(count * self.config.hiddenSize * 2))
                return resident[first..<(first + count), 0...]
            }
        }
        return try DeepSeekV41Head.streamedLogits(
            normalized: normalized,
            artifact: global,
            vocabularySize: config.vocabularySize,
            windowRows: headWindowRows,
            boundsLiveWindows: boundsLiveOperands,
            cancellationCheck: cancellationCheck,
            phaseAccounting: phaseAccounting)
    }

    public func withBlock<R>(
        _ index: Int, _ body: (DeepSeekV41BlockWeights) throws -> R
    ) throws -> R {
        lock.lock()
        let held = pinnedBlocks[index]
        lock.unlock()
        if let held {
            phaseAccounting.recordPinnedServed()
            readAccounting.recordPinnedServed(blocks[index].denseReadBytesPerPass)
            return try body(held)
        }
        // Streamed: read the block's dense tensors as one unit, run, and let
        // them go. `autoreleasepool` and the scope together are what make the
        // release happen before the next block's read starts.
        return try autoreleasepool {
            let weights = try loadBlock(index)
            defer { DeepSeekV4TransientMemory.reclaim() }
            return try body(weights)
        }
    }

    /// Issue the token's Engram page reads before block 0.
    ///
    /// One batch per Engram block, one read per *distinct* 4 KiB page — the
    /// `.engrampage` layout co-locates a row's 256 values and its own eight
    /// exponents in one page, so a row is one read and fifteen rows can share
    /// one. The addresses depend only on token ids, so this is issued before
    /// any compute starts: block 1's rows have one block to hide behind and
    /// block 14's have thirteen.
    public func prefetchEngramRows(_ rows: [[[Int]]]) throws {
        guard rows.count == config.engramLayerIDs.count else {
            throw DeepSeekV41Error.engram(
                "the pass presented rows for \(rows.count) Engram blocks; this "
                    + "checkpoint has \(config.engramLayerIDs.count)")
        }
        var loaded = [[UInt64: EngramRowPageContainer.Row]]()
        loaded.reserveCapacity(rows.count)
        for (ordinal, blockID) in config.engramLayerIDs.enumerated() {
            try cancellationCheck()
            let wanted = Array(Set(rows[ordinal].flatMap { $0 }.map(UInt64.init))).sorted()
            guard !wanted.isEmpty else {
                loaded.append([:])
                continue
            }
            let started = MonotonicClock.now()
            let read = try blocks[blockID].engramRows(wanted)
            phaseAccounting.recordEngramPageRead(
                nanoseconds: MonotonicClock.nanoseconds(
                    MonotonicClock.seconds(since: started)),
                pages: wanted.count)
            var bytes = UInt64Accounting.SaturatingSum()
            for row in read { bytes.add(UInt64(row.values.count + row.scales.count)) }
            readAccounting.recordDeterministic(bytes.value)
            var table = [UInt64: EngramRowPageContainer.Row]()
            table.reserveCapacity(read.count)
            for row in read { table[row.row] = row }
            loaded.append(table)
        }
        lock.lock()
        engramRows = loaded
        lock.unlock()
    }

    // MARK: - Loading one block

    private func loadBlock(_ index: Int) throws -> DeepSeekV41BlockWeights {
        let artifact = blocks[index]
        return try measuringPhase(
            phaseAccounting,
            excludingGPUBoundaryFrom: phaseAccounting.recordDeterministicRead(nanoseconds:)
        ) {
            func matrix(_ name: DeepSeekV41BlockMatrix) throws -> BlockFP8Weights {
                try artifact.loadBlockFP8(
                    name, verifyDigest: verifiesTileDigests,
                    cancellationCheck: cancellationCheck)
            }
            func optionalMatrix(_ name: DeepSeekV41BlockMatrix) throws -> BlockFP8Weights? {
                artifact.contains(name) ? try matrix(name) : nil
            }
            func vector(_ name: DeepSeekV41BlockVector) throws -> MLXArray {
                try artifact.loadVector(name, cancellationCheck: cancellationCheck)
            }
            func optionalVector(_ name: DeepSeekV41BlockVector) throws -> MLXArray? {
                artifact.contains(name) ? try vector(name) : nil
            }

            readAccounting.recordDeterministic(artifact.denseReadBytesPerPass)
            let hyper = try DeepSeekV41DenseBlock.HyperWeights(
                artifact: artifact, cancellationCheck: cancellationCheck)
            // `attn.wo_a` is FP8 [32, 32] in our container and bfloat16 in the
            // reference, which dequantizes it at conversion time and reads it
            // through an einsum with no activation quantization at all. This is
            // the one call that reaches the reference's tensor.
            let outputDown = try DeepSeekV41FP8Linear.dequantizedToBFloat16(
                try matrix(.attentionOutputDown))
            let attention = DeepSeekV41AttentionWeights(
                sink: try vector(.attentionSink),
                queryDown: try matrix(.attentionQueryDown),
                queryNorm: try vector(.attentionQueryNorm),
                queryUp: try matrix(.attentionQueryUp),
                keyValue: try matrix(.attentionKeyValue),
                keyValueNorm: try vector(.attentionKeyValueNorm),
                outputDown: outputDown,
                outputUp: try matrix(.attentionOutputUp),
                compressorKeyValue: try optionalVector(.compressorKeyValue),
                compressorGate: try optionalVector(.compressorGate),
                compressorNorm: try optionalVector(.compressorNorm),
                indexerKey: try optionalVector(.indexerKey),
                indexerKeyNorm: try optionalVector(.indexerKeyNorm),
                indexerQueryUp: try optionalMatrix(.indexerQueryUp),
                indexerWeights: try optionalVector(.indexerWeights))

            var engram: DeepSeekV41BlockWeights.Engram?
            if let ordinal = config.engramLayerIDs.firstIndex(of: index) {
                engram = DeepSeekV41BlockWeights.Engram(
                    keyValue: try matrix(.engramKeyValue),
                    queryWeight: try vector(.engramQuery),
                    keyWeight: try vector(.engramKey),
                    rowProvider: EngramRows(artifact: self, ordinal: ordinal))
            }

            return DeepSeekV41BlockWeights(
                hyper: hyper,
                attention: attention,
                gateWeight: try vector(.gateWeight),
                gateBias: try vector(.gateBias),
                sharedGate: try matrix(.sharedExpertGate),
                sharedDown: try matrix(.sharedExpertDown),
                sharedUp: try matrix(.sharedExpertUp),
                experts: DeepSeekV41PooledExpertSource(
                    artifact: artifact, pool: expertPool,
                    adoptsOperands: adoptsExpertOperands,
                    // Three calls a token per block, which is the finest
                    // cadence a budget check has inside a prefill: the runner's
                    // own sampler only sees block boundaries, and a block
                    // boundary is measured *after* the block's weights and
                    // operands have been released. See the 2026-09-11 append to
                    // the phase 3 record, §"Why the runner did not stop first".
                    operandGuard: cancellationCheck),
                engram: engram)
        }
    }

    /// The rows one Engram module reads, out of what
    /// ``prefetchEngramRows(_:)`` brought in before block 0.
    ///
    /// A row that is missing is a refusal and not a zero row: the addresses
    /// were computed from token ids the pass already validated, so a miss means
    /// the prefetch and the operator disagree about which token they are on,
    /// and 24 zero rows would be a plausible-looking wrong answer.
    private struct EngramRows: DeepSeekV41EngramRowProvider {
        let artifact: DeepSeekV41ModelArtifact
        let ordinal: Int

        func rows(_ indices: [Int]) throws -> (values: MLXArray, scales: MLXArray) {
            artifact.lock.lock()
            let table = artifact.engramRows.indices.contains(ordinal)
                ? artifact.engramRows[ordinal] : [:]
            artifact.lock.unlock()
            var values = [UInt8]()
            var scales = [UInt8]()
            var valueWidth = 0
            var scaleWidth = 0
            values.reserveCapacity(indices.count * 256)
            for index in indices {
                guard index >= 0, let row = table[UInt64(index)] else {
                    throw DeepSeekV41Error.engram(
                        "Engram block \(ordinal) asked for row \(index), which this "
                            + "token's page reads did not bring in")
                }
                if valueWidth == 0 {
                    valueWidth = row.values.count
                    scaleWidth = row.scales.count
                }
                guard row.values.count == valueWidth, row.scales.count == scaleWidth else {
                    throw DeepSeekV41Error.engram(
                        "Engram block \(ordinal) read rows of two different widths")
                }
                values.append(contentsOf: row.values)
                scales.append(contentsOf: row.scales)
            }
            return (
                MLXArray(values, [indices.count, valueWidth]),
                MLXArray(scales, [indices.count, scaleWidth])
            )
        }
    }
}
