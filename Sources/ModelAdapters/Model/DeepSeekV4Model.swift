import Foundation

/// One executable main-layer family selected from a verified V4 config.
///
/// The repository commit is intentionally absent. A live catalogue revision
/// supplies verified bytes; those bytes select an understood operator family
/// here or receive a named refusal. The `mtp.*` units — DSpark's speculative
/// stages, not a next-token predictor head; see ADR-0017 — are not part of the
/// causal language-model forward and therefore do not appear in this plan.
public enum DeepSeekV4LayerPlan: Sendable, Equatable {
    case hashWindow(DeepSeekV4HashWindowBlock.Geometry)
    case compressedIndexed(DeepSeekV4CompressedIndexedBlock.Geometry)
    case compressedLearned(DeepSeekV4CompressedLearnedBlock.Geometry)

    public var layer: Int {
        switch self {
        case .hashWindow(let geometry): geometry.layer
        case .compressedIndexed(let geometry): geometry.layer
        case .compressedLearned(let geometry): geometry.layer
        }
    }

    public var compressionRatio: Int {
        switch self {
        case .hashWindow: 0
        case .compressedIndexed: 4
        case .compressedLearned(let geometry): geometry.attention.compressionRatio
        }
    }

    public var usesHashRouting: Bool {
        switch self {
        case .hashWindow: true
        case .compressedIndexed(let geometry): geometry.routingKind == .tokenHash
        case .compressedLearned: false
        }
    }
}

/// The complete ordered main-layer schedule derived from verified config
/// bytes.
///
/// This is the first full-model admission boundary. It refuses a future V4
/// combination for which the adapter has no implementation instead of
/// silently mapping that layer to a nearby operator. The current published
/// architecture resolves to 43 consecutive layers spanning all four supported
/// routing/compression combinations.
public struct DeepSeekV4ModelPlan: Sendable, Equatable {
    public let config: DeepSeekV4Config
    public let layers: [DeepSeekV4LayerPlan]

    public init(config: DeepSeekV4Config) throws {
        var layers = [DeepSeekV4LayerPlan]()
        layers.reserveCapacity(config.numberOfLayers)

        for layer in 0..<config.numberOfLayers {
            let ratio = try config.compressionRatio(layer: layer)
            let usesHash = config.usesHashRouting(layer: layer)
            switch (usesHash, ratio) {
            case (true, 0):
                layers.append(.hashWindow(
                    try DeepSeekV4HashWindowBlock.Geometry(
                        config: config, layer: layer)))

            case (true, 4):
                layers.append(.compressedIndexed(
                    try DeepSeekV4CompressedIndexedBlock.Geometry(
                        config: config, layer: layer, routingKind: .tokenHash)))

            case (false, 4):
                layers.append(.compressedIndexed(
                    try DeepSeekV4CompressedIndexedBlock.Geometry(
                        config: config, layer: layer, routingKind: .learned)))

            case (false, 128):
                layers.append(.compressedLearned(
                    try DeepSeekV4CompressedLearnedBlock.Geometry(
                        config: config, layer: layer)))

            case (true, 128):
                throw DeepSeekV4Error.unsupportedArchitecture(
                    "layer \(layer) combines token-hash routing with ratio-128 compression")

            case (false, 0):
                // This is exactly the signature of a DSpark stage: index >=
                // num_hidden_layers, so outside num_hash_layers, a real
                // 256x4096 router, and `compress_ratios` 0 for its 128-wide
                // window (`DSparkAttention` asserts `compress_ratio == 0`).
                // Implementing that fourth family is blocker B1 of ADR-0017.
                throw DeepSeekV4Error.unsupportedArchitecture(
                    "layer \(layer) combines learned routing with uncompressed window attention")

            default:
                // DeepSeekV4Config currently admits only 0, 4, and 128. Keep
                // this explicit default so a later config expansion cannot
                // silently enter the model plan.
                throw DeepSeekV4Error.unsupportedArchitecture(
                    "layer \(layer) uses unsupported compression ratio \(ratio)")
            }
        }

        guard layers.count == config.numberOfLayers,
            layers.enumerated().allSatisfy({ $0.offset == $0.element.layer })
        else {
            throw DeepSeekV4Error.configuration(
                "main-layer plan is not a complete consecutive schedule")
        }
        self.config = config
        self.layers = layers
    }
}

/// The publication-level inventory decoded from a fully verified V4
/// `index.json`.
///
/// The source revision is evidence carried by the current publication, not a
/// value compiled into the adapter. Main-layer unit names, however, are an
/// executable layout contract: they must exactly cover the layers selected by
/// the verified config. Auxiliary units remain visible in ``units`` and in the
/// byte total but are not silently inserted into the causal forward.
public struct DeepSeekV4ArtifactIndex: Sendable, Equatable {
    public struct Unit: Sendable, Equatable {
        public let id: String
        public let fileCount: Int
        public let bytes: UInt64

        fileprivate init(id: String, fileCount: Int, bytes: UInt64) {
            self.id = id
            self.fileCount = fileCount
            self.bytes = bytes
        }
    }

    public let sourceRepository: String
    public let sourceRevision: String
    public let relationship: String
    public let units: [Unit]
    public let mainLayerUnits: [Unit]
    public let globalUnit: Unit
    public let totalBytes: UInt64

    public init(json data: Data, config: DeepSeekV4Config) throws {
        let document: Document
        do {
            document = try JSONDecoder().decode(Document.self, from: data)
        } catch {
            throw DeepSeekV4Error.artifact(
                "index.json could not be decoded: \(error)")
        }
        guard !document.sourceRepository.isEmpty,
            !document.sourceRevision.isEmpty
        else {
            throw DeepSeekV4Error.artifact(
                "index source repository and revision must be non-empty")
        }
        guard document.relationship == "byte-preserving repack" else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "index relationship '\(document.relationship)' is not a byte-preserving repack")
        }
        guard !document.units.isEmpty else {
            throw DeepSeekV4Error.artifact("index contains no artifact units")
        }

        var seen = Set<String>()
        var units = [Unit]()
        units.reserveCapacity(document.units.count)
        var accountedBytes: UInt64 = 0
        for record in document.units {
            guard Self.isSafeComponent(record.id), seen.insert(record.id).inserted else {
                throw DeepSeekV4Error.artifact(
                    "index contains an unsafe or repeated unit '\(record.id)'")
            }
            if record.id.hasPrefix("layers"), Self.layerNumber(record.id) == nil {
                throw DeepSeekV4Error.artifact(
                    "index contains malformed main-layer unit '\(record.id)'")
            }
            guard record.fileCount > 0, record.bytes > 0 else {
                throw DeepSeekV4Error.artifact(
                    "index unit '\(record.id)' must contain files and bytes")
            }
            let next = accountedBytes.addingReportingOverflow(record.bytes)
            guard !next.overflow else {
                throw DeepSeekV4Error.artifact(
                    "index unit byte total exceeds UInt64.max")
            }
            accountedBytes = next.partialValue
            units.append(Unit(
                id: record.id, fileCount: record.fileCount, bytes: record.bytes))
        }
        guard accountedBytes == document.totalBytes else {
            throw DeepSeekV4Error.artifact(
                "index units account for \(accountedBytes) bytes; total_bytes is "
                    + "\(document.totalBytes)")
        }

        let byID = Dictionary(uniqueKeysWithValues: units.map { ($0.id, $0) })
        let expectedMainIDs = (0..<config.numberOfLayers).map(Self.mainUnitID)
        let main = try expectedMainIDs.map { id in
            guard let unit = byID[id] else {
                throw DeepSeekV4Error.artifact(
                    "index has no required main-layer unit '\(id)'")
            }
            return unit
        }
        let publishedMainIDs = Set(units.compactMap { unit in
            Self.layerNumber(unit.id).map { _ in unit.id }
        })
        guard publishedMainIDs == Set(expectedMainIDs) else {
            let extras = publishedMainIDs.subtracting(expectedMainIDs).sorted()
            throw DeepSeekV4Error.artifact(
                "index main-layer units do not exactly match config"
                    + (extras.isEmpty ? "" : "; unexpected: \(extras.joined(separator: ", "))"))
        }
        guard let global = byID["global00"] else {
            throw DeepSeekV4Error.artifact(
                "index has no required global unit 'global00'")
        }
        let otherGlobal = units.map(\.id).filter {
            $0.hasPrefix("global") && $0 != "global00"
        }
        guard otherGlobal.isEmpty else {
            throw DeepSeekV4Error.artifact(
                "index contains unsupported global units: \(otherGlobal.joined(separator: ", "))")
        }

        sourceRepository = document.sourceRepository
        sourceRevision = document.sourceRevision
        relationship = document.relationship
        self.units = units
        mainLayerUnits = main
        globalUnit = global
        totalBytes = document.totalBytes
    }

    private struct Document: Decodable {
        struct Unit: Decodable {
            let id: String
            let fileCount: Int
            let bytes: UInt64

            enum CodingKeys: String, CodingKey {
                case id = "unit"
                case fileCount = "files"
                case bytes
            }
        }

        let sourceRepository: String
        let sourceRevision: String
        let relationship: String
        let units: [Unit]
        let totalBytes: UInt64

        enum CodingKeys: String, CodingKey {
            case sourceRepository = "source_repo"
            case sourceRevision = "source_revision"
            case relationship, units
            case totalBytes = "total_bytes"
        }
    }

    private static func mainUnitID(_ layer: Int) -> String {
        layer < 10 ? "layers0\(layer)" : "layers\(layer)"
    }

    private static func layerNumber(_ unit: String) -> Int? {
        guard unit.hasPrefix("layers") else { return nil }
        let suffix = unit.dropFirst("layers".count)
        guard !suffix.isEmpty, suffix.allSatisfy(\.isNumber) else { return nil }
        return Int(suffix)
    }

    private static func isSafeComponent(_ value: String) -> Bool {
        !value.isEmpty && value != "." && value != ".."
            && !value.contains("/") && !value.contains("\\")
            && !value.contains("\0")
    }
}

/// A complete, metadata-admitted V4 causal model rooted in one verified
/// publication.
///
/// Construction is intentionally expensive in metadata and cheap in payload:
/// every unit manifest is decoded and reconciled with `index.json`, every main
/// layer container header is opened beneath the caller's rooted
/// ``ModelFileAccess``, and every tensor contract is checked before this value
/// is returned. No matrix, expert tile, embedding row, or output-head row is
/// materialized. Execution can therefore walk ``layers`` one at a time without
/// discovering a missing layer-42 tensor after earlier payloads were consumed.
///
/// The repository commit is not an adapter constant. `sourceRevision` is the
/// identity declared by the currently verified publication; a future commit
/// is admitted whenever its config and complete artifact contract remain
/// compatible.
public final class DeepSeekV4ModelArtifact {
    public struct Layer {
        public let plan: DeepSeekV4LayerPlan
        public let artifact: DeepSeekV4LayerArtifact
        public let expertUnit: DeepSeekV4ExpertUnit

        fileprivate init(
            plan: DeepSeekV4LayerPlan,
            artifact: DeepSeekV4LayerArtifact,
            expertUnit: DeepSeekV4ExpertUnit
        ) {
            self.plan = plan
            self.artifact = artifact
            self.expertUnit = expertUnit
        }
    }

    public typealias ManifestLoader = (_ unitID: String) throws -> Data

    public let plan: DeepSeekV4ModelPlan
    public let index: DeepSeekV4ArtifactIndex
    public let layers: [Layer]
    public let global: DeepSeekV4GlobalArtifact
    public let auxiliaryUnits: [DeepSeekV4ArtifactIndex.Unit]
    public let readAccounting: DeepSeekV4ReadAccounting
    /// Run-scoped wall-time brackets for the phases one pass moves through.
    /// Cumulative like ``readAccounting``; a pass is the difference of two
    /// snapshots. See ``DeepSeekV4PhaseMetrics``.
    public let phaseAccounting: DeepSeekV4PhaseAccounting

    public var sourceRepository: String { index.sourceRepository }
    public var sourceRevision: String { index.sourceRevision }

    /// The memory dial's pinned tier, once a stated budget has installed one.
    private var pinnedTier: DeepSeekV4PinnedWeightCache?

    /// Hold `layers` — and, when the plan reached the globals rung, the output
    /// head table — resident for the life of this artifact.
    ///
    /// Called once, after the plan is accepted and before the first payload
    /// read. The tier fills lazily — each tensor is adopted by the pass that
    /// first reads it, through the same descriptor, the same `pread` loop and
    /// the same ``TileDigestPolicy`` as the serial path — so this call
    /// allocates nothing and cannot fail.
    public func installPinnedTier(
        layers: Set<Int>, budgetBytes: UInt64, outputHead: Bool = false
    ) {
        guard !layers.isEmpty || outputHead, pinnedTier == nil else { return }
        let cache = DeepSeekV4PinnedWeightCache(
            layers: layers, budgetBytes: budgetBytes, pinsOutputHead: outputHead)
        for layer in self.layers where layers.contains(layer.artifact.layer) {
            layer.artifact.installPinnedWeights(cache)
        }
        if outputHead { global.installPinnedTier(cache) }
        pinnedTier = cache
    }

    /// What the tier held and what it saved. Nil when nothing was pinned.
    public var pinnedTierMetrics: DeepSeekV4PinnedTierMetrics? { pinnedTier?.metrics }

    /// Free every pinned byte, the head table included. Idempotent; the metrics
    /// survive the release so a terminal record can still say what the tier did.
    public func releasePinnedTier() {
        global.releasePinnedOutputHead()
        pinnedTier?.release()
    }

    /// Build a complete model view from already-bounded metadata reads and one
    /// rooted payload opener. Product code obtains both from the same full
    /// verification authority; tests and command-line probes may use a private
    /// fixture root. `loadManifest("layers03")`, for example, must return the
    /// verified bytes of `layers03/manifest.json`.
    ///
    /// `tileDigestPolicy` reaches every layer artifact and defaults to
    /// ``TileDigestPolicy/verifyEveryLoad``. Only a caller that holds complete
    /// verification authority for this exact revision may state the other one.
    ///
    /// `diagnostics` reaches every layer and the global unit the same way, and
    /// defaults to ``DeepSeekV4Diagnostics/validating``. A caller that wants
    /// the model's arithmetic without the guard-only finiteness sweeps — the
    /// product runner does — states ``DeepSeekV4Diagnostics/off``.
    public init(
        configData: Data,
        indexData: Data,
        fileAccess: ModelFileAccess,
        tileDigestPolicy: TileDigestPolicy = .verifyEveryLoad,
        maximumHeadRowsPerRead: Int = 4_096,
        readAccounting: DeepSeekV4ReadAccounting = DeepSeekV4ReadAccounting(),
        phaseAccounting: DeepSeekV4PhaseAccounting = DeepSeekV4PhaseAccounting(),
        diagnostics: DeepSeekV4Diagnostics = .validating,
        cancellationCheck: () throws -> Void = {},
        loadManifest: ManifestLoader
    ) throws {
        let config = try DeepSeekV4Config(json: configData)
        let plan = try DeepSeekV4ModelPlan(config: config)
        let index = try DeepSeekV4ArtifactIndex(json: indexData, config: config)
        let manifests = try Self.admitPublicationMetadata(
            config: config,
            index: index,
            cancellationCheck: cancellationCheck,
            loadManifest: loadManifest)
        let expertGeometry = try DeepSeekV4ExpertGeometry(config: config)

        var layers = [Layer]()
        layers.reserveCapacity(plan.layers.count)
        for (layerPlan, unit) in zip(plan.layers, index.mainLayerUnits) {
            try cancellationCheck()
            guard let manifest = manifests[unit.id] else {
                throw DeepSeekV4Error.artifact(
                    "publication admission lost manifest for '\(unit.id)'")
            }
            let artifact = try DeepSeekV4LayerArtifact(
                manifestData: manifest,
                unitReference: unit.id,
                fileAccess: fileAccess,
                tileDigestPolicy: tileDigestPolicy,
                readAccounting: readAccounting,
                phaseAccounting: phaseAccounting,
                diagnostics: diagnostics,
                expectedSourceRepository: index.sourceRepository,
                expectedSourceRevision: index.sourceRevision)
            let expertUnit = try DeepSeekV4ExpertUnit(
                manifestData: manifest,
                unitReference: unit.id,
                fileAccess: fileAccess,
                geometry: expertGeometry,
                expectedSourceRepository: index.sourceRepository,
                expectedSourceRevision: index.sourceRevision)
            guard artifact.manifestFileCount == unit.fileCount,
                artifact.manifestBytes == unit.bytes
            else {
                throw DeepSeekV4Error.artifact(
                    "\(unit.id) payload accounting disagrees with index.json")
            }

            switch layerPlan {
            case .hashWindow(let geometry):
                try DeepSeekV4HashWindowBlock.validateArtifactContract(
                    artifact: artifact,
                    expertUnit: expertUnit,
                    geometry: geometry,
                    vocabularySize: config.vocabularySize)

            case .compressedIndexed(let geometry):
                let vocabularySize = geometry.routingKind == .tokenHash
                    ? config.vocabularySize : nil
                try DeepSeekV4CompressedIndexedBlock.validateArtifactContract(
                    artifact: artifact,
                    expertUnit: expertUnit,
                    geometry: geometry,
                    vocabularySize: vocabularySize)

            case .compressedLearned(let geometry):
                try DeepSeekV4CompressedLearnedBlock.validateArtifactContract(
                    artifact: artifact,
                    expertUnit: expertUnit,
                    geometry: geometry)
            }
            layers.append(Layer(
                plan: layerPlan, artifact: artifact, expertUnit: expertUnit))
        }
        guard layers.count == config.numberOfLayers,
            layers.enumerated().allSatisfy({
                $0.offset == $0.element.plan.layer
                    && $0.element.artifact.layer == $0.offset
                    && $0.element.expertUnit.layer == $0.offset
            })
        else {
            throw DeepSeekV4Error.artifact(
                "admitted layers are not the complete consecutive model schedule")
        }

        try cancellationCheck()
        let globalUnit = index.globalUnit
        guard let globalManifest = manifests[globalUnit.id] else {
            throw DeepSeekV4Error.artifact(
                "publication admission lost manifest for 'global00'")
        }
        let global = try DeepSeekV4GlobalArtifact(
            manifestData: globalManifest,
            unitReference: globalUnit.id,
            fileAccess: fileAccess,
            geometry: try DeepSeekV4GlobalArtifact.Geometry(
                config: config,
                maximumHeadRowsPerRead: maximumHeadRowsPerRead),
            readAccounting: readAccounting,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics,
            expectedSourceRepository: index.sourceRepository,
            expectedSourceRevision: index.sourceRevision)
        guard global.manifestFileCount == globalUnit.fileCount,
            global.manifestBytes == globalUnit.bytes
        else {
            throw DeepSeekV4Error.artifact(
                "global00 payload accounting disagrees with index.json")
        }

        self.plan = plan
        self.index = index
        self.layers = layers
        self.global = global
        self.auxiliaryUnits = Self.auxiliaryUnits(index: index)
        self.readAccounting = readAccounting
        self.phaseAccounting = phaseAccounting
    }

    private struct ManifestSummary: Decodable {
        struct File: Decodable {
            let name: String
            let bytes: UInt64
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

    /// Metadata-only half of full-model admission. Kept internal so tests can
    /// prove the current 47-unit publication is complete without fabricating
    /// 166 GB of payloads. The returned data is bounded as a whole and is the
    /// exact data later consumed by the concrete layer/global constructors.
    static func admitPublicationMetadata(
        config: DeepSeekV4Config,
        index: DeepSeekV4ArtifactIndex,
        maximumManifestBytes: Int = 8 << 20,
        maximumTotalManifestBytes: Int = 64 << 20,
        cancellationCheck: () throws -> Void = {},
        loadManifest: ManifestLoader
    ) throws -> [String: Data] {
        guard maximumManifestBytes > 0,
            maximumTotalManifestBytes >= maximumManifestBytes
        else {
            throw DeepSeekV4Error.configuration(
                "manifest metadata bounds must be positive and ordered")
        }
        _ = try validateAuxiliaryUnitSet(config: config, index: index)

        var result = [String: Data]()
        result.reserveCapacity(index.units.count)
        var totalManifestBytes = 0
        for unit in index.units {
            try cancellationCheck()
            let data = try loadManifest(unit.id)
            guard !data.isEmpty, data.count <= maximumManifestBytes else {
                throw DeepSeekV4Error.artifact(
                    "\(unit.id)/manifest.json is empty or exceeds "
                        + "\(maximumManifestBytes) bytes")
            }
            let nextTotal = totalManifestBytes.addingReportingOverflow(data.count)
            guard !nextTotal.overflow,
                nextTotal.partialValue <= maximumTotalManifestBytes
            else {
                throw DeepSeekV4Error.artifact(
                    "publication manifests exceed the \(maximumTotalManifestBytes)-byte bound")
            }
            totalManifestBytes = nextTotal.partialValue

            let summary: ManifestSummary
            do {
                summary = try JSONDecoder().decode(ManifestSummary.self, from: data)
            } catch {
                throw DeepSeekV4Error.artifact(
                    "\(unit.id)/manifest.json could not be decoded: \(error)")
            }
            guard summary.unit == unit.id,
                summary.sourceRepository == index.sourceRepository,
                summary.sourceRevision == index.sourceRevision,
                !summary.files.isEmpty
            else {
                throw DeepSeekV4Error.artifact(
                    "\(unit.id)/manifest.json does not share the index identity")
            }
            var names = Set<String>()
            var payloadBytes: UInt64 = 0
            for file in summary.files {
                guard Self.isSafeComponent(file.name),
                    names.insert(file.name).inserted,
                    file.bytes > 0
                else {
                    throw DeepSeekV4Error.artifact(
                        "\(unit.id) contains an unsafe, repeated, or empty payload")
                }
                let next = payloadBytes.addingReportingOverflow(file.bytes)
                guard !next.overflow else {
                    throw DeepSeekV4Error.artifact(
                        "\(unit.id) payload byte total exceeds UInt64.max")
                }
                payloadBytes = next.partialValue
            }
            guard summary.files.count == unit.fileCount,
                payloadBytes == unit.bytes
            else {
                throw DeepSeekV4Error.artifact(
                    "\(unit.id) manifest accounts for \(summary.files.count) files and "
                        + "\(payloadBytes) bytes; index.json declares \(unit.fileCount) "
                        + "files and \(unit.bytes) bytes")
            }
            result[unit.id] = data
        }
        guard result.count == index.units.count else {
            throw DeepSeekV4Error.artifact(
                "publication metadata did not produce one manifest per index unit")
        }
        return result
    }

    /// Admit the publication's non-main, non-global units.
    ///
    /// These are named `mtp00…`, and the name is the only thing about them that
    /// says "MTP": they are DeepSeek's **DSpark** speculative stages, stored
    /// under the checkpoint's `mtp.*` namespace (ADR-0017). The published
    /// artifact carries three of them — three complete blocks, each with its
    /// own 256 routed FP4 experts — while the transformers-side
    /// `num_nextn_predict_layers` is 1. That field therefore does **not** count
    /// these units, and the check below is deliberately presence-only: a
    /// publication either has DSpark stages and declares a predictor count, or
    /// has neither. What is enforced strictly is the unit-name contract —
    /// consecutive, canonically formatted, no unknown auxiliary unit — because
    /// that is what keeps an unrecognized unit from arriving unnoticed.
    @discardableResult
    static func validateAuxiliaryUnitSet(
        config: DeepSeekV4Config,
        index: DeepSeekV4ArtifactIndex
    ) throws -> [DeepSeekV4ArtifactIndex.Unit] {
        let main = Set(index.mainLayerUnits.map(\.id))
        let auxiliary = index.units.filter {
            !main.contains($0.id) && $0.id != index.globalUnit.id
        }
        let indices = auxiliary.compactMap { Self.mtpIndex($0.id) }.sorted()
        let canonical = indices.map(Self.mtpUnitID)
        guard canonical == auxiliary.map(\.id).sorted() else {
            let unknown = auxiliary.map(\.id).filter { Self.mtpIndex($0) == nil }.sorted()
            throw DeepSeekV4Error.artifact(
                "publication contains unsupported auxiliary units: "
                    + unknown.joined(separator: ", "))
        }
        let expectedIndices = Array(0..<indices.count)
        guard indices == expectedIndices else {
            let actual = Set(indices)
            let expected = Set(0...(indices.max() ?? 0))
            let missing = expected.subtracting(actual).sorted().map(Self.mtpUnitID)
            throw DeepSeekV4Error.artifact(
                "DSpark (mtp.*) publication units are not consecutive"
                    + (missing.isEmpty ? "" : "; missing: \(missing.joined(separator: ", "))"))
        }
        guard (config.numberOfMTPPredictors == 0) == auxiliary.isEmpty else {
            throw DeepSeekV4Error.artifact(
                "DSpark (mtp.*) publication presence disagrees with "
                    + "num_nextn_predict_layers")
        }
        return auxiliary.sorted { $0.id < $1.id }
    }

    private static func auxiliaryUnits(
        index: DeepSeekV4ArtifactIndex
    ) -> [DeepSeekV4ArtifactIndex.Unit] {
        let main = Set(index.mainLayerUnits.map(\.id))
        return index.units.filter {
            !main.contains($0.id) && $0.id != index.globalUnit.id
        }.sorted { $0.id < $1.id }
    }

    private static func isSafeComponent(_ value: String) -> Bool {
        !value.isEmpty && value != "." && value != ".."
            && !value.contains("/") && !value.contains("\\")
            && !value.unicodeScalars.contains(where: { $0.value == 0 })
    }

    private static func mtpIndex(_ unitID: String) -> Int? {
        guard unitID.hasPrefix("mtp") else { return nil }
        let suffix = unitID.dropFirst(3)
        guard suffix.count >= 2, suffix.allSatisfy(\.isNumber),
            let index = Int(suffix), mtpUnitID(index) == unitID
        else { return nil }
        return index
    }

    private static func mtpUnitID(_ index: Int) -> String {
        index < 10 ? "mtp0\(index)" : "mtp\(index)"
    }
}
