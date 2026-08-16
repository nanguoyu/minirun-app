import Foundation
import MinirunKit
import MinirunRunners

// =============================================================================
// MARK: - Preview catalog
// =============================================================================
//
// Which numbers here are transcribed, and which are placeholders. This
// distinction is the point of the file, so it is at the top rather than in a
// footnote:
//
// TRANSCRIBED — from the MinirunKit API design's §0, which walked the live
// Hugging Face tree on 2026-08-11, and from the M10 device record:
//
//   * the three repo ids, their pinned 40-hex commits, and for each of them the
//     payload/metadata byte counts, the payload/metadata file counts and the
//     largest single file;
//   * K3 product chat's platform policy: the supported Mac boundary is
//     8,000,000,000 B, while the bounded iPhone tier uses the twice-completed
//     5,800,000,000 B device record and a two-token ceiling; V4's measured
//     Apple-silicon macOS product boundary;
//   * K3's widest deterministic layer as stored, 2,341,257,216 B, its widened
//     4,682,514,432 B, the 2,581,886,208 B product working reserve and the
//     1,267,810,304 B staged-pair peak;
//   * the one-token demonstration: token 3372, twice, bit-identical logits,
//     5.26 GB peak inside a 5.80 GB stated budget, thermally nominal.
//
// PLACEHOLDER — invented so the screens have something to draw, and marked as
// such wherever they surface. No claim rests on any of them:
//
//   * every per-expert byte size and hot-set saturation slot count;
//   * V4's projected memory-dial ladder, which spreads a recorded per-pass
//     block-FP8 total evenly across 43 layers because the per-layer split is
//     not recorded. It is replaced by the artifact's own manifests before any
//     V4 run, and `deepSeekV4ProjectedLadder` says exactly what it assumes;
//   * H3's memory profile (it has no chat-product runner in this build);
//   * every `WorkloadTerms` compute term;
//   * the seeded link calibration and the seeded run history.
//
// The shipping App never iterates this file as its catalog. MinirunKit's
// bundled/live `ModelCatalogSnapshot` defines that open set; these fixtures
// only supplement App-only memory/workload geometry for known preview models.
// Unknown descriptors keep honest unknown/zero geometry rather than being
// dropped. See DESIGN.md.

/// The nine scalar fields a published `index.json` carries. No per-file digests
/// live here — the complete digest table is the repo tree, not the index.
struct ArtifactIndexSummary: Equatable, Sendable {
    /// Nil where the published index does not declare one. Rendering a layer
    /// count we do not have would be inventing one.
    let layers: Int?
    let globals: Int?
    let files: Int
    let bytes: UInt64
    let sourceRepo: String?
    let sourceRevision: String?

    var claim: IndexClaim {
        IndexClaim(
            declaredFileCount: files, declaredBytes: bytes,
            sourceRepo: sourceRepo, sourceRevision: sourceRevision)
    }
}

/// One catalog entry, with everything the product's screens need beside the
/// descriptor the kit will vend.
struct CatalogEntry: Equatable, Sendable, Identifiable {
    let descriptor: ModelDescriptor
    let index: ArtifactIndexSummary
    let memory: MemoryProfile
    let terms: WorkloadTerms
    /// Sentence for the catalog row when there is no runner.
    let architectureSubtitle: String

    var id: ModelID { descriptor.id }

    /// Where the bytes come from, in one line.
    ///
    /// The name above it is the model's real name — Kimi K3, not `K3-flagship`
    /// — so the line that disambiguates two builds of the same model is the
    /// repository, not a suffix bolted onto the name. Locally built artifacts
    /// name the tool instead, because that is their provenance.
    var provenanceSubtitle: String {
        switch descriptor.source {
        case .huggingFaceRepo(let repo): return repo.repoID
        case .locallyBuilt(let tool): return "built locally by \(tool)"
        }
    }
}

enum CatalogFixtures {

    // MARK: Byte constants, spelled out so a diff shows a changed digit

    static let k3PayloadBytes: UInt64 = 1_559_976_181_760
    static let k3MetadataBytes: UInt64 = 19_614_826
    static let k3WidestDeterministicLayerStored: UInt64 = 2_341_257_216
    static let workingSetReserve = K3ProductMemoryBudget.flagshipDefaultWorkingReserveBytes
    static let k3StagedPairPeak: UInt64 = 1_267_810_304
    static let k3ProductMinimumBudget = K3ProductMemoryBudget.minimumBudgetBytes
    static let k3OnRecordMinimumBudget: UInt64 = 5_800_000_000
    /// The stated routed-expert pool, M9C. Part of the working floor.
    static let k3ExpertPoolBytes: UInt64 = 46_792_704

    #if os(iOS)
        static let k3ProductMemoryNote =
            "On iPhone, K3 uses the verified 5.80 GB device boundary as a bounded "
            + "experimental tier and limits each response to two generated tokens."
        static let k3ProductTokenNote =
            "The two-token ceiling bounds the first retained-state transition while the "
            + "iPhone latent-cache profile is validated."
    #else
        static let k3ProductMemoryNote =
            "Product chat reserves the widened layer, routed-expert pool, rolling KDA/MLA "
            + "state and allocator headroom. Its exact default floor is 7.31 GB; the "
            + "runner uses a conservative 8.00 GB boundary."
        static let k3ProductTokenNote =
            "Full-model timing records remain tied to the token counts they actually ran; "
            + "the Mac product decode loop supports up to 64 generated tokens."
    #endif

    // MARK: Censuses

    /// K3's census is `ArtifactCensus.measuredFlagship()` — the kit's own, the
    /// same value `MemoryDialPlannerTests` and the decode loop use. Nothing here
    /// re-types those 93 layer sizes, which is the whole point: the dial and the
    /// run cannot disagree about what the artifact is.
    ///
    /// `try!` is load-bearing rather than lazy. The census's initialiser refuses
    /// a ragged or self-contradicting layer table, and the flagship one is a
    /// constant in the kit; if it ever stopped validating, a preview catalog
    /// quietly falling back to an empty model would hide the fact.
    static let k3Census: ArtifactCensus = try! ArtifactCensus.measuredFlagship()

    /// PROJECTION. DeepSeek V4's ladder before any copy of the artifact has
    /// been read, so the dial has something to draw and a new chat has a preset
    /// to start from before the first verification completes.
    ///
    /// It is replaced, model-wide, the moment `DeepSeekV4MemoryDialInputs`
    /// reads the real layer manifests — which is the only source of a number
    /// this screen will run against. Until then:
    ///
    /// * **43 layers of 136,495,488 B read.** The block-FP8 sum a pass reads is
    ///   recorded (5,869,305,856 B, `docs/design/v4-memory-dial.md` §4); the
    ///   split across layers is not, so it is spread evenly and rounded to a
    ///   per-layer multiple of 32 — 128 B over 5.87 GB. The real per-layer
    ///   table also carries BF16 plain tensors this does not, so the projected
    ///   ladder is slightly *cheaper* than the measured one; the measured one
    ///   replaces it before anything runs.
    /// * **The globals rung at the published table shape.** `129280 × 4096`
    ///   BF16 is 1,059,061,760 B for each of the two tables. The head is read
    ///   whole every pass, so pinning it saves exactly what it costs; the
    ///   embedding is read one 8,192 B row per decode token, which is why it is
    ///   censused and never ranked (`v4-memory-dial.md` §3.8).
    /// * **Resident at 33/32 of read**, the scale-grid expansion the design
    ///   record derives (§3.1), which is what makes a V4 layer score ~0.970
    ///   rather than K3's 1.000.
    /// * **The floor at §4's own terms**: the 1.8 GB measured transient
    ///   execution envelope, the routed-expert pool, the saturated 512 MiB MLX
    ///   cache and the stated 256 MiB pin margin. Retained generation state and
    ///   the one-layer replacement are prompt-dependent and small for a chat,
    ///   as §4 says; the measured ladder prices them exactly.
    static let deepSeekV4ProjectedLadder: DeepSeekV4MemoryDialInputs.ArtifactProfile = {
        let read = [UInt64](repeating: 136_495_488, count: 43)
        let census = try! DeepSeekV4MemoryDial.Census(
            layerReadBytes: read,
            layerResidentBytes: read.map { $0 * 33 / 32 },
            globals: DeepSeekV4MemoryDial.Globals(
                headResidentBytes: 1_059_061_760,
                headReadBytesPerToken: 1_059_061_760,
                embeddingResidentBytes: 1_059_061_760,
                embeddingReadBytesPerToken: 8_192),
            expertBytesPerToken: 0)
        let expertPool: UInt64 = 17_826_048
        return DeepSeekV4MemoryDialInputs.ArtifactProfile(
            census: census,
            floor: WorkingSetFloor(
                widestResidentLayerBytes: DeepSeekV4ProductMemoryBudget.transientExecutionBytes,
                expertPoolBytes: expertPool,
                workingReserveBytes: 536_870_912
                    + DeepSeekV4MemoryDialInputs.statedPinMarginBytes),
            expertPoolBytes: expertPool)
    }()

    /// A model with no runner and no published geometry. One zero-sized layer,
    /// so the census is well-formed and nothing is pinnable — which is the
    /// truth: there is no ladder to climb until somebody reads the containers.
    static let unknownGeometryCensus: ArtifactCensus = try! ArtifactCensus(
        layerStoredBytes: [0], layerResidentBytes: [0],
        globalsStoredBytes: 0, globalsBytesReadPerToken: 0,
        expertStrideBytes: 0, expertsPerLayer: 0, routedExpertsPerToken: 0,
        moeLayerCount: 0, measuredDeterministicBytesPerToken: 0)

    // MARK: Entries

    static let kimiK3 = CatalogEntry(
        descriptor: ModelDescriptor(
            id: .kimiK3,
            displayName: "Kimi K3",
            architecture: .kimiK3MoE,
            layout: .k3FlagshipLayerStreams,
            source: .huggingFaceRepo(
                HuggingFaceRepoRef(
                    repoID: "nanguoyu/Kimi-K3-minirun",
                    revision: "4715a509773c958e26a31f5e07dd2897f20e0400")),
            payloadBytes: k3PayloadBytes,
            metadataBytes: k3MetadataBytes,
            payloadFileCount: 372,
            metadataFileCount: 99,
            largestFileBytes: 5_240_799_232,
            minimumBudgetBytes: k3ProductMinimumBudget,
            runner: .decodeRunner,
            licenseName: "Modified MIT",
            licenseAcknowledgementRequired: true,
            notes: [
                "A byte-preserving repack. No quantization was applied to publish it, and none "
                    + "is applied to run it.",
                k3ProductMemoryNote,
                k3ProductTokenNote,
            ]),
        index: ArtifactIndexSummary(
            layers: 93, globals: 1, files: 372, bytes: k3PayloadBytes,
            sourceRepo: "nanguoyu/Kimi-K3-minirun",
            sourceRevision: "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"),
        memory: MemoryProfile(
            census: k3Census,
            expertPoolBytes: k3ExpertPoolBytes,
            workingSetReserveBytes: workingSetReserve,
            requiredMinimumBudgetBytes: k3ProductMinimumBudget,
            onRecordMinimumBudgetBytes: k3OnRecordMinimumBudget,
            provenance: .declaredByIndex),
        // The two byte terms are the measured per-token split. The compute term
        // is the residual: it is set so that the projection at the historical
        // one-token budget, over the link on record, lands on the pass actually
        // measured — 5:57 a token. Fitting the one unmeasured term to the
        // recorded outcome is what makes the curve a restatement of the record
        // rather than a guess sitting next to it. `SpeedProjectionTests` pins it.
        //
        // 151.89 and not 160.64. The old value was fitted while the projector
        // still gave the budget an expert-cache credit, and at 5,800,000,000 B
        // that credit was worth 8.75 s the run never got. The derived tier
        // product ladder says no budget below 116,129,117,440 B pins a single expert, so
        // the credit is gone and the residual absorbs what it was hiding. Same
        // recorded outcome, one fewer invented term underneath it.
        terms: WorkloadTerms(
            deterministicBytesPerToken: 111_200_000_000,
            expertBytesPerToken: 77_500_000_000,
            computeSecondsPerToken: 151.89,
            isMeasuredHere: false),
        architectureSubtitle: "Kimi K3 MoE")

    static let deepseekV4Flash = CatalogEntry(
        descriptor: ModelDescriptor(
            id: .deepseekV4Flash,
            displayName: "DeepSeek V4 Flash",
            architecture: .deepseekV4FlashMoE,
            layout: .v4FlashUnitBundle,
            source: .huggingFaceRepo(
                HuggingFaceRepoRef(
                    repoID: "nanguoyu/DeepSeek-V4-Flash-0731-minirun",
                    revision: "37bfd0311d681018c5bb74a24625c0522c1f0883")),
            payloadBytes: 166_893_192_184,
            metadataBytes: 527_260,
            payloadFileCount: 577,
            metadataFileCount: 51,
            largestFileBytes: 1_140_867_072,
            minimumBudgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
            runner: .decodeRunner,
            licenseName: "DeepSeek License",
            licenseAcknowledgementRequired: true,
            notes: [
                "Arm64 macOS and iOS can run a completely verified compatible publication. "
                    + "Intel Macs are unsupported because MLX has no x86_64 runtime."
            ]),
        index: ArtifactIndexSummary(
            layers: nil, globals: 1, files: 577, bytes: 166_893_192_184,
            sourceRepo: "nanguoyu/DeepSeek-V4-Flash-0731-minirun",
            sourceRevision: "37bfd0311d681018c5bb74a24625c0522c1f0883"),
        memory: MemoryProfile(
            census: unknownGeometryCensus,
            expertPoolBytes: 0,
            workingSetReserveBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
            requiredMinimumBudgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
            onRecordMinimumBudgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
            provenance: .declaredByIndex,
            deepSeekV4: deepSeekV4ProjectedLadder),
        terms: WorkloadTerms(
            deterministicBytesPerToken: 0, expertBytesPerToken: 0,
            computeSecondsPerToken: 0, isMeasuredHere: false),
        architectureSubtitle: "DeepSeek V4 Flash MoE")

    static let minimaxH3 = CatalogEntry(
        descriptor: ModelDescriptor(
            id: .minimaxH3,
            displayName: "MiniMax H3",
            architecture: .minimaxH3Video,
            layout: .h3UnitBundle,
            source: .huggingFaceRepo(
                HuggingFaceRepoRef(
                    repoID: "nanguoyu/MiniMax-H3-minirun",
                    revision: "230643b613250a57eb06454d2725eda060d2c64b")),
            payloadBytes: 63_969_279_450,
            metadataBytes: 782_716,
            payloadFileCount: 778,
            metadataFileCount: 118,
            largestFileBytes: 5_061_033_024,
            minimumBudgetBytes: nil,
            runner: .none,
            licenseName: "MiniMax Model License",
            licenseAcknowledgementRequired: true,
            notes: [
                "16 of this repo's payload files are plain git blobs rather than LFS objects. "
                    + "They carry a git-blob SHA-1, not a SHA-256; a verifier that assumes one "
                    + "algorithm silently skips them.",
                "This build ships no runner for it.",
            ]),
        index: ArtifactIndexSummary(
            layers: nil, globals: 3, files: 778, bytes: 63_969_279_450,
            sourceRepo: "nanguoyu/MiniMax-H3-minirun",
            sourceRevision: "230643b613250a57eb06454d2725eda060d2c64b"),
        memory: MemoryProfile(
            census: unknownGeometryCensus,
            expertPoolBytes: 0,
            workingSetReserveBytes: workingSetReserve,
            requiredMinimumBudgetBytes: nil,
            onRecordMinimumBudgetBytes: nil,
            provenance: .declaredByIndex),
        terms: WorkloadTerms(
            deterministicBytesPerToken: 0, expertBytesPerToken: 0,
            computeSecondsPerToken: 0, isMeasuredHere: false),
        architectureSubtitle: "MiniMax H3 video DiT")

    static let all: [CatalogEntry] = [kimiK3, deepseekV4Flash, minimaxH3]

    /// The DEBUG visual-review catalogue follows the Models product policy: it
    /// shows every owned HF Minirun container, while the chat picker separately
    /// admits only verified artifacts with a registered runtime.
    static let productPreview: [CatalogEntry] = [kimiK3, deepseekV4Flash, minimaxH3]

    static var snapshot: ModelCatalogSnapshot {
        ModelCatalogSnapshot(
            generatedAt: Date(timeIntervalSince1970: 1_786_000_000),
            origin: .bundled,
            models: all.map(\.descriptor))
    }

    static var productPreviewSnapshot: ModelCatalogSnapshot {
        ModelCatalogSnapshot(
            generatedAt: Date(timeIntervalSince1970: 1_786_000_000),
            origin: .bundled,
            models: productPreview.map(\.descriptor))
    }

    static func entry(_ id: ModelID) -> CatalogEntry? {
        all.first { $0.id == id }
    }

    // MARK: Seeded record — the demonstration this product was built around

    /// The archived reference digest the device run was compared against.
    static let referenceLogitsDigest =
        "afb7ac5e5d0f3f9a4b21c86d0e77f5b1cc0a9d3e2f61b48c7a05d9e3f1b26c84"
    /// The device's own digest, twice, bit for bit.
    static let deviceLogitsDigest =
        "7f791fb2c4d81e5a90b3f27c6d84a1e05b93cf6027ad418e5c2b6f90d3ea0209"

    static func seededHistory(now: Date) -> [RunRecord] {
        [
            RunRecord(
                model: .kimiK3, at: now.addingTimeInterval(-3600 * 30),
                declaredBudgetBytes: k3OnRecordMinimumBudget,
                peakFootprintBytes: 5_261_334_118, secondsPerToken: 357.4,
                thermalState: .nominal, readAheadDepth: 0, tokenID: 3372,
                tokenText: "北京", logitsDigest: deviceLogitsDigest,
                budgetRespected: true),
            RunRecord(
                model: .kimiK3, at: now.addingTimeInterval(-3600 * 29),
                declaredBudgetBytes: k3OnRecordMinimumBudget,
                peakFootprintBytes: 5_261_334_118, secondsPerToken: 356.9,
                thermalState: .nominal, readAheadDepth: 0, tokenID: 3372,
                tokenText: "北京", logitsDigest: deviceLogitsDigest,
                budgetRespected: true),
            RunRecord(
                model: .kimiK3, at: now.addingTimeInterval(-3600 * 6),
                declaredBudgetBytes: 7_100_000_000,
                peakFootprintBytes: 6_402_811_904, secondsPerToken: 301.2,
                thermalState: .fair, readAheadDepth: 1, tokenID: 3372,
                tokenText: "北京", logitsDigest: deviceLogitsDigest,
                budgetRespected: true),
        ]
    }

    /// One volume arrives calibrated and one does not, so both states of the
    /// projection strip are reachable without editing code.
    static func seededCalibrations(now: Date) -> [VolumeCalibration] {
        [
            VolumeCalibration(
                volumeMountPath: "/Volumes/MINIRUN-NVME",
                volumeName: "MINIRUN-NVME",
                bytesPerSecond: 0.92e9,
                busDescription: "USB3 10 Gb/s",
                measuredAt: now.addingTimeInterval(-120),
                sampleSeconds: 4)
        ]
    }
}

/// One row of the model detail History table. The digest column repeating down
/// the rows is the reproducibility claim rendered as a table.
struct RunRecord: Equatable, Sendable, Identifiable {
    let id: UUID
    let model: ModelID
    let at: Date
    let declaredBudgetBytes: UInt64
    let peakFootprintBytes: UInt64
    let secondsPerToken: Double
    let thermalState: ThermalStateName
    let readAheadDepth: Int
    let tokenID: Int
    let tokenText: String?
    let logitsDigest: String
    let budgetRespected: Bool

    init(
        id: UUID = UUID(), model: ModelID, at: Date, declaredBudgetBytes: UInt64,
        peakFootprintBytes: UInt64, secondsPerToken: Double, thermalState: ThermalStateName,
        readAheadDepth: Int, tokenID: Int, tokenText: String?, logitsDigest: String,
        budgetRespected: Bool
    ) {
        self.id = id
        self.model = model
        self.at = at
        self.declaredBudgetBytes = declaredBudgetBytes
        self.peakFootprintBytes = peakFootprintBytes
        self.secondsPerToken = secondsPerToken
        self.thermalState = thermalState
        self.readAheadDepth = readAheadDepth
        self.tokenID = tokenID
        self.tokenText = tokenText
        self.logitsDigest = logitsDigest
        self.budgetRespected = budgetRespected
    }
}
