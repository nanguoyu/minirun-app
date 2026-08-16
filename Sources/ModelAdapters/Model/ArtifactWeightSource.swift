import Darwin
import Foundation
import MLX
import MLXBridge
import StorageCore

/// A ``K3WeightSource`` backed by the published, translated K3 artifact.
///
/// M9A designed this seam and wrote none of it; M9B could not reach it because
/// the artifact would not open. With ``PublishedArtifactLayout`` in front, one
/// layer's non-expert weights are a set of row bands in that layer's
/// deterministic blob, and this materialises them on demand and drops them when
/// the decode loop says the layer is finished.
///
/// ## What is resident, and when
///
/// Exactly one layer at a time. `layerWeights(i)` reads that layer's blob into
/// float32 arrays; `releaseLayer(i)` drops the reference. The peak is therefore
///
/// ```text
/// max over layers (one layer's non-expert weights, widened) + expert pool
///   + one lm_head chunk + MLX working set
/// ```
///
/// and the first term is a property of the checkpoint rather than of this class:
/// layer 0 is the largest at 2.34 GB stored (its dense MLP is 33,792 × 7,168
/// three times over), which is ~4.6 GB widened. Nothing accumulates across
/// layers, which is the whole reason ``K3WeightSource/releaseLayer(_:)`` exists.
///
/// ## Why float32
///
/// The blob stores BF16 and F32. Widening BF16 to float32 is exact — every BF16
/// value is a float32 value — so it changes no arithmetic, only where the
/// rounding of the *products* happens. It is the convention
/// ``ExpertStreamEngine`` already uses when it adopts a unit ("widen to
/// float32"), and matching it is what lets a layer computed here be compared
/// against a layer computed there.
///
/// ## Why the tensors are read whole and the experts are not
///
/// The non-expert half is read in full every token no matter what the router
/// decides, so there is nothing to stream *around*: paging it would move the
/// same bytes in smaller pieces. The routed experts are the opposite — 16 of
/// 896 — and they go through ``FlagshipRoutedExpertBackend`` and the pool.
///
/// ## Read-ahead
///
/// With ``Configuration/deterministicReadAhead`` at 1 the bytes of layer *i+1*
/// are read on a background thread while layer *i* computes, and
/// ``layerWeights(_:)`` widens whatever arrived instead of reading it first.
/// The widening — and therefore every value — is untouched;
/// ``DeterministicReadAhead`` carries the argument for why, and the budget that
/// decides which pairs may do it.
///
/// ## The pinned tier
///
/// With ``Configuration/pinnedLayers`` non-empty the layers the memory dial
/// chose are read **once** and held in their stored BF16 form for the life of
/// the run (``PinnedLayerCache``), and every later pass widens them from RAM
/// instead of the drive. The order of precedence inside ``layerWeights(_:)`` is
/// pinned, then staged, then serial, and all three end in the same widening —
/// the pinned one through ``RawTensor/materialiseKeepingBytes()``, which is that
/// same instruction sequence under an ownership rule that leaves the bytes
/// where they are. Bit-identity is therefore structural here too.
///
/// The tier is a plan, not a policy: the set comes from ``MemoryDialPlanner``
/// before the run, the run refuses a set that does not fit the residency it was
/// given, and nothing is admitted or evicted while it runs.
public final class ArtifactWeightSource: K3WeightSource {

    /// Metadata-only geometry for the exact K3 artifact a run will open.
    ///
    /// This reads translated manifests and the fixed-size container headers
    /// they name. It never materialises a tensor or reads a weight payload. A
    /// product memory plan can therefore be derived from the current verified
    /// artifact instead of from a layer table compiled into the app.
    public struct MetadataCensus: Sendable, Equatable {
        public let layerStoredBytes: [UInt64]
        public let layerResidentBytes: [UInt64]
        public let expertStrideBytes: UInt64
        public let expertsPerLayer: Int
        public let routedExpertsPerToken: Int
        public let moeLayerCount: Int

        public init(
            layerStoredBytes: [UInt64], layerResidentBytes: [UInt64],
            expertStrideBytes: UInt64, expertsPerLayer: Int,
            routedExpertsPerToken: Int, moeLayerCount: Int
        ) {
            self.layerStoredBytes = layerStoredBytes
            self.layerResidentBytes = layerResidentBytes
            self.expertStrideBytes = expertStrideBytes
            self.expertsPerLayer = expertsPerLayer
            self.routedExpertsPerToken = routedExpertsPerToken
            self.moeLayerCount = moeLayerCount
        }
    }

    /// Inspect the same layer table ``init`` later validates and consumes.
    ///
    /// The tensor plan is shared with the runtime below. That is important for
    /// F32 vectors: their resident bytes are unchanged, while BF16 tensors
    /// widen twofold. Treating every stored byte as BF16 produced a plausible
    /// but wrong census and made a fully verified, updated artifact fail only
    /// after the user pressed Send.
    public static func inspectMetadataCensus(
        translated: String, config: TinyK3Config,
        fileAccess: ModelFileAccess? = nil
    ) throws -> MetadataCensus {
        let loaded = try FlagshipLayerStreams.loadAll(
            directory: translated, fileAccess: fileAccess)
        guard loaded.count == config.numHiddenLayers else {
            throw TinyK3Error.configuration(
                "the translated artifact has \(loaded.count) layers, the config declares "
                    + "\(config.numHiddenLayers)")
        }
        for (index, stream) in loaded.enumerated() where stream.layer != index {
            throw TinyK3Error.configuration(
                "layer \(index) of the translated artifact declares itself layer \(stream.layer)")
        }

        let plans = loaded.indices.map {
            tensorPlan(layer: $0, hasExperts: loaded[$0].hasExperts, config: config)
        }
        let bytes = loaded.indices.map {
            layerBytes(loaded[$0], planned: Set(plans[$0]))
        }

        let routed = loaded.filter(\.hasExperts)
        var expertStrides = Set<UInt64>()
        var expertCounts = Set<Int>()
        for stream in routed {
            var stride: UInt64 = 0
            for container in stream.containers.values {
                let product = container.tilesPerExpert.multipliedReportingOverflow(
                    by: container.layout.tileStride)
                guard !product.overflow, product.partialValue > 0 else {
                    throw TinyK3Error.configuration(
                        "layer \(stream.layer) has an unrepresentable expert stride")
                }
                let sum = stride.addingReportingOverflow(UInt64(product.partialValue))
                guard !sum.overflow else {
                    throw TinyK3Error.configuration(
                        "layer \(stream.layer) expert bytes exceed UInt64.max")
                }
                stride = sum.partialValue
            }
            expertStrides.insert(stride)
            expertCounts.insert(stream.experts)
        }
        guard expertStrides.count <= 1, expertCounts.count <= 1 else {
            throw TinyK3Error.configuration(
                "the routed layers do not share one expert geometry")
        }

        return MetadataCensus(
            layerStoredBytes: bytes.map(\.storedBytes),
            layerResidentBytes: bytes.map(\.residentBytes),
            expertStrideBytes: expertStrides.first ?? 0,
            expertsPerLayer: expertCounts.first ?? 0,
            routedExpertsPerToken: config.numExpertsPerToken,
            moeLayerCount: routed.count)
    }

    /// Exact references for the three dense global files. A rooted product
    /// runtime supplies repository-relative identities whose opener returns
    /// already-authorized descriptors instead of reopening mutable paths.
    public struct GlobalFiles: Sendable, Equatable {
        public let embedTokens: String
        public let lmHead: String
        public let final: String

        public init(embedTokens: String, lmHead: String, final: String) {
            self.embedTokens = embedTokens
            self.lmHead = lmHead
            self.final = final
        }
    }

    /// Read-ahead configuration. Everything here is refused rather than
    /// clamped, in the style of the pager's `evalEveryK + readAhead ≤ slotCount`
    /// invariant: a window that quietly shrinks to fit turns a configuration
    /// error into a performance mystery.
    public struct Configuration: Sendable, Equatable {

        /// Layers staged ahead of the one being computed. `0` is strictly
        /// serial; `1` is a double buffer. Nothing else is defined.
        ///
        /// The owner's decision of 2026-08-11 is that read-ahead is **on** by
        /// default, and it is implemented at every surface that states a
        /// budget: `K3DecodeScenario` and the flagship first-token harness both
        /// default to depth 1 against a budget they declare. This type-level
        /// default stays `0` on purpose, and not as a hedge: depth 1 requires a
        /// declared budget (``validated()``, spec §5.3 refuse-don't-clamp), and
        /// a library type cannot invent one for a machine it knows nothing
        /// about. Defaulting to 1 here would make every bare `Configuration()`
        /// throw — the refusal would be right, and useless. Serial is what a
        /// caller that has not stated a bound gets.
        public var deterministicReadAhead: Int

        /// The budget staging must fit inside, stated before the run (spec
        /// §5.3). Required at depth 1, because staged bytes are resident bytes.
        public var budgetBytes: UInt64?

        /// Held back from ``budgetBytes`` for everything that is not these two
        /// layers — expert pool, MLX cache, working set. Zero means the budget
        /// describes the two deterministic terms alone.
        public var reserveBytes: UInt64

        /// The deterministic layers the memory dial holds in RAM for the whole
        /// run, in their **stored** form (``PinnedLayerCache``). Empty is the
        /// behaviour every record before the dial measured.
        ///
        /// This is a decision taken before the run, not a policy applied during
        /// it: ``MemoryDialPlanner`` turns one budget into this set, and the run
        /// either honours it or refuses to start.
        public var pinnedLayers: Set<Int>

        /// What ``pinnedLayers`` may occupy, stated by whoever chose the set.
        /// The layers' real stored size is checked against it when the artifact
        /// is opened, and a set that does not fit is **refused**, not trimmed:
        /// a dial that quietly pinned fewer layers than it reported would make
        /// every later number in the record a fiction.
        public var pinnedBudgetBytes: UInt64

        /// Per-layer stored bytes from the census that produced a `PinPlan`.
        /// `nil` only for the lower-level/manual configuration used by focused
        /// streaming tests. A plan-backed run checks this table against the
        /// artifact so an out-of-date census cannot make the UI and runtime
        /// report different residency for the same pinned set.
        public var expectedPinnedLayerBytes: [Int: UInt64]?

        /// The complete plan behind the derived fields above. Kept only for a
        /// plan-backed configuration so opening the artifact can re-run the
        /// projected schedule against the manifests it actually found.
        var expectedPlan: PinPlan?

        public init(
            deterministicReadAhead: Int = 0,
            budgetBytes: UInt64? = nil,
            reserveBytes: UInt64 = 0,
            pinnedLayers: Set<Int> = [],
            pinnedBudgetBytes: UInt64 = 0,
            expectedPinnedLayerBytes: [Int: UInt64]? = nil
        ) {
            self.deterministicReadAhead = deterministicReadAhead
            self.budgetBytes = budgetBytes
            self.reserveBytes = reserveBytes
            self.pinnedLayers = pinnedLayers
            self.pinnedBudgetBytes = pinnedBudgetBytes
            self.expectedPinnedLayerBytes = expectedPinnedLayerBytes
            self.expectedPlan = nil
        }

        /// The configuration one ``PinPlan`` implies.
        ///
        /// Every term comes from the plan, so the run cannot be configured with
        /// one budget and reported with another: the pair budget is the dial's
        /// own ``PinPlan/budgetBytes``, the reserve is the plan's
        /// ``PinPlan/readAheadReserveBytes`` (pool + working set + everything
        /// the plan makes permanently resident), and the pinned set and its
        /// residency are the plan's decisions verbatim.
        ///
        /// Only the plan's **deterministic layers** are pinned here. The globals
        /// bundle and the expert hot set are other engines' tiers and this
        /// source holds neither; a plan that pins them still reserves their
        /// bytes (so the pair schedule stays honest) and this source simply does
        /// not serve them. At every budget below 114,047,231,232 B the ladder
        /// pins no such unit anyway (`docs/design/memory-dial.md` §7).
        public init(plan: PinPlan, deterministicReadAhead: Int = 1) throws {
            if let reason = plan.runtimeValidationError {
                throw TinyK3Error.configuration("the memory plan is invalid: \(reason)")
            }
            guard plan.readAheadDepth == deterministicReadAhead else {
                throw TinyK3Error.configuration(
                    "the memory plan projects read-ahead depth \(plan.readAheadDepth ?? -1), "
                        + "but this run requests depth \(deterministicReadAhead)")
            }
            var pinned = Set<Int>()
            var expected: [Int: UInt64] = [:]
            var pinnedBytes: UInt64 = 0
            for decision in plan.pinnedLayers {
                guard case .deterministicLayer(let layer) = decision.unit else {
                    throw TinyK3Error.configuration(
                        "the plan's pinned-layer tier contains a non-layer unit")
                }
                guard pinned.insert(layer).inserted else {
                    throw TinyK3Error.configuration(
                        "the memory plan pins layer \(layer) more than once")
                }
                expected[layer] = decision.residentBytes
                let sum = pinnedBytes.addingReportingOverflow(decision.residentBytes)
                guard !sum.overflow else {
                    throw TinyK3Error.configuration(
                        "the memory plan's pinned-layer byte total exceeds UInt64.max")
                }
                pinnedBytes = sum.partialValue
            }
            self.init(
                deterministicReadAhead: deterministicReadAhead,
                budgetBytes: plan.budgetBytes,
                reserveBytes: plan.readAheadReserveBytes,
                pinnedLayers: pinned,
                pinnedBudgetBytes: pinnedBytes,
                expectedPinnedLayerBytes: expected)
            self.expectedPlan = plan
        }

        /// The same configuration, or a named error saying why it cannot be
        /// used. Pure, so the refusal is testable without an artifact.
        public func validated() throws -> Configuration {
            guard (0...1).contains(deterministicReadAhead) else {
                throw TinyK3Error.configuration(
                    "deterministicReadAhead \(deterministicReadAhead) is neither 0 (serial, the "
                        + "default) nor 1 (one layer staged while one is resident). The depth is "
                        + "refused rather than clamped: a deeper window is a budget term nothing "
                        + "has measured, and silently reducing it would hide that.")
            }
            if deterministicReadAhead > 0 {
                guard let budgetBytes, budgetBytes > 0 else {
                    throw TinyK3Error.configuration(
                        "deterministicReadAhead \(deterministicReadAhead) needs a declared "
                            + "budget: staged bytes are resident bytes, and spec §5.3 requires "
                            + "that bound to be stated before the run rather than discovered "
                            + "during it.")
                }
            }
            if let stray = pinnedLayers.first(where: { $0 < 0 }) {
                throw TinyK3Error.configuration(
                    "layer \(stray) cannot be pinned: the pinned set names layers of this "
                        + "artifact, and there is no layer \(stray).")
            }
            if !pinnedLayers.isEmpty && pinnedBudgetBytes == 0 {
                throw TinyK3Error.configuration(
                    "\(pinnedLayers.count) layers are pinned against a declared pin budget of 0 B. "
                        + "Pinned bytes are resident bytes for the whole run, so the residency has "
                        + "to be stated before it (spec §5.3) — refused rather than assumed.")
            }
            if let expectedPinnedLayerBytes,
                Set(expectedPinnedLayerBytes.keys) != pinnedLayers
            {
                throw TinyK3Error.configuration(
                    "the memory plan's per-layer byte table does not name exactly its pinned set")
            }
            return self
        }
    }

    public let config: TinyK3Config
    public let streams: [FlagshipLayerStreams]
    public let configuration: Configuration
    public let outputAttnResScore: MLXArray
    public let finalNormWeight: MLXArray

    private let embedding: RowAddressedBlob
    private let lmHead: RowAddressedBlob
    private let successfulReadObserver: (@Sendable (UInt64) -> Void)?
    private var blobs: [Int: DeterministicBlob] = [:]

    /// The layer being computed, in the two halves the model consumes it in.
    ///
    /// A class rather than a tuple because it is mutated in place from three
    /// call sites — the attention widening, the MLP widening, and the release
    /// of either — and a value copied out and forgotten would leak the staged
    /// bytes it owns.
    private final class ResidentLayer {
        let index: Int
        let blob: DeterministicBlob
        let isPinned: Bool
        /// Whether this pass is the one that filled the pinned tier for this
        /// layer. Constant across both halves: the pass that read the bytes is
        /// the pass that is charged for them, whichever half widens them.
        let filledPinsNow: Bool
        /// The whole layer's raw bytes, staged in one piece. Only the widening
        /// is split; a tensor stays here until its half asks for it.
        var staged: StagedLayer?
        var attention: TinyK3AttentionWeights?
        var feedForward: TinyK3FeedForwardWeights?
        var attentionWidenedBytes: UInt64 = 0
        var feedForwardWidenedBytes: UInt64 = 0

        init(index: Int, blob: DeterministicBlob, isPinned: Bool, filledPinsNow: Bool,
             staged: StagedLayer?) {
            self.index = index
            self.blob = blob
            self.isPinned = isPinned
            self.filledPinsNow = filledPinsNow
            self.staged = staged
        }
    }

    private var resident: ResidentLayer?

    /// The tensors ``layerWeights(_:)`` reads for each layer, in the order it
    /// reads them. Shared with the stager so both arms move the same bytes.
    private let plans: [[String]]
    /// Stored and widened size of each layer's planned tensors — the two terms
    /// the schedule weighs.
    let layerBytes: [DeterministicReadAheadSchedule.LayerBytes]

    private let stager: DeterministicStager?
    /// The dial's pinned tier. `nil` when nothing is pinned — the tier is
    /// absent, not empty, so the unpinned path is byte for byte the old one.
    private let pins: PinnedLayerCache?
    private var pinnedSkippedLayerCount = 0
    private var stagedLayerCount = 0
    private var skippedLayerCount = 0
    private var missedLayerCount = 0
    private var missedTensorCount = 0
    private var unusedStagedTensorCount = 0
    private var materialisedStagedBytes: UInt64 = 0
    private var discardedStagedBytes: UInt64 = 0
    private var heldStagedBytes: UInt64 = 0
    private var readAheadAccountingOverflowed = false
    private var liveWidenedBytes: UInt64 = 0
    private var peakWidenedBytes: UInt64 = 0
    private var peakAttentionWidenedBytes: UInt64 = 0
    private var peakFeedForwardWidenedBytes: UInt64 = 0
    private var peakLayerWidenedBytes: UInt64 = 0
    private var coresidentHalfCount = 0
    private var widenedAccountingOverflowed = false
    private var prefetchWaitSeconds: Double = 0
    private var serialReadSeconds: Double = 0
    private var waitSecondsByLayer: [Double]
    /// The layer the budget refused to stage, so that its consume can be
    /// reported as a skip rather than as a miss.
    private var skippedLayer: Int?
    private var isShutDown = false

    /// How the most recent ``layerWeights(_:)`` got its bytes, and how long it
    /// waited for them. The per-layer trace the scenario records.
    public private(set) var lastLayerOutcome: DeterministicReadAheadOutcome = .serial
    public private(set) var lastLayerWaitSeconds: Double = 0

    /// Bytes read from the deterministic blobs and the two global tables.
    /// Reported as an observation; it is not a latency claim.
    ///
    /// The pinned tier does not change what this counts: a pinned layer's bytes
    /// are added **once**, by the pass that read them into the cache, exactly as
    /// a staged or serial read is. Bytes a later pass takes from residency are
    /// not reads and go to ``pinnedBytesServed`` instead — every byte in one
    /// bucket, which is what makes "pass 2 read nothing for these layers" a
    /// checkable statement rather than a hope.
    public private(set) var deterministicBytesRead: UInt64 = 0
    /// Sticky companion to ``deterministicBytesRead``. A saturated value is an
    /// explicit loss of evidence, never a smaller plausible byte count.
    public private(set) var deterministicByteAccountingOverflowed = false

    /// - Parameters:
    ///   - translated: the directory ``PublishedArtifactLayout/translate(artifact:into:config:)``
    ///     wrote — flat `layerNN-manifest.json` plus links.
    ///   - globals: the artifact's `global/` directory, read directly. It is not
    ///     translated because its published manifest is a 761-byte stub: the
    ///     publisher's sanitiser dropped the unit table the same way it dropped
    ///     the layers' container descriptors. The three blobs are recovered from
    ///     their geometry instead, and every recovery is checked against the
    ///     file's length (see ``RowAddressedBlob`` and ``finalTensors``).
    ///   - configuration: read-ahead depth and its budget. The default is the
    ///     serial behaviour, so an existing call site keeps its measurements.
    public init(
        translated: String, globals: String, config: TinyK3Config,
        globalFiles: GlobalFiles? = nil,
        fileAccess: ModelFileAccess? = nil,
        configuration: Configuration = Configuration(),
        successfulReadObserver: (@Sendable (UInt64) -> Void)? = nil
    ) throws {
        let validated = try configuration.validated()
        let access = fileAccess ?? .filesystem
        let loaded = try FlagshipLayerStreams.loadAll(
            directory: translated, fileAccess: fileAccess)
        self.config = config
        self.configuration = validated
        self.streams = loaded
        self.successfulReadObserver = successfulReadObserver
        guard loaded.count == config.numHiddenLayers else {
            throw TinyK3Error.configuration(
                "the translated artifact has \(loaded.count) layers, the config declares "
                    + "\(config.numHiddenLayers)")
        }
        for (index, stream) in loaded.enumerated() where stream.layer != index {
            throw TinyK3Error.configuration(
                "layer \(index) of the translated artifact declares itself layer \(stream.layer)")
        }

        let plans = loaded.indices.map {
            Self.tensorPlan(layer: $0, hasExperts: loaded[$0].hasExperts, config: config)
        }
        self.plans = plans
        self.layerBytes = loaded.indices.map { index in
            Self.layerBytes(loaded[index], planned: Set(plans[index]))
        }
        if let expected = validated.expectedPlan?.layerTableSHA256 {
            let actual = ArtifactCensus.layerTableSHA256(
                storedBytes: self.layerBytes.map(\.storedBytes),
                residentBytes: self.layerBytes.map(\.residentBytes))
            guard actual == expected else {
                throw TinyK3Error.configuration(
                    "the memory plan's layer census \(expected) does not match the opened "
                        + "artifact's layer census \(actual); the plan's census does not "
                        + "describe this artifact")
            }
        }
        self.waitSecondsByLayer = [Double](repeating: 0, count: loaded.count)
        // No thread at all in the serial default: the feature is off, not idle.
        self.stager = validated.deterministicReadAhead > 0 ? DeterministicStager() : nil

        // The pin plan against the artifact's own byte table. The planner works
        // from a census; this is the check that the census describes *this*
        // directory, and it refuses rather than trimming — a dial that pinned
        // fewer layers than it reported would make every saving in the record a
        // claim nothing measured (spec §5.3, §12.3).
        if validated.pinnedLayers.isEmpty {
            self.pins = nil
        } else {
            var pinnedStored: UInt64 = 0
            for layer in validated.pinnedLayers.sorted() {
                guard loaded.indices.contains(layer) else {
                    throw TinyK3Error.configuration(
                        "layer \(layer) is pinned but the artifact has \(loaded.count) layers")
                }
                let artifactBytes = self.layerBytes[layer].storedBytes
                if let expected = validated.expectedPinnedLayerBytes?[layer],
                    expected != artifactBytes
                {
                    throw TinyK3Error.configuration(
                        "the memory plan says layer \(layer) occupies \(expected) B, but this "
                            + "artifact contains \(artifactBytes) B for that layer. Refused: "
                            + "the plan's census does not describe this artifact.")
                }
                let sum = pinnedStored.addingReportingOverflow(artifactBytes)
                guard !sum.overflow else {
                    throw TinyK3Error.configuration(
                        "the pinned layers' stored byte total exceeds UInt64.max")
                }
                pinnedStored = sum.partialValue
            }
            if validated.expectedPlan != nil, pinnedStored != validated.pinnedBudgetBytes {
                throw TinyK3Error.configuration(
                    "the memory plan claims \(validated.pinnedBudgetBytes) B of pinned layers, "
                        + "but this artifact's exact pinned set occupies \(pinnedStored) B")
            }
            guard pinnedStored <= validated.pinnedBudgetBytes else {
                throw TinyK3Error.configuration(
                    "the \(validated.pinnedLayers.count) pinned layers of this artifact are "
                        + "\(pinnedStored) B stored, against a declared pin budget of "
                        + "\(validated.pinnedBudgetBytes) B — short by "
                        + "\(pinnedStored - validated.pinnedBudgetBytes) B. Refused rather than "
                        + "clamped: the set is the memory dial's decision and this is the check "
                        + "that the dial's census describes this artifact.")
            }
            self.pins = PinnedLayerCache(
                layers: validated.pinnedLayers, budgetBytes: validated.pinnedBudgetBytes)
        }

        let root = URL(fileURLWithPath: globals)
        let globalFiles = globalFiles ?? GlobalFiles(
            embedTokens: root.appendingPathComponent("embed_tokens.bin").path,
            lmHead: root.appendingPathComponent("lm_head.bin").path,
            final: root.appendingPathComponent("final.bin").path)
        // `embed_tokens` and `lm_head` are dense row-major tables with no unit
        // padding at all, and that is checked rather than assumed: the file
        // length has to be exactly `vocab * hidden * 2`. A padded or unit-tiled
        // layout would not be, so the check distinguishes the two readings.
        self.embedding = try RowAddressedBlob(
            path: globalFiles.embedTokens,
            rows: config.vocabSize, cols: config.hiddenSize, dtype: .bfloat16,
            fileAccess: access,
            successfulReadObserver: successfulReadObserver)
        self.lmHead = try RowAddressedBlob(
            path: globalFiles.lmHead,
            rows: config.vocabSize, cols: config.hiddenSize, dtype: .bfloat16,
            fileAccess: access,
            successfulReadObserver: successfulReadObserver)

        let finals = try Self.finalTensors(
            path: globalFiles.final, hidden: config.hiddenSize, fileAccess: access,
            successfulReadObserver: successfulReadObserver)
        let finalPayload = UInt64(config.hiddenSize).multipliedReportingOverflow(by: 2)
        let finalTotal = finalPayload.partialValue.multipliedReportingOverflow(by: 3)
        guard !finalPayload.overflow, !finalTotal.overflow else {
            throw TinyK3Error.configuration("final tensor byte count exceeds UInt64.max")
        }
        deterministicBytesRead = finalTotal.partialValue
        // `norm.weight * proj.weight[0]`, precomputed — the form the AttnRes
        // output stage wants and the only form these two are ever used in.
        self.outputAttnResScore = finals.attnResNorm * finals.attnResProj
        self.finalNormWeight = finals.norm
        outputAttnResScore.eval()
        finalNormWeight.eval()
    }

    deinit { shutdown() }

    /// Bind a plan's remaining runtime terms to the artifact and effective
    /// configuration that will actually execute it.
    ///
    /// Pure plan validation happens before a session is accepted. This second
    /// gate needs the opened manifests: it proves the widest layer and complete
    /// schedule, and it compares the routed-expert pool chosen by the effective
    /// knobs. No payload byte has been materialised at this point.
    public func validateRuntimePlan(
        expertPoolBytes: UInt64, minimumWorkingReserveBytes: UInt64
    ) throws {
        guard let plan = configuration.expectedPlan else { return }
        guard plan.layerCount == layerBytes.count else {
            throw TinyK3Error.configuration(
                "the memory plan names \(plan.layerCount ?? -1) layers, but the opened "
                    + "artifact has \(layerBytes.count)")
        }
        guard plan.floor.widestResidentLayerBytes
            == (layerBytes.map(\.residentBytes).max() ?? 0)
        else {
            throw TinyK3Error.configuration(
                "the memory plan's widest resident layer does not match the opened artifact")
        }
        guard plan.floor.expertPoolBytes == expertPoolBytes else {
            throw TinyK3Error.configuration(
                "the memory plan reserves \(plan.floor.expertPoolBytes) B for routed experts, "
                    + "but the effective expert configuration allocates \(expertPoolBytes) B")
        }
        guard plan.floor.workingReserveBytes >= minimumWorkingReserveBytes else {
            throw TinyK3Error.configuration(
                "the memory plan reserves \(plan.floor.workingReserveBytes) B for the working "
                    + "set, below the effective runtime minimum "
                    + "\(minimumWorkingReserveBytes) B")
        }

        // K3 currently accepts deterministic-layer plans only. Re-run that
        // tier's greedy fill from the real table so rank, order, remaining
        // budget and per-token saving cannot be forged independently.
        var remaining = plan.budgetBytes - plan.workingFloorBytes
        var canonicalPins: [PinDecision] = []
        for (layer, bytes) in layerBytes.enumerated() where bytes.storedBytes <= remaining {
            remaining -= bytes.storedBytes
            canonicalPins.append(
                PinDecision(
                    unit: .deterministicLayer(layer), rank: layer,
                    residentBytes: bytes.storedBytes,
                    bytesSavedPerToken: bytes.storedBytes,
                    savedPerResidentByte: bytes.storedBytes == 0 ? 0 : 1,
                    budgetRemainingAfterBytes: remaining))
        }
        guard canonicalPins == plan.pinnedLayers else {
            throw TinyK3Error.configuration(
                "the memory plan's pinned-layer decisions are not the canonical fill for "
                    + "this artifact and budget")
        }

        var stagedCount = 0
        var inlineCount = 0
        var refused: [DeterministicReadAheadSchedule.Decision] = []
        for layer in layerBytes.indices where !configuration.pinnedLayers.contains(layer) {
            guard configuration.deterministicReadAhead == 1, layer > 0 else {
                inlineCount += 1
                continue
            }
            let decision = DeterministicReadAheadSchedule.decide(
                residentLayer: layer - 1, resident: layerBytes[layer - 1],
                stagedLayer: layer, next: layerBytes[layer],
                budgetBytes: plan.budgetBytes, reserveBytes: plan.readAheadReserveBytes)
            if decision.isAllowed {
                stagedCount += 1
            } else {
                inlineCount += 1
                refused.append(decision)
            }
        }
        guard stagedCount == plan.projectedStagedLayerCount,
            inlineCount == plan.projectedInlineLayerCount,
            refused.count == plan.projectedRefusedPairCount,
            refused.count == plan.refusedPairs.count
        else {
            throw TinyK3Error.configuration(
                "the memory plan's staged, inline, or refused layer counts do not match "
                    + "the opened artifact")
        }
        for (actual, projected) in zip(refused, plan.refusedPairs) {
            guard actual.residentLayer == projected.residentLayer,
                actual.stagedLayer == projected.stagedLayer,
                actual.residentBytes == projected.residentBytes,
                actual.stagedBytes == projected.stagedBytes,
                actual.reserveBytes == projected.reserveBytes,
                actual.budgetBytes == projected.budgetBytes,
                actual.requiredBytes == projected.requiredBytes,
                actual.isAllowed == projected.isAllowed,
                actual.headroomBytes == projected.headroomBytes
            else {
                throw TinyK3Error.configuration(
                    "a projected read-ahead refusal does not match the opened artifact")
            }
        }
    }

    /// The largest deterministic layer after widening, from the opened
    /// manifests. Used by a runner that must reject an unplanned request below
    /// the physical working floor without guessing from catalog metadata.
    public var widestResidentLayerBytes: UInt64 {
        layerBytes.map(\.residentBytes).max() ?? 0
    }

    /// Stop the background reader, free anything it staged, and close the
    /// descriptors. Idempotent, and called from `deinit`: the stager's thread
    /// holds a strong reference to itself while it runs, so joining it is not
    /// optional (the rule ``TileReader`` states for the same reason).
    public func shutdown() {
        guard !isShutDown else { return }
        isShutDown = true
        // `releaseLayer` is normally called by the model's per-layer defer, but
        // terminal teardown is the ownership boundary. Drop the last widened
        // arrays here too — and the staged bytes the record owns, so the
        // read-ahead accounting still balances after an abandoned layer — so
        // observing `.finished` never means the previous layer is still
        // retained merely because the engine owns this source.
        if let record = resident { finish(record) }
        resident = nil
        stager?.shutdown()
        // The pinned tier is the largest allocation this object holds after the
        // resident layer, and it is held on purpose for the whole run — so the
        // end of the run is where it goes, on the same idempotent path as the
        // stager's thread.
        pins?.release()
        for blob in blobs.values { blob.close() }
        blobs.removeAll()
        embedding.close()
        lmHead.close()
    }

    /// `final.bin`, whose three tensors the published manifest no longer
    /// describes.
    ///
    /// The order is not guessed. `build_full_artifact.py` sorts a global's
    /// tensors by their byte offset in the source shard before laying them out,
    /// and in `model-00094-of-000096.safetensors` that order is `norm`,
    /// `output_attn_res_norm`, `output_attn_res_proj` — each `[7168]` BF16,
    /// 14,336 bytes padded to one 16,384-byte unit. Three units, 49,152 bytes,
    /// which is exactly the file's length.
    static func finalTensors(
        path: String, hidden: Int, fileAccess: ModelFileAccess = .filesystem,
        successfulReadObserver: (@Sendable (UInt64) -> Void)? = nil
    )
        throws -> (norm: MLXArray, attnResNorm: MLXArray, attnResProj: MLXArray)
    {
        let unit = 16384
        let payload = hidden * 2
        guard payload <= unit else {
            throw TinyK3Error.configuration(
                "a \(hidden)-wide BF16 vector does not fit one \(unit)-byte unit")
        }
        let descriptor = try fileAccess.open(path)
        defer { close(descriptor) }
        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            throw StorageCoreError.posix(operation: "fstat", path: path, code: errno)
        }
        guard Int(status.st_size) == 3 * unit else {
            throw TinyK3Error.configuration(
                "final.bin is \(status.st_size) bytes; three \(unit)-byte units is \(3 * unit). "
                    + "The three-tensor reading of this file does not hold.")
        }
        func vector(_ index: Int) throws -> MLXArray {
            let buffer = UnsafeMutableRawPointer.allocate(byteCount: payload, alignment: 16)
            var adopted = false
            defer { if !adopted { buffer.deallocate() } }
            let read = pread(descriptor, buffer, payload, off_t(index * unit))
            guard read == payload else {
                throw StorageCoreError.shortTransfer(
                    operation: "pread", path: path, offset: UInt64(index * unit),
                    expected: payload, actual: max(read, 0))
            }
            successfulReadObserver?(UInt64(payload))
            let raw = MLXArray(
                rawPointer: buffer, [hidden], dtype: .bfloat16,
                finalizer: { buffer.deallocate() })
            adopted = true
            let widened = raw.asType(.float32)
            widened.eval()
            return widened
        }
        return (try vector(0), try vector(1), try vector(2))
    }

    // MARK: - K3WeightSource

    public func embeddingRows(_ tokenIds: [Int]) throws -> MLXArray {
        let elements = tokenIds.count.multipliedReportingOverflow(by: config.hiddenSize)
        let bytes = elements.partialValue.multipliedReportingOverflow(by: 2)
        guard !elements.overflow, !bytes.overflow else {
            throw TinyK3Error.configuration("embedding row byte count exceeds Int.max")
        }
        let result = try embedding.rows(tokenIds)
        addDeterministicBytes(UInt64(bytes.partialValue))
        return result
    }

    public func lmHeadRows(from first: Int, count: Int) throws -> MLXArray {
        let elements = count.multipliedReportingOverflow(by: config.hiddenSize)
        let bytes = elements.partialValue.multipliedReportingOverflow(by: 2)
        guard count >= 0, !elements.overflow, !bytes.overflow else {
            throw TinyK3Error.configuration("lm-head row byte count exceeds Int.max")
        }
        let result = try lmHead.rowRange(from: first, count: count)
        addDeterministicBytes(UInt64(bytes.partialValue))
        return result
    }

    public func releaseLayer(_ index: Int) {
        guard let record = resident, record.index == index else { return }
        finish(record)
        resident = nil
        // The blob's descriptor stays open — it is one file descriptor, not a
        // thread pool, and reopening it 93 times a token buys nothing.
    }

    /// Drop the attention half's widened arrays, keeping the rest of the layer.
    ///
    /// The caller has evaluated everything computed from them, so this is the
    /// moment those bytes stop being resident — before the MLP half is widened
    /// rather than after it, which is the whole saving. Nothing is read, nothing
    /// is re-read, and the layer's raw staged bytes for the *other* half are
    /// untouched.
    public func releaseLayerAttention(_ index: Int) {
        guard let record = resident, record.index == index, record.attention != nil else {
            return
        }
        record.attention = nil
        liveWidenedBytes = subtractingWidenedBytes(
            record.attentionWidenedBytes, from: liveWidenedBytes)
    }

    /// The whole layer at once — both halves, resident together.
    ///
    /// This is what the decode loop used to ask for and what a caller that
    /// wants to *inspect* a layer still asks for. It is deliberately not what
    /// ``K3Model`` calls: computing through it would restore the peak this
    /// class was split to avoid, and ``subLayerResidencyMetrics`` says so —
    /// every call through here counts as a co-resident pair.
    public func layerWeights(_ index: Int) throws -> TinyK3LayerWeights {
        let attention = try layerAttentionWeights(index)
        let feedForward = try layerFeedForwardWeights(index)
        return TinyK3LayerWeights(attention: attention, feedForward: feedForward)
    }

    public func layerAttentionWeights(_ index: Int) throws -> TinyK3AttentionWeights {
        if let record = resident, record.index == index {
            if let attention = record.attention { return attention }
            // Widening it again would read bytes this pass has already been
            // charged for and would undo the release that had just been made.
            // Refused and named, in the style of every other configuration
            // error here: a silent re-read is a performance mystery.
            throw TinyK3Error.configuration(
                "layer \(index)'s attention weights were released and then asked for again; "
                    + "a layer is consumed once, attention half first")
        }
        guard streams.indices.contains(index) else {
            throw TinyK3Error.configuration(
                "layer \(index) requested but the artifact has \(streams.count)")
        }
        // Drop the previous layer before reading the next, never after: holding
        // two at once would double the term the budget is stated in.
        if let previous = resident {
            finish(previous)
            resident = nil
        }

        // Whatever the background read staged for this layer. A read that
        // failed there is rethrown here — as the error the serial path would
        // have thrown, from the call that would have thrown it.
        var staged: StagedLayer?
        var outcome = DeterministicReadAheadOutcome.serial
        let isPinned = pins?.isPinned(index) ?? false
        if let stager {
            let claim = try stager.claim(layer: index)
            prefetchWaitSeconds += claim.waitSeconds
            lastLayerWaitSeconds = claim.waitSeconds
            if waitSecondsByLayer.indices.contains(index) {
                waitSecondsByLayer[index] += claim.waitSeconds
            }
            if isPinned {
                // Nothing is ever staged for a pinned layer, so a set arriving
                // here means the schedule and the pin plan disagreed. It is
                // adopted anyway — the record below frees it and the counters
                // record it — rather than leaked past an accounting that would
                // then not balance.
                staged = claim.staged
                heldStagedBytes = addingReadAheadBytes(
                    claim.staged?.remainingBytes ?? 0, to: heldStagedBytes)
                outcome = .pinned
            } else if let ready = claim.staged {
                staged = ready
                heldStagedBytes = addingReadAheadBytes(
                    ready.remainingBytes, to: heldStagedBytes)
                outcome = .staged
                stagedLayerCount += 1
            } else if skippedLayer == index {
                outcome = .skipped
            } else {
                outcome = .missed
                missedLayerCount += 1
            }
        } else {
            lastLayerWaitSeconds = 0
            if isPinned { outcome = .pinned }
        }
        lastLayerOutcome = outcome
        // Until the record below owns it, a throw frees it here. Afterwards the
        // record does — the MLP half's raw bytes must outlive this call, which
        // is why the whole-layer `defer` this replaced could not stay.
        var adopted = false
        defer { if !adopted { discardStaged(staged) } }

        let blob = try self.blob(for: index)

        // The pinned tier's fill: one layer's stored bytes — both halves, since
        // a pinned layer is pinned whole — read by the same `rawTensor` loop the
        // serial path uses, on first touch. Assembled into a local first so a
        // read that throws part way leaves no half-filled layer behind.
        var filledPinsNow = false
        if let pins, isPinned, !pins.isLoaded(index) {
            let started = MonotonicClock.now()
            var tensors: [String: RawTensor] = [:]
            for name in plans[index] { tensors[name] = try blob.rawTensor(name) }
            // Charged as a serial read because that is what it is: on the
            // filling pass a pinned layer's bytes are in the critical path,
            // since the stager was told there was nothing to stage.
            serialReadSeconds += MonotonicClock.seconds(since: started)
            pins.fill(layer: index, tensors: tensors)
            filledPinsNow = true
        } else if let pins, isPinned {
            pins.noteLayerServed()
        }

        let record = ResidentLayer(
            index: index, blob: blob, isPinned: isPinned, filledPinsNow: filledPinsNow,
            staged: staged)
        resident = record
        adopted = true

        do {
            let half = try assembleAttention(record)
            record.attention = half
            record.attentionWidenedBytes = Self.widenedBytes(half.residentArrays)
            noteHalfWidened(record)
            // Both halves of the *read*, in the one accounting: what the
            // background thread moved for this layer, plus whatever this thread
            // had to read itself. Charged here because the layer is staged and
            // read whole; the MLP widening below adds only what it reads itself.
            if record.staged?.byteAccountingOverflowed == true || blob.byteAccountingOverflowed {
                deterministicByteAccountingOverflowed = true
                deterministicBytesRead = .max
            }
            addDeterministicBytes(record.staged?.bytesRead ?? 0)
            addDeterministicBytes(blob.bytesRead)
            // The same point in the pass the whole-layer version issued it from:
            // this layer's bytes are in hand and nothing of it has been computed
            // yet. The window it opens is *smaller* than the one the schedule
            // budgets for — attention widened plus the MLP half still stored is
            // below the whole layer widened — so the pair arithmetic stays a
            // conservative bound and is deliberately left alone here.
            scheduleReadAhead(after: index)
            return half
        } catch {
            releaseLayer(index)
            throw error
        }
    }

    public func layerFeedForwardWeights(_ index: Int) throws -> TinyK3FeedForwardWeights {
        guard let record = resident, record.index == index else {
            // The attention half comes first — it is what claims the staged
            // bytes and fills the pinned tier. A caller that wants only this
            // half gets it, by way of the half it skipped.
            _ = try layerAttentionWeights(index)
            return try layerFeedForwardWeights(index)
        }
        if let feedForward = record.feedForward { return feedForward }

        // The attention half's reads are already charged; count only this
        // half's from here.
        record.blob.resetAccounting()
        do {
            let half = try assembleFeedForward(record)
            record.feedForward = half
            record.feedForwardWidenedBytes = Self.widenedBytes(half.residentArrays)
            noteHalfWidened(record)
            if record.blob.byteAccountingOverflowed {
                deterministicByteAccountingOverflowed = true
                deterministicBytesRead = .max
            }
            addDeterministicBytes(record.blob.bytesRead)
            return half
        } catch {
            releaseLayer(index)
            throw error
        }
    }

    // MARK: - Widening one half

    private func assembleAttention(_ record: ResidentLayer) throws -> TinyK3AttentionWeights {
        let index = record.index
        let prefix = "language_model.model.layers.\(index)"
        let attention = "\(prefix).self_attn"
        func matrix(_ name: String) throws -> MLXArray { try widened(name, of: record) }
        func vector(_ name: String) throws -> MLXArray {
            try widened(name, of: record).reshaped([-1])
        }

        var kda: K3KDA.Weights?
        var mla: K3MLA.Weights?
        if config.isLinearAttention(layer: index) {
            kda = K3KDA.Weights(
                qProj: try matrix("\(attention).q_proj.weight"),
                kProj: try matrix("\(attention).k_proj.weight"),
                vProj: try matrix("\(attention).v_proj.weight"),
                // `[channels, 1, W]` on disk; the singleton group axis goes
                // because the convolution is depthwise.
                qConv: try matrix("\(attention).q_conv1d.weight")
                    .reshaped([-1, config.kdaShortConvKernelSize]),
                kConv: try matrix("\(attention).k_conv1d.weight")
                    .reshaped([-1, config.kdaShortConvKernelSize]),
                vConv: try matrix("\(attention).v_conv1d.weight")
                    .reshaped([-1, config.kdaShortConvKernelSize]),
                fAProj: try matrix("\(attention).f_a_proj.weight"),
                fBProj: try matrix("\(attention).f_b_proj.weight"),
                bProj: try matrix("\(attention).b_proj.weight"),
                gProj: try matrix("\(attention).g_proj.weight"),
                // Stored `[128]` and meant `[96]`. `resolveALog` slices it and
                // refuses any checkpoint where the tail is not provably padding
                // (M9B: the alternative reading is 0.7617 relative away).
                aLog: try K3KDA.resolveALog(
                    try vector("\(attention).A_log"),
                    heads: config.kdaNumHeads, headDim: config.kdaHeadDim),
                dtBias: try vector("\(attention).dt_bias"),
                oNormWeight: try vector("\(attention).o_norm.weight"),
                oProj: try matrix("\(attention).o_proj.weight"))
        } else {
            mla = K3MLA.Weights(
                qAProj: try matrix("\(attention).q_a_proj.weight"),
                qALayerNorm: try vector("\(attention).q_a_layernorm.weight"),
                qBProj: try matrix("\(attention).q_b_proj.weight"),
                kvAProjWithMQA: try matrix("\(attention).kv_a_proj_with_mqa.weight"),
                kvALayerNorm: try vector("\(attention).kv_a_layernorm.weight"),
                kvBProj: try matrix("\(attention).kv_b_proj.weight"),
                gProj: try matrix("\(attention).g_proj.weight"),
                oProj: try matrix("\(attention).o_proj.weight"))
        }

        let selfNorm = try vector("\(prefix).self_attention_res_norm.weight")
        let selfProj = try matrix("\(prefix).self_attention_res_proj.weight").reshaped([-1])
        let half = TinyK3AttentionWeights(
            inputLayerNorm: try vector("\(prefix).input_layernorm.weight"),
            selfAttentionResScore: selfNorm * selfProj,
            kda: kda, mla: mla)
        // Materialise here rather than at first use: the blob's buffers are
        // freed as soon as this returns, and a lazy graph node still pointing
        // into one would be reading freed memory.
        half.selfAttentionResScore.eval()
        return half
    }

    private func assembleFeedForward(_ record: ResidentLayer) throws -> TinyK3FeedForwardWeights {
        let index = record.index
        let stream = streams[index]
        let prefix = "language_model.model.layers.\(index)"
        func matrix(_ name: String) throws -> MLXArray { try widened(name, of: record) }
        func vector(_ name: String) throws -> MLXArray {
            try widened(name, of: record).reshaped([-1])
        }

        var dense: K3MLP.DenseWeights?
        var moe: K3MoE.Weights?
        if config.isDense(layer: index) {
            guard !stream.hasExperts else {
                throw TinyK3Error.configuration(
                    "layer \(index) is dense by config but ships expert containers")
            }
            dense = K3MLP.DenseWeights(
                gateProj: try matrix("\(prefix).mlp.gate_proj.weight"),
                upProj: try matrix("\(prefix).mlp.up_proj.weight"),
                downProj: try matrix("\(prefix).mlp.down_proj.weight"))
        } else {
            guard stream.hasExperts else {
                throw TinyK3Error.configuration(
                    "layer \(index) routes by config but ships no expert containers")
            }
            let block = "\(prefix).block_sparse_moe"
            // Flagship stores these dense BF16 where the tiny checkpoint stored
            // them MXFP4. `K3Projection` already expresses both, so this is a
            // different case of the same type rather than a different path.
            moe = K3MoE.Weights(
                gateWeight: try matrix("\(block).gate.weight"),
                correctionBias: try vector("\(block).gate.e_score_correction_bias"),
                routedDownProj: .dense(try matrix("\(block).routed_expert_down_proj.weight")),
                routedUpProj: .dense(try matrix("\(block).routed_expert_up_proj.weight")),
                routedNorm: try vector("\(block).routed_expert_norm.weight"),
                shared: K3MLP.QuantizedWeights(
                    gateProj: try matrix("\(block).shared_experts.gate_proj.weight"),
                    upProj: try matrix("\(block).shared_experts.up_proj.weight"),
                    downProj: try matrix("\(block).shared_experts.down_proj.weight")))
        }

        let mlpNorm = try vector("\(prefix).mlp_res_norm.weight")
        let mlpProj = try matrix("\(prefix).mlp_res_proj.weight").reshaped([-1])
        let half = TinyK3FeedForwardWeights(
            postAttentionLayerNorm: try vector("\(prefix).post_attention_layernorm.weight"),
            mlpResScore: mlpNorm * mlpProj,
            dense: dense, moe: moe)
        half.mlpResScore.eval()
        return half
    }

    /// Pinned bytes first, then staged bytes, then the serial read. All three
    /// hand the *same* bytes to the *same* widening — the pinned tier through
    /// ``RawTensor/materialiseKeepingBytes()``, which is that widening under the
    /// other ownership rule and nothing else.
    private func widened(_ name: String, of record: ResidentLayer) throws -> MLXArray {
        if let pins, record.isPinned,
            let widened = try pins.widened(
                layer: record.index, name: name, justFilled: record.filledPinsNow)
        {
            return widened
        }
        if let staged = record.staged, let hit = staged.take(name) {
            heldStagedBytes = subtractingReadAheadBytes(
                UInt64(hit.byteCount), from: heldStagedBytes)
            do {
                let widened = try hit.materialise()
                materialisedStagedBytes = addingReadAheadBytes(
                    UInt64(hit.byteCount), to: materialisedStagedBytes)
                return widened
            } catch {
                discardedStagedBytes = addingReadAheadBytes(
                    UInt64(hit.byteCount), to: discardedStagedBytes)
                throw error
            }
        }
        // A name the plan did not cover costs an overlap, never a value: the
        // read below is the one the serial arm makes.
        if record.staged != nil { missedTensorCount += 1 }
        let started = MonotonicClock.now()
        let raw = try record.blob.rawTensor(name)
        serialReadSeconds += MonotonicClock.seconds(since: started)
        return try raw.materialise()
    }

    // MARK: - The resident record's lifetime

    /// Free everything one resident layer still holds, and account for it.
    private func finish(_ record: ResidentLayer) {
        discardStaged(record.staged)
        record.staged = nil
        if record.attention != nil {
            record.attention = nil
            liveWidenedBytes = subtractingWidenedBytes(
                record.attentionWidenedBytes, from: liveWidenedBytes)
        }
        if record.feedForward != nil {
            record.feedForward = nil
            liveWidenedBytes = subtractingWidenedBytes(
                record.feedForwardWidenedBytes, from: liveWidenedBytes)
        }
    }

    private func discardStaged(_ staged: StagedLayer?) {
        guard let staged else { return }
        unusedStagedTensorCount += staged.remainingCount
        let left = staged.discardAll()
        heldStagedBytes = subtractingReadAheadBytes(left, from: heldStagedBytes)
        discardedStagedBytes = addingReadAheadBytes(left, to: discardedStagedBytes)
    }

    /// One half just became resident. Every peak this class reports is updated
    /// here and nowhere else.
    private func noteHalfWidened(_ record: ResidentLayer) {
        var live = UInt64Accounting.SaturatingSum(
            value: 0, didOverflow: widenedAccountingOverflowed)
        if record.attention != nil { live.add(record.attentionWidenedBytes) }
        if record.feedForward != nil { live.add(record.feedForwardWidenedBytes) }
        if live.didOverflow { widenedAccountingOverflowed = true }
        liveWidenedBytes = live.value
        peakWidenedBytes = max(peakWidenedBytes, liveWidenedBytes)
        peakAttentionWidenedBytes = max(peakAttentionWidenedBytes, record.attentionWidenedBytes)
        peakFeedForwardWidenedBytes = max(
            peakFeedForwardWidenedBytes, record.feedForwardWidenedBytes)
        // What this layer *would* occupy with both halves held at once — the
        // peak this class had before the split, measured on the same run rather
        // than quoted from an older one.
        var whole = UInt64Accounting.SaturatingSum(
            value: record.attentionWidenedBytes, didOverflow: widenedAccountingOverflowed)
        whole.add(record.feedForwardWidenedBytes)
        if whole.didOverflow { widenedAccountingOverflowed = true }
        peakLayerWidenedBytes = max(peakLayerWidenedBytes, whole.value)
        if record.attention != nil && record.feedForward != nil { coresidentHalfCount += 1 }
    }

    private static func widenedBytes(_ arrays: [MLXArray]) -> UInt64 {
        UInt64Accounting.saturatingSum(arrays.map { UInt64($0.nbytes) }).value
    }

    private func subtractingWidenedBytes(_ decrement: UInt64, from value: UInt64) -> UInt64 {
        guard decrement <= value, !widenedAccountingOverflowed else {
            widenedAccountingOverflowed = true
            return 0
        }
        return value - decrement
    }

    /// What the two-phase residency actually held, over this source's life.
    public var subLayerResidencyMetrics: SubLayerResidencyMetrics {
        var metrics = SubLayerResidencyMetrics()
        metrics.liveWidenedBytes = liveWidenedBytes
        metrics.peakWidenedBytes = peakWidenedBytes
        metrics.peakAttentionWidenedBytes = peakAttentionWidenedBytes
        metrics.peakFeedForwardWidenedBytes = peakFeedForwardWidenedBytes
        metrics.peakLayerWidenedBytes = peakLayerWidenedBytes
        metrics.coresidentHalfCount = coresidentHalfCount
        metrics.accountingOverflowed = widenedAccountingOverflowed
        return metrics
    }

    private func blob(for index: Int) throws -> DeterministicBlob {
        if let existing = blobs[index] {
            existing.resetAccounting()
            return existing
        }
        let blob = try DeterministicBlob(
            stream: streams[index],
            successfulReadObserver: successfulReadObserver)
        blobs[index] = blob
        return blob
    }

    // MARK: - Read-ahead

    /// Hand layer `index + 1` to the background reader, if the budget says the
    /// pair fits.
    ///
    /// Called once layer `index` is resident and about to be computed with,
    /// which is exactly the window the staging has to fit inside. A pair that
    /// does not fit is *skipped*, not clamped and not an error: at flagship
    /// shape the layer 0 pair provably does not fit the phone's budget, and a
    /// run that refused to start over it would be refusing to do the thing that
    /// works.
    private func scheduleReadAhead(after index: Int) {
        guard let stager else { return }
        skippedLayer = nil
        let next = index + 1
        guard streams.indices.contains(next), layerBytes.indices.contains(index) else { return }
        // A pinned layer has nothing to stage: its bytes are resident already,
        // or they will be read once by the pass that first touches it. Counted
        // separately from a budget skip, because the two are opposite news — a
        // budget skip says the plan did not fit, a pin skip says it worked.
        if pins?.isPinned(next) ?? false {
            pinnedSkippedLayerCount += 1
            return
        }
        let decision = DeterministicReadAheadSchedule.decide(
            residentLayer: index, resident: layerBytes[index],
            stagedLayer: next, next: layerBytes[next],
            budgetBytes: configuration.budgetBytes ?? 0,
            reserveBytes: configuration.reserveBytes)
        guard decision.isAllowed else {
            skippedLayerCount += 1
            skippedLayer = next
            return
        }
        do {
            stager.stage(layer: next, blob: try blob(for: next), tensors: plans[next])
        } catch {
            // Opening the next layer's blob failed. Nothing is masked by
            // declining to stage it: the serial path opens the same file at
            // consume time and throws there, which is where it throws today.
            skippedLayerCount += 1
            skippedLayer = next
        }
    }

    /// What the read-ahead did. All zeros at depth 0.
    public var readAheadMetrics: DeterministicReadAheadMetrics {
        var metrics = DeterministicReadAheadMetrics()
        metrics.depth = configuration.deterministicReadAhead
        metrics.declaredBudgetBytes = configuration.budgetBytes ?? 0
        metrics.reserveBytes = configuration.reserveBytes
        metrics.stagedLayerCount = stagedLayerCount
        metrics.skippedLayerCount = skippedLayerCount
        metrics.pinnedSkippedLayerCount = pinnedSkippedLayerCount
        metrics.missedLayerCount = missedLayerCount
        metrics.missedTensorCount = missedTensorCount
        metrics.unusedStagedTensorCount = unusedStagedTensorCount
        metrics.stagedBytes = stager?.stagedBytes ?? 0
        metrics.peakStagedBytes = stager?.peakStagedBytes ?? 0
        metrics.materialisedStagedBytes = materialisedStagedBytes
        var discarded = UInt64Accounting.SaturatingSum(
            value: discardedStagedBytes,
            didOverflow: readAheadAccountingOverflowed || (stager?.accountingOverflowed ?? false))
        discarded.add(stager?.discardedBytes ?? 0)
        var outstanding = UInt64Accounting.SaturatingSum(
            value: heldStagedBytes, didOverflow: readAheadAccountingOverflowed)
        outstanding.add(stager?.outstandingBytes ?? 0)
        metrics.discardedStagedBytes = discarded.value
        metrics.outstandingStagedBytes = outstanding.value
        metrics.accountingOverflowed = discarded.didOverflow || outstanding.didOverflow
        metrics.prefetchWaitSeconds = prefetchWaitSeconds
        metrics.stageReadSeconds = stager?.stageReadSeconds ?? 0
        metrics.serialReadSeconds = serialReadSeconds
        metrics.waitSecondsByLayer = waitSecondsByLayer
        metrics.errorCount = stager?.errorCount ?? 0
        metrics.firstError = stager?.firstError
        return metrics
    }

    /// What the memory dial's pinned tier held, and what it saved. All zeros —
    /// and `pinnedLayerCount` zero — when nothing is pinned, which is the arm
    /// every record before the dial measured.
    ///
    /// Note which counter a run should read: on the pass that *fills* the tier
    /// the bytes are read, so they land in ``deterministicBytesRead`` and in
    /// ``PinnedTierMetrics/bytesChargedAsRead``. Only the passes after it show
    /// up as ``PinnedTierMetrics/bytesServed``. A one-token run therefore
    /// reports a full fill and zero service, and that is the tier working, not
    /// failing.
    public var pinnedTierMetrics: PinnedTierMetrics { pins?.metrics ?? PinnedTierMetrics() }

    /// Bytes this run did not read because the dial pinned them. Zero on the
    /// filling pass by construction; the counter a multi-token run is judged on.
    public var pinnedBytesServed: UInt64 { pins?.bytesServed ?? 0 }

    /// Every decision the current budget would make over this artifact, without
    /// running it. The schedule is a property of the checkpoint and the budget,
    /// so it can be read — and recorded — before a token is decoded.
    public func readAheadSchedule() -> [DeterministicReadAheadSchedule.Decision] {
        DeterministicReadAheadSchedule.plan(
            layerBytes, budgetBytes: configuration.budgetBytes ?? 0,
            reserveBytes: configuration.reserveBytes)
    }

    private func addingReadAheadBytes(_ increment: UInt64, to value: UInt64) -> UInt64 {
        var sum = UInt64Accounting.SaturatingSum(
            value: value, didOverflow: readAheadAccountingOverflowed)
        sum.add(increment)
        if sum.didOverflow { readAheadAccountingOverflowed = true }
        return sum.value
    }

    private func subtractingReadAheadBytes(_ decrement: UInt64, from value: UInt64) -> UInt64 {
        guard decrement <= value, !readAheadAccountingOverflowed else {
            readAheadAccountingOverflowed = true
            return 0
        }
        return value - decrement
    }

    private func addDeterministicBytes(_ increment: UInt64) {
        var sum = UInt64Accounting.SaturatingSum(
            value: deterministicBytesRead,
            didOverflow: deterministicByteAccountingOverflowed)
        sum.add(increment)
        deterministicBytesRead = sum.value
        deterministicByteAccountingOverflowed = sum.didOverflow
    }

    // MARK: - The tensor plan

    /// The tensors one layer's **attention** half reads, in the order it reads
    /// them.
    ///
    /// This list and the assembly are two statements of the same fact, so they
    /// can disagree. They are not allowed to disagree *silently*: a name the
    /// assembly asks for and this list missed is counted as a tensor miss and
    /// read serially, and a name this list carries that the assembly never asks
    /// for is counted as an unused staged tensor. Both counters are in the run's
    /// JSON, and both are zero for every archetype the fixture covers.
    static func attentionTensorPlan(layer index: Int, config: TinyK3Config) -> [String] {
        let prefix = "language_model.model.layers.\(index)"
        let attention = "\(prefix).self_attn"
        var names: [String] = []
        if config.isLinearAttention(layer: index) {
            names += [
                "\(attention).q_proj.weight", "\(attention).k_proj.weight",
                "\(attention).v_proj.weight", "\(attention).q_conv1d.weight",
                "\(attention).k_conv1d.weight", "\(attention).v_conv1d.weight",
                "\(attention).f_a_proj.weight", "\(attention).f_b_proj.weight",
                "\(attention).b_proj.weight", "\(attention).g_proj.weight",
                "\(attention).A_log", "\(attention).dt_bias",
                "\(attention).o_norm.weight", "\(attention).o_proj.weight",
            ]
        } else {
            names += [
                "\(attention).q_a_proj.weight", "\(attention).q_a_layernorm.weight",
                "\(attention).q_b_proj.weight", "\(attention).kv_a_proj_with_mqa.weight",
                "\(attention).kv_a_layernorm.weight", "\(attention).kv_b_proj.weight",
                "\(attention).g_proj.weight", "\(attention).o_proj.weight",
            ]
        }
        // The AttnRes stage whose score these two make runs *before* attention,
        // and the input norm feeds attention itself: both are dead once
        // attention has produced its output.
        names += [
            "\(prefix).self_attention_res_norm.weight",
            "\(prefix).self_attention_res_proj.weight",
            "\(prefix).input_layernorm.weight",
        ]
        return names
    }

    /// The tensors one layer's **MLP / MoE** half reads, in the order it reads
    /// them.
    ///
    /// `mlp_res_norm` and `mlp_res_proj` are here rather than with the attention
    /// half although the AttnRes stage that consumes their product runs before
    /// the MLP: that stage is still *after* attention, which is the boundary the
    /// split is about. No tensor in a layer is read by both halves; one that
    /// were would belong here, because this half is released last.
    static func feedForwardTensorPlan(
        layer index: Int, hasExperts: Bool, config: TinyK3Config
    ) -> [String] {
        let prefix = "language_model.model.layers.\(index)"
        var names: [String] = []
        // Container presence, not the config's dense prefix: spec §5.10 says
        // the files decide, and the assembly refuses the two readings when they
        // disagree rather than choosing one.
        if config.isDense(layer: index) && !hasExperts {
            names += [
                "\(prefix).mlp.gate_proj.weight", "\(prefix).mlp.up_proj.weight",
                "\(prefix).mlp.down_proj.weight",
            ]
        } else if !config.isDense(layer: index) && hasExperts {
            let block = "\(prefix).block_sparse_moe"
            names += [
                "\(block).gate.weight", "\(block).gate.e_score_correction_bias",
                "\(block).routed_expert_down_proj.weight",
                "\(block).routed_expert_up_proj.weight",
                "\(block).routed_expert_norm.weight",
                "\(block).shared_experts.gate_proj.weight",
                "\(block).shared_experts.up_proj.weight",
                "\(block).shared_experts.down_proj.weight",
            ]
        }
        names += [
            "\(prefix).mlp_res_norm.weight", "\(prefix).mlp_res_proj.weight",
            "\(prefix).post_attention_layernorm.weight",
        ]
        return names
    }

    /// Both halves, in the order the layer is read.
    ///
    /// The whole layer is what the stager stages and what the byte census
    /// weighs: only the *widening* is split. The raw BF16 is the small half, and
    /// reading it in two pieces would move the same bytes in more requests.
    static func tensorPlan(layer index: Int, hasExperts: Bool, config: TinyK3Config) -> [String] {
        attentionTensorPlan(layer: index, config: config)
            + feedForwardTensorPlan(layer: index, hasExperts: hasExperts, config: config)
    }

    /// One layer's two byte totals, over the planned tensors only.
    ///
    /// Stored is what staging allocates and reads; resident is what the same
    /// tensors occupy once ``RawTensor/materialise()`` has widened them, which
    /// is `2 ×` for BF16 and unchanged for the handful of F32 vectors.
    static func layerBytes(_ stream: FlagshipLayerStreams, planned: Set<String>)
        -> DeterministicReadAheadSchedule.LayerBytes
    {
        var stored = UInt64Accounting.SaturatingSum()
        var resident = UInt64Accounting.SaturatingSum()
        for unit in stream.deterministic.units where planned.contains(unit.tensor) {
            let payload = UInt64(unit.payloadLength)
            stored.add(payload)
            if unit.dtype == "F32" {
                resident.add(payload)
            } else {
                let widened = payload.multipliedReportingOverflow(by: 2)
                resident.add(widened.overflow ? .max : widened.partialValue)
            }
        }
        return DeterministicReadAheadSchedule.LayerBytes(
            storedBytes: stored.value, residentBytes: resident.value)
    }
}

extension FlagshipLayerStreams {
    /// Whether this layer routes. Container presence, not a manifest flag.
    public var hasExperts: Bool { !containers.isEmpty }
}

// MARK: - Sub-layer residency

/// What the two-phase layer residency actually held, for the run's JSON
/// (AGENTS.md: a parameter that changes a measured result is reported next to
/// the result).
///
/// The bytes here are **widened** bytes, summed over the arrays the halves
/// really keep alive rather than over a plan's idea of them — the two differ,
/// because a pair like `self_attention_res_norm × _proj` is held as one product
/// and not as the two vectors that made it.
public struct SubLayerResidencyMetrics: Codable, Sendable, Equatable {

    /// Widened bytes held right now. Zero at rest, on every path.
    public var liveWidenedBytes: UInt64 = 0
    /// The high-water mark of the line above: the term a memory budget for the
    /// deterministic half is stated in.
    public var peakWidenedBytes: UInt64 = 0
    public var peakAttentionWidenedBytes: UInt64 = 0
    public var peakFeedForwardWidenedBytes: UInt64 = 0
    /// The widest layer's two halves *added*, whether or not they were ever
    /// resident together — what ``peakWidenedBytes`` would have been before the
    /// split, measured on this run instead of quoted from an older one.
    public var peakLayerWidenedBytes: UInt64 = 0
    /// How many times a second half was widened while the first was still held.
    /// Zero is the two-phase decode path; a whole-layer `layerWeights(_:)` call
    /// is exactly what makes it non-zero, and that is not an error — it is the
    /// inspection path saying so.
    public var coresidentHalfCount: Int = 0
    /// Sticky. A saturated byte total cannot prove a residency claim.
    public var accountingOverflowed: Bool = false

    public init() {}

    /// What the split bought at the widest moment: `1 - peak / wholeLayerPeak`.
    /// Zero when nothing has been widened, and zero for a run that only ever
    /// asked for whole layers.
    public var peakReductionRatio: Double {
        guard peakLayerWidenedBytes > 0 else { return 0 }
        return max(0, 1 - Double(peakWidenedBytes) / Double(peakLayerWidenedBytes))
    }

    /// The claim the split has to earn: the peak is one half, not two.
    public var halvesWereNeverCoresident: Bool {
        coresidentHalfCount == 0
            && peakWidenedBytes == max(peakAttentionWidenedBytes, peakFeedForwardWidenedBytes)
    }

    public var summaryLine: String {
        String(
            format: "[sublayer] peak %llu B (attn %llu B, mlp %llu B) against %llu B whole-layer, "
                + "%.1f%% lower, %d co-resident widenings",
            peakWidenedBytes, peakAttentionWidenedBytes, peakFeedForwardWidenedBytes,
            peakLayerWidenedBytes, 100 * peakReductionRatio, coresidentHalfCount)
    }
}

// MARK: - Reading

/// One layer's deterministic blob, addressed by tensor name.
///
/// The published manifest's `units[]` is a row-band decomposition of each
/// tensor: `q_proj.weight` at flagship is `[12288, 7168]` in 48 bands of 256
/// rows. A band is 16 KiB-aligned and padded to `length`, of which
/// `payload_length` is content — so a tensor is reassembled band by band into
/// its own buffer rather than read as one range, because the padding between
/// bands is not part of the tensor.
final class DeterministicBlob {

    private let path: String
    private let descriptor: Int32
    private let byTensor: [String: [FlagshipLayerStreams.DeterministicUnit]]
    private let successfulReadObserver: (@Sendable (UInt64) -> Void)?
    private(set) var bytesRead: UInt64 = 0
    private(set) var byteAccountingOverflowed = false

    init(
        stream: FlagshipLayerStreams,
        successfulReadObserver: (@Sendable (UInt64) -> Void)? = nil
    ) throws {
        self.path = stream.deterministic.path
        self.descriptor = try stream.deterministic.fileAccess.open(stream.deterministic.path)
        self.successfulReadObserver = successfulReadObserver
        var grouped: [String: [FlagshipLayerStreams.DeterministicUnit]] = [:]
        for unit in stream.deterministic.units { grouped[unit.tensor, default: []].append(unit) }
        for key in grouped.keys { grouped[key]?.sort { $0.rowStart < $1.rowStart } }
        self.byTensor = grouped
    }

    func close() { Darwin.close(descriptor) }
    func resetAccounting() {
        bytesRead = 0
        byteAccountingOverflowed = false
    }

    /// A `[rows, cols]` float32 array assembled from every band of `name`.
    ///
    /// Two steps, split so the read-ahead can take the first on a background
    /// thread and leave the second exactly where it is: ``rawTensor(_:)`` is
    /// `pread` and nothing else, ``RawTensor/materialise()`` is the widening.
    /// Calling them back to back — which is all this does — is what the serial
    /// path has always done.
    func tensor(_ name: String) throws -> MLXArray {
        try rawTensor(name).materialise()
    }

    /// Every band of `name`, assembled into one buffer. No MLX, no widening,
    /// no arithmetic: safe to run wherever the bytes are wanted from.
    func rawTensor(_ name: String) throws -> RawTensor {
        guard let units = byTensor[name], let first = units.first else {
            throw TinyK3Error.missingTensor("\(name) (in \(path))")
        }
        let cols = first.cols
        let elementBytes = first.dtype == "F32" ? 4 : 2
        var rows = 0
        for unit in units {
            guard unit.cols == cols, unit.dtype == first.dtype else {
                throw TinyK3Error.configuration(
                    "\(name): band \(unit.index) is \(unit.rows)x\(unit.cols) \(unit.dtype), "
                        + "the first band is \(first.rows)x\(cols) \(first.dtype)")
            }
            guard unit.rowStart == rows else {
                throw TinyK3Error.configuration(
                    "\(name): band \(unit.index) starts at row \(unit.rowStart), expected \(rows) "
                        + "— the bands do not tile the tensor")
            }
            guard unit.payloadLength == unit.rows * cols * elementBytes else {
                throw TinyK3Error.configuration(
                    "\(name): band \(unit.index) declares \(unit.payloadLength) payload bytes for "
                        + "\(unit.rows)x\(cols) \(unit.dtype)")
            }
            rows += unit.rows
        }

        let total = rows * cols * elementBytes
        let buffer = UnsafeMutableRawPointer.allocate(
            byteCount: max(total, 1), alignment: 16)
        var adopted = false
        defer { if !adopted { buffer.deallocate() } }
        for unit in units {
            var done = 0
            let destination = buffer + unit.rowStart * cols * elementBytes
            while done < unit.payloadLength {
                let got = pread(
                    descriptor, destination + done, unit.payloadLength - done,
                    off_t(unit.offset) + off_t(done))
                if got < 0 {
                    if errno == EINTR { continue }
                    throw StorageCoreError.posix(operation: "pread", path: path, code: errno)
                }
                if got == 0 {
                    throw StorageCoreError.shortTransfer(
                        operation: "pread", path: path, offset: unit.offset,
                        expected: unit.payloadLength, actual: done)
                }
                done += got
            }
            successfulReadObserver?(UInt64(unit.payloadLength))
            var sum = UInt64Accounting.SaturatingSum(
                value: bytesRead, didOverflow: byteAccountingOverflowed)
            sum.add(UInt64(unit.payloadLength))
            bytesRead = sum.value
            byteAccountingOverflowed = sum.didOverflow
        }

        let raw = RawTensor(
            name: name, rows: rows, cols: cols,
            dtype: first.dtype == "F32" ? .float32 : .bfloat16,
            byteCount: total, buffer: buffer)
        adopted = true
        return raw
    }

    /// A one-dimensional tensor, flattened. `[1, n]` and `[n]` both occur.
    func vector(_ name: String) throws -> MLXArray {
        let array = try tensor(name)
        return array.reshaped([-1])
    }
}

/// A dense row-major table read one row at a time.
///
/// `embed_tokens` and `lm_head` are 2.35 GB each — larger than everything else
/// in the model put together — so neither is ever materialised (spec 16.4,
/// 16.5). The published `global/manifest.json` no longer describes them, so the
/// layout is recovered from the geometry and then *checked*: a dense
/// `[vocab, hidden]` BF16 table is exactly `vocab * hidden * 2` bytes, and a
/// unit-tiled or padded one is not. The check is what distinguishes the reading
/// that was recovered from the reading that was guessed.
final class RowAddressedBlob {

    let path: String
    let rows: Int
    let cols: Int
    let dtype: DType
    private let descriptor: Int32
    private let elementBytes: Int
    private let successfulReadObserver: (@Sendable (UInt64) -> Void)?

    init(
        path: String, rows: Int, cols: Int, dtype: DType,
        fileAccess: ModelFileAccess = .filesystem,
        successfulReadObserver: (@Sendable (UInt64) -> Void)? = nil
    ) throws {
        self.path = path
        self.rows = rows
        self.cols = cols
        self.dtype = dtype
        self.elementBytes = dtype == .float32 ? 4 : 2
        self.successfulReadObserver = successfulReadObserver
        let descriptor = try fileAccess.open(path)
        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            let code = errno
            Darwin.close(descriptor)
            throw StorageCoreError.posix(operation: "fstat", path: path, code: code)
        }
        let expected = rows * cols * elementBytes
        guard Int(status.st_size) == expected else {
            Darwin.close(descriptor)
            throw TinyK3Error.configuration(
                "\(URL(fileURLWithPath: path).lastPathComponent) is \(status.st_size) bytes; a "
                    + "dense [\(rows), \(cols)] \(dtype) table is \(expected). The row-addressed "
                    + "reading of this file does not hold.")
        }
        self.descriptor = descriptor
    }

    func close() { Darwin.close(descriptor) }

    var rowBytes: Int { cols * elementBytes }

    /// The named rows, in the order given. Never the table.
    func rows(_ indices: [Int]) throws -> MLXArray {
        let buffer = UnsafeMutableRawPointer.allocate(
            byteCount: max(indices.count * rowBytes, 1), alignment: 16)
        var filled = false
        defer { if !filled { buffer.deallocate() } }
        for (position, row) in indices.enumerated() {
            guard row >= 0, row < rows else {
                throw TinyK3Error.configuration(
                    "row \(row) requested from a \(rows)-row table")
            }
            try read(into: buffer + position * rowBytes, count: rowBytes, at: row * rowBytes)
        }
        return try finish(buffer, rows: indices.count, filled: &filled)
    }

    /// `count` consecutive rows from `first` — one read, not `count`.
    func rowRange(from first: Int, count: Int) throws -> MLXArray {
        guard first >= 0, count >= 0, first + count <= rows else {
            throw TinyK3Error.configuration(
                "rows \(first)..<\(first + count) requested from a \(rows)-row table")
        }
        let bytes = count * rowBytes
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: max(bytes, 1), alignment: 16)
        var filled = false
        defer { if !filled { buffer.deallocate() } }
        try read(into: buffer, count: bytes, at: first * rowBytes)
        return try finish(buffer, rows: count, filled: &filled)
    }

    private func finish(
        _ buffer: UnsafeMutableRawPointer, rows count: Int, filled: inout Bool
    ) throws -> MLXArray {
        let raw = MLXArray(
            rawPointer: buffer, [count, cols], dtype: dtype,
            finalizer: { buffer.deallocate() })
        filled = true
        let widened = raw.asType(.float32)
        widened.eval()
        return widened
    }

    private func read(into destination: UnsafeMutableRawPointer, count: Int, at offset: Int) throws
    {
        var done = 0
        while done < count {
            let got = pread(descriptor, destination + done, count - done, off_t(offset + done))
            if got < 0 {
                if errno == EINTR { continue }
                throw StorageCoreError.posix(operation: "pread", path: path, code: errno)
            }
            if got == 0 {
                throw StorageCoreError.shortTransfer(
                    operation: "pread", path: path, offset: UInt64(offset),
                    expected: count, actual: done)
            }
            done += got
        }
        successfulReadObserver?(UInt64(count))
    }
}
