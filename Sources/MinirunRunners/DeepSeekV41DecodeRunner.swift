import Foundation
import MLXBridge
import MinirunKit
import ModelAdapters
import StorageCore

/// DeepSeek V4.1 Flash decoded from one completely verified unit-bundle
/// artifact.
///
/// The shape is ``DeepSeekV4DecodeRunner``'s, deliberately: two scales, the same
/// evidence gate before a session may start, the same refusal-not-clamp rule on
/// every knob, and the same rooted authority on every open. What differs is what
/// the knobs *mean*, because a V4.1 expert is half a tile and a V4 expert is a
/// whole one.
public struct DeepSeekV41DecodeRunner: RunnerFacade {
    public enum Scale: Sendable, Equatable {
        case stated(minimumBudgetBytes: UInt64, maximumNewTokens: Int)
        /// The measured product boundary.
        case product
    }

    /// Only knobs the V4.1 pool and output head consume.
    ///
    /// `expertProjectionPrefetch` and `expertCrossLayerPrefetch` are absent, and
    /// not by oversight: the V4.1 pool is keyed on `(unit, projection, tile)`
    /// and one `prefetch` call queues **all three** projections of the block's
    /// routed set, so projection prefetch is not an option a run can turn off.
    /// A cross-block window has nothing to queue for the same reason V4's needs
    /// run-scoped backends and V4.1 has no per-block backend to be scoped
    /// against: the pool is already the run's.
    public static let knobs: Set<String> = [
        "expertReadAhead", "queueDepth", "expertPoolSlots",
        "mlxCacheLimitBytes", "logitChunkRows", "expertTileAdoption",
    ]

    public let scale: Scale
    private let factory: any DeepSeekV41RunWorkloadFactory

    public init(scale: Scale) {
        self.init(scale: scale, factory: DeepSeekV41ArtifactWorkloadFactory())
    }

    init(scale: Scale, factory: any DeepSeekV41RunWorkloadFactory) {
        self.scale = scale
        self.factory = factory
    }

    public var capabilities: RunnerCapabilities {
        let minimum: UInt64
        let maximum: Int
        switch scale {
        case .stated(let minimumBudgetBytes, let maximumNewTokens):
            minimum = minimumBudgetBytes
            maximum = maximumNewTokens
        case .product:
            minimum = DeepSeekV41ProductMemoryBudget.minimumBudgetBytes
            maximum = DeepSeekV41ProductMemoryBudget.maximumNewTokens
        }
        return RunnerCapabilities(
            model: .deepseekV41Flash,
            layout: .v41FlashUnitBundle,
            supportedKnobs: Self.knobs,
            minimumBudgetBytes: minimum,
            maximumNewTokens: maximum,
            acceptsTextPrompts: false,
            requiresMLX: true,
            tokenEventGranularity: .perPass)
    }

    public func validate(_ request: RunRequest) throws {
        #if !arch(arm64)
            throw RunError.runnerUnavailable(
                .deepseekV41Flash,
                reason: "the MLX V4.1 runtime requires Apple silicon")
        #else
            try validateCommon(request)
            guard capabilities.maximumNewTokens >= 1 else {
                throw RunError.runnerUnavailable(
                    .deepseekV41Flash,
                    reason: "this runner was created without a positive token ceiling")
            }
            if request.artifact.scope?.isReleased == true {
                throw RunError.scopeRefused(path: request.artifact.root.path)
            }
            if scale == .product,
                case .tokenIDs(let tokenIDs) = request.prompt,
                tokenIDs.count > DeepSeekV41ProductMemoryBudget.maximumPromptTokens
            {
                throw RunError.promptNotSupported(
                    reason: "the verified V4.1 product runtime accepts at most "
                        + "\(DeepSeekV41ProductMemoryBudget.maximumPromptTokens) prompt tokens")
            }
            _ = try DeepSeekV41EffectiveKnobs.resolve(
                request.knobs, scale: scale,
                declaredBudgetBytes: request.memoryBudgetBytes)

            if factory.requiresRuntimeAuthority {
                guard let authority = request.artifact.runtimeAuthority else {
                    throw RunError.artifactNotReady(
                        "DeepSeek V4.1 requires complete verification and a rooted runtime "
                            + "authority")
                }
                guard authority.rootURL.standardizedFileURL.path
                    == request.artifact.root.standardizedFileURL.path
                else {
                    throw RunError.artifactNotReady(
                        "the rooted runtime authority does not name the requested V4.1 artifact")
                }
                try Self.validateExecutionEvidence(authority)
                try authority.validateCurrentBinding()
            }
        #endif
    }

    public func start(_ request: RunRequest) throws -> RunSession {
        try validate(request)
        let knobs = try DeepSeekV41EffectiveKnobs.resolve(
            request.knobs, scale: scale,
            declaredBudgetBytes: request.memoryBudgetBytes)
        let handle = RunHandleID()
        let cancellation = DeepSeekV4RunCancellation(
            boundaryReclaim: knobs.boundaryReclaim(
                declaredBudgetBytes: request.memoryBudgetBytes))
        let (events, continuation) = AsyncThrowingStream<RunEvent, Error>.makeStream()
        let engine = DeepSeekV41RunEngine(
            request: request, capabilities: capabilities, knobs: knobs,
            enforcesProductMemoryPolicy: scale == .product,
            handle: handle, cancellation: cancellation, factory: factory,
            continuation: continuation)

        continuation.onTermination = { _ in cancellation.cancel() }
        let thread = Thread { engine.run() }
        thread.name = "minirun.v41-decode"
        thread.qualityOfService = .userInitiated
        thread.stackSize = 8 << 20
        thread.start()

        return RunSession(
            handle: handle, events: events,
            payloadFlow: cancellation.payloadFlow.stream,
            cancel: { cancellation.cancel() })
    }

    /// The publication facts that can be checked without reading a model byte.
    static func validateExecutionEvidence(
        _ authority: ArtifactRuntimeAuthority
    ) throws {
        let evidence = authority.evidence
        guard evidence.model == .deepseekV41Flash else {
            throw RunError.artifactNotReady(
                "verification evidence belongs to \(evidence.model), not DeepSeek V4.1 Flash")
        }
        guard evidence.index.shape == .singleSourceUnits,
            evidence.index.repositories.count == 1,
            let source = evidence.index.repositories.first,
            let sourceRevision = source.revision,
            !sourceRevision.isEmpty
        else {
            throw RunError.artifactLayoutMismatch(
                expected: .v41FlashUnitBundle, atPath: authority.rootURL.path)
        }
        guard let configuration = evidence.index.configuration else {
            throw RunError.artifactNotReady(
                "index.json does not bind a verified DeepSeek V4.1 config")
        }
        guard let tokenizer = evidence.index.tokenizer else {
            throw RunError.artifactNotReady(
                "index.json does not bind a verified DeepSeek V4.1 tokenizer")
        }
        guard configuration.sourceRepo == source.repoID,
            configuration.sourceRevision == sourceRevision,
            tokenizer.sourceRepo == source.repoID,
            tokenizer.sourceRevision == sourceRevision
        else {
            throw RunError.artifactNotReady(
                "the V4.1 config, tokenizer, and unit index do not share one source revision")
        }
        let paths = Set(authority.verifiedRepositoryPaths)
        // `inference-config.json` joins V4's four. It is the file the reference
        // runner reads and the one ADR 0020 made the authority, so a tree that
        // verified everything else and not it is a tree this runner refuses.
        let required = [
            "index.json", configuration.file, tokenizer.file, tokenizer.license,
            "inference-config.json",
        ]
        guard required.allSatisfy(paths.contains) else {
            throw RunError.artifactNotReady(
                "the fully verified tree omits a V4.1 runtime metadata file")
        }
    }
}

struct DeepSeekV41EffectiveKnobs: Sendable, Equatable {
    /// Tiles the pool reads ahead when a block's routed set is named.
    var expertReadAhead: Int
    var queueDepth: Int
    var expertPoolSlots: Int
    var mlxCacheLimitBytes: Int
    var logitChunkRows: Int
    var expertTileAdoption: Bool
    /// Whether the routed gather's operands and the output head's row windows
    /// are evaluated as they are built.
    ///
    /// Off is what every phase 3 arm measured: nothing evaluates inside
    /// `perExpertOutputs`' token loop or `streamedLogits`' window loop, so an
    /// N-token prefill holds 3N gather operands and the whole 1.32 GB head
    /// table alive until the next block's router evaluates. On, at most one
    /// token's three operands and one window are alive, and the run's widest
    /// term stops scaling with the prompt.
    ///
    /// It changes no arithmetic — `eval` forces a computation MLX would perform
    /// anyway, in the same order — which is why the fixture's whole-model test
    /// asserts both modes against the same reference logits.
    ///
    /// Its neighbour ``expertTileAdoption`` says whether a gather operand
    /// adopts its own page-aligned allocation instead of being copied into
    /// MLX-owned arrays — V4's knob of the same name, for the term V4.1's phase
    /// 3 record measured at 11.10 s of a 66.30 s pinned decode.
    var boundsLiveOperands: Bool

    /// `w1`, `w3`, `w2` — the three projections one block reads for the same
    /// selected experts, and therefore the number of pool entries one expert
    /// costs.
    static let projectionsPerBlock = DeepSeekV41ExpertProjection.allCases.count

    /// The pool a run gets when the operator states none.
    ///
    /// V4's defaults, arrived at from V4's own arms and kept here **as a
    /// starting point that this record measures rather than inherits**: a
    /// read-ahead of 6 is one decode token's whole routed set, a queue depth of
    /// 4 lets that window drain concurrently, and 20 slots is
    /// 3 projections x 6 + 1 for the tile the consumer is inside. At V4.1's
    /// 12.5 MB pair tile that pool is 251 MB against V4's 20 x 44.6 MB = 892 MB,
    /// so the same slot count is a *smaller* reservation here.
    ///
    /// Since the iPhone-floor work these follow the **platform policy**, the way
    /// K3's do: `DeepSeekV41ProductMemoryBudget.currentPolicy` is the macOS
    /// policy on a Mac, so every number below is unchanged there, and the
    /// iPhone policy states its own.
    static var defaultExpertReadAhead: Int {
        DeepSeekV41ProductMemoryBudget.currentPolicy.expertReadAhead
    }
    static let defaultQueueDepth = 4
    static var defaultExpertPoolSlots: Int {
        DeepSeekV41ProductMemoryBudget.currentPolicy.expertPoolSlots
    }
    static var defaultLogitChunkRows: Int {
        DeepSeekV41ProductMemoryBudget.currentPolicy.headWindowRows
    }

    /// Whether a run that states nothing adopts its gather operands.
    ///
    /// **Off**, and that is a measurement rather than caution.
    /// `docs/experiments/2026-09-11-v41-phase3-runner.md` §6 ran the two modes
    /// from one binary, minutes apart, at 15 GB with all forty blocks and the
    /// head pinned. Adoption does what it was built to do — the copy and the
    /// operand build fall from 10.22 s to 5.38 s over seventeen decode passes —
    /// and the pass gets **slower**, 3.046 s a token against 4.160, because
    /// `gpuWaitSeconds` rises 22.53 s to 38.41 s and `expertIOWaitSeconds`
    /// 11.31 s to 18.20 s. A pinned V4.1 decode wraps 240 external buffers a
    /// pass, and what Metal charges for that is larger than the memcpy it
    /// replaces. ADR 0002 priced adoption's own cost at ~52 us per wrapped
    /// buffer for V4's pager, which streams far fewer, far larger tiles.
    ///
    /// At the product floor it does not even fit: that arm reached a peak of
    /// 4,163,426,176 B against the declared 3,400,000,000 and its own budget
    /// refused it, in prefill, before a token was decoded — an adopted operand
    /// is a separate mapping per *live* stack and a lazy graph keeps many more
    /// of them alive across an eleven-token prefill than across a decode step.
    ///
    /// The mode is kept, tested and bit-identical to the copy, which is the
    /// shape ADR 0002 keeps V4's copy mode in for the mirror-image reason: a
    /// retained alternative whose cost is a measured number.
    static let defaultExpertTileAdoption = false

    static let maximumDefaultMLXCacheBytes: UInt64 = DeepSeekV4EffectiveKnobs
        .maximumDefaultMLXCacheBytes

    static func defaultMLXCacheLimitBytes(
        scale: DeepSeekV41DecodeRunner.Scale, declaredBudgetBytes: UInt64
    ) -> Int {
        switch scale {
        case .product:
            return 0
        case .stated:
            return Int(clamping: min(declaredBudgetBytes / 10, maximumDefaultMLXCacheBytes))
        }
    }

    static func resolve(
        _ knobs: RunKnobs,
        scale: DeepSeekV41DecodeRunner.Scale,
        declaredBudgetBytes: UInt64
    ) throws -> DeepSeekV41EffectiveKnobs {
        let result = DeepSeekV41EffectiveKnobs(
            expertReadAhead: knobs.expertReadAhead ?? defaultExpertReadAhead,
            queueDepth: knobs.queueDepth ?? defaultQueueDepth,
            expertPoolSlots: knobs.expertPoolSlots ?? defaultExpertPoolSlots,
            mlxCacheLimitBytes: knobs.mlxCacheLimitBytes
                ?? defaultMLXCacheLimitBytes(
                    scale: scale, declaredBudgetBytes: declaredBudgetBytes),
            logitChunkRows: knobs.logitChunkRows ?? defaultLogitChunkRows,
            expertTileAdoption: knobs.expertTileAdoption ?? defaultExpertTileAdoption,
            // Not a `RunKnobs` field: whether the live set is bounded is a
            // property of the platform's stated memory policy, and the plan
            // that prices the gather term and the execution that produces it
            // have to be the same decision or the budget is a claim about a
            // different run. A stated-scale harness run gets the same policy;
            // what the scale changes is the residency, not the liveness.
            boundsLiveOperands: DeepSeekV41ProductMemoryBudget.currentPolicy
                .boundsLiveOperands)
        guard result.expertReadAhead >= 1 else {
            throw RunError.knobOutOfRange(
                name: "expertReadAhead", value: "\(result.expertReadAhead)",
                allowed: ">= 1 for the V4.1 tile pool")
        }
        guard (1...8).contains(result.queueDepth) else {
            throw RunError.knobOutOfRange(
                name: "queueDepth", value: "\(result.queueDepth)", allowed: "1...8")
        }
        // Every prefetch names all three projections at once, so the window in
        // flight is `3 x readAhead` tiles and the consumer holds one more.
        // Refused rather than clamped, and refused before 517 GB are opened: a
        // run that narrowed its own window would be recorded under a window it
        // did not use.
        let required = projectionsPerBlock * result.expertReadAhead + 1
        guard result.expertPoolSlots >= required else {
            throw RunError.knobOutOfRange(
                name: "expertPoolSlots", value: "\(result.expertPoolSlots)",
                allowed: ">= \(required), which is \(projectionsPerBlock) projections x "
                    + "expertReadAhead \(result.expertReadAhead) + 1 for the tile the "
                    + "consumer is inside")
        }
        guard result.logitChunkRows >= 1 else {
            throw RunError.knobOutOfRange(
                name: "logitChunkRows", value: "\(result.logitChunkRows)",
                allowed: ">= 1")
        }
        return result
    }

    func boundaryReclaim(
        declaredBudgetBytes: UInt64
    ) -> DeepSeekV4BoundaryReclaimPolicy {
        mlxCacheLimitBytes > 0
            ? DeepSeekV4BoundaryReclaimPolicy(nearBudgetBytes: declaredBudgetBytes)
            : DeepSeekV4BoundaryReclaimPolicy()
    }

    var asRunKnobs: RunKnobs {
        var result = RunKnobs()
        result.expertReadAhead = expertReadAhead
        result.queueDepth = queueDepth
        result.expertPoolSlots = expertPoolSlots
        result.mlxCacheLimitBytes = mlxCacheLimitBytes
        result.logitChunkRows = logitChunkRows
        result.expertTileAdoption = expertTileAdoption
        return result
    }

    var asStrings: [String: String] {
        [
            "expertReadAhead": "\(expertReadAhead)",
            "queueDepth": "\(queueDepth)",
            "expertPoolSlots": "\(expertPoolSlots)",
            "mlxCacheLimitBytes": "\(mlxCacheLimitBytes)",
            "logitChunkRows": "\(logitChunkRows)",
            "expertTileAdoption": "\(expertTileAdoption)",
            "boundsLiveOperands": "\(boundsLiveOperands)",
        ]
    }
}

struct DeepSeekV41WorkloadStep: Sendable, Equatable {
    let tokenID: Int
    let logits: [Float]
    let completedBlocks: Int
}

protocol DeepSeekV41RunWorkload: AnyObject {
    var blockCount: Int { get }
    var vocabularySize: Int { get }
    var eosTokenIDs: Set<Int> { get }
    var maximumPositionCount: Int { get }
    var sourceRepository: String { get }
    var sourceRevision: String { get }
    var expertPoolBudgetBytes: UInt64 { get }
    var readAccountingSnapshot: DeepSeekV4ReadAccountingSnapshot? { get }
    var phaseMetricsSnapshot: DeepSeekV4PhaseMetrics? { get }
    var deterministicCensus: DeepSeekV4MemoryDial.Census? { get }

    func productMemoryPlan(
        declaredBudgetBytes: UInt64,
        promptTokenCount: Int,
        maximumNewTokens: Int,
        mlxCacheBytes: UInt64,
        pinnedDeterministicBytes: UInt64,
        enforcesProductLimits: Bool
    ) throws -> DeepSeekV41ProductMemoryPlan

    /// Set the run's MLX cache ceiling and return everything the process was
    /// holding before it. Called once, after acceptance and **before** the
    /// pinned tier is installed.
    ///
    /// Its own call rather than a side effect of the first pass because of what
    /// runs before a gate's run: a complete verification of 517 GB, in this
    /// process, whose freed buffers macOS's allocator keeps in the zone. A run
    /// admitted against a declared ceiling has to start from a floor it owns,
    /// or it is measured against somebody else's high-water mark.
    func prepareForExecution() throws
    func installPinnedTier(blocks: Set<Int>, outputHead: Bool) throws
    var pinnedResidentBytes: UInt64 { get }
    var pinnedBlockCount: Int { get }
    var pinsOutputHead: Bool { get }
    var expertTileReads: Int { get }
    var expertTileHits: Int { get }

    func prefill(
        tokenIDs: [Int], decodePositionLimit: Int,
        cancellationCheck: () throws -> Void,
        onBlockCompleted: (_ completed: Int, _ total: Int) -> Void
    ) throws -> DeepSeekV41WorkloadStep

    func decode(
        tokenID: Int, cancellationCheck: () throws -> Void,
        onBlockCompleted: (_ completed: Int, _ total: Int) -> Void
    ) throws -> DeepSeekV41WorkloadStep

    func validateTerminalBinding() throws
    func shutdown()
}

extension DeepSeekV41RunWorkload {
    func prepareForExecution() throws {}
    var phaseMetricsSnapshot: DeepSeekV4PhaseMetrics? { nil }
    var deterministicCensus: DeepSeekV4MemoryDial.Census? { nil }
    func installPinnedTier(blocks: Set<Int>, outputHead: Bool) throws {}
    var pinnedResidentBytes: UInt64 { 0 }
    var pinnedBlockCount: Int { 0 }
    var pinsOutputHead: Bool { false }
    var expertTileReads: Int { 0 }
    var expertTileHits: Int { 0 }
}

/// One completed step of preparation, announced as it finishes.
///
/// Preparation used to be silent: the runner's first event of any kind was
/// `.accepted`, after the whole artifact had been opened and the model built,
/// so a process killed during it left a flight recorder holding a start line
/// and nothing else — which is exactly what the owner's iPhone produced three
/// times on 2026-09-11. With this, a kill lands **between two named steps** and
/// the trace says which one it was inside.
typealias DeepSeekV41PreparationReporter = (_ step: String) -> Void

protocol DeepSeekV41RunWorkloadFactory: Sendable {
    var requiresRuntimeAuthority: Bool { get }

    /// - Parameter sentry: the budget's watchman. It replaces the bare
    ///   cancellation check inside the artifact, so that a check at an
    ///   allocation seam is also a budget check. See
    ///   ``DeepSeekV41BudgetSentry``.
    /// - Parameter preparation: called with the name of each preparation step
    ///   as it completes. See ``DeepSeekV41PreparationReporter``.
    func prepare(
        request: RunRequest, knobs: DeepSeekV41EffectiveKnobs,
        cancellation: DeepSeekV4RunCancellation,
        sentry: DeepSeekV41BudgetSentry,
        preparation: @escaping DeepSeekV41PreparationReporter
    ) throws -> any DeepSeekV41RunWorkload
}
