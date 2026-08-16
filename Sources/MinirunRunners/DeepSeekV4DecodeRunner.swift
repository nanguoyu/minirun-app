import Foundation
import MLXBridge
import MinirunKit
import ModelAdapters
import StorageCore

/// DeepSeek V4 Flash decoded from one completely verified unit-bundle artifact.
///
/// The `.product` scale is the measured App boundary. The `.stated` scale
/// remains available to explicit harnesses whose own gate declares a floor and
/// token ceiling. Neither scale turns the machine's current free memory into a
/// budget, and neither silently loads the whole model.
public struct DeepSeekV4DecodeRunner: RunnerFacade {
    public enum Scale: Sendable, Equatable {
        case stated(minimumBudgetBytes: UInt64, maximumNewTokens: Int)
        /// The measured product boundary: 512 prompt tokens, 64 generated
        /// tokens, and request-derived cache/pool arithmetic before acceptance.
        case product
    }

    /// Only knobs the V4 pager and output head consume today.
    ///
    /// `expertsPerDispatch` is deliberately absent. The shared field names a
    /// count of experts, while V4's backend batches *container tiles* whose
    /// expert multiplicity belongs to the verified header. Reusing the field
    /// would make the recorded setting describe a different quantity.
    public static let knobs: Set<String> = [
        "expertReadAhead", "queueDepth", "expertPoolSlots",
        "mlxCacheLimitBytes", "logitChunkRows",
    ]

    public let scale: Scale
    private let factory: any DeepSeekV4RunWorkloadFactory

    public init(scale: Scale) {
        self.init(scale: scale, factory: DeepSeekV4ArtifactWorkloadFactory())
    }

    init(scale: Scale, factory: any DeepSeekV4RunWorkloadFactory) {
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
            minimum = DeepSeekV4ProductMemoryBudget.minimumBudgetBytes
            maximum = DeepSeekV4ProductMemoryBudget.maximumNewTokens
        }
        return RunnerCapabilities(
            model: .deepseekV4Flash,
            layout: .v4FlashUnitBundle,
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
                .deepseekV4Flash,
                reason: "the MLX V4 runtime requires Apple silicon")
        #else
            try validateCommon(request)
            guard capabilities.maximumNewTokens >= 1 else {
                throw RunError.runnerUnavailable(
                    .deepseekV4Flash,
                    reason: "this runner was created without a positive token ceiling")
            }
            if request.artifact.scope?.isReleased == true {
                throw RunError.scopeRefused(path: request.artifact.root.path)
            }
            if scale == .product,
                case .tokenIDs(let tokenIDs) = request.prompt,
                tokenIDs.count > DeepSeekV4ProductMemoryBudget.maximumPromptTokens
            {
                throw RunError.promptNotSupported(
                    reason: "the verified V4 product runtime accepts at most "
                        + "\(DeepSeekV4ProductMemoryBudget.maximumPromptTokens) prompt tokens")
            }
            _ = try DeepSeekV4EffectiveKnobs.resolve(
                request.knobs, scale: scale,
                declaredBudgetBytes: request.memoryBudgetBytes)

            if factory.requiresRuntimeAuthority {
                guard let authority = request.artifact.runtimeAuthority else {
                    throw RunError.artifactNotReady(
                        "DeepSeek V4 requires complete verification and a rooted runtime authority")
                }
                guard authority.rootURL.standardizedFileURL.path
                    == request.artifact.root.standardizedFileURL.path
                else {
                    throw RunError.artifactNotReady(
                        "the rooted runtime authority does not name the requested V4 artifact")
                }
                try Self.validateExecutionEvidence(authority)
                try authority.validateCurrentBinding()
            }
        #endif
    }

    public func start(_ request: RunRequest) throws -> RunSession {
        try validate(request)
        let knobs = try DeepSeekV4EffectiveKnobs.resolve(
            request.knobs, scale: scale,
            declaredBudgetBytes: request.memoryBudgetBytes)
        let handle = RunHandleID()
        let cancellation = DeepSeekV4RunCancellation(
            boundaryReclaim: knobs.boundaryReclaim(
                declaredBudgetBytes: request.memoryBudgetBytes))
        let (events, continuation) = AsyncThrowingStream<RunEvent, Error>.makeStream()
        let engine = DeepSeekV4RunEngine(
            request: request, capabilities: capabilities, knobs: knobs,
            enforcesProductMemoryPolicy: scale == .product,
            handle: handle, cancellation: cancellation, factory: factory,
            continuation: continuation)

        continuation.onTermination = { _ in cancellation.cancel() }
        let thread = Thread { engine.run() }
        thread.name = "minirun.v4-decode"
        thread.qualityOfService = .userInitiated
        thread.stackSize = 8 << 20
        thread.start()

        return RunSession(
            handle: handle, events: events,
            payloadFlow: cancellation.payloadFlow.stream,
            cancel: { cancellation.cancel() })
    }

    /// Validate the publication facts that can be checked without reading a
    /// model byte. The small declared files are rehashed by the workload
    /// factory before acceptance; this earlier gate keeps an impossible tree
    /// from starting a session at all.
    static func validateExecutionEvidence(
        _ authority: ArtifactRuntimeAuthority
    ) throws {
        let evidence = authority.evidence
        guard evidence.model == .deepseekV4Flash else {
            throw RunError.artifactNotReady(
                "verification evidence belongs to \(evidence.model), not DeepSeek V4 Flash")
        }
        guard evidence.index.shape == .singleSourceUnits,
            evidence.index.repositories.count == 1,
            let source = evidence.index.repositories.first,
            let sourceRevision = source.revision,
            !sourceRevision.isEmpty
        else {
            throw RunError.artifactLayoutMismatch(
                expected: .v4FlashUnitBundle, atPath: authority.rootURL.path)
        }
        guard let configuration = evidence.index.configuration else {
            throw RunError.artifactNotReady(
                "index.json does not bind a verified DeepSeek V4 config")
        }
        guard let tokenizer = evidence.index.tokenizer else {
            throw RunError.artifactNotReady(
                "index.json does not bind a verified DeepSeek V4 tokenizer")
        }
        guard configuration.sourceRepo == source.repoID,
            configuration.sourceRevision == sourceRevision,
            tokenizer.sourceRepo == source.repoID,
            tokenizer.sourceRevision == sourceRevision
        else {
            throw RunError.artifactNotReady(
                "the V4 config, tokenizer, and unit index do not share one source revision")
        }

        let paths = Set(authority.verifiedRepositoryPaths)
        let required = [
            "index.json", configuration.file, tokenizer.file, tokenizer.license,
        ]
        guard required.allSatisfy(paths.contains) else {
            throw RunError.artifactNotReady(
                "the fully verified tree omits a V4 runtime metadata file")
        }
    }
}

struct DeepSeekV4EffectiveKnobs: Sendable, Equatable {
    var expertReadAhead: Int
    var queueDepth: Int
    var expertPoolSlots: Int
    var mlxCacheLimitBytes: Int
    var logitChunkRows: Int

    /// The ceiling on the budget-linked MLX cache default. A recyclable cache
    /// larger than this buys nothing V4 has evidence for, whatever the budget.
    static let maximumDefaultMLXCacheBytes: UInt64 = 512 << 20

    /// Copy transfer returns every pager lease before compute is expressed, so
    /// the default pool is exactly the two-tile read-ahead window rather than
    /// that window plus a graph-held operand slot as adoption requires. Named
    /// because the memory dial prices the expert-pool term with it before a
    /// run exists to resolve knobs.
    static let defaultExpertPoolSlots = 2

    /// The MLX cache limit a run gets when the operator states none.
    ///
    /// `.product` keeps 0: MLX buffer recycling stays off, every boundary
    /// reclaims, and the accepted 2 GB floor and its accounting identity are
    /// byte-identical to before this default existed.
    ///
    /// `.stated` takes `min(declaredBudget / 10, 512 MiB)`. The budget is the
    /// operator's own declaration, so a tenth of it is memory the run was
    /// already given, and it is named in the plan rather than spent silently.
    static func defaultMLXCacheLimitBytes(
        scale: DeepSeekV4DecodeRunner.Scale, declaredBudgetBytes: UInt64
    ) -> Int {
        switch scale {
        case .product:
            return 0
        case .stated:
            let derived = min(
                declaredBudgetBytes / 10, maximumDefaultMLXCacheBytes)
            return Int(clamping: derived)
        }
    }

    static func resolve(
        _ knobs: RunKnobs,
        scale: DeepSeekV4DecodeRunner.Scale,
        declaredBudgetBytes: UInt64
    ) throws -> DeepSeekV4EffectiveKnobs {
        let result = DeepSeekV4EffectiveKnobs(
            expertReadAhead: knobs.expertReadAhead ?? 2,
            queueDepth: knobs.queueDepth ?? 2,
            expertPoolSlots: knobs.expertPoolSlots ?? defaultExpertPoolSlots,
            // A stated `mlxCacheLimitBytes` always wins, including an explicit
            // 0: §5.3 knobs are honoured, never silently overridden.
            mlxCacheLimitBytes: knobs.mlxCacheLimitBytes
                ?? defaultMLXCacheLimitBytes(
                    scale: scale, declaredBudgetBytes: declaredBudgetBytes),
            logitChunkRows: knobs.logitChunkRows ?? 4_096)
        guard result.expertReadAhead >= 1 else {
            throw RunError.knobOutOfRange(
                name: "expertReadAhead", value: "\(result.expertReadAhead)",
                allowed: ">= 1 for the V4 tile pager")
        }
        guard (1...8).contains(result.queueDepth) else {
            throw RunError.knobOutOfRange(
                name: "queueDepth", value: "\(result.queueDepth)", allowed: "1...8")
        }
        guard result.expertPoolSlots >= result.expertReadAhead else {
            throw RunError.knobOutOfRange(
                name: "expertPoolSlots", value: "\(result.expertPoolSlots)",
                allowed: ">= expertReadAhead (\(result.expertReadAhead)) for copy transfer")
        }
        return result
    }

    var expertConfiguration: PagedRoutedExpertBackend.Configuration {
        PagedRoutedExpertBackend.Configuration(
            slotCount: expertPoolSlots, queueDepth: queueDepth,
            readAhead: expertReadAhead, noCache: false,
            slotAcquireTimeout: 60, tilesPerDispatch: 1,
            transferMode: .copy)
    }

    /// Where this run reclaims. A zero cache limit recycles nothing, so it
    /// keeps the unconditional every-boundary reclaim; a non-zero limit would
    /// be defeated by that, and reclaims only at the policy's stated margin.
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
        return result
    }

    var asStrings: [String: String] {
        [
            "expertReadAhead": "\(expertReadAhead)",
            "queueDepth": "\(queueDepth)",
            "expertPoolSlots": "\(expertPoolSlots)",
            "mlxCacheLimitBytes": "\(mlxCacheLimitBytes)",
            "logitChunkRows": "\(logitChunkRows)",
            "expertTransferMode": BufferTransferMode.copy.rawValue,
        ]
    }
}

/// One terminal cancellation authority shared by the session, the blocking
/// loop, and the currently active V4 pager.
final class DeepSeekV4RunCancellation: @unchecked Sendable {
    let latch = RunCancellationLatch()
    let executionControl: DeepSeekV4ExecutionControl
    let payloadFlow = RunPayloadFlowMeter()

    private let lock = NSLock()
    private var overBudgetPeak: UInt64?
    private var artifactFailureReason: String?

    init(
        boundaryReclaim: DeepSeekV4BoundaryReclaimPolicy = DeepSeekV4BoundaryReclaimPolicy()
    ) {
        executionControl = DeepSeekV4ExecutionControl(
            boundaryReclaim: boundaryReclaim)
    }

    var isCancelled: Bool { latch.isCancelled }

    var budgetExceededPeak: UInt64? {
        lock.lock()
        defer { lock.unlock() }
        return overBudgetPeak
    }

    var artifactFailure: String? {
        lock.lock()
        defer { lock.unlock() }
        return artifactFailureReason
    }

    func cancel() {
        latch.cancel()
        executionControl.cancel()
    }

    func exceedBudget(peakBytes: UInt64) {
        lock.lock()
        if overBudgetPeak == nil { overBudgetPeak = peakBytes }
        lock.unlock()
        cancel()
    }

    func failArtifact(_ reason: String) {
        lock.lock()
        if artifactFailureReason == nil { artifactFailureReason = reason }
        lock.unlock()
        cancel()
    }

    func check() throws {
        try latch.check()
        try executionControl.throwIfCancelled()
    }
}

struct DeepSeekV4WorkloadStep: Sendable, Equatable {
    let tokenID: Int
    let logits: [Float]
    let completedLayers: Int
}

protocol DeepSeekV4RunWorkload: AnyObject {
    var layerCount: Int { get }
    var vocabularySize: Int { get }
    var eosTokenID: Int { get }
    var maximumPositionCount: Int { get }
    var sourceRepository: String { get }
    var sourceRevision: String { get }
    /// Exact resident storage reserved by the routed-expert tile pool. The
    /// workload derives it from admitted container geometry and the accepted
    /// slot count; the engine refuses the run before acceptance when the
    /// caller's stated budget cannot fund it.
    var expertPoolBudgetBytes: UInt64 { get }
    /// Cumulative generation-payload reads. Nil only for an explicit harness
    /// that does not implement byte observation; the product workload always
    /// returns a complete deterministic/expert split.
    var readAccountingSnapshot: DeepSeekV4ReadAccountingSnapshot? { get }
    /// Cumulative wall-time brackets for the phases a pass moves through. The
    /// engine subtracts consecutive samples to attribute one pass; nil for a
    /// harness workload that does not decompose its own time.
    var phaseMetricsSnapshot: DeepSeekV4PhaseMetrics? { get }

    func productMemoryPlan(
        declaredBudgetBytes: UInt64,
        promptTokenCount: Int,
        maximumNewTokens: Int,
        mlxCacheBytes: UInt64,
        pinnedDeterministicBytes: UInt64,
        enforcesProductLimits: Bool
    ) throws -> DeepSeekV4ProductMemoryPlan

    /// The memory dial's census of this artifact, from the layer manifests.
    /// Nil for a harness workload that has no manifests to census, which is
    /// then simply a run that pins nothing.
    var deterministicCensus: DeepSeekV4MemoryDial.Census? { get }

    /// Hold these layers — and, when the plan reached the globals rung, the
    /// output head table — resident for the run. Called once, after the plan is
    /// accepted and before the first payload read.
    func installPinnedTier(layers: Set<Int>, budgetBytes: UInt64, outputHead: Bool)

    /// What the tier held and what it saved. Nil when nothing was pinned.
    var pinnedTierMetrics: DeepSeekV4PinnedTierMetrics? { get }

    func prefill(
        tokenIDs: [Int], decodePositionLimit: Int,
        cancellationCheck: () throws -> Void,
        onLayerCompleted: (_ completed: Int, _ total: Int) -> Void
    ) throws -> DeepSeekV4WorkloadStep

    func decode(
        tokenID: Int, cancellationCheck: () throws -> Void,
        onLayerCompleted: (_ completed: Int, _ total: Int) -> Void
    ) throws -> DeepSeekV4WorkloadStep

    func validateTerminalBinding() throws
    func shutdown()
}

extension DeepSeekV4RunWorkload {
    /// A workload that measures nothing says so, rather than reporting a
    /// decomposition of zeros that would read as a measured zero.
    var phaseMetricsSnapshot: DeepSeekV4PhaseMetrics? { nil }

    /// A workload with no manifests cannot be censused, and a run that cannot
    /// census its artifact pins nothing rather than guessing at one.
    var deterministicCensus: DeepSeekV4MemoryDial.Census? { nil }
    func installPinnedTier(layers: Set<Int>, budgetBytes: UInt64, outputHead: Bool) {}
    var pinnedTierMetrics: DeepSeekV4PinnedTierMetrics? { nil }
}

protocol DeepSeekV4RunWorkloadFactory: Sendable {
    var requiresRuntimeAuthority: Bool { get }

    func prepare(
        request: RunRequest, knobs: DeepSeekV4EffectiveKnobs,
        cancellation: DeepSeekV4RunCancellation
    ) throws -> any DeepSeekV4RunWorkload
}
