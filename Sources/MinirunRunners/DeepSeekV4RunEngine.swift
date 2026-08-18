import BenchScenarios
import Darwin
import Foundation
import MLX
import MinirunKit
import ModelAdapters
import StorageCore

#if canImport(UIKit)
    import UIKit
#endif

/// Opens the exact V4 files named by full-verification evidence and performs
/// complete metadata/header admission before a run may be accepted.
struct DeepSeekV4ArtifactWorkloadFactory: DeepSeekV4RunWorkloadFactory {
    let requiresRuntimeAuthority = true

    func prepare(
        request: RunRequest, knobs: DeepSeekV4EffectiveKnobs,
        cancellation: DeepSeekV4RunCancellation
    ) throws -> any DeepSeekV4RunWorkload {
        guard let authority = request.artifact.runtimeAuthority else {
            throw RunError.artifactNotReady(
                "DeepSeek V4 requires a rooted runtime authority")
        }
        try cancellation.check()
        try DeepSeekV4DecodeRunner.validateExecutionEvidence(authority)

        let indexIdentity = authority.evidence.index
        guard let configurationIdentity = indexIdentity.configuration,
            let tokenizerIdentity = indexIdentity.tokenizer
        else {
            throw RunError.artifactNotReady(
                "the V4 execution identities disappeared after validation")
        }

        let configData = try Self.verifiedData(
            configurationIdentity, beneath: authority,
            maximumBytes: 1 << 20, cancellation: cancellation)
        let tokenizerData = try Self.verifiedData(
            tokenizerIdentity, beneath: authority,
            maximumBytes: 32 << 20, cancellation: cancellation)
        let indexData = try authority.openFile("index.json").readAll(
            maximumBytes: 8 << 20)
        try cancellation.check()

        let vocabulary = try autoreleasepool {
            try DeepSeekV4Vocabulary(data: tokenizerData)
        }
        // Every control is structural data. Checking the full set here means a
        // compatible-looking ordinary BPE cannot make it through product
        // admission and later fail while a chat is being assembled.
        for control in DeepSeekV4ChatControl.allCases {
            _ = try vocabulary.id(for: control.rawValue)
        }
        let beginningOfSentenceID = try vocabulary.id(
            for: DeepSeekV4ChatControl.beginningOfSentence.rawValue)
        let endOfSentenceID = try vocabulary.id(
            for: DeepSeekV4ChatControl.endOfSentence.rawValue)

        // Every open goes through the held authority: beneath the registered
        // root, `O_NOFOLLOW` on every component, and the complete recorded file
        // identity rechecked before the descriptor is returned. That is what
        // the readers below are told they may rely on.
        let access = ModelFileAccess(identityAssurance: .rootedDescriptorIdentity) {
            repositoryPath in
            try authority.openFileDescriptor(repositoryPath)
        }
        // This run has already hashed every one of these bytes: the complete
        // verification pass covered the exact revision the units declare, and
        // its authority is held open for the run's lifetime. The layer readers
        // therefore stop re-hashing each tile on every load and keep the
        // identity guards that catch a file replaced underneath them.
        let tileDigestPolicy = TileDigestPolicy.trustHeldAuthority(
            HeldVerificationAuthority(
                sourceRepository: configurationIdentity.sourceRepo,
                sourceRevision: configurationIdentity.sourceRevision,
                verifiedFileCount: authority.evidence.files.count,
                verifiedBytes: authority.evidence.selectedBytes,
                heldBy: authority))
        let readAccounting = DeepSeekV4ReadAccounting(
            successfulDeterministicReadObserver: {
                [payloadFlow = cancellation.payloadFlow] bytes in
                payloadFlow.recordDeterministic(bytes)
            },
            successfulExpertReadObserver: {
                [payloadFlow = cancellation.payloadFlow] bytes in
                payloadFlow.recordExpert(bytes)
            })
        // Both scales run the model's arithmetic without the guard-only
        // finiteness sweeps. They are diagnostics — 553 host syncs and 0.39 s
        // of the measured 6.6 s decode pass, none of which any result depends
        // on — and this run still fails closed on a corrupted one: the output
        // head refuses a logit window containing a non-finite value, and
        // `greedyToken` refuses non-finite logits before a token is chosen. A
        // run that wants the per-operator sweeps back states `.validating`.
        let artifact = try DeepSeekV4ModelArtifact(
            configData: configData,
            indexData: indexData,
            fileAccess: access,
            tileDigestPolicy: tileDigestPolicy,
            maximumHeadRowsPerRead: knobs.logitChunkRows,
            readAccounting: readAccounting,
            diagnostics: .off,
            cancellationCheck: cancellation.check,
            loadManifest: { unitID in
                try cancellation.check()
                return try authority.openFile("\(unitID)/manifest.json").readAll(
                    maximumBytes: 8 << 20)
            })
        let config = artifact.plan.config
        guard vocabulary.vocabularySize == config.vocabularySize else {
            throw RunError.artifactNotReady(
                "the V4 tokenizer and model config declare different vocabulary sizes")
        }
        guard beginningOfSentenceID == config.bosTokenID,
            endOfSentenceID == config.eosTokenID
        else {
            throw RunError.artifactNotReady(
                "the V4 tokenizer controls do not match config bos/eos token ids")
        }
        guard artifact.sourceRepository == configurationIdentity.sourceRepo,
            artifact.sourceRevision == configurationIdentity.sourceRevision,
            artifact.sourceRepository == tokenizerIdentity.sourceRepo,
            artifact.sourceRevision == tokenizerIdentity.sourceRevision
        else {
            throw RunError.artifactNotReady(
                "the opened V4 units, config, and tokenizer do not share one source identity")
        }

        // The licence path is part of the tokenizer identity and the complete
        // verification pass. Opening and immediately closing it proves the
        // declared object still belongs to this rooted authority.
        let licence = try authority.openFileDescriptor(tokenizerIdentity.license)
        _ = close(licence)
        try authority.validateCurrentBinding()
        try cancellation.check()

        return try DeepSeekV4ArtifactWorkload(
            artifact: artifact, authority: authority,
            knobs: knobs, control: cancellation.executionControl)
    }

    private static func verifiedData(
        _ identity: ArtifactConfigurationIdentity,
        beneath authority: ArtifactRuntimeAuthority,
        maximumBytes: UInt64,
        cancellation: DeepSeekV4RunCancellation
    ) throws -> Data {
        try verifiedData(
            file: identity.file, expectedBytes: identity.bytes,
            expectedSHA256: identity.sha256, beneath: authority,
            maximumBytes: maximumBytes, cancellation: cancellation)
    }

    private static func verifiedData(
        _ identity: ArtifactTokenizerIdentity,
        beneath authority: ArtifactRuntimeAuthority,
        maximumBytes: UInt64,
        cancellation: DeepSeekV4RunCancellation
    ) throws -> Data {
        try verifiedData(
            file: identity.file, expectedBytes: identity.bytes,
            expectedSHA256: identity.sha256, beneath: authority,
            maximumBytes: maximumBytes, cancellation: cancellation)
    }

    private static func verifiedData(
        file: String, expectedBytes: UInt64, expectedSHA256: String,
        beneath authority: ArtifactRuntimeAuthority,
        maximumBytes: UInt64,
        cancellation: DeepSeekV4RunCancellation
    ) throws -> Data {
        try cancellation.check()
        let data = try authority.openFile(file).readAll(maximumBytes: maximumBytes)
        guard UInt64(data.count) == expectedBytes,
            SHA256.hexString(SHA256.hash(data)) == expectedSHA256
        else {
            throw RunError.artifactNotReady(
                "\(file) does not match the size and SHA-256 recorded by index.json")
        }
        try cancellation.check()
        return data
    }
}

/// The production workload owns the opaque attention state and the completely
/// admitted artifact. Both are dropped by ``shutdown()`` before the engine
/// emits a terminal event.
final class DeepSeekV4ArtifactWorkload: DeepSeekV4RunWorkload {
    private var artifact: DeepSeekV4ModelArtifact?
    private var authority: ArtifactRuntimeAuthority?
    private var generationSession: DeepSeekV4GenerationSession?
    private let knobs: DeepSeekV4EffectiveKnobs
    private let control: DeepSeekV4ExecutionControl
    private var beganExecution = false

    let layerCount: Int
    let vocabularySize: Int
    let eosTokenID: Int
    let maximumPositionCount: Int
    let sourceRepository: String
    let sourceRevision: String
    let expertPoolBudgetBytes: UInt64
    private let readAccounting: DeepSeekV4ReadAccounting
    private let phaseAccounting: DeepSeekV4PhaseAccounting
    private let configuration: DeepSeekV4Config
    /// How many routed-expert tiles one gather dispatch covers.
    ///
    /// Read from the model's own `expertsPerToken`, because at V4's geometry a
    /// tile holds one expert and a decode token selects exactly that many: the
    /// width that turns a decode gather into a single dispatch. See
    /// ``DeepSeekV4EffectiveKnobs/expertConfiguration(tilesPerDispatch:)`` for
    /// why it is not 1 and what it costs.
    private let expertGatherTilesPerDispatch: Int
    /// Captured at shutdown so the metrics survive the artifact being released,
    /// exactly as the terminal read accounting is.
    private var releasedPinnedTierMetrics: DeepSeekV4PinnedTierMetrics?
    /// Every layer's expert backend, alive for the run over one shared pool.
    /// Nil for the per-layer lifetime, which is the default and every record
    /// before this one.
    private var expertResidency: DeepSeekV4RunExpertBackends?
    /// The hash-routed layers and their token tables, built once the artifact
    /// has read them, so a pass can name their experts before it starts.
    private var hashRoutedLayers: [DeepSeekV4RunExpertBackends.HashRoutedLayer] = []

    init(
        artifact: DeepSeekV4ModelArtifact,
        authority: ArtifactRuntimeAuthority,
        knobs: DeepSeekV4EffectiveKnobs,
        control: DeepSeekV4ExecutionControl
    ) throws {
        let maximumStride = artifact.layers.flatMap {
            $0.expertUnit.containers.entries.values.map(\.layout.tileStride)
        }.max() ?? 0
        guard maximumStride > 0,
            let stride = UInt64(exactly: maximumStride)
        else {
            throw RunError.artifactNotReady(
                "the V4 artifact has no representable routed-expert tile stride")
        }
        let pool = stride.multipliedReportingOverflow(
            by: UInt64(knobs.expertPoolSlots))
        guard !pool.overflow else {
            throw RunError.artifactNotReady(
                "the V4 expert pool byte count exceeds UInt64.max")
        }

        self.artifact = artifact
        self.authority = authority
        self.knobs = knobs
        self.control = control
        self.layerCount = artifact.layers.count
        self.vocabularySize = artifact.plan.config.vocabularySize
        self.eosTokenID = artifact.plan.config.eosTokenID
        self.maximumPositionCount = artifact.plan.config.maximumPositionCount
        self.sourceRepository = artifact.sourceRepository
        self.sourceRevision = artifact.sourceRevision
        self.expertPoolBudgetBytes = pool.partialValue
        self.readAccounting = artifact.readAccounting
        self.phaseAccounting = artifact.phaseAccounting
        self.configuration = artifact.plan.config
        // Refused rather than clamped, like every other width this run states:
        // a config that routes to no experts describes no gather, and silently
        // substituting 1 would reintroduce the per-tile dispatch this width
        // exists to remove without saying so anywhere.
        guard artifact.plan.config.expertsPerToken >= 1 else {
            throw RunError.artifactNotReady(
                "the V4 config routes each token to \(artifact.plan.config.expertsPerToken) "
                    + "experts; a routed-expert gather needs at least one")
        }
        self.expertGatherTilesPerDispatch = artifact.plan.config.expertsPerToken
    }

    var readAccountingSnapshot: DeepSeekV4ReadAccountingSnapshot? {
        readAccounting.snapshot
    }

    var phaseMetricsSnapshot: DeepSeekV4PhaseMetrics? {
        phaseAccounting.snapshot
    }

    func productMemoryPlan(
        declaredBudgetBytes: UInt64,
        promptTokenCount: Int,
        maximumNewTokens: Int,
        mlxCacheBytes: UInt64,
        pinnedDeterministicBytes: UInt64,
        enforcesProductLimits: Bool
    ) throws -> DeepSeekV4ProductMemoryPlan {
        try DeepSeekV4ProductMemoryBudget.plan(
            declaredBudgetBytes: declaredBudgetBytes,
            config: configuration,
            promptTokenCount: promptTokenCount,
            maximumNewTokens: maximumNewTokens,
            expertPoolBytes: expertPoolBudgetBytes,
            mlxCacheBytes: mlxCacheBytes,
            pinnedDeterministicBytes: pinnedDeterministicBytes,
            enforcesProductLimits: enforcesProductLimits)
    }

    /// The dial's census, transcribed from the layer manifests this workload
    /// already reconciled. No payload byte is read to build it.
    var deterministicCensus: DeepSeekV4MemoryDial.Census? {
        guard let artifact else { return nil }
        return try? DeepSeekV4MemoryDial.Census(
            layerReadBytes: artifact.layers.map(\.artifact.deterministicReadBytesPerPass),
            layerResidentBytes: artifact.layers.map(\.artifact.pinnedResidentBytesPerLayer),
            // The globals rung, off the same reconciled manifest the run will
            // read the tables through — not off the config, so the plan and the
            // artifact cannot disagree about the head's size.
            globals: DeepSeekV4MemoryDialInputs.globals(artifact.global.census),
            expertBytesPerToken: 0)
    }

    func installPinnedTier(layers: Set<Int>, budgetBytes: UInt64, outputHead: Bool) {
        artifact?.installPinnedTier(
            layers: layers, budgetBytes: budgetBytes, outputHead: outputHead)
    }

    var pinnedTierMetrics: DeepSeekV4PinnedTierMetrics? {
        releasedPinnedTierMetrics ?? artifact?.pinnedTierMetrics
    }

    func prefill(
        tokenIDs: [Int], decodePositionLimit: Int,
        cancellationCheck: () throws -> Void,
        onLayerCompleted: (Int, Int) -> Void
    ) throws -> DeepSeekV4WorkloadStep {
        try beginIfNeeded(cancellationCheck: cancellationCheck)
        guard let artifact else {
            throw RunError.artifactNotReady("the V4 workload was already released")
        }
        let result = try artifact.prefillGenerationSession(
            tokenIDs: tokenIDs,
            decodePositionLimit: decodePositionLimit,
            expertConfiguration: knobs.expertConfiguration(
                tilesPerDispatch: expertGatherTilesPerDispatch),
            logitChunkRows: knobs.logitChunkRows,
            control: control,
            onLayerCompleted: onLayerCompleted)
        generationSession = result.session
        // After prefill, because that is when the hash layers' `tid2eid`
        // tables have been read and retained. Building the plan here therefore
        // re-reads nothing; before prefill it would have pulled three tables
        // off the drive that the prefill was about to pull anyway.
        if knobs.expertCrossLayerPrefetch > 0 {
            hashRoutedLayers = artifact.hashRoutedExpertPlan(
                cancellationCheck: cancellationCheck)
        }
        return DeepSeekV4WorkloadStep(
            tokenID: result.greedyTokenID, logits: result.logits,
            completedLayers: result.completedLayers)
    }

    func decode(
        tokenID: Int, cancellationCheck: () throws -> Void,
        onLayerCompleted: (Int, Int) -> Void
    ) throws -> DeepSeekV4WorkloadStep {
        try beginIfNeeded(cancellationCheck: cancellationCheck)
        guard let artifact, let generationSession else {
            throw RunError.artifactNotReady(
                "V4 decode has no admitted artifact and prefill state")
        }
        // The one point in a decode pass at which the next token id is known
        // and no layer has run: everything a hash-routed layer will read is a
        // table lookup away, and queueing it here is the whole of the
        // cross-layer dividend. Costs nothing when the window is 0.
        if let expertResidency {
            try expertResidency.beginPass(
                tokenID: tokenID, hashRoutedLayers: hashRoutedLayers)
        }
        let result = try artifact.decode(
            tokenID: tokenID,
            session: generationSession,
            expertConfiguration: knobs.expertConfiguration(
                tilesPerDispatch: expertGatherTilesPerDispatch),
            logitChunkRows: knobs.logitChunkRows,
            control: control,
            onLayerCompleted: onLayerCompleted)
        return DeepSeekV4WorkloadStep(
            tokenID: result.greedyTokenID, logits: result.logits,
            completedLayers: result.completedLayers)
    }

    func validateTerminalBinding() throws {
        guard let authority else {
            throw RunError.artifactNotReady("the V4 runtime authority was released early")
        }
        try authority.validateAllFilesCurrent()
    }

    func shutdown() {
        control.cancel(reason: "V4 workload teardown")
        generationSession?.release()
        generationSession = nil
        // The pool and 129 readers are the largest thing a run-scoped
        // residency holds; a run that finished must not still hold them.
        control.installExpertResidency(nil)
        expertResidency?.shutdown()
        expertResidency = nil
        hashRoutedLayers = []
        // Pinned weights are the largest allocation a stated run holds, and a
        // run that finished should not still be holding them. The metrics are
        // captured first; the arrays do not survive.
        releasedPinnedTierMetrics = artifact?.pinnedTierMetrics
        artifact?.releasePinnedTier()
        artifact = nil
        authority = nil
        Self.reclaimTransientMemory()
    }

    private func beginIfNeeded(cancellationCheck: () throws -> Void) throws {
        try cancellationCheck()
        try control.throwIfCancelled()
        if !beganExecution {
            MLX.Memory.cacheLimit = knobs.mlxCacheLimitBytes
            Self.reclaimTransientMemory()
            try installExpertResidencyIfStated()
            beganExecution = true
        }
    }

    /// Build the run's one expert residency, if the run asked for it.
    ///
    /// Built here rather than in `init` for one reason that matters: the pool
    /// it allocates is the stated `expertPoolBudgetBytes`, and that term has to
    /// be *reserved before the memory dial plans* — which has already happened
    /// by the time the first pass runs and has not by the time the workload is
    /// constructed. Same bytes either way; this ordering is the one where the
    /// dial never sees a pool it did not price.
    private func installExpertResidencyIfStated() throws {
        guard knobs.expertRunScopedBackends, expertResidency == nil,
            let artifact
        else { return }
        let residency = try artifact.makeRunScopedExpertBackends(
            configuration: knobs.expertConfiguration(
                tilesPerDispatch: expertGatherTilesPerDispatch))
        // The one accounting check that matters: a residency that shared no
        // pool would have multiplied a reserved term by the layer count.
        guard UInt64(residency.poolBudgetBytes) == expertPoolBudgetBytes else {
            residency.shutdown()
            throw RunError.artifactNotReady(
                "the run-scoped expert residency reserved "
                    + "\(residency.poolBudgetBytes) B against the "
                    + "\(expertPoolBudgetBytes) B the memory plan priced")
        }
        expertResidency = residency
        control.installExpertResidency(residency)
    }

    private static func reclaimTransientMemory() {
        MLX.Memory.clearCache()
        _ = malloc_zone_pressure_relief(nil, 0)
    }
}

final class DeepSeekV4RunEngine {
    private struct FootprintPeak {
        private let lifetimeBaseline: UInt64?
        private(set) var bytes: UInt64 = 0

        init(lifetimeBaseline: UInt64?) {
            self.lifetimeBaseline = lifetimeBaseline
        }

        mutating func observe(current: UInt64, lifetime: UInt64) -> UInt64 {
            bytes = max(bytes, current)
            if let lifetimeBaseline, lifetime > lifetimeBaseline {
                bytes = max(bytes, lifetime)
            }
            return bytes
        }
    }

    struct TeardownRecord: Codable, Sendable, Equatable {
        var workloadWasPrepared = false
        var workloadReleased = false
        var scopeWasHeld = false
        var scopeReleased = false
        var mlxCacheBytesAfterClear: UInt64 = 0
    }

    private enum Outcome: String { case finished, cancelled }

    private let request: RunRequest
    private let capabilities: RunnerCapabilities
    private let knobs: DeepSeekV4EffectiveKnobs
    private let enforcesProductMemoryPolicy: Bool
    private let handle: RunHandleID
    private let cancellation: DeepSeekV4RunCancellation
    private let factory: any DeepSeekV4RunWorkloadFactory
    private let continuation: AsyncThrowingStream<RunEvent, Error>.Continuation

    private let started = Date()
    private let startedTick = MonotonicClock.now()
    private var phaseName = "starting"
    private var generationStage: RunGenerationStage = .preparing
    private var thermalTrail: [ThermalTransition] = []
    private var footprintPeak: FootprintPeak
    private var tokenIDs: [Int] = []
    private var passSeconds: [Double] = []
    private var lastLogits: [Float] = []
    /// The opt-in per-position logits sink, or nil — which is what it is unless
    /// this process was started with `MINIRUN_V4_DUMP_LOGITS`.
    private let logitsDump = DeepSeekV4LogitsDump.shared
    private var workload: (any DeepSeekV4RunWorkload)?
    private var environment: EnvironmentReport?
    private var sourceRepository = ""
    private var sourceRevision = ""
    private var productMemoryPlan: DeepSeekV4ProductMemoryPlan?
    /// The residency plan the stated budget implied. Nil at the product scale,
    /// which pins nothing.
    private var pinPlan: PinPlan?
    private var terminalPinnedTierMetrics: DeepSeekV4PinnedTierMetrics?
    private var completedTokenBytes: ByteAccounting?
    private var prefillBytes: ByteAccounting?
    /// Cumulative phase brackets as they stood when the current pass began.
    /// A pass is the difference against this, which is what keeps prefill and
    /// each decode pass separate without a second instrument.
    private var phaseBoundary: DeepSeekV4PhaseMetrics?
    private var prefillPhaseMetrics: DeepSeekV4PhaseMetrics?
    private var decodePhaseMetrics: [DeepSeekV4PhaseMetrics] = []
    /// Pinned-tier bytes as they stood when the current pass began, for the
    /// same reason `phaseBoundary` exists: the counter is run-scoped, and what
    /// a reader of a pinned arm needs is what a *later* pass was served, not
    /// what the whole run was — the filling pass and every pass after it are
    /// different measurements.
    private var pinnedServedBoundary: UInt64?
    private var prefillPinnedServedBytes: UInt64?
    private var decodePinnedServedBytes: [UInt64] = []
    private var terminalReadAccounting: DeepSeekV4ReadAccountingSnapshot?
    private var teardownRecord: TeardownRecord?

    init(
        request: RunRequest, capabilities: RunnerCapabilities,
        knobs: DeepSeekV4EffectiveKnobs,
        enforcesProductMemoryPolicy: Bool,
        handle: RunHandleID,
        cancellation: DeepSeekV4RunCancellation,
        factory: any DeepSeekV4RunWorkloadFactory,
        continuation: AsyncThrowingStream<RunEvent, Error>.Continuation
    ) {
        self.request = request
        self.capabilities = capabilities
        self.knobs = knobs
        self.enforcesProductMemoryPolicy = enforcesProductMemoryPolicy
        self.handle = handle
        self.cancellation = cancellation
        self.factory = factory
        self.continuation = continuation
        footprintPeak = FootprintPeak(
            lifetimeBaseline: ProcessFootprint.current()?.peakFootprintBytes)
    }

    func run() {
        recordThermal(.unknown, at: 0, phase: phaseName)
        do {
            try execute()
            let teardown = tearDown()
            continuation.yield(
                .finished(summary(outcome: .finished, teardown: teardown)))
            continuation.finish()
        } catch {
            let overBudget = cancellation.budgetExceededPeak
            let artifactFailure = cancellation.artifactFailure
            let wasCancelled = Self.isCancellation(error) || cancellation.isCancelled
            let teardown = tearDown()
            if let peak = overBudget {
                continuation.finish(
                    throwing: RunError.budgetExceeded(
                        peakBytes: peak, declaredBytes: request.memoryBudgetBytes))
            } else if let artifactFailure {
                continuation.finish(
                    throwing: RunError.artifactNotReady(artifactFailure))
            } else if wasCancelled {
                continuation.yield(
                    .cancelled(summary(outcome: .cancelled, teardown: teardown)))
                continuation.finish()
            } else {
                continuation.finish(throwing: error)
            }
        }
    }

    private func execute() throws {
        try cancellation.check()
        environment = EnvironmentReport.current(path: request.artifact.root.path)
        let prepared = try factory.prepare(
            request: request, knobs: knobs, cancellation: cancellation)
        workload = prepared
        sourceRepository = prepared.sourceRepository
        sourceRevision = prepared.sourceRevision

        guard prepared.layerCount > 0, prepared.vocabularySize > 0 else {
            throw RunError.artifactNotReady(
                "the admitted V4 workload has no layers or vocabulary")
        }
        guard let cacheBytes = UInt64(exactly: knobs.mlxCacheLimitBytes) else {
            throw RunError.knobOutOfRange(
                name: "mlxCacheLimitBytes", value: "\(knobs.mlxCacheLimitBytes)",
                allowed: "a non-negative UInt64 byte count")
        }
        let explicitlyReserved = prepared.expertPoolBudgetBytes.addingReportingOverflow(
            cacheBytes)
        guard !explicitlyReserved.overflow else {
            throw RunError.artifactNotReady(
                "the V4 expert pool and MLX cache reservation exceed UInt64.max")
        }
        guard case .tokenIDs(let promptTokenIDs) = request.prompt else {
            throw RunError.promptNotSupported(
                reason: "the V4 runner accepts only ids from its verified tokenizer")
        }
        guard promptTokenIDs.allSatisfy({
            $0 >= 0 && $0 < prepared.vocabularySize
        }) else {
            throw RunError.promptNotSupported(
                reason: "the prompt contains an id outside the verified V4 vocabulary")
        }
        let extraPositions = request.maximumNewTokens - 1
        let positionLimit = promptTokenIDs.count.addingReportingOverflow(extraPositions)
        guard !positionLimit.overflow,
            positionLimit.partialValue <= prepared.maximumPositionCount
        else {
            throw RunError.promptNotSupported(
                reason: "the prompt and requested response exceed the verified V4 position limit")
        }
        let requiredBudget: UInt64
        if enforcesProductMemoryPolicy {
            // The product scale pins nothing, so the eighth term is zero and
            // this plan is byte-identical to the one the 2 GB floor was
            // accepted against.
            let plan = try prepared.productMemoryPlan(
                declaredBudgetBytes: request.memoryBudgetBytes,
                promptTokenCount: promptTokenIDs.count,
                maximumNewTokens: request.maximumNewTokens,
                mlxCacheBytes: cacheBytes,
                pinnedDeterministicBytes: 0,
                enforcesProductLimits: true)
            productMemoryPlan = plan
            requiredBudget = plan.requiredBudgetBytes
        } else {
            let planned = try planPinnedTier(
                workload: prepared, promptTokenCount: promptTokenIDs.count,
                cacheBytes: cacheBytes)
            pinPlan = planned
            // A stated run funds what it reserves plus what it pins, and
            // nothing else: the 2 GB product floor is the product scale's
            // policy and a stated harness declares its own. Refused by name
            // below when the declared budget cannot cover the sum — never
            // clamped to a smaller pin set.
            let withPins = explicitlyReserved.partialValue.addingReportingOverflow(
                planned?.pinnedBytes ?? 0)
            guard !withPins.overflow else {
                throw RunError.artifactNotReady(
                    "the V4 reservation and pinned tier exceed UInt64.max")
            }
            requiredBudget = withPins.partialValue
        }
        guard requiredBudget <= request.memoryBudgetBytes else {
            throw RunError.budgetBelowMinimum(
                declared: request.memoryBudgetBytes,
                minimum: requiredBudget,
                model: capabilities.model)
        }
        try cancellation.check()

        if let pinPlan, pinPlan.pinnedBytes > 0 {
            prepared.installPinnedTier(
                layers: Set(
                    pinPlan.pinnedLayers.compactMap { decision in
                        guard case .deterministicLayer(let layer) = decision.unit else {
                            return nil
                        }
                        return layer
                    }),
                budgetBytes: pinPlan.pinnedBytes,
                outputHead: pinPlan.pinnedGlobals?.unit == .outputHead)
        }

        continuation.yield(
            .accepted(
                RunAcceptance(
                    handle: handle, model: capabilities.model,
                    declaredBudgetBytes: request.memoryBudgetBytes,
                    pinPlan: pinPlan, effectiveKnobs: knobs.asRunKnobs,
                    artifactRoot: request.artifact.root.path,
                    startedAt: started)))
        if let pinPlan {
            // The globals rung is named where it is pinned. "43 of 43 layers"
            // and "43 of 43 layers + output head" are different residencies and
            // different per-token reads, and the log is the only place a run
            // record says which one this pass had.
            let head = pinPlan.pinnedGlobals?.unit == .outputHead ? " + output head" : ""
            continuation.yield(
                .log(
                    "V4 memory dial: \(pinPlan.pinnedLayers.count) of "
                        + "\(pinPlan.layerCount ?? 0) layers\(head) pinned, "
                        + ByteSize.format(pinPlan.pinnedBytes) + " resident, "
                        + ByteSize.format(pinPlan.projectedBytesPerTokenSaved)
                        + " saved per later pass"))
        }
        continuation.yield(
            .log(
                "V4 source \(sourceRepository)@\(sourceRevision); "
                    + "\(prepared.layerCount) layers; expert pool "
                    + ByteSize.format(prepared.expertPoolBudgetBytes)))
        if let plan = productMemoryPlan {
            continuation.yield(
                .log(
                    "V4 bounded memory plan: retained state "
                        + ByteSize.format(plan.retainedStateBytes)
                        + ", one-layer replacement "
                        + ByteSize.format(plan.replacementLayerStateBytes)
                        + ", headroom " + ByteSize.format(plan.headroomBytes)))
        }

        phaseName = "prefill"
        generationStage = .prefill
        continuation.yield(
            .phase(
                RunPhase(
                    name: phaseName, generationStage: generationStage,
                    fraction: 0, detail: "reading prompt")))
        phaseBoundary = prepared.phaseMetricsSnapshot
        pinnedServedBoundary = pinnedServedReading()
        let prefillStarted = MonotonicClock.now()
        let first = try autoreleasepool {
            try prepared.prefill(
                tokenIDs: promptTokenIDs,
                decodePositionLimit: positionLimit.partialValue,
                cancellationCheck: cancellation.check,
                onLayerCompleted: observeLayer)
        }
        try record(
            first, seconds: MonotonicClock.seconds(since: prefillStarted), prefill: true)

        if first.tokenID != prepared.eosTokenID, request.maximumNewTokens > 1 {
            for index in 1..<request.maximumNewTokens {
                try cancellation.check()
                phaseName = "decode token \(index + 1)"
                generationStage = .decode
                continuation.yield(
                    .phase(
                        RunPhase(
                            name: phaseName, generationStage: generationStage,
                            fraction: 0, detail: "reading layers")))
                phaseBoundary = prepared.phaseMetricsSnapshot
                pinnedServedBoundary = pinnedServedReading()
                let passStarted = MonotonicClock.now()
                let next = try autoreleasepool {
                    try prepared.decode(
                        tokenID: tokenIDs.last!,
                        cancellationCheck: cancellation.check,
                        onLayerCompleted: observeLayer)
                }
                try record(
                    next, seconds: MonotonicClock.seconds(since: passStarted),
                    prefill: false)
                if next.tokenID == prepared.eosTokenID { break }
            }
        }

        try cancellation.check()
        try prepared.validateTerminalBinding()
        try cancellation.check()
    }

    /// Turn a stated budget into a residency plan, or refuse it by name.
    ///
    /// Only the `.stated` scale reaches this. The product scale is the 2 GB
    /// floor, whose whole case is that it does not depend on anything the
    /// conversation chose, and it pins nothing.
    ///
    /// The dial's floor is priced from the product plan's own terms — the
    /// transient execution envelope, the retained generation state, the
    /// one-layer replacement, the routed-expert pool and the MLX cache — built
    /// with an unbounded declared budget so it prices the run rather than
    /// admitting it. The 2 GB product *floor* is deliberately not part of it:
    /// that is the product scale's policy, and a stated harness declares its
    /// own. What the stated run is then admitted against is its reservation
    /// plus the pinned residency, in `execute()`.
    ///
    /// A budget that clears the product floor but not the dial's floor pins
    /// nothing and runs exactly as it did before the dial existed. A refusal to
    /// pin is not a refusal to run; a budget that cannot fund the pins it asked
    /// for *is* refused, by name, in `execute()`.
    private func planPinnedTier(
        workload: any DeepSeekV4RunWorkload, promptTokenCount: Int, cacheBytes: UInt64
    ) throws -> PinPlan? {
        guard let census = workload.deterministicCensus else { return nil }
        let priced = try workload.productMemoryPlan(
            declaredBudgetBytes: .max,
            promptTokenCount: promptTokenCount,
            maximumNewTokens: request.maximumNewTokens,
            mlxCacheBytes: cacheBytes,
            pinnedDeterministicBytes: 0,
            enforcesProductLimits: false)

        // The floor comes from the same shared derivation the memory dial's
        // screen prices its ladder with, so a preset and the run it starts
        // cannot disagree about what the artifact costs before a byte is
        // pinned.
        guard let floor = DeepSeekV4MemoryDialInputs.floor(pricing: priced) else { return nil }
        guard request.memoryBudgetBytes >= floor.totalBytes else { return nil }

        return try DeepSeekV4MemoryDial.plan(
            budgetBytes: request.memoryBudgetBytes,
            census: census,
            floor: floor,
            maximumNewTokens: request.maximumNewTokens)
    }

    static let statedPinMarginBytes = DeepSeekV4MemoryDialInputs.statedPinMarginBytes

    private func record(
        _ step: DeepSeekV4WorkloadStep, seconds: Double, prefill: Bool
    ) throws {
        guard step.completedLayers == workload?.layerCount,
            step.tokenID >= 0,
            step.tokenID < (workload?.vocabularySize ?? 0),
            !step.logits.isEmpty,
            step.logits.count == workload?.vocabularySize,
            step.logits.allSatisfy(\.isFinite)
        else {
            throw RunError.artifactNotReady(
                "the V4 generation step returned incomplete layers, token, or logits")
        }
        passSeconds.append(seconds)
        completedTokenBytes = byteAccounting()
        if prefill { prefillBytes = completedTokenBytes }
        recordPhaseSplit(seconds: seconds, prefill: prefill)
        recordPinnedServedSplit(prefill: prefill)
        tokenIDs.append(step.tokenID)
        lastLogits = step.logits
        // Off unless `MINIRUN_V4_DUMP_LOGITS` named a directory. The run
        // summary carries one digest of the last pass; an arm that has to say
        // *how far apart* two paths' logits are needs every position's vector,
        // and this is the only place in the run that holds one.
        logitsDump?.record(
            position: tokenIDs.count - 1, logits: step.logits, tokenID: step.tokenID,
            isPrefillToken: prefill)
        continuation.yield(
            .token(
                TokenEvent(
                    index: tokenIDs.count - 1, tokenID: step.tokenID,
                    text: nil, secondsSincePreviousToken: seconds,
                    isPrefillToken: prefill)))
        let telemetry = sample(phase: phaseName)
        recordThermal(
            telemetry.thermalState, at: telemetry.elapsed, phase: telemetry.phase)
        continuation.yield(.telemetry(telemetry))
        enforceBudget(telemetry)
        try cancellation.check()
    }

    /// Attribute one completed pass, or say nothing.
    ///
    /// The workload's brackets are cumulative, so the pass is the difference
    /// between the boundary taken before it started and the boundary now. The
    /// subtraction fails closed — a term that moved backwards, a saturated
    /// total, or terms that do not fit inside the measured wall all mean the
    /// two samples cannot be compared, and a decomposition that cannot be
    /// trusted is worse than none.
    private func recordPhaseSplit(seconds: Double, prefill: Bool) {
        guard let earlier = phaseBoundary,
            let current = workload?.phaseMetricsSnapshot,
            let split = current.subtracting(earlier)?.withPassSeconds(seconds),
            split.isAccountingBalanced
        else {
            phaseBoundary = nil
            return
        }
        phaseBoundary = nil
        if prefill {
            prefillPhaseMetrics = split
        } else {
            decodePhaseMetrics.append(split)
        }
        // The same measurement twice, for two audiences: the log line a run
        // record is read from, and the structured summary the product panel
        // draws and the conversation persists.
        if let summary = split.runPhaseSummary(
            passKind: prefill ? .prefill : .decode,
            passIndex: prefill ? 0 : decodePhaseMetrics.count)
        {
            continuation.yield(.phaseSummary(summary))
        }
        continuation.yield(.log(split.summaryLine))
    }

    /// What the memory dial's pinned tier served this pass, or nothing.
    ///
    /// The counter is cumulative and monotonic, so a pass is the difference of
    /// two readings, exactly as the phase split is. A reading that moved
    /// backwards, a saturated counter, or a workload that observes no bytes at
    /// all all mean the two cannot be compared, and the pass is left out rather
    /// than recorded as a zero — a zero here reads as "the tier served this
    /// pass nothing", which is a measurement, not a missing one.
    private func recordPinnedServedSplit(prefill: Bool) {
        let earlier = pinnedServedBoundary
        pinnedServedBoundary = nil
        guard let earlier, let current = pinnedServedReading(), current >= earlier
        else { return }
        if prefill {
            prefillPinnedServedBytes = current - earlier
        } else {
            decodePinnedServedBytes.append(current - earlier)
        }
    }

    private func pinnedServedReading() -> UInt64? {
        guard let snapshot = workload?.readAccountingSnapshot, !snapshot.didOverflow
        else { return nil }
        return snapshot.pinnedServedBytes
    }

    private func observeLayer(completed: Int, total: Int) {
        guard completed >= 1, total >= 1, completed <= total else {
            cancellation.failArtifact(
                "the V4 workload reported an invalid layer progress boundary")
            return
        }
        let telemetry = sample(phase: phaseName)
        recordThermal(
            telemetry.thermalState, at: telemetry.elapsed, phase: telemetry.phase)
        continuation.yield(
            .phase(
                RunPhase(
                    name: phaseName, generationStage: generationStage,
                    fraction: Double(completed) / Double(total),
                    detail: "layer \(completed)/\(total)")))
        continuation.yield(.telemetry(telemetry))
        enforceBudget(telemetry)
    }

    private func enforceBudget(_ telemetry: RunTelemetry) {
        if telemetry.peakFootprintBytes > request.memoryBudgetBytes {
            cancellation.exceedBudget(peakBytes: telemetry.peakFootprintBytes)
        }
    }

    private func sample(
        phase: String, stage: RunGenerationStage? = nil
    ) -> RunTelemetry {
        let harness = HarnessTelemetry.sample(
            declaredBudgetBytes: request.memoryBudgetBytes)
        let peak = footprintPeak.observe(
            current: harness.footprintBytes,
            lifetime: harness.peakFootprintBytes)
        // The boundary reclaim policy decides from the same footprint this
        // sample reports, so the record contains the observation the decision
        // was made from. The unconditional product policy ignores it.
        cancellation.executionControl.boundaryReclaim.observeFootprint(
            harness.footprintBytes)
        let elapsed = MonotonicClock.seconds(since: startedTick)
        let prefillSeconds = passSeconds.first
        let decodeSeconds = passSeconds.dropFirst().reduce(0, +)
        let decodeTokens = max(0, tokenIDs.count - 1)
        let observed = byteAccounting()
        let accounting = observed ?? ByteAccounting(totalBytesRead: 0)
        let completed = completedTokenBytes
        let decodeAccounting = completed.flatMap { completed in
            prefillBytes.flatMap { completed.subtracting($0) }
        }
        return RunTelemetry(
            at: Date(), elapsed: elapsed, phase: phase,
            tokensPerSecond: decodeTokens > 0 && decodeSeconds > 0
                ? Double(decodeTokens) / decodeSeconds : nil,
            generationStage: stage ?? generationStage,
            bytes: accounting,
            bytesPerSecond: observed != nil && elapsed > 0
                ? Double(accounting.totalBytesRead) / elapsed : nil,
            completedTokenBytes: completed,
            prefillBytes: prefillBytes,
            bytesPerToken: decodeTokens > 0
                ? decodeAccounting.map { $0.totalBytesRead / UInt64(decodeTokens) } : nil,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            decodeTokensCompleted: decodeTokens,
            byteAccountingReported: observed != nil,
            declaredBudgetBytes: request.memoryBudgetBytes,
            footprintBytes: harness.footprintBytes,
            peakFootprintBytes: peak,
            residentBytes: harness.residentBytes,
            availableBytes: harness.availableBytes,
            mlxActiveBytes: UInt64(max(0, MLX.Memory.activeMemory)),
            mlxCacheBytes: UInt64(max(0, MLX.Memory.cacheMemory)),
            mlxPeakBytes: UInt64(max(0, MLX.Memory.peakMemory)),
            thermalState: ThermalStateName(name: harness.thermalState),
            lowPowerMode: harness.lowPowerModeEnabled,
            batteryLevel: Self.batteryLevel(),
            instrumentation: nil)
    }

    @discardableResult
    private func tearDown() -> TeardownRecord {
        if let teardownRecord { return teardownRecord }
        var record = TeardownRecord()
        record.workloadWasPrepared = workload != nil
        terminalReadAccounting = workload?.readAccountingSnapshot
        terminalPinnedTierMetrics = workload?.pinnedTierMetrics
        workload?.shutdown()
        workload = nil
        record.workloadReleased = true
        // The workload has closed both deterministic files and every expert
        // reader, so no payload callback can race this final partial window.
        cancellation.payloadFlow.finish()

        if let scope = request.artifact.scope {
            record.scopeWasHeld = true
            scope.release()
            record.scopeReleased = scope.isReleased
        }
        Self.reclaimTransientMemory()
        record.mlxCacheBytesAfterClear = UInt64(max(0, MLX.Memory.cacheMemory))
        teardownRecord = record
        return record
    }

    private func summary(
        outcome: Outcome, teardown: TeardownRecord
    ) -> RunSummary {
        let telemetry = sample(phase: outcome.rawValue, stage: .terminal)
        let wall = MonotonicClock.seconds(since: startedTick)
        let prefillSeconds = passSeconds.first
        let decodeSeconds = passSeconds.dropFirst().reduce(0, +)
        let decodeTokens = max(0, tokenIDs.count - 1)
        let rate = decodeTokens > 0 && decodeSeconds > 0
            ? Double(decodeTokens) / decodeSeconds : nil
        let digest = lastLogits.isEmpty ? nil : Self.logitsDigest(lastLogits)
        let publication = request.artifact.runtimeAuthority?.evidence.repository
        let payload = Payload(
            runner: "deepseek-v4-decode",
            handle: handle.rawValue.uuidString,
            revision: BuildInfo.gitRevision,
            outcome: outcome.rawValue,
            environment: environment
                ?? EnvironmentReport.current(path: request.artifact.root.path),
            artifactRoot: request.artifact.root.path,
            publicationRepository: publication?.repoID ?? "",
            publicationRevision: publication?.revision ?? "",
            sourceRepository: sourceRepository,
            sourceRevision: sourceRevision,
            promptTokenIDs: {
                if case .tokenIDs(let ids) = request.prompt { return ids }
                return []
            }(),
            requestedNewTokens: request.maximumNewTokens,
            generatedTokenIDs: tokenIDs,
            logitsSHA256: digest,
            wallSeconds: wall,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            decodeTokensCompleted: decodeTokens,
            passSeconds: passSeconds,
            prefillPhaseMetrics: prefillPhaseMetrics,
            decodePhaseMetrics: decodePhaseMetrics,
            prefillPinnedServedBytes: prefillPinnedServedBytes,
            decodePinnedServedBytes: decodePinnedServedBytes,
            tokensPerSecond: rate,
            byteAccountingReported: telemetry.reportsByteAccounting,
            declaredBudgetBytes: request.memoryBudgetBytes,
            peakFootprintBytes: telemetry.peakFootprintBytes,
            budgetRespected: telemetry.budgetRespected,
            telemetry: telemetry,
            thermalTrail: thermalTrail,
            configuration: knobs.asStrings,
            productMemoryPlan: productMemoryPlan,
            pinPlan: pinPlan,
            pinnedTier: terminalPinnedTierMetrics ?? workload?.pinnedTierMetrics,
            teardown: teardown)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        let headline: String
        switch outcome {
        case .finished:
            headline = String(
                format: "%d V4 token%@ in %.1f s, peak %@",
                tokenIDs.count, tokenIDs.count == 1 ? "" : "s", wall,
                ByteSize.format(telemetry.peakFootprintBytes))
        case .cancelled:
            headline = String(
                format: "V4 stopped after %d of %d token%@, %.1f s, peak %@",
                tokenIDs.count, request.maximumNewTokens,
                request.maximumNewTokens == 1 ? "" : "s", wall,
                ByteSize.format(telemetry.peakFootprintBytes))
        }
        var lines = [
            "source \(sourceRepository)@\(sourceRevision)",
            "thermal " + thermalTrail.map(\.state.rawValue).joined(separator: " -> "),
            "teardown: workload released before terminal event",
        ]
        if telemetry.reportsByteAccounting {
            lines.insert(
                "payload read \(ByteSize.format(telemetry.bytes.totalBytesRead)); "
                    + "deterministic \(ByteSize.format(telemetry.bytes.deterministicBytesRead ?? 0)); "
                    + "expert \(ByteSize.format(telemetry.bytes.expertBytesRead ?? 0))",
                at: 1)
        } else {
            lines.insert(
                "payload byte flow unavailable: observer missing or accounting overflowed",
                at: 1)
        }
        if let last = decodePhaseMetrics.last ?? prefillPhaseMetrics {
            lines.insert(last.summaryLine, at: 1)
        }
        if let tier = terminalPinnedTierMetrics, tier.pinnedLayerCount > 0 {
            lines.insert(tier.summaryLine, at: 1)
        }
        if let digest { lines.insert("logits sha256 \(digest)", at: 1) }
        if let plan = productMemoryPlan {
            lines.insert(
                "bounded V4 state: one retained generation + one layer replacement; "
                    + "headroom \(ByteSize.format(plan.headroomBytes))",
                at: min(2, lines.count))
        }

        return RunSummary(
            handle: handle, model: capabilities.model,
            tokenIDs: tokenIDs, text: nil,
            wallSeconds: wall, tokensPerSecond: rate,
            finalTelemetry: telemetry,
            peakFootprintBytes: telemetry.peakFootprintBytes,
            budgetRespected: telemetry.budgetRespected,
            thermalTrail: thermalTrail,
            logitsDigest: tokenIDs.indices.last.flatMap { index in
                digest.map { RunLogitsDigest(tokenIndex: index, hex: $0) }
            },
            resultJSON: try? encoder.encode(payload),
            resultJSONFilename: "deepseek-v4-decode-run.json",
            headline: headline, lines: lines,
            gitRevision: BuildInfo.gitRevision)
    }

    private func byteAccounting() -> ByteAccounting? {
        guard let snapshot = terminalReadAccounting ?? workload?.readAccountingSnapshot,
            snapshot.isBalanced
        else { return nil }
        return ByteAccounting(
            totalBytesRead: snapshot.totalBytesRead,
            deterministicBytesRead: snapshot.deterministicBytesRead,
            expertBytesRead: snapshot.expertBytesRead)
    }

    private func recordThermal(
        _ state: ThermalStateName, at seconds: Double, phase: String
    ) {
        let resolved = state == .unknown
            ? ThermalStateName(name: ThermalState.currentName) : state
        guard thermalTrail.last?.state != resolved else { return }
        thermalTrail.append(
            ThermalTransition(state: resolved, atSeconds: seconds, phase: phase))
    }

    /// The published gate's digest: SHA-256 over the float32 bit patterns,
    /// little-endian, in vocabulary order. Shares its byte packing with
    /// ``DeepSeekV4LogitsDump/rawBytes(_:)`` so that a dumped vector and the
    /// digest of the same pass cannot describe different bytes.
    static func logitsDigest(_ logits: [Float]) -> String {
        SHA256.hexString(SHA256.hash(DeepSeekV4LogitsDump.rawBytes(logits)))
    }

    private static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        return (error as? RunError) == .cancelled
    }

    private static func reclaimTransientMemory() {
        MLX.Memory.clearCache()
        _ = malloc_zone_pressure_relief(nil, 0)
    }

    private static func batteryLevel() -> Float? {
        #if canImport(UIKit)
            let level = UIDevice.current.batteryLevel
            return level < 0 ? nil : level
        #else
            return nil
        #endif
    }

    private struct Payload: Codable {
        let runner: String
        let handle: String
        let revision: String
        let outcome: String
        let environment: EnvironmentReport
        let artifactRoot: String
        let publicationRepository: String
        let publicationRevision: String
        let sourceRepository: String
        let sourceRevision: String
        let promptTokenIDs: [Int]
        let requestedNewTokens: Int
        let generatedTokenIDs: [Int]
        let logitsSHA256: String?
        let wallSeconds: Double
        let prefillSeconds: Double?
        let decodeSeconds: Double
        let decodeTokensCompleted: Int
        let passSeconds: [Double]
        /// Where each pass's wall time went. Prefill and decode are different
        /// workloads and are reported apart, exactly as their seconds and
        /// bytes already are.
        let prefillPhaseMetrics: DeepSeekV4PhaseMetrics?
        let decodePhaseMetrics: [DeepSeekV4PhaseMetrics]
        /// Bytes the pinned tier served each pass, beside the seconds each pass
        /// took. `pinnedServedCount` in the phase metrics says how many loads
        /// that was; this says how many bytes those loads did not read. Zero at
        /// the product scale, which pins nothing.
        let prefillPinnedServedBytes: UInt64?
        let decodePinnedServedBytes: [UInt64]
        let tokensPerSecond: Double?
        let byteAccountingReported: Bool
        let declaredBudgetBytes: UInt64
        let peakFootprintBytes: UInt64
        let budgetRespected: Bool
        let telemetry: RunTelemetry
        let thermalTrail: [ThermalTransition]
        let configuration: [String: String]
        let productMemoryPlan: DeepSeekV4ProductMemoryPlan?
        /// What the dial decided, beside what the tier then did. A timing run
        /// records the plan that produced it (AGENTS.md reproducibility).
        let pinPlan: PinPlan?
        let pinnedTier: DeepSeekV4PinnedTierMetrics?
        let teardown: TeardownRecord
    }
}
