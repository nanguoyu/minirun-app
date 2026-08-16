import BenchScenarios
import Foundation
import MLX
import MinirunKit
import ModelAdapters
import StorageCore

#if canImport(UIKit)
    import UIKit
#endif

/// K3 response framing ends before the model starts emitting the chat
/// protocol's closing tags. The runner owns this boundary because only it can
/// stop another expensive pass; the product tokenizer independently hides the
/// same terminal marker from presented text.
enum K3GenerationBoundary {
    static let flagshipVocabularySize = 163_840

    static func endsResponse(tokenID: Int, vocabularySize: Int) -> Bool {
        guard vocabularySize == flagshipVocabularySize else { return false }
        let base = vocabularySize - TikTokenVocabulary.reservedSpecialTokens
        return tokenID == base + K3SpecialToken.endOfSequence.rawValue
            || tokenID == base + K3SpecialToken.endOfMessage.rawValue
            || tokenID == base + K3SpecialToken.close.rawValue
    }
}

/// One K3 run, from a resolved artifact to a terminal event.
///
/// Lives on its own thread for the whole run and is the only thing that touches
/// the source, the backend and the model. Three properties it has to keep, and
/// which the tests check rather than assume:
///
/// 1. **Every byte lands in exactly one bucket.** The deterministic stream and
///    the expert pager each count their own, and the total is their sum — not a
///    third counter that could drift from them.
/// 2. **The terminal event follows teardown.** A consumer that sees
///    `.finished` or `.cancelled` knows the readers are closed, the staged
///    bytes freed and the scope dropped, because the teardown record it carries
///    was written by the teardown itself.
/// 3. **Cancellation lands inside a forward pass.** `K3Model.onLayerCompleted`
///    may throw, and it unwinds through the model's own
///    `defer { source.releaseLayer(index) }`, so a stopped run cannot leak the
///    resident layer.
final class K3RunEngine {

    /// A process-lifetime high-water mark cannot describe one RunSession. A
    /// baseline taken before this run lets a newly raised kernel high-water
    /// mark capture a peak between layer callbacks without inheriting an older
    /// run's peak. Current samples remain the authority when the lifetime peak
    /// does not move (or no baseline was available).
    struct FootprintPeak {
        private let baselineProcessLifetimeBytes: UInt64?
        private(set) var bytes: UInt64 = 0

        init(processLifetimePeakBytes: UInt64?) {
            baselineProcessLifetimeBytes = processLifetimePeakBytes
        }

        @discardableResult
        mutating func observe(
            currentBytes: UInt64, processLifetimePeakBytes: UInt64
        ) -> UInt64 {
            bytes = max(bytes, currentBytes)
            if let baselineProcessLifetimeBytes,
                processLifetimePeakBytes > baselineProcessLifetimeBytes
            {
                bytes = max(bytes, processLifetimePeakBytes)
            }
            return bytes
        }
    }

    private let request: RunRequest
    private let capabilities: RunnerCapabilities
    private let knobs: K3EffectiveKnobs
    private let handle: RunHandleID
    private let latch: RunCancellationLatch
    private let continuation: AsyncThrowingStream<RunEvent, Error>.Continuation
    private let payloadFlow: RunPayloadFlowMeter
    private let usesProductMemoryBudget: Bool

    private let started = Date()
    private let startedTick = MonotonicClock.now()
    private var phaseName = "starting"
    private var generationStage: RunGenerationStage = .preparing
    private var thermalTrail: [ThermalTransition] = []
    private var footprintPeak: FootprintPeak
    private var tokenIDs: [Int] = []
    /// Exact counters at the most recent completed-token boundary. Progressive
    /// counters continue moving during the following pass; this snapshot keeps
    /// their partial work out of the completed-token average.
    private var completedTokenBytes: ByteAccounting?
    private var prefillBytes: ByteAccounting?
    private var passSeconds: [Double] = []
    /// Where each completed pass's wall time went, in the order the passes ran.
    /// A pass whose attribution could not be trusted is absent rather than
    /// present as zeros, so this may be shorter than `passSeconds`.
    private var phaseSummaries: [RunPhaseSummary] = []
    private var accepted = false
    private var preAcceptanceEvents: [RunEvent] = []

    private var source: ArtifactWeightSource?
    private var backend: FlagshipRoutedExpertBackend?
    private var rootedMirror: K3RootedArtifactMirror?
    /// Small value snapshots retained only while the terminal summary is
    /// assembled. The actual source/backend objects own multi-gigabyte working
    /// sets and are dropped during teardown, before the terminal event is
    /// emitted.
    private var terminalRuntime: TerminalRuntime?
    private var teardownRecord: TeardownRecord?
    /// Sampled while the scope is still held. `StorageInfo.describing(path:)`
    /// reads the volume through the artifact path, and after teardown that path
    /// may no longer be readable — which would cost the record exactly the
    /// storage-backend metadata spec §17.4 asks it to carry.
    private var environment: EnvironmentReport?

    /// The same reference spelling is used for translation and for the run
    /// record. Keeping it in one place prevents the JSON from describing an
    /// absolute-path run when the product actually consumed rooted artifact
    /// identities.
    private var effectiveBlobReference: PublishedArtifactLayout.BlobReference {
        request.artifact.runtimeAuthority == nil ? .absolutePath : .artifactReference
    }

    /// The run-scoped brackets as they stood when a pass began.
    ///
    /// Both instruments K3 owns count cumulatively over a run, so a pass is the
    /// difference of two readings — the same shape V4's `phaseBoundary` has,
    /// and for the same reason: without it the first pass's attribution would
    /// be charged to every pass after it.
    private struct PhaseBoundary {
        let expertPhaseSeconds: Double
        let expertIOWaitSeconds: Double
        let expertGatherComputeSeconds: Double
        let expertGatherCallCount: Int
        let expertReadCount: Int
        let serialReadSeconds: Double
        let prefetchWaitSeconds: Double
        let stageReadSeconds: Double
        /// Sticky, from either instrument. A saturated total cannot prove an
        /// identity, so a pass that straddles one is not attributed at all.
        let accountingOverflowed: Bool
    }

    private struct TerminalRuntime {
        let bytes: ByteAccounting
        let expertReadCount: Int
        let gatherCount: Int
        let expertPoolBudgetBytes: Int
        let deterministicReadAhead: DeterministicReadAheadMetrics
        let readAheadSkippedLayers: [Int]
        let pinnedTier: PinnedTierMetrics
        let instrumentation: RunInstrumentationTelemetry?
    }

    init(
        request: RunRequest, capabilities: RunnerCapabilities, knobs: K3EffectiveKnobs,
        handle: RunHandleID, latch: RunCancellationLatch,
        continuation: AsyncThrowingStream<RunEvent, Error>.Continuation,
        payloadFlow: RunPayloadFlowMeter,
        usesProductMemoryBudget: Bool
    ) {
        self.request = request
        self.capabilities = capabilities
        self.knobs = knobs
        self.handle = handle
        self.latch = latch
        self.continuation = continuation
        self.payloadFlow = payloadFlow
        self.usesProductMemoryBudget = usesProductMemoryBudget
        footprintPeak = FootprintPeak(
            processLifetimePeakBytes: ProcessFootprint.current()?.peakFootprintBytes)
    }

    // MARK: - The run

    func run() {
        thermalTrail.append(
            ThermalTransition(
                state: ThermalStateName(name: ThermalState.currentName), atSeconds: 0,
                phase: phaseName))

        do {
            let report = try decode()
            let teardown = tearDown()
            continuation.yield(
                .finished(summary(report, outcome: .finished, teardown: teardown)))
            continuation.finish()
        } catch let error as RunError where error == .cancelled {
            // A named ending, not a short result: the run says what it had
            // measured when it was stopped.
            let teardown = tearDown()
            continuation.yield(
                .cancelled(summary(nil, outcome: .cancelled, teardown: teardown)))
            continuation.finish()
        } catch {
            _ = tearDown()
            // `AsyncThrowingStream` already carries `Error`; keep the concrete
            // storage error intact. Turning a `StorageCoreError.shortTransfer`
            // into `RunError.underlying(String)` here discards the operation,
            // path, offset, expected count and actual count, forcing every UI
            // above the runner to parse prose to recover fields that already
            // existed.
            continuation.finish(throwing: error)
        }
    }

    /// What the decode loop produced, with no MLX array held past it: the
    /// summary is built after teardown, and holding a capture across teardown
    /// would keep alive exactly the memory teardown exists to release.
    private struct DecodeReport {
        var translateSeconds: Double = 0
        var openSeconds: Double = 0
        var layerSeconds: [Double] = []
        var logits: [Float] = []
        var minimumRelativeRoutingMargin: Double = .nan
        var routingDecisions: Int = 0
        var translatedDirectory: String = ""
        var globalsDirectory: String = ""
        var configOrigin: String = ""
    }

    /// The small values a product generation step carries into the next pass.
    /// The full TinyK3Capture is deliberately scoped inside `generateStep`, so
    /// its diagnostic MLX tensors are released before the following pass starts.
    private struct GenerationStep {
        let tokenID: Int
        let logits: [Float]
        let minimumRelativeRoutingMargin: Double
        let routingDecisions: Int
    }

    private func generateStep(
        model: K3Model, tokenIDs: [Int], state: inout TinyK3State
    ) throws -> GenerationStep {
        // MLX and Metal create autoreleased command-side objects as well as
        // Swift values. The decode thread lives for the whole turn, so without
        // a token-sized pool the first token's temporary objects can survive
        // into the second token even though `capture` is lexically gone.
        let step = try autoreleasepool {
            let capture = try model.forward(tokenIds: tokenIDs, state: &state)
            return GenerationStep(
                tokenID: capture.greedyTokenId,
                logits: capture.logits,
                minimumRelativeRoutingMargin: capture.minimumRelativeRoutingMargin,
                routingDecisions: capture.routerRelativeMargins.reduce(0) { $0 + $1.count })
        }
        // The configured cache limit is not an immediate reclamation command.
        // At a token boundary the previous pass's working buffers are no longer
        // useful: clear only MLX's recyclable cache while the evaluated KDA/MLA
        // state remains active for the next pass.
        MLX.Memory.clearCache()
        return step
    }

    private func decode() throws -> DecodeReport {
        var report = DecodeReport()
        try latch.check()

        environment = EnvironmentReport.current(path: request.artifact.root.path)
        let executionReference: ArtifactReference
        let fileAccess: ModelFileAccess?
        if request.artifact.runtimeAuthority != nil {
            let mirror = try K3RootedArtifactMirror(
                reference: request.artifact, workingDirectory: request.workingDirectory)
            rootedMirror = mirror
            executionReference = mirror.reference
            fileAccess = mirror.fileAccess
        } else {
            executionReference = request.artifact
            fileAccess = nil
        }
        let resolved = try K3ArtifactResolution.resolve(
            executionReference, fileAccess: fileAccess)
        let config = resolved.config
        report.globalsDirectory = resolved.globals.path
        report.configOrigin = resolved.configOrigin
        log("artifact: \(request.artifact.root.path)")
        log(
            "budget: \(ByteSize.format(request.memoryBudgetBytes)) declared, "
                + "MLX cache \(ByteSize.format(UInt64(knobs.mlxCacheLimitBytes))), "
                + "lm_head chunk \(knobs.logitChunkRows) rows")
        log(
            "config (\(resolved.configOrigin)): \(config.numHiddenLayers) layers, "
                + "\(config.numExperts) experts top-\(config.numExpertsPerToken), "
                + "hidden \(config.hiddenSize)")

        // MLX's cache is a second memory pool, and on a flagship run every
        // megabyte it holds is a megabyte layer 0's 4.68 GB does not have.
        MLX.Memory.cacheLimit = knobs.mlxCacheLimitBytes
        MLX.Memory.clearCache()

        // MARK: Translate

        let translated: URL
        switch resolved.shape {
        case .translated(let directory):
            translated = directory
            log("translated manifests: \(directory.path) (already present)")
        case .published(let artifact):
            phase(
                RunPhase(
                    name: "translating", generationStage: .preparing,
                    detail: "\(config.numHiddenLayers) manifests"))
            let destination = request.workingDirectory.appendingPathComponent("k3-translated")
            try FileManager.default.createDirectory(
                at: destination, withIntermediateDirectories: true)
            let clock = MonotonicClock.now()
            // `.absolutePath`, not a symlink farm: an iOS external volume
            // mounts under a per-mount UUID, so links written on internal
            // storage name a path that stops being true when the drive is
            // replugged.
            let translation = try PublishedArtifactLayout.translate(
                artifact: artifact.path, into: destination.path, config: config,
                blobReference: effectiveBlobReference,
                fileAccess: fileAccess)
            report.translateSeconds = MonotonicClock.seconds(since: clock)
            translated = destination
            log(
                "translated: \(translation.layers) layers (\(translation.denseLayers) dense, "
                    + "\(translation.moeLayers) MoE), \(translation.experts) experts, "
                    + "\(ByteSize.format(UInt64(translation.manifestBytes))) JSON")
        }
        report.translatedDirectory = translated.path
        try latch.check()

        // MARK: Open

        phase(
            RunPhase(
                name: "opening", generationStage: .preparing,
                detail: "\(config.numHiddenLayers) layer streams"))
        let openStarted = MonotonicClock.now()
        // Both of the following refuse rather than clamp: an undefined
        // read-ahead depth, a depth without a budget, a pool too small to serve
        // one dispatch plus its read-ahead. A clamped pool deadlocks on
        // `acquire` instead of running slowly, so the throw is the feature.
        let sourceConfiguration: ArtifactWeightSource.Configuration
        if let plan = request.pinPlan {
            sourceConfiguration = try ArtifactWeightSource.Configuration(
                plan: plan,
                deterministicReadAhead: knobs.deterministicReadAheadLayers)
            log(
                "memory plan: \(plan.pinnedLayers.count) deterministic layers pinned, "
                    + "\(ByteSize.format(plan.pinnedBytes)) resident, "
                    + "\(ByteSize.format(plan.readAheadReserveBytes)) read-ahead reserve")
        } else {
            sourceConfiguration = ArtifactWeightSource.Configuration(
                deterministicReadAhead: knobs.deterministicReadAheadLayers,
                budgetBytes: knobs.deterministicReadAheadLayers > 0
                    ? request.memoryBudgetBytes : nil,
                reserveBytes: knobs.readAheadReserveBytes)
        }
        let source = try ArtifactWeightSource(
            translated: translated.path,
            globals: resolved.globals.path,
            config: config,
            globalFiles: resolved.globalFiles,
            fileAccess: fileAccess,
            configuration: sourceConfiguration,
            successfulReadObserver: { [payloadFlow] bytes in
                payloadFlow.recordDeterministic(bytes)
            })
        self.source = source
        if knobs.deterministicReadAheadLayers > 0 {
            let schedule = source.readAheadSchedule()
            let refused = schedule.filter { !$0.isAllowed }.map(\.stagedLayer)
            log(
                "deterministic read-ahead: depth \(knobs.deterministicReadAheadLayers), "
                    + "reserve \(ByteSize.format(knobs.readAheadReserveBytes)), "
                    + "\(schedule.count - refused.count)/\(schedule.count) pairs fit"
                    + (refused.isEmpty ? "" : ", skipping layers \(refused)"))
        }
        let backend = try FlagshipRoutedExpertBackend(
            layers: source.streams,
            configuration: FlagshipRoutedExpertBackend.Configuration(
                granularity: knobs.fetchGranularity,
                queueDepth: knobs.queueDepth,
                readAhead: knobs.expertReadAhead,
                expertsPerDispatch: knobs.expertsPerDispatch),
            successfulReadObserver: { [payloadFlow] bytes in
                payloadFlow.recordExpert(bytes)
            })
        self.backend = backend
        let logitElements = UInt64(knobs.logitChunkRows).multipliedReportingOverflow(
            by: UInt64(config.hiddenSize))
        let logitBytes = logitElements.partialValue.multipliedReportingOverflow(
            by: UInt64(MemoryLayout<Float>.stride))
        let minimumWorking = UInt64(knobs.mlxCacheLimitBytes).addingReportingOverflow(
            logitBytes.partialValue)
        guard !logitElements.overflow, !logitBytes.overflow, !minimumWorking.overflow else {
            throw TinyK3Error.configuration(
                "the effective cache and logit-chunk working set exceeds UInt64.max")
        }
        let requiredWorkingReserve: UInt64
        if usesProductMemoryBudget {
            guard case .tokenIDs(let promptIDs) = request.prompt else {
                throw RunError.promptNotSupported(reason: "K3DecodeRunner carries no vocabulary")
            }
            let stateMemory = try K3GenerationStateMemory(config: config)
            let productReserve = try K3ProductMemoryBudget.workingReserveBytes(
                state: stateMemory, promptTokenCount: promptIDs.count,
                maximumNewTokens: request.maximumNewTokens)
            // The product reserve contains the default cache/chunk allowance,
            // but an explicitly larger effective knob remains authoritative.
            // Refuse its larger requirement rather than hiding it inside the
            // product constant.
            requiredWorkingReserve = max(productReserve, minimumWorking.partialValue)
            log(
                "product state reserve: \(ByteSize.format(requiredWorkingReserve)) "
                    + "(KDA \(ByteSize.format(stateMemory.kdaBytes)), MLA "
                    + "\(ByteSize.format(stateMemory.mlaBytesPerCachedPosition))/position)")
        } else {
            requiredWorkingReserve = minimumWorking.partialValue
        }
        try source.validateRuntimePlan(
            expertPoolBytes: UInt64(backend.budgetBytes),
            minimumWorkingReserveBytes: requiredWorkingReserve)
        if usesProductMemoryBudget, request.pinPlan == nil {
            let floor = UInt64Accounting.checkedSum([
                source.widestResidentLayerBytes,
                UInt64(backend.budgetBytes), requiredWorkingReserve,
            ])
            guard let floor else {
                throw TinyK3Error.configuration("the product working floor exceeds UInt64.max")
            }
            guard request.memoryBudgetBytes >= floor else {
                throw RunError.budgetBelowMinimum(
                    declared: request.memoryBudgetBytes, minimum: floor, model: .kimiK3)
            }
        }
        accept()
        report.openSeconds = MonotonicClock.seconds(since: openStarted)
        log(
            "expert pool: \(ByteSize.format(UInt64(backend.budgetBytes))) "
                + "(granularity \(knobs.fetchGranularity), readAhead \(knobs.expertReadAhead), "
                + "\(knobs.expertsPerDispatch)/dispatch)")
        try latch.check()

        let model = K3Model(config: config, source: source, expertBackend: backend)
        model.logitChunkRows = knobs.logitChunkRows
        // The oracle-facing capture is intentionally not the product working
        // set. It keeps evaluated tensors from every layer and, if retained
        // across passes, overlaps the previous token with the next token's
        // widest layer. Product generation retains only the small values copied
        // by `generateStep` above.
        model.captureDiagnosticTensors = false
        // Cleared on every exit path, cancellation included: the closure holds
        // this engine, and the model holds the closure.
        defer { model.onLayerCompleted = nil }
        let layerCount = config.numHiddenLayers
        model.onLayerCompleted = { [unowned self] index, seconds in
            try self.latch.check()
            self.observe(layer: index, of: layerCount, seconds: seconds)
        }

        // MARK: Decode

        guard case .tokenIDs(let promptIDs) = request.prompt else {
            throw RunError.promptNotSupported(reason: "K3DecodeRunner carries no vocabulary")
        }
        var state = TinyK3State()
        // One pass, one token. The prefill produces the first, so the pass
        // count *is* the new-token count and the ceiling cannot be exceeded by
        // forgetting a flag.
        for pass in 0..<request.maximumNewTokens {
            try latch.check()
            phaseName = pass == 0 ? "prefill" : "decode \(pass)"
            generationStage = pass == 0 ? .prefill : .decode
            phase(
                RunPhase(
                    name: phaseName, generationStage: generationStage,
                    fraction: 0, detail: "layer 0/\(layerCount)"))
            let phaseBoundary = phaseBoundaryReading()
            let passStarted = MonotonicClock.now()
            let ids = pass == 0 ? promptIDs : [tokenIDs[tokenIDs.count - 1]]
            let produced = try generateStep(model: model, tokenIDs: ids, state: &state)
            let seconds = MonotonicClock.seconds(since: passStarted)
            let reclaimSeconds = model.lastLayerReclaimSeconds.reduce(0, +)
            report.logits = produced.logits
            report.minimumRelativeRoutingMargin = produced.minimumRelativeRoutingMargin
            report.routingDecisions = produced.routingDecisions
            passSeconds.append(seconds)
            completedTokenBytes = bytes()
            if pass == 0 { prefillBytes = completedTokenBytes }
            tokenIDs.append(produced.tokenID)
            continuation.yield(
                .token(
                    TokenEvent(
                        index: pass, tokenID: produced.tokenID, text: nil,
                        secondsSincePreviousToken: seconds, isPrefillToken: pass == 0,
                        // Nil by contract: `captureRoutingMargins` is not a
                        // knob this runner honours, so nothing here was
                        // measured under it.
                        top1Top2Gap: nil)))
            log(
                "\(phaseName): token \(produced.tokenID) in "
                    + String(format: "%.1f s (layer reclamation %.3f s)", seconds, reclaimSeconds))
            recordPhaseSplit(
                since: phaseBoundary, model: model, passIndex: pass, seconds: seconds)
            if K3GenerationBoundary.endsResponse(
                tokenID: produced.tokenID, vocabularySize: config.vocabSize)
            {
                log("\(phaseName): response framing completed")
                break
            }
        }
        report.layerSeconds = model.lastLayerSeconds
        model.onLayerCompleted = nil

        // A full verification authorizes physical objects, not only the names
        // they occupied at launch. Recheck every held object and the bookmark
        // binding before a successful terminal event can be emitted.
        try rootedMirror?.validateCurrentBinding()
        return report
    }

    // MARK: - Progress

    private func accept() {
        guard !accepted else { return }
        continuation.yield(
            .accepted(
                RunAcceptance(
                    handle: handle, model: capabilities.model,
                    declaredBudgetBytes: request.memoryBudgetBytes,
                    pinPlan: request.pinPlan,
                    effectiveKnobs: knobs.asRunKnobs,
                    artifactRoot: request.artifact.root.path, startedAt: started)))
        accepted = true
        for event in preAcceptanceEvents { continuation.yield(event) }
        preAcceptanceEvents.removeAll(keepingCapacity: false)
    }

    private func emit(_ event: RunEvent) {
        if accepted {
            continuation.yield(event)
        } else {
            preAcceptanceEvents.append(event)
        }
    }

    private func log(_ line: String) { emit(.log(line)) }

    private func phase(_ phase: RunPhase) { emit(.phase(phase)) }

    /// One layer of one pass: a phase and a telemetry sample, the pair the
    /// screen needs to tell a slow run from a hung one.
    private func observe(layer index: Int, of count: Int, seconds: Double) {
        let telemetry = sample(phase: phaseName)
        if thermalTrail.last?.state != telemetry.thermalState {
            thermalTrail.append(
                ThermalTransition(
                    state: telemetry.thermalState,
                    atSeconds: MonotonicClock.seconds(since: startedTick), phase: phaseName))
        }
        continuation.yield(
            .phase(
                RunPhase(
                    name: phaseName, generationStage: generationStage,
                    fraction: Double(index + 1) / Double(count),
                    detail: "layer \(index + 1)/\(count) — " + String(format: "%.2f s", seconds))))
        continuation.yield(.telemetry(telemetry))
    }

    // MARK: - Where a pass went

    private func phaseBoundaryReading() -> PhaseBoundary? {
        guard let source, let backend else { return nil }
        let expert = backend.expertPhaseMetrics
        let readAhead = source.readAheadMetrics
        return PhaseBoundary(
            expertPhaseSeconds: expert.phaseSeconds,
            expertIOWaitSeconds: expert.ioWaitSeconds,
            expertGatherComputeSeconds: expert.gatherComputeSeconds,
            expertGatherCallCount: expert.gatherCallCount,
            expertReadCount: expert.expertReadCount,
            serialReadSeconds: readAhead.serialReadSeconds,
            prefetchWaitSeconds: readAhead.prefetchWaitSeconds,
            stageReadSeconds: readAhead.stageReadSeconds,
            accountingOverflowed: expert.timingAccountingOverflowed
                || readAhead.accountingOverflowed)
    }

    /// Attribute one completed pass, or say nothing.
    ///
    /// The two run-scoped instruments are differenced against the boundary this
    /// pass started from; the model's three per-layer arrays already belong to
    /// the pass that just finished and are summed as they stand. The
    /// subtraction fails closed for the same reasons V4's does — a term that
    /// moved backwards, a saturated total, or terms that do not fit inside the
    /// measured wall all mean the two readings cannot be compared, and a
    /// decomposition that cannot be trusted is worse than none.
    ///
    /// The terms are disjoint by construction, which is what lets them be
    /// summed: `serialReadSeconds` and `prefetchWaitSeconds` bracket the two
    /// ways the decode loop obtains deterministic weights, the attention
    /// barrier is its own `MLX.eval` between the two halves of a layer, the
    /// expert phase is entered from the MoE block after that barrier, and the
    /// reclaim runs at the layer boundary outside all of them. The stager's own
    /// read time and the layer loop's total are reported *beside* the pass —
    /// the first overlaps compute by design, the second contains several of the
    /// terms above — so neither is summed into the identity.
    private func recordPhaseSplit(
        since earlier: PhaseBoundary?, model: K3Model, passIndex: Int, seconds: Double
    ) {
        guard let earlier, let current = phaseBoundaryReading(),
            !earlier.accountingOverflowed, !current.accountingOverflowed,
            seconds.isFinite, seconds > 0
        else { return }

        let phase = current.expertPhaseSeconds - earlier.expertPhaseSeconds
        let ioWait = current.expertIOWaitSeconds - earlier.expertIOWaitSeconds
        let gather = current.expertGatherComputeSeconds - earlier.expertGatherComputeSeconds
        let expertOther = phase - ioWait - gather
        let serialRead = current.serialReadSeconds - earlier.serialReadSeconds
        let stagerWait = current.prefetchWaitSeconds - earlier.prefetchWaitSeconds
        let stagedRead = current.stageReadSeconds - earlier.stageReadSeconds
        let barrier = model.lastLayerAttentionBarrierSeconds.reduce(0, +)
        let reclaim = model.lastLayerReclaimSeconds.reduce(0, +)
        let layerTotal = model.lastLayerSeconds.reduce(0, +)
        guard [
            phase, ioWait, gather, expertOther, serialRead, stagerWait, stagedRead,
            barrier, reclaim, layerTotal,
        ].allSatisfy({ $0.isFinite && $0 >= -1e-6 }) else { return }

        let gathers = current.expertGatherCallCount - earlier.expertGatherCallCount
        let reads = current.expertReadCount - earlier.expertReadCount
        guard gathers >= 0, reads >= 0 else { return }

        let summary = RunPhaseSummary(
            passKind: passIndex == 0 ? .prefill : .decode,
            passIndex: passIndex,
            passSeconds: seconds,
            terms: [
                RunPhaseTerm(
                    name: RunPhaseTermName.deterministicRead, seconds: max(0, serialRead)),
                RunPhaseTerm(name: RunPhaseTermName.stagerWait, seconds: max(0, stagerWait)),
                RunPhaseTerm(
                    name: RunPhaseTermName.expertIOWait, seconds: max(0, ioWait), count: reads),
                RunPhaseTerm(
                    name: RunPhaseTermName.expertGatherCompute, seconds: max(0, gather),
                    count: gathers),
                RunPhaseTerm(name: RunPhaseTermName.expertOther, seconds: max(0, expertOther)),
                RunPhaseTerm(
                    name: RunPhaseTermName.attentionBarrier, seconds: max(0, barrier),
                    count: model.lastLayerAttentionBarrierSeconds.count),
                RunPhaseTerm(
                    name: RunPhaseTermName.reclaim, seconds: max(0, reclaim),
                    count: model.lastLayerReclaimSeconds.count),
                RunPhaseTerm(
                    name: RunPhaseTermName.stagedRead, seconds: max(0, stagedRead),
                    isInsidePass: false),
                RunPhaseTerm(
                    name: RunPhaseTermName.layerTotal, seconds: max(0, layerTotal),
                    count: model.lastLayerSeconds.count, isInsidePass: false),
            ])
        guard summary.isBalanced else { return }
        phaseSummaries.append(summary)
        emit(.phaseSummary(summary))
        emit(.log(Self.summaryLine(for: summary)))
    }

    /// One line for a run's log, in the shape V4's `[v4 phase]` line has: the
    /// pass, its named terms, and the residual.
    private static func summaryLine(for summary: RunPhaseSummary) -> String {
        let terms = summary.insideTerms
            .map { String(format: "%.2f s %@", $0.seconds, $0.name) }
            .joined(separator: " + ")
        let beside = summary.besideTerms
            .map { String(format: "%@ %.2f s", $0.name, $0.seconds) }
            .joined(separator: ", ")
        return String(
            format: "[k3 phase] %@ pass = %@ + %.2f s other; beside: %@",
            summary.passKind.rawValue, terms, summary.unattributedSeconds, beside)
    }

    /// The bytes so far, split by the counters that own them. The total is the
    /// sum, never a third counter: a run's byte accounting has to be an
    /// identity, not two numbers that agree most of the time.
    private func bytes() -> ByteAccounting {
        if let terminalRuntime { return terminalRuntime.bytes }
        let deterministic = source?.deterministicBytesRead ?? 0
        let expert = backend?.expertPhaseMetrics.expertBytesRead ?? 0
        let total = deterministic.addingReportingOverflow(expert)
        return ByteAccounting(
            // Telemetry must never turn an overflow into a small, apparently
            // balanced run. The split remains present, so `isBalanced` names
            // the impossible total as false.
            totalBytesRead: total.overflow ? .max : total.partialValue,
            deterministicBytesRead: deterministic, expertBytesRead: expert)
    }

    private func sample(
        phase: String, stage: RunGenerationStage? = nil
    ) -> RunTelemetry {
        // `HarnessTelemetry` is what the device harness quotes, and it is
        // sampled here rather than re-derived: the footprint iOS enforces its
        // limit against is one number, and two ways of reading it would be two
        // answers to the same question.
        let harness = HarnessTelemetry.sample(declaredBudgetBytes: request.memoryBudgetBytes)
        let runPeak = footprintPeak.observe(
            currentBytes: harness.footprintBytes,
            processLifetimePeakBytes: harness.peakFootprintBytes)
        let elapsed = MonotonicClock.seconds(since: startedTick)
        let accounting = bytes()
        let decodeTokens = max(0, tokenIDs.count - 1)
        let completed = completedTokenBytes
        let prefillElapsed = passSeconds.first
        let decodeElapsed = passSeconds.dropFirst().reduce(0, +)
        let decodeAccounting = completed.flatMap { completed in
            prefillBytes.flatMap { completed.subtracting($0) }
        }
        return RunTelemetry(
            at: Date(), elapsed: elapsed, phase: phase,
            tokensPerSecond: decodeTokens > 0 && decodeElapsed > 0
                ? Double(decodeTokens) / decodeElapsed : nil,
            generationStage: stage ?? generationStage,
            bytes: accounting,
            bytesPerSecond: elapsed > 0 ? Double(accounting.totalBytesRead) / elapsed : nil,
            completedTokenBytes: completed,
            prefillBytes: prefillBytes,
            bytesPerToken: decodeTokens > 0
                ? decodeAccounting.map { $0.totalBytesRead / UInt64(decodeTokens) } : nil,
            prefillSeconds: prefillElapsed,
            decodeSeconds: decodeElapsed,
            decodeTokensCompleted: decodeTokens,
            declaredBudgetBytes: request.memoryBudgetBytes,
            footprintBytes: harness.footprintBytes,
            peakFootprintBytes: runPeak,
            residentBytes: harness.residentBytes,
            availableBytes: harness.availableBytes,
            mlxActiveBytes: UInt64(max(0, MLX.GPU.activeMemory)),
            mlxCacheBytes: UInt64(max(0, MLX.GPU.cacheMemory)),
            mlxPeakBytes: UInt64(max(0, MLX.GPU.peakMemory)),
            thermalState: ThermalStateName(name: harness.thermalState),
            lowPowerMode: harness.lowPowerModeEnabled,
            batteryLevel: Self.batteryLevel(),
            instrumentation: instrumentation())
    }

    private func instrumentation() -> RunInstrumentationTelemetry? {
        if let terminal = terminalRuntime { return terminal.instrumentation }

        let expert: RunExpertTelemetry?
        if let backend {
            let phase = backend.expertPhaseMetrics
            let cache = backend.expertPoolStatistics
            expert = RunExpertTelemetry(
                phaseSeconds: phase.phaseSeconds,
                ioWaitSeconds: phase.ioWaitSeconds,
                bytesRead: phase.expertBytesRead,
                cachePolicy: cache.cachePolicyName,
                cacheHits: cache.cacheHitCount,
                cacheMisses: cache.cacheMissCount,
                cacheEvictions: cache.cacheEvictionCount,
                cacheRejectedAdmissions: cache.cacheRejectedAdmissionCount,
                pinnedExpertCount: 0)
        } else {
            expert = nil
        }

        let readAhead = source.map {
            Self.readAheadTelemetry($0.readAheadMetrics, hasFinalCensus: false)
        }
        guard expert != nil || readAhead != nil else { return nil }
        return RunInstrumentationTelemetry(expert: expert, readAhead: readAhead)
    }

    private static func readAheadTelemetry(
        _ metrics: DeterministicReadAheadMetrics, hasFinalCensus: Bool
    ) -> RunReadAheadTelemetry {
        RunReadAheadTelemetry(
            depth: metrics.depth,
            stagedBytes: metrics.stagedBytes,
            peakStagedBytes: metrics.peakStagedBytes,
            materialisedStagedBytes: metrics.materialisedStagedBytes,
            discardedStagedBytes: metrics.discardedStagedBytes,
            outstandingStagedBytes: metrics.outstandingStagedBytes,
            prefetchWaitSeconds: metrics.prefetchWaitSeconds,
            stageReadSeconds: metrics.stageReadSeconds,
            nonFreeSlots: hasFinalCensus
                ? (metrics.outstandingStagedBytes == 0 ? 0 : 1) : nil)
    }

    private static func batteryLevel() -> Float? {
        #if canImport(UIKit)
            let level = UIDevice.current.batteryLevel
            return level < 0 ? nil : level
        #else
            return nil
        #endif
    }

    // MARK: - Teardown

    /// What the run released, written by the release itself.
    ///
    /// Reported rather than assumed because "the reader was closed" is exactly
    /// the kind of claim that is true right up until an early return skips it,
    /// and a leaked reader on a 1.56 TB volume is a leaked file descriptor per
    /// container.
    struct TeardownRecord: Encodable, Sendable, Equatable {
        var sourceShutDown = false
        var expertBackendShutDown = false
        var scopeReleased = false
        var scopeWasHeld = false
        /// Staged bytes the deterministic read-ahead still owned. Zero after a
        /// clean teardown, cancellation included.
        var outstandingStagedBytes: UInt64 = 0
        var readAheadAccountingBalanced = true
        var mlxCacheBytesAfterClear: UInt64 = 0
        /// The large Swift owners, not only their file readers, were released
        /// before the terminal event crossed the stream.
        var deterministicWorkingSetReleased = false
        var expertWorkingSetReleased = false
    }

    /// Idempotent, and called from exactly two places: the terminal path, which
    /// wants the record, and the error path, which only wants it done.
    @discardableResult
    private func tearDown() -> TeardownRecord {
        if let existing = teardownRecord { return existing }
        var record = TeardownRecord()

        // Capture expert timing/cache values before shutdown. The reader
        // totals survive eviction, but the pool census belongs to the live run.
        let terminalExpert: RunExpertTelemetry?
        if let backend {
            let phase = backend.expertPhaseMetrics
            let cache = backend.expertPoolStatistics
            terminalExpert = RunExpertTelemetry(
                phaseSeconds: phase.phaseSeconds,
                ioWaitSeconds: phase.ioWaitSeconds,
                bytesRead: phase.expertBytesRead,
                cachePolicy: cache.cachePolicyName,
                cacheHits: cache.cacheHitCount,
                cacheMisses: cache.cacheMissCount,
                cacheEvictions: cache.cacheEvictionCount,
                cacheRejectedAdmissions: cache.cacheRejectedAdmissionCount,
                pinnedExpertCount: 0)
        } else {
            terminalExpert = nil
        }

        backend?.shutdown()
        record.expertBackendShutDown = backend != nil
        source?.shutdown()
        record.sourceShutDown = source != nil
        // Both owners are quiescent. Flush the final partial monotonic window
        // before the terminal RunEvent can cross the session.
        payloadFlow.finish()
        if let source, let backend {
            let readAhead = source.readAheadMetrics
            record.outstandingStagedBytes = readAhead.outstandingStagedBytes
            record.readAheadAccountingBalanced = readAhead.isAccountingBalanced
            terminalRuntime = TerminalRuntime(
                bytes: bytes(),
                expertReadCount: backend.expertReadCount,
                gatherCount: backend.gatherCount,
                expertPoolBudgetBytes: backend.budgetBytes,
                deterministicReadAhead: readAhead,
                readAheadSkippedLayers: source.readAheadSchedule()
                    .filter { !$0.isAllowed }.map(\.stagedLayer),
                pinnedTier: source.pinnedTierMetrics,
                instrumentation: RunInstrumentationTelemetry(
                    expert: terminalExpert,
                    readAhead: Self.readAheadTelemetry(
                        readAhead, hasFinalCensus: true)))
        } else {
            let terminalReadAhead = source.map {
                Self.readAheadTelemetry($0.readAheadMetrics, hasFinalCensus: true)
            }
            terminalRuntime = TerminalRuntime(
                bytes: bytes(),
                expertReadCount: backend?.expertReadCount ?? 0,
                gatherCount: backend?.gatherCount ?? 0,
                expertPoolBudgetBytes: backend?.budgetBytes ?? 0,
                deterministicReadAhead: source?.readAheadMetrics
                    ?? DeterministicReadAheadMetrics(),
                readAheadSkippedLayers: source?.readAheadSchedule()
                    .filter { !$0.isAllowed }.map(\.stagedLayer) ?? [],
                pinnedTier: source?.pinnedTierMetrics ?? PinnedTierMetrics(),
                instrumentation: terminalExpert != nil || terminalReadAhead != nil
                    ? RunInstrumentationTelemetry(
                        expert: terminalExpert, readAhead: terminalReadAhead)
                    : nil)
        }
        backend = nil
        source = nil
        record.expertWorkingSetReleased = backend == nil
        record.deterministicWorkingSetReleased = source == nil

        // Readers and their owners are gone before the rooted capability and
        // finally the operator's security scope are released.
        rootedMirror = nil
        // The scope is the operator's grant. Dropping it here — not in a
        // caller's `defer` — is what makes the failure paths release it too.
        if let scope = request.artifact.scope {
            record.scopeWasHeld = true
            scope.release()
            record.scopeReleased = scope.isReleased
        }
        MLX.Memory.clearCache()
        record.mlxCacheBytesAfterClear = UInt64(max(0, MLX.Memory.cacheMemory))

        teardownRecord = record
        return record
    }

    // MARK: - Summary

    private enum Outcome: String { case finished, cancelled }

    private func summary(
        _ report: DecodeReport?, outcome: Outcome, teardown: TeardownRecord
    ) -> RunSummary {
        defer { terminalRuntime = nil }
        let terminal = terminalRuntime
        let wall = MonotonicClock.seconds(since: startedTick)
        let telemetry = sample(phase: outcome.rawValue, stage: .terminal)
        let peak = telemetry.peakFootprintBytes
        let prefillSeconds = passSeconds.first
        let decodeSeconds = passSeconds.dropFirst().reduce(0, +)
        let decodeTokens = max(0, tokenIDs.count - 1)
        let rate = decodeTokens == 0 || decodeSeconds <= 0
            ? nil : Double(decodeTokens) / decodeSeconds

        var digest: String?
        var top: [TopLogit] = []
        var gap: Double?
        var nonFinite: Int?
        if let report, !report.logits.isEmpty {
            var raw = Data(capacity: report.logits.count * 4)
            for value in report.logits {
                withUnsafeBytes(of: value.bitPattern.littleEndian) { raw.append(contentsOf: $0) }
            }
            digest = SHA256.hash([UInt8](raw)).map { String(format: "%02x", $0) }.joined()
            let sorted = report.logits.enumerated().sorted { $0.element > $1.element }
            top = sorted.prefix(8).map { TopLogit(id: $0.offset, logit: Double($0.element)) }
            if sorted.count > 1 { gap = Double(sorted[0].element - sorted[1].element) }
            nonFinite = report.logits.filter { !$0.isFinite }.count
        }

        let payload = Payload(
            runner: "k3-decode",
            handle: handle.rawValue.uuidString,
            revision: BuildInfo.gitRevision,
            outcome: outcome.rawValue,
            environment: environment ?? EnvironmentReport.current(path: request.artifact.root.path),
            artifactRoot: request.artifact.root.path,
            translatedDirectory: report?.translatedDirectory ?? "",
            globalsDirectory: report?.globalsDirectory ?? "",
            configOrigin: report?.configOrigin ?? "",
            promptTokenIds: {
                if case .tokenIDs(let ids) = request.prompt { return ids }
                return []
            }(),
            requestedNewTokens: request.maximumNewTokens,
            generatedTokenIds: tokenIDs,
            logitsSHA256: digest,
            top8: top,
            top1Top2Gap: gap,
            nonFiniteLogits: nonFinite,
            minimumRelativeRoutingMargin: report?.minimumRelativeRoutingMargin,
            routingDecisions: report?.routingDecisions,
            wallSeconds: wall,
            translateSeconds: report?.translateSeconds ?? 0,
            openSeconds: report?.openSeconds ?? 0,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            passSeconds: passSeconds,
            phaseSummaries: phaseSummaries,
            layerSeconds: report?.layerSeconds ?? [],
            tokensPerSecond: rate,
            bytes: bytes(),
            bytesBalanced: bytes().isBalanced,
            expertReadCount: terminal?.expertReadCount ?? 0,
            gatherCount: terminal?.gatherCount ?? 0,
            expertPoolBudgetBytes: terminal?.expertPoolBudgetBytes ?? 0,
            mlxPeakBytes: max(0, MLX.GPU.peakMemory),
            deterministicReadAhead: terminal?.deterministicReadAhead
                ?? DeterministicReadAheadMetrics(),
            readAheadSkippedLayers: terminal?.readAheadSkippedLayers ?? [],
            pinPlan: request.pinPlan,
            pinnedTier: terminal?.pinnedTier ?? PinnedTierMetrics(),
            declaredBudgetBytes: request.memoryBudgetBytes,
            peakFootprintBytes: peak,
            budgetRespected: peak <= request.memoryBudgetBytes,
            telemetry: telemetry,
            thermalTrail: thermalTrail,
            configuration: knobs.asStrings(blobReference: effectiveBlobReference),
            teardown: teardown)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        var headline: String
        switch outcome {
        case .finished:
            headline = tokenIDs.isEmpty
                ? "no token"
                : String(
                    format: "token%@ %@ in %.1f s, peak %@",
                    tokenIDs.count == 1 ? "" : "s",
                    tokenIDs.map(String.init).joined(separator: ", "), wall,
                    ByteSize.format(peak))
        case .cancelled:
            headline = String(
                format: "cancelled after %d of %d token%@, %.1f s, peak %@",
                tokenIDs.count, request.maximumNewTokens,
                request.maximumNewTokens == 1 ? "" : "s", wall, ByteSize.format(peak))
        }
        if peak > request.memoryBudgetBytes { headline += ", OVER BUDGET" }

        let accounting = bytes()
        var lines = [
            "read \(ByteSize.format(accounting.deterministicBytesRead ?? 0)) deterministic + "
                + "\(ByteSize.format(accounting.expertBytesRead ?? 0)) expert",
            "expert pool \(ByteSize.format(UInt64(terminal?.expertPoolBudgetBytes ?? 0))), "
                + "MLX peak \(ByteSize.format(UInt64(max(0, MLX.GPU.peakMemory))))",
            "thermal " + thermalTrail.map(\.state.rawValue).joined(separator: " -> "),
            "teardown: readers closed, \(teardown.outstandingStagedBytes) B staged outstanding",
        ]
        if let digest { lines.insert("logits sha256 \(digest)", at: 2) }
        if let nonFinite, nonFinite > 0 { lines.append("\(nonFinite) NON-FINITE LOGITS") }

        return RunSummary(
            handle: handle, model: capabilities.model, tokenIDs: tokenIDs,
            // No vocabulary travels with this runner, so there is no text to
            // report — and inventing one from a different tokenizer would be a
            // claim about the model that nothing here can support.
            text: nil,
            wallSeconds: wall, tokensPerSecond: rate, finalTelemetry: telemetry,
            peakFootprintBytes: peak, budgetRespected: peak <= request.memoryBudgetBytes,
            thermalTrail: thermalTrail,
            logitsDigest: tokenIDs.indices.last.flatMap { tokenIndex in
                digest.map { RunLogitsDigest(tokenIndex: tokenIndex, hex: $0) }
            },
            resultJSON: try? encoder.encode(payload),
            resultJSONFilename: "k3-decode-run.json", headline: headline, lines: lines,
            gitRevision: BuildInfo.gitRevision)
    }

    // MARK: - The record

    struct TopLogit: Encodable {
        let id: Int
        let logit: Double
    }

    /// Spec §17.4: a documented result carries its own metadata, so the run
    /// emits the file rather than the record's author re-typing the numbers.
    struct Payload: Encodable {
        let runner: String
        let handle: String
        let revision: String
        let outcome: String
        let environment: EnvironmentReport
        let artifactRoot: String
        let translatedDirectory: String
        let globalsDirectory: String
        let configOrigin: String
        let promptTokenIds: [Int]
        let requestedNewTokens: Int
        let generatedTokenIds: [Int]
        let logitsSHA256: String?
        let top8: [TopLogit]
        let top1Top2Gap: Double?
        let nonFiniteLogits: Int?
        let minimumRelativeRoutingMargin: Double?
        let routingDecisions: Int?
        /// The whole run, translation and opening included. `decodeSeconds` is
        /// the part a tok/s figure may be quoted from.
        let wallSeconds: Double
        let translateSeconds: Double
        let openSeconds: Double
        let prefillSeconds: Double?
        let decodeSeconds: Double
        let passSeconds: [Double]
        /// Where each pass's wall time went, beside how long it took. A pass
        /// the run could not attribute is absent, so this is not padded to the
        /// length of `passSeconds`.
        let phaseSummaries: [RunPhaseSummary]
        let layerSeconds: [Double]
        let tokensPerSecond: Double?
        let bytes: ByteAccounting
        let bytesBalanced: Bool
        let expertReadCount: Int
        let gatherCount: Int
        let expertPoolBudgetBytes: Int
        let mlxPeakBytes: Int
        let deterministicReadAhead: DeterministicReadAheadMetrics
        let readAheadSkippedLayers: [Int]
        /// The exact pre-run decision and the post-teardown counters that show
        /// whether the runtime honoured and released it.
        let pinPlan: PinPlan?
        let pinnedTier: PinnedTierMetrics
        let declaredBudgetBytes: UInt64
        let peakFootprintBytes: UInt64
        let budgetRespected: Bool
        let telemetry: RunTelemetry
        let thermalTrail: [ThermalTransition]
        let configuration: [String: String]
        let teardown: TeardownRecord
    }
}
