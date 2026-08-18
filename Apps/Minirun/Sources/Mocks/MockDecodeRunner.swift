#if DEBUG
import Foundation
import MinirunKit
import MinirunRunners
import StorageCore

/// A `RunnerFacade` that reads no bytes and computes nothing.
///
/// What it *does* do is behave like the real one where behaviour is the
/// contract: it refuses before a byte is read, it refuses knobs by name, it
/// emits tokens at pass boundaries rather than per step, it reports a peak that
/// can exceed the stated budget, and it ends a cancellation as a named failure
/// class rather than as a short result.
///
/// Replaced at integration by `MinirunRunners.K3DecodeRunner` /
/// `.DeepSeekV4DecodeRunner`, which wrap the existing scenarios on a dedicated
/// `Thread` — `HarnessScenario.run` blocks by design and a 357-second blocking
/// call on the cooperative pool would starve it.
final class MockDecodeRunner: InstrumentedRunner, @unchecked Sendable {

    /// What the world does to this run, and when.
    enum Fault: Sendable, Equatable {
        case volumeDisconnects(volume: String, atLayer: Int)
        case shortRead(atLayer: Int)
        /// The peak overshoots the stated budget. The run still completes; the
        /// number is still written down.
        case budgetOvershoot(byBytes: UInt64)
        case thermalEscalation
    }

    let capabilities: RunnerCapabilities
    private let entry: CatalogEntry
    /// Wall seconds one simulated pass should take to review.
    private let reviewSeconds: Double
    var fault: Fault?
    /// Which turn of a conversation this run answers. It selects where in the
    /// preview continuation ladder the reply starts, so turn 2 continues turn 1
    /// instead of repeating it. Zero — a fresh single-turn run — reproduces the
    /// archived record's token exactly, which is what the tests pin.
    var turnIndex = 0

    private let lock = NSLock()
    private var instrumentContinuations:
        [RunHandleID: AsyncStream<InstrumentSample>.Continuation] = [:]
    private var cancelled: Set<RunHandleID> = []

    init(entry: CatalogEntry, reviewSeconds: Double = 26) {
        self.entry = entry
        self.reviewSeconds = reviewSeconds
        switch entry.id {
        case .kimiK3:
            capabilities = RunnerCapabilities(
                model: .kimiK3, layout: .k3FlagshipLayerStreams,
                supportedKnobs: [
                    "expertReadAhead", "expertsPerDispatch", "queueDepth", "granularity",
                    "mlxCacheLimitBytes", "logitChunkRows", "deterministicReadAheadLayers",
                    "readAheadReserveBytes",
                ],
                minimumBudgetBytes: K3ProductMemoryBudget.minimumBudgetBytes,
                // The platform's tier, not a compiled-in 64. The budget floor
                // beside it was already policy-derived, so the ceiling being a
                // literal made the review build promise an iPhone 64 output
                // tokens where the product policy admits 2 (ADR 0011).
                maximumNewTokens: K3ProductMemoryBudget.maximumNewTokens,
                acceptsTextPrompts: false, requiresMLX: true,
                supportsPinPlan: true,
                tokenEventGranularity: .perPass)
        case .deepseekV4Flash:
            capabilities = RunnerCapabilities(
                model: .deepseekV4Flash, layout: .v4FlashUnitBundle,
                supportedKnobs: DeepSeekV4DecodeRunner.knobs,
                minimumBudgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
                maximumNewTokens: DeepSeekV4ProductMemoryBudget.maximumNewTokens,
                acceptsTextPrompts: false, requiresMLX: true,
                tokenEventGranularity: .perPass)
        default:
            // A fixture with no product runner of its own. It states only the
            // geometry its own descriptor carries rather than borrowing a
            // shipping model's floor.
            capabilities = RunnerCapabilities(
                model: entry.id, layout: entry.descriptor.layout,
                supportedKnobs: [
                    "expertPoolSlots", "expertsPerDispatch", "captureRoutingMargins",
                ],
                minimumBudgetBytes: entry.descriptor.minimumBudgetBytes ?? 4_000_000_000,
                maximumNewTokens: 64, acceptsTextPrompts: false, requiresMLX: true,
                tokenEventGranularity: .perToken)
        }
    }

    // MARK: Refusals, before a byte is read

    func validate(_ request: RunRequest) throws {
        try validateCommon(request)
        if request.model != capabilities.model {
            throw RunError.runnerUnavailable(
                request.model, reason: "this runner serves \(capabilities.model.rawValue)")
        }
        if entry.id == .kimiK3 {
            let setting = ReadAheadSetting(request.knobs.deterministicReadAheadLayers)
            guard setting.isProductSupported else {
                throw RunError.knobOutOfRange(
                    name: "deterministicReadAheadLayers",
                    value: "\(setting.effectiveDepth)",
                    allowed: "0 (off) or 1 (one layer)")
            }
        }
    }

    // MARK: The run

    func start(_ request: RunRequest) throws -> RunSession {
        do {
            try validate(request)
        } catch {
            // A session was never created, so ownership cannot pass to its run
            // bracket. Release the caller's grant on this failure path.
            request.artifact.scope?.release()
            throw error
        }
        let handle = RunHandleID()

        lock.lock()
        cancelled.remove(handle)
        lock.unlock()

        let (events, continuation) = AsyncThrowingStream<RunEvent, Error>.makeStream(
            bufferingPolicy: .bufferingNewest(64))

        // The session owns the simulated run until its stream reaches a
        // terminal event. Capturing weakly let a temporary runner disappear
        // before this task was scheduled under a busy full test suite, leaving
        // callers on a stream that could never finish.
        let task = Task {
            await self.drive(request: request, handle: handle, continuation: continuation)
        }

        let cancel: @Sendable () -> Void = { [weak self] in
            self?.markCancelled(handle)
            task.cancel()
        }

        return RunSession(handle: handle, events: events, cancel: cancel)
    }

    /// One subscriber per handle. Subscribing again replaces the sink, because
    /// the app has exactly one instrument panel and a second one would be a bug
    /// worth failing loudly on rather than fanning out to.
    func instrumentation(for handle: RunHandleID) -> AsyncStream<InstrumentSample> {
        lock.lock()
        defer { lock.unlock() }
        instrumentContinuations[handle]?.finish()
        let (stream, continuation) = AsyncStream<InstrumentSample>.makeStream(
            bufferingPolicy: .bufferingNewest(8))
        instrumentContinuations[handle] = continuation
        return stream
    }

    private func markCancelled(_ handle: RunHandleID) {
        lock.lock()
        cancelled.insert(handle)
        lock.unlock()
    }

    private func isCancelled(_ handle: RunHandleID) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled.contains(handle)
    }

    private func publish(_ sample: InstrumentSample, for handle: RunHandleID) {
        lock.lock()
        let continuation = instrumentContinuations[handle]
        lock.unlock()
        continuation?.yield(sample)
    }

    private func closeInstruments(_ handle: RunHandleID) {
        lock.lock()
        let continuation = instrumentContinuations.removeValue(forKey: handle)
        lock.unlock()
        continuation?.finish()
    }

    // swiftlint:disable:next function_body_length
    private func drive(
        request: RunRequest, handle: RunHandleID,
        continuation: AsyncThrowingStream<RunEvent, Error>.Continuation
    ) async {
        // The run owns the grant after `start` returns a session. Every exit —
        // success, cancellation, disconnect and short transfer — crosses this
        // one bracket. `StorageScope.release()` is idempotent, so deinit remains
        // a safe last resort without becoming a second stop-access call.
        defer {
            request.artifact.scope?.release()
            closeInstruments(handle)
        }
        let startedAt = Date()
        let profile = entry.memory
        let terms = entry.terms
        let layerCount = max(1, profile.layerCount)

        // The simulated pass length, and how much simulated time one wall tick
        // buys. The numbers the panel prints are on the simulated clock; only
        // the reviewer's patience is on the wall one.
        let passSeconds = terms.computeSecondsPerToken
            + Double(terms.totalBytesPerToken) / 0.92e9
        let tickHz = 2.0
        let ticks = max(1, Int(reviewSeconds * tickHz))
        let simulatedPerTick = passSeconds / Double(ticks)

        continuation.yield(
            .accepted(
                RunAcceptance(
                    handle: handle, model: request.model,
                    declaredBudgetBytes: request.memoryBudgetBytes,
                    pinPlan: request.pinPlan,
                    effectiveKnobs: request.knobs,
                    artifactRoot: request.artifact.root.path, startedAt: startedAt)))
        continuation.yield(.log("stated budget \(MRFormat.bytesDecimal(request.memoryBudgetBytes))"))
        continuation.yield(
            .log("artifact \(request.artifact.root.lastPathComponent) · \(layerCount) layers"))

        var ladder = (0..<layerCount).map { index in
            LayerCell(
                index: index, status: .pending,
                storedBytes: profile.widestDeterministicLayerStoredBytes,
                widenedBytes: profile.widenedResidentBytes, wallSeconds: nil,
                wasStaged: false)
        }
        // An unset knob means the runner chooses, and since spec v0.6.19 the
        // runner chooses to prefetch. A mock that modelled unset as off would
        // produce a serial timing for a run the real runner would stage.
        let readAheadSetting = ReadAheadSetting(request.knobs.deterministicReadAheadLayers)
        let readAheadOn = entry.id == .kimiK3
            && readAheadSetting.isProductSupported
            && readAheadSetting.effectiveDepth == 1

        var elapsed = 0.0
        var bytesRead: UInt64 = 0
        var deterministicBytes: UInt64 = 0
        var expertBytes: UInt64 = 0
        var peakFootprint: UInt64 = 0
        var thermalTrail: [ThermalTransition] = []
        var currentThermal: ThermalStateName = .nominal
        // Pass boundaries, published the way the real engines publish them:
        // prompt ingestion is its own pass with its own byte and time totals,
        // and a decode rate is only ever divided by the passes that came after
        // it. Without these the panel has no prefill stage to render and no
        // way to keep the completed-token average stable.
        var emitted: [TokenEvent] = []
        var lastTokenAt = 0.0
        var completedTokenBytes: ByteAccounting?
        var prefillBytes: ByteAccounting?
        var prefillSeconds: Double?

        func accounting() -> ByteAccounting {
            ByteAccounting(
                totalBytesRead: bytesRead,
                deterministicBytesRead: deterministicBytes,
                expertBytesRead: expertBytes)
        }

        func telemetry(phase: String, stage: RunGenerationStage) -> RunTelemetry {
            let footprint = footprintCurve(
                elapsed: elapsed, passSeconds: passSeconds, budget: request.memoryBudgetBytes,
                profile: profile, readAheadOn: readAheadOn)
            peakFootprint = max(peakFootprint, footprint)
            let decodeTokens = max(0, emitted.count - 1)
            let decodeSeconds = prefillSeconds.map { max(0, elapsed - $0) }
            let decodeBytes = completedTokenBytes.flatMap { completed in
                prefillBytes.flatMap { completed.subtracting($0) }
            }
            var decodeRate: Double?
            if decodeTokens > 0, let seconds = decodeSeconds, seconds > 0 {
                decodeRate = Double(decodeTokens) / seconds
            }
            return RunTelemetry(
                at: Date(), elapsed: elapsed, phase: phase,
                // Until a decode pass has completed there is no decode rate, and
                // a rate computed from zero tokens would be a zero that reads as
                // a measurement.
                tokensPerSecond: decodeRate,
                generationStage: stage,
                bytes: accounting(),
                bytesPerSecond: elapsed > 0 ? Double(bytesRead) / elapsed : nil,
                completedTokenBytes: completedTokenBytes,
                prefillBytes: prefillBytes,
                bytesPerToken: decodeTokens > 0
                    ? decodeBytes.map { $0.totalBytesRead / UInt64(decodeTokens) } : nil,
                prefillSeconds: prefillSeconds,
                decodeSeconds: decodeSeconds,
                decodeTokensCompleted: decodeTokens,
                declaredBudgetBytes: request.memoryBudgetBytes,
                footprintBytes: footprint, peakFootprintBytes: peakFootprint,
                residentBytes: UInt64(Double(footprint) * 0.94),
                availableBytes: nil,
                mlxActiveBytes: UInt64(Double(footprint) * 0.78),
                mlxCacheBytes: UInt64(request.knobs.mlxCacheLimitBytes ?? 0),
                mlxPeakBytes: UInt64(Double(peakFootprint) * 0.80),
                thermalState: currentThermal,
                lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
                batteryLevel: nil)
        }

        continuation.yield(
            .phase(RunPhase(name: "translating", fraction: 0, detail: "reading manifests")))
        try? await Task.sleep(for: .milliseconds(500))
        if isCancelled(handle) {
            finishCancelled(
                continuation: continuation, handle: handle, request: request,
                telemetry: telemetry(phase: "translating", stage: .preparing), elapsed: elapsed,
                thermalTrail: thermalTrail)
            return
        }
        continuation.yield(
            .phase(RunPhase(name: "opening", fraction: 0.02, detail: "opening containers")))
        try? await Task.sleep(for: .milliseconds(400))

        let deterministicPerLayer = terms.deterministicBytesPerToken / UInt64(layerCount)
        let expertPerLayer = terms.expertBytesPerToken / UInt64(layerCount)

        // The reply this turn produces. One entry per new token; K3's ceiling is
        // one, and a runner with a higher ceiling streams the rest at its own
        // pass boundaries.
        let reply = PreviewReplies.tokens(
            model: request.model, turnIndex: turnIndex,
            count: max(1, min(request.maximumNewTokens, capabilities.maximumNewTokens)))
        // A turn is passes, not one long slide. Prompt ingestion is one whole
        // walk of the layers and produces the first token; every later pass
        // walks them again and produces one more. The review clock gives
        // prefill a visible share of the run because that is also its real
        // shape — on a flagship artifact prompt ingestion is the longest single
        // pass of a turn, and it is the stretch during which the panel has only
        // prefill numbers to show.
        let passCount = max(1, reply.count)
        let decodePasses = passCount - 1
        let prefillTicks = decodePasses == 0
            ? ticks : max(1, Int((Double(ticks) * 0.4).rounded()))
        let decodeTicksPerPass = decodePasses == 0
            ? 0 : max(1, (ticks - prefillTicks) / decodePasses)
        let totalLayerSteps = Double(passCount * layerCount)

        for pass in 0..<passCount {
            let isPrefill = pass == 0
            let stage: RunGenerationStage = isPrefill ? .prefill : .decode
            let phaseName = isPrefill ? "prefill" : "decode \(pass)"
            let passTicks = isPrefill ? prefillTicks : decodeTicksPerPass

            for tick in 0..<passTicks {
                if Task.isCancelled || isCancelled(handle) {
                    finishCancelled(
                        continuation: continuation, handle: handle, request: request,
                        telemetry: telemetry(phase: phaseName, stage: stage), elapsed: elapsed,
                        thermalTrail: thermalTrail)
                    return
                }
                try? await Task.sleep(for: .milliseconds(Int(1000 / tickHz)))
                elapsed += simulatedPerTick

                // The ladder cursor is a position in THIS pass — every pass
                // reads every layer — while the byte totals accumulate across
                // all of them.
                let passFraction = Double(tick + 1) / Double(passTicks)
                let layerCursor = min(layerCount - 1, Int(passFraction * Double(layerCount)))
                let layerSteps = pass * layerCount + layerCursor + 1
                let runFraction = min(1, Double(layerSteps) / totalLayerSteps)

                for index in 0..<layerCount {
                    if index < layerCursor {
                        ladder[index].status = .done
                        ladder[index].wallSeconds =
                            simulatedPerTick * Double(passTicks) / Double(layerCount)
                        ladder[index].wasStaged = readAheadOn
                    } else if index == layerCursor {
                        ladder[index].status = .computing
                    } else if readAheadOn && index == layerCursor + 1 {
                        ladder[index].status = .staged
                    } else {
                        ladder[index].status = .pending
                    }
                }

                deterministicBytes = deterministicPerLayer * UInt64(layerSteps)
                expertBytes = expertPerLayer * UInt64(layerSteps)
                bytesRead = deterministicBytes &+ expertBytes

                // The `Stalls` pair, kept consistent with the read-ahead arm:
                // with depth 1 the model-layer read hides behind compute, so the
                // loop waits less on storage.
                let stalls = StallAccounting(
                    computeSeconds: elapsed * (readAheadOn ? 0.72 : 0.43),
                    computeWaitedOnIOSeconds: elapsed * (readAheadOn ? 0.11 : 0.55),
                    wallSeconds: elapsed,
                    achievedBytesPerSecond: elapsed > 0 ? Double(bytesRead) / elapsed : 0)
                let cache = CacheStrip(
                    policy: "lru",
                    hits: Int(runFraction * 1_840), misses: Int(runFraction * 9_120),
                    evictions: Int(runFraction * 640),
                    rejectedAdmissions: readAheadOn ? 0 : Int(runFraction * 24),
                    pinnedExpertCount: 6)

                if case .thermalEscalation = fault, runFraction > 0.55,
                    currentThermal != .serious
                {
                    currentThermal = .serious
                    thermalTrail.append(
                        ThermalTransition(state: .serious, atSeconds: elapsed, phase: phaseName))
                } else if runFraction > 0.30 && currentThermal == .nominal && fault == nil {
                    currentThermal = .fair
                    thermalTrail.append(
                        ThermalTransition(state: .fair, atSeconds: elapsed, phase: phaseName))
                }

                let sample = telemetry(phase: phaseName, stage: stage)
                continuation.yield(
                    .phase(
                        RunPhase(
                            name: phaseName, generationStage: stage, fraction: passFraction,
                            detail: "layer \(layerCursor + 1) of \(layerCount)")))
                continuation.yield(.telemetry(sample))

                publish(
                    InstrumentSample(
                        at: Date(), ladder: ladder,
                        flow: FlowSample(
                            at: Date(),
                            deterministicBytesPerSecond: Double(deterministicPerLayer)
                                / max(0.001, simulatedPerTick) * (readAheadOn ? 0.45 : 1.0),
                            expertBytesPerSecond: Double(expertPerLayer)
                                / max(0.001, simulatedPerTick)),
                        stalls: stalls, cache: cache,
                        readAhead: readAheadOn
                            ? ReadAheadAccounting(
                                depth: 1,
                                stagedBytes: profile.stagedPairPeakBytes,
                                peakStagedBytes: profile.stagedPairPeakBytes,
                                materialisedStagedBytes: UInt64(
                                    Double(profile.stagedPairPeakBytes) * 0.86),
                                discardedStagedBytes: UInt64(
                                    Double(profile.stagedPairPeakBytes) * 0.09),
                                outstandingStagedBytes: profile.stagedPairPeakBytes
                                    - UInt64(Double(profile.stagedPairPeakBytes) * 0.86)
                                    - UInt64(Double(profile.stagedPairPeakBytes) * 0.09),
                                prefetchWaitSeconds: elapsed * 0.004,
                                stageReadSeconds: elapsed * 0.62,
                                nonFreeSlots: 0)
                            : nil),
                    for: handle)

                // Faults strike at a layer boundary, which is also the only
                // place a Stop can take effect.
                if let fault {
                    switch fault {
                    case .volumeDisconnects(let volume, let atLayer) where layerCursor >= atLayer:
                        let error = StorageCoreError.posix(
                            operation: "pread",
                            path: "/Volumes/\(volume)/\(entry.descriptor.id.rawValue)/layer\(atLayer)/layer\(atLayer)-w1.mxfp4tile",
                            code: EIO)
                        continuation.yield(.log(error.description))
                        continuation.finish(throwing: error)
                        return
                    case .shortRead(let atLayer) where layerCursor >= atLayer:
                        let error = StorageCoreError.shortTransfer(
                            operation: "pread",
                            path: "\(request.artifact.root.path)/layer\(atLayer)/layer\(atLayer)-w1.mxfp4tile",
                            offset: 1_073_741_824, expected: 4_194_304, actual: 2_097_152)
                        continuation.yield(.log(error.description))
                        continuation.finish(throwing: error)
                        return
                    default:
                        break
                    }
                }
            }

            // Tokens arrive at PASS boundaries, not per step, and the byte and
            // time totals for the pass are sealed in the same breath — prefill's
            // separately, because prompt ingestion is not decode throughput.
            completedTokenBytes = accounting()
            if isPrefill {
                prefillBytes = completedTokenBytes
                prefillSeconds = elapsed
            }
            // A pass boundary is also where the real engines publish where the
            // pass went. The shares below are the shape the measured records
            // have — storage wait dominates, and the residual is large — not a
            // decomposition of anything this runner did, which read no bytes.
            continuation.yield(
                .phaseSummary(
                    Self.previewPhaseSummary(
                        model: request.model, passIndex: pass,
                        passSeconds: max(0.001, elapsed - lastTokenAt),
                        layerCount: layerCount, readAheadOn: readAheadOn)))

            if pass < reply.count {
                let next = reply[pass]
                let event = TokenEvent(
                    index: pass, tokenID: next.tokenID, text: next.text,
                    secondsSincePreviousToken: elapsed - lastTokenAt,
                    isPrefillToken: isPrefill, top1Top2Gap: 0.8024)
                lastTokenAt = elapsed
                emitted.append(event)
                continuation.yield(.token(event))
            }
        }

        if case .budgetOvershoot(let over) = fault {
            peakFootprint = request.memoryBudgetBytes + over
        }

        let final = telemetry(phase: "finished", stage: .terminal)
        let replyText = emitted.compactMap { $0.text }.joined()
        let summary = RunSummary(
            handle: handle, model: request.model, tokenIDs: emitted.map { $0.tokenID },
            text: replyText.isEmpty ? nil : replyText,
            wallSeconds: elapsed,
            tokensPerSecond: elapsed > 0 ? Double(emitted.count) / elapsed : nil,
            finalTelemetry: final, peakFootprintBytes: peakFootprint,
            budgetRespected: peakFootprint <= request.memoryBudgetBytes,
            thermalTrail: thermalTrail,
            resultJSON: resultJSON(
                request: request, elapsed: elapsed, peak: peakFootprint,
                bytesRead: bytesRead, tokenIDs: emitted.map { $0.tokenID }),
            resultJSONFilename:
                "\(entry.descriptor.id.rawValue)-decode-\(Int(startedAt.timeIntervalSince1970)).json",
            headline:
                "\(entry.descriptor.displayName): "
                + (emitted.count == 1
                    ? "token \(emitted[0].tokenID)" : "\(emitted.count) tokens")
                + " in \(MRFormat.clock(elapsed)), "
                + "peak \(MRFormat.bytesDecimal(peakFootprint)) inside "
                + "\(MRFormat.bytesDecimal(request.memoryBudgetBytes))",
            lines: [
                "data read \(MRFormat.bytesDecimal(bytesRead))",
                "model layers \(MRFormat.bytesDecimal(deterministicBytes)) · experts "
                    + "\(MRFormat.bytesDecimal(expertBytes))",
            ],
            gitRevision: BuildRevision.current)
        continuation.yield(.telemetry(final))
        continuation.yield(.finished(summary))
        continuation.finish()
    }

    private func finishCancelled(
        continuation: AsyncThrowingStream<RunEvent, Error>.Continuation,
        handle: RunHandleID, request: RunRequest, telemetry: RunTelemetry, elapsed: Double,
        thermalTrail: [ThermalTransition]
    ) {
        // A cancelled run ends as a named failure class, never as a silently
        // short result: the partial telemetry is reported, not discarded.
        let summary = RunSummary(
            handle: handle, model: request.model, tokenIDs: [], text: nil,
            wallSeconds: elapsed, tokensPerSecond: nil, finalTelemetry: telemetry,
            peakFootprintBytes: telemetry.peakFootprintBytes,
            budgetRespected: telemetry.budgetRespected, thermalTrail: thermalTrail,
            resultJSON: nil, resultJSONFilename: nil,
            headline: "Stopped at a layer boundary after \(MRFormat.clock(elapsed)).",
            lines: ["the run was cancelled; the telemetry above is what it had reached"],
            gitRevision: BuildRevision.current)
        continuation.yield(.cancelled(summary))
        continuation.finish()
    }

    /// A decomposition shaped like the measured ones, for a run that measured
    /// nothing.
    ///
    /// K3's terms and V4's are different sets, and the preview publishes each
    /// model's own: a panel that drew V4's fifteen terms for a K3 run would
    /// review a screen the product cannot produce. The shares come from the
    /// records — `2026-08-11-expert-overlap.md` for K3's storage-bound expert
    /// phase, `2026-08-15-v4-residual-attributed.md` for V4's — and the terms
    /// are deliberately left to sum to less than the pass, because that
    /// residual is the thing the section exists to show.
    private static func previewPhaseSummary(
        model: ModelID, passIndex: Int, passSeconds: Double,
        layerCount: Int, readAheadOn: Bool
    ) -> RunPhaseSummary {
        func term(_ name: String, _ share: Double, count: Int? = nil) -> RunPhaseTerm {
            RunPhaseTerm(name: name, seconds: passSeconds * share, count: count)
        }
        let terms: [RunPhaseTerm]
        switch model {
        case .deepseekV4Flash:
            terms = [
                term(RunPhaseTermName.deterministicRead, 0.11, count: layerCount * 3),
                term(RunPhaseTermName.tileDigest, 0.02, count: layerCount),
                term(RunPhaseTermName.expertIOWait, 0.09, count: layerCount * 8),
                term(RunPhaseTermName.expertGatherCompute, 0.04, count: layerCount),
                term(RunPhaseTermName.expertOther, 0.01),
                term(RunPhaseTermName.outputHeadRead, 0.06, count: 4),
                term(RunPhaseTermName.outputHeadCompute, 0.03),
                term(RunPhaseTermName.reclaim, 0.02, count: layerCount),
                term(RunPhaseTermName.expertBackendLifecycle, 0.05, count: layerCount),
                term(RunPhaseTermName.layerArtifactSetup, 0.04, count: layerCount),
                term(RunPhaseTermName.tileAdoption, 0.07, count: layerCount * 3),
                term(RunPhaseTermName.activationScaleSync, 0.06, count: 480),
                term(RunPhaseTermName.finitenessSweep, 0.01, count: 553),
                // Routing and the indexer are milliseconds of host sort once
                // the wait behind them is its own term, and the preview says
                // so: a mock that kept the old shares would train the eye on a
                // split the real engine no longer produces.
                term(RunPhaseTermName.routingSelect, 0.002, count: layerCount),
                term(RunPhaseTermName.sparseAttention, 0.05, count: layerCount),
                term(RunPhaseTermName.lightningIndexer, 0.001, count: layerCount),
                term(RunPhaseTermName.gpuWait, 0.19, count: layerCount * 2 + 33),
                term(RunPhaseTermName.gpuSubmit, 0.09, count: layerCount * 5),
            ]
        default:
            terms = [
                term(RunPhaseTermName.deterministicRead, readAheadOn ? 0.03 : 0.24),
                term(RunPhaseTermName.stagerWait, readAheadOn ? 0.05 : 0),
                term(
                    RunPhaseTermName.expertIOWait, readAheadOn ? 0.41 : 0.34,
                    count: layerCount * 48),
                term(RunPhaseTermName.expertGatherCompute, 0.14, count: layerCount * 3),
                term(RunPhaseTermName.expertOther, 0.03),
                term(RunPhaseTermName.attentionBarrier, 0.06, count: layerCount),
                term(RunPhaseTermName.reclaim, 0.01, count: layerCount),
                RunPhaseTerm(
                    name: RunPhaseTermName.stagedRead,
                    seconds: passSeconds * (readAheadOn ? 0.45 : 0), isInsidePass: false),
                RunPhaseTerm(
                    name: RunPhaseTermName.layerTotal, seconds: passSeconds * 0.93,
                    count: layerCount, isInsidePass: false),
            ]
        }
        return RunPhaseSummary(
            passKind: passIndex == 0 ? .prefill : .decode,
            passIndex: passIndex, passSeconds: passSeconds, terms: terms)
    }

    /// The footprint curve: the floor is resident almost immediately, the
    /// staged pair adds a second step when read-ahead is on, and the hot set
    /// creeps. Never above the stated budget unless a fault says so.
    private func footprintCurve(
        elapsed: Double, passSeconds: Double, budget: UInt64, profile: MemoryProfile,
        readAheadOn: Bool
    ) -> UInt64 {
        let fraction = passSeconds > 0 ? min(1, elapsed / passSeconds) : 0
        let ramp = min(1, fraction * 8)
        var bytes = Double(profile.arithmeticFloorBytes) * ramp
        if readAheadOn {
            bytes += Double(profile.stagedPairPeakBytes) * min(1, max(0, (fraction - 0.05) * 6))
        }
        bytes += Double(profile.expertStoredBytes) * 6 * fraction
        return UInt64(min(Double(budget) * 0.93, bytes))
    }

    private func resultJSON(
        request: RunRequest, elapsed: Double, peak: UInt64, bytesRead: UInt64,
        tokenIDs: [Int]
    ) -> Data? {
        let object: [String: Any] = [
            "model": request.model.rawValue,
            "declaredBudgetBytes": request.memoryBudgetBytes,
            "peakFootprintBytes": peak,
            "wallSeconds": elapsed,
            "bytesRead": bytesRead,
            "tokenIDs": tokenIDs,
            "gitRevision": BuildRevision.current,
            "source": "preview runner — no bytes were read",
        ]
        return try? JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
    }
}

/// The build revision every result carries. Real when the build was stamped,
/// and honest about it when it was not.
enum BuildRevision {
    static let current: String = {
        Bundle.main.object(forInfoDictionaryKey: "MinirunGitRevision") as? String
            ?? "unstamped-working-tree"
    }()
}
#endif
