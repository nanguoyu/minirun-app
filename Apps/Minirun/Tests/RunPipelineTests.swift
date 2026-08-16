import MinirunKit
import MinirunRunners
import StorageCore
import XCTest
import iOSBench

@testable import MinirunApp

/// One end-to-end trip through the runner seam, plus the refusals that happen
/// before a byte is read.
///
/// The mock reads nothing, so what is under test is the pipeline: that the
/// facade refuses when it should, that the stream produces the events the panel
/// renders, and that a cancelled run ends as a named class rather than as a
/// silently short result.
@MainActor
final class RunPipelineTests: XCTestCase {

    private func request(
        budget: UInt64 = CatalogFixtures.k3ProductMinimumBudget,
        tokens: Int = 1,
        knobs: RunKnobs = RunKnobs(),
        scope: StorageScope? = nil
    ) -> RunRequest {
        RunRequest(
            model: .kimiK3,
            artifact: ArtifactReference(
                root: URL(fileURLWithPath: "/tmp/k3"), scope: scope),
            prompt: .tokenIDs([17529]), memoryBudgetBytes: budget,
            maximumNewTokens: tokens, knobs: knobs,
            workingDirectory: URL(fileURLWithPath: NSTemporaryDirectory()))
    }

    // MARK: Refusals before a byte is read

    func testBudgetBelowTheMinimumIsRefusedByName() {
        let runner = MockDecodeRunner(entry: CatalogFixtures.kimiK3)
        XCTAssertThrowsError(try runner.validate(request(budget: 4_000_000_000))) { error in
            guard case RunError.budgetBelowMinimum(let declared, let minimum, let model)? =
                error as? RunError
            else { return XCTFail("expected budgetBelowMinimum, got \(error)") }
            XCTAssertEqual(declared, 4_000_000_000)
            XCTAssertEqual(minimum, 8_000_000_000)
            XCTAssertEqual(model, .kimiK3)
        }
    }

    /// A knob the runner cannot honour is refused BY NAME, never dropped.
    func testUnsupportedKnobIsRefusedByName() {
        let runner = MockDecodeRunner(entry: CatalogFixtures.kimiK3)
        var knobs = RunKnobs()
        knobs.expertPoolSlots = 8  // a V4 knob, on a K3 runner
        XCTAssertThrowsError(try runner.validate(request(knobs: knobs))) { error in
            guard case RunError.knobNotSupported(let name, _)? = error as? RunError else {
                return XCTFail("expected knobNotSupported, got \(error)")
            }
            XCTAssertEqual(name, "expertPoolSlots")
        }
    }

    func testTextPromptIsRefusedRatherThanTokenizedByAGuess() {
        let runner = MockDecodeRunner(entry: CatalogFixtures.kimiK3)
        let textual = RunRequest(
            model: .kimiK3, artifact: ArtifactReference(root: URL(fileURLWithPath: "/tmp/k3")),
            prompt: .text("北京"), memoryBudgetBytes: CatalogFixtures.k3ProductMinimumBudget,
            maximumNewTokens: 1, workingDirectory: URL(fileURLWithPath: NSTemporaryDirectory()))
        XCTAssertThrowsError(try runner.validate(textual)) { error in
            guard case RunError.promptNotSupported? = error as? RunError else {
                return XCTFail("expected promptNotSupported, got \(error)")
            }
        }
    }

    /// K3's product ceiling is finite; asking for more is
    /// refused rather than quietly truncated.
    func testTooManyTokensIsRefused() {
        let runner = MockDecodeRunner(entry: CatalogFixtures.kimiK3)
        // The ceiling is the platform's product policy, not a literal: 64 on
        // the supported Mac and 2 on the bounded iPhone tier (ADR 0011).
        let ceiling = K3ProductMemoryBudget.currentPolicy.maximumNewTokens
        XCTAssertThrowsError(try runner.validate(request(tokens: ceiling + 1))) { error in
            guard case RunError.tooManyTokens(let requested, let maximum, _)? =
                error as? RunError
            else { return XCTFail("expected tooManyTokens, got \(error)") }
            XCTAssertEqual(requested, ceiling + 1)
            XCTAssertEqual(maximum, ceiling)
        }
    }

    func testTheProductMinimumBudgetIsAccepted() {
        XCTAssertNoThrow(
            try MockDecodeRunner(entry: CatalogFixtures.kimiK3).validate(request()))
    }

    func testK3PreviewRefusesAnUnimplementedReadAheadDepthBeforeStarting() {
        var knobs = RunKnobs()
        knobs.deterministicReadAheadLayers = 2
        XCTAssertThrowsError(
            try MockDecodeRunner(entry: CatalogFixtures.kimiK3).validate(
                request(knobs: knobs))
        ) { error in
            guard case RunError.knobOutOfRange(let name, let value, let allowed)? =
                error as? RunError
            else { return XCTFail("expected knobOutOfRange, got \(error)") }
            XCTAssertEqual(name, "deterministicReadAheadLayers")
            XCTAssertEqual(value, "2")
            XCTAssertTrue(allowed.contains("0"))
            XCTAssertTrue(allowed.contains("1"))
        }
    }

    // MARK: A whole run

    func testARunProducesAcceptancePhasesTelemetryAndATokenAtThePassBoundary() async throws {
        let runner = MockDecodeRunner(entry: CatalogFixtures.kimiK3, reviewSeconds: 1.5)
        let session = try runner.start(request())

        var accepted: RunAcceptance?
        var telemetry: [RunTelemetry] = []
        var tokens: [TokenEvent] = []
        var summary: RunSummary?

        for try await event in session.events {
            switch event {
            case .accepted(let value): accepted = value
            case .telemetry(let value): telemetry.append(value)
            case .token(let value): tokens.append(value)
            case .finished(let value): summary = value
            default: break
            }
        }

        XCTAssertEqual(accepted?.declaredBudgetBytes, CatalogFixtures.k3ProductMinimumBudget)
        XCTAssertFalse(telemetry.isEmpty, "the panel has nothing to render without telemetry")
        XCTAssertEqual(tokens.count, 1, "K3 emits one token, at the pass boundary")
        XCTAssertEqual(tokens.first?.tokenID, 3372)

        let final = try XCTUnwrap(summary)
        XCTAssertEqual(final.tokenIDs, [3372])
        XCTAssertNil(final.logitsDigest, "the preview runner computed no logits")
        XCTAssertTrue(final.budgetRespected)
        XCTAssertLessThanOrEqual(
            final.peakFootprintBytes, CatalogFixtures.k3ProductMinimumBudget)
        XCTAssertNotNil(final.resultJSON)

        // Every byte lands in exactly one bucket.
        XCTAssertTrue(final.finalTelemetry.bytes.isBalanced)
        // The peak only ever moves right.
        let peaks = telemetry.map(\.peakFootprintBytes)
        XCTAssertEqual(peaks, peaks.sorted(), "the peak watermark moved backwards")
    }

    /// Reporting a breach honestly beats hiding it: the run completes, the
    /// number is written down, and `budgetRespected` is false forever after.
    func testABreachedBudgetIsReportedAndTheRunIsNotDiscarded() async throws {
        let runner = MockDecodeRunner(entry: CatalogFixtures.kimiK3, reviewSeconds: 1)
        runner.fault = .budgetOvershoot(byBytes: 214_748_364)
        let session = try runner.start(request())

        var summary: RunSummary?
        for try await event in session.events {
            if case .finished(let value) = event { summary = value }
        }
        let final = try XCTUnwrap(summary)
        XCTAssertFalse(final.budgetRespected)
        XCTAssertEqual(
            final.peakFootprintBytes,
            CatalogFixtures.k3ProductMinimumBudget + 214_748_364)
        XCTAssertEqual(final.tokenIDs, [3372], "the run still completed")
    }

    /// A disconnect preserves the storage layer's operation, path and errno.
    /// The description remains verbatim, but it is no longer the data model.
    func testAVolumeDisconnectEndsTheStreamWithTheStructuredPosixError() async {
        let runner = MockDecodeRunner(entry: CatalogFixtures.kimiK3, reviewSeconds: 1)
        runner.fault = .volumeDisconnects(volume: "MINIRUN-NVME", atLayer: 4)
        do {
            let session = try runner.start(request())
            for try await _ in session.events {}
            XCTFail("expected the stream to fail")
        } catch let error as StorageCoreError {
            guard case .posix(let operation, let path, let code) = error else {
                return XCTFail("expected posix, got \(error)")
            }
            XCTAssertEqual(operation, "pread")
            XCTAssertTrue(path.hasPrefix("/Volumes/MINIRUN-NVME/"))
            XCTAssertEqual(code, EIO)
            XCTAssertTrue(error.description.contains("pread failed"))
            XCTAssertTrue(error.description.contains("/Volumes/MINIRUN-NVME/"))
        } catch {
            XCTFail("expected a StorageCoreError, got \(error)")
        }
    }

    /// Cancellation ends as a named class carrying partial telemetry, never as
    /// a short result that looks like a finished one.
    func testCancellationReportsPartialTelemetry() async throws {
        let runner = MockDecodeRunner(entry: CatalogFixtures.kimiK3, reviewSeconds: 6)
        let session = try runner.start(request())

        let collector = Task { () -> RunSummary? in
            var cancelled: RunSummary?
            for try await event in session.events {
                if case .cancelled(let value) = event { cancelled = value }
            }
            return cancelled
        }
        try await Task.sleep(for: .milliseconds(900))
        session.cancel()

        let summary = try await collector.value
        let final = try XCTUnwrap(summary, "a cancelled run must still report a summary")
        XCTAssertTrue(final.tokenIDs.isEmpty)
        XCTAssertTrue(final.headline.contains("Stopped at a layer boundary"))
    }

    func testControllerShowsStoppingUntilSafeTeardownAndReleasesItsDecoder() async throws {
        let controller = RunController()
        let ended = expectation(description: "cancelled turn ended")
        controller.onTurnEnded = { _ in ended.fulfill() }

        var decoder: DecoderLifetimeProbe? = DecoderLifetimeProbe()
        weak var retainedDecoder = decoder
        controller.start(
            runner: MockDecodeRunner(entry: CatalogFixtures.kimiK3, reviewSeconds: 6),
            request: request(
                tokens: min(8, K3ProductMemoryBudget.currentPolicy.maximumNewTokens)),
            linkCeiling: nil,
            layerCount: 93, storedBytesPerLayer: 1, widenedBytesPerLayer: 2,
            decodeTokens: { [decoder] ids in decoder?.decode(ids) ?? "" })
        decoder = nil

        XCTAssertTrue(controller.isRunning)
        XCTAssertFalse(controller.isStopping)
        XCTAssertNotNil(retainedDecoder, "the active turn dropped its verified decoder")

        controller.stop(clearSnapshot: false)
        XCTAssertTrue(controller.isRunning, "Stop claimed completion before runner teardown")
        XCTAssertTrue(controller.isStopping)
        controller.stop(clearSnapshot: false)  // A second click is an idempotent no-op.

        await fulfillment(of: [ended], timeout: 10)
        XCTAssertEqual(controller.snapshot.state, .cancelled)
        XCTAssertFalse(controller.isRunning)
        XCTAssertFalse(controller.isStopping)
        XCTAssertNil(
            retainedDecoder,
            "a terminal turn retained its tokenizer/vocabulary closure in memory")
    }

    func testARefusedTurnDoesNotRetainItsDecoder() {
        let controller = RunController()
        var decoder: DecoderLifetimeProbe? = DecoderLifetimeProbe()
        weak var retainedDecoder = decoder
        controller.start(
            runner: MockDecodeRunner(entry: CatalogFixtures.kimiK3),
            request: request(budget: 1), linkCeiling: nil,
            layerCount: 93, storedBytesPerLayer: 1, widenedBytesPerLayer: 2,
            decodeTokens: { [decoder] ids in decoder?.decode(ids) ?? "" })
        decoder = nil

        guard case .refused = controller.snapshot.state else {
            return XCTFail("the invalid request was not refused")
        }
        XCTAssertNil(retainedDecoder, "a refused turn retained its decoder")
    }

    // MARK: The controller

    func testControllerLatchesARefusalWithoutStartingAnything() {
        let controller = RunController()
        let runner = MockDecodeRunner(entry: CatalogFixtures.kimiK3)
        controller.start(
            runner: runner, request: request(budget: 1_000_000), linkCeiling: nil,
            layerCount: 93, storedBytesPerLayer: 1, widenedBytesPerLayer: 2)
        guard case .refused(let error) = controller.snapshot.state else {
            return XCTFail("expected the controller to latch a refusal")
        }
        // The kit spells its refusals as sentences rather than as case names,
        // and the sentence is what the refusal card shows. What matters is that
        // it names both numbers and the model, so an operator can see the gap
        // without opening a log.
        guard case RunError.budgetBelowMinimum(let declared, let minimum, let model) = error else {
            return XCTFail("expected budgetBelowMinimum, got \(error)")
        }
        XCTAssertEqual(declared, 1_000_000)
        XCTAssertEqual(model, .kimiK3)
        XCTAssertTrue(error.description.contains("\(declared)"))
        XCTAssertTrue(error.description.contains("\(minimum)"))
        XCTAssertTrue(error.description.contains(model.rawValue))
        XCTAssertNil(controller.statedBudgetBytes, "nothing was stated, so nothing was started")
    }

    func testControllerKeepsEveryShortTransferFieldWithoutParsingErrorText() async throws {
        let error = StorageCoreError.shortTransfer(
            operation: "readv",
            path: "/Volumes/FIELD-TEST/a path/weights.bin",
            offset: 987_654_321,
            expected: 12_345,
            actual: 678)
        let controller = RunController()
        controller.start(
            runner: ImmediateStorageFailureRunner(error), request: request(), linkCeiling: nil,
            layerCount: 3, storedBytesPerLayer: 1, widenedBytesPerLayer: 2)

        for _ in 0..<100 where controller.snapshot.fault == nil {
            try await Task.sleep(for: .milliseconds(10))
        }
        let fault = try XCTUnwrap(controller.snapshot.fault)
        guard case .shortRead(
            let operation, let path, let offset, let expected, let actual) = fault.kind
        else { return XCTFail("expected shortRead, got \(fault.kind)") }
        XCTAssertEqual(operation, "readv")
        XCTAssertEqual(path, "/Volumes/FIELD-TEST/a path/weights.bin")
        XCTAssertEqual(offset, 987_654_321)
        XCTAssertEqual(expected, 12_345)
        XCTAssertEqual(actual, 678)
        XCTAssertEqual(fault.namedError, error.description)
        XCTAssertNil(
            fault.bytesReadSoFar,
            "a runner that emitted no byte measurement must not be presented as reading zero")
    }

    func testControllerShowsOnlyARunnerSuppliedDigest() async throws {
        let preview = RunController()
        let previewEnded = expectation(description: "preview ended")
        preview.onTurnEnded = { _ in previewEnded.fulfill() }
        preview.start(
            runner: MockDecodeRunner(entry: CatalogFixtures.kimiK3, reviewSeconds: 0.02),
            request: request(), linkCeiling: nil,
            layerCount: 2, storedBytesPerLayer: 1, widenedBytesPerLayer: 2,
            previousDigest: CatalogFixtures.deviceLogitsDigest)
        await fulfillment(of: [previewEnded], timeout: 10)
        XCTAssertEqual(preview.snapshot.state, .finished)
        XCTAssertNil(preview.snapshot.reproducibility)
        XCTAssertNil(preview.snapshot.summary?.logitsDigest)

        let measured = RunController()
        let measuredEnded = expectation(description: "measured run ended")
        measured.onTurnEnded = { _ in measuredEnded.fulfill() }
        let digest = String(repeating: "a", count: 64)
        measured.start(
            runner: ImmediateDigestRunner(digest: digest), request: request(), linkCeiling: nil,
            layerCount: 1, storedBytesPerLayer: 1, widenedBytesPerLayer: 2,
            previousDigest: digest)
        await fulfillment(of: [measuredEnded], timeout: 2)
        XCTAssertEqual(measured.snapshot.reproducibility?.logitsDigest, digest)
        XCTAssertEqual(measured.snapshot.reproducibility?.matchesPreviousRun, true)
    }

    func testControllerConsumesRunnerOwnedPayloadFlow() async throws {
        let controller = RunController()
        controller.start(
            runner: ImmediatePayloadFlowRunner(), request: request(), linkCeiling: nil,
            layerCount: 2, storedBytesPerLayer: 1, widenedBytesPerLayer: 2)

        for _ in 0..<100 where controller.snapshot.flow.isEmpty {
            try await Task.sleep(for: .milliseconds(10))
        }
        let sample = try XCTUnwrap(controller.snapshot.flow.last)
        XCTAssertEqual(sample.sequence, 7)
        XCTAssertEqual(sample.deterministicBytesPerSecond, 600, accuracy: 0.0001)
        XCTAssertEqual(sample.expertBytesPerSecond, 200, accuracy: 0.0001)
        controller.clear()
    }

    /// The pass decomposition travels the same stream as the tokens, and the
    /// controller keeps every pass rather than the last one: a turn's decode
    /// aggregate is only honest if none of its passes was overwritten.
    func testControllerRecordsOnePhaseSummaryPerPass() async throws {
        let controller = RunController()
        let ended = expectation(description: "turn ended")
        controller.onTurnEnded = { _ in ended.fulfill() }
        let runner = MockDecodeRunner(entry: CatalogFixtures.kimiK3, reviewSeconds: 1.5)
        let tokens = min(2, K3ProductMemoryBudget.currentPolicy.maximumNewTokens)
        controller.start(
            runner: runner, request: request(tokens: tokens), linkCeiling: nil,
            layerCount: 4, storedBytesPerLayer: 1, widenedBytesPerLayer: 2)

        await fulfillment(of: [ended], timeout: 30)
        let snapshot = controller.snapshot
        XCTAssertEqual(snapshot.phaseSummaries.count, tokens)
        XCTAssertEqual(snapshot.phaseSummaries.first?.passKind, .prefill)
        XCTAssertNotNil(snapshot.prefillPhaseSummary)
        for summary in snapshot.phaseSummaries {
            XCTAssertTrue(summary.isBalanced, "the terms do not fit inside the pass")
            XCTAssertGreaterThan(summary.passSeconds, 0)
            XCTAssertNotNil(summary.seconds(of: RunPhaseTermName.expertIOWait))
        }
        controller.clear()
    }

    /// The iOS grant is one bracket around the session. Every terminal path
    /// crosses it, and an extra defensive release remains a no-op rather than a
    /// second `stopAccessingSecurityScopedResource()`.
    func testSecurityScopeReleasesExactlyOnceOnRefusalFailureDisconnectAndCancel() async throws {
        do {
            let coding = CountingScopeCoding()
            let scope = StorageScope(url: URL(fileURLWithPath: "/tmp/k3"), coding: coding)
            XCTAssertThrowsError(
                try MockDecodeRunner(entry: CatalogFixtures.kimiK3).start(
                    request(budget: 1, scope: scope)))
            assertReleasedOnce(scope, coding)
        }

        for fault: MockDecodeRunner.Fault in [
            .shortRead(atLayer: 0),
            .volumeDisconnects(volume: "SCOPE-TEST", atLayer: 0),
        ] {
            let coding = CountingScopeCoding()
            let scope = StorageScope(url: URL(fileURLWithPath: "/tmp/k3"), coding: coding)
            let runner = MockDecodeRunner(entry: CatalogFixtures.kimiK3, reviewSeconds: 0.05)
            runner.fault = fault
            do {
                let session = try runner.start(request(scope: scope))
                for try await _ in session.events {}
                XCTFail("expected \(fault) to fail the stream")
            } catch is StorageCoreError {
                // Expected. The assertion below is the lifecycle gate.
            }
            assertReleasedOnce(scope, coding)
        }

        do {
            let coding = CountingScopeCoding()
            let scope = StorageScope(url: URL(fileURLWithPath: "/tmp/k3"), coding: coding)
            let runner = MockDecodeRunner(entry: CatalogFixtures.kimiK3, reviewSeconds: 10)
            let session = try runner.start(request(scope: scope))
            session.cancel()
            for try await _ in session.events {}
            assertReleasedOnce(scope, coding)
        }
    }

    private func assertReleasedOnce(
        _ scope: StorageScope, _ coding: CountingScopeCoding,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertTrue(scope.isReleased, file: file, line: line)
        XCTAssertEqual(coding.starts, 1, file: file, line: line)
        XCTAssertEqual(coding.stops, 1, file: file, line: line)
        scope.release()
        XCTAssertEqual(coding.stops, 1, "a second release decremented the grant", file: file, line: line)
    }

    /// UPDATED for conversations: a budget is now stated by a CHAT, not by a
    /// global per-model setting. The claim under test is unchanged — whatever
    /// was stated is what the request carries, refused or not.
    func testAConversationsStatedBudgetReachesTheRequestUnchanged() {
        let model = AppModel.preview()
        let id = model.newConversation()!
        model.setBudget(3_000_000_000, in: id)
        let conversation = model.conversation(id)!
        let plan = model.budgetPlan(for: conversation)
        XCTAssertEqual(plan?.budgetBytes, 3_000_000_000)
        XCTAssertFalse(plan?.isRunnable ?? true)
        XCTAssertEqual(model.makeRequest(for: conversation)?.memoryBudgetBytes, 3_000_000_000)
    }

    func testAConversationHandsTheExactDisplayedPlanToTheRunnerRequest() throws {
        let model = AppModel.preview()
        let id = try XCTUnwrap(model.newConversation())
        model.setBudget(8 << 30, in: id)
        let conversation = try XCTUnwrap(model.conversation(id))
        let displayed = try XCTUnwrap(model.budgetPlan(for: conversation)?.pinPlan)
        let request = try XCTUnwrap(model.makeRequest(for: conversation))
        XCTAssertEqual(request.pinPlan, displayed)
        XCTAssertEqual(request.pinPlan?.budgetBytes, request.memoryBudgetBytes)
        XCTAssertEqual(request.pinPlan?.requestedMaximumNewTokens, request.maximumNewTokens)
        XCTAssertEqual(displayed.requestedMaximumNewTokens, conversation.settings.maximumNewTokens)
    }

    /// A V4 request states a budget and no `PinPlan`, and that is still true
    /// now that the dial has one to show.
    ///
    /// The two are not in tension. `supportsPinPlan` is false for V4 because
    /// the runner builds its own residency plan from the budget it was handed —
    /// against the exact opened artifact, not a screen's projection — so
    /// sending one would be the App deciding what the run holds. What changed
    /// is only that the dial can now draw the same ladder the runner will walk.
    func testV4RequestCarriesTheBudgetWithoutSendingAResidencyPlan() throws {
        let model = AppModel.freshForTests(installedModel: .deepseekV4Flash)
        model.setDefaultModel(.deepseekV4Flash)
        let id = try XCTUnwrap(model.newConversation())
        let conversation = try XCTUnwrap(model.conversation(id))
        let displayed = try XCTUnwrap(model.budgetPlan(for: conversation))
        XCTAssertEqual(displayed.strategy, .boundedLayerStreaming)

        let request = try XCTUnwrap(model.makeRequest(for: conversation))
        XCTAssertEqual(request.model, .deepseekV4Flash)
        XCTAssertEqual(request.memoryBudgetBytes, displayed.budgetBytes)
        XCTAssertNil(request.pinPlan, "the V4 runner plans against the artifact it opens")
        XCTAssertFalse(try XCTUnwrap(model.capabilities(for: .deepseekV4Flash)).supportsPinPlan)
    }

    /// A new V4 chat starts at the preset its platform's evidence supports, and
    /// the budget it inherits is exactly that preset.
    func testANewV4ChatStartsAtItsPlatformsDefaultPreset() throws {
        let model = AppModel.freshForTests(installedModel: .deepseekV4Flash)
        model.setDefaultModel(.deepseekV4Flash)
        let id = try XCTUnwrap(model.newConversation())
        let conversation = try XCTUnwrap(model.conversation(id))
        let plan = try XCTUnwrap(model.budgetPlan(for: conversation))
        let preset = AppModel.defaultPreset(for: .deepseekV4Flash)

        XCTAssertEqual(conversation.settings.memoryBudgetBytes, plan.budget(for: preset))
        XCTAssertEqual(
            model.defaultBudgetPlan(for: .deepseekV4Flash)?.budgetBytes, plan.budget(for: preset))
        #if os(macOS)
            XCTAssertEqual(preset, .balanced)
            XCTAssertGreaterThan(plan.pinnedLayerCount, 0, "the Mac default holds layers")
        #else
            XCTAssertEqual(preset, .floor)
        #endif
    }

    /// K3 is untouched by the V4 default: it still starts at its own floor.
    func testANewK3ChatStillStartsAtItsFloor() throws {
        let model = AppModel.freshForTests(installedModel: .kimiK3)
        let id = try XCTUnwrap(model.newConversation())
        let conversation = try XCTUnwrap(model.conversation(id))
        let plan = try XCTUnwrap(model.budgetPlan(for: conversation))

        XCTAssertEqual(conversation.settings.memoryBudgetBytes, plan.budget(for: .floor))
        XCTAssertEqual(
            conversation.settings.memoryBudgetBytes, plan.profile.refusalThresholdBytes)
    }

    func testPreparingAnUpdatedArtifactReplacesHistoricalPlanGeometry() throws {
        let historical = CatalogFixtures.k3Census
        var currentResident = historical.layerResidentBytes
        currentResident[0] = historical.layerStoredBytes[0]
        let current = try ArtifactCensus(
            layerStoredBytes: historical.layerStoredBytes,
            layerResidentBytes: currentResident,
            globalsStoredBytes: historical.globalsStoredBytes,
            globalsBytesReadPerToken: historical.globalsBytesReadPerToken,
            expertStrideBytes: historical.expertStrideBytes,
            expertsPerLayer: historical.expertsPerLayer,
            routedExpertsPerToken: historical.routedExpertsPerToken,
            moeLayerCount: historical.moeLayerCount,
            measuredDeterministicBytesPerToken:
                historical.measuredDeterministicBytesPerToken)
        XCTAssertNotEqual(current.layerTableSHA256, historical.layerTableSHA256)

        let model = AppModel.freshForTests(runtimeCensus: current)
        let id = try XCTUnwrap(model.newConversation())
        model.setBudget(8 << 30, in: id)
        let conversation = try XCTUnwrap(model.conversation(id))
        let historicalPlan = try XCTUnwrap(model.budgetPlan(for: conversation)?.pinPlan)
        XCTAssertEqual(historicalPlan.layerTableSHA256, historical.layerTableSHA256)

        let request = try XCTUnwrap(model.makeRequest(for: conversation))
        let displayedAfterPreparation = try XCTUnwrap(
            model.budgetPlan(for: conversation)?.pinPlan)
        XCTAssertEqual(request.pinPlan, displayedAfterPreparation)
        XCTAssertEqual(request.pinPlan?.layerTableSHA256, current.layerTableSHA256)
        XCTAssertNotEqual(request.pinPlan?.layerTableSHA256, historicalPlan.layerTableSHA256)
    }

    /// Only scan-backed full verification flips provenance. A mock download
    /// reaching `ready` is history, not evidence about the mounted artifact.
    func testOnlyScannedFullVerificationFlipsProvenanceToMeasured() {
        let unverified = AppModel.freshForTests(verification: .unverified)
        XCTAssertEqual(
            unverified.defaultBudgetPlan(for: .kimiK3)?.profile.provenance,
            .declaredByIndex)
        unverified.download(.kimiK3)?.markReadyForPreview()
        XCTAssertEqual(
            unverified.defaultBudgetPlan(for: .kimiK3)?.profile.provenance,
            .declaredByIndex)

        let verified = AppModel.freshForTests(verification: .fullyVerified)
        XCTAssertFalse(verified.state(of: .kimiK3).isReady)
        XCTAssertEqual(
            verified.defaultBudgetPlan(for: .kimiK3)?.profile.provenance,
            .measured)
    }
}

/// The cockpit never promotes absence to success. These cases exercise the
/// evidence policy independently of SwiftUI so a later visual refactor cannot
/// turn 0/0 back into a percentage or an idle snapshot back into a green dot.
final class InstrumentEvidenceTests: XCTestCase {

    func testCanonicalPhaseEventsDriveHonestLayerProgress() {
        var snapshot = RunSnapshot()
        snapshot.state = .running
        snapshot.ladder = (0..<3).map {
            LayerCell(
                index: $0, status: .pending, storedBytes: 10, widenedBytes: 20,
                wallSeconds: nil, wasStaged: false)
        }

        snapshot.apply(RunPhase(name: "prefill", fraction: 0, detail: "layer 0/3"))
        XCTAssertEqual(snapshot.ladder.map(\.status), [.computing, .pending, .pending])

        snapshot.apply(
            RunPhase(name: "prefill", fraction: 1.0 / 3.0, detail: "layer 1/3"))
        XCTAssertEqual(snapshot.ladder.map(\.status), [.done, .computing, .pending])

        snapshot.apply(RunPhase(name: "decode 1", fraction: 0, detail: "layer 0/3"))
        XCTAssertEqual(
            snapshot.ladder.map(\.status), [.computing, .pending, .pending],
            "a new token pass owns a new 3-layer progress cycle")

        snapshot.apply(RunPhase(name: "decode 1", fraction: 1, detail: "layer 3/3"))
        XCTAssertEqual(snapshot.ladder.map(\.status), [.done, .done, .done])

        snapshot.apply(
            RunPhase(
                name: "model-specific-pass", generationStage: .decode,
                fraction: 1.0 / 3.0, detail: "layer 1/3"))
        XCTAssertEqual(
            snapshot.ladder.map(\.status), [.done, .computing, .pending],
            "structured stage identity must not depend on a model-specific phase name")
    }

    // MARK: Where the time went

    /// Two bars, and which passes each one covers. The decode aggregate is a
    /// sum that states its pass count, so the panel can read a mean without
    /// inventing a pass nobody ran.
    func testTheSnapshotSplitsPassesIntoTheTwoBarsThePanelDraws() {
        var snapshot = RunSnapshot()
        snapshot.phaseSummaries = [
            RunPhaseSummary(
                passKind: .prefill, passIndex: 0, passSeconds: 100,
                terms: [RunPhaseTerm(name: RunPhaseTermName.expertIOWait, seconds: 60)]),
            RunPhaseSummary(
                passKind: .decode, passIndex: 1, passSeconds: 20,
                terms: [RunPhaseTerm(name: RunPhaseTermName.expertIOWait, seconds: 8)]),
            RunPhaseSummary(
                passKind: .decode, passIndex: 2, passSeconds: 30,
                terms: [RunPhaseTerm(name: RunPhaseTermName.expertIOWait, seconds: 12)]),
        ]

        XCTAssertEqual(snapshot.prefillPhaseSummary?.passSeconds, 100)
        let decode = snapshot.decodePhaseAggregate
        XCTAssertEqual(decode?.passCount, 2)
        XCTAssertEqual(decode?.passSeconds, 50)
        XCTAssertEqual(decode?.secondsPerPass, 25)
        XCTAssertEqual(decode?.seconds(of: RunPhaseTermName.expertIOWait), 20)
        XCTAssertEqual(decode?.unattributedSeconds, 30)
        XCTAssertEqual(snapshot.phaseBars.map(\.passKind), [.prefill, .decode])
    }

    func testASnapshotWithNoDecompositionDrawsNoBars() {
        XCTAssertTrue(RunSnapshot().phaseBars.isEmpty)
        var prefillOnly = RunSnapshot()
        prefillOnly.phaseSummaries = [
            RunPhaseSummary(passKind: .prefill, passIndex: 0, passSeconds: 4, terms: [])
        ]
        XCTAssertEqual(prefillOnly.phaseBars.count, 1, "one attributed pass is one bar")
    }

    // MARK: The prefill pass has its own readouts

    /// Prompt ingestion is minutes long, and for all of it the decode pair is
    /// honestly "—". What the run does publish in that time is a byte total, a
    /// clock and a layer cursor, and these are the three derivations the panel
    /// shows instead.
    func testPrefillPublishesThroughputProgressAndAnEstimateWhileDecodeHasNothing() {
        var snapshot = RunSnapshot()
        snapshot.state = .running
        snapshot.ladder = (0..<10).map {
            LayerCell(
                index: $0, status: .pending, storedBytes: 10, widenedBytes: 20,
                wallSeconds: nil, wasStaged: false)
        }
        snapshot.apply(
            RunPhase(name: "prefill", generationStage: .prefill, fraction: 0.4, detail: ""))
        snapshot.apply(
            prefillTelemetry(elapsed: 60, totalBytesRead: 6_000_000_000, bytesPerSecond: 1e8))

        XCTAssertEqual(snapshot.generationStage, .prefill)
        XCTAssertEqual(try XCTUnwrap(snapshot.readBytesPerSecond), 1e8, accuracy: 1)
        XCTAssertEqual(snapshot.passLayerProgress?.completed, 4)
        XCTAssertEqual(snapshot.passLayerProgress?.total, 10)
        // 60 s bought four layers, so six more are estimated at 90 s.
        XCTAssertEqual(try XCTUnwrap(snapshot.prefillRemainingSecondsEstimate), 90, accuracy: 0.001)

        // The decode pair remains unreported: no decode token has completed.
        XCTAssertNil(snapshot.tokensPerSecond)
        XCTAssertNil(snapshot.bytesPerToken)
    }

    func testTheRemainingEstimateIsWithheldUntilItCanBeDerived() {
        var snapshot = RunSnapshot()
        snapshot.state = .running
        snapshot.ladder = (0..<10).map {
            LayerCell(
                index: $0, status: .pending, storedBytes: 10, widenedBytes: 20,
                wallSeconds: nil, wasStaged: false)
        }
        snapshot.apply(
            RunPhase(name: "prefill", generationStage: .prefill, fraction: 0.1, detail: ""))
        snapshot.apply(
            prefillTelemetry(elapsed: 12, totalBytesRead: 1_000, bytesPerSecond: 80))
        XCTAssertNil(
            snapshot.prefillRemainingSecondsEstimate,
            "one finished layer is not a pace")

        snapshot.apply(
            RunPhase(name: "prefill", generationStage: .prefill, fraction: 1, detail: ""))
        XCTAssertNil(
            snapshot.prefillRemainingSecondsEstimate,
            "a finished pass has nothing remaining")
    }

    /// A runner without a complete read observer publishes zeroes it does not
    /// stand behind. They must not become a measured throughput.
    func testUnreportedByteAccountingHasNoReadThroughput() {
        var snapshot = RunSnapshot()
        snapshot.state = .running
        snapshot.telemetry = telemetry(
            bytes: ByteAccounting(totalBytesRead: 0), bytesPerToken: nil,
            byteAccountingReported: false)
        XCTAssertNil(snapshot.readBytesPerSecond)
    }

    func testTheByteCountChipSaysWhatItMeansInProductWords() {
        XCTAssertEqual(AccountingStatusChip(status: .notReported).title, "Byte counts pending")
        XCTAssertEqual(AccountingStatusChip(status: .valid).title, "Byte counts balanced")
        XCTAssertEqual(AccountingStatusChip(status: .failed("x")).title, "Byte counts mismatch")
    }

    private func prefillTelemetry(
        elapsed: TimeInterval, totalBytesRead: UInt64, bytesPerSecond: Double
    ) -> RunTelemetry {
        RunTelemetry(
            at: Date(), elapsed: elapsed, phase: "prefill", tokensPerSecond: nil,
            generationStage: .prefill,
            bytes: ByteAccounting(
                totalBytesRead: totalBytesRead,
                deterministicBytesRead: totalBytesRead, expertBytesRead: 0),
            bytesPerSecond: bytesPerSecond, bytesPerToken: nil,
            declaredBudgetBytes: 8_000_000_000, footprintBytes: 1, peakFootprintBytes: 1,
            residentBytes: 1, availableBytes: nil, mlxActiveBytes: 0,
            mlxCacheBytes: 0, mlxPeakBytes: 0, thermalState: .nominal,
            lowPowerMode: false, batteryLevel: nil)
    }

    func testIdleMetricsAreNotReportedRatherThanHealthyZeroes() {
        let snapshot = RunSnapshot()

        XCTAssertEqual(InstrumentPanelPresentation(snapshot: snapshot), .idle)
        XCTAssertEqual(snapshot.accountingStatus, .notReported)
        XCTAssertNil(snapshot.stalls)
        XCTAssertNil(snapshot.cache)
        XCTAssertNil(snapshot.tokensPerSecond)
        XCTAssertNil(snapshot.bytesPerToken)

        let emptyStalls = StallAccounting(
            computeSeconds: 0, computeWaitedOnIOSeconds: 0, wallSeconds: 0,
            achievedBytesPerSecond: 0)
        XCTAssertNil(emptyStalls.ioWaitFraction)
        XCTAssertNil(emptyStalls.overlapFraction, "0/0 is not 100% overlap")

        let emptyCache = CacheStrip(
            hits: 0, misses: 0, evictions: 0, rejectedAdmissions: 0,
            pinnedExpertCount: 0)
        XCTAssertNil(emptyCache.hitRate, "0/0 is not a cache hit rate")
        XCTAssertEqual(emptyCache.line, "No expert requests yet")
        XCTAssertEqual(emptyCache.accessibilityLine, "no expert requests yet")
        XCTAssertFalse(emptyCache.line.contains("%"))

        var zeroBeforeAnyToken = RunSnapshot()
        zeroBeforeAnyToken.telemetry = telemetry(
            bytes: ByteAccounting(totalBytesRead: 0), bytesPerToken: 0)
        XCTAssertNil(
            zeroBeforeAnyToken.bytesPerToken,
            "a runner's pre-token zero is not a per-token measurement")
    }

    func testInstrumentPanelHasOneTruthfulStateBeforeAValidatedRun() {
        var preparing = RunSnapshot()
        preparing.state = .validating
        XCTAssertEqual(InstrumentPanelPresentation(snapshot: preparing), .preparing)

        preparing.acceptance = acceptance()
        XCTAssertEqual(InstrumentPanelPresentation(snapshot: preparing), .measurements)

        var refused = RunSnapshot()
        refused.state = .refused(.underlying("runner refused"))
        XCTAssertEqual(InstrumentPanelPresentation(snapshot: refused), .unavailable)
    }

    func testInstrumentHelpExplainsProductMeaningInsteadOfInternalHistory() {
        XCTAssertTrue(InstrumentHelpCopy.byteFlow.contains("fixed half-second"))
        XCTAssertTrue(InstrumentHelpCopy.byteFlow.contains("not drive or network speed"))
        XCTAssertTrue(InstrumentHelpCopy.overlap.contains("not blocked waiting for model data"))
        XCTAssertTrue(InstrumentHelpCopy.overlap.contains("not a disk-utilization measure"))
        XCTAssertTrue(InstrumentHelpCopy.overlap.contains("largest staging buffer"))
        XCTAssertTrue(InstrumentHelpCopy.expertReuse.contains("reused from memory"))
        XCTAssertTrue(InstrumentHelpCopy.expertReuse.contains("says Off"))
        XCTAssertFalse(InstrumentHelpCopy.byteFlow.localizedCaseInsensitiveContains("spec"))
        XCTAssertFalse(InstrumentHelpCopy.overlap.localizedCaseInsensitiveContains("experiment"))
    }

    func testPrefillAndInFlightBytesDoNotInflateDecodeTokenAverage() throws {
        var snapshot = RunSnapshot()
        snapshot.state = .running
        snapshot.tokens = [
            TokenEvent(
                index: 0, tokenID: 1, text: "x", secondsSincePreviousToken: 1,
                isPrefillToken: true, top1Top2Gap: nil),
            TokenEvent(
                index: 1, tokenID: 2, text: "y", secondsSincePreviousToken: 2,
                isPrefillToken: false, top1Top2Gap: nil),
        ]
        let prefill = ByteAccounting(
            totalBytesRead: 100, deterministicBytesRead: 80, expertBytesRead: 20)
        let completed = ByteAccounting(
            totalBytesRead: 200, deterministicBytesRead: 150, expertBytesRead: 50)
        snapshot.telemetry = telemetry(
            bytes: ByteAccounting(
                totalBytesRead: 275, deterministicBytesRead: 210, expertBytesRead: 65),
            completedTokenBytes: completed, prefillBytes: prefill,
            bytesPerToken: nil, decodeTokensCompleted: 1)

        XCTAssertEqual(try XCTUnwrap(snapshot.tokensPerSecond), 0.5, accuracy: 0.0001)
        XCTAssertEqual(snapshot.bytesPerToken, 100)
        XCTAssertEqual(snapshot.byteSplitPerToken?.deterministic, 70)
        XCTAssertEqual(snapshot.byteSplitPerToken?.expert, 30)
        XCTAssertEqual(
            try XCTUnwrap(snapshot.currentTokenBytes),
            ByteAccounting(
                totalBytesRead: 75, deterministicBytesRead: 60, expertBytesRead: 15))
    }

    func testUnreportedByteAccountingNeverAppearsAsAMeasuredZero() {
        var snapshot = RunSnapshot()
        snapshot.state = .running
        snapshot.acceptance = acceptance()
        snapshot.telemetry = telemetry(
            bytes: ByteAccounting(totalBytesRead: 0), bytesPerToken: nil,
            byteAccountingReported: false)

        XCTAssertEqual(snapshot.accountingStatus, .notReported)
        XCTAssertNil(snapshot.bytesPerToken)
        XCTAssertNil(snapshot.currentTokenBytes)

        snapshot.tokens = [
            TokenEvent(
                index: 0, tokenID: 1, text: "x", secondsSincePreviousToken: 1,
                isPrefillToken: true, top1Top2Gap: nil)
        ]
        snapshot.state = .finished
        XCTAssertNil(
            snapshot.bytesPerToken,
            "an unavailable V4 counter must not become 0 MB/token at the terminal event")
        XCTAssertNil(snapshot.currentTokenBytes)
    }

    func testPartialEvidenceKeepsEachMetricIndependent() throws {
        var snapshot = RunSnapshot()
        snapshot.acceptance = acceptance()
        snapshot.telemetry = telemetry(
            bytes: ByteAccounting(totalBytesRead: 100), bytesPerToken: nil)
        snapshot.merge(
            InstrumentSample(
                at: Date(),
                stalls: StallAccounting(
                    computeSeconds: 8, computeWaitedOnIOSeconds: 2, wallSeconds: 10,
                    achievedBytesPerSecond: 50),
                cache: CacheStrip(
                    hits: 0, misses: 0, evictions: nil, rejectedAdmissions: nil,
                    pinnedExpertCount: nil)))

        XCTAssertEqual(snapshot.accountingStatus, .notReported)
        XCTAssertEqual(try XCTUnwrap(snapshot.stalls?.overlapFraction), 0.8, accuracy: 0.0001)
        XCTAssertNil(snapshot.cache?.hitRate)
        XCTAssertEqual(snapshot.cache?.line, "No expert requests yet")
        XCTAssertNil(snapshot.bytesPerToken, "a zero-token denominator must stay unknown")

        snapshot.merge(
            InstrumentSample(
                at: Date(),
                flow: FlowSample(
                    at: Date(), deterministicBytesPerSecond: 10,
                    expertBytesPerSecond: 20)))
        XCTAssertNotNil(snapshot.stalls, "a partial sample erased earlier stall evidence")
        XCTAssertNotNil(snapshot.cache, "a partial sample erased earlier cache evidence")
        XCTAssertEqual(snapshot.flow.count, 1)
    }

    func testCompleteValidEvidenceBecomesBalanced() throws {
        var snapshot = RunSnapshot()
        snapshot.acceptance = acceptance()
        snapshot.telemetry = telemetry(
            bytes: ByteAccounting(
                totalBytesRead: 100, deterministicBytesRead: 60, expertBytesRead: 40),
            completedTokenBytes: ByteAccounting(
                totalBytesRead: 100, deterministicBytesRead: 60, expertBytesRead: 40),
            bytesPerToken: 100, decodeTokensCompleted: 1)
        snapshot.readAhead = ReadAheadAccounting(
            depth: 1, stagedBytes: 100, peakStagedBytes: 100,
            materialisedStagedBytes: 70, discardedStagedBytes: 20,
            outstandingStagedBytes: 10, prefetchWaitSeconds: 2,
            stageReadSeconds: 10, nonFreeSlots: 0)
        snapshot.cache = CacheStrip(
            policy: "lru",
            hits: 3, misses: 1, evictions: 2, rejectedAdmissions: 0,
            pinnedExpertCount: 6)

        XCTAssertEqual(snapshot.byteAccountingStatus, .valid)
        XCTAssertEqual(snapshot.readAheadAccountingStatus, .valid)
        XCTAssertEqual(snapshot.accountingStatus, .valid)
        XCTAssertEqual(try XCTUnwrap(snapshot.readAhead?.overlapEfficiency), 0.8, accuracy: 0.0001)
        XCTAssertEqual(try XCTUnwrap(snapshot.cache?.hitRate), 0.75, accuracy: 0.0001)
        XCTAssertEqual(snapshot.bytesPerToken, 100)

        var serialKnobs = RunKnobs()
        serialKnobs.deterministicReadAheadLayers = 0
        var serial = RunSnapshot()
        serial.acceptance = acceptance(knobs: serialKnobs)
        serial.telemetry = telemetry(
            bytes: ByteAccounting(
                totalBytesRead: 100, deterministicBytesRead: 60, expertBytesRead: 40),
            completedTokenBytes: ByteAccounting(
                totalBytesRead: 100, deterministicBytesRead: 60, expertBytesRead: 40),
            bytesPerToken: nil)
        XCTAssertEqual(
            serial.accountingStatus, .valid,
            "a serial run does not owe read-ahead accounting it never enabled")
    }

    /// `pending` is a statement about passes, not about acceptance: the chip's
    /// own ⓘ says the counts settle at a pass boundary and that until the run
    /// finishes its first pass there is nothing to check.
    func testByteCountsStayPendingUntilTheFirstPassReports() {
        var snapshot = RunSnapshot()
        snapshot.state = .running
        snapshot.acceptance = acceptance()
        snapshot.telemetry = telemetry(
            bytes: ByteAccounting(
                totalBytesRead: 40, deterministicBytesRead: 25, expertBytesRead: 15),
            bytesPerToken: nil)

        XCTAssertNil(
            snapshot.byteSplitPerToken,
            "no pass has reported, so there is no per-token split on screen either")
        XCTAssertEqual(snapshot.byteAccountingStatus, .notReported)
        XCTAssertEqual(snapshot.accountingStatus, .notReported)
    }

    /// The bug the owner photographed on a live V4 run: the panel was already
    /// showing decode data/token and its model-layers/experts split while the
    /// chip beside it still read `Byte counts pending`.
    ///
    /// V4 has no deterministic read-ahead — the knob is not in its supported
    /// set, so its acceptance leaves the field nil — and it publishes no
    /// instrumentation. The chip must not wait on staged-slot evidence that
    /// this run will never produce; it reports the partition the run reported.
    func testAReportedPerTokenSplitBalancesTheChipMidRun() throws {
        var snapshot = RunSnapshot()
        snapshot.state = .running
        snapshot.acceptance = acceptance(model: .deepseekV4Flash)
        snapshot.telemetry = telemetry(
            bytes: ByteAccounting(
                totalBytesRead: 260, deterministicBytesRead: 190, expertBytesRead: 70),
            completedTokenBytes: ByteAccounting(
                totalBytesRead: 200, deterministicBytesRead: 150, expertBytesRead: 50),
            prefillBytes: ByteAccounting(
                totalBytesRead: 100, deterministicBytesRead: 80, expertBytesRead: 20),
            bytesPerToken: nil, decodeTokensCompleted: 1)

        XCTAssertNil(snapshot.readAhead, "V4 publishes no read-ahead accounting")
        XCTAssertEqual(snapshot.readAheadAccountingStatus, .notReported)
        XCTAssertEqual(try XCTUnwrap(snapshot.byteSplitPerToken).deterministic, 70)
        XCTAssertEqual(try XCTUnwrap(snapshot.byteSplitPerToken).expert, 30)
        XCTAssertEqual(snapshot.byteAccountingStatus, .valid)
        XCTAssertEqual(
            snapshot.accountingStatus, .valid,
            "the run reported an exact partition at a pass boundary")
    }

    /// K3 does publish read-ahead accounting, but its final slot census only
    /// exists at the terminal event. A byte chip that waited for it read
    /// pending for every pass of every run.
    func testAPendingSlotCensusDoesNotHoldTheByteCountsPending() {
        var snapshot = RunSnapshot()
        snapshot.state = .running
        snapshot.acceptance = acceptance()
        snapshot.telemetry = telemetry(
            bytes: ByteAccounting(
                totalBytesRead: 100, deterministicBytesRead: 60, expertBytesRead: 40),
            completedTokenBytes: ByteAccounting(
                totalBytesRead: 100, deterministicBytesRead: 60, expertBytesRead: 40),
            bytesPerToken: nil)
        snapshot.readAhead = ReadAheadAccounting(
            depth: 1, stagedBytes: 100, peakStagedBytes: 100,
            materialisedStagedBytes: 70, discardedStagedBytes: 20,
            outstandingStagedBytes: 10, prefetchWaitSeconds: 2,
            stageReadSeconds: 10, nonFreeSlots: nil)

        XCTAssertEqual(snapshot.readAheadAccountingStatus, .notReported)
        XCTAssertEqual(snapshot.accountingStatus, .valid)
    }

    /// The reported boundary is checked, not only the cumulative counter, so a
    /// pass that double-counts a read is a mismatch rather than a quiet pass.
    func testAnUnbalancedReportedPassIsAMismatch() {
        var snapshot = RunSnapshot()
        snapshot.state = .running
        snapshot.acceptance = acceptance()
        snapshot.telemetry = telemetry(
            bytes: ByteAccounting(
                totalBytesRead: 260, deterministicBytesRead: 190, expertBytesRead: 70),
            completedTokenBytes: ByteAccounting(
                totalBytesRead: 200, deterministicBytesRead: 150, expertBytesRead: 60),
            bytesPerToken: nil)

        XCTAssertEqual(
            snapshot.byteAccountingStatus,
            .failed("model layers + experts ≠ total bytes read"))
        XCTAssertEqual(
            snapshot.accountingStatus,
            .failed("model layers + experts ≠ total bytes read"))
    }

    func testCanonicalTelemetryFeedsOverlapReadAheadAndCacheButNotPayloadFlow() throws {
        let start = Date(timeIntervalSince1970: 100)
        let expert = RunExpertTelemetry(
            phaseSeconds: 4, ioWaitSeconds: 1, bytesRead: 120,
            cachePolicy: "lru", cacheHits: 2, cacheMisses: 6,
            cacheEvictions: 1, cacheRejectedAdmissions: 0,
            pinnedExpertCount: 0)
        let liveReadAhead = RunReadAheadTelemetry(
            depth: 1, stagedBytes: 100, peakStagedBytes: 60,
            materialisedStagedBytes: 70, discardedStagedBytes: 10,
            outstandingStagedBytes: 20, prefetchWaitSeconds: 1,
            stageReadSeconds: 4, nonFreeSlots: nil)
        let finalReadAhead = RunReadAheadTelemetry(
            depth: 1, stagedBytes: 100, peakStagedBytes: 60,
            materialisedStagedBytes: 70, discardedStagedBytes: 30,
            outstandingStagedBytes: 0, prefetchWaitSeconds: 1,
            stageReadSeconds: 4, nonFreeSlots: 0)

        func sample(
            at: Date, deterministic: UInt64, expertBytes: UInt64,
            readAhead: RunReadAheadTelemetry
        ) -> RunTelemetry {
            let accounting = ByteAccounting(
                totalBytesRead: deterministic + expertBytes,
                deterministicBytesRead: deterministic,
                expertBytesRead: expertBytes)
            return RunTelemetry(
                at: at, elapsed: at.timeIntervalSince(start), phase: "prefill",
                tokensPerSecond: nil,
                bytes: accounting,
                bytesPerSecond: nil, completedTokenBytes: accounting,
                bytesPerToken: nil,
                declaredBudgetBytes: 1_000, footprintBytes: 500,
                peakFootprintBytes: 500, residentBytes: 400,
                availableBytes: nil, mlxActiveBytes: 0, mlxCacheBytes: 0,
                mlxPeakBytes: 0, thermalState: .nominal,
                lowPowerMode: false, batteryLevel: nil,
                instrumentation: RunInstrumentationTelemetry(
                    expert: expert, readAhead: readAhead))
        }

        var snapshot = RunSnapshot()
        snapshot.state = .running
        snapshot.acceptance = acceptance()
        snapshot.apply(
            sample(at: start, deterministic: 100, expertBytes: 40, readAhead: liveReadAhead))
        snapshot.apply(
            sample(
                at: start.addingTimeInterval(2), deterministic: 220,
                expertBytes: 100, readAhead: liveReadAhead))

        XCTAssertTrue(
            snapshot.flow.isEmpty,
            "telemetry delivery time is not a comparable model-data-flow clock")
        snapshot.apply(
            RunPayloadFlowSample(
                sequence: 1, at: start.addingTimeInterval(2), intervalSeconds: 2,
                deterministicBytes: 120, expertBytes: 60))
        let flow = try XCTUnwrap(snapshot.flow.last)
        XCTAssertEqual(flow.deterministicBytesPerSecond, 60, accuracy: 0.0001)
        XCTAssertEqual(flow.expertBytesPerSecond, 30, accuracy: 0.0001)
        XCTAssertEqual(try XCTUnwrap(snapshot.stalls?.overlapFraction), 0.75, accuracy: 0.0001)
        XCTAssertEqual(try XCTUnwrap(snapshot.cache?.hitRate), 0.25, accuracy: 0.0001)
        XCTAssertEqual(snapshot.readAhead?.peakStagedBytes, 60)
        XCTAssertEqual(snapshot.readAheadAccountingStatus, .notReported)

        snapshot.apply(
            sample(
                at: start.addingTimeInterval(3), deterministic: 220,
                expertBytes: 100, readAhead: finalReadAhead))
        XCTAssertEqual(snapshot.readAheadAccountingStatus, .valid)
        XCTAssertEqual(snapshot.accountingStatus, .valid)
    }

    func testPayloadFlowRejectsInvalidDuplicateAndOutOfOrderWindows() {
        var snapshot = RunSnapshot()
        snapshot.apply(
            RunPayloadFlowSample(
                sequence: 2, at: Date(), intervalSeconds: 0.5,
                deterministicBytes: 50, expertBytes: 25))
        snapshot.apply(
            RunPayloadFlowSample(
                sequence: 2, at: Date(), intervalSeconds: 0.5,
                deterministicBytes: 500, expertBytes: 250))
        snapshot.apply(
            RunPayloadFlowSample(
                sequence: 1, at: Date(), intervalSeconds: 0.5,
                deterministicBytes: 500, expertBytes: 250))
        snapshot.apply(
            RunPayloadFlowSample(
                sequence: 3, at: Date(), intervalSeconds: 0,
                deterministicBytes: 1, expertBytes: 1))

        XCTAssertEqual(snapshot.flow.count, 1)
        XCTAssertEqual(snapshot.flow.first?.sequence, 2)
        XCTAssertEqual(snapshot.flow.first?.totalBytesPerSecond, 150)
    }

    func testPayloadFlowKeepsExactlyOneMinuteOfHalfSecondWindows() {
        var snapshot = RunSnapshot()
        let start = Date(timeIntervalSince1970: 100)
        for sequence in 1...121 {
            snapshot.apply(
                RunPayloadFlowSample(
                    sequence: UInt64(sequence),
                    at: start.addingTimeInterval(Double(sequence) * 0.5),
                    intervalSeconds: 0.5,
                    deterministicBytes: UInt64(sequence), expertBytes: 0))
        }

        XCTAssertEqual(snapshot.flow.count, 120)
        XCTAssertEqual(snapshot.flow.first?.sequence, 2)
        XCTAssertEqual(snapshot.flow.last?.sequence, 121)
    }

    func testIncompleteStallIdentityDoesNotClaimStorageOverlap() {
        let incomplete = StallAccounting(
            computeSeconds: 8, computeWaitedOnIOSeconds: 1, wallSeconds: 10,
            achievedBytesPerSecond: 100)
        XCTAssertNil(incomplete.overlapFraction)
    }

    func testNoCachePolicyReportsOffInsteadOfAFakeZeroPercentHitRate() {
        let cache = CacheStrip(
            policy: "no-cache", hits: 0, misses: 132_960, evictions: 0,
            rejectedAdmissions: 0, pinnedExpertCount: 0)

        XCTAssertTrue(cache.isDisabled)
        XCTAssertNil(cache.hitRate)
        XCTAssertEqual(cache.line, "Off · 132,960 loaded from storage")
        XCTAssertEqual(cache.accessibilityLine, "off, 132960 loaded from storage")
        XCTAssertFalse(cache.line.contains("0.0%"))
    }

    func testEnabledCacheWithRequestsCanTruthfullyReportZeroReuse() throws {
        let cache = CacheStrip(
            policy: "lru", hits: 0, misses: 8, evictions: 1,
            rejectedAdmissions: 2, pinnedExpertCount: 0)

        XCTAssertEqual(try XCTUnwrap(cache.hitRate), 0, accuracy: 0.0001)
        XCTAssertTrue(cache.line.contains("0.0% hit rate"))
        XCTAssertTrue(cache.line.contains("8 loaded"))
    }

    func testFailedIdentityWinsOverOtherValidOrMissingMetrics() {
        var brokenBytes = RunSnapshot()
        brokenBytes.acceptance = acceptance()
        brokenBytes.telemetry = telemetry(
            bytes: ByteAccounting(
                totalBytesRead: 100, deterministicBytesRead: 60, expertBytesRead: 50),
            bytesPerToken: nil)
        // The identity is named in the product's vocabulary, because the chip
        // shows this string verbatim.
        XCTAssertEqual(
            brokenBytes.accountingStatus,
            .failed("model layers + experts ≠ total bytes read"))

        var brokenReadAhead = RunSnapshot()
        brokenReadAhead.acceptance = acceptance()
        brokenReadAhead.telemetry = telemetry(
            bytes: ByteAccounting(
                totalBytesRead: 100, deterministicBytesRead: 60, expertBytesRead: 40),
            bytesPerToken: nil)
        brokenReadAhead.readAhead = ReadAheadAccounting(
            depth: 1, stagedBytes: 100, peakStagedBytes: 100,
            materialisedStagedBytes: 70, discardedStagedBytes: 20,
            outstandingStagedBytes: 9, prefetchWaitSeconds: 0,
            stageReadSeconds: 0, nonFreeSlots: 0)
        XCTAssertEqual(
            brokenReadAhead.accountingStatus,
            .failed("materialised + discarded + outstanding ≠ staged"))
        XCTAssertNil(
            brokenReadAhead.readAhead?.overlapEfficiency,
            "a failed accounting sample with 0/0 timing must not claim 0% overlap")
    }

    func testAnAcceptedUnsupportedReadAheadDepthIsAnAccountingFailure() {
        var knobs = RunKnobs()
        knobs.deterministicReadAheadLayers = 2
        var snapshot = RunSnapshot()
        snapshot.acceptance = acceptance(knobs: knobs)
        snapshot.telemetry = telemetry(
            bytes: ByteAccounting(
                totalBytesRead: 100, deterministicBytesRead: 60, expertBytesRead: 40),
            bytesPerToken: 100)

        XCTAssertEqual(
            snapshot.accountingStatus,
            .failed("accepted read-ahead depth is outside 0...1"))
    }

    private func acceptance(
        knobs: RunKnobs = RunKnobs(), model: ModelID = .kimiK3
    ) -> RunAcceptance {
        RunAcceptance(
            handle: RunHandleID(), model: model, declaredBudgetBytes: 1_000,
            effectiveKnobs: knobs, artifactRoot: "/tmp/k3", startedAt: Date())
    }

    private func telemetry(
        bytes: ByteAccounting, completedTokenBytes: ByteAccounting? = nil,
        prefillBytes: ByteAccounting? = nil,
        bytesPerToken: UInt64?, byteAccountingReported: Bool? = true,
        decodeTokensCompleted: Int? = nil
    ) -> RunTelemetry {
        RunTelemetry(
            at: Date(), elapsed: 0, phase: "idle", tokensPerSecond: nil,
            bytes: bytes, bytesPerSecond: nil, completedTokenBytes: completedTokenBytes,
            prefillBytes: prefillBytes, bytesPerToken: bytesPerToken,
            decodeTokensCompleted: decodeTokensCompleted,
            byteAccountingReported: byteAccountingReported,
            declaredBudgetBytes: 1_000, footprintBytes: 0, peakFootprintBytes: 0,
            residentBytes: 0, availableBytes: nil, mlxActiveBytes: 0,
            mlxCacheBytes: 0, mlxPeakBytes: 0, thermalState: .unknown,
            lowPowerMode: false, batteryLevel: nil)
    }
}

/// A test seam that represents the one fact the preview runner cannot: logits
/// bytes were captured and hashed by the runner itself.
private final class ImmediateDigestRunner: RunnerFacade, @unchecked Sendable {
    let capabilities = MockDecodeRunner(entry: CatalogFixtures.kimiK3).capabilities
    private let digest: String

    init(digest: String) { self.digest = digest }

    func validate(_ request: RunRequest) throws { try validateCommon(request) }

    func start(_ request: RunRequest) throws -> RunSession {
        try validate(request)
        let handle = RunHandleID()
        let now = Date()
        let telemetry = RunTelemetry(
            at: now, elapsed: 1, phase: "finished", tokensPerSecond: 1,
            bytes: ByteAccounting(totalBytesRead: 1), bytesPerSecond: 1, bytesPerToken: 1,
            declaredBudgetBytes: request.memoryBudgetBytes,
            footprintBytes: 1, peakFootprintBytes: 1, residentBytes: 1,
            availableBytes: nil, mlxActiveBytes: 0, mlxCacheBytes: 0, mlxPeakBytes: 0,
            thermalState: .nominal, lowPowerMode: false, batteryLevel: nil)
        let token = TokenEvent(
            index: 0, tokenID: 42, text: "x", secondsSincePreviousToken: 1,
            isPrefillToken: true, top1Top2Gap: 0.5)
        let summary = RunSummary(
            handle: handle, model: request.model, tokenIDs: [token.tokenID], text: token.text,
            wallSeconds: 1, tokensPerSecond: 1, finalTelemetry: telemetry,
            peakFootprintBytes: 1, budgetRespected: true, thermalTrail: [],
            logitsDigest: RunLogitsDigest(tokenIndex: 0, hex: digest),
            resultJSON: nil, resultJSONFilename: nil,
            headline: "measured", lines: [], gitRevision: "test")
        let (events, continuation) = AsyncThrowingStream<RunEvent, Error>.makeStream()
        continuation.yield(
            .accepted(
                RunAcceptance(
                    handle: handle, model: request.model,
                    declaredBudgetBytes: request.memoryBudgetBytes,
                    pinPlan: request.pinPlan, effectiveKnobs: request.knobs,
                    artifactRoot: request.artifact.root.path, startedAt: now)))
        // The terminal summary is authoritative evidence even if a transport
        // drops the otherwise useful progressive token event.
        continuation.yield(.finished(summary))
        continuation.finish()
        return RunSession(handle: handle, events: events, cancel: {})
    }
}

private final class ImmediatePayloadFlowRunner: RunnerFacade, @unchecked Sendable {
    let capabilities = MockDecodeRunner(entry: CatalogFixtures.kimiK3).capabilities

    func validate(_ request: RunRequest) throws { try validateCommon(request) }

    func start(_ request: RunRequest) throws -> RunSession {
        try validate(request)
        let handle = RunHandleID()
        let now = Date()
        let (events, eventContinuation) = AsyncThrowingStream<RunEvent, Error>.makeStream()
        let (payloadFlow, payloadContinuation) =
            AsyncStream<RunPayloadFlowSample>.makeStream()
        eventContinuation.yield(
            .accepted(
                RunAcceptance(
                    handle: handle, model: request.model,
                    declaredBudgetBytes: request.memoryBudgetBytes,
                    pinPlan: request.pinPlan, effectiveKnobs: request.knobs,
                    artifactRoot: request.artifact.root.path, startedAt: now)))
        eventContinuation.finish()
        payloadContinuation.yield(
            RunPayloadFlowSample(
                sequence: 7, at: now, intervalSeconds: 0.5,
                deterministicBytes: 300, expertBytes: 100))
        payloadContinuation.finish()
        return RunSession(
            handle: handle, events: events, payloadFlow: payloadFlow, cancel: {})
    }
}

private final class ImmediateStorageFailureRunner: RunnerFacade, @unchecked Sendable {
    let capabilities = MockDecodeRunner(entry: CatalogFixtures.kimiK3).capabilities
    private let failure: StorageCoreError

    init(_ failure: StorageCoreError) { self.failure = failure }

    func validate(_ request: RunRequest) throws { try validateCommon(request) }

    func start(_ request: RunRequest) throws -> RunSession {
        try validate(request)
        let handle = RunHandleID()
        let (events, continuation) = AsyncThrowingStream<RunEvent, Error>.makeStream()
        continuation.finish(throwing: failure)
        return RunSession(
            handle: handle, events: events,
            cancel: { request.artifact.scope?.release() })
    }
}

private final class CountingScopeCoding: SecurityScopedURLCoding, @unchecked Sendable {
    private let lock = NSLock()
    private var startCount = 0
    private var stopCount = 0

    var starts: Int { lock.withLock { startCount } }
    var stops: Int { lock.withLock { stopCount } }

    func bookmarkData(for url: URL) throws -> Data { Data(url.path.utf8) }
    func resolve(_ data: Data) throws -> ResolvedBookmark {
        throw CocoaError(.fileReadCorruptFile)
    }
    func startAccessing(_ url: URL) -> Bool {
        lock.withLock { startCount += 1 }
        return true
    }
    func stopAccessing(_ url: URL) {
        lock.withLock { stopCount += 1 }
    }
}

private final class DecoderLifetimeProbe {
    func decode(_ ids: [Int]) -> String { ids.map(String.init).joined(separator: " ") }
}

/// The projection refuses to guess, and its two discontinuities are real.
final class SpeedProjectionTests: XCTestCase {

    private let entry = CatalogFixtures.kimiK3

    /// A 32 GiB ceiling, which is the development Mac's. It has to be at least
    /// that for the pinned tier to have anywhere to go: at 12 GiB the ladder
    /// gets five layers in and the curve has no interesting range left.
    private func plan(budget: UInt64, readAhead: Int) -> BudgetPlan {
        BudgetPlan(
            model: .kimiK3, modelName: "Kimi K3", profile: entry.memory,
            budgetBytes: budget, maximumNewTokens: 64,
            deviceCeilingBytes: 32 << 30, readAheadDepth: readAhead)
    }

    private var link: VolumeCalibration {
        VolumeCalibration(
            volumeMountPath: "/Volumes/MINIRUN-NVME", volumeName: "MINIRUN-NVME",
            bytesPerSecond: 0.92e9, busDescription: "USB3 10 Gb/s", measuredAt: Date(),
            sampleSeconds: 4)
    }

    func testProductProjectionEvidenceRequiresModelRevisionDeviceAndLocalTerms() {
        let candidate = plan(budget: 8_000_000_000, readAhead: 0)
        let complete = ProjectionEvidence(
            model: .kimiK3,
            artifactRevision: "revision-a",
            deviceIdentifier: "device-a")
        let measuredHere = WorkloadTerms(
            deterministicBytesPerToken: entry.terms.deterministicBytesPerToken,
            expertBytesPerToken: entry.terms.expertBytesPerToken,
            computeSecondsPerToken: entry.terms.computeSecondsPerToken,
            isMeasuredHere: true)

        XCTAssertFalse(complete.matches(plan: candidate, terms: entry.terms))
        XCTAssertTrue(complete.matches(plan: candidate, terms: measuredHere))
        XCTAssertFalse(
            ProjectionEvidence(
                model: .deepseekV4Flash,
                artifactRevision: "revision-a",
                deviceIdentifier: "device-a"
            ).matches(plan: candidate, terms: measuredHere))
        XCTAssertFalse(
            ProjectionEvidence(
                model: .kimiK3,
                artifactRevision: " ",
                deviceIdentifier: "device-a"
            ).matches(plan: candidate, terms: measuredHere))
        XCTAssertFalse(
            ProjectionEvidence(
                model: .kimiK3,
                artifactRevision: "revision-a",
                deviceIdentifier: ""
            ).matches(plan: candidate, terms: measuredHere))
    }

    func testNoCalibrationMeansNoProjectionAtAll() {
        XCTAssertNil(
            SpeedProjector.project(
                plan: plan(budget: 8_000_000_000, readAhead: 0), terms: entry.terms, link: nil))
    }

    func testARefusedBudgetHasNoProjection() {
        XCTAssertNil(
            SpeedProjector.project(
                plan: plan(budget: 3_000_000_000, readAhead: 0), terms: entry.terms, link: link))
    }

    /// The measured pass on the reference rig was 5:57 a token. The projection
    /// at that budget, on that link, has to land there — otherwise the curve is
    /// decorative.
    func testTheSerialProjectionReproducesTheMeasuredPass() throws {
        let projection = try XCTUnwrap(
            SpeedProjector.project(
                plan: plan(budget: 8_000_000_000, readAhead: 0), terms: entry.terms, link: link))
        XCTAssertEqual(projection.mid, 357, accuracy: 2)
        XCTAssertFalse(projection.deterministicHidden)
        XCTAssertLessThan(projection.secondsPerTokenLow, projection.mid)
        XCTAssertGreaterThan(projection.secondsPerTokenHigh, projection.mid)
    }

    /// The step down where the staged pair first fits.
    func testFundingTheStagedPairStepsTheCurveDown() throws {
        let serial = try XCTUnwrap(
            SpeedProjector.project(
                plan: plan(budget: 8_000_000_000, readAhead: 1), terms: entry.terms, link: link))
        let staged = try XCTUnwrap(
            SpeedProjector.project(
                plan: plan(budget: 9_000_000_000, readAhead: 1), terms: entry.terms, link: link))
        XCTAssertFalse(serial.deterministicHidden)
        XCTAssertTrue(staged.deterministicHidden)
        XCTAssertLessThan(staged.mid, serial.mid)
    }

    /// There is no hot-set slope, and there must not be one.
    ///
    /// The test this replaces asserted that the curve bent gently upward as a
    /// pinned expert cache grew with the budget. The derived tier ladder says
    /// that cache does not exist at any budget a device here can state: every
    /// deterministic layer returns 1.000 bytes saved per resident byte and the
    /// best hot set ever measured returns 0.230, so experts rank last, behind
    /// all 93 layers *and* the globals bundle. The old slope was a projection of
    /// a tier that is never funded.
    func testNoBudgetBelowFullResidencyPinsASingleExpert() {
        for budget: UInt64 in [6_000_000_000, 9_000_000_000, 12 << 30] {
            XCTAssertEqual(
                plan(budget: budget, readAhead: 0).hotSetBytes, 0,
                "an expert was pinned at \(budget), below full deterministic residency")
        }
        XCTAssertGreaterThan(
            entry.memory.firstExpertBudgetBytes, 116_129_117_440,
            "the first expert cannot be reachable before every layer is resident")
    }

    /// Pinning must never make the projected curve promise a speedup.
    ///
    /// The arithmetic says a pinned layer saves its own bytes every token, and
    /// the only measurement anybody has of a pinned tier
    /// (`docs/experiments/2026-08-11-dial-decode.md`) is a single-pass token
    /// that came out 11.31 s SLOWER. Until a second token has been decoded with
    /// a tier filled, the curve stays flat across the pinned range and the strip
    /// says so in words. A curve that fell here would be this product promising
    /// something nobody has seen.
    /// Both budgets sit above the staged step, so the only thing that differs
    /// between them is how many layers are pinned. 16 GiB is where every
    /// read-ahead pair first becomes affordable; 24 GiB pins eight more layers
    /// than it and must project exactly the same seconds.
    func testTheProjectedCurveDoesNotFallAsThePinnedTierGrows() throws {
        let lower = plan(budget: 16 << 30, readAhead: 1)
        let higher = plan(budget: 24 << 30, readAhead: 1)
        XCTAssertTrue(lower.stagedTierIsFunded, "both arms must be past the staged step")
        XCTAssertTrue(higher.stagedTierIsFunded)
        XCTAssertGreaterThan(
            higher.pinnedBytes, lower.pinnedBytes,
            "the higher budget must actually pin more, or this proves nothing")

        let low = try XCTUnwrap(
            SpeedProjector.project(plan: lower, terms: entry.terms, link: link))
        let high = try XCTUnwrap(
            SpeedProjector.project(plan: higher, terms: entry.terms, link: link))
        XCTAssertEqual(
            high.mid, low.mid, accuracy: 0.001,
            "the curve bent across the pinned tier; no arm supports that")
    }
}

/// The formatting rules the whole product's honesty rests on.
final class FormattingTests: XCTestCase {

    /// The response limit is typed now, so what a commit does with what was
    /// typed is a product rule rather than a stepper's arithmetic.
    func testATypedIntegerCommitsClampedAndNeverInventsAZero() {
        let range = 1...64
        XCTAssertEqual(IntegerField.committed("16", in: range, current: 8), 16)
        XCTAssertEqual(IntegerField.committed("900", in: range, current: 8), 64)
        XCTAssertEqual(IntegerField.committed("0", in: range, current: 8), 1)
        XCTAssertEqual(
            IntegerField.committed("", in: range, current: 8), 8,
            "an emptied field restores the stated value instead of becoming zero")
    }

    func testAdaptiveRateFormatting() {
        XCTAssertEqual(MRFormat.tokenRate(1.4), "1.4 tok/s")
        XCTAssertEqual(MRFormat.tokenRate(0.4), "0.4 tok/s (2.5 s/tok)")
        XCTAssertEqual(MRFormat.tokenRate(1.0 / 357.0), "5:57 / token")
        XCTAssertEqual(MRFormat.tokenRate(nil), "—")
        XCTAssertEqual(MRFormat.tokenRate(0), "—")
    }

    func testDecimalAndBinaryScalesAreNotInterchangeable() {
        XCTAssertEqual(MRFormat.bytesDecimal(UInt64(5_800_000_000)), "5.80 GB")
        XCTAssertEqual(MRFormat.bytesDecimal(UInt64(5_261_334_118)), "5.26 GB")
        XCTAssertEqual(MRFormat.bytesDecimal(UInt64(1_559_976_181_760)), "1.56 TB")
    }

    func testExactBytesAreGroupedAndNeverRounded() {
        XCTAssertEqual(MRFormat.exactBytes(UInt64(1_560_411_238_400)), "1,560,411,238,400 B")
        XCTAssertEqual(MRFormat.exactBytes(UInt64(1_182_514_432)), "1,182,514,432 B")
    }

    func testDigestsTruncateInTheMiddleAndNeverLookWhole() {
        let digest = CatalogFixtures.deviceLogitsDigest
        let short = MRFormat.shortDigest(digest)
        XCTAssertTrue(short.contains("…"))
        XCTAssertTrue(digest.hasPrefix(String(short.prefix(8))))
        XCTAssertTrue(digest.hasSuffix(String(short.suffix(5))))
    }
}
