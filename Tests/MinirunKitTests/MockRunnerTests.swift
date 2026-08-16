import Foundation
import XCTest

@testable import MinirunKit

final class MockRunnerTests: XCTestCase {

    private func request(
        budget: UInt64 = 6_000_000_000, tokens: Int = 1, knobs: RunKnobs = RunKnobs(),
        prompt: Prompt = .tokenIDs([1, 2, 3]), pinPlan: PinPlan? = nil
    ) -> RunRequest {
        RunRequest(
            model: .kimiK3,
            artifact: ArtifactReference(root: URL(fileURLWithPath: "/tmp/artifact")),
            prompt: prompt, memoryBudgetBytes: budget, pinPlan: pinPlan,
            maximumNewTokens: tokens,
            knobs: knobs, workingDirectory: URL(fileURLWithPath: NSTemporaryDirectory()))
    }

    private func plan(budget: UInt64) throws -> PinPlan {
        let census = try ArtifactCensus.measuredFlagship()
        return try MemoryDialPlanner.plan(
            budgetBytes: budget, census: census,
            floor: WorkingSetFloor.measuredFlagship(census))
    }

    // MARK: - Refusals, all before a byte is read

    func testABudgetBelowTheMeasuredMinimumIsRefusedNotClamped() {
        let runner = MockRunner()
        XCTAssertThrowsError(try runner.validate(request(budget: 1_000_000_000))) { error in
            XCTAssertEqual(
                error as? RunError,
                .budgetBelowMinimum(
                    declared: 1_000_000_000, minimum: 5_800_000_000, model: .kimiK3))
        }
        // And nothing ran: `start` calls `validate` first.
        XCTAssertThrowsError(try runner.start(request(budget: 1_000_000_000)))
    }

    func testAKnobTheRunnerCannotHonourIsRefusedByName() {
        var knobs = RunKnobs()
        knobs.expertPoolSlots = 8  // a V4 knob, on a K3-shaped runner
        XCTAssertThrowsError(try MockRunner().validate(request(knobs: knobs))) { error in
            XCTAssertEqual(
                error as? RunError, .knobNotSupported(name: "expertPoolSlots", byModel: .kimiK3))
        }
    }

    func testAKnobOutOfRangeNamesTheRangeItMissed() {
        var knobs = RunKnobs()
        knobs.queueDepth = 0
        XCTAssertThrowsError(try MockRunner().validate(request(knobs: knobs))) { error in
            XCTAssertEqual(
                error as? RunError,
                .knobOutOfRange(name: "queueDepth", value: "0", allowed: ">= 1"))
        }
    }

    func testATextPromptIsRefusedRatherThanTokenizedByAGuess() {
        XCTAssertThrowsError(try MockRunner().validate(request(prompt: .text("hello")))) { error in
            guard case .promptNotSupported(let reason)? = error as? RunError else {
                return XCTFail("expected promptNotSupported, got \(error)")
            }
            XCTAssertTrue(reason.contains("token ids"))
        }
    }

    func testAnEmptyPromptIsRefused() {
        XCTAssertThrowsError(try MockRunner().validate(request(prompt: .tokenIDs([]))))
    }

    func testAskingForMoreTokensThanTheRunnerProducesIsRefused() {
        XCTAssertThrowsError(try MockRunner().validate(request(tokens: 8))) { error in
            XCTAssertEqual(
                error as? RunError, .tooManyTokens(requested: 8, maximum: 1, model: .kimiK3))
        }
    }

    func testAPlanForAnotherBudgetIsRefusedRatherThanSilentlyReplanned() throws {
        let otherBudget: UInt64 = 5_900_000_000
        let mismatched = try plan(budget: otherBudget)
        XCTAssertThrowsError(try MockRunner().validate(request(pinPlan: mismatched))) { error in
            XCTAssertEqual(
                error as? RunError,
                .pinPlanBudgetMismatch(
                    planBytes: otherBudget, declaredBytes: 6_000_000_000))
        }
    }

    func testAPlanForAnotherResponseHorizonIsRefused() throws {
        let census = try ArtifactCensus.measuredFlagship()
        let mismatched = try MemoryDialPlanner.plan(
            budgetBytes: 6_000_000_000, census: census,
            floor: WorkingSetFloor.measuredFlagship(census), maximumNewTokens: 2)
        XCTAssertThrowsError(try MockRunner().validate(request(pinPlan: mismatched))) { error in
            guard case RunError.pinPlanInvalid(let reason) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertTrue(reason.contains("2 new tokens"), reason)
            XCTAssertTrue(reason.contains("requests 1"), reason)
        }
    }

    func testAnUnbalancedPlanIsRefusedBeforeTheRunnerStarts() throws {
        let valid = try plan(budget: 6_000_000_000)
        let encoded = try JSONEncoder().encode(valid)
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object["unusedBudgetBytes"] = 0
        let corrupted = try JSONDecoder().decode(
            PinPlan.self, from: JSONSerialization.data(withJSONObject: object))

        XCTAssertThrowsError(try MockRunner().start(request(pinPlan: corrupted))) { error in
            XCTAssertEqual(error as? RunError, .pinPlanAccountingMismatch)
        }
    }

    func testSelfBalancingTopLevelBucketsCannotHideForgedPinDecisions() throws {
        let budget: UInt64 = 8 << 30
        let valid = try plan(budget: budget)
        XCTAssertFalse(valid.pinnedLayers.isEmpty)
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(valid))
                as? [String: Any])
        object["pinnedBytes"] = 0
        object["unusedBudgetBytes"] = NSNumber(
            value: valid.unusedBudgetBytes + valid.pinnedBytes)
        let forged = try JSONDecoder().decode(
            PinPlan.self, from: JSONSerialization.data(withJSONObject: object))
        XCTAssertTrue(forged.isResidencyBalanced, "the attack keeps the old shallow identity")

        XCTAssertThrowsError(try MockRunner().start(request(budget: budget, pinPlan: forged))) {
            guard case RunError.pinPlanInvalid(let reason) = $0 else {
                return XCTFail("\($0)")
            }
            XCTAssertTrue(reason.contains("pin decisions"), reason)
        }
    }

    func testAReadAheadReserveCannotBeForgedIndependently() throws {
        let budget: UInt64 = 8 << 30
        let valid = try plan(budget: budget)
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(valid))
                as? [String: Any])
        object["readAheadReserveBytes"] = NSNumber(value: valid.readAheadReserveBytes + 1)
        let forged = try JSONDecoder().decode(
            PinPlan.self, from: JSONSerialization.data(withJSONObject: object))
        XCTAssertTrue(forged.isResidencyBalanced)
        XCTAssertTrue(forged.isAccountingBalanced)

        XCTAssertThrowsError(try MockRunner().validate(request(budget: budget, pinPlan: forged))) {
            guard case RunError.pinPlanInvalid(let reason) = $0 else {
                return XCTFail("\($0)")
            }
            XCTAssertTrue(reason.contains("reserve"), reason)
        }
    }

    func testARequestForAnotherModelIsRefused() {
        let other = RunRequest(
            model: .deepseekV4Flash,
            artifact: ArtifactReference(root: URL(fileURLWithPath: "/tmp")),
            prompt: .tokenIDs([1]), memoryBudgetBytes: 6_000_000_000, maximumNewTokens: 1,
            workingDirectory: URL(fileURLWithPath: NSTemporaryDirectory()))
        XCTAssertThrowsError(try MockRunner().validate(other)) { error in
            guard case .runnerUnavailable(let model, _)? = error as? RunError else {
                return XCTFail("expected runnerUnavailable, got \(error)")
            }
            XCTAssertEqual(model, .deepseekV4Flash)
        }
    }

    // MARK: - A run

    func testAValidRunAcceptsThenProducesTokensThenFinishes() async throws {
        let runner = MockRunner(scriptedTokenIDs: [4242])
        let pinPlan = try plan(budget: 6_000_000_000)
        let session = try runner.start(request(pinPlan: pinPlan))

        var accepted: RunAcceptance?
        var tokens: [TokenEvent] = []
        var summary: RunSummary?
        for try await event in session.events {
            switch event {
            case .accepted(let acceptance): accepted = acceptance
            case .token(let token): tokens.append(token)
            case .finished(let done): summary = done
            default: break
            }
        }

        let acceptance = try XCTUnwrap(accepted)
        XCTAssertEqual(acceptance.handle, session.handle)
        XCTAssertEqual(acceptance.declaredBudgetBytes, 6_000_000_000)
        XCTAssertEqual(acceptance.pinPlan, pinPlan)
        XCTAssertEqual(tokens.map(\.tokenID), [4242])
        XCTAssertTrue(tokens[0].isPrefillToken)
        XCTAssertNil(tokens[0].top1Top2Gap, "routing margins were not asked for")

        let done = try XCTUnwrap(summary)
        XCTAssertEqual(done.tokenIDs, [4242])
        XCTAssertEqual(done.finalTelemetry.declaredBudgetBytes, 6_000_000_000)
        XCTAssertFalse(done.gitRevision.isEmpty)
        XCTAssertTrue(done.lines.contains { $0.contains("no byte was read") })
    }

    func testTheMockNeverInventsAnMLXNumberOrAByteItDidNotRead() async throws {
        let session = try MockRunner().start(request())
        var telemetry: RunTelemetry?
        for try await event in session.events {
            if case .telemetry(let sample) = event { telemetry = sample }
        }
        let sample = try XCTUnwrap(telemetry)
        XCTAssertEqual(sample.bytes.totalBytesRead, 0)
        XCTAssertEqual(sample.mlxActiveBytes, 0)
        XCTAssertEqual(sample.mlxCacheBytes, 0)
        XCTAssertEqual(sample.mlxPeakBytes, 0)
        XCTAssertTrue(sample.bytes.isBalanced)
        XCTAssertEqual(sample.bytes.unattributedBytes, 0)
    }

    func testCancelEndsTheRunAsCancelledRatherThanShort() async throws {
        let runner = MockRunner(scriptedTokenIDs: [1], secondsPerToken: 0.4)
        let session = try runner.start(request())
        session.cancel()

        var outcome: String?
        for try await event in session.events {
            if case .cancelled = event { outcome = "cancelled" }
            if case .finished = event { outcome = "finished" }
        }
        XCTAssertEqual(outcome, "cancelled")
    }

    /// K3's runner does not offer routing margins, so asking a K3-shaped mock
    /// for them is refused by name — and a runner that *does* offer them
    /// captures them. Both halves matter: the knob must never be quietly
    /// ignored, and capturing must never happen unasked.
    func testRoutingMarginsAreRefusedWhereUnsupportedAndCapturedWhereOffered() async throws {
        var knobs = RunKnobs()
        knobs.captureRoutingMargins = true
        XCTAssertThrowsError(try MockRunner().validate(request(knobs: knobs))) { error in
            XCTAssertEqual(
                error as? RunError,
                .knobNotSupported(name: "captureRoutingMargins", byModel: .kimiK3))
        }

        var capabilities = MockRunner.defaultCapabilities
        capabilities = RunnerCapabilities(
            model: capabilities.model, layout: capabilities.layout,
            supportedKnobs: capabilities.supportedKnobs.union(["captureRoutingMargins"]),
            minimumBudgetBytes: capabilities.minimumBudgetBytes,
            maximumNewTokens: capabilities.maximumNewTokens,
            acceptsTextPrompts: capabilities.acceptsTextPrompts,
            requiresMLX: capabilities.requiresMLX,
            tokenEventGranularity: capabilities.tokenEventGranularity)
        let session = try MockRunner(capabilities: capabilities).start(request(knobs: knobs))
        var gaps: [Double?] = []
        for try await event in session.events {
            if case .token(let token) = event { gaps.append(token.top1Top2Gap) }
        }
        XCTAssertEqual(gaps.compactMap { $0 }.count, gaps.count)
    }

    // MARK: - Accounting identities

    func testByteAccountingCatchesADoubleCountedRead() {
        let honest = ByteAccounting(
            totalBytesRead: 100, deterministicBytesRead: 40, expertBytesRead: 60)
        XCTAssertTrue(honest.isBalanced)
        XCTAssertEqual(honest.unattributedBytes, 0)

        let doubleCounted = ByteAccounting(
            totalBytesRead: 100, deterministicBytesRead: 80, expertBytesRead: 60)
        XCTAssertFalse(doubleCounted.isBalanced)

        let notSplitYet = ByteAccounting(totalBytesRead: 100)
        XCTAssertTrue(notSplitYet.isBalanced, "an unavailable split is not an imbalance")
        XCTAssertNil(notSplitYet.unattributedBytes)
    }

    func testByteAccountingCannotWrapAnOverflowIntoABalancedRun() {
        let overflow = ByteAccounting(
            totalBytesRead: 0,
            deterministicBytesRead: UInt64.max,
            expertBytesRead: 1)
        XCTAssertFalse(overflow.isBalanced)
        XCTAssertNil(overflow.unattributedBytes)
    }

    func testBudgetRespectedComparesThePeakAgainstThePromise() {
        func telemetry(peak: UInt64, budget: UInt64) -> RunTelemetry {
            RunTelemetry(
                at: Date(), elapsed: 1, phase: "p", tokensPerSecond: nil,
                bytes: ByteAccounting(totalBytesRead: 0), bytesPerSecond: nil, bytesPerToken: nil,
                declaredBudgetBytes: budget, footprintBytes: peak, peakFootprintBytes: peak,
                residentBytes: 0, availableBytes: nil, mlxActiveBytes: 0, mlxCacheBytes: 0,
                mlxPeakBytes: 0, thermalState: .nominal, lowPowerMode: false, batteryLevel: nil)
        }
        XCTAssertTrue(telemetry(peak: 99, budget: 100).budgetRespected)
        XCTAssertTrue(telemetry(peak: 100, budget: 100).budgetRespected)
        XCTAssertFalse(telemetry(peak: 101, budget: 100).budgetRespected)
        XCTAssertEqual(telemetry(peak: 50, budget: 100).budgetFraction, 0.5)
    }

    func testCapabilitiesStateWhenTokensArriveRatherThanImplyingIt() {
        XCTAssertEqual(MockRunner.defaultCapabilities.tokenEventGranularity, .perPass)
        XCTAssertFalse(MockRunner.defaultCapabilities.acceptsTextPrompts)
        XCTAssertEqual(MockRunner.defaultCapabilities.maximumNewTokens, 1)
        XCTAssertTrue(MockRunner.defaultCapabilities.supportsPinPlan)
    }
}
