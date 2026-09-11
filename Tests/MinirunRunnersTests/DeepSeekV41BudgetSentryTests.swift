import Foundation
import MinirunKit
import ModelAdapters
import StorageCore
import XCTest

@testable import MinirunRunners

/// The refusal has to fire **inside** a prefill, and this is what says it does.
///
/// The arm this would ideally be is the floor arm at a deliberately too-small
/// budget over the real container. That needs the drive, and on 2026-09-11 the
/// drive was plugged into the phone; the arm is the record's first open item.
/// What can be established without it is the mechanism, and the mechanism is
/// the part that was broken: a run that grows inside one block used to reach
/// no sample at all, because the engine looked once per *completed* block —
/// after `withBlock`'s pool and reclaim had already returned everything the
/// block held.
///
/// So this drives the real ``DeepSeekV41RunEngine`` over a workload that
/// allocates for real inside its prefill and asks the same question the artifact
/// asks: it calls the `cancellationCheck` the engine handed it. That closure is
/// ``DeepSeekV41BudgetSentry/check()``.
final class DeepSeekV41BudgetSentryTests: XCTestCase {
    private var root: URL!
    private var working: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "v41-sentry-\(UUID().uuidString)", isDirectory: true)
        working = root.appendingPathComponent("work", isDirectory: true)
        try FileManager.default.createDirectory(
            at: working, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: - The unit

    func testTheSentryDoesNotEnforceBeforeItKnowsItsFloor() {
        var exceeded: UInt64?
        let sentry = DeepSeekV41BudgetSentry(
            declaredBudgetBytes: 1, cancellationCheck: {}, exceeded: { exceeded = $0 })
        // Unarmed: a process footprint of gigabytes against a one-byte budget
        // must not refuse, because `peak - floor` is not yet `peak - anything`.
        XCTAssertNoThrow(try sentry.check())
        XCTAssertNil(exceeded)
        XCTAssertEqual(sentry.samples, 0)
    }

    func testTheSentryCountsItsSamplesAndItsWorstOvershoot() {
        var exceeded = [UInt64]()
        let sentry = DeepSeekV41BudgetSentry(
            declaredBudgetBytes: 1_000, cancellationCheck: {},
            exceeded: { exceeded.append($0) })
        sentry.setFloor(10_000)
        XCTAssertEqual(sentry.observe(current: 10_500, lifetime: 0), 500)
        XCTAssertEqual(sentry.worstOvershoot, 0)
        XCTAssertEqual(sentry.observe(current: 11_750, lifetime: 0), 1_750)
        XCTAssertEqual(sentry.worstOvershoot, 750)
        // A peak never falls, so a later, smaller reading does not erase it.
        XCTAssertEqual(sentry.observe(current: 10_100, lifetime: 0), 1_750)
        XCTAssertEqual(sentry.samples, 3)
        XCTAssertEqual(sentry.absolute, 11_750)
        XCTAssertEqual(sentry.floor, 10_000)
        XCTAssertTrue(exceeded.isEmpty, "observe() records; check() is what refuses")
    }

    // MARK: - The engine

    /// A run that grows past its ceiling **inside a block of prefill** is
    /// refused there, before a token exists.
    ///
    /// The workload never completes a block, so `onBlockCompleted` — the engine's
    /// old sampling point — is never called. Under the previous code this run
    /// would have allocated without limit until the OS intervened, which is
    /// exactly what the iPhone did.
    func testARunThatGrowsInsideAPrefillBlockIsRefusedInsideThatBlock() async throws {
        let state = SentryFakeState(
            allocationBytes: 64 << 20, allocationsBeforeCompletingABlock: 64)
        let runner = DeepSeekV41DecodeRunner(
            scale: .stated(minimumBudgetBytes: 1 << 20, maximumNewTokens: 4),
            factory: SentryFakeFactory(state: state))
        let session = try runner.start(
            RunRequest(
                model: .deepseekV41Flash,
                artifact: ArtifactReference(root: root),
                prompt: .tokenIDs([1, 2, 3]),
                memoryBudgetBytes: 512 << 20,
                maximumNewTokens: 2,
                knobs: RunKnobs(),
                workingDirectory: working))

        var sawToken = false
        var failure: Error?
        do {
            for try await event in session.events {
                if case .token = event { sawToken = true }
            }
        } catch {
            failure = error
        }

        let error = try XCTUnwrap(failure as? RunError)
        guard case .budgetExceeded(let peak, let declared) = error else {
            return XCTFail("expected a budget refusal, got \(error)")
        }
        XCTAssertEqual(declared, 512 << 20)
        XCTAssertGreaterThan(peak, declared)
        XCTAssertFalse(sawToken, "the refusal must fire before a token exists")
        XCTAssertTrue(state.enteredPrefill, "and inside the prefill")
        XCTAssertEqual(
            state.completedBlocks, 0,
            "no block completed, so the engine's per-block sampler never ran; the "
                + "refusal came from the seam inside the block")

        // The overshoot is bounded by what one allocation between two checks can
        // add. The workload allocates 64 MB a step and checks between steps, so
        // the peak may stand at most one step — plus the allocator's own slack —
        // above the ceiling.
        let overshoot = peak - declared
        XCTAssertLessThan(
            overshoot, 2 * UInt64(state.allocationBytes),
            "a refusal that arrives \(overshoot) B late is not a bounded overshoot")
    }

    /// A run inside its ceiling is not refused, and the record says how often it
    /// was looked at.
    func testARunInsideItsCeilingFinishesAndIsCheckedManyTimes() async throws {
        let state = SentryFakeState(
            allocationBytes: 1 << 20, allocationsBeforeCompletingABlock: 4)
        let runner = DeepSeekV41DecodeRunner(
            scale: .stated(minimumBudgetBytes: 1 << 20, maximumNewTokens: 4),
            factory: SentryFakeFactory(state: state))
        let session = try runner.start(
            RunRequest(
                model: .deepseekV41Flash,
                artifact: ArtifactReference(root: root),
                prompt: .tokenIDs([1, 2, 3]),
                memoryBudgetBytes: 2 << 30,
                maximumNewTokens: 2,
                knobs: RunKnobs(),
                workingDirectory: working))

        var summary: RunSummary?
        for try await event in session.events {
            if case .finished(let done) = event { summary = done }
        }
        let done = try XCTUnwrap(summary)
        XCTAssertTrue(done.budgetRespected)
        XCTAssertGreaterThan(
            state.checkCount, 6,
            "one prefill block and one decode block are eight allocations, and each is "
                + "followed by a check")
        XCTAssertTrue(
            done.lines.contains { $0.hasPrefix("budget checked ") },
            "a record that states a peak without stating the cadence cannot tell a run "
                + "that stayed inside its ceiling from a run nobody measured: "
                + "\(done.lines)")
    }
}

// MARK: - A workload that allocates for real

private final class SentryFakeState: @unchecked Sendable {
    let allocationBytes: Int
    private let allocationsBeforeCompletingABlock: Int
    private let lock = NSLock()
    private var held: [[UInt8]] = []
    private var checks = 0
    private var completed = 0
    private var prefillEntered = false

    init(allocationBytes: Int, allocationsBeforeCompletingABlock: Int) {
        self.allocationBytes = allocationBytes
        self.allocationsBeforeCompletingABlock = allocationsBeforeCompletingABlock
    }

    var checkCount: Int { lock.withLock { checks } }
    var completedBlocks: Int { lock.withLock { completed } }
    var enteredPrefill: Bool { lock.withLock { prefillEntered } }

    func markPrefill() { lock.withLock { prefillEntered = true } }

    /// One block's worth of work: allocate, then ask whether to stop — the order
    /// the real block reader and gather use, and the order that makes the
    /// overshoot one allocation rather than none.
    func runBlock(
        cancellationCheck: () throws -> Void,
        onBlockCompleted: (Int, Int) -> Void
    ) throws {
        for _ in 0..<allocationsBeforeCompletingABlock {
            let page = [UInt8](repeating: 7, count: allocationBytes)
            lock.lock()
            held.append(page)
            checks += 1
            lock.unlock()
            try cancellationCheck()
        }
        lock.withLock { completed += 1 }
        onBlockCompleted(1, 1)
    }

    func release() { lock.withLock { held.removeAll() } }
}

private struct SentryFakeFactory: DeepSeekV41RunWorkloadFactory {
    let state: SentryFakeState
    let requiresRuntimeAuthority = false

    func prepare(
        request: RunRequest, knobs: DeepSeekV41EffectiveKnobs,
        cancellation: DeepSeekV4RunCancellation,
        sentry: DeepSeekV41BudgetSentry,
        preparation: @escaping DeepSeekV41PreparationReporter
    ) throws -> any DeepSeekV41RunWorkload {
        try cancellation.check()
        preparation("fake workload prepared")
        return SentryFakeWorkload(state: state)
    }
}

private final class SentryFakeWorkload: DeepSeekV41RunWorkload {
    private let state: SentryFakeState

    init(state: SentryFakeState) { self.state = state }

    let blockCount = 1
    let vocabularySize = 32
    let eosTokenIDs: Set<Int> = [31]
    let maximumPositionCount = 64
    let sourceRepository = "deepseek-ai/DeepSeek-V4.1-Flash"
    let sourceRevision = "df42c109f1defefcbfcedbe7d905718a12266e40"
    let expertPoolBudgetBytes: UInt64 = 1 << 20
    var readAccountingSnapshot: DeepSeekV4ReadAccountingSnapshot? { nil }

    func productMemoryPlan(
        declaredBudgetBytes: UInt64,
        promptTokenCount: Int,
        maximumNewTokens: Int,
        mlxCacheBytes: UInt64,
        pinnedDeterministicBytes: UInt64,
        enforcesProductLimits: Bool
    ) throws -> DeepSeekV41ProductMemoryPlan {
        throw RunError.artifactNotReady("this fake runs at the stated scale only")
    }

    func prefill(
        tokenIDs: [Int], decodePositionLimit: Int,
        cancellationCheck: () throws -> Void,
        onBlockCompleted: (Int, Int) -> Void
    ) throws -> DeepSeekV41WorkloadStep {
        state.markPrefill()
        try state.runBlock(
            cancellationCheck: cancellationCheck, onBlockCompleted: onBlockCompleted)
        return step()
    }

    func decode(
        tokenID: Int, cancellationCheck: () throws -> Void,
        onBlockCompleted: (Int, Int) -> Void
    ) throws -> DeepSeekV41WorkloadStep {
        try state.runBlock(
            cancellationCheck: cancellationCheck, onBlockCompleted: onBlockCompleted)
        return step()
    }

    func validateTerminalBinding() throws {}

    func shutdown() { state.release() }

    /// A token that is not end-of-sentence, so the engine runs the decode pass
    /// too and the cadence is a cadence rather than a single sample.
    private func step() -> DeepSeekV41WorkloadStep {
        var logits = [Float](repeating: 0, count: vocabularySize)
        logits[1] = 1
        return DeepSeekV41WorkloadStep(tokenID: 1, logits: logits, completedBlocks: 1)
    }
}
