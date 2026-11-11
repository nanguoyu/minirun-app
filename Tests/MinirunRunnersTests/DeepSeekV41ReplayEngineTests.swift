import Foundation
import MinirunKit
import ModelAdapters
import StorageCore
import XCTest

@testable import MinirunRunners

/// The engine's side of ADR 0023: a prompt is a block and then one pass per
/// id, and every one of those passes is a pass the engine sees.
///
/// The arithmetic is the model's and is held bit for bit on the mini checkpoint
/// (`DeepSeekV41SessionContinuationTests`) and on the container
/// (`DeepSeekV41ConversationGateTests`). What is held here is the plumbing: the
/// engine issues exactly the schedule's calls, says which id of how many a
/// replay is on, stops between passes, and records the path.
final class DeepSeekV41ReplayEngineTests: XCTestCase {
    private var root: URL!
    private var working: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "v41-replay-\(UUID().uuidString)", isDirectory: true)
        working = root.appendingPathComponent("work", isDirectory: true)
        try FileManager.default.createDirectory(
            at: working, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: - The engine

    func testAReplayIsTheBlockThenOnePassPerIdAndSaysWhichIdItIsOn() async throws {
        let prompt = [1, 2, 3, 4, 5, 6, 7]
        let fake = ReplayFakeState(
            schedule: try DeepSeekV41PromptSchedule.make(
                prompt: prompt, firstBlockTokens: 3, restored: nil,
                unusableReason: "no session state was found"))
        let events = try await run(fake, prompt: prompt, newTokens: 2)

        XCTAssertEqual(fake.prefills, [[1, 2, 3]])
        XCTAssertEqual(fake.decodes, [4, 5, 6, 7, ReplayFakeWorkload.generated])
        let replays = events.compactMap { event -> RunReplayProgress? in
            if case .phase(let phase) = event { return phase.replay }
            return nil
        }
        let distinct = replays.map(\.token).reduce(into: [Int]()) { seen, token in
            if seen.last != token { seen.append(token) }
        }
        XCTAssertEqual(distinct, [3, 4, 5, 6, 7])
        XCTAssertTrue(replays.allSatisfy { $0.of == 7 })
        // The phase events of the decode that follows carry no replay.
        let lastPhase = events.last { if case .phase = $0 { return true } else { return false } }
        if case .phase(let phase) = lastPhase { XCTAssertNil(phase.replay) }
        XCTAssertTrue(
            events.contains {
                if case .telemetry(let sample) = $0 {
                    return sample.phase == "prefill, replay token 7 of 7"
                }
                return false
            })

        let tokens = events.compactMap { event -> TokenEvent? in
            if case .token(let token) = event { return token }
            return nil
        }
        XCTAssertEqual(tokens.count, 2, "replay passes are not generated tokens")
        XCTAssertTrue(tokens[0].isPrefillToken)
        XCTAssertFalse(tokens[1].isPrefillToken)
        XCTAssertTrue(
            events.contains {
                if case .log(let line) = $0 { return line.hasPrefix("V4.1 replay: 3-id first block") }
                return false
            })
    }

    func testAContinuedTurnRunsOnlyItsNewIds() async throws {
        let prompt = [1, 2, 3, 4, 5]
        let fake = ReplayFakeState(
            schedule: try DeepSeekV41PromptSchedule.make(
                prompt: prompt, firstBlockTokens: 2, restored: ([1, 2, 3], 2)))
        let events = try await run(fake, prompt: prompt, newTokens: 1)
        XCTAssertEqual(fake.prefills, [])
        XCTAssertEqual(fake.decodes, [4, 5])
        XCTAssertFalse(
            events.contains {
                if case .phase(let phase) = $0 { return phase.replay != nil }
                return false
            },
            "a continued turn is not a replay and does not say it is")
    }

    func testARunWithoutAConversationIsOnePrefillAsBefore() async throws {
        let prompt = [1, 2, 3]
        let fake = ReplayFakeState(schedule: nil)
        _ = try await run(fake, prompt: prompt, newTokens: 2)
        XCTAssertEqual(fake.prefills, [prompt])
        XCTAssertEqual(fake.decodes, [ReplayFakeWorkload.generated])
    }

    /// Stop during a replay: the engine stops between two of its passes like
    /// any decode, and no token is produced.
    func testAReplayStopsBetweenPasses() async throws {
        let prompt = Array(1...20)
        let fake = ReplayFakeState(
            schedule: try DeepSeekV41PromptSchedule.make(
                prompt: prompt, firstBlockTokens: 2, restored: nil),
            blockSeconds: 0.01)
        let runner = DeepSeekV41DecodeRunner(
            scale: .stated(minimumBudgetBytes: 1 << 20, maximumNewTokens: 4),
            factory: ReplayFakeFactory(state: fake))
        let session = try runner.start(request(prompt: prompt, newTokens: 2))
        var cancelled = false
        var sawToken = false
        for try await event in session.events {
            switch event {
            case .phase(let phase):
                if phase.replay?.token == 6 { session.cancel() }
            case .token: sawToken = true
            case .cancelled: cancelled = true
            default: break
            }
        }
        XCTAssertTrue(cancelled)
        XCTAssertFalse(sawToken)
        XCTAssertLessThan(fake.decodes.count, prompt.count - 2)
    }

    // MARK: - The logits dump sees every pass

    /// A replay's passes are the positions a teacher-forced arm compares, so
    /// each one writes its vector, named by the prompt id it consumed.
    func testAReplayDumpsEveryPromptPassByThePromptIdItConsumed() async throws {
        let prompt = [1, 2, 3, 4, 5, 6, 7]
        let fake = ReplayFakeState(
            schedule: try DeepSeekV41PromptSchedule.make(
                prompt: prompt, firstBlockTokens: 3, restored: nil))
        let dumped = try await dumpedFiles(fake, prompt: prompt, newTokens: 2)
        XCTAssertEqual(
            dumped,
            [
                "logits-0", "logits-1",
                "prompt-2", "prompt-3", "prompt-4", "prompt-5",
            ])
        let block = try sidecar("prompt-2")
        XCTAssertEqual(block.promptPass, true)
        XCTAssertTrue(block.isPrefillToken, "the block's own pass")
        XCTAssertEqual(block.position, 2)
        let pass = try sidecar("prompt-5")
        XCTAssertFalse(pass.isPrefillToken)
        XCTAssertEqual(pass.tokenID, ReplayFakeWorkload.generated)
        XCTAssertNil(try sidecar("logits-0").promptPass)
    }

    func testAContinuedTurnDumpsOnlyThePassesItRan() async throws {
        let prompt = [1, 2, 3, 4, 5]
        let fake = ReplayFakeState(
            schedule: try DeepSeekV41PromptSchedule.make(
                prompt: prompt, firstBlockTokens: 2, restored: ([1, 2, 3], 2)))
        let dumped = try await dumpedFiles(fake, prompt: prompt, newTokens: 1)
        XCTAssertEqual(dumped, ["logits-0", "prompt-3"])
    }

    func testAOneBlockPrefillDumpsNoPromptPass() async throws {
        let fake = ReplayFakeState(schedule: nil)
        let dumped = try await dumpedFiles(fake, prompt: [1, 2, 3], newTokens: 2)
        XCTAssertEqual(dumped, ["logits-0", "logits-1"])
    }

    // MARK: - The ceilings switch

    /// The product scale holds every runner the App builds to the policy's
    /// ceilings; only the internal initializer a measurement arm uses lifts
    /// them, and lifting them changes nothing else that is validated.
    func testOnlyTheInternalSwitchLiftsTheProductCeilings() throws {
        let policy = DeepSeekV41ProductMemoryBudget.currentPolicy
        let fake = ReplayFakeState(schedule: nil)
        func product(_ ceilings: Bool) -> DeepSeekV41DecodeRunner {
            DeepSeekV41DecodeRunner(
                scale: .product, factory: ReplayFakeFactory(state: fake),
                enforcesProductTokenCeilings: ceilings)
        }
        func request(promptCount: Int, newTokens: Int) -> RunRequest {
            RunRequest(
                model: .deepseekV41Flash,
                artifact: ArtifactReference(root: root),
                prompt: .tokenIDs(Array(repeating: 1, count: promptCount)),
                memoryBudgetBytes: policy.minimumBudgetBytes,
                maximumNewTokens: newTokens,
                knobs: RunKnobs(),
                workingDirectory: working)
        }
        let longReply = request(promptCount: 19, newTokens: policy.maximumNewTokens + 1)
        let longPrompt = request(promptCount: policy.maximumPromptTokens + 1, newTokens: 1)

        XCTAssertThrowsError(try product(true).validate(longReply))
        XCTAssertThrowsError(try product(true).validate(longPrompt))
        XCTAssertEqual(
            DeepSeekV41DecodeRunner(scale: .product).capabilities.maximumNewTokens,
            policy.maximumNewTokens)

        XCTAssertNoThrow(try product(false).validate(longReply))
        XCTAssertNoThrow(try product(false).validate(longPrompt))
        XCTAssertThrowsError(
            try product(false).validate(
                RunRequest(
                    model: .deepseekV41Flash, artifact: ArtifactReference(root: root),
                    prompt: .tokenIDs([1]),
                    memoryBudgetBytes: policy.minimumBudgetBytes - 1,
                    maximumNewTokens: 1, knobs: RunKnobs(), workingDirectory: working)),
            "the floor still binds")
    }

    private var dumpDirectory: URL { root.appendingPathComponent("dump", isDirectory: true) }

    private func dumpedFiles(
        _ fake: ReplayFakeState, prompt: [Int], newTokens: Int
    ) async throws -> [String] {
        let dump = try XCTUnwrap(
            DeepSeekV4LogitsDump(directoryPath: dumpDirectory.path, environment: [:]))
        let runner = DeepSeekV41DecodeRunner(
            scale: .stated(minimumBudgetBytes: 1 << 20, maximumNewTokens: 4),
            factory: ReplayFakeFactory(state: fake), logitsDump: dump)
        let session = try runner.start(request(prompt: prompt, newTokens: newTokens))
        for try await _ in session.events {}
        let names = try FileManager.default.contentsOfDirectory(atPath: dumpDirectory.path)
        XCTAssertEqual(
            Set(names.filter { $0.hasSuffix(".bin") }.map { String($0.dropLast(4)) }),
            Set(names.filter { $0.hasSuffix(".json") }.map { String($0.dropLast(5)) }),
            "every vector has its sidecar")
        return names.filter { $0.hasSuffix(".bin") }.map { String($0.dropLast(4)) }.sorted()
    }

    private func sidecar(_ name: String) throws -> DeepSeekV4LogitsDump.Sidecar {
        try JSONDecoder().decode(
            DeepSeekV4LogitsDump.Sidecar.self,
            from: Data(contentsOf: dumpDirectory.appendingPathComponent("\(name).json")))
    }

    private func request(prompt: [Int], newTokens: Int) -> RunRequest {
        RunRequest(
            model: .deepseekV41Flash,
            artifact: ArtifactReference(root: root),
            prompt: .tokenIDs(prompt),
            memoryBudgetBytes: 2 << 30,
            maximumNewTokens: newTokens,
            knobs: RunKnobs(),
            workingDirectory: working)
    }

    private func run(
        _ fake: ReplayFakeState, prompt: [Int], newTokens: Int
    ) async throws -> [RunEvent] {
        let runner = DeepSeekV41DecodeRunner(
            scale: .stated(minimumBudgetBytes: 1 << 20, maximumNewTokens: 4),
            factory: ReplayFakeFactory(state: fake), logitsDump: nil)
        let session = try runner.start(request(prompt: prompt, newTokens: newTokens))
        var events: [RunEvent] = []
        for try await event in session.events { events.append(event) }
        XCTAssertTrue(
            events.contains { if case .finished = $0 { return true } else { return false } })
        return events
    }

    // MARK: - The record

    func testTheRecordNamesThePath() throws {
        var record = DeepSeekV41SessionRecord()
        record.record(
            try .make(prompt: [1, 2, 3], firstBlockTokens: 3, restored: nil),
            statedFirstTurn: true)
        XCTAssertEqual(record.path, .firstTurn)
        XCTAssertEqual(record.firstBlockTokens, 3)

        record = DeepSeekV41SessionRecord()
        record.record(
            try .make(prompt: [1, 2, 3], firstBlockTokens: 3, restored: nil),
            statedFirstTurn: false)
        XCTAssertEqual(record.path, .fullPrefill)

        record = DeepSeekV41SessionRecord()
        record.record(
            try .make(
                prompt: [1, 2, 3, 4, 5], firstBlockTokens: 2, restored: nil,
                unusableReason: "no session state was found"),
            statedFirstTurn: true)
        XCTAssertEqual(record.path, .replayed)
        XCTAssertEqual(record.replayedPasses, 3)
        XCTAssertEqual(record.fallbackReason, "no session state was found")

        record = DeepSeekV41SessionRecord()
        record.record(
            try .make(prompt: [1, 2, 3, 4, 5], firstBlockTokens: 2, restored: ([1, 2, 3], 2)),
            statedFirstTurn: true)
        XCTAssertEqual(record.path, .continued)
        XCTAssertEqual(record.reusedPositions, 3)
        XCTAssertEqual(record.newTokens, 2)
        XCTAssertNil(record.firstBlockTokens)
    }

    // MARK: - Checkpoint cadence

    func testTheDefaultCadenceIsEveryPassOnAPhoneAndAtReleaseOnAMac() {
        #if os(iOS)
            XCTAssertEqual(DeepSeekV41SessionOptions().checkpoint, .everyPass)
        #else
            XCTAssertEqual(DeepSeekV41SessionOptions().checkpoint, .atRelease)
        #endif
        XCTAssertEqual(
            DeepSeekV41SessionOptions(checkpoint: .everyPass).checkpoint, .everyPass)
    }

    func testEveryPassWritesAfterEachPassAndNotAgainAtRelease() {
        let url = root.appendingPathComponent("a.v41session")
        let checkpointer = DeepSeekV41SessionCheckpointer(url: url, cadence: .everyPass)
        var written: [Int] = []
        func writer(_ positions: Int) -> DeepSeekV41SessionCheckpointer.Write {
            { _ in
                written.append(positions)
                return DeepSeekV41SessionSnapshotSummary(
                    fileBytes: 10, tensorBytes: 8, positions: positions, tokenSHA256: "x")
            }
        }
        for positions in [11, 12, 13] {
            checkpointer.passCompleted(positions: positions, write: writer(positions))
        }
        checkpointer.released(positions: 13, write: writer(13))
        XCTAssertEqual(written, [11, 12, 13])

        // A pass that threw drops its state; the last finished pass stays.
        checkpointer.released(positions: nil, write: writer(-1))
        var record = DeepSeekV41SessionRecord()
        checkpointer.fill(&record)
        XCTAssertEqual(record.snapshotPositions, 13)
        XCTAssertNil(record.snapshotRefusal)
        XCTAssertEqual(record.snapshotWrites, 3)
        XCTAssertEqual(record.checkpoint, .everyPass)
    }

    func testAtReleaseWritesOnceAndSaysWhenThereIsNothingToWrite() {
        let url = root.appendingPathComponent("b.v41session")
        var written = 0
        let write: DeepSeekV41SessionCheckpointer.Write = { _ in
            written += 1
            return DeepSeekV41SessionSnapshotSummary(
                fileBytes: 10, tensorBytes: 8, positions: 20, tokenSHA256: "x")
        }
        let checkpointer = DeepSeekV41SessionCheckpointer(url: url, cadence: .atRelease)
        checkpointer.passCompleted(positions: 19, write: write)
        checkpointer.passCompleted(positions: 20, write: write)
        XCTAssertEqual(written, 0)
        checkpointer.released(positions: 20, write: write)
        XCTAssertEqual(written, 1)

        let nothing = DeepSeekV41SessionCheckpointer(url: url, cadence: .atRelease)
        nothing.released(positions: nil, write: write)
        var record = DeepSeekV41SessionRecord()
        nothing.fill(&record)
        XCTAssertEqual(record.snapshotRefusal, "no complete generation state to write")
        XCTAssertNil(record.snapshotWrittenTo)
    }
}

// MARK: - A workload that records what it was asked to run

private final class ReplayFakeState: @unchecked Sendable {
    let schedule: DeepSeekV41PromptSchedule?
    let blockSeconds: Double
    private let lock = NSLock()
    private var prefillCalls: [[Int]] = []
    private var decodeCalls: [Int] = []

    init(schedule: DeepSeekV41PromptSchedule?, blockSeconds: Double = 0) {
        self.schedule = schedule
        self.blockSeconds = blockSeconds
    }

    var prefills: [[Int]] { lock.withLock { prefillCalls } }
    var decodes: [Int] { lock.withLock { decodeCalls } }
    func prefilled(_ ids: [Int]) { lock.withLock { prefillCalls.append(ids) } }
    func decoded(_ id: Int) { lock.withLock { decodeCalls.append(id) } }
}

private struct ReplayFakeFactory: DeepSeekV41RunWorkloadFactory {
    let state: ReplayFakeState
    let requiresRuntimeAuthority = false

    func prepare(
        request: RunRequest, knobs: DeepSeekV41EffectiveKnobs,
        cancellation: DeepSeekV4RunCancellation,
        sentry: DeepSeekV41BudgetSentry,
        preparation: @escaping DeepSeekV41PreparationReporter
    ) throws -> any DeepSeekV41RunWorkload {
        try cancellation.check()
        return ReplayFakeWorkload(state: state)
    }
}

private final class ReplayFakeWorkload: DeepSeekV41RunWorkload {
    static let generated = 9
    private let state: ReplayFakeState

    init(state: ReplayFakeState) { self.state = state }

    let blockCount = 2
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

    func schedulePrompt(
        tokenIDs: [Int], decodePositionLimit: Int
    ) throws -> DeepSeekV41PromptSchedule {
        if let schedule = state.schedule { return schedule }
        return try DeepSeekV41PromptSchedule.make(
            prompt: tokenIDs, firstBlockTokens: tokenIDs.count, restored: nil)
    }

    func prefill(
        tokenIDs: [Int], decodePositionLimit: Int,
        cancellationCheck: () throws -> Void,
        onBlockCompleted: (Int, Int) -> Void
    ) throws -> DeepSeekV41WorkloadStep {
        state.prefilled(tokenIDs)
        return try pass(cancellationCheck: cancellationCheck, onBlockCompleted: onBlockCompleted)
    }

    func decode(
        tokenID: Int, cancellationCheck: () throws -> Void,
        onBlockCompleted: (Int, Int) -> Void
    ) throws -> DeepSeekV41WorkloadStep {
        state.decoded(tokenID)
        return try pass(cancellationCheck: cancellationCheck, onBlockCompleted: onBlockCompleted)
    }

    private func pass(
        cancellationCheck: () throws -> Void, onBlockCompleted: (Int, Int) -> Void
    ) throws -> DeepSeekV41WorkloadStep {
        for block in 1...blockCount {
            try cancellationCheck()
            if state.blockSeconds > 0 { Thread.sleep(forTimeInterval: state.blockSeconds) }
            onBlockCompleted(block, blockCount)
        }
        var logits = [Float](repeating: 0, count: vocabularySize)
        logits[Self.generated] = 1
        return DeepSeekV41WorkloadStep(
            tokenID: Self.generated, logits: logits, completedBlocks: blockCount)
    }

    func validateTerminalBinding() throws {}
    func shutdown() {}
}
