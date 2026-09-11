import Foundation
import MinirunKit
import MinirunRunners
import XCTest

@testable import MinirunApp

/// The flight recorder: what it writes, when it writes it, and what it keeps.
///
/// Every assertion here is about a file that survives the process, because that
/// is the whole of the feature. A Jetsam kill produces no terminal event and no
/// chance to flush, so a trace is only worth having if the line that returned
/// is the line the kernel has.
final class RunTraceRecorderTests: XCTestCase {

    private var directory = URL(fileURLWithPath: NSTemporaryDirectory())

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-run-traces-\(UUID().uuidString)", isDirectory: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func start(at date: Date = Date()) -> RunTraceRecorder.Start {
        RunTraceRecorder.Start(
            at: date, conversation: UUID().uuidString, model: "deepseek-v41-flash",
            scale: "product", declaredBudgetBytes: 1_900_000_000, maximumNewTokens: 64,
            promptTokenCount: 11, policy: nil, pinPlan: nil,
            footprintBytes: 1_000, residentBytes: 2_000, availableBytes: 5_000_000_000,
            gitRevision: "abcdef")
    }

    private func sample(elapsed: TimeInterval, block: Int?) -> RunTraceRecorder.Sample {
        RunTraceRecorder.Sample(
            elapsed: elapsed, stage: "prefill", phase: "prefill", block: block,
            blockCount: 40, footprintBytes: 3_000, residentBytes: 4_000,
            budgetedFootprintBytes: 1_500, peakFootprintBytes: 1_600,
            entryFootprintBytes: 900, declaredBudgetBytes: 1_900_000_000,
            availableBytes: 4_000_000_000, mlxActiveBytes: 10, mlxCacheBytes: 20,
            sentryChecks: 7, sentryWorstOvershootBytes: 0, tokens: 0)
    }

    private func lines(in url: URL) throws -> [[String: Any]] {
        let text = try String(contentsOf: url, encoding: .utf8)
        return try text.split(separator: "\n").map { line in
            let object = try JSONSerialization.jsonObject(
                with: Data(line.utf8), options: [])
            return try XCTUnwrap(object as? [String: Any])
        }
    }

    /// A trace is NDJSON: one object per line, `start`, then samples, then
    /// `end`, and each of them readable on its own.
    func testATraceIsOneJSONObjectPerLineInTheOrderTheRunProducedThem() throws {
        let recorder = try XCTUnwrap(
            RunTraceRecorder(
                directory: directory, conversationID: UUID(), startedAt: Date()))
        recorder.record(start())
        recorder.record(sample(elapsed: 0.5, block: 3))
        recorder.record(sample(elapsed: 1.5, block: 14))
        recorder.record(
            RunTraceRecorder.End(
                at: Date(), elapsed: 2, outcome: "finished", tokens: 1,
                peakFootprintBytes: 1_600, budgetRespected: true, namedError: nil))

        let objects = try lines(in: recorder.url)
        XCTAssertEqual(objects.map { $0["kind"] as? String }, [
            "start", "sample", "sample", "end",
        ])
        XCTAssertEqual(objects[0]["model"] as? String, "deepseek-v41-flash")
        XCTAssertEqual(objects[0]["scale"] as? String, "product")
        XCTAssertEqual(objects[1]["block"] as? Int, 3)
        // Both footprints, on every sample: `phys_footprint` is what iOS
        // enforces against and `resident_size` is roughly what a Jetsam report
        // prints, and a trace that carried one of them could not say which grew.
        XCTAssertEqual(objects[2]["footprintBytes"] as? UInt64, 3_000)
        XCTAssertEqual(objects[2]["residentBytes"] as? UInt64, 4_000)
        XCTAssertEqual(objects[2]["budgetedFootprintBytes"] as? UInt64, 1_500)
        XCTAssertEqual(objects[2]["sentryChecks"] as? UInt64, 7)
        XCTAssertEqual(objects[3]["outcome"] as? String, "finished")
    }

    /// **A killed run leaves the lines it had already written.**
    ///
    /// Checked by reading the file from a second handle while the recorder is
    /// still open, which is the closest a test can stand to `SIGKILL`: if a
    /// line were buffered in the process, it would not be there.
    func testEveryLineIsOnDiskBeforeTheCallReturns() throws {
        let recorder = try XCTUnwrap(
            RunTraceRecorder(
                directory: directory, conversationID: UUID(), startedAt: Date()))
        recorder.record(start())
        XCTAssertEqual(try lines(in: recorder.url).count, 1)
        recorder.record(sample(elapsed: 0.5, block: 1))
        XCTAssertEqual(try lines(in: recorder.url).count, 2)
        recorder.record(sample(elapsed: 0.9, block: 2))
        let midRun = try lines(in: recorder.url)
        XCTAssertEqual(midRun.count, 3)
        XCTAssertEqual(midRun.last?["kind"] as? String, "sample")
        XCTAssertNil(
            midRun.last?["outcome"],
            "a trace with no end line is what a kill looks like, and that is the point")
    }

    /// The newest twenty stay. An app container a person has to copy off a
    /// phone is not a place to accumulate a file per chat forever.
    func testOnlyTheNewestTracesAreKept() throws {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        var written: [URL] = []
        for index in 0..<25 {
            let url = directory.appendingPathComponent("trace-\(index).ndjson")
            try Data("{}\n".utf8).write(to: url)
            try manager.setAttributes(
                [.modificationDate: Date(timeIntervalSince1970: Double(index))], ofItemAtPath: url.path)
            written.append(url)
        }
        // A recorder prunes to `keep - 1` before opening its own file, so the
        // directory holds exactly `keep` traces once the new one exists.
        let recorder = try XCTUnwrap(
            RunTraceRecorder(
                directory: directory, conversationID: UUID(), startedAt: Date()))
        recorder.record(start())

        let remaining = try manager.contentsOfDirectory(atPath: directory.path)
            .filter { $0.hasSuffix(".ndjson") }
        XCTAssertEqual(remaining.count, RunTraceRecorder.Retention.keep)
        XCTAssertTrue(remaining.contains(recorder.url.lastPathComponent))
        XCTAssertTrue(
            remaining.contains("trace-24.ndjson"), "the newest of the old traces stays")
        XCTAssertFalse(
            remaining.contains("trace-0.ndjson"), "the oldest does not")
    }

    /// The name a `devicectl` pull produces: the conversation, then when the
    /// run started, with no colon in it.
    func testTheFilenameNamesTheConversationAndTheStart() {
        let id = UUID(uuidString: "3F2504E0-4F89-11D3-9A0C-0305E82C3301")!
        let name = RunTraceRecorder.filename(
            conversationID: id, startedAt: Date(timeIntervalSince1970: 1_789_000_000))
        XCTAssertTrue(name.hasPrefix("3F2504E0-4F89-11D3-9A0C-0305E82C3301-"))
        XCTAssertTrue(name.hasSuffix(".ndjson"))
        XCTAssertFalse(name.contains(":"))
        XCTAssertTrue(name.contains("T"), "the timestamp is ISO 8601's basic form")
    }

    // MARK: The heartbeat

    /// **The silence this closes.**
    ///
    /// The recorder used to write a `sample` per telemetry event, and a V4.1
    /// runner published its first telemetry after the first block of the first
    /// pass — after the artifact was opened, forty-one manifests reconciled and
    /// the model built. On 2026-09-11 the owner's iPhone was killed three times
    /// inside that silence and left traces with a start line and nothing else.
    /// The heartbeat samples whether or not the runner has said anything.
    func testTheHeartbeatWritesSamplesWithNoRunnerEventAtAll() throws {
        let recorder = try XCTUnwrap(
            RunTraceRecorder(
                directory: directory, conversationID: UUID(), startedAt: Date()))
        recorder.record(start())
        var footprint: UInt64 = 1_000_000
        recorder.beginHeartbeat(
            declaredBudgetBytes: 1_900_000_000,
            startedAt: Date(timeIntervalSince1970: 100),
            interval: 3_600,  // never fires on its own; the body is driven below
            memory: {
                footprint += 268_435_456
                return (footprint: footprint, resident: footprint / 2, available: 6_000_000_000)
            },
            now: { Date(timeIntervalSince1970: 100.4) })

        // Three heartbeats, all of them before the runner has produced one
        // event: this is exactly the stretch the phone died in.
        for _ in 0..<3 { recorder.recordHeartbeatSample() }

        let objects = try lines(in: recorder.url)
        XCTAssertEqual(objects.map { $0["kind"] as? String }, [
            "start", "sample", "sample", "sample",
        ])
        XCTAssertEqual(
            objects.dropFirst().map { $0["source"] as? String },
            ["timer", "timer", "timer"])
        // The stage nobody had told it yet, which is the truthful one.
        XCTAssertEqual(objects[1]["stage"] as? String, "preparing")
        XCTAssertEqual(objects[1]["phase"] as? String, "submitted")
        XCTAssertEqual(objects[1]["footprintBytes"] as? UInt64, 269_435_456)
        XCTAssertEqual(objects[3]["footprintBytes"] as? UInt64, 806_306_368)
        XCTAssertEqual(objects[2]["residentBytes"] as? UInt64, 268_935_456)
        XCTAssertEqual(
            try XCTUnwrap(objects[1]["elapsed"] as? Double), 0.4, accuracy: 1e-6)
        XCTAssertEqual(objects[1]["declaredBudgetBytes"] as? UInt64, 1_900_000_000)
        XCTAssertNil(objects[1]["block"], "preparation is not inside a block")
    }

    /// A timer line repeats the run-basis numbers the runner last published,
    /// and says so with `source`, so a reader is never left joining lines to
    /// find out what a footprint should be compared against.
    func testATimerLineCarriesTheBasisTheRunnerLastPublished() throws {
        let recorder = try XCTUnwrap(
            RunTraceRecorder(
                directory: directory, conversationID: UUID(), startedAt: Date()))
        recorder.record(start())
        recorder.beginHeartbeat(
            declaredBudgetBytes: 1_900_000_000,
            startedAt: Date(timeIntervalSince1970: 0),
            interval: 3_600,
            memory: { (footprint: 7_000, resident: 8_000, available: nil) },
            now: { Date(timeIntervalSince1970: 2) })

        recorder.noteStage(.prefill, phase: "prefill", block: (block: 12, count: 40))
        recorder.noteRunnerState(
            telemetry(stage: .prefill, phase: "prefill"),
            block: (block: 12, count: 40), tokens: 0)
        let line = try XCTUnwrap(recorder.recordHeartbeatSample())

        XCTAssertEqual(line.source, "timer")
        XCTAssertEqual(line.stage, "prefill")
        XCTAssertEqual(line.block, 12)
        XCTAssertEqual(line.blockCount, 40)
        // Measured by the recorder itself.
        XCTAssertEqual(line.footprintBytes, 7_000)
        XCTAssertEqual(line.residentBytes, 8_000)
        // Carried from the runner: 3,000 absolute less a 900-byte floor.
        XCTAssertEqual(line.budgetedFootprintBytes, 2_100)
        XCTAssertEqual(line.peakFootprintBytes, 1_600)
        XCTAssertEqual(line.entryFootprintBytes, 900)
        XCTAssertEqual(line.elapsed, 2)
        // Not visible from the App, and a zero that looked measured would be
        // worse than one `source` tells a reader to skip.
        XCTAssertEqual(line.mlxActiveBytes, 0)
        XCTAssertEqual(line.mlxCacheBytes, 0)
    }

    /// A real `DispatchSourceTimer`, actually firing, on a queue that is not the
    /// main one — because the main actor is exactly what is blocked when this
    /// matters.
    func testTheHeartbeatFiresOnItsOwnQueueWithoutBeingDriven() throws {
        let recorder = try XCTUnwrap(
            RunTraceRecorder(
                directory: directory, conversationID: UUID(), startedAt: Date()))
        recorder.record(start())
        let fired = expectation(description: "the heartbeat sampled on its own")
        // Four, so that three of them have certainly finished writing by the
        // time the wait returns: the reading is taken before the line is
        // appended, not after.
        fired.expectedFulfillmentCount = 4
        fired.assertForOverFulfill = false
        recorder.beginHeartbeat(
            declaredBudgetBytes: 1_900_000_000,
            interval: 0.02,
            memory: {
                fired.fulfill()
                return (footprint: 1, resident: 2, available: nil)
            })
        wait(for: [fired], timeout: 5)

        recorder.endHeartbeat()
        // Let whatever reading was already in flight finish its line.
        Thread.sleep(forTimeInterval: 0.15)
        let afterStopping = try lines(in: recorder.url).count
        XCTAssertGreaterThanOrEqual(
            afterStopping, 4, "a start line and at least three heartbeats")

        // Nothing more is written once it is stopped, and stopping twice is not
        // an error: the terminal path may reach it from either direction.
        recorder.endHeartbeat()
        Thread.sleep(forTimeInterval: 0.15)
        XCTAssertEqual(try lines(in: recorder.url).count, afterStopping)
    }

    /// The end line stops the heartbeat, so a trace that has said how the run
    /// ended cannot grow another sample after it.
    func testTheEndLineStopsTheHeartbeat() throws {
        let recorder = try XCTUnwrap(
            RunTraceRecorder(
                directory: directory, conversationID: UUID(), startedAt: Date()))
        recorder.record(start())
        recorder.beginHeartbeat(
            declaredBudgetBytes: 1_900_000_000,
            interval: 0.01,
            memory: { (footprint: 1, resident: 2, available: nil) })
        recorder.record(
            RunTraceRecorder.End(
                at: Date(), elapsed: 1, outcome: "finished", tokens: 1,
                peakFootprintBytes: 10, budgetRespected: true, namedError: nil))
        Thread.sleep(forTimeInterval: 0.2)

        let objects = try lines(in: recorder.url)
        XCTAssertEqual(objects.last?["kind"] as? String, "end")
        XCTAssertEqual(
            objects.filter { ($0["kind"] as? String) == "end" }.count, 1)
    }

    private func telemetry(
        stage: RunGenerationStage, phase: String
    ) -> RunTelemetry {
        RunTelemetry(
            at: Date(), elapsed: 1, phase: phase, tokensPerSecond: nil,
            generationStage: stage,
            bytes: ByteAccounting(totalBytesRead: 0), bytesPerSecond: nil,
            bytesPerToken: nil,
            declaredBudgetBytes: 1_900_000_000,
            footprintBytes: 3_000, peakFootprintBytes: 1_600,
            entryFootprintBytes: 900,
            residentBytes: 4_000, availableBytes: nil,
            mlxActiveBytes: 10, mlxCacheBytes: 20, mlxPeakBytes: 30,
            thermalState: .nominal, lowPowerMode: false, batteryLevel: nil)
    }

    /// The block index comes out of the phase detail the runners already
    /// publish, so a trace line can say where in the pass a footprint was taken.
    func testTheBlockIndexIsReadFromThePhaseDetail() {
        XCTAssertEqual(
            RunTraceRecorder.blockProgress(in: "block 12/40")?.block, 12)
        XCTAssertEqual(
            RunTraceRecorder.blockProgress(in: "block 12/40")?.count, 40)
        XCTAssertNil(RunTraceRecorder.blockProgress(in: "reading prompt"))
        XCTAssertNil(RunTraceRecorder.blockProgress(in: "block twelve/40"))
        XCTAssertNil(RunTraceRecorder.blockProgress(in: ""))
    }
}
