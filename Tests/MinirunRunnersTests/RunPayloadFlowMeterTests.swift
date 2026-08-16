import Foundation
import MinirunKit
import StorageCore
import XCTest

@testable import MinirunRunners

final class RunPayloadFlowMeterTests: XCTestCase {
    private final class TestClock: @unchecked Sendable {
        private let lock = NSLock()
        private var tick: UInt64
        private var date: Date

        init(tick: UInt64 = 1_000, date: Date = Date(timeIntervalSince1970: 10)) {
            self.tick = tick
            self.date = date
        }

        func now() -> UInt64 {
            lock.lock()
            defer { lock.unlock() }
            return tick
        }

        func wallNow() -> Date {
            lock.lock()
            defer { lock.unlock() }
            return date
        }

        func advance(nanoseconds: UInt64, wallSeconds: TimeInterval = 0.5) {
            lock.lock()
            tick += nanoseconds
            date = date.addingTimeInterval(wallSeconds)
            lock.unlock()
        }
    }

    func testFixedWindowsKeepCategoriesAndUseMonotonicTime() async throws {
        let clock = TestClock()
        let meter = RunPayloadFlowMeter(
            startsAutomatically: false,
            clock: { clock.now() }, wallClock: { clock.wallNow() })
        var iterator = meter.stream.makeAsyncIterator()

        meter.recordDeterministic(300)
        meter.recordExpert(200)
        clock.advance(nanoseconds: 500_000_000, wallSeconds: 8_000)
        meter.emitWindow(final: false)

        let firstValue = await iterator.next()
        let first = try XCTUnwrap(firstValue)
        XCTAssertEqual(first.sequence, 0)
        XCTAssertEqual(first.intervalSeconds, 0.5, accuracy: 0.000_001)
        XCTAssertEqual(first.deterministicBytes, 300)
        XCTAssertEqual(first.expertBytes, 200)
        XCTAssertEqual(first.totalBytes, 500)
        XCTAssertEqual(first.deterministicBytesPerSecond, 600)
        XCTAssertEqual(first.expertBytesPerSecond, 400)
        XCTAssertTrue(first.isValid)

        clock.advance(nanoseconds: 500_000_000)
        meter.emitWindow(final: false)
        let zeroValue = await iterator.next()
        let zero = try XCTUnwrap(zeroValue)
        XCTAssertEqual(zero.sequence, 1)
        XCTAssertEqual(zero.totalBytes, 0, "idle windows remain visible to the chart")

        XCTAssertEqual(
            meter.snapshot,
            RunPayloadFlowMeter.Snapshot(
                deterministicBytes: 300, expertBytes: 200, didOverflow: false))
        meter.finish()
        let end = await iterator.next()
        XCTAssertNil(end)
    }

    func testFinalPartialWindowFlushesExactlyOnceAndIgnoresLateReads() async throws {
        let clock = TestClock()
        let meter = RunPayloadFlowMeter(
            startsAutomatically: false,
            clock: { clock.now() }, wallClock: { clock.wallNow() })
        var iterator = meter.stream.makeAsyncIterator()

        meter.recordExpert(17)
        clock.advance(nanoseconds: 125_000_000, wallSeconds: 0.125)
        meter.finish()
        meter.finish()
        meter.recordExpert(99)

        let finalValue = await iterator.next()
        let final = try XCTUnwrap(finalValue)
        XCTAssertEqual(final.sequence, 0)
        XCTAssertEqual(final.intervalSeconds, 0.125, accuracy: 0.000_001)
        XCTAssertEqual(final.expertBytes, 17)
        let end = await iterator.next()
        XCTAssertNil(end)
        XCTAssertEqual(meter.snapshot.expertBytes, 17)
    }

    func testOverflowIsStickyAndInvalidatesEveryLaterSample() async throws {
        let clock = TestClock()
        let meter = RunPayloadFlowMeter(
            startsAutomatically: false,
            clock: { clock.now() }, wallClock: { clock.wallNow() })
        var iterator = meter.stream.makeAsyncIterator()

        meter.recordDeterministic(.max)
        meter.recordDeterministic(1)
        clock.advance(nanoseconds: 500_000_000)
        meter.emitWindow(final: false)
        let overflowValue = await iterator.next()
        let overflow = try XCTUnwrap(overflowValue)
        XCTAssertTrue(overflow.didOverflow)
        XCTAssertNil(overflow.totalBytes)
        XCTAssertFalse(overflow.isValid)

        clock.advance(nanoseconds: 500_000_000)
        meter.emitWindow(final: false)
        let laterValue = await iterator.next()
        let later = try XCTUnwrap(laterValue)
        XCTAssertTrue(later.didOverflow, "exactness cannot recover later in the same run")
        XCTAssertFalse(later.isValid)
        XCTAssertTrue(meter.snapshot.didOverflow)
        meter.finish()
    }

    /// Keeps the callback added to every successful payload read visible as a
    /// benchmark rather than assuming that one lock is free. The result is a
    /// local regression diagnostic, not a product throughput claim.
    func testRecordHotPathBenchmark() {
        let meter = RunPayloadFlowMeter(startsAutomatically: false)
        measure(metrics: [XCTClockMetric()]) {
            for _ in 0..<100_000 { meter.recordDeterministic(4_096) }
        }
        meter.finish()
    }
}
