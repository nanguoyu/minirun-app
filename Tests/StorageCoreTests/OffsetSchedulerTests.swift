import Foundation
import XCTest

@testable import StorageCore

final class OffsetSchedulerTests: XCTestCase {
    private func scheduler(
        fileSize: UInt64,
        readSize: Int,
        pattern: AccessPattern,
        worker: Int = 0,
        workers: Int = 1,
        seed: UInt64 = 42
    ) -> OffsetScheduler {
        OffsetScheduler(
            fileSizeBytes: fileSize, readSizeBytes: readSize, pattern: pattern,
            workerIndex: worker, workerCount: workers, seed: seed)
    }

    func testBlockCountIgnoresTrailingPartialBlock() {
        XCTAssertEqual(OffsetScheduler.blockCount(fileSizeBytes: 1000, readSizeBytes: 100), 10)
        XCTAssertEqual(OffsetScheduler.blockCount(fileSizeBytes: 1050, readSizeBytes: 100), 10)
        XCTAssertEqual(OffsetScheduler.blockCount(fileSizeBytes: 99, readSizeBytes: 100), 0)
        XCTAssertEqual(OffsetScheduler.blockCount(fileSizeBytes: 100, readSizeBytes: 100), 1)
    }

    /// The core safety property: a scheduled read can never run past EOF, so a
    /// short read always means something went wrong rather than "the file
    /// ended".
    func testRandomOffsetsNeverCrossEndOfFile() {
        let fileSize: UInt64 = 1_048_576 + 777  // deliberately not a multiple
        let readSize = 4096
        var scheduler = scheduler(fileSize: fileSize, readSize: readSize, pattern: .random)

        for _ in 0..<20_000 {
            let offset = try! XCTUnwrap(scheduler.nextOffset())
            XCTAssertLessThanOrEqual(offset + UInt64(readSize), fileSize)
            XCTAssertEqual(offset % UInt64(readSize), 0, "offsets must be read-size aligned")
        }
    }

    func testSequentialOffsetsNeverCrossEndOfFile() {
        let fileSize: UInt64 = 65_536 + 13
        let readSize = 1024
        for worker in 0..<4 {
            var scheduler = scheduler(
                fileSize: fileSize, readSize: readSize, pattern: .sequential,
                worker: worker, workers: 4)
            for _ in 0..<500 {
                let offset = try! XCTUnwrap(scheduler.nextOffset())
                XCTAssertLessThanOrEqual(offset + UInt64(readSize), fileSize)
                XCTAssertEqual(offset % UInt64(readSize), 0)
            }
        }
    }

    func testSequentialWorkerAdvancesOneBlockAtATime() {
        var scheduler = scheduler(fileSize: 10_240, readSize: 1024, pattern: .sequential)
        var expected: UInt64 = 0
        for _ in 0..<10 {
            XCTAssertEqual(scheduler.nextOffset(), expected)
            expected += 1024
        }
        // Wraps to the start of its own range rather than running off the end.
        XCTAssertEqual(scheduler.nextOffset(), 0)
    }

    func testSequentialWorkersGetDisjointRanges() {
        let workers = 4
        let readSize = 1024
        let fileSize: UInt64 = UInt64(readSize) * 40
        var seen: [Set<UInt64>] = []

        for worker in 0..<workers {
            var scheduler = scheduler(
                fileSize: fileSize, readSize: readSize, pattern: .sequential,
                worker: worker, workers: workers)
            var offsets = Set<UInt64>()
            for _ in 0..<10 { offsets.insert(try! XCTUnwrap(scheduler.nextOffset())) }
            XCTAssertEqual(offsets.count, 10, "worker \(worker) should cover 10 distinct blocks")
            seen.append(offsets)
        }

        for a in 0..<workers {
            for b in (a + 1)..<workers {
                XCTAssertTrue(
                    seen[a].isDisjoint(with: seen[b]),
                    "workers \(a) and \(b) overlap")
            }
        }
        // Together they cover the whole file.
        XCTAssertEqual(seen.reduce(into: Set<UInt64>()) { $0.formUnion($1) }.count, 40)
    }

    /// More workers than blocks must not leave a worker with an empty range,
    /// which would divide by zero or stall it.
    func testMoreWorkersThanBlocksFallBackToWholeFile() {
        let readSize = 1024
        let fileSize = UInt64(readSize) * 2
        for worker in 0..<8 {
            var scheduler = scheduler(
                fileSize: fileSize, readSize: readSize, pattern: .sequential,
                worker: worker, workers: 8)
            let offset = try! XCTUnwrap(scheduler.nextOffset())
            XCTAssertLessThanOrEqual(offset + UInt64(readSize), fileSize)
        }
    }

    func testEmptyFileYieldsNoOffsets() {
        var random = scheduler(fileSize: 0, readSize: 4096, pattern: .random)
        XCTAssertNil(random.nextOffset())
        var short = scheduler(fileSize: 100, readSize: 4096, pattern: .sequential)
        XCTAssertNil(short.nextOffset())
    }

    func testRandomSequenceIsReproducibleForASeed() {
        var first = scheduler(fileSize: 1 << 20, readSize: 4096, pattern: .random, seed: 7)
        var second = scheduler(fileSize: 1 << 20, readSize: 4096, pattern: .random, seed: 7)
        for _ in 0..<100 {
            XCTAssertEqual(first.nextOffset(), second.nextOffset())
        }
    }

    func testDifferentSeedsAndWorkersGiveDifferentSequences() {
        var a = scheduler(fileSize: 1 << 20, readSize: 4096, pattern: .random, seed: 7)
        var b = scheduler(fileSize: 1 << 20, readSize: 4096, pattern: .random, seed: 8)
        var c = scheduler(
            fileSize: 1 << 20, readSize: 4096, pattern: .random, worker: 1, workers: 4, seed: 7)

        let first = (0..<50).map { _ in a.nextOffset() }
        let second = (0..<50).map { _ in b.nextOffset() }
        let third = (0..<50).map { _ in c.nextOffset() }
        XCTAssertNotEqual(first, second)
        XCTAssertNotEqual(first, third, "workers must not walk the same random sequence")
    }

    /// Offset selection is derived from the run seed, but must not simply be
    /// the file-content stream: a scheduler that walked the content stream
    /// would correlate offsets with the bytes stored there.
    func testOffsetStreamDiffersFromContentStream() {
        var scheduler = self.scheduler(fileSize: 1 << 24, readSize: 8, pattern: .random, seed: 3)
        let offsets = (0..<32).map { _ in scheduler.nextOffset()! / 8 }
        let content = (0..<32).map { SplitMix64.value(seed: 3, index: UInt64($0)) % (1 << 21) }
        XCTAssertNotEqual(offsets, content)
    }

    func testRandomOffsetsCoverTheFileBroadly() {
        let readSize = 4096
        let blocks: UInt64 = 256
        var scheduler = scheduler(
            fileSize: UInt64(readSize) * blocks, readSize: readSize, pattern: .random)
        var seen = Set<UInt64>()
        for _ in 0..<4096 { seen.insert(scheduler.nextOffset()!) }
        // With 4096 draws over 256 blocks, missing any is vanishingly unlikely.
        XCTAssertEqual(seen.count, Int(blocks))
    }
}

final class LatencyStatisticsTests: XCTestCase {
    func testPercentilesOnKnownInput() {
        // 1...100 nanoseconds; nearest-rank percentiles are exact values.
        let sorted = (1...100).map { UInt64($0) }
        XCTAssertEqual(LatencyStatistics.percentile(50, ofSorted: sorted), 50)
        XCTAssertEqual(LatencyStatistics.percentile(95, ofSorted: sorted), 95)
        XCTAssertEqual(LatencyStatistics.percentile(99, ofSorted: sorted), 99)
        XCTAssertEqual(LatencyStatistics.percentile(100, ofSorted: sorted), 100)
        XCTAssertEqual(LatencyStatistics.percentile(1, ofSorted: sorted), 1)
        XCTAssertEqual(LatencyStatistics.percentile(0, ofSorted: sorted), 1)
    }

    func testPercentilesOnSingleSample() {
        let sorted: [UInt64] = [7]
        for percent in [0.0, 50.0, 99.0, 100.0] {
            XCTAssertEqual(LatencyStatistics.percentile(percent, ofSorted: sorted), 7)
        }
    }

    func testPercentileOnEmptyInput() {
        XCTAssertEqual(LatencyStatistics.percentile(50, ofSorted: []), 0)
    }

    func testComputeConvertsNanosecondsToMicroseconds() throws {
        var samples: [UInt64] = [1_000, 2_000, 3_000, 4_000]
        let stats = try XCTUnwrap(
            LatencyStatistics.compute(nanosecondSamples: &samples))
        XCTAssertEqual(stats.sampleCount, 4)
        XCTAssertEqual(stats.minUs, 1.0, accuracy: 1e-9)
        XCTAssertEqual(stats.maxUs, 4.0, accuracy: 1e-9)
        XCTAssertEqual(stats.meanUs, 2.5, accuracy: 1e-9)
        XCTAssertEqual(stats.p50Us, 2.0, accuracy: 1e-9)
    }

    func testComputeSortsUnorderedInput() throws {
        var samples: [UInt64] = [5_000, 1_000, 4_000, 2_000, 3_000]
        let stats = try XCTUnwrap(LatencyStatistics.compute(nanosecondSamples: &samples))
        XCTAssertEqual(stats.minUs, 1.0, accuracy: 1e-9)
        XCTAssertEqual(stats.maxUs, 5.0, accuracy: 1e-9)
        XCTAssertEqual(stats.p50Us, 3.0, accuracy: 1e-9)
        XCTAssertEqual(samples, samples.sorted(), "compute sorts in place")
    }

    func testEmptySamplesProduceNoStatistics() {
        var samples: [UInt64] = []
        XCTAssertNil(LatencyStatistics.compute(nanosecondSamples: &samples))
    }

    func testPercentilesAreObservedValues() throws {
        // No interpolation: every reported percentile must appear in the input.
        var samples: [UInt64] = [10, 20, 30, 40, 50, 60, 70]
        let set = Set(samples)
        let stats = try XCTUnwrap(LatencyStatistics.compute(nanosecondSamples: &samples))
        for value in [stats.p50Us, stats.p95Us, stats.p99Us, stats.minUs, stats.maxUs] {
            XCTAssertTrue(set.contains(UInt64(value * 1_000)), "\(value) us is not an observed sample")
        }
    }
}
