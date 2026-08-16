import Darwin
import Foundation
import XCTest

@testable import StorageCore

/// The threaded read path: read-ahead, backpressure, short reads, cancellation.
final class TileReaderTests: XCTestCase {

    private final class SuccessfulReadRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var recordedBytes: [UInt64] = []

        func record(_ bytes: UInt64) {
            lock.lock()
            recordedBytes.append(bytes)
            lock.unlock()
        }

        var bytes: [UInt64] {
            lock.lock()
            defer { lock.unlock() }
            return recordedBytes
        }
    }

    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-tile-reader-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let directory, FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.removeItem(at: directory)
        }
    }

    /// A file of `regionCount` regions, each `regionBytes` long, where region i
    /// is filled with the byte `i + 1`.
    private func makeFile(regionCount: Int, regionBytes: Int) throws -> String {
        let path = directory.appendingPathComponent("regions.bin").path
        var data = Data(capacity: regionCount * regionBytes)
        for index in 0..<regionCount {
            data.append(
                contentsOf: [UInt8](
                    repeating: UInt8(truncatingIfNeeded: index + 1), count: regionBytes))
        }
        try data.write(to: URL(fileURLWithPath: path))
        return path
    }

    private func regions(count: Int, bytes: Int) -> [RegionDescriptor] {
        (0..<count).map {
            RegionDescriptor(
                identity: RegionIdentity(sourceID: 1, index: UInt64($0)),
                offset: UInt64($0 * bytes),
                length: bytes)
        }
    }

    private func makePool(slots: Int, bytes: Int) throws -> BufferPool {
        try BufferPool(
            configuration: BufferPoolConfiguration(slotCount: slots, slotBytes: bytes))
    }

    // MARK: - Happy path

    func testStreamsEveryRegionInOrderWithCorrectBytes() throws {
        let regionBytes = 4096
        let count = 40
        let path = try makeFile(regionCount: count, regionBytes: regionBytes)
        let pool = try makePool(slots: 3, bytes: regionBytes)
        let recorder = SuccessfulReadRecorder()
        let reader = try TileReader(
            path: path, pool: pool,
            configuration: TileReaderConfiguration(queueDepth: 2, readAhead: 2),
            successfulReadObserver: recorder.record)
        defer { reader.shutdown() }

        let descriptors = regions(count: count, bytes: regionBytes)
        try reader.schedule(descriptors)

        for (index, descriptor) in descriptors.enumerated() {
            let lease = try reader.acquire(descriptor)
            XCTAssertEqual(lease.region.identity, descriptor.identity)
            XCTAssertEqual(lease.contents.count, regionBytes)
            let expected = UInt8(truncatingIfNeeded: index + 1)
            XCTAssertTrue(
                lease.contents.allSatisfy { $0 == expected },
                "region \(index) contents are wrong")
            try lease.releaseChecked()
        }

        let statistics = reader.statistics
        XCTAssertEqual(statistics.readCount, count)
        XCTAssertEqual(statistics.bytesRead, UInt64(count * regionBytes))
        XCTAssertEqual(statistics.shortReadCount, 0)
        XCTAssertEqual(statistics.errorCount, 0)
        XCTAssertEqual(recorder.bytes.count, count)
        XCTAssertEqual(recorder.bytes.reduce(0, +), UInt64(count * regionBytes))
        XCTAssertNotNil(statistics.latency)
        XCTAssertEqual(pool.census.inUse, 0)
        XCTAssertEqual(pool.statistics.faults, [])
    }

    /// A consumer that releases every lease synchronously needs no slot in
    /// addition to the read-ahead window. This is the lower-level progress
    /// argument used by the V4 product's copying expert path: the reader fills
    /// both slots, delivery/release of the head wakes the dispatcher, and the
    /// remaining regions continue without an unevaluated graph retaining a
    /// slot.
    func testSynchronousConsumerStreamsWithSlotsEqualToReadAhead() throws {
        let regionBytes = 4096
        let count = 32
        let path = try makeFile(regionCount: count, regionBytes: regionBytes)
        let pool = try makePool(slots: 2, bytes: regionBytes)
        let reader = try TileReader(
            path: path, pool: pool,
            configuration: TileReaderConfiguration(
                queueDepth: 2, readAhead: 2, slotAcquireTimeout: 5))
        defer { reader.shutdown() }

        let descriptors = regions(count: count, bytes: regionBytes)
        try reader.schedule(descriptors)
        for (index, descriptor) in descriptors.enumerated() {
            let lease = try reader.acquire(descriptor)
            XCTAssertTrue(
                lease.contents.allSatisfy { $0 == UInt8(truncatingIfNeeded: index + 1) })
            try lease.releaseChecked()
        }

        XCTAssertEqual(reader.statistics.readCount, count)
        XCTAssertEqual(pool.statistics.peakInUseSlots, 2)
        XCTAssertEqual(pool.census.free, 2)
        XCTAssertTrue(pool.statistics.faults.isEmpty)
    }

    func testOwningDescriptorReadsTheAuthorizedObjectAndClosesItOnShutdown() throws {
        let regionBytes = 4096
        let path = try makeFile(regionCount: 2, regionBytes: regionBytes)
        let descriptor = Darwin.open(path, O_RDONLY | O_CLOEXEC)
        XCTAssertGreaterThanOrEqual(descriptor, 0)
        let pool = try makePool(slots: 2, bytes: regionBytes)
        let reader = try TileReader(
            owningFileDescriptor: descriptor, path: "authorized-regions", pool: pool,
            configuration: TileReaderConfiguration(queueDepth: 1, readAhead: 1))

        try Data(repeating: 9, count: 2 * regionBytes).write(
            to: URL(fileURLWithPath: path), options: .atomic)
        let descriptorToRead = regions(count: 2, bytes: regionBytes)[0]
        let lease = try reader.acquire(descriptorToRead)
        XCTAssertTrue(lease.contents.allSatisfy { $0 == 1 })
        try lease.releaseChecked()

        reader.shutdown()
        errno = 0
        XCTAssertEqual(fcntl(descriptor, F_GETFD), -1)
        XCTAssertEqual(errno, EBADF)
    }

    /// Ten times as many regions as slots, with the read-ahead window filled.
    /// The pool must run exactly full and must never exceed the budget.
    func testBoundedPoolStreamsManyMoreRegionsThanSlots() throws {
        let regionBytes = 8192
        let slots = 4
        let count = slots * 10
        let path = try makeFile(regionCount: count, regionBytes: regionBytes)
        let pool = try makePool(slots: slots, bytes: regionBytes)
        let reader = try TileReader(
            path: path, pool: pool,
            configuration: TileReaderConfiguration(
                queueDepth: 3, readAhead: 3, slotAcquireTimeout: 30))
        defer { reader.shutdown() }

        let descriptors = regions(count: count, bytes: regionBytes)
        try reader.schedule(descriptors)

        // Hold `slots - 1` leases at a time, standing in for the tiles an
        // unevaluated compute graph would be pinning.
        //
        // The count is chosen so that backpressure is structural rather than
        // lucky. With 3 of 4 slots held, the reader can never have more than
        // one in flight, while its read-ahead window asks for three — so the
        // dispatcher is always back in `pool.acquire` with nothing available.
        // An earlier version of this test held only one lease and asserted the
        // same thing; it passed almost always and then failed, because with a
        // warm 320 KiB file the reader could refill fast enough that no
        // acquisition ever had to wait. Reaching peak occupancy does not imply
        // anyone blocked: the pool can touch 4 and drop back before the next
        // request arrives.
        let concurrentHolds = slots - 1
        var held: [SlotLease] = []
        for descriptor in descriptors {
            held.append(try reader.acquire(descriptor))
            if held.count > concurrentHolds - 1 {
                try held.removeFirst().releaseChecked()
            }
        }
        for lease in held { try lease.releaseChecked() }

        let statistics = pool.statistics
        XCTAssertEqual(statistics.peakInUseSlots, slots, "the pool should have run exactly full")
        XCTAssertLessThanOrEqual(statistics.peakInUseSlots, slots)
        XCTAssertGreaterThan(
            statistics.blockedAcquireCount, 0, "backpressure never applied — not a bounded run")
        XCTAssertGreaterThan(reader.statistics.slotWaitSeconds, 0)
        XCTAssertEqual(reader.statistics.readCount, count)
        XCTAssertEqual(pool.census.free, slots)
    }

    /// Acquiring in an order the caller did not schedule: the routed-expert
    /// shape, where the router decides after the prefetch was issued.
    func testAcquireOutOfScheduleOrder() throws {
        let regionBytes = 4096
        let count = 8
        let path = try makeFile(regionCount: count, regionBytes: regionBytes)
        let pool = try makePool(slots: 3, bytes: regionBytes)
        let reader = try TileReader(
            path: path, pool: pool,
            configuration: TileReaderConfiguration(queueDepth: 2, readAhead: 2))
        defer { reader.shutdown() }

        let descriptors = regions(count: count, bytes: regionBytes)
        try reader.schedule(descriptors)

        for index in [7, 0, 5, 2, 6, 1, 4, 3] {
            let lease = try reader.acquire(descriptors[index])
            let expected = UInt8(truncatingIfNeeded: index + 1)
            XCTAssertTrue(
                lease.contents.allSatisfy { $0 == expected }, "region \(index) contents are wrong")
            try lease.releaseChecked()
        }
        XCTAssertEqual(pool.census.inUse, 0)
    }

    func testAcquiringAnUnscheduledRegionReadsItOnDemand() throws {
        let regionBytes = 4096
        let path = try makeFile(regionCount: 4, regionBytes: regionBytes)
        let pool = try makePool(slots: 2, bytes: regionBytes)
        let reader = try TileReader(
            path: path, pool: pool,
            configuration: TileReaderConfiguration(queueDepth: 1, readAhead: 1))
        defer { reader.shutdown() }

        let descriptors = regions(count: 4, bytes: regionBytes)
        let lease = try reader.acquire(descriptors[2])
        XCTAssertTrue(lease.contents.allSatisfy { $0 == 3 })
        try lease.releaseChecked()
    }

    // MARK: - Explicit failure

    func testRegionLargerThanASlotIsRejected() throws {
        let path = try makeFile(regionCount: 2, regionBytes: 8192)
        let pool = try makePool(slots: 2, bytes: 4096)
        let reader = try TileReader(path: path, pool: pool)
        defer { reader.shutdown() }

        let oversized = RegionDescriptor(
            identity: RegionIdentity(sourceID: 1, index: 0), offset: 0, length: 8192)
        XCTAssertThrowsError(try reader.schedule([oversized])) { error in
            guard case PagerError.regionTooLarge(_, let length, let slotBytes) = error else {
                return XCTFail("expected regionTooLarge, got \(error)")
            }
            XCTAssertEqual(length, 8192)
            XCTAssertEqual(slotBytes, 4096)
        }
        XCTAssertThrowsError(try reader.acquire(oversized))
    }

    /// An already-invalid public range is refused before it reaches a worker.
    func testOutOfBoundsRegionIsRejectedByName() throws {
        let regionBytes = 4096
        let path = try makeFile(regionCount: 4, regionBytes: regionBytes)
        let pool = try makePool(slots: 2, bytes: regionBytes)
        let reader = try TileReader(
            path: path, pool: pool,
            configuration: TileReaderConfiguration(queueDepth: 1, readAhead: 1))
        defer { reader.shutdown() }

        // Region 5 of a 4-region file: the last 4 KiB do not exist.
        let beyond = RegionDescriptor(
            identity: RegionIdentity(sourceID: 1, index: 5),
            offset: UInt64(5 * regionBytes),
            length: regionBytes)
        XCTAssertThrowsError(try reader.schedule([beyond])) { error in
            guard case PagerError.invalidRegion(_, let offset, let length, let sourceBytes, _) = error
            else { return XCTFail("expected invalidRegion, got \(error)") }
            XCTAssertEqual(offset, UInt64(5 * regionBytes))
            XCTAssertEqual(length, regionBytes)
            XCTAssertEqual(sourceBytes, UInt64(4 * regionBytes))
        }
        XCTAssertThrowsError(try reader.acquire(beyond))
        XCTAssertEqual(reader.statistics.readCount, 0)
    }

    func testInvalidLengthsAndOverflowingEndAreRejectedByName() throws {
        let regionBytes = 4096
        let path = try makeFile(regionCount: 4, regionBytes: regionBytes)
        let pool = try makePool(slots: 2, bytes: regionBytes)
        let reader = try TileReader(
            path: path, pool: pool,
            configuration: TileReaderConfiguration(queueDepth: 1, readAhead: 1))
        defer { reader.shutdown() }

        let invalid = [
            RegionDescriptor(
                identity: RegionIdentity(sourceID: 1, index: 8), offset: 0, length: -1),
            RegionDescriptor(
                identity: RegionIdentity(sourceID: 1, index: 9), offset: 0, length: 0),
            RegionDescriptor(
                identity: RegionIdentity(sourceID: 1, index: 10),
                offset: UInt64.max - UInt64(regionBytes / 2), length: regionBytes),
        ]
        for region in invalid {
            XCTAssertThrowsError(try reader.schedule([region])) { error in
                guard case PagerError.invalidRegion = error else {
                    return XCTFail("expected invalidRegion, got \(error)")
                }
            }
        }
        XCTAssertEqual(reader.statistics.readCount, 0)
    }

    /// A range that was valid when opened can still short-read after a live
    /// source is truncated or disconnected. That remains a worker error rather
    /// than being confused with an invalid manifest range.
    func testSourceTruncatedAfterOpenSurfacesShortRead() throws {
        let regionBytes = 4096
        let path = try makeFile(regionCount: 4, regionBytes: regionBytes)
        let pool = try makePool(slots: 2, bytes: regionBytes)
        let recorder = SuccessfulReadRecorder()
        let reader = try TileReader(
            path: path, pool: pool,
            configuration: TileReaderConfiguration(queueDepth: 1, readAhead: 1),
            successfulReadObserver: recorder.record)
        defer { reader.shutdown() }

        let last = regions(count: 4, bytes: regionBytes)[3]
        let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
        try handle.truncate(atOffset: UInt64(3 * regionBytes + regionBytes / 2))
        try handle.close()

        try reader.schedule([last])
        XCTAssertThrowsError(try reader.acquire(last)) { error in
            guard case StorageCoreError.shortTransfer(_, _, _, let expected, let actual) = error
            else { return XCTFail("expected shortTransfer, got \(error)") }
            XCTAssertEqual(expected, regionBytes)
            XCTAssertEqual(actual, regionBytes / 2)
        }
        XCTAssertEqual(reader.statistics.shortReadCount, 1)
        XCTAssertEqual(
            recorder.bytes, [],
            "a partial transfer must not be presented as completed model payload")
        XCTAssertEqual(pool.census.free, 2)
    }

    // MARK: - Cancellation

    func testCancellationDuringFillingReleasesEverySlot() throws {
        let regionBytes = 1 << 20
        let count = 64
        let path = try makeFile(regionCount: count, regionBytes: regionBytes)
        let pool = try makePool(slots: 4, bytes: regionBytes)
        let reader = try TileReader(
            path: path, pool: pool,
            configuration: TileReaderConfiguration(queueDepth: 2, readAhead: 3))

        let descriptors = regions(count: count, bytes: regionBytes)
        try reader.schedule(descriptors)
        // Consume a few, then cancel with reads still in flight.
        for index in 0..<3 {
            try reader.acquire(descriptors[index]).releaseChecked()
        }
        reader.cancel(reason: "drive removed")

        XCTAssertThrowsError(try reader.acquire(descriptors[10])) { error in
            guard case PagerError.cancelled(let reason) = error else {
                return XCTFail("expected cancelled, got \(error)")
            }
            XCTAssertEqual(reason, "drive removed")
        }

        reader.shutdown()
        let census = pool.census
        XCTAssertEqual(census.inUse, 0, "slots leaked after cancellation: \(census)")
        XCTAssertEqual(census.free, 4)
        XCTAssertEqual(pool.statistics.faults, [])
    }

    /// Cancellation while a consumer is blocked waiting for a region.
    func testCancellationWakesALeaseWaiter() throws {
        let regionBytes = 4096
        let path = try makeFile(regionCount: 4, regionBytes: regionBytes)
        let pool = try makePool(slots: 2, bytes: regionBytes)
        let reader = try TileReader(
            path: path, pool: pool,
            configuration: TileReaderConfiguration(queueDepth: 1, readAhead: 1))

        // Never scheduled and never readable: the waiter will sit in acquire.
        let descriptors = regions(count: 4, bytes: regionBytes)
        try reader.schedule([descriptors[0]])
        try reader.acquire(descriptors[0]).releaseChecked()

        let woke = expectation(description: "waiter woke with cancellation")
        // Hold both slots so the dispatcher cannot make progress on the region
        // the waiter wants.
        let blocker = try pool.acquire()
        let blocker2 = try pool.acquire()
        try reader.schedule([descriptors[1]])

        let thread = Thread {
            do {
                _ = try reader.acquire(descriptors[1])
                XCTFail("acquire should not have succeeded")
            } catch PagerError.cancelled {
                woke.fulfill()
            } catch {
                XCTFail("expected cancelled, got \(error)")
            }
        }
        thread.start()
        XCTAssertTrue(
            reader.waitUntilConsumerIsWaiting(timeout: 2),
            "the consumer never entered TileReader.acquire's condition wait")
        reader.cancel(reason: "cancelled while waiting")
        wait(for: [woke], timeout: 10)

        try pool.discard(blocker)
        try pool.discard(blocker2)
        reader.shutdown()
        XCTAssertEqual(pool.census.inUse, 0)
    }

    /// A second run on a fresh reader over the same pool and file must work,
    /// which is the "repeated runs do not corrupt reused slots" exit criterion
    /// at the StorageCore level.
    func testPoolIsReusableAfterCancellation() throws {
        let regionBytes = 4096
        let count = 16
        let path = try makeFile(regionCount: count, regionBytes: regionBytes)
        let pool = try makePool(slots: 3, bytes: regionBytes)
        let descriptors = regions(count: count, bytes: regionBytes)

        let first = try TileReader(
            path: path, pool: pool,
            configuration: TileReaderConfiguration(queueDepth: 2, readAhead: 2))
        try first.schedule(descriptors)
        try first.acquire(descriptors[0]).releaseChecked()
        first.cancel(reason: "abort")
        first.shutdown()
        XCTAssertEqual(pool.census.inUse, 0)

        pool.resume()
        try pool.resetStatistics()

        let second = try TileReader(
            path: path, pool: pool,
            configuration: TileReaderConfiguration(queueDepth: 2, readAhead: 2))
        defer { second.shutdown() }
        try second.schedule(descriptors)
        for (index, descriptor) in descriptors.enumerated() {
            let lease = try second.acquire(descriptor)
            let expected = UInt8(truncatingIfNeeded: index + 1)
            XCTAssertTrue(
                lease.contents.allSatisfy { $0 == expected },
                "region \(index) is corrupt on the second run")
            try lease.releaseChecked()
        }
        XCTAssertEqual(pool.census.free, 3)
        XCTAssertEqual(pool.statistics.faults, [])
    }

    // MARK: - Configuration

    func testConfigurationBounds() throws {
        let path = try makeFile(regionCount: 1, regionBytes: 4096)
        let pool = try makePool(slots: 1, bytes: 4096)
        for depth in [0, 9] {
            XCTAssertThrowsError(
                try TileReader(
                    path: path, pool: pool,
                    configuration: TileReaderConfiguration(queueDepth: depth)))
        }
        XCTAssertThrowsError(
            try TileReader(
                path: path, pool: pool,
                configuration: TileReaderConfiguration(queueDepth: 2, readAhead: 0)))
    }

    func testMissingFileFailsAtConstruction() throws {
        let pool = try makePool(slots: 1, bytes: 4096)
        XCTAssertThrowsError(
            try TileReader(path: directory.appendingPathComponent("absent.bin").path, pool: pool)
        ) { error in
            guard case StorageCoreError.posix(let operation, _, _) = error else {
                return XCTFail("expected posix error, got \(error)")
            }
            XCTAssertEqual(operation, "open")
        }
    }
}
