import Foundation
import XCTest

@testable import StorageCore

/// The slot state machine and the lease refcount, exercised directly.
///
/// These tests are deliberately MLX-free: they are the ones that run under
/// ThreadSanitizer and on a CI runner with plain `swift test`, which is where a
/// lifetime bug in a pager is cheapest to find.
final class BufferPoolTests: XCTestCase {

    private func makePool(
        slots: Int = 4,
        bytes: Int = 4096,
        retain: Bool = false,
        poison: Bool? = nil
    ) throws -> BufferPool {
        try BufferPool(
            configuration: BufferPoolConfiguration(
                slotCount: slots, slotBytes: bytes, retainOnRelease: retain,
                poisonOnRelease: poison))
    }

    private func region(_ index: UInt64, length: Int = 4096) -> RegionDescriptor {
        RegionDescriptor(
            identity: RegionIdentity(sourceID: 7, index: index),
            offset: UInt64(index) * 4096,
            length: length)
    }

    // MARK: - Configuration

    func testRejectsEmptyConfigurations() {
        XCTAssertThrowsError(try makePool(slots: 0))
        XCTAssertThrowsError(try makePool(bytes: 0))
    }

    func testSlotsArePageAligned() throws {
        let pool = try makePool()
        let pageSize = UInt(AlignedBuffer.pageSize)
        for _ in 0..<4 {
            let slot = try pool.acquire()
            XCTAssertEqual(UInt(bitPattern: slot.pointer) % pageSize, 0)
        }
    }

    // MARK: - Legal path

    func testFullLifecycleReturnsTheSlot() throws {
        let pool = try makePool(slots: 1)
        let slot = try pool.acquire()
        XCTAssertEqual(pool.inspect(slot).state, .filling)

        try pool.markReady(slot, region: region(0))
        XCTAssertEqual(pool.inspect(slot).state, .ready)

        let lease = try pool.lease(slot)
        XCTAssertEqual(pool.inspect(slot).refcount, 1)

        try lease.retain()
        XCTAssertEqual(pool.inspect(slot).refcount, 2)

        try lease.releaseChecked()
        XCTAssertEqual(pool.inspect(slot).state, .leased, "one holder left, slot must stay leased")

        try lease.releaseChecked()
        XCTAssertEqual(pool.inspect(slot).state, .free)
        XCTAssertEqual(pool.census.free, 1)
    }

    // MARK: - Illegal transitions (WP-008)

    func testMarkReadyOnlyFromFilling() throws {
        let pool = try makePool(slots: 2)
        let slot = try pool.acquire()
        try pool.markReady(slot, region: region(0))
        // ready -> ready
        XCTAssertThrowsError(try pool.markReady(slot, region: region(0))) { error in
            guard case PagerError.illegalTransition(_, let from, let to) = error else {
                return XCTFail("expected illegalTransition, got \(error)")
            }
            XCTAssertEqual(from, .ready)
            XCTAssertEqual(to, "ready")
        }
        // leased -> ready
        let lease = try pool.lease(slot)
        XCTAssertThrowsError(try pool.markReady(slot, region: region(0)))
        try lease.releaseChecked()
        // free -> ready
        XCTAssertThrowsError(try pool.markReady(slot, region: region(0)))
    }

    func testLeaseOnlyFromReadyOrCache() throws {
        let pool = try makePool(slots: 2)
        let filling = try pool.acquire()
        // filling -> leased: the partial-fill hazard spec §12.4 forbids.
        XCTAssertThrowsError(try pool.lease(filling)) { error in
            guard case PagerError.illegalTransition(_, let from, _) = error else {
                return XCTFail("expected illegalTransition, got \(error)")
            }
            XCTAssertEqual(from, .filling)
        }
        try pool.markReady(filling, region: region(0))
        let lease = try pool.lease(filling)
        // leased -> leased (a second independent lease, not a retain)
        XCTAssertThrowsError(try pool.lease(filling))
        try lease.releaseChecked()
        // free -> leased
        XCTAssertThrowsError(try pool.lease(filling))
    }

    func testRetainOnlyWhileLeased() throws {
        let pool = try makePool(slots: 2)
        let slot = try pool.acquire()
        XCTAssertThrowsError(try pool.retain(slot))  // filling
        try pool.markReady(slot, region: region(0))
        XCTAssertThrowsError(try pool.retain(slot))  // ready
        let lease = try pool.lease(slot)
        try pool.retain(slot)  // leased: legal
        try lease.releaseChecked()
        try lease.releaseChecked()
        XCTAssertThrowsError(try pool.retain(slot))  // free
    }

    func testReleaseOnNonLeasedSlotThrows() throws {
        let pool = try makePool(slots: 2)
        let slot = try pool.acquire()
        XCTAssertThrowsError(try pool.release(slot))  // filling
        try pool.markReady(slot, region: region(0))
        XCTAssertThrowsError(try pool.release(slot))  // ready
    }

    /// The double release. A boolean lease would silently allow this and hand
    /// the slot to a new read while a second consumer still points at it.
    func testDoubleReleaseTraps() throws {
        let pool = try makePool(slots: 1)
        let slot = try pool.acquire()
        try pool.markReady(slot, region: region(0))
        let lease = try pool.lease(slot)
        try lease.releaseChecked()
        XCTAssertEqual(pool.inspect(slot).state, .free)

        XCTAssertThrowsError(try lease.releaseChecked()) { error in
            guard case PagerError.illegalTransition(_, let from, let to) = error else {
                return XCTFail("expected illegalTransition, got \(error)")
            }
            XCTAssertEqual(from, .free)
            XCTAssertEqual(to, "released")
        }
    }

    /// The non-throwing release path records rather than swallows.
    func testNonThrowingReleaseRecordsAFault() throws {
        let pool = try makePool(slots: 1)
        let slot = try pool.acquire()
        try pool.markReady(slot, region: region(0))
        let lease = try pool.lease(slot)
        try lease.releaseChecked()
        XCTAssertTrue(pool.statistics.faults.isEmpty)

        lease.release()
        XCTAssertEqual(pool.statistics.faults.count, 1)
        XCTAssertTrue(pool.statistics.faults[0].contains("illegal transition"))
    }

    func testDiscardOnlyFromFillingOrReady() throws {
        let pool = try makePool(slots: 2)
        let slot = try pool.acquire()
        try pool.discard(slot)  // filling -> free
        XCTAssertEqual(pool.inspect(slot).state, .free)
        XCTAssertThrowsError(try pool.discard(slot))  // free -> free

        let second = try pool.acquire()
        try pool.markReady(second, region: region(1))
        try pool.discard(second)  // ready -> free
        XCTAssertEqual(pool.inspect(second).state, .free)

        let third = try pool.acquire()
        try pool.markReady(third, region: region(2))
        let lease = try pool.lease(third)
        XCTAssertThrowsError(try pool.discard(third))  // leased -> free
        try lease.releaseChecked()
    }

    // MARK: - Cache state

    func testCachedSlotIsReusableByIdentityAndEvictable() throws {
        let pool = try makePool(slots: 1, retain: true)
        let slot = try pool.acquire()
        try pool.markReady(slot, region: region(11))
        let lease = try pool.lease(slot)
        try lease.releaseChecked()
        XCTAssertEqual(pool.inspect(slot).state, .cached)
        XCTAssertEqual(pool.census.cached, 1)

        let hit = try XCTUnwrap(pool.leaseCached(region(11).identity), "cached slot should hit")
        XCTAssertEqual(pool.inspect(slot).state, .leased)
        XCTAssertNil(pool.leaseCached(region(11).identity), "a leased slot is no longer cached")
        try hit.releaseChecked()

        XCTAssertNil(pool.leaseCached(region(99).identity), "different identity must miss")

        // Only slot in the pool: acquiring must evict it rather than block.
        let reused = try pool.acquire(timeout: 1)
        XCTAssertEqual(reused.index, slot.index)
        XCTAssertEqual(pool.statistics.cacheEvictionCount, 1)
        XCTAssertEqual(pool.statistics.cacheHitCount, 1)
    }

    func testEvictAllClearsTheCache() throws {
        let pool = try makePool(slots: 2, retain: true)
        for index in 0..<2 {
            let slot = try pool.acquire()
            try pool.markReady(slot, region: region(UInt64(index)))
            try pool.lease(slot).releaseChecked()
        }
        XCTAssertEqual(pool.census.cached, 2)
        pool.evictAll()
        XCTAssertEqual(pool.census.cached, 0)
        XCTAssertEqual(pool.census.free, 2)
        XCTAssertNil(pool.leaseCached(region(0).identity))
    }

    // MARK: - Poison

    func testReleasedSlotIsPoisoned() throws {
        let pool = try makePool(slots: 1, bytes: 8192, poison: true)
        let slot = try pool.acquire()
        memset(slot.pointer, 0x5A, slot.capacity)
        try pool.markReady(slot, region: region(0, length: 8192))
        let lease = try pool.lease(slot)
        XCTAssertEqual(lease.contents[0], 0x5A, "live data must survive the lease")
        try lease.releaseChecked()

        let bytes = UnsafeRawBufferPointer(start: slot.pointer, count: slot.capacity)
        XCTAssertTrue(
            bytes.allSatisfy { $0 == BufferPool.poisonByte },
            "released slot memory must be poisoned to catch use-after-release")
    }

    /// The other half: poison must never be visible in data a consumer holds.
    /// A slot is refilled after release and the lease's contents checked, which
    /// is exactly the sequence a use-after-release would corrupt.
    func testPoisonNeverAppearsInLiveData() throws {
        let pool = try makePool(slots: 2, bytes: 4096, poison: true)
        for round in 0..<50 {
            let slot = try pool.acquire()
            memset(slot.pointer, Int32(round % 200 + 1), slot.capacity)
            try pool.markReady(slot, region: region(UInt64(round)))
            let lease = try pool.lease(slot)
            XCTAssertFalse(
                lease.contents.contains(BufferPool.poisonByte),
                "round \(round): poison byte observed in leased data")
            try lease.releaseChecked()
        }
    }

    // MARK: - Backpressure

    func testAcquireBlocksUntilASlotIsReleased() throws {
        let pool = try makePool(slots: 1)
        let slot = try pool.acquire()
        try pool.markReady(slot, region: region(0))
        let lease = try pool.lease(slot)

        let acquired = expectation(description: "second acquire completes")
        let thread = Thread {
            let second = try? pool.acquire(timeout: 10)
            XCTAssertNotNil(second)
            acquired.fulfill()
        }
        thread.start()

        // Give the waiter time to actually block, then release.
        Thread.sleep(forTimeInterval: 0.1)
        XCTAssertEqual(pool.census.free, 0)
        try lease.releaseChecked()

        wait(for: [acquired], timeout: 10)
        XCTAssertGreaterThanOrEqual(pool.statistics.blockedAcquireCount, 1)
        XCTAssertGreaterThan(pool.statistics.blockedSeconds, 0)
    }

    func testAcquireTimesOutWithTheWaitRecorded() throws {
        let pool = try makePool(slots: 1)
        let slot = try pool.acquire()
        try pool.markReady(slot, region: region(0))
        let lease = try pool.lease(slot)

        XCTAssertThrowsError(try pool.acquire(timeout: 0.2)) { error in
            guard case PagerError.timedOut(let operation, let waited) = error else {
                return XCTFail("expected timedOut, got \(error)")
            }
            XCTAssertEqual(operation, "BufferPool.acquire")
            XCTAssertGreaterThan(waited, 0.1)
        }
        try lease.releaseChecked()
    }

    func testPeakAccountingNeverExceedsTheBudget() throws {
        let pool = try makePool(slots: 3, bytes: 4096)
        var leases: [SlotLease] = []
        for index in 0..<3 {
            let slot = try pool.acquire()
            try pool.markReady(slot, region: region(UInt64(index)))
            leases.append(try pool.lease(slot))
        }
        let statistics = pool.statistics
        XCTAssertEqual(statistics.peakInUseSlots, 3)
        XCTAssertEqual(statistics.peakInUseBytes, 3 * 4096)
        XCTAssertEqual(statistics.budgetBytes, 3 * 4096)
        XCTAssertTrue(statistics.isBudgetRespected)
        for lease in leases { try lease.releaseChecked() }
    }

    func testResetStatisticsRefusesWhileSlotsAreInUse() throws {
        let pool = try makePool(slots: 2)
        let slot = try pool.acquire()
        XCTAssertThrowsError(try pool.resetStatistics())
        try pool.discard(slot)
        try pool.resetStatistics()
        XCTAssertEqual(pool.statistics.peakInUseSlots, 0)
        XCTAssertEqual(pool.statistics.acquireCount, 0)
    }

    // MARK: - Cancellation

    func testCancellationWakesWaitersWithACleanError() throws {
        let pool = try makePool(slots: 1)
        let slot = try pool.acquire()
        try pool.markReady(slot, region: region(0))
        let lease = try pool.lease(slot)

        let woke = expectation(description: "waiter woke")
        woke.expectedFulfillmentCount = 3
        for _ in 0..<3 {
            let thread = Thread {
                do {
                    _ = try pool.acquire()
                    XCTFail("acquire should not have succeeded")
                } catch PagerError.cancelled(let reason) {
                    XCTAssertEqual(reason, "drive removed")
                    woke.fulfill()
                } catch {
                    XCTFail("expected cancelled, got \(error)")
                }
            }
            thread.start()
        }
        Thread.sleep(forTimeInterval: 0.1)
        pool.cancel(reason: "drive removed")
        wait(for: [woke], timeout: 10)

        XCTAssertTrue(pool.isCancelled)
        // A lease handed out before cancellation stays valid: cancelling under
        // a live consumer is the eviction spec §11.1 forbids.
        XCTAssertEqual(pool.inspect(slot).state, .leased)
        try lease.releaseChecked()

        pool.resume()
        XCTAssertFalse(pool.isCancelled)
        _ = try pool.acquire(timeout: 1)
    }

    // MARK: - Concurrency

    /// Many threads running the whole lifecycle at random for a few seconds,
    /// with the pool's own invariants checked continuously. Under
    /// ThreadSanitizer this is the test that matters.
    func testConcurrentLifecycleStress() throws {
        let slotCount = 4
        let pool = try makePool(slots: slotCount, bytes: 16384, poison: true)
        let deadline = Date().addingTimeInterval(3.0)
        let threadCount = 8
        let done = expectation(description: "workers finished")
        done.expectedFulfillmentCount = threadCount
        let failures = FailureLog()

        for workerIndex in 0..<threadCount {
            let thread = Thread {
                var generator = SplitMix64(seed: UInt64(workerIndex) &+ 0xBEEF)
                var round: UInt64 = 0
                while Date() < deadline {
                    round += 1
                    do {
                        let slot = try pool.acquire(timeout: 30)
                        // 1...200, so a marker can never coincidentally equal
                        // the poison byte (0xDE = 222) and make the poison
                        // check fire on legitimate data.
                        let marker = UInt8(
                            truncatingIfNeeded: (round &+ UInt64(workerIndex)) % 200 &+ 1)
                        memset(slot.pointer, Int32(marker), slot.capacity)

                        // Sometimes a read "fails" and the slot is discarded
                        // without ever becoming ready.
                        if generator.next() % 8 == 0 {
                            try pool.discard(slot)
                            continue
                        }

                        let descriptor = RegionDescriptor(
                            identity: RegionIdentity(
                                sourceID: UInt64(workerIndex), index: round),
                            offset: round * 4096,
                            length: slot.capacity)
                        try pool.markReady(slot, region: descriptor)
                        let lease = try pool.lease(slot)

                        // Two extra holders, like the packed + scales adoption.
                        try lease.retain()
                        try lease.retain()

                        if lease.contents.contains(BufferPool.poisonByte) {
                            failures.record("poison observed in live data on worker \(workerIndex)")
                        }
                        if lease.contents.contains(where: { $0 != marker }) {
                            failures.record("foreign bytes in slot on worker \(workerIndex)")
                        }

                        // Release the three holders from three different
                        // threads, which is how MLX finalizers really arrive.
                        let group = DispatchGroup()
                        for _ in 0..<2 {
                            group.enter()
                            DispatchQueue.global().async {
                                lease.releaseChecked2(failures)
                                group.leave()
                            }
                        }
                        group.wait()
                        lease.releaseChecked2(failures)

                        let census = pool.census
                        if census.total != slotCount || census.inUse > slotCount {
                            failures.record("census broken: \(census)")
                        }
                    } catch {
                        failures.record("worker \(workerIndex): \(error)")
                    }
                }
                done.fulfill()
            }
            thread.name = "stress.\(workerIndex)"
            thread.start()
        }

        wait(for: [done], timeout: 60)

        XCTAssertEqual(failures.messages, [], "invariant violations during stress")
        let statistics = pool.statistics
        XCTAssertEqual(statistics.faults, [])
        XCTAssertLessThanOrEqual(statistics.peakInUseSlots, slotCount)
        XCTAssertEqual(statistics.peakInUseSlots, slotCount, "the pool should have run full")
        XCTAssertGreaterThan(statistics.blockedAcquireCount, 0, "backpressure never applied")
        let census = pool.census
        XCTAssertEqual(census.inUse, 0, "slots leaked: \(census)")
        XCTAssertEqual(census.free, slotCount)
    }
}

/// Thread-safe collector, so a failure inside a worker survives to the assertion.
final class FailureLog: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []

    func record(_ message: String) {
        lock.lock()
        // One of each is enough to fail the test and keeps the output readable.
        if !storage.contains(message) { storage.append(message) }
        lock.unlock()
    }

    var messages: [String] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

extension SlotLease {
    /// `releaseChecked()` from a context that cannot throw, recording instead.
    fileprivate func releaseChecked2(_ log: FailureLog) {
        do { try releaseChecked() } catch { log.record("release failed: \(error)") }
    }
}
