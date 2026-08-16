import Foundation
import XCTest

@testable import StorageCore

/// The three cache arms M8 compares (spec §11.3), exercised on the pool alone.
///
/// MLX-free and file-free on purpose: this is the suite that runs under
/// ThreadSanitizer, and the property that matters — a cold region can never
/// displace a hot one — is a statement about the pool's bookkeeping, not about
/// storage.
final class CachePolicyTests: XCTestCase {

    private func makePool(
        slots: Int = 4,
        bytes: Int = 4096,
        policy: CachePolicy
    ) throws -> BufferPool {
        try BufferPool(
            configuration: BufferPoolConfiguration(
                slotCount: slots, slotBytes: bytes, cachePolicy: policy, poisonOnRelease: false))
    }

    private func identity(_ index: UInt64) -> RegionIdentity {
        RegionIdentity(sourceID: 3, index: index)
    }

    private func region(_ index: UInt64, length: Int = 4096) -> RegionDescriptor {
        RegionDescriptor(identity: identity(index), offset: index * 4096, length: length)
    }

    /// Fill one slot with `index` and drop the caller's reference, which is the
    /// only moment the admission policy is consulted.
    @discardableResult
    private func cycle(_ pool: BufferPool, _ index: UInt64) throws -> BufferSlot {
        let slot = try pool.acquire(timeout: 5)
        try pool.markReady(slot, region: region(index))
        try pool.lease(slot).releaseChecked()
        return slot
    }

    // MARK: - Admission

    func testNoCachePolicyRetainsNothing() throws {
        let pool = try makePool(policy: .none)
        try cycle(pool, 0)
        XCTAssertEqual(pool.census.cached, 0)
        XCTAssertEqual(pool.census.free, 4)
        XCTAssertNil(pool.leaseCached(identity(0)))

        let statistics = pool.statistics
        XCTAssertEqual(statistics.cachePolicyName, "no-cache")
        XCTAssertEqual(statistics.cacheAdmissionCount, 0)
        XCTAssertEqual(
            statistics.cacheRejectedAdmissionCount, 0,
            "no-cache never attempts admission, so a rejection would be miscounted work")
        XCTAssertEqual(statistics.cacheHitCount, 0)
        XCTAssertEqual(statistics.cacheMissCount, 1)
    }

    func testLeastRecentlyUsedAdmitsEverything() throws {
        let pool = try makePool(policy: .leastRecentlyUsed)
        for index in 0..<3 { try cycle(pool, UInt64(index)) }
        XCTAssertEqual(pool.census.cached, 3)
        XCTAssertEqual(pool.statistics.cacheAdmissionCount, 3)
        XCTAssertEqual(pool.statistics.cacheRejectedAdmissionCount, 0)
    }

    func testPinnedHotSetAdmitsOnlyTheHotSet() throws {
        let hot: Set<RegionIdentity> = [identity(1), identity(2)]
        let pool = try makePool(policy: .pinnedHotSet(hot))

        try cycle(pool, 1)
        try cycle(pool, 5)  // cold
        try cycle(pool, 2)
        try cycle(pool, 6)  // cold

        XCTAssertEqual(pool.census.cached, 2)
        XCTAssertNotNil(pool.leaseCached(identity(1)))
        XCTAssertNotNil(pool.leaseCached(identity(2)))
        XCTAssertNil(pool.leaseCached(identity(5)))

        let statistics = pool.statistics
        XCTAssertEqual(statistics.cachePolicyName, "pinned-hot-set")
        XCTAssertEqual(statistics.cacheAdmissionCount, 2)
        XCTAssertEqual(statistics.cacheRejectedAdmissionCount, 2)
    }

    // MARK: - Replacement

    func testLeastRecentlyUsedEvictsTheLeastRecentlyReleased() throws {
        let pool = try makePool(slots: 2, policy: .leastRecentlyUsed)
        try cycle(pool, 0)
        try cycle(pool, 1)

        // Touch 0 so 1 becomes the least recently released.
        let hit = try XCTUnwrap(pool.leaseCached(identity(0)))
        try hit.releaseChecked()

        _ = try pool.acquire(timeout: 5)
        XCTAssertNotNil(pool.leaseCached(identity(0)), "0 was touched most recently and must stay")
        XCTAssertNil(pool.leaseCached(identity(1)), "1 was least recently released and must go")
        XCTAssertEqual(pool.statistics.cacheEvictionCount, 1)
    }

    /// The property the whole policy exists for. Under LRU a stream of
    /// one-shot regions walks the hot entries out of the cache; under the
    /// pinned policy it cannot, because it never gets in.
    func testColdStreamCannotDisplaceTheHotSetButUnderLRUItDoes() throws {
        let hot: Set<RegionIdentity> = [identity(0), identity(1)]

        let pinned = try makePool(slots: 4, policy: .pinnedHotSet(hot))
        try cycle(pinned, 0)
        try cycle(pinned, 1)
        for cold in 100..<140 { try cycle(pinned, UInt64(cold)) }
        XCTAssertEqual(pinned.census.cached, 2)
        XCTAssertNotNil(pinned.leaseCached(identity(0)))
        XCTAssertNotNil(pinned.leaseCached(identity(1)))
        XCTAssertEqual(pinned.statistics.cacheEvictionCount, 0, "nothing hot was ever evicted")

        let lru = try makePool(slots: 4, policy: .leastRecentlyUsed)
        try cycle(lru, 0)
        try cycle(lru, 1)
        for cold in 100..<140 { try cycle(lru, UInt64(cold)) }
        XCTAssertNil(lru.leaseCached(identity(0)), "LRU lets the cold stream evict the hot entry")
        XCTAssertNil(lru.leaseCached(identity(1)))
        XCTAssertGreaterThan(lru.statistics.cacheEvictionCount, 0)
    }

    // MARK: - Configuration

    func testHotSetMustLeaveAtLeastOneSlotFree() throws {
        let full: Set<RegionIdentity> = Set((0..<4).map { identity(UInt64($0)) })
        XCTAssertThrowsError(try makePool(slots: 4, policy: .pinnedHotSet(full))) { error in
            XCTAssertTrue(
                "\(error)".contains("pins 4 regions in a pool of 4 slots"),
                "the rejection must name both numbers, got: \(error)")
        }
        XCTAssertNoThrow(
            try makePool(
                slots: 4, policy: .pinnedHotSet(Set((0..<3).map { identity(UInt64($0)) }))))
    }

    func testSwappingThePolicyDropsEntriesTheNewPolicyWouldNotAdmit() throws {
        let pool = try makePool(slots: 4, policy: .leastRecentlyUsed)
        for index in 0..<3 { try cycle(pool, UInt64(index)) }
        XCTAssertEqual(pool.census.cached, 3)

        try pool.setCachePolicy(.pinnedHotSet([identity(2)]))
        XCTAssertEqual(pool.census.cached, 1, "0 and 1 are no longer admissible")
        XCTAssertNotNil(pool.leaseCached(identity(2)))

        try pool.setCachePolicy(.none)
        XCTAssertEqual(pool.census.cached, 0)
    }

    func testRetainOnReleaseStillSelectsLeastRecentlyUsed() throws {
        let pool = try BufferPool(
            configuration: BufferPoolConfiguration(
                slotCount: 2, slotBytes: 4096, retainOnRelease: true))
        XCTAssertEqual(pool.cachePolicy, .leastRecentlyUsed)
        XCTAssertTrue(pool.configuration.retainOnRelease)
    }

    func testHitRateCountsEveryLookup() throws {
        let pool = try makePool(policy: .leastRecentlyUsed)
        try cycle(pool, 0)
        _ = pool.leaseCached(identity(0))  // hit
        _ = pool.leaseCached(identity(9))  // miss
        _ = pool.leaseCached(identity(9))  // miss
        let statistics = pool.statistics
        XCTAssertEqual(statistics.cacheHitCount, 1)
        XCTAssertEqual(statistics.cacheMissCount, 2)
        XCTAssertEqual(statistics.cacheHitRate, 1.0 / 3.0, accuracy: 1e-12)
    }

    // MARK: - Concurrency

    /// Admission and replacement run under the pool's monitor like every other
    /// transition. This is the case ThreadSanitizer is pointed at.
    func testConcurrentCyclingUnderAPinnedPolicyKeepsTheHotSetIntact() throws {
        let hot: Set<RegionIdentity> = [identity(0), identity(1), identity(2)]
        let pool = try makePool(slots: 8, policy: .pinnedHotSet(hot))

        let group = DispatchGroup()
        for worker in 0..<4 {
            DispatchQueue.global().async(group: group) {
                for step in 0..<200 {
                    let index = step % 5 == 0 ? UInt64(step % 3) : UInt64(1000 + worker * 500 + step)
                    guard let slot = try? pool.acquire(timeout: 10) else { continue }
                    try? pool.markReady(slot, region: self.region(index))
                    if let lease = try? pool.lease(slot) { lease.release() }
                }
            }
        }
        XCTAssertEqual(group.wait(timeout: .now() + 120), .success)

        let statistics = pool.statistics
        XCTAssertTrue(statistics.faults.isEmpty, "\(statistics.faults)")
        XCTAssertEqual(pool.census.inUse, 0)
        // The bound the policy promises. Duplicate concurrent fills of one hot
        // identity must retire the older slot rather than leave it cached and
        // unreachable, or "pin three" would quietly occupy more than three.
        XCTAssertLessThanOrEqual(pool.census.cached, hot.count)
        XCTAssertTrue(statistics.isBudgetRespected)
        XCTAssertGreaterThan(statistics.cacheRejectedAdmissionCount, 0)
    }
}
