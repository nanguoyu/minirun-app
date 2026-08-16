import Darwin
import Foundation

/// One reusable, page-aligned buffer owned by a ``BufferPool``.
///
/// The slot is a handle, not a value: the pool identifies it by `index` and
/// keeps its mutable lifecycle fields behind the pool's monitor. Nothing here
/// is safe to read without the pool.
public final class BufferSlot: @unchecked Sendable {
    public let index: Int
    /// Allocated size. Every slot in a pool is the same size — a pool with one
    /// budget and mixed slot sizes has an emergent footprint, not a stated one.
    public let capacity: Int

    private let buffer: AlignedBuffer

    /// Base address. Page-aligned, which both the direct read path and MLX's
    /// external-buffer adoption require.
    public var pointer: UnsafeMutableRawPointer { buffer.pointer }

    public var mutableBytes: UnsafeMutableRawBufferPointer { buffer.mutableBytes }
    public var bytes: UnsafeRawBufferPointer { buffer.bytes }

    /// A page-aligned subrange, used to address the two operand regions that
    /// share one slot.
    public func subregion(offset: Int, count: Int) -> UnsafeMutableRawBufferPointer {
        precondition(
            offset >= 0 && count >= 0 && offset + count <= capacity,
            "subregion [\(offset), \(offset + count)) is outside slot capacity \(capacity)")
        return UnsafeMutableRawBufferPointer(start: buffer.pointer + offset, count: count)
    }

    // Guarded by the owning pool's monitor. Never touch these outside it.
    fileprivate var state: SlotState = .free
    fileprivate var refcount: Int = 0
    fileprivate var region: RegionDescriptor?
    fileprivate var cacheSequence: UInt64 = 0

    fileprivate init(index: Int, capacity: Int, alignment: Int) throws {
        self.index = index
        self.capacity = capacity
        self.buffer = try AlignedBuffer(count: capacity, alignment: alignment)
    }
}

/// A live reference to a filled slot.
///
/// A lease is a *refcount*, not a flag. ``retain()`` adds a holder and
/// ``release()`` removes one; the slot returns to the pool only at zero. The
/// lease keeps the pool alive by a strong reference on purpose: releases arrive
/// from MLX finalizers on arbitrary threads, potentially after every other
/// reference to the pool has gone, and a dangling `unowned` pool would turn a
/// late release into a crash.
public final class SlotLease: @unchecked Sendable {
    public let slot: BufferSlot
    public let region: RegionDescriptor
    private let pool: BufferPool

    fileprivate init(slot: BufferSlot, region: RegionDescriptor, pool: BufferPool) {
        self.slot = slot
        self.region = region
        self.pool = pool
    }

    /// The region's bytes, exactly `region.length` of them.
    public var contents: UnsafeRawBufferPointer {
        UnsafeRawBufferPointer(start: slot.pointer, count: region.length)
    }

    /// Add a holder. Throws if the slot is no longer leased, which means the
    /// caller retained after the last release — a use-after-free in waiting.
    public func retain() throws {
        try pool.retain(slot)
    }

    /// Drop a holder. Non-throwing because the intended caller is an MLX
    /// finalizer, which cannot propagate an error; a fault is recorded on the
    /// pool and surfaced in ``BufferPool/statistics`` instead of being lost.
    public func release() {
        pool.releaseRecordingFaults(slot)
    }

    /// Throwing form, for callers that can handle it and for the tests that
    /// assert the double-release trap.
    public func releaseChecked() throws {
        try pool.release(slot)
    }
}

/// Which filled slots survive their last release, and which one goes back
/// first when the pool needs a slot (spec §11.3).
///
/// The three cases are deliberately the three M8 compares, and they differ
/// along two axes that are easy to conflate:
///
/// | | admission | replacement |
/// | --- | --- | --- |
/// | ``none`` | nothing | n/a |
/// | ``leastRecentlyUsed`` | everything | least recently released |
/// | ``pinnedHotSet(_:)`` | hot set only | least recently released |
///
/// `leastRecentlyUsed` is pure replacement policy: every region competes for
/// every slot, so a long tail of one-shot reads evicts the entries that were
/// about to be reused. `pinnedHotSet` is *admission* control — a region the
/// routing trace says is cold never enters the cache at all, and therefore can
/// never displace a hot one. Spec §11.3 lists both "cost-aware admission" and
/// "expert-specific routing statistics" as candidates; this is the two
/// together, because at 896 experts and top-16 they are the same idea.
public enum CachePolicy: Sendable, Equatable {
    /// Retain nothing. Every request reaches storage.
    case none
    /// Retain every completed region; evict the least recently released.
    case leastRecentlyUsed
    /// Retain only regions named in the set. Everything else goes straight
    /// back to `free`, so a cold region cannot displace a hot one.
    case pinnedHotSet(Set<RegionIdentity>)

    /// Whether a region is allowed into the cache at all.
    public func admits(_ identity: RegionIdentity) -> Bool {
        switch self {
        case .none: return false
        case .leastRecentlyUsed: return true
        case .pinnedHotSet(let hot): return hot.contains(identity)
        }
    }

    /// Slots the policy can occupy indefinitely. `nil` means "any", which is
    /// what makes ``leastRecentlyUsed`` unbounded-by-identity.
    public var residentCapacity: Int? {
        switch self {
        case .none: return 0
        case .leastRecentlyUsed: return nil
        case .pinnedHotSet(let hot): return hot.count
        }
    }

    /// A short label for a metrics record.
    public var name: String {
        switch self {
        case .none: return "no-cache"
        case .leastRecentlyUsed: return "lru"
        case .pinnedHotSet: return "pinned-hot-set"
        }
    }
}

/// How a pool is sized and how it behaves at the edges.
public struct BufferPoolConfiguration: Sendable {
    /// Number of slots. This times ``slotBytes`` is the pager's entire weight
    /// footprint — the budget in spec §5.3 and §11.1.
    public var slotCount: Int
    /// Size of every slot.
    public var slotBytes: Int
    public var alignment: Int
    /// What survives a last release (spec §11.3). M4 shipped the mechanism
    /// with the most trivial policy that exercises it; M8 compares three.
    public var cachePolicy: CachePolicy
    /// Fill a slot with `0xDE` whenever it returns to `free`. Defaults on in
    /// debug builds: it converts a use-after-release from an intermittent wrong
    /// answer into an obviously wrong one.
    public var poisonOnRelease: Bool

    /// Retain contents after the last release, so a later request for the same
    /// identity is a hit. Kept as the plain-boolean spelling of the two
    /// policies that predate M8.
    public var retainOnRelease: Bool { cachePolicy != .none }

    public init(
        slotCount: Int,
        slotBytes: Int,
        alignment: Int = AlignedBuffer.pageSize,
        retainOnRelease: Bool = false,
        poisonOnRelease: Bool? = nil
    ) {
        self.init(
            slotCount: slotCount,
            slotBytes: slotBytes,
            alignment: alignment,
            cachePolicy: retainOnRelease ? .leastRecentlyUsed : CachePolicy.none,
            poisonOnRelease: poisonOnRelease)
    }

    public init(
        slotCount: Int,
        slotBytes: Int,
        alignment: Int = AlignedBuffer.pageSize,
        cachePolicy: CachePolicy,
        poisonOnRelease: Bool? = nil
    ) {
        self.slotCount = slotCount
        self.slotBytes = slotBytes
        self.alignment = alignment
        self.cachePolicy = cachePolicy
        #if DEBUG
            self.poisonOnRelease = poisonOnRelease ?? true
        #else
            self.poisonOnRelease = poisonOnRelease ?? false
        #endif
    }

    public var budgetBytes: Int { slotCount * slotBytes }
}

/// Slot counts by state at one instant.
public struct SlotCensus: Codable, Equatable, Sendable {
    public var free = 0
    public var filling = 0
    public var ready = 0
    public var leased = 0
    public var cached = 0

    /// Slots the pool cannot hand to a new read without evicting something.
    public var inUse: Int { filling + ready + leased }
    public var total: Int { free + filling + ready + leased + cached }
}

/// Everything the pool can say about itself, for the instrumentation the M4
/// exit criteria ask to be observable.
public struct BufferPoolStatistics: Codable, Equatable, Sendable {
    public var slotCount: Int
    public var slotBytes: Int
    public var budgetBytes: Int
    public var census: SlotCensus
    /// Highest simultaneous `filling + ready + leased`. The number that has to
    /// stay equal to `slotCount` (fully used) and never exceed it.
    public var peakInUseSlots: Int
    public var peakLeasedSlots: Int
    public var peakCachedSlots: Int
    /// `peakInUseSlots * slotBytes`. The pager-managed peak byte figure.
    public var peakInUseBytes: Int
    public var acquireCount: Int
    /// Acquisitions that had to wait for a slot — i.e. backpressure actually
    /// applied. Zero here means the pool was never the constraint.
    public var blockedAcquireCount: Int
    public var blockedSeconds: Double
    /// Sticky. Saturated blocked time must not be interpreted as an exact duration.
    public var blockedTimeOverflowed: Bool
    public var cacheHitCount: Int
    /// Lookups that found nothing. `hit + miss` is every identity the consumer
    /// asked for, so a hit rate can be computed without the caller counting.
    public var cacheMissCount: Int
    public var cacheEvictionCount: Int
    /// Releases the policy let into the cache.
    public var cacheAdmissionCount: Int
    /// Releases the policy turned away. Zero under ``CachePolicy/none`` (no
    /// admission is ever attempted) and under ``CachePolicy/leastRecentlyUsed``
    /// (everything is admitted); non-zero only under an admission policy.
    public var cacheRejectedAdmissionCount: Int
    public var cachePolicyName: String
    /// Non-throwing release paths that detected an illegal state. Must be empty.
    public var faults: [String]

    public var isBudgetRespected: Bool { peakInUseSlots <= slotCount }

    public var cacheHitRate: Double {
        let looks = cacheHitCount + cacheMissCount
        return looks > 0 ? Double(cacheHitCount) / Double(looks) : 0
    }
}

/// Fixed-size pool of page-aligned buffer slots with refcounted leases.
///
/// ## Why acquisition blocks
///
/// `acquire` waits rather than failing or allocating. That is the mechanism by
/// which the working set stays bounded (spec §5.3, §12.3): the reader cannot
/// run ahead of the consumer, because there is nowhere to put the bytes. A pool
/// that grew on demand would turn a stated budget back into an emergent one.
///
/// ## Thread safety
///
/// One `NSCondition` guards every mutable field. Releases arrive from MLX
/// finalizers on threads this module never created, so there is no
/// single-threaded fast path to protect and no benefit in splitting the lock;
/// the critical sections are a handful of integer updates. The one exception is
/// the debug poison memset, which does run under the lock — deliberately, so a
/// slot cannot be reacquired mid-poison — and which is compiled out of release
/// builds.
public final class BufferPool: @unchecked Sendable {
    public let configuration: BufferPoolConfiguration

    private let monitor = NSCondition()
    private var slots: [BufferSlot] = []
    private var cachedByIdentity: [RegionIdentity: BufferSlot] = [:]
    private var cancellationReason: String?
    private var interruptGeneration: UInt64 = 0
    private var nextCacheSequence: UInt64 = 0

    private var peakInUseSlots = 0
    private var peakLeasedSlots = 0
    private var peakCachedSlots = 0
    private var acquireCount = 0
    private var blockedAcquireCount = 0
    private var blockedNanoseconds: UInt64 = 0
    private var blockedTimeOverflowed = false
    private var cacheHitCount = 0
    private var cacheMissCount = 0
    private var cacheEvictionCount = 0
    private var cacheAdmissionCount = 0
    private var cacheRejectedAdmissionCount = 0
    private var policy: CachePolicy
    private var faults: [String] = []

    /// Byte written over a slot's contents when it returns to `free`.
    public static let poisonByte: UInt8 = 0xDE

    public init(configuration: BufferPoolConfiguration) throws {
        guard configuration.slotCount > 0 else {
            throw PagerError.configuration("slotCount must be positive")
        }
        guard configuration.slotBytes > 0 else {
            throw PagerError.configuration("slotBytes must be positive")
        }
        try Self.validate(configuration.cachePolicy, slotCount: configuration.slotCount)
        self.configuration = configuration
        self.policy = configuration.cachePolicy
        self.slots = try (0..<configuration.slotCount).map {
            try BufferSlot(
                index: $0, capacity: configuration.slotBytes, alignment: configuration.alignment)
        }
        if configuration.poisonOnRelease {
            for slot in slots { memset(slot.pointer, Int32(Self.poisonByte), slot.capacity) }
        }
    }

    /// A hot set that fills the pool leaves nothing to read into: the next miss
    /// must evict a pinned entry, and the policy degenerates into a slower LRU.
    /// Rejected at configuration time rather than clamped, for the same reason
    /// every other budget here is (spec §5.3).
    private static func validate(_ policy: CachePolicy, slotCount: Int) throws {
        guard let capacity = policy.residentCapacity else { return }
        guard capacity < slotCount else {
            throw PagerError.configuration(
                """
                cache policy '\(policy.name)' pins \(capacity) regions in a pool of \
                \(slotCount) slots. At least one slot must stay available for the region \
                being read, or every miss evicts a pinned entry. Pin at most \
                \(slotCount - 1), or raise slotCount.
                """)
        }
    }

    /// Swap the policy between runs. Retained entries the new policy would not
    /// admit are dropped immediately, so the cache never contains something the
    /// active policy says should not be there.
    public func setCachePolicy(_ newPolicy: CachePolicy) throws {
        try Self.validate(newPolicy, slotCount: configuration.slotCount)
        monitor.lock()
        defer { monitor.unlock() }
        policy = newPolicy
        for (identity, slot) in cachedByIdentity where !newPolicy.admits(identity) {
            cachedByIdentity.removeValue(forKey: identity)
            cacheEvictionCount += 1
            returnToFreeLocked(slot)
        }
        monitor.broadcast()
    }

    public var cachePolicy: CachePolicy {
        monitor.lock()
        defer { monitor.unlock() }
        return policy
    }

    // MARK: - Transitions

    /// `free -> filling`, evicting a cached slot first if that is the only way.
    ///
    /// Blocks while every slot is `filling`, `ready`, or `leased`. Returns as
    /// soon as one is available; `timeout` bounds a stall rather than being an
    /// expected path, so it defaults to no limit.
    public func acquire(timeout: TimeInterval? = nil) throws -> BufferSlot {
        monitor.lock()
        defer { monitor.unlock() }

        let started = MonotonicClock.now()
        let deadline = timeout.map { Date().addingTimeInterval($0) }
        let entryInterrupt = interruptGeneration
        var blocked = false

        while true {
            if let reason = cancellationReason {
                if blocked { finishBlockedWaitLocked(since: started) }
                throw PagerError.cancelled(reason: reason)
            }
            if interruptGeneration != entryInterrupt {
                if blocked { finishBlockedWaitLocked(since: started) }
                throw PagerError.interrupted
            }
            if let slot = takeAvailableSlotLocked() {
                slot.state = .filling
                slot.region = nil
                acquireCount += 1
                if blocked { finishBlockedWaitLocked(since: started) }
                updatePeaksLocked()
                return slot
            }
            blocked = true
            if let deadline {
                if !monitor.wait(until: deadline) {
                    finishBlockedWaitLocked(since: started)
                    throw PagerError.timedOut(
                        operation: "BufferPool.acquire",
                        waitedSeconds: MonotonicClock.seconds(since: started))
                }
            } else {
                monitor.wait()
            }
        }
    }

    /// `filling -> ready`. The region is recorded here and not at acquisition,
    /// because a slot only *means* something once its bytes are complete.
    public func markReady(_ slot: BufferSlot, region: RegionDescriptor) throws {
        monitor.lock()
        defer { monitor.unlock() }
        guard slot.state == .filling else {
            throw PagerError.illegalTransition(slot: slot.index, from: slot.state, to: "ready")
        }
        slot.state = .ready
        slot.region = region
    }

    /// `ready -> leased(1)`.
    public func lease(_ slot: BufferSlot) throws -> SlotLease {
        monitor.lock()
        defer { monitor.unlock() }
        guard slot.state == .ready, let region = slot.region else {
            throw PagerError.illegalTransition(slot: slot.index, from: slot.state, to: "leased")
        }
        slot.state = .leased
        slot.refcount = 1
        updatePeaksLocked()
        return SlotLease(slot: slot, region: region, pool: self)
    }

    /// `cached -> leased(1)` by identity: the cache hit path. Returns nil on a
    /// miss, which is not an error.
    public func leaseCached(_ identity: RegionIdentity) -> SlotLease? {
        monitor.lock()
        defer { monitor.unlock() }
        guard let slot = cachedByIdentity[identity], slot.state == .cached,
            let region = slot.region
        else {
            cacheMissCount += 1
            return nil
        }
        cachedByIdentity.removeValue(forKey: identity)
        slot.state = .leased
        slot.refcount = 1
        cacheHitCount += 1
        updatePeaksLocked()
        return SlotLease(slot: slot, region: region, pool: self)
    }

    /// Whether an identity is cached right now, without taking it.
    ///
    /// A scheduler needs this: queuing a read for bytes the pool already holds
    /// makes the reader fetch them anyway, and the redundant read evicts the
    /// very entry the request would have hit. Advisory by nature — the answer
    /// can be stale the moment it is returned — which is safe, because a
    /// consumer that skipped scheduling and then misses simply issues an urgent
    /// read instead.
    public func isCached(_ identity: RegionIdentity) -> Bool {
        monitor.lock()
        defer { monitor.unlock() }
        return cachedByIdentity[identity]?.state == .cached
    }

    /// `leased(n) -> leased(n + 1)`.
    public func retain(_ slot: BufferSlot) throws {
        monitor.lock()
        defer { monitor.unlock() }
        guard slot.state == .leased, slot.refcount >= 1 else {
            throw PagerError.illegalTransition(slot: slot.index, from: slot.state, to: "retained")
        }
        slot.refcount += 1
    }

    /// `leased(n) -> leased(n - 1)`, and at zero to `cached` or `free`.
    public func release(_ slot: BufferSlot) throws {
        monitor.lock()
        defer { monitor.unlock() }
        guard slot.state == .leased else {
            throw PagerError.illegalTransition(slot: slot.index, from: slot.state, to: "released")
        }
        guard slot.refcount >= 1 else {
            throw PagerError.refcountUnderflow(slot: slot.index, state: slot.state)
        }
        slot.refcount -= 1
        guard slot.refcount == 0 else { return }

        if let region = slot.region, policy.admits(region.identity) {
            // Two readers can legitimately fill two slots with the same
            // identity — nothing serialises a duplicate request across the
            // whole system. Without this the older slot stays `cached` while
            // the map points at the newer one, so it is reachable only by
            // eviction and the policy's resident capacity stops bounding
            // anything. Retire the duplicate instead.
            if let previous = cachedByIdentity[region.identity], previous !== slot {
                cacheEvictionCount += 1
                returnToFreeLocked(previous)
            }
            slot.state = .cached
            slot.cacheSequence = nextCacheSequence
            nextCacheSequence += 1
            cachedByIdentity[region.identity] = slot
            cacheAdmissionCount += 1
        } else {
            if slot.region != nil, policy != .none { cacheRejectedAdmissionCount += 1 }
            returnToFreeLocked(slot)
        }
        updatePeaksLocked()
        monitor.broadcast()
    }

    /// `filling -> free` or `ready -> free`: the read failed, or the run was
    /// cancelled before anyone leased the bytes. A partially filled slot must
    /// never become `ready` (spec §12.4), so this is its only exit.
    public func discard(_ slot: BufferSlot) throws {
        monitor.lock()
        defer { monitor.unlock() }
        guard slot.state == .filling || slot.state == .ready else {
            throw PagerError.illegalTransition(slot: slot.index, from: slot.state, to: "free")
        }
        returnToFreeLocked(slot)
        monitor.broadcast()
    }

    /// Drop every cached slot. Not an error when the cache is empty.
    public func evictAll() {
        monitor.lock()
        defer { monitor.unlock() }
        for slot in slots where slot.state == .cached {
            cacheEvictionCount += 1
            returnToFreeLocked(slot)
        }
        cachedByIdentity.removeAll()
        monitor.broadcast()
    }

    // MARK: - Cancellation

    /// Wake every waiter with ``PagerError/cancelled(reason:)`` and refuse new
    /// acquisitions until ``resume()``. Leases already handed out stay valid —
    /// their holders still have to release them, and cancelling underneath a
    /// live GPU operand would be exactly the eviction spec §11.1 forbids.
    public func cancel(reason: String) {
        monitor.lock()
        defer { monitor.unlock() }
        cancellationReason = reason
        monitor.broadcast()
    }

    /// Wake every current waiter with ``PagerError/interrupted`` without
    /// cancelling anything.
    ///
    /// This exists because a component that owns a thread blocked in
    /// ``acquire(timeout:)`` needs a way to get that thread back when *it* is
    /// shutting down, and using ``cancel(reason:)`` for that would leave a
    /// caller-owned pool permanently cancelled — the next run on the same pool
    /// would fail before reading a byte. Interruption is not sticky: a waiter
    /// that still wants a slot just calls again.
    public func interrupt() {
        monitor.lock()
        defer { monitor.unlock() }
        interruptGeneration &+= 1
        monitor.broadcast()
    }

    /// Clear cancellation so the pool can be reused for another run.
    public func resume() {
        monitor.lock()
        defer { monitor.unlock() }
        cancellationReason = nil
        monitor.broadcast()
    }

    public var isCancelled: Bool {
        monitor.lock()
        defer { monitor.unlock() }
        return cancellationReason != nil
    }

    // MARK: - Observation

    public var census: SlotCensus {
        monitor.lock()
        defer { monitor.unlock() }
        return censusLocked()
    }

    public var statistics: BufferPoolStatistics {
        monitor.lock()
        defer { monitor.unlock() }
        return BufferPoolStatistics(
            slotCount: configuration.slotCount,
            slotBytes: configuration.slotBytes,
            budgetBytes: configuration.budgetBytes,
            census: censusLocked(),
            peakInUseSlots: peakInUseSlots,
            peakLeasedSlots: peakLeasedSlots,
            peakCachedSlots: peakCachedSlots,
            peakInUseBytes: peakInUseSlots * configuration.slotBytes,
            acquireCount: acquireCount,
            blockedAcquireCount: blockedAcquireCount,
            blockedSeconds: Double(blockedNanoseconds) / 1_000_000_000,
            blockedTimeOverflowed: blockedTimeOverflowed,
            cacheHitCount: cacheHitCount,
            cacheMissCount: cacheMissCount,
            cacheEvictionCount: cacheEvictionCount,
            cacheAdmissionCount: cacheAdmissionCount,
            cacheRejectedAdmissionCount: cacheRejectedAdmissionCount,
            cachePolicyName: policy.name,
            faults: faults)
    }

    /// Zero the peaks and counters. Refuses while anything is still in use, so
    /// a "peaks reset" assertion cannot accidentally paper over a leaked slot.
    public func resetStatistics() throws {
        monitor.lock()
        defer { monitor.unlock() }
        let current = censusLocked()
        guard current.inUse == 0 else {
            throw PagerError.configuration(
                "resetStatistics with \(current.inUse) slots still in use "
                    + "(filling \(current.filling), ready \(current.ready), leased \(current.leased))")
        }
        peakInUseSlots = 0
        peakLeasedSlots = 0
        peakCachedSlots = cachedByIdentity.count
        acquireCount = 0
        blockedAcquireCount = 0
        blockedNanoseconds = 0
        blockedTimeOverflowed = false
        cacheHitCount = 0
        cacheMissCount = 0
        cacheEvictionCount = 0
        cacheAdmissionCount = 0
        cacheRejectedAdmissionCount = 0
        faults.removeAll()
    }

    /// State and refcount of one slot, for tests and diagnostics.
    public func inspect(_ slot: BufferSlot) -> (state: SlotState, refcount: Int) {
        monitor.lock()
        defer { monitor.unlock() }
        return (slot.state, slot.refcount)
    }

    // MARK: - Locked helpers

    private func takeAvailableSlotLocked() -> BufferSlot? {
        if let free = slots.first(where: { $0.state == .free }) { return free }
        // Replacement, in two tiers. Anything the *current* policy would no
        // longer admit goes first — that is only reachable after the policy was
        // swapped mid-life, but evicting a demoted entry before a live one is
        // the obviously right order. Within a tier: least recently released.
        let cached = slots.filter { $0.state == .cached }
        let demoted = cached.filter { slot in
            slot.region.map { !policy.admits($0.identity) } ?? true
        }
        guard
            let oldest = (demoted.isEmpty ? cached : demoted)
                .min(by: { $0.cacheSequence < $1.cacheSequence })
        else { return nil }
        if let identity = oldest.region?.identity {
            cachedByIdentity.removeValue(forKey: identity)
        }
        cacheEvictionCount += 1
        returnToFreeLocked(oldest)
        return oldest
    }

    private func returnToFreeLocked(_ slot: BufferSlot) {
        if configuration.poisonOnRelease {
            memset(slot.pointer, Int32(Self.poisonByte), slot.capacity)
        }
        slot.state = .free
        slot.refcount = 0
        slot.region = nil
    }

    private func censusLocked() -> SlotCensus {
        var census = SlotCensus()
        for slot in slots {
            switch slot.state {
            case .free: census.free += 1
            case .filling: census.filling += 1
            case .ready: census.ready += 1
            case .leased: census.leased += 1
            case .cached: census.cached += 1
            }
        }
        return census
    }

    private func updatePeaksLocked() {
        let current = censusLocked()
        peakInUseSlots = max(peakInUseSlots, current.inUse)
        peakLeasedSlots = max(peakLeasedSlots, current.leased)
        peakCachedSlots = max(peakCachedSlots, current.cached)
    }

    private func finishBlockedWaitLocked(since start: UInt64) {
        blockedAcquireCount += 1
        var total = UInt64Accounting.SaturatingSum(
            value: blockedNanoseconds, didOverflow: blockedTimeOverflowed)
        total.add(MonotonicClock.now() &- start)
        blockedNanoseconds = total.value
        blockedTimeOverflowed = total.didOverflow
    }

    /// Release for callers that cannot throw — an MLX finalizer is the reason
    /// this exists. A fault is recorded and surfaced through ``statistics``
    /// rather than trapping: a `fatalError` here would abort the process from a
    /// Metal completion handler with no useful context, and spec §5.8 asks for
    /// an explicit report, not a crash. Callers are expected to assert the
    /// fault list is empty at the end of a run, and the tests do.
    func releaseRecordingFaults(_ slot: BufferSlot) {
        do {
            try release(slot)
        } catch {
            monitor.lock()
            faults.append("\(error)")
            monitor.unlock()
        }
    }
}
