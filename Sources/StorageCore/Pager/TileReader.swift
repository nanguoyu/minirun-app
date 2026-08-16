import Darwin
import Foundation

/// Sizing of the read path (ADR-0001, spec §12.3).
public struct TileReaderConfiguration: Sendable {
    /// `pread` worker threads. M1a measured the internal SSD saturating at
    /// queue depth 2–4 for full-slice reads, with throughput flat and latency
    /// rising above that, so the useful range is small and the default sits in
    /// the measured plateau.
    public var queueDepth: Int
    /// How many regions the reader may hold slots for at once. This is one half
    /// of the deadlock invariant: the consumer needs `slotCount - readAhead`
    /// slots left over to hold tiles that are referenced by an unevaluated
    /// compute graph. See ``PagedReadPlan``.
    public var readAhead: Int
    /// Ask the kernel not to populate the page cache. Required for any honest
    /// storage measurement (spec §5.7); irrelevant to correctness tests.
    public var noCache: Bool
    /// Bounds a stall rather than expressing an expectation. A real deadlock
    /// then surfaces as an error naming the wait instead of hanging a test run.
    public var slotAcquireTimeout: TimeInterval?

    public init(
        queueDepth: Int = 3,
        readAhead: Int = 2,
        noCache: Bool = false,
        slotAcquireTimeout: TimeInterval? = 120
    ) {
        self.queueDepth = queueDepth
        self.readAhead = readAhead
        self.noCache = noCache
        self.slotAcquireTimeout = slotAcquireTimeout
    }

    func validated() throws -> TileReaderConfiguration {
        guard (1...8).contains(queueDepth) else {
            throw PagerError.configuration("queueDepth \(queueDepth) is outside 1...8")
        }
        guard readAhead >= 1 else {
            throw PagerError.configuration("readAhead must be at least 1")
        }
        return self
    }
}

/// What the read path did, for the M4 instrumentation requirement.
// Not `Sendable`: `LatencyStatistics` predates this milestone and is not marked
// as such, and retrofitting a conformance onto the benchmark's types is not
// this change's business. The reader hands out copies under its own lock.
public struct TileReaderStatistics: Codable, Equatable {
    public var readCount = 0
    public var bytesRead: UInt64 = 0
    /// Sticky. `bytesRead == UInt64.max` is a saturation, not a wrapped total.
    public var byteCountOverflowed = false
    public var shortReadCount = 0
    public var errorCount = 0
    public var firstError: String?
    public var latency: LatencyStatistics?
    /// Wall time the dispatcher spent waiting for a slot: the pool applying
    /// backpressure to storage, which is the intended steady state rather than
    /// a fault (spec §12.3).
    public var slotWaitSeconds: Double = 0
}

/// Reads regions into pool slots on a `pread` worker pool.
///
/// ## Shape of the API
///
/// ``schedule(_:)`` takes region descriptors and ``acquire(_:)`` returns a
/// lease for one of them. There is no cursor: the caller decides both what to
/// prefetch and what to wait for, in whatever order it likes. A dense linear
/// schedules its tiles in row order and acquires them in the same order; a
/// routed MoE layer schedules the experts its router picked. Same two calls.
///
/// ## Why a single dispatcher thread
///
/// Slots are assigned to regions by one thread, in schedule order, *before* the
/// work reaches a worker. The alternative — every worker dequeuing a region and
/// then acquiring a slot — deadlocks: worker A can take region *i+3*, block on
/// an empty pool, and sit ahead of worker B holding region *i* in the pool's
/// wait queue, while the consumer waits for region *i* and therefore never
/// releases the slots that would let A proceed. In-order assignment removes
/// that ordering hazard entirely: region *i* always holds a slot before region
/// *i+1* asks for one. The workers then only do the blocking `pread`, which is
/// what the queue depth is actually for (ADR-0001).
///
/// ## Lifetime
///
/// The reader owns threads and a file descriptor, and its worker threads hold a
/// strong reference to it while they run. ``shutdown()`` is therefore not
/// optional — without it the threads never exit and the object never
/// deallocates. Call it from a `defer`.
public final class TileReader: @unchecked Sendable {

    private enum EntryState {
        case queued
        case dispatched(BufferSlot)
        case ready(BufferSlot)
        case failed(Error)
        case delivered
        case abandoned
    }

    private final class Entry {
        let region: RegionDescriptor
        var state: EntryState = .queued
        /// Set when a consumer is already blocked on this region: it then jumps
        /// the queue and ignores the read-ahead window, because prefetching
        /// past a request that is actively blocking progress buys nothing.
        var urgent = false
        init(region: RegionDescriptor) { self.region = region }
    }

    public let path: String
    public let configuration: TileReaderConfiguration

    private let pool: BufferPool
    private let successfulReadObserver: (@Sendable (UInt64) -> Void)?
    private var fileDescriptor: Int32
    private let sourceBytes: UInt64
    private let monitor = NSCondition()

    /// Every region known to the reader and not yet handed over or failed. This
    /// is the authoritative set for reclaiming slots at shutdown; the two
    /// queues below are only work lists.
    private var entries: [RegionIdentity: Entry] = [:]
    private var pending: [Entry] = []
    private var dispatched: [Entry] = []
    private var inFlight = 0
    private var finished = false
    private var closed = false
    private var cancellationReason: String?
    private var liveThreads = 0
    /// Callers currently asleep in ``acquire(_:)`` waiting for a queued or
    /// dispatched region. Kept under `monitor`; the internal wait helper below
    /// lets a concurrency test synchronize on the state it intends to cancel
    /// instead of guessing with a wall-clock sleep.
    private var waitingConsumers = 0

    private var statisticsAccumulator = TileReaderStatistics()
    private var latencySamplesNs: [UInt64] = []

    public convenience init(
        path: String,
        pool: BufferPool,
        configuration: TileReaderConfiguration = TileReaderConfiguration(),
        successfulReadObserver: (@Sendable (UInt64) -> Void)? = nil
    ) throws {
        let descriptor = open(path, O_RDONLY)
        guard descriptor >= 0 else {
            throw StorageCoreError.posix(operation: "open", path: path, code: errno)
        }
        try self.init(
            owningFileDescriptor: descriptor, path: path, pool: pool,
            configuration: configuration,
            successfulReadObserver: successfulReadObserver)
    }

    /// Build a reader around a descriptor supplied by a rooted authority.
    ///
    /// Ownership transfers at entry, including failure paths. The descriptor
    /// is never reopened by pathname and is closed by ``shutdown()``.
    public init(
        owningFileDescriptor descriptor: Int32,
        path: String,
        pool: BufferPool,
        configuration: TileReaderConfiguration = TileReaderConfiguration(),
        successfulReadObserver: (@Sendable (UInt64) -> Void)? = nil
    ) throws {
        guard descriptor >= 0 else {
            throw StorageCoreError.posix(operation: "open", path: path, code: EBADF)
        }
        let validated: TileReaderConfiguration
        do {
            validated = try configuration.validated()
        } catch {
            close(descriptor)
            throw error
        }
        self.path = path
        self.pool = pool
        self.successfulReadObserver = successfulReadObserver
        self.configuration = validated

        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            let code = errno
            close(descriptor)
            throw StorageCoreError.posix(operation: "fstat", path: path, code: code)
        }
        guard status.st_size >= 0 else {
            close(descriptor)
            throw PagerError.configuration("'\(path)' reports a negative byte length")
        }
        self.sourceBytes = UInt64(status.st_size)
        if validated.noCache {
            // Advisory: a failure changes what is being measured, not whether
            // the read is correct, so it is not fatal.
            _ = fcntl(descriptor, F_NOCACHE, 1)
        }
        self.fileDescriptor = descriptor

        startThreads()
    }

    deinit {
        // Only reachable once the threads have already exited and dropped their
        // strong references, i.e. after shutdown(). Kept as a backstop for the
        // descriptor.
        if fileDescriptor >= 0 { close(fileDescriptor) }
    }

    // MARK: - Request side

    /// Queue regions for background filling, in the given order.
    public func schedule(_ regions: [RegionDescriptor]) throws {
        for region in regions { try validate(region) }
        monitor.lock()
        defer { monitor.unlock() }
        if let reason = cancellationReason {
            throw PagerError.cancelled(reason: reason)
        }
        guard !finished else {
            throw PagerError.configuration("schedule after shutdown")
        }
        for region in regions {
            if let existing = entries[region.identity] {
                guard existing.region == region else {
                    throw PagerError.invalidRegion(
                        identity: region.identity, offset: region.offset, length: region.length,
                        sourceBytes: sourceBytes,
                        reason: "identity is already assigned to a different byte range")
                }
            } else {
                let entry = Entry(region: region)
                entries[region.identity] = entry
                pending.append(entry)
            }
        }
        monitor.broadcast()
    }

    /// Wait for a region's bytes and take a lease on the slot holding them.
    ///
    /// The lease comes back with refcount 1, held by the caller. Every further
    /// consumer — each `MLXArray` adopted out of the slot — adds its own.
    public func acquire(_ region: RegionDescriptor) throws -> SlotLease {
        try validate(region)
        // A hit needs no read at all. With the default pool policy there are
        // never hits in a single streaming pass; the path exists because the
        // routed-expert case is where it starts paying (spec §16.2).
        if let hit = pool.leaseCached(region.identity) {
            guard hit.region == region else {
                try? hit.releaseChecked()
                throw PagerError.invalidRegion(
                    identity: region.identity, offset: region.offset, length: region.length,
                    sourceBytes: sourceBytes,
                    reason: "identity names cached bytes from a different range")
            }
            return hit
        }

        monitor.lock()
        let entry: Entry
        if let existing = entries[region.identity] {
            guard existing.region == region else {
                monitor.unlock()
                throw PagerError.invalidRegion(
                    identity: region.identity, offset: region.offset, length: region.length,
                    sourceBytes: sourceBytes,
                    reason: "identity is scheduled for a different byte range")
            }
            entry = existing
        } else {
            if let reason = cancellationReason {
                monitor.unlock()
                throw PagerError.cancelled(reason: reason)
            }
            guard !finished else {
                monitor.unlock()
                throw PagerError.regionNotScheduled(identity: region.identity)
            }
            let created = Entry(region: region)
            entries[region.identity] = created
            pending.insert(created, at: 0)
            entry = created
        }
        if case .queued = entry.state {
            entry.urgent = true
            if let position = pending.firstIndex(where: { $0 === entry }), position != 0 {
                pending.remove(at: position)
                pending.insert(entry, at: 0)
            }
        }
        monitor.broadcast()

        while true {
            switch entry.state {
            case .ready(let slot):
                entry.state = .delivered
                entries.removeValue(forKey: region.identity)
                inFlight -= 1
                monitor.broadcast()
                monitor.unlock()
                // Taken outside the reader's monitor: the pool has its own, and
                // nesting them in one order here and the other order in the
                // dispatcher is how lock-order inversions get built.
                return try pool.lease(slot)
            case .failed(let error):
                entries.removeValue(forKey: region.identity)
                inFlight -= 1
                monitor.broadcast()
                monitor.unlock()
                throw error
            case .delivered, .abandoned:
                monitor.unlock()
                throw PagerError.regionNotScheduled(identity: region.identity)
            case .queued, .dispatched:
                if let reason = cancellationReason {
                    monitor.unlock()
                    throw PagerError.cancelled(reason: reason)
                }
                waitingConsumers += 1
                monitor.broadcast()
                monitor.wait()
                waitingConsumers -= 1
                monitor.broadcast()
            }
        }
    }

    // MARK: - Cancellation and shutdown

    /// Stop issuing reads and wake every waiter with a cancellation error.
    ///
    /// In-flight reads are allowed to finish — a `pread` cannot be un-issued —
    /// and their slots go back to the pool rather than to a consumer. Leases
    /// already handed out are left alone: they belong to the consumer now, and
    /// reclaiming a slot a GPU command buffer might still read is exactly the
    /// eviction spec §11.1 forbids.
    public func cancel(reason: String) {
        monitor.lock()
        cancellationReason = reason
        monitor.broadcast()
        monitor.unlock()
        pool.cancel(reason: reason)
    }

    /// Stop the threads, reclaim every slot the reader still holds, and close
    /// the descriptor. Idempotent.
    public func shutdown() {
        monitor.lock()
        if closed {
            monitor.unlock()
            return
        }
        finished = true
        monitor.broadcast()
        monitor.unlock()

        // A dispatcher blocked inside `pool.acquire` only wakes on the pool's
        // own condition, so it has to be woken there too. Deliberately
        // `interrupt` and not `cancel`: the pool may belong to the caller and
        // outlive this reader, and cancelling it here would leave the next run
        // on that pool failing before it read a byte.
        pool.interrupt()

        monitor.lock()
        while liveThreads > 0 {
            monitor.wait()
        }
        // With the threads gone, `entries` is stable and authoritative.
        let reclaimable: [BufferSlot] = entries.values.compactMap { entry in
            switch entry.state {
            case .ready(let slot), .dispatched(let slot):
                entry.state = .abandoned
                return slot
            default:
                return nil
            }
        }
        entries.removeAll()
        pending.removeAll()
        dispatched.removeAll()
        inFlight = 0
        closed = true
        let descriptor = fileDescriptor
        fileDescriptor = -1
        monitor.unlock()

        for slot in reclaimable { try? pool.discard(slot) }
        if descriptor >= 0 { close(descriptor) }
    }

    // MARK: - Observation

    public var statistics: TileReaderStatistics {
        monitor.lock()
        defer { monitor.unlock() }
        var result = statisticsAccumulator
        var samples = latencySamplesNs
        result.latency = LatencyStatistics.compute(nanosecondSamples: &samples)
        return result
    }

    /// Raw per-read latencies, nanoseconds, in completion order.
    ///
    /// ``statistics`` summarises one reader. A caller running several readers
    /// over several files needs the *pooled* distribution, and pooling
    /// summaries is not a thing that can be done: a p99 cannot be recovered
    /// from a mean and a count. So the samples themselves are available, and
    /// the caller concatenates before summarising.
    public var latencySamples: [UInt64] {
        monitor.lock()
        defer { monitor.unlock() }
        return latencySamplesNs
    }

    /// Wait until at least one consumer is actually blocked inside
    /// ``acquire(_:)``. Internal rather than public: this is a synchronization
    /// seam for deterministic concurrency tests, not product telemetry.
    func waitUntilConsumerIsWaiting(timeout: TimeInterval) -> Bool {
        monitor.lock()
        defer { monitor.unlock() }
        if waitingConsumers > 0 { return true }

        let deadline = Date().addingTimeInterval(timeout)
        while waitingConsumers == 0, !finished, cancellationReason == nil {
            if !monitor.wait(until: deadline) { break }
        }
        return waitingConsumers > 0
    }

    public func resetStatistics() {
        monitor.lock()
        defer { monitor.unlock() }
        statisticsAccumulator = TileReaderStatistics()
        latencySamplesNs.removeAll(keepingCapacity: true)
    }

    // MARK: - Threads

    private func startThreads() {
        monitor.lock()
        liveThreads = configuration.queueDepth + 1
        monitor.unlock()

        // Strong capture on purpose: a worker mid-`pread` must not be able to
        // outlive the object holding its file descriptor and its pool. The
        // cycle is broken by shutdown(), which is why shutdown() is mandatory.
        let dispatcher = Thread { self.dispatcherLoop() }
        dispatcher.name = "minirun.pager.dispatch"
        dispatcher.qualityOfService = .userInitiated
        dispatcher.stackSize = 512 << 10
        dispatcher.start()

        for index in 0..<configuration.queueDepth {
            let worker = Thread { self.workerLoop() }
            worker.name = "minirun.pager.read.\(index)"
            worker.qualityOfService = .userInitiated
            worker.stackSize = 512 << 10
            worker.start()
        }
    }

    private func threadDidExit() {
        monitor.lock()
        liveThreads -= 1
        monitor.broadcast()
        monitor.unlock()
    }

    /// Assigns slots to regions strictly in schedule order. See the type-level
    /// note for why this is one thread and not `queueDepth` of them.
    private func dispatcherLoop() {
        defer { threadDidExit() }
        while true {
            monitor.lock()
            while true {
                if finished || cancellationReason != nil {
                    monitor.unlock()
                    return
                }
                if let head = pending.first, head.urgent || inFlight < configuration.readAhead {
                    break
                }
                monitor.wait()
            }
            let entry = pending.removeFirst()
            inFlight += 1
            monitor.unlock()

            let waitStart = MonotonicClock.now()
            let slot: BufferSlot
            do {
                slot = try pool.acquire(timeout: configuration.slotAcquireTimeout)
            } catch {
                monitor.lock()
                statisticsAccumulator.slotWaitSeconds += MonotonicClock.seconds(since: waitStart)
                if finished || cancellationReason != nil {
                    // Shutdown or cancellation woke this wait. Nobody will come
                    // for the region, so the reader stops accounting for it and
                    // does not file it as a read error.
                    inFlight -= 1
                    entry.state = .abandoned
                    monitor.broadcast()
                    monitor.unlock()
                    return
                }
                entry.state = .failed(error)
                recordErrorLocked(error)
                monitor.broadcast()
                monitor.unlock()
                continue
            }

            monitor.lock()
            statisticsAccumulator.slotWaitSeconds += MonotonicClock.seconds(since: waitStart)
            if finished || cancellationReason != nil {
                inFlight -= 1
                entry.state = .abandoned
                monitor.broadcast()
                monitor.unlock()
                try? pool.discard(slot)
                return
            }
            entry.state = .dispatched(slot)
            dispatched.append(entry)
            monitor.broadcast()
            monitor.unlock()
        }
    }

    /// The only thing a worker does is one blocking `pread` per region
    /// (ADR-0001). No allocation, no scheduling decisions.
    private func workerLoop() {
        defer { threadDidExit() }
        while true {
            monitor.lock()
            while true {
                if finished || cancellationReason != nil {
                    monitor.unlock()
                    return
                }
                if !dispatched.isEmpty { break }
                monitor.wait()
            }
            let entry = dispatched.removeFirst()
            guard case .dispatched(let slot) = entry.state else {
                monitor.unlock()
                continue
            }
            let region = entry.region
            monitor.unlock()

            switch performRead(region: region, into: slot) {
            case .success(let latencyNs, let bytes):
                // Completion means the entire requested payload is now in this
                // process. Count here, not when it was scheduled or acquired:
                // prefetched-and-discarded bytes remain real traffic, while a
                // short/failed read never masquerades as successful flow.
                successfulReadObserver?(UInt64(bytes))
                // markReady is taken outside the reader's monitor, like every
                // other pool call, to keep a single lock order.
                let readyError: Error?
                do {
                    try pool.markReady(slot, region: region)
                    readyError = nil
                } catch {
                    readyError = error
                }
                monitor.lock()
                statisticsAccumulator.readCount += 1
                var byteTotal = UInt64Accounting.SaturatingSum(
                    value: statisticsAccumulator.bytesRead,
                    didOverflow: statisticsAccumulator.byteCountOverflowed)
                byteTotal.add(UInt64(bytes))
                statisticsAccumulator.bytesRead = byteTotal.value
                statisticsAccumulator.byteCountOverflowed = byteTotal.didOverflow
                latencySamplesNs.append(latencyNs)
                if let readyError {
                    entry.state = .failed(readyError)
                    recordErrorLocked(readyError)
                } else {
                    entry.state = .ready(slot)
                }
                monitor.broadcast()
                monitor.unlock()

            case .failure(let error, let isShort):
                monitor.lock()
                if isShort { statisticsAccumulator.shortReadCount += 1 }
                recordErrorLocked(error)
                entry.state = .failed(error)
                monitor.broadcast()
                monitor.unlock()
                // A partially filled slot must never reach a consumer
                // (spec §12.4). It goes straight back to the pool.
                try? pool.discard(slot)
            }
        }
    }

    private enum ReadOutcome {
        case success(latencyNs: UInt64, bytes: Int)
        case failure(Error, isShort: Bool)
    }

    private func validate(_ region: RegionDescriptor) throws {
        guard region.length > 0 else {
            throw PagerError.invalidRegion(
                identity: region.identity, offset: region.offset, length: region.length,
                sourceBytes: sourceBytes, reason: "length must be positive")
        }
        guard region.length <= pool.configuration.slotBytes else {
            throw PagerError.regionTooLarge(
                identity: region.identity, length: region.length,
                slotBytes: pool.configuration.slotBytes)
        }
        let length = UInt64(region.length)
        let end = region.offset.addingReportingOverflow(length)
        guard !end.overflow else {
            throw PagerError.invalidRegion(
                identity: region.identity, offset: region.offset, length: region.length,
                sourceBytes: sourceBytes, reason: "offset + length exceeds UInt64.max")
        }
        guard end.partialValue <= sourceBytes else {
            throw PagerError.invalidRegion(
                identity: region.identity, offset: region.offset, length: region.length,
                sourceBytes: sourceBytes,
                reason: "end offset \(end.partialValue) is beyond end of source")
        }
        guard region.offset <= UInt64(Int64.max), end.partialValue <= UInt64(Int64.max) else {
            throw PagerError.invalidRegion(
                identity: region.identity, offset: region.offset, length: region.length,
                sourceBytes: sourceBytes, reason: "range cannot be represented by pread's off_t")
        }
    }

    private func performRead(region: RegionDescriptor, into slot: BufferSlot) -> ReadOutcome {
        var total = 0
        let start = MonotonicClock.now()
        while total < region.length {
            let bytes = pread(
                fileDescriptor,
                slot.pointer + total,
                region.length - total,
                off_t(region.offset) + off_t(total))
            if bytes < 0 {
                if errno == EINTR { continue }
                return .failure(
                    PagerError.readFailed(
                        identity: region.identity,
                        underlying: StorageCoreError.posix(
                            operation: "pread", path: path, code: errno).description),
                    isShort: false)
            }
            if bytes == 0 {
                // End of file before the region was complete: a truncated
                // container, or a drive that went away mid-read. Either way the
                // error names the exact bytes that are missing (spec §12.4).
                return .failure(
                    StorageCoreError.shortTransfer(
                        operation: "pread",
                        path: path,
                        offset: region.offset,
                        expected: region.length,
                        actual: total),
                    isShort: true)
            }
            total += bytes
        }
        return .success(latencyNs: MonotonicClock.now() &- start, bytes: total)
    }

    private func recordErrorLocked(_ error: Error) {
        statisticsAccumulator.errorCount += 1
        if statisticsAccumulator.firstError == nil {
            statisticsAccumulator.firstError = "\(error)"
        }
    }
}
