import Darwin
import Dispatch
import Foundation

/// Random-access `DispatchIO` channels with `queue_depth` reads in flight.
///
/// The counterpart to `PreadEngine` for the M1 experiment that compares a
/// `pread` worker pool against Dispatch I/O. The two engines walk the same
/// offset sequence — this one keeps `queue_depth` schedulers and cycles through
/// them, so the offsets it issues are the ones `queue_depth` pread workers
/// would have issued — and record the same tallies, so a difference in the
/// result is a difference in the mechanism.
///
/// `configuration.dispatchIOChannels` selects the channel topology: one shared
/// channel, or one per in-flight slot. **Only the topology changes between
/// them.** The offset sequence stays `schedulers[issued % depth]` in both
/// modes, so a `one` result and a `per-slot` result differ in exactly one
/// variable and the comparison actually isolates what it claims to.
///
/// Latency is measured from the moment a slot frees up (the analogue of a
/// pread worker becoming free) to the completion of the read, so both engines
/// report the same thing: how long a caller waited for its bytes.
enum DispatchIOEngine {
    static func run(
        configuration: BenchmarkConfiguration,
        fileDescriptor: Int32,
        fileSizeBytes: UInt64,
        window: RunWindow
    ) throws -> ReadAccounting {
        let depth = configuration.queueDepth
        let readSize = configuration.readSizeBytes
        let accumulator = SharedAccounting()

        // One serial queue per in-flight slot. DispatchIO may invoke a handler
        // several times for a single read (partial deliveries); a serial queue
        // per slot keeps those invocations ordered without serializing
        // different reads against each other.
        let channelQueue = DispatchQueue(
            label: "minirun.dispatchio.channel", qos: .userInitiated)
        let slotQueues = (0..<depth).map { slot in
            DispatchQueue(label: "minirun.dispatchio.slot.\(slot)", qos: .userInitiated)
        }

        let teardown = DispatchGroup()
        let channels = try makeChannels(
            configuration: configuration,
            fileDescriptor: fileDescriptor,
            queue: channelQueue,
            teardown: teardown)
        for channel in channels {
            // Deliver a read in one piece rather than dribbling partial buffers.
            channel.setLimit(lowWater: readSize)
            channel.setLimit(highWater: readSize)
        }

        var schedulers = (0..<depth).map { workerIndex in
            OffsetScheduler(
                fileSizeBytes: fileSizeBytes,
                readSizeBytes: readSize,
                pattern: configuration.pattern,
                workerIndex: workerIndex,
                workerCount: depth,
                seed: configuration.seed)
        }

        let slots = SlotPool(count: depth)
        var issued = 0
        var readsSinceVerification = 0

        issuing: while true {
            let slot = slots.acquire()

            // Stop submitting once a read has failed, matching what a pread
            // worker does when it hits a non-EINTR error. Without this, an
            // unplugged drive collects one failure per issue for the rest of
            // the window rather than one failure and a stop (spec §5.8, §12.4).
            if accumulator.hasError {
                slots.release(slot)
                break issuing
            }

            let start = MonotonicClock.now()
            if window.isFinished(at: start) {
                slots.release(slot)
                break issuing
            }
            let measuring = window.isMeasuring(at: start)

            guard let offset = schedulers[issued % depth].nextOffset() else {
                slots.release(slot)
                break issuing
            }
            issued += 1

            var shouldVerify = false
            if configuration.verify, measuring {
                readsSinceVerification += 1
                if readsSinceVerification >= configuration.verifySampleInterval {
                    readsSinceVerification = 0
                    shouldVerify = true
                }
            }

            let pending = InFlightRead(
                offset: offset, start: start, measuring: measuring, shouldVerify: shouldVerify)

            // `one` mode has a single channel and lands on index 0 for every
            // slot; `per-slot` mode has one per slot, and slots are unique
            // across in-flight reads, so no channel serves two reads at once.
            let channel = channels[slot % channels.count]

            channel.read(offset: off_t(offset), length: readSize, queue: slotQueues[slot]) {
                done, data, error in
                if let data, !data.isEmpty {
                    if pending.shouldVerify {
                        data.enumerateBytes { region, byteIndex, _ in
                            pending.verification.merge(
                                DeterministicContent.verify(
                                    UnsafeRawBufferPointer(region),
                                    seed: configuration.seed,
                                    fileOffset: pending.offset &+ UInt64(byteIndex)))
                        }
                    }
                    pending.bytesDelivered += data.count
                }

                guard done else { return }
                let end = MonotonicClock.now()

                accumulator.withLock { accounting in
                    if error != 0 {
                        let failure = StorageCoreError.posix(
                            operation: "DispatchIO.read", path: configuration.path, code: error)
                        accounting.recordError(failure.description)
                    } else if pending.measuring {
                        accounting.readCount += 1
                        accounting.bytesRequested += UInt64(readSize)
                        accounting.bytesRead += UInt64(pending.bytesDelivered)
                        accounting.latencySamplesNs.append(end &- pending.start)
                        if pending.bytesDelivered < readSize {
                            accounting.shortReadCount += 1
                        }
                        if pending.shouldVerify {
                            accounting.verification.merge(pending.verification)
                            accounting.verifiedReadCount += 1
                        }
                    }
                }
                slots.release(slot)
            }
        }

        // Let every outstanding read finish before reporting.
        slots.drain()
        for channel in channels { channel.close() }

        // Wait for the cleanup handlers, which is where per-slot mode closes
        // the descriptors it duplicated. The timeout is a backstop only: the
        // reads have already drained, so the handlers should fire immediately,
        // and a stuck one must not hang a finished benchmark.
        _ = teardown.wait(timeout: .now() + 10)

        return accumulator.snapshot
    }

    /// Builds the channel set for the configured topology.
    ///
    /// Descriptor ownership differs by mode, and getting it wrong is the way
    /// this code would corrupt a run:
    ///
    /// - `one` uses the caller's descriptor directly, so its cleanup handler
    ///   does nothing. `StorageBenchmark` opened that descriptor and closes it;
    ///   a channel closing it as well would be a double close, and on a
    ///   long-lived process could take down whatever inherited the number.
    /// - `per-slot` gives each channel its own `dup`, which that channel owns
    ///   and closes in its cleanup handler. Duplicating rather than reopening
    ///   keeps every channel on the same open file description, so the
    ///   `F_NOCACHE` state the caller established still applies; it is
    ///   reapplied anyway, since a redundant `fcntl` is cheaper than depending
    ///   on that being true.
    private static func makeChannels(
        configuration: BenchmarkConfiguration,
        fileDescriptor: Int32,
        queue: DispatchQueue,
        teardown: DispatchGroup
    ) throws -> [DispatchIO] {
        guard configuration.mode == .dispatchio else { return [] }

        if configuration.dispatchIOChannels == .one {
            teardown.enter()
            let channel = DispatchIO(
                type: .random,
                fileDescriptor: fileDescriptor,
                queue: queue,
                cleanupHandler: { _ in
                    // The caller owns this descriptor. Do not close it here.
                    teardown.leave()
                })
            return [channel]
        }

        // Duplicate every descriptor first, so a failure part-way leaves
        // nothing half-built for a cleanup handler to close.
        var duplicates: [Int32] = []
        duplicates.reserveCapacity(configuration.queueDepth)
        do {
            for _ in 0..<configuration.queueDepth {
                let duplicate = dup(fileDescriptor)
                guard duplicate >= 0 else {
                    throw StorageCoreError.posix(
                        operation: "dup", path: configuration.path, code: errno)
                }
                if configuration.disableCache {
                    _ = fcntl(duplicate, F_NOCACHE, 1)
                }
                duplicates.append(duplicate)
            }
        } catch {
            for duplicate in duplicates { close(duplicate) }
            throw error
        }

        return duplicates.map { duplicate in
            teardown.enter()
            return DispatchIO(
                type: .random,
                fileDescriptor: duplicate,
                queue: queue,
                cleanupHandler: { _ in
                    // This channel owns its duplicate and hands it back here.
                    close(duplicate)
                    teardown.leave()
                })
        }
    }
}

/// State of a single outstanding read.
private final class InFlightRead: @unchecked Sendable {
    let offset: UInt64
    let start: UInt64
    let measuring: Bool
    let shouldVerify: Bool
    /// Mutated only from the read's own serial slot queue.
    var bytesDelivered = 0
    var verification = DeterministicContent.Verification.clean

    init(offset: UInt64, start: UInt64, measuring: Bool, shouldVerify: Bool) {
        self.offset = offset
        self.start = start
        self.measuring = measuring
        self.shouldVerify = shouldVerify
    }
}

/// Bounds in-flight reads and hands out a distinct slot for each one, so that
/// no two concurrent reads share a completion queue.
private final class SlotPool: @unchecked Sendable {
    private let semaphore: DispatchSemaphore
    private let lock = NSLock()
    private var free: [Int]
    private let count: Int

    init(count: Int) {
        self.count = count
        self.semaphore = DispatchSemaphore(value: count)
        self.free = Array(0..<count)
    }

    func acquire() -> Int {
        semaphore.wait()
        lock.lock()
        defer { lock.unlock() }
        return free.removeLast()
    }

    func release(_ slot: Int) {
        lock.lock()
        free.append(slot)
        lock.unlock()
        semaphore.signal()
    }

    /// Blocks until every slot is free.
    ///
    /// The permits taken here are handed straight back: libdispatch traps if a
    /// semaphore is deallocated while its value is below the one it was created
    /// with, so draining by acquiring everything and keeping it would turn a
    /// finished run into a crash.
    func drain() {
        for _ in 0..<count { semaphore.wait() }
        for _ in 0..<count { semaphore.signal() }
    }
}
