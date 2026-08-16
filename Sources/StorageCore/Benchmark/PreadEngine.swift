import Darwin
import Foundation

/// `queue_depth` threads, each issuing blocking `pread` calls.
///
/// The descriptor is shared: `pread` carries its own offset and does not touch
/// the file's position, so concurrent workers cannot interfere. Each worker
/// owns one page-aligned buffer for the whole run, so no allocation happens
/// between reads.
///
/// This is the mode that mirrors how the eventual pager would read — spec §4.5
/// records that prior on-device work settled on `pread` over `mmap` precisely
/// because a removable drive disappearing then surfaces as a short read instead
/// of a crash.
enum PreadEngine {
    static func run(
        configuration: BenchmarkConfiguration,
        fileDescriptor: Int32,
        fileSizeBytes: UInt64,
        window: RunWindow
    ) throws -> ReadAccounting {
        let accumulator = SharedAccounting()
        let group = DispatchGroup()

        for workerIndex in 0..<configuration.queueDepth {
            group.enter()
            let thread = Thread {
                defer { group.leave() }
                var local = ReadAccounting()
                do {
                    try workerLoop(
                        configuration: configuration,
                        fileDescriptor: fileDescriptor,
                        fileSizeBytes: fileSizeBytes,
                        window: window,
                        workerIndex: workerIndex,
                        accounting: &local)
                } catch {
                    local.recordError("\(error)")
                }
                accumulator.merge(local)
            }
            thread.name = "minirun.pread.\(workerIndex)"
            thread.qualityOfService = .userInitiated
            thread.stackSize = 512 << 10
            thread.start()
        }

        group.wait()
        return accumulator.snapshot
    }

    private static func workerLoop(
        configuration: BenchmarkConfiguration,
        fileDescriptor: Int32,
        fileSizeBytes: UInt64,
        window: RunWindow,
        workerIndex: Int,
        accounting: inout ReadAccounting
    ) throws {
        var scheduler = OffsetScheduler(
            fileSizeBytes: fileSizeBytes,
            readSizeBytes: configuration.readSizeBytes,
            pattern: configuration.pattern,
            workerIndex: workerIndex,
            workerCount: configuration.queueDepth,
            seed: configuration.seed)

        let buffer = try AlignedBuffer(count: configuration.readSizeBytes)
        // Latency samples are appended after the timed region, but reserving
        // keeps most reallocation out of the run entirely.
        accounting.latencySamplesNs.reserveCapacity(1 << 16)
        var readsSinceVerification = 0

        while true {
            let start = MonotonicClock.now()
            if window.isFinished(at: start) { break }
            let measuring = window.isMeasuring(at: start)

            guard let offset = scheduler.nextOffset() else { break }

            let bytes = pread(
                fileDescriptor, buffer.pointer, configuration.readSizeBytes, off_t(offset))
            let end = MonotonicClock.now()

            if bytes < 0 {
                if errno == EINTR { continue }
                let failure = StorageCoreError.posix(
                    operation: "pread", path: configuration.path, code: errno)
                accounting.recordError(failure.description)
                // Spec §5.8: stop rather than spin on a failing descriptor.
                return
            }

            guard measuring else { continue }

            accounting.readCount += 1
            accounting.bytesRequested += UInt64(configuration.readSizeBytes)
            accounting.bytesRead += UInt64(bytes)
            accounting.latencySamplesNs.append(end &- start)
            if bytes < configuration.readSizeBytes {
                accounting.shortReadCount += 1
            }

            if configuration.verify, bytes > 0 {
                readsSinceVerification += 1
                if readsSinceVerification >= configuration.verifySampleInterval {
                    readsSinceVerification = 0
                    let observed = UnsafeRawBufferPointer(start: buffer.pointer, count: bytes)
                    accounting.verification.merge(
                        DeterministicContent.verify(
                            observed, seed: configuration.seed, fileOffset: offset))
                    accounting.verifiedReadCount += 1
                }
            }
        }
    }
}
