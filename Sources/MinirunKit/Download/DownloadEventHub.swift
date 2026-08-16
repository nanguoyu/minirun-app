import Foundation
import StorageCore

/// Fan-out for a job's events, with a short replay.
///
/// The replay is not a convenience. `start(_:into:options:)` returns a job id
/// and the transfer begins immediately, so a caller that then asks for
/// `events(for:)` would otherwise miss `planned`, `preflight` and the first
/// files — and, on a fast local transfer, the terminal event too, leaving it
/// awaiting a stream that has already finished. Late subscribers get the
/// history and then the live feed.
///
/// The history is capped. A terabyte job emits progress continuously and an
/// unbounded log of it would be a slow memory leak with a good excuse, so the
/// oldest entries fall off — and, because the terminal event is what a late
/// subscriber most needs, it is kept out of that eviction.
final class DownloadEventHub: @unchecked Sendable {
    static let historyLimit = 512

    private let lock = NSLock()
    private var continuations: [DownloadJobID: [UUID: AsyncStream<DownloadEvent>.Continuation]] = [:]
    private var history: [DownloadJobID: [DownloadEvent]] = [:]
    private var terminal: [DownloadJobID: DownloadEvent] = [:]

    func emit(_ event: DownloadEvent, for job: DownloadJobID) {
        lock.lock()
        // A lifecycle has one terminal truth. A late sibling that ignored task
        // cancellation must not append progress or a competing terminal after
        // the job has already ended. `restart` deliberately opens a new
        // lifecycle for repair/resume and clears this guard.
        guard terminal[job] == nil else {
            lock.unlock()
            return
        }
        var log = history[job] ?? []
        log.append(event)
        if log.count > Self.historyLimit { log.removeFirst(log.count - Self.historyLimit) }
        history[job] = log
        if event.isTerminal { terminal[job] = event }
        let listeners = continuations[job]?.values.map { $0 } ?? []
        let isTerminal = event.isTerminal
        if isTerminal { continuations[job] = nil }
        lock.unlock()

        for listener in listeners {
            listener.yield(event)
            if isTerminal { listener.finish() }
        }
    }

    func stream(for job: DownloadJobID) -> AsyncStream<DownloadEvent> {
        AsyncStream(bufferingPolicy: .unbounded) { continuation in
            lock.lock()
            let replay = history[job] ?? []
            let finished = terminal[job] != nil
            let id = UUID()
            if !finished {
                continuations[job, default: [:]][id] = continuation
            }
            lock.unlock()

            for event in replay { continuation.yield(event) }
            if finished {
                continuation.finish()
                return
            }
            continuation.onTermination = { [weak self] _ in
                guard let self else { return }
                self.lock.lock()
                self.continuations[job]?[id] = nil
                self.lock.unlock()
            }
        }
    }

    /// Reopen a job that has already ended, for a repair or a resume.
    ///
    /// The terminal marker is what makes a late subscriber's stream finish
    /// immediately; leaving it in place would mean a repaired job's events
    /// could never be observed, and the caller would be handed the *previous*
    /// run's failure as though nothing had happened. The old log goes with it:
    /// the story starting now is the one that matters.
    func restart(_ job: DownloadJobID) {
        lock.lock()
        history[job] = nil
        terminal[job] = nil
        lock.unlock()
    }

    /// Drop a finished job's log. Called when its state is removed.
    func forget(_ job: DownloadJobID) {
        lock.lock()
        defer { lock.unlock() }
        history[job] = nil
        terminal[job] = nil
        continuations[job] = nil
    }
}

/// Bytes counted off the wire, updated from the transport's callback and read
/// from the actor. A class because the callback is `@Sendable` and synchronous:
/// it cannot `await` its way onto an actor, and a progress counter that lags a
/// hop behind the bytes it counts is a progress counter that lies at the end.
final class JobCounters: @unchecked Sendable {
    private let lock = NSLock()
    private var network: UInt64 = 0
    private var wasted: UInt64 = 0
    private var networkOverflowed = false
    private var wastedOverflowed = false

    func addNetwork(_ bytes: UInt64) {
        lock.lock()
        var sum = UInt64Accounting.SaturatingSum(
            value: network, didOverflow: networkOverflowed)
        sum.add(bytes)
        network = sum.value
        networkOverflowed = sum.didOverflow
        lock.unlock()
    }

    func addWasted(_ bytes: UInt64) {
        lock.lock()
        var sum = UInt64Accounting.SaturatingSum(
            value: wasted, didOverflow: wastedOverflowed)
        sum.add(bytes)
        wasted = sum.value
        wastedOverflowed = sum.didOverflow
        lock.unlock()
    }

    var networkBytes: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return network
    }

    var wastedBytes: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return wasted
    }

    var didOverflow: Bool {
        lock.lock()
        defer { lock.unlock() }
        return networkOverflowed || wastedOverflowed
    }
}

/// The latches a running job checks.
///
/// Separate from the job record because the transfer reads them from outside
/// the actor's isolation, at chunk boundaries, without a hop — a Stop button
/// whose effect is queued behind a five-gigabyte file is not a Stop button.
final class JobControl: @unchecked Sendable {
    private let lock = NSLock()
    private var paused = false
    private var cancelled = false
    private var persistenceAborted = false
    private var keepPartials = true

    var isPaused: Bool {
        lock.lock()
        defer { lock.unlock() }
        return paused
    }

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    var keepsPartialFiles: Bool {
        lock.lock()
        defer { lock.unlock() }
        return keepPartials
    }

    var mustAbortForPersistenceFailure: Bool {
        lock.lock()
        defer { lock.unlock() }
        return persistenceAborted
    }

    func pause() {
        lock.lock()
        paused = true
        lock.unlock()
    }

    func unpause() {
        lock.lock()
        paused = false
        lock.unlock()
    }

    func cancel(keepPartialFiles: Bool) {
        lock.lock()
        cancelled = true
        keepPartials = keepPartialFiles
        lock.unlock()
    }

    /// State persistence is an internal safety latch, distinct from user
    /// cancellation. Once raised, sibling transfers may finish an unavoidable
    /// transport write into their part files, but may not hash, promote, or
    /// persist another result.
    func abortForPersistenceFailure() {
        lock.lock()
        persistenceAborted = true
        lock.unlock()
    }
}
