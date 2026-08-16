import Foundation

/// Per-run tallies produced by an I/O engine.
///
/// Identical for both engines so that a `pread` result and a Dispatch I/O
/// result are comparable line for line, which is the point of the M1
/// experiment that pits them against each other.
public struct ReadAccounting {
    public var readCount = 0
    public var bytesRequested: UInt64 = 0
    public var bytesRead: UInt64 = 0
    /// Reads that returned fewer bytes than asked for. The scheduler never
    /// crosses EOF, so on a local file this should be zero; anything else is a
    /// signal, most likely a disconnected drive (spec §12.4).
    public var shortReadCount = 0
    public var errorCount = 0
    public var firstError: String?
    /// Latencies of measured reads, nanoseconds. Warm-up reads are absent.
    public var latencySamplesNs: [UInt64] = []
    public var verifiedReadCount = 0
    public var verification = DeterministicContent.Verification.clean

    public init() {}

    public mutating func recordError(_ description: String) {
        errorCount += 1
        if firstError == nil { firstError = description }
    }

    public mutating func merge(_ other: ReadAccounting) {
        readCount += other.readCount
        bytesRequested += other.bytesRequested
        bytesRead += other.bytesRead
        shortReadCount += other.shortReadCount
        errorCount += other.errorCount
        if firstError == nil { firstError = other.firstError }
        latencySamplesNs.append(contentsOf: other.latencySamplesNs)
        verifiedReadCount += other.verifiedReadCount
        verification.merge(other.verification)
    }
}

/// Lock-guarded accumulator shared by workers or completion handlers.
final class SharedAccounting: @unchecked Sendable {
    private let lock = NSLock()
    private var value = ReadAccounting()

    func merge(_ other: ReadAccounting) {
        lock.lock()
        defer { lock.unlock() }
        value.merge(other)
    }

    /// Applies `body` to the shared value under the lock. Used by the Dispatch
    /// I/O engine, whose completion handlers record one read at a time.
    func withLock<Result>(_ body: (inout ReadAccounting) -> Result) -> Result {
        lock.lock()
        defer { lock.unlock() }
        return body(&value)
    }

    var snapshot: ReadAccounting {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    /// Whether any read has failed yet.
    ///
    /// Lets an issuing loop stop submitting work once the descriptor has gone
    /// bad. One uncontended lock acquisition per issued read costs tens of
    /// nanoseconds against a read measured in tens of microseconds, and it
    /// saves an engine from logging thousands of identical failures against an
    /// unplugged drive.
    var hasError: Bool {
        lock.lock()
        defer { lock.unlock() }
        return value.errorCount > 0
    }
}

/// Timing boundaries shared by the engines.
struct RunWindow {
    /// Reads that start before this are warm-up and are not accounted for.
    let warmupDeadline: UInt64
    /// No read is started at or after this.
    let stopDeadline: UInt64

    func isMeasuring(at instant: UInt64) -> Bool { instant >= warmupDeadline }
    func isFinished(at instant: UInt64) -> Bool { instant >= stopDeadline }
}
