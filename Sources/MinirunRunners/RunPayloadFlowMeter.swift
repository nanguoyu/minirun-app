import Dispatch
import Foundation
import MinirunKit
import StorageCore

/// Run-scoped fixed-window payload accounting.
///
/// Read paths call only ``recordDeterministic(_:)`` or ``recordExpert(_:)``.
/// Both are one lock plus two saturating additions; formatting and stream
/// delivery happen on a private 2 Hz timer. The timer's interval comes from
/// `CLOCK_UPTIME_RAW`, so wall-clock changes and device sleep cannot invent a
/// throughput spike.
final class RunPayloadFlowMeter: @unchecked Sendable {
    enum Bucket { case deterministic, expert }

    struct Snapshot: Equatable {
        let deterministicBytes: UInt64
        let expertBytes: UInt64
        let didOverflow: Bool

        var accounting: ByteAccounting {
            let total = deterministicBytes.addingReportingOverflow(expertBytes)
            return ByteAccounting(
                totalBytesRead: didOverflow || total.overflow ? .max : total.partialValue,
                deterministicBytesRead: deterministicBytes,
                expertBytesRead: expertBytes)
        }
    }

    let stream: AsyncStream<RunPayloadFlowSample>

    private let lock = NSLock()
    private let continuation: AsyncStream<RunPayloadFlowSample>.Continuation
    private let intervalSeconds: TimeInterval
    private let clock: @Sendable () -> UInt64
    private let wallClock: @Sendable () -> Date
    private var timer: DispatchSourceTimer?

    private var lastTick: UInt64
    private var sequence: UInt64 = 0
    private var deterministicWindow = UInt64Accounting.SaturatingSum()
    private var expertWindow = UInt64Accounting.SaturatingSum()
    private var deterministicTotal = UInt64Accounting.SaturatingSum()
    private var expertTotal = UInt64Accounting.SaturatingSum()
    private var finished = false

    init(
        intervalSeconds: TimeInterval = 0.5,
        startsAutomatically: Bool = true,
        clock: @escaping @Sendable () -> UInt64 = { MonotonicClock.now() },
        wallClock: @escaping @Sendable () -> Date = { Date() }
    ) {
        precondition(intervalSeconds.isFinite && intervalSeconds > 0)
        let pair = AsyncStream<RunPayloadFlowSample>.makeStream(
            bufferingPolicy: .bufferingNewest(120))
        stream = pair.stream
        continuation = pair.continuation
        self.intervalSeconds = intervalSeconds
        self.clock = clock
        self.wallClock = wallClock
        lastTick = clock()
        if startsAutomatically { start() }
    }

    func start() {
        lock.lock()
        guard !finished, timer == nil else {
            lock.unlock()
            return
        }
        let source = DispatchSource.makeTimerSource(
            queue: DispatchQueue(label: "dev.minirun.payload-flow", qos: .utility))
        timer = source
        lock.unlock()

        source.schedule(
            deadline: .now() + intervalSeconds,
            repeating: intervalSeconds,
            leeway: .milliseconds(25))
        source.setEventHandler { [weak self] in self?.emitWindow(final: false) }
        source.resume()
    }

    @inline(__always)
    func recordDeterministic(_ bytes: UInt64) {
        record(bytes, bucket: .deterministic)
    }

    @inline(__always)
    func recordExpert(_ bytes: UInt64) {
        record(bytes, bucket: .expert)
    }

    private func record(_ bytes: UInt64, bucket: Bucket) {
        lock.lock()
        guard !finished else {
            lock.unlock()
            return
        }
        switch bucket {
        case .deterministic:
            deterministicWindow.add(bytes)
            deterministicTotal.add(bytes)
        case .expert:
            expertWindow.add(bytes)
            expertTotal.add(bytes)
        }
        lock.unlock()
    }

    var snapshot: Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            deterministicBytes: deterministicTotal.value,
            expertBytes: expertTotal.value,
            didOverflow: deterministicTotal.didOverflow || expertTotal.didOverflow)
    }

    /// Flushes the final partial window and closes the stream. Idempotent.
    func finish() {
        emitWindow(final: true)
    }

    /// Internal deterministic seam for tests; production calls it from the
    /// timer and from terminal teardown only.
    func emitWindow(final: Bool) {
        let tick = clock()
        let date = wallClock()

        lock.lock()
        guard !finished else {
            lock.unlock()
            return
        }
        let elapsed = MonotonicClock.seconds(from: lastTick, to: tick)
        let hasBytes = deterministicWindow.value > 0 || expertWindow.value > 0
        let overflow = deterministicWindow.didOverflow || expertWindow.didOverflow
            || deterministicTotal.didOverflow || expertTotal.didOverflow
        let shouldEmit = elapsed.isFinite && elapsed > 0 && (!final || hasBytes || overflow)
        let sample: RunPayloadFlowSample?
        if shouldEmit {
            sample = RunPayloadFlowSample(
                sequence: sequence, at: date, intervalSeconds: elapsed,
                deterministicBytes: deterministicWindow.value,
                expertBytes: expertWindow.value, didOverflow: overflow)
            sequence = sequence == .max ? .max : sequence + 1
        } else {
            sample = nil
        }
        deterministicWindow = UInt64Accounting.SaturatingSum()
        expertWindow = UInt64Accounting.SaturatingSum()
        lastTick = tick
        let source = final ? timer : nil
        if final {
            finished = true
            timer = nil
        }
        lock.unlock()

        if let sample { continuation.yield(sample) }
        if final {
            source?.cancel()
            continuation.finish()
        }
    }
}
