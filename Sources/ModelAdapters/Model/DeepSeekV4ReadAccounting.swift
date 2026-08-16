import Foundation
import StorageCore

/// One cumulative, run-scoped census of V4 execution-payload reads.
///
/// Configuration, tokenizer, manifests, and container headers belong to
/// admission. This counter starts at the first weight payload consumed by
/// generation, matching the K3 runner's payload-byte meaning. Deterministic
/// matrices/tables and routed-expert tiles are disjoint buckets; a sticky
/// overflow makes the whole snapshot unavailable rather than wrapping into a
/// plausible smaller number.
public struct DeepSeekV4ReadAccountingSnapshot: Sendable, Equatable {
    public let deterministicBytesRead: UInt64
    public let expertBytesRead: UInt64
    public let totalBytesRead: UInt64

    /// Deterministic bytes a pass did **not** read because the memory dial's
    /// pinned tier already held them.
    ///
    /// Deliberately outside ``totalBytesRead`` and outside ``isBalanced``. The
    /// read identity is a statement about bytes that crossed the link, and a
    /// served byte did not cross it — folding this in would make the total
    /// grow while the drive did less work, which is the one thing this counter
    /// exists to prevent. It is reported *beside* the identity so
    /// ``deterministicBytesRead`` can fall honestly and a reader can still see
    /// what the fall was bought with.
    public let pinnedServedBytes: UInt64
    public let didOverflow: Bool

    public init(
        deterministicBytesRead: UInt64,
        expertBytesRead: UInt64,
        pinnedServedBytes: UInt64 = 0,
        didOverflow: Bool = false
    ) {
        self.deterministicBytesRead = deterministicBytesRead
        self.expertBytesRead = expertBytesRead
        self.pinnedServedBytes = pinnedServedBytes
        let total = UInt64Accounting.saturatingSum([
            deterministicBytesRead, expertBytesRead,
        ])
        self.totalBytesRead = total.value
        self.didOverflow = didOverflow || total.didOverflow
    }

    public var isBalanced: Bool {
        guard !didOverflow else { return false }
        return UInt64Accounting.checkedSum([
            deterministicBytesRead, expertBytesRead,
        ]) == totalBytesRead
    }

    /// What one pass would have read with nothing pinned: the bytes it did read
    /// plus the bytes the tier served it. Stated so a pinned run and an unpinned
    /// run of the same prompt can be compared on the quantity that did not
    /// change.
    public var deterministicBytesWithoutPins: UInt64 {
        UInt64Accounting.saturatingSum([
            deterministicBytesRead, pinnedServedBytes,
        ]).value
    }
}

/// Thread-safe because expert reads complete on pager workers while the decode
/// thread samples telemetry at layer and token boundaries.
public final class DeepSeekV4ReadAccounting: @unchecked Sendable {
    private let lock = NSLock()
    private let successfulDeterministicReadObserver: (@Sendable (UInt64) -> Void)?
    private let successfulExpertReadObserver: (@Sendable (UInt64) -> Void)?
    private var deterministic = UInt64Accounting.SaturatingSum()
    private var expert = UInt64Accounting.SaturatingSum()
    private var pinnedServed = UInt64Accounting.SaturatingSum()

    public init(
        successfulDeterministicReadObserver: (@Sendable (UInt64) -> Void)? = nil,
        successfulExpertReadObserver: (@Sendable (UInt64) -> Void)? = nil
    ) {
        self.successfulDeterministicReadObserver = successfulDeterministicReadObserver
        self.successfulExpertReadObserver = successfulExpertReadObserver
    }

    public var snapshot: DeepSeekV4ReadAccountingSnapshot {
        lock.lock()
        defer { lock.unlock() }
        return DeepSeekV4ReadAccountingSnapshot(
            deterministicBytesRead: deterministic.value,
            expertBytesRead: expert.value,
            pinnedServedBytes: pinnedServed.value,
            didOverflow: deterministic.didOverflow || expert.didOverflow
                || pinnedServed.didOverflow)
    }

    func recordDeterministic(_ bytes: UInt64) {
        lock.lock()
        deterministic.add(bytes)
        lock.unlock()
        successfulDeterministicReadObserver?(bytes)
    }

    /// One pinned-tier hit: bytes a load did not have to read.
    ///
    /// No payload-flow observer fires. The flow meter reports bytes moving
    /// across the link for the progress instruments, and nothing moved.
    func recordPinnedServed(_ bytes: UInt64) {
        lock.lock()
        pinnedServed.add(bytes)
        lock.unlock()
    }

    func recordExpert(_ bytes: UInt64, didOverflow: Bool) {
        lock.lock()
        expert.add(bytes)
        if didOverflow {
            expert = UInt64Accounting.SaturatingSum(
                value: expert.value, didOverflow: true)
        }
        lock.unlock()
        if bytes > 0 { successfulExpertReadObserver?(bytes) }
    }
}
