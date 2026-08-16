import Foundation
import StorageCore

/// A bounded, least-recently-used set of open ``TileReader``s.
///
/// ## The hazard this exists to remove (M9B, 2026-08-10)
///
/// `ExpertStreamEngine` opened a reader per container and kept it forever. At
/// M8's two layers that is 8 readers and nobody noticed. At M9's 93 layers the
/// same code reaches **372 readers** — 93 × 3 expert containers plus 93
/// deterministic blobs — and a reader is not a file descriptor, it is a
/// descriptor *plus a worker pool*: `queueDepth + 1` threads each, so
/// **~1,488 threads** at the default queue depth of 3. That is the scaling
/// hazard M9A catalogued and did not address.
///
/// The fix is not a bigger pool, it is a bounded one. Readers are cheap to
/// reopen (an `open(2)` and a few thread spawns) and the decode loop only ever
/// touches one layer at a time, so a cache sized for the layers actually in
/// flight costs nothing and bounds both threads and descriptors by a stated
/// number rather than by the model's depth.
///
/// ## Why eviction is safe
///
/// A reader may only be closed while it holds no lease. The engine acquires and
/// releases within one layer before moving to the next, so the readers with
/// outstanding work are always the most recently touched. `limit` is floored at
/// ``minimumLimit`` — one layer's worth of concurrently-active readers — which
/// is what makes "least recently used" and "not currently in use" the same set.
/// Evicting below that floor could close a reader mid-lease, so the floor is
/// enforced here rather than trusted to the caller.
///
/// ## Why it keeps books on the dead
///
/// The run's I/O accounting is summed off live readers. If eviction simply
/// dropped them, every byte an evicted reader moved would vanish from the
/// metrics, and a 93-layer run would under-report its own traffic by however
/// much it evicted — silently, and in the direction that flatters it. So a
/// reader's statistics are absorbed into ``retired`` before it is shut down,
/// and the reporting path adds the two together.
struct RetiredReaderStatistics {
    var bytesRead: UInt64 = 0
    var byteCountOverflowed = false
    var readCount = 0
    var shortReadCount = 0
    var errorCount = 0
    var firstError: String?
    var slotWaitSeconds: Double = 0
    var latencySamples: [UInt64] = []

    mutating func absorb(_ reader: TileReader) {
        let statistics = reader.statistics
        var total = UInt64Accounting.SaturatingSum(
            value: bytesRead,
            didOverflow: byteCountOverflowed || statistics.byteCountOverflowed)
        total.add(statistics.bytesRead)
        bytesRead = total.value
        byteCountOverflowed = total.didOverflow
        readCount += statistics.readCount
        shortReadCount += statistics.shortReadCount
        errorCount += statistics.errorCount
        if firstError == nil { firstError = statistics.firstError }
        slotWaitSeconds += statistics.slotWaitSeconds
        latencySamples.append(contentsOf: reader.latencySamples)
    }
}

final class BoundedReaderCache {

    /// One layer's concurrently-active readers. The expert path touches all
    /// three projections while assembling a dispatch; the deterministic path
    /// touches one blob. Three covers both, and is the smallest limit at which
    /// the least-recently-used entry is provably idle.
    static let minimumLimit = 3

    let limit: Int
    private var readers: [String: TileReader] = [:]
    /// Least-recently-used first. Linear, which is correct at these sizes: the
    /// limit is a handful of entries, so a search is cheaper than the bookkeeping
    /// a heap would need.
    private var order: [String] = []

    private(set) var retired = RetiredReaderStatistics()
    private(set) var evictionCount = 0
    private(set) var openCount = 0

    init(limit: Int) {
        self.limit = max(Self.minimumLimit, limit)
    }

    /// The readers currently open. Reporting folds these together with
    /// ``retired``; neither is the whole picture on its own.
    var live: some Collection<TileReader> { readers.values }

    var liveCount: Int { readers.count }

    /// The reader for `key`, opening one if it is not already cached and
    /// evicting the least recently used to make room.
    func reader(for key: String, make: () throws -> TileReader) throws -> TileReader {
        if let existing = readers[key] {
            touch(key)
            return existing
        }
        while readers.count >= limit, let victim = order.first {
            evict(victim)
        }
        let reader = try make()
        readers[key] = reader
        order.append(key)
        openCount += 1
        return reader
    }

    private func touch(_ key: String) {
        guard let index = order.firstIndex(of: key) else { return }
        order.remove(at: index)
        order.append(key)
    }

    private func evict(_ key: String) {
        guard let reader = readers.removeValue(forKey: key) else { return }
        if let index = order.firstIndex(of: key) { order.remove(at: index) }
        retired.absorb(reader)
        reader.shutdown()
        evictionCount += 1
    }

    func shutdownAll() {
        for key in order {
            if let reader = readers[key] {
                retired.absorb(reader)
                reader.shutdown()
            }
        }
        readers.removeAll()
        order.removeAll()
    }
}
