import Darwin
import Foundation
import MLX
import StorageCore

/// Staging layer *i+1*'s deterministic bytes while layer *i* computes.
///
/// ## What this is for
///
/// The decode loop is strictly serial per layer: read layer *i*'s deterministic
/// blob, widen it to float32, compute, and only then start reading layer *i+1*.
/// Measured that way a Mac token is ~113 s — roughly 59 s of I/O and 54 s of
/// compute, added rather than overlapped — and spec §12.2's stated pipeline goal
/// is the other shape:
///
/// ```text
/// I/O fills slot N+1
/// GPU consumes slot N
/// ```
///
/// So while layer *i* is resident and computing, a background thread reads layer
/// *i+1*'s bytes into its own buffers. Wall time then tends to
/// `max(I/O, compute)` instead of their sum. Nothing else changes: the bytes are
/// the same bytes, read by the same `pread` loop over the same unit table, and
/// the **widening to float32 stays in the consume path**, unchanged, so every
/// computed value is bit-identical with the feature on or off. Only the moment
/// the bytes arrive moves.
///
/// ## What it costs, and why the budget decides per pair
///
/// Staged bytes are resident bytes. While layer *i* is computing, the process
/// holds layer *i*'s **widened float32** weights *and* layer *i+1*'s **stored**
/// (BF16) bytes at once, and spec §5.3 requires that to be a stated budget
/// rather than an emergent footprint. At flagship shape one pair does not fit:
/// layer 0 widens to 4,682,514,432 B, the next layer stores 1,267,810,304 B, and
/// 5.95 GB is over the phone's declared 5.8 GB. That pair therefore runs serial
/// and the run says so — a skip is normal operation, not an error
/// (``DeterministicReadAheadSchedule``).
///
/// The consume phase needs no extra bound. Each staged tensor's buffer is handed
/// to MLX and freed as its float32 copy is made, so the total moves from
/// `stored` to `2 × stored` monotonically and peaks at exactly the widened
/// layer — the same peak the serial path has. The pair term above is the only
/// new one.
public enum DeterministicReadAheadSchedule {

    /// What one layer's deterministic stream costs, in both forms it exists in.
    public struct LayerBytes: Sendable, Equatable, Codable {
        /// Bytes as written: BF16 for nearly everything, F32 for a few vectors.
        /// This is what staging allocates and reads.
        public let storedBytes: UInt64
        /// Bytes after ``ArtifactWeightSource`` widens every tensor to float32.
        /// This is what the layer occupies while it is the resident one.
        public let residentBytes: UInt64

        public init(storedBytes: UInt64, residentBytes: UInt64) {
            self.storedBytes = storedBytes
            self.residentBytes = residentBytes
        }
    }

    /// Whether one `(resident, staged)` pair fits, and the arithmetic that said
    /// so. Recorded rather than recomputed, so a skip in a run's JSON can be
    /// read back without the layer table.
    public struct Decision: Sendable, Equatable, Codable {
        public let residentLayer: Int
        public let stagedLayer: Int
        public let residentBytes: UInt64
        public let stagedBytes: UInt64
        /// Everything else the budget must also cover — expert pool, MLX cache,
        /// working set. Stated by the caller; zero means "this budget describes
        /// the two deterministic terms alone".
        public let reserveBytes: UInt64
        public let budgetBytes: UInt64

        /// `resident + staged + reserve`, saturating rather than trapping: a
        /// nonsensical layer table must produce a refusal, not a crash.
        public var requiredBytes: UInt64 {
            required.value
        }

        public var isAllowed: Bool {
            let sum = residentBytes.addingReportingOverflow(stagedBytes)
            guard !sum.overflow else { return false }
            let total = sum.partialValue.addingReportingOverflow(reserveBytes)
            guard !total.overflow else { return false }
            return total.partialValue <= budgetBytes
        }

        /// Budget minus requirement. Negative on a skip, which is the number
        /// worth reading: it says how far over the pair was.
        public var headroomBytes: Int64 {
            let required = required
            guard !required.overflow else { return .min }
            if required.value > budgetBytes {
                let deficit = required.value - budgetBytes
                guard deficit <= UInt64(Int64.max) else { return .min }
                return -Int64(deficit)
            }
            return Int64(clamping: budgetBytes - required.value)
        }

        private var required: (value: UInt64, overflow: Bool) {
            let first = residentBytes.addingReportingOverflow(stagedBytes)
            guard !first.overflow else { return (.max, true) }
            let second = first.partialValue.addingReportingOverflow(reserveBytes)
            guard !second.overflow else { return (.max, true) }
            return (second.partialValue, false)
        }
    }

    /// Does staging `next` while `resident` is resident fit inside `budget`?
    ///
    /// Pure, and deliberately so: the decision is the one thing in this feature
    /// that can be checked against the real checkpoint's numbers without a
    /// checkpoint, a drive, or a phone.
    public static func decide(
        residentLayer: Int,
        resident: LayerBytes,
        stagedLayer: Int,
        next: LayerBytes,
        budgetBytes: UInt64,
        reserveBytes: UInt64 = 0
    ) -> Decision {
        Decision(
            residentLayer: residentLayer,
            stagedLayer: stagedLayer,
            residentBytes: resident.residentBytes,
            stagedBytes: next.storedBytes,
            reserveBytes: reserveBytes,
            budgetBytes: budgetBytes)
    }

    /// Every consecutive pair of a layer table, in order. `layers.count - 1`
    /// decisions; an empty or single-layer table has none.
    public static func plan(
        _ layers: [LayerBytes], budgetBytes: UInt64, reserveBytes: UInt64 = 0
    ) -> [Decision] {
        guard layers.count > 1 else { return [] }
        return (0..<(layers.count - 1)).map { index in
            decide(
                residentLayer: index, resident: layers[index],
                stagedLayer: index + 1, next: layers[index + 1],
                budgetBytes: budgetBytes, reserveBytes: reserveBytes)
        }
    }
}

// MARK: - Metrics

/// What the read-ahead did, for the run's JSON (AGENTS.md: a parameter that
/// changes a measured result is reported next to the result).
///
/// The counters distinguish three things that look alike in a wall time and are
/// not alike at all: a **skip** (the budget refused the pair — normal), a
/// **miss** (nothing was staged for the layer that was asked for, e.g. a caller
/// that jumps layers) and a **tensor miss** (the layer was staged but a tensor
/// the consume path wanted was not in the plan). All three fall back to the
/// serial read and none of them changes a value.
public struct DeterministicReadAheadMetrics: Codable, Sendable, Equatable {

    /// 0 or 1. Zero is the serial default and every field below is then zero.
    public var depth: Int = 0
    /// The budget the schedule was evaluated against (spec §5.3).
    public var declaredBudgetBytes: UInt64 = 0
    /// Held back from the budget for everything that is not these two layers.
    public var reserveBytes: UInt64 = 0

    public var stagedLayerCount: Int = 0
    public var skippedLayerCount: Int = 0
    /// Layers the stager was **not** asked to stage because the memory dial
    /// pinned them: there is nothing to stage for a layer that is already
    /// resident. Deliberately a different counter from ``skippedLayerCount``,
    /// which is a pair the *budget* refused — the two look identical in a wall
    /// time and mean opposite things about whether the plan is working.
    public var pinnedSkippedLayerCount: Int = 0
    public var missedLayerCount: Int = 0
    public var missedTensorCount: Int = 0
    /// Staged and never asked for. Non-zero means the plan is a strict superset
    /// of what the consume path reads, which would mean the two arms move
    /// different bytes — so it is reported rather than assumed to be zero.
    public var unusedStagedTensorCount: Int = 0

    /// Bytes the background thread read, over the whole run.
    public var stagedBytes: UInt64 = 0
    /// Largest single layer staged. This is the term the budget bounds.
    public var peakStagedBytes: UInt64 = 0
    /// Staged bytes still allocated right now. Zero at rest, on every path.
    public var outstandingStagedBytes: UInt64 = 0
    /// Staged bytes handed to MLX and widened.
    public var materialisedStagedBytes: UInt64 = 0
    /// Staged bytes freed without being used: a wrong-layer set, an unused
    /// plan entry, or a partially read layer whose staging failed.
    public var discardedStagedBytes: UInt64 = 0
    /// Sticky. Saturated byte totals cannot prove a lifecycle identity.
    public var accountingOverflowed: Bool = false

    /// Wall time the decode loop spent waiting for staged bytes that had not
    /// arrived yet. The part of the read that failed to hide.
    public var prefetchWaitSeconds: Double = 0
    /// Wall time the background thread spent reading, in total.
    public var stageReadSeconds: Double = 0
    /// Wall time the decode loop spent reading deterministic bytes itself —
    /// every layer at depth 0, and the skipped/missed ones at depth 1.
    public var serialReadSeconds: Double = 0
    /// Per layer, summed over passes. Index is the layer number.
    public var waitSecondsByLayer: [Double] = []

    public var errorCount: Int = 0
    public var firstError: String?

    /// Fraction of the background read that landed behind compute:
    /// `1 - wait / stageRead`. 1.0 means the reads were entirely hidden, 0
    /// means the loop waited for all of them and nothing was gained.
    public var overlapEfficiency: Double {
        guard stageReadSeconds > 0 else { return 0 }
        return max(0, 1 - prefetchWaitSeconds / stageReadSeconds)
    }

    /// Staged bytes that reached a value, as a fraction of all staged bytes.
    /// Anything below 1 with a completed run means bytes were read and thrown
    /// away, which is the wasteful failure mode this feature could have.
    public var stagedByteUtilisation: Double {
        guard stagedBytes > 0 else { return 0 }
        return Double(materialisedStagedBytes) / Double(stagedBytes)
    }

    /// Every staged byte is on exactly one path: widened, discarded, or still
    /// held. The accounting identity the lifecycle tests assert.
    public var isAccountingBalanced: Bool {
        guard !accountingOverflowed,
            let accounted = UInt64Accounting.checkedSum([
                materialisedStagedBytes, discardedStagedBytes, outstandingStagedBytes,
            ])
        else { return false }
        return accounted == stagedBytes
    }

    enum CodingKeys: String, CodingKey {
        case depth, declaredBudgetBytes, reserveBytes
        case stagedLayerCount, skippedLayerCount, pinnedSkippedLayerCount
        case missedLayerCount, missedTensorCount
        case unusedStagedTensorCount
        case stagedBytes, peakStagedBytes, outstandingStagedBytes
        case materialisedStagedBytes, discardedStagedBytes
        case accountingOverflowed
        case prefetchWaitSeconds, stageReadSeconds, serialReadSeconds, waitSecondsByLayer
        case errorCount, firstError
        case overlapEfficiency, stagedByteUtilisation
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        depth = try container.decode(Int.self, forKey: .depth)
        declaredBudgetBytes = try container.decode(UInt64.self, forKey: .declaredBudgetBytes)
        reserveBytes = try container.decode(UInt64.self, forKey: .reserveBytes)
        stagedLayerCount = try container.decode(Int.self, forKey: .stagedLayerCount)
        skippedLayerCount = try container.decode(Int.self, forKey: .skippedLayerCount)
        // Absent from every record written before the pinned tier existed, and
        // zero is what those runs measured — so it decodes rather than failing.
        pinnedSkippedLayerCount = try container.decodeIfPresent(
            Int.self, forKey: .pinnedSkippedLayerCount) ?? 0
        missedLayerCount = try container.decode(Int.self, forKey: .missedLayerCount)
        missedTensorCount = try container.decode(Int.self, forKey: .missedTensorCount)
        unusedStagedTensorCount = try container.decode(
            Int.self, forKey: .unusedStagedTensorCount)
        stagedBytes = try container.decode(UInt64.self, forKey: .stagedBytes)
        peakStagedBytes = try container.decode(UInt64.self, forKey: .peakStagedBytes)
        outstandingStagedBytes = try container.decode(
            UInt64.self, forKey: .outstandingStagedBytes)
        materialisedStagedBytes = try container.decode(
            UInt64.self, forKey: .materialisedStagedBytes)
        discardedStagedBytes = try container.decode(UInt64.self, forKey: .discardedStagedBytes)
        accountingOverflowed = try container.decodeIfPresent(
            Bool.self, forKey: .accountingOverflowed) ?? false
        prefetchWaitSeconds = try container.decode(Double.self, forKey: .prefetchWaitSeconds)
        stageReadSeconds = try container.decode(Double.self, forKey: .stageReadSeconds)
        serialReadSeconds = try container.decode(Double.self, forKey: .serialReadSeconds)
        waitSecondsByLayer = try container.decode([Double].self, forKey: .waitSecondsByLayer)
        errorCount = try container.decode(Int.self, forKey: .errorCount)
        firstError = try container.decodeIfPresent(String.self, forKey: .firstError)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(depth, forKey: .depth)
        try container.encode(declaredBudgetBytes, forKey: .declaredBudgetBytes)
        try container.encode(reserveBytes, forKey: .reserveBytes)
        try container.encode(stagedLayerCount, forKey: .stagedLayerCount)
        try container.encode(skippedLayerCount, forKey: .skippedLayerCount)
        try container.encode(pinnedSkippedLayerCount, forKey: .pinnedSkippedLayerCount)
        try container.encode(missedLayerCount, forKey: .missedLayerCount)
        try container.encode(missedTensorCount, forKey: .missedTensorCount)
        try container.encode(unusedStagedTensorCount, forKey: .unusedStagedTensorCount)
        try container.encode(stagedBytes, forKey: .stagedBytes)
        try container.encode(peakStagedBytes, forKey: .peakStagedBytes)
        try container.encode(outstandingStagedBytes, forKey: .outstandingStagedBytes)
        try container.encode(materialisedStagedBytes, forKey: .materialisedStagedBytes)
        try container.encode(discardedStagedBytes, forKey: .discardedStagedBytes)
        try container.encode(accountingOverflowed, forKey: .accountingOverflowed)
        try container.encode(prefetchWaitSeconds, forKey: .prefetchWaitSeconds)
        try container.encode(stageReadSeconds, forKey: .stageReadSeconds)
        try container.encode(serialReadSeconds, forKey: .serialReadSeconds)
        try container.encode(waitSecondsByLayer, forKey: .waitSecondsByLayer)
        try container.encode(errorCount, forKey: .errorCount)
        try container.encodeIfPresent(firstError, forKey: .firstError)
        // Derived, written so a reader of the JSON does not have to redo the
        // division — and never read back, so the two can never disagree.
        try container.encode(overlapEfficiency, forKey: .overlapEfficiency)
        try container.encode(stagedByteUtilisation, forKey: .stagedByteUtilisation)
    }
}

/// How one layer's weights were obtained, for the per-layer trace.
public enum DeterministicReadAheadOutcome: String, Codable, Sendable {
    /// Read serially by the decode loop. The only outcome at depth 0.
    case serial
    /// Consumed from bytes a background read had already staged.
    case staged
    /// The budget refused the pair, so the layer was read serially.
    case skipped
    /// Served from the memory dial's pinned tier: the bytes were already
    /// resident and nothing was read for this layer on this pass.
    case pinned
    /// Read-ahead was on and nothing was staged for this layer.
    case missed
}

// MARK: - Staged bytes

/// One tensor's bytes, read but **not** widened.
///
/// The split matters: everything before this point is a `pread` and can happen
/// on any thread, and everything after it is MLX arithmetic that must stay
/// exactly where it was. ``materialise()`` is the tail of what
/// ``DeterministicBlob/tensor(_:)`` always did, moved into a method so that the
/// staged path and the serial path run the *same* instructions over the same
/// bytes rather than two copies of them.
final class RawTensor: @unchecked Sendable {

    let name: String
    let rows: Int
    let cols: Int
    let dtype: DType
    let byteCount: Int
    private var buffer: UnsafeMutableRawPointer?

    init(
        name: String, rows: Int, cols: Int, dtype: DType, byteCount: Int,
        buffer: UnsafeMutableRawPointer
    ) {
        self.name = name
        self.rows = rows
        self.cols = cols
        self.dtype = dtype
        self.byteCount = byteCount
        self.buffer = buffer
    }

    /// Frees the bytes if nobody took them. Every path — success, error, an
    /// abandoned loop — ends here.
    deinit { discard() }

    /// Bytes still held. Zero once the buffer has been widened or discarded.
    var heldBytes: Int { buffer == nil ? 0 : byteCount }

    /// The `[rows, cols]` float32 array these bytes stand for, **consuming** the
    /// bytes.
    ///
    /// Widening BF16 to float32 is exact and it *copies*, so the array returned
    /// does not point into the staged buffer and the buffer dies with the
    /// temporary. That is what makes `releaseLayer` the only thing keeping the
    /// peak down.
    ///
    /// This is one of the two ownership modes of ``widened(retainingBytes:)``;
    /// ``materialiseKeepingBytes()`` is the other. See that method for why the
    /// pair is one function with a flag rather than two similar ones.
    func materialise() throws -> MLXArray {
        try widened(retainingBytes: false)
    }

    /// The same `[rows, cols]` float32 array, **keeping** the bytes.
    ///
    /// The pinned tier holds a layer's stored bytes for the life of the run and
    /// widens them once per use, so it needs the widening without the transfer
    /// of ownership: the finalizer is a no-op, `buffer` is not nilled, and this
    /// object stays the owner. `asType` copies when it widens, so the array
    /// returned is independent of the retained buffer and the next call over the
    /// same bytes produces a bit-identical array.
    ///
    /// The two modes are one function on purpose. They are not two ways of
    /// making an array; they are one widening under two ownership rules, and any
    /// drift between their arithmetic would be a value change disguised as a
    /// memory-management change — exactly what the read-ahead's bit-identity
    /// argument was built to exclude.
    func materialiseKeepingBytes() throws -> MLXArray {
        try widened(retainingBytes: true)
    }

    /// The one widening, in both ownership modes.
    ///
    /// `MLXArray(rawPointer:) → asType(.float32) → eval()`, once, so the pinned
    /// path and the staged path cannot compute different values.
    ///
    /// The one asymmetry is stated rather than hidden: `asType` returns *self*
    /// when the type already matches (mlx-swift `MLXArray.asType`), so for the
    /// handful of tensors stored F32 the "widening" is not a copy and the array
    /// would alias the retained buffer. Retaining mode therefore hands MLX a
    /// duplicate of those bytes and keeps the pinned original — the same
    /// instructions over the same bytes, so the values are identical, and the
    /// caller's buffer is nobody else's to free. BF16 — every deterministic
    /// tensor of the flagship — takes the copy `asType` already makes.
    private func widened(retainingBytes: Bool) throws -> MLXArray {
        guard let buffer else {
            throw TinyK3Error.configuration(
                "the staged bytes for '\(name)' were already consumed")
        }
        let handed: UnsafeMutableRawPointer
        let finalizer: () -> Void
        if retainingBytes {
            if dtype == .float32 {
                let copy = UnsafeMutableRawPointer.allocate(
                    byteCount: max(byteCount, 1), alignment: 16)
                copy.copyMemory(from: buffer, byteCount: byteCount)
                handed = copy
                finalizer = { copy.deallocate() }
            } else {
                handed = buffer
                finalizer = {}
            }
        } else {
            self.buffer = nil
            handed = buffer
            finalizer = { buffer.deallocate() }
        }
        let raw = MLXArray(
            rawPointer: handed, [rows, cols], dtype: dtype, finalizer: finalizer)
        let result = raw.asType(.float32)
        result.eval()
        return result
    }

    func discard() {
        buffer?.deallocate()
        buffer = nil
    }
}

/// One layer's deterministic tensors, read but not widened.
final class StagedLayer: @unchecked Sendable {

    let layer: Int
    /// Bytes the reader charged to this layer, in the accounting
    /// ``ArtifactWeightSource/deterministicBytesRead`` is stated in.
    let bytesRead: UInt64
    /// Sticky companion to `bytesRead`; a saturated read total is not evidence.
    let byteAccountingOverflowed: Bool
    private var tensors: [String: RawTensor]

    init(
        layer: Int, tensors: [String: RawTensor], bytesRead: UInt64,
        byteAccountingOverflowed: Bool = false
    ) {
        self.layer = layer
        self.tensors = tensors
        self.bytesRead = bytesRead
        self.byteAccountingOverflowed = byteAccountingOverflowed
    }

    /// Bytes still allocated.
    var remainingBytes: UInt64 {
        UInt64Accounting.saturatingSum(tensors.values.lazy.map { UInt64($0.heldBytes) }).value
    }

    var remainingCount: Int { tensors.count }

    /// Hand over one tensor's bytes, removing it from the set.
    func take(_ name: String) -> RawTensor? { tensors.removeValue(forKey: name) }

    /// Free whatever is left, and say how much that was.
    @discardableResult
    func discardAll() -> UInt64 {
        let bytes = remainingBytes
        for tensor in tensors.values { tensor.discard() }
        tensors.removeAll()
        return bytes
    }
}

// MARK: - The background reader

/// A one-layer-deep background reader for the deterministic stream.
///
/// Depth is exactly one — a double buffer — because that is what the pipeline
/// goal asks for and what the budget can state: at most one layer is staged
/// while at most one is resident. A second staged layer would be a third term in
/// a budget that already has ~10% headroom on the phone, and no measurement asks
/// for it.
///
/// ## Threading
///
/// One `Thread` and one `NSCondition`, the idiom `TileReader` uses, for the same
/// reason: the work is a blocking `pread` and the alternative is a queue that
/// hides where the blocking happens. The thread holds a strong reference to this
/// object while it runs, so ``shutdown()`` is not optional.
///
/// The blob a request names is touched **only** by this thread until the
/// matching ``claim(layer:)`` returns, and the consume path never asks for the
/// layer being staged. One owner at a time, enforced by the protocol rather than
/// by a lock around the file descriptor.
final class DeterministicStager: @unchecked Sendable {

    /// What a consume-side claim found.
    struct Claim {
        let staged: StagedLayer?
        /// Bytes freed because a set was staged for a different layer.
        let droppedBytes: UInt64
        let waitSeconds: Double
    }

    private struct Request {
        let layer: Int
        let blob: DeterministicBlob
        let tensors: [String]
    }

    private let monitor = NSCondition()
    private var queued: Request?
    private var activeLayer: Int?
    private var completed: StagedLayer?
    private var failure: (layer: Int, error: Error)?
    private var finished = false
    private var threadLive = false

    private(set) var stagedBytes: UInt64 = 0
    private(set) var peakStagedBytes: UInt64 = 0
    private(set) var discardedBytes: UInt64 = 0
    private(set) var stageReadSeconds: Double = 0
    private(set) var errorCount = 0
    private(set) var firstError: String?
    private(set) var accountingOverflowed = false

    init() {
        monitor.lock()
        threadLive = true
        monitor.unlock()
        let thread = Thread { self.loop() }
        thread.name = "minirun.deterministic.readahead"
        thread.qualityOfService = .userInitiated
        thread.stackSize = 512 << 10
        thread.start()
    }

    deinit { shutdown() }

    // MARK: Request side

    /// Queue one layer. Depth is one, so anything already queued or completed
    /// for another layer is dropped first — it cannot be waited for any more.
    func stage(layer: Int, blob: DeterministicBlob, tensors: [String]) {
        monitor.lock()
        defer { monitor.unlock() }
        guard !finished else { return }
        if let stale = completed {
            discardedBytes = adding(stale.discardAll(), to: discardedBytes)
            completed = nil
        }
        queued = Request(layer: layer, blob: blob, tensors: tensors)
        monitor.broadcast()
    }

    /// Take the staged bytes for `layer`, waiting for an in-flight read of it.
    ///
    /// A read that failed is rethrown here, exactly as the serial path would
    /// have thrown it — the error is not swallowed and there is no silent
    /// fallback that would turn a drive disconnection into a slow run.
    func claim(layer: Int) throws -> Claim {
        monitor.lock()
        let started = MonotonicClock.now()
        var waited = false
        while !finished && (activeLayer == layer || queued?.layer == layer) {
            waited = true
            monitor.wait()
        }
        let waitSeconds = waited ? MonotonicClock.seconds(since: started) : 0

        if let failure, failure.layer == layer {
            self.failure = nil
            monitor.unlock()
            throw failure.error
        }
        var dropped: UInt64 = 0
        var staged: StagedLayer?
        if let ready = completed {
            completed = nil
            if ready.layer == layer {
                staged = ready
            } else {
                dropped = ready.discardAll()
                discardedBytes = adding(dropped, to: discardedBytes)
            }
        }
        monitor.unlock()
        return Claim(staged: staged, droppedBytes: dropped, waitSeconds: waitSeconds)
    }

    /// Bytes held by a completed-but-unclaimed set. The consume side tracks the
    /// rest; together they are the read-ahead's outstanding allocation.
    var outstandingBytes: UInt64 {
        monitor.lock()
        defer { monitor.unlock() }
        return completed?.remainingBytes ?? 0
    }

    /// Stop the thread and free anything staged. Idempotent, and mandatory:
    /// the thread holds a strong reference to this object.
    func shutdown() {
        monitor.lock()
        if !threadLive && queued == nil && completed == nil {
            monitor.unlock()
            return
        }
        finished = true
        queued = nil
        monitor.broadcast()
        while threadLive {
            monitor.wait()
        }
        if let staged = completed {
            discardedBytes = adding(staged.discardAll(), to: discardedBytes)
            completed = nil
        }
        failure = nil
        monitor.unlock()
    }

    // MARK: The thread

    private func loop() {
        while true {
            monitor.lock()
            while !finished && queued == nil {
                monitor.wait()
            }
            if finished {
                threadLive = false
                monitor.broadcast()
                monitor.unlock()
                return
            }
            let request = queued!
            queued = nil
            activeLayer = request.layer
            monitor.unlock()

            let started = MonotonicClock.now()
            var tensors: [String: RawTensor] = [:]
            var readBytes: UInt64 = 0
            var thrown: Error?
            // The blob is this thread's alone until the claim returns, so its
            // own byte accounting is safe to reset and read here.
            request.blob.resetAccounting()
            for name in request.tensors {
                do {
                    let tensor = try request.blob.rawTensor(name)
                    tensors[name] = tensor
                } catch {
                    thrown = error
                    break
                }
            }
            readBytes = request.blob.bytesRead
            let readByteAccountingOverflowed = request.blob.byteAccountingOverflowed
            let elapsed = MonotonicClock.seconds(since: started)

            monitor.lock()
            if readByteAccountingOverflowed { accountingOverflowed = true }
            stageReadSeconds += elapsed
            let stagedSum = UInt64Accounting.saturatingSum(
                tensors.values.map { UInt64($0.byteCount) })
            if stagedSum.didOverflow { accountingOverflowed = true }
            let staged = stagedSum.value
            stagedBytes = adding(staged, to: stagedBytes)
            peakStagedBytes = max(peakStagedBytes, staged)
            activeLayer = nil
            if let thrown {
                // Partial bytes never reach a consumer (spec §12.4). They are
                // freed here and the error is held for the claim.
                for tensor in tensors.values { tensor.discard() }
                discardedBytes = adding(staged, to: discardedBytes)
                errorCount += 1
                if firstError == nil { firstError = "\(thrown)" }
                failure = (request.layer, thrown)
            } else if finished {
                for tensor in tensors.values { tensor.discard() }
                discardedBytes = adding(staged, to: discardedBytes)
            } else {
                // A set nobody claimed — the consumer jumped past its layer
                // while this read was already in flight. Freed here and
                // counted, rather than left to a destructor the accounting
                // cannot see.
                if let stale = completed {
                    discardedBytes = adding(stale.discardAll(), to: discardedBytes)
                }
                completed = StagedLayer(
                    layer: request.layer, tensors: tensors, bytesRead: readBytes,
                    byteAccountingOverflowed: readByteAccountingOverflowed)
            }
            monitor.broadcast()
            monitor.unlock()
        }
    }

    private func adding(_ increment: UInt64, to value: UInt64) -> UInt64 {
        var sum = UInt64Accounting.SaturatingSum(
            value: value, didOverflow: accountingOverflowed)
        sum.add(increment)
        if sum.didOverflow { accountingOverflowed = true }
        return sum.value
    }
}
