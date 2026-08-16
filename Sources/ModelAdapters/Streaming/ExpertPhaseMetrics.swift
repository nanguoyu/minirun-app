import Foundation
import StorageCore

/// Where the time inside the routed-expert phase went.
///
/// ## Why this exists
///
/// After deterministic read-ahead landed (v0.6.17) the flagship token is 85.6 s
/// on the USB4 rig, of which the deterministic stream accounts for 0.7 s inline:
/// 99.4% of a 38.3 s read is hidden behind compute. The residual was decomposed
/// **by subtraction** in that record — ≈64 s of "compute plus expert path" and
/// ≈21 s of link contention — and subtraction is not a measurement. Inside the
/// expert phase nobody had measured how much is storage wait, how much is Metal,
/// and how much is neither.
///
/// The expert stream is 77.5 GB per token — 16 routed experts × 3 projections ×
/// 92 MoE layers × 3 positions, 13,248 reads — against 108.8 GB deterministic.
/// It cannot be staged a layer ahead: the router decides the ids inside the
/// layer that needs them, so cross-layer speculation has nothing to speculate
/// on. Whatever overlap the expert stream can have is *in-layer*, and sizing it
/// needs the split below rather than a guess.
///
/// ## What it found (`2026-08-11-expert-overlap.md`)
///
/// On the first cold flagship arm: a 60.1 s phase of an 84.2 s token, split
/// 45.4 s storage wait / 11.7 s Metal / 3.0 s remainder, with
/// ``fetchConcurrency`` at **0.98** — the expert reader was busy only while the
/// consumer was blocked, over an entire token. Opening the window
/// (``FlagshipRoutedExpertBackend/Configuration/prefetchExpertReads``) removes
/// 24.8 s of that wait and does not move the wall, because on this link the
/// wait was the port being busy rather than idle. On the cold arms it did move
/// the wall — 73.5 s against 84.2/91.1 — and on that evidence the owner turned
/// the knob on by default (2026-08-11); zero here now means a run that opted
/// back out.
///
/// ## The accounting identity
///
/// Every second the decode thread spends inside
/// ``RoutedExpertBackend/gather(_:layer:projection:expertIds:)`` lands in
/// exactly one of three buckets:
///
/// ```text
/// phaseSeconds == ioWaitSeconds + gatherComputeSeconds + unattributedSeconds
/// ```
///
/// - ``ioWaitSeconds`` is measured around `TileReader.acquire`, which is the one
///   place the consumer blocks on storage;
/// - ``gatherComputeSeconds`` is measured around the `eval` calls that force the
///   lazy graph, which is the one place Metal work is actually waited on;
/// - ``unattributedSeconds`` is the remainder — expression building, adoption,
///   `MXFP4Weights` construction, slot bookkeeping. It is *derived*, so it can
///   go negative if the two measured terms are ever taken outside the phase,
///   and ``isAccountingBalanced`` is the assertion that they are not.
///
/// The two terms that describe the *other* side of the pipe are reported beside
/// them but are deliberately **not** in the identity, because they are not the
/// consumer's wall time:
///
/// - ``fetchBusySeconds`` is `pread` busy time summed over the reader's worker
///   threads. With `queueDepth` workers it can exceed the phase wall; that is
///   the point of reporting it, since `fetchBusy > ioWait` is what concurrency
///   looks like and `fetchBusy ≈ ioWait` is what a strictly demand-driven read
///   looks like.
/// - ``backpressureStallSeconds`` is the pool applying backpressure to storage —
///   the reader waiting for a slot the consumer has not released. It is the
///   opposite direction of waiting from ``ioWaitSeconds``, and a lever that
///   raises read-ahead without raising the slot budget converts one into the
///   other rather than removing it.
public struct ExpertPhaseMetrics: Codable, Sendable, Equatable {

    /// One MoE layer's share of the phase, summed over the three projections and
    /// over every position in the pass.
    public struct LayerSplit: Codable, Sendable, Equatable {
        public var layer: Int
        public var phaseSeconds: Double
        public var ioWaitSeconds: Double
        public var gatherComputeSeconds: Double
        public var dispatchCount: Int
        public var readCount: Int
        public var bytesRequested: UInt64

        public init(
            layer: Int, phaseSeconds: Double = 0, ioWaitSeconds: Double = 0,
            gatherComputeSeconds: Double = 0, dispatchCount: Int = 0, readCount: Int = 0,
            bytesRequested: UInt64 = 0
        ) {
            self.layer = layer
            self.phaseSeconds = phaseSeconds
            self.ioWaitSeconds = ioWaitSeconds
            self.gatherComputeSeconds = gatherComputeSeconds
            self.dispatchCount = dispatchCount
            self.readCount = readCount
            self.bytesRequested = bytesRequested
        }

        public var unattributedSeconds: Double {
            phaseSeconds - ioWaitSeconds - gatherComputeSeconds
        }
    }

    // MARK: Parameters that change the measured result (AGENTS.md §17.4)

    public var granularity: String = FetchGranularity.perExpert.rawValue
    /// `pread` worker threads per reader.
    public var queueDepth: Int = 0
    /// Regions one reader may hold slots for at once. Dead weight unless
    /// something schedules regions before they are demanded — see
    /// ``prefetchExpertsAhead``.
    public var readAhead: Int = 0
    public var expertsPerDispatch: Int = 1
    /// Experts whose regions are queued before the first one is consumed, per
    /// projection. Zero is the strictly demand-driven path every record before
    /// this one measured; a positive value is the in-layer window.
    public var prefetchExpertsAhead: Int = 0
    public var poolSlotCount: Int = 0
    public var poolSlotBytes: Int = 0
    /// `poolSlotCount × poolSlotBytes` — the routed-expert working set, stated
    /// rather than emergent (spec §5.3).
    public var poolBudgetBytes: Int = 0

    // MARK: Work

    public var gatherCallCount: Int = 0
    /// `mxfp4GatherMM` calls on the routed path. Dispatches, not experts and not
    /// reads — the quantity a batched gather would move.
    public var dispatchCount: Int = 0
    public var expertReadCount: Int = 0
    /// Region lengths the consumer asked for.
    public var expertBytesRequested: UInt64 = 0
    /// Bytes the readers actually moved, live and retired. Equal to the request
    /// when nothing was prefetched and abandoned and nothing short-read.
    public var expertBytesRead: UInt64 = 0
    /// Sticky. Saturated byte totals cannot prove request/read identities.
    public var byteAccountingOverflowed: Bool = false
    public var readErrorCount: Int = 0

    // MARK: The split

    /// Wall time the decode thread spent inside the gather entry point.
    public var phaseSeconds: Double = 0
    /// Of that, blocked in `acquire` waiting for expert bytes.
    public var ioWaitSeconds: Double = 0
    /// Of that, blocked in `eval` forcing the gather graph.
    public var gatherComputeSeconds: Double = 0
    /// Sticky. Saturated nanosecond totals cannot prove the phase identity.
    public var timingAccountingOverflowed: Bool = false
    /// `pread` busy time, summed over worker threads. Not part of the identity.
    public var fetchBusySeconds: Double = 0
    /// The reader waiting for a free slot. Not part of the identity either: it
    /// happens on the dispatcher thread, and it only becomes the consumer's
    /// problem indirectly, as a later ``ioWaitSeconds``.
    public var backpressureStallSeconds: Double = 0
    public var backpressureStallCount: Int = 0

    // MARK: Occupancy

    /// Highest simultaneous `filling + ready + leased`. Against
    /// ``poolSlotCount``, this is the budget's high-water mark.
    public var poolPeakInUseSlots: Int = 0
    public var poolPeakInUseBytes: Int = 0
    public var poolFaults: [String] = []

    public var perLayer: [LayerSplit] = []

    public init() {}

    // MARK: Derived

    /// The remainder of the phase: expression building, adoption, bookkeeping.
    /// Derived, never stored, so it cannot disagree with the three terms.
    public var unattributedSeconds: Double {
        phaseSeconds - ioWaitSeconds - gatherComputeSeconds
    }

    public var ioWaitFraction: Double {
        phaseSeconds > 0 ? ioWaitSeconds / phaseSeconds : 0
    }

    public var gatherComputeFraction: Double {
        phaseSeconds > 0 ? gatherComputeSeconds / phaseSeconds : 0
    }

    /// `fetchBusy / ioWait`. One means every byte was waited for as it was read
    /// — a strictly serial demand path. Above one means reads overlapped each
    /// other or overlapped compute.
    public var fetchConcurrency: Double {
        ioWaitSeconds > 0 ? fetchBusySeconds / ioWaitSeconds : 0
    }

    /// Bytes per second the expert stream achieved against the time the consumer
    /// waited for it. The link rate the expert path is actually getting.
    public var ioWaitBytesPerSecond: Double {
        ioWaitSeconds > 0 ? Double(expertBytesRead) / ioWaitSeconds : 0
    }

    public var poolOccupancyFraction: Double {
        poolSlotCount > 0 ? Double(poolPeakInUseSlots) / Double(poolSlotCount) : 0
    }

    /// Every second inside the phase is in exactly one bucket, and the two
    /// measured buckets are inside the phase. A tolerance of a microsecond
    /// absorbs the clock reads themselves.
    public var isAccountingBalanced: Bool {
        !timingAccountingOverflowed && ioWaitSeconds >= 0 && gatherComputeSeconds >= 0
            && unattributedSeconds >= -1e-6
    }

    /// Every requested byte was read, and no byte was read that nobody asked
    /// for. False after a prefetch that was abandoned, or a short read — both of
    /// which are reported rather than hidden.
    public var isByteAccountingBalanced: Bool {
        !byteAccountingOverflowed && expertBytesRequested == expertBytesRead
    }

    /// The per-layer table sums to the totals. The layer table is what localises
    /// a stall, so it has to be the same measurement and not a second one.
    public var isLayerAccountingBalanced: Bool {
        let phase = perLayer.reduce(0) { $0 + $1.phaseSeconds }
        let io = perLayer.reduce(0) { $0 + $1.ioWaitSeconds }
        let compute = perLayer.reduce(0) { $0 + $1.gatherComputeSeconds }
        let dispatches = perLayer.reduce(0) { $0 + $1.dispatchCount }
        let reads = perLayer.reduce(0) { $0 + $1.readCount }
        guard !byteAccountingOverflowed,
            let bytes = UInt64Accounting.checkedSum(perLayer.lazy.map(\.bytesRequested))
        else { return false }
        return abs(phase - phaseSeconds) < 1e-6 && abs(io - ioWaitSeconds) < 1e-6
            && abs(compute - gatherComputeSeconds) < 1e-6 && dispatches == dispatchCount
            && reads == expertReadCount && bytes == expertBytesRequested
    }

    /// One line for a run's log. The three-way split, the pipe, and the budget.
    public var summaryLine: String {
        String(
            format: "[expert] %.1f s phase = %.1f s ioWait (%.0f%%) + %.1f s gather (%.0f%%) "
                + "+ %.1f s other; %d dispatches, %d reads, %@ read at %.2f GB/s waited; "
                + "fetch busy %.1f s (%.2fx), backpressure %.1f s over %d stalls; "
                + "pool %d/%d slots (%@)",
            phaseSeconds, ioWaitSeconds, ioWaitFraction * 100, gatherComputeSeconds,
            gatherComputeFraction * 100, unattributedSeconds, dispatchCount, expertReadCount,
            ByteSize.format(expertBytesRead), ioWaitBytesPerSecond / 1e9, fetchBusySeconds,
            fetchConcurrency, backpressureStallSeconds, backpressureStallCount,
            poolPeakInUseSlots, poolSlotCount, ByteSize.format(UInt64(poolBudgetBytes)))
    }

    enum CodingKeys: String, CodingKey {
        case granularity, queueDepth, readAhead, expertsPerDispatch, prefetchExpertsAhead
        case poolSlotCount, poolSlotBytes, poolBudgetBytes
        case gatherCallCount, dispatchCount, expertReadCount
        case expertBytesRequested, expertBytesRead, byteAccountingOverflowed, readErrorCount
        case phaseSeconds, ioWaitSeconds, gatherComputeSeconds, timingAccountingOverflowed
        case unattributedSeconds
        case fetchBusySeconds, backpressureStallSeconds, backpressureStallCount
        case poolPeakInUseSlots, poolPeakInUseBytes, poolFaults
        case perLayer
        case ioWaitFraction, gatherComputeFraction, fetchConcurrency, ioWaitBytesPerSecond
        case poolOccupancyFraction
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        granularity = try container.decode(String.self, forKey: .granularity)
        queueDepth = try container.decode(Int.self, forKey: .queueDepth)
        readAhead = try container.decode(Int.self, forKey: .readAhead)
        expertsPerDispatch = try container.decode(Int.self, forKey: .expertsPerDispatch)
        prefetchExpertsAhead = try container.decode(Int.self, forKey: .prefetchExpertsAhead)
        poolSlotCount = try container.decode(Int.self, forKey: .poolSlotCount)
        poolSlotBytes = try container.decode(Int.self, forKey: .poolSlotBytes)
        poolBudgetBytes = try container.decode(Int.self, forKey: .poolBudgetBytes)
        gatherCallCount = try container.decode(Int.self, forKey: .gatherCallCount)
        dispatchCount = try container.decode(Int.self, forKey: .dispatchCount)
        expertReadCount = try container.decode(Int.self, forKey: .expertReadCount)
        expertBytesRequested = try container.decode(UInt64.self, forKey: .expertBytesRequested)
        expertBytesRead = try container.decode(UInt64.self, forKey: .expertBytesRead)
        byteAccountingOverflowed = try container.decodeIfPresent(
            Bool.self, forKey: .byteAccountingOverflowed) ?? false
        readErrorCount = try container.decode(Int.self, forKey: .readErrorCount)
        phaseSeconds = try container.decode(Double.self, forKey: .phaseSeconds)
        ioWaitSeconds = try container.decode(Double.self, forKey: .ioWaitSeconds)
        gatherComputeSeconds = try container.decode(Double.self, forKey: .gatherComputeSeconds)
        timingAccountingOverflowed = try container.decodeIfPresent(
            Bool.self, forKey: .timingAccountingOverflowed) ?? false
        fetchBusySeconds = try container.decode(Double.self, forKey: .fetchBusySeconds)
        backpressureStallSeconds = try container.decode(
            Double.self, forKey: .backpressureStallSeconds)
        backpressureStallCount = try container.decode(Int.self, forKey: .backpressureStallCount)
        poolPeakInUseSlots = try container.decode(Int.self, forKey: .poolPeakInUseSlots)
        poolPeakInUseBytes = try container.decode(Int.self, forKey: .poolPeakInUseBytes)
        poolFaults = try container.decode([String].self, forKey: .poolFaults)
        perLayer = try container.decode([LayerSplit].self, forKey: .perLayer)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(granularity, forKey: .granularity)
        try container.encode(queueDepth, forKey: .queueDepth)
        try container.encode(readAhead, forKey: .readAhead)
        try container.encode(expertsPerDispatch, forKey: .expertsPerDispatch)
        try container.encode(prefetchExpertsAhead, forKey: .prefetchExpertsAhead)
        try container.encode(poolSlotCount, forKey: .poolSlotCount)
        try container.encode(poolSlotBytes, forKey: .poolSlotBytes)
        try container.encode(poolBudgetBytes, forKey: .poolBudgetBytes)
        try container.encode(gatherCallCount, forKey: .gatherCallCount)
        try container.encode(dispatchCount, forKey: .dispatchCount)
        try container.encode(expertReadCount, forKey: .expertReadCount)
        try container.encode(expertBytesRequested, forKey: .expertBytesRequested)
        try container.encode(expertBytesRead, forKey: .expertBytesRead)
        try container.encode(byteAccountingOverflowed, forKey: .byteAccountingOverflowed)
        try container.encode(readErrorCount, forKey: .readErrorCount)
        try container.encode(phaseSeconds, forKey: .phaseSeconds)
        try container.encode(ioWaitSeconds, forKey: .ioWaitSeconds)
        try container.encode(gatherComputeSeconds, forKey: .gatherComputeSeconds)
        try container.encode(timingAccountingOverflowed, forKey: .timingAccountingOverflowed)
        try container.encode(fetchBusySeconds, forKey: .fetchBusySeconds)
        try container.encode(backpressureStallSeconds, forKey: .backpressureStallSeconds)
        try container.encode(backpressureStallCount, forKey: .backpressureStallCount)
        try container.encode(poolPeakInUseSlots, forKey: .poolPeakInUseSlots)
        try container.encode(poolPeakInUseBytes, forKey: .poolPeakInUseBytes)
        try container.encode(poolFaults, forKey: .poolFaults)
        try container.encode(perLayer, forKey: .perLayer)
        // Derived, written so a reader of the JSON does not have to redo the
        // arithmetic — and never read back, so the two can never disagree.
        try container.encode(unattributedSeconds, forKey: .unattributedSeconds)
        try container.encode(ioWaitFraction, forKey: .ioWaitFraction)
        try container.encode(gatherComputeFraction, forKey: .gatherComputeFraction)
        try container.encode(fetchConcurrency, forKey: .fetchConcurrency)
        try container.encode(ioWaitBytesPerSecond, forKey: .ioWaitBytesPerSecond)
        try container.encode(poolOccupancyFraction, forKey: .poolOccupancyFraction)
    }
}
