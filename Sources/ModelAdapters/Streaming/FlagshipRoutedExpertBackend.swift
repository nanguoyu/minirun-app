import Foundation
import MLX
import MLXBridge
import StorageCore

/// Streams routed experts out of the published artifact's containers.
///
/// ``PagedRoutedExpertBackend`` does the same job over ``RoutedExpertContainers``
/// — the tiny adapter's layout, where one tile stacks several experts. At
/// flagship a tile is a row band of exactly *one* expert and the containers are
/// per layer, so the addressing is different enough to be its own type and the
/// pager machinery underneath is identical.
///
/// ## The budget, stated rather than emergent
///
/// Two bounds, and neither is model depth:
///
/// - **Readers.** A `TileReader` is a descriptor *plus* `queueDepth + 1` worker
///   threads, so one reader per container would be 372 readers and ~1,500
///   threads at 93 layers (M9B §6). ``BoundedReaderCache`` caps live readers at
///   ``Configuration/maxLiveReaders``, and the floor is one layer's concurrently
///   active readers — three, one per projection — which is what makes
///   "least recently used" and "provably idle" the same set: the decode loop
///   finishes a layer before it touches the next.
/// - **Slots.** `2 × slotsPerDispatch + readers × readAhead`, the formula
///   ``ExpertStreamEngine`` uses. It is checked at construction and *rejected*,
///   never clamped: silently shrinking a read-ahead window to fit a pool turns a
///   configuration error into a performance mystery.
public final class FlagshipRoutedExpertBackend: RoutedExpertBackend {

    public struct Configuration: Sendable {
        public var granularity: FetchGranularity
        public var queueDepth: Int
        public var readAhead: Int
        /// Queue a gather's regions as soon as the router has named them,
        /// instead of demanding them one at a time.
        ///
        /// **On by default** since the owner's decision of 2026-08-11: 73.5 s —
        /// the fastest flagship token recorded — was measured with this on, the
        /// logits digest was identical across all 15 runs, and MLX's peak never
        /// moved. There is nothing to state before turning it on: `slotCount`
        /// left `nil` takes the floor the window implies.
        ///
        /// Off is now the opt-out, and what it reproduces is the measurement
        /// conditions of every record before v0.6.18. It makes
        /// ``readAhead`` and ``queueDepth`` inert: `TileReader.acquire` inserts
        /// an unscheduled region at the head of the pending queue and marks it
        /// urgent, so `inFlight` never exceeds one and the second worker thread
        /// never has anything to do. Measured on the flagship token
        /// (`2026-08-11-expert-overlap.md`): 45.4 s of the 60.1 s expert phase
        /// was the consumer blocked in `acquire`, against 44.6 s of `pread` busy
        /// time — a ratio of 0.98, which is what "strictly alternating" looks
        /// like in a number.
        ///
        /// On, the window is ``readAhead`` deep and the pool has to hold it;
        /// ``slotCount`` states how much pool that is.
        ///
        /// This is *in-layer* prefetch and can be nothing else: the router
        /// consumes the layer's own hidden state to choose its ids, so there is
        /// nothing to fetch before the layer starts.
        public var prefetchExpertReads: Bool
        /// Experts covered by one gather dispatch.
        ///
        /// Sizes the pool; it does **not** yet change the program, which is one
        /// dispatch per (expert, row band) whatever this says.
        public var expertsPerDispatch: Int
        public var maxLiveReaders: Int
        public var noCache: Bool
        public var slotAcquireTimeout: TimeInterval?
        /// The pool's slot count — the routed-expert working set, stated rather
        /// than inferred (spec §5.3).
        ///
        /// `nil` takes the floor `2 × slotsPerDispatch + readers × readAhead`.
        /// A number below that floor is **rejected**, not clamped: a pool that
        /// cannot serve one dispatch plus the read-ahead window deadlocks on
        /// `acquire` rather than running slowly (spec §12.3).
        public var slotCount: Int?

        public init(
            granularity: FetchGranularity = .perExpert,
            queueDepth: Int = 3,
            readAhead: Int = 2,
            prefetchExpertReads: Bool = true,
            expertsPerDispatch: Int = 1,
            maxLiveReaders: Int = 6,
            noCache: Bool = false,
            slotAcquireTimeout: TimeInterval? = 120,
            slotCount: Int? = nil
        ) {
            self.granularity = granularity
            self.queueDepth = queueDepth
            self.readAhead = readAhead
            self.prefetchExpertReads = prefetchExpertReads
            self.expertsPerDispatch = max(1, expertsPerDispatch)
            self.maxLiveReaders = maxLiveReaders
            self.noCache = noCache
            self.slotAcquireTimeout = slotAcquireTimeout
            self.slotCount = slotCount
        }
    }

    public let configuration: Configuration
    public let layers: [FlagshipLayerStreams]
    private var readers: BoundedReaderCache
    private let pool: BufferPool
    private let slotBytes: Int
    private let slotCount: Int
    private let successfulReadObserver: (@Sendable (UInt64) -> Void)?

    public private(set) var expertBytesRequested: UInt64 = 0
    private var byteAccountingOverflowed = false
    public private(set) var expertReadCount: Int = 0
    public private(set) var gatherCount: Int = 0

    // The expert-phase decomposition. Three clocks, disjoint by construction:
    // `phase` brackets the whole gather entry point, `ioWait` brackets the one
    // call that blocks on storage, `gatherCompute` brackets the `eval`s that
    // force the graph. See ``ExpertPhaseMetrics``.
    private var phaseNanoseconds: UInt64 = 0
    private var ioWaitNanoseconds: UInt64 = 0
    private var gatherComputeNanoseconds: UInt64 = 0
    private var timingAccountingOverflowed = false
    private var dispatchCount: Int = 0
    /// Per layer, in the order the layers were first gathered from.
    private var layerSplits: [Int: ExpertPhaseMetrics.LayerSplit] = [:]
    private var layerOrder: [Int] = []

    public init(
        layers: [FlagshipLayerStreams],
        configuration: Configuration = Configuration(),
        successfulReadObserver: (@Sendable (UInt64) -> Void)? = nil
    ) throws
    {
        self.layers = layers
        self.configuration = configuration
        self.successfulReadObserver = successfulReadObserver

        let routing = layers.filter { !$0.containers.isEmpty }
        guard !routing.isEmpty else {
            throw TinyK3Error.configuration("no layer in this artifact has expert containers")
        }
        // The slot has to hold the largest single request any layer can make,
        // which at `perExpert` is a whole expert of the widest projection.
        slotBytes = routing.reduce(0) {
            max($0, $1.maximumRegionBytes(granularity: configuration.granularity))
        }
        // One reader per projection is what a layer touches at once; the cache
        // may hold more, and the pool has to cover whatever it holds.
        let concurrentReaders = ExpertProjection.allCases.count
        guard configuration.maxLiveReaders >= concurrentReaders else {
            throw TinyK3Error.configuration(
                "maxLiveReaders \(configuration.maxLiveReaders) is below the \(concurrentReaders) "
                    + "readers one layer holds at once; the cache would evict a reader with an "
                    + "outstanding lease")
        }
        let slotsPerDispatch = configuration.expertsPerDispatch
            * (configuration.granularity == .perExpert ? 1 : 1)
        // The floor, and the default budget: a dispatch's operands, the
        // previous dispatch's not-yet-reclaimed operands, and every reader's
        // window. Prefetching does not widen it — the window is what bounds
        // in-flight regions, whether they were scheduled or demanded.
        let floor = 2 * slotsPerDispatch + concurrentReaders * configuration.readAhead
        slotCount = configuration.slotCount ?? floor
        // Reject, do not clamp — twice over. `PagedReadPlan` needs
        // `evalEveryK + readAhead` slots to make progress, and a pool that
        // cannot supply them deadlocks on `acquire` rather than running slowly;
        // a stated budget below the floor is the same error said earlier.
        guard slotCount >= floor else {
            throw TinyK3Error.configuration(
                "a stated pool of \(slotCount) slots is below the \(floor) this configuration "
                    + "needs — \(2 * slotsPerDispatch) for a dispatch and its predecessor plus "
                    + "\(concurrentReaders) readers × \(configuration.readAhead) read-ahead")
        }
        guard slotsPerDispatch + configuration.readAhead <= slotCount else {
            throw TinyK3Error.configuration(
                "a pool of \(slotCount) slots cannot serve \(slotsPerDispatch) per dispatch plus "
                    + "\(configuration.readAhead) read-ahead")
        }
        pool = try BufferPool(
            configuration: BufferPoolConfiguration(
                slotCount: slotCount, slotBytes: slotBytes))
        readers = BoundedReaderCache(limit: configuration.maxLiveReaders)
    }

    public func shutdown() { readers.shutdownAll() }

    /// Bytes the pool holds — the whole routed-expert weight budget.
    public var budgetBytes: Int { slotCount * slotBytes }

    /// The routed expert pool's live counters. Exposed as a value snapshot so
    /// a product runner can report cache behavior without leaking the pool or
    /// granting another owner mutation authority.
    public var expertPoolStatistics: BufferPoolStatistics { pool.statistics }

    // MARK: - RoutedExpertBackend

    public func gather(
        _ x: MLXArray, layer: Int, projection: ExpertProjection, expertIds: [[Int]]
    ) throws -> MLXArray {
        guard layers.indices.contains(layer) else {
            throw TinyK3Error.configuration("layer \(layer) is outside this artifact")
        }
        let stream = layers[layer]
        let container = try stream.container(projection)
        gatherCount += 1

        // The phase bracket. `defer` and not a straight-line assignment: a read
        // that throws still spent the time, and a phase total that silently
        // dropped the failing layer would be the flattering kind of wrong.
        let phaseStart = MonotonicClock.now()
        let ioBefore = ioWaitNanoseconds
        let computeBefore = gatherComputeNanoseconds
        let dispatchesBefore = dispatchCount
        let readsBefore = expertReadCount
        let bytesBefore = expertBytesRequested
        let bytesOverflowedBefore = byteAccountingOverflowed
        defer {
            let elapsed = MonotonicClock.now() &- phaseStart
            addPhaseNanoseconds(elapsed)
            if layerSplits[layer] == nil {
                layerSplits[layer] = ExpertPhaseMetrics.LayerSplit(layer: layer)
                layerOrder.append(layer)
            }
            var split = layerSplits[layer]!
            split.phaseSeconds += Double(elapsed) / 1e9
            let ioDelta = ioWaitNanoseconds.subtractingReportingOverflow(ioBefore)
            let computeDelta = gatherComputeNanoseconds.subtractingReportingOverflow(computeBefore)
            if ioDelta.overflow || computeDelta.overflow { timingAccountingOverflowed = true }
            split.ioWaitSeconds += Double(ioDelta.overflow ? 0 : ioDelta.partialValue) / 1e9
            split.gatherComputeSeconds +=
                Double(computeDelta.overflow ? 0 : computeDelta.partialValue) / 1e9
            split.dispatchCount += dispatchCount - dispatchesBefore
            split.readCount += expertReadCount - readsBefore
            let byteDelta = expertBytesRequested.subtractingReportingOverflow(bytesBefore)
            var layerBytes = UInt64Accounting.SaturatingSum(
                value: split.bytesRequested,
                didOverflow: bytesOverflowedBefore || byteAccountingOverflowed
                    || byteDelta.overflow)
            layerBytes.add(byteDelta.overflow ? .max : byteDelta.partialValue)
            split.bytesRequested = layerBytes.value
            if layerBytes.didOverflow { byteAccountingOverflowed = true }
            layerSplits[layer] = split
        }

        let tokens = expertIds.count

        // Queue the whole gather's regions before consuming any of them. The
        // router has already named every expert this call will touch, so there
        // is nothing to predict — only a window to open. Without this the
        // reader cannot start region *n+1* until the consumer asks for it, and
        // `readAhead` and the second worker thread are both dead weight.
        //
        // Scheduling is *not* a claim on slots: the dispatcher admits a pending
        // region only while `inFlight < readAhead`, so the pending list may be
        // the whole layer while the pool holds the stated window and no more.
        if configuration.prefetchExpertReads {
            var regions: [RegionDescriptor] = []
            regions.reserveCapacity(tokens * (expertIds.first?.count ?? 0))
            for token in 0..<tokens {
                for expert in expertIds[token] {
                    let slot = try stream.slot(forExpert: expert)
                    switch configuration.granularity {
                    case .perExpert:
                        regions.append(container.expertRegion(slot: slot))
                    case .perTile:
                        for band in 0..<container.tilesPerExpert {
                            regions.append(container.tileRegion(slot: slot, band: band))
                        }
                    }
                }
            }
            try self.reader(for: container).schedule(regions)
        }

        var rows: [MLXArray] = []
        rows.reserveCapacity(tokens)
        for token in 0..<tokens {
            let input = x[token..<(token + 1), 0...]
            var perExpert: [MLXArray] = []
            perExpert.reserveCapacity(expertIds[token].count)
            for expert in expertIds[token] {
                let slot = try stream.slot(forExpert: expert)
                perExpert.append(
                    try product(input, stream: stream, container: container, slot: slot))
            }
            rows.append(stacked(perExpert, axis: 0))
        }
        let out = stacked(rows, axis: 0)
        let assembled = MonotonicClock.now()
        out.eval()
        addGatherComputeNanoseconds(MonotonicClock.now() &- assembled)
        return out
    }

    /// One expert's `[1, outFeaturesPerExpert]` contribution.
    private func product(
        _ input: MLXArray,
        stream: FlagshipLayerStreams,
        container: FlagshipLayerStreams.ExpertContainer,
        slot: Int
    ) throws -> MLXArray {
        let reader = try self.reader(for: container)
        let index = MLXArray([Int32(0)], [1, 1])
        var parts: [MLXArray] = []
        parts.reserveCapacity(container.tilesPerExpert)

        switch configuration.granularity {
        case .perExpert:
            // One read spanning every band of the expert. Legal only because
            // tiles are expert-major with a uniform stride, so the bands are
            // physically adjacent.
            let region = container.expertRegion(slot: slot)
            addRequestedBytes(UInt64(region.length))
            expertReadCount += 1
            let lease = try acquire(reader, region)
            var held: [MXFP4Weights] = []
            held.reserveCapacity(container.tilesPerExpert)
            for band in 0..<container.tilesPerExpert {
                try lease.retain()
                try lease.retain()
                let base = lease.slot.pointer + band * container.layout.tileStride
                let weights = try MXFP4Weights.adopting(
                    packedPointer: base,
                    scalesPointer: base + container.layout.scaleOffsetInTile,
                    experts: 1,
                    outFeatures: container.rowsPerTile,
                    inFeatures: container.inFeatures,
                    packedFinalizer: { lease.release() },
                    scalesFinalizer: { lease.release() })
                held.append(weights)
                dispatchCount += 1
                parts.append(
                    mxfp4GatherMM(input, weights, rhsIndices: index)
                        .reshaped([1, container.rowsPerTile]))
            }
            // Force the lazy graph before the slot can be recycled: every
            // expression above points into this lease's buffer.
            let evaluate = MonotonicClock.now()
            MLX.eval(parts)
            addGatherComputeNanoseconds(MonotonicClock.now() &- evaluate)
            held.removeAll()
            lease.release()

        case .perTile:
            for band in 0..<container.tilesPerExpert {
                let region = container.tileRegion(slot: slot, band: band)
                addRequestedBytes(UInt64(region.length))
                expertReadCount += 1
                let lease = try acquire(reader, region)
                try lease.retain()
                try lease.retain()
                let weights = try MXFP4Weights.adopting(
                    packedPointer: lease.slot.pointer,
                    scalesPointer: lease.slot.pointer + container.layout.scaleOffsetInTile,
                    experts: 1,
                    outFeatures: container.rowsPerTile,
                    inFeatures: container.inFeatures,
                    packedFinalizer: { lease.release() },
                    scalesFinalizer: { lease.release() })
                dispatchCount += 1
                let part = mxfp4GatherMM(input, weights, rhsIndices: index)
                    .reshaped([1, container.rowsPerTile])
                let evaluate = MonotonicClock.now()
                part.eval()
                addGatherComputeNanoseconds(MonotonicClock.now() &- evaluate)
                parts.append(part)
                lease.release()
            }
        }
        // Row bands concatenate with no cross-band accumulation, so the result
        // is the same arithmetic in the same order as a resident weight would
        // give — which is what lets the two granularities be required to agree
        // bit for bit rather than merely closely.
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: -1)
    }

    private func reader(for container: FlagshipLayerStreams.ExpertContainer) throws -> TileReader {
        try readers.reader(for: container.path) {
            let descriptor = try container.fileAccess.open(container.path)
            return try TileReader(
                owningFileDescriptor: descriptor, path: container.path, pool: pool,
                configuration: TileReaderConfiguration(
                    queueDepth: configuration.queueDepth,
                    readAhead: configuration.readAhead,
                    noCache: configuration.noCache,
                    slotAcquireTimeout: configuration.slotAcquireTimeout),
                successfulReadObserver: successfulReadObserver)
        }
    }

    /// Every `acquire` goes through here, so the one place the consumer blocks
    /// on storage is charged exactly once and in exactly one metric.
    private func acquire(_ reader: TileReader, _ region: RegionDescriptor) throws -> SlotLease {
        let start = MonotonicClock.now()
        defer { addIOWaitNanoseconds(MonotonicClock.now() &- start) }
        return try reader.acquire(region)
    }

    // MARK: - The expert-phase split

    /// Where the routed-expert phase's time went. See ``ExpertPhaseMetrics``.
    ///
    /// Read it **before** ``shutdown()``: the reader statistics survive
    /// eviction (``BoundedReaderCache`` absorbs them into `retired`), but the
    /// pool's census is only meaningful while the run's slots are still the
    /// run's.
    public var expertPhaseMetrics: ExpertPhaseMetrics {
        var metrics = ExpertPhaseMetrics()
        metrics.granularity = configuration.granularity.rawValue
        metrics.queueDepth = configuration.queueDepth
        metrics.readAhead = configuration.readAhead
        metrics.expertsPerDispatch = configuration.expertsPerDispatch
        // One number that says both things: the window is `readAhead` deep when
        // regions are queued ahead of demand, and zero when they are not —
        // because an unscheduled region is demanded one at a time however
        // `readAhead` is set.
        metrics.prefetchExpertsAhead = configuration.prefetchExpertReads
            ? configuration.readAhead : 0
        metrics.poolSlotCount = slotCount
        metrics.poolSlotBytes = slotBytes
        metrics.poolBudgetBytes = budgetBytes

        metrics.gatherCallCount = gatherCount
        metrics.dispatchCount = dispatchCount
        metrics.expertReadCount = expertReadCount
        metrics.expertBytesRequested = expertBytesRequested
        metrics.byteAccountingOverflowed = byteAccountingOverflowed

        metrics.phaseSeconds = Double(phaseNanoseconds) / 1e9
        metrics.ioWaitSeconds = Double(ioWaitNanoseconds) / 1e9
        metrics.gatherComputeSeconds = Double(gatherComputeNanoseconds) / 1e9
        metrics.timingAccountingOverflowed = timingAccountingOverflowed

        // Live and retired together. Dropping the retired half would
        // under-report the run's own traffic in the flattering direction, which
        // is the hazard spec §12.3 names for exactly this reader cache.
        var bytes = UInt64Accounting.SaturatingSum(
            value: readers.retired.bytesRead,
            didOverflow: readers.retired.byteCountOverflowed)
        var errors = readers.retired.errorCount
        var slotWait = readers.retired.slotWaitSeconds
        var busyNanoseconds = UInt64Accounting.saturatingSum(
            readers.retired.latencySamples)
        for reader in readers.live {
            let statistics = reader.statistics
            if statistics.byteCountOverflowed {
                bytes = UInt64Accounting.SaturatingSum(value: .max, didOverflow: true)
            } else {
                bytes.add(statistics.bytesRead)
            }
            errors += statistics.errorCount
            slotWait += statistics.slotWaitSeconds
            let readerBusy = UInt64Accounting.saturatingSum(reader.latencySamples)
            if readerBusy.didOverflow {
                busyNanoseconds = UInt64Accounting.SaturatingSum(value: .max, didOverflow: true)
            } else {
                busyNanoseconds.add(readerBusy.value)
            }
        }
        metrics.expertBytesRead = bytes.value
        metrics.byteAccountingOverflowed = metrics.byteAccountingOverflowed || bytes.didOverflow
        metrics.readErrorCount = errors
        metrics.fetchBusySeconds = Double(busyNanoseconds.value) / 1e9
        metrics.timingAccountingOverflowed =
            metrics.timingAccountingOverflowed || busyNanoseconds.didOverflow

        let statistics = pool.statistics
        metrics.timingAccountingOverflowed =
            metrics.timingAccountingOverflowed || statistics.blockedTimeOverflowed
        // The pool's own count of blocked acquisitions is the authority on how
        // often storage was held back; the dispatcher's `slotWaitSeconds` is the
        // same wait timed from the other side, and is the one that includes the
        // non-blocking acquisitions at zero.
        metrics.backpressureStallCount = statistics.blockedAcquireCount
        metrics.backpressureStallSeconds = max(statistics.blockedSeconds, slotWait)
        metrics.poolPeakInUseSlots = statistics.peakInUseSlots
        metrics.poolPeakInUseBytes = statistics.peakInUseBytes
        metrics.poolFaults = statistics.faults

        metrics.perLayer = layerOrder.compactMap { layerSplits[$0] }
        return metrics
    }

    private func addRequestedBytes(_ increment: UInt64) {
        var total = UInt64Accounting.SaturatingSum(
            value: expertBytesRequested, didOverflow: byteAccountingOverflowed)
        total.add(increment)
        expertBytesRequested = total.value
        byteAccountingOverflowed = total.didOverflow
    }

    private func addPhaseNanoseconds(_ increment: UInt64) {
        let total = saturatedTiming(phaseNanoseconds, adding: increment)
        phaseNanoseconds = total
    }

    private func addIOWaitNanoseconds(_ increment: UInt64) {
        let total = saturatedTiming(ioWaitNanoseconds, adding: increment)
        ioWaitNanoseconds = total
    }

    private func addGatherComputeNanoseconds(_ increment: UInt64) {
        let total = saturatedTiming(gatherComputeNanoseconds, adding: increment)
        gatherComputeNanoseconds = total
    }

    private func saturatedTiming(_ value: UInt64, adding increment: UInt64) -> UInt64 {
        var total = UInt64Accounting.SaturatingSum(
            value: value, didOverflow: timingAccountingOverflowed)
        total.add(increment)
        if total.didOverflow { timingAccountingOverflowed = true }
        return total.value
    }
}
