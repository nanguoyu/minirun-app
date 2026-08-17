import Foundation
import MLXBridge
import StorageCore

/// Routed-expert backends that outlive the layer that built them.
///
/// ## What this is for
///
/// V4 built and destroyed one ``PagedRoutedExpertBackend`` per layer per pass
/// (`DeepSeekV4ModelExecution.withLayerBackend`). Two things follow from that,
/// and the second is the larger:
///
/// 1. **Nothing can be in flight across a layer boundary.** Layer L+1's reader
///    does not exist while layer L computes, so however wide the window is set,
///    the first tiles of every layer are read with the decode thread already
///    waiting for them. The 2026-08-17 overlap record names this the single
///    largest remaining serialisation and calls it "a lifetime problem, not a
///    scheduling one".
/// 2. **The pool was allocated 43 times a pass.** `expertPoolSlots x tileStride`
///    is 89 MB at the current default; `posix_memalign` of that size is fresh
///    anonymous memory, so every slot's first `pread` faults its pages in, and
///    the layer boundary's `malloc_zone_pressure_relief` hands them straight
///    back. Those faults are taken inside the `pread`, which means they are
///    charged to `expertIOWaitSeconds` — a storage number that is partly the
///    allocator.
///
/// ## The memory statement
///
/// **One pool, shared.** Every layer's backend built here is handed the same
/// ``BufferPool``, so ``poolBudgetBytes`` is still `tileStride x slotCount` and
/// the eight-term accounting identity is untouched: the residency does not
/// multiply a reserved term by the number of layers it keeps alive. A
/// per-layer pool would have made the same overlap possible and cost 43 x 89 MB
/// = 3.8 GB, which is not a trade this project may make quietly (spec §5.3).
///
/// What the residency *does* add to the stated pool is
/// ``PagedRoutedExpertBackend/Configuration/crossLayerPrefetchTiles``: slots
/// held by a layer that has not started. It is refused rather than clamped by
/// ``PagedRoutedExpertBackend/validatedReadAhead(configuration:readerCount:)``,
/// which the residency calls once, before it opens a descriptor.
///
/// What it adds outside the pool is threads and descriptors, and that cost is
/// real and is stated rather than hidden: one ``TileReader`` per container per
/// layer, opened on first use and kept, is **43 x 3 = 129 descriptors and
/// 129 x (queueDepth + 1) = 645 threads** at the published geometry and the
/// current default depth, against 3 descriptors and 15 threads alive at a time
/// before — but *created and joined 43 times a pass*. The exchange is a
/// constant standing cost for a per-pass one, and the standing cost is bounded
/// by the model's layer count, not by the run's length. `kern.num_taskthreads`
/// is 8192 on this machine; a geometry that made 645 into 8000 would need this
/// bounded again, which is why the number is written down here.
public final class DeepSeekV4RunExpertBackends: RoutedExpertScheduleCoordinator {

    /// One layer's containers, as the residency needs to see them.
    public struct LayerUnit {
        public let layer: Int
        public let containers: RoutedExpertContainers

        public init(layer: Int, containers: RoutedExpertContainers) {
            self.layer = layer
            self.containers = containers
        }
    }

    /// A hash-routed layer, and the expert ids it will select for a given
    /// token. Hash routing reads its ids out of a `[vocabulary, k]` table, so
    /// they are a function of the token id alone and are knowable before the
    /// pass runs — the only layers in V4 for which that is true.
    public struct HashRoutedLayer {
        public let layer: Int
        public let map: DeepSeekV4TokenExpertMap

        public init(layer: Int, map: DeepSeekV4TokenExpertMap) {
            self.layer = layer
            self.map = map
        }
    }

    public let configuration: PagedRoutedExpertBackend.Configuration
    public let slotBytes: Int
    /// `tileStride x slotCount` — the one routed-expert weight term, whatever
    /// the number of live layers.
    public var poolBudgetBytes: Int { pool.configuration.budgetBytes }

    private let pool: BufferPool
    private let units: [Int: RoutedExpertContainers]
    private let successfulReadObserver: (@Sendable (UInt64) -> Void)?
    private let phaseObserver: (any RoutedExpertPhaseObserver)?
    private let buildObserver: (@Sendable (UInt64) -> Void)?
    private let retireObserver: (@Sendable (UInt64) -> Void)?

    private let lock = NSLock()
    private var backends: [Int: PagedRoutedExpertBackend] = [:]
    private var didShutdown = false
    private var buildCount = 0

    /// Regions computed at pass start for layers whose ids are known in
    /// advance, in the order they will be consumed, and how many of them have
    /// been handed to a reader.
    private struct AheadEntry {
        let layer: Int
        let projection: ExpertProjection
        let regions: [RegionDescriptor]
    }
    private var ahead: [AheadEntry] = []
    private var aheadCursor = 0
    /// How much of `ahead[aheadCursor]` has already been queued. The window is
    /// a tile count, not a projection count, so a stated window narrower than
    /// one projection still schedules what it can rather than nothing.
    private var aheadOffset = 0
    /// Issued and not yet consumed, per layer. A layer's tiles stop counting
    /// against the cross-layer window the moment its own router runs, because
    /// its gather is then the next thing that will acquire them.
    private var outstandingAhead: [Int: Int] = [:]

    public init(
        units: [LayerUnit],
        configuration: PagedRoutedExpertBackend.Configuration,
        successfulReadObserver: (@Sendable (UInt64) -> Void)? = nil,
        phaseObserver: (any RoutedExpertPhaseObserver)? = nil,
        buildObserver: (@Sendable (UInt64) -> Void)? = nil,
        retireObserver: (@Sendable (UInt64) -> Void)? = nil
    ) throws {
        guard !units.isEmpty else {
            throw PagerError.configuration(
                "a run-scoped expert residency needs at least one layer's containers")
        }
        var byLayer: [Int: RoutedExpertContainers] = [:]
        var slotBytes = 0
        var readerCount = 0
        for unit in units {
            guard byLayer[unit.layer] == nil else {
                throw PagerError.configuration(
                    "layer \(unit.layer) appears twice in the expert residency")
            }
            byLayer[unit.layer] = unit.containers
            slotBytes = max(
                slotBytes, unit.containers.entries.values.map(\.layout.tileStride).max() ?? 0)
            readerCount = max(readerCount, unit.containers.entries.count)
        }
        guard slotBytes > 0 else {
            throw TinyK3Error.configuration("expert containers describe a zero-byte tile")
        }
        // Priced before a descriptor is opened, and refused rather than
        // narrowed: a run that quietly shortened its own cross-layer window
        // would be recorded under a window it did not use (spec §12.3).
        _ = try PagedRoutedExpertBackend.validatedReadAhead(
            configuration: configuration, readerCount: readerCount)

        self.units = byLayer
        self.configuration = configuration
        self.slotBytes = slotBytes
        self.successfulReadObserver = successfulReadObserver
        self.phaseObserver = phaseObserver
        self.buildObserver = buildObserver
        self.retireObserver = retireObserver
        self.pool = try BufferPool(
            configuration: BufferPoolConfiguration(
                slotCount: configuration.slotCount,
                slotBytes: slotBytes,
                alignment: max(AlignedBuffer.pageSize, MXFP4TileContainer.regionAlignment)))
    }

    deinit { shutdown() }

    /// How many backends this residency has built. One per layer for a run that
    /// keeps them, against `layers x passes` for a run that does not — the
    /// number that says the lifetime change actually took.
    public var backendBuildCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return buildCount
    }

    public var livePoolCensus: SlotCensus { pool.census }
    public var poolStatistics: BufferPoolStatistics { pool.statistics }

    /// The backend for one layer, built on first use and kept.
    public func backend(for layer: Int) throws -> PagedRoutedExpertBackend {
        lock.lock()
        if didShutdown {
            lock.unlock()
            throw PagerError.configuration("expert residency used after shutdown")
        }
        if let existing = backends[layer] {
            lock.unlock()
            return existing
        }
        guard let containers = units[layer] else {
            lock.unlock()
            throw PagerError.configuration(
                "layer \(layer) has no containers in this expert residency")
        }
        lock.unlock()

        let start = MonotonicClock.now()
        let backend = try PagedRoutedExpertBackend(
            containers: containers,
            configuration: configuration,
            sharedPool: pool,
            successfulReadObserver: successfulReadObserver,
            phaseObserver: phaseObserver)
        backend.scheduleCoordinator = self
        buildObserver?(MonotonicClock.now() &- start)

        lock.lock()
        if didShutdown {
            lock.unlock()
            backend.shutdown()
            throw PagerError.configuration("expert residency used after shutdown")
        }
        if let raced = backends[layer] {
            lock.unlock()
            backend.shutdown()
            return raced
        }
        backends[layer] = backend
        buildCount += 1
        lock.unlock()
        return backend
    }

    /// Cancel every live backend and the pool they share.
    public func cancel(reason: String) {
        lock.lock()
        let live = Array(backends.values)
        lock.unlock()
        pool.cancel(reason: reason)
        for backend in live { backend.cancel(reason: reason) }
    }

    public func shutdown() {
        lock.lock()
        guard !didShutdown else {
            lock.unlock()
            return
        }
        didShutdown = true
        let live = Array(backends.values)
        backends.removeAll()
        ahead.removeAll()
        outstandingAhead.removeAll()
        lock.unlock()
        for backend in live { retire(backend) }
        pool.evictAll()
    }

    // MARK: - Cross-pass scheduling for hash-routed layers

    /// Queue what this pass already knows it will read.
    ///
    /// A hash-routed layer's experts are `tid2eid[token]` — a table lookup, not
    /// a function of any hidden state — so at the top of a decode pass the ids
    /// for every hash layer are already determined. Their tiles are queued here,
    /// bounded by ``PagedRoutedExpertBackend/Configuration/crossLayerPrefetchTiles``
    /// and topped back up as each layer's router runs, so the reads proceed
    /// underneath the embedding lookup and the preceding layers' compute
    /// instead of at the moment the gather asks for them.
    ///
    /// Nothing here is speculative and nothing here waits: these are exactly the
    /// tiles those layers are about to acquire, scheduled by the same region
    /// identity, so the gather's own `schedule` finds the entry already queued
    /// and `readCount` does not move.
    public func beginPass(tokenID: Int, hashRoutedLayers: [HashRoutedLayer]) throws {
        guard configuration.crossLayerPrefetchTiles > 0, !hashRoutedLayers.isEmpty else {
            return
        }
        var planned: [AheadEntry] = []
        for hash in hashRoutedLayers.sorted(by: { $0.layer < $1.layer }) {
            guard units[hash.layer] != nil else { continue }
            let ids = try hash.map.experts(for: [tokenID])
            let backend = try self.backend(for: hash.layer)
            for projection in ExpertProjection.allCases {
                planned.append(
                    AheadEntry(
                        layer: hash.layer,
                        projection: projection,
                        regions: try backend.plannedRegions(
                            layer: hash.layer, projection: projection, expertIds: ids)))
            }
        }
        lock.lock()
        ahead = planned
        aheadCursor = 0
        aheadOffset = 0
        outstandingAhead.removeAll()
        lock.unlock()
        try issueAhead(currentLayer: -1)
    }

    func routedLayerDidSelect(layer: Int) throws {
        try issueAhead(currentLayer: layer)
    }

    /// Hand the reader as much of the pre-computed schedule as the stated
    /// cross-layer window still has room for.
    ///
    /// The window counts tiles belonging to layers *after* `currentLayer`,
    /// because those are the ones occupying a slot that the current layer's own
    /// gather cannot free. Tiles for the current layer stop counting the moment
    /// its router runs: its gather is the next thing that acquires them.
    private func issueAhead(currentLayer: Int) throws {
        while true {
            lock.lock()
            if didShutdown || aheadCursor >= ahead.count {
                lock.unlock()
                return
            }
            let entry = ahead[aheadCursor]
            // The schedule has caught up with the layer that is running. Its
            // own routing prefetch owns those tiles from here, so they stop
            // counting against the cross-layer window.
            if entry.layer <= currentLayer {
                aheadCursor += 1
                aheadOffset = 0
                outstandingAhead[entry.layer] = nil
                lock.unlock()
                continue
            }
            let held = outstandingAhead.reduce(0) { total, pair in
                pair.key > currentLayer ? total + pair.value : total
            }
            let remaining = entry.regions.count - aheadOffset
            let take = min(configuration.crossLayerPrefetchTiles - held, remaining)
            guard take > 0 else {
                lock.unlock()
                return
            }
            let slice = Array(entry.regions[aheadOffset..<(aheadOffset + take)])
            if take == remaining {
                aheadCursor += 1
                aheadOffset = 0
            } else {
                aheadOffset += take
            }
            outstandingAhead[entry.layer, default: 0] += take
            lock.unlock()

            let backend = try self.backend(for: entry.layer)
            try backend.scheduleRegions(
                slice, layer: entry.layer, projection: entry.projection)
        }
    }

    private func retire(_ backend: PagedRoutedExpertBackend) {
        let start = MonotonicClock.now()
        backend.shutdown()
        retireObserver?(MonotonicClock.now() &- start)
    }
}
