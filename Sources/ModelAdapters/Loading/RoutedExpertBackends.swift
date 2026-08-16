import Foundation
import MLX
import MLXBridge
import StorageCore

/// What a paged run can be asked about afterwards.
public struct RoutedExpertReport: Codable {
    /// Sum of every expert container's size on disk.
    public var containerTotalBytes: Int
    /// `slotCount * slotBytes` — the whole routed-expert weight budget.
    public var budgetBytes: Int
    public var slotCount: Int
    public var slotBytes: Int
    public var queueDepth: Int
    public var readAhead: Int
    /// Container bytes divided by the budget. The number the milestone's
    /// "budget smaller than model size" claim is made of.
    public var budgetRatio: Double
    public var readCount: Int
    public var bytesRead: UInt64
    public var shortReadCount: Int
    public var gatherCount: Int
    public var tileFetchCount: Int
    /// `peakInUseSlots * slotBytes`: the pager-managed high-water mark.
    public var poolPeakBytes: Int
    public var poolPeakSlots: Int
    public var poolFaults: [String]
    public var budgetRespected: Bool
    /// MLX's own high-water allocation. Advisory and *not* the routed-expert
    /// budget: it covers every resident weight and every activation in the
    /// process, so it answers a different question and is reported separately
    /// rather than folded in.
    public var mlxPeakBytes: Int
    /// Process-lifetime high-water resident set. Advisory in the same way, and
    /// additionally cumulative across everything the test binary has done.
    public var processPeakResidentBytes: UInt64?
}

/// Nanosecond brackets the routed-expert gather reports to whoever is
/// measuring the pass it runs inside.
///
/// The pager backend is generic infrastructure and must not name a model's
/// metrics type, so it reports through this protocol instead. The three
/// brackets follow K3's rule exactly (see ``ExpertPhaseMetrics``):
/// ``recordExpertPhase(nanoseconds:)`` is the whole gather entry point,
/// ``recordExpertIOWait(nanoseconds:)`` is only `acquire` — the one call that
/// blocks on storage — and ``recordExpertGatherCompute(nanoseconds:)`` is only
/// the `eval`s that force the gather graph.
public protocol RoutedExpertPhaseObserver: AnyObject, Sendable {
    func recordExpertPhase(nanoseconds: UInt64)
    func recordExpertIOWait(nanoseconds: UInt64)
    func recordExpertGatherCompute(nanoseconds: UInt64)
}

/// Streams routed-expert tiles through a bounded pool.
///
/// One `BufferPool` for the whole forward pass, one `TileReader` per container
/// file, kept alive so a repeated pass does not pay for thread creation 21
/// times. Readers hold no memory of their own — a slot is only held between
/// the read that fills it and the `acquire` that hands it over — so keeping
/// them all alive does not widen the budget.
public final class PagedRoutedExpertBackend: RoutedExpertBackend {

    public struct Configuration: Sendable {
        /// Slots in the pool. Times the tile size, this is the whole
        /// routed-expert weight budget.
        public var slotCount: Int
        /// `pread` worker threads (ADR-0001; M1a's plateau is 2–4).
        public var queueDepth: Int
        /// Tiles the reader may hold slots for at once.
        public var readAhead: Int
        /// `F_NOCACHE`. Off by default: this is a correctness path, and a warm
        /// cache only makes it faster.
        public var noCache: Bool
        public var slotAcquireTimeout: TimeInterval?
        /// Tiles covered by one `gatherQuantizedMM`.
        ///
        /// Under ``BufferTransferMode/adopt``, every tile in a dispatch is
        /// leased for the whole of it. The pool must therefore cover this
        /// width plus the read-ahead window. Under
        /// ``BufferTransferMode/copy``, each lease is returned synchronously
        /// before the next tile is acquired, so the dispatch keeps no pager
        /// slot and only the read-ahead window has to fit. Defaults to 1, the
        /// program M6 measured; raising it is a knob and a measurement, not a
        /// silent edit.
        public var tilesPerDispatch: Int
        /// How one filled pager slot becomes an MLX operand.
        ///
        /// Adoption avoids a copy, but the slot lease then follows MLX's
        /// graph/finalizer lifetime. Copying returns the lease synchronously
        /// before compute is expressed. Product generation chooses copying so
        /// one token cannot retain a layer-sized pool through the next token;
        /// benchmark/reference callers keep the historical adoption default.
        public var transferMode: BufferTransferMode

        public init(
            slotCount: Int = 4, queueDepth: Int = 2, readAhead: Int = 2,
            noCache: Bool = false, slotAcquireTimeout: TimeInterval? = 60,
            tilesPerDispatch: Int = 1,
            transferMode: BufferTransferMode = .adopt
        ) {
            self.slotCount = slotCount
            self.queueDepth = queueDepth
            self.readAhead = readAhead
            self.noCache = noCache
            self.slotAcquireTimeout = slotAcquireTimeout
            self.tilesPerDispatch = max(1, tilesPerDispatch)
            self.transferMode = transferMode
        }
    }

    public let containers: RoutedExpertContainers
    public let configuration: Configuration
    private let pool: BufferPool
    private let effectiveReadAhead: Int
    private let successfulReadObserver: (@Sendable (UInt64) -> Void)?
    private let phaseObserver: (any RoutedExpertPhaseObserver)?
    private let lifecycle = NSLock()
    private var readers: [String: TileReader] = [:]
    private var cancellationReason: String?
    private var didShutdown = false
    private var gatherCount = 0
    private var tileFetchCount = 0
    private var counterOverflowed = false
    private var retiredBytes = UInt64Accounting.SaturatingSum()

    public init(
        containers: RoutedExpertContainers,
        configuration: Configuration = Configuration(),
        successfulReadObserver: (@Sendable (UInt64) -> Void)? = nil,
        phaseObserver: (any RoutedExpertPhaseObserver)? = nil
    ) throws {
        self.containers = containers
        self.configuration = configuration
        self.successfulReadObserver = successfulReadObserver
        self.phaseObserver = phaseObserver

        // Every container in this set has the same tile stride, but that is a
        // property of this model rather than of the format, so the pool is
        // sized from the largest one actually present.
        let slotBytes = containers.entries.values.map(\.layout.tileStride).max() ?? 0
        guard slotBytes > 0 else {
            throw TinyK3Error.configuration("expert containers describe a zero-byte tile")
        }
        // The deadlock invariant is conditional on ownership. Adoption leaves
        // every tile in a dispatch leased until MLX evaluates the graph, so it
        // retains the generic `evalEveryK + readAhead <= slotCount` check.
        // Copying creates MLX-owned bytes and releases each lease inside
        // `copyOut`, before the gather loop can ask for its next tile. It has
        // zero graph-held pager slots, so `readAhead <= slotCount` is both the
        // necessary and sufficient pool bound. Do not apply the smaller bound
        // to adoption: that recreates the real deadlock PagedReadPlan exists to
        // reject.
        switch configuration.transferMode {
        case .adopt:
            let plan = try PagedReadPlan(
                slotCount: configuration.slotCount,
                readAhead: configuration.readAhead,
                evalEveryK: configuration.tilesPerDispatch)
            self.effectiveReadAhead = plan.readAhead
        case .copy:
            guard configuration.slotCount >= 1 else {
                throw PagerError.configuration("slotCount must be at least 1")
            }
            guard configuration.readAhead >= 1 else {
                throw PagerError.configuration("readAhead must be at least 1")
            }
            guard configuration.readAhead <= configuration.slotCount else {
                throw PagerError.configuration(
                    "copy transfer releases each lease synchronously, but readAhead "
                        + "(\(configuration.readAhead)) still exceeds slotCount "
                        + "(\(configuration.slotCount)); raise slotCount to at least "
                        + "\(configuration.readAhead), or lower readAhead")
            }
            self.effectiveReadAhead = configuration.readAhead
        }
        self.pool = try BufferPool(
            configuration: BufferPoolConfiguration(
                slotCount: configuration.slotCount,
                slotBytes: slotBytes,
                alignment: max(AlignedBuffer.pageSize, MXFP4TileContainer.regionAlignment)))
    }

    deinit { shutdown() }

    public func shutdown() {
        lifecycle.lock()
        guard !didShutdown else {
            lifecycle.unlock()
            return
        }
        didShutdown = true
        let closing = Array(readers.values)
        readers.removeAll()
        lifecycle.unlock()
        for reader in closing { reader.shutdown() }
        var retired = UInt64Accounting.SaturatingSum()
        for reader in closing {
            let statistics = reader.statistics
            retired.add(statistics.bytesRead)
            if statistics.byteCountOverflowed {
                retired = UInt64Accounting.SaturatingSum(
                    value: retired.value, didOverflow: true)
            }
        }
        lifecycle.lock()
        retiredBytes.add(retired.value)
        if retired.didOverflow {
            retiredBytes = UInt64Accounting.SaturatingSum(
                value: retiredBytes.value, didOverflow: true)
        }
        lifecycle.unlock()
        pool.evictAll()
    }

    /// Interrupt a routed-expert pass without allocating a replacement path.
    /// Readers wake blocked acquires; already-issued `pread`s finish into their
    /// bounded slots and are discarded. The backend is terminal after this.
    public func cancel(reason: String) {
        lifecycle.lock()
        if cancellationReason == nil { cancellationReason = reason }
        let recorded = cancellationReason ?? reason
        let active = Array(readers.values)
        lifecycle.unlock()
        pool.cancel(reason: recorded)
        for reader in active { reader.cancel(reason: recorded) }
    }

    public var poolCensus: SlotCensus { pool.census }
    public var poolStatistics: BufferPoolStatistics { pool.statistics }

    /// Execution-payload bytes survive reader teardown so the model can merge
    /// one layer's expert count after releasing every file descriptor and pool
    /// slot. A sticky overflow is carried separately from the saturated value.
    var expertReadAccounting: UInt64Accounting.SaturatingSum {
        lifecycle.lock()
        let active = Array(readers.values)
        var result = retiredBytes
        lifecycle.unlock()
        for reader in active {
            let statistics = reader.statistics
            result.add(statistics.bytesRead)
            if statistics.byteCountOverflowed {
                result = UInt64Accounting.SaturatingSum(
                    value: result.value, didOverflow: true)
            }
        }
        return result
    }

    private func reader(for entry: RoutedExpertContainers.Entry) throws -> TileReader {
        lifecycle.lock()
        defer { lifecycle.unlock() }
        if let reason = cancellationReason {
            throw PagerError.cancelled(reason: reason)
        }
        guard !didShutdown else {
            throw PagerError.configuration("routed-expert backend used after shutdown")
        }
        if let existing = readers[entry.path] { return existing }
        let descriptor = try entry.fileAccess.open(entry.path)
        let reader = try TileReader(
            owningFileDescriptor: descriptor, path: entry.path, pool: pool,
            configuration: TileReaderConfiguration(
                queueDepth: configuration.queueDepth,
                readAhead: effectiveReadAhead,
                noCache: configuration.noCache,
                slotAcquireTimeout: configuration.slotAcquireTimeout),
            successfulReadObserver: successfulReadObserver)
        readers[entry.path] = reader
        return reader
    }

    private func throwIfUnavailable() throws {
        lifecycle.lock()
        defer { lifecycle.unlock() }
        if let reason = cancellationReason {
            throw PagerError.cancelled(reason: reason)
        }
        guard !didShutdown else {
            throw PagerError.configuration("routed-expert backend used after shutdown")
        }
    }

    public func gather(
        _ x: MLXArray, layer: Int, projection: ExpertProjection, expertIds: [[Int]]
    ) throws -> MLXArray {
        // The phase bracket. `defer` and not a straight-line assignment: a
        // gather that throws still spent the time, and a phase total that
        // silently dropped the failing layer would be the flattering kind of
        // wrong.
        let phaseStart = MonotonicClock.now()
        defer {
            phaseObserver?.recordExpertPhase(
                nanoseconds: MonotonicClock.now() &- phaseStart)
        }
        let entry = try containers.entry(layer: layer, projection: projection)
        let reader = try self.reader(for: entry)
        lifecycle.lock()
        let nextGather = gatherCount.addingReportingOverflow(1)
        gatherCount = nextGather.overflow ? .max : nextGather.partialValue
        counterOverflowed = counterOverflowed || nextGather.overflow
        lifecycle.unlock()

        // Regions are scheduled up front so the read-ahead window has something
        // to work on; only the tiles the routing decision actually needs are
        // asked for, which is what makes `readCount` a demand-driven number.
        let plan = try TilePlan(entry: entry, expertIds: expertIds)
        // `sourceID` keeps identities from different containers apart in the
        // pool's cache map — a tile index alone would collide across the 21
        // files.
        let sourceID = Self.stableSourceID(entry.path)
        let regions = plan.tiles.map { entry.layout.region($0, sourceID: sourceID) }
        try reader.schedule(regions)

        var byTile: [Int: RegionDescriptor] = [:]
        for (position, tile) in plan.tiles.enumerated() { byTile[tile] = regions[position] }

        let result = try gatherOverTiles(
            x, entry: entry, expertIds: expertIds,
            tilesPerDispatch: configuration.tilesPerDispatch,
            evalObserver: phaseObserver.map { observer in
                { nanoseconds in observer.recordExpertGatherCompute(nanoseconds: nanoseconds) }
            }
        ) { tile in
            let lease = try self.acquire(reader, byTile[tile]!)
            self.lifecycle.lock()
            let nextFetch = self.tileFetchCount.addingReportingOverflow(1)
            self.tileFetchCount = nextFetch.overflow ? .max : nextFetch.partialValue
            self.counterOverflowed = self.counterOverflowed || nextFetch.overflow
            self.lifecycle.unlock()
            switch configuration.transferMode {
            case .adopt:
                return try Self.adopt(lease: lease, entry: entry)
            case .copy:
                return try Self.copyOut(lease: lease, entry: entry)
            }
        }
        try throwIfUnavailable()
        return result
    }

    /// Every `acquire` goes through here, so the one place the consumer blocks
    /// on storage is charged exactly once and in exactly one metric.
    private func acquire(_ reader: TileReader, _ region: RegionDescriptor) throws -> SlotLease {
        guard let phaseObserver else { return try reader.acquire(region) }
        let start = MonotonicClock.now()
        defer {
            phaseObserver.recordExpertIOWait(nanoseconds: MonotonicClock.now() &- start)
        }
        return try reader.acquire(region)
    }

    /// Swift's `hashValue` is process-randomized and `abs(Int.min)` traps. A
    /// region identity belongs in persisted/debuggable telemetry, so derive it
    /// deterministically from the repository-relative container identity.
    private static func stableSourceID(_ path: String) -> UInt64 {
        var value: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in path.utf8 {
            value ^= UInt64(byte)
            value &*= 0x0000_0100_0000_01b3
        }
        return value
    }

    /// Wrap one filled slot as two `MLXArray`s without copying: the tile's
    /// packed rows at the slot's base, its scales at the aligned tail offset,
    /// reinterpreted as a stack of `expertsPerTile` experts.
    ///
    /// That reinterpretation is the whole point of the container holding four
    /// experts per tile: the bytes are already expert-major, so
    /// `[expertsPerTile, rowsPerExpert, inFeatures]` is the same byte image as
    /// `[expertsPerTile * rowsPerExpert, inFeatures]`. No repack, no reorder.
    ///
    /// Two retains are taken before the arrays exist, because a finalizer can
    /// fire as soon as one does; this function's own reference is dropped last.
    private static func adopt(
        lease: SlotLease, entry: RoutedExpertContainers.Entry
    ) throws -> MXFP4Weights {
        try lease.retain()  // packed
        try lease.retain()  // scales

        let weights: MXFP4Weights
        do {
            weights = try MXFP4Weights.adopting(
                packedPointer: lease.slot.pointer,
                scalesPointer: lease.slot.pointer + entry.layout.scaleOffsetInTile,
                experts: entry.expertsPerTile,
                outFeatures: entry.rowsPerExpert,
                inFeatures: entry.inFeatures,
                packedFinalizer: { lease.release() },
                scalesFinalizer: { lease.release() })
        } catch {
            // Shapes are validated when the container index is built, so this
            // is unreachable in practice; only this function's own reference
            // has to be dropped here.
            lease.release()
            throw error
        }
        lease.release()  // the loop's own reference; MLX holds the other two
        return weights
    }

    /// Copy one tile into MLX-owned arrays and return its slot immediately.
    ///
    /// `MXFP4Weights`' copying initializer is eager (the adopting initializer
    /// is the separate raw-pointer API), so the deferred release cannot race a
    /// lazy read. This is the bounded-lifetime product path: a returned result
    /// may keep its MLX graph, but cannot keep the pager pool that supplied the
    /// bytes.
    private static func copyOut(
        lease: SlotLease, entry: RoutedExpertContainers.Entry
    ) throws -> MXFP4Weights {
        defer { lease.release() }
        let base = UnsafeRawPointer(lease.slot.pointer)
        return try MXFP4Weights(
            packedBytes: UnsafeRawBufferPointer(
                start: base, count: entry.layout.geometry.packedBytes),
            scaleBytes: UnsafeRawBufferPointer(
                start: base + entry.layout.scaleOffsetInTile,
                count: entry.layout.geometry.scaleBytes),
            experts: entry.expertsPerTile,
            outFeatures: entry.rowsPerExpert,
            inFeatures: entry.inFeatures)
    }

    /// Slots come back from MLX finalizers, which can run on a Metal
    /// completion thread after `eval` returns, so the census is read after a
    /// bounded wait rather than the instant the loop ends.
    public func report() -> RoutedExpertReport {
        let deadline = Date().addingTimeInterval(5)
        while pool.census.inUse > 0 && Date() < deadline { usleep(500) }

        lifecycle.lock()
        let reportingReaders = Array(readers.values)
        let reportingGatherCount = gatherCount
        let reportingTileFetchCount = tileFetchCount
        let reportingCounterOverflow = counterOverflowed
        lifecycle.unlock()
        var readCount = 0
        var bytesRead: UInt64 = 0
        var shortReads = 0
        var byteCountOverflowed = false
        var eventCountOverflowed = reportingCounterOverflow
        for reader in reportingReaders {
            let statistics = reader.statistics
            let reads = readCount.addingReportingOverflow(statistics.readCount)
            readCount = reads.overflow ? .max : reads.partialValue
            eventCountOverflowed = eventCountOverflowed || reads.overflow
            let sum = bytesRead.addingReportingOverflow(statistics.bytesRead)
            bytesRead = sum.overflow ? .max : sum.partialValue
            byteCountOverflowed = byteCountOverflowed || statistics.byteCountOverflowed
                || sum.overflow
            let shorts = shortReads.addingReportingOverflow(statistics.shortReadCount)
            shortReads = shorts.overflow ? .max : shorts.partialValue
            eventCountOverflowed = eventCountOverflowed || shorts.overflow
        }
        let statistics = pool.statistics
        var faults = statistics.faults
        if byteCountOverflowed { faults.append("expert byte accounting overflowed") }
        if eventCountOverflowed { faults.append("expert event accounting overflowed") }
        return RoutedExpertReport(
            containerTotalBytes: containers.totalBytes,
            budgetBytes: statistics.budgetBytes,
            slotCount: statistics.slotCount,
            slotBytes: statistics.slotBytes,
            queueDepth: configuration.queueDepth,
            readAhead: effectiveReadAhead,
            budgetRatio: Double(containers.totalBytes) / Double(statistics.budgetBytes),
            readCount: readCount,
            bytesRead: bytesRead,
            shortReadCount: shortReads,
            gatherCount: reportingGatherCount,
            tileFetchCount: reportingTileFetchCount,
            poolPeakBytes: statistics.peakInUseBytes,
            poolPeakSlots: statistics.peakInUseSlots,
            poolFaults: faults,
            budgetRespected: statistics.isBudgetRespected
                && !byteCountOverflowed && !eventCountOverflowed,
            mlxPeakBytes: MLX.Memory.peakMemory,
            processPeakResidentBytes: MemoryUsage.current()?.peakResidentBytes)
    }
}

/// The same computation with every container tile read into ordinary MLX-owned
/// arrays up front.
///
/// The question this answers is *does paging the bytes change the answer*, so
/// the only thing that may differ between the two paths is where the bytes came
/// from: same tile order, same gather geometry, same merge, same evaluation
/// cadence — and now the same ``tilesPerDispatch``.
///
/// That last one used to be fixed at "one tile per call", on the grounds that a
/// wider stack changes the kernel's launch geometry and a mismatch would then
/// no longer point at the pager. The hazard is real; the remedy was aimed at
/// the wrong thing. Holding the width at 1 protects the comparison only by
/// forbidding the width, whereas giving both arms the *same* width protects it
/// while leaving the width free — which is why this is a parameter the caller
/// sets for both, exactly as `expertsPerDispatch` is on the Qwen path.
public final class ResidentRoutedExpertBackend: RoutedExpertBackend {

    public let containers: RoutedExpertContainers
    /// Tiles per `gatherQuantizedMM`. Must match the paged backend's value
    /// whenever the two are compared, which is what
    /// ``PagedExpertGatherTests/testBatchedTileGatherIsBitIdenticalToPerTile``
    /// sets it for.
    public let tilesPerDispatch: Int
    private var tiles: [String: [MXFP4Weights]] = [:]

    public init(containers: RoutedExpertContainers, tilesPerDispatch: Int = 1) throws {
        self.containers = containers
        self.tilesPerDispatch = max(1, tilesPerDispatch)
        for entry in containers.entries.values {
            let loaded = try entry.fileAccess.withDescriptor(entry.path) { descriptor in
                let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
                var loaded: [MXFP4Weights] = []
                loaded.reserveCapacity(entry.layout.tileCount)
                for tile in 0..<entry.layout.tileCount {
                    try handle.seek(toOffset: entry.layout.tileOffset(tile))
                    guard let data = try handle.read(upToCount: entry.layout.tileStride),
                        data.count == entry.layout.tileStride
                    else {
                        throw TileContainerError.truncated(
                            path: entry.path, expectedBytes: entry.layout.totalBytes,
                            actualBytes: entry.layout.tileOffset(tile))
                    }
                    loaded.append(
                        try data.withUnsafeBytes { bytes -> MXFP4Weights in
                            let base = bytes.baseAddress!
                            return try MXFP4Weights(
                                packedBytes: UnsafeRawBufferPointer(
                                    start: base, count: entry.layout.geometry.packedBytes),
                                scaleBytes: UnsafeRawBufferPointer(
                                    start: base + entry.layout.scaleOffsetInTile,
                                    count: entry.layout.geometry.scaleBytes),
                                experts: entry.expertsPerTile,
                                outFeatures: entry.rowsPerExpert,
                                inFeatures: entry.inFeatures)
                        })
                }
                return loaded
            }
            tiles["\(entry.layer).\(entry.projection.rawValue)"] = loaded
        }
    }

    public func gather(
        _ x: MLXArray, layer: Int, projection: ExpertProjection, expertIds: [[Int]]
    ) throws -> MLXArray {
        let entry = try containers.entry(layer: layer, projection: projection)
        let loaded = tiles["\(layer).\(projection.rawValue)"]!
        return try gatherOverTiles(
            x, entry: entry, expertIds: expertIds, tilesPerDispatch: tilesPerDispatch
        ) { loaded[$0] }
    }
}
