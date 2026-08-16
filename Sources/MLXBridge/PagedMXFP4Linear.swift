import Foundation
import MLX
import StorageCore

/// How a filled pool slot reaches MLX.
///
/// The two ends of spec §14.2's "copied or exposed": ``adopt`` is the
/// integration path M4 measured at scale, ``copy`` is the conservative bridge
/// that needs nothing from MLX beyond its ordinary copying initializer. M5
/// exists to price one against the other rather than assume the answer.
public enum BufferTransferMode: String, Codable, Sendable, CaseIterable {
    /// Wrap the slot's bytes as `MLXArray`s in place. The lease lives until
    /// MLX's finalizers fire, which is after the graph that used them is
    /// evaluated.
    case adopt
    /// Copy the slot's bytes into fresh MLX-owned arrays and hand the slot back
    /// before any compute is expressed. Costs a memcpy per tile and a second
    /// live copy of the tile; buys a lease that ends at a statement boundary
    /// rather than at an MLX finalizer on an unknown thread.
    case copy
}

/// How a paged run is budgeted and instrumented.
public struct PagedLinearConfiguration: Sendable {
    /// Slots in the pool. Times the tile size, this is the whole weight budget.
    public var slotCount: Int
    /// `pread` worker threads (ADR-0001; M1a's plateau is 2–4).
    public var queueDepth: Int
    /// Regions the reader may hold slots for at once.
    public var readAhead: Int
    /// Evaluate the graph every this many tiles. See ``PagedReadPlan`` for why
    /// this cannot be raised freely against a fixed slot count.
    public var evalEveryK: Int
    /// `F_NOCACHE`. On for measurement, off for correctness tests where a warm
    /// cache only makes the run faster.
    public var noCache: Bool
    /// MLX's allocator keeps freed buffers in a cache that is not returned to
    /// the system. Left unbounded it grows to whatever the largest transient
    /// was and makes "peak memory" describe MLX's reuse policy rather than this
    /// run's working set, so the harness pins it and reports what it pinned.
    public var mlxCacheLimitBytes: Int
    /// Bounds a stall. A genuine deadlock then names the wait instead of
    /// hanging the run.
    public var slotAcquireTimeout: TimeInterval?
    /// How a filled slot reaches MLX. Defaults to ``BufferTransferMode/adopt``,
    /// which is what every result before M5 was measured with.
    public var transferMode: BufferTransferMode

    public init(
        slotCount: Int = 8,
        queueDepth: Int = 3,
        readAhead: Int = 3,
        evalEveryK: Int = 2,
        noCache: Bool = false,
        mlxCacheLimitBytes: Int = 64 << 20,
        slotAcquireTimeout: TimeInterval? = 120,
        transferMode: BufferTransferMode = .adopt
    ) {
        self.slotCount = slotCount
        self.queueDepth = queueDepth
        self.readAhead = readAhead
        self.evalEveryK = evalEveryK
        self.noCache = noCache
        self.mlxCacheLimitBytes = mlxCacheLimitBytes
        self.slotAcquireTimeout = slotAcquireTimeout
        self.transferMode = transferMode
    }

    /// Validates the deadlock invariant. Throws; never clamps.
    public func plan() throws -> PagedReadPlan {
        try PagedReadPlan(
            slotCount: slotCount, readAhead: readAhead, evalEveryK: evalEveryK)
    }
}

/// Everything one paged run can be asked about afterwards.
public struct PagedLinearReport: Codable {
    // Shape of the problem.
    public var tileCount: Int
    public var tileRows: Int
    public var tileCols: Int
    public var totalRows: Int
    public var tileBytes: Int
    public var containerBytes: UInt64

    // Budget.
    public var slotCount: Int
    public var queueDepth: Int
    public var readAhead: Int
    public var evalEveryK: Int
    public var budgetBytes: Int
    public var noCache: Bool
    public var mlxCacheLimitBytes: Int
    /// How each slot reached MLX. Every figure below is conditional on it, so
    /// it is recorded next to the budget rather than left to the reader's
    /// memory of what was configured. Result files written before M5 have no
    /// such field and are all `adopt`.
    public var transferMode: BufferTransferMode

    // I/O.
    public var readCount: Int
    public var bytesRead: UInt64
    public var shortReadCount: Int
    public var readLatency: LatencyStatistics?

    // Where the time went.
    public var wallSeconds: Double
    /// Time the compute loop spent inside `acquire`, i.e. waiting on storage.
    public var computeWaitOnIOSeconds: Double
    /// Time the reader spent waiting for a free slot, i.e. storage waiting on
    /// compute. The two together say which side is the constraint.
    public var readerWaitOnSlotSeconds: Double
    /// Time inside `eval`, summed over every evaluation boundary in the loop.
    /// This is the blocking GPU cost; `wallSeconds` minus this and
    /// `computeWaitOnIOSeconds` is loop overhead and graph construction.
    public var evalSeconds: Double
    /// Time spent copying slots into MLX-owned arrays, summed over every tile.
    /// Zero under ``BufferTransferMode/adopt``, where no such copy exists —
    /// which is the point of reporting it separately from the loop overhead it
    /// would otherwise disappear into.
    public var copySeconds: Double

    // Memory.
    public var pool: BufferPoolStatistics
    /// `peakInUseSlots * slotBytes`: the pager-managed high-water mark, which
    /// is the number that must stay flat as the file grows.
    public var poolPeakBytes: Int
    public var mlxPeakBytes: Int
    public var mlxActiveBytesAtEnd: Int
    public var mlxCacheBytesAtEnd: Int
    /// Resident set at the end of this run. Advisory: it includes the whole
    /// process, MLX's allocator and the test harness with it.
    public var processResidentBytes: UInt64?
    /// Process-lifetime high-water resident set. Advisory in the same way, and
    /// additionally cumulative — in a process that has already run a larger
    /// case, this says nothing about the current one.
    public var processPeakResidentBytes: UInt64?

    public var throughputMBPerSecond: Double {
        guard wallSeconds > 0 else { return 0 }
        return Double(bytesRead) / wallSeconds / 1_000_000
    }
}

/// A dense MXFP4 linear whose weight matrix never fits in the budget.
///
/// ## What "paged" means here
///
/// The matrix is stored as output-row tiles. Each tile is read whole into one
/// pool slot, adopted as two `MLXArray`s without a copy, multiplied, and the
/// slot handed back — while the next tiles are already being read. The budget
/// is `slotCount` tiles, whatever the matrix weighs.
///
/// ``BufferTransferMode/copy`` substitutes a memcpy into MLX-owned arrays for
/// that adoption, releasing the slot before the matmul is even expressed. It is
/// the same arithmetic over the same bytes — the two modes are asserted
/// bit-identical — and exists so the cost of the conservative bridge is a
/// measured number rather than an assumption (spec §14.2, §14.3).
///
/// ## Why the answer is bit-identical and not merely close
///
/// Tiling is along output rows only. Output row *r* is a dot product over the
/// full reduction axis regardless of which tile it lands in, so no partial sums
/// cross a tile boundary and no summation order depends on the tiling. The
/// paged result is therefore the *same arithmetic in the same order* as the
/// resident one, and the correctness gate can demand bit-for-bit equality
/// rather than a tolerance. If tiles were ever cut along the reduction axis
/// instead, that gate would have to weaken to a tolerance — which is the main
/// reason not to.
///
/// ## The lifetime that makes it work
///
/// MLX graphs are lazy, and a graph node holds its operands. Expressing a
/// tile's matmul does not free the tile's bytes: the slot stays leased until
/// the graph is *evaluated*. That is why the loop evaluates every `evalEveryK`
/// tiles and why ``PagedReadPlan`` refuses a configuration where the
/// unevaluated tiles plus the read-ahead window could exceed the pool.
public enum PagedMXFP4Linear {

    /// Stream a container's tiles through a bounded pool, computing `x · W^T`.
    ///
    /// - Parameters:
    ///   - x: `[M, inFeatures]` activations.
    ///   - containerPath: a container written by ``MXFP4TileContainer``.
    /// - Returns: `[M, totalRows]` and a report of what it cost.
    public static func run(
        _ x: MLXArray,
        containerPath: String,
        configuration: PagedLinearConfiguration = PagedLinearConfiguration()
    ) throws -> (output: MLXArray, report: PagedLinearReport) {
        // Every rejection that can happen before a byte is read happens here:
        // a truncated container, a nonsensical geometry, or a budget that could
        // deadlock must not be discovered halfway through a stream.
        let layout = try MXFP4TileContainer.open(path: containerPath)
        let plan = try configuration.plan()
        try validate(x: x, layout: layout)

        let pool = try BufferPool(
            configuration: BufferPoolConfiguration(
                slotCount: plan.slotCount,
                slotBytes: layout.tileStride,
                alignment: max(AlignedBuffer.pageSize, MXFP4TileContainer.regionAlignment)))
        defer { pool.evictAll() }

        return try run(
            x, containerPath: containerPath, layout: layout, pool: pool,
            configuration: configuration)
    }

    /// Same, over a caller-owned pool. Used to prove a pool survives a
    /// cancelled run and is reusable afterwards.
    public static func run(
        _ x: MLXArray,
        containerPath: String,
        layout: MXFP4TileContainer.Layout,
        pool: BufferPool,
        configuration: PagedLinearConfiguration = PagedLinearConfiguration()
    ) throws -> (output: MLXArray, report: PagedLinearReport) {
        let plan = try configuration.plan()
        try validate(x: x, layout: layout)
        guard pool.configuration.slotBytes >= layout.tileStride else {
            throw PagerError.configuration(
                "pool slots are \(pool.configuration.slotBytes) bytes, a tile is \(layout.tileStride)")
        }
        guard pool.configuration.slotCount == plan.slotCount else {
            throw PagerError.configuration(
                "pool has \(pool.configuration.slotCount) slots, the plan assumes \(plan.slotCount)")
        }

        MLX.Memory.cacheLimit = configuration.mlxCacheLimitBytes
        MLX.Memory.clearCache()
        MLX.Memory.peakMemory = 0

        let reader = try TileReader(
            path: containerPath,
            pool: pool,
            configuration: TileReaderConfiguration(
                queueDepth: configuration.queueDepth,
                readAhead: plan.readAhead,
                noCache: configuration.noCache,
                slotAcquireTimeout: configuration.slotAcquireTimeout))
        defer { reader.shutdown() }

        let regions = (0..<layout.tileCount).map { layout.region($0) }
        try reader.schedule(regions)

        var waitOnIONanoseconds: UInt64 = 0
        var evalNanoseconds: UInt64 = 0
        var copyNanoseconds: UInt64 = 0
        let started = MonotonicClock.now()

        let output = try executeTiles(
            x: x,
            tileCount: layout.tileCount,
            evalEveryK: plan.evalEveryK,
            evalNanoseconds: &evalNanoseconds,
            weightsForTile: { index in
                let waitStart = MonotonicClock.now()
                let lease = try reader.acquire(regions[index])
                waitOnIONanoseconds &+= MonotonicClock.now() &- waitStart

                switch configuration.transferMode {
                case .adopt:
                    return try adopt(lease: lease, layout: layout)
                case .copy:
                    let copyStart = MonotonicClock.now()
                    defer { copyNanoseconds &+= MonotonicClock.now() &- copyStart }
                    return try copyOut(lease: lease, layout: layout)
                }
            })
        // The loop already evaluated the last batch, so this is a no-op except
        // when tileCount is not a multiple of evalEveryK.
        let tailStart = MonotonicClock.now()
        output.eval()
        evalNanoseconds &+= MonotonicClock.now() &- tailStart
        let wallSeconds = MonotonicClock.seconds(since: started)

        // Slots come back from MLX finalizers, which may run on a Metal
        // completion thread after eval returns. Give them a bounded moment
        // before reporting the census, or the report accuses itself of a leak.
        waitForSlotsToDrain(pool: pool, timeout: 5)

        let readerStatistics = reader.statistics
        let report = PagedLinearReport(
            tileCount: layout.tileCount,
            tileRows: layout.geometry.rows,
            tileCols: layout.geometry.cols,
            totalRows: layout.totalRows,
            tileBytes: layout.tileStride,
            containerBytes: layout.totalBytes,
            slotCount: plan.slotCount,
            queueDepth: configuration.queueDepth,
            readAhead: plan.readAhead,
            evalEveryK: plan.evalEveryK,
            budgetBytes: pool.configuration.budgetBytes,
            noCache: configuration.noCache,
            mlxCacheLimitBytes: configuration.mlxCacheLimitBytes,
            transferMode: configuration.transferMode,
            readCount: readerStatistics.readCount,
            bytesRead: readerStatistics.bytesRead,
            shortReadCount: readerStatistics.shortReadCount,
            readLatency: readerStatistics.latency,
            wallSeconds: wallSeconds,
            computeWaitOnIOSeconds: Double(waitOnIONanoseconds) / 1_000_000_000,
            readerWaitOnSlotSeconds: readerStatistics.slotWaitSeconds,
            evalSeconds: Double(evalNanoseconds) / 1_000_000_000,
            copySeconds: Double(copyNanoseconds) / 1_000_000_000,
            pool: pool.statistics,
            poolPeakBytes: pool.statistics.peakInUseSlots * pool.configuration.slotBytes,
            mlxPeakBytes: MLX.Memory.peakMemory,
            mlxActiveBytesAtEnd: MLX.Memory.activeMemory,
            mlxCacheBytesAtEnd: MLX.Memory.cacheMemory,
            processResidentBytes: MemoryUsage.current()?.residentBytes,
            processPeakResidentBytes: MemoryUsage.current()?.peakResidentBytes)
        return (output, report)
    }

    /// The same computation with the whole matrix resident: every tile read
    /// into ordinary MLX-owned arrays up front, then the identical tile loop.
    ///
    /// Deliberately *not* a single `mxfp4MM` over the concatenated matrix. The
    /// question this reference answers is "does paging the bytes change the
    /// answer", so the only thing that may differ between the two paths is
    /// where the bytes came from — same tile count, same per-tile operator,
    /// same concatenation, same evaluation cadence. A single whole-matrix call
    /// would additionally change the kernel's launch geometry, and a mismatch
    /// would no longer point at the pager. (That comparison is worth making
    /// too, and ``residentWholeMatrix(_:containerPath:)`` makes it, but it is
    /// not the gate.)
    public static func resident(
        _ x: MLXArray,
        containerPath: String,
        evalEveryK: Int = 2
    ) throws -> MLXArray {
        let layout = try MXFP4TileContainer.open(path: containerPath)
        try validate(x: x, layout: layout)
        let tiles = try loadResidentTiles(containerPath: containerPath, layout: layout)
        var evalNanoseconds: UInt64 = 0
        return try executeTiles(
            x: x, tileCount: layout.tileCount, evalEveryK: max(1, evalEveryK),
            evalNanoseconds: &evalNanoseconds, weightsForTile: { tiles[$0] })
    }

    /// The whole matrix as one `mxfp4MM`. Not the correctness gate — see
    /// ``resident(_:containerPath:evalEveryK:)`` — but a useful second opinion
    /// on whether the tiling itself perturbs anything.
    public static func residentWholeMatrix(
        _ x: MLXArray,
        containerPath: String
    ) throws -> MLXArray {
        let layout = try MXFP4TileContainer.open(path: containerPath)
        try validate(x: x, layout: layout)
        let tiles = try loadResidentTiles(containerPath: containerPath, layout: layout)
        let packed = concatenated(tiles.map(\.packed), axis: 0)
        let scales = concatenated(tiles.map(\.scales), axis: 0)
        let whole = try MXFP4Weights(packed: packed, scales: scales)
        let output = mxfp4MM(x, whole)
        output.eval()
        return output
    }

    // MARK: - Shared loop

    /// The one place the tile loop is written. Both the paged and the resident
    /// path go through it, so the two runs cannot drift apart in accumulation
    /// order, evaluation cadence, or concatenation strategy.
    private static func executeTiles(
        x: MLXArray,
        tileCount: Int,
        evalEveryK: Int,
        evalNanoseconds: inout UInt64,
        weightsForTile: (Int) throws -> MXFP4Weights
    ) rethrows -> MLXArray {
        var accumulated: MLXArray?
        var pending: [MLXArray] = []
        pending.reserveCapacity(evalEveryK)

        for index in 0..<tileCount {
            let weights = try weightsForTile(index)
            pending.append(mxfp4MM(x, weights))
            // `weights` goes out of scope here. The graph node keeps the
            // operands alive until it is evaluated; nothing is freed yet.

            if pending.count == evalEveryK || index == tileCount - 1 {
                let merged =
                    accumulated.map { concatenated([$0] + pending, axis: -1) }
                    ?? concatenated(pending, axis: -1)
                // Evaluating is what releases the tiles' slots. Without it the
                // loop would hold every tile it has ever expressed.
                let evalStart = MonotonicClock.now()
                merged.eval()
                evalNanoseconds &+= MonotonicClock.now() &- evalStart
                accumulated = merged
                pending.removeAll(keepingCapacity: true)
            }
        }
        // tileCount >= 1 is checked by the container, so this is never nil.
        return accumulated!
    }

    // MARK: - Adoption

    /// Wrap one filled slot as two `MLXArray`s without copying: packed weights
    /// at the slot's base, scales at the tile's aligned tail offset.
    ///
    /// The lease arrives with refcount 1 (this function's own). Two retains are
    /// taken *before* the arrays exist, because a finalizer can fire as soon as
    /// one does; the own reference is dropped last, leaving exactly the two
    /// holders MLX will release.
    private static func adopt(
        lease: SlotLease,
        layout: MXFP4TileContainer.Layout
    ) throws -> MXFP4Weights {
        try lease.retain()  // packed
        try lease.retain()  // scales

        let weights: MXFP4Weights
        do {
            weights = try MXFP4Weights.adopting(
                packedPointer: lease.slot.pointer,
                scalesPointer: lease.slot.pointer + layout.scaleOffsetInTile,
                outFeatures: layout.geometry.rows,
                inFeatures: layout.geometry.cols,
                packedFinalizer: { lease.release() },
                scalesFinalizer: { lease.release() })
        } catch {
            // Shapes are validated before the run starts, so this is
            // unreachable in practice. If it ever fires, the arrays that were
            // constructed own the pointers and will release their own holds;
            // only this function's reference has to be dropped here.
            lease.release()
            throw error
        }
        lease.release()  // the loop's own reference; MLX holds the other two
        return weights
    }

    // MARK: - Copy

    /// Copy one filled slot into fresh MLX-owned arrays and hand the slot back
    /// before any compute is expressed — spec §14.2's "copied" alternative to
    /// the adoption above, priced in `docs/experiments/2026-08-08-copy-vs-adopt.md`.
    ///
    /// The release is a `defer`, so it happens after `MXFP4Weights`' copying
    /// initializer has returned and on the throwing path too. That ordering is
    /// only safe because the copy is *eager*: `MLXArray(UnsafeBufferPointer:_:)`
    /// reaches `mlx_array_new_data`, which allocates and copies at construction
    /// (the adopting entry point is a different function taking a deleter). The
    /// debug builds check the claim rather than trusting it — a released slot is
    /// poisoned with `0xDE` and may be refilled immediately, so a lazy copy
    /// would corrupt the CI-sized bit-identity test loudly instead of subtly.
    private static func copyOut(
        lease: SlotLease,
        layout: MXFP4TileContainer.Layout
    ) throws -> MXFP4Weights {
        defer { lease.release() }
        let base = UnsafeRawPointer(lease.slot.pointer)
        return try MXFP4Weights(
            packedBytes: UnsafeRawBufferPointer(
                start: base, count: layout.geometry.packedBytes),
            scaleBytes: UnsafeRawBufferPointer(
                start: base + layout.scaleOffsetInTile, count: layout.geometry.scaleBytes),
            outFeatures: layout.geometry.rows,
            inFeatures: layout.geometry.cols)
    }

    // MARK: - Helpers

    private static func validate(x: MLXArray, layout: MXFP4TileContainer.Layout) throws {
        guard x.ndim == 2 else {
            throw MXFP4BridgeError.shapeMismatch(
                reason: "x must be [M, inFeatures]; got shape \(x.shape)")
        }
        guard x.shape[1] == layout.geometry.cols else {
            throw MXFP4BridgeError.shapeMismatch(
                reason: """
                    x has \(x.shape[1]) input features, the container's tiles have \
                    \(layout.geometry.cols)
                    """)
        }
    }

    /// Read every tile into MLX-owned arrays. Only for the reference path and
    /// for containers small enough to fit — which is the whole point of the
    /// paged path existing.
    private static func loadResidentTiles(
        containerPath: String,
        layout: MXFP4TileContainer.Layout
    ) throws -> [MXFP4Weights] {
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: containerPath))
        defer { try? handle.close() }

        var tiles: [MXFP4Weights] = []
        tiles.reserveCapacity(layout.tileCount)
        for index in 0..<layout.tileCount {
            try handle.seek(toOffset: layout.tileOffset(index))
            guard let data = try handle.read(upToCount: layout.tileStride),
                data.count == layout.tileStride
            else {
                throw TileContainerError.truncated(
                    path: containerPath,
                    expectedBytes: layout.totalBytes,
                    actualBytes: layout.tileOffset(index))
            }
            let weights = try data.withUnsafeBytes { bytes -> MXFP4Weights in
                let base = bytes.baseAddress!
                return try MXFP4Weights(
                    packedBytes: UnsafeRawBufferPointer(
                        start: base, count: layout.geometry.packedBytes),
                    scaleBytes: UnsafeRawBufferPointer(
                        start: base + layout.scaleOffsetInTile,
                        count: layout.geometry.scaleBytes),
                    outFeatures: layout.geometry.rows,
                    inFeatures: layout.geometry.cols)
            }
            tiles.append(weights)
        }
        return tiles
    }

    /// Wait, bounded, for every slot to come back.
    ///
    /// Releasing an adopted buffer is not guaranteed to happen on the thread
    /// that dropped the last Swift reference: MLX holds operand buffers until
    /// the command buffer that used them completes and can release from that
    /// completion handler (measured in M3). Reading the census the instant
    /// `eval` returns is therefore a race against the pager's own accounting.
    private static func waitForSlotsToDrain(pool: BufferPool, timeout: TimeInterval) {
        let deadline = Date().addingTimeInterval(timeout)
        while pool.census.inUse > 0 && Date() < deadline {
            usleep(500)
        }
    }
}
