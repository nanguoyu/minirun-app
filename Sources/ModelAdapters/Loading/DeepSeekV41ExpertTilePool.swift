import Darwin
import Foundation
import MLX
import MLXBridge
import StorageCore

/// A bounded, run-scoped residency for V4.1 routed-expert **pair tiles**.
///
/// ## Why a slot is a tile and not an expert
///
/// A V4.1 routed expert is *half* a published tile: `w1` is `[2304, 5120]` FP4,
/// whose scale region is 22.5 alignment units, so the container pairs two
/// experts per tile and `expert_order` maps an id to its tile and half
/// (ADR 0020). The read granularity, the recorded SHA-256 and therefore the
/// unit of residency are all the pair. A pool that held experts would pay for a
/// half it did not use, and a read-ahead that counted experts would issue one
/// read too many on the 3.92% of decode tokens whose routed set collides in a
/// tile (`docs/experiments/2026-09-11-v41-phase2-moe.md` §7).
///
/// So the key is `(unit, projection, tile)`, exactly as that record's first
/// open item asks, and the value is one `tileStride` byte image.
///
/// ## What it does with time
///
/// ``prefetch(unit:placements:)`` submits the reads a block is *certain* to
/// need — all three projections read the same experts, so every queued byte is
/// a byte the block will ask for — onto `queueDepth` workers, and returns
/// without waiting. ``withTile(_:_:)`` then either finds the tile resident,
/// waits for the read already in flight, or reads it on the calling thread.
/// The wait is bracketed as ``DeepSeekV4PhaseAccounting`` expert I/O wait, so a
/// pass can say how much of it the prefetch hid.
///
/// ## What it does with memory
///
/// `slots` tiles, never more. Eviction is least-recently-used among tiles no
/// caller is inside, which is what makes the residency a *stated* number the
/// memory dial can price rather than a cache that grows to the working set.
public final class DeepSeekV41ExpertTilePool: @unchecked Sendable {
    /// One residency, addressed the way the container addresses it.
    ///
    /// `half` is nil when the entry is a whole pair tile and the half's index
    /// when it is one expert's two byte runs. Which of the two a pool holds is
    /// decided once, by ``verifiesTileDigests``, and is a property of the pool
    /// rather than of a request — see ``Request``.
    public struct Key: Hashable, Sendable {
        public let unit: String
        public let projection: DeepSeekV41ExpertProjection
        public let tile: Int
        public let half: Int?

        public init(
            unit: String, projection: DeepSeekV41ExpertProjection, tile: Int,
            half: Int? = nil
        ) {
            self.unit = unit
            self.projection = projection
            self.tile = tile
            self.half = half
        }
    }

    /// What one read needs, taken from a ``DeepSeekV41ExpertPlacement`` so the
    /// pool never derives an offset of its own.
    ///
    /// ## Why a half is worth reading on its own
    ///
    /// A pair tile is two experts and a decode token draws six of 384, so it
    /// wants *both* halves of a tile only 3.92% of the time
    /// (`docs/experiments/2026-09-11-v41-phase2-moe.md` §7). Reading the pair
    /// therefore costs **twice** the expert bandwidth on 96% of draws: 9.0 GB a
    /// token against the 4.5 GB the design note priced.
    ///
    /// The pair is the read granularity for one reason only — the container's
    /// recorded SHA-256 covers the pair, so a half cannot be checked against
    /// it. A run holding a completed verification authority is not checking it
    /// (ADR 0013, and ADR 0021 §5 for V4.1), and for that run the two byte runs
    /// of one half are contiguous inside the packed region and inside the scale
    /// region, so a half is two `pread`s rather than one. A run that *is*
    /// verifying reads the pair, because that is what the digest covers.
    public struct Request: Sendable {
        public let key: Key
        public let fileReference: String
        /// The whole pair tile.
        public let tileOffset: UInt64
        public let tileStride: Int
        /// This expert's two byte runs inside it.
        public let packedOffsetInTile: Int
        public let packedBytes: Int
        public let scaleOffsetInTile: Int
        public let scaleBytes: Int

        public init(key: Key, placement: DeepSeekV41ExpertPlacement) {
            self.key = key
            self.fileReference = placement.fileReference
            self.tileOffset = placement.tileOffset
            self.tileStride = placement.tileStride
            self.packedOffsetInTile = placement.packedOffsetInTile
            self.packedBytes = placement.packedBytes
            self.scaleOffsetInTile = placement.scaleOffsetInTile
            self.scaleBytes = placement.scaleBytes
        }

        /// Bytes this request reads: the pair, or the half's two runs.
        func byteCount(wholeTile: Bool) -> Int {
            wholeTile ? tileStride : packedBytes + scaleBytes
        }
    }

    private final class Entry {
        var bytes: [UInt8]
        var readers = 0
        var stamp: UInt64
        init(bytes: [UInt8], stamp: UInt64) {
            self.bytes = bytes
            self.stamp = stamp
        }
    }

    public let slots: Int
    /// Bytes one slot holds — a whole pair tile, or one expert's half of one,
    /// depending on ``verifiesTileDigests``. Zero until the first read, because
    /// the width is a property of the unit and not of the pool's configuration.
    public private(set) var tileStrideBytes: Int = 0

    /// Whether an entry is a whole pair tile. False halves both the residency a
    /// slot costs and the bytes a token reads.
    public var holdsWholeTiles: Bool { verifiesTileDigests }

    private let fileAccess: ModelFileAccess
    private let verifiesTileDigests: Bool
    private let readAccounting: DeepSeekV4ReadAccounting?
    /// Not private: ``DeepSeekV41PooledExpertSource`` brackets the gather's own
    /// host work against the same accounting the pool charges its waits to, and
    /// a second reference threaded through the initializer could drift from it.
    let phaseAccounting: DeepSeekV4PhaseAccounting?
    private let condition = NSCondition()
    private var resident: [Key: Entry] = [:]
    private var inFlight: Set<Key> = []
    private var clock: UInt64 = 0
    private let queue: DispatchQueue
    private let semaphore: DispatchSemaphore
    private var shuttingDown = false

    /// Tiles read since the pool was opened, and tiles a caller found resident.
    public private(set) var tileReads = 0
    public private(set) var tileHits = 0

    public init(
        slots: Int,
        queueDepth: Int,
        fileAccess: ModelFileAccess,
        verifiesTileDigests: Bool,
        readAccounting: DeepSeekV4ReadAccounting? = nil,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil
    ) throws {
        guard slots >= 1, queueDepth >= 1 else {
            throw DeepSeekV41Error.experts(
                "an expert tile pool needs at least one slot and one worker; got "
                    + "\(slots) and \(queueDepth)")
        }
        self.slots = slots
        self.fileAccess = fileAccess
        self.verifiesTileDigests = verifiesTileDigests
        self.readAccounting = readAccounting
        self.phaseAccounting = phaseAccounting
        self.queue = DispatchQueue(
            label: "minirun.v41-expert-pool", qos: .userInitiated, attributes: .concurrent)
        self.semaphore = DispatchSemaphore(value: queueDepth)
    }

    /// The residency this pool reserves once it knows its stride.
    public var budgetBytes: UInt64 {
        UInt64(max(0, tileStrideBytes)).multipliedReportingOverflow(
            by: UInt64(slots)
        ).partialValue
    }

    /// Queue reads for tiles that are neither resident nor already in flight.
    ///
    /// Returns without waiting for any of them. Never queues more than the pool
    /// can hold: a read-ahead deeper than the residency would evict the tile the
    /// consumer is about to ask for, which is slower than not prefetching.
    public func prefetch(_ requests: [Request]) {
        condition.lock()
        guard !shuttingDown else {
            condition.unlock()
            return
        }
        var queued = [Request]()
        var budget = slots - 1
        for request in requests where budget > 0 {
            if resident[request.key] != nil || inFlight.contains(request.key) { continue }
            inFlight.insert(request.key)
            queued.append(request)
            budget -= 1
        }
        condition.unlock()

        for request in queued {
            queue.async { [weak self] in
                guard let self else { return }
                self.semaphore.wait()
                defer { self.semaphore.signal() }
                let bytes = try? self.readTile(request)
                self.condition.lock()
                self.inFlight.remove(request.key)
                if let bytes { self.publish(request.key, bytes: bytes) }
                self.condition.broadcast()
                self.condition.unlock()
            }
        }
    }

    /// Run `body` with the tile's bytes, reading or waiting as needed.
    ///
    /// The tile is pinned for the duration, so a concurrent prefetch cannot
    /// evict what a gather is reading out of.
    public func withTile<R>(
        _ request: Request, _ body: (UnsafeRawBufferPointer) throws -> R
    ) throws -> R {
        let started = MonotonicClock.now()
        condition.lock()
        while inFlight.contains(request.key), resident[request.key] == nil {
            condition.wait()
        }
        if let entry = resident[request.key] {
            entry.readers += 1
            clock &+= 1
            entry.stamp = clock
            tileHits += 1
            condition.unlock()
            phaseAccounting?.recordExpertIOWait(
                nanoseconds: MonotonicClock.nanoseconds(MonotonicClock.seconds(since: started)))
            defer { release(request.key) }
            return try entry.bytes.withUnsafeBytes(body)
        }
        inFlight.insert(request.key)
        condition.unlock()

        let bytes: [UInt8]
        do {
            bytes = try readTile(request)
        } catch {
            condition.lock()
            inFlight.remove(request.key)
            condition.broadcast()
            condition.unlock()
            throw error
        }
        condition.lock()
        inFlight.remove(request.key)
        publish(request.key, bytes: bytes)
        let entry = resident[request.key]
        entry?.readers += 1
        condition.broadcast()
        condition.unlock()
        phaseAccounting?.recordExpertIOWait(
            nanoseconds: MonotonicClock.nanoseconds(MonotonicClock.seconds(since: started)))
        defer { if entry != nil { release(request.key) } }
        // `publish` may have refused to keep the tile (a pool of one whose
        // single slot is held); the bytes are still this call's to use.
        return try (entry?.bytes ?? bytes).withUnsafeBytes(body)
    }

    /// Drop every resident tile. A pool that finished a run must not still hold
    /// its slots — the same rule V4's expert residency follows at teardown.
    public func shutdown() {
        condition.lock()
        shuttingDown = true
        resident.removeAll(keepingCapacity: false)
        spareOperandBuffers.removeAll(keepingCapacity: false)
        condition.broadcast()
        condition.unlock()
    }

    // MARK: - Gather operand buffers

    /// Page-aligned allocations a gather operand adopted and MLX has since
    /// finalized, kept for the next operand of the same width.
    ///
    /// ## Why a free list and not a ring
    ///
    /// A fresh 37.6 MB allocation is `mmap`'d, so the `memcpy` that fills it
    /// faults in every one of its 2,304 pages — which is most of what filling
    /// it costs. Reusing the allocation pays that once.
    ///
    /// It is a *free list* and never a bounded ring, deliberately. MLX graphs
    /// are lazy and an operand lives until the graph that used it is evaluated,
    /// so the number alive at once is a property of where the run's evals fall
    /// and not of this pool. A ring would have to block when it ran out, which
    /// is the deadlock class ``PagedReadPlan`` exists to refuse for V4; this
    /// list simply allocates when it is empty. What it bounds is only how many
    /// allocations are *retained* after their arrays died — ``operandSpareCap``
    /// of them, so the residency this adds is stated and small rather than the
    /// high-water mark of a lazy graph.
    private var spareOperandBuffers: [Int: [AlignedBuffer]] = [:]

    /// Retained spare allocations, per width. Four is three projections of one
    /// token plus one, which is the concurrency a decode step actually reaches;
    /// a run whose graph holds more allocates them and hands them back.
    public static let operandSpareCap = 4

    /// An allocation of `count` page-aligned bytes, reused when one is spare.
    func takeOperandBuffer(count: Int) throws -> AlignedBuffer {
        condition.lock()
        if var spares = spareOperandBuffers[count], let buffer = spares.popLast() {
            spareOperandBuffers[count] = spares
            condition.unlock()
            return buffer
        }
        condition.unlock()
        return try AlignedBuffer(count: count)
    }

    /// Hand one back. Called from an MLX finalizer, which may be any thread.
    func returnOperandBuffer(_ buffer: AlignedBuffer) {
        condition.lock()
        defer { condition.unlock() }
        guard !shuttingDown else { return }
        var spares = spareOperandBuffers[buffer.count] ?? []
        guard spares.count < Self.operandSpareCap else { return }
        spares.append(buffer)
        spareOperandBuffers[buffer.count] = spares
    }

    // MARK: - Internals

    private func release(_ key: Key) {
        condition.lock()
        if let entry = resident[key] { entry.readers = max(0, entry.readers - 1) }
        condition.broadcast()
        condition.unlock()
    }

    /// Called with the lock held.
    private func publish(_ key: Key, bytes: [UInt8]) {
        if tileStrideBytes == 0 { tileStrideBytes = bytes.count }
        if resident[key] != nil { return }
        while resident.count >= slots {
            guard let victim = resident
                .filter({ $0.value.readers == 0 })
                .min(by: { $0.value.stamp < $1.value.stamp })?.key
            else { return }
            resident.removeValue(forKey: victim)
        }
        clock &+= 1
        resident[key] = Entry(bytes: bytes, stamp: clock)
    }

    /// The bytes one entry holds: a whole pair tile when this pool verifies
    /// digests, and one expert's packed run followed by its scale run when it
    /// does not.
    private func readTile(_ request: Request) throws -> [UInt8] {
        let length = request.byteCount(wholeTile: verifiesTileDigests)
        var buffer = [UInt8](repeating: 0, count: length)
        try fileAccess.withDescriptor(request.fileReference) { descriptor in
            let layout = try QuantizedTileContainer.open(
                fileDescriptor: descriptor, path: request.fileReference)
            guard layout.tileStride == request.tileStride else {
                throw DeepSeekV41Error.experts(
                    "\(request.fileReference) no longer describes the tiles the manifest "
                        + "was reconciled against")
            }
            try buffer.withUnsafeMutableBytes { destination in
                guard let base = destination.baseAddress else { return }
                if verifiesTileDigests {
                    try Self.readExactly(
                        descriptor: descriptor, reference: request.fileReference,
                        offset: request.tileOffset, length: request.tileStride,
                        destination: base)
                } else {
                    try Self.readExactly(
                        descriptor: descriptor, reference: request.fileReference,
                        offset: request.tileOffset + UInt64(request.packedOffsetInTile),
                        length: request.packedBytes, destination: base)
                    try Self.readExactly(
                        descriptor: descriptor, reference: request.fileReference,
                        offset: request.tileOffset + UInt64(request.scaleOffsetInTile),
                        length: request.scaleBytes,
                        destination: base + request.packedBytes)
                }
            }
            if verifiesTileDigests {
                try buffer.withUnsafeBytes { bytes in
                    try QuantizedTileContainer.verifyTileDigest(
                        bytes, layout: layout, tile: request.key.tile)
                }
            }
        }
        condition.lock()
        tileReads += 1
        condition.unlock()
        readAccounting?.recordExpert(UInt64(length), didOverflow: false)
        return buffer
    }

    private static func readExactly(
        descriptor: Int32, reference: String, offset: UInt64, length: Int,
        destination: UnsafeMutableRawPointer
    ) throws {
        guard length > 0, let start = off_t(exactly: offset) else {
            throw DeepSeekV41Error.experts(
                "\(reference) has an unrepresentable expert tile range")
        }
        var completed = 0
        while completed < length {
            let result = pread(
                descriptor, destination + completed, length - completed,
                start + off_t(completed))
            if result < 0 {
                if errno == EINTR { continue }
                throw StorageCoreError.posix(
                    operation: "pread", path: reference, code: errno)
            }
            guard result > 0 else {
                throw StorageCoreError.shortTransfer(
                    operation: "pread", path: reference,
                    offset: offset + UInt64(completed),
                    expected: length - completed, actual: 0)
            }
            completed += result
        }
    }
}

/// One block's routed experts, served out of a shared pool.
///
/// The pool is the run's; this is the per-block view of it, which is what
/// ``DeepSeekV41MoE`` is handed. Every read goes through the pool, so a tile a
/// previous block or a previous token already brought in is not read again, and
/// the residency stays the pool's stated slots rather than the union of forty
/// blocks' working sets.
public final class DeepSeekV41PooledExpertSource: DeepSeekV41RoutedExpertSource {
    private let artifact: DeepSeekV41BlockArtifact
    private let pool: DeepSeekV41ExpertTilePool
    private let adoptsOperands: Bool
    /// Called before each gather operand is allocated.
    ///
    /// The seam a memory budget needs and did not have. A V4.1 prefill's widest
    /// allocation is this one, it happens three times a token inside a block,
    /// and the runner's own footprint sampler runs at block *boundaries* — that
    /// is, after `withBlock`'s `autoreleasepool` and reclaim have already
    /// returned everything the block was holding. So the sampler saw the
    /// narrowest moment of every block and never the widest, and a run could
    /// grow past its declared ceiling and be killed by the OS without a single
    /// sample above the budget. Checking here is what makes the refusal fire
    /// inside a prefill; the most a run can add between two checks is one
    /// operand.
    private let operandGuard: (() throws -> Void)?

    public init(
        artifact: DeepSeekV41BlockArtifact, pool: DeepSeekV41ExpertTilePool,
        adoptsOperands: Bool = true,
        operandGuard: (() throws -> Void)? = nil
    ) {
        self.artifact = artifact
        self.pool = pool
        self.adoptsOperands = adoptsOperands
        self.operandGuard = operandGuard
    }

    public func prefetch(_ experts: [Int]) throws {
        var requests = [DeepSeekV41ExpertTilePool.Request]()
        // All three projections, because all three read the same experts and a
        // block asks for every one of them before it finishes.
        for projection in DeepSeekV41ExpertProjection.allCases {
            for expert in experts {
                requests.append(try request(expert, projection: projection))
            }
        }
        measuringPhase(
            pool.phaseAccounting,
            excludingGPUBoundaryFrom: pool.phaseAccounting?.recordExpertPrefetch(nanoseconds:)
        ) {
            pool.prefetch(requests)
        }
    }

    /// One expert's residency request, keyed the way this pool holds it: by the
    /// pair tile when it holds pairs, and by the half when it holds halves.
    private func request(
        _ expert: Int, projection: DeepSeekV41ExpertProjection
    ) throws -> DeepSeekV41ExpertTilePool.Request {
        let placement = try artifact.expert(expert, projection: projection)
        return .init(
            key: .init(
                unit: artifact.unitID, projection: projection, tile: placement.tile,
                half: pool.holdsWholeTiles ? nil : placement.half),
            placement: placement)
    }

    public func stack(
        _ experts: [Int], projection: DeepSeekV41ExpertProjection
    ) throws -> MXFP4Weights {
        guard !experts.isEmpty else {
            throw DeepSeekV41Error.experts("expert source: asked for no experts")
        }
        try operandGuard?()
        let placements = try experts.map { try artifact.expert($0, projection: projection) }
        let first = placements[0]
        guard placements.allSatisfy({
            $0.fileReference == first.fileReference && $0.rows == first.rows
                && $0.columns == first.columns && $0.packedBytes == first.packedBytes
                && $0.scaleBytes == first.scaleBytes && $0.tileStride == first.tileStride
        }) else {
            throw DeepSeekV41Error.experts(
                "expert source: one projection's experts do not share a container geometry")
        }

        // The collision rate, measured rather than drawn from a urn. One
        // record per token per block: `stack` is called with one token's own
        // routed set, and `.gate` is picked so the three projections of the
        // same set count once. Phase 3's eighth open item.
        let accounting = pool.phaseAccounting
        if projection == .gate, let accounting {
            var members = [Int: Int]()
            for placement in placements { members[placement.tile, default: 0] += 1 }
            accounting.recordExpertRoutedSet(
                collisions: members.values.filter { $0 > 1 }.count)
        }

        // Ascending by tile, so the reads walk the file forward whatever order
        // the router named the experts in. A pair a token drew both halves of
        // is one entry when the pool holds pairs and two when it holds halves;
        // either way the pool decides, and this loop only asks.
        let order = placements.indices.sorted(by: {
            (placements[$0].tile, placements[$0].half)
                < (placements[$1].tile, placements[$1].half)
        })
        // Every nanosecond of the fill, charged once. A bracket per half would
        // be seven events for one operand and would say nothing the operand
        // count does not already say.
        var copyNanoseconds: UInt64 = 0

        if adoptsOperands {
            let layout = try StackLayout(experts: experts.count, placement: first)
            let buffer = try pool.takeOperandBuffer(count: layout.totalBytes)
            let base = buffer.pointer
            for slot in order {
                let placement = placements[slot]
                let whole = pool.holdsWholeTiles
                try pool.withTile(try request(experts[slot], projection: projection)) { bytes in
                    let start = MonotonicClock.now()
                    defer { copyNanoseconds &+= MonotonicClock.now() &- start }
                    let packedStart = whole ? placement.packedOffsetInTile : 0
                    let scaleStart = whole ? placement.scaleOffsetInTile : placement.packedBytes
                    guard let source = bytes.baseAddress else { return }
                    (base + slot * layout.packedStride).copyMemory(
                        from: source + packedStart, byteCount: placement.packedBytes)
                    (base + layout.scaleOffset + slot * layout.scaleStride).copyMemory(
                        from: source + scaleStart, byteCount: placement.scaleBytes)
                }
            }
            accounting?.recordExpertStackCopy(nanoseconds: copyNanoseconds)

            let buildStart = MonotonicClock.now()
            defer {
                accounting?.recordExpertOperandBuild(
                    nanoseconds: MonotonicClock.now() &- buildStart, adopted: true)
            }
            // Two arrays, one allocation, and the allocation lives exactly as
            // long as the two of them: MLX calls each finalizer exactly once
            // (ADR 0002), and the second call is what frees the buffer. This is
            // V4's `transferMode: .adopt` in the shape a *stack* can take —
            // see ``StackLayout`` for why a pool entry cannot be adopted
            // directly.
            let holder = AdoptedStack(buffer: buffer, holds: 2, pool: pool)
            return try MXFP4Weights.adopting(
                packedPointer: base,
                scalesPointer: base + layout.scaleOffset,
                experts: experts.count == 1 ? nil : experts.count,
                outFeatures: first.rows, inFeatures: first.columns,
                packedFinalizer: { holder.release() },
                scalesFinalizer: { holder.release() })
        }

        let allocationStart = MonotonicClock.now()
        var packed = [UInt8](repeating: 0, count: experts.count * first.packedBytes)
        var scales = [UInt8](repeating: 0, count: experts.count * first.scaleBytes)
        copyNanoseconds &+= MonotonicClock.now() &- allocationStart
        for slot in order {
            let placement = placements[slot]
            let whole = pool.holdsWholeTiles
            try pool.withTile(try request(experts[slot], projection: projection)) { bytes in
                // Inside the body, so the wait for the tile — which the pool
                // charges to `expertIOWaitSeconds` before it calls this — is
                // not in the copy's own bracket.
                let copyStart = MonotonicClock.now()
                defer { copyNanoseconds &+= MonotonicClock.now() &- copyStart }
                let packedStart = whole ? placement.packedOffsetInTile : 0
                let scaleStart = whole ? placement.scaleOffsetInTile : placement.packedBytes
                packed.replaceSubrange(
                    (slot * first.packedBytes)..<((slot + 1) * first.packedBytes),
                    with: UnsafeRawBufferPointer(
                        rebasing: bytes[packedStart..<(packedStart + placement.packedBytes)]))
                scales.replaceSubrange(
                    (slot * first.scaleBytes)..<((slot + 1) * first.scaleBytes),
                    with: UnsafeRawBufferPointer(
                        rebasing: bytes[scaleStart..<(scaleStart + placement.scaleBytes)]))
            }
        }
        accounting?.recordExpertStackCopy(nanoseconds: copyNanoseconds)

        let buildStart = MonotonicClock.now()
        defer {
            accounting?.recordExpertOperandBuild(
                nanoseconds: MonotonicClock.now() &- buildStart, adopted: false)
        }
        return try packed.withUnsafeBytes { packedBytes in
            try scales.withUnsafeBytes { scaleBytes in
                try MXFP4Weights(
                    packedBytes: packedBytes, scaleBytes: scaleBytes,
                    experts: experts.count == 1 ? nil : experts.count,
                    outFeatures: first.rows, inFeatures: first.columns)
            }
        }
    }

    /// Where one gather operand's two regions sit inside one page-aligned
    /// allocation.
    ///
    /// ## Why a pool entry cannot be adopted directly
    ///
    /// V4 adopts a pool slot because a V4 gather's operand *is* one slot: one
    /// tile, one matrix. A V4.1 gather's operand is a **stack** — `gather_qmm`
    /// indexes experts along a leading axis, so `packed` must be one array of
    /// `[experts, rows, cols / 8]` and the six experts must be contiguous in
    /// it. The six halves a token routes to live in six independent pool
    /// entries that the router chose at this token and nothing can make
    /// adjacent. So the half's own two byte runs, which *are* contiguous inside
    /// the entry, still cannot be the array MLX multiplies.
    ///
    /// The cheapest correct shape is therefore one adopted allocation per
    /// stack, filled by six `memcpy`s out of the entries. Against the copying
    /// path that removes three of the four passes over the bytes — the Swift
    /// array's zero-fill, its bounds-checked fill, and MLX's own copy into
    /// array storage — and leaves one.
    ///
    /// Both regions start on a page boundary because MLX wraps an adopted
    /// pointer for the Metal backend, which requires page alignment
    /// (``AffineWeights/adopting(...)`` states the same requirement for the
    /// tile container's three sub-regions).
    struct StackLayout {
        let packedStride: Int
        let scaleStride: Int
        let scaleOffset: Int
        let totalBytes: Int

        init(experts: Int, placement: DeepSeekV41ExpertPlacement) throws {
            guard experts >= 1, placement.packedBytes > 0, placement.scaleBytes > 0 else {
                throw DeepSeekV41Error.experts(
                    "expert source: a gather operand needs at least one expert with "
                        + "a non-empty packed and scale run")
            }
            packedStride = placement.packedBytes
            scaleStride = placement.scaleBytes
            let page = AlignedBuffer.pageSize
            func roundUp(_ value: Int) -> Int { (value + page - 1) / page * page }
            scaleOffset = roundUp(experts * packedStride)
            totalBytes = scaleOffset + roundUp(experts * scaleStride)
        }
    }

    /// One allocation, two `MLXArray`s, freed when both finalizers have fired.
    ///
    /// The counter is V4's `SlotLease` retain/release in miniature and exists
    /// for the same reason (ADR 0016's lifetime rule): a finalizer can run on a
    /// Metal completion thread after `eval` returns, the two can run in either
    /// order, and the buffer must outlive both.
    private final class AdoptedStack: @unchecked Sendable {
        private let lock = NSLock()
        private var buffer: AlignedBuffer?
        private var holds: Int
        private let pool: DeepSeekV41ExpertTilePool

        init(buffer: AlignedBuffer, holds: Int, pool: DeepSeekV41ExpertTilePool) {
            self.buffer = buffer
            self.holds = holds
            self.pool = pool
        }

        func release() {
            lock.lock()
            holds -= 1
            let spare = holds <= 0 ? buffer : nil
            if holds <= 0 { buffer = nil }
            lock.unlock()
            // Outside the lock: the pool takes its own, and a finalizer can be
            // on a Metal completion thread that must not wait on this one.
            if let spare { pool.returnOperandBuffer(spare) }
        }
    }
}
