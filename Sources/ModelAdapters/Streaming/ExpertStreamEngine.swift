import Foundation
import MLX
import MLXBridge
import StorageCore

/// A simulated decode loop over real K3 MoE layers, streamed from storage.
///
/// This is the skeleton M9 reuses, so the shape of it matters as much as the
/// numbers it produces. Per token, per layer, it does what a decode step does:
///
/// 1. read the router, in row bands, and select 16 of 896 experts;
/// 2. schedule those experts' reads *immediately*, so they overlap step 3;
/// 3. read the latent down-projection and compute the latent activation;
/// 4. read the rest of the deterministic stream — attention bytes go through
///    the pager and are **not** computed with, the shared expert is;
/// 5. consume the expert bytes as they arrive, situ, weighted-sum;
/// 6. read the latent norm and up-projection and finish the block.
///
/// ## What is and is not computed
///
/// Attention weights are read and dropped. Streaming them is what makes the
/// per-token byte budget real — they are 70% of the deterministic class — but
/// computing KDA or MLA at flagship dims would be M9's job and would not change
/// a single byte of I/O. The record must say so, and the metrics carry the
/// role split so it cannot be forgotten.
///
/// ## Two pools, on purpose
///
/// The expert stream is routing-dependent and reusable across tokens; the
/// deterministic stream is read in full every token and has nothing to reuse.
/// Giving them one pool would let a 1.2 GiB once-per-token sweep evict the
/// expert cache the experiment is trying to measure — and the "cache budget"
/// column would then describe neither. Two pools, two stated budgets, summed
/// in the report.
public final class ExpertStreamEngine {

    public enum CacheArm: String, Codable, Sendable, CaseIterable {
        case none = "no-cache"
        case lru = "lru"
        case pinnedHotSet = "pinned-hot-set"
    }

    /// How the routed-expert arithmetic is handed to the kernel.
    ///
    /// These are two programs, not two settings of one, and separating them
    /// from ``Configuration/expertsPerDispatch`` matters because batching does
    /// **two** independent things and they do not cost the same:
    ///
    /// - it fuses an expert's 13 row bands into one operand, which removes
    ///   dispatches and costs nothing in overlap — the bands of one expert
    ///   arrive in one read either way;
    /// - it stacks several *experts* into one operand, which removes more
    ///   dispatches but makes the consumer wait for the whole chunk before it
    ///   can compute with any of it, where the per-expert loop overlaps the
    ///   arrival of expert *n+1* with the arithmetic on expert *n*.
    ///
    /// Folding both into a width would make the second effect unmeasurable:
    /// there would be no arm with the first and not the second, and the
    /// measurement in `2026-08-08-partial-k3-streaming.md` turns on exactly
    /// that arm existing.
    public enum GatherProgram: String, Codable, Sendable, CaseIterable {
        /// One dispatch per (expert, row band). What M8 measured, kept intact
        /// so the batched form has an unchanged baseline to be read against.
        case perExpertPerBand = "per-expert"
        /// One dispatch per chunk of ``Configuration/expertsPerDispatch``
        /// experts, with each expert's row bands fused. At a width of 1 this is
        /// band fusion alone.
        case batched
    }

    public struct Configuration: Sendable {
        public var granularity: FetchGranularity
        public var cacheArm: CacheArm
        /// Expert-cache budget, counted in whole experts. The pool's byte
        /// budget is this times the bytes one expert occupies, so the three
        /// arms and both granularities are compared at the same memory.
        public var cacheBudgetExperts: Int
        public var queueDepth: Int
        /// Per reader. There are three expert readers per layer, so the pool
        /// must hold `3 * readAhead + 1` slots before anything else.
        public var readAhead: Int
        public var noCache: Bool
        public var deterministicSlotCount: Int
        public var streamDeterministic: Bool
        public var computeDeterministic: Bool
        public var slotAcquireTimeout: TimeInterval?
        public var mlxCacheLimitBytes: Int
        /// Tokens per reported throughput window. Three windows (first, middle,
        /// last) are what a stability claim needs.
        public var windowTokens: Int
        public var hiddenStateSeed: UInt64
        /// Which of the two programs runs. See ``GatherProgram``.
        public var gatherProgram: GatherProgram
        /// Experts covered by one `gatherQuantizedMM` under
        /// ``GatherProgram/batched``; ignored by the other program.
        ///
        /// Per layer, a decode step issues `3 × ceil(16 / k)` dispatches: 48 at
        /// `k = 1` (band fusion only), 3 at the top-k. Against 208 for
        /// ``GatherProgram/perExpertPerBand``.
        ///
        /// This is a claim on the pool as well as a dispatch count: every
        /// expert in a dispatch is leased for the whole of it, and
        /// ``ExpertStreamEngine/peakSlotsInFlight(readerCount:readAhead:slotsPerDispatch:)``
        /// is stated in exactly that quantity. Lowering it trades dispatches
        /// back for slots — and, measurably, for overlap.
        ///
        /// Clamped at run time to the number of experts the router selects: a
        /// dispatch cannot lease more experts than there are.
        public var expertsPerDispatch: Int

        /// Open readers held at once, per stream class. See
        /// ``BoundedReaderCache``: a reader is a descriptor plus `queueDepth + 1`
        /// threads, and the previous unbounded behaviour reached 372 readers and
        /// ~1,488 threads at 93 layers. This bounds both by a stated number
        /// instead of by the model's depth. Floored at
        /// ``BoundedReaderCache/minimumLimit`` so the least-recently-used entry
        /// is always idle.
        public var maxLiveReaders: Int

        public init(
            granularity: FetchGranularity = .perExpert,
            cacheArm: CacheArm = .none,
            cacheBudgetExperts: Int = 8,
            queueDepth: Int = 3,
            readAhead: Int = 2,
            noCache: Bool = true,
            deterministicSlotCount: Int = 6,
            streamDeterministic: Bool = true,
            computeDeterministic: Bool = true,
            slotAcquireTimeout: TimeInterval? = 300,
            mlxCacheLimitBytes: Int = 64 << 20,
            windowTokens: Int = 0,
            hiddenStateSeed: UInt64 = 0x4B38_0001,
            // Stays the program M8 measured. Batching was implemented, gated
            // and measured on the real layers, and it does not pay here: the
            // dispatch count falls 69× and throughput does not move (best
            // 1.03× at width 2, inside the run-to-run spread; 0.87× at the
            // top-k). This workload is I/O-bound — 46% I/O wait, and a
            // deterministic stream 4.12:1 over the expert stream — so
            // dispatches were never what was costing it. Flipping the default
            // on that evidence would be claiming a speedup the measurement does
            // not support, and at the top-k it would ship a 13% regression and
            // a 2.1× pool budget with it.
            gatherProgram: GatherProgram = .perExpertPerBand,
            expertsPerDispatch: Int = 1,
            // Two layers' worth. One layer's three projections are what can be
            // active at once, so this keeps the previous layer's readers warm
            // across a layer boundary without letting the count track depth.
            maxLiveReaders: Int = 6
        ) {
            self.gatherProgram = gatherProgram
            self.expertsPerDispatch = max(1, expertsPerDispatch)
            self.maxLiveReaders = maxLiveReaders
            self.granularity = granularity
            self.cacheArm = cacheArm
            self.cacheBudgetExperts = cacheBudgetExperts
            self.queueDepth = queueDepth
            self.readAhead = readAhead
            self.noCache = noCache
            self.deterministicSlotCount = deterministicSlotCount
            self.streamDeterministic = streamDeterministic
            self.computeDeterministic = computeDeterministic
            self.slotAcquireTimeout = slotAcquireTimeout
            self.mlxCacheLimitBytes = mlxCacheLimitBytes
            self.windowTokens = windowTokens
            self.hiddenStateSeed = hiddenStateSeed
        }

        public var label: String {
            "\(granularity.rawValue)/\(cacheArm.rawValue)/\(cacheBudgetExperts)e"
                + "/qd\(queueDepth)/ra\(readAhead)\(noCache ? "/nocache" : "/cached")"
                + "/\(gatherProgram.rawValue)"
                + (gatherProgram == .batched ? "/x\(expertsPerDispatch)" : "")
        }
    }

    /// Model constants the engine needs. Read from the containers where
    /// possible; the rest are the checkpoint's own config values.
    public struct ModelShape: Sendable {
        public var hiddenSize: Int
        public var latentSize: Int
        public var expertsPerToken: Int
        public var situBeta: Float
        public var situLinearBeta: Float?
        public var rmsNormEps: Float
        public var renormalize: Bool
        public var routedScalingFactor: Float

        public init(
            hiddenSize: Int = 7168,
            latentSize: Int = 3584,
            expertsPerToken: Int = 16,
            situBeta: Float = 4.0,
            situLinearBeta: Float? = 25.0,
            rmsNormEps: Float = 1e-5,
            renormalize: Bool = true,
            routedScalingFactor: Float = 1.0
        ) {
            self.hiddenSize = hiddenSize
            self.latentSize = latentSize
            self.expertsPerToken = expertsPerToken
            self.situBeta = situBeta
            self.situLinearBeta = situLinearBeta
            self.rmsNormEps = rmsNormEps
            self.renormalize = renormalize
            self.routedScalingFactor = routedScalingFactor
        }
    }

    // MARK: - State

    public let layers: [FlagshipLayerStreams]
    public let configuration: Configuration
    public let shape: ModelShape

    private let expertPool: BufferPool
    private let deterministicPool: BufferPool
    private var expertReaders: BoundedReaderCache
    private var deterministicReaders: BoundedReaderCache

    private let expertSlotBytes: Int
    private let deterministicSlotBytes: Int
    private var pinnedExpertCount = 0
    private var pinnedRegionCount = 0

    /// ``Configuration/expertsPerDispatch`` clamped to the top-k. One means the
    /// per-expert program.
    public let dispatchWidth: Int
    /// `gatherQuantizedMM` calls issued on the routed-expert path — dispatches,
    /// not experts and not reads. The number this restructuring is about, so it
    /// is counted rather than derived from the configuration.
    private var expertGatherDispatches = 0

    private var expertBytesRequested: UInt64 = 0
    private var deterministicBytesRequested: UInt64 = 0
    private var expertByteAccountingOverflowed = false
    private var deterministicByteAccountingOverflowed = false
    private var acquireWaitNanoseconds: UInt64 = 0
    private var computeNanoseconds: UInt64 = 0
    private var timingAccountingOverflowed = false
    private var notes: [String] = []
    /// What the run actually routed to, layer by layer. For a pre-selected
    /// trace this reproduces the trace; for a hidden-state trace it is the only
    /// record of what the real router chose, and therefore the only way to say
    /// what locality the checkpoint's own router produces.
    private var realizedSelections: [Int: [[Int]]] = [:]

    /// Phase assignment of every deterministic unit, precomputed once.
    private struct DeterministicSchedule {
        let router: [FlagshipLayerStreams.DeterministicUnit]
        let routerBias: FlagshipLayerStreams.DeterministicUnit?
        let latentDown: [FlagshipLayerStreams.DeterministicUnit]
        let overlap: [FlagshipLayerStreams.DeterministicUnit]
        let latentNorm: FlagshipLayerStreams.DeterministicUnit?
        let latentUp: [FlagshipLayerStreams.DeterministicUnit]
        let sharedGate: [FlagshipLayerStreams.DeterministicUnit]
        let sharedUp: [FlagshipLayerStreams.DeterministicUnit]
        let sharedDown: [FlagshipLayerStreams.DeterministicUnit]
    }
    private var schedules: [Int: DeterministicSchedule] = [:]

    // MARK: - Construction

    public init(
        layers: [FlagshipLayerStreams],
        configuration: Configuration,
        shape: ModelShape = ModelShape(),
        trace: RoutingTrace? = nil
    ) throws {
        guard !layers.isEmpty else {
            throw TinyK3Error.configuration("the engine needs at least one layer")
        }
        self.layers = layers
        self.configuration = configuration
        self.shape = shape

        self.expertSlotBytes = layers.map {
            $0.maximumRegionBytes(granularity: configuration.granularity)
        }.max() ?? 0
        guard expertSlotBytes > 0 else {
            throw TinyK3Error.configuration("expert containers describe a zero-byte read unit")
        }
        self.deterministicSlotBytes =
            layers.flatMap { $0.deterministic.units.map(\.length) }.max() ?? 0

        // Three expert readers per layer share one pool and each may hold
        // `readAhead` slots. The deadlock argument in `PagedReadPlan` assumes
        // ONE reader, so the plan is built with the aggregate window: unless
        // every reader's in-flight window plus the tiles the consumer holds
        // fits, a reader can be waiting for a slot only the consumer can free
        // while the consumer waits for a tile only that reader can deliver.
        let readerCount = ExpertProjection.allCases.count
        let aggregateReadAhead = readerCount * configuration.readAhead
        let bytesPerExpert = layers.map(\.bytesPerExpert).max() ?? expertSlotBytes
        let budgetBytes = configuration.cacheBudgetExperts * bytesPerExpert

        // A dispatch can never lease more experts than the router selects, so
        // the configured width is a ceiling and the top-k is the other one.
        // Clamping here rather than at the call site is what keeps the pool
        // sized for what will actually happen: a 16-wide default must not size
        // a 2-expert model's pool for 16.
        self.dispatchWidth = max(1, min(configuration.expertsPerDispatch, shape.expertsPerToken))
        // Regions one expert's projection occupies. `perExpert` reads the whole
        // projection in one, so one slot; `perTile` takes a slot per row band,
        // and a batch of them multiplies.
        let regionsPerProjection: Int
        switch configuration.granularity {
        case .perExpert:
            regionsPerProjection = 1
        case .perTile:
            regionsPerProjection =
                layers.flatMap { layer in
                    ExpertProjection.allCases.compactMap { try? layer.container($0).tilesPerExpert }
                }.max() ?? 1
        }
        // The per-expert program leases exactly one region at a time whichever
        // granularity it runs at — one whole expert's slot at `perExpert`, one
        // band's at `perTile` — because it evaluates and releases before asking
        // for the next one. Only the batched program holds a dispatch's worth,
        // so only it multiplies.
        let slotsPerDispatch =
            configuration.gatherProgram == .batched
            ? self.dispatchWidth * regionsPerProjection : 1

        let inFlight = Self.peakSlotsInFlight(
            readerCount: readerCount, readAhead: configuration.readAhead,
            slotsPerDispatch: slotsPerDispatch)
        let floor = Self.slotsForPeak(inFlight)
        let slotCount = max(floor, budgetBytes / expertSlotBytes)
        // `evalEveryK` is `slotsPerDispatch` and not 1: a batched gather leases
        // every operand of a dispatch before evaluating any of them, and that
        // is the quantity `PagedReadPlan`'s invariant is stated in. Batching is
        // a change to the budget arithmetic, not only to the dispatch count.
        _ = try PagedReadPlan(
            slotCount: slotCount, readAhead: aggregateReadAhead, evalEveryK: slotsPerDispatch)

        var policy = CachePolicy.none
        var pinnedRegions = 0
        var pinnedExperts = 0
        switch configuration.cacheArm {
        case .none:
            policy = .none
        case .lru:
            policy = .leastRecentlyUsed
        case .pinnedHotSet:
            guard let trace else {
                throw TinyK3Error.configuration(
                    "the pinned-hot-set arm needs the routing trace it is built from")
            }
            // Slots left over once the read window is reserved, converted into
            // whole experts: pinning half an expert would guarantee a miss on
            // the other half and buy nothing.
            let regionsPerExpert = Self.regionsPerExpert(
                layers[0], granularity: configuration.granularity)
            let spare = max(0, slotCount - floor)
            let expertsPinnable = spare / max(1, regionsPerExpert * layers.count)
            var hot: Set<RegionIdentity> = []
            for layer in layers {
                for expert in trace.hottestExperts(layer: layer.layer, count: expertsPinnable) {
                    let slot = try layer.slot(forExpert: expert)
                    hot.formUnion(
                        try Self.regions(
                            layer: layer, slot: slot,
                            granularity: configuration.granularity
                        ).map(\.identity))
                    pinnedExperts += 1
                }
            }
            pinnedRegions = hot.count
            policy = .pinnedHotSet(hot)
        }
        self.pinnedExpertCount = pinnedExperts
        self.pinnedRegionCount = pinnedRegions

        self.expertPool = try BufferPool(
            configuration: BufferPoolConfiguration(
                slotCount: slotCount,
                slotBytes: expertSlotBytes,
                alignment: max(AlignedBuffer.pageSize, MXFP4TileContainer.regionAlignment),
                cachePolicy: policy,
                poisonOnRelease: false))

        // The deterministic stream is swept in full every token, so nothing in
        // it is ever reused inside a budget of this size: no cache, and the
        // slots exist purely to give the reader somewhere to read ahead into.
        // One deterministic reader is active at a time (layers run in
        // sequence), so its window plus the unit being consumed is the floor.
        // Rejected rather than clamped, like every other budget here.
        let deterministicFloor = configuration.readAhead + 1
        guard configuration.deterministicSlotCount >= deterministicFloor else {
            throw TinyK3Error.configuration(
                "deterministicSlotCount \(configuration.deterministicSlotCount) is below the "
                    + "read-ahead window plus one (\(deterministicFloor)); the reader would wait "
                    + "for a slot only the consumer can free")
        }
        self.deterministicPool = try BufferPool(
            configuration: BufferPoolConfiguration(
                slotCount: max(deterministicFloor, configuration.deterministicSlotCount),
                slotBytes: max(1, deterministicSlotBytes),
                alignment: max(AlignedBuffer.pageSize, MXFP4TileContainer.regionAlignment),
                cachePolicy: .none,
                poisonOnRelease: false))

        self.expertReaders = BoundedReaderCache(limit: configuration.maxLiveReaders)
        self.deterministicReaders = BoundedReaderCache(limit: configuration.maxLiveReaders)

        for layer in layers { schedules[layer.layer] = try Self.schedule(layer) }

        if configuration.cacheArm == .pinnedHotSet && pinnedExperts == 0 {
            notes.append(
                "pinned-hot-set had no room to pin a whole expert at this budget; "
                    + "it degenerates to no-cache")
        }
    }

    deinit { shutdown() }

    public func shutdown() {
        expertReaders.shutdownAll()
        deterministicReaders.shutdownAll()
        expertPool.evictAll()
        deterministicPool.evictAll()
    }

    // MARK: - Pool sizing

    /// Fraction of the expert pool that must still be free when it is fullest.
    ///
    /// A pool at its ceiling is not yet a fault — the budget held, nothing
    /// faulted — but it is a pool with no room for the next thing anyone adds,
    /// and it starts blocking acquires. M8.5 phase C hit exactly that: 24 of 24
    /// slots on two of six prompts with one blocked acquire, at a configuration
    /// whose two-term floor said 17 (`2026-08-09-qwen-mac-timing.md`, addendum
    /// §6). A quarter is the margin this project sizes to; it is a choice, and
    /// naming it here is what makes it one rather than an accident of defaults.
    static let requiredFreeSlotFraction = 0.25

    /// Slots that can be `filling`, `ready` or `leased` at the same instant.
    ///
    /// Three terms. The first two are the ones the old two-term floor had:
    ///
    /// - `slotsPerDispatch` — what the consumer leases for the gather it is
    ///   currently expressing. One, in the per-expert program; a whole batch,
    ///   in the batched one, because every operand of a dispatch is leased for
    ///   the whole of it.
    /// - `readerCount * readAhead` — each of the three projection readers may
    ///   run its own window ahead of demand (`TileReader` admits a pending
    ///   region only while `inFlight < readAhead`).
    ///
    /// The third term is the one that was missing, and it was found by
    /// measurement rather than by argument:
    ///
    /// - `slotsPerDispatch` again — **the previous dispatch's leases, not yet
    ///   reclaimed.** A gather's operands are adopted pager slots, and the
    ///   adoption's reference is dropped by an MLX finalizer that runs when the
    ///   graph does: after `eval` returns, on another thread. So `w3`'s
    ///   operands are being read into slots while `w1`'s are still leased.
    ///
    /// Against the Qwen measurement: 8 leased + 9 read-ahead + 8 unreclaimed is
    /// 25, and 24 of 24 is what that run peaked at. The two-term floor said 17
    /// and was wrong by the whole of the third term.
    ///
    /// `SlotCensus.inUse` deliberately excludes `cached`, so this is a count of
    /// slots that genuinely cannot be handed to anyone, not of slots that
    /// merely hold something evictable.
    static func peakSlotsInFlight(readerCount: Int, readAhead: Int, slotsPerDispatch: Int) -> Int {
        2 * slotsPerDispatch + readerCount * readAhead
    }

    /// The smallest pool that leaves ``requiredFreeSlotFraction`` free at
    /// `peak`. Ceiling division, so the margin is met rather than approached.
    static func slotsForPeak(_ peak: Int) -> Int {
        let numerator = Double(peak)
        return Int((numerator / (1 - requiredFreeSlotFraction)).rounded(.up))
    }

    /// The expert pool's stated budget and its high-water mark. Readable
    /// without a run, so the sizing can be asserted on its own.
    public var expertPoolStatistics: BufferPoolStatistics { expertPool.statistics }

    /// Open expert readers right now. Bounded by
    /// ``Configuration/maxLiveReaders`` rather than by the layer count; exposed
    /// so a test can assert the bound instead of inferring it from thread
    /// counts.
    public var liveExpertReaderCount: Int { expertReaders.liveCount }

    /// Expert readers closed to stay inside the bound. Zero means the cache
    /// never had to evict, which is what makes it usable as a control arm.
    public var expertReaderEvictionCount: Int { expertReaders.evictionCount }

    static func regionsPerExpert(
        _ layer: FlagshipLayerStreams, granularity: FetchGranularity
    ) -> Int {
        switch granularity {
        case .perExpert: return ExpertProjection.allCases.count
        case .perTile: return layer.tilesPerExpert
        }
    }

    static func regions(
        layer: FlagshipLayerStreams, slot: Int, granularity: FetchGranularity
    ) throws -> [RegionDescriptor] {
        var regions: [RegionDescriptor] = []
        for projection in ExpertProjection.allCases {
            let container = try layer.container(projection)
            switch granularity {
            case .perExpert:
                regions.append(container.expertRegion(slot: slot))
            case .perTile:
                for band in 0..<container.tilesPerExpert {
                    regions.append(container.tileRegion(slot: slot, band: band))
                }
            }
        }
        return regions
    }

    /// Split the deterministic units into the order the decode step needs them.
    ///
    /// The blob is written in checkpoint file order, which puts the latent
    /// up-projection *before* the shared expert and long before the expert
    /// bytes it consumes. Reading in file order would mean buffering 51 MB of
    /// up-projection until the experts finish. Reading in dependency order
    /// costs a handful of extra seeks per token and no extra bytes, so that is
    /// what this does — and each tensor is still one contiguous run.
    private static func schedule(_ layer: FlagshipLayerStreams) throws
        -> DeterministicSchedule
    {
        let units = layer.deterministic.units
        func suffix(_ name: String) -> [FlagshipLayerStreams.DeterministicUnit] {
            units.filter { $0.tensor.hasSuffix(name) }
        }
        let router = suffix("gate.weight")
        let bias = suffix("gate.e_score_correction_bias").first
        let latentDown = suffix("routed_expert_down_proj.weight")
        let latentUp = suffix("routed_expert_up_proj.weight")
        let latentNorm = suffix("routed_expert_norm.weight").first
        let sharedGate = suffix("shared_experts.gate_proj.weight")
        let sharedUp = suffix("shared_experts.up_proj.weight")
        let sharedDown = suffix("shared_experts.down_proj.weight")

        var named: [FlagshipLayerStreams.DeterministicUnit] = []
        named.append(contentsOf: router)
        named.append(contentsOf: latentDown)
        named.append(contentsOf: latentUp)
        named.append(contentsOf: sharedGate)
        named.append(contentsOf: sharedUp)
        named.append(contentsOf: sharedDown)
        var claimed = Set<Int>(named.map(\.index))
        if let bias { claimed.insert(bias.index) }
        if let latentNorm { claimed.insert(latentNorm.index) }
        let overlap = units.filter { !claimed.contains($0.index) }

        for required in [
            ("router", router), ("latent down", latentDown), ("latent up", latentUp),
            ("shared gate", sharedGate), ("shared up", sharedUp), ("shared down", sharedDown),
        ] where required.1.isEmpty {
            throw TinyK3Error.configuration(
                "layer \(layer.layer) deterministic blob has no \(required.0) units")
        }
        guard bias != nil, latentNorm != nil else {
            throw TinyK3Error.configuration(
                "layer \(layer.layer) deterministic blob is missing the router bias or latent norm")
        }
        return DeterministicSchedule(
            router: router, routerBias: bias, latentDown: latentDown, overlap: overlap,
            latentNorm: latentNorm, latentUp: latentUp,
            sharedGate: sharedGate, sharedUp: sharedUp, sharedDown: sharedDown)
    }

    // MARK: - Readers

    private func expertReader(_ container: FlagshipLayerStreams.ExpertContainer) throws
        -> TileReader
    {
        try expertReaders.reader(for: container.path) {
            try TileReader(
                path: container.path, pool: expertPool,
                configuration: TileReaderConfiguration(
                    queueDepth: configuration.queueDepth,
                    readAhead: configuration.readAhead,
                    noCache: configuration.noCache,
                    slotAcquireTimeout: configuration.slotAcquireTimeout))
        }
    }

    private func deterministicReader(_ layer: FlagshipLayerStreams) throws -> TileReader {
        try deterministicReaders.reader(for: layer.deterministic.path) {
            try TileReader(
                path: layer.deterministic.path, pool: deterministicPool,
                configuration: TileReaderConfiguration(
                    queueDepth: configuration.queueDepth,
                    readAhead: configuration.readAhead,
                    noCache: configuration.noCache,
                    slotAcquireTimeout: configuration.slotAcquireTimeout))
        }
    }

    /// Every `acquire` goes through here so the wait is charged exactly once,
    /// to the one metric that can say "the loop was blocked on storage".
    private func acquire(_ reader: TileReader, _ region: RegionDescriptor) throws -> SlotLease {
        let start = MonotonicClock.now()
        let lease = try reader.acquire(region)
        addAcquireWaitNanoseconds(MonotonicClock.now() &- start)
        return lease
    }

    // MARK: - Deterministic compute

    /// Adopt a BF16 or F32 unit out of its slot with no copy, widen to float32,
    /// and hand back the row band as `[rows, cols]`.
    private func adopt(_ lease: SlotLease, _ unit: FlagshipLayerStreams.DeterministicUnit)
        -> MLXArray
    {
        let dtype: DType = unit.dtype == "F32" ? .float32 : .bfloat16
        return MLXArray(
            rawPointer: lease.slot.pointer, [unit.rows, unit.cols], dtype: dtype,
            finalizer: { lease.release() })
    }

    /// `x · Wᵀ` accumulated over the row bands of one tensor, each band read
    /// through the pager and released before the next is asked for.
    private func streamedLinear(
        _ x: MLXArray,
        units: [FlagshipLayerStreams.DeterministicUnit],
        layer: FlagshipLayerStreams,
        reader: TileReader
    ) throws -> MLXArray {
        var parts: [MLXArray] = []
        parts.reserveCapacity(units.count)
        for unit in units {
            let region = layer.deterministic.region(unit)
            addDeterministicBytesRequested(UInt64(region.length))
            let lease = try acquire(reader, region)
            try lease.retain()
            let weight = adopt(lease, unit)
            let compute = MonotonicClock.now()
            let part = matmul(x, weight.asType(.float32).transposed(1, 0))
            part.eval()
            addComputeNanoseconds(MonotonicClock.now() &- compute)
            parts.append(part)
            lease.release()
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: -1)
    }

    /// Read units and drop them. The bytes still move — that is the point —
    /// but nothing is computed from them.
    private func streamOnly(
        units: [FlagshipLayerStreams.DeterministicUnit],
        layer: FlagshipLayerStreams,
        reader: TileReader
    ) throws {
        for unit in units {
            let region = layer.deterministic.region(unit)
            addDeterministicBytesRequested(UInt64(region.length))
            let lease = try acquire(reader, region)
            lease.release()
        }
    }

    private func readVector(
        _ unit: FlagshipLayerStreams.DeterministicUnit,
        layer: FlagshipLayerStreams,
        reader: TileReader
    ) throws -> MLXArray {
        let region = layer.deterministic.region(unit)
        addDeterministicBytesRequested(UInt64(region.length))
        let lease = try acquire(reader, region)
        try lease.retain()  // the adopted array's own reference
        let array = adopt(lease, unit).asType(.float32).reshaped([unit.rows * unit.cols])
        array.eval()
        // Exactly one release here: this function's reference. The adopted
        // array holds the other and drops it in its finalizer.
        lease.release()
        return array
    }

    // MARK: - Expert compute

    /// One expert's contribution, streamed. Returns `[1, latentSize]`.
    private func expertOutput(
        latent z: MLXArray, layer: FlagshipLayerStreams, expert: Int
    ) throws -> MLXArray {
        let slot = try layer.slot(forExpert: expert)
        let gate = try projectionOutput(z, layer: layer, projection: .w1, slot: slot)
        let up = try projectionOutput(z, layer: layer, projection: .w3, slot: slot)
        let compute = MonotonicClock.now()
        let activated = K3Activation.situ(
            concatenated([gate, up], axis: -1),
            beta: shape.situBeta, linearBeta: shape.situLinearBeta)
        activated.eval()
        addComputeNanoseconds(MonotonicClock.now() &- compute)
        return try projectionOutput(activated, layer: layer, projection: .w2, slot: slot)
    }

    /// All row bands of one projection of one expert, concatenated.
    ///
    /// The two granularities differ here and nowhere else: `perExpert` takes
    /// one lease and adopts every band out of sub-offsets of that one slot;
    /// `perTile` takes one lease per band. Same bytes, same arithmetic, same
    /// order — only the request shape changes, which is what makes the
    /// comparison a statement about the read plan.
    private func projectionOutput(
        _ input: MLXArray,
        layer: FlagshipLayerStreams,
        projection: ExpertProjection,
        slot: Int
    ) throws -> MLXArray {
        let container = try layer.container(projection)
        let reader = try expertReader(container)
        let index = MLXArray([Int32(0)], [1, 1])
        var parts: [MLXArray] = []
        parts.reserveCapacity(container.tilesPerExpert)

        switch configuration.granularity {
        case .perExpert:
            let region = container.expertRegion(slot: slot)
            addExpertBytesRequested(UInt64(region.length))
            let lease = try acquire(reader, region)
            // Held until after the evaluation below. The lazy graph keeps its
            // own reference, but relying on that to outlive the Swift wrapper
            // is the kind of assumption that fails once and is unfindable.
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
                parts.append(bandProduct(input, weights, index: index, container: container))
            }
            // `MLX.eval` forces the lazy graph — it is not code evaluation.
            // Everything expressed from this slot must be computed before the
            // lease goes, or the graph would read a recycled buffer.
            let evaluate = MonotonicClock.now()
            MLX.eval(parts)
            addComputeNanoseconds(MonotonicClock.now() &- evaluate)
            held.removeAll()
            lease.release()

        case .perTile:
            for band in 0..<container.tilesPerExpert {
                let region = container.tileRegion(slot: slot, band: band)
                addExpertBytesRequested(UInt64(region.length))
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
                let part = bandProduct(input, weights, index: index, container: container)
                let evaluate = MonotonicClock.now()
                part.eval()
                addComputeNanoseconds(MonotonicClock.now() &- evaluate)
                parts.append(part)
                lease.release()
            }
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: -1)
    }

    private func bandProduct(
        _ input: MLXArray,
        _ weights: MXFP4Weights,
        index: MLXArray,
        container: FlagshipLayerStreams.ExpertContainer
    ) -> MLXArray {
        // A stack of one, gathered with one index. See ``batchedProjection``
        // for why this is no longer the only option at flagship geometry.
        expertGatherDispatches += 1
        return mxfp4GatherMM(input, weights, rhsIndices: index)
            .reshaped([1, container.rowsPerTile])
    }

    // MARK: - Batched expert compute

    /// Every selected expert's contribution, in `ceil(k / dispatchWidth)`
    /// dispatches per projection instead of `k × tilesPerExpert`. Returns
    /// `[slots, latentSize]`.
    ///
    /// ## What M8 said, and why it was wrong
    ///
    /// M8 recorded that batching was not available here: *"a container tile is
    /// a row band of a SINGLE expert, so the gather has nothing to gather
    /// across; batching the 16 selected experts into one dispatch would need
    /// them contiguous in memory, which 16 independent pager slots are not."*
    ///
    /// The second half is a fact and the first half does not follow from it.
    /// Pager slots are not contiguous, so the batch is **assembled** — the same
    /// answer M8.5 phase C reached on the Qwen path, where the identical
    /// objection had been raised twice and retired by measuring rather than
    /// arguing.
    ///
    /// Two things had to be true, and both are measured in
    /// `MLXBridgeTests.BatchedGatherEquivalenceTests` rather than assumed:
    /// concatenating an expert's row bands into one operand does not
    /// reassociate the kernel's reduction, and neither does lengthening the
    /// expert axis. Bit-for-bit on both streams.
    ///
    /// ## What the assembly costs
    ///
    /// One copy of the selected experts' bytes per layer, device-side: 267.8
    /// MiB at flagship geometry. It is not avoidable at this container version
    /// because a tile interleaves its packed block with its scale block, so an
    /// expert's packed rows are not contiguous even inside one whole-expert
    /// read. A format that grouped all of an expert's packed bytes and then all
    /// of its scales would make this stack adoptable with no copy at all.
    ///
    /// ## Why a chunk boundary cannot move a value
    ///
    /// Chunks partition the **slots**, not the experts, and each chunk produces
    /// the rows for its own slots. They are concatenated in slot order, never
    /// summed, so there is nothing to reassociate across a boundary. The one
    /// sum on this path — the weighted sum over slots — happens after all of
    /// them, in ``step(token:layer:hidden:trace:)``, in slot order, exactly as
    /// the per-expert program does it.
    private func batchedExpertOutputs(
        latent z: MLXArray, layer: FlagshipLayerStreams, ids: [Int]
    ) throws -> MLXArray {
        let gate = try batchedProjection(
            z, layer: layer, projection: .w1, ids: ids, rowPerSlot: false)
        let up = try batchedProjection(
            z, layer: layer, projection: .w3, ids: ids, rowPerSlot: false)

        let compute = MonotonicClock.now()
        let activated = K3Activation.situ(
            concatenated([gate, up], axis: -1),
            beta: shape.situBeta, linearBeta: shape.situLinearBeta)
        // `[1, slots, intermediate]` -> `[slots, intermediate]`. From here each
        // slot is its own row with its own expert, which is exactly what the
        // down projection gathers over — the same reshape `K3MoE.forward` does
        // for the same reason.
        let rows = activated.reshaped([ids.count, activated.shape[activated.ndim - 1]])
        rows.eval()
        addComputeNanoseconds(MonotonicClock.now() &- compute)

        let down = try batchedProjection(
            rows, layer: layer, projection: .w2, ids: ids, rowPerSlot: true)
        return down.reshaped([ids.count, shape.latentSize])
    }

    /// One projection of every selected expert.
    ///
    /// `w1` and `w3` take the single latent row and produce one output per slot
    /// (`rowPerSlot: false`, `[1, slots, out]`); `w2` takes one activation row
    /// per slot and produces one output for it (`rowPerSlot: true`,
    /// `[slots, 1, out]`). Both shapes reach `mxfp4GatherMM` as a rank-2 `x`,
    /// so the square-broadcast trap the wrapper exists to close stays closed —
    /// a batched gather is precisely where presenting `x` with its own
    /// broadcast axes becomes expressible.
    ///
    /// `rowPerSlot` is a parameter and not something inferred from `input`'s
    /// shape on purpose. The two cases are distinguishable by shape only while
    /// `tokens != slots`, which is the exact condition the trap turns on, and a
    /// silent wrong answer is what it costs.
    private func batchedProjection(
        _ input: MLXArray,
        layer: FlagshipLayerStreams,
        projection: ExpertProjection,
        ids: [Int],
        rowPerSlot: Bool
    ) throws -> MLXArray {
        let container = try layer.container(projection)
        let reader = try expertReader(container)
        let words = container.inFeatures / 8
        let groups = container.inFeatures / MXFP4Weights.groupSize

        var chunks: [MLXArray] = []
        var first = 0
        while first < ids.count {
            let count = min(dispatchWidth, ids.count - first)

            // Obtaining the operands. The waiting inside this is charged to
            // `acquireWaitNanoseconds` by `acquire` itself, once per region.
            var leases: [SlotLease] = []
            var held: [MXFP4Weights] = []
            var packedParts: [MLXArray] = []
            var scaleParts: [MLXArray] = []
            leases.reserveCapacity(count)
            for expert in ids[first..<(first + count)] {
                try adopt(
                    expert: expert, layer: layer, container: container, reader: reader,
                    leases: &leases, held: &held, packed: &packedParts, scales: &scaleParts)
            }

            let mark = MonotonicClock.now()
            // Expert-major, band-minor, row-minor: exactly the row-major order
            // of `[experts, outFeatures, ...]`, so one concatenation on the row
            // axis plus a reshape is the whole assembly. Two copies per
            // dispatch, not two per band.
            let stacked = try MXFP4Weights(
                packed: concatenated(packedParts, axis: 1)
                    .reshaped([count, container.outFeaturesPerExpert, words]),
                scales: concatenated(scaleParts, axis: 1)
                    .reshaped([count, container.outFeaturesPerExpert, groups]))
            let x = rowPerSlot ? input[first..<(first + count), 0...] : input
            let indices =
                rowPerSlot
                ? MLXArray((0..<count).map(Int32.init), [count, 1])
                : MLXArray((0..<count).map(Int32.init), [1, count])
            let produced = mxfp4GatherMM(x, stacked, rhsIndices: indices)
            expertGatherDispatches += 1
            // Everything expressed from these slots must be computed before the
            // leases go, or the graph would read a recycled buffer. Same rule
            // as the per-expert program, one dispatch later.
            produced.eval()
            addComputeNanoseconds(MonotonicClock.now() &- mark)

            held.removeAll()
            for lease in leases { lease.release() }
            chunks.append(produced)
            first += count
        }
        // Chunks cover disjoint runs of slots in order, so reassembling them is
        // a concatenation on the slot axis and not a merge.
        return chunks.count == 1
            ? chunks[0] : concatenated(chunks, axis: rowPerSlot ? 0 : 1)
    }

    /// Lease one expert's projection and adopt its row bands out of the slots.
    ///
    /// The two granularities differ here and nowhere else, exactly as they do
    /// in ``projectionOutput(_:layer:projection:slot:)``: `perExpert` takes one
    /// lease and adopts every band from sub-offsets of it, `perTile` takes a
    /// lease per band. Same bytes, same order, same arithmetic.
    private func adopt(
        expert: Int,
        layer: FlagshipLayerStreams,
        container: FlagshipLayerStreams.ExpertContainer,
        reader: TileReader,
        leases: inout [SlotLease],
        held: inout [MXFP4Weights],
        packed: inout [MLXArray],
        scales: inout [MLXArray]
    ) throws {
        let slot = try layer.slot(forExpert: expert)
        func adoptBand(_ lease: SlotLease, _ base: UnsafeMutableRawPointer) throws {
            try lease.retain()
            try lease.retain()
            let weights = try MXFP4Weights.adopting(
                packedPointer: base,
                scalesPointer: base + container.layout.scaleOffsetInTile,
                experts: 1,
                outFeatures: container.rowsPerTile,
                inFeatures: container.inFeatures,
                packedFinalizer: { lease.release() },
                scalesFinalizer: { lease.release() })
            held.append(weights)
            packed.append(weights.packed)
            scales.append(weights.scales)
        }

        switch configuration.granularity {
        case .perExpert:
            let region = container.expertRegion(slot: slot)
            addExpertBytesRequested(UInt64(region.length))
            let lease = try acquire(reader, region)
            leases.append(lease)
            for band in 0..<container.tilesPerExpert {
                try adoptBand(lease, lease.slot.pointer + band * container.layout.tileStride)
            }
        case .perTile:
            for band in 0..<container.tilesPerExpert {
                let region = container.tileRegion(slot: slot, band: band)
                addExpertBytesRequested(UInt64(region.length))
                let lease = try acquire(reader, region)
                leases.append(lease)
                try adoptBand(lease, lease.slot.pointer)
            }
        }
    }

    // MARK: - The loop

    public func run(trace: RoutingTrace, tokens requested: Int? = nil) throws
        -> ExpertStreamMetrics
    {
        MLX.GPU.set(cacheLimit: configuration.mlxCacheLimitBytes)
        MLX.GPU.clearCache()

        let tokenCount = min(requested ?? trace.tokens, trace.tokens)
        var hiddenGenerator = SplitMix64(seed: configuration.hiddenStateSeed)
        var windows: [ExpertStreamMetrics.Window] = []
        let windowSize = configuration.windowTokens > 0
            ? configuration.windowTokens : max(1, tokenCount / 3)

        var windowStart = MonotonicClock.now()
        var windowFirstToken = 0
        var windowBase = totalBytesRead()

        let started = MonotonicClock.now()
        for token in 0..<tokenCount {
            let x = randomHidden(&hiddenGenerator, width: shape.hiddenSize)
            for layer in layers {
                _ = try step(token: token, layer: layer, hidden: x, trace: trace)
            }
            let done = token + 1
            if done % windowSize == 0 || done == tokenCount {
                let seconds = MonotonicClock.seconds(since: windowStart)
                let currentBytes = totalBytesRead()
                let byteDelta = currentBytes.value.subtractingReportingOverflow(windowBase.value)
                let windowByteOverflow = windowBase.didOverflow || currentBytes.didOverflow
                    || byteDelta.overflow
                let bytes = windowByteOverflow ? 0 : byteDelta.partialValue
                windows.append(
                    ExpertStreamMetrics.Window(
                        label: "tokens \(windowFirstToken)..<\(done)",
                        firstToken: windowFirstToken,
                        tokenCount: done - windowFirstToken,
                        seconds: seconds,
                        tokensPerSecond: seconds > 0
                            ? Double(done - windowFirstToken) / seconds : 0,
                        bytesRead: bytes,
                        byteAccountingOverflowed: windowByteOverflow,
                        throughputMBPerSecond: seconds > 0
                            ? Double(bytes) / seconds / 1e6 : 0,
                        thermalState: ProcessInfo.processInfo.thermalState.label))
                windowStart = MonotonicClock.now()
                windowFirstToken = done
                windowBase = currentBytes
            }
        }
        let wall = MonotonicClock.seconds(since: started)
        return report(trace: trace, tokens: tokenCount, wallSeconds: wall, windows: windows)
    }

    /// One layer of one token.
    @discardableResult
    public func step(
        token: Int, layer: FlagshipLayerStreams, hidden x: MLXArray, trace: RoutingTrace
    ) throws -> MLXArray {
        let plan = schedules[layer.layer]!
        let reader = try deterministicReader(layer)

        // Queue the whole deterministic stream, in the order it will be
        // consumed, before consuming any of it. Without this every unit is
        // demanded one at a time and the read-ahead window never fills: the
        // reader cannot start unit n+1 until the loop asks for it, which pins
        // 2 GiB per token at queue depth 1. Scheduling in consumption order
        // means the window is always working on what comes next.
        if configuration.streamDeterministic {
            var ordered: [FlagshipLayerStreams.DeterministicUnit] = plan.router
            if let bias = plan.routerBias { ordered.append(bias) }
            ordered += plan.latentDown
            ordered += plan.overlap.filter { $0.role != "shared_expert" }
            ordered += plan.sharedGate + plan.sharedUp + plan.sharedDown
            if let norm = plan.latentNorm { ordered.append(norm) }
            ordered += plan.latentUp
            try reader.schedule(ordered.map { layer.deterministic.region($0) })
        }

        // 1. Router, in row bands.
        var ids: [Int]
        var weights: MLXArray
        if configuration.streamDeterministic {
            let logits = try streamedLinear(x, units: plan.router, layer: layer, reader: reader)
            let bias = try readVector(plan.routerBias!, layer: layer, reader: reader)
            if let preselected = trace.ids(layer: layer.layer, token: token) {
                ids = preselected
                // The real router still ran — the trace only overrides which
                // experts are fetched, so that locality is a controlled
                // variable. Its weights are used for the weighted sum.
                let selection = K3Router.select(
                    logits: logits, correctionBias: bias, numExperts: layer.experts,
                    topK: shape.expertsPerToken, renormalize: shape.renormalize,
                    scalingFactor: shape.routedScalingFactor)
                weights = selection.weights
            } else {
                let selection = K3Router.select(
                    logits: logits, correctionBias: bias, numExperts: layer.experts,
                    topK: shape.expertsPerToken, renormalize: shape.renormalize,
                    scalingFactor: shape.routedScalingFactor)
                ids = selection.ids[0]
                weights = selection.weights
            }
        } else {
            guard let preselected = trace.ids(layer: layer.layer, token: token) else {
                throw TinyK3Error.configuration(
                    "the trace has no ids for layer \(layer.layer) token \(token), and the "
                        + "deterministic stream that would produce them is switched off")
            }
            ids = preselected
            weights = MLXArray.ones([1, ids.count], dtype: .float32) / Float(ids.count)
        }

        realizedSelections[layer.layer, default: []].append(ids)

        // 2. Expert reads start now, so they overlap steps 3 and 4.
        for projection in ExpertProjection.allCases {
            let container = try layer.container(projection)
            let regions = try ids.map { expert -> [RegionDescriptor] in
                let slot = try layer.slot(forExpert: expert)
                switch configuration.granularity {
                case .perExpert:
                    return [container.expertRegion(slot: slot)]
                case .perTile:
                    return (0..<container.tilesPerExpert).map {
                        container.tileRegion(slot: slot, band: $0)
                    }
                }
            }.flatMap { $0 }
            // Do not queue a read for bytes the pool is already holding. The
            // reader would fetch them regardless, and the redundant read takes
            // a slot — evicting, in the worst case, the entry the request was
            // about to hit. Skipping is safe: if the entry is evicted before
            // `acquire`, that call schedules it urgently instead.
            try expertReader(container).schedule(
                regions.filter { !expertPool.isCached($0.identity) })
        }

        // 3. Latent down-projection.
        var z: MLXArray
        if configuration.streamDeterministic {
            z = try streamedLinear(x, units: plan.latentDown, layer: layer, reader: reader)
        } else {
            z = MLXArray.zeros([1, shape.latentSize], dtype: .float32)
        }

        // 4. The rest of the deterministic stream, overlapping the expert reads.
        var shared: MLXArray?
        if configuration.streamDeterministic {
            try streamOnly(
                units: plan.overlap.filter { $0.role != "shared_expert" },
                layer: layer, reader: reader)
            if configuration.computeDeterministic {
                let gate = try streamedLinear(
                    x, units: plan.sharedGate, layer: layer, reader: reader)
                let up = try streamedLinear(x, units: plan.sharedUp, layer: layer, reader: reader)
                let activated = K3Activation.situ(
                    concatenated([gate, up], axis: -1),
                    beta: shape.situBeta, linearBeta: shape.situLinearBeta)
                shared = try streamedLinear(
                    activated, units: plan.sharedDown, layer: layer, reader: reader)
            } else {
                try streamOnly(
                    units: plan.sharedGate + plan.sharedUp + plan.sharedDown,
                    layer: layer, reader: reader)
            }
        }

        // 5. Experts.
        weights.eval()
        let weightValues = weights.asArray(Float.self)
        var accumulator = MLXArray.zeros([1, shape.latentSize], dtype: .float32)
        if configuration.gatherProgram == .batched {
            let outputs = try batchedExpertOutputs(latent: z, layer: layer, ids: ids)
            let compute = MonotonicClock.now()
            // Slot 0 first, then slot 1, and so on: the *same* left-to-right
            // accumulation the per-expert loop below performs. This is the one
            // reduction on the routed path, so its order is the one thing
            // batching must not be allowed to change — and it is written out
            // rather than handed to `sum`, which makes no order promise.
            for position in ids.indices {
                accumulator =
                    accumulator
                    + outputs[position].reshaped([1, shape.latentSize])
                    * MLXArray(weightValues[position])
            }
            accumulator.eval()
            addComputeNanoseconds(MonotonicClock.now() &- compute)
        } else {
            for (position, expert) in ids.enumerated() {
                let output = try expertOutput(latent: z, layer: layer, expert: expert)
                let compute = MonotonicClock.now()
                accumulator = accumulator + output * MLXArray(weightValues[position])
                accumulator.eval()
                addComputeNanoseconds(MonotonicClock.now() &- compute)
            }
        }

        // 6. Latent norm and up-projection.
        guard configuration.streamDeterministic else { return accumulator }
        let normWeight = try readVector(plan.latentNorm!, layer: layer, reader: reader)
        let compute = MonotonicClock.now()
        let normed = K3Norm.rms(accumulator, weight: normWeight, eps: shape.rmsNormEps)
        addComputeNanoseconds(MonotonicClock.now() &- compute)
        var output = try streamedLinear(normed, units: plan.latentUp, layer: layer, reader: reader)
        if let shared { output = output + shared }
        output.eval()
        return output
    }

    private func randomHidden(_ generator: inout SplitMix64, width: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: width)
        for index in 0..<width {
            // Uniform on [-1, 1) from the top 24 bits, so the hidden state is
            // reproducible from the seed alone and independent of MLX's RNG.
            let bits = generator.next() >> 40
            values[index] = Float(bits) / Float(1 << 23) - 1
        }
        return MLXArray(values, [1, width])
    }

    // MARK: - Reporting

    private func totalBytesRead() -> UInt64Accounting.SaturatingSum {
        let expert = byteTotal(readers: expertReaders.live, retired: expertReaders.retired)
        let deterministic = byteTotal(
            readers: deterministicReaders.live, retired: deterministicReaders.retired)
        var total = UInt64Accounting.SaturatingSum(
            value: expert.value,
            didOverflow: expert.didOverflow || deterministic.didOverflow)
        total.add(deterministic.value)
        return total
    }

    private func byteTotal(
        readers: some Collection<TileReader>, retired: RetiredReaderStatistics
    ) -> UInt64Accounting.SaturatingSum {
        var total = UInt64Accounting.SaturatingSum(
            value: retired.bytesRead, didOverflow: retired.byteCountOverflowed)
        for reader in readers {
            let statistics = reader.statistics
            if statistics.byteCountOverflowed {
                total = UInt64Accounting.SaturatingSum(value: .max, didOverflow: true)
            } else {
                total.add(statistics.bytesRead)
            }
        }
        return total
    }

    private func streamIO(
        readers: some Collection<TileReader>, retired: RetiredReaderStatistics,
        requested: UInt64, requestedOverflowed: Bool, tokens: Int, wall: Double
    ) -> ExpertStreamMetrics.StreamIO {
        let bytes = byteTotal(readers: readers, retired: retired)
        var reads = retired.readCount
        var short = retired.shortReadCount
        var errors = retired.errorCount
        var firstError: String? = retired.firstError
        var samples: [UInt64] = retired.latencySamples
        for reader in readers {
            let statistics = reader.statistics
            reads += statistics.readCount
            short += statistics.shortReadCount
            errors += statistics.errorCount
            if firstError == nil { firstError = statistics.firstError }
            // Genuinely pooled: the raw samples from every reader concatenated,
            // then summarised once. Summarising each reader and averaging the
            // summaries would report a p99 that no read ever had.
            samples.append(contentsOf: reader.latencySamples)
        }
        var pooled = samples
        return ExpertStreamMetrics.StreamIO(
            bytesRequested: requested,
            bytesRead: bytes.value,
            byteAccountingOverflowed: requestedOverflowed || bytes.didOverflow,
            readCount: reads,
            shortReadCount: short,
            errorCount: errors,
            firstError: firstError,
            latency: LatencyStatistics.compute(nanosecondSamples: &pooled),
            bytesPerToken: tokens > 0 ? Double(bytes.value) / Double(tokens) : 0,
            throughputMBPerSecond: wall > 0 ? Double(bytes.value) / wall / 1e6 : 0)
    }

    private func report(
        trace: RoutingTrace, tokens: Int, wallSeconds: Double,
        windows: [ExpertStreamMetrics.Window]
    ) -> ExpertStreamMetrics {
        // Slots come back from MLX finalizers, which can run on a completion
        // thread after `eval` returns, so the census is read after a bounded
        // wait rather than the instant the loop ends.
        let deadline = Date().addingTimeInterval(5)
        while (expertPool.census.inUse > 0 || deterministicPool.census.inUse > 0)
            && Date() < deadline
        {
            usleep(500)
        }

        let expertStatistics = expertPool.statistics
        let deterministicStatistics = deterministicPool.statistics
        let expertIO = streamIO(
            readers: expertReaders.live, retired: expertReaders.retired,
            requested: expertBytesRequested,
            requestedOverflowed: expertByteAccountingOverflowed,
            tokens: tokens, wall: wallSeconds)
        let deterministicIO = streamIO(
            readers: deterministicReaders.live, retired: deterministicReaders.retired,
            requested: deterministicBytesRequested,
            requestedOverflowed: deterministicByteAccountingOverflowed,
            tokens: tokens, wall: wallSeconds)

        let combinedBytes = UInt64Accounting.checkedSum([
            expertIO.bytesRead, deterministicIO.bytesRead,
        ])
        let combinedByteAccountingOverflowed = expertIO.byteAccountingOverflowed
            || deterministicIO.byteAccountingOverflowed || combinedBytes == nil

        var slotWait = expertReaders.retired.slotWaitSeconds
            + deterministicReaders.retired.slotWaitSeconds
        for reader in expertReaders.live { slotWait += reader.statistics.slotWaitSeconds }
        for reader in deterministicReaders.live { slotWait += reader.statistics.slotWaitSeconds }
        let ioWait = Double(acquireWaitNanoseconds) / 1e9
        let compute = Double(computeNanoseconds) / 1e9

        var census = expertStatistics.census
        let other = deterministicStatistics.census
        census.free += other.free
        census.filling += other.filling
        census.ready += other.ready
        census.leased += other.leased
        census.cached += other.cached

        let bytesPerExpert = layers.map(\.bytesPerExpert).max() ?? 1
        let memory = MemoryUsage.current()
        // Locality of what was actually routed, which is the trace's own
        // locality when the trace pre-selected, and a measurement of the
        // checkpoint's router when it did not.
        let realized = RoutingTrace(
            name: "\(trace.name)+realized",
            numExperts: layers.first?.experts ?? trace.numExperts,
            expertsPerToken: shape.expertsPerToken,
            tokens: tokens,
            generator: trace.generator,
            selections: realizedSelections)

        return ExpertStreamMetrics(
            run: ExpertStreamMetrics.Run(
                timestamp: ISO8601DateFormatter().string(from: Date()),
                gitRevision: BuildInfo.gitRevision,
                device: DeviceInfo.current(),
                os: OSInfo.current(),
                configuration: configuration.label,
                thermalState: ProcessInfo.processInfo.thermalState.label),
            workload: ExpertStreamMetrics.Workload(
                tokens: tokens,
                layers: layers.map(\.layer),
                expertsPerToken: shape.expertsPerToken,
                expertsPerLayer: layers.first?.experts ?? 0,
                trace: trace.name,
                traceDistribution: trace.generator.distribution,
                traceSeed: trace.generator.seed,
                routedByRealRouter: configuration.streamDeterministic,
                granularity: configuration.granularity.rawValue,
                readsPerExpert: configuration.granularity.readsPerExpert,
                computedDeterministicRoles: configuration.computeDeterministic
                    ? ["router", "latent", "latent_norm", "shared_expert"] : ["router", "latent"],
                streamedOnlyDeterministicRoles: ["attention", "norm"],
                gatherProgram: configuration.gatherProgram.rawValue,
                expertsPerDispatch: configuration.gatherProgram == .batched ? dispatchWidth : 1,
                expertGatherDispatches: expertGatherDispatches,
                expertGatherDispatchesPerToken: tokens > 0
                    ? Double(expertGatherDispatches) / Double(tokens) : 0),
            deterministic: deterministicIO,
            expert: expertIO,
            combinedThroughputMBPerSecond: wallSeconds > 0
                ? Double(combinedBytes ?? 0) / wallSeconds / 1e6 : 0,
            deterministicToExpertByteRatio: expertIO.bytesRead > 0
                ? Double(deterministicIO.bytesRead) / Double(expertIO.bytesRead) : 0,
            timingAccountingOverflowed: timingAccountingOverflowed
                || expertStatistics.blockedTimeOverflowed
                || deterministicStatistics.blockedTimeOverflowed,
            cache: ExpertStreamMetrics.Cache(
                policy: expertStatistics.cachePolicyName,
                budgetBytes: expertStatistics.budgetBytes,
                budgetExpertsEquivalent: Double(expertStatistics.budgetBytes)
                    / Double(bytesPerExpert),
                slotCount: expertStatistics.slotCount,
                slotBytes: expertStatistics.slotBytes,
                pinnedRegionCount: pinnedRegionCount,
                pinnedExpertCount: pinnedExpertCount,
                hits: expertStatistics.cacheHitCount,
                misses: expertStatistics.cacheMissCount,
                hitRate: expertStatistics.cacheHitRate,
                evictions: expertStatistics.cacheEvictionCount,
                admissions: expertStatistics.cacheAdmissionCount,
                rejectedAdmissions: expertStatistics.cacheRejectedAdmissionCount),
            memory: ExpertStreamMetrics.Memory(
                expertPoolBudgetBytes: expertStatistics.budgetBytes,
                expertPoolPeakBytes: expertStatistics.peakInUseBytes,
                expertPoolPeakSlots: expertStatistics.peakInUseSlots,
                deterministicPoolBudgetBytes: deterministicStatistics.budgetBytes,
                deterministicPoolPeakBytes: deterministicStatistics.peakInUseBytes,
                totalPagerBudgetBytes: expertStatistics.budgetBytes
                    + deterministicStatistics.budgetBytes,
                mlxPeakBytes: MLX.GPU.peakMemory,
                mlxCacheLimitBytes: configuration.mlxCacheLimitBytes,
                processPeakResidentBytes: memory?.peakResidentBytes,
                processResidentBytes: memory?.residentBytes,
                poolFaults: expertStatistics.faults + deterministicStatistics.faults,
                budgetRespected: expertStatistics.isBudgetRespected
                    && deterministicStatistics.isBudgetRespected,
                finalCensus: census),
            stalls: ExpertStreamMetrics.Stalls(
                computeWaitedOnIOSeconds: ioWait,
                readerWaitedOnSlotSeconds: slotWait,
                computeSeconds: compute,
                wallSeconds: wallSeconds,
                ioWaitFraction: wallSeconds > 0 ? ioWait / wallSeconds : 0,
                overlapFraction: wallSeconds > 0 ? max(0, 1 - ioWait / wallSeconds) : 0),
            timing: windows,
            tokensPerSecond: wallSeconds > 0 ? Double(tokens) / wallSeconds : 0,
            secondsPerToken: tokens > 0 ? wallSeconds / Double(tokens) : 0,
            wallSeconds: wallSeconds,
            localityByLayer: layers.compactMap { trace.locality(layer: $0.layer) },
            realizedLocality: layers.compactMap { realized.locality(layer: $0.layer) },
            notes: combinedByteAccountingOverflowed
                ? notes + ["byte accounting saturated; byte-derived conclusions are indeterminate"]
                : notes)
    }

    private func addExpertBytesRequested(_ increment: UInt64) {
        var total = UInt64Accounting.SaturatingSum(
            value: expertBytesRequested, didOverflow: expertByteAccountingOverflowed)
        total.add(increment)
        expertBytesRequested = total.value
        expertByteAccountingOverflowed = total.didOverflow
    }

    private func addDeterministicBytesRequested(_ increment: UInt64) {
        var total = UInt64Accounting.SaturatingSum(
            value: deterministicBytesRequested,
            didOverflow: deterministicByteAccountingOverflowed)
        total.add(increment)
        deterministicBytesRequested = total.value
        deterministicByteAccountingOverflowed = total.didOverflow
    }

    private func addAcquireWaitNanoseconds(_ increment: UInt64) {
        acquireWaitNanoseconds = addingTiming(increment, to: acquireWaitNanoseconds)
    }

    private func addComputeNanoseconds(_ increment: UInt64) {
        computeNanoseconds = addingTiming(increment, to: computeNanoseconds)
    }

    private func addingTiming(_ increment: UInt64, to value: UInt64) -> UInt64 {
        var total = UInt64Accounting.SaturatingSum(
            value: value, didOverflow: timingAccountingOverflowed)
        total.add(increment)
        if total.didOverflow { timingAccountingOverflowed = true }
        return total.value
    }
}
