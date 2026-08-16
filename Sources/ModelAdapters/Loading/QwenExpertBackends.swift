import Foundation
import MLX
import MLXBridge
import StorageCore

/// Where the gather loop's wall time went, split at the boundaries that call
/// for different fixes.
///
/// The M8.5 phase B record left 0.23 s/token of non-wait time unexplained:
/// turning read-ahead on moved time out of work that is not `acquire`, which
/// read-ahead has no business touching. `acquireWaitSeconds` alone cannot say
/// where it went, because "everything else" was one bucket. These are the four
/// things that bucket contains.
///
/// Accumulated without a lock. The gather loop is single-threaded and this
/// counts *its* time, exactly like ``QwenStreamedExpertBackend/expertReads``.
public final class QwenGatherPhases: @unchecked Sendable {
    /// Obtaining the operands: for the streamed backend an `acquire` plus three
    /// zero-copy adoptions, for the resident one a slice of a mapped stack.
    public private(set) var weightsSeconds: Double = 0
    /// Building the index and mask arrays and *expressing* the gather. Pure
    /// host work on a lazy graph — no device is involved yet.
    public private(set) var expressSeconds: Double = 0
    /// `eval()`. A device round trip, and the only phase that waits for the GPU.
    public private(set) var evalSeconds: Double = 0
    /// Giving the operands back: a pager slot release, or nothing.
    public private(set) var releaseSeconds: Double = 0
    /// Dispatches issued — `gatherQuantizedMM` calls, not experts.
    public private(set) var dispatches = 0

    public init() {}

    @inline(__always) func addWeights(_ start: UInt64) {
        weightsSeconds += MonotonicClock.seconds(since: start)
    }
    @inline(__always) func addExpress(_ start: UInt64) {
        expressSeconds += MonotonicClock.seconds(since: start)
    }
    @inline(__always) func addEval(_ start: UInt64) {
        evalSeconds += MonotonicClock.seconds(since: start)
    }
    @inline(__always) func addRelease(_ start: UInt64) {
        releaseSeconds += MonotonicClock.seconds(since: start)
    }
    @inline(__always) func countDispatch() { dispatches += 1 }
}

/// The one place the Qwen routed-expert loop is written.
///
/// Both backends go through it, so the streamed and resident runs cannot drift
/// apart in expert order, gather geometry, merge order or evaluation cadence —
/// which is what makes "bit-identical" a statement about the byte source and
/// nothing else. Same discipline as `gatherOverTiles` for K3, for the same
/// reason.
///
/// ## Per-expert, or batched
///
/// `expertsPerDispatch` selects between two programs, and both backends are
/// given the same value by their caller for the reason above.
///
/// - `1` is what phase A and phase B measured: one `gatherQuantizedMM` per
///   selected expert over a stack of one, merged with `which`. A decode step
///   issues 48 × 3 × 8 = **1152** of them, and phase B priced that at 0.425 ms
///   each — 89% of a step at the best measured configuration.
/// - `k > 1` assembles up to `k` experts into one contiguous stack and issues
///   **one** dispatch for them. At `k = 8` a decode step issues 48 × 3 = 144.
///
/// The batched form was deferred twice, in phase A and again in phase B, on the
/// grounds that assembling the stack would make the streamed arm copy where the
/// resident arm does not and so forfeit the bit-identity gate. That reasoning
/// had one flaw: it assumed only the streamed arm would have to assemble. Both
/// arms assemble here, from ``AffineWeights`` values their own byte source
/// produced, so the gate still compares one program against one program.
///
/// Whether the assembly moves a bit is a separate question, and it is measured
/// rather than argued — see `testBatchedGatherIsBitIdenticalToPerExpert`.
func gatherRoutedExperts(
    _ x: MLXArray,
    expertIds: [[Int]],
    outFeatures: Int,
    expertsPerDispatch: Int = 1,
    phases: QwenGatherPhases? = nil,
    /// Must return a **stack of one** (`experts == 1`), because the gather
    /// indexes into an expert axis. Both backends satisfy this the same way and
    /// for the same reason: a pager slot holds one expert.
    weightsForExpert: (Int) throws -> AffineWeights,
    releaseExpert: (Int) -> Void = { _ in }
) throws -> MLXArray {
    let tokens = expertIds.count
    let slots = expertIds.first?.count ?? 0
    guard tokens > 0, slots > 0 else {
        throw TinyK3Error.configuration("gatherRoutedExperts needs at least one (token, slot)")
    }
    guard expertsPerDispatch >= 1 else {
        throw TinyK3Error.configuration(
            "expertsPerDispatch must be at least 1, got \(expertsPerDispatch)")
    }

    // Ascending expert order, so the sequence of dispatches is a function of the
    // routing decision and not of iteration order over a dictionary. Stated once
    // here rather than in each of the two programs below, because it is the
    // property that makes either of them reproducible.
    var needed: Set<Int> = []
    for row in expertIds { needed.formUnion(row) }
    let ordered = needed.sorted()

    if expertsPerDispatch == 1 {
        return try gatherPerExpert(
            x, expertIds: expertIds, ordered: ordered, tokens: tokens, slots: slots,
            outFeatures: outFeatures, phases: phases,
            weightsForExpert: weightsForExpert, releaseExpert: releaseExpert)
    }
    return try gatherBatchedExperts(
        x, expertIds: expertIds, ordered: ordered, tokens: tokens, slots: slots,
        outFeatures: outFeatures, expertsPerDispatch: expertsPerDispatch, phases: phases,
        weightsForExpert: weightsForExpert, releaseExpert: releaseExpert)
}

/// One dispatch per selected expert. The program phase A and phase B measured,
/// kept intact so the batched form has an unchanged baseline to be read against.
private func gatherPerExpert(
    _ x: MLXArray,
    expertIds: [[Int]],
    ordered: [Int],
    tokens: Int,
    slots: Int,
    outFeatures: Int,
    phases: QwenGatherPhases?,
    weightsForExpert: (Int) throws -> AffineWeights,
    releaseExpert: (Int) -> Void
) throws -> MLXArray {
    var accumulator = MLXArray.zeros([tokens, slots, outFeatures], dtype: .float32)
    let index = MLXArray([Int32](repeating: 0, count: tokens), [tokens, 1])

    for expert in ordered {
        var mark = MonotonicClock.now()
        let weights = try weightsForExpert(expert)
        phases?.addWeights(mark)

        mark = MonotonicClock.now()
        // A stack of one, gathered with one index: at this geometry a pager slot
        // holds exactly one expert.
        let produced = affineGatherMM(x, weights, rhsIndices: index)
            .reshaped([tokens, 1, outFeatures])

        var maskValues = [Bool](repeating: false, count: tokens * slots)
        for (token, row) in expertIds.enumerated() {
            for (slot, chosen) in row.enumerated() where chosen == expert {
                maskValues[token * slots + slot] = true
            }
        }
        let mask = MLXArray(maskValues, [tokens, slots, 1])
        accumulator = which(mask, produced, accumulator)
        phases?.addExpress(mark)
        phases?.countDispatch()

        // Evaluating here is what releases a streamed expert's slot. Without it
        // the loop would hold every expert it has ever expressed, and the budget
        // would describe the graph rather than the working set.
        mark = MonotonicClock.now()
        accumulator.eval()
        phases?.addEval(mark)

        mark = MonotonicClock.now()
        releaseExpert(expert)
        phases?.addRelease(mark)
    }
    return accumulator
}

/// One dispatch per chunk of up to `expertsPerDispatch` experts.
///
/// ## Why it is chunked at all, rather than one dispatch for the whole layer
///
/// Every expert in a chunk is an operand of the same dispatch, so all of them
/// are leased at once and the pool must have room for the lot. A decode step
/// selects 8 experts per (layer, projection) and one chunk covers it; a prefill
/// selects the *union* over its tokens, which phase B measured at up to 58.6
/// distinct experts per (layer, projection) on a 31-token prompt. Batching that
/// unbounded would size the pool from the prompt length, which is precisely the
/// "budget as an emergent property" spec §5.3 forbids. So the chunk is the
/// budgeted number of experts, and the dispatch count falls out of it.
///
/// ## Why the chunk boundary cannot move a value
///
/// Each `(token, slot)` is produced by exactly one expert, so exactly one chunk
/// contributes to it and the merge is a selection rather than a sum. There is no
/// accumulation across chunks to reassociate. Rows whose expert is not in the
/// current chunk are pointed at local index 0 and discarded by the mask —
/// arithmetic that is computed and thrown away, never arithmetic that is added.
private func gatherBatchedExperts(
    _ x: MLXArray,
    expertIds: [[Int]],
    ordered: [Int],
    tokens: Int,
    slots: Int,
    outFeatures: Int,
    expertsPerDispatch: Int,
    phases: QwenGatherPhases?,
    weightsForExpert: (Int) throws -> AffineWeights,
    releaseExpert: (Int) -> Void
) throws -> MLXArray {
    let chunks = stride(from: 0, to: ordered.count, by: expertsPerDispatch).map {
        Array(ordered[$0..<min($0 + expertsPerDispatch, ordered.count)])
    }
    // One chunk covers every (token, slot), so there is nothing to merge into and
    // the gather's own output *is* the answer. This is the decode case, and
    // skipping the accumulator there is what takes the step from 1152 dispatches
    // to 144 rather than to 288.
    let single = chunks.count == 1
    var accumulator: MLXArray? =
        single ? nil : MLXArray.zeros([tokens, slots, outFeatures], dtype: .float32)

    for chunk in chunks {
        var position: [Int: Int] = [:]
        position.reserveCapacity(chunk.count)
        for (local, expert) in chunk.enumerated() { position[expert] = local }

        var mark = MonotonicClock.now()
        var parts: [AffineWeights] = []
        parts.reserveCapacity(chunk.count)
        for expert in chunk { parts.append(try weightsForExpert(expert)) }
        phases?.addWeights(mark)

        mark = MonotonicClock.now()
        let stacked = try stackExperts(parts)

        var indexValues = [Int32](repeating: 0, count: tokens * slots)
        var maskValues = [Bool](repeating: false, count: tokens * slots)
        for (token, row) in expertIds.enumerated() {
            for (slot, chosen) in row.enumerated() {
                guard let local = position[chosen] else { continue }
                indexValues[token * slots + slot] = Int32(local)
                maskValues[token * slots + slot] = true
            }
        }

        // `affineGatherMM` inserts the broadcast axes itself. That is not a
        // convenience here: `x` presented as `[tokens, 1, inFeatures]` is a legal
        // broadcast whenever `tokens == slots` and silently computes
        // `W[ids[t, j]] · x[j]`, which M3 measured at ~1.5 relative to the output
        // scale. A batched gather is exactly where that mistake becomes
        // expressible, so it goes through the wrapper that cannot express it.
        let produced = affineGatherMM(
            x, stacked, rhsIndices: MLXArray(indexValues, [tokens, slots]))

        let merged: MLXArray
        if let current = accumulator {
            merged = which(MLXArray(maskValues, [tokens, slots, 1]), produced, current)
        } else {
            merged = produced
        }
        phases?.addExpress(mark)
        phases?.countDispatch()

        // Same cadence rule as the per-expert loop, one chunk later: the slots
        // under this chunk's operands are not free until the graph reading them
        // has run.
        mark = MonotonicClock.now()
        merged.eval()
        phases?.addEval(mark)
        accumulator = merged

        mark = MonotonicClock.now()
        for expert in chunk { releaseExpert(expert) }
        phases?.addRelease(mark)
    }

    guard let result = accumulator else {
        throw TinyK3Error.configuration("a routing decision selected no experts")
    }
    return result
}

/// One gather operand out of `parts.count` stacks of one.
///
/// `gatherQuantizedMM` indexes a contiguous expert axis, and a streamed expert
/// arrives in its own pager slot, so the batch has to be assembled. Both
/// backends assemble it the same way and from the same shape — the resident arm
/// concatenates slices of a mapped stack, the streamed arm concatenates adopted
/// slots — because an assembly performed by only one of them would be a
/// difference in the program rather than in the byte source, which is the one
/// thing the bit-identity gate must not have to absorb.
///
/// This is the batched path's whole cost: three concatenations totalling one
/// expert-stack's bytes per dispatch, against the 7 dispatches it removes.
private func stackExperts(_ parts: [AffineWeights]) throws -> AffineWeights {
    guard let first = parts.first else {
        throw TinyK3Error.configuration("a batched gather needs at least one expert")
    }
    guard parts.count > 1 else { return first }
    return try AffineWeights(
        packed: concatenated(parts.map(\.packed), axis: 0),
        scales: concatenated(parts.map(\.scales), axis: 0),
        biases: concatenated(parts.map(\.biases), axis: 0),
        groupSize: first.groupSize,
        bits: first.bits)
}

// MARK: - Resident

/// Routed experts held in memory, read from the checkpoint's stacked tensors.
///
/// The reference arm for the streaming comparison, and the only one that can run
/// without a converted artifact. It keeps at most ``layerBudget`` layers'
/// experts — 340 MB each — because holding all 48 would be 15.2 GB and would
/// make "resident" mean "the thing this project exists to avoid".
public final class QwenResidentExpertBackend: QwenExpertBackend {

    public let checkpoint: QwenCheckpoint
    /// Layers whose expert stacks may be held at once.
    public let layerBudget: Int
    /// Experts per `gatherQuantizedMM`. Must match the streamed backend's value
    /// whenever the two are compared, which is what
    /// ``QwenMoETests/testStreamedEqualsResidentBitForBit`` sets it for.
    public let expertsPerDispatch: Int
    /// Where this backend's gather time went. Read per decode step by the bench.
    public let phases = QwenGatherPhases()

    private var stacks: [String: AffineWeights] = [:]
    private var order: [String] = []
    private let lock = NSLock()

    public init(checkpoint: QwenCheckpoint, layerBudget: Int = 1, expertsPerDispatch: Int = 8) {
        self.checkpoint = checkpoint
        self.layerBudget = max(1, layerBudget)
        self.expertsPerDispatch = max(1, expertsPerDispatch)
    }

    private func key(_ layer: Int, _ projection: QwenExpertProjection) -> String {
        "\(layer).\(projection.rawValue)"
    }

    private func stack(_ layer: Int, _ projection: QwenExpertProjection) throws -> AffineWeights {
        lock.lock()
        defer { lock.unlock() }
        let key = self.key(layer, projection)
        if let existing = stacks[key] { return existing }
        let weights = try checkpoint.affine(
            QwenTensors.experts(layer, projection.rawValue),
            groupSize: checkpoint.config.quantGroupSize,
            bits: checkpoint.config.quantBits,
            stacked: true)
        stacks[key] = weights
        order.append(key)
        // Three projections per layer, so the eviction unit is a whole layer's
        // worth of entries.
        while order.count > layerBudget * QwenExpertProjection.allCases.count {
            stacks.removeValue(forKey: order.removeFirst())
        }
        return weights
    }

    public func gather(
        _ x: MLXArray, layer: Int, projection: QwenExpertProjection, expertIds: [[Int]]
    ) throws -> MLXArray {
        let stacked = try stack(layer, projection)
        return try gatherRoutedExperts(
            x, expertIds: expertIds, outFeatures: stacked.outFeatures,
            expertsPerDispatch: expertsPerDispatch, phases: phases,
            weightsForExpert: { try stacked.expertStack($0) })
    }
}

// MARK: - Containers

/// The affine expert containers written by
/// `Tools/qwen_containers/convert_qwen_experts.py`, indexed by
/// `(layer, projection)`.
///
/// One tile is one whole expert. That is not a convention this reader imposes —
/// a Qwen expert projection is 1,572,864 elements, exactly three times the
/// format's 524,288-element alignment floor, so it lands with no remainder. The
/// mapping is still read from the manifest rather than assumed, for the reason
/// M6A gave: a future geometry change then shows up as different data instead of
/// as a silently wrong row offset.
public final class QwenExpertContainers {

    public struct Entry: Sendable {
        public let layer: Int
        public let projection: QwenExpertProjection
        public let path: String
        public let layout: QuantizedTileContainer.Layout
        public let outFeatures: Int
        public let inFeatures: Int
        /// `tile -> the single expert it holds`.
        public let tileExperts: [Int]

        public func tile(holding expert: Int) -> Int? {
            tileExperts.firstIndex(of: expert)
        }

        /// The pager request for one expert: one whole tile, one read, one slot.
        public func region(expert: Int, sourceID: UInt64) throws -> RegionDescriptor {
            guard let tile = tile(holding: expert) else {
                throw TinyK3Error.configuration(
                    "expert \(expert) is in no tile of layer \(layer) \(projection.rawValue)")
            }
            return layout.region(tile, sourceID: sourceID)
        }
    }

    public let directory: String
    public let entries: [String: Entry]
    public let totalBytes: Int
    public let layers: [Int]
    /// The largest tile in the set: the pager slot has to fit it.
    public let maximumTileBytes: Int

    private static func key(layer: Int, projection: QwenExpertProjection) -> String {
        "\(layer).\(projection.rawValue)"
    }

    public func entry(layer: Int, projection: QwenExpertProjection) throws -> Entry {
        guard let entry = entries[Self.key(layer: layer, projection: projection)] else {
            throw TinyK3Error.configuration(
                "no expert container for layer \(layer) \(projection.rawValue) in \(directory)")
        }
        return entry
    }

    public init(directory: String) throws {
        self.directory = directory
        let manifestURL = URL(fileURLWithPath: directory).appendingPathComponent("manifest.json")
        guard
            let root = try JSONSerialization.jsonObject(with: try Data(contentsOf: manifestURL))
                as? [String: Any],
            let containers = root["containers"] as? [[String: Any]]
        else {
            throw TinyK3Error.configuration("\(manifestURL.path) has no 'containers' list")
        }

        var entries: [String: Entry] = [:]
        var totalBytes = 0
        var layers: Set<Int> = []
        var maximumTileBytes = 0

        let slotToProjection = Dictionary(
            uniqueKeysWithValues: QwenExpertProjection.allCases.map { ($0.slot, $0) })

        for record in containers {
            guard
                let layer = record["layer"] as? Int,
                let slot = record["projection"] as? String,
                let projection = slotToProjection[slot],
                let file = record["file"] as? String,
                let outFeatures = record["out_features_per_expert"] as? Int,
                let inFeatures = record["in_features"] as? Int,
                let expertsPerTile = record["experts_per_tile"] as? Int,
                let tiles = record["tiles"] as? [[String: Any]]
            else {
                throw TinyK3Error.configuration("malformed container record: \(record)")
            }
            guard expertsPerTile == 1 else {
                throw TinyK3Error.configuration(
                    "\(file) packs \(expertsPerTile) experts per tile; this reader implements the "
                        + "whole-expert geometry (ADR-0003) and will not silently sub-index a tile")
            }
            let path = URL(fileURLWithPath: directory).appendingPathComponent(file).path
            // Re-read and re-derive rather than trust the manifest: `open` checks
            // the file is as long as it claims and that every derivable field
            // agrees with the geometry.
            let layout = try QuantizedTileContainer.open(path: path)
            guard layout.geometry.quantMode == .affine else {
                throw TinyK3Error.configuration(
                    "\(file) is a \(layout.geometry.quantMode.name) container; Qwen experts are affine")
            }
            guard layout.contentKind == .checkpointDerived else {
                throw TinyK3Error.configuration(
                    "\(file) declares \(layout.contentKind.name) content; expert containers are "
                        + "built from a checkpoint and must say so")
            }
            guard layout.geometry.rows == outFeatures, layout.geometry.cols == inFeatures else {
                throw TinyK3Error.configuration(
                    "\(file): header geometry \(layout.geometry.rows)x\(layout.geometry.cols) "
                        + "contradicts the manifest's \(outFeatures)x\(inFeatures)")
            }

            var tileExperts = [Int](repeating: -1, count: layout.tileCount)
            for tile in tiles {
                guard let index = tile["tile"] as? Int, let members = tile["experts"] as? [Int],
                    members.count == 1, index >= 0, index < layout.tileCount
                else {
                    throw TinyK3Error.configuration("malformed tile record in \(file): \(tile)")
                }
                tileExperts[index] = members[0]
            }
            guard !tileExperts.contains(-1) else {
                throw TinyK3Error.configuration("\(file): a tile lists no expert")
            }

            entries[Self.key(layer: layer, projection: projection)] = Entry(
                layer: layer, projection: projection, path: path, layout: layout,
                outFeatures: outFeatures, inFeatures: inFeatures, tileExperts: tileExperts)
            totalBytes += Int(layout.totalBytes)
            maximumTileBytes = max(maximumTileBytes, layout.tileStride)
            layers.insert(layer)
        }
        guard !entries.isEmpty else {
            throw TinyK3Error.configuration("\(manifestURL.path) lists no containers")
        }
        self.entries = entries
        self.totalBytes = totalBytes
        self.layers = layers.sorted()
        self.maximumTileBytes = maximumTileBytes
    }
}

// MARK: - Streamed

/// Routed experts streamed through the pager, one whole expert per read.
///
/// This is what M8.5 is for. The pool is a stated byte budget (spec §5.3), the
/// read unit is a whole expert (ADR-0003), and the arithmetic is
/// ``gatherPerExpert`` — the same function the resident backend calls, so the
/// only difference between the two arms is where the bytes came from.
public final class QwenStreamedExpertBackend: QwenExpertBackend {

    /// How far ahead of demand reads are scheduled.
    ///
    /// This is the knob the decode measurement turned out to be about. A Qwen
    /// expert tile is 884,736 B, and `2026-08-08-mac-internal-storage-curves`
    /// found this SSD's throughput is governed by *bytes in flight*: under
    /// 1 MiB it delivers 7–63% of its 6.9 GB/s peak, and 92% needs 4 MiB. One
    /// expert is 0.84 MiB, so a pager that has exactly one read outstanding is
    /// pinned to the bottom band by geometry — which is what a demand-driven
    /// loop has, no matter how large its slot budget.
    public enum PrefetchScope: String, Sendable, CaseIterable {
        /// Nothing is scheduled ahead of demand; each expert is read when the
        /// gather asks for it. What phase A measured, by omission.
        case none
        /// Schedule the experts of the projection about to be consumed. This is
        /// what the K3 streamed backend does inside its own `gather`.
        case projection
        /// Schedule all three projections as soon as the router has decided.
        /// The only scope that can hold more than one reader's window open at
        /// once, and therefore the only one that can reach the flat part of the
        /// device curve at this tile size.
        case layer
    }

    public struct Configuration: Sendable {
        /// Expert-cache budget, counted in whole experts.
        public var cacheBudgetExperts: Int
        public var queueDepth: Int
        public var readAhead: Int
        public var noCache: Bool
        public var cachePolicy: CachePolicy
        public var slotAcquireTimeout: TimeInterval?
        /// Check every streamed tile against the SHA-256 in its container
        /// header. Off by default because it hashes 864 KiB per expert on the
        /// hot path; on, it is the strongest statement available that the bytes
        /// the kernel saw are the bytes the converter wrote.
        public var verifyDigests: Bool
        /// Defaults to ``PrefetchScope/none``, which is the behaviour phase A
        /// measured. The measured cost of that default is in
        /// `docs/experiments/2026-08-09-qwen-mac-timing.md`; it is left as the
        /// default here so this change is a knob and a measurement rather than
        /// a silent performance edit, and the flip is its own commit.
        public var prefetchScope: PrefetchScope
        /// Experts covered by one `gatherQuantizedMM`.
        ///
        /// Every expert in a dispatch is leased for the whole of it, so this is
        /// a claim on the pool as well as a dispatch count: the plan below
        /// requires `expertsPerDispatch + 3 × readAhead ≤ slotCount`. Raising it
        /// past the top-k buys nothing on a decode step, which never selects
        /// more than 8 experts per (layer, projection); it only shortens a
        /// prefill's chunk count.
        ///
        /// Defaults to the top-k, 8. Measured in M8.5 phase C: 1.67× at the
        /// shipped prefetch scope and 1.07× with read-ahead on, no logit moved
        /// on any of the three screened prompts, and the same 1152 whole-expert
        /// reads per token. The pool peak it costs is real and worth knowing
        /// before combining it with read-ahead — 14 of 24 slots at scope `none`,
        /// but 21 of 24 at scope `layer`, which is 87% of the stated budget.
        public var expertsPerDispatch: Int

        public init(
            cacheBudgetExperts: Int = 24,
            queueDepth: Int = 4,
            readAhead: Int = 3,
            noCache: Bool = true,
            cachePolicy: CachePolicy = .leastRecentlyUsed,
            slotAcquireTimeout: TimeInterval? = 300,
            verifyDigests: Bool = false,
            prefetchScope: PrefetchScope = .none,
            expertsPerDispatch: Int = 8
        ) {
            self.prefetchScope = prefetchScope
            self.expertsPerDispatch = max(1, expertsPerDispatch)
            self.cacheBudgetExperts = cacheBudgetExperts
            self.queueDepth = queueDepth
            self.readAhead = readAhead
            self.noCache = noCache
            self.cachePolicy = cachePolicy
            self.slotAcquireTimeout = slotAcquireTimeout
            self.verifyDigests = verifyDigests
        }
    }

    public let containers: QwenExpertContainers
    public let configuration: Configuration
    public let config: QwenMoEConfig

    private let pool: BufferPool
    private var readers: [String: TileReader] = [:]
    private var sourceIDs: [String: UInt64] = [:]
    private let lock = NSLock()

    public private(set) var bytesRequested: UInt64 = 0
    /// Sticky. A saturated request total is reported, never wrapped smaller.
    public private(set) var byteAccountingOverflowed = false
    public private(set) var expertReads = 0
    public private(set) var digestChecks = 0
    /// Wall time the gather loop spent inside ``TileReader/acquire(_:)``.
    ///
    /// One half of the pair spec §18.1 insists on: this is *compute blocked on
    /// storage*, and `readerStatistics`' `slotWaitSeconds` is *storage blocked
    /// on the budget*. Either one alone misattributes a stall — a run can be
    /// slow because the bytes are not there yet, or because there is nowhere to
    /// put the bytes, and those call for opposite fixes.
    ///
    /// Accumulated without a lock, like the two counters above and for the same
    /// reason: the gather loop is single-threaded and this counts *its* waiting.
    public private(set) var acquireWaitSeconds: Double = 0
    /// The rest of the split: obtaining operands, expressing the gather,
    /// evaluating it, releasing the slots. ``acquireWaitSeconds`` is the part of
    /// ``QwenGatherPhases/weightsSeconds`` that was spent waiting for bytes.
    public let phases = QwenGatherPhases()

    public init(
        containers: QwenExpertContainers,
        config: QwenMoEConfig,
        configuration: Configuration = Configuration()
    ) throws {
        self.containers = containers
        self.config = config
        self.configuration = configuration

        // Three readers are live per layer (one per projection) and each may hold
        // `readAhead` slots. `PagedReadPlan`'s deadlock argument assumes one
        // reader, so the plan is built with the aggregate window: unless every
        // reader's in-flight window plus the experts the consumer holds fits, a
        // reader can wait for a slot only the consumer can free while the
        // consumer waits for bytes only that reader can deliver.
        //
        // `evalEveryK` is `expertsPerDispatch` and not 1, because a batched
        // gather leases every expert of a dispatch before evaluating any of
        // them. That is exactly the quantity the plan's invariant is stated in,
        // so batching is a change to the budget arithmetic and not merely to the
        // dispatch count.
        let readerCount = QwenExpertProjection.allCases.count
        let aggregateReadAhead = readerCount * configuration.readAhead
        let floor = aggregateReadAhead + configuration.expertsPerDispatch
        let slotCount = max(floor, configuration.cacheBudgetExperts)
        _ = try PagedReadPlan(
            slotCount: slotCount, readAhead: aggregateReadAhead,
            evalEveryK: configuration.expertsPerDispatch)

        self.pool = try BufferPool(
            configuration: BufferPoolConfiguration(
                slotCount: slotCount,
                slotBytes: containers.maximumTileBytes,
                alignment: max(AlignedBuffer.pageSize, QuantizedTileContainer.regionAlignment),
                cachePolicy: configuration.cachePolicy,
                poisonOnRelease: false))

        // A distinct source id per container, so two layers' tile 7 are not the
        // same cache key. Assigned once, deterministically, from the sorted
        // manifest keys.
        for (index, key) in containers.entries.keys.sorted().enumerated() {
            sourceIDs[key] = UInt64(index)
        }
    }

    deinit { shutdown() }

    public func shutdown() {
        lock.lock()
        defer { lock.unlock() }
        for reader in readers.values { reader.shutdown() }
        readers.removeAll()
        pool.evictAll()
    }

    /// The pool's stated budget and what it actually peaked at.
    public var statistics: BufferPoolStatistics { pool.statistics }

    /// Every reader's I/O, pooled. Summarising per reader and averaging the
    /// summaries would report percentiles no read ever had (M8 finding).
    public var readerStatistics: [TileReaderStatistics] {
        readers.values.map(\.statistics)
    }

    private func reader(for entry: QwenExpertContainers.Entry) throws -> TileReader {
        let key = "\(entry.layer).\(entry.projection.rawValue)"
        if let existing = readers[key] { return existing }
        let reader = try TileReader(
            path: entry.path, pool: pool,
            configuration: TileReaderConfiguration(
                queueDepth: configuration.queueDepth,
                readAhead: configuration.readAhead,
                noCache: configuration.noCache,
                slotAcquireTimeout: configuration.slotAcquireTimeout))
        readers[key] = reader
        return reader
    }

    private func sourceID(_ entry: QwenExpertContainers.Entry) -> UInt64 {
        sourceIDs["\(entry.layer).\(entry.projection.rawValue)"] ?? 0
    }

    /// Queue the reads for a routing decision before any of them is consumed.
    ///
    /// Without this every expert is demanded one at a time and the read-ahead
    /// window never fills: the reader cannot start expert n+1 until the loop asks
    /// for it. Scheduling in consumption order means the window is always working
    /// on what comes next.
    public func prefetch(layer: Int, expertIds: [[Int]]) throws {
        guard configuration.prefetchScope == .layer else { return }
        for projection in QwenExpertProjection.allCases {
            try schedule(layer: layer, projection: projection, expertIds: expertIds)
        }
    }

    /// Queue one projection's selected experts, skipping what the pool already
    /// holds.
    ///
    /// The filter is not an optimisation: the reader would fetch a cached
    /// region regardless, and that redundant read takes a slot — evicting, in
    /// the worst case, the very entry the request was about to hit.
    private func schedule(
        layer: Int, projection: QwenExpertProjection, expertIds: [[Int]]
    ) throws {
        var needed: Set<Int> = []
        for row in expertIds { needed.formUnion(row) }
        let entry = try containers.entry(layer: layer, projection: projection)
        let identity = sourceID(entry)
        let regions = try needed.sorted().map { try entry.region(expert: $0, sourceID: identity) }
        try reader(for: entry).schedule(regions.filter { !pool.isCached($0.identity) })
    }

    public func gather(
        _ x: MLXArray, layer: Int, projection: QwenExpertProjection, expertIds: [[Int]]
    ) throws -> MLXArray {
        let entry = try containers.entry(layer: layer, projection: projection)
        let reader = try self.reader(for: entry)
        let layout = entry.layout
        let identity = sourceID(entry)

        if configuration.prefetchScope == .projection {
            try schedule(layer: layer, projection: projection, expertIds: expertIds)
        }

        // Leases are held until the graph that reads from them has been
        // evaluated: at most one per call in the per-expert program, and up to
        // `expertsPerDispatch` in the batched one. The release is explicit
        // rather than relying on the lazy graph's own reference, which is the
        // kind of assumption that fails once and is unfindable.
        var live: [Int: (SlotLease, AffineWeights)] = [:]

        let result = try gatherRoutedExperts(
            x, expertIds: expertIds, outFeatures: entry.outFeatures,
            expertsPerDispatch: configuration.expertsPerDispatch, phases: phases,
            weightsForExpert: { expert in
                let region = try entry.region(expert: expert, sourceID: identity)
                var total = UInt64Accounting.SaturatingSum(
                    value: self.bytesRequested,
                    didOverflow: self.byteAccountingOverflowed)
                total.add(UInt64(region.length))
                self.bytesRequested = total.value
                self.byteAccountingOverflowed = total.didOverflow
                self.expertReads += 1
                let waitStarted = MonotonicClock.now()
                let lease = try reader.acquire(region)
                self.acquireWaitSeconds += MonotonicClock.seconds(since: waitStarted)

                if self.configuration.verifyDigests {
                    let bytes = UnsafeRawBufferPointer(
                        start: lease.slot.pointer, count: layout.tileStride)
                    guard let tile = entry.tile(holding: expert) else {
                        throw TinyK3Error.configuration("expert \(expert) has no tile")
                    }
                    try QuantizedTileContainer.verifyTileDigest(
                        bytes, layout: layout, tile: tile)
                    self.digestChecks += 1
                }

                // Three adoptions out of one slot, which is exactly why the
                // container aligns all three sub-regions independently.
                try lease.retain()
                try lease.retain()
                try lease.retain()
                let base = lease.slot.pointer
                let weights = try AffineWeights.adopting(
                    packedPointer: base,
                    scalesPointer: base + layout.scaleOffsetInTile,
                    biasesPointer: base + (layout.biasOffsetInTile ?? 0),
                    scaleDType: Self.dtype(layout.geometry.scaleDType),
                    experts: 1,
                    outFeatures: entry.outFeatures,
                    inFeatures: entry.inFeatures,
                    groupSize: layout.geometry.groupSize,
                    bits: layout.geometry.bits,
                    packedFinalizer: { lease.release() },
                    scalesFinalizer: { lease.release() },
                    biasesFinalizer: { lease.release() })
                live[expert] = (lease, weights)
                return weights
            },
            releaseExpert: { expert in
                if let (lease, _) = live.removeValue(forKey: expert) { lease.release() }
            })

        return result
    }

    private static func dtype(_ scaleDType: TileScaleDType) -> DType {
        switch scaleDType {
        case .uint8E8M0: return .uint8
        case .float16: return .float16
        case .bfloat16: return .bfloat16
        case .float32: return .float32
        }
    }
}
