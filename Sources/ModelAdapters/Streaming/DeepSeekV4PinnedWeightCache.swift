import Foundation
import MLX
import MLXBridge
import StorageCore

/// The deterministic V4 weights the memory dial decided to hold in RAM.
///
/// ## What is held, and why that form
///
/// K3's pinned tier holds a layer's **stored** BF16 bytes and widens per use, so
/// a pinned layer costs exactly the bytes it stops reading — ratio 1.000, the
/// ceiling of that ladder (`docs/design/memory-dial.md` §2). V4 holds the
/// **loaded** form instead: the ``BlockFP8Weights`` value, whose `packed` array
/// adopts the tile's packed region unchanged and whose `expandedScales` array is
/// the 128×128 block grid expanded to MLX's one-scale-per-32-columns layout.
///
/// That is a deliberate departure and it is the one rule of K3's design this
/// tier does not inherit. The reason is that a V4 load is not only a read. Every
/// `loadBlockFP8` also runs `BlockFP8Weights.adopting`, which scans the whole
/// packed matrix for non-finite E4M3 encodings and rebuilds the expanded scale
/// grid — host work, on the decode thread. Holding the stored tile and
/// re-adopting per use would return ratio 1.000 and pay both again on every
/// pass; holding the loaded form pays neither. The price is the expanded scale
/// grid:
/// `packed/32` against a stored tile's `packed/16384`, so residency is about
/// **1.031× stored** and a pinned V4 layer scores ~0.970 rather than 1.000.
/// Still 4.2× better than the best expert hot set ever measured (0.230, M8), so
/// the tier order the ladder produces is unchanged.
///
/// The adoption half of that argument got much smaller on 2026-08-15: the
/// vectorized scan and the memcpy-run scale expansion took a 32 MiB matrix from
/// 22.9 ms to 1.5–2.2 ms, and a trusted run's second and later adoptions of a
/// tensor to 0.07 ms (`docs/experiments/2026-08-15-v4-adoption-fast-path.md`).
/// The decision is unchanged — the tier's main saving was always the read, and
/// re-expanding the grid per pass would still be pure waste — but the margin it
/// wins by is now the read, not the adoption.
///
/// Numerical identity is structural rather than compared. A pinned tensor is the
/// *same* `MLXArray` on every pass — MLX arrays are reference-counted values and
/// the projection reads them without consuming them — so a pinned layer cannot
/// compute a different value from the streamed layer that produced it. There is
/// no second widening path to drift against, which is the property K3 had to
/// build `materialiseKeepingBytes()` to obtain.
///
/// ## What it costs to fill
///
/// The bytes are read once, by the pass that first touches the tensor, through
/// the same descriptor and the same `pread` loop the serial path uses, and under
/// the same ``TileDigestPolicy``. A pinned tensor is therefore digested exactly
/// as often as the run's policy says it should be — once, if the policy is
/// `.trustHeldAuthority`, which is what the product run carries — and **never
/// re-digested afterwards**, because there is no second read to digest.
///
/// The fill is inline: on the filling pass the bytes arrive in the critical path
/// and the pass pays for them, exactly as K3's does. Every pass after it pays
/// nothing. A single-pass run therefore buys nothing from the tier and pays its
/// fill, and the counters below are split so a record can say which of the two
/// it measured.
final class DeepSeekV4PinnedWeightCache: @unchecked Sendable {

    /// The layers the plan pinned. Fixed for the life of the run: the dial is a
    /// decision made *before* the run, not a cache policy discovered during it
    /// (spec §5.3, and M8's finding that LRU under scan pollution is worse than
    /// no cache at all).
    let layers: Set<Int>

    /// Whether the plan pinned the output head — the ladder's one non-layer
    /// rung. The table itself lives in ``DeepSeekV4GlobalArtifact``, which is
    /// what owns the door it is served through; this type owns the budget, the
    /// counters and the refusal, exactly as it does for a layer tensor.
    let pinsOutputHead: Bool

    /// What the plan said these layers may occupy. Refused, never clamped: a
    /// fill that would cross it is declined and counted, and the tensor is read
    /// serially on every pass instead. The plan's census is a conservative
    /// upper bound on what a layer actually loads, so in a healthy run this
    /// never fires — which is why it is a counter rather than an assertion.
    let budgetBytes: UInt64

    private let lock = NSLock()
    private var blockFP8: [LayerTensor: BlockFP8Weights] = [:]
    private var floating: [LayerTensor: MLXArray] = [:]

    private struct LayerTensor: Hashable {
        let layer: Int
        let tensor: String
    }

    private(set) var residentBytes: UInt64 = 0
    private(set) var freedBytes: UInt64 = 0
    /// Bytes read into this cache. They are counted in
    /// ``DeepSeekV4ReadAccountingSnapshot/deterministicBytesRead`` **once**, by
    /// the pass that read them, exactly like any other deterministic read.
    private(set) var bytesLoaded: UInt64 = 0
    /// Bytes a load took from this cache on a pass that did *not* read them.
    /// These are the bytes the tier saved, and they are never added to bytes
    /// read — that is the whole point of the counter existing separately.
    private(set) var bytesServed: UInt64 = 0
    private(set) var servedCount = 0
    private(set) var filledTensorCount = 0
    /// A fill declined because it would cross ``budgetBytes``. The tensor is
    /// read serially every pass after that, which is correct and slow, so it is
    /// counted rather than absorbed.
    private(set) var refusedFillCount = 0
    private(set) var accountingOverflowed = false

    init(layers: Set<Int>, budgetBytes: UInt64, pinsOutputHead: Bool = false) {
        self.layers = layers
        self.budgetBytes = budgetBytes
        self.pinsOutputHead = pinsOutputHead
    }

    func isPinned(_ layer: Int) -> Bool { layers.contains(layer) }

    // MARK: The output head

    private var outputHeadResident: UInt64 = 0

    /// Reserve the head table's residency, or refuse it and count the refusal.
    ///
    /// Called once, by the pass that starts filling the table. Refused rather
    /// than clamped for the same reason a layer fill is: the plan declared what
    /// this tier may hold, and a tier that quietly held more would make the
    /// run's own accounting a guess. A refused head is read every pass, which
    /// is correct and slow.
    func admitOutputHead(residentBytes cost: UInt64) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard pinsOutputHead, outputHeadResident == 0 else { return false }
        guard admits(cost) else {
            refusedFillCount += 1
            return false
        }
        outputHeadResident = cost
        residentBytes = adding(cost, to: residentBytes)
        filledTensorCount += 1
        return true
    }

    /// Bytes read into the head table. They are already inside
    /// `deterministicBytesRead` — the filling pass read them — so this counter
    /// says what the fill cost, never what it saved.
    func noteOutputHeadLoaded(readBytes: UInt64) {
        lock.lock()
        bytesLoaded = adding(readBytes, to: bytesLoaded)
        lock.unlock()
    }

    /// One head window served without a read: the bytes the rung saved, in the
    /// unit `deterministicBytesRead` is stated in.
    func noteOutputHeadServed(readBytes: UInt64) {
        lock.lock()
        bytesServed = adding(readBytes, to: bytesServed)
        servedCount += 1
        lock.unlock()
    }

    /// The head table was freed. Its bytes move from resident to freed, like
    /// every other byte this tier held.
    func noteOutputHeadReleased() {
        lock.lock()
        freedBytes = adding(outputHeadResident, to: freedBytes)
        residentBytes = residentBytes >= outputHeadResident
            ? residentBytes - outputHeadResident : 0
        outputHeadResident = 0
        lock.unlock()
    }

    // MARK: Block-FP8 matrices

    /// The pinned ``BlockFP8Weights`` for this tensor, or `nil` if the layer is
    /// not pinned or the tensor has not been filled yet.
    func blockFP8(layer: Int, tensor: String) -> BlockFP8Weights? {
        guard layers.contains(layer) else { return nil }
        lock.lock()
        defer { lock.unlock() }
        guard let weights = blockFP8[LayerTensor(layer: layer, tensor: tensor)] else {
            return nil
        }
        return weights
    }

    /// Note that a pinned block-FP8 tensor was served without a read.
    ///
    /// `readBytes` is what the serial path would have read for it — one tile
    /// stride — so the counter states the bytes the tier saved in the same unit
    /// `deterministicBytesRead` is stated in.
    func noteBlockFP8Served(readBytes: UInt64) {
        lock.lock()
        bytesServed = adding(readBytes, to: bytesServed)
        servedCount += 1
        lock.unlock()
    }

    /// Adopt one block-FP8 tensor the caller has just read.
    ///
    /// `readBytes` is the tile stride the read charged to
    /// `deterministicBytesRead`; `residentBytes` is what holding the loaded form
    /// costs, which is larger by the scale expansion.
    /// - Returns: whether this tier now holds these weights. False for an
    ///   unpinned layer and for a refused fill — in both cases the caller's
    ///   graph is still the only owner of the bytes, which is what decides
    ///   whether its projection has to be forced before the loader returns.
    ///   True also when the tensor was already held, because the tier owns it
    ///   either way.
    @discardableResult
    func fill(
        layer: Int, tensor: String, weights: BlockFP8Weights,
        readBytes: UInt64, residentBytes cost: UInt64
    ) -> Bool {
        guard layers.contains(layer) else { return false }
        let key = LayerTensor(layer: layer, tensor: tensor)
        lock.lock()
        defer { lock.unlock() }
        guard blockFP8[key] == nil else { return true }
        guard admits(cost) else {
            refusedFillCount += 1
            return false
        }
        blockFP8[key] = weights
        residentBytes = adding(cost, to: residentBytes)
        bytesLoaded = adding(readBytes, to: bytesLoaded)
        filledTensorCount += 1
        return true
    }

    // MARK: Plain BF16/F32 tensors

    func floatingTensor(layer: Int, tensor: String) -> MLXArray? {
        guard layers.contains(layer) else { return nil }
        lock.lock()
        defer { lock.unlock() }
        return floating[LayerTensor(layer: layer, tensor: tensor)]
    }

    func noteFloatingServed(readBytes: UInt64) {
        lock.lock()
        bytesServed = adding(readBytes, to: bytesServed)
        servedCount += 1
        lock.unlock()
    }

    /// - Returns: whether this tier now holds this array, on the same terms as
    ///   the block-FP8 fill above.
    @discardableResult
    func fill(
        layer: Int, tensor: String, array: MLXArray,
        readBytes: UInt64, residentBytes cost: UInt64
    ) -> Bool {
        guard layers.contains(layer) else { return false }
        let key = LayerTensor(layer: layer, tensor: tensor)
        lock.lock()
        defer { lock.unlock() }
        guard floating[key] == nil else { return true }
        guard admits(cost) else {
            refusedFillCount += 1
            return false
        }
        floating[key] = array
        residentBytes = adding(cost, to: residentBytes)
        bytesLoaded = adding(readBytes, to: bytesLoaded)
        filledTensorCount += 1
        return true
    }

    // MARK: Lifecycle

    /// Free every pinned byte. Idempotent, and called from the workload's
    /// shutdown: pinned bytes are the largest allocation a stated V4 run holds,
    /// and a run that finished should not still be holding them.
    func release() {
        lock.lock()
        blockFP8.removeAll()
        floating.removeAll()
        freedBytes = adding(residentBytes, to: freedBytes)
        residentBytes = 0
        lock.unlock()
    }

    /// `resident + cost <= budget`, saturating rather than trapping. Called
    /// under `lock`.
    private func admits(_ cost: UInt64) -> Bool {
        let next = residentBytes.addingReportingOverflow(cost)
        return !next.overflow && next.partialValue <= budgetBytes
    }

    private func adding(_ increment: UInt64, to value: UInt64) -> UInt64 {
        var sum = UInt64Accounting.SaturatingSum(
            value: value, didOverflow: accountingOverflowed)
        sum.add(increment)
        if sum.didOverflow { accountingOverflowed = true }
        return sum.value
    }

    var metrics: DeepSeekV4PinnedTierMetrics {
        lock.lock()
        defer { lock.unlock() }
        var metrics = DeepSeekV4PinnedTierMetrics()
        metrics.pinnedLayerCount = layers.count
        metrics.pinnedLayers = layers.sorted()
        metrics.pinsOutputHead = pinsOutputHead
        metrics.outputHeadResidentBytes = outputHeadResident
        metrics.budgetBytes = budgetBytes
        metrics.residentBytes = residentBytes
        metrics.freedBytes = freedBytes
        metrics.bytesLoaded = bytesLoaded
        metrics.bytesServed = bytesServed
        metrics.servedCount = servedCount
        metrics.filledTensorCount = filledTensorCount
        metrics.refusedFillCount = refusedFillCount
        metrics.accountingOverflowed = accountingOverflowed
        return metrics
    }
}

// MARK: - Metrics

/// What the V4 pinned tier held and what it saved, for the run's JSON
/// (AGENTS.md: a parameter that changes a measured result is reported next to
/// the result).
///
/// Two identities hold over these counters and both are asserted rather than
/// assumed, because between them they are the claim that the tier does not
/// invent or lose bytes:
///
/// - **residency**: every resident byte is held or freed, and never exceeds the
///   budget the plan declared.
/// - **service**: a byte can only be served after it was loaded, and served
///   bytes are outside `deterministicBytesRead` by construction.
public struct DeepSeekV4PinnedTierMetrics: Codable, Sendable, Equatable {

    /// Zero everywhere is the unpinned default — the product scale, and what
    /// every V4 record before this feature measured.
    public var pinnedLayerCount: Int = 0
    public var pinnedLayers: [Int] = []
    /// Whether the plan's globals rung was pinned: the output head table.
    public var pinsOutputHead: Bool = false
    /// What the head table occupies once its fill has been admitted. Zero while
    /// the fill has not started, and zero when it was refused.
    public var outputHeadResidentBytes: UInt64 = 0
    /// The residency the plan declared for these layers. A fill that would
    /// cross it is refused and counted; the tier never trims the set silently.
    public var budgetBytes: UInt64 = 0

    public var residentBytes: UInt64 = 0
    public var freedBytes: UInt64 = 0
    /// Bytes read to fill the tier, in the unit `deterministicBytesRead` uses.
    public var bytesLoaded: UInt64 = 0
    /// Bytes a later pass did **not** read because the tier served them.
    public var bytesServed: UInt64 = 0

    public var servedCount: Int = 0
    public var filledTensorCount: Int = 0
    public var refusedFillCount: Int = 0

    /// Sticky. Once true, no equality over the saturated counters is evidence.
    public var accountingOverflowed: Bool = false

    /// `resident + freed == resident-ever`, and residency never crossed the
    /// declared budget.
    public var isResidencyBalanced: Bool {
        guard !accountingOverflowed else { return false }
        guard UInt64Accounting.checkedSum([residentBytes, freedBytes]) != nil else {
            return false
        }
        return residentBytes <= budgetBytes
    }

    /// No byte is served before something was loaded, and a tier that holds
    /// nothing serves nothing.
    public var isServiceBalanced: Bool {
        guard !accountingOverflowed else { return false }
        return bytesServed == 0 || bytesLoaded > 0
    }

    public var isAccountingBalanced: Bool { isResidencyBalanced && isServiceBalanced }

    public init() {}

    /// One line for a console, in the shape the phase summary uses.
    public var summaryLine: String {
        guard pinnedLayerCount > 0 || pinsOutputHead else { return "[v4 pins] none" }
        let units = pinsOutputHead
            ? "\(pinnedLayerCount) layers + output head" : "\(pinnedLayerCount) layers"
        return String(
            format: "[v4 pins] %@, %llu B resident of %llu B budget, "
                + "%llu B loaded / %llu B served over %d hits, %d tensors held",
            units, residentBytes, budgetBytes, bytesLoaded, bytesServed,
            servedCount, filledTensorCount)
    }
}
