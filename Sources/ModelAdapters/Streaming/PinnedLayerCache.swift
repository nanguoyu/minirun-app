import Foundation
import MLX
import StorageCore

/// The deterministic layers the memory dial decided to hold in RAM.
///
/// ## What is held, and why that form
///
/// A pinned layer holds its **stored** bytes — BF16, exactly as written — and
/// the consume path still widens them per use. That is the decision the dial's
/// whole ladder rests on (`docs/design/memory-dial.md` §2): because the resident
/// cost is the stored size and the saving is the same stored size, a pinned
/// layer returns 1.000 bytes saved per resident byte, which is the ceiling and
/// the reason no expert hot set can outrank one.
///
/// Holding the *widened* form instead would double the residency for the same
/// saving — 0.5 on the ladder, below the globals bundle — and would also mean
/// the pinned tier computed something the streamed tier did not. Widening per
/// use keeps one code path: ``RawTensor/materialiseKeepingBytes()`` is the same
/// instruction sequence as ``RawTensor/materialise()``, so a pinned layer's
/// values are the streamed layer's values by construction rather than by
/// comparison.
///
/// ## What it costs to fill
///
/// The bytes are read once, by the pass that first touches the layer, through
/// the same `pread` loop the serial path uses. The read-ahead is told to *skip*
/// scheduling a pinned layer — there is nothing to stage for a layer that will
/// be resident — so on the filling pass a pinned layer's bytes arrive **inline**
/// and the pass pays for them in the critical path. Every pass after it pays
/// nothing. A single-token run therefore buys nothing from the tier and pays its
/// fill; the tier is for the second token onwards, and the counters below are
/// split so a record can say which of the two it measured.
final class PinnedLayerCache {

    /// The layers the plan pinned. Fixed for the life of the run: the dial is a
    /// decision made *before* the run, not a cache policy discovered during it
    /// (spec §5.3, and M8's finding that LRU under scan pollution is worse than
    /// no cache at all).
    let layers: Set<Int>
    /// What the plan said these layers may occupy. Refused, never clamped, and
    /// checked against the artifact's own byte table before a run starts.
    let budgetBytes: UInt64

    private var held: [Int: [String: RawTensor]] = [:]

    private(set) var residentBytes: UInt64 = 0
    private(set) var freedBytes: UInt64 = 0
    /// Bytes read into this cache. They are counted in
    /// ``ArtifactWeightSource/deterministicBytesRead`` **once**, by the pass
    /// that read them, exactly like any other deterministic read.
    private(set) var bytesLoaded: UInt64 = 0
    /// Bytes a widening took from this cache on a pass that did *not* read
    /// them. These are the bytes the tier saved, and they are never added to
    /// bytes read — that is the whole point of the counter existing separately.
    private(set) var bytesServed: UInt64 = 0
    /// Bytes a widening took on the pass that loaded them. Already inside bytes
    /// read; carried so the two halves of ``bytesWidened`` add up.
    private(set) var bytesChargedAsRead: UInt64 = 0
    private(set) var hitCount = 0
    /// A tensor the assembly asked a pinned layer for that the tensor plan did
    /// not cover. It is read serially every pass, which is correct and slow, so
    /// it is counted rather than absorbed.
    private(set) var missedTensorCount = 0
    private(set) var filledLayerCount = 0
    private(set) var servedLayerCount = 0
    private var accountingOverflowed = false

    init(layers: Set<Int>, budgetBytes: UInt64) {
        self.layers = layers
        self.budgetBytes = budgetBytes
    }

    deinit { release() }

    func isPinned(_ layer: Int) -> Bool { layers.contains(layer) }

    /// Has this layer's fill already happened?
    func isLoaded(_ layer: Int) -> Bool { held[layer] != nil }

    /// Adopt one layer's stored bytes. The caller has just read them, so they
    /// are already charged to ``ArtifactWeightSource/deterministicBytesRead``.
    func fill(layer: Int, tensors: [String: RawTensor]) {
        guard held[layer] == nil else { return }
        let sum = UInt64Accounting.saturatingSum(tensors.values.map { UInt64($0.byteCount) })
        if sum.didOverflow { accountingOverflowed = true }
        let bytes = sum.value
        held[layer] = tensors
        residentBytes = adding(bytes, to: residentBytes)
        bytesLoaded = adding(bytes, to: bytesLoaded)
        filledLayerCount += 1
    }

    /// The `[rows, cols]` float32 array for `name`, or `nil` if this layer is
    /// not pinned or the plan never held that tensor.
    ///
    /// - Parameter justFilled: whether this pass is the one that read the bytes.
    ///   It decides which bucket the widening is charged to, and nothing else:
    ///   the arithmetic is identical either way.
    func widened(layer: Int, name: String, justFilled: Bool) throws -> MLXArray? {
        guard let tensors = held[layer] else { return nil }
        guard let tensor = tensors[name] else {
            missedTensorCount += 1
            return nil
        }
        let widened = try tensor.materialiseKeepingBytes()
        hitCount += 1
        if justFilled {
            bytesChargedAsRead = adding(UInt64(tensor.byteCount), to: bytesChargedAsRead)
        } else {
            bytesServed = adding(UInt64(tensor.byteCount), to: bytesServed)
        }
        return widened
    }

    /// Note that a whole layer was served from residency — one layer's worth of
    /// deterministic bytes that never reached the link on this pass.
    func noteLayerServed() { servedLayerCount += 1 }

    /// Free every pinned byte. Idempotent, and called from
    /// ``ArtifactWeightSource/shutdown()``: pinned bytes are the largest
    /// allocation the process holds after layer 0, and a run that finished
    /// should not still be holding them.
    func release() {
        for tensors in held.values {
            for tensor in tensors.values { tensor.discard() }
        }
        held.removeAll()
        freedBytes = adding(residentBytes, to: freedBytes)
        residentBytes = 0
    }

    private func adding(_ increment: UInt64, to value: UInt64) -> UInt64 {
        var sum = UInt64Accounting.SaturatingSum(
            value: value, didOverflow: accountingOverflowed)
        sum.add(increment)
        if sum.didOverflow { accountingOverflowed = true }
        return sum.value
    }

    var metrics: PinnedTierMetrics {
        var metrics = PinnedTierMetrics()
        metrics.pinnedLayerCount = layers.count
        metrics.pinnedLayers = layers.sorted()
        metrics.budgetBytes = budgetBytes
        metrics.residentBytes = residentBytes
        metrics.freedBytes = freedBytes
        metrics.bytesLoaded = bytesLoaded
        metrics.bytesServed = bytesServed
        metrics.bytesChargedAsRead = bytesChargedAsRead
        metrics.hitCount = hitCount
        metrics.missedTensorCount = missedTensorCount
        metrics.filledLayerCount = filledLayerCount
        metrics.servedLayerCount = servedLayerCount
        metrics.accountingOverflowed = accountingOverflowed
        return metrics
    }
}

// MARK: - Metrics

/// What the pinned tier held and what it saved, for the run's JSON (AGENTS.md:
/// a parameter that changes a measured result is reported next to the result).
///
/// Two identities hold over these counters and both are asserted rather than
/// assumed, because between them they are the claim that the tier does not
/// invent or lose bytes:
///
/// - **residency**: every loaded byte is resident or freed.
/// - **service**: every byte handed to a widening was either read by the pass
///   that widened it (and is therefore inside `deterministicBytesRead`) or
///   served from residency (and is therefore *not*).
public struct PinnedTierMetrics: Codable, Sendable, Equatable {

    /// Zero everywhere is the unpinned default, which is what every record
    /// before this feature measured.
    public var pinnedLayerCount: Int = 0
    public var pinnedLayers: [Int] = []
    /// The residency the plan declared for these layers. The run refuses a plan
    /// whose layers do not fit it; it never trims the set to make it true.
    public var budgetBytes: UInt64 = 0

    public var residentBytes: UInt64 = 0
    public var freedBytes: UInt64 = 0
    public var bytesLoaded: UInt64 = 0
    public var bytesServed: UInt64 = 0
    public var bytesChargedAsRead: UInt64 = 0

    public var hitCount: Int = 0
    public var missedTensorCount: Int = 0
    public var filledLayerCount: Int = 0
    public var servedLayerCount: Int = 0

    /// Sticky. Once true, no equality over the saturated counters is evidence.
    public var accountingOverflowed: Bool = false

    /// Bytes that reached a widening, from either bucket.
    public var bytesWidened: UInt64 {
        UInt64Accounting.saturatingSum([bytesChargedAsRead, bytesServed]).value
    }

    /// `resident + freed == loaded`.
    public var isResidencyBalanced: Bool {
        guard !accountingOverflowed,
            let accounted = UInt64Accounting.checkedSum([residentBytes, freedBytes])
        else { return false }
        return accounted == bytesLoaded
    }

    /// Every widened byte is charged exactly once, and no byte is served before
    /// it is loaded.
    public var isServiceBalanced: Bool {
        !accountingOverflowed
            && UInt64Accounting.checkedSum([bytesChargedAsRead, bytesServed]) != nil
            && bytesChargedAsRead <= bytesLoaded && (bytesServed == 0 || bytesLoaded > 0)
    }

    public var isAccountingBalanced: Bool { isResidencyBalanced && isServiceBalanced }

    enum CodingKeys: String, CodingKey {
        case pinnedLayerCount, pinnedLayers, budgetBytes
        case residentBytes, freedBytes, bytesLoaded, bytesServed, bytesChargedAsRead
        case hitCount, missedTensorCount, filledLayerCount, servedLayerCount
        case accountingOverflowed
        case bytesWidened, isAccountingBalanced
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        pinnedLayerCount = try container.decode(Int.self, forKey: .pinnedLayerCount)
        pinnedLayers = try container.decode([Int].self, forKey: .pinnedLayers)
        budgetBytes = try container.decode(UInt64.self, forKey: .budgetBytes)
        residentBytes = try container.decode(UInt64.self, forKey: .residentBytes)
        freedBytes = try container.decode(UInt64.self, forKey: .freedBytes)
        bytesLoaded = try container.decode(UInt64.self, forKey: .bytesLoaded)
        bytesServed = try container.decode(UInt64.self, forKey: .bytesServed)
        bytesChargedAsRead = try container.decode(UInt64.self, forKey: .bytesChargedAsRead)
        hitCount = try container.decode(Int.self, forKey: .hitCount)
        missedTensorCount = try container.decode(Int.self, forKey: .missedTensorCount)
        filledLayerCount = try container.decode(Int.self, forKey: .filledLayerCount)
        servedLayerCount = try container.decode(Int.self, forKey: .servedLayerCount)
        accountingOverflowed = try container.decodeIfPresent(
            Bool.self, forKey: .accountingOverflowed) ?? false
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(pinnedLayerCount, forKey: .pinnedLayerCount)
        try container.encode(pinnedLayers, forKey: .pinnedLayers)
        try container.encode(budgetBytes, forKey: .budgetBytes)
        try container.encode(residentBytes, forKey: .residentBytes)
        try container.encode(freedBytes, forKey: .freedBytes)
        try container.encode(bytesLoaded, forKey: .bytesLoaded)
        try container.encode(bytesServed, forKey: .bytesServed)
        try container.encode(bytesChargedAsRead, forKey: .bytesChargedAsRead)
        try container.encode(hitCount, forKey: .hitCount)
        try container.encode(missedTensorCount, forKey: .missedTensorCount)
        try container.encode(filledLayerCount, forKey: .filledLayerCount)
        try container.encode(servedLayerCount, forKey: .servedLayerCount)
        try container.encode(accountingOverflowed, forKey: .accountingOverflowed)
        // Derived, written so a reader of the JSON does not have to redo the
        // addition — and never read back, so the two cannot disagree.
        try container.encode(bytesWidened, forKey: .bytesWidened)
        try container.encode(isAccountingBalanced, forKey: .isAccountingBalanced)
    }

    /// One line for a console, in the shape the expert phase's own summary uses.
    public var summaryLine: String {
        guard pinnedLayerCount > 0 else { return "[pins] none" }
        return String(
            format: "[pins] %d layers, %llu B resident of %llu B budget, "
                + "%llu B loaded / %llu B served, %d hits over %d filled + %d served passes",
            pinnedLayerCount, residentBytes, budgetBytes, bytesLoaded, bytesServed, hitCount,
            filledLayerCount, servedLayerCount)
    }
}
