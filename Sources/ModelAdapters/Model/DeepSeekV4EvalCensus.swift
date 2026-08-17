import Foundation
import StorageCore

/// Where a DeepSeek V4 pass forces its graph, and which of those points make
/// the host wait for the GPU.
///
/// ## Why this exists
///
/// ``DeepSeekV4PhaseMetrics`` says where a pass's *seconds* went. It cannot say
/// how often the pass stopped. Those are different questions: a host sync that
/// costs 0.1 ms of its own bracket still drains the command queue, so the
/// arithmetic that was pending behind it lands in whatever bracket happens to
/// be open — usually the residual. A pass whose residual is "the model's own
/// arithmetic" and a pass whose residual is a thousand pipeline bubbles look
/// identical in the seconds table.
///
/// This census counts the stops. Each site below is one *named* place where the
/// decode thread forces a graph. The count is per site and cumulative over a
/// run, and a pass is the difference of two snapshots — the same rule
/// ``DeepSeekV4PhaseMetrics`` already uses, for the same reason.
///
/// ## Blocking and deferred are counted apart
///
/// A forcing point is one of two things, and they cost different amounts:
///
/// - **blocking** — `eval()`, `MLX.eval([...])`, `.item()`, or an
///   `asArray`/`asData` of an array that was not already evaluated. The decode
///   thread submits the graph *and waits for the GPU round trip*, which
///   `docs/experiments/2026-08-17-v4-cpu-profile.md` measured at ~1 ms of
///   completion-event latency each. While it waits the pipe drains.
/// - **deferred** — `MLX.asyncEval([...])`. MLX walks the graph, encodes it,
///   commits the command buffer and **returns**; the arrays are realised when
///   the GPU reaches them. The graph is bounded exactly as a blocking `eval`
///   bounds it — `eval_impl` detaches every taped array either way — so a site
///   that exists only to stop the lazy graph growing across a pass boundary
///   gets the whole of its guarantee without the wait.
///
/// The two are counted in separate tables. ``DeepSeekV4EvalCensus/counts`` is
/// the blocking one, so every count in a record written before deferral existed
/// keeps its meaning and stays comparable; ``DeepSeekV4EvalCensus/asyncCounts``
/// is the new one and is absent (zero) from those records, which is true.
///
/// ## What counts as one sync
///
/// One *forcing point*, not one MLX call. `x.eval()` immediately followed by
/// `x.asArray(Float.self)` is **one** sync: the array is already materialised
/// when the copy happens, so the second call waits for nothing. Recording two
/// would inflate the census exactly where the code is most careful. Where a
/// site pulls two independent arrays (`base` and `function` in the head
/// collapse) it is still recorded once, because the guard is one decision and
/// the second pull cannot drain a queue the first one already drained.
///
/// ## Two sites that are deliberately absent
///
/// Every case below is recorded somewhere, so a zero in the census means
/// "counted none" and never "was not instrumented". Two operators the phase
/// table names have no case here because they contain no host sync at all:
///
/// - `DeepSeekV4SparseAttention.expressing` builds ~64 micro-graphs per head
///   and forces none of them; its arithmetic lands on whichever later sync
///   evaluates it. ``DeepSeekV4PhaseMetrics/sparseAttentionSeconds`` already
///   says this.
/// - `DeepSeekV4FP8ActivationReference.exactPowerOfTwoScales` and its FP4 twin
///   choose the scale on the GPU since `c6b9877`; the `eval`/`asArray` pair
///   that measured 471 syncs at `b7c2092` is gone. What remains under
///   ``DeepSeekV4PhaseMetrics/activationScaleSyncSeconds`` is graph
///   construction. A regression there would reintroduce a sync, and the
///   reintroduced line would need a case here.
///
/// ## Cost
///
/// Off by default. ``DeepSeekV4EvalCensusCounter/isEnabled`` is read once from
/// `MINIRUN_V4_EVAL_CENSUS` at construction, and every recording site is a
/// `guard isEnabled else { return }` behind an already-optional accounting
/// reference. An instrument that changes the thing it measures is not one, and
/// a census of ~1,000 syncs at a locked increment each is ~50 µs on a pass that
/// takes seconds — measurable in principle, and stated here so a reader can
/// decide whether that matters rather than having to assume.
public enum DeepSeekV4EvalSite: String, CaseIterable, Codable, Sendable {

    // MARK: Structural — the execution sequence itself
    //
    // Every site in this group exists to *bound the lazy graph* and to realise
    // state that outlives the pass. None of them produces a host value, so
    // none of them needs the decode thread to wait: they are deferred, and are
    // counted in the census's async table.

    /// The residual between two layers in ``DeepSeekV4ModelExecution``'s
    /// sequence drivers, and the per-layer residual that ends one generation
    /// layer. One per layer per pass by construction. **Deferred.**
    case layerBoundary

    /// The submission inside a block's decode/forward that realises the
    /// attention output together with the cache tensors it must persist. One
    /// per attention block per pass. **Deferred.**
    case blockCache

    /// The compressed-state updates in ``DeepSeekV4CompressedAttention``'s
    /// helpers — `[compressed, values, scores]` and its siblings.
    /// **Deferred.**
    case compressedState

    /// `DeepSeekV4MoEFFN`'s closing residual. **Deferred.**
    case ffnResidual

    // MARK: Decisions — a host value the arithmetic then consumes

    /// `splitSinkhorn`'s three scale floats. Load-bearing: `scaleValues[0..2]`
    /// are Swift multipliers in the coefficients below them, so the pull
    /// happens whatever the diagnostics say. Three floats, and a full drain of
    /// everything expressed since the last sync.
    case hyperConnectionSplit

    /// `collapseHead`'s one scale float, same shape of argument, once per pass.
    case hyperConnectionHead

    /// `DeepSeekV4Router.select`: the score pull the host top-k reads.
    /// **Blocking, and irreducibly so** — the chosen expert ids decide which
    /// container tiles the pager must read, so they have to be concrete host
    /// values before the gather can be expressed at all.
    ///
    /// One per selection, not two. Learned routing needs both the sqrt-softplus
    /// scores (the weights) and the bias-ranked scores (the choice), and it
    /// used to pull them in that order, which cost two round trips because the
    /// second graph was built after the first had already been forced. Building
    /// the ranked score *before* the pull makes them one `MLX.eval([scores,
    /// ranked])` — the same two arrays, the same values, one wait.
    case routingSelect

    /// `DeepSeekV4LightningIndexer.select`: the score pull before the host
    /// top-k.
    case indexer

    /// Guard-only sweeps — `isFinite(...).all().item(Bool.self)` and the head
    /// table pulls that exist only to validate. Zero unless
    /// ``DeepSeekV4Diagnostics/validateFiniteness`` is on.
    case finitenessSweep

    // MARK: Weights and experts

    /// The forcing point that closes one dispatch chunk of the routed-expert
    /// gather loop, once per dispatch.
    ///
    /// **Blocking, and the reason is memory rather than correctness.** Under
    /// `BufferTransferMode.adopt` the operand *is* the pager's slot and the
    /// wait is what stops the GPU reading a lease that has been handed back.
    /// Under `.copy` — the product path — the lease is already back and the
    /// wait buys only the graph bound, which `asyncEval` would give for free;
    /// deferring it was implemented and **measured**, and it put
    /// `mlxActiveBytes` +53.5 MB above the first decode pass on an ordinary
    /// token and +157.6 MB on the worst of forty, against the 64 MB the
    /// product gate allows. One gather's transient is 53.5 MB, so the in-flight
    /// depth that fits the stated envelope is zero.
    /// `docs/experiments/2026-08-17-v4-async-forcing.md` §arm T1.
    case expertGather

    /// A load-time forcing point that materialises a tensor just read from
    /// storage — `widened`, `normalized`, `array` in the layer and global
    /// artifacts. On the unpinned layers this happens inside the pass; a pinned
    /// load returns before it and records nothing.
    ///
    /// **Blocking when the bytes are borrowed, deferred when they are not.** A
    /// tensor wrapped over a `malloc`'d read buffer with a deallocating
    /// finalizer must not outlive the wait, exactly as an adopted expert tile
    /// must not; a window sliced out of the *pinned* head table is MLX-owned
    /// and is only a graph bound.
    case weightMaterialisation

    /// One projection forced to completion: the `projected.eval()` that closes
    /// `projectBlockFP8Reference`, `projectGroupedBlockFP8Reference`, and the
    /// blocks' BF16 projection closures.
    ///
    /// This is the census's whole reason for existing. Every FP8 and BF16
    /// linear in the model ends by draining the queue, so the model's
    /// arithmetic is not one graph per layer but one graph per *matrix* — and
    /// the seconds those drains cost land in ``DeepSeekV4PhaseMetrics``'s
    /// residual, where they are indistinguishable from the arithmetic itself.
    /// The `eval` is not gratuitous — it is what keeps an adopted tile alive
    /// through GPU consumption and lets the slot be released afterwards — so
    /// this count is a statement about the streaming design, not a bug report.
    case projection

    // MARK: The head

    /// One `lm_head` row window: the projection `eval()` and its host copy.
    case outputHead

    /// An embedding row pulled to the host.
    ///
    /// There is deliberately no separate greedy-token site: the token is chosen
    /// from the `[Float]` the head windows already copied, so it forces
    /// nothing of its own.
    case embedding

    /// A top-level forcing point that closes a whole prefill or decode pass,
    /// or that realises the final hidden state before the head walk.
    /// **Deferred** — the head's own per-window pull is the wait.
    case passBoundary

    /// This site's slot in the counter's fixed array.
    ///
    /// Written out rather than derived from `allCases.firstIndex(of:)`: the
    /// derived form is a linear scan on every recorded sync, which is exactly
    /// the kind of cost an instrument must not add to the thing it measures.
    /// The switch is total, so adding a case without a slot will not compile.
    var slot: Int {
        switch self {
        case .layerBoundary: 0
        case .blockCache: 1
        case .compressedState: 2
        case .ffnResidual: 3
        case .hyperConnectionSplit: 4
        case .hyperConnectionHead: 5
        case .routingSelect: 6
        case .indexer: 7
        case .finitenessSweep: 8
        case .expertGather: 9
        case .weightMaterialisation: 10
        case .outputHead: 11
        case .embedding: 12
        case .passBoundary: 13
        case .projection: 14
        }
    }
}

/// The census as a value: one count per site, plus the total.
///
/// Codable so it rides the phase-metrics JSON the gate already writes. Absent
/// keys decode as zero for the same reason ``DeepSeekV4PhaseMetrics`` treats
/// them that way: a record written before a site existed measured zero of it,
/// not an unknown amount.
public struct DeepSeekV4EvalCensus: Codable, Sendable, Equatable {

    /// **Blocking** forcing points by ``DeepSeekV4EvalSite/rawValue`` — the
    /// places the decode thread waited for a GPU round trip. Sites that never
    /// fired are present with zero rather than omitted, so a reader can tell
    /// "counted none" from "was not instrumented".
    ///
    /// The name and the JSON key are unchanged from the census's first version,
    /// when every forcing point was blocking. A record from then reads the same
    /// way now.
    public var counts: [String: Int]

    /// **Deferred** forcing points — `MLX.asyncEval` submissions, which bound
    /// the graph without waiting. Absent from records written before deferral
    /// existed, and absent decodes as zero, which is what those runs made.
    public var asyncCounts: [String: Int]

    /// Sticky. A saturated count cannot prove a per-pass rate, exactly as
    /// ``DeepSeekV4PhaseMetrics/eventAccountingOverflowed`` cannot.
    public var overflowed: Bool

    public init(
        counts: [String: Int] = [:],
        asyncCounts: [String: Int] = [:],
        overflowed: Bool = false
    ) {
        self.counts = Self.completed(counts)
        self.asyncCounts = Self.completed(asyncCounts)
        self.overflowed = overflowed
    }

    private static func completed(_ counts: [String: Int]) -> [String: Int] {
        var complete = counts
        for site in DeepSeekV4EvalSite.allCases where complete[site.rawValue] == nil {
            complete[site.rawValue] = 0
        }
        return complete
    }

    /// Blocking count at `site`.
    public subscript(site: DeepSeekV4EvalSite) -> Int {
        counts[site.rawValue] ?? 0
    }

    /// Deferred count at `site`.
    public func async(_ site: DeepSeekV4EvalSite) -> Int {
        asyncCounts[site.rawValue] ?? 0
    }

    /// Every **blocking** sync the pass made, across all sites. Unchanged in
    /// meaning: this is the number the four 2026-08-17 records quote.
    public var total: Int { Self.sum(counts) }

    /// Every **deferred** submission the pass made, across all sites.
    public var asyncTotal: Int { Self.sum(asyncCounts) }

    /// Blocking plus deferred: every place the pass forced a graph at all.
    public var forcingPointTotal: Int {
        let next = total.addingReportingOverflow(asyncTotal)
        return next.overflow ? .max : next.partialValue
    }

    private static func sum(_ counts: [String: Int]) -> Int {
        counts.values.reduce(0) { sum, value in
            let next = sum.addingReportingOverflow(value)
            return next.overflow ? .max : next.partialValue
        }
    }

    /// The sites whose **blocking** count fired, largest first, then by name so
    /// the order is stable across runs.
    public var ranked: [(site: DeepSeekV4EvalSite, count: Int)] {
        Self.ranking { self[$0] }
    }

    /// The same, for the deferred table.
    public var rankedAsync: [(site: DeepSeekV4EvalSite, count: Int)] {
        Self.ranking { self.async($0) }
    }

    private static func ranking(
        _ count: (DeepSeekV4EvalSite) -> Int
    ) -> [(site: DeepSeekV4EvalSite, count: Int)] {
        DeepSeekV4EvalSite.allCases
            .map { (site: $0, count: count($0)) }
            .filter { $0.count > 0 }
            .sorted { left, right in
                left.count == right.count
                    ? left.site.rawValue < right.site.rawValue
                    : left.count > right.count
            }
    }

    /// The counts completed after an earlier cumulative snapshot.
    ///
    /// Fails closed on a count that moved backwards or on either side having
    /// saturated, for the reason ``DeepSeekV4PhaseMetrics/subtracting(_:)``
    /// does: a difference of two clamped numbers is not a count. Both tables
    /// have to difference cleanly — a pass that can account for its blocking
    /// syncs but not its deferred ones is still a pass that cannot be trusted.
    public func subtracting(_ earlier: DeepSeekV4EvalCensus) -> DeepSeekV4EvalCensus? {
        guard !overflowed, !earlier.overflowed else { return nil }
        var blocking = [String: Int]()
        var deferred = [String: Int]()
        for site in DeepSeekV4EvalSite.allCases {
            guard self[site] >= earlier[site],
                async(site) >= earlier.async(site)
            else { return nil }
            blocking[site.rawValue] = self[site] - earlier[site]
            deferred[site.rawValue] = async(site) - earlier.async(site)
        }
        return DeepSeekV4EvalCensus(counts: blocking, asyncCounts: deferred)
    }

    /// One line for a run's log: the blocking total, then the deferred total,
    /// then the sites that fired in each.
    public var summaryLine: String {
        let blocking = ranked
            .map { "\($0.site.rawValue) \($0.count)" }
            .joined(separator: ", ")
        let deferred = rankedAsync
            .map { "\($0.site.rawValue) \($0.count)" }
            .joined(separator: ", ")
        return "[v4 eval census] \(total) blocking syncs"
            + (blocking.isEmpty ? "" : "; " + blocking)
            + " | \(asyncTotal) async submissions"
            + (deferred.isEmpty ? "" : "; " + deferred)
            + (overflowed ? " (COUNT OVERFLOWED)" : "")
    }

    // MARK: Codable

    private enum CodingKeys: String, CodingKey {
        case counts, asyncCounts, overflowed
    }

    /// Written by hand for one reason: `asyncCounts` must decode as *absent*
    /// rather than as a failure in every record written before deferral
    /// existed. Those runs deferred nothing, so zero is the measurement and not
    /// a default that hides one.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            counts: try container.decodeIfPresent([String: Int].self, forKey: .counts) ?? [:],
            asyncCounts: try container.decodeIfPresent(
                [String: Int].self, forKey: .asyncCounts) ?? [:],
            overflowed: try container.decodeIfPresent(Bool.self, forKey: .overflowed) ?? false)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(counts, forKey: .counts)
        try container.encode(asyncCounts, forKey: .asyncCounts)
        try container.encode(overflowed, forKey: .overflowed)
    }
}

/// The counter behind the census.
///
/// Cumulative over a run and safe to call from any thread: the routed-expert
/// gather's `eval` runs on the decode thread, but a backend's reader threads
/// touch the same accounting object.
public final class DeepSeekV4EvalCensusCounter: @unchecked Sendable {

    /// `1`, `true`, or `yes` turns the census on. Anything else, including an
    /// unset variable, leaves every site a no-op.
    public static let environmentKey = "MINIRUN_V4_EVAL_CENSUS"

    public let isEnabled: Bool

    private let lock = NSLock()
    /// Indexed by ``DeepSeekV4EvalSite/allCases`` position. A fixed array
    /// rather than a dictionary so a recording site is an add, not a hash.
    private var counts: [Int]
    /// The deferred table, indexed the same way. Two arrays rather than one
    /// array of pairs: a site is almost always one kind or the other, and
    /// keeping them apart means neither total can be produced by accident.
    private var asyncCounts: [Int]
    private var overflowed = false

    public init(enabled: Bool) {
        isEnabled = enabled
        counts = [Int](repeating: 0, count: DeepSeekV4EvalSite.allCases.count)
        asyncCounts = [Int](repeating: 0, count: DeepSeekV4EvalSite.allCases.count)
    }

    public convenience init(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        let raw = environment[Self.environmentKey]?
            .trimmingCharacters(in: .whitespaces).lowercased()
        self.init(enabled: raw == "1" || raw == "true" || raw == "yes")
    }

    /// One **blocking** host sync at `site`. A no-op, and inlined to a single
    /// predictable branch, when the census is off.
    @inline(__always)
    public func record(_ site: DeepSeekV4EvalSite) {
        guard isEnabled else { return }
        increment(site, blocking: true)
    }

    /// One **deferred** submission at `site` — the graph was bounded, the
    /// decode thread did not wait.
    @inline(__always)
    public func recordAsync(_ site: DeepSeekV4EvalSite) {
        guard isEnabled else { return }
        increment(site, blocking: false)
    }

    private func increment(_ site: DeepSeekV4EvalSite, blocking: Bool) {
        let index = site.slot
        lock.lock()
        let next =
            blocking
            ? counts[index].addingReportingOverflow(1)
            : asyncCounts[index].addingReportingOverflow(1)
        let clamped = next.overflow ? Int.max : next.partialValue
        if blocking { counts[index] = clamped } else { asyncCounts[index] = clamped }
        overflowed = overflowed || next.overflow
        lock.unlock()
    }

    /// The census so far, or nil when the census is off — an absent field in
    /// the run JSON says "not measured", which is true, where a table of zeros
    /// would say "measured none", which is not.
    public var snapshot: DeepSeekV4EvalCensus? {
        guard isEnabled else { return nil }
        lock.lock()
        defer { lock.unlock() }
        var blocking = [String: Int]()
        var deferred = [String: Int]()
        for site in DeepSeekV4EvalSite.allCases {
            blocking[site.rawValue] = counts[site.slot]
            deferred[site.rawValue] = asyncCounts[site.slot]
        }
        return DeepSeekV4EvalCensus(
            counts: blocking, asyncCounts: deferred, overflowed: overflowed)
    }
}
