import Foundation

/// The memory dial for DeepSeek V4, in the shape ``MemoryDialPlanner`` already
/// established for K3 and with the same rules — but over V4's own census.
///
/// ## Why V4 needs its own census rather than ``ArtifactCensus``
///
/// The rules are shared and are **not** forked here: candidates are ranked by
/// bytes saved per resident byte, ties break by layer order, the fill *skips*
/// a unit it cannot afford instead of stopping, a budget below the floor is
/// refused by name, and nothing is ever clamped. What does not carry over is
/// K3's arithmetic identity `resident == saved`.
///
/// K3 pins a layer's **stored** BF16 bytes and widens per use, so a pinned
/// layer costs exactly the bytes it stops reading and every deterministic layer
/// sits at ratio 1.000 — the property `ArtifactCensus` bakes in by deriving
/// both columns from `layerStoredBytes`. V4's deterministic weights are
/// block-FP8 tiles, and what a pinned V4 layer holds is the **loaded**
/// ``BlockFP8Weights`` form: the packed matrix, adopted unchanged, plus the
/// *expanded* scale grid. Expansion goes from one scale per 128×128 block to
/// one per row per 32 columns, so the resident form is `packed + packed/32`
/// against a stored tile of `packed + packed/16384` — about **1.031× stored**.
///
/// That is a deliberate trade and it is why the ladder is no longer flat at
/// 1.000: V4 pins the loaded form because doing so removes the scale expansion
/// and the whole-matrix finite-weight scan from every later pass as well as the
/// read, and those are host work inside the pass's own unattributed residual.
/// Holding the stored tile instead would return ratio 1.000 and re-run both on
/// every use. The cost of the choice is ~3.1% more residency, stated here
/// rather than hidden, and the consequence for the ladder is that a V4
/// deterministic layer scores ~0.970 — still far above anything an expert hot
/// set has ever measured (0.230 at best, M8), so the tier order is unchanged.
///
/// Feeding these numbers through `ArtifactCensus` was the first thing tried and
/// it cannot be done honestly: that type derives resident and saved from one
/// array, so a V4 plan built through it either understates residency by 3.1%
/// (`layerStoredBytes` = stored) or overstates the saving by 3.1%
/// (`layerStoredBytes` = resident). A refusal boundary computed from either is
/// wrong in the direction that matters. Hence a second census with two
/// independent columns, and the same rules over it.
///
/// ## The globals rung
///
/// The ladder is not layers alone. After every deterministic layer comes the
/// **output head** — the whole `[vocabulary, hidden]` BF16 table, which a pass
/// walks in row windows for every token it produces. Pinning it holds exactly
/// the bytes it stops reading, so it scores 1.000 on the same column; the
/// **embedding table** is the same size and a decode token reads one row of it,
/// so it scores ~0.0000077 and is censused but never ranked. See ``Globals``
/// for the arithmetic and `rank(_:)` for why the head sits behind the layers
/// rather than ahead of them.
///
/// ## Purity
///
/// No clock, no filesystem, no MLX. `UInt64` arithmetic over a census the
/// caller transcribed from manifests, so every fixture in
/// `DeepSeekV4MemoryDialTests` runs with no drive and no checkpoint attached.
public enum DeepSeekV4MemoryDial {

    // MARK: - Census

    /// The two `global00` tables, each in the two forms the dial reasons about.
    ///
    /// K3's census carries one `globalsStoredBytes` for `embed_tokens +
    /// lm_head + final` together and says why: splitting the bundle needs a
    /// per-unit read census that did not exist. V4 has one — the two tables are
    /// separate files in `global00`'s manifest and the run reads them through
    /// two different doors — so the split is made here, and the arithmetic
    /// decides which half is a rung:
    ///
    /// ```
    /// unit         resident B      read B/token   saved per resident byte
    /// head        1,059,061,760   1,059,061,760   1.000   (whole table, every pass)
    /// embedding   1,059,061,760           8,192   0.0000077 (one row per token)
    /// ```
    ///
    /// So the head is a rung and **the embedding is not**: pinning 1.06 GB to
    /// stop reading 8 KB a token is 130,000× worse than a deterministic layer,
    /// and K3's design reached the same verdict from the same column
    /// (`embed_tokens` at ~0.000006 there). It is censused anyway, because the
    /// row it reads is a real per-token read and the plan's read total should
    /// contain it, and because a number stated is a number a later reader can
    /// re-check rather than re-derive.
    ///
    /// One row per token is the *decode* rate. Prefill reads one row per prompt
    /// position; the ladder is a per-decode-token statement, like every other
    /// column in this census.
    public struct Globals: Sendable, Equatable {

        /// The `[vocabulary, hidden]` BF16 head table as one resident array.
        /// Zero when the census cannot state it — an unstated resident form is
        /// not a pin candidate, not a free one.
        public let headResidentBytes: UInt64
        /// The whole head table, re-read in row windows on every pass.
        public let headReadBytesPerToken: UInt64
        /// The `[vocabulary, hidden]` BF16 embedding table. Stated, never
        /// ranked — see the type note.
        public let embeddingResidentBytes: UInt64
        /// One embedding row per decode token.
        public let embeddingReadBytesPerToken: UInt64

        /// A census that names no pinnable globals: everything the pass reads
        /// outside the layers, and nothing the dial may hold.
        public static func readOnly(bytesPerToken: UInt64) -> Globals {
            Globals(
                headResidentBytes: 0, headReadBytesPerToken: bytesPerToken,
                embeddingResidentBytes: 0, embeddingReadBytesPerToken: 0)
        }

        public init(
            headResidentBytes: UInt64,
            headReadBytesPerToken: UInt64,
            embeddingResidentBytes: UInt64,
            embeddingReadBytesPerToken: UInt64
        ) {
            self.headResidentBytes = headResidentBytes
            self.headReadBytesPerToken = headReadBytesPerToken
            self.embeddingResidentBytes = embeddingResidentBytes
            self.embeddingReadBytesPerToken = embeddingReadBytesPerToken
        }

        /// Everything a pass reads from `global00`.
        public var bytesReadPerToken: UInt64 {
            PlanningUInt64Arithmetic.add(
                headReadBytesPerToken, embeddingReadBytesPerToken).value
        }

        /// The head is a candidate only when the census states both columns.
        var isHeadPinnable: Bool { headResidentBytes > 0 && headReadBytesPerToken > 0 }

        /// 1.000 for the published shape: the pinned table is exactly the bytes
        /// it stops reading, because the widening stays per window.
        public var headSavedPerResidentByte: Double {
            guard headResidentBytes > 0 else { return 0 }
            return Double(headReadBytesPerToken) / Double(headResidentBytes)
        }

        /// ~0.0000077 for the published shape. Recorded so the decision not to
        /// rank it is a number rather than an opinion.
        public var embeddingSavedPerResidentByte: Double {
            guard embeddingResidentBytes > 0 else { return 0 }
            return Double(embeddingReadBytesPerToken) / Double(embeddingResidentBytes)
        }
    }

    /// What one V4 artifact costs, in the two forms the dial reasons about.
    ///
    /// Both columns come from the layer manifests — file lengths and container
    /// geometry — never from a measured run, so a census exists before the
    /// first payload byte is read.
    public struct Census: Sendable, Equatable {

        /// Per layer, the deterministic bytes **one pass reads**: every
        /// block-FP8 tile plus every BF16/F32 plain tensor the layer loads
        /// again on every token.
        ///
        /// Routed-expert tiles are deliberately absent — they are not
        /// deterministic, they are chosen by the router per token, and the
        /// expert pool is a separate budget term. So is anything the artifact
        /// already retains for the life of the run (the token→expert table, the
        /// output-head parameters): a unit that is read once cannot be saved
        /// twice, and pinning it would book a saving that already exists.
        public let layerReadBytes: [UInt64]

        /// Per layer, the bytes a **pinned** layer holds: the adopted packed
        /// matrix plus the expanded scale grid, plus the MLX copy of each plain
        /// tensor. Larger than ``layerReadBytes`` by the scale expansion — see
        /// the type's own note for why that is the chosen form.
        public let layerResidentBytes: [UInt64]

        /// Deterministic bytes a pass reads that belong to no layer: the output
        /// head's walk and the token's embedding row, each with the residency
        /// pinning it would cost. The head is a rung of the ladder; the
        /// embedding is not, and ``Globals`` states the arithmetic that decided
        /// which is which.
        public let globals: Globals

        /// Deterministic bytes a pass reads that belong to no layer.
        public var globalsBytesReadPerToken: UInt64 { globals.bytesReadPerToken }

        /// Routed-expert bytes one pass reads. Carried so the plan can state
        /// what fraction of the pass's reads the dial actually addresses, and
        /// never a pin candidate: no V4 budget in the product range comes near
        /// an expert hot set, and M8 refused LRU outright.
        public let expertBytesPerToken: UInt64

        /// A census whose globals are read and never held — the shape every
        /// caller wrote before the head became a rung, and still the honest one
        /// for a caller that cannot state what holding them would cost.
        public init(
            layerReadBytes: [UInt64],
            layerResidentBytes: [UInt64],
            globalsBytesReadPerToken: UInt64,
            expertBytesPerToken: UInt64
        ) throws {
            try self.init(
                layerReadBytes: layerReadBytes,
                layerResidentBytes: layerResidentBytes,
                globals: .readOnly(bytesPerToken: globalsBytesReadPerToken),
                expertBytesPerToken: expertBytesPerToken)
        }

        public init(
            layerReadBytes: [UInt64],
            layerResidentBytes: [UInt64],
            globals: Globals,
            expertBytesPerToken: UInt64
        ) throws {
            guard !layerReadBytes.isEmpty else { throw MemoryDialError.layerTableEmpty }
            guard layerReadBytes.count == layerResidentBytes.count else {
                throw MemoryDialError.layerTableRagged(
                    storedCount: layerReadBytes.count,
                    residentCount: layerResidentBytes.count)
            }
            // A layer that costs less resident than it reads would be a census
            // describing a compression the loader does not perform, and every
            // ratio above 1.0 it produced would outrank a real one.
            for (layer, read) in layerReadBytes.enumerated()
            where layerResidentBytes[layer] < read {
                throw MemoryDialError.residentSmallerThanStored(
                    layer: layer, stored: read, resident: layerResidentBytes[layer])
            }
            // The same rule as a layer's, for the same reason: a globals unit
            // that held less than it saves would rank above 1.0 on a ladder
            // where 1.0 means "every byte held stops a byte being read", and no
            // loader performs that compression. Only checked when the census
            // states a resident form; not stating one means "not a candidate",
            // which is a different claim from "free".
            if globals.headResidentBytes > 0,
                globals.headResidentBytes < globals.headReadBytesPerToken
            {
                throw MemoryDialError.globalUnitResidentSmallerThanRead(
                    unit: "output head",
                    read: globals.headReadBytesPerToken,
                    resident: globals.headResidentBytes)
            }
            let deterministic = PlanningUInt64Arithmetic.sum(layerReadBytes)
            let globalsRead = PlanningUInt64Arithmetic.add(
                globals.headReadBytesPerToken, globals.embeddingReadBytesPerToken)
            let withGlobals = PlanningUInt64Arithmetic.add(
                deterministic.value, globalsRead.value)
            let total = PlanningUInt64Arithmetic.add(withGlobals.value, expertBytesPerToken)
            guard !deterministic.overflow, !globalsRead.overflow, !withGlobals.overflow,
                !total.overflow
            else {
                throw MemoryDialError.byteCountUnrepresentable(
                    context: "V4 census bytes per token")
            }

            self.layerReadBytes = layerReadBytes
            self.layerResidentBytes = layerResidentBytes
            self.globals = globals
            self.expertBytesPerToken = expertBytesPerToken
        }

        /// Deterministic layer bytes one pass reads: 5,869,305,856 B of
        /// block-FP8 plus the layers' BF16 tensors, at published shape.
        public var deterministicLayerBytesPerToken: UInt64 {
            PlanningUInt64Arithmetic.sum(layerReadBytes).value
        }

        /// Everything one pass reads, from any bucket.
        public var bytesPerToken: UInt64 {
            PlanningUInt64Arithmetic.sum([
                deterministicLayerBytesPerToken, globalsBytesReadPerToken,
                expertBytesPerToken,
            ]).value
        }

        /// The identity a ``PinPlan`` built from this census carries, so a
        /// decoded plan can be rejected when the manifests it names have moved.
        /// Shared with K3's census on purpose: it is a hash of two byte tables
        /// and nothing about it is model-specific.
        public var layerTableSHA256: String {
            ArtifactCensus.layerTableSHA256(
                storedBytes: layerReadBytes, residentBytes: layerResidentBytes)
        }
    }

    // MARK: - The plan

    /// One candidate, the single column that ranks it, and the rung it sits on.
    private struct Candidate {
        /// The two rungs this ladder has. Every layer outranks the head, which
        /// outranks nothing — see ``DeepSeekV4MemoryDial/rank(_:)`` for why
        /// that is stated rather than left to the ratio.
        enum Rung: Int, Comparable {
            case deterministicLayer = 0
            case outputHead = 1

            static func < (lhs: Rung, rhs: Rung) -> Bool { lhs.rawValue < rhs.rawValue }
        }

        let unit: PinDecision.Unit
        let rung: Rung
        /// Tie-break inside a rung: layer order, so two runs of one budget
        /// produce the same plan.
        let orderKey: Int
        let residentBytes: UInt64
        let readBytesPerToken: UInt64

        var savedPerResidentByte: Double {
            guard residentBytes > 0 else { return 0 }
            return Double(readBytesPerToken) / Double(residentBytes)
        }
    }

    /// (budget, floor, census) → ``PinPlan``.
    ///
    /// `floor` is everything the run costs with nothing pinned — for V4 the
    /// product plan's own terms: the transient execution envelope, the retained
    /// generation state, the one-layer replacement, the routed-expert pool, the
    /// MLX cache, and the stated margin. `pinBudgetBytes` is what is left of the
    /// declared budget after all of it, computed by the caller from those same
    /// terms so the two cannot drift.
    ///
    /// - Throws: ``MemoryDialError/budgetBelowWorkingFloor(budgetBytes:workingFloorBytes:shortfallBytes:widestResidentLayerBytes:expertPoolBytes:workingReserveBytes:)``
    ///   when the declared budget cannot pay for the run before a single byte
    ///   is pinned. Refused by name; never clamped to fit.
    public static func plan(
        budgetBytes: UInt64,
        census: Census,
        floor: WorkingSetFloor,
        maximumNewTokens: Int
    ) throws -> PinPlan {
        guard maximumNewTokens >= 1 else {
            throw MemoryDialError.invalidMaximumNewTokens(maximumNewTokens)
        }
        guard floor.isRepresentable else {
            throw MemoryDialError.byteCountUnrepresentable(context: "V4 working-set floor")
        }
        let floorBytes = floor.totalBytes
        guard budgetBytes >= floorBytes else {
            throw MemoryDialError.budgetBelowWorkingFloor(
                budgetBytes: budgetBytes,
                workingFloorBytes: floorBytes,
                shortfallBytes: floorBytes - budgetBytes,
                widestResidentLayerBytes: floor.widestResidentLayerBytes,
                expertPoolBytes: floor.expertPoolBytes,
                workingReserveBytes: floor.workingReserveBytes)
        }

        let ranked = rank(census)
        let fill = self.fill(ranked: ranked, headroomBytes: budgetBytes - floorBytes)

        // The head is not a layer, and the shared ``PinPlan`` already has the
        // slot for it: K3 puts its globals bundle in `pinnedGlobals`, counts it
        // inside `pinnedBytes`, and keeps the layer-scoped fill and reuse
        // fields layer-scoped. V4's head follows that split exactly, which is
        // why the accounting identity needs **no new term**: the pinned bucket
        // is defined over every decision's saved bytes, and the head's whole
        // walk simply moves from inline to pinned.
        var layerDecisions: [PinDecision] = []
        var headDecision: PinDecision?
        for decision in fill.decisions {
            switch decision.unit {
            case .deterministicLayer: layerDecisions.append(decision)
            case .outputHead: headDecision = decision
            case .globalsBundle, .expertHotSet: continue
            }
        }

        let layerResidency = PlanningUInt64Arithmetic.sum(
            layerDecisions.lazy.map(\.residentBytes))
        let pinnedResidency = PlanningUInt64Arithmetic.add(
            layerResidency.value, headDecision?.residentBytes ?? 0)
        let saved = PlanningUInt64Arithmetic.sum(
            fill.decisions.lazy.map(\.bytesSavedPerToken))
        guard !layerResidency.overflow, !pinnedResidency.overflow, !saved.overflow else {
            throw MemoryDialError.byteCountUnrepresentable(context: "V4 pinned residency")
        }
        guard let laterPasses = UInt64(exactly: maximumNewTokens - 1) else {
            throw MemoryDialError.byteCountUnrepresentable(
                context: "V4 later response-token count")
        }
        // Layer-scoped, like the field's name and like K3's plan: the head's
        // fill is not in it, and the shared validator checks that agreement.
        let servedAtLimit = PlanningUInt64Arithmetic.multiply(
            layerResidency.value, laterPasses)
        guard !servedAtLimit.overflow else {
            throw MemoryDialError.byteCountUnrepresentable(
                context: "V4 pinned bytes served at the response limit")
        }
        let reserve = PlanningUInt64Arithmetic.sum([
            floor.expertPoolBytes, floor.workingReserveBytes, pinnedResidency.value,
        ])
        let peak = PlanningUInt64Arithmetic.add(floorBytes, pinnedResidency.value)
        guard !reserve.overflow, !peak.overflow else {
            throw MemoryDialError.byteCountUnrepresentable(context: "V4 projected peak")
        }

        // Every per-token byte lands in exactly one bucket. Commit 1 stages
        // nothing, so the buckets are pinned and inline; the read-ahead adds the
        // third and re-partitions the same total.
        let pinnedLayers = Set(
            layerDecisions.compactMap { decision -> Int? in
                guard case .deterministicLayer(let layer) = decision.unit else { return nil }
                return layer
            })
        let inlineLayerBytes = PlanningUInt64Arithmetic.sum(
            census.layerReadBytes.enumerated().lazy
                .filter { !pinnedLayers.contains($0.offset) }
                .map(\.element))
        // The embedding row is read on every pass whatever the budget: it is
        // never a candidate, so it never leaves this bucket.
        let inlineGlobals = headDecision == nil
            ? census.globalsBytesReadPerToken
            : census.globals.embeddingReadBytesPerToken
        let inlineBytes = PlanningUInt64Arithmetic.sum([
            inlineLayerBytes.value, inlineGlobals, census.expertBytesPerToken,
        ])
        guard !inlineLayerBytes.overflow, !inlineBytes.overflow else {
            throw MemoryDialError.byteCountUnrepresentable(
                context: "V4 inline bytes per token")
        }

        return PinPlan(
            budgetBytes: budgetBytes,
            floor: floor,
            layerTableSHA256: census.layerTableSHA256,
            layerCount: census.layerReadBytes.count,
            // Commit 1 is serial. The V4 deterministic read-ahead raises this
            // to 1 and moves layers out of the inline bucket into the staged
            // one; nothing else in this plan changes shape.
            readAheadDepth: 0,
            requestedMaximumNewTokens: maximumNewTokens,
            // Residency, not bytes read: V4's pinned form is larger than its
            // stored form (see the type note), and this field's shared
            // invariant is defined against the pinned *layer* decisions'
            // residency. The bytes the fill pass actually *reads* are
            // `projectedBytesPerTokenSaved`, which is the stored column.
            projectedFirstPassPinnedLayerFillBytes: layerResidency.value,
            projectedPinnedLayerBytesServedAtTokenLimit: servedAtLimit.value,
            pinnedLayers: layerDecisions,
            pinnedGlobals: headDecision,
            expertHotSet: nil,
            pinnedBytes: pinnedResidency.value,
            expertHotSetBytes: 0,
            unusedBudgetBytes: fill.remainingBytes,
            // The smallest budget that pins strictly *more* than this one, not
            // the size of the next unit: the fill has already spent everything
            // it could, so what a caller needs is this budget plus the
            // cheapest remaining shortfall.
            nextPinBudgetBytes: fill.nextPin.flatMap { next in
                let raised = budgetBytes.addingReportingOverflow(next.deficitBytes)
                return raised.overflow ? nil : raised.partialValue
            },
            nextPinUnit: fill.nextPin.flatMap { next in
                let raised = budgetBytes.addingReportingOverflow(next.deficitBytes)
                return raised.overflow ? nil : next.unit
            },
            readAheadReserveBytes: reserve.value,
            projectedStagedLayerCount: 0,
            projectedInlineLayerCount: census.layerReadBytes.count - layerDecisions.count,
            projectedRefusedPairCount: 0,
            refusedPairs: [],
            projectedBytesPerTokenSaved: saved.value,
            projectedBytesPerTokenRead: census.bytesPerToken
                - min(census.bytesPerToken, saved.value),
            pinnedBytesPerToken: saved.value,
            stagedBytesPerToken: 0,
            inlineBytesPerToken: inlineBytes.value,
            projectedPeakBytes: peak.value,
            // The V4 link has no calibration record, and a borrowed second
            // would be a claim no measurement supports.
            projectedTokenSeconds: nil,
            calibrationLabel: nil,
            baselineTokenSeconds: nil)
    }

    /// Budgets at which the plan changes. The floor first — below it the dial
    /// refuses — then each successive prefix of the ranked ladder.
    public static func snapPoints(census: Census, floor: WorkingSetFloor) -> [UInt64] {
        var points = [floor.totalBytes]
        var running = floor.totalBytes
        for candidate in rank(census) {
            let next = running.addingReportingOverflow(candidate.residentBytes)
            guard !next.overflow else { break }
            running = next.partialValue
            points.append(running)
        }
        return points
    }

    // MARK: Ranking and fill

    /// The candidate ladder, in priority order: **every deterministic layer
    /// first, in schedule order, then the output head**; inside each rung,
    /// bytes saved per resident byte descending with layer order as the
    /// tie-break, so two runs of one budget produce the same plan.
    ///
    /// ## Why the rung is stated and not left to the ratio
    ///
    /// On the byte column alone the head would go *first*: it reads its whole
    /// `[vocabulary, hidden]` table every pass and a pinned copy stops exactly
    /// those bytes, so it scores **1.000**, while a V4 layer scores ~0.970
    /// because its resident form carries the expanded scale grid (§3.1 of
    /// `docs/design/v4-memory-dial.md`).
    ///
    /// That 3% is precisely the part of a layer's value the byte column cannot
    /// see. The expansion is not overhead the pin fails to avoid — it is what
    /// the pin *buys*: holding the loaded form also removes the whole-matrix
    /// finite-weight scan and the scale rebuild from every later pass, host
    /// work inside the residual the 2026-08-15 gate could not attribute. A
    /// layer's 0.970 is a lower bound on what pinning it removes; the head's
    /// 1.000 is the whole of it, because the head's walk is a `pread` and a
    /// matmul and the per-window widening stays where it is. Ranking a
    /// complete 1.000 above an understated 0.970 would reverse the order on the
    /// strength of the understatement.
    ///
    /// It is also the order that keeps the ladder monotone in the thing the
    /// dial is for: at every budget the phone can reach, a layer displaced by
    /// the head would be a layer still read serially — 43 small units the
    /// budget can spend exactly, against one 1.06 GB unit it usually cannot.
    ///
    /// So the rule is one sentence: **layers, in schedule order, then the
    /// head** — and the ratio each decision carries records the honest number
    /// either way.
    private static func rank(_ census: Census) -> [Candidate] {
        var candidates = census.layerReadBytes.enumerated().compactMap {
            layer, read -> Candidate? in
            guard read > 0, census.layerResidentBytes[layer] > 0 else { return nil }
            return Candidate(
                unit: .deterministicLayer(layer),
                rung: .deterministicLayer,
                orderKey: layer,
                residentBytes: census.layerResidentBytes[layer],
                readBytesPerToken: read)
        }
        candidates.sort {
            $0.savedPerResidentByte == $1.savedPerResidentByte
                ? $0.orderKey < $1.orderKey
                : $0.savedPerResidentByte > $1.savedPerResidentByte
        }
        if census.globals.isHeadPinnable {
            candidates.append(
                Candidate(
                    unit: .outputHead,
                    rung: .outputHead,
                    orderKey: 0,
                    residentBytes: census.globals.headResidentBytes,
                    readBytesPerToken: census.globals.headReadBytesPerToken))
        }
        // The embedding table is deliberately absent: 1.06 GB resident against
        // one 8 KB row a token is ~0.0000077 bytes saved per resident byte,
        // five orders of magnitude below the worst expert hot set ever
        // measured. `Globals` states the arithmetic.
        return candidates
    }

    private struct Fill {
        let decisions: [PinDecision]
        let remainingBytes: UInt64
        /// How much *more* budget the cheapest skipped candidate needs, and
        /// which one it is. Always at least 1 — a candidate is only skipped
        /// when it does not fit — so the budget derived from it is strictly
        /// above the one that produced this fill.
        let nextPin: (deficitBytes: UInt64, unit: PinDecision.Unit)?
    }

    /// Walk the ranked ladder and pin every unit that still fits, **skipping**
    /// the ones that do not rather than stopping at the first miss.
    ///
    /// Skipping is the same rule K3's fill uses and it changes the answer: a
    /// budget that cannot take the next candidate can still take a smaller one
    /// further down, and the dial maximises bytes saved rather than units held.
    private static func fill(ranked: [Candidate], headroomBytes: UInt64) -> Fill {
        var remaining = headroomBytes
        var pinned: [(candidate: Candidate, decision: PinDecision)] = []
        var skipped: [Candidate] = []

        for (rank, candidate) in ranked.enumerated() {
            guard candidate.residentBytes <= remaining else {
                skipped.append(candidate)
                continue
            }
            remaining -= candidate.residentBytes
            pinned.append((
                candidate,
                PinDecision(
                    unit: candidate.unit,
                    rank: rank,
                    residentBytes: candidate.residentBytes,
                    bytesSavedPerToken: candidate.readBytesPerToken,
                    savedPerResidentByte: candidate.savedPerResidentByte,
                    budgetRemainingAfterBytes: remaining)))
        }
        // Report pins in rung order and then in layer order — the head last,
        // after every layer it was ranked behind. The rank each decision
        // carries is the ladder position, so this ordering loses nothing and
        // makes `pinnedLayerRanges` and the run's JSON readable.
        pinned.sort {
            $0.candidate.rung == $1.candidate.rung
                ? $0.candidate.orderKey < $1.candidate.orderKey
                : $0.candidate.rung < $1.candidate.rung
        }

        // The smallest budget that pins strictly more than this one. Each
        // skipped candidate's deficit is a lower bound; the smallest of them is
        // the answer, because admitting it cannot displace an already-pinned
        // unit — the fill is a single pass over a fixed ladder.
        let nextPin = skipped.map { candidate -> (UInt64, PinDecision.Unit) in
            (candidate.residentBytes - remaining, candidate.unit)
        }.min { $0.0 < $1.0 }

        return Fill(
            decisions: pinned.map(\.decision),
            remainingBytes: remaining,
            nextPin: nextPin.map { (deficit, unit) in
                (deficitBytes: deficit, unit: unit)
            })
    }
}
