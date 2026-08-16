import Foundation
import MinirunKit
import MinirunRunners
import StorageCore

/// The per-model arithmetic the memory dial is drawn from.
///
/// Every field here is a byte count that came from somewhere nameable, and
/// `provenance` says where. That matters because the dial's refusal card quotes
/// these numbers back to the operator as the reason a configuration was
/// refused, and a refusal argued from a guess would be worse than no refusal.
///
/// ## The census is the profile
///
/// This type used to carry five loose scalars — a widest layer, a reserve, a
/// staged-pair peak, an expert size, a hot-set slot count — and the dial did its
/// own arithmetic over them. That arithmetic predated ``MemoryDialPlanner`` and
/// disagreed with it in the way that matters most: it had **no pinned tier at
/// all**, so every byte of budget above the staged pair went to a hot set and
/// then to free, and dragging the slider up appeared to buy nothing but empty
/// space. The planner's derived ladder says the opposite — every byte of surplus
/// below full deterministic residency buys pinned layers, at 1.000 bytes saved
/// per resident byte, and free only grows once all 93 layers are held.
///
/// So the profile now carries the planner's own input, an ``ArtifactCensus``,
/// and every scalar the screens used to store is derived from it. There is one
/// ladder, in one place, and the dial cannot disagree with the run.
struct MemoryProfile: Equatable, Sendable {

    enum Provenance: String, Sendable {
        /// Re-derived from the container headers on disk. Rendered mono.
        case measured
        /// Read out of the published `index.json`, unverified. Rendered italic.
        case declaredByIndex
        /// No revision-matched geometry or run record exists. Zero-valued
        /// fields in this profile are absence, not a measurement.
        case unknown

        var isVerified: Bool { self == .measured }
    }

    /// What the artifact is, in the form the planner reasons about: the stored
    /// and widened size of every deterministic blob, the globals bundle, and the
    /// routed-expert geometry.
    let census: ArtifactCensus
    /// The stated routed-expert pool. Part of the working floor, and not part of
    /// anything the dial can pin.
    let expertPoolBytes: UInt64
    /// KV, session, MLX's own cache and process overhead.
    let workingSetReserveBytes: UInt64
    /// A conservative product safety boundary that may be stricter than the
    /// arithmetic floor. Unlike the field below, this is not presented as a
    /// completed run record.
    let requiredMinimumBudgetBytes: UInt64?
    /// The smallest budget under which this model is ON RECORD as having run.
    /// Nil means no run exists; the dial then refuses at the arithmetic floor.
    let onRecordMinimumBudgetBytes: UInt64?
    let provenance: Provenance

    /// DeepSeek V4's own ladder input, when this profile describes V4.
    ///
    /// V4 cannot be described by ``ArtifactCensus``: that type derives resident
    /// and saved from one array, and a pinned V4 layer holds ~1.031× what it
    /// stops reading — see `docs/design/v4-memory-dial.md` §3.1. So V4 carries
    /// the kit's second census beside the first rather than being squeezed
    /// through it, and every V4 ladder question below is answered from this
    /// field. Nil for every other model, and nil for a V4 artifact whose
    /// manifests have not been read yet.
    let deepSeekV4: DeepSeekV4MemoryDialInputs.ArtifactProfile?

    init(
        census: ArtifactCensus,
        expertPoolBytes: UInt64,
        workingSetReserveBytes: UInt64,
        requiredMinimumBudgetBytes: UInt64?,
        onRecordMinimumBudgetBytes: UInt64?,
        provenance: Provenance,
        deepSeekV4: DeepSeekV4MemoryDialInputs.ArtifactProfile? = nil
    ) {
        self.census = census
        self.expertPoolBytes = expertPoolBytes
        self.workingSetReserveBytes = workingSetReserveBytes
        self.requiredMinimumBudgetBytes = requiredMinimumBudgetBytes
        self.onRecordMinimumBudgetBytes = onRecordMinimumBudgetBytes
        self.provenance = provenance
        self.deepSeekV4 = deepSeekV4
    }

    /// The planner's floor, built from this profile's own terms.
    var floor: WorkingSetFloor {
        WorkingSetFloor(
            widestResidentLayerBytes: census.layerResidentBytes.max() ?? 0,
            expertPoolBytes: expertPoolBytes,
            workingReserveBytes: workingSetReserveBytes)
    }

    /// The widest deterministic layer as it is STORED, in BF16.
    var widestDeterministicLayerStoredBytes: UInt64 { census.layerStoredBytes.max() ?? 0 }

    /// BF16 → float32. Widening is a correctness requirement, not a choice:
    /// it is what makes this device's arithmetic match the reference.
    var widenedResidentBytes: UInt64 { census.layerResidentBytes.max() ?? 0 }

    /// The arithmetic floor: the widest widened layer, the expert pool, and the
    /// working reserve. Below this the model cannot be resident at all.
    ///
    /// The expert pool is in here now and was not before. That was a real gap,
    /// not a rounding difference: a floor that omits a term the run allocates is
    /// a floor the run can walk through.
    var arithmeticFloorBytes: UInt64 { floor.totalBytes }

    /// What the dial actually refuses below.
    ///
    /// The arithmetic floor is the exact sum of known live terms. A product
    /// safety boundary may deliberately sit above it; the historical on-record
    /// minimum remains separate evidence rather than silently weakening either
    /// requirement.
    var refusalThresholdBytes: UInt64 {
        max(
            arithmeticFloorBytes,
            max(requiredMinimumBudgetBytes ?? 0, onRecordMinimumBudgetBytes ?? 0))
    }

    var layerCount: Int { deepSeekV4?.layerCount ?? census.layerStoredBytes.count }
    var routedLayerCount: Int { census.moeLayerCount }
    var denseLayerCount: Int { max(0, layerCount - routedLayerCount) }
    /// One expert as stored — the granularity a hot set would pin at.
    var expertStoredBytes: UInt64 { census.expertStrideBytes }
    /// Every deterministic blob, as written. What full residency costs.
    var deterministicStoredBytes: UInt64 { census.deterministicStoredBytes }

    /// The largest stored layer that can ever be *staged*.
    ///
    /// Layer 0 is excluded because nothing precedes it: it has no pair to hide
    /// behind and is read inline whenever it is not pinned. Every read-ahead
    /// record since the feature existed reports exactly that one structural
    /// miss.
    var stagedPairPeakBytes: UInt64 { census.layerStoredBytes.dropFirst().max() ?? 0 }

    /// The budgets at which the plan changes — the floor, then each successive
    /// prefix of the ranked ladder. The dial's presets are chosen out of these
    /// so a preset lands on a boundary that means something.
    ///
    /// Both branches ask a planner in the kit; neither computes a ladder here.
    /// That is the property the K3 relocation was for, and V4 keeps it: the
    /// screen cannot arrive at a boundary the run would not.
    var snapPoints: [UInt64] {
        if let v4 = deepSeekV4 {
            return DeepSeekV4MemoryDial.snapPoints(census: v4.census, floor: v4.floor)
        }
        return MemoryDialPlanner.snapPoints(census: census, floor: floor)
    }

    /// The smallest budget at which the ladder reaches an expert at all: full
    /// deterministic residency plus the globals bundle plus a hot set. Quoted so
    /// an empty hot-set tier can say *why* it is empty rather than just being
    /// empty.
    var firstExpertBudgetBytes: UInt64 {
        let points = snapPoints
        // The ladder is layers (93), then the globals bundle, then a hot set —
        // so the first budget that could pin an expert is one step past the
        // globals bundle, which is the last point `snapPoints` reports when no
        // measured hot set is offered.
        return points.last ?? arithmeticFloorBytes
    }
}

/// The read-ahead pair decision, and the storage layer's own sentence for a
/// refusal of it.
///
/// The struct that used to be here mirrored `DeterministicReadAhead.Decision`
/// field for field, by hand, with a comment saying so. It is now the kit's own
/// type — `MemoryDialPlanner`'s pair schedule — so "Show the arithmetic"
/// discloses the same values the planner used rather than a transcription of
/// them.
typealias ReadAheadDecision = ReadAheadPairSchedule.Decision

extension ReadAheadPairSchedule.Decision {
    /// The `StorageCoreError` a refusal produces downstream, spelled the way the
    /// storage layer spells it.
    var refusalError: StorageCoreError {
        .invalidArgument(
            "read-ahead budget \(MRFormat.bytesDecimal(budgetBytes)) is below the "
                + "\(MRFormat.bytesDecimal(requiredBytes)) required to hold layer "
                + "\(residentLayer) resident (\(MRFormat.bytesDecimal(residentBytes))) while "
                + "staging layer \(stagedLayer) (\(MRFormat.bytesDecimal(stagedBytes))) with a "
                + "\(MRFormat.bytesDecimal(reserveBytes)) reserve")
    }
}

/// Why a stated budget was refused. Each case carries the numbers its sentence
/// needs; none of them is ever resolved by clamping the budget.
enum BudgetRefusal: Equatable, Sendable {
    /// Below the arithmetic floor — the layer cannot be made resident.
    case belowArithmeticFloor(required: UInt64, stated: UInt64)
    /// Above the arithmetic floor, but below the smallest budget any run of
    /// this model has been made at.
    case belowOnRecordMinimum(minimum: UInt64, stated: UInt64)
    /// Above the calculated floor, but below a conservative product safety
    /// boundary established from observed process memory.
    case belowRequiredMinimum(minimum: UInt64, stated: UInt64)
    /// The device cannot offer the floor at all.
    case aboveDeviceCeiling(ceiling: UInt64, stated: UInt64)

    var deficitBytes: UInt64 {
        switch self {
        case .belowArithmeticFloor(let required, let stated):
            return required > stated ? required - stated : 0
        case .belowOnRecordMinimum(let minimum, let stated):
            return minimum > stated ? minimum - stated : 0
        case .belowRequiredMinimum(let minimum, let stated):
            return minimum > stated ? minimum - stated : 0
        case .aboveDeviceCeiling(let ceiling, let stated):
            return stated > ceiling ? stated - ceiling : 0
        }
    }

    /// What raising to would clear this refusal.
    var suggestedBudgetBytes: UInt64? {
        switch self {
        case .belowArithmeticFloor(let required, _): return required
        case .belowOnRecordMinimum(let minimum, _): return minimum
        case .belowRequiredMinimum(let minimum, _): return minimum
        case .aboveDeviceCeiling: return nil
        }
    }

    func runError(model: ModelID) -> RunError {
        switch self {
        case .belowArithmeticFloor(let required, let stated):
            return .budgetBelowMinimum(declared: stated, minimum: required, model: model)
        case .belowOnRecordMinimum(let minimum, let stated),
            .belowRequiredMinimum(let minimum, let stated):
            return .budgetBelowMinimum(declared: stated, minimum: minimum, model: model)
        case .aboveDeviceCeiling(let ceiling, let stated):
            return .pinPlanInvalid(
                "the stated budget \(stated) exceeds the device ceiling \(ceiling)")
        }
    }
}

/// A saved read-ahead value that this product cannot honour.
///
/// The wire type remains `Int?`, so an older conversation can contain a depth
/// that a previous UI offered. Preserving that value is important: silently
/// rewriting it would make the document claim a different run. The current
/// product instead refuses it before planning and offers an explicit recovery
/// choice in the dial.
struct ReadAheadRefusal: Equatable, Sendable {
    let requestedDepth: Int

    var message: String {
        "Saved read-ahead depth \(requestedDepth) is unavailable. Minirun currently supports "
            + "Auto (one layer) or Off. Choose Auto or Off before running."
    }

    var namedError: String {
        "deterministicReadAheadLayers=\(requestedDepth) is outside the supported range 0...1"
    }
}

/// What the read-ahead control states, and what that means in layers.
///
/// The app used to offer a stepper over `Int?` and label nil "unset — the
/// runner's own default". That was opaque in the way only a technically true
/// label can be — it named a default without saying what the default *was* —
/// and since spec v0.6.19 it was also stale: both prefetchers default **on**
/// wherever a budget is stated. So the control now states the two product
/// choices the runtime can honour: Auto (one layer) and Off (serial). The
/// explicit case remains only so an older persisted value can be preserved and
/// either honoured at depth one or refused visibly rather than clamped.
///
/// The wire form is unchanged and still `Int?`, because the knob the runner
/// takes is unchanged: nil is "the runner decides", and nil is a different
/// statement from 0.
enum ReadAheadSetting: Equatable, Sendable {
    /// The knob is not set. The runner prefetches, which is its default.
    case auto
    /// Explicitly zero: serial. The condition every pre-v0.6.19 measurement was
    /// taken under, which is the reason to keep it reachable at all.
    case off
    /// A depth stored by an older build or supplied through the request seam.
    case explicit(Int)

    /// What the runner's default does, in layers. One, per spec v0.6.19.
    static let automaticDepth = 1
    static let supportedDepths = 0...automaticDepth

    init(_ knob: Int?) {
        switch knob {
        case nil: self = .auto
        case 0: self = .off
        case let depth?: self = .explicit(depth)
        }
    }

    /// The value that travels in `RunKnobs.deterministicReadAheadLayers`.
    var knobValue: Int? {
        switch self {
        case .auto: return nil
        case .off: return 0
        case .explicit(let depth): return depth
        }
    }

    /// The depth the dial should draw, which is what the run will actually do.
    var effectiveDepth: Int {
        switch self {
        case .auto: return Self.automaticDepth
        case .off: return 0
        case .explicit(let depth): return depth
        }
    }

    /// Whether K3's product runtime and memory planner can honour this exact
    /// value. Nil/Auto resolves to depth one, so it is covered by the same gate.
    var isProductSupported: Bool {
        Self.supportedDepths.contains(effectiveDepth)
    }

    var refusal: ReadAheadRefusal? {
        isProductSupported ? nil : ReadAheadRefusal(requestedDepth: effectiveDepth)
    }

    var title: String {
        switch self {
        case .auto: return "Auto (on — recommended)"
        case .off: return "Off (serial)"
        case .explicit(1): return "On (one layer)"
        case .explicit(let depth): return "Unsupported depth \(depth)"
        }
    }

    var explanation: String {
        switch self {
        case .auto:
            return "Prepares the next layer while the current layer is processed. Recommended "
                + "for most storage."
        case .off:
            return "Reads one layer at a time. This can help troubleshoot storage, but is "
                + "usually slower."
        case .explicit(1):
            return "Prepares exactly one layer while the current layer is processed."
        case .explicit(let depth):
            return ReadAheadRefusal(requestedDepth: depth).message
        }
    }
}

/// The model-specific execution contract drawn by the memory dial.
///
/// K3 turns surplus into a concrete ``PinPlan``. DeepSeek V4 has a different
/// lifecycle: weights stream and the causal cache is replaced one layer at a
/// time beneath a hard process ceiling. Keeping those strategies explicit is
/// what stops a V4 chat from being described as a one-layer K3 artifact.
struct BudgetPlan: Equatable, Sendable {

    enum Strategy: Equatable, Sendable {
        case residency
        case boundedLayerStreaming
    }

    let model: ModelID
    let modelName: String
    let profile: MemoryProfile
    /// The stated budget. Never derived from anything; the operator says it.
    let budgetBytes: UInt64
    /// The response horizon whose first-fill and later-token reuse the plan
    /// describes. It must be the same value the eventual `RunRequest` carries.
    let maximumNewTokens: Int
    /// `os_proc_available_memory()`, or physical RAM on the Mac.
    let deviceCeilingBytes: UInt64
    let readAheadDepth: Int
    let strategy: Strategy
    /// Non-nil only for a K3-style residency plan whose saved depth cannot be
    /// represented by the runtime's measured 0/1 schedule.
    let readAheadRefusal: ReadAheadRefusal?

    /// The residency plan for this budget, from whichever kit planner owns this
    /// model's ladder.
    ///
    /// V4 used to have none, and that was true when it was written: a stated
    /// budget above the 2 GB product floor bought a 512 MiB MLX cache and
    /// nothing else. Since the V4 memory dial merged, a stated V4 budget pins
    /// deterministic layers out of its own headroom, so the plan exists for
    /// both strategies and the tiers below read it for both.
    let pinPlan: PinPlan?

    init(
        model: ModelID, modelName: String, profile: MemoryProfile, budgetBytes: UInt64,
        maximumNewTokens: Int, deviceCeilingBytes: UInt64, readAheadDepth: Int
    ) {
        self.model = model
        self.modelName = modelName
        self.profile = profile
        self.budgetBytes = budgetBytes
        self.maximumNewTokens = maximumNewTokens
        self.deviceCeilingBytes = deviceCeilingBytes
        self.readAheadDepth = readAheadDepth
        let strategy: Strategy = model == .deepseekV4Flash
            ? .boundedLayerStreaming : .residency
        self.strategy = strategy
        self.readAheadRefusal = strategy == .residency
            ? ReadAheadSetting.explicit(readAheadDepth).refusal
            : nil

        // The app's own refusal rules are stricter than the planner's: the
        // planner refuses below the arithmetic floor, and the dial additionally
        // refuses below a product safety boundary, below the smallest budget
        // on record, and above the device ceiling. So the refusal is decided
        // first, and the plan is only asked for after all gates pass.
        let refusal = Self.refusal(
            budgetBytes: budgetBytes, deviceCeilingBytes: deviceCeilingBytes, profile: profile)
        if refusal != nil || readAheadRefusal != nil {
            self.pinPlan = nil
        } else if let v4 = profile.deepSeekV4 {
            // V4's dial floor sits above the 2 GB product floor, and a budget
            // between the two is a real, runnable configuration that pins
            // nothing. The planner refuses it by name; `try?` is that refusal
            // becoming "no pinned tier", which is exactly what the run does.
            self.pinPlan = try? DeepSeekV4MemoryDial.plan(
                budgetBytes: budgetBytes,
                census: v4.census,
                floor: v4.floor,
                maximumNewTokens: maximumNewTokens)
        } else if strategy == .residency {
            self.pinPlan = try? MemoryDialPlanner.plan(
                budgetBytes: budgetBytes,
                census: profile.census,
                floor: profile.floor,
                readAhead: readAheadDepth >= 1 ? .depthOne : .off,
                maximumNewTokens: maximumNewTokens)
        } else {
            self.pinPlan = nil
        }
    }

    // MARK: Tiers, filled left to right in priority order

    /// Immovable: widest resident deterministic layer, widened, plus the expert
    /// pool and the working reserve.
    ///
    /// Zero when the budget is refused. A refused configuration allocates
    /// nothing, and a legend that reported a working floor beside a bar drawn
    /// as empty hatching would be two answers to the same question.
    /// The floor, less whatever the staged pair is currently occupying of it.
    var floorBytes: UInt64 {
        if strategy == .boundedLayerStreaming {
            guard isRunnable else { return 0 }
            // Above V4's dial floor the plan states the envelope exactly, in
            // the same terms the run reserves it. Below it there is no plan,
            // and the admitted product envelope is what the bar can honestly
            // draw.
            if let plan = pinPlan { return plan.workingFloorBytes }
            return min(profile.refusalThresholdBytes, budgetBytes)
        }
        let total = pinPlan?.workingFloorBytes ?? 0
        return total > stagedBytes ? total - stagedBytes : 0
    }

    /// The read-ahead pair's peak staged bytes — drawn as a slice **of the
    /// floor**, not of the slack above it.
    ///
    /// This is where the old bar was wrong in a second, quieter way. It charged
    /// the staged pair against the surplus, as though funding read-ahead
    /// competed with holding layers. It does not: the floor already reserves the
    /// *widest* widened layer, and while a narrower layer is the resident one
    /// that reservation has room to spare — which is exactly what the next
    /// layer's stored bytes are read into. The planner says the same thing by
    /// leaving read-ahead out of the floor and out of the residency identity,
    /// and by deciding each pair against the budget rather than against a tier.
    ///
    /// So the segment is drawn inside the floor: same total, and the operator
    /// can see what the reservation is being used for.
    var stagedBytes: UInt64 {
        guard strategy == .residency else { return 0 }
        guard let plan = pinPlan, readAheadDepth >= 1, plan.projectedStagedLayerCount > 0 else {
            return 0
        }
        return min(profile.stagedPairPeakBytes, plan.workingFloorBytes)
    }

    /// The pinned deterministic tier: whole layers held in RAM for the life of
    /// the run, so a later token reads none of their bytes.
    ///
    /// This is the tier the dial was missing. At 1.000 bytes saved per resident
    /// byte it outranks every other candidate, which is why raising the budget
    /// buys layers long before it buys anything else.
    var pinnedBytes: UInt64 { pinPlan?.pinnedBytes ?? 0 }

    var pinnedLayerCount: Int { pinPlan?.pinnedLayers.count ?? 0 }

    /// Whether the plan reached the ladder's globals rung: V4's output head,
    /// the `[vocabulary, hidden]` table a pass walks for every token.
    var pinsOutputHead: Bool { pinPlan?.pinnedGlobals?.unit == .outputHead }

    /// What is held, in units — the sentence the panel's metric shows.
    ///
    /// "43 of 43" and "43 of 43 + output head" are different residencies and
    /// different per-token reads, and the difference is 1.06 GB, so the head is
    /// named wherever the layer count is.
    var residentUnitsLabel: String {
        guard pinnedLayerCount > 0 || pinsOutputHead else { return "None" }
        let layers = "\(pinnedLayerCount) of \(profile.layerCount)"
        return pinsOutputHead ? "\(layers) + output head" : layers
    }

    /// Bytes read once to populate the deterministic resident tier. This is
    /// deliberately separate from the later-token read projection.
    var firstPassPinnedLayerFillBytes: UInt64 {
        pinPlan?.projectedFirstPassPinnedLayerFillBytes ?? 0
    }

    /// Maximum pinned-layer bytes served from memory after the first token if
    /// generation reaches this conversation's response limit.
    var pinnedLayerBytesServedAtTokenLimit: UInt64 {
        pinPlan?.projectedPinnedLayerBytesServedAtTokenLimit ?? 0
    }

    var pinReuseSentence: String? {
        guard pinnedLayerCount > 0 else { return nil }
        let fill = MRFormat.bytesDecimal(firstPassPinnedLayerFillBytes)
        guard maximumNewTokens > 1 else {
            return "The first token fills \(fill); a one-token response cannot reuse it."
        }
        return "The first token fills \(fill). Up to "
            + "\(MRFormat.bytesDecimal(pinnedLayerBytesServedAtTokenLimit)) can be reused "
            + "across tokens 2–\(maximumNewTokens)."
    }

    /// The layers held, as ranges — the shape the legend prints.
    var pinnedLayerRanges: [ClosedRange<Int>] { pinPlan?.pinnedLayerRanges ?? [] }

    /// The pinned expert hot set. Zero at every budget any device here can
    /// offer: the ladder reaches an expert only after all 93 deterministic
    /// layers *and* the globals bundle, because the best hot set ever measured
    /// returns 0.230 bytes per resident byte against a layer's 1.000.
    var hotSetBytes: UInt64 { pinPlan?.expertHotSetBytes ?? 0 }

    /// Headroom the run declared but does not allocate. Below full
    /// deterministic residency this is only ever the remainder under the next
    /// layer's size — the ladder spends surplus on layers, not on slack.
    var freeBytes: UInt64 {
        if strategy == .boundedLayerStreaming {
            guard isRunnable else { return 0 }
            if let plan = pinPlan { return plan.unusedBudgetBytes }
            return budgetBytes - floorBytes
        }
        return pinPlan?.unusedBudgetBytes ?? 0
    }

    /// The identity every tier drawing rests on: the tiers account for exactly
    /// what was allocated, which is the whole budget when it was accepted and
    /// nothing at all when it was refused. If this is false the dial is lying
    /// about where the bytes went.
    var tiersAccountForBudget: Bool {
        var total: UInt64 = 0
        for value in [floorBytes, stagedBytes, pinnedBytes, hotSetBytes, freeBytes] {
            let next = total.addingReportingOverflow(value)
            guard !next.overflow else { return false }
            total = next.partialValue
        }
        return total == (isRunnable ? budgetBytes : 0)
    }

    /// Every pair the budget was asked about was affordable.
    ///
    /// Not "the staged segment is as wide as the peak" — that was a byte
    /// comparison standing in for a decision the planner already makes, pair by
    /// pair, with the same function the run will call. A refused pair is the
    /// thing that costs seconds, so a refused pair is what "not funded" should
    /// mean. Layer 0 is not counted against it: nothing precedes layer 0, so it
    /// is read inline until it is *pinned*, which is a different lever.
    var stagedTierIsFunded: Bool {
        guard strategy == .residency else { return false }
        guard isRunnable, readAheadDepth >= 1, let plan = pinPlan else { return false }
        return plan.projectedRefusedPairCount == 0
    }

    /// The pairs the budget refused, and how far over each one was. A skip is
    /// normal operation, not an error — but it costs inline seconds, so it is
    /// reported rather than left implicit.
    var refusedPairCount: Int { pinPlan?.projectedRefusedPairCount ?? 0 }

    /// What the next GB of budget would actually buy, from the plan itself.
    var nextPinBudgetBytes: UInt64? { pinPlan?.nextPinBudgetBytes }

    /// The sentence the dial puts under the composition bar: which unit the next
    /// increase admits, and what it costs.
    var nextPinSentence: String? {
        guard let plan = pinPlan, let budget = plan.nextPinBudgetBytes, budget > budgetBytes,
            let unit = plan.nextPinUnit
        else { return nil }
        let name: String
        switch unit {
        case .deterministicLayer(let layer): name = "layer \(layer)"
        case .globalsBundle: name = "the globals bundle"
        case .outputHead: name = "the output head"
        case .expertHotSet(let experts): name = "an expert hot set of \(experts)/layer"
        }
        return "+\(MRFormat.bytesDecimal(budget - budgetBytes)) pins \(name)."
    }

    /// Per-token bytes this plan still reads, and what it saves. Both come from
    /// the plan, so the dial and a run's JSON cannot disagree.
    var bytesReadPerToken: UInt64 {
        if let plan = pinPlan { return plan.projectedBytesPerTokenRead }
        // A V4 budget between the product floor and the dial's floor is
        // runnable and has no plan, because it pins nothing. It still reads,
        // and the whole census is what it reads — reporting 0 there said the
        // opposite of the truth.
        if isRunnable, strategy == .boundedLayerStreaming, let v4 = profile.deepSeekV4 {
            return v4.census.bytesPerToken
        }
        return 0
    }
    var bytesSavedPerToken: UInt64 { pinPlan?.projectedBytesPerTokenSaved ?? 0 }

    // MARK: Decision and refusal

    /// The pair the budget is hardest on: layer 0 resident, layer 1 staged.
    /// The decision the storage layer would print, from the kit's own schedule.
    var decision: ReadAheadDecision {
        ReadAheadPairSchedule.decide(
            residentLayer: 0,
            resident: .init(
                storedBytes: profile.census.layerStoredBytes.first ?? 0,
                residentBytes: profile.census.layerResidentBytes.first ?? 0),
            stagedLayer: 1,
            next: .init(
                storedBytes: readAheadDepth >= 1
                    ? (profile.census.layerStoredBytes.dropFirst().first ?? 0) : 0,
                residentBytes: profile.census.layerResidentBytes.dropFirst().first ?? 0),
            budgetBytes: budgetBytes,
            reserveBytes: profile.workingSetReserveBytes)
    }

    private static func refusal(
        budgetBytes: UInt64, deviceCeilingBytes: UInt64, profile: MemoryProfile
    ) -> BudgetRefusal? {
        if budgetBytes > deviceCeilingBytes {
            return .aboveDeviceCeiling(ceiling: deviceCeilingBytes, stated: budgetBytes)
        }
        let threshold = profile.refusalThresholdBytes
        guard budgetBytes < threshold else { return nil }

        // Name the highest gate, not merely the first gate the stated budget
        // crossed. Otherwise an old 5.80 GB K3 chat is first offered the
        // 7.31 GB arithmetic floor and immediately refused again at the
        // 8.00 GB product boundary. A single recovery action must reach the
        // configuration that can actually run. Prompt-specific arithmetic can
        // still win when its exact MLA state grows past the product minimum.
        if profile.requiredMinimumBudgetBytes == threshold {
            return .belowRequiredMinimum(minimum: threshold, stated: budgetBytes)
        }
        if profile.onRecordMinimumBudgetBytes == threshold {
            return .belowOnRecordMinimum(minimum: threshold, stated: budgetBytes)
        }
        return .belowArithmeticFloor(required: threshold, stated: budgetBytes)
    }

    var refusal: BudgetRefusal? {
        Self.refusal(
            budgetBytes: budgetBytes, deviceCeilingBytes: deviceCeilingBytes, profile: profile)
    }

    var isRunnable: Bool { refusal == nil && readAheadRefusal == nil }

    /// True when the whole domain is refused — the device cannot host the model
    /// at any budget it can offer.
    var deviceCannotHostModel: Bool {
        deviceCeilingBytes < profile.refusalThresholdBytes
    }

    // MARK: Presets

    /// Three positions on the ladder, each landing one staged pair above a real
    /// pin boundary.
    ///
    /// The presets this replaces were "floor", "floor + staged pair" and "floor
    /// + staged pair + a full hot set + 10 %", and the last two of those bought
    /// nothing the ladder recognises: there is no reachable hot set, and 10 %
    /// over is not a boundary. A preset should land where the plan *changes*, so
    /// each one is now a snap point — a budget at which one more unit becomes
    /// affordable — plus the staged pair, so the tier the operator is being
    /// offered is actually funded rather than one byte short of it.
    enum Preset: String, CaseIterable, Identifiable, Sendable {
        case floor = "Floor"
        case balanced = "Balanced"
        case generous = "Generous"

        var id: String { rawValue }
    }

    func budget(for preset: Preset) -> UInt64 {
        if strategy == .boundedLayerStreaming {
            return v4Budget(for: preset)
        }
        let points = profile.snapPoints
        let pad = profile.stagedPairPeakBytes
        switch preset {
        case .floor:
            return profile.refusalThresholdBytes
        case .balanced:
            return boundary(in: points, atOrBelow: deviceCeilingBytes * 3 / 5, fallbackIndex: 1)
                &+ pad
        case .generous:
            return boundary(
                in: points, atOrBelow: deviceCeilingBytes > pad ? deviceCeilingBytes - pad : 0,
                fallbackIndex: 4) &+ pad
        }
    }

    /// V4's three positions, under K3's rules and on V4's own ladder.
    ///
    /// What this replaces was written before V4 had a memory dial: Balanced was
    /// a flat three-fifths of whatever the device offered, and its label called
    /// that "a moderate hard ceiling with additional headroom". Both were true
    /// then and neither is now. Since the dial merged, a stated budget pins
    /// deterministic layers out of its own headroom, so 20.6 GB on a 34.4 GB
    /// Mac is ~11 GB the run cannot spend: **every byte above the last rung
    /// buys nothing today**, because the ladder ends there — V4's rungs are the
    /// 43 deterministic layers and then the output head, and no expert hot set
    /// is reachable at any budget these devices offer.
    ///
    /// So Balanced is the point where all of them are held. On a device that
    /// cannot reach it, the rule is K3's unchanged: the largest snap point
    /// under three-fifths of the ceiling, and if not even the first pin fits,
    /// the first pin boundary anyway — over the ceiling, shown disabled with
    /// its deficit rather than quietly collapsing onto the floor.
    ///
    /// There is no staged pair to pad with. V4's deterministic read-ahead is
    /// named and not built (`docs/design/v4-memory-dial.md` §6), so a preset
    /// that reserved for one would be reserving for a tier the run has no code
    /// to fill.
    private func v4Budget(for preset: Preset) -> UInt64 {
        switch preset {
        case .floor:
            // The admitted product envelope, which is what a V4 chat that never
            // touched the dial carries. Nothing is pinned here.
            return profile.refusalThresholdBytes
        case .balanced:
            let points = profile.snapPoints
            guard points.count > 1 else { return profile.refusalThresholdBytes }
            let target = deviceCeilingBytes * 3 / 5
            // The last snap point is the whole ladder: every deterministic
            // layer and then the output head, which is the last rung there is.
            if let everyUnit = points.last, everyUnit <= target { return everyUnit }
            return boundary(in: points, atOrBelow: target, fallbackIndex: 1)
        case .generous:
            return max(profile.refusalThresholdBytes, deviceCeilingBytes)
        }
    }

    /// The largest snap point that fits under `target`.
    ///
    /// When even the first pin is out of reach the preset does **not** collapse
    /// onto the floor: it reports a fixed position on the ladder, which is over
    /// the ceiling and therefore shown disabled with its deficit. A preset that
    /// quietly became the floor on a small device would tell the operator that
    /// the device can do something it cannot.
    private func boundary(in points: [UInt64], atOrBelow target: UInt64, fallbackIndex: Int)
        -> UInt64
    {
        guard points.count > 1 else { return profile.refusalThresholdBytes }
        if let best = points.dropFirst().last(where: { $0 <= target }) { return best }
        return points[min(fallbackIndex, points.count - 1)]
    }

    /// What a preset buys, in the ladder's own terms. Computed by re-planning at
    /// the preset's budget, so the label cannot promise a tier the plan does not
    /// contain.
    func explanation(for preset: Preset) -> String {
        let target = budget(for: preset)
        if strategy == .boundedLayerStreaming {
            if case .floor = preset {
                return "The minimum admitted V4 envelope. Nothing is held resident: every layer "
                    + "streams, every token."
            }
            let candidate = with(budgetBytes: target)
            let layers = candidate.pinnedLayerCount
            let total = profile.layerCount
            guard layers > 0 else {
                return "The first pin boundary this ladder has. This device cannot reach it."
            }
            let saved = MRFormat.bytesDecimal(candidate.bytesSavedPerToken)
            // The head is the rung after the last layer, and it is worth 1.06 GB
            // a token, so a preset that reached it says so.
            let head = candidate.pinsOutputHead ? " + output head" : ""
            if layers >= total {
                return "All \(total) layers\(head) resident — about \(saved) fewer bytes "
                    + "per token."
            }
            return "\(layers) of \(total) layers\(head) resident — about \(saved) fewer bytes "
                + "per token."
        }
        switch preset {
        case .floor:
            return "The minimum that runs. Below this Minirun refuses. Nothing is pinned: "
                + "every layer streams, every token."
        case .balanced, .generous:
            let candidate = with(budgetBytes: target)
            let layers = candidate.pinnedLayerCount
            guard layers > 0 else {
                return "The first pin boundary this ladder has, plus the staged pair. This "
                    + "device cannot reach it."
            }
            let share = Double(candidate.pinnedBytes) / Double(max(1, profile.deterministicStoredBytes))
            return "Keeps \(layers) of \(profile.layerCount) model layers in memory "
                + "(\(MRFormat.bytesDecimal(candidate.pinnedBytes)), "
                + "\(Int((share * 100).rounded())) % of their stored size) and prepares the "
                + "next layer."
        }
    }

    /// A preset above the device ceiling is shown disabled with the deficit,
    /// never hidden.
    func presetDeficit(_ preset: Preset) -> UInt64? {
        let wanted = budget(for: preset)
        return wanted > deviceCeilingBytes ? wanted - deviceCeilingBytes : nil
    }

    func with(budgetBytes: UInt64) -> BudgetPlan {
        BudgetPlan(
            model: model, modelName: modelName, profile: profile, budgetBytes: budgetBytes,
            maximumNewTokens: maximumNewTokens,
            deviceCeilingBytes: deviceCeilingBytes, readAheadDepth: readAheadDepth)
    }

    func with(readAheadDepth: Int) -> BudgetPlan {
        BudgetPlan(
            model: model, modelName: modelName, profile: profile, budgetBytes: budgetBytes,
            maximumNewTokens: maximumNewTokens,
            deviceCeilingBytes: deviceCeilingBytes, readAheadDepth: readAheadDepth)
    }
}
