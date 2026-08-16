import CryptoKit
import Foundation

/// The planner accepts byte counts at a public boundary, including from JSON.
/// Keep overflow handling in one place so an aggregate either remains exact or
/// saturates to the only representable conservative value. Callers that need
/// an exact plan inspect `overflow` and refuse it by name.
enum PlanningUInt64Arithmetic {
    struct Result {
        let value: UInt64
        let overflow: Bool
    }

    static func add(_ lhs: UInt64, _ rhs: UInt64) -> Result {
        let sum = lhs.addingReportingOverflow(rhs)
        return Result(value: sum.overflow ? .max : sum.partialValue, overflow: sum.overflow)
    }

    static func sum<S: Sequence>(_ values: S) -> Result where S.Element == UInt64 {
        var value: UInt64 = 0
        var overflow = false
        for next in values {
            let addition = value.addingReportingOverflow(next)
            overflow = overflow || addition.overflow
            value = addition.overflow ? .max : addition.partialValue
        }
        return Result(value: value, overflow: overflow)
    }

    static func multiply(_ lhs: UInt64, _ rhs: UInt64) -> Result {
        let product = lhs.multipliedReportingOverflow(by: rhs)
        return Result(value: product.overflow ? .max : product.partialValue, overflow: product.overflow)
    }
}

/// One number from the user, turned into a *stated* residency plan before the
/// run starts: what is pinned, what streams, and what the token should cost.
///
/// ## Why a planner exists at all
///
/// Spec §5.3 asks for the working set to be a declared budget rather than an
/// emergent footprint, and §5.8 asks a run to refuse what it cannot honour
/// instead of discovering it late. A dial that hands the decode loop a byte
/// count satisfies neither on its own: the loop would still find out, layer by
/// layer, what fits. So the arithmetic happens once, up front, and produces a
/// ``PinPlan`` that says which units are resident, which stream, how many bytes
/// per token that leaves, and — only where a link has been measured — how many
/// seconds. The plan is the thing a run records, so the screen and the run's
/// JSON cannot disagree.
///
/// ## The one rule
///
/// Priority is **bytes saved per resident byte**, ties broken by layer order.
/// Because a pinned deterministic layer holds its **stored** BF16 bytes and the
/// consume path still widens per use, pinning a layer saves exactly the bytes it
/// stops reading — a ratio of 1.000. Everything else measured so far scores
/// worse: the globals bundle 0.499 (4.70 GB resident buys back the 2.34 GB
/// `lm_head` walk), and the best expert hot set ever measured 0.230. The tier
/// ladder is therefore *derived* from one column rather than assumed, which is
/// what lets it disagree with spec §11.5's table where the arithmetic says so
/// (see `docs/design/memory-dial.md` §9).
///
/// ## Purity
///
/// No clock, no filesystem, no MLX, no Metal: arithmetic over `UInt64`. The same
/// inputs produce the same plan on a phone, on the Mac, and in a unit test with
/// neither drive nor checkpoint attached — which is why every row of the design
/// document's table is a fixture in `MemoryDialPlannerTests` rather than a run.

// MARK: - Census

/// What the artifact is, in the two forms the dial has to reason about.
///
/// Derived from the files (spec §5.10 — where files and metadata disagree, the
/// files win), never from a manifest field. Every number the flagship census
/// carries is transcribed from a record: the layer census reproduces the
/// published 108,817,924,096 B total exactly, which is the check that makes it
/// a measurement rather than a model.
public struct ArtifactCensus: Sendable, Equatable, Codable {

    private enum CodingKeys: String, CodingKey {
        case layerStoredBytes
        case layerResidentBytes
        case globalsStoredBytes
        case globalsBytesReadPerToken
        case expertStrideBytes
        case expertsPerLayer
        case routedExpertsPerToken
        case moeLayerCount
        case measuredDeterministicBytesPerToken
    }

    /// Per deterministic blob, as written: BF16 for nearly everything. 93
    /// entries at flagship shape. This is what a pinned layer costs *and* what
    /// it stops reading, which is why its ratio is exactly 1.
    public let layerStoredBytes: [UInt64]
    /// Per deterministic blob after the consume path widens it to float32. Only
    /// the largest matters to the floor, but the whole table is carried because
    /// the read-ahead pair decision needs the widened size of layer *i*.
    public let layerResidentBytes: [UInt64]
    /// `embed_tokens` + `lm_head` + `final`, as written: 4,697,669,632 B.
    public let globalsStoredBytes: UInt64
    /// What the globals actually cost per token: 2,342,806,528 B, measured, and
    /// **not** the same as ``globalsStoredBytes`` — `embed_tokens` is 2.35 GB of
    /// file serving a ~14 KB row gather. Pinning the bundle buys back this
    /// number, not its size.
    public let globalsBytesReadPerToken: UInt64
    /// One routed expert, in bytes: 17,547,264 (M8's measured stride).
    public let expertStrideBytes: UInt64
    /// Experts held per MoE layer: 896. The ceiling on a hot set.
    public let expertsPerLayer: Int
    /// Experts read per layer per token: 16.
    public let routedExpertsPerToken: Int
    /// Layers with a routed expert stream: 92.
    public let moeLayerCount: Int
    /// The run's own per-token deterministic counter, stated so the census can
    /// be checked against it rather than trusted: 111,160,730,624 B.
    public let measuredDeterministicBytesPerToken: UInt64

    /// Stable identity of the complete deterministic layer table.
    ///
    /// A `PinPlan` is only meaningful for the artifact census that produced
    /// it. The runtime can derive this value from the manifests it actually
    /// opened, which catches drift in unpinned layers as well as in the pinned
    /// subset. Counts are included separately so a ragged table cannot collide
    /// with a differently partitioned one.
    public var layerTableSHA256: String {
        Self.layerTableSHA256(
            storedBytes: layerStoredBytes, residentBytes: layerResidentBytes)
    }

    public static func layerTableSHA256(
        storedBytes: [UInt64], residentBytes: [UInt64]
    ) -> String {
        var bytes = Data("minirun.layer-census.v1\0".utf8)

        func append(_ value: UInt64) {
            var littleEndian = value.littleEndian
            withUnsafeBytes(of: &littleEndian) { bytes.append(contentsOf: $0) }
        }

        append(UInt64(storedBytes.count))
        for value in storedBytes { append(value) }
        append(UInt64(residentBytes.count))
        for value in residentBytes { append(value) }
        return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }

    public init(
        layerStoredBytes: [UInt64],
        layerResidentBytes: [UInt64],
        globalsStoredBytes: UInt64,
        globalsBytesReadPerToken: UInt64,
        expertStrideBytes: UInt64,
        expertsPerLayer: Int,
        routedExpertsPerToken: Int,
        moeLayerCount: Int,
        measuredDeterministicBytesPerToken: UInt64
    ) throws {
        guard !layerStoredBytes.isEmpty else { throw MemoryDialError.layerTableEmpty }
        guard layerStoredBytes.count == layerResidentBytes.count else {
            throw MemoryDialError.layerTableRagged(
                storedCount: layerStoredBytes.count, residentCount: layerResidentBytes.count)
        }
        for (index, stored) in layerStoredBytes.enumerated() where
            layerResidentBytes[index] < stored
        {
            throw MemoryDialError.residentSmallerThanStored(
                layer: index, stored: stored, resident: layerResidentBytes[index])
        }
        guard expertsPerLayer >= 0, routedExpertsPerToken >= 0, moeLayerCount >= 0,
            routedExpertsPerToken <= expertsPerLayer
        else {
            throw MemoryDialError.invalidExpertGeometry(
                expertsPerLayer: expertsPerLayer,
                routedExpertsPerToken: routedExpertsPerToken,
                moeLayerCount: moeLayerCount)
        }

        let deterministic = PlanningUInt64Arithmetic.sum(layerStoredBytes)
        let accounted = PlanningUInt64Arithmetic.add(
            deterministic.value, globalsBytesReadPerToken)
        guard !deterministic.overflow, !accounted.overflow else {
            throw MemoryDialError.byteCountUnrepresentable(
                context: "deterministic census bytes per token")
        }
        if accounted.value > measuredDeterministicBytesPerToken {
            throw MemoryDialError.censusDisagreesWithMeasuredTotal(
                censusBytes: accounted.value,
                measuredBytes: measuredDeterministicBytesPerToken,
                differenceBytes: Self.signedDifference(
                    measuredDeterministicBytesPerToken, accounted.value))
        }

        let perLayerExperts = PlanningUInt64Arithmetic.multiply(
            UInt64(routedExpertsPerToken), expertStrideBytes)
        let expertTotal = PlanningUInt64Arithmetic.multiply(
            perLayerExperts.value, UInt64(moeLayerCount))
        guard !perLayerExperts.overflow, !expertTotal.overflow else {
            throw MemoryDialError.byteCountUnrepresentable(
                context: "routed expert bytes per token")
        }
        let measuredTotal = PlanningUInt64Arithmetic.add(
            measuredDeterministicBytesPerToken, expertTotal.value)
        guard !measuredTotal.overflow else {
            throw MemoryDialError.byteCountUnrepresentable(
                context: "total measured bytes per token")
        }

        self.layerStoredBytes = layerStoredBytes
        self.layerResidentBytes = layerResidentBytes
        self.globalsStoredBytes = globalsStoredBytes
        self.globalsBytesReadPerToken = globalsBytesReadPerToken
        self.expertStrideBytes = expertStrideBytes
        self.expertsPerLayer = expertsPerLayer
        self.routedExpertsPerToken = routedExpertsPerToken
        self.moeLayerCount = moeLayerCount
        self.measuredDeterministicBytesPerToken = measuredDeterministicBytesPerToken
    }

    /// Synthesised `Decodable` initialisation writes stored properties directly
    /// and would bypass every census invariant above. Route archived or remote
    /// input through the same validating boundary as in-process callers.
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            layerStoredBytes: values.decode([UInt64].self, forKey: .layerStoredBytes),
            layerResidentBytes: values.decode([UInt64].self, forKey: .layerResidentBytes),
            globalsStoredBytes: values.decode(UInt64.self, forKey: .globalsStoredBytes),
            globalsBytesReadPerToken: values.decode(
                UInt64.self, forKey: .globalsBytesReadPerToken),
            expertStrideBytes: values.decode(UInt64.self, forKey: .expertStrideBytes),
            expertsPerLayer: values.decode(Int.self, forKey: .expertsPerLayer),
            routedExpertsPerToken: values.decode(Int.self, forKey: .routedExpertsPerToken),
            moeLayerCount: values.decode(Int.self, forKey: .moeLayerCount),
            measuredDeterministicBytesPerToken: values.decode(
                UInt64.self, forKey: .measuredDeterministicBytesPerToken))
    }

    private static func signedDifference(_ lhs: UInt64, _ rhs: UInt64) -> Int64 {
        if lhs >= rhs { return Int64(clamping: lhs - rhs) }
        let magnitude = rhs - lhs
        guard magnitude <= UInt64(Int64.max) else { return .min }
        return -Int64(magnitude)
    }

    /// All deterministic blobs, as written: 108,817,924,096 B at flagship shape.
    public var deterministicStoredBytes: UInt64 {
        PlanningUInt64Arithmetic.sum(layerStoredBytes).value
    }

    /// Routed-expert bytes one token reads: 92 × 16 × 17,547,264 =
    /// 25,829,572,608. Three positions of that is 77,488,717,824 B — exactly the
    /// prefill total the flagship run recorded, which is the check on this line.
    public var expertBytesPerToken: UInt64 {
        let perLayer = PlanningUInt64Arithmetic.multiply(
            UInt64(max(0, routedExpertsPerToken)), expertStrideBytes)
        let total = PlanningUInt64Arithmetic.multiply(
            perLayer.value, UInt64(max(0, moeLayerCount)))
        return perLayer.overflow || total.overflow ? .max : total.value
    }

    /// Per-token bytes the run measured that no unit in this census explains:
    /// `measured − (blobs + globals read)`.
    ///
    /// Zero at flagship shape, and the initialiser refuses a negative residual,
    /// so this is only ever a positive gap. A positive gap is carried into the
    /// **inline** bucket rather than absorbed into a unit — unattributed bytes
    /// cannot be pinned, because nothing knows what they are — which keeps the
    /// bucket identity honest instead of tidy.
    public var unattributedBytesPerToken: Int64 {
        Int64(clamping: unattributedBytesPerTokenExact)
    }

    fileprivate var unattributedBytesPerTokenExact: UInt64 {
        let accounted = PlanningUInt64Arithmetic.add(
            deterministicStoredBytes, globalsBytesReadPerToken)
        guard !accounted.overflow, measuredDeterministicBytesPerToken >= accounted.value else {
            return 0
        }
        return measuredDeterministicBytesPerToken - accounted.value
    }

    /// Everything one token reads: deterministic plus routed experts.
    /// 136,990,303,232 B at flagship shape.
    public var measuredBytesPerToken: UInt64 {
        PlanningUInt64Arithmetic.add(
            measuredDeterministicBytesPerToken, expertBytesPerToken).value
    }

    /// The flagship artifact, from the records — layer 0 dense, 68 KDA+MoE
    /// layers, 24 MLA+MoE layers at 0-based `{3, 7, …, 91} ∪ {92}` (M9C §239).
    ///
    /// The alternation has a period of four, which is why a pin set is not
    /// always a clean prefix: at a budget that cannot take the next KDA layer an
    /// MLA layer three positions later still fits.
    public static func measuredFlagship() throws -> ArtifactCensus {
        let denseLayerZero: UInt64 = 2_341_257_216
        let kdaLayer: UInt64 = 1_267_810_304
        let mlaLayer: UInt64 = 844_398_592
        let mlaLayers = Set(stride(from: 3, through: 91, by: 4)).union([92])
        var stored: [UInt64] = [denseLayerZero]
        for layer in 1...92 { stored.append(mlaLayers.contains(layer) ? mlaLayer : kdaLayer) }
        return try ArtifactCensus(
            layerStoredBytes: stored,
            // Every deterministic tensor is BF16 and widens to float32, so the
            // resident form is exactly twice the stored form — the same
            // doubling `DeterministicReadAheadTests` states for layer 0.
            layerResidentBytes: stored.map { PlanningUInt64Arithmetic.multiply($0, 2).value },
            globalsStoredBytes: 4_697_669_632,
            globalsBytesReadPerToken: 2_342_806_528,
            expertStrideBytes: 17_547_264,
            expertsPerLayer: 896,
            routedExpertsPerToken: 16,
            moeLayerCount: 92,
            measuredDeterministicBytesPerToken: 111_160_730_624)
    }
}

// MARK: - Floor

/// What the budget must hold before a single byte is pinned (spec §5.3).
///
/// The floor is what the run costs with **nothing** pinned. It is stated rather
/// than emergent, and it does not shrink when the dial pins things — pinning
/// adds residency, it never removes the widened layer the loop is computing on.
///
/// Read-ahead is deliberately *not* in the floor. Staging is opportunistic and
/// already decided per pair by ``ReadAheadPairSchedule``; folding its
/// 1,267,810,304 B term in here would refuse the phone's own measured, working
/// configuration.
public struct WorkingSetFloor: Sendable, Equatable, Codable {

    /// The largest layer *after* widening: 4,682,514,432 B, layer 0. One layer
    /// is always resident in this form, whatever the dial says.
    public let widestResidentLayerBytes: UInt64
    /// The stated routed-expert pool: 46,792,704 B (44.6 MB, M9C).
    public let expertPoolBytes: UInt64
    /// KV, session, MLX's own cache and process overhead: 500,000,000 B, the
    /// term `K3DecodeScenario.workingSetReserveBytes` already declares.
    public let workingReserveBytes: UInt64

    public init(
        widestResidentLayerBytes: UInt64, expertPoolBytes: UInt64, workingReserveBytes: UInt64
    ) {
        self.widestResidentLayerBytes = widestResidentLayerBytes
        self.expertPoolBytes = expertPoolBytes
        self.workingReserveBytes = workingReserveBytes
    }

    /// Saturating rather than trapping: a nonsensical census must produce a
    /// refusal, not a crash.
    public var totalBytes: UInt64 {
        PlanningUInt64Arithmetic.sum(
            [widestResidentLayerBytes, expertPoolBytes, workingReserveBytes]).value
    }

    /// Internal rather than `fileprivate` since ``DeepSeekV4MemoryDial`` — a
    /// second planner over the same floor type, in its own file — has to make
    /// the same refusal before it does any arithmetic.
    var isRepresentable: Bool {
        !PlanningUInt64Arithmetic.sum(
            [widestResidentLayerBytes, expertPoolBytes, workingReserveBytes]).overflow
    }

    /// The flagship floor: 5,229,307,136 B.
    ///
    /// Worth keeping next to it: the Mac's measured `mlx_peak_bytes` for the
    /// flagship token is 5,166,220,288 B (M9C). This formula predicts 1.2%
    /// high — the conservative side. A floor that under-predicted the measured
    /// peak would be a bug.
    public static func measuredFlagship(_ census: ArtifactCensus) -> WorkingSetFloor {
        WorkingSetFloor(
            widestResidentLayerBytes: census.layerResidentBytes.max() ?? 0,
            expertPoolBytes: 46_792_704,
            workingReserveBytes: 500_000_000)
    }
}

// MARK: - Link calibration

/// Seconds per byte on a link, and the compute that is not the link.
///
/// There is exactly one of these per measured rig, and **no default**. A
/// projection is only ever as good as the arms it was fitted to, and the phone
/// has no such arms — so on the phone the planner emits no seconds at all
/// rather than a number borrowed from a Mac.
public struct LinkCalibration: Sendable, Equatable, Codable {

    /// How the record names the rig, e.g. "USB4 rig".
    public let label: String
    /// Seconds per byte read in the critical path.
    public let inlineSecondsPerByte: Double
    /// Seconds per byte read behind compute. Lower than inline but not zero:
    /// staging still contends for the link.
    public let stagedSecondsPerByte: Double
    /// Everything that is not the link, per token.
    public let computeSeconds: Double
    /// The unbounded-budget token time, for the UI's "was".
    public let baselineTokenSeconds: Double

    public init(
        label: String, inlineSecondsPerByte: Double, stagedSecondsPerByte: Double,
        computeSeconds: Double, baselineTokenSeconds: Double
    ) {
        self.label = label
        self.inlineSecondsPerByte = inlineSecondsPerByte
        self.stagedSecondsPerByte = stagedSecondsPerByte
        self.computeSeconds = computeSeconds
        self.baselineTokenSeconds = baselineTokenSeconds
    }

    /// Fitted to the two USB4 arms of `docs/experiments/2026-08-11-readahead-mac.md`.
    ///
    /// - `inlineSecondsPerByte`: 108,817,924,096 B / 38.3 s = 2.84 GB/s, the
    ///   serial arm's deterministic term.
    /// - `stagedSecondsPerByte`: the residue of the 85.6 s staged arm once
    ///   compute and the inline terms are removed — 4.958 GB/s effective.
    /// - `computeSeconds`: 63.3 s (= 101.6 − 38.3) minus the inline expert
    ///   (9.10 s) and globals (0.82 s) terms.
    ///
    /// Two independent facts fall out and both check: the staged arm's inline
    /// deterministic residue is layer 0 alone (2,341,257,216 B × the inline rate
    /// = 0.82 s against the 0.7 s reported), and `computeSeconds` lands at
    /// 53.4 s against the "~54 s of compute" the read-ahead record states from a
    /// different decomposition. Both endpoints of the interpolation are
    /// measurements: the null plan returns 85.60 s and the all-pinned plan
    /// 63.30 s, the serial arm's own compute+expert path.
    public static let usb4Rig2026_08_11 = LinkCalibration(
        label: "USB4 rig",
        inlineSecondsPerByte: 3.5211e-10,
        stagedSecondsPerByte: 2.0169e-10,
        computeSeconds: 53.38,
        baselineTokenSeconds: 85.6)
}

// MARK: - Options

/// An expert hot set is only admissible with a measured hit rate behind it.
///
/// M8 measured LRU as *worse* than no cache under read-ahead scan pollution, so
/// no eviction policy appears here at all — only a pinned set, and only one
/// whose hit rate came from a named trace.
public enum ExpertHotSetOption: Sendable, Equatable, Codable {
    case none
    case measured(expertsPerLayer: Int, hitRate: Double, traceLabel: String)
}

/// Whether layer *i+1* is read while layer *i* computes.
public enum ReadAheadOption: Sendable, Equatable, Codable {
    case off
    case depthOne
}

// MARK: - Decisions

/// Why one unit is in the plan. Recorded rather than recomputed, so a run's JSON
/// explains itself without the layer table.
public struct PinDecision: Sendable, Equatable, Codable {

    public enum Unit: Sendable, Equatable, Codable {
        case deterministicLayer(Int)
        /// `embed_tokens` + `lm_head` + `final`, pinned or streamed together —
        /// splitting them needs a per-unit read census that does not exist yet.
        /// K3's globals unit; V4 states its two tables separately and pins
        /// ``outputHead`` alone.
        case globalsBundle
        /// The output head table on its own — the whole `[vocabulary, hidden]`
        /// matrix a pass walks to produce logits.
        ///
        /// The embedding table is deliberately *not* in this unit and is not a
        /// unit of its own: it is the same size as the head and a decode token
        /// reads one row of it, so it ranks near zero on the one column that
        /// orders this ladder. Splitting the two is exactly the per-unit read
        /// census `globalsBundle` says it lacks; V4's census has it, so V4 pins
        /// the half that is worth pinning.
        case outputHead
        case expertHotSet(expertsPerLayer: Int)
    }

    public let unit: Unit
    /// Position in the ranked candidate list, 0-based. Rank is a property of the
    /// artifact, not of the budget: unit *n* has the same rank at every budget.
    public let rank: Int
    public let residentBytes: UInt64
    public let bytesSavedPerToken: UInt64
    /// The ordering key: `bytesSavedPerToken / residentBytes`. 1.000 for any
    /// deterministic layer, 0.499 for the globals bundle, at most 0.230 for a
    /// measured expert hot set.
    public let savedPerResidentByte: Double
    /// Budget still unspent once this unit is pinned.
    public let budgetRemainingAfterBytes: UInt64

    public init(
        unit: Unit, rank: Int, residentBytes: UInt64, bytesSavedPerToken: UInt64,
        savedPerResidentByte: Double, budgetRemainingAfterBytes: UInt64
    ) {
        self.unit = unit
        self.rank = rank
        self.residentBytes = residentBytes
        self.bytesSavedPerToken = bytesSavedPerToken
        self.savedPerResidentByte = savedPerResidentByte
        self.budgetRemainingAfterBytes = budgetRemainingAfterBytes
    }
}

// MARK: - The plan

/// The residency plan one budget implies. Everything the screen shows and
/// everything the run's JSON records comes from here, so the two cannot
/// disagree.
public struct PinPlan: Sendable, Equatable, Codable {

    public let budgetBytes: UInt64
    public let floor: WorkingSetFloor
    public var workingFloorBytes: UInt64 { floor.totalBytes }

    /// Identity and shape of the complete deterministic census used by the
    /// planner. Optional only so older run records remain decodable; a runtime
    /// refuses a plan that predates this binding rather than guessing which
    /// artifact it described.
    public let layerTableSHA256: String?
    public let layerCount: Int?
    /// The read-ahead arm whose staged/inline projection is recorded below:
    /// zero is serial and one is the measured single-layer window.
    public let readAheadDepth: Int?
    /// The response horizon this plan was computed for. Optional only so old
    /// run records remain decodable; a runtime refuses an unbound plan.
    public let requestedMaximumNewTokens: Int?
    /// Stored bytes read once to populate the pinned deterministic tier on the
    /// first pass. These bytes are not a steady-state saving on token one.
    public let projectedFirstPassPinnedLayerFillBytes: UInt64?
    /// The greatest number of pinned deterministic bytes that later passes can
    /// serve from memory if the response reaches its requested token limit.
    /// This is an exact byte bound, not a wall-clock or break-even claim.
    public let projectedPinnedLayerBytesServedAtTokenLimit: UInt64?

    /// Which deterministic layers are held resident, and why. In pin order,
    /// which is rank order, which is layer order.
    public let pinnedLayers: [PinDecision]
    public let pinnedGlobals: PinDecision?
    public let expertHotSet: PinDecision?
    /// Resident bytes of the pinned deterministic tier: layers (stored form)
    /// plus the globals bundle if it is pinned.
    public let pinnedBytes: UInt64
    /// Resident bytes of the pinned expert set, kept separate because it is
    /// handed to a different engine.
    public let expertHotSetBytes: UInt64
    /// `budget − floor − pinned − hot set`. The dial's slack.
    public let unusedBudgetBytes: UInt64
    /// The next budget at which this plan pins strictly more units, and the
    /// unit it admits. `nil` when everything is already pinned.
    public let nextPinBudgetBytes: UInt64?
    public let nextPinUnit: PinDecision.Unit?

    /// Hand this to the runtime stager's
    /// `DeterministicReadAheadSchedule.plan(_:budgetBytes:reserveBytes:)`.
    /// Pool + reserve + everything this plan makes permanently resident.
    public let readAheadReserveBytes: UInt64
    public let projectedStagedLayerCount: Int
    public let projectedInlineLayerCount: Int
    public let projectedRefusedPairCount: Int
    /// The pairs the budget refused, with the arithmetic that refused them. A
    /// skip is normal operation, not an error — but it costs inline seconds, so
    /// it is reported rather than left implicit.
    public let refusedPairs: [ReadAheadPairSchedule.Decision]

    public let projectedBytesPerTokenSaved: UInt64
    public let projectedBytesPerTokenRead: UInt64
    /// Bytes a token no longer reads because the unit is resident. Zero seconds.
    public let pinnedBytesPerToken: UInt64
    /// Bytes read behind compute. They cost link contention, not latency.
    public let stagedBytesPerToken: UInt64
    /// Bytes read in the critical path.
    public let inlineBytesPerToken: UInt64
    public let projectedPeakBytes: UInt64
    /// `nil` without a calibration — the phone has none, and a borrowed second
    /// would be a claim the record cannot support.
    public let projectedTokenSeconds: Double?
    public let calibrationLabel: String?
    /// The calibrated rig's unbounded-budget token time, carried so the screen's
    /// "was" comes from a plan field like every other number it shows. `nil`
    /// exactly when ``projectedTokenSeconds`` is.
    public let baselineTokenSeconds: Double?

    /// A pure, artifact-independent check for decoded or remote plans.
    ///
    /// `PinPlan` intentionally carries redundant fields so a run record can be
    /// understood without the original census. Redundancy is also an input
    /// boundary: every copy must agree before a runner is allowed to allocate
    /// from it. Artifact-specific equality is checked separately against
    /// ``layerTableSHA256`` when manifests are opened.
    public var runtimeValidationError: String? {
        guard let layerTableSHA256,
            layerTableSHA256.count == 64,
            layerTableSHA256.utf8.allSatisfy({
                (48...57).contains($0) || (97...102).contains($0)
            })
        else {
            return "the plan has no valid deterministic layer-census SHA-256"
        }
        guard let layerCount, layerCount > 0 else {
            return "the plan has no positive deterministic layer count"
        }
        guard let readAheadDepth, (0...1).contains(readAheadDepth) else {
            return "the plan does not name a supported read-ahead depth"
        }
        guard let requestedMaximumNewTokens, requestedMaximumNewTokens >= 1 else {
            return "the plan has no positive response-token limit"
        }
        guard let projectedFirstPassPinnedLayerFillBytes,
            let projectedPinnedLayerBytesServedAtTokenLimit
        else {
            return "the plan does not account for first-pass pin fill and later-token reuse"
        }
        guard floor.isRepresentable else {
            return "the plan's working-set floor exceeds UInt64.max"
        }
        guard isResidencyBalanced else {
            return "the plan's residency buckets do not sum to its budget"
        }
        guard isAccountingBalanced else {
            return "the plan's per-token byte buckets do not balance"
        }
        guard projectedStagedLayerCount >= 0, projectedInlineLayerCount >= 0,
            projectedRefusedPairCount >= 0
        else {
            return "the plan contains a negative layer count"
        }
        guard projectedRefusedPairCount == refusedPairs.count,
            projectedRefusedPairCount <= projectedInlineLayerCount
        else {
            return "the plan's refused read-ahead count disagrees with its decisions"
        }

        var seenLayers = Set<Int>()
        var seenRanks = Set<Int>()
        for decision in pinnedLayers {
            guard case .deterministicLayer(let layer) = decision.unit, layer >= 0,
                layer < layerCount
            else {
                return "the pinned-layer tier contains an invalid layer unit"
            }
            guard seenLayers.insert(layer).inserted else {
                return "the memory plan pins layer \(layer) more than once"
            }
            guard decision.rank >= 0, seenRanks.insert(decision.rank).inserted else {
                return "the memory plan contains an invalid or duplicate pin rank"
            }
            guard decision.budgetRemainingAfterBytes <= budgetBytes else {
                return "a pin decision claims more remaining bytes than the plan budget"
            }
        }
        if let pinnedGlobals {
            // Either whole-bundle form is admissible here: K3 pins
            // `embed_tokens + lm_head + final` together because it has no
            // per-unit read census, and V4 pins the output head alone because
            // it has one. Both are "the globals tier" as far as this plan's
            // residency and per-token identities are concerned.
            switch pinnedGlobals.unit {
            case .globalsBundle, .outputHead: break
            case .deterministicLayer, .expertHotSet:
                return "the globals tier contains a non-globals unit"
            }
            guard pinnedGlobals.rank >= 0, seenRanks.insert(pinnedGlobals.rank).inserted else {
                return "the memory plan contains an invalid or duplicate pin rank"
            }
        }
        if let expertHotSet {
            guard case .expertHotSet(let experts) = expertHotSet.unit, experts > 0 else {
                return "the expert tier contains an invalid expert-hot-set unit"
            }
            guard expertHotSet.rank >= 0, seenRanks.insert(expertHotSet.rank).inserted else {
                return "the memory plan contains an invalid or duplicate pin rank"
            }
            guard expertHotSet.residentBytes == expertHotSetBytes else {
                return "the expert hot-set decision disagrees with its resident byte total"
            }
        } else if expertHotSetBytes != 0 {
            return "the plan reserves expert hot-set bytes without an expert decision"
        }

        let streamed = projectedStagedLayerCount.addingReportingOverflow(
            projectedInlineLayerCount)
        guard !streamed.overflow else {
            return "the plan's streamed layer count exceeds Int.max"
        }
        let accountedLayers = pinnedLayers.count.addingReportingOverflow(streamed.partialValue)
        guard !accountedLayers.overflow, accountedLayers.partialValue == layerCount else {
            return "the plan's pinned, staged, and inline layers do not cover its layer table"
        }

        let layerResidency = PlanningUInt64Arithmetic.sum(
            pinnedLayers.lazy.map(\.residentBytes))
        let deterministicResidency = PlanningUInt64Arithmetic.add(
            layerResidency.value, pinnedGlobals?.residentBytes ?? 0)
        guard !layerResidency.overflow, !deterministicResidency.overflow,
            deterministicResidency.value == pinnedBytes
        else {
            return "the pin decisions disagree with the plan's resident byte total"
        }

        guard projectedFirstPassPinnedLayerFillBytes == layerResidency.value else {
            return "the first-pass pin fill does not equal the pinned-layer residency"
        }
        guard let laterPasses = UInt64(exactly: requestedMaximumNewTokens - 1) else {
            return "the plan's later-token count cannot be represented"
        }
        let servedAtLimit = PlanningUInt64Arithmetic.multiply(
            layerResidency.value, laterPasses)
        guard !servedAtLimit.overflow,
            projectedPinnedLayerBytesServedAtTokenLimit == servedAtLimit.value
        else {
            return "the pinned-layer reuse bound disagrees with the response-token limit"
        }

        let decisions = pinnedLayers
            + (pinnedGlobals.map { [$0] } ?? [])
            + (expertHotSet.map { [$0] } ?? [])
        let saved = PlanningUInt64Arithmetic.sum(decisions.lazy.map(\.bytesSavedPerToken))
        guard !saved.overflow, saved.value == projectedBytesPerTokenSaved,
            saved.value == pinnedBytesPerToken
        else {
            return "the pin decisions disagree with the projected bytes saved per token"
        }

        let reserve = PlanningUInt64Arithmetic.sum([
            floor.expertPoolBytes, floor.workingReserveBytes, pinnedBytes,
            expertHotSetBytes,
        ])
        guard !reserve.overflow, reserve.value == readAheadReserveBytes else {
            return "the read-ahead reserve does not equal pool + working + pinned residency"
        }
        let peak = PlanningUInt64Arithmetic.sum([
            workingFloorBytes, pinnedBytes, expertHotSetBytes,
        ])
        guard !peak.overflow, peak.value == projectedPeakBytes else {
            return "the projected peak does not equal floor + pinned residency"
        }

        for decision in refusedPairs {
            guard !decision.isAllowed,
                decision.budgetBytes == budgetBytes,
                decision.reserveBytes == readAheadReserveBytes,
                decision.residentLayer >= 0,
                decision.stagedLayer > decision.residentLayer,
                decision.stagedLayer < layerCount
            else {
                return "a refused read-ahead decision does not describe this plan"
            }
        }
        if readAheadDepth == 0,
            projectedStagedLayerCount != 0 || projectedRefusedPairCount != 0
                || !refusedPairs.isEmpty
        {
            return "a serial plan contains read-ahead decisions"
        }

        switch (nextPinBudgetBytes, nextPinUnit) {
        case (nil, nil): break
        case (.some(let next), .some(_)):
            guard next > budgetBytes else {
                return "the next pin budget is not above the current budget"
            }
        default:
            return "the next pin budget and unit are not both present or both absent"
        }

        switch (projectedTokenSeconds, calibrationLabel, baselineTokenSeconds) {
        case (nil, nil, nil): break
        case (.some(let projected), .some(let label), .some(let baseline)):
            guard projected.isFinite, projected >= 0, baseline.isFinite, baseline >= 0,
                !label.isEmpty
            else {
                return "the plan contains an invalid calibrated time projection"
            }
        default:
            return "the calibrated time fields are only partially present"
        }
        return nil
    }

    /// Every per-token byte lands in exactly one bucket. The accounting identity
    /// the metrics carry, asserted rather than assumed.
    public var isAccountingBalanced: Bool {
        let buckets = pinnedBytesPerToken.addingReportingOverflow(stagedBytesPerToken)
        guard !buckets.overflow else { return false }
        let total = buckets.partialValue.addingReportingOverflow(inlineBytesPerToken)
        guard !total.overflow else { return false }
        let measured = projectedBytesPerTokenRead.addingReportingOverflow(
            projectedBytesPerTokenSaved)
        guard !measured.overflow else { return false }
        return total.partialValue == measured.partialValue
            && pinnedBytesPerToken == projectedBytesPerTokenSaved
    }

    /// `floor + pinned + hot set + unused == budget`. The residency identity, in
    /// the same shape as the byte one: every byte of the dial is spent on
    /// something named, or it is slack.
    public var isResidencyBalanced: Bool {
        let spent = workingFloorBytes.addingReportingOverflow(pinnedBytes)
        guard !spent.overflow else { return false }
        let withExperts = spent.partialValue.addingReportingOverflow(expertHotSetBytes)
        guard !withExperts.overflow else { return false }
        let total = withExperts.partialValue.addingReportingOverflow(unusedBudgetBytes)
        guard !total.overflow else { return false }
        return total.partialValue == budgetBytes
    }

    /// Layers held resident, as ranges — the shape the screen draws.
    public var pinnedLayerRanges: [ClosedRange<Int>] {
        var ranges: [ClosedRange<Int>] = []
        for decision in pinnedLayers {
            guard case .deterministicLayer(let layer) = decision.unit else { continue }
            let successor = ranges.last?.upperBound.addingReportingOverflow(1)
            if let last = ranges.last, let successor, !successor.overflow,
                successor.partialValue == layer
            {
                ranges[ranges.count - 1] = last.lowerBound...layer
            } else {
                ranges.append(layer...layer)
            }
        }
        return ranges
    }

    /// Layers the plan does not hold: staged plus inline.
    public var streamedLayerCount: Int {
        let total = projectedStagedLayerCount.addingReportingOverflow(projectedInlineLayerCount)
        if total.overflow {
            return projectedStagedLayerCount >= 0 && projectedInlineLayerCount >= 0
                ? .max : .min
        }
        return total.partialValue
    }
}

// MARK: - Errors

/// Every refusal names the term it failed against. Nothing here is clamped:
/// silently shrinking a budget to fit would put the run back in the business of
/// discovering its own footprint (spec §5.3).
public enum MemoryDialError: Error, CustomStringConvertible, LocalizedError, Equatable {

    case budgetBelowWorkingFloor(
        budgetBytes: UInt64, workingFloorBytes: UInt64, shortfallBytes: UInt64,
        widestResidentLayerBytes: UInt64, expertPoolBytes: UInt64,
        workingReserveBytes: UInt64)
    case censusDisagreesWithMeasuredTotal(
        censusBytes: UInt64, measuredBytes: UInt64, differenceBytes: Int64)
    case layerTableEmpty
    case layerTableRagged(storedCount: Int, residentCount: Int)
    case expertHotSetWithoutMeasuredTrace(expertsPerLayer: Int)
    case residentSmallerThanStored(layer: Int, stored: UInt64, resident: UInt64)
    /// The same refusal for a unit that is not a layer — V4's output head.
    case globalUnitResidentSmallerThanRead(unit: String, read: UInt64, resident: UInt64)
    case invalidExpertGeometry(
        expertsPerLayer: Int, routedExpertsPerToken: Int, moeLayerCount: Int)
    case invalidMaximumNewTokens(Int)
    case byteCountUnrepresentable(context: String)

    public var description: String {
        switch self {
        case .budgetBelowWorkingFloor(
            let budget, let floor, let shortfall, let widest, let pool, let reserve):
            return """
                declared budget \(MemoryDialFormat.grouped(budget)) B is below the working \
                floor \(MemoryDialFormat.grouped(floor)) B by \
                \(MemoryDialFormat.grouped(shortfall)) B — refused rather than clamped. The \
                floor is the widest widened layer \(MemoryDialFormat.grouped(widest)) B + \
                expert pool \(MemoryDialFormat.grouped(pool)) B + working reserve \
                \(MemoryDialFormat.grouped(reserve)) B, and it is what the run costs with \
                nothing pinned
                """
        case .censusDisagreesWithMeasuredTotal(let census, let measured, let difference):
            return """
                the census accounts for \(MemoryDialFormat.grouped(census)) B per token but \
                the run measured \(MemoryDialFormat.grouped(measured)) B — a difference of \
                \(difference) B. A census that claims more bytes than the run read describes \
                a different artifact
                """
        case .layerTableEmpty:
            return "the layer table is empty: there is nothing to pin and no floor to state"
        case .layerTableRagged(let stored, let resident):
            return """
                the layer table is ragged: \(stored) stored sizes against \(resident) resident \
                sizes. Both forms of every layer are required
                """
        case .expertHotSetWithoutMeasuredTrace(let experts):
            return """
                an expert hot set of \(experts)/layer was requested without a measured hit \
                rate and a named trace behind it. M8 measured LRU as worse than no cache, so \
                a hot set is admissible only on evidence
                """
        case .residentSmallerThanStored(let layer, let stored, let resident):
            return """
                layer \(layer) is \(MemoryDialFormat.grouped(stored)) B stored but only \
                \(MemoryDialFormat.grouped(resident)) B resident — widening never shrinks a \
                tensor, so the census is wrong
                """
        case .globalUnitResidentSmallerThanRead(let unit, let read, let resident):
            return """
                \(unit) reads \(MemoryDialFormat.grouped(read)) B a token but claims only \
                \(MemoryDialFormat.grouped(resident)) B resident — no loader compresses a table \
                by holding it, so the census is wrong
                """
        case .invalidExpertGeometry(let experts, let routed, let layers):
            return """
                invalid expert geometry: \(experts) experts/layer, \(routed) routed/token, \
                \(layers) MoE layers. Counts must be nonnegative and routed experts cannot \
                exceed the experts available in one layer
                """
        case .invalidMaximumNewTokens(let count):
            return "maximum new tokens is \(count); a memory plan requires at least one"
        case .byteCountUnrepresentable(let context):
            return "\(context) exceeds UInt64.max — refused because a wrapped byte count "
                + "would describe a smaller, fictitious plan"
        }
    }

    public var errorDescription: String? { description }
}

// MARK: - The planner

public enum MemoryDialPlanner {

    /// One candidate for the pinned tier, and the single column that ranks it.
    private struct Candidate {
        let unit: PinDecision.Unit
        /// Tie-break within an equal ratio: layer order first, then the globals
        /// bundle, then the hot set. Deterministic, so two runs of the same
        /// budget produce the same plan.
        let orderKey: Int
        let residentBytes: UInt64
        let savedBytesPerToken: UInt64

        var savedPerResidentByte: Double {
            guard residentBytes > 0 else { return 0 }
            return Double(savedBytesPerToken) / Double(residentBytes)
        }
    }

    /// (budget, geometry, floor) -> ``PinPlan``.
    ///
    /// Pure: no clock, no filesystem, no MLX. Same inputs, same plan, on a phone
    /// or in a unit test.
    ///
    /// - Throws: ``MemoryDialError/budgetBelowWorkingFloor(budgetBytes:workingFloorBytes:shortfallBytes:widestResidentLayerBytes:expertPoolBytes:workingReserveBytes:)``
    ///   when the budget cannot pay for the run with nothing pinned, and
    ///   ``MemoryDialError/expertHotSetWithoutMeasuredTrace(expertsPerLayer:)``
    ///   when a hot set arrives without evidence.
    public static func plan(
        budgetBytes: UInt64,
        census: ArtifactCensus,
        floor: WorkingSetFloor,
        readAhead: ReadAheadOption = .depthOne,
        expertHotSet: ExpertHotSetOption = .none,
        calibration: LinkCalibration? = nil,
        maximumNewTokens: Int = 1
    ) throws -> PinPlan {
        guard maximumNewTokens >= 1 else {
            throw MemoryDialError.invalidMaximumNewTokens(maximumNewTokens)
        }
        guard floor.isRepresentable else {
            throw MemoryDialError.byteCountUnrepresentable(context: "working-set floor")
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

        let ranked = try candidates(census: census, expertHotSet: expertHotSet)
        let fill = self.fill(ranked: ranked, headroomBytes: budgetBytes - floorBytes)

        var pinnedLayers: [PinDecision] = []
        var pinnedGlobals: PinDecision?
        var hotSet: PinDecision?
        var pinnedLayerIndices = Set<Int>()
        for decision in fill.decisions {
            switch decision.unit {
            case .deterministicLayer(let layer):
                pinnedLayers.append(decision)
                pinnedLayerIndices.insert(layer)
            case .globalsBundle, .outputHead:
                pinnedGlobals = decision
            case .expertHotSet:
                hotSet = decision
            }
        }

        let hotSetBytes = hotSet?.residentBytes ?? 0
        let pinnedLayerBytes = PlanningUInt64Arithmetic.sum(
            pinnedLayers.lazy.map(\.residentBytes))
        let pinnedTotal = PlanningUInt64Arithmetic.add(
            pinnedLayerBytes.value, pinnedGlobals?.residentBytes ?? 0)
        guard !pinnedLayerBytes.overflow, !pinnedTotal.overflow else {
            throw MemoryDialError.byteCountUnrepresentable(context: "pinned residency")
        }
        let pinnedBytes = pinnedTotal.value
        guard let laterPasses = UInt64(exactly: maximumNewTokens - 1) else {
            throw MemoryDialError.byteCountUnrepresentable(
                context: "later response-token count")
        }
        let servedAtLimit = PlanningUInt64Arithmetic.multiply(
            pinnedLayerBytes.value, laterPasses)
        guard !servedAtLimit.overflow else {
            throw MemoryDialError.byteCountUnrepresentable(
                context: "pinned-layer bytes served at the response limit")
        }
        let savedTotal = PlanningUInt64Arithmetic.sum(
            fill.decisions.lazy.map(\.bytesSavedPerToken))
        guard !savedTotal.overflow else {
            throw MemoryDialError.byteCountUnrepresentable(
                context: "projected bytes saved per token")
        }
        let savedBytes = savedTotal.value

        // Everything the plan makes permanently resident, held back from the
        // budget before the read-ahead schedule considers a single pair.
        let reserve = PlanningUInt64Arithmetic.sum([
            floor.expertPoolBytes, floor.workingReserveBytes, pinnedBytes, hotSetBytes,
        ])
        guard !reserve.overflow else {
            throw MemoryDialError.byteCountUnrepresentable(context: "read-ahead reserve")
        }
        let reserveBytes = reserve.value

        let staging = self.staging(
            census: census, pinned: pinnedLayerIndices, budgetBytes: budgetBytes,
            reserveBytes: reserveBytes, readAhead: readAhead)

        // Bucket every per-token byte exactly once. Staged and inline are summed
        // over units here rather than derived from the total, so
        // `isAccountingBalanced` is a real check on this arithmetic.
        let stagedTotal = PlanningUInt64Arithmetic.sum(
            staging.stagedLayers.lazy.map { census.layerStoredBytes[$0] })
        let inlineLayerTotal = PlanningUInt64Arithmetic.sum(
            staging.inlineLayers.lazy.map { census.layerStoredBytes[$0] })
        guard !stagedTotal.overflow, !inlineLayerTotal.overflow else {
            throw MemoryDialError.byteCountUnrepresentable(
                context: "streamed deterministic bytes per token")
        }
        let stagedBytesPerToken = stagedTotal.value
        var inlineBytesPerToken = inlineLayerTotal.value
        if pinnedGlobals == nil {
            let withGlobals = PlanningUInt64Arithmetic.add(
                inlineBytesPerToken, census.globalsBytesReadPerToken)
            guard !withGlobals.overflow else {
                throw MemoryDialError.byteCountUnrepresentable(
                    context: "inline bytes per token")
            }
            inlineBytesPerToken = withGlobals.value
        }
        // Routed experts are read in the critical path; a pinned hot set removes
        // its measured share of them and nothing else.
        let expertsRead = census.expertBytesPerToken - min(
            census.expertBytesPerToken, hotSet?.bytesSavedPerToken ?? 0)
        let withExperts = PlanningUInt64Arithmetic.add(inlineBytesPerToken, expertsRead)
        guard !withExperts.overflow else {
            throw MemoryDialError.byteCountUnrepresentable(context: "inline bytes per token")
        }
        inlineBytesPerToken = withExperts.value
        // Bytes no unit explains cannot be pinned or staged — they are read, in
        // the critical path, by something this census does not name.
        if census.unattributedBytesPerTokenExact > 0 {
            let withUnattributed = PlanningUInt64Arithmetic.add(
                inlineBytesPerToken, census.unattributedBytesPerTokenExact)
            guard !withUnattributed.overflow else {
                throw MemoryDialError.byteCountUnrepresentable(
                    context: "inline bytes per token")
            }
            inlineBytesPerToken = withUnattributed.value
        }

        let usableCalibration = calibration.flatMap { calibration -> LinkCalibration? in
            let terms = [
                calibration.inlineSecondsPerByte, calibration.stagedSecondsPerByte,
                calibration.computeSeconds, calibration.baselineTokenSeconds,
            ]
            return terms.allSatisfy { $0.isFinite && $0 >= 0 } ? calibration : nil
        }
        let seconds = usableCalibration.flatMap { calibration -> Double? in
            let projection = calibration.computeSeconds
                + calibration.inlineSecondsPerByte * Double(inlineBytesPerToken)
                + calibration.stagedSecondsPerByte * Double(stagedBytesPerToken)
            return projection.isFinite && projection >= 0 ? projection : nil
        }

        let next = nextPin(
            ranked: ranked, floorBytes: floorBytes, budgetBytes: budgetBytes,
            pinnedCount: fill.decisions.count)

        let projectedPeak = PlanningUInt64Arithmetic.sum(
            [floorBytes, pinnedBytes, hotSetBytes])
        guard !projectedPeak.overflow else {
            throw MemoryDialError.byteCountUnrepresentable(context: "projected peak residency")
        }

        return PinPlan(
            budgetBytes: budgetBytes,
            floor: floor,
            layerTableSHA256: census.layerTableSHA256,
            layerCount: census.layerStoredBytes.count,
            readAheadDepth: readAhead == .depthOne ? 1 : 0,
            requestedMaximumNewTokens: maximumNewTokens,
            projectedFirstPassPinnedLayerFillBytes: pinnedLayerBytes.value,
            projectedPinnedLayerBytesServedAtTokenLimit: servedAtLimit.value,
            pinnedLayers: pinnedLayers,
            pinnedGlobals: pinnedGlobals,
            expertHotSet: hotSet,
            pinnedBytes: pinnedBytes,
            expertHotSetBytes: hotSetBytes,
            unusedBudgetBytes: fill.remainingBytes,
            nextPinBudgetBytes: next?.budgetBytes,
            nextPinUnit: next?.unit,
            readAheadReserveBytes: reserveBytes,
            projectedStagedLayerCount: staging.stagedLayers.count,
            projectedInlineLayerCount: staging.inlineLayers.count,
            projectedRefusedPairCount: staging.refused.count,
            refusedPairs: staging.refused,
            projectedBytesPerTokenSaved: savedBytes,
            projectedBytesPerTokenRead: census.measuredBytesPerToken
                - min(census.measuredBytesPerToken, savedBytes),
            pinnedBytesPerToken: savedBytes,
            stagedBytesPerToken: stagedBytesPerToken,
            inlineBytesPerToken: inlineBytesPerToken,
            projectedPeakBytes: projectedPeak.value,
            projectedTokenSeconds: seconds,
            calibrationLabel: seconds == nil ? nil : usableCalibration?.label,
            baselineTokenSeconds: seconds == nil ? nil : usableCalibration?.baselineTokenSeconds)
    }

    /// Budgets at which the plan changes — the dial's snap points.
    ///
    /// The floor first (below it the dial refuses), then the budget at which
    /// each successive prefix of the ranked ladder is affordable in full. At
    /// flagship shape the last two are 114,047,231,232 B (every deterministic
    /// layer) and 118,744,900,864 B (the globals bundle as well), which is where
    /// the derived ladder disagrees with spec §11.5's table.
    ///
    /// `readAhead` does not move the ladder and is accepted so that a caller
    /// cannot configure ``plan(budgetBytes:census:floor:readAhead:expertHotSet:calibration:)``
    /// one way and the dial's stops another: staging decides which bucket a byte
    /// lands in, never whether a unit is pinnable.
    public static func snapPoints(
        census: ArtifactCensus,
        floor: WorkingSetFloor,
        readAhead: ReadAheadOption = .depthOne,
        expertHotSet: ExpertHotSetOption = .none
    ) -> [UInt64] {
        guard let ranked = try? candidates(census: census, expertHotSet: expertHotSet) else {
            return [floor.totalBytes]
        }
        var points = [floor.totalBytes]
        var running = floor.totalBytes
        for candidate in ranked {
            let next = running.addingReportingOverflow(candidate.residentBytes)
            guard !next.overflow else { break }
            running = next.partialValue
            points.append(running)
        }
        return points
    }

    // MARK: Ranking and fill

    /// The candidate ladder: every unit the dial can pin, in priority order.
    private static func candidates(
        census: ArtifactCensus, expertHotSet: ExpertHotSetOption
    ) throws -> [Candidate] {
        var candidates: [Candidate] = []
        for (layer, stored) in census.layerStoredBytes.enumerated() where stored > 0 {
            // A pinned layer holds its stored bytes and stops reading exactly
            // those bytes, so its ratio is 1.000 — the ceiling of the ladder.
            candidates.append(
                Candidate(
                    unit: .deterministicLayer(layer), orderKey: layer,
                    residentBytes: stored, savedBytesPerToken: stored))
        }
        if census.globalsStoredBytes > 0 {
            candidates.append(
                Candidate(
                    unit: .globalsBundle, orderKey: Int.max - 1,
                    residentBytes: census.globalsStoredBytes,
                    savedBytesPerToken: census.globalsBytesReadPerToken))
        }
        if case .measured(let experts, let hitRate, let traceLabel) = expertHotSet {
            let named = !traceLabel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            guard named, hitRate > 0, hitRate <= 1, experts > 0,
                experts <= census.expertsPerLayer
            else {
                throw MemoryDialError.expertHotSetWithoutMeasuredTrace(expertsPerLayer: experts)
            }
            let expertCount = PlanningUInt64Arithmetic.multiply(
                UInt64(experts), UInt64(census.moeLayerCount))
            let resident = PlanningUInt64Arithmetic.multiply(
                expertCount.value, census.expertStrideBytes)
            guard !expertCount.overflow, !resident.overflow else {
                throw MemoryDialError.byteCountUnrepresentable(
                    context: "expert hot-set residency")
            }
            // Floor, not round: a hit rate is a measurement with a trace behind
            // it, and rounding up would claim bytes the trace did not save.
            let saved: UInt64
            if hitRate == 1 {
                saved = census.expertBytesPerToken
            } else {
                let product = (Double(census.expertBytesPerToken) * hitRate).rounded(.down)
                saved = product >= Double(UInt64.max) ? .max : UInt64(product)
            }
            if resident.value > 0 {
                candidates.append(
                    Candidate(
                        unit: .expertHotSet(expertsPerLayer: experts), orderKey: Int.max,
                        residentBytes: resident.value, savedBytesPerToken: saved))
            }
        }
        return candidates.sorted {
            if $0.savedPerResidentByte != $1.savedPerResidentByte {
                return $0.savedPerResidentByte > $1.savedPerResidentByte
            }
            return $0.orderKey < $1.orderKey
        }
    }

    private struct Fill {
        let decisions: [PinDecision]
        let remainingBytes: UInt64
    }

    /// Walk the ranked list and pin everything that still fits, **skipping**
    /// what does not rather than stopping at the first miss.
    ///
    /// Skipping is worth stating because it changes the answer: at 24 GiB a
    /// stop-at-first-miss fill pins 16 layers and leaves 875,731,712 B unused,
    /// while the skip fill reaches past three layers it cannot afford to pin
    /// layer 19 as well — one more layer, 844,398,592 B/token saved. Both are
    /// pure and deterministic; skip wins on the only axis the dial has.
    private static func fill(ranked: [Candidate], headroomBytes: UInt64) -> Fill {
        var remaining = headroomBytes
        var decisions: [PinDecision] = []
        for (rank, candidate) in ranked.enumerated() where candidate.residentBytes <= remaining {
            remaining -= candidate.residentBytes
            decisions.append(
                PinDecision(
                    unit: candidate.unit, rank: rank, residentBytes: candidate.residentBytes,
                    bytesSavedPerToken: candidate.savedBytesPerToken,
                    savedPerResidentByte: candidate.savedPerResidentByte,
                    budgetRemainingAfterBytes: remaining))
        }
        return Fill(decisions: decisions, remainingBytes: remaining)
    }

    /// The smallest budget above this one that pins strictly more units, and the
    /// unit it admits.
    ///
    /// Not "this budget plus the next unit's size": at the phone's 5.8 GB the
    /// first pin is an *MLA* layer 844,398,592 B, of which 570,692,864 B is
    /// already paid for, so the answer is 6,073,705,728 B — the number the
    /// design document quotes. A budget increase only ever changes the fill by
    /// covering some skipped candidate's deficit, so the deficits are the only
    /// budgets worth testing, and each is tested by re-running the fill rather
    /// than by assuming the count went up (it does not always: admitting an
    /// earlier candidate can displace a later one).
    private static func nextPin(
        ranked: [Candidate], floorBytes: UInt64, budgetBytes: UInt64, pinnedCount: Int
    ) -> (budgetBytes: UInt64, unit: PinDecision.Unit)? {
        guard pinnedCount < ranked.count else { return nil }
        var remaining = budgetBytes - floorBytes
        var deficits: Set<UInt64> = []
        var pinned: [Bool] = []
        for candidate in ranked {
            if candidate.residentBytes <= remaining {
                remaining -= candidate.residentBytes
                pinned.append(true)
            } else {
                deficits.insert(candidate.residentBytes - remaining)
                pinned.append(false)
            }
        }
        for deficit in deficits.sorted() {
            let candidateBudget = budgetBytes.addingReportingOverflow(deficit)
            guard !candidateBudget.overflow else { continue }
            let retry = fill(ranked: ranked, headroomBytes: candidateBudget.partialValue - floorBytes)
            guard retry.decisions.count > pinnedCount else { continue }
            let admitted = retry.decisions.first { !pinned[$0.rank] }
            guard let unit = admitted?.unit else { continue }
            return (candidateBudget.partialValue, unit)
        }
        return nil
    }

    // MARK: Staging

    private struct Staging {
        let stagedLayers: [Int]
        let inlineLayers: [Int]
        let refused: [ReadAheadPairSchedule.Decision]
    }

    /// Which streamed layers arrive behind compute and which arrive in the way.
    ///
    /// The pair arithmetic is ``ReadAheadPairSchedule``'s own, which
    /// `ReadAheadScheduleAgreementTests` pins field for field to the runtime
    /// stager's `DeterministicReadAheadSchedule` — so the plan cannot promise a
    /// staging the schedule then refuses. Two rules shape the list: a pinned
    /// layer is
    /// neither staged nor inline (it is already there), and layer 0 has no
    /// predecessor to hide behind, so unpinned it is always inline.
    private static func staging(
        census: ArtifactCensus, pinned: Set<Int>, budgetBytes: UInt64, reserveBytes: UInt64,
        readAhead: ReadAheadOption
    ) -> Staging {
        var staged: [Int] = []
        var inline: [Int] = []
        var refused: [ReadAheadPairSchedule.Decision] = []
        for layer in census.layerStoredBytes.indices where !pinned.contains(layer) {
            guard readAhead == .depthOne, layer > 0 else {
                inline.append(layer)
                continue
            }
            let previous = layer - 1
            let decision = ReadAheadPairSchedule.decide(
                residentLayer: previous,
                // The widened copy of layer i is resident whether or not layer i
                // is pinned: `materialise()` widens per use and `asType` copies.
                resident: .init(
                    storedBytes: census.layerStoredBytes[previous],
                    residentBytes: census.layerResidentBytes[previous]),
                stagedLayer: layer,
                next: .init(
                    storedBytes: census.layerStoredBytes[layer],
                    residentBytes: census.layerResidentBytes[layer]),
                budgetBytes: budgetBytes,
                reserveBytes: reserveBytes)
            if decision.isAllowed {
                staged.append(layer)
            } else {
                inline.append(layer)
                refused.append(decision)
            }
        }
        return Staging(stagedLayers: staged, inlineLayers: inline, refused: refused)
    }

    // MARK: Summary

    public static func summary(_ plan: PinPlan) -> MemoryDialSummary {
        let layerCountAddition = plan.pinnedLayers.count.addingReportingOverflow(
            plan.streamedLayerCount)
        let layerCount = layerCountAddition.overflow ? Int.max : layerCountAddition.partialValue
        let measuredTotal = PlanningUInt64Arithmetic.add(
            plan.projectedBytesPerTokenRead, plan.projectedBytesPerTokenSaved).value

        let headline: String
        if plan.pinnedLayers.isEmpty {
            headline = "no layers held in RAM — \(layerCount) stream"
        } else {
            headline = "\(plan.pinnedLayers.count) of \(layerCount) layers held in RAM"
        }

        var residency = "\(MemoryDialFormat.gib(plan.pinnedBytes)) GiB pinned"
        if plan.expertHotSetBytes > 0 {
            residency += " + \(MemoryDialFormat.gib(plan.expertHotSetBytes)) GiB experts"
        }
        residency += " + \(MemoryDialFormat.gib(plan.workingFloorBytes)) GiB floor"

        var secondsLabel: String?
        if let seconds = plan.projectedTokenSeconds, seconds.isFinite, seconds >= 0,
            let label = plan.calibrationLabel
        {
            let baseline = plan.baselineTokenSeconds.flatMap {
                $0.isFinite && $0 >= 0 ? $0 : nil
            } ?? seconds
            secondsLabel = String(
                format: "≈%.1f s/token (%@, baseline %.1f s)", seconds, label,
                baseline)
        }

        let readAheadLabel: String
        if plan.streamedLayerCount == 0 {
            readAheadLabel = "read-ahead: nothing streams"
        } else {
            readAheadLabel = "read-ahead: \(plan.projectedStagedLayerCount) of "
                + "\(plan.streamedLayerCount) streamed layers staged"
        }

        var nextStep: String?
        if let budget = plan.nextPinBudgetBytes, let unit = plan.nextPinUnit,
            budget > plan.budgetBytes
        {
            nextStep = "+\(MemoryDialFormat.gib(budget - plan.budgetBytes)) GiB pins "
                + MemoryDialFormat.describe(unit)
        }

        var warnings: [String] = []
        if plan.pinnedLayers.isEmpty && plan.pinnedGlobals == nil && plan.expertHotSet == nil {
            var warning = "nothing fits above the floor: "
                + "\(MemoryDialFormat.grouped(plan.unusedBudgetBytes)) B of headroom"
            if let budget = plan.nextPinBudgetBytes {
                warning += ", and the first pin needs a declared budget of "
                    + "\(MemoryDialFormat.grouped(budget)) B"
            }
            warnings.append(warning)
        }
        for decision in plan.refusedPairs.prefix(3) {
            warnings.append(
                "layer \(decision.residentLayer) read-ahead pair refused (needs "
                    + "\(MemoryDialFormat.grouped(decision.headroomBytes.magnitude)) B more)")
        }
        if plan.refusedPairs.count > 3 {
            warnings.append("and \(plan.refusedPairs.count - 3) more refused read-ahead pairs")
        }
        if !plan.isAccountingBalanced {
            warnings.append("per-token byte buckets do not sum to the measured total")
        }

        return MemoryDialSummary(
            budgetLabel: MemoryDialFormat.budget(plan.budgetBytes),
            headline: headline,
            pinnedLayerRanges: plan.pinnedLayerRanges,
            residencyLabel: residency,
            perTokenLabel: "\(MemoryDialFormat.gb(plan.projectedBytesPerTokenRead)) GB read "
                + "per token, was \(MemoryDialFormat.gb(measuredTotal)) GB",
            savedFraction: measuredTotal == 0
                ? 0 : Double(plan.projectedBytesPerTokenSaved) / Double(measuredTotal),
            tokenSecondsLabel: secondsLabel,
            readAheadLabel: readAheadLabel,
            nextStepLabel: nextStep,
            nextStepBudgetBytes: plan.nextPinBudgetBytes,
            floorLabel: "\(MemoryDialFormat.gib(plan.workingFloorBytes)) GiB minimum — "
                + "widest layer widened + expert pool + working reserve",
            warnings: warnings)
    }
}

// MARK: - The dial screen

/// What the dial screen shows. Every byte figure here comes from a ``PinPlan``
/// field, so the screen and the run's JSON cannot disagree.
///
/// The floor is drawn as a hard left stop rather than a slider region: below it
/// the slider will not go, and the error text names the term the budget failed
/// against. ``tokenSecondsLabel`` is *absent* — not zero, not "unknown" — when
/// the link has no calibration.
public struct MemoryDialSummary: Sendable, Equatable {
    public let budgetLabel: String
    public let headline: String
    public let pinnedLayerRanges: [ClosedRange<Int>]
    public let residencyLabel: String
    public let perTokenLabel: String
    public let savedFraction: Double
    public let tokenSecondsLabel: String?
    public let readAheadLabel: String
    public let nextStepLabel: String?
    public let nextStepBudgetBytes: UInt64?
    public let floorLabel: String
    public let warnings: [String]
}

// MARK: - Formatting

/// Number formatting for the dial, done by hand and not by `NumberFormatter`:
/// the strings appear in tests and in run records, and neither may change with
/// the device's locale.
enum MemoryDialFormat {

    static func grouped(_ value: UInt64) -> String {
        var digits = Array(String(value))
        var index = digits.count - 3
        while index > 0 {
            digits.insert(",", at: index)
            index -= 3
        }
        return String(digits)
    }

    /// Three significant figures, which is what the screen has room for:
    /// 19,1 GiB reads as "19.1", 4.87 GiB as "4.87".
    static func gib(_ bytes: UInt64) -> String {
        let value = Double(bytes) / 1_073_741_824
        if value >= 100 { return String(format: "%.0f", value) }
        if value >= 10 { return String(format: "%.1f", value) }
        return String(format: "%.2f", value)
    }

    /// Decimal GB, one place: the unit the per-token numbers are recorded in.
    static func gb(_ bytes: UInt64) -> String {
        String(format: "%.1f", Double(bytes) / 1_000_000_000)
    }

    /// A whole number of GiB stays whole — a dial set to "24 GiB" should not
    /// read back as "24.0 GiB".
    static func budget(_ bytes: UInt64) -> String {
        if bytes % 1_073_741_824 == 0 { return "\(bytes / 1_073_741_824) GiB" }
        return "\(gib(bytes)) GiB"
    }

    static func describe(_ unit: PinDecision.Unit) -> String {
        switch unit {
        case .deterministicLayer(let layer): return "layer \(layer)"
        case .globalsBundle: return "the globals bundle"
        case .outputHead: return "the output head"
        case .expertHotSet(let experts): return "an expert hot set of \(experts)/layer"
        }
    }
}
