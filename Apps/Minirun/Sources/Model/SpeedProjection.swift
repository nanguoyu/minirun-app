import Foundation
import MinirunKit

/// A measured read ceiling for one volume.
///
/// Nothing in this product projects a speed without one of these. A projection
/// derived from a datasheet number would be a claim about hardware; a
/// projection derived from this is a claim about *this* rig, which is the only
/// kind the record accepts.
///
/// Called `LinkCalibration` until the memory dial started importing the kit's
/// planner, which has a public type of that name for a different thing — a
/// seconds-per-byte fit over one rig's inline and staged arms. Two calibrations
/// of one link, at two levels of detail, and the collision was ambiguous
/// wherever both modules were in scope. This one is per *volume* and the name
/// now says so.
struct VolumeCalibration: Equatable, Sendable, Codable, Identifiable {
    let volumeMountPath: String
    let volumeName: String?
    /// The measured ceiling. Not a bus rating.
    let bytesPerSecond: Double
    /// `USB4 40 Gb/s`, `USB3 10 Gb/s`, `internal NVMe` — reported, not inferred
    /// from the rate.
    let busDescription: String
    let measuredAt: Date
    let sampleSeconds: Double

    var id: String { volumeMountPath }

    /// Older than a week, or measured against a different volume. Labelled
    /// stale, never reused silently.
    func isStale(now: Date = Date()) -> Bool {
        now.timeIntervalSince(measuredAt) > 7 * 86400
    }

    var provenanceSentence: String {
        "≈ from this device's measured link (\(MRFormat.throughput(bytesPerSecond)), "
            + "\(busDescription), measured \(MRFormat.relative(measuredAt))) and this model's "
            + "measured compute term. Not a promise."
    }
}

/// The per-token byte and compute terms a projection needs.
struct WorkloadTerms: Equatable, Sendable {
    /// Deterministic bytes a token reads.
    let deterministicBytesPerToken: UInt64
    /// Routed-expert bytes a token reads.
    let expertBytesPerToken: UInt64
    /// Time the GPU spends, measured, with the reads taken out.
    let computeSecondsPerToken: Double
    /// Whether these came from a run on this device or from the archive.
    let isMeasuredHere: Bool

    var totalBytesPerToken: UInt64 { deterministicBytesPerToken &+ expertBytesPerToken }
}

/// The identity a product speed projection must be measured against. A link
/// rate plus archived workload terms is not evidence for the model artifact on
/// the current machine; all three fields must be present and the workload terms
/// must have been measured here before a projection can leave the model layer.
struct ProjectionEvidence: Equatable, Sendable {
    let model: ModelID
    let artifactRevision: String
    let deviceIdentifier: String

    func matches(plan: BudgetPlan, terms: WorkloadTerms) -> Bool {
        model == plan.model
            && !artifactRevision.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !deviceIdentifier.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && terms.isMeasuredHere
    }
}

/// A projected seconds-per-token, always a range, always prefixed `≈`.
struct SpeedProjection: Equatable, Sendable {
    let secondsPerTokenLow: Double
    let secondsPerTokenHigh: Double
    let mid: Double
    /// The staged read-ahead pair is funded at this budget, so the
    /// deterministic stream hides behind compute. This is the step in the curve.
    let deterministicHidden: Bool
    /// Share of expert bytes the hot set serves from cache at this budget.
    let expertCacheFraction: Double

    var displayString: String {
        "≈ \(MRFormat.clock(secondsPerTokenLow)) – \(MRFormat.clock(secondsPerTokenHigh)) / token"
    }

    var compactString: String {
        "≈ \(MRFormat.clock(mid))/token"
    }

    var spokenString: String {
        "approximately \(MRFormat.tokenRateSpoken(1 / mid)), a projection, not a measurement"
    }
}

/// Turns a budget into a projected seconds-per-token, or refuses to.
///
/// ## The one discontinuity this curve is entitled to draw
///
/// A **step down** where the staged pair first fits, because the deterministic
/// read stops being serial with compute. That step is measured, twice, on two
/// rigs.
///
/// ## What it deliberately does not draw
///
/// A slope as the **pinned tier** grows. The dial's arithmetic says a pinned
/// layer saves its own bytes every token, and that is true — from token 2. The
/// only measurement anyone has of pinning is `2026-08-11-dial-decode.md`, a
/// single-pass token, and there the tier made the token **11.31 s slower**
/// (83.09 s pinned against 71.78 s unpinned, both fifteen-minute-idle keepers):
/// the fill is inline because the stager skips a pinned layer, and 18.4 GB of
/// residency on a 32 GiB host cost 18.2 s of kernel time to page-reclaim
/// pressure. The steady-state payoff is real arithmetic and an **unmeasured**
/// outcome, so the curve stays flat across the pinned tier and the strip says
/// why in words. A line that sloped down there would be this product promising
/// a speedup nobody has seen.
///
/// ## And no slope for the hot set either
///
/// `expertCacheFraction` is gone. It modelled a pinned expert cache growing with
/// the budget, and the product ladder says no budget below 116,129,117,440 B
/// pins a single expert — every deterministic layer outranks the best measured
/// hot set by 4.3× on bytes saved per resident byte. The slope was drawing a
/// tier that does not exist at any budget a device here can offer.
enum SpeedProjector {

    /// The ±12 % the range spans. Wide enough to be honest about a projection
    /// built from two measured terms and no model of thermal drift.
    static let relativeUncertainty = 0.12

    /// Nil when there is no verified link measurement for the volume. The
    /// strip then reads `Not measured` and shows no curve — a guessed curve
    /// would be worse than an empty one.
    static func project(
        plan: BudgetPlan, terms: WorkloadTerms, link: VolumeCalibration?
    ) -> SpeedProjection? {
        guard let link, link.bytesPerSecond > 0 else { return nil }
        guard plan.isRunnable else { return nil }

        // No cache credit. The ladder reaches an expert only past full
        // deterministic residency, so at every budget this control can state,
        // every expert byte is read.
        let hitFraction = 0.0
        let deterministicSeconds = Double(terms.deterministicBytesPerToken) / link.bytesPerSecond
        let expertSeconds =
            Double(terms.expertBytesPerToken) * (1 - hitFraction) / link.bytesPerSecond

        let hidden = plan.stagedTierIsFunded
        // Serial: compute, then the deterministic read, then the expert read.
        // Staged: the deterministic read runs alongside everything else, so the
        // token costs whichever of the two paths is longer.
        let mid =
            hidden
            ? max(terms.computeSecondsPerToken + expertSeconds, deterministicSeconds)
            : terms.computeSecondsPerToken + deterministicSeconds + expertSeconds

        return SpeedProjection(
            secondsPerTokenLow: mid * (1 - relativeUncertainty),
            secondsPerTokenHigh: mid * (1 + relativeUncertainty),
            mid: mid,
            deterministicHidden: hidden,
            expertCacheFraction: hitFraction)
    }

    /// The curve the projection strip draws: `samples` points across the dial's
    /// domain, each either a projection or nil where the budget is refused.
    static func curve(
        plan: BudgetPlan, terms: WorkloadTerms, link: VolumeCalibration?, samples: Int = 96
    ) -> [(budgetBytes: UInt64, projection: SpeedProjection?)] {
        guard samples > 1, plan.deviceCeilingBytes > 0 else { return [] }
        return (0..<samples).map { step in
            let fraction = Double(step) / Double(samples - 1)
            let budget = UInt64(fraction * Double(plan.deviceCeilingBytes))
            let candidate = plan.with(budgetBytes: budget)
            return (budget, project(plan: candidate, terms: terms, link: link))
        }
    }
}

/// A run that actually happened, drawn on the strip as a solid dot. Measured
/// and projected are never the same visual object.
struct MeasuredPoint: Equatable, Sendable, Identifiable {
    let id: UUID
    let budgetBytes: UInt64
    let secondsPerToken: Double
    let at: Date
    let logitsDigest: String?

    init(
        id: UUID = UUID(), budgetBytes: UInt64, secondsPerToken: Double, at: Date,
        logitsDigest: String?
    ) {
        self.id = id
        self.budgetBytes = budgetBytes
        self.secondsPerToken = secondsPerToken
        self.at = at
        self.logitsDigest = logitsDigest
    }
}
