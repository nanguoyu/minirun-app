import Foundation

/// The per-token terms a projection needs, as the caller measured them.
///
/// Nothing here is invented by this module. The two byte terms are the measured
/// split of a token's reads; the compute term is the GPU time with the reads
/// taken out. A caller that has none of these gets no projection.
public struct TokenWorkload: Sendable, Equatable, Codable {
    public let deterministicBytesPerToken: UInt64
    public let expertBytesPerToken: UInt64
    public let computeSecondsPerToken: Double
    /// Share of expert bytes the hot set serves from memory at the budget the
    /// caller is planning. Zero unless the caller has a budget to state — a
    /// cache fraction assumed on a storage screen would be a second projection
    /// hiding inside the first.
    public let expertCacheFraction: Double
    /// Whether the staged read-ahead pair is funded, so the deterministic read
    /// runs alongside compute instead of in front of it. This is the step in the
    /// curve, and it belongs to the budget, not to the drive.
    public let deterministicOverlapsCompute: Bool

    public init(
        deterministicBytesPerToken: UInt64, expertBytesPerToken: UInt64,
        computeSecondsPerToken: Double, expertCacheFraction: Double = 0,
        deterministicOverlapsCompute: Bool = false
    ) {
        self.deterministicBytesPerToken = deterministicBytesPerToken
        self.expertBytesPerToken = expertBytesPerToken
        self.computeSecondsPerToken = computeSecondsPerToken
        self.expertCacheFraction = expertCacheFraction
        self.deterministicOverlapsCompute = deterministicOverlapsCompute
    }

    public var totalBytesPerToken: UInt64 {
        deterministicBytesPerToken &+ expertBytesPerToken
    }
}

/// A projected seconds-per-token. Always a range, never a point.
public struct ProjectedTokenTime: Sendable, Equatable, Codable {
    public let secondsPerTokenLow: Double
    public let secondsPerTokenHigh: Double
    public let mid: Double
    /// The rate this was computed from, so a stale report can be read against
    /// the link it was measured on.
    public let bytesPerSecond: Double
    public let deterministicHidden: Bool

    public init(
        secondsPerTokenLow: Double, secondsPerTokenHigh: Double, mid: Double,
        bytesPerSecond: Double, deterministicHidden: Bool
    ) {
        self.secondsPerTokenLow = secondsPerTokenLow
        self.secondsPerTokenHigh = secondsPerTokenHigh
        self.mid = mid
        self.bytesPerSecond = bytesPerSecond
        self.deterministicHidden = deterministicHidden
    }
}

/// Turns a measured read rate into a projected token time, or refuses to.
///
/// This is the app's own projection arithmetic, moved to where the assessment
/// can reach it and left identical: the same serial-versus-staged shape, the
/// same ±12 % band, and the same refusal to project without a measurement. It
/// is not a second model of the machine, and `SpeedProjectionAgreementTests`
/// exists to keep it from becoming one.
public enum TokenTimeProjector {

    /// The ±12 % the range spans. Wide enough to be honest about a projection
    /// built from two measured terms and no model of thermal drift.
    public static let relativeUncertainty = 0.12

    /// Nil when there is no measured rate. A guessed curve is worse than an
    /// empty one, and this product shows the empty one.
    public static func project(
        workload: TokenWorkload, bytesPerSecond: Double?
    ) -> ProjectedTokenTime? {
        guard let rate = bytesPerSecond, rate > 0, rate.isFinite else { return nil }

        let deterministicSeconds = Double(workload.deterministicBytesPerToken) / rate
        let hitFraction = min(1, max(0, workload.expertCacheFraction))
        let expertSeconds = Double(workload.expertBytesPerToken) * (1 - hitFraction) / rate

        // Serial: compute, then the deterministic read, then the expert read.
        // Staged: the deterministic read runs alongside everything else, so the
        // token costs whichever of the two paths is longer.
        let mid =
            workload.deterministicOverlapsCompute
            ? max(workload.computeSecondsPerToken + expertSeconds, deterministicSeconds)
            : workload.computeSecondsPerToken + deterministicSeconds + expertSeconds

        return ProjectedTokenTime(
            secondsPerTokenLow: mid * (1 - relativeUncertainty),
            secondsPerTokenHigh: mid * (1 + relativeUncertainty),
            mid: mid,
            bytesPerSecond: rate,
            deterministicHidden: workload.deterministicOverlapsCompute)
    }
}
