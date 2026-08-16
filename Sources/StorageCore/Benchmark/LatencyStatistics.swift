import Foundation

/// Latency distribution for a run, in microseconds.
///
/// Spec §18.1 asks for p50/p95/p99 rather than an average, because the tail is
/// what a decode loop actually waits on: a mean hides the read that stalled the
/// GPU. Percentiles use the nearest-rank definition on the sorted samples —
/// no interpolation, so every reported percentile is a latency that was really
/// observed.
public struct LatencyStatistics: Codable, Equatable {
    public let sampleCount: Int
    public let minUs: Double
    public let meanUs: Double
    public let p50Us: Double
    public let p95Us: Double
    public let p99Us: Double
    public let maxUs: Double

    /// Nearest-rank percentile: index `ceil(p/100 * n) - 1`, clamped.
    public static func percentile(_ percent: Double, ofSorted sorted: [UInt64]) -> UInt64 {
        guard !sorted.isEmpty else { return 0 }
        let rank = Int((percent / 100 * Double(sorted.count)).rounded(.up))
        let index = min(max(rank - 1, 0), sorted.count - 1)
        return sorted[index]
    }

    /// Sorts `samples` in place and summarizes them. Returns nil for an empty
    /// set: a run with no completed reads has no distribution, and reporting
    /// zeros would look like an instant one.
    public static func compute(nanosecondSamples samples: inout [UInt64]) -> LatencyStatistics? {
        guard !samples.isEmpty else { return nil }
        samples.sort()

        let total = samples.reduce(UInt64(0), &+)
        let toMicroseconds = { (nanoseconds: UInt64) in Double(nanoseconds) / 1_000 }

        return LatencyStatistics(
            sampleCount: samples.count,
            minUs: toMicroseconds(samples[0]),
            meanUs: Double(total) / Double(samples.count) / 1_000,
            p50Us: toMicroseconds(percentile(50, ofSorted: samples)),
            p95Us: toMicroseconds(percentile(95, ofSorted: samples)),
            p99Us: toMicroseconds(percentile(99, ofSorted: samples)),
            maxUs: toMicroseconds(samples[samples.count - 1])
        )
    }
}
