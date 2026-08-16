import Foundation

/// Every number the product prints goes through here.
///
/// Product byte values use one decimal scale. A budget and the footprint it is
/// compared with must never make the reader mentally convert GB to GiB in the
/// same sentence. Binary formatting remains available for machine-facing
/// diagnostics and old experiment assertions, not for product surfaces.
public enum MRFormat {

    // MARK: - Bytes

    /// Decimal — GB/TB, 1000-based. Stated budgets, artifact sizes, link rates.
    public static func bytesDecimal(_ bytes: UInt64, precision: Int? = nil) -> String {
        scale(
            Double(bytes), base: 1000,
            units: ["B", "kB", "MB", "GB", "TB", "PB"], minimumUnitIndex: 2,
            precision: precision)
    }

    public static func bytesDecimal(_ bytes: Int64, precision: Int? = nil) -> String {
        let sign = bytes < 0 ? "-" : ""
        return sign
            + scale(
                Double(bytes.magnitude), base: 1000,
                units: ["B", "kB", "MB", "GB", "TB", "PB"], minimumUnitIndex: 2,
                precision: precision)
    }

    /// Binary — GiB, 1024-based. Measured footprints, peaks, spare.
    public static func bytesBinary(_ bytes: UInt64, precision: Int? = nil) -> String {
        scale(
            Double(bytes), base: 1024,
            units: ["B", "KiB", "MiB", "GiB", "TiB", "PiB"], minimumUnitIndex: 2,
            precision: precision)
    }

    /// A catalog size. Zero is the catalog's sentinel for a repository whose
    /// tree did not publish a usable total; presenting it as `0 B` would turn
    /// missing metadata into a very precise claim.
    public static func publishedBytes(_ bytes: UInt64, precision: Int? = nil) -> String {
        guard bytes > 0 else { return "not published" }
        return bytesDecimal(bytes, precision: precision)
    }

    /// A measured artifact size. Absence and an empty tally both mean that no
    /// useful measurement is available on the product surface.
    public static func measuredBytes(_ bytes: UInt64?, precision: Int? = nil) -> String {
        guard let bytes, bytes > 0 else { return "not measured" }
        return bytesDecimal(bytes, precision: precision)
    }

    private static func scale(
        _ value: Double, base: Double, units: [String], minimumUnitIndex: Int,
        precision: Int?
    ) -> String {
        if value == 0 { return "0 \(units[minimumUnitIndex])" }
        if value < pow(base, Double(minimumUnitIndex)) {
            return "<1 \(units[minimumUnitIndex])"
        }
        var magnitude = value
        var index = 0
        while (magnitude >= base || index < minimumUnitIndex) && index < units.count - 1 {
            magnitude /= base
            index += 1
        }
        let digits = precision ?? (magnitude >= 100 ? 0 : (magnitude >= 10 ? 1 : 2))
        return String(format: "%.\(digits)f %@", magnitude, units[index])
    }

    /// The exact count for machine-facing diagnostics and fixture assertions.
    /// Product surfaces use the human-scale formatters above.
    public static func exactBytes(_ bytes: UInt64) -> String {
        "\(grouped(bytes)) B"
    }

    public static func exactBytes(_ bytes: Int64) -> String {
        "\(grouped(bytes)) B"
    }

    public static func grouped(_ value: UInt64) -> String {
        Self.integerFormatter.string(from: NSNumber(value: value)) ?? "\(value)"
    }

    public static func grouped(_ value: Int64) -> String {
        Self.integerFormatter.string(from: NSNumber(value: value)) ?? "\(value)"
    }

    public static func grouped(_ value: Int) -> String {
        Self.integerFormatter.string(from: NSNumber(value: value)) ?? "\(value)"
    }

    private static let integerFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.groupingSeparator = ","
        formatter.maximumFractionDigits = 0
        return formatter
    }()

    // MARK: - Rates

    /// Adaptive, because `0.003 tok/s` is useless.
    ///
    /// - `>= 1.0` → `1.4 tok/s`
    /// - `0.2 … 1.0` → `0.4 tok/s (2.5 s/tok)`
    /// - `< 0.2` → `5:57 / token`
    public static func tokenRate(_ tokensPerSecond: Double?) -> String {
        guard let rate = tokensPerSecond, rate > 0, rate.isFinite else { return "—" }
        if rate >= 1.0 { return String(format: "%.1f tok/s", rate) }
        if rate >= 0.2 {
            return String(format: "%.1f tok/s (%.1f s/tok)", rate, 1 / rate)
        }
        return "\(clock(1 / rate)) / token"
    }

    /// Accessibility spelling of the same, in words.
    public static func tokenRateSpoken(_ tokensPerSecond: Double?) -> String {
        guard let rate = tokensPerSecond, rate > 0, rate.isFinite else {
            return "rate not yet measured"
        }
        if rate >= 1.0 { return String(format: "%.1f tokens per second", rate) }
        let seconds = 1 / rate
        if seconds < 90 { return String(format: "%.0f seconds per token", seconds) }
        return String(format: "%.1f minutes per token", seconds / 60)
    }

    /// One term of a phase decomposition, which spans four orders of magnitude
    /// in the same table: a 0.004 s reclaim beside a 3-minute expert wait.
    /// `clock` rounds the first to `0:00`, which reads as a measured zero, so a
    /// short term keeps its own resolution and a long one becomes a clock.
    public static func phaseSeconds(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "—" }
        if seconds < 1 { return String(format: "%.2f s", seconds) }
        if seconds < 90 { return String(format: "%.1f s", seconds) }
        return clock(seconds)
    }

    public static func phaseSecondsSpoken(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "not reported" }
        if seconds < 90 { return String(format: "%.1f seconds", seconds) }
        if seconds < 5400 { return String(format: "%.1f minutes", seconds / 60) }
        return String(format: "%.1f hours", seconds / 3600)
    }

    /// `m:ss` for anything under an hour, `h:mm:ss` above it.
    public static func clock(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "—" }
        let total = Int(seconds.rounded())
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let secs = total % 60
        if hours > 0 { return String(format: "%d:%02d:%02d", hours, minutes, secs) }
        return String(format: "%d:%02d", minutes, secs)
    }

    /// Human duration for a download that may run for days.
    public static func duration(_ seconds: TimeInterval?) -> String {
        guard let seconds, seconds.isFinite, seconds >= 0 else { return "—" }
        if seconds < 90 { return String(format: "%.0f s", seconds) }
        if seconds < 5400 { return String(format: "%.0f min", seconds / 60) }
        if seconds < 172_800 { return String(format: "%.1f h", seconds / 3600) }
        return String(format: "%.1f days", seconds / 86400)
    }

    public static func throughput(_ bytesPerSecond: Double?) -> String {
        guard let rate = bytesPerSecond, rate > 0, rate.isFinite else { return "—" }
        if rate >= 1e9 { return String(format: "%.2f GB/s", rate / 1e9) }
        return String(format: "%.0f MB/s", rate / 1e6)
    }

    public static func percent(_ fraction: Double, digits: Int = 1) -> String {
        String(format: "%.\(digits)f%%", fraction * 100)
    }

    // MARK: - Digests

    /// `7f791fb2…a0209` — head and tail, never a silent truncation that could
    /// pass for a whole digest.
    public static func shortDigest(_ hex: String, head: Int = 8, tail: Int = 5) -> String {
        guard hex.count > head + tail + 1 else { return hex }
        return "\(hex.prefix(head))…\(hex.suffix(tail))"
    }

    // MARK: - Dates

    public static func timestamp(_ date: Date) -> String {
        Self.timestampFormatter.string(from: date)
    }

    private static let timestampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        return formatter
    }()

    public static func relative(_ date: Date, now: Date = Date()) -> String {
        let seconds = now.timeIntervalSince(date)
        if seconds < 60 { return "just now" }
        if seconds < 3600 { return "\(Int(seconds / 60)) min ago" }
        if seconds < 86400 { return "\(Int(seconds / 3600)) h ago" }
        return "\(Int(seconds / 86400)) days ago"
    }
}
