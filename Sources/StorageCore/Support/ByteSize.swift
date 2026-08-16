import Foundation

/// Parsing and formatting of byte counts on the command line.
///
/// Benchmark arguments are byte counts, and byte counts written in full are
/// easy to get wrong by a factor of ten. Suffixes are accepted so that a
/// mistyped size is visible; the canonical form in JSON output stays a plain
/// integer number of bytes.
public enum ByteSize {
    /// Accepts a plain byte count (`268435456`), a binary suffix (`256MiB`,
    /// `256M`, `256m`), or a decimal suffix (`256MB`). Underscores and internal
    /// spaces are ignored so `1_048_576` and `256 MiB` both parse.
    public static func parse(_ text: String) throws -> UInt64 {
        let normalized = text
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: " ", with: "")
            .trimmingCharacters(in: .whitespaces)
        guard !normalized.isEmpty else {
            throw StorageCoreError.invalidArgument("empty byte count")
        }

        let digits = normalized.prefix { $0.isNumber }
        let suffix = normalized.dropFirst(digits.count).lowercased()
        guard !digits.isEmpty, let value = UInt64(digits) else {
            throw StorageCoreError.invalidArgument("'\(text)' is not a byte count")
        }

        let multiplier: UInt64
        switch suffix {
        case "", "b": multiplier = 1
        case "k", "kib": multiplier = 1 << 10
        case "m", "mib": multiplier = 1 << 20
        case "g", "gib": multiplier = 1 << 30
        case "t", "tib": multiplier = 1 << 40
        case "kb": multiplier = 1_000
        case "mb": multiplier = 1_000_000
        case "gb": multiplier = 1_000_000_000
        case "tb": multiplier = 1_000_000_000_000
        default:
            throw StorageCoreError.invalidArgument(
                "unknown size suffix '\(suffix)' in '\(text)' "
                    + "(use B, KiB/MiB/GiB/TiB, or KB/MB/GB/TB)")
        }

        let (product, overflow) = value.multipliedReportingOverflow(by: multiplier)
        guard !overflow else {
            throw StorageCoreError.invalidArgument("byte count '\(text)' overflows 64 bits")
        }
        return product
    }

    /// Human-readable binary size, e.g. `256.0 MiB`. Reporting only; JSON keeps raw bytes.
    public static func format(_ bytes: UInt64) -> String {
        let units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
        var value = Double(bytes)
        var unit = 0
        while value >= 1024, unit < units.count - 1 {
            value /= 1024
            unit += 1
        }
        return unit == 0
            ? "\(bytes) B"
            : String(format: "%.1f %@", value, units[unit])
    }
}
