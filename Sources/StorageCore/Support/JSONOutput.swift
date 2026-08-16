import Foundation

/// Shared JSON conventions for machine-readable output (spec §18.3).
///
/// Keys are snake_case and sorted. Sorting is not cosmetic: benchmark results
/// get committed, compared, and diffed across runs, and Foundation's encoder
/// otherwise emits keys in hash order, so two runs of the same command would
/// produce files that differ everywhere and nowhere. Sorted output makes a
/// diff show only the numbers that changed.
public enum JSONOutput {
    public static func encoder(pretty: Bool = true) -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting =
            pretty
            ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            : [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }

    public static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }

    public static func string<Value: Encodable>(_ value: Value, pretty: Bool = true) throws
        -> String
    {
        let data = try encoder(pretty: pretty).encode(value)
        return String(decoding: data, as: UTF8.self)
    }

    /// ISO 8601 with fractional seconds, so runs a second apart stay ordered.
    public static func timestamp(_ date: Date = Date()) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }
}

extension KeyedEncodingContainer {
    /// Encodes `value`, writing an explicit `null` when it is nil.
    ///
    /// Codable omits nil optionals by default. For a versioned result schema
    /// that is the wrong default: a consumer cannot distinguish a field absent
    /// because this machine did not report it from a field absent because the
    /// tool that wrote the file predates it. Every optional in a result
    /// document goes through here, so key presence is stable across machines
    /// and runs, and analysis code can index rather than probe.
    public mutating func encodeAlways<Value: Encodable>(
        _ value: Value?, forKey key: Key
    ) throws {
        if let value {
            try encode(value, forKey: key)
        } else {
            try encodeNil(forKey: key)
        }
    }
}

/// Names for `ProcessInfo.ThermalState`.
///
/// Spec §17.4 asks for thermal state alongside every documented result, and
/// §18.2 asks for runs long enough to expose throttling. Prior on-device work
/// (spec §4.5) found that this value lags actual heat, so treat it as a coarse
/// annotation on a run, not a measurement.
public enum ThermalState {
    public static func name(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    public static var currentName: String {
        name(ProcessInfo.processInfo.thermalState)
    }
}
