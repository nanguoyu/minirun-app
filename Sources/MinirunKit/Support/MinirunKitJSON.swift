import Foundation

/// JSON conventions for MinirunKit's own documents.
///
/// Deliberately *not* `StorageCore.JSONOutput`. That encoder converts keys to
/// snake_case, and the round trip is lossy for any property whose name holds a
/// run of capitals: `repoID` encodes to `repo_id` and decodes back to `repoId`,
/// which no longer matches the coding key, so the document fails to read. The
/// benchmark result schema (spec §18.3) is snake_case by requirement and stays
/// where it is; these documents — the catalog snapshot, the download job state
/// — are this package's own and use its property names verbatim.
///
/// Keys are sorted for the same reason `JSONOutput` sorts them: a cached
/// snapshot that gets diffed should show the numbers that changed and nothing
/// else. Dates are ISO 8601 so a cache file stays readable by a human deciding
/// whether it is stale.
public enum MinirunKitJSON {
    public static func encoder(pretty: Bool = true) -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting =
            pretty
            ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            : [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }

    public static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
