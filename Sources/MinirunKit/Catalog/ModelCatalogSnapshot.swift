import Foundation

/// A whole catalog as of one moment, and where that moment came from.
public struct ModelCatalogSnapshot: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1

    public enum Origin: String, Codable, Sendable {
        /// Read from the Hugging Face API just now.
        case live
        /// Read from this machine's cache of a previous live fetch.
        case cached
        /// Read from the copy compiled into this build.
        case bundled
    }

    public let schemaVersion: Int
    public let generatedAt: Date
    public let origin: Origin
    public let models: [ModelDescriptor]

    public init(
        schemaVersion: Int = ModelCatalogSnapshot.currentSchemaVersion,
        generatedAt: Date, origin: Origin, models: [ModelDescriptor]
    ) {
        self.schemaVersion = schemaVersion
        self.generatedAt = generatedAt
        self.origin = origin
        self.models = models
    }

    public func descriptor(_ id: ModelID) -> ModelDescriptor? {
        models.first { $0.id == id }
    }

    /// The same snapshot, relabelled. Used when a cached document is read back:
    /// the bytes say `live` because that is how they were fetched, and the
    /// caller needs to be told it is looking at a cache.
    public func labelled(_ origin: Origin) -> ModelCatalogSnapshot {
        ModelCatalogSnapshot(
            schemaVersion: schemaVersion, generatedAt: generatedAt, origin: origin, models: models)
    }

    /// Field-by-field disagreement with `other`.
    ///
    /// Reported, never resolved. A repo whose byte count or commit moved under
    /// a pinned reference is news — it means the numbers this build was
    /// designed around describe different bytes — and news that a library
    /// silently reconciles is news nobody hears.
    public func divergence(from other: ModelCatalogSnapshot) -> [CatalogDivergence] {
        var out: [CatalogDivergence] = []
        for live in models {
            guard let bundled = other.descriptor(live.id) else {
                out.append(
                    CatalogDivergence(
                        model: live.id, field: "presence", bundled: "absent", live: "present"))
                continue
            }
            func compare(_ field: String, _ a: String, _ b: String) {
                if a != b {
                    out.append(CatalogDivergence(model: live.id, field: field, bundled: a, live: b))
                }
            }
            compare("payloadBytes", "\(bundled.payloadBytes)", "\(live.payloadBytes)")
            compare("metadataBytes", "\(bundled.metadataBytes)", "\(live.metadataBytes)")
            compare("payloadFileCount", "\(bundled.payloadFileCount)", "\(live.payloadFileCount)")
            compare(
                "metadataFileCount", "\(bundled.metadataFileCount)", "\(live.metadataFileCount)")
            compare("largestFileBytes", "\(bundled.largestFileBytes)", "\(live.largestFileBytes)")
            compare("layout", bundled.layout.rawValue, live.layout.rawValue)
            if let a = bundled.source.repo, let b = live.source.repo {
                compare("revision", a.revision, b.revision)
                compare("repoID", a.repoID, b.repoID)
            }
        }
        for bundled in other.models where descriptor(bundled.id) == nil {
            out.append(
                CatalogDivergence(
                    model: bundled.id, field: "presence", bundled: "present", live: "absent"))
        }
        return out
    }
}

/// One field, two answers.
public struct CatalogDivergence: Codable, Sendable, Equatable, Identifiable {
    public let model: ModelID
    public let field: String
    public let bundled: String
    public let live: String

    /// Model and field together, because a divergence list holds one row per
    /// field per model and the field alone repeats across models.
    public var id: String { "\(model.rawValue).\(field)" }

    public init(model: ModelID, field: String, bundled: String, live: String) {
        self.model = model
        self.field = field
        self.bundled = bundled
        self.live = live
    }
}

public enum RefreshPolicy: Sendable, Equatable {
    /// Ask the network, and fall back only if it fails.
    case forceLive
    /// Use the cache when it is younger than this many seconds.
    case preferCachedWithin(TimeInterval)
    /// Never touch the network or the cache.
    case bundledOnly
}

/// Where a fetched snapshot is kept between launches.
public protocol CatalogCache: Sendable {
    func load() throws -> ModelCatalogSnapshot?
    func save(_ snapshot: ModelCatalogSnapshot) throws
}

/// A cache that is one JSON file.
public struct FileCatalogCache: CatalogCache {
    public let url: URL

    public init(url: URL) { self.url = url }

    public static var `default`: FileCatalogCache {
        let base =
            FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return FileCatalogCache(
            url:
                base
                .appendingPathComponent("minirun", isDirectory: true)
                .appendingPathComponent("model-catalog.json"))
    }

    public func load() throws -> ModelCatalogSnapshot? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let data = try Data(contentsOf: url)
        return try MinirunKitJSON.decoder().decode(ModelCatalogSnapshot.self, from: data)
    }

    public func save(_ snapshot: ModelCatalogSnapshot) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = try MinirunKitJSON.encoder().encode(snapshot)
        try data.write(to: url, options: .atomic)
    }
}

public enum CatalogError: Error, Equatable, CustomStringConvertible {
    case transport(status: Int, url: String)
    case malformedResponse(String)
    /// A revision that is not a 40-hex commit. Refused rather than resolved,
    /// because a branch name lets the bytes move under a running transfer.
    case unpinnedRevision(repoID: String, given: String)
    case unknownModel(ModelID)
    /// A `*-minirun` repo this build has no layout for. Surfaced, not guessed.
    case unrecognizedLayout(repoID: String)
    /// The model has no repository to plan against — it is built by a local
    /// tool instead.
    case notDownloadable(ModelID, builtBy: String)

    public var description: String {
        switch self {
        case .transport(let status, let url):
            return "HTTP \(status) from \(url)"
        case .malformedResponse(let what):
            return "malformed response: \(what)"
        case .unpinnedRevision(let repoID, let given):
            return "\(repoID) is pinned to '\(given)', which is not a 40-hex commit"
        case .unknownModel(let id):
            return "no model '\(id)' in this catalog"
        case .unrecognizedLayout(let repoID):
            return "\(repoID) uses a layout this build does not know how to lay out"
        case .notDownloadable(let id, let tool):
            return "\(id) has no published repository; it is built by \(tool)"
        }
    }
}
