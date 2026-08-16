import CoreFoundation
import Foundation

/// One repository an artifact's `index.json` names as a source.
///
/// The revision is optional because only some shapes state one, and the role is
/// optional because only H3 has more than one upstream to distinguish. Neither
/// is invented when the document is silent: a provenance that was filled in by
/// a defaulting rule is a provenance nobody published.
public struct ArtifactSourceRepo: Codable, Sendable, Equatable, Hashable {
    public let repoID: String
    public let revision: String?
    /// H3's `sources.<name>.role`, verbatim. Nil for the single-source shapes.
    public let role: String?

    public init(repoID: String, revision: String? = nil, role: String? = nil) {
        self.repoID = repoID
        self.revision = revision
        self.role = role
    }
}

/// The tokenizer an artifact publishes beside its weights.
///
/// This is execution evidence, so a partially stated object is not exposed.
/// The file must be a canonical repository-relative root file, the source must
/// name an immutable commit, and the digest must be a complete lowercaseable
/// SHA-256. A future publisher may add unrelated tokenizer fields without
/// changing this value; a malformed required field makes the whole identity
/// absent rather than guessed.
public struct ArtifactTokenizerIdentity: Codable, Sendable, Equatable, Hashable {
    public let file: String
    public let bytes: UInt64
    public let sha256: String
    public let sourceRepo: String
    public let sourceRevision: String
    public let license: String

    public init(
        file: String, bytes: UInt64, sha256: String, sourceRepo: String,
        sourceRevision: String, license: String
    ) {
        self.file = file
        self.bytes = bytes
        self.sha256 = sha256.lowercased()
        self.sourceRepo = sourceRepo
        self.sourceRevision = sourceRevision
        self.license = license
    }

    fileprivate static func parse(_ value: Any?) -> ArtifactTokenizerIdentity? {
        guard let object = value as? [String: Any],
            let file = canonicalRootFile(object["file"]),
            let bytes = unsignedInteger(object["bytes"]), bytes > 0,
            let sha256 = object["sha256"] as? String,
            sha256.count == 64,
            sha256.utf8.allSatisfy({
                (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0)
            }),
            let sourceRepo = object["source_repo"] as? String,
            HuggingFaceRepoRef.isCanonicalRepositoryID(sourceRepo),
            let sourceRevision = object["source_revision"] as? String,
            sourceRevision.count == 40,
            sourceRevision.utf8.allSatisfy({
                (48...57).contains($0) || (97...102).contains($0)
            }),
            let license = canonicalRootFile(object["license"])
        else { return nil }
        return ArtifactTokenizerIdentity(
            file: file, bytes: bytes, sha256: sha256, sourceRepo: sourceRepo,
            sourceRevision: sourceRevision, license: license)
    }

    private static func canonicalRootFile(_ value: Any?) -> String? {
        guard let value = value as? String, !value.isEmpty,
            !value.contains("/"), value != ".", value != "..",
            value.utf8.allSatisfy({
                (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0)
                    || $0 == 45 || $0 == 46 || $0 == 95
            })
        else { return nil }
        return value
    }

    private static func unsignedInteger(_ value: Any?) -> UInt64? {
        guard let number = value as? NSNumber,
            CFGetTypeID(number) != CFBooleanGetTypeID()
        else { return nil }
        return UInt64(number.stringValue)
    }
}

/// The model configuration published beside an artifact's weights.
///
/// Like tokenizer identity, this is execution evidence rather than display
/// metadata. A product runner must open this exact root file through the
/// artifact's full-verification authority; a repository name or a compatible
/// looking JSON document is not a substitute for the recorded bytes.
public struct ArtifactConfigurationIdentity: Codable, Sendable, Equatable, Hashable {
    public let file: String
    public let bytes: UInt64
    public let sha256: String
    public let sourceRepo: String
    public let sourceRevision: String

    public init(
        file: String, bytes: UInt64, sha256: String, sourceRepo: String,
        sourceRevision: String
    ) {
        self.file = file
        self.bytes = bytes
        self.sha256 = sha256.lowercased()
        self.sourceRepo = sourceRepo
        self.sourceRevision = sourceRevision
    }

    fileprivate static func parse(_ value: Any?) -> ArtifactConfigurationIdentity? {
        guard let object = value as? [String: Any],
            let file = canonicalRootFile(object["file"]),
            let bytes = unsignedInteger(object["bytes"]), bytes > 0,
            let sha256 = object["sha256"] as? String,
            sha256.count == 64,
            sha256.utf8.allSatisfy({
                (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0)
            }),
            let sourceRepo = object["source_repo"] as? String,
            HuggingFaceRepoRef.isCanonicalRepositoryID(sourceRepo),
            let sourceRevision = object["source_revision"] as? String,
            sourceRevision.count == 40,
            sourceRevision.utf8.allSatisfy({
                (48...57).contains($0) || (97...102).contains($0)
            })
        else { return nil }
        return ArtifactConfigurationIdentity(
            file: file, bytes: bytes, sha256: sha256, sourceRepo: sourceRepo,
            sourceRevision: sourceRevision)
    }

    private static func canonicalRootFile(_ value: Any?) -> String? {
        guard let value = value as? String, !value.isEmpty,
            !value.contains("/"), value != ".", value != "..",
            value.utf8.allSatisfy({
                (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0)
                    || $0 == 45 || $0 == 46 || $0 == 95
            })
        else { return nil }
        return value
    }

    private static func unsignedInteger(_ value: Any?) -> UInt64? {
        guard let number = value as? NSNumber,
            CFGetTypeID(number) != CFBooleanGetTypeID()
        else { return nil }
        return UInt64(number.stringValue)
    }
}

/// Which of the published `index.json` documents this one looks like.
///
/// Three shapes exist across the three artifacts on record and there is no
/// schema to appeal to, so the reader recognises each by what it carries rather
/// than by a version field none of them has. A fourth shape is not a crash — it
/// is ``unrecognized``, which the UI shows as an unidentified artifact.
public enum ArtifactIndexShape: String, Codable, Sendable, CaseIterable {
    /// K3: `source_repo`, `source_revision`, `files`, `bytes` at the top level.
    case singleSourceTotals
    /// V4-Flash: `source_repo`, `source_revision`, `units[]`, `total_bytes`.
    case singleSourceUnits
    /// H3: a `sources` map of several upstreams, `units[]`, `total_bytes`.
    case multiSourceUnits
    /// A document that parsed as JSON and named no repository this reader
    /// understands. Reported, never guessed at.
    case unrecognized
}

/// What an artifact's own `index.json` says about itself.
///
/// Parsed with `JSONSerialization` rather than `Codable` on purpose: the three
/// documents disagree about nearly every key, and a decoder written against one
/// of them throws on the other two. Nothing here throws — an `index.json` that
/// cannot be read is an artifact whose identity is unknown, which is a state the
/// UI can show, unlike an exception during a directory walk.
public struct ArtifactIndexIdentity: Codable, Sendable, Equatable {
    public let shape: ArtifactIndexShape
    /// Every repository the document names, in document order.
    public let repositories: [ArtifactSourceRepo]
    /// K3's `format`, where one is stated.
    public let format: String?
    /// The publisher's own sentence about what the repack did.
    public let relationship: String?
    /// Payload files the index claims, from `files` or summed over `units[]`.
    public let declaredFileCount: Int?
    /// Payload bytes the index claims, from `bytes` or `total_bytes`.
    public let declaredBytes: UInt64?
    /// A complete tokenizer provenance object, or nil when the artifact does
    /// not publish one or any required field is malformed.
    public let tokenizer: ArtifactTokenizerIdentity?
    /// A complete model-configuration provenance object, under the same
    /// fail-closed parsing rule as the tokenizer.
    public let configuration: ArtifactConfigurationIdentity?
    /// The top-level keys, sorted. Shown when the shape is ``ArtifactIndexShape/unrecognized``
    /// so that an operator looking at a strange artifact sees what the document
    /// actually contained instead of "unknown".
    public let topLevelKeys: [String]

    public init(
        shape: ArtifactIndexShape, repositories: [ArtifactSourceRepo], format: String?,
        relationship: String?, declaredFileCount: Int?, declaredBytes: UInt64?,
        tokenizer: ArtifactTokenizerIdentity? = nil,
        configuration: ArtifactConfigurationIdentity? = nil,
        topLevelKeys: [String]
    ) {
        self.shape = shape
        self.repositories = repositories
        self.format = format
        self.relationship = relationship
        self.declaredFileCount = declaredFileCount
        self.declaredBytes = declaredBytes
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.topLevelKeys = topLevelKeys
    }

    /// The identity of a document that is not JSON at all, or is JSON that is
    /// not an object. Kept as a value rather than an optional so that a caller
    /// never has to branch on nil before it can show a row.
    public static let unreadable = ArtifactIndexIdentity(
        shape: .unrecognized, repositories: [], format: nil, relationship: nil,
        declaredFileCount: nil, declaredBytes: nil, tokenizer: nil, configuration: nil,
        topLevelKeys: [])

    /// Reads whatever this document is willing to say. Never throws.
    public static func parse(_ data: Data) -> ArtifactIndexIdentity {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .unreadable
        }
        let keys = root.keys.sorted()

        var repositories: [ArtifactSourceRepo] = []
        var shape: ArtifactIndexShape = .unrecognized

        if let repoID = root["source_repo"] as? String, !repoID.isEmpty {
            repositories = [
                ArtifactSourceRepo(
                    repoID: repoID, revision: root["source_revision"] as? String, role: nil)
            ]
            shape = root["units"] is [Any] ? .singleSourceUnits : .singleSourceTotals
        } else if let sources = root["sources"] as? [String: Any] {
            // The map's own key order is not stable, so the roles are sorted by
            // their key. H3 lists dit, text_encoder, official, lora, tae; a list
            // whose order changed between two scans of the same directory would
            // read as the artifact having changed.
            repositories =
                sources.keys.sorted().compactMap { name -> ArtifactSourceRepo? in
                    guard let entry = sources[name] as? [String: Any],
                        let repoID = entry["repo"] as? String, !repoID.isEmpty
                    else { return nil }
                    return ArtifactSourceRepo(
                        repoID: repoID, revision: entry["revision"] as? String,
                        role: entry["role"] as? String ?? name)
                }
            if !repositories.isEmpty { shape = .multiSourceUnits }
        }

        let claim = IndexClaim.parse(data)

        return ArtifactIndexIdentity(
            shape: shape,
            repositories: repositories,
            format: root["format"] as? String,
            relationship: root["relationship"] as? String,
            declaredFileCount: claim?.declaredFileCount,
            declaredBytes: claim?.declaredBytes,
            tokenizer: ArtifactTokenizerIdentity.parse(root["tokenizer"]),
            configuration: ArtifactConfigurationIdentity.parse(root["configuration"]),
            topLevelKeys: keys)
    }
}

/// Turns the repositories an `index.json` names into a catalog model.
///
/// Two tables and not one, because the two names are different facts. The
/// catalog knows a model by the `*-minirun` repository that *publishes* the
/// repack; the artifact's own index names the repository the bytes were
/// repacked *from*. A directory fetched from `nanguoyu/Kimi-K3-minirun` carries
/// an index that says `moonshotai/Kimi-K3` and nothing else, so matching on the
/// catalog's repo id alone identifies none of the three artifacts that actually
/// exist on the drive.
///
/// The upstream table is written down here rather than derived, because it
/// cannot be derived: nothing in the published bytes states which repack repo
/// consumed them. It is short, it is checked by a test against the three real
/// documents, and an upstream it does not know produces an unidentified
/// artifact rather than a wrong one.
public struct ArtifactIdentityMatcher: Sendable {
    /// Upstream repository id (lowercased) → the model it belongs to.
    public static let upstreamRepositories: [String: ModelID] = [
        "moonshotai/kimi-k3": .kimiK3,
        "deepseek-ai/deepseek-v4-flash-0731": .deepseekV4Flash,
        // H3 is assembled from five upstreams. Any one of them identifies it;
        // the DiT repo is the one that carries the weights a run reads.
        "pipenetwork/minimax-h3-mlx-8bit": .minimaxH3,
        "minimaxai/minimax-h3": .minimaxH3,
        "ddalcu/minimax-h3-fl2va-mlx-serve-8bit": .minimaxH3,
        "lightx2v/minimax-h3-turbo": .minimaxH3,
        "kijai/minimax-h3-tae": .minimaxH3,
    ]

    private let publishedRepositories: [String: ModelID]

    public init(catalog: ModelCatalogSnapshot) {
        var published: [String: ModelID] = [:]
        for model in catalog.models {
            if let repo = model.source.repo {
                published[repo.repoID.lowercased()] = model.id
            }
        }
        self.publishedRepositories = published
    }

    /// The model this index belongs to, or nil when nothing in it is recognised.
    ///
    /// The publishing repo is checked first: if an index ever does name the
    /// repack repo directly, that is a stronger statement than an upstream a
    /// second model might one day share.
    public func match(_ identity: ArtifactIndexIdentity) -> ModelID? {
        for repo in identity.repositories {
            if let model = publishedRepositories[repo.repoID.lowercased()] { return model }
        }
        for repo in identity.repositories {
            if let model = Self.upstreamRepositories[repo.repoID.lowercased()] { return model }
        }
        return nil
    }
}
