import Foundation

/// How hard to look.
public enum VerificationDepth: String, Sendable, Codable, Equatable {
    /// Sizes only. Fast, and catches truncation, which is the common failure.
    case sizeOnly
    /// Digest every payload file; size-check the metadata. The default for a
    /// finished job: payload is what a run reads.
    case digestPayload
    /// Digest everything, metadata included.
    case digestEverything

    func digests(_ file: RepoFile) -> Bool {
        switch self {
        case .sizeOnly: return false
        case .digestPayload: return file.isPayload
        case .digestEverything: return true
        }
    }
}

/// What was found on disk, file by file.
public struct VerificationReport: Sendable, Codable, Equatable {
    public let job: DownloadJobID
    public let model: ModelID
    public let depth: VerificationDepth
    public let checkedAt: Date
    public let ok: [String]
    public let missing: [String]
    public let wrongSize: [SizeMismatch]
    public let wrongDigest: [DigestMismatch]
    /// Path → the reason it could not be read.
    public let unreadable: [String: String]
    /// On disk, not in the plan. Reported and never deleted: this manager did
    /// not put them there and does not know what they are.
    public let extraneous: [String]
    /// Bytes ``pathsToRefetch`` add up to. Stored rather than computed because
    /// only the manager holds the plan that prices it, and a report that had to
    /// be paired with a plan to answer "how much will the repair cost" would be
    /// a report nobody could hand to a UI.
    public let bytesToRefetch: UInt64
    /// What an earlier revision's record spared this pass, when it spared
    /// anything. Nil for every download-side verification and for any artifact
    /// pass that read its whole plan. See ADR 0014.
    public let carryForward: ArtifactCarryForwardSummary?

    public init(
        job: DownloadJobID, model: ModelID, depth: VerificationDepth, checkedAt: Date,
        ok: [String], missing: [String], wrongSize: [SizeMismatch],
        wrongDigest: [DigestMismatch], unreadable: [String: String], extraneous: [String],
        bytesToRefetch: UInt64 = 0,
        carryForward: ArtifactCarryForwardSummary? = nil
    ) {
        self.job = job
        self.model = model
        self.depth = depth
        self.checkedAt = checkedAt
        self.ok = ok
        self.missing = missing
        self.wrongSize = wrongSize
        self.wrongDigest = wrongDigest
        self.unreadable = unreadable
        self.extraneous = extraneous
        self.bytesToRefetch = bytesToRefetch
        self.carryForward = carryForward
    }

    /// Every planned file is present and correct. Extraneous files do not make
    /// a download incomplete — they make it untidy, which is the operator's
    /// business and not this type's.
    public var isComplete: Bool {
        missing.isEmpty && wrongSize.isEmpty && wrongDigest.isEmpty && unreadable.isEmpty
    }

    /// Exactly the paths a repair must fetch again, in plan order within each
    /// category and deduplicated across them.
    public var pathsToRefetch: [String] {
        var seen = Set<String>()
        var out: [String] = []
        for path in missing + wrongSize.map(\.path) + wrongDigest.map(\.path)
            + unreadable.keys.sorted()
        where seen.insert(path).inserted {
            out.append(path)
        }
        return out
    }

}
