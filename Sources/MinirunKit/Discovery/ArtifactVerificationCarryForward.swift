import Foundation

/// What a pass did not have to read again, and where that permission came from.
///
/// Attached to ``VerificationReport`` whenever a pass spent evidence recorded
/// against an earlier published revision of the same repository (ADR 0014). It
/// is reporting, not authority: nothing consumes it to decide anything, and a
/// record written by a carried pass is structurally identical to one written by
/// a pass that read every byte.
public struct ArtifactCarryForwardSummary: Sendable, Codable, Equatable {
    /// The pinned commit whose record supplied the carried evidence.
    public let sourceRevision: String
    /// How much that record covered. A spot-checked source can only ever
    /// contribute the payloads it sampled, and makes no claim about the
    /// complete tree — which is why the two tree-shaped counts below are zero
    /// for one.
    public let sourceState: ArtifactVerification
    /// Files whose bytes were not read, because the same published digest was
    /// already digested from the same unmoved object.
    public let carriedForward: Int
    /// Files this pass digested.
    public let reread: Int
    /// Of the files this pass planned, those the source revision did not
    /// publish at all. Zero for a spot-checked source.
    public let newFiles: Int
    /// Files the source revision published and this one does not. Zero for a
    /// spot-checked source.
    public let removedFiles: Int

    public init(
        sourceRevision: String, sourceState: ArtifactVerification,
        carriedForward: Int, reread: Int, newFiles: Int, removedFiles: Int
    ) {
        self.sourceRevision = sourceRevision
        self.sourceState = sourceState
        self.carriedForward = carriedForward
        self.reread = reread
        self.newFiles = newFiles
        self.removedFiles = removedFiles
    }

    /// The operator-facing clause, or nil when nothing was carried. Kept beside
    /// the counts so the row, the notification and the stored sentence cannot
    /// drift into three spellings of one fact.
    public var sentence: String? {
        guard carriedForward > 0 else { return nil }
        var clause =
            "\(reread) file\(reread == 1 ? "" : "s") re-read, "
            + "\(carriedForward) carried forward from the previous revision"
        if removedFiles > 0 {
            clause += ", \(removedFiles) no longer published"
        }
        return clause + "."
    }
}

/// Which of a plan's files an earlier revision's record already answers.
///
/// The comparison is between two pinned commits' published digest tables. A
/// file whose published `(size, digest, payload flag)` is identical at both is
/// the same content by definition, so an entry that was digested against it
/// once — from an object whose durable identity has not moved since — needs no
/// second reading. Every one of those identities is still checked, by the
/// verifier, against the descriptor it opens beneath the rooted artifact fd.
struct ArtifactCarryForwardPlan {
    let sourceRepository: HuggingFaceRepoRef
    let sourceState: ArtifactVerification
    /// Path → the recorded evidence eligible to be spent for it.
    let offered: [String: ArtifactVerifiedFile]
    let newFileCount: Int
    let removedFileCount: Int

    /// Nil when nothing may be carried, which is every case that is not
    /// provably safe: a record for another model or repository, a record whose
    /// evidence predates the current schema or is not structurally valid, a
    /// record without durable volume binding, a record about a different
    /// artifact directory object, or a tree with no unchanged entry in common.
    static func make(
        source record: ArtifactVerificationRecord,
        model: ModelID,
        repository: HuggingFaceRepoRef,
        published: [RepoFile],
        selected: [RepoFile],
        persistentVolume: ArtifactPersistentVolumeIdentity,
        rootIdentity: ArtifactFilesystemIdentity
    ) -> ArtifactCarryForwardPlan? {
        guard record.state != .unverified,
            let evidence = record.evidence,
            evidence.isStructurallyValid(for: record.state),
            evidence.model == model,
            // Only the revision may differ. Another repository publishing the
            // same digests is not this repository.
            evidence.repository.repoID == repository.repoID,
            // Durable volume binding is required rather than tolerated. The
            // legacy exact-`dev` semantics of a record without it cannot
            // survive the remount this feature is otherwise indifferent to,
            // and a carried file is not the place to reason about that.
            let volume = evidence.persistentVolume, volume == persistentVolume,
            // Not the whole root identity: rewriting a file in the artifact
            // root moves the directory's own mtime/ctime, and a rewritten
            // README.md in the root is the exact case this exists for. The
            // object must still be the same object.
            evidence.root.inode == rootIdentity.inode
        else { return nil }

        let recorded = Dictionary(
            evidence.files.map { ($0.path, $0) }, uniquingKeysWith: { first, _ in first })
        var offered: [String: ArtifactVerifiedFile] = [:]
        for file in selected {
            guard let stored = recorded[file.path],
                stored.expectedSizeBytes == file.sizeBytes,
                stored.digest == file.digest,
                stored.isPayload == file.isPayload
            else { continue }
            offered[file.path] = stored
        }
        guard !offered.isEmpty else { return nil }

        // A spot check covers a chosen subset of payloads and no metadata, so
        // "new" and "removed" cannot be derived from it: everything it did not
        // sample would count as new, which would be a number that means
        // nothing. Only a full record describes a complete tree.
        let describesCompleteTree = record.state == .fullyVerified
        let publishedPaths = Set(published.map(\.path))
        return ArtifactCarryForwardPlan(
            sourceRepository: evidence.repository,
            sourceState: record.state,
            offered: offered,
            newFileCount: describesCompleteTree
                ? selected.filter { recorded[$0.path] == nil }.count : 0,
            removedFileCount: describesCompleteTree
                ? recorded.keys.filter { !publishedPaths.contains($0) }.count : 0)
    }

    func summary(carriedForward: Int, reread: Int) -> ArtifactCarryForwardSummary? {
        // A carry-forward that carried nothing — every offered object had
        // moved — is not news, and reporting it would put a source revision on
        // a report that spent none of it.
        guard carriedForward > 0 else { return nil }
        return ArtifactCarryForwardSummary(
            sourceRevision: sourceRepository.revision, sourceState: sourceState,
            carriedForward: carriedForward, reread: reread,
            newFiles: newFileCount, removedFiles: removedFileCount)
    }
}
