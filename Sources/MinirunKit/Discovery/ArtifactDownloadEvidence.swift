import Foundation

/// Why a finished transfer's own verification could not be turned into artifact
/// verification evidence.
///
/// Every case here is a *refusal*, not a downgrade. The ledger holds one kind of
/// positive record and it is the kind ``ArtifactVerifier`` writes; a hand-off
/// that cannot produce exactly that record writes nothing at all and says which
/// fact was missing. See ADR 0019.
public enum ArtifactDownloadEvidenceError: Error, Equatable, CustomStringConvertible {
    /// The closing verification did not pass, so there is nothing to hand over.
    case transferDidNotPass(model: ModelID)
    /// The summary and the plan disagree about which job, model, repository or
    /// destination they describe.
    case summaryDoesNotDescribePlan(reason: String)
    /// The transfer never read these files' bytes against a published digest.
    /// A size check is not a digest and is never recorded as one.
    case bytesNeverDigested(paths: [String])
    /// The plan is not the repository's complete published tree, so a record
    /// built from it could not honestly claim every published file.
    case planIsNotTheCompleteTree(reason: String)
    /// The catalog no longer pins the revision this transfer fetched.
    case repositoryMoved(planned: String, cataloged: String)
    /// The object on disk is not the object the closing pass checked.
    case identityMoved(path: String)
    case noPublishedDigests(ModelID)

    public var description: String {
        switch self {
        case .transferDidNotPass(let model):
            return
                "\(model)'s transfer did not pass its own verification, so it hands over nothing"
        case .summaryDoesNotDescribePlan(let reason):
            return "the finished transfer does not describe this plan: \(reason)"
        case .bytesNeverDigested(let paths):
            let first = paths.first ?? ""
            return
                "the transfer never digested \(paths.count) published file(s), the first being '\(first)'; "
                + "a size check is not a digest and is not recorded as one"
        case .planIsNotTheCompleteTree(let reason):
            return
                "the transfer's plan is not the repository's complete published tree: \(reason)"
        case .repositoryMoved(let planned, let cataloged):
            return
                "the transfer fetched \(planned) and the catalog now pins \(cataloged); "
                + "evidence is recorded against the revision that was read"
        case .identityMoved(let path):
            return "'\(path)' is no longer the object the transfer's closing pass checked"
        case .noPublishedDigests(let model):
            return "\(model) is built locally; no published digests exist to record"
        }
    }
}

/// What a finished transfer proved about the bytes it left on disk.
///
/// This is the download side of the hand-off in ADR 0019. It carries one entry
/// per published file, and an entry exists only where **this job read that
/// file's bytes and matched them against the digest the repository publishes** —
/// as the file landed, or in the closing pass, or both. The filesystem identity
/// on each entry is the one the closing pass observed with the descriptor it
/// had open; the recorder stats every object again and refuses if anything has
/// moved since.
///
/// A file the transfer only size-checked has no entry, and the recorder refuses
/// by name rather than writing a record that claims more than was read.
public struct DownloadVerificationProof: Sendable, Equatable {
    public let model: ModelID
    public let repository: HuggingFaceRepoRef
    /// The artifact directory the transfer wrote into.
    public let rootPath: String
    /// One entry per published file whose bytes were digested and matched.
    public let files: [ArtifactVerifiedFile]
    /// When the closing pass finished. The ledger records this rather than the
    /// moment the hand-off happened: it is when the bytes were last checked.
    public let checkedAt: Date

    public init(
        model: ModelID, repository: HuggingFaceRepoRef, rootPath: String,
        files: [ArtifactVerifiedFile], checkedAt: Date
    ) {
        self.model = model
        self.repository = repository
        self.rootPath = rootPath
        self.files = files.sorted { $0.path < $1.path }
        self.checkedAt = checkedAt
    }

    /// Read a finished job's own summary as a proof, or say why it is not one.
    ///
    /// The plan is required beside the summary because the summary states what
    /// happened and the plan states what was supposed to happen; a proof that
    /// took both from the same value could not detect the two disagreeing.
    public init(plan: DownloadPlan, summary: DownloadSummary) throws {
        guard summary.model == plan.model else {
            throw ArtifactDownloadEvidenceError.summaryDoesNotDescribePlan(
                reason: "the summary is about \(summary.model) and the plan about \(plan.model)")
        }
        guard summary.repo == plan.repo else {
            throw ArtifactDownloadEvidenceError.summaryDoesNotDescribePlan(
                reason:
                    "the summary is about \(summary.repo.repoID)@\(summary.repo.revision) and the plan about \(plan.repo.repoID)@\(plan.repo.revision)")
        }
        guard summary.verification.model == plan.model, summary.verification.job == summary.job
        else {
            throw ArtifactDownloadEvidenceError.summaryDoesNotDescribePlan(
                reason: "the closing report belongs to another job or model")
        }
        guard summary.verification.isComplete else {
            throw ArtifactDownloadEvidenceError.transferDidNotPass(model: plan.model)
        }
        let evidenced = Dictionary(
            summary.filesVerified.map { ($0.path, $0) }, uniquingKeysWith: { first, _ in first })
        var files: [ArtifactVerifiedFile] = []
        var undigested: [String] = []
        for file in plan.files.sorted(by: { $0.path < $1.path }) {
            guard let entry = evidenced[file.path],
                entry.expectedSizeBytes == file.sizeBytes,
                entry.digest == file.digest,
                entry.isPayload == file.isPayload,
                entry.filesystem.sizeBytes == file.sizeBytes
            else {
                undigested.append(file.path)
                continue
            }
            files.append(entry)
        }
        guard undigested.isEmpty else {
            throw ArtifactDownloadEvidenceError.bytesNeverDigested(paths: undigested)
        }
        self.init(
            model: plan.model, repository: plan.repo,
            rootPath: summary.destinationPath, files: files,
            checkedAt: summary.verification.checkedAt)
    }

    /// The sentence the ledger files and the model page shows. It says what was
    /// read and by whom, because "verified" with no provenance is the thing this
    /// hand-off is most likely to be accused of inventing.
    var ledgerSentence: String {
        "every published file matched the digests \(model) publishes, "
            + "read by the transfer that fetched them."
    }
}
