import Darwin
import Foundation

/// How much of an artifact to digest, and what the operator asked for.
public enum ArtifactVerificationRequest: Sendable, Equatable {
    /// Digest a bounded sample. Payloads are considered largest-first, but an
    /// oversized entry is skipped so the sum never exceeds the byte budget.
    /// Catches the failure that actually happens — a truncated or
    /// half-written tile — for the price of a few gigabytes of reading rather
    /// than the price of the whole artifact.
    case spotCheck(maximumBytes: UInt64 = 2 << 30)
    /// Digest every published file, including manifests and root metadata. On
    /// K3 this reads about 1.56 TB. Never started
    /// except by an explicit act.
    case full

    var resultingState: ArtifactVerification {
        switch self {
        case .spotCheck: return .spotChecked
        case .full: return .fullyVerified
        }
    }

    /// Which checkpoint slot this request's progress belongs to.
    public var kind: ArtifactVerificationRequestKind {
        switch self {
        case .spotCheck: return .spotCheck
        case .full: return .full
        }
    }
}

/// Whether a pass may spend the progress an interrupted pass recorded.
///
/// The default resumes, because the reason a checkpoint exists is that a
/// 1.56 TB pass is half an hour of reading and a phone will suspend the app
/// long before it ends. `fromScratch` is the operator saying "read it all
/// again", and it discards the checkpoint before the first descriptor opens.
public enum ArtifactVerificationResumption: Sendable, Equatable {
    case resumeIfPossible
    case fromScratch
}

/// Progress of a verification pass, so a 1.5 TB check is not a spinner.
public struct ArtifactVerificationProgress: Sendable, Equatable {
    public let filesChecked: Int
    public let filesToCheck: Int
    public let bytesChecked: UInt64
    public let bytesToCheck: UInt64
    public let currentPath: String

    public init(
        filesChecked: Int, filesToCheck: Int, bytesChecked: UInt64, bytesToCheck: UInt64,
        currentPath: String
    ) {
        self.filesChecked = filesChecked
        self.filesToCheck = filesToCheck
        self.bytesChecked = bytesChecked
        self.bytesToCheck = bytesToCheck
        self.currentPath = currentPath
    }

    public var fraction: Double {
        bytesToCheck == 0 ? 1 : min(1, Double(bytesChecked) / Double(bytesToCheck))
    }
}

public enum ArtifactVerificationError: Error, Equatable, CustomStringConvertible {
    /// The artifact's `index.json` named nothing this build recognises, so
    /// there is no repository to get a digest table from.
    case unidentifiedArtifact(rootPath: String)
    /// The model is built locally and has no published tree to check against.
    case noPublishedDigests(ModelID)
    /// Untrusted repository sizes could not be represented without wrapping.
    case arithmeticOverflow(context: String)
    /// The tree cannot be the complete tree promised by the descriptor.
    case publishedTreeMismatch(model: ModelID, reason: String)
    case emptyPublishedTree
    /// No positive-size payload fits within the requested hard ceiling.
    case budgetTooSmall(maximumBytes: UInt64)
    /// One verifier serialises passes so detached hashes never overlap.
    case verificationAlreadyInProgress
    /// Durable dev/inode evidence is unsafe on a volume without persistent IDs.
    case persistentFileIDsUnavailable(path: String)
    /// The object checked before hashing was not the object present afterwards.
    case filesystemChanged(path: String, reason: String)
    /// The verifier could not establish filesystem authority for the object.
    case filesystemUnavailable(path: String, reason: String)
    case cancelled

    public var description: String {
        switch self {
        case .unidentifiedArtifact(let path):
            return
                "'\(path)' does not name a model this build knows, so there is no digest table to check it against"
        case .noPublishedDigests(let model):
            return "\(model) is built locally; no published digests exist to check these bytes against"
        case .arithmeticOverflow(let context):
            return "verification refused because \(context) exceeds UInt64"
        case .publishedTreeMismatch(let model, let reason):
            return "verification refused because the published tree for \(model) is incomplete: \(reason)"
        case .emptyPublishedTree:
            return "verification refused because the published tree or its payload set is empty"
        case .budgetTooSmall(let maximumBytes):
            return "verification refused because no positive-size payload fits the \(maximumBytes)-byte budget"
        case .verificationAlreadyInProgress:
            return "verification refused because this verifier is already running a pass"
        case .persistentFileIDsUnavailable(let path):
            return "verification refused because the volume containing '\(path)' does not report persistent file identifiers"
        case .filesystemChanged(let path, let reason):
            return "verification refused because '\(path)' changed: \(reason)"
        case .filesystemUnavailable(let path, let reason):
            return "verification could not establish the identity of '\(path)': \(reason)"
        case .cancelled:
            return "cancelled"
        }
    }
}

/// A test seam at the same cancellation checkpoint the streaming digester
/// polls between chunks. It is internal, absent in production construction,
/// and exists so mutation/cancellation tests do not depend on scheduler luck.
struct ArtifactVerifierHooks: Sendable {
    var digestCheckpoint: @Sendable (String) -> Void
    var digestStarted: @Sendable (String) -> Void
    var digestFinished: @Sendable (String) -> Void
    var volumeSupportsPersistentIDs: @Sendable (URL) throws -> Bool?
    var descriptorSupportsPersistentIDs: @Sendable (Int32, String) throws -> Bool?
    var descriptorPersistentVolumeIdentity:
        @Sendable (Int32, String) throws -> ArtifactPersistentVolumeIdentity?

    init(
        digestCheckpoint: @escaping @Sendable (String) -> Void = { _ in },
        digestStarted: @escaping @Sendable (String) -> Void = { _ in },
        digestFinished: @escaping @Sendable (String) -> Void = { _ in },
        volumeSupportsPersistentIDs: @escaping @Sendable (URL) throws -> Bool? = {
            try ArtifactFilesystemEvidence.persistentIDCapability(at: $0)
        },
        descriptorSupportsPersistentIDs:
            @escaping @Sendable (Int32, String) throws -> Bool? = {
                try ArtifactFilesystemEvidence.persistentIDCapability(of: $0, path: $1)
            },
        descriptorPersistentVolumeIdentity:
            @escaping @Sendable (Int32, String) throws
                -> ArtifactPersistentVolumeIdentity? = {
                    try ArtifactFilesystemEvidence.persistentVolumeIdentity(
                        of: $0, path: $1)
        }
    ) {
        self.digestCheckpoint = digestCheckpoint
        self.digestStarted = digestStarted
        self.digestFinished = digestFinished
        self.volumeSupportsPersistentIDs = volumeSupportsPersistentIDs
        self.descriptorSupportsPersistentIDs = descriptorSupportsPersistentIDs
        self.descriptorPersistentVolumeIdentity = descriptorPersistentVolumeIdentity
    }

    static let none = ArtifactVerifierHooks()
}

struct ArtifactPublishedTreeTotals: Sendable, Equatable {
    let fileCount: Int
    let payloadFileCount: Int
    let metadataFileCount: Int
    let bytes: UInt64
    let payloadBytes: UInt64
    let metadataBytes: UInt64
}

private actor ArtifactVerificationPassCoordinator {
    static let shared = ArtifactVerificationPassCoordinator()

    private var rootsInProgress = Set<String>()

    func claim(_ root: String) -> Bool {
        rootsInProgress.insert(root).inserted
    }

    func release(_ root: String) {
        rootsInProgress.remove(root)
    }
}

/// Digests an artifact on disk against the digests its repository publishes.
///
/// The digest table is the repository tree and not the artifact's own
/// `index.json`, for the reason `HuggingFaceTreeClient` already states: no
/// published `index.json` carries a complete per-file digest table, and the
/// tree is the only thing that does. That makes a verification a networked act
/// — which is fine, because it is an act somebody asked for, unlike a scan.
///
/// Nothing here is ever called by ``ArtifactLocator``. That separation is the
/// whole point: a screen appearing must not start a terabyte of hashing.
public actor ArtifactVerifier {
    private let tree: HuggingFaceTreeClient
    private let catalog: ModelCatalogSnapshot
    private let ledger: any ArtifactVerificationLedger
    private let checkpoints: (any ArtifactVerificationCheckpointStore)?
    private let hooks: ArtifactVerifierHooks
    private var verificationInProgress = false

    public init(
        tree: HuggingFaceTreeClient, catalog: ModelCatalogSnapshot,
        ledger: any ArtifactVerificationLedger,
        checkpoints: (any ArtifactVerificationCheckpointStore)? = nil
    ) {
        self.tree = tree
        self.catalog = catalog
        self.ledger = ledger
        self.checkpoints = checkpoints
        self.hooks = .none
    }

    init(
        tree: HuggingFaceTreeClient, catalog: ModelCatalogSnapshot,
        ledger: any ArtifactVerificationLedger,
        checkpoints: (any ArtifactVerificationCheckpointStore)? = nil,
        hooks: ArtifactVerifierHooks
    ) {
        self.tree = tree
        self.catalog = catalog
        self.ledger = ledger
        self.checkpoints = checkpoints
        self.hooks = hooks
    }

    /// Check an artifact and record the result.
    ///
    /// The report's `job` is a fresh id that names *this check*. It is not a
    /// download job: these bytes may never have been fetched by this app at
    /// all, which is exactly the case discovery exists to handle.
    @discardableResult
    public func verify(
        _ artifact: DiscoveredArtifact,
        _ request: ArtifactVerificationRequest,
        resumption: ArtifactVerificationResumption = .resumeIfPossible,
        onProgress: (@Sendable (ArtifactVerificationProgress) -> Void)? = nil
    ) async throws -> VerificationReport {
        try await verify(
            artifact, request, rootedAt: nil, resumption: resumption,
            onProgress: onProgress)
    }

    /// Check an artifact using an already-open root capability.
    ///
    /// This is the path for a scan result beneath a security-scoped storage
    /// location. Every payload is opened relative to the capability's final
    /// artifact fd; the URL is retained only for display and ledger identity.
    @discardableResult
    public func verify(
        _ artifact: DiscoveredArtifact,
        _ request: ArtifactVerificationRequest,
        rootedAt authority: ArtifactVerificationRoot,
        resumption: ArtifactVerificationResumption = .resumeIfPossible,
        onProgress: (@Sendable (ArtifactVerificationProgress) -> Void)? = nil
    ) async throws -> VerificationReport {
        try await verify(
            artifact, request, rootedAt: Optional(authority), resumption: resumption,
            onProgress: onProgress)
    }

    private func verify(
        _ artifact: DiscoveredArtifact,
        _ request: ArtifactVerificationRequest,
        rootedAt authority: ArtifactVerificationRoot?,
        resumption: ArtifactVerificationResumption,
        onProgress: (@Sendable (ArtifactVerificationProgress) -> Void)?
    ) async throws -> VerificationReport {
        guard !verificationInProgress else {
            throw ArtifactVerificationError.verificationAlreadyInProgress
        }
        verificationInProgress = true
        defer { verificationInProgress = false }
        let coordinationRoot = authority?.rootURL.path
            ?? artifact.url.standardizedFileURL.path
        guard await ArtifactVerificationPassCoordinator.shared.claim(coordinationRoot) else {
            throw ArtifactVerificationError.verificationAlreadyInProgress
        }
        do {
            let report = try await verifyPass(
                artifact, request, rootedAt: authority, resumption: resumption,
                onProgress: onProgress)
            await ArtifactVerificationPassCoordinator.shared.release(coordinationRoot)
            return report
        } catch is CancellationError {
            await ArtifactVerificationPassCoordinator.shared.release(coordinationRoot)
            throw ArtifactVerificationError.cancelled
        } catch {
            await ArtifactVerificationPassCoordinator.shared.release(coordinationRoot)
            throw error
        }
    }

    private func verifyPass(
        _ artifact: DiscoveredArtifact,
        _ request: ArtifactVerificationRequest,
        rootedAt authority: ArtifactVerificationRoot?,
        resumption: ArtifactVerificationResumption,
        onProgress: (@Sendable (ArtifactVerificationProgress) -> Void)?
    ) async throws -> VerificationReport {
        guard let model = artifact.model else {
            throw ArtifactVerificationError.unidentifiedArtifact(rootPath: artifact.rootPath)
        }
        guard let descriptor = catalog.descriptor(model), let repo = descriptor.source.repo else {
            throw ArtifactVerificationError.noPublishedDigests(model)
        }

        let completeTree = try await tree.completeTree(in: repo)
        let published = completeTree.files
        // `completeTree` has already reconciled the paginated digest table
        // against the pinned repository-info sibling set. Descriptor totals
        // are a second consistency check, never the tree's completeness
        // authority: a live/open descriptor may have been derived from this
        // same tree response.
        let treeTotals = try Self.reconcile(
            published, with: descriptor, model: model)
        let treeIdentity = ArtifactDigestPlanIdentity.compute(published)
        // A pinned commit is immutable. If an explicit tree fetch nevertheless
        // returns different metadata, an earlier positive record is no longer
        // attributable to the plan this verifier is about to use.
        _ = ledger.record(
            matching: ArtifactVerificationLookup(
                rootPath: artifact.rootPath, model: model, repository: repo,
                index: artifact.index, currentTreeIdentity: treeIdentity))

        let selected = try Self.select(published, for: request)
        let selectedPlanIdentity = ArtifactDigestPlanIdentity.compute(selected)
        let checkpointKey = ArtifactVerificationCheckpointKey(
            rootPath: artifact.rootPath, model: model, repository: repo,
            treeIdentity: treeIdentity, selectedPlanIdentity: selectedPlanIdentity,
            requestKind: request.kind)
        if resumption == .fromScratch {
            checkpoints?.discard(rootPath: artifact.rootPath, kind: request.kind)
        }
        let root = artifact.url
        let totalBytes = try Self.checkedByteTotal(
            selected, context: "the selected verification byte total")
        let rootDescriptor: Int32
        let rootIdentity: ArtifactFilesystemIdentity
        do {
            if let authority {
                guard authority.rootURL.standardizedFileURL.path
                    == root.standardizedFileURL.path
                else {
                    throw ArtifactFilesystemEvidenceError.changed(path: root.path)
                }
                try authority.validateCurrentBinding()
                let descriptor = try authority.duplicateArtifactDescriptor()
                do {
                    let identity = try ArtifactFilesystemEvidence.identity(
                        of: descriptor, path: root.path, expectedDirectory: true)
                    guard identity == authority.artifactIdentity else {
                        throw ArtifactFilesystemEvidenceError.changed(path: root.path)
                    }
                    rootDescriptor = descriptor
                    rootIdentity = identity
                } catch {
                    _ = close(descriptor)
                    throw error
                }
            } else {
                let opened = try ArtifactFilesystemEvidence.openRootAndIdentity(at: root)
                rootDescriptor = opened.descriptor
                rootIdentity = opened.identity
            }
        } catch {
            throw Self.filesystemError(error, fallbackPath: root.path, changed: false)
        }
        defer { _ = close(rootDescriptor) }
        let persistentVolume: ArtifactPersistentVolumeIdentity
        do {
            let supportsPersistentIDs: Bool?
            if let authority {
                supportsPersistentIDs = try authority.persistentIDCapability(
                    using: hooks.descriptorSupportsPersistentIDs)
            } else {
                supportsPersistentIDs = try hooks.volumeSupportsPersistentIDs(root)
            }
            guard supportsPersistentIDs == true else {
                throw ArtifactVerificationError.persistentFileIDsUnavailable(path: root.path)
            }
            guard let identity = try hooks.descriptorPersistentVolumeIdentity(
                rootDescriptor, root.path)
            else {
                throw ArtifactVerificationError.persistentFileIDsUnavailable(path: root.path)
            }
            persistentVolume = identity
        } catch let error as ArtifactVerificationError {
            throw error
        } catch {
            throw ArtifactVerificationError.filesystemUnavailable(
                path: root.path, reason: String(describing: error))
        }

        // Progress an earlier interrupted pass recorded, offered per file and
        // spent only where the object opened below still carries the identity
        // whose bytes were digested. A checkpoint from another volume, or from
        // a different artifact directory object, is not offered at all.
        var resumable: [String: ArtifactVerifiedFile] = [:]
        if resumption == .resumeIfPossible,
            let saved = checkpoints?.checkpoint(matching: checkpointKey),
            saved.persistentVolume == persistentVolume,
            saved.root.inode == rootIdentity.inode
        {
            resumable = Dictionary(
                saved.files.map { ($0.path, $0) }, uniquingKeysWith: { first, _ in first })
        }
        var matched: [ArtifactVerifiedFile] = []

        var ok: [String] = []
        var missing: [String] = []
        var wrongSize: [SizeMismatch] = []
        var wrongDigest: [DigestMismatch] = []
        var unreadable: [String: String] = [:]
        var bytesChecked: UInt64 = 0
        var verifiedFiles: [ArtifactVerifiedFile] = []
        var seenObjects: [ArtifactFilesystemObjectKey: String] = [:]

        for (offset, file) in selected.enumerated() {
            try Task.checkCancellation()
            onProgress?(
                ArtifactVerificationProgress(
                    filesChecked: offset, filesToCheck: selected.count,
                    bytesChecked: bytesChecked, bytesToCheck: totalBytes,
                    currentPath: file.path))

            let descriptor: Int32
            do {
                descriptor = try ArtifactFilesystemEvidence.openPayload(
                    file.path, beneath: rootDescriptor)
            } catch let error as ArtifactFilesystemEvidenceError where error.isMissing {
                missing.append(file.path)
                continue
            } catch {
                unreadable[file.path] = String(describing: error)
                continue
            }
            defer { _ = close(descriptor) }

            let before: ArtifactFilesystemIdentity
            do {
                before = try ArtifactFilesystemEvidence.identity(
                    of: descriptor, path: file.path, expectedDirectory: false)
            } catch {
                unreadable[file.path] = String(describing: error)
                continue
            }
            guard before.sizeBytes == file.sizeBytes else {
                wrongSize.append(
                    SizeMismatch(
                        path: file.path, expected: file.sizeBytes,
                        found: before.sizeBytes))
                bytesChecked = try Self.checkedAdding(
                    bytesChecked, before.sizeBytes,
                    context: "bytes checked after '\(file.path)'")
                continue
            }
            guard before.device == rootIdentity.device else {
                ledger.forget(rootPath: artifact.rootPath)
                checkpoints?.discard(rootPath: artifact.rootPath, kind: nil)
                throw ArtifactVerificationError.filesystemChanged(
                    path: file.path,
                    reason: "the file is mounted from a different volume than the artifact root")
            }
            let objectKey = ArtifactFilesystemObjectKey(before)
            if let firstPath = seenObjects.updateValue(file.path, forKey: objectKey) {
                ledger.forget(rootPath: artifact.rootPath)
                checkpoints?.discard(rootPath: artifact.rootPath, kind: nil)
                throw ArtifactVerificationError.filesystemUnavailable(
                    path: file.path,
                    reason: ArtifactFilesystemEvidenceError.aliased(
                        path: file.path, firstPath: firstPath).description)
            }
            // The whole of resumption. The digest is skipped only because it
            // was already computed against an object whose durable identity —
            // inode, size, and both nanosecond timestamps, on a volume whose
            // UUID still matches — is the identity this descriptor reports now.
            // Any rewrite, replacement or truncation moves one of those fields,
            // and the file is hashed again.
            if let saved = resumable[file.path],
                saved.expectedSizeBytes == file.sizeBytes,
                saved.digest == file.digest,
                saved.isPayload == file.isPayload,
                ArtifactFilesystemEvidence.durableIdentityMatches(
                    current: before, stored: saved.filesystem,
                    persistentVolume: persistentVolume)
            {
                let carried = ArtifactVerifiedFile(
                    path: file.path, expectedSizeBytes: file.sizeBytes,
                    digest: file.digest, isPayload: file.isPayload,
                    filesystem: before)
                ok.append(file.path)
                verifiedFiles.append(carried)
                matched.append(carried)
                bytesChecked = try Self.checkedAdding(
                    bytesChecked, file.sizeBytes,
                    context: "bytes checked after '\(file.path)'")
                continue
            }

            do {
                let digest = try await digestOffActor(
                    descriptor: descriptor, file: file)
                let after = try ArtifactFilesystemEvidence.identity(
                    of: descriptor, path: file.path, expectedDirectory: false)
                guard after == before else {
                    ledger.forget(rootPath: artifact.rootPath)
                    checkpoints?.discard(rootPath: artifact.rootPath, kind: nil)
                    throw ArtifactVerificationError.filesystemChanged(
                        path: file.path,
                        reason: "device, inode, size, modification time or status-change time moved during hashing")
                }
                let checked = ArtifactVerifiedFile(
                    path: file.path, expectedSizeBytes: file.sizeBytes,
                    digest: file.digest, isPayload: file.isPayload,
                    filesystem: after)
                if digest == file.digest {
                    ok.append(file.path)
                    matched.append(checked)
                    // Per file, and not per batch. One K3 payload is minutes of
                    // reading over USB and the record is a few hundred bytes
                    // per entry, so the write is free next to the work it
                    // protects — and a pass killed mid-file loses only that
                    // file rather than a window of them.
                    checkpoints?.record(
                        ArtifactVerificationCheckpoint(
                            rootPath: artifact.rootPath, model: model, repository: repo,
                            treeIdentity: treeIdentity,
                            selectedPlanIdentity: selectedPlanIdentity,
                            requestKind: request.kind,
                            persistentVolume: persistentVolume, root: rootIdentity,
                            updatedAt: Date(), files: matched))
                } else {
                    wrongDigest.append(
                        DigestMismatch(
                            path: file.path, algorithm: file.digest.algorithm,
                            expected: file.digest.hex, found: digest.hex))
                }
                verifiedFiles.append(checked)
            } catch is CancellationError {
                throw CancellationError()
            } catch let error as ArtifactVerificationError {
                throw error
            } catch {
                unreadable[file.path] = String(describing: error)
            }
            bytesChecked = try Self.checkedAdding(
                bytesChecked, file.sizeBytes,
                context: "bytes checked after '\(file.path)'")
        }

        try Task.checkCancellation()

        let report = VerificationReport(
            job: DownloadJobID(), model: model,
            depth: request == .full ? .digestEverything : .digestPayload,
            checkedAt: Date(),
            ok: ok, missing: missing, wrongSize: wrongSize, wrongDigest: wrongDigest,
            unreadable: unreadable, extraneous: [],
            bytesToRefetch: try Self.bytesToRefetch(
                published,
                paths: missing + wrongSize.map(\.path) + wrongDigest.map(\.path)
                    + Array(unreadable.keys)))

        // A failed check does not become a verified state, and it does not
        // silently leave the old one standing either: whatever was believed
        // about these bytes before is now known to be wrong.
        let state: ArtifactVerification = report.isComplete ? request.resultingState : .unverified
        let evidence: ArtifactVerificationEvidence?
        if report.isComplete {
            let rootAfter = try ArtifactFilesystemEvidence.identity(
                of: rootDescriptor, path: root.path, expectedDirectory: true)
            guard rootAfter == rootIdentity else {
                ledger.forget(rootPath: artifact.rootPath)
                checkpoints?.discard(rootPath: artifact.rootPath, kind: nil)
                throw ArtifactVerificationError.filesystemChanged(
                    path: root.path,
                    reason: "the artifact directory metadata moved during verification")
            }
            let built = ArtifactVerificationEvidence(
                model: model, repository: repo,
                treeIdentity: treeIdentity,
                completenessAuthorityIdentity: completeTree.completenessAuthorityIdentity,
                selectedPlanIdentity: selectedPlanIdentity,
                treeFileCount: treeTotals.fileCount,
                treePayloadFileCount: treeTotals.payloadFileCount,
                treeMetadataFileCount: treeTotals.metadataFileCount,
                treeBytes: treeTotals.bytes, treePayloadBytes: treeTotals.payloadBytes,
                treeMetadataBytes: treeTotals.metadataBytes,
                selectedBytes: totalBytes, index: artifact.index,
                persistentVolume: persistentVolume,
                root: rootIdentity,
                files: verifiedFiles.sorted { $0.path < $1.path })
            do {
                if let authority {
                    try ArtifactFilesystemEvidence.validate(
                        evidence: built, rootDescriptor: rootDescriptor,
                        rootPath: root.path)
                    try authority.validateCurrentBinding()
                } else {
                    try ArtifactFilesystemEvidence.validate(evidence: built, at: root)
                }
            } catch {
                ledger.forget(rootPath: artifact.rootPath)
                checkpoints?.discard(rootPath: artifact.rootPath, kind: nil)
                throw Self.filesystemError(error, fallbackPath: root.path, changed: true)
            }
            evidence = built
        } else {
            evidence = nil
        }
        try Task.checkCancellation()
        if let authority {
            do {
                // Keep this as the last filesystem act before the synchronous
                // ledger write. A path replacement during the digest can make
                // the pass fail, but cannot redirect any payload read or earn
                // positive evidence for its replacement.
                try authority.validateCurrentBinding()
            } catch {
                ledger.forget(rootPath: artifact.rootPath)
                checkpoints?.discard(rootPath: artifact.rootPath, kind: nil)
                throw Self.filesystemError(error, fallbackPath: root.path, changed: true)
            }
        }
        ledger.record(
            ArtifactVerificationRecord(
                rootPath: artifact.rootPath, state: state, checkedAt: report.checkedAt,
                detail: Self.sentence(report, request: request, selected: selected.count),
                evidence: evidence))
        if report.isComplete {
            // The durable record now covers every file the checkpoint held.
            // A pass that ended with faults keeps its progress instead: the
            // files that matched still matched, and the operator's next act is
            // usually to repair one file rather than to re-read a terabyte.
            checkpoints?.discard(rootPath: artifact.rootPath, kind: request.kind)
        }
        return report
    }

    /// Run the synchronous streaming hash on a detached utility task. The
    /// descriptor remains owned by the actor frame until this await returns;
    /// parent cancellation is forwarded to the detached task, whose closure is
    /// polled once per bounded digest chunk.
    private func digestOffActor(descriptor: Int32, file: RepoFile) async throws -> FileDigest {
        let checkpoint = hooks.digestCheckpoint
        let started = hooks.digestStarted
        let finished = hooks.digestFinished
        let hashing = Task.detached(priority: .utility) {
            started(file.path)
            defer { finished(file.path) }
            return try FileDigestComputer.digest(
                ofOpenFileDescriptor: descriptor, algorithm: file.digest.algorithm,
                cancellationRequested: {
                    checkpoint(file.path)
                    return Task.isCancelled
                })
        }
        return try await withTaskCancellationHandler {
            try await hashing.value
        } onCancel: {
            hashing.cancel()
        }
    }

    /// Which files a request digests.
    ///
    /// Full verification covers the complete published tree. Spot checking is
    /// payload-only and considers the largest files first, skipping an entry
    /// that cannot fit the remaining hard budget instead of exceeding it.
    static func select(
        _ files: [RepoFile], for request: ArtifactVerificationRequest
    ) throws -> [RepoFile] {
        let payload = files.filter(\.isPayload)
        switch request {
        case .full:
            guard !files.isEmpty, !payload.isEmpty else {
                throw ArtifactVerificationError.emptyPublishedTree
            }
            let selected = files.sorted { $0.path < $1.path }
            _ = try checkedByteTotal(selected, context: "the full published-file selection")
            return selected
        case .spotCheck(let budget):
            guard budget > 0 else {
                throw ArtifactVerificationError.budgetTooSmall(maximumBytes: budget)
            }
            var spent: UInt64 = 0
            var chosen: [RepoFile] = []
            for file in payload.filter({ $0.sizeBytes > 0 }).sorted(by: {
                $0.sizeBytes == $1.sizeBytes ? $0.path < $1.path : $0.sizeBytes > $1.sizeBytes
            }) {
                guard file.sizeBytes <= budget - spent else { continue }
                spent = try checkedAdding(
                    spent, file.sizeBytes, context: "the spot-check payload selection")
                chosen.append(file)
                if spent == budget { break }
            }
            guard !chosen.isEmpty else {
                throw ArtifactVerificationError.budgetTooSmall(maximumBytes: budget)
            }
            return chosen.sorted { $0.path < $1.path }
        }
    }

    static func reconcile(
        _ files: [RepoFile], with descriptor: ModelDescriptor, model: ModelID
    ) throws -> ArtifactPublishedTreeTotals {
        guard !files.isEmpty else {
            throw ArtifactVerificationError.publishedTreeMismatch(
                model: model, reason: "zero files were returned")
        }
        let payload = files.filter(\.isPayload)
        guard !payload.isEmpty else {
            throw ArtifactVerificationError.publishedTreeMismatch(
                model: model, reason: "zero payload files were returned")
        }
        let metadata = files.filter { !$0.isPayload }
        let payloadBytes = try checkedByteTotal(
            payload, context: "the published payload byte total")
        let metadataBytes = try checkedByteTotal(
            metadata, context: "the published metadata byte total")
        let bytes = try checkedAdding(
            payloadBytes, metadataBytes, context: "the complete published tree byte total")
        let expectedBytes = try checkedAdding(
            descriptor.payloadBytes, descriptor.metadataBytes,
            context: "the descriptor's complete byte total")
        let fileCount = try checkedCountAdding(
            payload.count, metadata.count, context: "the complete published tree file count")
        let expectedFileCount = try checkedCountAdding(
            descriptor.payloadFileCount, descriptor.metadataFileCount,
            context: "the descriptor's complete file count")
        let totals = ArtifactPublishedTreeTotals(
            fileCount: fileCount, payloadFileCount: payload.count,
            metadataFileCount: metadata.count, bytes: bytes,
            payloadBytes: payloadBytes, metadataBytes: metadataBytes)
        guard payload.count == descriptor.payloadFileCount,
            metadata.count == descriptor.metadataFileCount,
            fileCount == expectedFileCount,
            payloadBytes == descriptor.payloadBytes,
            metadataBytes == descriptor.metadataBytes,
            bytes == expectedBytes
        else {
            throw ArtifactVerificationError.publishedTreeMismatch(
                model: model,
                reason:
                    "expected \(descriptor.payloadFileCount) payload/\(descriptor.metadataFileCount) metadata files and \(descriptor.payloadBytes)/\(descriptor.metadataBytes) bytes, got \(payload.count)/\(metadata.count) files and \(payloadBytes)/\(metadataBytes) bytes")
        }
        return totals
    }

    static func bytesToRefetch<S: Sequence>(
        _ files: [RepoFile], paths: S
    ) throws -> UInt64 where S.Element == String {
        let wanted = Set(paths)
        return try checkedByteTotal(
            files.filter { wanted.contains($0.path) },
            context: "the repair byte total")
    }

    static func checkedByteTotal(
        _ files: [RepoFile], context: String
    ) throws -> UInt64 {
        var total: UInt64 = 0
        for file in files {
            total = try checkedAdding(total, file.sizeBytes, context: context)
        }
        return total
    }

    private static func checkedAdding(
        _ lhs: UInt64, _ rhs: UInt64, context: String
    ) throws -> UInt64 {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else {
            throw ArtifactVerificationError.arithmeticOverflow(context: context)
        }
        return sum
    }

    private static func checkedCountAdding(
        _ lhs: Int, _ rhs: Int, context: String
    ) throws -> Int {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else {
            throw ArtifactVerificationError.arithmeticOverflow(context: context)
        }
        return sum
    }

    private static func filesystemError(
        _ error: Error, fallbackPath: String, changed: Bool
    ) -> ArtifactVerificationError {
        let path: String
        if case ArtifactFilesystemEvidenceError.changed(let changedPath) = error {
            path = changedPath
        } else if case ArtifactFilesystemEvidenceError.cannotOpen(let failedPath, _) = error {
            path = failedPath
        } else if case ArtifactFilesystemEvidenceError.cannotStat(let failedPath, _) = error {
            path = failedPath
        } else if case ArtifactFilesystemEvidenceError.wrongKind(let failedPath, _) = error {
            path = failedPath
        } else if case ArtifactFilesystemEvidenceError.negativeSize(let failedPath) = error {
            path = failedPath
        } else if case ArtifactFilesystemEvidenceError.multipleLinks(let failedPath, _) = error {
            path = failedPath
        } else if case ArtifactFilesystemEvidenceError.aliased(let failedPath, _) = error {
            path = failedPath
        } else if case ArtifactFilesystemEvidenceError.persistentIDsUnavailable(let failedPath) = error {
            path = failedPath
        } else if case ArtifactFilesystemEvidenceError.invalidPath(let failedPath) = error {
            path = failedPath
        } else {
            path = fallbackPath
        }
        if changed {
            return .filesystemChanged(path: path, reason: String(describing: error))
        }
        return .filesystemUnavailable(path: path, reason: String(describing: error))
    }

    /// The operator-facing sentence for a report. Public because the same
    /// sentence is filed in the ledger and shown on screen, and two spellings of
    /// one result is one spelling too many.
    public static func sentence(
        _ report: VerificationReport, request: ArtifactVerificationRequest, selected: Int
    ) -> String {
        let scope: String
        switch request {
        case .full: scope = "every published file"
        case .spotCheck: scope = "\(selected) of the largest payload files"
        }
        if report.isComplete {
            return "\(scope) matched the digests \(report.model) publishes."
        }
        var faults: [String] = []
        if !report.missing.isEmpty { faults.append("\(report.missing.count) missing") }
        if !report.wrongSize.isEmpty { faults.append("\(report.wrongSize.count) the wrong size") }
        if !report.wrongDigest.isEmpty {
            faults.append("\(report.wrongDigest.count) with a wrong digest")
        }
        if !report.unreadable.isEmpty { faults.append("\(report.unreadable.count) unreadable") }
        return "Checked \(scope): " + faults.joined(separator: ", ") + "."
    }
}
