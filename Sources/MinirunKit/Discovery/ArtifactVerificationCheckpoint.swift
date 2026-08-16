import Foundation

/// Which pass a checkpoint belongs to.
///
/// Progress from a spot check can never be spent on a full pass, and the two
/// are filed under separate keys so a 2 GB sample cannot evict a terabyte of
/// full-pass progress by starting after it.
public enum ArtifactVerificationRequestKind: String, Codable, Sendable {
    case full
    case spotCheck
}

/// Everything that has to agree before recorded progress may be spent.
///
/// The plan identity already names every path, size, digest and payload flag
/// the pass will check, so a re-published tree, a different budget or a
/// different request produces a different key rather than a partial match.
public struct ArtifactVerificationCheckpointKey: Sendable, Equatable {
    public let rootPath: String
    public let model: ModelID
    public let repository: HuggingFaceRepoRef
    /// Canonical SHA-256 of every file entry returned for the pinned tree.
    public let treeIdentity: String
    /// Canonical SHA-256 of exactly the entries this pass will digest.
    public let selectedPlanIdentity: String
    public let requestKind: ArtifactVerificationRequestKind

    public init(
        rootPath: String, model: ModelID, repository: HuggingFaceRepoRef,
        treeIdentity: String, selectedPlanIdentity: String,
        requestKind: ArtifactVerificationRequestKind
    ) {
        self.rootPath = rootPath
        self.model = model
        self.repository = repository
        self.treeIdentity = treeIdentity
        self.selectedPlanIdentity = selectedPlanIdentity
        self.requestKind = requestKind
    }
}

/// The files one interrupted pass has already digested, and the physical
/// objects those digests were computed from.
///
/// This is deliberately *not* a verification state. Nothing reads it except the
/// verifier that wrote it, and reading it never produces a positive row: it
/// answers exactly one question — may this file's digest be skipped, because it
/// was already computed against bytes whose filesystem identity has not moved
/// since. Every entry was written only after the digest matched the published
/// value *and* the pre-hash and post-hash `stat` of the open descriptor agreed,
/// which is the same evidence ``ArtifactVerifiedFile`` carries into a completed
/// record.
public struct ArtifactVerificationCheckpoint: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1

    public let schemaVersion: Int
    public let rootPath: String
    public let model: ModelID
    public let repository: HuggingFaceRepoRef
    public let treeIdentity: String
    public let selectedPlanIdentity: String
    public let requestKind: ArtifactVerificationRequestKind
    /// The volume that supplied the persistent object ids below. Unlike the
    /// evidence record this is never optional: a checkpoint is only ever
    /// written by a pass that already proved the capability.
    public let persistentVolume: ArtifactPersistentVolumeIdentity
    /// The artifact directory the files were opened beneath.
    public let root: ArtifactFilesystemIdentity
    public let updatedAt: Date
    public let files: [ArtifactVerifiedFile]

    public init(
        schemaVersion: Int = ArtifactVerificationCheckpoint.currentSchemaVersion,
        rootPath: String, model: ModelID, repository: HuggingFaceRepoRef,
        treeIdentity: String, selectedPlanIdentity: String,
        requestKind: ArtifactVerificationRequestKind,
        persistentVolume: ArtifactPersistentVolumeIdentity,
        root: ArtifactFilesystemIdentity, updatedAt: Date,
        files: [ArtifactVerifiedFile]
    ) {
        self.schemaVersion = schemaVersion
        self.rootPath = rootPath
        self.model = model
        self.repository = repository
        self.treeIdentity = treeIdentity
        self.selectedPlanIdentity = selectedPlanIdentity
        self.requestKind = requestKind
        self.persistentVolume = persistentVolume
        self.root = root
        self.updatedAt = updatedAt
        self.files = files
    }

    var key: ArtifactVerificationCheckpointKey {
        ArtifactVerificationCheckpointKey(
            rootPath: rootPath, model: model, repository: repository,
            treeIdentity: treeIdentity, selectedPlanIdentity: selectedPlanIdentity,
            requestKind: requestKind)
    }

    /// Whether this record was written by a pass asking the same question.
    ///
    /// Everything here is metadata agreement. It decides only whether the
    /// per-file identities below are worth *offering* to the verifier; each one
    /// is still checked against the object opened beneath the current rooted
    /// descriptor before a single digest is skipped.
    func answers(_ lookup: ArtifactVerificationCheckpointKey) -> Bool {
        schemaVersion == Self.currentSchemaVersion
            && ArtifactVerificationPathIdentity.canonical(rootPath)
                == ArtifactVerificationPathIdentity.canonical(lookup.rootPath)
            && model == lookup.model
            && repository == lookup.repository
            && treeIdentity == lookup.treeIdentity
            && selectedPlanIdentity == lookup.selectedPlanIdentity
            && requestKind == lookup.requestKind
    }
}

/// Where an interrupted verification's progress lives between passes.
///
/// Separate from ``ArtifactVerificationLedger`` on purpose. The ledger holds
/// results an operator is shown and a runtime may consume; this holds work in
/// progress, which is never shown and never consumed. Losing everything in it
/// costs reading time and nothing else, which is why it is allowed to be
/// discarded on any doubt.
public protocol ArtifactVerificationCheckpointStore: Sendable {
    /// The recorded progress for exactly this pass, or nil.
    func checkpoint(
        matching lookup: ArtifactVerificationCheckpointKey
    ) -> ArtifactVerificationCheckpoint?
    func record(_ checkpoint: ArtifactVerificationCheckpoint)
    /// Throw progress away. A nil `kind` discards every pass kind for the root,
    /// which is what artifact removal and a contradicted tree both want.
    func discard(rootPath: String, kind: ArtifactVerificationRequestKind?)
}

public struct UserDefaultsVerificationCheckpointStore: ArtifactVerificationCheckpointStore {
    private static let coordinationLock = NSRecursiveLock()

    private final class DefaultsBox: @unchecked Sendable {
        let defaults: UserDefaults
        init(_ defaults: UserDefaults) { self.defaults = defaults }
    }

    private let box: DefaultsBox
    private let prefix: String

    public init(
        defaults: UserDefaults = .standard, prefix: String = "minirun.kit.verifying."
    ) {
        self.box = DefaultsBox(defaults)
        self.prefix = prefix
    }

    private func key(_ path: String, _ kind: ArtifactVerificationRequestKind) -> String {
        prefix + kind.rawValue + "." + ArtifactVerificationPathIdentity.canonical(path)
    }

    public func checkpoint(
        matching lookup: ArtifactVerificationCheckpointKey
    ) -> ArtifactVerificationCheckpoint? {
        Self.coordinationLock.lock()
        let stored = box.defaults.data(forKey: key(lookup.rootPath, lookup.requestKind))
        Self.coordinationLock.unlock()
        guard let stored,
            let decoded = try? MinirunKitJSON.decoder().decode(
                ArtifactVerificationCheckpoint.self, from: stored)
        else { return nil }
        guard decoded.answers(lookup) else { return nil }
        return decoded
    }

    public func record(_ checkpoint: ArtifactVerificationCheckpoint) {
        guard let data = try? MinirunKitJSON.encoder(pretty: false).encode(checkpoint)
        else { return }
        Self.coordinationLock.lock()
        defer { Self.coordinationLock.unlock() }
        box.defaults.set(data, forKey: key(checkpoint.rootPath, checkpoint.requestKind))
        // Deliberately no `synchronize()`. This is a cache of reading work, not
        // security evidence: a flush per digested file would cost the
        // preferences daemon a write per file for something whose loss costs
        // only time. The final record, which *is* evidence, still synchronizes.
    }

    public func discard(rootPath: String, kind: ArtifactVerificationRequestKind?) {
        let kinds: [ArtifactVerificationRequestKind] =
            kind.map { [$0] } ?? [.full, .spotCheck]
        Self.coordinationLock.lock()
        defer { Self.coordinationLock.unlock() }
        for kind in kinds {
            for path in ArtifactVerificationPathIdentity.storageAliases(for: rootPath) {
                box.defaults.removeObject(forKey: key(path, kind))
            }
        }
        box.defaults.synchronize()
    }
}

public final class InMemoryVerificationCheckpointStore:
    ArtifactVerificationCheckpointStore, @unchecked Sendable
{
    private struct Slot: Hashable {
        let path: String
        let kind: ArtifactVerificationRequestKind
    }

    private let lock = NSLock()
    private var storage: [Slot: ArtifactVerificationCheckpoint] = [:]

    public init() {}

    private func slot(
        _ path: String, _ kind: ArtifactVerificationRequestKind
    ) -> Slot {
        Slot(path: ArtifactVerificationPathIdentity.canonical(path), kind: kind)
    }

    public func checkpoint(
        matching lookup: ArtifactVerificationCheckpointKey
    ) -> ArtifactVerificationCheckpoint? {
        lock.lock()
        let stored = storage[slot(lookup.rootPath, lookup.requestKind)]
        lock.unlock()
        guard let stored, stored.answers(lookup) else { return nil }
        return stored
    }

    public func record(_ checkpoint: ArtifactVerificationCheckpoint) {
        lock.lock()
        defer { lock.unlock() }
        storage[slot(checkpoint.rootPath, checkpoint.requestKind)] = checkpoint
    }

    public func discard(rootPath: String, kind: ArtifactVerificationRequestKind?) {
        let kinds: [ArtifactVerificationRequestKind] =
            kind.map { [$0] } ?? [.full, .spotCheck]
        lock.lock()
        defer { lock.unlock() }
        for kind in kinds { storage.removeValue(forKey: slot(rootPath, kind)) }
    }
}
