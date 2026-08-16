import Darwin
import Foundation

/// Metadata identity of one filesystem object at the instant it was checked.
///
/// The verifier deliberately does not use paths as identities. A removable
/// volume may be re-mounted at the same path, and a payload may be atomically
/// replaced without changing its name or length. `dev + ino` names the object;
/// size, modification time and status-change time detect in-place writes. The
/// times are the nanosecond fields returned by Darwin `stat(2)`, not rounded
/// Foundation dates. `device` is deliberately retained as the exact identity
/// of one mounted session; durable evidence across an unmount/remount pairs the
/// persistent object id (`inode`) with ``ArtifactPersistentVolumeIdentity``.
public struct ArtifactFilesystemIdentity: Codable, Sendable, Equatable {
    public let device: UInt64
    public let inode: UInt64
    public let sizeBytes: UInt64
    public let modificationSeconds: Int64
    public let modificationNanoseconds: Int64
    public let statusChangeSeconds: Int64
    public let statusChangeNanoseconds: Int64

    public init(
        device: UInt64, inode: UInt64, sizeBytes: UInt64,
        modificationSeconds: Int64, modificationNanoseconds: Int64,
        statusChangeSeconds: Int64, statusChangeNanoseconds: Int64
    ) {
        self.device = device
        self.inode = inode
        self.sizeBytes = sizeBytes
        self.modificationSeconds = modificationSeconds
        self.modificationNanoseconds = modificationNanoseconds
        self.statusChangeSeconds = statusChangeSeconds
        self.statusChangeNanoseconds = statusChangeNanoseconds
    }
}

/// Stable identity of the volume which supplied persistent filesystem object
/// identifiers for one verification pass.
///
/// Darwin's `st_dev` identifies a mounted session and may change when the same
/// removable volume is attached again. `ATTR_VOL_UUID`, by contrast, identifies
/// the volume, while `VOL_CAP_FMT_PERSISTENTOBJECTIDS` authorizes reuse of file
/// object ids across mounts. Both facts are required before Minirun ignores a
/// changed `st_dev`; neither a path nor a file size is durable authority.
public struct ArtifactPersistentVolumeIdentity: Codable, Sendable, Equatable {
    public let uuid: String

    public init(uuid: String) {
        self.uuid = uuid.lowercased()
    }

    var isCanonical: Bool {
        guard let parsed = UUID(uuidString: uuid) else { return false }
        return parsed.uuidString.lowercased() == uuid
    }
}

/// The physical object and published digest one successful pass checked.
/// A full pass includes both payloads and the metadata a runner consumes.
public struct ArtifactVerifiedFile: Codable, Sendable, Equatable {
    public let path: String
    public let expectedSizeBytes: UInt64
    public let digest: FileDigest
    public let isPayload: Bool
    public let filesystem: ArtifactFilesystemIdentity

    public init(
        path: String, expectedSizeBytes: UInt64, digest: FileDigest,
        isPayload: Bool = true, filesystem: ArtifactFilesystemIdentity
    ) {
        self.path = path
        self.expectedSizeBytes = expectedSizeBytes
        self.digest = digest
        self.isPayload = isPayload
        self.filesystem = filesystem
    }

    var repoFile: RepoFile {
        RepoFile(
            path: path, sizeBytes: expectedSizeBytes, digest: digest,
            isPayload: isPayload)
    }

    private enum CodingKeys: String, CodingKey {
        case path, expectedSizeBytes, digest, isPayload, filesystem
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            path: try container.decode(String.self, forKey: .path),
            expectedSizeBytes: try container.decode(UInt64.self, forKey: .expectedSizeBytes),
            digest: try container.decode(FileDigest.self, forKey: .digest),
            // Schema 1 stored payloads only and therefore had no discriminator.
            isPayload: try container.decodeIfPresent(Bool.self, forKey: .isPayload) ?? true,
            filesystem: try container.decode(
                ArtifactFilesystemIdentity.self, forKey: .filesystem))
    }
}

@available(*, deprecated, renamed: "ArtifactVerifiedFile")
public typealias ArtifactVerifiedPayload = ArtifactVerifiedFile

/// Evidence that makes a remembered verification applicable to one artifact,
/// rather than to whichever directory later occupies the same pathname.
public struct ArtifactVerificationEvidence: Codable, Sendable, Equatable {
    /// Schema 3 records that the digest plan's path set was reconciled with
    /// the pinned repository-info `siblings` authority. Schema 2 positives
    /// predate that independent completeness gate and are deliberately
    /// downgraded on read.
    public static let currentSchemaVersion = 3

    public let schemaVersion: Int
    public let model: ModelID
    public let repository: HuggingFaceRepoRef
    /// Canonical SHA-256 of every file entry returned for the pinned tree.
    public let treeIdentity: String
    /// Canonical SHA-256 of the independent pinned repository-info path set.
    public let completenessAuthorityIdentity: String
    /// Canonical SHA-256 of exactly the entries selected for this pass.
    public let selectedPlanIdentity: String
    public let treeFileCount: Int
    public let treePayloadFileCount: Int
    public let treeMetadataFileCount: Int
    public let treeBytes: UInt64
    public let treePayloadBytes: UInt64
    public let treeMetadataBytes: UInt64
    public let selectedBytes: UInt64
    public let index: ArtifactIndexIdentity
    /// Nil only for schema-3 records written before durable volume binding was
    /// introduced. Those legacy records retain exact `dev + ino` semantics and
    /// therefore cannot cross a remount without another explicit digest pass.
    public let persistentVolume: ArtifactPersistentVolumeIdentity?
    public let root: ArtifactFilesystemIdentity
    public let files: [ArtifactVerifiedFile]

    public init(
        schemaVersion: Int = ArtifactVerificationEvidence.currentSchemaVersion,
        model: ModelID, repository: HuggingFaceRepoRef,
        treeIdentity: String, completenessAuthorityIdentity: String = "",
        selectedPlanIdentity: String,
        treeFileCount: Int, treePayloadFileCount: Int, treeMetadataFileCount: Int,
        treeBytes: UInt64, treePayloadBytes: UInt64, treeMetadataBytes: UInt64,
        selectedBytes: UInt64,
        index: ArtifactIndexIdentity,
        persistentVolume: ArtifactPersistentVolumeIdentity? = nil,
        root: ArtifactFilesystemIdentity,
        files: [ArtifactVerifiedFile]
    ) {
        self.schemaVersion = schemaVersion
        self.model = model
        self.repository = repository
        self.treeIdentity = treeIdentity
        self.completenessAuthorityIdentity = completenessAuthorityIdentity
        self.selectedPlanIdentity = selectedPlanIdentity
        self.treeFileCount = treeFileCount
        self.treePayloadFileCount = treePayloadFileCount
        self.treeMetadataFileCount = treeMetadataFileCount
        self.treeBytes = treeBytes
        self.treePayloadBytes = treePayloadBytes
        self.treeMetadataBytes = treeMetadataBytes
        self.selectedBytes = selectedBytes
        self.index = index
        self.persistentVolume = persistentVolume
        self.root = root
        self.files = files
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion, model, repository, treeIdentity, completenessAuthorityIdentity
        case selectedPlanIdentity
        case treeFileCount, treePayloadFileCount, treeMetadataFileCount
        case treeBytes, treePayloadBytes, treeMetadataBytes, selectedBytes
        case index, persistentVolume, root, files
        // Schema 1 names. They are decode-only so old records can be
        // deliberately downgraded instead of becoming undecodable debris.
        case selectedPayloadBytes, payloads
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let schema = try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        let decodedFiles: [ArtifactVerifiedFile]
        let fileCount: Int
        let payloadCount: Int
        let metadataCount: Int
        let allBytes: UInt64
        let payloadBytes: UInt64
        let metadataBytes: UInt64
        let selected: UInt64
        if schema >= 2 {
            decodedFiles = try container.decode([ArtifactVerifiedFile].self, forKey: .files)
            fileCount = try container.decode(Int.self, forKey: .treeFileCount)
            payloadCount = try container.decode(Int.self, forKey: .treePayloadFileCount)
            metadataCount = try container.decode(Int.self, forKey: .treeMetadataFileCount)
            allBytes = try container.decode(UInt64.self, forKey: .treeBytes)
            payloadBytes = try container.decode(UInt64.self, forKey: .treePayloadBytes)
            metadataBytes = try container.decode(UInt64.self, forKey: .treeMetadataBytes)
            selected = try container.decode(UInt64.self, forKey: .selectedBytes)
        } else {
            decodedFiles = try container.decodeIfPresent(
                [ArtifactVerifiedFile].self, forKey: .payloads) ?? []
            payloadCount = try container.decodeIfPresent(
                Int.self, forKey: .treePayloadFileCount) ?? decodedFiles.count
            metadataCount = 0
            fileCount = payloadCount
            selected = try container.decodeIfPresent(
                UInt64.self, forKey: .selectedPayloadBytes) ?? 0
            allBytes = selected
            payloadBytes = selected
            metadataBytes = 0
        }
        self.init(
            schemaVersion: schema,
            model: try container.decode(ModelID.self, forKey: .model),
            repository: try container.decode(HuggingFaceRepoRef.self, forKey: .repository),
            treeIdentity: try container.decode(String.self, forKey: .treeIdentity),
            completenessAuthorityIdentity: try container.decodeIfPresent(
                String.self, forKey: .completenessAuthorityIdentity) ?? "",
            selectedPlanIdentity: try container.decode(
                String.self, forKey: .selectedPlanIdentity),
            treeFileCount: fileCount, treePayloadFileCount: payloadCount,
            treeMetadataFileCount: metadataCount, treeBytes: allBytes,
            treePayloadBytes: payloadBytes, treeMetadataBytes: metadataBytes,
            selectedBytes: selected,
            index: try container.decode(ArtifactIndexIdentity.self, forKey: .index),
            persistentVolume: try container.decodeIfPresent(
                ArtifactPersistentVolumeIdentity.self, forKey: .persistentVolume),
            root: try container.decode(ArtifactFilesystemIdentity.self, forKey: .root),
            files: decodedFiles)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(model, forKey: .model)
        try container.encode(repository, forKey: .repository)
        try container.encode(treeIdentity, forKey: .treeIdentity)
        try container.encode(completenessAuthorityIdentity, forKey: .completenessAuthorityIdentity)
        try container.encode(selectedPlanIdentity, forKey: .selectedPlanIdentity)
        try container.encode(treeFileCount, forKey: .treeFileCount)
        try container.encode(treePayloadFileCount, forKey: .treePayloadFileCount)
        try container.encode(treeMetadataFileCount, forKey: .treeMetadataFileCount)
        try container.encode(treeBytes, forKey: .treeBytes)
        try container.encode(treePayloadBytes, forKey: .treePayloadBytes)
        try container.encode(treeMetadataBytes, forKey: .treeMetadataBytes)
        try container.encode(selectedBytes, forKey: .selectedBytes)
        try container.encode(index, forKey: .index)
        try container.encodeIfPresent(persistentVolume, forKey: .persistentVolume)
        try container.encode(root, forKey: .root)
        try container.encode(files, forKey: .files)
    }

    func isStructurallyValid(for state: ArtifactVerification) -> Bool {
        guard schemaVersion == Self.currentSchemaVersion,
            repository.isPinnedCommit,
            HuggingFaceRepoRef.isCanonicalRepositoryID(repository.repoID),
            Self.isSHA256(treeIdentity),
            Self.isSHA256(completenessAuthorityIdentity),
            Self.isSHA256(selectedPlanIdentity),
            treeFileCount > 0,
            treePayloadFileCount > 0,
            treeMetadataFileCount >= 0,
            treeFileCount >= files.count,
            !files.isEmpty,
            state != .unverified,
            Self.checkedCount(treePayloadFileCount, treeMetadataFileCount) == treeFileCount,
            Self.checkedBytes(treePayloadBytes, treeMetadataBytes) == treeBytes,
            persistentVolume?.isCanonical ?? true
        else { return false }

        var seenPaths = Set<String>()
        var seenObjects = Set<ArtifactFilesystemObjectKey>()
        var bytes: UInt64 = 0
        var selectedPayloadCount = 0
        var selectedMetadataCount = 0
        var selectedPayloadBytes: UInt64 = 0
        var selectedMetadataBytes: UInt64 = 0
        for file in files {
            guard HuggingFaceTreeClient.isSafeRepositoryPath(file.path),
                seenPaths.insert(file.path).inserted,
                seenObjects.insert(ArtifactFilesystemObjectKey(file.filesystem)).inserted,
                file.expectedSizeBytes == file.filesystem.sizeBytes,
                persistentVolume == nil || file.filesystem.device == root.device
            else { return false }
            let (next, overflow) = bytes.addingReportingOverflow(file.expectedSizeBytes)
            guard !overflow else { return false }
            bytes = next
            if file.isPayload {
                selectedPayloadCount += 1
                guard let next = Self.checkedBytes(
                    selectedPayloadBytes, file.expectedSizeBytes) else { return false }
                selectedPayloadBytes = next
            } else {
                selectedMetadataCount += 1
                guard let next = Self.checkedBytes(
                    selectedMetadataBytes, file.expectedSizeBytes) else { return false }
                selectedMetadataBytes = next
            }
        }
        guard bytes == selectedBytes,
            ArtifactDigestPlanIdentity.compute(files.map(\.repoFile)) == selectedPlanIdentity
        else { return false }
        switch state {
        case .fullyVerified:
            return treeIdentity == selectedPlanIdentity
                && files.count == treeFileCount
                && selectedPayloadCount == treePayloadFileCount
                && selectedMetadataCount == treeMetadataFileCount
                && selectedPayloadBytes == treePayloadBytes
                && selectedMetadataBytes == treeMetadataBytes
                && selectedBytes == treeBytes
        case .spotChecked:
            return selectedMetadataCount == 0 && files.allSatisfy { $0.isPayload && $0.expectedSizeBytes > 0 }
                && selectedPayloadCount <= treePayloadFileCount
                && selectedPayloadBytes <= treePayloadBytes
        case .unverified:
            return false
        }
    }

    private static func checkedCount(_ lhs: Int, _ rhs: Int) -> Int? {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? nil : sum
    }

    private static func checkedBytes(_ lhs: UInt64, _ rhs: UInt64) -> UInt64? {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? nil : sum
    }

    private static func isSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy {
            (48...57).contains($0) || (97...102).contains($0)
        }
    }
}

struct ArtifactFilesystemObjectKey: Hashable {
    let device: UInt64
    let inode: UInt64

    init(_ identity: ArtifactFilesystemIdentity) {
        device = identity.device
        inode = identity.inode
    }
}

/// The facts available during a metadata-only scan. A current tree identity is
/// optional because scanning must not make a network request; the pinned commit
/// makes that tree immutable. The verifier supplies it after its explicit tree
/// fetch, which detects a server/cache returning a different tree for the same
/// nominal commit.
public struct ArtifactVerificationLookup: Sendable, Equatable {
    public let rootPath: String
    public let model: ModelID?
    public let repository: HuggingFaceRepoRef?
    public let index: ArtifactIndexIdentity
    public let currentTreeIdentity: String?

    public init(
        rootPath: String, model: ModelID?, repository: HuggingFaceRepoRef?,
        index: ArtifactIndexIdentity, currentTreeIdentity: String? = nil
    ) {
        self.rootPath = rootPath
        self.model = model
        self.repository = repository
        self.index = index
        self.currentTreeIdentity = currentTreeIdentity
    }
}

/// Stable, unambiguous identity for a repository digest plan.
///
/// This hashes metadata only. Each field is length-prefixed, so neither a path
/// separator nor a future multi-digit size can make two different plans share
/// the same byte stream. Sorting makes pagination order irrelevant.
enum ArtifactDigestPlanIdentity {
    static func compute(_ files: [RepoFile]) -> String {
        var canonical = Data("minirun-artifact-digest-plan-v1".utf8)
        appendField(Data(), to: &canonical)
        for file in files.sorted(by: { $0.path < $1.path }) {
            appendField(Data(file.path.utf8), to: &canonical)
            appendField(Data(String(file.sizeBytes).utf8), to: &canonical)
            appendField(Data(file.digest.algorithm.rawValue.utf8), to: &canonical)
            appendField(Data(file.digest.hex.utf8), to: &canonical)
            appendField(Data(file.isPayload ? "1".utf8 : "0".utf8), to: &canonical)
        }
        return FileDigestComputer.sha256(of: canonical)
    }

    private static func appendField(_ field: Data, to output: inout Data) {
        output.append(Data(String(field.count).utf8))
        output.append(0x3A)  // ':'
        output.append(field)
        output.append(0x0A)  // '\n'
    }
}

enum ArtifactFilesystemEvidenceError: Error, CustomStringConvertible {
    case invalidPath(String)
    case cannotOpen(path: String, code: Int32)
    case cannotStat(path: String, code: Int32)
    case wrongKind(path: String, expected: String)
    case negativeSize(path: String)
    case multipleLinks(path: String, count: UInt64)
    case aliased(path: String, firstPath: String)
    case persistentIDsUnavailable(path: String)
    case changed(path: String)

    var description: String {
        switch self {
        case .invalidPath(let path):
            return "'\(path)' is not a canonical repository-relative path"
        case .cannotOpen(let path, let code):
            return "cannot open '\(path)': \(String(cString: strerror(code)))"
        case .cannotStat(let path, let code):
            return "cannot stat '\(path)': \(String(cString: strerror(code)))"
        case .wrongKind(let path, let expected):
            return "'\(path)' is not a \(expected)"
        case .negativeSize(let path):
            return "'\(path)' reports a negative size"
        case .multipleLinks(let path, let count):
            return "'\(path)' has \(count) hard links; durable identity requires exactly one"
        case .aliased(let path, let firstPath):
            return "'\(path)' aliases the same filesystem object as '\(firstPath)'"
        case .persistentIDsUnavailable(let path):
            return "the volume containing '\(path)' does not report persistent file identifiers"
        case .changed(let path):
            return "'\(path)' changed while its verification evidence was being checked"
        }
    }

    var isMissing: Bool {
        if case .cannotOpen(_, let code) = self { return code == ENOENT }
        return false
    }

    /// Only a contradiction in durable identity invalidates remembered
    /// evidence. Missing/unreadable media and unavailable persistent-ID
    /// capability suppress the positive state for this lookup but may be
    /// transient (especially for a removable volume), so the ledger retains
    /// the record for a later successful pass.
    var invalidatesRememberedEvidence: Bool {
        switch self {
        case .changed, .wrongKind, .negativeSize, .multipleLinks, .aliased, .invalidPath:
            return true
        case .cannotOpen, .cannotStat, .persistentIDsUnavailable:
            return false
        }
    }
}

struct ArtifactFilesystemValidationHooks: Sendable {
    var betweenIdentityPasses: @Sendable () -> Void
    var persistentIDCapability: @Sendable (Int32, String) throws -> Bool? = {
        try ArtifactFilesystemEvidence.persistentIDCapability(of: $0, path: $1)
    }
    var persistentVolumeIdentity:
        @Sendable (Int32, String) throws -> ArtifactPersistentVolumeIdentity? = {
            try ArtifactFilesystemEvidence.persistentVolumeIdentity(of: $0, path: $1)
        }
    var filesystemIdentity:
        @Sendable (Int32, String, Bool) throws -> ArtifactFilesystemIdentity = {
            try ArtifactFilesystemEvidence.identity(
                of: $0, path: $1, expectedDirectory: $2)
        }

    static let none = ArtifactFilesystemValidationHooks(betweenIdentityPasses: {})
}

/// An opened, rooted capability for one artifact beneath a registered storage
/// location.
///
/// A URL cannot provide this guarantee: an intermediate directory can be
/// replaced after a safe walk and before a verifier opens the URL again. This
/// object keeps both the registered-location directory and the final artifact
/// directory open. Verification duplicates only the final descriptor for its
/// payload reads, then asks the capability to replay the relative path from the
/// still-open registered root before it records positive evidence.
///
/// Raw descriptors are deliberately not public. The caller owns a capability,
/// not general filesystem authority, and descriptor lifetime ends in `deinit`.
public final class ArtifactVerificationRoot: @unchecked Sendable {
    public let rootURL: URL
    public let registeredLocationURL: URL

    private let relativeComponents: [String]
    private let registeredLocationDescriptor: Int32
    fileprivate let artifactDescriptor: Int32
    private let registeredLocationIdentity: ArtifactFilesystemIdentity
    let artifactIdentity: ArtifactFilesystemIdentity

    private init(
        rootURL: URL, registeredLocationURL: URL, relativeComponents: [String],
        registeredLocationDescriptor: Int32, artifactDescriptor: Int32,
        registeredLocationIdentity: ArtifactFilesystemIdentity,
        artifactIdentity: ArtifactFilesystemIdentity
    ) {
        self.rootURL = rootURL
        self.registeredLocationURL = registeredLocationURL
        self.relativeComponents = relativeComponents
        self.registeredLocationDescriptor = registeredLocationDescriptor
        self.artifactDescriptor = artifactDescriptor
        self.registeredLocationIdentity = registeredLocationIdentity
        self.artifactIdentity = artifactIdentity
    }

    /// Open one canonical relative directory beneath the location selected by
    /// the operator. Every component is opened with `O_NOFOLLOW`.
    public static func open(
        relativeComponents: [String], beneath locationURL: URL
    ) throws -> ArtifactVerificationRoot {
        let location = locationURL.standardizedFileURL
        let openedLocation = try ArtifactFilesystemEvidence.openRootAndIdentity(at: location)
        do {
            let artifactDescriptor = try ArtifactFilesystemEvidence.openDirectory(
                relativeComponents, beneath: openedLocation.descriptor,
                path: relativeComponents.isEmpty
                    ? location.path : relativeComponents.joined(separator: "/"))
            do {
                let artifactIdentity = try ArtifactFilesystemEvidence.identity(
                    of: artifactDescriptor,
                    path: relativeComponents.isEmpty
                        ? location.path : relativeComponents.joined(separator: "/"),
                    expectedDirectory: true)
                let rootURL = relativeComponents.reduce(location) { root, component in
                    root.appendingPathComponent(component, isDirectory: true)
                }.standardizedFileURL
                return ArtifactVerificationRoot(
                    rootURL: rootURL, registeredLocationURL: location,
                    relativeComponents: relativeComponents,
                    registeredLocationDescriptor: openedLocation.descriptor,
                    artifactDescriptor: artifactDescriptor,
                    registeredLocationIdentity: openedLocation.identity,
                    artifactIdentity: artifactIdentity)
            } catch {
                _ = close(artifactDescriptor)
                throw error
            }
        } catch {
            _ = close(openedLocation.descriptor)
            throw error
        }
    }

    deinit {
        _ = close(artifactDescriptor)
        _ = close(registeredLocationDescriptor)
    }

    func duplicateArtifactDescriptor() throws -> Int32 {
        try ArtifactFilesystemEvidence.duplicate(
            artifactDescriptor, path: rootURL.path)
    }

    func persistentIDCapability(
        using provider: (Int32, String) throws -> Bool?
    ) throws -> Bool? {
        // A registered folder may itself contain another mounted filesystem.
        // Ask the descriptor that owns the artifact bytes, rather than the
        // bookmark root, so the persistence claim applies to the identities
        // the verifier will actually record.
        try provider(artifactDescriptor, rootURL.path)
    }

    /// Prove that the held artifact is still the object reached by a fresh,
    /// no-follow walk from the exact registered location. Holding the old fd is
    /// what keeps reads safe; this replay is what prevents those safe reads of
    /// a now-unlinked object from authorizing whichever object later occupies
    /// the pathname.
    func validateCurrentBinding() throws {
        let heldLocation = try ArtifactFilesystemEvidence.identity(
            of: registeredLocationDescriptor, path: registeredLocationURL.path,
            expectedDirectory: true)
        guard ArtifactFilesystemObjectKey(heldLocation)
            == ArtifactFilesystemObjectKey(registeredLocationIdentity)
        else {
            throw ArtifactFilesystemEvidenceError.changed(path: registeredLocationURL.path)
        }

        let heldArtifact = try ArtifactFilesystemEvidence.identity(
            of: artifactDescriptor, path: rootURL.path, expectedDirectory: true)
        guard heldArtifact == artifactIdentity else {
            throw ArtifactFilesystemEvidenceError.changed(path: rootURL.path)
        }

        let replayed = try ArtifactFilesystemEvidence.openDirectory(
            relativeComponents, beneath: registeredLocationDescriptor, path: rootURL.path)
        defer { _ = close(replayed) }
        let replayedIdentity = try ArtifactFilesystemEvidence.identity(
            of: replayed, path: rootURL.path, expectedDirectory: true)
        guard replayedIdentity == artifactIdentity else {
            throw ArtifactFilesystemEvidenceError.changed(path: rootURL.path)
        }

        // The held registered-root fd alone would continue to work after its
        // pathname was replaced. Reopening the bookmarked URL binds the final
        // evidence to the location the app will resolve on its next launch.
        let reopenedLocation = try ArtifactFilesystemEvidence.openRootAndIdentity(
            at: registeredLocationURL)
        defer { _ = close(reopenedLocation.descriptor) }
        guard ArtifactFilesystemObjectKey(reopenedLocation.identity)
            == ArtifactFilesystemObjectKey(registeredLocationIdentity)
        else {
            throw ArtifactFilesystemEvidenceError.changed(path: registeredLocationURL.path)
        }
    }

    /// Permanently remove the exact artifact directory held by this capability.
    ///
    /// Removal is deliberately descriptor-relative. Before a byte is removed,
    /// the directory is atomically swapped with a private empty placeholder in
    /// its registered parent and the object at the private name is compared with
    /// the already-open artifact descriptor. If another process replaces the
    /// visible pathname during that exchange, the swap is reversed and nothing
    /// is deleted. Descendants are then traversed from the held artifact
    /// descriptor with `O_NOFOLLOW`, so a symlink can only be unlinked; it can
    /// never redirect deletion outside the selected artifact.
    ///
    /// The registered location itself is never a valid target. Callers are
    /// expected to obtain this capability from a fresh user-authorized storage
    /// scope and to present their own destructive confirmation before invoking
    /// the operation.
    public func removeRecursively() throws -> ArtifactRemovalReceipt {
        try removeRecursively(hooks: .none)
    }

    func removeRecursively(hooks: ArtifactRemovalHooks) throws -> ArtifactRemovalReceipt {
        guard !relativeComponents.isEmpty, let leaf = relativeComponents.last else {
            throw ArtifactRemovalError.refusesRegisteredLocation(
                path: registeredLocationURL.path)
        }
        try Task.checkCancellation()
        try validateCurrentBinding()
        let census = try ArtifactRemovalFilesystem.census(
            beneath: artifactDescriptor, rootDevice: artifactIdentity.device,
            displayPath: rootURL.path)
        try Task.checkCancellation()

        let parentComponents = Array(relativeComponents.dropLast())
        let parentDescriptor = try ArtifactFilesystemEvidence.openDirectory(
            parentComponents, beneath: registeredLocationDescriptor,
            path: rootURL.deletingLastPathComponent().path)
        defer { _ = close(parentDescriptor) }

        // Revalidate after the potentially long census. The hook is test-only
        // and creates the exact check-to-rename replacement the swap protocol
        // must reject without touching either object.
        try validateCurrentBinding()
        hooks.beforeAtomicIsolation()
        try Task.checkCancellation()

        let quarantine = ".minirun-removing-\(UUID().uuidString)"
        guard quarantine.utf8.count <= Int(NAME_MAX) else {
            throw ArtifactRemovalError.invalidTarget(path: rootURL.path)
        }
        let created = quarantine.withCString { mkdirat(parentDescriptor, $0, 0o700) }
        guard created == 0 else {
            throw ArtifactRemovalError.cannotPrepare(path: rootURL.path, code: errno)
        }
        var placeholderAtQuarantine = true
        defer {
            if placeholderAtQuarantine {
                _ = quarantine.withCString {
                    unlinkat(parentDescriptor, $0, AT_REMOVEDIR)
                }
            }
        }

        let swapResult = leaf.withCString { leafPath in
            quarantine.withCString { quarantinePath in
                renameatx_np(
                    parentDescriptor, leafPath, parentDescriptor, quarantinePath,
                    UInt32(RENAME_SWAP))
            }
        }
        guard swapResult == 0 else {
            throw ArtifactRemovalError.atomicIsolationUnavailable(
                path: rootURL.path, code: errno)
        }
        placeholderAtQuarantine = false

        do {
            let isolated = try ArtifactRemovalFilesystem.identity(
                named: quarantine, beneath: parentDescriptor, displayPath: rootURL.path)
            guard isolated.device == artifactIdentity.device,
                isolated.inode == artifactIdentity.inode
            else {
                try ArtifactRemovalFilesystem.reverseSwap(
                    leaf: leaf, quarantine: quarantine, parent: parentDescriptor,
                    displayPath: rootURL.path)
                placeholderAtQuarantine = true
                throw ArtifactRemovalError.changed(path: rootURL.path)
            }

            // The visible artifact name currently holds only the empty
            // placeholder we created. Removing it makes a concurrent scan see
            // absence rather than an unidentified empty model directory.
            try ArtifactRemovalFilesystem.unlink(
                name: leaf, beneath: parentDescriptor, directory: true,
                displayPath: rootURL.path)

            do {
                try ArtifactRemovalFilesystem.removeContents(
                    beneath: artifactDescriptor, rootDevice: artifactIdentity.device,
                    displayPath: rootURL.path)
                let finalIdentity = try ArtifactRemovalFilesystem.identity(
                    named: quarantine, beneath: parentDescriptor, displayPath: rootURL.path)
                guard finalIdentity.device == artifactIdentity.device,
                    finalIdentity.inode == artifactIdentity.inode
                else {
                    throw ArtifactRemovalError.changed(path: rootURL.path)
                }
                try ArtifactRemovalFilesystem.unlink(
                    name: quarantine, beneath: parentDescriptor, directory: true,
                    displayPath: rootURL.path)
            } catch {
                // Best effort makes a partially removed artifact visible again.
                // `RENAME_EXCL` refuses to overwrite anything that appeared at
                // the original name while deletion was in progress.
                _ = quarantine.withCString { quarantinePath in
                    leaf.withCString { leafPath in
                        renameatx_np(
                            parentDescriptor, quarantinePath, parentDescriptor, leafPath,
                            UInt32(RENAME_EXCL))
                    }
                }
                throw error
            }
        } catch {
            throw error
        }

        return ArtifactRemovalReceipt(
            rootPath: rootURL.path, removedEntryCount: census.entryCount,
            removedRegularFileBytes: census.regularFileBytes)
    }
}

/// What one descriptor-bound artifact removal actually reclaimed.
public struct ArtifactRemovalReceipt: Sendable, Equatable {
    public let rootPath: String
    public let removedEntryCount: UInt64
    public let removedRegularFileBytes: UInt64

    public init(
        rootPath: String, removedEntryCount: UInt64, removedRegularFileBytes: UInt64
    ) {
        self.rootPath = rootPath
        self.removedEntryCount = removedEntryCount
        self.removedRegularFileBytes = removedRegularFileBytes
    }
}

/// Named fail-closed reasons for local artifact removal.
public enum ArtifactRemovalError: Error, Sendable, Equatable, CustomStringConvertible {
    case refusesRegisteredLocation(path: String)
    case invalidTarget(path: String)
    case cannotPrepare(path: String, code: Int32)
    case atomicIsolationUnavailable(path: String, code: Int32)
    case cannotRead(path: String, code: Int32)
    case cannotRemove(path: String, code: Int32)
    case crossedFilesystem(path: String)
    case unsupportedEntry(path: String)
    case accountingOverflow(path: String)
    case changed(path: String)

    public var description: String {
        switch self {
        case .refusesRegisteredLocation(let path):
            return "removal refused because '\(path)' is the registered storage location itself"
        case .invalidTarget(let path):
            return "removal refused because '\(path)' is not a safe artifact target"
        case .cannotPrepare(let path, let code):
            return "removal could not prepare '\(path)' (errno \(code))"
        case .atomicIsolationUnavailable(let path, let code):
            return "removal of '\(path)' requires an atomic filesystem swap that is unavailable (errno \(code))"
        case .cannotRead(let path, let code):
            return "removal could not inspect '\(path)' (errno \(code))"
        case .cannotRemove(let path, let code):
            return "removal could not delete '\(path)' (errno \(code))"
        case .crossedFilesystem(let path):
            return "removal refused because '\(path)' crosses into another filesystem"
        case .unsupportedEntry(let path):
            return "removal refused because '\(path)' is not a regular file, directory, or symbolic link"
        case .accountingOverflow(let path):
            return "removal refused because the contents of '\(path)' cannot be counted safely"
        case .changed(let path):
            return "removal refused because '\(path)' changed after it was selected"
        }
    }
}

struct ArtifactRemovalHooks: Sendable {
    var beforeAtomicIsolation: @Sendable () -> Void
    static let none = ArtifactRemovalHooks(beforeAtomicIsolation: {})
}

private enum ArtifactRemovalFilesystem {
    struct Census {
        var entryCount: UInt64 = 0
        var regularFileBytes: UInt64 = 0
    }

    static func census(
        beneath descriptor: Int32, rootDevice: UInt64, displayPath: String, depth: Int = 0
    ) throws -> Census {
        try Task.checkCancellation()
        guard depth <= 128 else {
            throw ArtifactRemovalError.invalidTarget(path: displayPath)
        }
        var result = Census()
        for name in try entries(beneath: descriptor, displayPath: displayPath) {
            try Task.checkCancellation()
            let path = displayPath + "/" + name
            let status = try status(named: name, beneath: descriptor, displayPath: path)
            guard UInt64(status.st_dev) == rootDevice else {
                throw ArtifactRemovalError.crossedFilesystem(path: path)
            }
            let (entryCount, entryOverflow) = result.entryCount.addingReportingOverflow(1)
            guard !entryOverflow else {
                throw ArtifactRemovalError.accountingOverflow(path: displayPath)
            }
            result.entryCount = entryCount
            switch status.st_mode & S_IFMT {
            case S_IFDIR:
                let child = try openDirectory(
                    named: name, beneath: descriptor, displayPath: path)
                let nested: Census
                do {
                    nested = try census(
                        beneath: child, rootDevice: rootDevice, displayPath: path,
                        depth: depth + 1)
                    _ = close(child)
                } catch {
                    _ = close(child)
                    throw error
                }
                let (nestedCount, countOverflow) = result.entryCount.addingReportingOverflow(
                    nested.entryCount)
                let (nestedBytes, bytesOverflow) = result.regularFileBytes.addingReportingOverflow(
                    nested.regularFileBytes)
                guard !countOverflow, !bytesOverflow else {
                    throw ArtifactRemovalError.accountingOverflow(path: displayPath)
                }
                result.entryCount = nestedCount
                result.regularFileBytes = nestedBytes
            case S_IFREG:
                guard status.st_size >= 0 else {
                    throw ArtifactRemovalError.cannotRead(path: path, code: EOVERFLOW)
                }
                let (bytes, overflow) = result.regularFileBytes.addingReportingOverflow(
                    UInt64(status.st_size))
                guard !overflow else {
                    throw ArtifactRemovalError.accountingOverflow(path: displayPath)
                }
                result.regularFileBytes = bytes
            case S_IFLNK:
                break
            default:
                throw ArtifactRemovalError.unsupportedEntry(path: path)
            }
        }
        return result
    }

    static func removeContents(
        beneath descriptor: Int32, rootDevice: UInt64, displayPath: String, depth: Int = 0
    ) throws {
        try Task.checkCancellation()
        guard depth <= 128 else {
            throw ArtifactRemovalError.invalidTarget(path: displayPath)
        }
        for name in try entries(beneath: descriptor, displayPath: displayPath) {
            try Task.checkCancellation()
            let path = displayPath + "/" + name
            let item = try status(named: name, beneath: descriptor, displayPath: path)
            guard UInt64(item.st_dev) == rootDevice else {
                throw ArtifactRemovalError.crossedFilesystem(path: path)
            }
            if item.st_mode & S_IFMT == S_IFDIR {
                let child = try openDirectory(
                    named: name, beneath: descriptor, displayPath: path)
                do {
                    try removeContents(
                        beneath: child, rootDevice: rootDevice, displayPath: path,
                        depth: depth + 1)
                    _ = close(child)
                } catch {
                    _ = close(child)
                    throw error
                }
                try unlink(name: name, beneath: descriptor, directory: true, displayPath: path)
            } else {
                // `fstatat(..., AT_SYMLINK_NOFOLLOW)` established that this is a
                // leaf. `unlinkat` removes the directory entry and never follows
                // a symlink, including one introduced after preflight.
                try unlink(name: name, beneath: descriptor, directory: false, displayPath: path)
            }
        }
    }

    static func identity(
        named name: String, beneath descriptor: Int32, displayPath: String
    ) throws -> (device: UInt64, inode: UInt64) {
        let item = try status(named: name, beneath: descriptor, displayPath: displayPath)
        guard item.st_mode & S_IFMT == S_IFDIR else {
            throw ArtifactRemovalError.changed(path: displayPath)
        }
        return (UInt64(item.st_dev), UInt64(item.st_ino))
    }

    static func reverseSwap(
        leaf: String, quarantine: String, parent: Int32, displayPath: String
    ) throws {
        let result = leaf.withCString { leafPath in
            quarantine.withCString { quarantinePath in
                renameatx_np(
                    parent, leafPath, parent, quarantinePath, UInt32(RENAME_SWAP))
            }
        }
        guard result == 0 else {
            throw ArtifactRemovalError.cannotPrepare(path: displayPath, code: errno)
        }
    }

    static func unlink(
        name: String, beneath descriptor: Int32, directory: Bool, displayPath: String
    ) throws {
        let flags = directory ? AT_REMOVEDIR : 0
        while true {
            let result = name.withCString { unlinkat(descriptor, $0, flags) }
            if result == 0 { return }
            if errno == EINTR { continue }
            throw ArtifactRemovalError.cannotRemove(path: displayPath, code: errno)
        }
    }

    private static func status(
        named name: String, beneath descriptor: Int32, displayPath: String
    ) throws -> stat {
        var value = stat()
        let result = name.withCString {
            fstatat(descriptor, $0, &value, AT_SYMLINK_NOFOLLOW)
        }
        guard result == 0 else {
            throw ArtifactRemovalError.cannotRead(path: displayPath, code: errno)
        }
        return value
    }

    private static func openDirectory(
        named name: String, beneath descriptor: Int32, displayPath: String
    ) throws -> Int32 {
        let opened = name.withCString {
            openat(descriptor, $0, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
        }
        guard opened >= 0 else {
            throw ArtifactRemovalError.cannotRead(path: displayPath, code: errno)
        }
        return opened
    }

    private static func entries(beneath descriptor: Int32, displayPath: String) throws
        -> [String]
    {
        // `dup` would share the directory stream offset with the held
        // capability. A preflight census could then leave the removal pass at
        // end-of-directory and falsely conclude that the artifact was empty.
        // Opening `.` relative to the held fd creates a new open-file
        // description with an independent offset while remaining bound to the
        // same directory object.
        let copy = openat(
            descriptor, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
        guard copy >= 0 else {
            throw ArtifactRemovalError.cannotRead(path: displayPath, code: errno)
        }
        guard let directory = fdopendir(copy) else {
            let code = errno
            _ = close(copy)
            throw ArtifactRemovalError.cannotRead(path: displayPath, code: code)
        }
        defer { _ = closedir(directory) }

        var names: [String] = []
        errno = 0
        while let entry = readdir(directory) {
            let name = withUnsafePointer(to: &entry.pointee.d_name) { pointer in
                pointer.withMemoryRebound(to: CChar.self, capacity: Int(NAME_MAX) + 1) {
                    String(cString: $0)
                }
            }
            if name != "." && name != ".." { names.append(name) }
        }
        let code = errno
        guard code == 0 else {
            throw ArtifactRemovalError.cannotRead(path: displayPath, code: code)
        }
        return names.sorted()
    }
}

/// A regular file opened beneath a verified artifact root.
///
/// The descriptor, rather than the pathname, is the authority consumed by a
/// runtime. The object keeps it alive, supports bounded metadata reads, and can
/// revalidate its physical identity before the run emits a terminal result.
public final class ArtifactRootedFile: @unchecked Sendable {
    public let repositoryPath: String
    public let expectedSizeBytes: UInt64

    private let descriptor: Int32
    private let expectedIdentity: ArtifactFilesystemIdentity

    fileprivate init(
        repositoryPath: String, descriptor: Int32,
        expectedIdentity: ArtifactFilesystemIdentity
    ) {
        self.repositoryPath = repositoryPath
        self.descriptor = descriptor
        self.expectedIdentity = expectedIdentity
        self.expectedSizeBytes = expectedIdentity.sizeBytes
    }

    deinit { _ = close(descriptor) }

    /// Refuse if the held object changed after it was opened.
    public func validateCurrentIdentity() throws {
        let current = try ArtifactFilesystemEvidence.identity(
            of: descriptor, path: repositoryPath, expectedDirectory: false)
        guard current == expectedIdentity else {
            throw ArtifactFilesystemEvidenceError.changed(path: repositoryPath)
        }
    }

    /// Read a small verified metadata file without reopening its pathname.
    /// Large payloads use ``ArtifactRuntimeAuthority/openFileDescriptor(_:)``
    /// and are never materialized into `Data` by this adapter.
    public func readAll(maximumBytes: UInt64) throws -> Data {
        guard expectedSizeBytes <= maximumBytes,
            let count = Int(exactly: expectedSizeBytes)
        else {
            throw ArtifactFilesystemEvidenceError.wrongKind(
                path: repositoryPath,
                expected: "regular file no larger than \(maximumBytes) bytes")
        }
        var data = Data(count: count)
        var offset = 0
        while offset < count {
            let readCount = data.withUnsafeMutableBytes { buffer -> Int in
                guard let base = buffer.baseAddress else { return 0 }
                return pread(descriptor, base.advanced(by: offset), count - offset, off_t(offset))
            }
            guard readCount > 0 else {
                let code = readCount < 0 ? errno : EIO
                throw ArtifactFilesystemEvidenceError.cannotOpen(
                    path: repositoryPath, code: code)
            }
            offset += readCount
        }
        try validateCurrentIdentity()
        return data
    }
}

/// The exact full-verification evidence a model runtime may consume.
///
/// A scan result and a URL are insufficient: either can be stale by the time a
/// run starts. This capability carries the already-open artifact root plus the
/// physical identity of every file covered by the complete digest pass. Each
/// runtime file is opened with `openat(..., O_NOFOLLOW)` beneath that root and
/// must match the recorded object before it is handed to a loader.
public final class ArtifactRuntimeAuthority: @unchecked Sendable {
    public let rootURL: URL
    public let evidence: ArtifactVerificationEvidence

    private let root: ArtifactVerificationRoot
    private let expectedFiles: [String: ArtifactVerifiedFile]

    public init(
        root: ArtifactVerificationRoot, evidence: ArtifactVerificationEvidence
    ) throws {
        guard evidence.isStructurallyValid(for: .fullyVerified) else {
            throw ArtifactFilesystemEvidenceError.changed(path: root.rootURL.path)
        }
        guard try ArtifactFilesystemEvidence.persistentIDCapability(
            of: root.artifactDescriptor, path: root.rootURL.path) == true
        else {
            throw ArtifactFilesystemEvidenceError.persistentIDsUnavailable(
                path: root.rootURL.path)
        }
        if let storedVolume = evidence.persistentVolume {
            guard let currentVolume = try ArtifactFilesystemEvidence.persistentVolumeIdentity(
                of: root.artifactDescriptor, path: root.rootURL.path)
            else {
                throw ArtifactFilesystemEvidenceError.persistentIDsUnavailable(
                    path: root.rootURL.path)
            }
            guard storedVolume == currentVolume else {
                throw ArtifactFilesystemEvidenceError.changed(path: root.rootURL.path)
            }
        }
        guard ArtifactFilesystemEvidence.durableIdentityMatches(
            current: root.artifactIdentity, stored: evidence.root,
            persistentVolume: evidence.persistentVolume)
        else {
            throw ArtifactFilesystemEvidenceError.changed(path: root.rootURL.path)
        }
        try root.validateCurrentBinding()
        self.rootURL = root.rootURL
        self.root = root
        self.evidence = evidence
        self.expectedFiles = Dictionary(
            uniqueKeysWithValues: evidence.files.map { ($0.path, $0) })
    }

    public var verifiedRepositoryPaths: [String] {
        expectedFiles.keys.sorted()
    }

    /// Open one file whose exact digest and physical object were included in
    /// the full pass. Unknown paths cannot be introduced by an index or a
    /// manifest after verification.
    public func openFile(_ repositoryPath: String) throws -> ArtifactRootedFile {
        let opened = try openValidatedFileDescriptor(repositoryPath)
        return ArtifactRootedFile(
            repositoryPath: repositoryPath, descriptor: opened.descriptor,
            expectedIdentity: opened.currentIdentity)
    }

    /// Open one verified file and transfer ownership of its descriptor.
    ///
    /// This is the narrow bridge used by bounded model readers in another
    /// package module. It never opens a caller-supplied path: the reference
    /// must be in the complete digest evidence and is opened beneath the held
    /// artifact root with every component protected by `O_NOFOLLOW`.
    public func openFileDescriptor(_ repositoryPath: String) throws -> Int32 {
        try openValidatedFileDescriptor(repositoryPath).descriptor
    }

    private func openValidatedFileDescriptor(
        _ repositoryPath: String
    ) throws -> (descriptor: Int32, currentIdentity: ArtifactFilesystemIdentity) {
        guard let expected = expectedFiles[repositoryPath] else {
            throw ArtifactFilesystemEvidenceError.invalidPath(repositoryPath)
        }
        let descriptor = try ArtifactFilesystemEvidence.openPayload(
            repositoryPath, beneath: root.artifactDescriptor)
        do {
            let current = try ArtifactFilesystemEvidence.identity(
                of: descriptor, path: repositoryPath, expectedDirectory: false)
            guard current.device == root.artifactIdentity.device,
                ArtifactFilesystemEvidence.durableIdentityMatches(
                    current: current, stored: expected.filesystem,
                    persistentVolume: evidence.persistentVolume)
            else {
                throw ArtifactFilesystemEvidenceError.changed(path: repositoryPath)
            }
            return (descriptor, current)
        } catch {
            _ = close(descriptor)
            throw error
        }
    }

    /// Rebind the held root to the registered bookmark path before a terminal
    /// result may claim it consumed the verified artifact.
    public func validateCurrentBinding() throws {
        try root.validateCurrentBinding()
    }

    /// Revalidate the complete tree without retaining one descriptor per file.
    ///
    /// Keeping 471 descriptors alive would exceed the common 256-descriptor
    /// launchd soft limit before a K3 run opened its bounded readers. A final
    /// pass instead opens, identities and closes one evidence path at a time;
    /// the runner calls it immediately before a successful terminal event.
    public func validateAllFilesCurrent() throws {
        try root.validateCurrentBinding()
        for path in verifiedRepositoryPaths {
            let descriptor = try openFileDescriptor(path)
            _ = close(descriptor)
        }
        try root.validateCurrentBinding()
    }
}

/// `openat`-based filesystem authority shared by verification and ledger
/// revalidation. Every repository-relative descendant below the already-open
/// registered root, including the final component, is opened with `O_NOFOLLOW`,
/// so the object whose metadata is checked is the object whose descriptor is
/// hashed. The OS-resolved security-scoped bookmark is authority for the
/// registered root itself; this layer does not independently authenticate the
/// ancestor components of that OS-vended URL.
enum ArtifactFilesystemEvidence {
    private struct VolumeCapabilitiesBuffer {
        var length: UInt32 = 0
        var value = vol_capabilities_attr_t()
    }

    static func openRoot(at url: URL) throws -> Int32 {
        try url.withUnsafeFileSystemRepresentation { path in
            guard let path else {
                throw ArtifactFilesystemEvidenceError.invalidPath(url.path)
            }
            let descriptor = open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
            guard descriptor >= 0 else {
                throw ArtifactFilesystemEvidenceError.cannotOpen(path: url.path, code: errno)
            }
            return descriptor
        }
    }

    /// Duplicate a descriptor without leaking it through a future `exec`.
    static func duplicate(_ descriptor: Int32, path: String) throws -> Int32 {
        let copy = dup(descriptor)
        guard copy >= 0 else {
            throw ArtifactFilesystemEvidenceError.cannotOpen(path: path, code: errno)
        }
        guard fcntl(copy, F_SETFD, FD_CLOEXEC) == 0 else {
            let code = errno
            _ = close(copy)
            throw ArtifactFilesystemEvidenceError.cannotOpen(path: path, code: code)
        }
        return copy
    }

    /// Open a directory by walking a canonical relative path beneath an
    /// already-open directory. The returned descriptor is attached to the
    /// final object; every component is `O_NOFOLLOW`.
    static func openDirectory(
        _ relativeComponents: [String], beneath root: Int32, path: String
    ) throws -> Int32 {
        guard relativeComponents.allSatisfy({
            !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("/")
        }) else {
            throw ArtifactFilesystemEvidenceError.invalidPath(
                relativeComponents.joined(separator: "/"))
        }
        var directory = try duplicate(root, path: path)
        for component in relativeComponents {
            let next = component.withCString {
                openat(directory, $0, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
            }
            let code = errno
            guard next >= 0 else {
                _ = close(directory)
                throw ArtifactFilesystemEvidenceError.cannotOpen(
                    path: path, code: code)
            }
            _ = close(directory)
            directory = next
        }
        return directory
    }

    /// Couples `open` and the first `fstat`: if the latter fails, ownership is
    /// not leaked before the caller has a chance to install its own `defer`.
    static func openRootAndIdentity(
        at url: URL,
        identityProvider: (Int32, String) throws -> ArtifactFilesystemIdentity = {
            try identity(of: $0, path: $1, expectedDirectory: true)
        }
    ) throws -> (descriptor: Int32, identity: ArtifactFilesystemIdentity) {
        let descriptor = try openRoot(at: url)
        do {
            return (descriptor, try identityProvider(descriptor, url.path))
        } catch {
            _ = close(descriptor)
            throw error
        }
    }

    static func persistentIDCapability(at rootURL: URL) throws -> Bool? {
        let values = try rootURL.resourceValues(forKeys: [.volumeSupportsPersistentIDsKey])
        return values.volumeSupportsPersistentIDs
    }

    /// Query volume identity semantics from the descriptor whose objects will
    /// actually be evidenced. A path-based `URLResourceValues` query can race
    /// onto another mounted volume even while the verifier safely holds the
    /// original directory fd.
    static func persistentIDCapability(
        of descriptor: Int32, path: String
    ) throws -> Bool? {
        var attributes = attrlist()
        attributes.bitmapcount = UInt16(ATTR_BIT_MAP_COUNT)
        attributes.volattr = UInt32(ATTR_VOL_INFO) | UInt32(ATTR_VOL_CAPABILITIES)
        var buffer = VolumeCapabilitiesBuffer()
        guard fgetattrlist(
            descriptor, &attributes, &buffer,
            MemoryLayout<VolumeCapabilitiesBuffer>.size, 0) == 0
        else {
            throw ArtifactFilesystemEvidenceError.cannotStat(path: path, code: errno)
        }
        let bit = UInt32(VOL_CAP_FMT_PERSISTENTOBJECTIDS)
        guard buffer.value.valid.0 & bit != 0 else { return nil }
        return buffer.value.capabilities.0 & bit != 0
    }

    /// Read the volume UUID from the already-open artifact descriptor. The
    /// returned bytes are packed immediately after getattrlist's length word;
    /// using unaligned loads avoids relying on a Swift struct's padding.
    static func persistentVolumeIdentity(
        of descriptor: Int32, path: String
    ) throws -> ArtifactPersistentVolumeIdentity? {
        var attributes = attrlist()
        attributes.bitmapcount = UInt16(ATTR_BIT_MAP_COUNT)
        attributes.volattr = UInt32(ATTR_VOL_INFO) | UInt32(ATTR_VOL_UUID)
        var buffer = [UInt8](repeating: 0, count: MemoryLayout<UInt32>.size + 16)
        let result = buffer.withUnsafeMutableBytes { bytes in
            fgetattrlist(descriptor, &attributes, bytes.baseAddress, bytes.count, 0)
        }
        guard result == 0 else {
            throw ArtifactFilesystemEvidenceError.cannotStat(path: path, code: errno)
        }
        let returnedLength = buffer.withUnsafeBytes {
            $0.loadUnaligned(as: UInt32.self)
        }
        guard returnedLength >= UInt32(buffer.count) else { return nil }
        let uuidBytes = Array(buffer.dropFirst(MemoryLayout<UInt32>.size).prefix(16))
        guard uuidBytes.count == 16, uuidBytes.contains(where: { $0 != 0 }) else {
            return nil
        }
        let hex = uuidBytes.map { String(format: "%02x", $0) }
        let canonical = [
            hex[0...3].joined(), hex[4...5].joined(), hex[6...7].joined(),
            hex[8...9].joined(), hex[10...15].joined(),
        ].joined(separator: "-")
        return ArtifactPersistentVolumeIdentity(uuid: canonical)
    }

    static func durableIdentityMatches(
        current: ArtifactFilesystemIdentity,
        stored: ArtifactFilesystemIdentity,
        persistentVolume: ArtifactPersistentVolumeIdentity?
    ) -> Bool {
        guard persistentVolume != nil else { return current == stored }
        return current.inode == stored.inode
            && current.sizeBytes == stored.sizeBytes
            && current.modificationSeconds == stored.modificationSeconds
            && current.modificationNanoseconds == stored.modificationNanoseconds
            && current.statusChangeSeconds == stored.statusChangeSeconds
            && current.statusChangeNanoseconds == stored.statusChangeNanoseconds
    }

    static func openPayload(_ path: String, beneath root: Int32) throws -> Int32 {
        guard HuggingFaceTreeClient.isSafeRepositoryPath(path) else {
            throw ArtifactFilesystemEvidenceError.invalidPath(path)
        }
        let components = path.split(separator: "/").map(String.init)
        var directory = dup(root)
        guard directory >= 0 else {
            throw ArtifactFilesystemEvidenceError.cannotOpen(path: path, code: errno)
        }

        for component in components.dropLast() {
            let next = component.withCString {
                openat(directory, $0, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
            }
            let code = errno
            _ = close(directory)
            guard next >= 0 else {
                throw ArtifactFilesystemEvidenceError.cannotOpen(path: path, code: code)
            }
            directory = next
        }
        defer { _ = close(directory) }

        let descriptor = components.last!.withCString {
            // O_NONBLOCK prevents an untrusted non-regular object such as a
            // FIFO from hanging before `identity` can reject its kind.
            openat(directory, $0, O_RDONLY | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW)
        }
        guard descriptor >= 0 else {
            throw ArtifactFilesystemEvidenceError.cannotOpen(path: path, code: errno)
        }
        return descriptor
    }

    static func identity(
        of descriptor: Int32, path: String, expectedDirectory: Bool
    ) throws -> ArtifactFilesystemIdentity {
        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            throw ArtifactFilesystemEvidenceError.cannotStat(path: path, code: errno)
        }
        let kind = status.st_mode & S_IFMT
        if expectedDirectory {
            guard kind == S_IFDIR else {
                throw ArtifactFilesystemEvidenceError.wrongKind(path: path, expected: "directory")
            }
        } else {
            guard kind == S_IFREG else {
                throw ArtifactFilesystemEvidenceError.wrongKind(path: path, expected: "regular file")
            }
            guard status.st_nlink == 1 else {
                throw ArtifactFilesystemEvidenceError.multipleLinks(
                    path: path, count: UInt64(status.st_nlink))
            }
        }
        guard status.st_size >= 0 else {
            throw ArtifactFilesystemEvidenceError.negativeSize(path: path)
        }
        return ArtifactFilesystemIdentity(
            // `dev_t` is signed on Darwin even though its bits are an opaque
            // identifier. Preserve those bits instead of using a conversion
            // that could trap if the high bit is set.
            device: UInt64(bitPattern: Int64(status.st_dev)), inode: UInt64(status.st_ino),
            sizeBytes: UInt64(status.st_size),
            modificationSeconds: Int64(status.st_mtimespec.tv_sec),
            modificationNanoseconds: Int64(status.st_mtimespec.tv_nsec),
            statusChangeSeconds: Int64(status.st_ctimespec.tv_sec),
            statusChangeNanoseconds: Int64(status.st_ctimespec.tv_nsec))
    }

    static func validate(
        evidence: ArtifactVerificationEvidence, at rootURL: URL,
        hooks: ArtifactFilesystemValidationHooks = .none
    ) throws {
        let opened = try openRootAndIdentity(at: rootURL) { descriptor, path in
            try hooks.filesystemIdentity(descriptor, path, true)
        }
        let root = opened.descriptor
        defer { _ = close(root) }
        // Full verification asks the opened artifact directory whether its
        // filesystem supplies persistent identities. A later metadata-only
        // lookup must ask the same object through the same fd-shaped seam.
        // Re-querying the URL here is both a pathname race and, in an App
        // Sandbox, can produce different resource-value semantics from the
        // security-scoped object which was successfully opened.
        guard try hooks.persistentIDCapability(root, rootURL.path) == true else {
            throw ArtifactFilesystemEvidenceError.persistentIDsUnavailable(path: rootURL.path)
        }
        if let storedVolume = evidence.persistentVolume {
            guard let currentVolume = try hooks.persistentVolumeIdentity(root, rootURL.path) else {
                throw ArtifactFilesystemEvidenceError.persistentIDsUnavailable(
                    path: rootURL.path)
            }
            guard storedVolume == currentVolume else {
                throw ArtifactFilesystemEvidenceError.changed(path: rootURL.path)
            }
        }
        let rootBefore = opened.identity
        guard durableIdentityMatches(
            current: rootBefore, stored: evidence.root,
            persistentVolume: evidence.persistentVolume)
        else {
            throw ArtifactFilesystemEvidenceError.changed(path: rootURL.path)
        }

        try validate(
            evidence: evidence, rootDescriptor: root, rootPath: rootURL.path,
            hooks: hooks)

        // An open directory descriptor remains attached to the old directory
        // after a rename. Re-open the pathname so same-path replacement cannot
        // make the old descriptor look current.
        let current = try openRootAndIdentity(at: rootURL) { descriptor, path in
            try hooks.filesystemIdentity(descriptor, path, true)
        }
        defer { _ = close(current.descriptor) }
        let pathIdentity = current.identity
        guard pathIdentity == rootBefore else {
            throw ArtifactFilesystemEvidenceError.changed(path: rootURL.path)
        }
    }

    /// Validate evidence using a caller-owned root descriptor. This is the
    /// verifier's rooted-capability path: no URL is reopened for payloads.
    static func validate(
        evidence: ArtifactVerificationEvidence, rootDescriptor root: Int32,
        rootPath: String, hooks: ArtifactFilesystemValidationHooks = .none
    ) throws {
        guard try hooks.persistentIDCapability(root, rootPath) == true else {
            throw ArtifactFilesystemEvidenceError.persistentIDsUnavailable(path: rootPath)
        }
        if let storedVolume = evidence.persistentVolume {
            guard let currentVolume = try hooks.persistentVolumeIdentity(root, rootPath) else {
                throw ArtifactFilesystemEvidenceError.persistentIDsUnavailable(path: rootPath)
            }
            guard storedVolume == currentVolume else {
                throw ArtifactFilesystemEvidenceError.changed(path: rootPath)
            }
        }
        let rootBefore = try hooks.filesystemIdentity(root, rootPath, true)
        guard durableIdentityMatches(
            current: rootBefore, stored: evidence.root,
            persistentVolume: evidence.persistentVolume)
        else {
            throw ArtifactFilesystemEvidenceError.changed(path: rootPath)
        }

        func validateFiles(
            exactCurrentIdentities: [String: ArtifactFilesystemIdentity]? = nil
        ) throws -> [String: ArtifactFilesystemIdentity] {
            var seen: [ArtifactFilesystemObjectKey: String] = [:]
            var identities: [String: ArtifactFilesystemIdentity] = [:]
            for file in evidence.files {
                let descriptor = try openPayload(file.path, beneath: root)
                let current: ArtifactFilesystemIdentity
                do {
                    current = try hooks.filesystemIdentity(descriptor, file.path, false)
                } catch {
                    _ = close(descriptor)
                    throw error
                }
                _ = close(descriptor)
                let matchesFirstPass = exactCurrentIdentities.map {
                    $0[file.path] == current
                } ?? true
                guard current.device == rootBefore.device,
                    durableIdentityMatches(
                        current: current, stored: file.filesystem,
                        persistentVolume: evidence.persistentVolume),
                    matchesFirstPass
                else {
                    throw ArtifactFilesystemEvidenceError.changed(path: file.path)
                }
                let key = ArtifactFilesystemObjectKey(current)
                if let first = seen.updateValue(file.path, forKey: key) {
                    throw ArtifactFilesystemEvidenceError.aliased(
                        path: file.path, firstPath: first)
                }
                identities[file.path] = current
            }
            return identities
        }
        // Two complete passes close the hole where file A changes while a
        // later file B is being inspected. This remains point-in-time evidence;
        // the runtime must bind it again immediately before consuming bytes.
        let firstPass = try validateFiles()
        hooks.betweenIdentityPasses()
        _ = try validateFiles(exactCurrentIdentities: firstPass)

        let rootAfter = try hooks.filesystemIdentity(root, rootPath, true)
        guard rootAfter == rootBefore else {
            throw ArtifactFilesystemEvidenceError.changed(path: rootPath)
        }
    }
}
