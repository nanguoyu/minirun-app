import Foundation

/// How hard anybody has actually looked at these bytes.
///
/// Three states and not two, because "checked" is not one thing. Nothing here
/// is ever set by a scan: a scan counts files and sums their sizes, which is
/// metadata work, and a 1.56 TB digest pass must not start because a screen
/// appeared. Both non-default states are the result of an operator asking.
public enum ArtifactVerification: String, Codable, Sendable, CaseIterable {
    /// Nobody has digested anything here. What a scan always reports.
    case unverified
    /// A bounded sample of files was digested and matched.
    case spotChecked
    /// Every published payload and metadata file was digested and matched.
    case fullyVerified

    public var label: String {
        switch self {
        case .unverified: return "unverified"
        case .spotChecked: return "spot-checked"
        case .fullyVerified: return "verified"
        }
    }
}

/// A verification state remembered between launches, with the moment it was
/// established.
public struct ArtifactVerificationRecord: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 2

    public let schemaVersion: Int
    public let rootPath: String
    public let state: ArtifactVerification
    public let checkedAt: Date
    /// The operator-facing sentence from the check that produced this. Shown
    /// verbatim.
    public let detail: String
    /// Nil only for a legacy record or an explicitly unverified result. A
    /// positive state without evidence is decoded for migration and then
    /// downgraded by the ledger; it is never returned as verified.
    public let evidence: ArtifactVerificationEvidence?

    public init(
        rootPath: String, state: ArtifactVerification, checkedAt: Date, detail: String,
        evidence: ArtifactVerificationEvidence? = nil
    ) {
        self.schemaVersion = Self.currentSchemaVersion
        self.rootPath = rootPath
        self.state = state
        self.checkedAt = checkedAt
        self.detail = detail
        self.evidence = evidence
    }

    private init(
        schemaVersion: Int, rootPath: String, state: ArtifactVerification,
        checkedAt: Date, detail: String, evidence: ArtifactVerificationEvidence?
    ) {
        self.schemaVersion = schemaVersion
        self.rootPath = rootPath
        self.state = state
        self.checkedAt = checkedAt
        self.detail = detail
        self.evidence = evidence
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion, rootPath, state, checkedAt, detail, evidence
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            schemaVersion: try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1,
            rootPath: try container.decode(String.self, forKey: .rootPath),
            state: try container.decode(ArtifactVerification.self, forKey: .state),
            checkedAt: try container.decode(Date.self, forKey: .checkedAt),
            detail: try container.decode(String.self, forKey: .detail),
            evidence: try container.decodeIfPresent(
                ArtifactVerificationEvidence.self, forKey: .evidence))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(rootPath, forKey: .rootPath)
        try container.encode(state, forKey: .state)
        try container.encode(checkedAt, forKey: .checkedAt)
        try container.encode(detail, forKey: .detail)
        try container.encodeIfPresent(evidence, forKey: .evidence)
    }

    var needsEvidenceMigration: Bool {
        schemaVersion != Self.currentSchemaVersion
            || (state != .unverified
                && (evidence == nil
                    || evidence?.schemaVersion != ArtifactVerificationEvidence.currentSchemaVersion))
    }

    var migratedWithoutTrustingLegacyVerification: ArtifactVerificationRecord {
        ArtifactVerificationRecord(
            rootPath: rootPath, state: .unverified, checkedAt: checkedAt,
            detail: "Previous verification predates the current evidence authority; verify again.")
    }

    /// Whether this record answers one lookup, and — when it does not — whether
    /// the disagreement is grounds for destroying it.
    ///
    /// The question that decides between withholding and deleting is *whose*
    /// fact disagrees. A contradiction in the physical object is a fact about
    /// the volume: the bytes that were digested are no longer the bytes on
    /// disk, so the remembered result is worthless for every future lookup and
    /// the record goes. A disagreement with the catalog the caller happens to
    /// be holding is a fact about the *caller*: it says this record does not
    /// answer the question being asked right now, and says nothing whatever
    /// about the disk. Those records are withheld and kept, so the same lookup
    /// made with the right catalog — the live one, once it is fetched — finds
    /// the evidence still there.
    ///
    /// The two were one case until 2026-08-15, when both of these deleted
    /// complete full-verification evidence for two artifacts totalling 1.7 TB,
    /// whose 1,101 recorded file identities all still matched the volume
    /// exactly:
    ///
    /// - a launch with no cached catalog scanned under an empty snapshot, which
    ///   has no descriptor for anything, so every lookup arrived with
    ///   `repository: nil` — and with `model: nil` for any artifact whose index
    ///   the compiled-in upstream table does not happen to name;
    /// - the bundled catalog pins the V4 repository at an older revision than
    ///   both the evidence and the live catalog, so a catalog that was merely
    ///   behind the disk erased a record about the disk.
    ///
    /// Deleting on those inputs is not fail-closed, it is fail-destructive.
    /// Withholding costs one more verification pass if the caller was right;
    /// deleting costs a terabyte of re-reading when the caller was wrong.
    func applicability(to lookup: ArtifactVerificationLookup) -> ArtifactRecordApplicability {
        // Darwin vends the same iOS LiveFiles object through both
        // `/private/var/...` (security-scoped bookmark resolution) and
        // `/var/...` (the opened descriptor's canonical path). Those are
        // system aliases, not two artifact identities. Physical evidence
        // below still has to match every dev/inode/time tuple exactly.
        //
        // A record written by `record(_:)` is stored under the canonical form
        // of its own `rootPath`, and every storage alias of a path canonicalizes
        // to that same string, so a record reached through the ledger's key can
        // not fail this guard. Reaching it means the stored payload disagrees
        // with the key it was filed under — corrupt bookkeeping about which
        // directory was checked, not a catalog holding a different opinion.
        guard ArtifactVerificationPathIdentity.canonical(rootPath)
            == ArtifactVerificationPathIdentity.canonical(lookup.rootPath)
        else { return .contradicted }
        guard state != .unverified else { return .applies }
        // Structural invalidity is self-contradiction: no future catalog or
        // volume can make these bytes decode into usable evidence.
        guard let evidence, evidence.isStructurallyValid(for: state)
        else { return .contradicted }
        // Catalog-side. `model` and `repository` are what the caller currently
        // believes; an empty, stale or bundled snapshot supplies nil or an
        // older pinned revision without anything on disk having moved.
        //
        // `index` is disk-derived — the caller parsed `index.json` just now —
        // but it does not earn a delete either. For a full pass `index.json` is
        // in `evidence.files` and a genuine replacement is caught below as a
        // physical contradiction, so the delete path loses nothing. For a spot
        // check it can not be: `isStructurallyValid(for: .spotChecked)` requires
        // zero metadata entries, so no physical check covers `index.json` and
        // the inequality is unproven. An unreadable `index.json` also parses to
        // `.unreadable` and would compare unequal after a transient read
        // failure, which is the same class `cannotOpen` is already excused for.
        guard evidence.model == lookup.model,
            evidence.repository == lookup.repository,
            evidence.index == lookup.index
        else { return .notApplicable }
        // Not catalog-side: `currentTreeIdentity` is supplied only by the
        // verifier, only from its own fetch, and only for the pinned commit
        // this record already agreed with on the line above. A pinned commit is
        // immutable, so one commit presenting two trees contradicts the record
        // itself rather than describing a different question.
        if let currentTreeIdentity = lookup.currentTreeIdentity,
            evidence.treeIdentity != currentTreeIdentity
        {
            return .contradicted
        }
        do {
            try ArtifactFilesystemEvidence.validate(
                evidence: evidence, at: URL(fileURLWithPath: lookup.rootPath))
            return .applies
        } catch let error as ArtifactFilesystemEvidenceError {
            return error.invalidatesRememberedEvidence ? .contradicted : .temporarilyUnavailable
        } catch {
            // An unclassified I/O failure is not evidence that the remembered
            // physical object changed. Do not surface the positive state, but
            // retain it so a later successful identity pass can decide.
            return .temporarilyUnavailable
        }
    }
}

enum ArtifactRecordApplicability {
    /// This record describes this lookup and the physical objects still match.
    case applies
    /// The record is about a different question than this lookup asks. Withhold
    /// the state; keep the record.
    case notApplicable
    /// The record contradicts either the volume or itself. Delete it.
    case contradicted
    /// The answer could not be established right now — missing media, an
    /// unreadable path, a volume that will not report persistent identifiers.
    /// Withhold the state; keep the record.
    case temporarilyUnavailable
}

/// Where verification results live between launches.
public protocol ArtifactVerificationLedger: Sendable {
    func record(_ record: ArtifactVerificationRecord)
    /// Returns a positive record only after its model, pinned repository,
    /// digest-plan identity and current filesystem objects still match.
    ///
    /// Returning nil is not the same as forgetting. A record is removed only
    /// when it contradicts the volume or itself; a record that simply does not
    /// answer *this* caller's question is withheld and kept. See
    /// ``ArtifactRecordApplicability``.
    func record(matching lookup: ArtifactVerificationLookup) -> ArtifactVerificationRecord?
    func forget(rootPath: String)
}

public extension ArtifactVerificationLedger {
    /// This intentionally is not a protocol requirement. Keeping it solely as
    /// an extension preserves source compatibility while making calls through
    /// an `ArtifactVerificationLedger` existential or generic constraint
    /// dispatch here instead of to a legacy conformer's path-only authority.
    @available(*, deprecated, message: "Use record(matching:) with physical evidence")
    func record(forRootPath rootPath: String) -> ArtifactVerificationRecord? { nil }

    /// Old external conformers remain source-compatible, but their path-only
    /// records are not silently promoted to filesystem evidence.
    func record(matching lookup: ArtifactVerificationLookup) -> ArtifactVerificationRecord? { nil }
}

public struct UserDefaultsVerificationLedger: ArtifactVerificationLedger {
    /// Different value wrappers can address the same defaults domain (for
    /// example while a view refresh races a verifier). UserDefaults serializes
    /// individual calls, but the read/validate/remove transaction is ours, so
    /// all wrappers share one coordination lock.
    private static let coordinationLock = NSRecursiveLock()

    private final class DefaultsBox: @unchecked Sendable {
        let defaults: UserDefaults
        init(_ defaults: UserDefaults) { self.defaults = defaults }
    }

    private let box: DefaultsBox
    private let prefix: String
    private let hooks: UserDefaultsVerificationLedgerHooks

    public init(defaults: UserDefaults = .standard, prefix: String = "minirun.kit.verified.") {
        self.box = DefaultsBox(defaults)
        self.prefix = prefix
        self.hooks = .none
    }

    init(
        defaults: UserDefaults, prefix: String,
        hooks: UserDefaultsVerificationLedgerHooks
    ) {
        self.box = DefaultsBox(defaults)
        self.prefix = prefix
        self.hooks = hooks
    }

    private struct StoredSnapshot {
        let path: String
        let data: Data
    }

    private func key(_ path: String) -> String { prefix + path }

    private func snapshot(for path: String) -> StoredSnapshot? {
        Self.coordinationLock.lock()
        defer { Self.coordinationLock.unlock() }
        for candidate in ArtifactVerificationPathIdentity.storageAliases(for: path) {
            if let data = box.defaults.data(forKey: key(candidate)) {
                return StoredSnapshot(path: candidate, data: data)
            }
        }
        return nil
    }

    private func isCurrent(_ snapshot: StoredSnapshot) -> Bool {
        Self.coordinationLock.lock()
        defer { Self.coordinationLock.unlock() }
        return box.defaults.data(forKey: key(snapshot.path)) == snapshot.data
    }

    /// Compare and mutate as one transaction across every wrapper addressing
    /// this defaults domain. `NSRecursiveLock` is deliberate: UserDefaults may
    /// synchronously notify an observer that re-enters this ledger while
    /// `set`/`removeObject` is still on the stack.
    private func compareAndReplace(
        _ snapshot: StoredSnapshot, with replacement: Data?
    ) -> Bool {
        hooks.beforeCompareAndSwap(snapshot.path)
        Self.coordinationLock.lock()
        defer { Self.coordinationLock.unlock() }
        guard box.defaults.data(forKey: key(snapshot.path)) == snapshot.data else { return false }
        if let replacement {
            box.defaults.set(replacement, forKey: key(snapshot.path))
        } else {
            box.defaults.removeObject(forKey: key(snapshot.path))
        }
        // Verification can finish immediately before the user quits. This is
        // security evidence, not an eventually-consistent UI preference: make
        // the accepted mutation visible to the preferences daemon before
        // reporting success to the caller.
        box.defaults.synchronize()
        return true
    }

    public func record(_ record: ArtifactVerificationRecord) {
        guard let data = try? MinirunKitJSON.encoder(pretty: false).encode(record) else { return }
        Self.coordinationLock.lock()
        defer { Self.coordinationLock.unlock() }
        let aliases = ArtifactVerificationPathIdentity.storageAliases(for: record.rootPath)
        let canonicalPath = ArtifactVerificationPathIdentity.canonical(record.rootPath)
        box.defaults.set(data, forKey: key(canonicalPath))
        // A legacy alias must not survive beside a newer record and later
        // reappear after the current record is invalidated.
        for alternate in aliases where alternate != canonicalPath {
            box.defaults.removeObject(forKey: key(alternate))
        }
        box.defaults.synchronize()
    }

    @available(*, deprecated, message: "Use record(matching:) with physical evidence")
    public func record(forRootPath rootPath: String) -> ArtifactVerificationRecord? {
        guard let snapshot = snapshot(for: rootPath) else { return nil }
        guard let record = try? MinirunKitJSON.decoder().decode(
            ArtifactVerificationRecord.self, from: snapshot.data),
            record.state == .unverified
        else { return nil }
        return record
    }

    public func record(matching lookup: ArtifactVerificationLookup) -> ArtifactVerificationRecord? {
        // Snapshot under the lock, then do decoding and potentially hundreds
        // of open/fstat calls without serialising unrelated ledgers.
        guard let snapshot = snapshot(for: lookup.rootPath),
            let decoded = try? MinirunKitJSON.decoder().decode(
                ArtifactVerificationRecord.self, from: snapshot.data)
        else { return nil }
        if decoded.needsEvidenceMigration {
            let migrated = decoded.migratedWithoutTrustingLegacyVerification
            guard let migratedData = try? MinirunKitJSON.encoder(pretty: false).encode(migrated)
            else { return nil }
            return compareAndReplace(snapshot, with: migratedData)
                ? migrated : nil
        }
        switch decoded.applicability(to: lookup) {
        case .contradicted:
            _ = compareAndReplace(snapshot, with: nil)
            return nil
        case .notApplicable, .temporarilyUnavailable:
            // Withheld, not erased. The next lookup that asks the question this
            // record answers must still find it.
            return nil
        case .applies:
            guard isCurrent(snapshot) else { return nil }
            return decoded
        }
    }

    public func forget(rootPath: String) {
        Self.coordinationLock.lock()
        defer { Self.coordinationLock.unlock() }
        for path in ArtifactVerificationPathIdentity.storageAliases(for: rootPath) {
            box.defaults.removeObject(forKey: key(path))
        }
        box.defaults.synchronize()
    }
}

struct UserDefaultsVerificationLedgerHooks: Sendable {
    var beforeCompareAndSwap: @Sendable (String) -> Void

    static let none = UserDefaultsVerificationLedgerHooks(beforeCompareAndSwap: { _ in })
}

public final class InMemoryVerificationLedger: ArtifactVerificationLedger, @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String: ArtifactVerificationRecord] = [:]

    public init() {}

    public func record(_ record: ArtifactVerificationRecord) {
        lock.lock()
        defer { lock.unlock() }
        storage[ArtifactVerificationPathIdentity.canonical(record.rootPath)] = record
    }

    @available(*, deprecated, message: "Use record(matching:) with physical evidence")
    public func record(forRootPath rootPath: String) -> ArtifactVerificationRecord? {
        lock.lock()
        defer { lock.unlock() }
        guard let record = storage[ArtifactVerificationPathIdentity.canonical(rootPath)],
            record.state == .unverified
        else { return nil }
        return record
    }

    public func record(matching lookup: ArtifactVerificationLookup) -> ArtifactVerificationRecord? {
        let canonicalPath = ArtifactVerificationPathIdentity.canonical(lookup.rootPath)
        lock.lock()
        guard let decoded = storage[canonicalPath] else {
            lock.unlock()
            return nil
        }
        lock.unlock()
        if decoded.needsEvidenceMigration {
            let migrated = decoded.migratedWithoutTrustingLegacyVerification
            lock.lock()
            defer { lock.unlock() }
            guard storage[canonicalPath] == decoded else { return nil }
            storage[canonicalPath] = migrated
            return migrated
        }
        switch decoded.applicability(to: lookup) {
        case .contradicted:
            lock.lock()
            defer { lock.unlock() }
            if storage[canonicalPath] == decoded {
                storage.removeValue(forKey: canonicalPath)
            }
            return nil
        case .notApplicable, .temporarilyUnavailable:
            return nil
        case .applies:
            lock.lock()
            defer { lock.unlock() }
            guard storage[canonicalPath] == decoded else { return nil }
            return decoded
        }
    }

    public func forget(rootPath: String) {
        lock.lock()
        defer { lock.unlock() }
        storage.removeValue(forKey: ArtifactVerificationPathIdentity.canonical(rootPath))
    }
}

/// Stable lexical identity for Darwin's system-provided `/var` alias.
///
/// This deliberately does not resolve arbitrary symlinks: a user-controlled
/// path alias must never inherit verification authority. Apple exposes the
/// same iOS security-scoped LiveFiles directory as `/private/var/...` when a
/// bookmark resolves and `/var/...` when an opened descriptor reports its
/// path. Only that fixed system alias is normalized here; dev/inode evidence
/// remains the actual authority.
enum ArtifactVerificationPathIdentity {
    private static let privateVar = "/private/var"
    private static let publicVar = "/var"

    static func canonical(_ path: String) -> String {
        // `standardizedFileURL` consults the filesystem and can collapse
        // arbitrary symlinks. `standardized` is deliberately lexical: only
        // `.` and `..` are removed before the fixed Darwin alias below.
        let standardized = URL(fileURLWithPath: path).standardized.path
        if standardized == privateVar { return publicVar }
        if standardized.hasPrefix(privateVar + "/") {
            return String(standardized.dropFirst("/private".count))
        }
        return standardized
    }

    /// The caller's spelling first, followed by its one equivalent Darwin
    /// spelling. Reads preserve the actual stored key for compare-and-swap;
    /// new records converge on ``canonical(_:)``.
    static func storageAliases(for path: String) -> [String] {
        let standardized = URL(fileURLWithPath: path).standardized.path
        let canonical = canonical(path)
        var aliases = [standardized]
        if canonical == publicVar || canonical.hasPrefix(publicVar + "/") {
            let privateAlias = "/private" + canonical
            let alternate = standardized == privateAlias ? canonical : privateAlias
            if alternate != standardized { aliases.append(alternate) }
        }
        return aliases
    }
}

/// One artifact root found on disk.
///
/// Installed-ness is a fact about a directory, so everything here is measured
/// from the directory: the bytes are the sum of the file sizes actually present,
/// the count is the files actually present, and the catalog only supplies what
/// was *expected*. Nothing is inferred from a registry, because a registry
/// describes what was once downloaded rather than what is currently on the
/// volume.
public struct DiscoveredArtifact: Sendable, Equatable, Identifiable {
    public var id: String { rootPath }
    public let rootPath: String
    /// The location this artifact was found under.
    public let locationPath: String
    public let index: ArtifactIndexIdentity
    /// Nil when nothing in `index.json` matched the catalog. The artifact is
    /// still listed — an unidentified artifact taking up 60 GB is worth seeing.
    public let model: ModelID?
    public let displayName: String
    /// Sum of the sizes of every file under the root, hidden bookkeeping
    /// excluded. Metadata only: nothing is opened or read.
    public let bytesOnDisk: UInt64
    public let fileCount: Int
    /// True when filesystem metadata exceeded the numeric range above. The
    /// displayed total is then saturated, and no completeness claim is made.
    public let metadataOverflowed: Bool
    /// What the catalog says a complete artifact holds. Nil for an unidentified
    /// artifact, and for a model the catalog has no entry for.
    public let expectedFileCount: Int?
    public let expectedBytes: UInt64?
    public let verification: ArtifactVerification
    public let verifiedAt: Date?
    public let scannedAt: Date

    public init(
        rootPath: String, locationPath: String, index: ArtifactIndexIdentity, model: ModelID?,
        displayName: String, bytesOnDisk: UInt64, fileCount: Int, expectedFileCount: Int?,
        expectedBytes: UInt64?, verification: ArtifactVerification, verifiedAt: Date?,
        scannedAt: Date, metadataOverflowed: Bool = false
    ) {
        self.rootPath = rootPath
        self.locationPath = locationPath
        self.index = index
        self.model = model
        self.displayName = displayName
        self.bytesOnDisk = bytesOnDisk
        self.fileCount = fileCount
        self.metadataOverflowed = metadataOverflowed
        self.expectedFileCount = expectedFileCount
        self.expectedBytes = expectedBytes
        self.verification = verification
        self.verifiedAt = verifiedAt
        self.scannedAt = scannedAt
    }

    public var url: URL { URL(fileURLWithPath: rootPath) }

    public var isIdentified: Bool { model != nil }

    /// Files the catalog expected and the directory does not hold. Negative
    /// means the directory holds more than the catalog expected, which is news
    /// rather than an error. Nil when there is no expectation to compare with.
    public var missingFileCount: Int? {
        guard let expectedFileCount else { return nil }
        return expectedFileCount - fileCount
    }

    /// Every expected file is present. Nil when nothing is expected — a
    /// completeness verdict with no expectation behind it would be a guess.
    ///
    /// This is a *count*, not a digest: it says the directory holds as many
    /// files as the catalog names, and says nothing whatever about their
    /// contents. That is what ``verification`` is for.
    public var isComplete: Bool? {
        guard !metadataOverflowed else { return nil }
        guard let missing = missingFileCount else { return nil }
        return missing <= 0
    }

    /// Fraction of the expected bytes present, clamped to 0…1. Nil without an
    /// expectation.
    public var fractionOnDisk: Double? {
        guard !metadataOverflowed else { return nil }
        guard let expectedBytes, expectedBytes > 0 else { return nil }
        return min(1, Double(bytesOnDisk) / Double(expectedBytes))
    }
}

/// One registered storage location, and what a scan of it found.
public struct LocationScan: Sendable, Equatable, Identifiable {
    public var id: String { rootPath }
    public let rootPath: String
    /// The volume name where there is one, else the directory's last component.
    public let displayName: String
    /// The volume is present and the directory could be read. A location on an
    /// ejected drive is reported, not dropped: the operator asked for it to be
    /// remembered.
    public let isMounted: Bool
    public let storageKey: StorageKey?
    public let artifacts: [DiscoveredArtifact]
    /// Path → the reason it could not be walked. Surfaced, never swallowed.
    public let unreadable: [String: String]
    public let scannedAt: Date

    public init(
        rootPath: String, displayName: String, isMounted: Bool, storageKey: StorageKey?,
        artifacts: [DiscoveredArtifact], unreadable: [String: String], scannedAt: Date
    ) {
        self.rootPath = rootPath
        self.displayName = displayName
        self.isMounted = isMounted
        self.storageKey = storageKey
        self.artifacts = artifacts
        self.unreadable = unreadable
        self.scannedAt = scannedAt
    }

    public func artifacts(for model: ModelID) -> [DiscoveredArtifact] {
        artifacts.filter { $0.model == model }
    }
}

/// Every location's scan, and the questions the UI actually asks of them.
public struct DiscoveryReport: Sendable, Equatable {
    public let locations: [LocationScan]
    public let scannedAt: Date

    public init(locations: [LocationScan], scannedAt: Date) {
        self.locations = locations
        self.scannedAt = scannedAt
    }

    public static let empty = DiscoveryReport(locations: [], scannedAt: .distantPast)

    /// Every place this model was found, mounted or not.
    public func installations(of model: ModelID) -> [DiscoveredArtifact] {
        locations.flatMap { $0.artifacts(for: model) }
    }

    /// The model has an artifact somewhere, mounted or not.
    public func isInstalled(_ model: ModelID) -> Bool {
        !installations(of: model).isEmpty
    }

    /// The artifact is present **and** the volume holding it is mounted right
    /// now. This is the only sense in which a model can be run here, and it is
    /// deliberately not the same question as "is it installed": a 1.5 TB
    /// artifact on an unplugged drive is installed and unrunnable.
    public func isRunnableHere(_ model: ModelID) -> Bool {
        locations.contains { $0.isMounted && !$0.artifacts(for: model).isEmpty }
    }

    public var unidentifiedArtifacts: [DiscoveredArtifact] {
        locations.flatMap { $0.artifacts }.filter { !$0.isIdentified }
    }
}
