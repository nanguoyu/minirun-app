import Foundation
import StorageCore
import iOSBench

/// A name a remembered location is filed under.
public struct StorageKey: RawRepresentable, Hashable, Codable, Sendable {
    public let rawValue: String
    public init(rawValue: String) { self.rawValue = rawValue }
    public init(_ rawValue: String) { self.rawValue = rawValue }

    /// Where artifacts are read from.
    public static let artifactRoot = StorageKey("artifact-root")
}

/// Free space on a volume, with the optimistic number and the dependable one
/// kept apart.
public struct FreeSpace: Sendable, Codable, Equatable {
    /// `volumeAvailableCapacityForImportantUsage`. Includes space the system
    /// would purge if pressed. Reported to the operator, and never consulted by
    /// ``StorageManaging/canHold(bytes:at:headroomBytes:)`` — a terabyte that
    /// only fits because the system might evict somebody else's cache is not a
    /// terabyte that fits.
    public let optimisticBytes: Int64?
    /// `volumeAvailableCapacity`. The only number a decision uses.
    public let dependableBytes: Int64?
    public let totalBytes: Int64?
    public let volumeName: String?
    public let isReadOnly: Bool
    public let isRemovable: Bool

    public init(
        optimisticBytes: Int64?, dependableBytes: Int64?, totalBytes: Int64?,
        volumeName: String?, isReadOnly: Bool, isRemovable: Bool
    ) {
        self.optimisticBytes = optimisticBytes
        self.dependableBytes = dependableBytes
        self.totalBytes = totalBytes
        self.volumeName = volumeName
        self.isReadOnly = isReadOnly
        self.isRemovable = isRemovable
    }

    public init(_ info: StorageInfo) {
        self.init(
            optimisticBytes: info.volumeAvailableBytesForImportantUsage,
            dependableBytes: info.volumeAvailableBytes,
            totalBytes: info.volumeTotalBytes,
            volumeName: info.volumeName,
            isReadOnly: info.volumeIsReadOnly ?? false,
            isRemovable: info.volumeIsRemovable ?? false)
    }
}

/// Whether a volume can take a given number of bytes.
public enum SpaceVerdict: Sendable, Equatable {
    case fits(spareBytes: Int64)
    case refused(DownloadError)
    /// The volume could not be probed. The caller must not proceed — "unknown"
    /// is not "fine", and a download that starts on an unprobeable volume finds
    /// out how much room there was by running out of it.
    case unknown(reason: String)

    public var isFit: Bool { if case .fits = self { return true } else { return false } }
}

/// A mounted volume, and whether this build will write to it.
public struct VolumeDescriptor: Sendable, Codable, Equatable, Identifiable {
    public var id: String { mountPath }
    public let mountPath: String
    public let name: String?
    public let filesystemType: String?
    public let space: FreeSpace
    public let isInternal: Bool?
    public let isEjectable: Bool?
    /// Non-nil when this build refuses to write here, with the reason shown
    /// verbatim.
    public let writeRefusal: String?

    public init(
        mountPath: String, name: String?, filesystemType: String?, space: FreeSpace,
        isInternal: Bool?, isEjectable: Bool?, writeRefusal: String?
    ) {
        self.mountPath = mountPath
        self.name = name
        self.filesystemType = filesystemType
        self.space = space
        self.isInternal = isInternal
        self.isEjectable = isEjectable
        self.writeRefusal = writeRefusal
    }

    public var isWritable: Bool { writeRefusal == nil }
}

/// The ledger entry for one remembered location.
public struct StorageBookmark: Codable, Sendable, Equatable {
    public let key: StorageKey
    public let createdAt: Date
    public let recordedPath: String
    public let byteCount: Int
    /// Where the bookmark last resolved to.
    ///
    /// Kept because M10 found the iOS `userfsd` mount UUID stable across a
    /// replug *and* a reboot. A path that resolved to something different is
    /// therefore news about the volume, not routine drift, and it is surfaced
    /// rather than smoothed over.
    public let lastResolvedPath: String?
    public let lastResolvedAt: Date?
    public let resolveCount: Int
    public let staleRewriteCount: Int

    public init(
        key: StorageKey, createdAt: Date, recordedPath: String, byteCount: Int,
        lastResolvedPath: String? = nil, lastResolvedAt: Date? = nil,
        resolveCount: Int = 0, staleRewriteCount: Int = 0
    ) {
        self.key = key
        self.createdAt = createdAt
        self.recordedPath = recordedPath
        self.byteCount = byteCount
        self.lastResolvedPath = lastResolvedPath
        self.lastResolvedAt = lastResolvedAt
        self.resolveCount = resolveCount
        self.staleRewriteCount = staleRewriteCount
    }

    /// Whether the last resolution landed somewhere other than the path that
    /// was recorded.
    public var pathMoved: Bool {
        guard let lastResolvedPath else { return false }
        return lastResolvedPath != recordedPath
    }
}

public enum StorageError: Error, Equatable, CustomStringConvertible {
    case noBookmark(StorageKey)
    case bookmarkUnresolvable(StorageKey, reason: String)
    case accessRefused(path: String)
    case bookmarkCreationFailed(path: String, reason: String)
    case bookmarkDataUnresolvable(reason: String)
    case bookmarkRefreshFailed(path: String, reason: String)
    case volumeUnavailable(path: String)
    case scopeAlreadyReleased(path: String)

    public var description: String {
        switch self {
        case .noBookmark(let key):
            return "nothing is remembered under '\(key.rawValue)'"
        case .bookmarkUnresolvable(let key, let reason):
            return "'\(key.rawValue)' could not be resolved: \(reason)"
        case .accessRefused(let path):
            return "the sandbox refused access to '\(path)'"
        case .bookmarkCreationFailed(let path, let reason):
            return "could not bookmark '\(path)': \(reason)"
        case .bookmarkDataUnresolvable(let reason):
            return "the saved destination bookmark could not be resolved: \(reason)"
        case .bookmarkRefreshFailed(let path, let reason):
            return "the stale destination bookmark for '\(path)' could not be refreshed: \(reason)"
        case .volumeUnavailable(let path):
            return "'\(path)' is not mounted"
        case .scopeAlreadyReleased(let path):
            return "the scope over '\(path)' was already released"
        }
    }

    /// Each `SecurityScopedBookmarkError` maps to exactly one case here, so a
    /// caller never has to catch two error types for one failure.
    public init(_ error: SecurityScopedBookmarkError, key: StorageKey) {
        switch error {
        case .noStoredBookmark:
            self = .noBookmark(key)
        case .resolutionFailed(_, let reason):
            self = .bookmarkUnresolvable(key, reason: reason)
        case .accessRefused(let path):
            self = .accessRefused(path: path)
        case .bookmarkCreationFailed(let path, let reason):
            self = .bookmarkCreationFailed(path: path, reason: reason)
        }
    }
}

/// A raw per-job bookmark resolved under a live access grant.
///
/// `bookmarkData` is the original blob when it was fresh and the replacement
/// blob when Foundation reported it stale. The caller persists that value
/// before it relies on the job surviving another process death.
public struct ResolvedStorageBookmark: Sendable {
    public let scope: StorageScope
    public let bookmarkData: Data
    public let wasStale: Bool

    public init(scope: StorageScope, bookmarkData: Data, wasStale: Bool) {
        self.scope = scope
        self.bookmarkData = bookmarkData
        self.wasStale = wasStale
    }
}

/// Where ledger entries live between launches.
public protocol BookmarkLedger: Sendable {
    func record(_ bookmark: StorageBookmark)
    func bookmark(_ key: StorageKey) -> StorageBookmark?
    func keys() -> [StorageKey]
    func forget(_ key: StorageKey)
}

public struct UserDefaultsBookmarkLedger: BookmarkLedger {
    /// `UserDefaults` is documented as thread-safe but is not marked
    /// `Sendable`, so it travels in a box that says so out loud rather than
    /// silencing the checker on the whole type.
    private final class DefaultsBox: @unchecked Sendable {
        let defaults: UserDefaults
        init(_ defaults: UserDefaults) { self.defaults = defaults }
    }

    private let box: DefaultsBox
    private let prefix: String
    private let indexKey: String

    private var defaults: UserDefaults { box.defaults }

    public init(defaults: UserDefaults = .standard, prefix: String = "minirun.kit.ledger.") {
        self.box = DefaultsBox(defaults)
        self.prefix = prefix
        self.indexKey = prefix + "__keys"
    }

    public func record(_ bookmark: StorageBookmark) {
        guard let data = try? MinirunKitJSON.encoder(pretty: false).encode(bookmark) else { return }
        defaults.set(data, forKey: prefix + bookmark.key.rawValue)
        var known = Set(defaults.stringArray(forKey: indexKey) ?? [])
        known.insert(bookmark.key.rawValue)
        defaults.set(known.sorted(), forKey: indexKey)
    }

    public func bookmark(_ key: StorageKey) -> StorageBookmark? {
        guard let data = defaults.data(forKey: prefix + key.rawValue) else { return nil }
        return try? MinirunKitJSON.decoder().decode(StorageBookmark.self, from: data)
    }

    public func keys() -> [StorageKey] {
        (defaults.stringArray(forKey: indexKey) ?? []).map(StorageKey.init(rawValue:))
    }

    public func forget(_ key: StorageKey) {
        defaults.removeObject(forKey: prefix + key.rawValue)
        var known = Set(defaults.stringArray(forKey: indexKey) ?? [])
        known.remove(key.rawValue)
        defaults.set(known.sorted(), forKey: indexKey)
    }
}

/// A ledger with no side effects outside the instance.
public final class InMemoryBookmarkLedger: BookmarkLedger, @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [StorageKey: StorageBookmark] = [:]

    public init() {}

    public func record(_ bookmark: StorageBookmark) {
        lock.lock()
        defer { lock.unlock() }
        storage[bookmark.key] = bookmark
    }

    public func bookmark(_ key: StorageKey) -> StorageBookmark? {
        lock.lock()
        defer { lock.unlock() }
        return storage[key]
    }

    public func keys() -> [StorageKey] {
        lock.lock()
        defer { lock.unlock() }
        return storage.keys.sorted { $0.rawValue < $1.rawValue }
    }

    public func forget(_ key: StorageKey) {
        lock.lock()
        defer { lock.unlock() }
        storage.removeValue(forKey: key)
    }
}

public protocol StorageManaging: Sendable {
    func volumes() throws -> [VolumeDescriptor]
    func describe(_ url: URL) -> StorageInfo
    func freeSpace(at url: URL) -> FreeSpace
    func canHold(bytes: UInt64, at url: URL, headroomBytes: UInt64) -> SpaceVerdict
    /// A scope over a URL the user just picked, before anything is remembered.
    func scope(for url: URL) throws -> StorageScope
    @discardableResult
    func remember(_ url: URL, as key: StorageKey) throws -> StorageBookmark
    func resolve(_ key: StorageKey) throws -> StorageScope
    func bookmark(_ key: StorageKey) -> StorageBookmark?
    func forget(_ key: StorageKey)
    func knownLocations() -> [StorageKey]
    /// Raw bookmark data for a URL, so a long-lived job can carry its *own*
    /// destination across a relaunch rather than sharing hidden global state.
    func bookmarkData(for url: URL) throws -> Data
    /// Resolve one job's raw bookmark and hold its security scope. A saved
    /// bookmark that cannot be resolved or granted is an error; callers must
    /// never fall back to its recorded path.
    func resolveBookmarkData(_ data: Data) throws -> ResolvedStorageBookmark
}

extension StorageManaging {
    public func resolveBookmarkData(_ data: Data) throws -> ResolvedStorageBookmark {
        throw StorageError.bookmarkDataUnresolvable(
            reason: "this storage provider does not support raw bookmark resolution")
    }
}

/// Volume facts and remembered locations.
///
/// **Wraps** `iOSBench.SecurityScopedLocationStore`; it does not reimplement
/// bookmarking. Everything about `.withSecurityScope` being macOS-only, about
/// resolving a stale bookmark inside an access bracket, and about pairing
/// `stop` only with a successful `start` already lives there and was already
/// verified on the device. What this type adds is the ledger — what was
/// remembered, when it last resolved, and to where — and ``StorageScope``,
/// which is the one new mechanism, needed because a bracket cannot span an
/// async transfer.
public struct StorageManager: StorageManaging {
    private final class Box: @unchecked Sendable {
        let store: SecurityScopedLocationStore
        let coding: any SecurityScopedURLCoding
        init(store: SecurityScopedLocationStore, coding: any SecurityScopedURLCoding) {
            self.store = store
            self.coding = coding
        }
    }

    private let box: Box
    private let ledger: any BookmarkLedger

    public init(
        store: SecurityScopedLocationStore = SecurityScopedLocationStore(
            persistence: UserDefaultsBookmarkPersistence(prefix: "minirun.kit.location.")),
        bookmarkLedger: any BookmarkLedger = UserDefaultsBookmarkLedger(),
        coding: any SecurityScopedURLCoding = SystemSecurityScopedURLCoding()
    ) {
        self.box = Box(store: store, coding: coding)
        self.ledger = bookmarkLedger
    }

    // MARK: - Volumes

    public func volumes() throws -> [VolumeDescriptor] {
        let keys: [URLResourceKey] = [
            .volumeNameKey, .volumeTypeNameKey, .volumeIsInternalKey, .volumeIsEjectableKey,
            .volumeIsReadOnlyKey,
        ]
        let mounted =
            FileManager.default.mountedVolumeURLs(
                includingResourceValuesForKeys: keys,
                options: [.skipHiddenVolumes]) ?? []
        return
            mounted
            .map { url -> VolumeDescriptor in
                let info = StorageInfo.describing(path: url.path)
                let values = try? url.resourceValues(forKeys: Set(keys))
                let readOnly = values?.volumeIsReadOnly ?? info.volumeIsReadOnly ?? false
                let refusal = readOnly ? "the volume is mounted read-only." : nil
                return VolumeDescriptor(
                    mountPath: url.path,
                    name: values?.volumeName ?? info.volumeName,
                    filesystemType: values?.volumeTypeName ?? info.filesystemType,
                    space: FreeSpace(info),
                    isInternal: values?.volumeIsInternal ?? info.volumeIsInternal,
                    isEjectable: values?.volumeIsEjectable ?? info.volumeIsEjectable,
                    writeRefusal: refusal)
            }
            .sorted { $0.mountPath < $1.mountPath }
    }

    public func describe(_ url: URL) -> StorageInfo {
        StorageInfo.describing(path: url.path)
    }

    public func freeSpace(at url: URL) -> FreeSpace {
        FreeSpace(describe(url))
    }

    public func canHold(bytes: UInt64, at url: URL, headroomBytes: UInt64) -> SpaceVerdict {
        let info = describe(url)
        if info.volumeIsReadOnly == true {
            return .refused(.destinationIsReadOnlyVolume(path: info.resolvedPath))
        }
        guard let dependable = info.volumeAvailableBytes else {
            return .unknown(
                reason:
                    "the volume behind '\(info.resolvedPath)' reported no available capacity, so its free space is not known")
        }
        // Keep untrusted public inputs unsigned until they have been proved to
        // fit the signed capacity returned by the filesystem. `Int64(bitPattern:)`
        // would turn UInt64.max into -1, making an impossible request look as if
        // it fit with room to spare; unchecked addition can likewise wrap before
        // the comparison. A negative filesystem capacity is not usable evidence.
        guard dependable >= 0 else {
            return .unknown(
                reason:
                    "the volume behind '\(info.resolvedPath)' reported a negative available capacity, so its free space is not usable")
        }
        let (need, overflowed) = bytes.addingReportingOverflow(headroomBytes)
        guard !overflowed, need <= UInt64(dependable) else {
            return .refused(
                .insufficientFreeSpace(
                    needBytes: bytes, headroomBytes: headroomBytes,
                    availableBytes: dependable, volume: info.volumeName))
        }
        return .fits(spareBytes: dependable - Int64(need))
    }

    // MARK: - Remembered locations

    public func scope(for url: URL) throws -> StorageScope {
        StorageScope(url: url, coding: box.coding)
    }

    @discardableResult
    public func remember(_ url: URL, as key: StorageKey) throws -> StorageBookmark {
        let data: Data
        do {
            data = try box.store.remember(url, forKey: key.rawValue)
        } catch let error as SecurityScopedBookmarkError {
            throw StorageError(error, key: key)
        }
        let bookmark = StorageBookmark(
            key: key, createdAt: Date(), recordedPath: url.path, byteCount: data.count)
        ledger.record(bookmark)
        return bookmark
    }

    public func resolve(_ key: StorageKey) throws -> StorageScope {
        let resolved: ResolvedBookmark
        do {
            resolved = try box.store.resolve(key: key.rawValue)
        } catch let error as SecurityScopedBookmarkError {
            throw StorageError(error, key: key)
        }
        let scope = StorageScope(
            url: resolved.url, coding: box.coding, isStaleBookmark: resolved.isStale)
        // A *bookmarked* location that refuses the grant is an error, unlike a
        // freshly picked one inside the app container: something was
        // remembered, and it is no longer reachable. This is the same rule
        // `SecurityScopedLocationStore.withAccess(toKey:_:)` applies, kept
        // identical so the two do not disagree about what a refusal means.
        guard scope.isScoped else {
            scope.release()
            throw StorageError.accessRefused(path: resolved.url.path)
        }
        if let previous = ledger.bookmark(key) {
            ledger.record(
                StorageBookmark(
                    key: key, createdAt: previous.createdAt, recordedPath: previous.recordedPath,
                    byteCount: previous.byteCount,
                    lastResolvedPath: resolved.url.path, lastResolvedAt: Date(),
                    resolveCount: previous.resolveCount + 1,
                    staleRewriteCount: previous.staleRewriteCount + (resolved.isStale ? 1 : 0)))
        }
        return scope
    }

    public func bookmark(_ key: StorageKey) -> StorageBookmark? { ledger.bookmark(key) }

    public func forget(_ key: StorageKey) {
        box.store.forget(key: key.rawValue)
        ledger.forget(key)
    }

    public func knownLocations() -> [StorageKey] { ledger.keys() }

    public func bookmarkData(for url: URL) throws -> Data {
        do {
            return try box.coding.bookmarkData(for: url)
        } catch {
            throw StorageError.bookmarkCreationFailed(
                path: url.path, reason: String(describing: error))
        }
    }

    public func resolveBookmarkData(_ data: Data) throws -> ResolvedStorageBookmark {
        let resolved: ResolvedBookmark
        do {
            resolved = try box.coding.resolve(data)
        } catch {
            throw StorageError.bookmarkDataUnresolvable(reason: String(describing: error))
        }

        let scope = StorageScope(
            url: resolved.url, coding: box.coding, isStaleBookmark: resolved.isStale)
        guard scope.isScoped else {
            scope.release()
            throw StorageError.accessRefused(path: resolved.url.path)
        }

        guard resolved.isStale else {
            return ResolvedStorageBookmark(
                scope: scope, bookmarkData: data, wasStale: false)
        }
        do {
            let refreshed = try scope.withAccess { try box.coding.bookmarkData(for: $0) }
            return ResolvedStorageBookmark(
                scope: scope, bookmarkData: refreshed, wasStale: true)
        } catch {
            scope.release()
            throw StorageError.bookmarkRefreshFailed(
                path: resolved.url.path, reason: String(describing: error))
        }
    }
}
