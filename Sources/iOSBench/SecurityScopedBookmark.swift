import Foundation

/// A URL recovered from a persisted bookmark.
public struct ResolvedBookmark: Equatable {
    public let url: URL
    /// The system re-derived this URL but wants the bookmark rewritten. Apple
    /// treats a stale bookmark as still usable *once*, for long enough to make
    /// a fresh one; ``SecurityScopedLocationStore`` does exactly that rather
    /// than surfacing staleness to the caller as a failure.
    public let isStale: Bool

    public init(url: URL, isStale: Bool) {
        self.url = url
        self.isStale = isStale
    }
}

/// The four Foundation calls that behave differently on iOS than on macOS, and
/// that no unit test on a Mac can drive through a real document picker.
///
/// This is a protocol for one reason: spec §4.3 says external-storage semantics
/// on iOS must be *verified*, not assumed, and a fake lets the failure modes
/// worth verifying (stale bookmark, refused access, resolution error) be
/// exercised deterministically on the development Mac. The real
/// implementation, ``SystemSecurityScopedURLCoding``, is exercised in the same
/// suite against a temporary directory, where macOS scoped bookmarks do work.
public protocol SecurityScopedURLCoding {
    /// Bookmark `url` so it survives process death.
    func bookmarkData(for url: URL) throws -> Data
    /// Recover a URL from bookmark data.
    func resolve(_ data: Data) throws -> ResolvedBookmark
    /// `startAccessingSecurityScopedResource()`. `false` means the sandbox
    /// refused *or* the URL never needed scoping; the caller must not pair a
    /// `stop` with a `false` here (see ``SecurityScopedLocationStore``).
    func startAccessing(_ url: URL) -> Bool
    /// `stopAccessingSecurityScopedResource()`.
    func stopAccessing(_ url: URL)
}

/// Foundation's implementation, with the one real API difference between the
/// platforms isolated to this file.
///
/// ## `.withSecurityScope` is macOS-only
///
/// `URL.BookmarkCreationOptions.withSecurityScope` and
/// `URL.BookmarkResolutionOptions.withSecurityScope` are annotated
/// `@available(macOS 10.7, *)` and are *not* available on iOS — referencing
/// them in iOS-bound code is a compile error, not a runtime no-op. On iOS the
/// security scope is implicit: a bookmark made from a URL the document picker
/// vended is security-scoped by construction, and
/// `startAccessingSecurityScopedResource()` is what activates it. The
/// documented iOS creation option is `.minimalBookmark`, and resolution takes
/// no scope option at all.
///
/// Getting this wrong is silent on one platform and fatal on the other, which
/// is why the difference lives here and nowhere else: everything above this
/// type sees one API.
public struct SystemSecurityScopedURLCoding: SecurityScopedURLCoding {
    public init() {}

    public func bookmarkData(for url: URL) throws -> Data {
        #if os(macOS)
            return try url.bookmarkData(
                options: [.withSecurityScope],
                includingResourceValuesForKeys: nil,
                relativeTo: nil)
        #else
            // `.minimalBookmark` keeps the blob small; a full bookmark embeds
            // resource values this harness never reads, and on a removable
            // volume those go stale faster than the path does.
            return try url.bookmarkData(
                options: [.minimalBookmark],
                includingResourceValuesForKeys: nil,
                relativeTo: nil)
        #endif
    }

    public func resolve(_ data: Data) throws -> ResolvedBookmark {
        var isStale = false
        #if os(macOS)
            let options: URL.BookmarkResolutionOptions = [.withSecurityScope]
        #else
            let options: URL.BookmarkResolutionOptions = []
        #endif
        let url = try URL(
            resolvingBookmarkData: data,
            options: options,
            relativeTo: nil,
            bookmarkDataIsStale: &isStale)
        return ResolvedBookmark(url: url, isStale: isStale)
    }

    public func startAccessing(_ url: URL) -> Bool {
        url.startAccessingSecurityScopedResource()
    }

    public func stopAccessing(_ url: URL) {
        url.stopAccessingSecurityScopedResource()
    }
}

/// Where bookmark blobs live between launches.
public protocol BookmarkPersistence {
    func data(forKey key: String) -> Data?
    func setData(_ data: Data, forKey key: String)
    func removeData(forKey key: String)
}

/// `UserDefaults`-backed persistence. A bookmark is a few hundred bytes and
/// there is at most one per benchmark location, so a defaults key is the right
/// size of mechanism; nothing here justifies a file or a database.
public struct UserDefaultsBookmarkPersistence: BookmarkPersistence {
    private let defaults: UserDefaults
    private let prefix: String

    public init(defaults: UserDefaults = .standard, prefix: String = "minirun.bookmark.") {
        self.defaults = defaults
        self.prefix = prefix
    }

    public func data(forKey key: String) -> Data? {
        defaults.data(forKey: prefix + key)
    }

    public func setData(_ data: Data, forKey key: String) {
        defaults.set(data, forKey: prefix + key)
    }

    public func removeData(forKey key: String) {
        defaults.removeObject(forKey: prefix + key)
    }
}

/// Persistence with no side effects outside the instance. Used by the tests,
/// and by any caller that wants a picked location to last only as long as the
/// process.
public final class InMemoryBookmarkPersistence: BookmarkPersistence, @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String: Data] = [:]

    public init(storage: [String: Data] = [:]) {
        self.storage = storage
    }

    public func data(forKey key: String) -> Data? {
        lock.lock()
        defer { lock.unlock() }
        return storage[key]
    }

    public func setData(_ data: Data, forKey key: String) {
        lock.lock()
        defer { lock.unlock() }
        storage[key] = data
    }

    public func removeData(forKey key: String) {
        lock.lock()
        defer { lock.unlock() }
        storage.removeValue(forKey: key)
    }

    /// Keys currently held. Lets a test assert that a stale bookmark was
    /// actually rewritten rather than merely resolved.
    public var keys: [String] {
        lock.lock()
        defer { lock.unlock() }
        return storage.keys.sorted()
    }
}

public enum SecurityScopedBookmarkError: Error, CustomStringConvertible, LocalizedError, Equatable {
    /// Nothing has been bookmarked under this key yet.
    case noStoredBookmark(key: String)
    /// The bookmark exists but the system could not turn it back into a URL —
    /// the usual cause on iOS is that the volume is not attached.
    case resolutionFailed(key: String, reason: String)
    /// The bookmark resolved but the sandbox refused access. Spec §5.8: report
    /// it, never fall back to reading a path that happens to be readable.
    case accessRefused(path: String)
    /// Creating the bookmark failed.
    case bookmarkCreationFailed(path: String, reason: String)

    public var description: String {
        switch self {
        case .noStoredBookmark(let key):
            return "no bookmark stored under '\(key)'"
        case .resolutionFailed(let key, let reason):
            return "bookmark '\(key)' could not be resolved: \(reason)"
        case .accessRefused(let path):
            return "startAccessingSecurityScopedResource refused '\(path)'"
        case .bookmarkCreationFailed(let path, let reason):
            return "could not bookmark '\(path)': \(reason)"
        }
    }

    public var errorDescription: String? { description }
}

/// Remembers a user-picked location across launches and hands it out only
/// inside a balanced access bracket.
///
/// Spec §4.3 requires the iOS spike to verify "security-scoped URL behavior"
/// before assuming desktop filesystem semantics. Two behaviours it pins down:
///
/// 1. **Access is bracketed, and the bracket is scoped.** Every read of the
///    picked location happens inside ``withAccess(toKey:_:)``, whose `defer`
///    stops access on the throwing path as well as the returning one. A leaked
///    `start` is not a visible bug — it is a kernel-side reference that outlives
///    the app's interest in the volume, and on a removable volume it is what
///    keeps an ejection from completing.
/// 2. **`stop` is called only when `start` returned `true`.** Foundation
///    reference-counts these, so an unmatched `stop` decrements someone else's
///    grant. A URL inside the app's own container legitimately returns `false`,
///    which is why "false" cannot simply be treated as an error here — but for
///    a *bookmarked* location it is one, and is reported as
///    ``SecurityScopedBookmarkError/accessRefused(path:)``.
public final class SecurityScopedLocationStore {
    private let coding: SecurityScopedURLCoding
    private let persistence: BookmarkPersistence

    public init(
        coding: SecurityScopedURLCoding = SystemSecurityScopedURLCoding(),
        persistence: BookmarkPersistence
    ) {
        self.coding = coding
        self.persistence = persistence
    }

    /// Bookmark `url` under `key`, replacing anything stored there.
    ///
    /// The document picker vends a URL that is already accessible for the
    /// duration of the delegate callback, so this does not open its own
    /// bracket — doing so would nest a second grant inside the picker's and
    /// tell us nothing. Callers that bookmark a URL from anywhere else should
    /// go through ``withAccess(to:_:)``.
    @discardableResult
    public func remember(_ url: URL, forKey key: String) throws -> Data {
        let data: Data
        do {
            data = try coding.bookmarkData(for: url)
        } catch {
            throw SecurityScopedBookmarkError.bookmarkCreationFailed(
                path: url.path, reason: String(describing: error))
        }
        persistence.setData(data, forKey: key)
        return data
    }

    public func hasBookmark(forKey key: String) -> Bool {
        persistence.data(forKey: key) != nil
    }

    public func forget(key: String) {
        persistence.removeData(forKey: key)
    }

    /// Resolve the stored bookmark, rewriting it if the system reports it
    /// stale.
    ///
    /// The rewrite happens under an access bracket. That is not defensive
    /// styling: on iOS, creating a bookmark for a security-scoped URL outside
    /// the bracket fails, so "resolve stale, then re-bookmark" is only correct
    /// in this order. A rewrite that fails is *not* fatal — the resolved URL is
    /// still usable this launch — so the failure is swallowed and reported
    /// through ``ResolvedBookmark/isStale`` staying `true`.
    public func resolve(key: String) throws -> ResolvedBookmark {
        guard let data = persistence.data(forKey: key) else {
            throw SecurityScopedBookmarkError.noStoredBookmark(key: key)
        }
        let resolved: ResolvedBookmark
        do {
            resolved = try coding.resolve(data)
        } catch {
            throw SecurityScopedBookmarkError.resolutionFailed(
                key: key, reason: String(describing: error))
        }
        guard resolved.isStale else { return resolved }

        let refreshed = try? withAccess(to: resolved.url) { url -> Data in
            try coding.bookmarkData(for: url)
        }
        if let refreshed {
            persistence.setData(refreshed, forKey: key)
            return ResolvedBookmark(url: resolved.url, isStale: false)
        }
        return resolved
    }

    /// Run `body` with the bookmarked location accessible, then stop access.
    public func withAccess<T>(toKey key: String, _ body: (URL) throws -> T) throws -> T {
        let resolved = try resolve(key: key)
        return try withAccess(to: resolved.url, body)
    }

    /// Run `body` with `url` accessible, then stop access — including when
    /// `body` throws.
    public func withAccess<T>(to url: URL, _ body: (URL) throws -> T) throws -> T {
        guard coding.startAccessing(url) else {
            throw SecurityScopedBookmarkError.accessRefused(path: url.path)
        }
        defer { coding.stopAccessing(url) }
        return try body(url)
    }
}
