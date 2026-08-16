import Foundation
import iOSBench

/// A live security-scoped grant.
///
/// An object rather than a closure, and that is the whole reason it exists.
/// `SecurityScopedLocationStore.withAccess(to:_:)` is the right shape for a
/// synchronous read and the wrong shape for a transfer: a 1.5 TB download runs
/// for hours across many `await` points, and nothing can be wrapped in a
/// synchronous closure for that long. So the grant becomes a value that is
/// started once and stopped once.
///
/// Balanced by construction: `startAccessingSecurityScopedResource()` in
/// `init`, `stop` exactly once in ``release()`` or in `deinit`. Foundation
/// reference-counts these calls, so an unmatched `stop` decrements somebody
/// else's grant and an unmatched `start` is a kernel-side reference that
/// outlives the app's interest in the volume — which on a removable volume is
/// what stops an ejection from completing.
public final class StorageScope: @unchecked Sendable {
    public let url: URL
    /// Whether the system actually issued a grant. `false` for a URL inside the
    /// app's own container, where no scope is needed — not an error, and the
    /// reason `false` cannot simply be treated as one.
    public let isScoped: Bool
    /// The bookmark resolved but asked to be rewritten. It was rewritten.
    public let isStaleBookmark: Bool

    private let coding: any SecurityScopedURLCoding
    private let lock = NSLock()
    private var released = false

    public var isReleased: Bool {
        lock.lock()
        defer { lock.unlock() }
        return released
    }

    public init(
        url: URL, coding: any SecurityScopedURLCoding = SystemSecurityScopedURLCoding(),
        isStaleBookmark: Bool = false
    ) {
        self.url = url
        self.coding = coding
        self.isStaleBookmark = isStaleBookmark
        self.isScoped = coding.startAccessing(url)
    }

    /// Stop access. Idempotent — calling it twice is a no-op rather than a
    /// second decrement.
    public func release() {
        lock.lock()
        let shouldStop = !released && isScoped
        released = true
        lock.unlock()
        if shouldStop { coding.stopAccessing(url) }
    }

    deinit { release() }

    /// Run `body` with the URL. Opens no second grant: this scope already holds
    /// one, and nesting a second would tell us nothing and unbalance the count.
    public func withAccess<T>(_ body: (URL) throws -> T) rethrows -> T {
        try body(url)
    }
}
