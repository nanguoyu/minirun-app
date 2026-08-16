import Foundation
import XCTest

@testable import iOSBench

/// Bookmark lifecycle against the injected fake: round trip, staleness, and
/// the bracket's behaviour on every exit path.
final class SecurityScopedBookmarkTests: XCTestCase {

    private func makeStore(
        _ coding: FakeSecurityScopedURLCoding = FakeSecurityScopedURLCoding()
    ) -> (SecurityScopedLocationStore, FakeSecurityScopedURLCoding, InMemoryBookmarkPersistence) {
        let persistence = InMemoryBookmarkPersistence()
        return (
            SecurityScopedLocationStore(coding: coding, persistence: persistence),
            coding, persistence
        )
    }

    func testRememberThenResolveReturnsTheSameURL() throws {
        let (store, _, persistence) = makeStore()
        let url = URL(fileURLWithPath: "/Volumes/NVMe/artifact")

        XCTAssertFalse(store.hasBookmark(forKey: "model"))
        try store.remember(url, forKey: "model")
        XCTAssertTrue(store.hasBookmark(forKey: "model"))
        XCTAssertNotNil(persistence.data(forKey: "model"))

        let resolved = try store.resolve(key: "model")
        XCTAssertEqual(resolved.url, url)
        XCTAssertFalse(resolved.isStale)
    }

    func testResolveWithoutBookmarkNamesTheMissingKey() {
        let (store, _, _) = makeStore()
        XCTAssertThrowsError(try store.resolve(key: "model")) { error in
            XCTAssertEqual(
                error as? SecurityScopedBookmarkError, .noStoredBookmark(key: "model"))
        }
    }

    func testForgetRemovesTheStoredBookmark() throws {
        let (store, _, persistence) = makeStore()
        try store.remember(URL(fileURLWithPath: "/tmp/x"), forKey: "model")
        store.forget(key: "model")
        XCTAssertFalse(store.hasBookmark(forKey: "model"))
        XCTAssertNil(persistence.data(forKey: "model"))
    }

    // MARK: - Staleness

    func testStaleBookmarkIsRewrittenAndReportedFresh() throws {
        let (store, coding, persistence) = makeStore()
        let url = URL(fileURLWithPath: "/Volumes/NVMe/artifact")
        try store.remember(url, forKey: "model")
        let original = persistence.data(forKey: "model")

        coding.staleURLs = [url]
        let resolved = try store.resolve(key: "model")

        // The caller sees a usable URL and no staleness, because the store
        // already dealt with it.
        XCTAssertEqual(resolved.url, url)
        XCTAssertFalse(resolved.isStale)
        XCTAssertNotEqual(persistence.data(forKey: "model"), original)
        // The rewrite happened inside a bracket, as iOS requires.
        XCTAssertEqual(coding.accessLog, ["start artifact", "stop artifact"])
        XCTAssertTrue(coding.isBalanced)
    }

    func testStaleBookmarkThatCannotBeRewrittenStaysMarkedStale() throws {
        let (store, coding, persistence) = makeStore()
        let url = URL(fileURLWithPath: "/Volumes/NVMe/artifact")
        try store.remember(url, forKey: "model")
        let original = persistence.data(forKey: "model")

        coding.staleURLs = [url]
        coding.failBookmarkCreation = true
        let resolved = try store.resolve(key: "model")

        // Still usable this launch — a failed rewrite is not a failed resolve.
        XCTAssertEqual(resolved.url, url)
        XCTAssertTrue(resolved.isStale)
        XCTAssertEqual(persistence.data(forKey: "model"), original)
        XCTAssertTrue(coding.isBalanced)
    }

    func testStaleRewriteThatIsRefusedAccessLeavesNoDanglingGrant() throws {
        let (store, coding, persistence) = makeStore()
        let url = URL(fileURLWithPath: "/Volumes/NVMe/artifact")
        try store.remember(url, forKey: "model")
        let original = persistence.data(forKey: "model")

        coding.staleURLs = [url]
        coding.refuseAccess = true
        let resolved = try store.resolve(key: "model")

        XCTAssertTrue(resolved.isStale)
        XCTAssertEqual(persistence.data(forKey: "model"), original)
        XCTAssertEqual(coding.startCount, 0)
        XCTAssertEqual(coding.stopCount, 0, "stop must not follow a start that returned false")
    }

    func testResolutionFailureNamesTheKey() throws {
        let (store, coding, _) = makeStore()
        try store.remember(URL(fileURLWithPath: "/Volumes/NVMe/artifact"), forKey: "model")
        coding.failResolution = true

        XCTAssertThrowsError(try store.resolve(key: "model")) { error in
            guard case .resolutionFailed(let key, _)? = error as? SecurityScopedBookmarkError
            else {
                return XCTFail("expected resolutionFailed, got \(error)")
            }
            XCTAssertEqual(key, "model")
        }
    }

    // MARK: - Access bracketing

    func testAccessIsBalancedOnTheReturningPath() throws {
        let (store, coding, _) = makeStore()
        let url = URL(fileURLWithPath: "/Volumes/NVMe/artifact")

        let value = try store.withAccess(to: url) { _ in 42 }

        XCTAssertEqual(value, 42)
        XCTAssertEqual(coding.startCount, 1)
        XCTAssertEqual(coding.stopCount, 1)
        XCTAssertEqual(coding.accessLog, ["start artifact", "stop artifact"])
    }

    func testAccessIsBalancedOnTheThrowingPath() {
        let (store, coding, _) = makeStore()
        struct Boom: Error {}

        XCTAssertThrowsError(
            try store.withAccess(to: URL(fileURLWithPath: "/Volumes/NVMe/artifact")) { _ in
                throw Boom()
            }
        ) { XCTAssertTrue($0 is Boom, "the body's error must not be swallowed") }

        XCTAssertEqual(coding.startCount, 1)
        XCTAssertEqual(coding.stopCount, 1, "a throwing body must still stop access")
    }

    func testRefusedAccessThrowsAndDoesNotStop() {
        let (store, coding, _) = makeStore()
        coding.refuseAccess = true
        var bodyRan = false

        XCTAssertThrowsError(
            try store.withAccess(to: URL(fileURLWithPath: "/Volumes/NVMe/artifact")) { _ in
                bodyRan = true
            }
        ) { error in
            XCTAssertEqual(
                error as? SecurityScopedBookmarkError,
                .accessRefused(path: "/Volumes/NVMe/artifact"))
        }

        XCTAssertFalse(bodyRan)
        XCTAssertEqual(coding.startCount, 0)
        XCTAssertEqual(
            coding.stopCount, 0,
            "Foundation reference-counts these; an unmatched stop drops someone else's grant")
    }

    func testRepeatedBracketsStayBalanced() throws {
        let (store, coding, _) = makeStore()
        let url = URL(fileURLWithPath: "/Volumes/NVMe/artifact")
        for _ in 0..<8 {
            _ = try? store.withAccess(to: url) { _ -> Int in throw CocoaError(.fileNoSuchFile) }
            _ = try store.withAccess(to: url) { _ in 0 }
        }
        XCTAssertEqual(coding.startCount, 16)
        XCTAssertEqual(coding.stopCount, 16)
    }

    func testWithAccessToKeyResolvesThenBrackets() throws {
        let (store, coding, _) = makeStore()
        let url = URL(fileURLWithPath: "/Volumes/NVMe/artifact")
        try store.remember(url, forKey: "model")

        let seen = try store.withAccess(toKey: "model") { $0 }

        XCTAssertEqual(seen, url)
        XCTAssertEqual(coding.accessLog, ["start artifact", "stop artifact"])
    }
}

/// The real Foundation implementation, run where it works.
///
/// macOS scoped bookmarks are the only place this code can be exercised
/// without a device; `.withSecurityScope` does not exist on iOS and
/// `.minimalBookmark` cannot be driven from a Mac test. What these tests pin
/// is the half that is shared: that the store's create/resolve/refresh logic
/// is correct against a real Foundation bookmark, so the only untested
/// difference on the phone is the option set itself.
///
/// Measured on this machine (Xcode 26.3, macOS 15.7.7) while writing them:
/// a `.withSecurityScope` bookmark of a temporary directory is 820 bytes,
/// `startAccessingSecurityScopedResource()` returns `true` for an unsandboxed
/// test process even for a URL that was never scoped, and *moving* the
/// directory is enough to make resolution report `isStale` while still
/// returning the new location. That last fact is what makes the stale path
/// testable for real rather than only through the fake.
final class SystemSecurityScopedURLCodingTests: XCTestCase {

    func testRealBookmarkRoundTrip() throws {
        let directory = TemporaryDirectory(self)
        let coding = SystemSecurityScopedURLCoding()

        let data = try coding.bookmarkData(for: directory.url)
        XCTAssertFalse(data.isEmpty)

        let resolved = try coding.resolve(data)
        XCTAssertFalse(resolved.isStale)
        XCTAssertEqual(
            resolved.url.resolvingSymlinksInPath().standardizedFileURL,
            directory.url.resolvingSymlinksInPath().standardizedFileURL)
    }

    func testRealStaleBookmarkIsRewrittenByTheStore() throws {
        let parent = TemporaryDirectory(self)
        let original = parent.url.appendingPathComponent("artifact")
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)

        let persistence = InMemoryBookmarkPersistence()
        let store = SecurityScopedLocationStore(
            coding: SystemSecurityScopedURLCoding(), persistence: persistence)
        try store.remember(original, forKey: "model")
        let before = persistence.data(forKey: "model")

        // Moving the directory is what makes Foundation call the bookmark
        // stale: the file id still resolves, the recorded path no longer
        // matches.
        let moved = parent.url.appendingPathComponent("artifact-moved")
        try FileManager.default.moveItem(at: original, to: moved)

        let resolved = try store.resolve(key: "model")
        XCTAssertEqual(
            resolved.url.resolvingSymlinksInPath().standardizedFileURL,
            moved.resolvingSymlinksInPath().standardizedFileURL,
            "a stale bookmark should still find the item it was made for")
        XCTAssertFalse(resolved.isStale, "the store refreshes rather than passing staleness up")
        XCTAssertNotEqual(persistence.data(forKey: "model"), before, "the blob must be rewritten")

        // And the refreshed blob is not itself stale.
        let again = try store.resolve(key: "model")
        XCTAssertFalse(again.isStale)
    }

    func testRealBookmarkOfAMissingPathFailsExplicitly() {
        let missing = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("iosbench-absent-\(UUID().uuidString)")
        let store = SecurityScopedLocationStore(
            coding: SystemSecurityScopedURLCoding(), persistence: InMemoryBookmarkPersistence())

        XCTAssertThrowsError(try store.remember(missing, forKey: "model")) { error in
            guard case .bookmarkCreationFailed? = error as? SecurityScopedBookmarkError else {
                return XCTFail("expected bookmarkCreationFailed, got \(error)")
            }
        }
    }
}

final class UserDefaultsBookmarkPersistenceTests: XCTestCase {

    func testStoresUnderAPrefixedKeyAndRemoves() throws {
        let suite = "iosbench.tests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        addTeardownBlock { defaults.removePersistentDomain(forName: suite) }

        let persistence = UserDefaultsBookmarkPersistence(defaults: defaults, prefix: "p.")
        XCTAssertNil(persistence.data(forKey: "model"))

        persistence.setData(Data([1, 2, 3]), forKey: "model")
        XCTAssertEqual(persistence.data(forKey: "model"), Data([1, 2, 3]))
        XCTAssertEqual(defaults.data(forKey: "p.model"), Data([1, 2, 3]))

        persistence.removeData(forKey: "model")
        XCTAssertNil(persistence.data(forKey: "model"))
    }
}
