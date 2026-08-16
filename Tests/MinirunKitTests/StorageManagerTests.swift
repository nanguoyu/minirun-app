import Foundation
import XCTest
import iOSBench

@testable import MinirunKit

/// A scriptable stand-in for the four Foundation calls that behave differently
/// on iOS than on macOS. Mirrors the one `iOSBenchTests` already uses; it is
/// duplicated rather than shared because test targets do not export their
/// helpers, and the alternative — making it public production code — would put
/// a test double in the shipping module.
final class ScriptedURLCoding: SecurityScopedURLCoding, @unchecked Sendable {
    private let lock = NSLock()
    var staleURLs: Set<URL> = []
    var refusedURLs: Set<URL> = []
    var bookmarkFailureURLs: Set<URL> = []
    private(set) var startCount = 0
    private(set) var stopCount = 0

    func bookmarkData(for url: URL) throws -> Data {
        if bookmarkFailureURLs.contains(url) {
            throw CocoaError(.fileWriteNoPermission)
        }
        return Data(url.path.utf8)
    }

    func resolve(_ data: Data) throws -> ResolvedBookmark {
        let url = URL(fileURLWithPath: String(decoding: data, as: UTF8.self))
        return ResolvedBookmark(url: url, isStale: staleURLs.contains(url))
    }

    func startAccessing(_ url: URL) -> Bool {
        lock.lock()
        startCount += 1
        lock.unlock()
        return !refusedURLs.contains(url)
    }

    func stopAccessing(_ url: URL) {
        lock.lock()
        stopCount += 1
        lock.unlock()
    }

    var balance: Int {
        lock.lock()
        defer { lock.unlock() }
        return startCount - stopCount
    }
}

final class StorageManagerTests: XCTestCase {

    private func makeManager(
        _ coding: ScriptedURLCoding = ScriptedURLCoding()
    ) -> (StorageManager, ScriptedURLCoding, InMemoryBookmarkLedger) {
        let ledger = InMemoryBookmarkLedger()
        let store = SecurityScopedLocationStore(
            coding: coding, persistence: InMemoryBookmarkPersistence())
        return (
            StorageManager(store: store, bookmarkLedger: ledger, coding: coding), coding, ledger
        )
    }

    // MARK: - Scope lifecycle

    func testAScopeStartsOnceAndStopsExactlyOnce() {
        let (manager, coding, _) = makeManager()
        let url = URL(fileURLWithPath: "/Volumes/NVMe/artifact")
        let scope = try! manager.scope(for: url)
        XCTAssertTrue(scope.isScoped)
        XCTAssertEqual(coding.startCount, 1)
        XCTAssertEqual(coding.stopCount, 0)

        scope.release()
        XCTAssertTrue(scope.isReleased)
        XCTAssertEqual(coding.stopCount, 1)

        // Idempotent: a second release must not decrement somebody else's grant.
        scope.release()
        scope.release()
        XCTAssertEqual(coding.stopCount, 1)
        XCTAssertEqual(coding.balance, 0)
    }

    func testAURLThatNeedsNoScopeIsNotAnErrorAndIsNeverStopped() {
        let coding = ScriptedURLCoding()
        let inContainer = URL(fileURLWithPath: "/tmp/app-container")
        coding.refusedURLs = [inContainer]
        let (manager, _, _) = makeManager(coding)

        let scope = try! manager.scope(for: inContainer)
        XCTAssertFalse(scope.isScoped)
        scope.release()
        XCTAssertEqual(
            coding.stopCount, 0,
            "stop must never be paired with a start that returned false")
    }

    func testWithAccessDoesNotNestASecondGrant() {
        let (manager, coding, _) = makeManager()
        let scope = try! manager.scope(for: URL(fileURLWithPath: "/Volumes/NVMe/artifact"))
        let path = scope.withAccess { $0.lastPathComponent }
        XCTAssertEqual(path, "artifact")
        XCTAssertEqual(coding.startCount, 1)
        scope.release()
    }

    // MARK: - Bookmarks and the ledger

    func testRememberThenResolveRecordsWhereItLanded() throws {
        let (manager, _, ledger) = makeManager()
        let url = URL(fileURLWithPath: "/Volumes/NVMe/artifact")

        let bookmark = try manager.remember(url, as: .artifactRoot)
        XCTAssertEqual(bookmark.key, .artifactRoot)
        XCTAssertEqual(bookmark.recordedPath, url.path)
        XCTAssertEqual(bookmark.resolveCount, 0)
        XCTAssertNil(bookmark.lastResolvedPath)
        XCTAssertEqual(manager.knownLocations(), [.artifactRoot])

        let scope = try manager.resolve(.artifactRoot)
        XCTAssertEqual(scope.url, url)
        scope.release()

        let after = try XCTUnwrap(ledger.bookmark(.artifactRoot))
        XCTAssertEqual(after.resolveCount, 1)
        XCTAssertEqual(after.lastResolvedPath, url.path)
        XCTAssertNotNil(after.lastResolvedAt)
        XCTAssertFalse(after.pathMoved)
    }

    /// M10 found the iOS mount UUID stable across a replug and a reboot, so a
    /// resolved path that differs from the recorded one is news. `pathMoved`
    /// is what surfaces it; here it is exercised directly, because the double
    /// resolves a bookmark back to the path it was made from and cannot move
    /// a mount on its own.
    func testTheLedgerReportsAResolvedPathThatDiffersFromTheRecordedOne() {
        let bookmark = StorageBookmark(
            key: .artifactRoot, createdAt: Date(), recordedPath: "/Volumes/NVMe/artifact",
            byteCount: 12, lastResolvedPath: "/Volumes/NVMe-1/artifact", lastResolvedAt: Date(),
            resolveCount: 1, staleRewriteCount: 0)
        XCTAssertTrue(bookmark.pathMoved)

        let steady = StorageBookmark(
            key: .artifactRoot, createdAt: Date(), recordedPath: "/Volumes/NVMe/artifact",
            byteCount: 12, lastResolvedPath: "/Volumes/NVMe/artifact", lastResolvedAt: Date(),
            resolveCount: 3, staleRewriteCount: 0)
        XCTAssertFalse(steady.pathMoved)
    }

    /// A stale bookmark is rewritten *below* this layer — `SecurityScopedLocationStore`
    /// resolves it, re-bookmarks it inside an access bracket, and reports it
    /// fresh — so the scope this manager hands out is not marked stale and the
    /// grant count still balances. Asserting that here keeps the two layers
    /// from quietly disagreeing about who does the rewrite.
    func testAStaleBookmarkIsRewrittenBeforeTheScopeIsHandedOut() throws {
        let coding = ScriptedURLCoding()
        let url = URL(fileURLWithPath: "/Volumes/NVMe/artifact")
        coding.staleURLs = [url]
        let (manager, _, ledger) = makeManager(coding)
        let destination = StorageKey("test-download-destination")
        try manager.remember(url, as: destination)

        let scope = try manager.resolve(destination)
        XCTAssertFalse(scope.isStaleBookmark)
        scope.release()
        XCTAssertEqual(ledger.bookmark(destination)?.staleRewriteCount, 0)
        XCTAssertEqual(coding.balance, 0, "the rewrite's own bracket must balance too")
    }

    func testRawJobBookmarkResolutionHoldsOneBalancedGrant() throws {
        let (manager, coding, _) = makeManager()
        let url = URL(fileURLWithPath: "/Volumes/NVMe/download")
        let bookmark = try coding.bookmarkData(for: url)

        let resolved = try manager.resolveBookmarkData(bookmark)
        XCTAssertEqual(resolved.scope.url, url)
        XCTAssertEqual(resolved.bookmarkData, bookmark)
        XCTAssertFalse(resolved.wasStale)
        XCTAssertEqual(coding.balance, 1)

        resolved.scope.release()
        XCTAssertEqual(coding.balance, 0)
    }

    func testRawStaleJobBookmarkIsRefreshedInsideTheHeldGrant() throws {
        let coding = ScriptedURLCoding()
        let url = URL(fileURLWithPath: "/Volumes/NVMe/download")
        coding.staleURLs = [url]
        let (manager, _, _) = makeManager(coding)
        let bookmark = Data(url.path.utf8)

        let resolved = try manager.resolveBookmarkData(bookmark)
        XCTAssertEqual(resolved.scope.url, url)
        XCTAssertEqual(resolved.bookmarkData, Data(url.path.utf8))
        XCTAssertTrue(resolved.wasStale)
        XCTAssertEqual(coding.balance, 1)

        resolved.scope.release()
        XCTAssertEqual(coding.balance, 0)
    }

    func testRawBookmarkRefreshFailureReleasesTheGrant() {
        let coding = ScriptedURLCoding()
        let url = URL(fileURLWithPath: "/Volumes/NVMe/download")
        coding.staleURLs = [url]
        coding.bookmarkFailureURLs = [url]
        let (manager, _, _) = makeManager(coding)

        XCTAssertThrowsError(try manager.resolveBookmarkData(Data(url.path.utf8))) { error in
            guard case .bookmarkRefreshFailed(let path, _) = error as? StorageError else {
                return XCTFail("expected bookmarkRefreshFailed, got \(error)")
            }
            XCTAssertEqual(path, url.path)
        }
        XCTAssertEqual(coding.balance, 0)
    }

    func testRawBookmarkThatCannotAcquireItsGrantIsRefusedWithoutAStop() {
        let coding = ScriptedURLCoding()
        let url = URL(fileURLWithPath: "/Volumes/Gone/download")
        coding.refusedURLs = [url]
        let (manager, _, _) = makeManager(coding)

        XCTAssertThrowsError(try manager.resolveBookmarkData(Data(url.path.utf8))) { error in
            XCTAssertEqual(error as? StorageError, .accessRefused(path: url.path))
        }
        XCTAssertEqual(coding.stopCount, 0)
    }

    func testResolvingWithoutABookmarkNamesTheKey() {
        let (manager, _, _) = makeManager()
        XCTAssertThrowsError(try manager.resolve(.artifactRoot)) { error in
            XCTAssertEqual(error as? StorageError, .noBookmark(.artifactRoot))
        }
    }

    func testARefusedGrantOnABookmarkedLocationIsAnError() throws {
        let coding = ScriptedURLCoding()
        let url = URL(fileURLWithPath: "/Volumes/Gone/artifact")
        let (manager, _, _) = makeManager(coding)
        try manager.remember(url, as: .artifactRoot)
        coding.refusedURLs = [url]

        XCTAssertThrowsError(try manager.resolve(.artifactRoot)) { error in
            XCTAssertEqual(error as? StorageError, .accessRefused(path: url.path))
        }
        // No grant was issued, so none is stopped. Pairing a `stop` with a
        // `start` that returned false would decrement somebody else's grant.
        XCTAssertEqual(coding.stopCount, 0)
    }

    func testForgetClearsBothTheBookmarkAndTheLedger() throws {
        let (manager, _, ledger) = makeManager()
        try manager.remember(URL(fileURLWithPath: "/tmp/x"), as: .artifactRoot)
        manager.forget(.artifactRoot)
        XCTAssertNil(ledger.bookmark(.artifactRoot))
        XCTAssertEqual(manager.knownLocations(), [])
        XCTAssertThrowsError(try manager.resolve(.artifactRoot))
    }

    func testEverySecurityScopedErrorMapsToExactlyOneStorageError() {
        XCTAssertEqual(
            StorageError(.noStoredBookmark(key: "k"), key: .artifactRoot),
            .noBookmark(.artifactRoot))
        XCTAssertEqual(
            StorageError(.resolutionFailed(key: "k", reason: "gone"), key: .artifactRoot),
            .bookmarkUnresolvable(.artifactRoot, reason: "gone"))
        XCTAssertEqual(
            StorageError(.accessRefused(path: "/p"), key: .artifactRoot),
            .accessRefused(path: "/p"))
        XCTAssertEqual(
            StorageError(.bookmarkCreationFailed(path: "/p", reason: "why"), key: .artifactRoot),
            .bookmarkCreationFailed(path: "/p", reason: "why"))
    }

    // MARK: - Space

    func testCanHoldUsesTheDependableNumberAndRefusesRatherThanRounding() {
        let (manager, _, _) = makeManager()
        let temporary = URL(fileURLWithPath: NSTemporaryDirectory())

        let space = manager.freeSpace(at: temporary)
        XCTAssertNotNil(space.dependableBytes)
        if let dependable = space.dependableBytes, let optimistic = space.optimisticBytes {
            XCTAssertGreaterThanOrEqual(
                optimistic, dependable,
                "purgeable-inclusive free space cannot be smaller than the dependable figure")
        }

        switch manager.canHold(bytes: 1024, at: temporary, headroomBytes: 1024) {
        case .fits(let spare):
            XCTAssertGreaterThanOrEqual(spare, 0)
        case .refused(let error):
            XCTFail("a kilobyte should fit in the temporary directory: \(error)")
        case .unknown(let reason):
            XCTFail("the temporary directory should be probeable: \(reason)")
        }

        // Refused, and the refusal quotes the arithmetic rather than a verdict.
        switch manager.canHold(
            bytes: UInt64(Int64.max / 4), at: temporary, headroomBytes: UInt64(Int64.max / 4))
        {
        case .refused(.insufficientFreeSpace(let need, let headroom, let available, _)):
            XCTAssertEqual(need, UInt64(Int64.max / 4))
            XCTAssertEqual(headroom, UInt64(Int64.max / 4))
            XCTAssertLessThan(available, Int64.max / 4)
        default:
            XCTFail("half of Int64.max cannot fit anywhere")
        }
    }

    func testCanHoldRefusesUnsignedValuesThatCannotBeRepresentedAsCapacity() {
        let (manager, _, _) = makeManager()
        let temporary = URL(fileURLWithPath: NSTemporaryDirectory())

        for (bytes, headroom) in [
            (UInt64.max, UInt64(0)),
            (UInt64(0), UInt64.max),
            (UInt64.max, UInt64.max),
        ] {
            guard case .refused(
                .insufficientFreeSpace(
                    needBytes: let reportedBytes,
                    headroomBytes: let reportedHeadroom,
                    availableBytes: let available,
                    volume: _)
            ) = manager.canHold(bytes: bytes, at: temporary, headroomBytes: headroom) else {
                return XCTFail("an unrepresentable request must be refused")
            }
            XCTAssertEqual(reportedBytes, bytes)
            XCTAssertEqual(reportedHeadroom, headroom)
            XCTAssertGreaterThanOrEqual(available, 0)
        }
    }

    /// Product policy follows the filesystem rather than a volume name. This
    /// regression is environment-gated because CI does not carry the owner's
    /// external drive; when it is mounted writable, neither enumeration nor
    /// the capacity gate may manufacture a read-only refusal for it.
    func testWritableK3NVMEIsNotRefusedByName() throws {
        let root = URL(fileURLWithPath: "/Volumes/K3NVME", isDirectory: true)
        guard FileManager.default.fileExists(atPath: root.path) else {
            throw XCTSkip("K3NVME is not mounted")
        }
        let (manager, _, _) = makeManager()
        guard manager.describe(root).volumeIsReadOnly == false else {
            throw XCTSkip("K3NVME is currently mounted read-only")
        }
        let volume = try XCTUnwrap(manager.volumes().first { $0.mountPath == root.path })
        XCTAssertNil(volume.writeRefusal)
        if case .refused(.destinationIsReadOnlyVolume(let path)) = manager.canHold(
            bytes: 1, at: root, headroomBytes: 0)
        {
            XCTFail("a writable volume was misreported as read-only at \(path)")
        }
    }

    func testVolumesAreEnumeratedAndCarryTheirRefusals() throws {
        let (manager, _, _) = makeManager()
        let volumes = try manager.volumes()
        XCTAssertFalse(volumes.isEmpty, "this machine has at least a root volume")
        XCTAssertTrue(volumes.contains { $0.mountPath == "/" })
        for volume in volumes {
            if volume.space.isReadOnly {
                XCTAssertNotNil(
                    volume.writeRefusal,
                    "\(volume.mountPath) is read-only and must carry a refusal")
            } else {
                XCTAssertNil(
                    volume.writeRefusal,
                    "\(volume.mountPath) is writable and must not be refused by identity")
            }
        }
    }

    func testDescribeReusesStorageCoreRatherThanRederiving() {
        let (manager, _, _) = makeManager()
        let info = manager.describe(URL(fileURLWithPath: NSTemporaryDirectory()))
        XCTAssertTrue(info.pathExists)
        XCTAssertTrue(info.pathIsDirectory)
        XCTAssertEqual(FreeSpace(info).dependableBytes, info.volumeAvailableBytes)
        XCTAssertEqual(
            FreeSpace(info).optimisticBytes, info.volumeAvailableBytesForImportantUsage)
    }

    func testBookmarkLedgerPersistsThroughUserDefaults() {
        let suite = "minirun.kit.tests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let ledger = UserDefaultsBookmarkLedger(defaults: defaults, prefix: "test.")
        let bookmark = StorageBookmark(
            key: .artifactRoot, createdAt: Date(), recordedPath: "/p", byteCount: 12)
        ledger.record(bookmark)
        let read = ledger.bookmark(.artifactRoot)
        XCTAssertEqual(read?.key, .artifactRoot)
        XCTAssertEqual(read?.recordedPath, "/p")
        XCTAssertEqual(read?.byteCount, 12)
        XCTAssertEqual(read?.resolveCount, 0)
        // ISO 8601 keeps whole seconds, so the timestamp is compared at the
        // resolution it is stored at rather than to the nanosecond.
        XCTAssertEqual(
            read.map { Int($0.createdAt.timeIntervalSince1970) },
            Int(bookmark.createdAt.timeIntervalSince1970))
        XCTAssertEqual(ledger.keys(), [.artifactRoot])
        ledger.forget(.artifactRoot)
        XCTAssertNil(ledger.bookmark(.artifactRoot))
        XCTAssertEqual(ledger.keys(), [])
    }
}
