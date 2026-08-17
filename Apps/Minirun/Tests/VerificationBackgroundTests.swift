import Foundation
import MinirunKit
import iOSBench
import StorageCore
import XCTest

@testable import MinirunApp

/// A storage double that hands back the real fixture directory. Same shape as
/// the one in `InstalledModelsTests`; both are private to their file.
private final class BackgroundFixtureStorage: StorageManaging, @unchecked Sendable {
    private let lock = NSLock()
    private var remembered: [StorageKey: URL] = [:]
    private let scopeCoding = BackgroundScopeCoding()
    private var resolvedScopes: [StorageScope] = []

    var resolvedScopesSnapshot: [StorageScope] {
        lock.lock()
        defer { lock.unlock() }
        return resolvedScopes
    }

    func volumes() throws -> [VolumeDescriptor] { [] }
    func describe(_ url: URL) -> StorageInfo { StorageInfo.describing(path: url.path) }
    func freeSpace(at url: URL) -> FreeSpace { FreeSpace(describe(url)) }
    func canHold(bytes: UInt64, at url: URL, headroomBytes: UInt64) -> SpaceVerdict {
        .fits(spareBytes: 0)
    }
    func scope(for url: URL) throws -> StorageScope {
        StorageScope(url: url, coding: scopeCoding)
    }

    @discardableResult
    func remember(_ url: URL, as key: StorageKey) throws -> StorageBookmark {
        lock.lock()
        remembered[key] = url
        lock.unlock()
        return StorageBookmark(
            key: key, createdAt: Date(), recordedPath: url.path, byteCount: 0)
    }

    func resolve(_ key: StorageKey) throws -> StorageScope {
        lock.lock()
        let url = remembered[key]
        lock.unlock()
        guard let url else { throw StorageError.noBookmark(key) }
        let scope = StorageScope(url: url, coding: scopeCoding)
        lock.lock()
        resolvedScopes.append(scope)
        lock.unlock()
        return scope
    }

    func bookmark(_ key: StorageKey) -> StorageBookmark? {
        lock.lock()
        let url = remembered[key]
        lock.unlock()
        return url.map {
            StorageBookmark(key: key, createdAt: Date(), recordedPath: $0.path, byteCount: 0)
        }
    }

    func forget(_ key: StorageKey) {
        lock.lock()
        remembered.removeValue(forKey: key)
        lock.unlock()
    }

    func knownLocations() -> [StorageKey] {
        lock.lock()
        defer { lock.unlock() }
        return remembered.keys.sorted { $0.rawValue < $1.rawValue }
    }

    func bookmarkData(for url: URL) throws -> Data { Data(url.path.utf8) }
}

private final class BackgroundScopeCoding: SecurityScopedURLCoding, @unchecked Sendable {
    func bookmarkData(for url: URL) throws -> Data { Data(url.path.utf8) }
    func resolve(_ data: Data) throws -> ResolvedBookmark {
        throw CocoaError(.fileReadCorruptFile)
    }
    func startAccessing(_ url: URL) -> Bool { true }
    func stopAccessing(_ url: URL) {}
}

/// Stands in for `BGContinuedProcessingTask`. The expiration closure is exactly
/// the one the framework calls when it takes its background time back, so a
/// test can drive that moment without a device.
@MainActor
private final class FakeBackgroundWork: VerificationBackgroundWork {
    private(set) var isRunning = false
    private(set) var refusal: String?
    private(set) var beginCount = 0
    private(set) var endedSuccessfully: [Bool] = []
    private(set) var lastTitle = ""
    private(set) var lastTotalBytes: UInt64 = 0
    private(set) var lastCompletedBytes: UInt64 = 0
    private var expiration: (@MainActor () -> Void)?
    /// Whether the system accepts the request at all.
    var accepts = true

    @discardableResult
    func begin(
        title: String, subtitle: String, totalBytes: UInt64,
        onExpiration: @escaping @MainActor () -> Void
    ) -> Bool {
        beginCount += 1
        lastTitle = title
        lastTotalBytes = totalBytes
        guard accepts else {
            refusal = "the system declined"
            return false
        }
        expiration = onExpiration
        isRunning = true
        return true
    }

    func update(subtitle: String, completedBytes: UInt64, totalBytes: UInt64) {
        lastCompletedBytes = completedBytes
        lastTotalBytes = totalBytes
    }

    func end(success: Bool) {
        endedSuccessfully.append(success)
        expiration = nil
        isRunning = false
    }

    /// What iOS does when the continued-processing task's time is up.
    func expire() {
        let handler = expiration
        expiration = nil
        isRunning = false
        handler?()
    }
}

/// Records what the app decided to tell the operator.
@MainActor
private final class RecordingAnnouncer: VerificationAnnouncing {
    private(set) var prepareCount = 0
    private(set) var announcements: [VerificationAnnouncement] = []

    func prepare() { prepareCount += 1 }
    func announce(_ announcement: VerificationAnnouncement) {
        announcements.append(announcement)
    }
}

private struct VerificationRefusal: Error, CustomStringConvertible {
    let description: String
}

/// Whether the operator is looking at the app, under the test's control.
@MainActor
private final class FrontmostFlag {
    var value = true
}

/// Drives an injected verification: it reports when it started, blocks until
/// released, and honours cancellation the way the real verifier does.
private final class VerificationDriver: @unchecked Sendable {
    enum Outcome {
        case matched
        /// Matched, having spent an earlier revision's evidence for all but a
        /// few files — what an edited model card produces (ADR 0014).
        case matchedCarryingForward
        case wrongDigest
        case refused(String)
    }

    private let lock = NSLock()
    private var starts: [ArtifactVerificationResumption] = []
    private var released = false
    private var stored: Outcome = .matched

    var outcome: Outcome {
        get {
            lock.lock()
            defer { lock.unlock() }
            return stored
        }
        set {
            lock.lock()
            stored = newValue
            lock.unlock()
        }
    }

    var startCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return starts.count
    }

    var resumptions: [ArtifactVerificationResumption] {
        lock.lock()
        defer { lock.unlock() }
        return starts
    }

    func noteStart(_ resumption: ArtifactVerificationResumption) {
        lock.lock()
        starts.append(resumption)
        lock.unlock()
    }

    func release() {
        lock.lock()
        released = true
        lock.unlock()
    }

    private var isReleased: Bool {
        lock.lock()
        defer { lock.unlock() }
        return released
    }

    func run(
        _ resumption: ArtifactVerificationResumption, model: ModelID
    ) async throws -> VerificationReport {
        noteStart(resumption)
        while !isReleased {
            // Throws `CancellationError` the moment the task is cancelled,
            // which is what expiration does to the real pass.
            try await Task.sleep(for: .milliseconds(5))
        }
        switch outcome {
        case .matched:
            return VerificationReport(
                job: DownloadJobID(), model: model, depth: .digestEverything,
                checkedAt: Date(), ok: ["global/embed_tokens.bin"], missing: [],
                wrongSize: [], wrongDigest: [], unreadable: [:], extraneous: [])
        case .matchedCarryingForward:
            return VerificationReport(
                job: DownloadJobID(), model: model, depth: .digestEverything,
                checkedAt: Date(), ok: ["global/embed_tokens.bin"], missing: [],
                wrongSize: [], wrongDigest: [], unreadable: [:], extraneous: [],
                carryForward: ArtifactCarryForwardSummary(
                    sourceRevision: String(repeating: "c", count: 40),
                    sourceState: .fullyVerified, carriedForward: 627, reread: 3,
                    newFiles: 0, removedFiles: 0))
        case .wrongDigest:
            return VerificationReport(
                job: DownloadJobID(), model: model, depth: .digestEverything,
                checkedAt: Date(), ok: [], missing: [], wrongSize: [],
                wrongDigest: [
                    DigestMismatch(
                        path: "global/embed_tokens.bin", algorithm: .sha256,
                        expected: String(repeating: "a", count: 64),
                        found: String(repeating: "b", count: 64))
                ],
                unreadable: [:], extraneous: [])
        case .refused(let reason):
            throw VerificationRefusal(description: reason)
        }
    }

    func waitUntilStarted(_ count: Int = 1) async throws {
        for _ in 0..<400 {
            if startCount >= count { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("the verification never started")
    }
}

/// Leaving the app used to end a terabyte of hashing. These are the app-side
/// rules that replaced that: the system keeps the work running, an expiration
/// is a pause rather than a loss, returning to the app resumes it where it
/// stopped, and none of it depends on which screen is on top.
@MainActor
final class VerificationBackgroundTests: XCTestCase {
    private var root: URL!
    private var storage: BackgroundFixtureStorage!
    private var backgroundWork: FakeBackgroundWork!
    private var driver: VerificationDriver!
    private var announcer: RecordingAnnouncer!
    private var frontmost: FrontmostFlag!
    private var installed: InstalledModels!

    private let k3Index = """
        {"format":"quantized_tile_container","source_repo":"moonshotai/Kimi-K3",
         "source_revision":"9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
         "files":372,"bytes":1559976181760}
        """

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-verification-background-\(UUID().uuidString)")
        let artifact = root.appendingPathComponent("k3-artifact")
        try FileManager.default.createDirectory(
            at: artifact.appendingPathComponent("global"), withIntermediateDirectories: true)
        try Data(k3Index.utf8).write(to: artifact.appendingPathComponent("index.json"))
        try Data(repeating: 0x41, count: 4096).write(
            to: artifact.appendingPathComponent("global/embed_tokens.bin"))

        storage = BackgroundFixtureStorage()
        backgroundWork = FakeBackgroundWork()
        driver = VerificationDriver()
        announcer = RecordingAnnouncer()
        frontmost = FrontmostFlag()
        let driver = self.driver!
        let frontmost = self.frontmost!
        installed = InstalledModels(
            storage: storage, catalog: ModelCatalog.bundled,
            ledger: InMemoryVerificationLedger(),
            checkpoints: InMemoryVerificationCheckpointStore(),
            backgroundWork: backgroundWork,
            announcer: announcer,
            applicationIsFrontmost: { frontmost.value },
            verificationOperation: { artifact, _, _, resumption, _ in
                try await driver.run(resumption, model: artifact.model ?? .kimiK3)
            })
    }

    override func tearDownWithError() throws {
        driver?.release()
        installed = nil
        if let root { try? FileManager.default.removeItem(at: root) }
    }

    private func scan() async throws {
        installed.refresh()
        for _ in 0..<400 {
            if !installed.isScanning && installed.scanCount > 0 { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("the scan did not finish")
    }

    private func registeredArtifact() async throws -> DiscoveredArtifact {
        installed.addLocation(root)
        try await scan()
        return try XCTUnwrap(installed.preferredArtifact(for: .kimiK3))
    }

    // MARK: - Start, expire, resume, complete

    func testStartingAVerificationAsksTheSystemToKeepItRunning() async throws {
        let artifact = try await registeredArtifact()
        let verification = Task { await installed.verify(artifact, .full) }
        try await driver.waitUntilStarted()

        XCTAssertEqual(backgroundWork.beginCount, 1)
        XCTAssertTrue(backgroundWork.isRunning)
        XCTAssertTrue(
            backgroundWork.lastTitle.contains(artifact.displayName),
            "the system panel names the model being verified: \(backgroundWork.lastTitle)")

        driver.release()
        await verification.value
        XCTAssertEqual(backgroundWork.endedSuccessfully, [true])
        XCTAssertNil(installed.pausedVerification)
    }

    func testExpirationPausesTheVerificationAndForegroundResumesIt() async throws {
        let artifact = try await registeredArtifact()
        let verification = Task { await installed.verify(artifact, .full) }
        try await driver.waitUntilStarted()

        // Exactly what the framework's expiration handler does.
        backgroundWork.expire()
        await verification.value

        let paused = try XCTUnwrap(installed.pausedVerification)
        XCTAssertEqual(paused.rootPath, artifact.rootPath)
        XCTAssertEqual(paused.request, .full)
        XCTAssertNil(installed.verificationInFlight)
        XCTAssertEqual(
            installed.verificationOutcome[artifact.rootPath],
            "Paused where it stopped; open Minirun to continue from there.")
        XCTAssertEqual(
            backgroundWork.endedSuccessfully, [false],
            "an expired pass is not reported as a success")

        // Coming back to the app resumes it, and resuming means resuming: the
        // second pass is allowed to spend the checkpoint the first one wrote.
        driver.release()
        installed.resumePausedVerification()
        try await driver.waitUntilStarted(2)
        for _ in 0..<400 where installed.verificationInFlight != nil {
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertEqual(driver.resumptions, [.resumeIfPossible, .resumeIfPossible])
        XCTAssertNil(installed.pausedVerification)
        XCTAssertEqual(backgroundWork.beginCount, 2)
        XCTAssertEqual(backgroundWork.endedSuccessfully, [false, true])
    }

    func testCancellingIsStillACancellationAndNotAPause() async throws {
        let artifact = try await registeredArtifact()
        let verification = Task { await installed.verify(artifact, .full) }
        try await driver.waitUntilStarted()

        installed.cancelVerification()
        await verification.value

        XCTAssertNil(
            installed.pausedVerification,
            "the operator stopping the pass is not a pause the app resumes by itself")
        XCTAssertEqual(installed.verificationOutcome[artifact.rootPath], "Cancelled")
    }

    // MARK: - Leaving the foreground

    func testLeavingTheForegroundDoesNotStopAContinuedVerification() async throws {
        let artifact = try await registeredArtifact()
        let verification = Task { await installed.verify(artifact, .full) }
        try await driver.waitUntilStarted()

        installed.applicationWillLeaveForeground()
        try await Task.sleep(for: .milliseconds(30))

        XCTAssertEqual(
            installed.verificationInFlight, artifact.rootPath,
            "a pass the system agreed to continue keeps running behind the app")
        XCTAssertNil(installed.pausedVerification)

        driver.release()
        await verification.value
        XCTAssertEqual(installed.verificationOutcome[artifact.rootPath]?.isEmpty, false)
    }

    func testLeavingTheForegroundWithoutAContinuationPausesInsteadOfFreezing() async throws {
        backgroundWork.accepts = false
        let artifact = try await registeredArtifact()
        let verification = Task { await installed.verify(artifact, .full) }
        try await driver.waitUntilStarted()

        installed.applicationWillLeaveForeground()
        await verification.value

        let paused = try XCTUnwrap(installed.pausedVerification)
        XCTAssertEqual(paused.rootPath, artifact.rootPath)
        XCTAssertNil(installed.verificationInFlight)
    }

    func testSceneTransitionsRouteThroughTheSameRules() async throws {
        let artifact = try await registeredArtifact()
        let model = AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            store: .ephemeral(),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            persistsDefaults: false,
            storage: storage,
            downloadServices: .unavailable("not part of this test"),
            installed: installed, startDiscovery: false)
        model.transitionScene(to: .active)
        let verification = Task { await installed.verify(artifact, .full) }
        try await driver.waitUntilStarted()

        model.transitionScene(to: .background)
        try await Task.sleep(for: .milliseconds(30))
        XCTAssertEqual(
            installed.verificationInFlight, artifact.rootPath,
            "backgrounding no longer cancels a verification the system is continuing")

        backgroundWork.expire()
        await verification.value
        XCTAssertNotNil(installed.pausedVerification)

        driver.release()
        model.transitionScene(to: .active)
        try await driver.waitUntilStarted(2)
        for _ in 0..<400 where installed.verificationInFlight != nil {
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertNil(
            installed.pausedVerification,
            "returning to the app resumes the pass the system paused")
    }

    // MARK: - The screen is not the owner

    func testNavigatingAwayFromStorageDoesNotStopTheVerification() async throws {
        let artifact = try await registeredArtifact()
        let model = AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            store: .ephemeral(),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            persistsDefaults: false,
            storage: storage,
            downloadServices: .unavailable("not part of this test"),
            installed: installed, startDiscovery: false)
        model.openSettings(.storage)
        let verification = Task { await installed.verify(artifact, .full) }
        try await driver.waitUntilStarted()

        // Every navigation the product offers, while a terabyte is hashing.
        model.openSettings(.models)
        model.openSettingsRoot()
        model.destination = .chats
        try await Task.sleep(for: .milliseconds(30))

        XCTAssertEqual(installed.verificationInFlight, artifact.rootPath)
        XCTAssertNil(installed.pausedVerification)
        XCTAssertTrue(backgroundWork.isRunning)

        driver.release()
        await verification.value
        XCTAssertNil(installed.verificationInFlight)
        XCTAssertEqual(backgroundWork.endedSuccessfully, [true])
    }

    // MARK: - Telling the operator it finished

    func testPermissionIsAskedForWhenAVerificationStartsAndNotBefore() async throws {
        let artifact = try await registeredArtifact()
        XCTAssertEqual(
            announcer.prepareCount, 0,
            "nothing has been started, so there is nothing to ask permission about")

        driver.release()
        await installed.verify(artifact, .full)
        XCTAssertEqual(announcer.prepareCount, 1)
    }

    func testACompletedVerificationIsAnnouncedOnlyWhenTheAppIsElsewhere() async throws {
        let artifact = try await registeredArtifact()
        driver.release()

        frontmost.value = true
        await installed.verify(artifact, .full)
        XCTAssertEqual(
            announcer.announcements, [],
            "the row changing in front of the operator is the whole announcement")

        frontmost.value = false
        await installed.verify(artifact, .full)
        let announcement = try XCTUnwrap(announcer.announcements.last)
        XCTAssertEqual(announcer.announcements.count, 1)
        XCTAssertEqual(announcement.rootPath, artifact.rootPath)
        XCTAssertEqual(announcement.title, "\(artifact.displayName) verified")
        XCTAssertFalse(announcement.isFailure)
        XCTAssertTrue(
            announcement.body.contains("1 files") || announcement.body.contains("1 file"),
            "the body carries the size and the file count: \(announcement.body)")
        XCTAssertTrue(announcement.body.contains(","), announcement.body)
    }

    func testAFailedCheckAndARefusalBothNameTheirReason() async throws {
        let artifact = try await registeredArtifact()
        frontmost.value = false
        driver.release()

        driver.outcome = .wrongDigest
        await installed.verify(artifact, .full)
        let failure = try XCTUnwrap(announcer.announcements.last)
        XCTAssertTrue(failure.isFailure)
        XCTAssertEqual(failure.title, "\(artifact.displayName) did not verify")
        XCTAssertTrue(
            failure.body.contains("wrong digest"),
            "the notification says what was wrong: \(failure.body)")

        driver.outcome = .refused("the drive was unplugged mid-pass")
        await installed.verify(artifact, .full)
        let refusal = try XCTUnwrap(announcer.announcements.last)
        XCTAssertTrue(refusal.isFailure)
        XCTAssertEqual(refusal.body, "the drive was unplugged mid-pass")
    }

    func testAPausedVerificationAnnouncesNothingBecauseItHasNotEnded() async throws {
        let artifact = try await registeredArtifact()
        frontmost.value = false
        let verification = Task { await installed.verify(artifact, .full) }
        try await driver.waitUntilStarted()
        backgroundWork.expire()
        await verification.value

        XCTAssertNotNil(installed.pausedVerification)
        XCTAssertEqual(
            announcer.announcements, [],
            "a pass that will resume by itself has nothing to report yet")
    }

    func testACancelledVerificationAnnouncesNothing() async throws {
        let artifact = try await registeredArtifact()
        frontmost.value = false
        let verification = Task { await installed.verify(artifact, .full) }
        try await driver.waitUntilStarted()
        installed.cancelVerification()
        await verification.value

        XCTAssertEqual(announcer.announcements, [])
    }

    func testTheSecurityScopeStaysOpenForTheWholeVerification() async throws {
        let artifact = try await registeredArtifact()
        let verification = Task { await installed.verify(artifact, .full) }
        try await driver.waitUntilStarted()

        let scope = try XCTUnwrap(storage.resolvedScopesSnapshot.last)
        XCTAssertFalse(
            scope.isReleased,
            "the registered location's grant is held for the pass's whole lifetime")

        driver.release()
        await verification.value
        XCTAssertTrue(scope.isReleased, "and released once, when the pass ends")
    }

    func testACarriedForwardPassSaysWhatItReadAndWhatItStoodOn() async throws {
        driver.outcome = .matchedCarryingForward
        frontmost.value = false
        let artifact = try await registeredArtifact()
        let verification = Task { await installed.verify(artifact, .full) }
        try await driver.waitUntilStarted()
        driver.release()
        await verification.value

        let outcome = try XCTUnwrap(installed.verificationOutcome[artifact.rootPath])
        XCTAssertTrue(
            outcome.hasSuffix("3 files re-read, 627 carried forward from the previous revision."),
            "the row says why a terabyte finished in seconds: \(outcome)")
        let announcement = try XCTUnwrap(announcer.announcements.last)
        XCTAssertFalse(announcement.isFailure)
        XCTAssertEqual(
            announcement.body, "3 re-read, 627 carried forward",
            "and the notification does not claim a terabyte was read")
    }
}
