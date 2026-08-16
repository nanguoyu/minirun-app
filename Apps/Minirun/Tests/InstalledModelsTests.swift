import Foundation
import iOSBench
import MinirunKit
import StorageCore
import XCTest

@testable import MinirunApp

/// A `StorageManaging` over plain directories.
///
/// The real one mints security-scoped bookmarks, which a sandboxed test host
/// cannot do for a temporary directory without a user having picked it. What is
/// under test here is not bookmarking — `MinirunKitTests` covers that — but what
/// the app concludes from a scan, so the locations are simply remembered by
/// path.
private final class FixtureStorage: StorageManaging, @unchecked Sendable {
    private let lock = NSLock()
    private var remembered: [StorageKey: URL] = [:]
    private let scopeCoding = VerificationScopeCoding()
    private var resolvedScopes: [StorageScope] = []

    var lastResolvedScope: StorageScope? {
        lock.lock()
        defer { lock.unlock() }
        return resolvedScopes.last
    }

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

private final class VerificationScopeCoding: SecurityScopedURLCoding, @unchecked Sendable {
    func bookmarkData(for url: URL) throws -> Data { Data(url.path.utf8) }
    func resolve(_ data: Data) throws -> ResolvedBookmark {
        throw CocoaError(.fileReadCorruptFile)
    }
    func startAccessing(_ url: URL) -> Bool { true }
    func stopAccessing(_ url: URL) {}
}

private struct RootedVerificationTreeTransport: HTTPTransport {
    let treeBody: Data
    let paths: [String]

    func data(for request: HTTPRequest) async throws -> HTTPResponse {
        if request.url.path.contains("/revision/") {
            let revision = request.url.path.split(separator: "/").last.map(String.init) ?? ""
            let siblings = paths.map { ["rfilename": $0] }
            let body = try JSONSerialization.data(withJSONObject: [
                "sha": revision, "siblings": siblings,
            ])
            return HTTPResponse(statusCode: 200, headers: [:], body: body)
        }
        return HTTPResponse(statusCode: 200, headers: [:], body: treeBody)
    }

    func download(_ request: HTTPRequest, into sink: HTTPDownloadSink) async throws
        -> HTTPResponse
    {
        throw HTTPTransportError.notHTTP(url: request.url.absoluteString)
    }
}

/// Installed-ness comes from a directory. These are the app-side consequences:
/// the pill, the counter and the two filters all have to change when the disk
/// does, and none of them may consult the download history to decide.
@MainActor
final class InstalledModelsTests: XCTestCase {

    private var root: URL!
    private var storage: FixtureStorage!
    private var installed: InstalledModels!

    /// K3's real index, trimmed to the fields identity needs.
    private let k3Index = """
        {"format":"quantized_tile_container","source_repo":"moonshotai/Kimi-K3",
         "source_revision":"9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
         "files":372,"bytes":1559976181760}
        """

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-installed-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        storage = FixtureStorage()
        installed = InstalledModels(
            storage: storage, catalog: ModelCatalog.bundled,
            ledger: InMemoryVerificationLedger())
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func makeK3Artifact(payloadBytes: Int = 4096) throws {
        let directory = root.appendingPathComponent("k3-artifact")
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("global"), withIntermediateDirectories: true)
        try Data(k3Index.utf8).write(to: directory.appendingPathComponent("index.json"))
        try Data(repeating: 0x41, count: payloadBytes)
            .write(to: directory.appendingPathComponent("global/embed_tokens.bin"))
    }

    /// Waits for the detached scan. The scan runs off the main actor on purpose
    /// — walking a real volume is half a second of `stat` calls — so the test
    /// has to wait for it rather than read a value that is not there yet.
    private func scan() async throws {
        installed.refresh()
        for _ in 0..<200 {
            if !installed.isScanning && installed.scanCount > 0 { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("the scan did not finish")
    }

    func testAddingALocationIsWhatMakesAModelInstalled() async throws {
        try makeK3Artifact()
        try await scan()
        XCTAssertFalse(
            installed.isInstalled(.kimiK3),
            "with no location registered, nothing on any disk is visible")

        installed.addLocation(root)
        try await scan()
        XCTAssertTrue(installed.isInstalled(.kimiK3))
        XCTAssertTrue(installed.isRunnableHere(.kimiK3), "the fixture directory is mounted")
        XCTAssertEqual(installed.installedCount, 1)
        XCTAssertFalse(installed.isInstalled(.deepseekV4Flash))
    }

    func testRemovingTheLocationTakesTheModelWithIt() async throws {
        try makeK3Artifact()
        installed.addLocation(root)
        try await scan()
        XCTAssertTrue(installed.isInstalled(.kimiK3))

        let key = try XCTUnwrap(installed.locationKeys.first)
        installed.removeLocation(key)
        try await scan()
        XCTAssertFalse(
            installed.isInstalled(.kimiK3),
            "the bytes are still on disk; the app was simply told to stop looking there")
    }

    /// The pill's numbers are measured, not the catalog's. A partial artifact
    /// must not borrow the catalog's 1.56 TB and read as complete.
    func testTheReportedSizeIsTheSizeOnDiskAndTheShortfallIsNamed() async throws {
        try makeK3Artifact(payloadBytes: 4096)
        installed.addLocation(root)
        try await scan()

        let artifact = try XCTUnwrap(installed.preferredArtifact(for: .kimiK3))
        let expected = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        XCTAssertEqual(artifact.bytesOnDisk, UInt64(4096 + k3Index.utf8.count))
        XCTAssertNotEqual(artifact.bytesOnDisk, expected.totalBytes)
        XCTAssertEqual(artifact.missingFileCount, expected.totalFileCount - 2)
        XCTAssertEqual(artifact.isComplete, false)
        XCTAssertEqual(artifact.verification, .unverified)
    }

    func testAnArtifactWhoseIndexNamesNothingKnownIsListedAsUnidentified() async throws {
        let directory = root.appendingPathComponent("something-else")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(#"{"schema":"v9"}"#.utf8).write(to: directory.appendingPathComponent("index.json"))

        installed.addLocation(root)
        try await scan()
        XCTAssertEqual(installed.installedCount, 0)
        let unidentified = installed.report.unidentifiedArtifacts
        XCTAssertEqual(unidentified.count, 1)
        XCTAssertEqual(unidentified.first?.displayName, "something-else")
        XCTAssertEqual(unidentified.first?.index.topLevelKeys, ["schema"])
    }

    /// Catalog rows and artifact identity must change together. An index that is
    /// unknown to the bundled snapshot becomes identified after the App applies
    /// a refreshed snapshot; no relaunch and no second catalog owner is involved.
    func testUpdatingTheCatalogRescansDiscoveryAgainstThatSameSnapshot() async throws {
        let id = ModelID("future-model")
        let repoID = "nanguoyu/Future-Model-minirun"
        let directory = root.appendingPathComponent("future-artifact")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(
            """
            {"source_repo":"\(repoID)","source_revision":"\(String(repeating: "a", count: 40))",
             "files":0,"bytes":0}
            """.utf8
        ).write(to: directory.appendingPathComponent("index.json"))
        installed.addLocation(root)
        try await scan()
        XCTAssertFalse(installed.isInstalled(id))
        XCTAssertEqual(installed.report.unidentifiedArtifacts.count, 1)

        let descriptor = ModelDescriptor(
            id: id,
            displayName: "Future Model",
            architecture: .deepseekV4FlashMoE,
            layout: .v4FlashUnitBundle,
            source: .huggingFaceRepo(
                HuggingFaceRepoRef(repoID: repoID, revision: String(repeating: "a", count: 40))),
            payloadBytes: 0,
            metadataBytes: 0,
            payloadFileCount: 0,
            metadataFileCount: 1,
            largestFileBytes: 0,
            minimumBudgetBytes: nil,
            runner: .none,
            licenseName: "test",
            licenseAcknowledgementRequired: false,
            notes: [])
        let baseline = installed.scanCount
        installed.updateCatalog(
            ModelCatalogSnapshot(
                generatedAt: Date(),
                origin: .live,
                models: ModelCatalog.bundled.models + [descriptor]))
        for _ in 0..<200 {
            if !installed.isScanning && installed.scanCount > baseline { break }
            try await Task.sleep(for: .milliseconds(10))
        }

        XCTAssertGreaterThan(installed.scanCount, baseline)
        XCTAssertTrue(installed.isInstalled(id))
        XCTAssertTrue(installed.report.unidentifiedArtifacts.isEmpty)
    }

    func testARefreshRequestedDuringAScanAlwaysRunsOneFollowUpPass() async throws {
        let firstDate = Date(timeIntervalSince1970: 1)
        let secondDate = Date(timeIntervalSince1970: 2)
        let gate = InstalledModelScanGate(
            first: DiscoveryReport(locations: [], scannedAt: firstDate),
            second: DiscoveryReport(locations: [], scannedAt: secondDate))
        let coalescing = InstalledModels(
            storage: storage, catalog: ModelCatalog.bundled,
            ledger: InMemoryVerificationLedger(),
            scanOperation: { _, _, _ in await gate.scan() })

        coalescing.refresh()
        await gate.waitUntilFirstScanStarted()
        coalescing.refresh()
        await gate.releaseFirstScan()

        for _ in 0..<200 {
            if coalescing.scanCount == 2 { break }
            try await Task.sleep(for: .milliseconds(10))
        }

        XCTAssertEqual(coalescing.scanCount, 2)
        XCTAssertEqual(coalescing.report.scannedAt, secondDate)
        let callCount = await gate.callCount
        XCTAssertEqual(callCount, 2)
    }

    // MARK: - What the catalog screen shows

    func testTheCatalogStaysWholeAndTheLeadingStateFollowsTheDisk() async throws {
        try makeK3Artifact()
        installed.addLocation(root)
        try await scan()

        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunTest-\(UUID().uuidString)", isDirectory: true)
        let model = AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: CatalogFixtures.all),
            installed: installed,
            startDiscovery: false)
        defer { try? FileManager.default.removeItem(at: directory) }

        // Remote discovery remains whole, while the primary Models page is a
        // local inventory containing only the artifact the scan found.
        XCTAssertEqual(model.visibleEntries.count, model.entries.count)
        XCTAssertTrue(model.visibleEntries.contains { $0.id == .kimiK3 })
        XCTAssertEqual(model.localEntriesForPresentation.map(\.id), [.kimiK3])

        // Even a conflicting preview-download state must not overwrite what the
        // scan measured. This fixture has only two of K3's expected files, so
        // the leading symbol is incomplete rather than the mock's ready seal.
        model.download(.kimiK3)?.markReadyForPreview()
        let downloadState = model.state(of: .kimiK3)
        XCTAssertTrue(downloadState.isReady)
        let leading = ModelRowLeadingState.resolve(
            installations: installed.installations(of: .kimiK3),
            hasStorageLocation: installed.hasLocation,
            downloadState: downloadState)
        XCTAssertEqual(leading, .scannedIncomplete)
        XCTAssertEqual(leading.glyphName, "cube")
        XCTAssertFalse(leading.glyphName.localizedCaseInsensitiveContains("drive"))
        XCTAssertNotEqual(leading.glyphName, downloadState.glyphName)
        XCTAssertEqual(model.modelAvailability(for: .kimiK3), .incompleteArtifact)
        XCTAssertNil(
            model.newConversation(),
            "a mock ready state must not create a chat over a scan-known incomplete artifact")
    }

    /// Runner availability belongs in the row/detail, not in a filter that makes
    /// an installed model disappear from the catalog.
    func testTheSingleCatalogKeepsAnInstalledModelWithNoRunnerVisible() async throws {
        let directory = root.appendingPathComponent("h3-artifact")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(
            #"{"sources":{"dit":{"repo":"pipenetwork/MiniMax-H3-MLX-8bit"}},"total_bytes":1,"units":[]}"#
                .utf8
        ).write(to: directory.appendingPathComponent("index.json"))

        installed.addLocation(root)
        try await scan()
        XCTAssertTrue(installed.isInstalled(.minimaxH3))

        let store = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunTest-\(UUID().uuidString)", isDirectory: true)
        let model = AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            store: ConversationStore(directory: store),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: CatalogFixtures.all),
            installed: installed,
            startDiscovery: false)
        defer { try? FileManager.default.removeItem(at: store) }

        XCTAssertTrue(model.visibleEntries.contains { $0.id == .minimaxH3 })
        XCTAssertEqual(model.visibleEntries.count, model.entries.count)
        XCTAssertEqual(model.localEntriesForPresentation.map(\.id), [.minimaxH3])
        XCTAssertNil(model.capabilities(for: .minimaxH3), "this build still ships no H3 runner")
        XCTAssertEqual(
            ModelRowLeadingState.resolve(
                installations: installed.installations(of: .minimaxH3),
                hasStorageLocation: installed.hasLocation,
                downloadState: model.state(of: .minimaxH3)),
            .scannedIncomplete)
    }

    /// Models may manage a published H3 container, but the chat picker contains
    /// only fully verified artifacts with a runtime in this build.
    func testModelPickerChoicesContainOnlyVerifiedRunnableChatModels() throws {
        let allModels = ModelCatalog.bundled
        let snapshot = ProductCatalogPolicy.applying(to: allModels)
        let k3 = try scannedArtifact(
            try XCTUnwrap(allModels.descriptor(.kimiK3)),
            root: "/Volumes/Test/k3",
            verification: .fullyVerified)
        let h3 = try scannedArtifact(
            try XCTUnwrap(allModels.descriptor(.minimaxH3)),
            root: "/Volumes/Test/h3",
            verification: .fullyVerified)
        let unverifiedV4 = try scannedArtifact(
            try XCTUnwrap(allModels.descriptor(.deepseekV4Flash)),
            root: "/Volumes/Test/v4",
            verification: .spotChecked)
        let unknown = DiscoveredArtifact(
            rootPath: "/Volumes/Test/mystery",
            locationPath: "/Volumes/Test",
            index: .unreadable,
            model: nil,
            displayName: "mystery",
            bytesOnDisk: 12,
            fileCount: 1,
            expectedFileCount: nil,
            expectedBytes: nil,
            verification: .unverified,
            verifiedAt: nil,
            scannedAt: Date())
        let report = DiscoveryReport(
            locations: [
                LocationScan(
                    rootPath: "/Volumes/Test",
                    displayName: "Test",
                    isMounted: true,
                    storageKey: nil,
                    artifacts: [k3, h3, unverifiedV4, unknown],
                    unreadable: [:],
                    scannedAt: Date())
            ],
            scannedAt: Date())
        let scanned = InstalledModels(
            storage: storage,
            catalog: snapshot,
            ledger: InMemoryVerificationLedger(),
            initialReport: report)
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunPickerTest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(
            catalogSnapshot: snapshot,
            runtimes: .preview(entries: AppCatalogAdapter.entries(from: snapshot)),
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: AppCatalogAdapter.entries(from: snapshot)),
            installed: scanned,
            seedRecordedRuns: false,
            startDiscovery: false)

        let choices = model.modelPickerChoices(current: .kimiK3)
        let k3Choice = try XCTUnwrap(choices.first { $0.modelID == .kimiK3 })
        XCTAssertTrue(k3Choice.isSelectable)
        XCTAssertEqual(k3Choice.availability, .available)

        XCTAssertFalse(choices.contains { $0.modelID == .minimaxH3 })
        XCTAssertFalse(choices.contains { $0.modelID == .deepseekV4Flash })
        XCTAssertFalse(choices.contains { $0.modelID == nil })

        // A persisted choice that is no longer scanned stays in the document,
        // but it cannot leak into a menu promised to contain verified models.
        let retired = ModelID("retired-model")
        model.defaults.model = retired
        XCTAssertFalse(
            model.modelPickerChoices(current: retired).contains {
                $0.modelID == retired
            })
        XCTAssertEqual(model.defaults.model, retired)
    }

    /// Presence and byte-count completeness are not a correctness gate. Until
    /// the scan ledger says fully verified, the model remains visible but may
    /// not become a request or a new conversation.
    func testACompleteButUnverifiedArtifactIsVisibleAndCannotRun() throws {
        let snapshot = ModelCatalog.bundled
        let artifact = try scannedArtifact(
            try XCTUnwrap(snapshot.descriptor(.kimiK3)),
            root: "/Volumes/Test/k3-unverified",
            verification: .spotChecked)
        let report = DiscoveryReport(
            locations: [
                LocationScan(
                    rootPath: "/Volumes/Test",
                    displayName: "Test",
                    isMounted: true,
                    storageKey: nil,
                    artifacts: [artifact],
                    unreadable: [:],
                    scannedAt: Date())
            ],
            scannedAt: Date())
        let scanned = InstalledModels(
            storage: storage,
            catalog: snapshot,
            ledger: InMemoryVerificationLedger(),
            initialReport: report)
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunUnverifiedPickerTest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(
            catalogSnapshot: snapshot,
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: AppCatalogAdapter.entries(from: snapshot)),
            installed: scanned,
            seedRecordedRuns: false,
            startDiscovery: false)

        XCTAssertTrue(model.installed.isInstalled(.kimiK3))
        XCTAssertNil(model.installed.runnableArtifact(for: .kimiK3))
        XCTAssertEqual(model.modelAvailability(for: .kimiK3), .unverifiedArtifact)
        XCTAssertNil(model.newConversation())
        XCTAssertFalse(
            model.modelPickerChoices(current: .kimiK3).contains {
                $0.modelID == .kimiK3
            })
    }

    func testFullVerificationConfirmationNamesTheReadAndItsCost() throws {
        let descriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let artifact = try scannedArtifact(
            descriptor, root: "/Volumes/Test/k3-full-verification")

        XCTAssertEqual(FullVerificationConfirmation.bytes(for: artifact), descriptor.totalBytes)
        let message = FullVerificationConfirmation.message(for: artifact)
        XCTAssertTrue(message.contains(MRFormat.publishedBytes(descriptor.totalBytes)))
        XCTAssertTrue(message.contains("may take a long time"))
        XCTAssertTrue(message.contains("does not modify model files"))
        XCTAssertTrue(message.contains("can be cancelled"))
    }

    func testRemovalConfirmationNamesExactModelPathSizeAndPermanence() throws {
        let descriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let artifact = try scannedArtifact(
            descriptor, root: "/Volumes/Test/k3-removal",
            location: "/Volumes/Test")

        let message = LocalCopyRemovalConfirmation.message(
            modelName: descriptor.displayName, artifact: artifact)

        XCTAssertTrue(message.contains(descriptor.displayName))
        XCTAssertTrue(message.contains(artifact.rootPath))
        XCTAssertTrue(message.contains(MRFormat.measuredBytes(artifact.bytesOnDisk)))
        XCTAssertTrue(message.localizedCaseInsensitiveContains("permanently"))
        XCTAssertTrue(message.localizedCaseInsensitiveContains("cannot be undone"))
        XCTAssertTrue(message.contains("Other copies"))
    }

    func testLocalCopyRemovalReplaysStorageAuthorityAndForgetsVerificationEvidence()
        async throws
    {
        let descriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let registered = root.appendingPathComponent("removal-location", isDirectory: true)
        let artifactURL = registered.appendingPathComponent("published/k3", isDirectory: true)
        try FileManager.default.createDirectory(
            at: artifactURL, withIntermediateDirectories: true)
        try Data("fixture".utf8).write(to: artifactURL.appendingPathComponent("payload.bin"))
        let artifact = try scannedArtifact(
            descriptor, root: artifactURL.path, location: registered.path)
        let key = StorageKey(InstalledModels.locationKeyPrefix + "removal")
        _ = try storage.remember(registered, as: key)
        let ledger = RecordingVerificationLedger()
        let probe = RemovalArtifactProbe()
        let fixtureStorage = try XCTUnwrap(storage)
        let scoped = InstalledModels(
            storage: storage, catalog: ModelCatalog.bundled, ledger: ledger,
            initialReport: DiscoveryReport(
                locations: [
                    LocationScan(
                        rootPath: registered.path, displayName: "Removal",
                        isMounted: true, storageKey: key, artifacts: [artifact],
                        unreadable: [:], scannedAt: Date())
                ], scannedAt: Date()),
            removalOperation: { authority in
                await probe.record(
                    rootURL: authority.rootURL, scope: fixtureStorage.lastResolvedScope)
                return ArtifactRemovalReceipt(
                    rootPath: authority.rootURL.path, removedEntryCount: 1,
                    removedRegularFileBytes: 7)
            },
            scanOperation: { _, _, _ in .empty })

        let receipt = try await scoped.remove(artifact)

        XCTAssertEqual(receipt.rootPath, artifactURL.standardizedFileURL.path)
        let observedRoot = await probe.rootURL
        let scopeWasActive = await probe.scopeWasActive
        XCTAssertEqual(observedRoot, artifactURL.standardizedFileURL)
        XCTAssertTrue(scopeWasActive)
        XCTAssertTrue(storage.lastResolvedScope?.isReleased == true)
        XCTAssertEqual(ledger.forgottenPaths, [artifactURL.standardizedFileURL.path])
        XCTAssertNil(scoped.removalInFlight)
        XCTAssertTrue(
            scoped.removalOutcome[artifact.rootPath]?.contains(MRFormat.measuredBytes(7))
                == true)
    }

    func testAppRemovalDeletesTheExactDirectoryAndMatchingDurableJobRecord() async throws {
        try makeK3Artifact(payloadBytes: 64)
        installed.addLocation(root)
        try await scan()
        let artifact = try XCTUnwrap(installed.preferredArtifact(for: .kimiK3))
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let stateStore = InMemoryDownloadStateStore()
        let job = DownloadJobID()
        try stateStore.save(
            DownloadJobState(
                job: job, plan: plan, destinationPath: artifact.rootPath,
                destinationBookmark: Data("bookmark".utf8),
                artifactRootPath: artifact.rootPath,
                artifactRootDevice: nil, artifactRootInode: nil,
                options: DownloadOptions(), fileStates: [:], runState: .finished,
                createdAt: Date(), updatedAt: Date()))
        let conversations = root.appendingPathComponent(
            "RemovalConversations", isDirectory: true)
        let model = AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            store: ConversationStore(directory: conversations),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            storage: storage,
            downloadServices: AppDownloadServices(
                manager: MockDownloadManager(entries: CatalogFixtures.all),
                stateStore: stateStore, restoresPersistedJobs: false, setupError: nil),
            installed: installed, startDiscovery: false)

        let removed = await model.removeLocalCopy(artifact)

        XCTAssertTrue(removed)
        XCTAssertFalse(FileManager.default.fileExists(atPath: artifact.rootPath))
        XCTAssertNil(try stateStore.load(job))
        XCTAssertTrue(
            model.localCopyRemovalMessage(for: artifact)?.contains("Removed") == true)
    }

    func testVerificationResolvesTheOwningLocationAndRemapsTheArtifact() async throws {
        let descriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let scannedLocation = "/Volumes/OLD-MOUNT/models"
        let scannedArtifact = try scannedArtifact(
            descriptor,
            root: scannedLocation + "/published/k3",
            location: scannedLocation)
        let key = StorageKey(InstalledModels.locationKeyPrefix + "verification-remap")
        let resolvedLocation = root.appendingPathComponent("new-mount/models", isDirectory: true)
        try FileManager.default.createDirectory(
            at: resolvedLocation.appendingPathComponent("published/k3", isDirectory: true),
            withIntermediateDirectories: true)
        _ = try storage.remember(resolvedLocation, as: key)
        let fixtureStorage = try XCTUnwrap(storage)
        let report = DiscoveryReport(
            locations: [
                LocationScan(
                    rootPath: scannedLocation,
                    displayName: "OLD-MOUNT",
                    isMounted: true,
                    storageKey: key,
                    artifacts: [scannedArtifact],
                    unreadable: [:],
                    scannedAt: Date())
            ],
            scannedAt: Date())
        let observed = VerificationArtifactProbe()
        let scoped = InstalledModels(
            storage: storage,
            catalog: ModelCatalog.bundled,
            ledger: InMemoryVerificationLedger(),
            initialReport: report,
            verificationOperation: { artifact, _, authority, _, _ in
                await observed.record(
                    artifact, scope: fixtureStorage.lastResolvedScope)
                XCTAssertEqual(authority.rootURL.path, artifact.rootPath)
                XCTAssertFalse(
                    fixtureStorage.lastResolvedScope?.isReleased ?? true,
                    "the security scope must remain live while the verifier reads")
                return Self.successfulVerificationReport(for: descriptor.id)
            })

        await scoped.verify(scannedArtifact, .spotCheck())

        let remapped = await observed.artifact
        XCTAssertEqual(
            remapped?.rootPath,
            resolvedLocation.appendingPathComponent("published/k3", isDirectory: true)
                .standardizedFileURL.path)
        XCTAssertEqual(remapped?.locationPath, resolvedLocation.standardizedFileURL.path)
        let observedScope = await observed.scope
        let verificationScope = try XCTUnwrap(observedScope)
        XCTAssertTrue(verificationScope.isReleased)
    }

    func testConfirmedVerificationReplacesTheVisibleScanRowImmediately() throws {
        let descriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(.deepseekV4Flash))
        let artifact = try scannedArtifact(
            descriptor, root: "/Volumes/Test/v4", verification: .unverified)
        let report = DiscoveryReport(
            locations: [
                LocationScan(
                    rootPath: artifact.locationPath,
                    displayName: "Test",
                    isMounted: true,
                    storageKey: StorageKey("test"),
                    artifacts: [artifact], unreadable: [:], scannedAt: Date())
            ],
            scannedAt: Date())
        let checkedAt = Date(timeIntervalSince1970: 123_456)

        let rescannedArtifact = try scannedArtifact(
            descriptor, root: artifact.rootPath, verification: .unverified)
        let rescannedReport = DiscoveryReport(
            locations: [
                LocationScan(
                    rootPath: artifact.locationPath,
                    displayName: "Test",
                    isMounted: true,
                    storageKey: StorageKey("test"),
                    artifacts: [rescannedArtifact], unreadable: [:],
                    scannedAt: Date(timeIntervalSince1970: 123_455))
            ],
            scannedAt: Date(timeIntervalSince1970: 123_455))

        let updated = InstalledModels.applyingVerification(
            .fullyVerified, checkedAt: checkedAt, to: artifact, in: rescannedReport)

        let visible = try XCTUnwrap(updated.installations(of: .deepseekV4Flash).first)
        XCTAssertEqual(visible.verification, .fullyVerified)
        XCTAssertEqual(visible.verifiedAt, checkedAt)
        XCTAssertEqual(visible.bytesOnDisk, artifact.bytesOnDisk)
        XCTAssertEqual(visible.index, artifact.index)
    }

    func testRegisteredRunReferenceRequiresMatchingCompleteEvidence() throws {
        let descriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let location = root.appendingPathComponent("run-location", isDirectory: true)
        let artifactURL = location.appendingPathComponent("k3", isDirectory: true)
        try FileManager.default.createDirectory(at: artifactURL, withIntermediateDirectories: true)
        let artifact = try scannedArtifact(
            descriptor, root: artifactURL.path, location: location.path,
            verification: .fullyVerified)
        let key = StorageKey(InstalledModels.locationKeyPrefix + "run-rooted-authority")
        _ = try storage.remember(location, as: key)
        let scoped = InstalledModels(
            storage: storage, catalog: ModelCatalog.bundled,
            ledger: InMemoryVerificationLedger(),
            initialReport: DiscoveryReport(
                locations: [
                    LocationScan(
                        rootPath: location.path, displayName: "run-location",
                        isMounted: true, storageKey: key, artifacts: [artifact],
                        unreadable: [:], scannedAt: Date())
                ], scannedAt: Date()))

        XCTAssertThrowsError(try scoped.runnableArtifactReference(for: descriptor.id)) { error in
            guard case InstalledArtifactAccessError.verificationEvidenceUnavailable(let path) = error
            else { return XCTFail("unexpected error: \(error)") }
            XCTAssertEqual(path, artifact.rootPath)
        }
        XCTAssertTrue(storage.lastResolvedScope?.isReleased == true)
    }

    func testRootedVerificationSucceedsAndRecordsEvidenceWithoutURLReopen() async throws {
        let bytes = Data("rooted success".utf8)
        let digest = FileDigest.sha256(hex: FileDigestComputer.sha256(of: bytes))
        let file = RepoFile(
            path: "payload/weights.bin", sizeBytes: UInt64(bytes.count), digest: digest,
            isPayload: true)
        let treeBody = try JSONSerialization.data(withJSONObject: [[
            "type": "file", "path": file.path,
            "size": NSNumber(value: file.sizeBytes),
            "lfs": ["oid": digest.hex, "size": NSNumber(value: file.sizeBytes)],
        ]], options: [.sortedKeys])
        let base = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let descriptor = ModelDescriptor(
            id: base.id, displayName: base.displayName, architecture: base.architecture,
            layout: base.layout,
            source: .huggingFaceRepo(
                HuggingFaceRepoRef(
                    repoID: "fixture/K3-minirun", revision: String(repeating: "b", count: 40))),
            payloadBytes: file.sizeBytes, metadataBytes: 0,
            payloadFileCount: 1, metadataFileCount: 0,
            largestFileBytes: file.sizeBytes,
            minimumBudgetBytes: base.minimumBudgetBytes, runner: base.runner,
            licenseName: base.licenseName,
            licenseAcknowledgementRequired: base.licenseAcknowledgementRequired,
            notes: [])
        let catalog = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .bundled, models: [descriptor])
        let registered = root.appendingPathComponent("rooted-success", isDirectory: true)
        let artifactURL = registered.appendingPathComponent("k3", isDirectory: true)
        try FileManager.default.createDirectory(
            at: artifactURL.appendingPathComponent("payload", isDirectory: true),
            withIntermediateDirectories: true)
        try bytes.write(to: artifactURL.appendingPathComponent(file.path))
        let authority = try ArtifactVerificationRoot.open(
            relativeComponents: ["k3"], beneath: registered)
        let artifact = try scannedArtifact(
            descriptor, root: artifactURL.path, location: registered.path)
        let ledger = RecordingVerificationLedger()
        let verifier = ArtifactVerifier(
            tree: HuggingFaceTreeClient(
                transport: RootedVerificationTreeTransport(
                    treeBody: treeBody, paths: [file.path]),
                endpoint: URL(string: "https://verification.invalid")!),
            catalog: catalog, ledger: ledger)

        let report = try await verifier.verify(artifact, .full, rootedAt: authority)

        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(ledger.recordCount, 1)
    }

    func testIntermediateReplacementAfterCapabilityOpenCannotEarnVerificationEvidence()
        async throws
    {
        let base = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let revision = String(repeating: "a", count: 40)
        let registered = root.appendingPathComponent("race-registered", isDirectory: true)
        let intermediate = registered.appendingPathComponent("models", isDirectory: true)
        let artifactURL = intermediate.appendingPathComponent("k3", isDirectory: true)
        let payloadPath = "payload/weights.bin"
        let payloadURL = artifactURL.appendingPathComponent(payloadPath)
        let expected = Data("trusted artifact".utf8)
        try FileManager.default.createDirectory(
            at: payloadURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try expected.write(to: payloadURL)

        let outsideParent = root.appendingPathComponent("race-outside", isDirectory: true)
        let outsideArtifact = outsideParent.appendingPathComponent("k3", isDirectory: true)
        try FileManager.default.createDirectory(
            at: outsideArtifact.appendingPathComponent("payload", isDirectory: true),
            withIntermediateDirectories: true)
        try Data("attacker bytes!!".utf8).write(
            to: outsideArtifact.appendingPathComponent(payloadPath))

        let digest = FileDigest.sha256(hex: FileDigestComputer.sha256(of: expected))
        let file = RepoFile(
            path: payloadPath, sizeBytes: UInt64(expected.count), digest: digest,
            isPayload: true)
        let treeObject: [[String: Any]] = [[
            "type": "file", "path": file.path,
            "size": NSNumber(value: file.sizeBytes),
            "lfs": ["oid": digest.hex, "size": NSNumber(value: file.sizeBytes)],
        ]]
        let treeBody = try JSONSerialization.data(
            withJSONObject: treeObject, options: [.sortedKeys])
        let descriptor = ModelDescriptor(
            id: base.id, displayName: base.displayName, architecture: base.architecture,
            layout: base.layout,
            source: .huggingFaceRepo(
                HuggingFaceRepoRef(repoID: "fixture/K3-minirun", revision: revision)),
            payloadBytes: file.sizeBytes, metadataBytes: 0,
            payloadFileCount: 1, metadataFileCount: 0,
            largestFileBytes: file.sizeBytes,
            minimumBudgetBytes: base.minimumBudgetBytes, runner: base.runner,
            licenseName: base.licenseName,
            licenseAcknowledgementRequired: base.licenseAcknowledgementRequired,
            notes: [])
        let catalog = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .bundled, models: [descriptor])
        let artifact = try scannedArtifact(
            descriptor, root: artifactURL.path, location: registered.path)
        let key = StorageKey(InstalledModels.locationKeyPrefix + "verification-race")
        _ = try storage.remember(registered, as: key)
        let ledger = RecordingVerificationLedger()
        let verificationGate = RootedVerificationProgressGate()
        let verifier = ArtifactVerifier(
            tree: HuggingFaceTreeClient(
                transport: RootedVerificationTreeTransport(
                    treeBody: treeBody, paths: [file.path]),
                endpoint: URL(string: "https://verification.invalid")!),
            catalog: catalog, ledger: ledger)
        let heldIntermediate = root.appendingPathComponent(
            "race-held-models", isDirectory: true)
        let scoped = InstalledModels(
            storage: storage, catalog: catalog, ledger: ledger,
            initialReport: DiscoveryReport(
                locations: [
                    LocationScan(
                        rootPath: registered.path, displayName: "race-registered",
                        isMounted: true, storageKey: key, artifacts: [artifact],
                        unreadable: [:], scannedAt: Date())
                ], scannedAt: Date()),
            verificationOperation: { artifact, request, authority, resumption, progress in
                await verificationGate.pauseBeforeVerification()
                return try await verifier.verify(
                    artifact, request, rootedAt: authority, resumption: resumption,
                    onProgress: progress)
            })

        let verification = Task { await scoped.verify(artifact, .full) }
        await verificationGate.waitUntilStarted()
        let verificationScope = try XCTUnwrap(storage.lastResolvedScope)
        // The verifier has opened and checked its held artifact root. Replace
        // an intermediate component before its first payload open; every read
        // must still be relative to that held descriptor.
        try FileManager.default.moveItem(at: intermediate, to: heldIntermediate)
        try FileManager.default.createSymbolicLink(
            at: intermediate, withDestinationURL: outsideParent)
        await verificationGate.release()
        await verification.value

        XCTAssertEqual(ledger.recordCount, 0)
        XCTAssertTrue(verificationScope.isReleased)
        let outcome = try XCTUnwrap(scoped.verificationOutcome[artifact.rootPath])
        XCTAssertTrue(
            outcome.contains("could not establish") || outcome.contains("changed"),
            "unexpected outcome: \(outcome)")
    }

    func testVerificationRefusesAnArtifactOutsideItsOwningLocationWithoutCallingVerifier()
        async throws
    {
        let descriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let scannedLocation = "/Volumes/Registered"
        let escapingArtifact = try scannedArtifact(
            descriptor,
            root: scannedLocation + "/../Other/k3",
            location: scannedLocation)
        let key = StorageKey(InstalledModels.locationKeyPrefix + "verification-escape")
        _ = try storage.remember(root, as: key)
        let report = DiscoveryReport(
            locations: [
                LocationScan(
                    rootPath: scannedLocation,
                    displayName: "Registered",
                    isMounted: true,
                    storageKey: key,
                    artifacts: [escapingArtifact],
                    unreadable: [:],
                    scannedAt: Date())
            ],
            scannedAt: Date())
        let ledger = RecordingVerificationLedger()
        let probe = VerificationOperationProbe()
        let scoped = InstalledModels(
            storage: storage,
            catalog: ModelCatalog.bundled,
            ledger: ledger,
            initialReport: report,
            verificationOperation: { _, _, _, _, _ in
                await probe.markCalled()
                return Self.successfulVerificationReport(for: descriptor.id)
            })

        await scoped.verify(escapingArtifact, .full)

        let operationWasCalled = await probe.wasCalled
        XCTAssertFalse(operationWasCalled)
        XCTAssertEqual(ledger.recordCount, 0)
        XCTAssertTrue(
            storage.resolvedScopesSnapshot.first?.isReleased ?? false,
            "the verification scope must be released even if the follow-up scan resolves another")
        let outcome = try XCTUnwrap(scoped.verificationOutcome[escapingArtifact.rootPath])
        XCTAssertTrue(outcome.contains("outside its registered storage location"))
    }

    func testVerificationRefusesAnIntermediateSymlinkEscapeWithoutCallingVerifier() async throws {
        let descriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let registered = root.appendingPathComponent("registered", isDirectory: true)
        let outside = root.appendingPathComponent("outside", isDirectory: true)
        let outsideArtifact = outside.appendingPathComponent("k3", isDirectory: true)
        try FileManager.default.createDirectory(
            at: outsideArtifact, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: registered, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: registered.appendingPathComponent("link", isDirectory: true),
            withDestinationURL: outside)
        let scannedRoot = registered.appendingPathComponent("link/k3", isDirectory: true).path
        let artifact = try scannedArtifact(
            descriptor, root: scannedRoot, location: registered.path)
        let key = StorageKey(InstalledModels.locationKeyPrefix + "verification-symlink")
        _ = try storage.remember(registered, as: key)
        let ledger = RecordingVerificationLedger()
        let probe = VerificationOperationProbe()
        let scoped = InstalledModels(
            storage: storage, catalog: ModelCatalog.bundled, ledger: ledger,
            initialReport: DiscoveryReport(
                locations: [
                    LocationScan(
                        rootPath: registered.path, displayName: "registered",
                        isMounted: true, storageKey: key, artifacts: [artifact],
                        unreadable: [:], scannedAt: Date())
                ], scannedAt: Date()),
            verificationOperation: { _, _, _, _, _ in
                await probe.markCalled()
                return Self.successfulVerificationReport(for: descriptor.id)
            })

        await scoped.verify(artifact, .spotCheck())

        let verifierWasCalled = await probe.wasCalled
        XCTAssertFalse(verifierWasCalled)
        XCTAssertEqual(ledger.recordCount, 0)
        let verificationScope = try XCTUnwrap(storage.resolvedScopesSnapshot.first)
        XCTAssertTrue(verificationScope.isReleased)
        XCTAssertTrue(
            scoped.verificationOutcome[artifact.rootPath]?
                .contains("outside its registered storage location") == true)
    }

    func testVerificationWithoutRegisteredStorageAuthorityFailsClosed() async throws {
        let descriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let artifact = try scannedArtifact(
            descriptor, root: "/Volumes/Preview/k3", location: "/Volumes/Preview")
        let report = DiscoveryReport(
            locations: [
                LocationScan(
                    rootPath: "/Volumes/Preview",
                    displayName: "Preview",
                    isMounted: true,
                    storageKey: nil,
                    artifacts: [artifact],
                    unreadable: [:],
                    scannedAt: Date())
            ],
            scannedAt: Date())
        let ledger = RecordingVerificationLedger()
        let probe = VerificationOperationProbe()
        let scoped = InstalledModels(
            storage: storage,
            catalog: ModelCatalog.bundled,
            ledger: ledger,
            initialReport: report,
            verificationOperation: { _, _, _, _, _ in
                await probe.markCalled()
                return Self.successfulVerificationReport(for: descriptor.id)
            })

        await scoped.verify(artifact, .spotCheck())

        let operationWasCalled = await probe.wasCalled
        XCTAssertFalse(operationWasCalled)
        XCTAssertEqual(ledger.recordCount, 0)
        let outcome = try XCTUnwrap(scoped.verificationOutcome[artifact.rootPath])
        XCTAssertTrue(outcome.contains("no registered storage bookmark owns"))
    }

    func testCancellingVerificationStopsTheRetainedTaskAndWritesNoPositiveRecord() async throws {
        let descriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let location = root.appendingPathComponent("cancel-location", isDirectory: true)
        let artifactURL = location.appendingPathComponent(
            "k3-cancel-verification", isDirectory: true)
        try FileManager.default.createDirectory(
            at: artifactURL, withIntermediateDirectories: true)
        let artifact = try scannedArtifact(
            descriptor, root: artifactURL.path, location: location.path)
        let key = StorageKey(InstalledModels.locationKeyPrefix + "verification-cancel")
        _ = try storage.remember(location, as: key)
        let fixtureStorage = try XCTUnwrap(storage)
        let initialReport = DiscoveryReport(
            locations: [
                LocationScan(
                    rootPath: location.path,
                    displayName: "Test",
                    isMounted: true,
                    storageKey: key,
                    artifacts: [artifact],
                    unreadable: [:],
                    scannedAt: Date())
            ],
            scannedAt: Date())
        let probe = VerificationCancellationProbe()
        let ledger = RecordingVerificationLedger()
        let cancellable = InstalledModels(
            storage: storage, catalog: ModelCatalog.bundled, ledger: ledger,
            initialReport: initialReport,
            verificationOperation: { _, _, _, _, _ in
                await probe.markStarted()
                XCTAssertFalse(fixtureStorage.lastResolvedScope?.isReleased ?? true)
                do {
                    try await Task.sleep(for: .seconds(60))
                    throw CancellationError()
                } catch {
                    if error is CancellationError { await probe.markCancelled() }
                    throw error
                }
            })

        let verification = Task { await cancellable.verify(artifact, .full) }
        await probe.waitUntilStarted()
        XCTAssertEqual(cancellable.verificationInFlight, artifact.rootPath)
        let verificationScope = try XCTUnwrap(storage.lastResolvedScope)

        cancellable.cancelVerification()
        await verification.value

        let didCancel = await probe.didCancel
        XCTAssertTrue(didCancel)
        XCTAssertNil(cancellable.verificationInFlight)
        XCTAssertNil(cancellable.verificationProgress)
        XCTAssertEqual(cancellable.verificationOutcome[artifact.rootPath], "Cancelled")
        XCTAssertEqual(ledger.recordCount, 0)
        XCTAssertTrue(verificationScope.isReleased)
    }

    private func scannedArtifact(
        _ descriptor: ModelDescriptor,
        root: String,
        location: String = "/Volumes/Test",
        verification: ArtifactVerification = .unverified
    ) throws
        -> DiscoveredArtifact
    {
        DiscoveredArtifact(
            rootPath: root,
            locationPath: location,
            index: .unreadable,
            model: descriptor.id,
            displayName: descriptor.displayName,
            bytesOnDisk: descriptor.totalBytes,
            fileCount: descriptor.totalFileCount,
            expectedFileCount: descriptor.totalFileCount,
            expectedBytes: descriptor.totalBytes,
            verification: verification,
            verifiedAt: verification == .unverified ? nil : Date(),
            scannedAt: Date())
    }

    nonisolated private static func successfulVerificationReport(
        for model: ModelID
    ) -> VerificationReport {
        VerificationReport(
            job: DownloadJobID(), model: model, depth: .digestPayload, checkedAt: Date(),
            ok: ["payload.bin"], missing: [], wrongSize: [], wrongDigest: [], unreadable: [:],
            extraneous: [])
    }
}

private actor VerificationArtifactProbe {
    private(set) var artifact: DiscoveredArtifact?
    private(set) var scope: StorageScope?

    func record(_ artifact: DiscoveredArtifact, scope: StorageScope? = nil) {
        self.artifact = artifact
        self.scope = scope
    }
}

private actor RemovalArtifactProbe {
    private(set) var rootURL: URL?
    private(set) var scopeWasActive = false

    func record(rootURL: URL, scope: StorageScope?) {
        self.rootURL = rootURL
        scopeWasActive = !(scope?.isReleased ?? true)
    }
}

private actor InstalledModelScanGate {
    private let first: DiscoveryReport
    private let second: DiscoveryReport
    private var calls = 0
    private var firstStarted = false
    private var firstReleased = false
    private var startedWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

    init(first: DiscoveryReport, second: DiscoveryReport) {
        self.first = first
        self.second = second
    }

    var callCount: Int { calls }

    func scan() async -> DiscoveryReport {
        calls += 1
        if calls == 1 {
            firstStarted = true
            let waiters = startedWaiters
            startedWaiters.removeAll()
            for waiter in waiters { waiter.resume() }
            if !firstReleased {
                await withCheckedContinuation { releaseWaiters.append($0) }
            }
            return first
        }
        return second
    }

    func waitUntilFirstScanStarted() async {
        guard !firstStarted else { return }
        await withCheckedContinuation { startedWaiters.append($0) }
    }

    func releaseFirstScan() {
        firstReleased = true
        let waiters = releaseWaiters
        releaseWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }
}

private actor RootedVerificationProgressGate {
    private var didStart = false
    private var didRelease = false
    private var startedWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

    func pauseBeforeVerification() async {
        if !didStart {
            didStart = true
            let waiters = startedWaiters
            startedWaiters.removeAll()
            for waiter in waiters { waiter.resume() }
        }
        guard !didRelease else { return }
        await withCheckedContinuation { releaseWaiters.append($0) }
    }

    func waitUntilStarted() async {
        guard !didStart else { return }
        await withCheckedContinuation { startedWaiters.append($0) }
    }

    func release() {
        didRelease = true
        let waiters = releaseWaiters
        releaseWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }
}

private actor VerificationOperationProbe {
    private(set) var wasCalled = false
    func markCalled() { wasCalled = true }
}

private actor VerificationCancellationProbe {
    private var started = false
    private var cancelled = false
    private var startedWaiters: [CheckedContinuation<Void, Never>] = []

    var didCancel: Bool { cancelled }

    func markStarted() {
        started = true
        let waiters = startedWaiters
        startedWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }

    func waitUntilStarted() async {
        guard !started else { return }
        await withCheckedContinuation { startedWaiters.append($0) }
    }

    func markCancelled() { cancelled = true }
}

private final class RecordingVerificationLedger: ArtifactVerificationLedger, @unchecked Sendable {
    private let lock = NSLock()
    private var records: [ArtifactVerificationRecord] = []
    private var forgotten: [String] = []

    var recordCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return records.count
    }

    var forgottenPaths: [String] {
        lock.lock()
        defer { lock.unlock() }
        return forgotten
    }

    func record(_ record: ArtifactVerificationRecord) {
        lock.lock()
        records.append(record)
        lock.unlock()
    }

    func record(matching lookup: ArtifactVerificationLookup) -> ArtifactVerificationRecord? {
        nil
    }

    func forget(rootPath: String) {
        lock.lock()
        forgotten.append(rootPath)
        lock.unlock()
    }
}
