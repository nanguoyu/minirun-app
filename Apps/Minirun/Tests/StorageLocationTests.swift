import Foundation
import iOSBench
import MinirunKit
import Observation
import StorageCore
import XCTest

@testable import MinirunApp

/// Registering a folder, and the states around it.
///
/// These exist because of a real failure: the reviewed build had **no storage
/// location bookmark in its container at all**, so the scanner had never run,
/// every model read as not installed, and no screen said why. Three things had
/// to become true, and each has a test here — the app knows when it has nowhere
/// to look, an add says out loud how it ended, and a grant that was minted once
/// is still there after a relaunch.
@MainActor
final class StorageLocationTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-location-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: - The empty state

    /// The one the reviewed build got wrong. With nothing registered, "not
    /// installed" is a statement about the app, and `hasLocation` is what lets
    /// a screen say so instead.
    func testWithNothingRegisteredTheAppKnowsItHasNowhereToLook() async throws {
        let installed = makeInstalled()
        XCTAssertFalse(installed.hasLocation)
        XCTAssertNil(installed.lastAddOutcome)

        let locationChanged = expectation(
            description: "the accepted folder invalidates the storage UI")
        withObservationTracking {
            _ = installed.hasLocation
        } onChange: {
            locationChanged.fulfill()
        }
        installed.addLocation(root)
        await fulfillment(of: [locationChanged], timeout: 1)
        XCTAssertTrue(installed.hasLocation)
    }

    func testAddingAFolderScansItAtOnceAndSaysWhatItFound() async throws {
        let installed = makeInstalled()
        try makeK3Artifact()

        installed.addLocation(root)
        XCTAssertEqual(installed.lastAddOutcome, .added(path: root.path))
        try await waitForScan(installed)

        // The scan the add started is the one that made the model installed.
        // Nothing here pressed a refresh button.
        XCTAssertTrue(installed.isInstalled(.kimiK3))
        let location = try XCTUnwrap(
            installed.report.locations.first { $0.rootPath == root.path })
        XCTAssertEqual(location.artifacts.count, 1)
        XCTAssertGreaterThan(try XCTUnwrap(location.artifacts.first).bytesOnDisk, 0)
    }

    /// A dismissed picker used to be indistinguishable from a broken button.
    func testACancelledPickIsRecordedRatherThanIgnored() {
        let installed = makeInstalled()
        installed.noteAddCancelled()
        XCTAssertEqual(installed.lastAddOutcome, .cancelled)
        XCTAssertTrue(installed.lastAddOutcome?.isFailure ?? false)
        XCTAssertFalse(installed.hasLocation)
    }

    func testAFailedGrantIsNamed() {
        let installed = makeInstalled()
        installed.noteAddFailed("the sandbox refused access to '/Volumes/Nope'")
        XCTAssertEqual(
            installed.lastAddOutcome, .failed("the sandbox refused access to '/Volumes/Nope'"))
        XCTAssertEqual(installed.locationError, "the sandbox refused access to '/Volumes/Nope'")
    }

    func testReviewStoragePolicyCannotInvokeTheFolderPickerAction() {
        var calls = 0

        let performed = StorageLocationActionPolicy.perform(isReviewPreview: true) {
            calls += 1
        }

        XCTAssertFalse(performed)
        XCTAssertEqual(calls, 0)
        XCTAssertEqual(
            StorageLocationActionPolicy.reviewDisabledMessage,
            "Sample review — storage access disabled")
    }

    func testDownloadDestinationTransfersOnlyLiveScopeWithoutHiddenBookmark() throws {
        let storage = DestinationAuthorizationStorage()

        let destination = try DownloadDestinationAuthorization.authorize(
            root, storage: storage)

        XCTAssertEqual(storage.calls, ["scope"])
        XCTAssertTrue(storage.rememberedKeys.isEmpty)
        XCTAssertTrue(storage.forgottenKeys.isEmpty)
        XCTAssertEqual(destination.directory, root)
        XCTAssertFalse(destination.scope.isReleased)
        XCTAssertEqual(storage.scopeCoding.starts, 1)
        XCTAssertEqual(storage.scopeCoding.stops, 0)
        destination.scope.release()
        destination.scope.release()
        XCTAssertEqual(storage.scopeCoding.stops, 1)
    }

    func testDownloadDestinationSelectionReleasesReplacementButTransfersFinalScope() throws {
        let storage = DestinationAuthorizationStorage()
        let first = try DownloadDestinationAuthorization.authorize(
            root.appendingPathComponent("first", isDirectory: true), storage: storage)
        let second = try DownloadDestinationAuthorization.authorize(
            root.appendingPathComponent("second", isDirectory: true), storage: storage)
        let selection = DownloadDestinationSelection()

        selection.replace(with: first)
        selection.replace(with: second)

        XCTAssertEqual(storage.scopeCoding.starts, 2)
        XCTAssertEqual(storage.scopeCoding.stops, 1)
        let transferred = try XCTUnwrap(selection.take())
        selection.cancel()
        XCTAssertEqual(storage.scopeCoding.stops, 1)
        XCTAssertEqual(transferred.directory.lastPathComponent, "second")

        transferred.scope.release()
        XCTAssertEqual(storage.scopeCoding.stops, 2)
    }

    func testDownloadDestinationSelectionClosesGrantWhenSheetLeaves() throws {
        let storage = DestinationAuthorizationStorage()
        let destination = try DownloadDestinationAuthorization.authorize(
            root, storage: storage)
        let selection = DownloadDestinationSelection()

        selection.replace(with: destination)
        selection.cancel()
        selection.cancel()

        XCTAssertNil(selection.destination)
        XCTAssertEqual(storage.scopeCoding.starts, 1)
        XCTAssertEqual(storage.scopeCoding.stops, 1)
    }

    func testDownloadDestinationReleasesScopeExactlyOnceWhenPlanningFailsBeforeStart() async
        throws
    {
        let storage = DestinationAuthorizationStorage()
        let destination = try DownloadDestinationAuthorization.authorize(root, storage: storage)
        let controller = DownloadController(
            entry: CatalogFixtures.kimiK3, manager: MockDownloadManager(entries: []))

        await controller.start(
            destination: destination.directory, scope: destination.scope,
            options: DownloadOptions())

        XCTAssertTrue(destination.scope.isReleased)
        XCTAssertEqual(storage.scopeCoding.starts, 1)
        XCTAssertEqual(storage.scopeCoding.stops, 1)
        XCTAssertTrue(storage.rememberedKeys.isEmpty)
        XCTAssertTrue(storage.forgottenKeys.isEmpty)
        XCTAssertNil(controller.plan)
    }

    func testDownloadCapacityPassesUnsignedExtremesToCheckedStorageDecision() {
        let storage = DestinationCapacityStorage()
        let volume = VolumeDescriptor(
            mountPath: root.path, name: "Test", filesystemType: "APFS",
            space: FreeSpace(
                optimisticBytes: Int64.max, dependableBytes: Int64.max,
                totalBytes: Int64.max, volumeName: "Test", isReadOnly: false,
                isRemovable: true),
            isInternal: false, isEjectable: true, writeRefusal: nil)

        let verdict = DownloadDestinationCapacity.verdict(
            for: volume, requiredBytes: .max, headroomBytes: .max, storage: storage)

        XCTAssertFalse(verdict.isFit)
        XCTAssertEqual(storage.capacityRequests.count, 1)
        XCTAssertEqual(storage.capacityRequests.first?.bytes, UInt64.max)
        XCTAssertEqual(storage.capacityRequests.first?.headroom, UInt64.max)
    }

    func testPersistentDownloadServicesUseApplicationSupportJobDirectoryAndRealManager() throws {
        let storage = InMemoryLocationStorage()
        let services = try AppDownloadServices.persistent(
            storage: storage, applicationSupportRoot: root)

        XCTAssertTrue(services.manager is DownloadManager)
        let store = try XCTUnwrap(services.stateStore as? FileDownloadStateStore)
        XCTAssertEqual(
            store.directory,
            root.appendingPathComponent("DownloadJobs", isDirectory: true))
        XCTAssertTrue(services.restoresPersistedJobs)
        XCTAssertNil(services.setupError)
    }

    func testOverflowedProgressIsNeitherBalancedNorShownAsComplete() {
        let progress = ProgressSnapshot(
            totalBytes: UInt64.max, verifiedBytes: UInt64.max,
            fetchedUnverifiedBytes: UInt64.max, inFlightBytes: UInt64.max,
            filesTotal: 3, filesDone: 0, currentFilePath: nil,
            currentFileBytes: 0, currentFileOffset: 0, bytesPerSecond: 0,
            estimatedTimeRemaining: nil, networkBytes: 0, wastedBytes: 0)

        XCTAssertFalse(progress.accountsForEveryByte)
        XCTAssertEqual(progress.fraction, 0)
    }

    func testFormalDownloadDetailsNeverSynthesizeAPreviewPlan() {
        let directory = root.appendingPathComponent("Conversations", isDirectory: true)
        let model = AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot.labelled(.live),
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: CatalogFixtures.all),
            seedRecordedRuns: false,
            startDiscovery: false)

        XCTAssertNil(model.download(.kimiK3)?.plan)
        XCTAssertNil(model.downloadPlanForPresentation(.kimiK3))
    }

    func testConversationStorageFailureUsesEphemeralStoreAndNamesDataLoss() {
        struct TestFailure: Error {}
        let result = ConversationPersistence.open { throw TestFailure() }

        XCTAssertFalse(result.store.isPersistent)
        XCTAssertNotNil(result.error)
        XCTAssertTrue(result.error?.contains("will not be saved") ?? false)
        XCTAssertTrue(result.store.directory.path.contains("MinirunVisualReview-"))
        XCTAssertFalse(result.store.directory.path.contains("MinirunConversations"))
        try? result.store.save(
            Conversation(
                settings: ConversationSettings(
                    model: .kimiK3,
                    memoryBudgetBytes: CatalogFixtures.k3OnRecordMinimumBudget)))
        XCTAssertFalse(FileManager.default.fileExists(atPath: result.store.directory.path))
    }

    func testLiveFactoryMakesConversationPersistenceFailureVisibleWithoutTempFallback() {
        struct TestFailure: Error {}
        let candidate = root.appendingPathComponent("PretendApplicationSupport", isDirectory: true)
        let defaults = UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!
        let model = AppModel.live(
            userDefaults: defaults,
            conversationFactory: { throw TestFailure() },
            downloadServicesFactory: { _ in .unavailable("test download service") },
            storageFactory: { InMemoryLocationStorage() },
            startDiscovery: false)

        XCTAssertFalse(model.conversationStore.isPersistent)
        XCTAssertTrue(model.conversationStorageError?.contains("will not be saved") ?? false)
        XCTAssertFalse(model.conversationStore.directory.path.hasPrefix(candidate.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: candidate.path))
    }

    func testFormalLaunchRevokesTheObsoleteGlobalDownloadGrant() throws {
        let storage = InMemoryLocationStorage()
        let legacy = AppStorageMigrations.legacyGlobalDownloadDestination
        let registered = StorageKey(InstalledModels.locationKeyPrefix + "kept")
        try storage.remember(root.appendingPathComponent("Legacy"), as: legacy)
        try storage.remember(root.appendingPathComponent("Artifact"), as: registered)
        let empty = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .bundled, models: [])

        _ = AppModel.live(
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            catalogService: ModelCatalog(
                live: SavedTransferCatalogSource(snapshot: empty),
                bundled: empty, cache: nil),
            conversationFactory: { .ephemeral() },
            downloadServicesFactory: { _ in .unavailable("test download service") },
            storageFactory: { storage },
            startDiscovery: false)

        XCTAssertNil(storage.bookmark(legacy))
        XCTAssertNotNil(storage.bookmark(registered))
        XCTAssertEqual(storage.knownLocations(), [registered])
    }

    func testStartReleasesScopeWhenAConformerRejectsBeforeTakingOwnership() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let manager = AppDownloadManagerSpy(plan: plan)
        manager.startError = DownloadError.transport(status: 503, path: "index.json", attempt: 1)
        let controller = DownloadController(entry: CatalogFixtures.kimiK3, manager: manager)
        let storage = DestinationAuthorizationStorage()
        let destination = try DownloadDestinationAuthorization.authorize(root, storage: storage)

        await controller.start(
            destination: destination.directory, scope: destination.scope,
            options: DownloadOptions())

        XCTAssertTrue(destination.scope.isReleased)
        XCTAssertEqual(storage.scopeCoding.starts, 1)
        XCTAssertEqual(storage.scopeCoding.stops, 1)
        XCTAssertTrue(storage.rememberedKeys.isEmpty)
        XCTAssertTrue(storage.forgottenKeys.isEmpty)
        XCTAssertNotNil(controller.operationError)
    }

    func testStartFailureAfterManagerReleasesOwnedScopeStillStopsExactlyOnce() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let manager = AppDownloadManagerSpy(plan: plan)
        manager.startError = .destinationNotWritable(path: root.path, reason: "preflight refused")
        manager.releaseScopeBeforeStartError = true
        let controller = DownloadController(entry: CatalogFixtures.kimiK3, manager: manager)
        let storage = DestinationAuthorizationStorage()
        let destination = try DownloadDestinationAuthorization.authorize(root, storage: storage)

        await controller.start(
            destination: destination.directory, scope: destination.scope,
            options: DownloadOptions())

        XCTAssertTrue(destination.scope.isReleased)
        XCTAssertEqual(storage.scopeCoding.starts, 1)
        XCTAssertEqual(storage.scopeCoding.stops, 1)
        XCTAssertTrue(storage.rememberedKeys.isEmpty)
        XCTAssertTrue(storage.forgottenKeys.isEmpty)
        XCTAssertNotNil(controller.operationError)
    }

    func testRestoreCoalescesConcurrentCallersAndNeverAutoResumes() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let state = makeDownloadJobState(plan: plan, destination: root, bookmark: Data("one".utf8))
        let store = InMemoryDownloadStateStore()
        try store.save(state)
        let gate = RestoreGate()
        let manager = AppDownloadManagerSpy(
            plan: plan,
            restoreReport: DownloadRestoreReport(restoredJobs: [state.job]),
            summaries: [makeDownloadSummary(state: state)],
            restoreGate: gate)
        let model = makeDownloadAppModel(
            manager: manager, stateStore: store, storage: InMemoryLocationStorage())

        let first = Task { await model.restoreDownloads() }
        await gate.waitUntilEntered()
        let second = Task { await model.restoreDownloads() }
        await Task.yield()

        XCTAssertTrue(model.isRecoveringDownloads)
        XCTAssertNotNil(model.downloadStartRefusal)
        XCTAssertEqual(manager.restoreCount, 1)
        XCTAssertEqual(manager.resumeCount, 0)

        await gate.release()
        await first.value
        await second.value

        XCTAssertFalse(model.isRecoveringDownloads)
        XCTAssertEqual(manager.restoreCount, 1)
        XCTAssertEqual(manager.resumeCount, 0)
    }

    func testRestoredTransferRemainsGloballyVisibleWhileCatalogMetadataIsUnavailable()
        async throws
    {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let state = makeDownloadJobState(
            plan: plan, destination: root, bookmark: Data("pending-catalog".utf8))
        let store = InMemoryDownloadStateStore()
        try store.save(state)
        let manager = AppDownloadManagerSpy(
            plan: plan,
            restoreReport: DownloadRestoreReport(restoredJobs: [state.job]),
            summaries: [makeDownloadSummary(state: state)])
        let empty = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .bundled, models: [])
        let model = AppModel(
            catalogSnapshot: empty,
            catalogService: ModelCatalog(bundled: empty, cache: nil),
            allowsLiveCatalogRefresh: false,
            store: ConversationStore(
                directory: root.appendingPathComponent(
                    "PendingCatalogConversations", isDirectory: true)),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            storage: InMemoryLocationStorage(),
            downloadServices: AppDownloadServices(
                manager: manager, stateStore: store,
                restoresPersistedJobs: true, setupError: nil),
            seedRecordedRuns: false, startDiscovery: false)

        await model.restoreDownloads()

        XCTAssertEqual(model.savedTransfersForPresentation.count, 1)
        let transfer = try XCTUnwrap(model.savedTransfersForPresentation.first)
        XCTAssertEqual(transfer.job, state.job)
        XCTAssertEqual(transfer.model, state.plan.model)
        XCTAssertEqual(transfer.destinationPath, state.destinationPath)
        XCTAssertEqual(transfer.state, "paused")
        XCTAssertEqual(transfer.reason, "Waiting for catalog metadata.")
        let globalTransfer = try XCTUnwrap(model.transfersForPresentation.first)
        XCTAssertEqual(globalTransfer.job, state.job)
        XCTAssertEqual(globalTransfer.state.compactTransferStatus, "Paused · 0%")
        XCTAssertFalse(globalTransfer.hasCatalogEntry)
        XCTAssertTrue(model.downloadJobs(for: state.plan.model).isEmpty)
        XCTAssertFalse(model.canRetryDownloadRecovery)
        XCTAssertEqual(manager.resumeCount, 0)
    }

    func testCataloglessTerminalTransfersKeepTheirPersistedState() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        for terminal: DownloadJobSummary.JobState in [.failed, .cancelled] {
            let state = makeDownloadJobState(
                plan: plan, destination: root,
                bookmark: Data("terminal-\(terminal.rawValue)".utf8),
                runState: terminal)
            let store = InMemoryDownloadStateStore()
            try store.save(state)
            let manager = AppDownloadManagerSpy(
                plan: plan,
                restoreReport: DownloadRestoreReport(restoredJobs: [state.job]),
                summaries: [makeDownloadSummary(state: state, reportedState: terminal)])
            let empty = ModelCatalogSnapshot(
                generatedAt: Date(), origin: .bundled, models: [])
            let model = AppModel(
                catalogSnapshot: empty,
                catalogService: ModelCatalog(bundled: empty, cache: nil),
                allowsLiveCatalogRefresh: false,
                store: ConversationStore(
                    directory: root.appendingPathComponent(
                        "Terminal-\(terminal.rawValue)-\(UUID().uuidString)",
                        isDirectory: true)),
                userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
                storage: InMemoryLocationStorage(),
                downloadServices: AppDownloadServices(
                    manager: manager, stateStore: store,
                    restoresPersistedJobs: true, setupError: nil),
                seedRecordedRuns: false, startDiscovery: false)

            await model.restoreDownloads()

            XCTAssertEqual(model.savedTransfersForPresentation.first?.state, terminal.rawValue)
            XCTAssertTrue(model.downloadRecoveryIssues.isEmpty)
            XCTAssertEqual(manager.resumeCount, 0)
        }
    }

    func testTerminalTransfersStayTerminalWhenCatalogArrivesAndEventsReplay() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let empty = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .bundled, models: [])
        let live = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .live,
            models: [try XCTUnwrap(CatalogFixtures.snapshot.descriptor(.kimiK3))])

        for terminal: DownloadJobSummary.JobState in [.failed, .cancelled] {
            let state = makeDownloadJobState(
                plan: plan,
                destination: root.appendingPathComponent("attached-\(terminal.rawValue)"),
                bookmark: Data("attached-\(terminal.rawValue)".utf8),
                runState: terminal)
            let store = InMemoryDownloadStateStore()
            try store.save(state)
            let progress = makeDownloadProgress(job: state.job, plan: plan)
            let event: DownloadEvent =
                terminal == .failed
                ? .failed(
                    .interrupted(
                        path: state.destinationPath, atByte: 0,
                        underlying: "restored persisted job was already failed"),
                    progress)
                : .cancelled(progress)
            let manager = AppDownloadManagerSpy(
                plan: plan,
                restoreReport: DownloadRestoreReport(restoredJobs: [state.job]),
                summaries: [makeDownloadSummary(state: state, reportedState: terminal)],
                events: [state.job: [event]])
            let model = AppModel(
                catalogSnapshot: empty,
                catalogService: ModelCatalog(
                    live: SavedTransferCatalogSource(snapshot: live),
                    bundled: empty, cache: nil),
                store: ConversationStore(
                    directory: root.appendingPathComponent(
                        "TerminalAttach-\(terminal.rawValue)-\(UUID().uuidString)",
                        isDirectory: true)),
                userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
                storage: InMemoryLocationStorage(),
                downloadServices: AppDownloadServices(
                    manager: manager, stateStore: store,
                    restoresPersistedJobs: true, setupError: nil),
                seedRecordedRuns: false, startDiscovery: false)

            await model.restoreDownloads()
            await model.refreshCatalog()
            try await Task.sleep(for: .milliseconds(20))

            let controller = try XCTUnwrap(model.downloadJobs(for: .kimiK3).first)
            if terminal == .failed {
                guard case .failed = controller.state else {
                    XCTFail("failed durable state became \(controller.state.phrase)")
                    continue
                }
            } else {
                guard case .cancelled = controller.state else {
                    XCTFail("cancelled durable state became \(controller.state.phrase)")
                    continue
                }
            }
            XCTAssertTrue(model.downloadRecoveryIssues.isEmpty)
            XCTAssertEqual(manager.resumeCount, 0)
        }
    }

    func testContradictoryTerminalReplayCannotOverwriteDurableTerminalTruth() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        for terminal: DownloadJobSummary.JobState in [.failed, .cancelled] {
            let state = makeDownloadJobState(
                plan: plan,
                destination: root.appendingPathComponent(
                    "terminal-replay-\(terminal.rawValue)"),
                bookmark: Data(terminal.rawValue.utf8), runState: terminal)
            let progress = makeDownloadProgress(job: state.job, plan: plan)
            let injectedEvent: DownloadEvent =
                terminal == .failed ? .cancelled(progress) : .failed(.cancelled, progress)
            let store = InMemoryDownloadStateStore()
            try store.save(state)
            let manager = AppDownloadManagerSpy(
                plan: plan,
                restoreReport: DownloadRestoreReport(restoredJobs: [state.job]),
                summaries: [makeDownloadSummary(state: state, reportedState: terminal)],
                events: [state.job: [injectedEvent]])
            let model = makeDownloadAppModel(
                manager: manager, stateStore: store, storage: InMemoryLocationStorage())

            await model.restoreDownloads()
            try await Task.sleep(for: .milliseconds(20))

            let controller = try XCTUnwrap(model.downloadJobs(for: .kimiK3).first)
            if terminal == .failed {
                guard case .failed = controller.state else {
                    XCTFail("failed durable state accepted a contradictory replay")
                    continue
                }
            } else {
                guard case .cancelled = controller.state else {
                    XCTFail("cancelled durable state accepted a contradictory replay")
                    continue
                }
            }
            XCTAssertEqual(controller.updatedAt, state.updatedAt)
            XCTAssertEqual(manager.resumeCount, 0)
        }
    }

    func testRestoreRejectsMalformedProgressBeforeConstructingAController() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let invalidProgresses: [(String, (DownloadJobID) -> DownloadProgress)] = [
            (
                "negative files",
                { job in
                    DownloadProgress(
                        job: job, planTotalBytes: plan.totalBytes,
                        verifiedBytes: 0, fetchedUnverifiedBytes: 0, inFlightBytes: 0,
                        remainingBytes: plan.totalBytes,
                        filesTotal: plan.files.count, filesVerified: -1, filesFailed: 0,
                        networkBytes: 0, wastedBytes: 0,
                        instantaneousBytesPerSecond: 0, smoothedBytesPerSecond: 0,
                        estimatedTimeRemaining: nil, startedAt: Date(), elapsed: 0)
                }
            ),
            (
                "unbalanced bytes",
                { job in
                    DownloadProgress(
                        job: job, planTotalBytes: plan.totalBytes,
                        verifiedBytes: 1, fetchedUnverifiedBytes: 0, inFlightBytes: 0,
                        remainingBytes: plan.totalBytes,
                        filesTotal: plan.files.count, filesVerified: 0, filesFailed: 0,
                        networkBytes: 0, wastedBytes: 0,
                        instantaneousBytesPerSecond: 0, smoothedBytesPerSecond: 0,
                        estimatedTimeRemaining: nil, startedAt: Date(), elapsed: 0)
                }
            ),
        ]

        for (label, makeProgress) in invalidProgresses {
            let state = makeDownloadJobState(
                plan: plan,
                destination: root.appendingPathComponent("invalid-progress-\(label)"),
                bookmark: Data(label.utf8))
            let store = InMemoryDownloadStateStore()
            try store.save(state)
            let manager = AppDownloadManagerSpy(
                plan: plan,
                restoreReport: DownloadRestoreReport(restoredJobs: [state.job]),
                summaries: [
                    makeDownloadSummary(
                        state: state, progress: makeProgress(state.job))
                ])
            let model = makeDownloadAppModel(
                manager: manager, stateStore: store, storage: InMemoryLocationStorage())

            await model.restoreDownloads()

            XCTAssertTrue(model.downloadJobs(for: .kimiK3).isEmpty, label)
            XCTAssertTrue(model.canRetryDownloadRecovery, label)
            XCTAssertEqual(model.downloadRecoveryIssues.map(\.job), [state.job], label)
        }
    }

    func testCatalogRefreshAtomicallyAttachesPendingTransferWithoutDuplicationOrResume()
        async throws
    {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let state = makeDownloadJobState(
            plan: plan, destination: root, bookmark: Data("attach-after-catalog".utf8))
        let store = InMemoryDownloadStateStore()
        try store.save(state)
        let manager = AppDownloadManagerSpy(
            plan: plan,
            restoreReport: DownloadRestoreReport(restoredJobs: [state.job]),
            summaries: [makeDownloadSummary(state: state)])
        let empty = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .bundled, models: [])
        let live = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .live,
            models: [try XCTUnwrap(CatalogFixtures.snapshot.descriptor(.kimiK3))])
        let model = AppModel(
            catalogSnapshot: empty,
            catalogService: ModelCatalog(
                live: SavedTransferCatalogSource(snapshot: live), bundled: empty, cache: nil),
            store: ConversationStore(
                directory: root.appendingPathComponent(
                    "AttachCatalogConversations", isDirectory: true)),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            storage: InMemoryLocationStorage(),
            downloadServices: AppDownloadServices(
                manager: manager, stateStore: store,
                restoresPersistedJobs: true, setupError: nil),
            seedRecordedRuns: false, startDiscovery: false)

        await model.restoreDownloads()
        XCTAssertEqual(model.savedTransfersForPresentation.count, 1)

        await model.refreshCatalog()
        await model.refreshCatalog()

        XCTAssertEqual(model.downloadJobs(for: .kimiK3).compactMap(\.jobID), [state.job])
        XCTAssertTrue(model.savedTransfersForPresentation.isEmpty)
        XCTAssertEqual(model.transfersForPresentation.map(\.job), [state.job])
        XCTAssertTrue(try XCTUnwrap(model.transfersForPresentation.first).hasCatalogEntry)
        XCTAssertEqual(manager.restoreCount, 1)
        XCTAssertEqual(manager.resumeCount, 0)
    }

    func testAttachedTransferRemainsVisibleWhenALaterCatalogRemovesItsModel() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let state = makeDownloadJobState(
            plan: plan, destination: root, bookmark: Data("catalog-removal".utf8))
        let store = InMemoryDownloadStateStore()
        try store.save(state)
        let manager = AppDownloadManagerSpy(
            plan: plan,
            restoreReport: DownloadRestoreReport(restoredJobs: [state.job]),
            summaries: [makeDownloadSummary(state: state)])
        let present = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .live,
            models: [try XCTUnwrap(CatalogFixtures.snapshot.descriptor(.kimiK3))])
        let removed = ModelCatalogSnapshot(generatedAt: Date(), origin: .live, models: [])
        let source = SequencedSavedTransferCatalogSource(snapshots: [removed])
        let model = AppModel(
            catalogSnapshot: present,
            catalogService: ModelCatalog(live: source, bundled: removed, cache: nil),
            store: ConversationStore(
                directory: root.appendingPathComponent(
                    "RemovedCatalogConversations", isDirectory: true)),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            storage: InMemoryLocationStorage(),
            downloadServices: AppDownloadServices(
                manager: manager, stateStore: store,
                restoresPersistedJobs: true, setupError: nil),
            seedRecordedRuns: false, startDiscovery: false)

        await model.restoreDownloads()
        XCTAssertTrue(model.savedTransfersForPresentation.isEmpty)

        await model.refreshCatalog()

        XCTAssertEqual(model.savedTransfersForPresentation.map(\.job), [state.job])
        XCTAssertEqual(model.downloadJobs(for: .kimiK3).compactMap(\.jobID), [state.job])
        XCTAssertEqual(manager.restoreCount, 1)
        XCTAssertEqual(manager.resumeCount, 0)
    }

    func testFailedRestoreCanBeRetriedAndOnlySuccessCompletesRecovery() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let manager = AppDownloadManagerSpy(plan: plan, restoreFailuresRemaining: 1)
        let model = makeDownloadAppModel(
            manager: manager, stateStore: InMemoryDownloadStateStore(),
            storage: InMemoryLocationStorage())

        await model.restoreDownloads()
        XCTAssertTrue(model.canRetryDownloadRecovery)
        XCTAssertNotNil(model.downloadServiceError)
        XCTAssertEqual(manager.restoreCount, 1)

        await model.restoreDownloads()
        XCTAssertFalse(model.canRetryDownloadRecovery)
        XCTAssertNil(model.downloadServiceError)
        XCTAssertEqual(manager.restoreCount, 2)
    }

    func testOneBrokenSavedJobDoesNotDisableNewTransfersForOtherModels() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let state = makeDownloadJobState(
            plan: plan, destination: root, bookmark: Data("broken-k3".utf8))
        let store = InMemoryDownloadStateStore()
        try store.save(state)
        let manager = AppDownloadManagerSpy(
            plan: plan,
            restoreReport: DownloadRestoreReport(
                failures: [
                    DownloadRestoreFailure(
                        job: state.job, destinationPath: state.destinationPath,
                        reason: "the saved bookmark no longer resolves")
                ]))
        let model = makeDownloadAppModel(
            manager: manager, stateStore: store, storage: InMemoryLocationStorage())

        await model.restoreDownloads()

        XCTAssertNotNil(model.downloadServiceError)
        XCTAssertNil(model.downloadSetupError)
        XCTAssertNil(model.downloadStartRefusal(for: .deepseekV4Flash))
        XCTAssertNotNil(model.downloadStartRefusal(for: .kimiK3))
        XCTAssertTrue(model.canRetryDownloadRecovery)
    }

    func testPendingSuccessDoesNotHideTheRetryForAnotherFailedRestore() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let restored = makeDownloadJobState(
            plan: plan, destination: root.appendingPathComponent("restored"),
            bookmark: Data("restored".utf8))
        let failed = makeDownloadJobState(
            plan: plan, destination: root.appendingPathComponent("failed"),
            bookmark: Data("failed".utf8))
        let store = InMemoryDownloadStateStore()
        try store.save(restored)
        try store.save(failed)
        let manager = AppDownloadManagerSpy(
            plan: plan,
            restoreReport: DownloadRestoreReport(
                restoredJobs: [restored.job],
                failures: [
                    DownloadRestoreFailure(
                        job: failed.job, destinationPath: failed.destinationPath,
                        reason: "bookmark unavailable")
                ]),
            summaries: [makeDownloadSummary(state: restored)])
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let model = AppModel(
            catalogSnapshot: empty,
            catalogService: ModelCatalog(bundled: empty, cache: nil),
            allowsLiveCatalogRefresh: false,
            store: ConversationStore(
                directory: root.appendingPathComponent("PartialRestore", isDirectory: true)),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            storage: InMemoryLocationStorage(),
            downloadServices: AppDownloadServices(
                manager: manager, stateStore: store,
                restoresPersistedJobs: true, setupError: nil),
            seedRecordedRuns: false, startDiscovery: false)

        await model.restoreDownloads()

        XCTAssertEqual(model.savedTransfersForPresentation.map(\.job), [restored.job])
        XCTAssertEqual(model.downloadRecoveryIssues.map(\.job), [failed.job])
        XCTAssertTrue(model.canRetryDownloadRecovery)
        XCTAssertEqual(manager.resumeCount, 0)
    }

    func testRestoreRejectsDuplicateDurableJobIDsWithoutCallingManager() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let first = makeDownloadJobState(
            plan: plan, destination: root.appendingPathComponent("A"),
            bookmark: Data("A".utf8))
        let second = DownloadJobState(
            job: first.job, plan: first.plan,
            destinationPath: root.appendingPathComponent("B").path,
            destinationBookmark: Data("B".utf8),
            artifactRootPath: root.appendingPathComponent("B").path,
            artifactRootDevice: 1, artifactRootInode: 2,
            options: DownloadOptions(), fileStates: [:], runState: .paused,
            createdAt: first.createdAt, updatedAt: first.updatedAt)
        let store = DuplicateDownloadStateStore(states: [first, second])
        let manager = AppDownloadManagerSpy(plan: plan)
        let model = makeDownloadAppModel(
            manager: manager, stateStore: store, storage: InMemoryLocationStorage())

        await model.restoreDownloads()

        XCTAssertEqual(manager.restoreCount, 0)
        XCTAssertEqual(model.downloadRecoveryIssues.count, 1)
        XCTAssertTrue(model.canRetryDownloadRecovery)
        XCTAssertNotNil(model.downloadServiceError)
    }

    func testRestoreRejectsManagerReportThatSilentlyOmitsASavedJob() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let state = makeDownloadJobState(
            plan: plan, destination: root, bookmark: Data("omitted".utf8))
        let store = InMemoryDownloadStateStore()
        try store.save(state)
        let manager = AppDownloadManagerSpy(plan: plan)
        let model = makeDownloadAppModel(
            manager: manager, stateStore: store, storage: InMemoryLocationStorage())

        await model.restoreDownloads()

        XCTAssertTrue(model.downloadJobs(for: .kimiK3).isEmpty)
        XCTAssertTrue(model.canRetryDownloadRecovery)
        XCTAssertEqual(model.downloadRecoveryIssues.map(\.job), [state.job])
        XCTAssertTrue(
            model.downloadRecoveryIssues.first?.reason.contains("omitted this saved job") ?? false)
        XCTAssertTrue(model.downloadServiceError?.contains("every durable job") ?? false)
    }

    func testRestoreRejectsAJobReportedAsBothRestoredAndFailed() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let state = makeDownloadJobState(
            plan: plan, destination: root, bookmark: Data("contradictory".utf8))
        let store = InMemoryDownloadStateStore()
        try store.save(state)
        let manager = AppDownloadManagerSpy(
            plan: plan,
            restoreReport: DownloadRestoreReport(
                restoredJobs: [state.job],
                failures: [
                    DownloadRestoreFailure(
                        job: state.job, destinationPath: state.destinationPath,
                        reason: "injected contradictory report")
                ]),
            summaries: [makeDownloadSummary(state: state)])
        let model = makeDownloadAppModel(
            manager: manager, stateStore: store, storage: InMemoryLocationStorage())

        await model.restoreDownloads()

        XCTAssertTrue(model.downloadJobs(for: .kimiK3).isEmpty)
        XCTAssertTrue(model.canRetryDownloadRecovery)
        XCTAssertEqual(model.downloadRecoveryIssues.map(\.job), [state.job])
        XCTAssertTrue(
            model.downloadRecoveryIssues.first?.reason.contains("both restored and failed")
                ?? false)
    }

    func testRestoreRejectsRestoredJobWithoutAUniqueSummary() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let state = makeDownloadJobState(
            plan: plan, destination: root, bookmark: Data("no-summary".utf8))
        let store = InMemoryDownloadStateStore()
        try store.save(state)
        let manager = AppDownloadManagerSpy(
            plan: plan,
            restoreReport: DownloadRestoreReport(restoredJobs: [state.job]))
        let model = makeDownloadAppModel(
            manager: manager, stateStore: store, storage: InMemoryLocationStorage())

        await model.restoreDownloads()

        XCTAssertTrue(model.downloadJobs(for: .kimiK3).isEmpty)
        XCTAssertTrue(model.canRetryDownloadRecovery)
        XCTAssertTrue(
            model.downloadRecoveryIssues.first?.reason.contains("without a summary") ?? false)
    }

    func testRestoreRejectsRestoredJobMissingFromPostRestoreState() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let state = makeDownloadJobState(
            plan: plan, destination: root, bookmark: Data("no-post-state".utf8))
        let store = ChangingDownloadStateStore(before: [state], after: [])
        let manager = AppDownloadManagerSpy(
            plan: plan,
            restoreReport: DownloadRestoreReport(restoredJobs: [state.job]),
            summaries: [makeDownloadSummary(state: state)])
        let model = makeDownloadAppModel(
            manager: manager, stateStore: store, storage: InMemoryLocationStorage())

        await model.restoreDownloads()

        XCTAssertTrue(model.downloadJobs(for: .kimiK3).isEmpty)
        XCTAssertTrue(model.canRetryDownloadRecovery)
        XCTAssertTrue(
            model.downloadRecoveryIssues.first?.reason.contains("post-restore durable state")
                ?? false)
    }

    func testRestoreRejectsAnExtraDurableJobCreatedBehindTheReport() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let original = makeDownloadJobState(
            plan: plan, destination: root.appendingPathComponent("original"),
            bookmark: Data("original".utf8))
        let injected = makeDownloadJobState(
            plan: plan, destination: root.appendingPathComponent("injected"),
            bookmark: Data("injected".utf8))
        let store = ChangingDownloadStateStore(before: [original], after: [original, injected])
        let manager = AppDownloadManagerSpy(
            plan: plan,
            restoreReport: DownloadRestoreReport(restoredJobs: [original.job]),
            summaries: [makeDownloadSummary(state: original)])
        let model = makeDownloadAppModel(
            manager: manager, stateStore: store, storage: InMemoryLocationStorage())

        await model.restoreDownloads()

        XCTAssertTrue(model.downloadJobs(for: .kimiK3).isEmpty)
        XCTAssertTrue(model.canRetryDownloadRecovery)
        XCTAssertEqual(model.downloadRecoveryIssues.map(\.job), [injected.job])
        XCTAssertTrue(
            model.downloadRecoveryIssues.first?.reason.contains("did not exist before") ?? false)
        XCTAssertEqual(manager.resumeCount, 0)
    }

    func testRestoreRejectsSummaryStateThatDisagreesWithDurableState() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let failed = makeDownloadJobState(
            plan: plan, destination: root.appendingPathComponent("failed-summary-mismatch"),
            bookmark: Data("failed-summary-mismatch".utf8), runState: .failed)
        let store = InMemoryDownloadStateStore()
        try store.save(failed)
        let manager = AppDownloadManagerSpy(
            plan: plan,
            restoreReport: DownloadRestoreReport(restoredJobs: [failed.job]),
            summaries: [makeDownloadSummary(state: failed, reportedState: .paused)])
        let model = makeDownloadAppModel(
            manager: manager, stateStore: store, storage: InMemoryLocationStorage())

        await model.restoreDownloads()

        XCTAssertTrue(model.downloadJobs(for: .kimiK3).isEmpty)
        XCTAssertTrue(model.canRetryDownloadRecovery)
        XCTAssertEqual(model.downloadRecoveryIssues.map(\.job), [failed.job])
        XCTAssertTrue(
            model.downloadRecoveryIssues.first?.reason.contains("durable state says failed")
                ?? false)
        XCTAssertEqual(manager.resumeCount, 0)
    }

    func testRestoreRejectsMutatedDestinationAndPhysicalRootAuthority() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let original = makeDownloadJobState(
            plan: plan, destination: root.appendingPathComponent("original-authority"),
            bookmark: Data("original-authority".utf8))
        let changed = DownloadJobState(
            job: original.job, plan: original.plan,
            destinationPath: root.appendingPathComponent("substituted-destination").path,
            destinationBookmark: original.destinationBookmark,
            artifactRootPath: root.appendingPathComponent("substituted-root").path,
            artifactRootDevice: 900, artifactRootInode: 901,
            options: original.options, fileStates: original.fileStates,
            runState: .paused, createdAt: original.createdAt, updatedAt: Date())
        let store = ChangingDownloadStateStore(before: [original], after: [changed])
        let manager = AppDownloadManagerSpy(
            plan: plan,
            restoreReport: DownloadRestoreReport(restoredJobs: [original.job]),
            summaries: [makeDownloadSummary(state: changed)])
        let model = makeDownloadAppModel(
            manager: manager, stateStore: store, storage: InMemoryLocationStorage())

        await model.restoreDownloads()

        XCTAssertTrue(model.downloadJobs(for: .kimiK3).isEmpty)
        XCTAssertTrue(model.canRetryDownloadRecovery)
        let reason = try XCTUnwrap(model.downloadRecoveryIssues.first?.reason)
        XCTAssertTrue(reason.contains("destination path"))
        XCTAssertTrue(reason.contains("physical artifact identity"))
        XCTAssertEqual(manager.resumeCount, 0)
    }

    func testRestoreFailureCannotMutateTheDurableJobItClaimsWasNotRestored() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let original = makeDownloadJobState(
            plan: plan, destination: root.appendingPathComponent("failed-original"),
            bookmark: Data("failed-original".utf8))
        let changed = DownloadJobState(
            job: original.job, plan: original.plan,
            destinationPath: original.destinationPath,
            destinationBookmark: Data("substituted-bookmark".utf8),
            artifactRootPath: original.artifactRootPath,
            artifactRootDevice: original.artifactRootDevice,
            artifactRootInode: original.artifactRootInode,
            options: original.options, fileStates: [:], runState: .failed,
            createdAt: original.createdAt, updatedAt: Date())
        let store = ChangingDownloadStateStore(before: [original], after: [changed])
        let manager = AppDownloadManagerSpy(
            plan: plan,
            restoreReport: DownloadRestoreReport(
                failures: [
                    DownloadRestoreFailure(
                        job: original.job, destinationPath: original.destinationPath,
                        reason: "injected restore failure")
                ]))
        let model = makeDownloadAppModel(
            manager: manager, stateStore: store, storage: InMemoryLocationStorage())

        await model.restoreDownloads()

        XCTAssertTrue(model.downloadJobs(for: .kimiK3).isEmpty)
        XCTAssertTrue(model.canRetryDownloadRecovery)
        let reason = try XCTUnwrap(model.downloadRecoveryIssues.first?.reason)
        XCTAssertTrue(reason.contains("changed durable state"), reason)
        XCTAssertEqual(manager.resumeCount, 0)
    }

    func testPostRestoreStoreReadFailureDoesNotReuseAStaleSnapshotAndCanRetry() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let state = makeDownloadJobState(
            plan: plan, destination: root, bookmark: Data("post-restore".utf8))
        let store = SequencedDownloadStateStore(states: [state], failingAllCalls: [2])
        let manager = AppDownloadManagerSpy(
            plan: plan,
            restoreReport: DownloadRestoreReport(restoredJobs: [state.job]),
            summaries: [makeDownloadSummary(state: state)])
        let model = makeDownloadAppModel(
            manager: manager, stateStore: store, storage: InMemoryLocationStorage())

        await model.restoreDownloads()

        XCTAssertTrue(model.downloadJobs(for: .kimiK3).isEmpty)
        XCTAssertTrue(model.canRetryDownloadRecovery)
        XCTAssertTrue(model.downloadServiceError?.contains("read back") ?? false)
        XCTAssertEqual(manager.restoreCount, 1)

        await model.restoreDownloads()

        XCTAssertEqual(model.downloadJobs(for: .kimiK3).compactMap(\.jobID), [state.job])
        XCTAssertFalse(model.canRetryDownloadRecovery)
        XCTAssertNil(model.downloadServiceError)
        XCTAssertEqual(manager.restoreCount, 2)
        XCTAssertEqual(manager.resumeCount, 0)
    }

    func testMultipleRestoredJobsStayPausedUntilExplicitResumeUsesTheirOwnBookmarks()
        async throws
    {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let firstDestination = root.appendingPathComponent("Drive-A", isDirectory: true)
        let secondDestination = root.appendingPathComponent("Drive-B", isDirectory: true)
        try FileManager.default.createDirectory(
            at: firstDestination, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: secondDestination, withIntermediateDirectories: true)
        let firstBookmark = Data("bookmark-A".utf8)
        let secondBookmark = Data("bookmark-B".utf8)
        let first = makeDownloadJobState(
            plan: plan, destination: firstDestination, bookmark: firstBookmark,
            updatedAt: Date(timeIntervalSince1970: 1))
        let second = makeDownloadJobState(
            plan: plan, destination: secondDestination, bookmark: secondBookmark,
            updatedAt: Date(timeIntervalSince1970: 2))
        let store = InMemoryDownloadStateStore()
        try store.save(first)
        try store.save(second)
        let firstFinished = makeFinishedEvent(state: first)
        let secondFinished = makeFinishedEvent(state: second)
        let manager = AppDownloadManagerSpy(
            plan: plan,
            restoreReport: DownloadRestoreReport(restoredJobs: [first.job, second.job]),
            summaries: [makeDownloadSummary(state: first), makeDownloadSummary(state: second)],
            events: [first.job: [firstFinished], second.job: [secondFinished]])
        let storage = JobBookmarkStorage(
            bookmarks: [firstBookmark: firstDestination, secondBookmark: secondDestination])
        let model = makeDownloadAppModel(manager: manager, stateStore: store, storage: storage)

        await model.restoreDownloads()

        XCTAssertEqual(
            Set(model.downloadJobs(for: .kimiK3).compactMap(\.jobID)), [first.job, second.job])
        XCTAssertTrue(
            model.downloadJobs(for: .kimiK3).allSatisfy {
                if case .paused = $0.state { return true }
                return false
            })
        XCTAssertTrue(storage.resolvedBookmarkData.isEmpty)
        XCTAssertTrue(storage.registeredPaths.isEmpty)
        XCTAssertEqual(manager.verifyCount, 0)
        XCTAssertEqual(manager.resumeCount, 0)

        for controller in model.downloadJobs(for: .kimiK3) {
            await controller.resume()
        }
        try await waitUntil {
            model.downloadJobs(for: .kimiK3).allSatisfy { $0.state.isReady }
                && storage.resolvedBookmarkData.count >= 2
        }

        XCTAssertEqual(Set(model.downloadJobs(for: .kimiK3).compactMap(\.jobID)), [first.job, second.job])
        XCTAssertEqual(Set(storage.resolvedBookmarkData), [firstBookmark, secondBookmark])
        XCTAssertEqual(
            Set(storage.registeredPaths),
            Set([firstDestination.standardizedFileURL.path, secondDestination.standardizedFileURL.path]))
        XCTAssertEqual(manager.verifyCount, 0)
        XCTAssertEqual(manager.resumeCount, 2)
        XCTAssertNil(model.downloadServiceError)
        XCTAssertNil(model.artifactRegistrationError)
    }

    func testRestoreNeverPresentsContradictoryRunningOrVerifyingSummaryAsActive() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        for reportedState in [
            DownloadJobSummary.JobState.running, .verifying,
        ] {
            let state = makeDownloadJobState(
                plan: plan,
                destination: root.appendingPathComponent(reportedState.rawValue),
                bookmark: Data(reportedState.rawValue.utf8),
                runState: reportedState)
            let store = InMemoryDownloadStateStore()
            try store.save(state)
            let manager = AppDownloadManagerSpy(
                plan: plan,
                restoreReport: DownloadRestoreReport(restoredJobs: [state.job]),
                summaries: [makeDownloadSummary(state: state, reportedState: reportedState)],
                events: [
                    state.job: [
                        .progress(makeDownloadProgress(job: state.job, plan: state.plan))
                    ]
                ])
            let model = makeDownloadAppModel(
                manager: manager, stateStore: store, storage: InMemoryLocationStorage())

            await model.restoreDownloads()

            XCTAssertTrue(model.downloadJobs(for: state.plan.model).isEmpty)
            XCTAssertTrue(model.canRetryDownloadRecovery)
            XCTAssertTrue(
                model.downloadRecoveryIssues.contains {
                    $0.job == state.job && $0.reason.contains("zero-network restore")
                })
            XCTAssertEqual(manager.resumeCount, 0)
        }
    }

    func testCapacityUnknownFailureIsNotMisreportedAsNetworkInterruption() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let job = DownloadJobID()
        let progress = makeDownloadProgress(job: job, plan: plan)
        let manager = AppDownloadManagerSpy(
            plan: plan, startJobID: job,
            events: [
                job: [
                    .failed(
                        .spaceUnknown(path: root.path, reason: "capacity metadata unavailable"),
                        progress)
                ]
            ])
        let controller = DownloadController(entry: CatalogFixtures.kimiK3, manager: manager)

        await controller.start(destination: root, scope: nil, options: DownloadOptions())
        try await waitUntil {
            if case .failed = controller.state { return true }
            return false
        }

        XCTAssertTrue(controller.state.phrase.contains("free space"))
        XCTAssertFalse(controller.state.phrase.localizedCaseInsensitiveContains("network"))
    }

    func testCancellingManualVerificationRestoresThePreviousReadyState() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let job = DownloadJobID()
        let state = makeDownloadJobState(
            plan: plan, destination: root, bookmark: Data("verify-cancel".utf8))
        let manager = AppDownloadManagerSpy(
            plan: plan, startJobID: job,
            events: [job: [makeFinishedEvent(state: state, job: job)]],
            verificationDelay: .seconds(60))
        let controller = DownloadController(entry: CatalogFixtures.kimiK3, manager: manager)

        await controller.start(destination: root, scope: nil, options: DownloadOptions())
        try await waitUntil { controller.state.isReady }
        let readyState = controller.state

        controller.verify()
        try await waitUntil { manager.verifyCount == 1 }
        guard case .verifying = controller.state else {
            return XCTFail("manual verification did not enter verifying")
        }

        controller.cancelVerification()
        try await waitUntil { controller.state == readyState }

        XCTAssertNil(controller.operationError)
        XCTAssertEqual(controller.state, readyState)
    }

    func testCancellingManualVerificationRestoresThePreviousPausedState() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let restored = makeDownloadJobState(
            plan: plan, destination: root, bookmark: Data("paused-verify-cancel".utf8))
        let manager = AppDownloadManagerSpy(
            plan: plan, verificationDelay: .seconds(60))
        let controller = DownloadController(
            entry: CatalogFixtures.kimiK3, manager: manager,
            restoredState: restored, summary: makeDownloadSummary(state: restored))
        let pausedState = controller.state

        controller.verify()
        try await waitUntil { manager.verifyCount == 1 }
        controller.cancelVerification()
        try await waitUntil { controller.state == pausedState }

        XCTAssertNil(controller.operationError)
        XCTAssertEqual(controller.state, pausedState)
    }

    func testCancellingManualVerificationRestoresThePreviousIncompleteState() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let job = DownloadJobID()
        let incomplete = VerificationReport(
            job: job, model: plan.model, depth: .digestPayload, checkedAt: Date(),
            ok: Array(plan.files.dropFirst()).map(\.path), missing: [plan.files[0].path],
            wrongSize: [], wrongDigest: [], unreadable: [:], extraneous: [],
            bytesToRefetch: plan.files[0].sizeBytes)
        let manager = AppDownloadManagerSpy(
            plan: plan, startJobID: job,
            events: [job: [.verificationFinished(incomplete)]],
            verificationDelay: .seconds(60))
        let controller = DownloadController(entry: CatalogFixtures.kimiK3, manager: manager)

        await controller.start(destination: root, scope: nil, options: DownloadOptions())
        try await waitUntil {
            if case .incomplete = controller.state { return true }
            return false
        }
        let incompleteState = controller.state

        controller.verify()
        try await waitUntil { manager.verifyCount == 1 }
        controller.cancelVerification()
        try await waitUntil { controller.state == incompleteState }

        XCTAssertNil(controller.operationError)
        XCTAssertEqual(controller.state, incompleteState)
    }

    func testManagerCancelledErrorAlsoRestoresThePreviousReadyState() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let job = DownloadJobID()
        let state = makeDownloadJobState(
            plan: plan, destination: root, bookmark: Data("manager-cancelled".utf8))
        let manager = AppDownloadManagerSpy(
            plan: plan, startJobID: job,
            events: [job: [makeFinishedEvent(state: state, job: job)]],
            verificationError: .cancelled)
        let controller = DownloadController(entry: CatalogFixtures.kimiK3, manager: manager)

        await controller.start(destination: root, scope: nil, options: DownloadOptions())
        try await waitUntil { controller.state.isReady }
        let readyState = controller.state

        controller.verify()
        try await waitUntil { manager.verifyCount == 1 && controller.state == readyState }

        XCTAssertNil(controller.operationError)
        XCTAssertEqual(controller.state, readyState)
    }

    func testCancellationRaceDoesNotHideANewerIncompleteVerificationReport() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let job = DownloadJobID()
        let state = makeDownloadJobState(
            plan: plan, destination: root, bookmark: Data("verify-race".utf8))
        let incomplete = VerificationReport(
            job: job, model: plan.model, depth: .digestPayload, checkedAt: Date(),
            ok: Array(plan.files.dropFirst()).map(\.path), missing: [plan.files[0].path],
            wrongSize: [], wrongDigest: [], unreadable: [:], extraneous: [],
            bytesToRefetch: plan.files[0].sizeBytes)
        let manager = AppDownloadManagerSpy(
            plan: plan, startJobID: job,
            events: [job: [makeFinishedEvent(state: state, job: job)]],
            verificationGate: IgnoringCancellationVerificationGate(),
            verificationReport: incomplete)
        let controller = DownloadController(entry: CatalogFixtures.kimiK3, manager: manager)

        await controller.start(destination: root, scope: nil, options: DownloadOptions())
        try await waitUntil { controller.state.isReady }
        controller.verify()
        try await waitUntil { manager.verifyCount == 1 }
        controller.cancelVerification()
        await manager.releaseVerification()
        try await waitUntil {
            if case .incomplete = controller.state { return true }
            return false
        }

        guard case .incomplete(let files, _) = controller.state else {
            return XCTFail("the newer verification report was not applied")
        }
        XCTAssertEqual(files, 1)
    }

    func testManualVerificationRejectsEmptyCoverageAndAnotherJobWithoutGettingStuck()
        async throws
    {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let invalidReports: [(String, (DownloadJobID) -> VerificationReport)] = [
            (
                "empty coverage",
                { job in
                    VerificationReport(
                        job: job, model: plan.model, depth: .digestPayload,
                        checkedAt: Date(), ok: [], missing: [], wrongSize: [],
                        wrongDigest: [], unreadable: [:], extraneous: [])
                }
            ),
            (
                "another job",
                { _ in
                    VerificationReport(
                        job: DownloadJobID(), model: plan.model, depth: .digestPayload,
                        checkedAt: Date(), ok: plan.files.map(\.path), missing: [],
                        wrongSize: [], wrongDigest: [], unreadable: [:], extraneous: [])
                }
            ),
        ]

        for (label, makeInvalidReport) in invalidReports {
            let job = DownloadJobID()
            let state = makeDownloadJobState(
                plan: plan, destination: root,
                bookmark: Data("invalid-report-\(label)".utf8))
            let manager = AppDownloadManagerSpy(
                plan: plan, startJobID: job,
                events: [job: [makeFinishedEvent(state: state, job: job)]],
                verificationReport: makeInvalidReport(job))
            let controller = DownloadController(
                entry: CatalogFixtures.kimiK3, manager: manager)
            var verifiedCallbacks = 0
            controller.onArtifactVerified = { _, _ in verifiedCallbacks += 1 }

            await controller.start(destination: root, scope: nil, options: DownloadOptions())
            try await waitUntil { controller.state.isReady }
            XCTAssertEqual(verifiedCallbacks, 1, label)

            controller.verify()
            try await waitUntil {
                if case .failed = controller.state { return true }
                return false
            }

            XCTAssertEqual(verifiedCallbacks, 1, label)
            XCTAssertNotNil(controller.operationError, label)
            if case .verifying = controller.state {
                XCTFail("invalid report left verification stuck: \(label)")
            }
        }
    }

    func testSizeOnlyTerminalReportsCannotMakeAnArtifactReady() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        for terminalKind in ["verification event", "finished summary"] {
            let job = DownloadJobID()
            let state = makeDownloadJobState(
                plan: plan, destination: root,
                bookmark: Data("size-only-\(terminalKind)".utf8))
            let report = VerificationReport(
                job: job, model: plan.model, depth: .sizeOnly, checkedAt: Date(),
                ok: plan.files.map(\.path), missing: [], wrongSize: [], wrongDigest: [],
                unreadable: [:], extraneous: [])
            let event: DownloadEvent
            if terminalKind == "verification event" {
                event = .verificationFinished(report)
            } else {
                event = .finished(
                    DownloadSummary(
                        job: job, model: plan.model, repo: plan.repo,
                        destinationPath: state.destinationPath,
                        bytesWritten: plan.totalBytes, networkBytes: plan.totalBytes,
                        wastedBytes: 0, wallSeconds: 1, meanBytesPerSecond: 1,
                        verification: report))
            }
            let manager = AppDownloadManagerSpy(
                plan: plan, startJobID: job, events: [job: [event]])
            let controller = DownloadController(
                entry: CatalogFixtures.kimiK3, manager: manager)
            var verifiedCallbacks = 0
            controller.onArtifactVerified = { _, _ in verifiedCallbacks += 1 }

            await controller.start(destination: root, scope: nil, options: DownloadOptions())
            try await waitUntil {
                if case .failed = controller.state { return true }
                return false
            }

            XCTAssertEqual(verifiedCallbacks, 0, terminalKind)
            XCTAssertFalse(controller.state.isReady, terminalKind)
            XCTAssertTrue(controller.operationError?.contains("depth") == true, terminalKind)
        }
    }

    func testDownloadReverificationConfirmationNamesHumanReadableReadSize() {
        let message = DownloadVerificationConfirmation.message(
            readBytes: CatalogFixtures.kimiK3.descriptor.totalBytes)

        XCTAssertTrue(message.contains("1.56 TB"))
        XCTAssertTrue(message.contains("read"))
        XCTAssertTrue(message.contains("cancel"))
        XCTAssertFalse(message.contains("1560000000000"))
    }

    func testNamedReadInterruptionKeepsItsCauseInsteadOfInventingNetworkFailure() async throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.kimiK3))
        let job = DownloadJobID()
        let progress = makeDownloadProgress(job: job, plan: plan)
        let manager = AppDownloadManagerSpy(
            plan: plan, startJobID: job,
            events: [
                job: [
                    .failed(
                        .interrupted(
                            path: "weights.bin", atByte: 42,
                            underlying: "the removable drive stopped responding"),
                        progress)
                ]
            ])
        let controller = DownloadController(entry: CatalogFixtures.kimiK3, manager: manager)

        await controller.start(destination: root, scope: nil, options: DownloadOptions())
        try await waitUntil {
            if case .interrupted = controller.state { return true }
            return false
        }

        XCTAssertTrue(controller.state.phrase.contains("removable drive stopped responding"))
        XCTAssertFalse(controller.state.phrase.localizedCaseInsensitiveContains("network"))
    }

    func testTerminalInterruptionCopyExplainsThatKeptFilesWillBeReused() {
        for reason: InterruptionReason in [
            .volumeDisappeared("K3NVME"), .network, .thermal,
        ] {
            XCTAssertTrue(reason.recoverySentence.localizedCaseInsensitiveContains("continue"))
            XCTAssertTrue(reason.recoverySentence.localizedCaseInsensitiveContains("kept"))
            XCTAssertFalse(reason.recoverySentence.contains("Resume"))
        }
    }

    func testVolumeShortcutCannotStartFromARawVolumesPath() throws {
        let appRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: appRoot.appendingPathComponent("Sources/Screens/DestinationPicker.swift"),
            encoding: .utf8)

        XCTAssertFalse(source.contains("appendingPathComponent(\"minirun\")"))
        XCTAssertFalse(source.contains("scope: nil"))
        XCTAssertFalse(source.contains(".downloadDestination"))
        XCTAssertFalse(source.contains("remembered:"))
        XCTAssertTrue(source.contains("switch FolderChooser.run("))
    }

    func testStorageSeparatesFolderAuthorityFromMountedDriveInventory() throws {
        let appRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let addSource = try String(
            contentsOf: appRoot.appendingPathComponent("Sources/Screens/AddStorageLocation.swift"),
            encoding: .utf8)
        let storageSource = try String(
            contentsOf: appRoot.appendingPathComponent("Sources/Screens/StorageSettingsView.swift"),
            encoding: .utf8)

        XCTAssertFalse(addSource.contains("Menu(\"Mounted drives\")"))
        XCTAssertTrue(storageSource.contains("title: \"Folders Minirun can access\""))
        XCTAssertTrue(storageSource.contains("SectionHeader(title: \"Mounted drives\""))
        XCTAssertTrue(storageSource.contains("struct StorageRefreshButton"))
        XCTAssertTrue(storageSource.contains("Text(\"Rescan\")"))
        XCTAssertTrue(storageSource.contains("model.refreshVolumes()"))
        XCTAssertTrue(storageSource.contains("model.installed.refresh()"))
    }

    func testAddingTheSameFolderTwiceSaysSoInsteadOfDoingNothing() async throws {
        let installed = makeInstalled()
        installed.addLocation(root)
        try await waitForScan(installed)
        installed.addLocation(root)
        XCTAssertEqual(installed.lastAddOutcome, .alreadyRegistered(path: root.path))
        XCTAssertEqual(installed.locationKeys.count, 1)
    }

    func testLocationAddRollsBackABookmarkThatCannotImmediatelyResolve() {
        let storage = ReauthorizingLocationStorage(failEveryResolve: true)
        let installed = InstalledModels(
            storage: storage, catalog: ModelCatalog.bundled,
            ledger: InMemoryVerificationLedger())

        installed.addLocation(root)

        XCTAssertFalse(installed.hasLocation)
        XCTAssertTrue(installed.lastAddOutcome?.isFailure ?? false)
        XCTAssertEqual(storage.forgetCount, 1)
        XCTAssertTrue(storage.pickerScope?.isReleased ?? false)
    }

    func testFreshPickerGrantReplacesBrokenBookmarkAtTheSamePath() {
        let storage = ReauthorizingLocationStorage(existingBrokenURL: root)
        let installed = InstalledModels(
            storage: storage, catalog: ModelCatalog.bundled,
            ledger: InMemoryVerificationLedger())

        installed.addLocation(root)

        XCTAssertEqual(installed.lastAddOutcome, .added(path: root.path))
        XCTAssertEqual(installed.locationKeys.count, 1)
        XCTAssertEqual(storage.forgetCount, 1)
        XCTAssertEqual(storage.rememberCount, 1)
    }

    /// Missing capacity is not zero capacity, and ordinary storage UI uses a
    /// readable decimal unit rather than an ungrouped byte count.
    func testCapacityCopyDistinguishesUnknownFromZeroAndUsesReadableUnits() {
        XCTAssertEqual(StorageCapacityPresentation.value(nil), "capacity unknown")
        XCTAssertEqual(StorageCapacityPresentation.freeSpace(nil), "free space unknown")
        XCTAssertEqual(StorageCapacityPresentation.value(0), "capacity unknown")
        XCTAssertEqual(StorageCapacityPresentation.freeSpace(0), "no free space")
        XCTAssertEqual(
            StorageCapacityPresentation.value(512_000_000_000),
            "512 GB")
        XCTAssertEqual(
            StorageCapacityPresentation.freeSpace(6_440_000_000),
            "6.44 GB free")
        XCTAssertEqual(
            StorageCapacityPresentation.reclaimableTotal(135_000_000_000),
            "Up to 135 GB available after cleanup")
        XCTAssertNotEqual(
            StorageCapacityPresentation.value(nil), MRFormat.bytesDecimal(Int64(0)))
        XCTAssertEqual(
            StorageVolumePresentation.systemImage(isInternal: true, isSelected: false),
            "internaldrive")
        XCTAssertEqual(
            StorageVolumePresentation.systemImage(isInternal: false, isSelected: false),
            "externaldrive")
        XCTAssertEqual(
            StorageVolumePresentation.systemImage(isInternal: false, isSelected: true),
            "externaldrive.fill")
    }

    func testProductByteCopyNeverTurnsMissingMetadataIntoZeroBytes() {
        XCTAssertEqual(MRFormat.publishedBytes(0), "not published")
        XCTAssertEqual(MRFormat.measuredBytes(nil), "not measured")
        XCTAssertEqual(MRFormat.measuredBytes(0), "not measured")
        XCTAssertEqual(MRFormat.bytesDecimal(UInt64(0)), "0 MB")
        XCTAssertEqual(MRFormat.bytesDecimal(UInt64(999_999)), "<1 MB")
        XCTAssertEqual(MRFormat.bytesDecimal(UInt64(1_000_000)), "1.00 MB")
    }

    func testGeneralAndAboutCopyDoNotLeakImplementationDetails() {
        XCTAssertEqual(
            ConversationStoragePresentation.folderDescription,
            "Saved privately in Minirun's app data")
        XCTAssertFalse(ConversationStoragePresentation.folderDescription.contains("/"))
        XCTAssertEqual(AboutPresentation.versionLine("1.2.3"), "Version 1.2.3")
        XCTAssertFalse(AboutPresentation.versionLine("1.2.3").contains("("))
        XCTAssertEqual(AboutPresentation.linksTitle, "Links")
        XCTAssertEqual(AboutPresentation.websiteTitle, "Website")
        XCTAssertEqual(AboutPresentation.websiteLabel, "minirun.dev")
        XCTAssertEqual(AboutPresentation.websiteURL.scheme, "https")
        XCTAssertEqual(AboutPresentation.websiteURL.host, "minirun.dev")
        XCTAssertEqual(AboutPresentation.privacyLabel, "minirun.dev/privacy")
        XCTAssertEqual(AboutPresentation.privacyURL.path, "/privacy")
        XCTAssertEqual(AboutPresentation.termsLabel, "minirun.dev/terms")
        XCTAssertEqual(AboutPresentation.termsURL.path, "/terms")
        XCTAssertEqual(
            GeneralSettingsPresentation.subtitle,
            "Choose how new chats start and where conversations are kept.")
        XCTAssertFalse(GeneralSettingsPresentation.subtitle.contains("runner"))
        XCTAssertFalse(ModelRunAvailability.runtimeUnavailable.productReason.contains("tokenizer"))
        XCTAssertFalse(ModelRunAvailability.runtimeUnavailable.productReason.contains("registry"))
        XCTAssertNil(ModelRunAvailability.available.productDetail)
        XCTAssertEqual(
            ModelRunAvailability.runtimeUnavailable.productDetail,
            ModelRunAvailability.runtimeUnavailable.productReason)
        XCTAssertEqual(ProjectionPresentation.notMeasuredTitle, "Not measured")
        XCTAssertFalse(
            ProjectionPresentation.notMeasuredMessage.localizedCaseInsensitiveContains(
                "calibrate"))
        XCTAssertFalse(
            ProjectionPresentation.notMeasuredMessage.localizedCaseInsensitiveContains(
                "run once"))
    }

    func testVolumeAssessmentsFollowTheCurrentOpenEndedCatalog() async throws {
        let storage = InMemoryLocationStorage()
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let assessments = VolumeAssessments(
            storage: storage, catalog: empty, ledger: InMemoryAssessmentLedger())
        XCTAssertTrue(assessments.catalogModelIDs.isEmpty)

        let base = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let futureID = ModelID.discoveredHuggingFaceRepository("example/Future-minirun")
        let future = ModelDescriptor(
            id: futureID, displayName: "Future Model",
            architecture: base.architecture, layout: base.layout,
            source: .huggingFaceRepo(
                HuggingFaceRepoRef(
                    repoID: "example/Future-minirun",
                    revision: String(repeating: "a", count: 40))),
            payloadBytes: 1_000_000, metadataBytes: 1_000,
            payloadFileCount: 1, metadataFileCount: 1,
            largestFileBytes: 1_000_000, minimumBudgetBytes: nil,
            runner: .none, licenseName: "test",
            licenseAcknowledgementRequired: false, notes: [])
        let live = ModelCatalogSnapshot(generatedAt: Date(), origin: .live, models: [future])
        assessments.updateCatalog(live)
        XCTAssertEqual(assessments.catalogModelIDs, [futureID])

        let key = StorageKey(InstalledModels.locationKeyPrefix + "assessment")
        try storage.remember(root, as: key)
        await assessments.assess(key: key, scan: nil, workloads: [:])

        let report = try XCTUnwrap(assessments.report(forLocationPath: root.path))
        XCTAssertEqual(report.verdicts.map(\.fit.model), [futureID])
        XCTAssertFalse(report.verdicts.contains { $0.fit.model == .kimiK3 })
    }

    func testProductMemorySurfacesDoNotEmbedArchivedExperimentNarrative() throws {
        let appRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let files = [
            "Sources/DesignSystem/Components/ProjectionStrip.swift",
            "Sources/DesignSystem/Components/RefusalCard.swift",
            "Sources/Screens/MemoryDialView.swift",
        ]
        let productSource = try files.map {
            try String(contentsOf: appRoot.appendingPathComponent($0), encoding: .utf8)
        }.joined(separator: "\n")
        let archivedFragments = [
            "11.31", "83.09", "71.78", "18.2", "arms C", "arms D", "0.230", "1.000",
            "Show the arithmetic", "Memory details",
        ]

        for fragment in archivedFragments {
            XCTAssertFalse(
                productSource.localizedCaseInsensitiveContains(fragment),
                "product memory surfaces must not expose archived experiment text: \(fragment)")
        }
    }

    func testModelDetailDoesNotRenderBundledEngineeringNotes() throws {
        let appRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: appRoot.appendingPathComponent("Sources/Screens/ModelDetailView.swift"),
            encoding: .utf8)

        XCTAssertFalse(source.contains("notes(entry)"))
        XCTAssertFalse(source.contains("descriptor.notes"))
        XCTAssertFalse(source.contains("Panel(title: \"Notes\")"))
    }

    func testModelArtifactStatusKeepsCompletenessAndVerificationSeparate() {
        let incomplete = discoveredArtifact(
            files: 3, expectedFiles: 5, verification: .fullyVerified)
        XCTAssertEqual(
            ModelArtifactPresentation.status(incomplete),
            "Incomplete — 2 files missing")

        let unverified = discoveredArtifact(
            files: 5, expectedFiles: 5, verification: .unverified)
        XCTAssertEqual(
            ModelArtifactPresentation.status(unverified),
            "5 files · Not verified")
    }

    // MARK: - It has to survive a relaunch

    /// A relaunch, as far as the storage layer is concerned, is a fresh
    /// `StorageManager` over the same defaults. The bookmark is minted for a
    /// directory inside this process's own container, which is what a sandboxed
    /// test may bookmark without a user having picked it; the user-picked case
    /// is the same call and is covered on the device record.
    func testAGrantedLocationIsStillThereAfterARelaunch() throws {
        let suiteName = "minirun.tests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let container = try containerDirectory()
        defer { try? FileManager.default.removeItem(at: container) }

        let key = StorageKey(InstalledModels.locationKeyPrefix + "relaunch")
        let before = makeStorage(defaults)
        let bookmark = try before.remember(container, as: key)
        XCTAssertEqual(bookmark.recordedPath, container.path)

        // Nothing is carried over but the defaults — the same thing a relaunch
        // carries over.
        let after = makeStorage(defaults)
        XCTAssertEqual(after.knownLocations(), [key])
        XCTAssertEqual(after.bookmark(key)?.recordedPath, container.path)

        let scope = try after.resolve(key)
        defer { scope.release() }
        XCTAssertEqual(scope.url.resolvingSymlinksInPath(), container.resolvingSymlinksInPath())

        // And an `InstalledModels` built over it — which is what the app does on
        // launch — sees the location without anybody adding it again.
        let installed = InstalledModels(
            storage: after, catalog: ModelCatalog.bundled,
            ledger: InMemoryVerificationLedger())
        XCTAssertTrue(installed.hasLocation)
        XCTAssertEqual(installed.locationKeys, [key])

        after.forget(key)
    }

    // MARK: - Helpers

    private func makeDownloadJobState(
        plan: DownloadPlan, destination: URL, bookmark: Data,
        updatedAt: Date = Date(),
        runState: DownloadJobSummary.JobState = .paused
    ) -> DownloadJobState {
        let createdAt = updatedAt.addingTimeInterval(-1)
        return DownloadJobState(
            job: DownloadJobID(), plan: plan, destinationPath: destination.path,
            destinationBookmark: bookmark, artifactRootPath: destination.path,
            artifactRootDevice: 1, artifactRootInode: 2,
            options: DownloadOptions(),
            fileStates: Dictionary(
                uniqueKeysWithValues: plan.files.map { ($0.path, FileState.pending) }),
            runState: runState, createdAt: createdAt, updatedAt: updatedAt)
    }

    private func makeDownloadSummary(
        state: DownloadJobState,
        reportedState: DownloadJobSummary.JobState = .paused,
        progress: DownloadProgress? = nil
    ) -> DownloadJobSummary {
        DownloadJobSummary(
            job: state.job, model: state.plan.model, state: reportedState,
            progress: progress ?? makeDownloadProgress(job: state.job, plan: state.plan))
    }

    private func makeDownloadProgress(job: DownloadJobID, plan: DownloadPlan) -> DownloadProgress {
        DownloadProgress(
            job: job, planTotalBytes: plan.totalBytes,
            verifiedBytes: 0, fetchedUnverifiedBytes: 0, inFlightBytes: 0,
            remainingBytes: plan.totalBytes,
            filesTotal: plan.files.count, filesVerified: 0, filesFailed: 0,
            networkBytes: 0, wastedBytes: 0,
            instantaneousBytesPerSecond: 0, smoothedBytesPerSecond: 0,
            estimatedTimeRemaining: nil, startedAt: Date(), elapsed: 0)
    }

    private func makeFinishedEvent(
        state: DownloadJobState, job: DownloadJobID? = nil
    ) -> DownloadEvent {
        let job = job ?? state.job
        let report = VerificationReport(
            job: job, model: state.plan.model, depth: .digestPayload,
            checkedAt: Date(), ok: state.plan.files.map(\.path), missing: [],
            wrongSize: [], wrongDigest: [], unreadable: [:], extraneous: [])
        return .finished(
            DownloadSummary(
                job: job, model: state.plan.model, repo: state.plan.repo,
                destinationPath: state.destinationPath,
                bytesWritten: state.plan.totalBytes, networkBytes: state.plan.totalBytes,
                wastedBytes: 0, wallSeconds: 1, meanBytesPerSecond: 1,
                verification: report))
    }

    private func makeDownloadAppModel(
        manager: any DownloadManaging, stateStore: any DownloadStateStore,
        storage: any StorageManaging
    ) -> AppModel {
        let snapshot = CatalogFixtures.snapshot.labelled(.live)
        let installed = InstalledModels(
            storage: storage, catalog: snapshot, ledger: InMemoryVerificationLedger())
        return AppModel(
            entries: CatalogFixtures.all, catalogSnapshot: snapshot,
            store: ConversationStore(
                directory: root.appendingPathComponent(
                    "Conversations-\(UUID().uuidString)", isDirectory: true)),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            storage: storage,
            downloadServices: AppDownloadServices(
                manager: manager, stateStore: stateStore,
                restoresPersistedJobs: true, setupError: nil),
            installed: installed, seedRecordedRuns: false, startDiscovery: false)
    }

    private func waitUntil(
        _ condition: @MainActor () -> Bool,
        file: StaticString = #filePath, line: UInt = #line
    ) async throws {
        for _ in 0..<200 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("condition did not become true", file: file, line: line)
    }

    private func makeInstalled() -> InstalledModels {
        InstalledModels(
            storage: InMemoryLocationStorage(), catalog: ModelCatalog.bundled,
            ledger: InMemoryVerificationLedger())
    }

    private func discoveredArtifact(
        files: Int, expectedFiles: Int, verification: ArtifactVerification
    ) -> DiscoveredArtifact {
        DiscoveredArtifact(
            rootPath: "/Volumes/Test/model", locationPath: "/Volumes/Test",
            index: .unreadable, model: .kimiK3, displayName: "Kimi K3",
            bytesOnDisk: 1_000_000_000, fileCount: files,
            expectedFileCount: expectedFiles, expectedBytes: 1_000_000_000,
            verification: verification, verifiedAt: nil, scannedAt: Date())
    }

    private func makeStorage(_ defaults: UserDefaults) -> StorageManager {
        StorageManager(
            store: SecurityScopedLocationStore(
                persistence: UserDefaultsBookmarkPersistence(
                    defaults: defaults, prefix: "minirun.tests.location.")),
            bookmarkLedger: UserDefaultsBookmarkLedger(
                defaults: defaults, prefix: "minirun.tests.ledger."))
    }

    /// A directory inside the app container. `Application Support` is where the
    /// app already writes, so a bookmark for something under it is a bookmark
    /// the sandbox will mint.
    private func containerDirectory() throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let directory = base.appendingPathComponent(
            "MinirunLocationTest-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func makeK3Artifact() throws {
        let index = """
            {"format":"quantized_tile_container","source_repo":"moonshotai/Kimi-K3",
             "source_revision":"9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
             "files":372,"bytes":1559976181760}
            """
        let directory = root.appendingPathComponent("k3-artifact")
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("global"), withIntermediateDirectories: true)
        try Data(index.utf8).write(to: directory.appendingPathComponent("index.json"))
        try Data(repeating: 0x41, count: 4096)
            .write(to: directory.appendingPathComponent("global/embed_tokens.bin"))
    }

    private func waitForScan(_ installed: InstalledModels) async throws {
        for _ in 0..<200 {
            if !installed.isScanning && installed.scanCount > 0 { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("the scan did not finish")
    }
}

/// A `StorageManaging` that remembers paths and mints nothing.
///
/// The bookmarking itself is covered by the relaunch test above and by
/// `MinirunKitTests`; what the rest of this file is about is what the app
/// concludes from a location existing.
private final class InMemoryLocationStorage: StorageManaging, @unchecked Sendable {
    private let lock = NSLock()
    private var remembered: [StorageKey: URL] = [:]

    func volumes() throws -> [VolumeDescriptor] { [] }
    func describe(_ url: URL) -> StorageInfo { StorageInfo.describing(path: url.path) }
    func freeSpace(at url: URL) -> FreeSpace { FreeSpace(describe(url)) }
    func canHold(bytes: UInt64, at url: URL, headroomBytes: UInt64) -> SpaceVerdict {
        .fits(spareBytes: 0)
    }
    func scope(for url: URL) throws -> StorageScope { StorageScope(url: url) }

    @discardableResult
    func remember(_ url: URL, as key: StorageKey) throws -> StorageBookmark {
        lock.lock()
        remembered[key] = url
        lock.unlock()
        return StorageBookmark(key: key, createdAt: Date(), recordedPath: url.path, byteCount: 0)
    }

    func resolve(_ key: StorageKey) throws -> StorageScope {
        lock.lock()
        let url = remembered[key]
        lock.unlock()
        guard let url else { throw StorageError.noBookmark(key) }
        return StorageScope(url: url)
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

private final class DestinationAuthorizationStorage: StorageManaging, @unchecked Sendable {
    private(set) var calls: [String] = []
    private(set) var rememberedKeys: [StorageKey] = []
    private(set) var forgottenKeys: [StorageKey] = []
    let scopeCoding = DestinationScopeCoding()

    func volumes() throws -> [VolumeDescriptor] { [] }
    func describe(_ url: URL) -> StorageInfo { StorageInfo.describing(path: url.path) }
    func freeSpace(at url: URL) -> FreeSpace { FreeSpace(describe(url)) }
    func canHold(bytes: UInt64, at url: URL, headroomBytes: UInt64) -> SpaceVerdict {
        .fits(spareBytes: 0)
    }
    func scope(for url: URL) throws -> StorageScope {
        calls.append("scope")
        return StorageScope(url: url, coding: scopeCoding)
    }
    func remember(_ url: URL, as key: StorageKey) throws -> StorageBookmark {
        calls.append("remember")
        rememberedKeys.append(key)
        return StorageBookmark(
            key: key, createdAt: Date(), recordedPath: url.path, byteCount: 1)
    }
    func resolve(_ key: StorageKey) throws -> StorageScope { throw StorageError.noBookmark(key) }
    func bookmark(_ key: StorageKey) -> StorageBookmark? { nil }
    func forget(_ key: StorageKey) {
        calls.append("forget")
        forgottenKeys.append(key)
    }
    func knownLocations() -> [StorageKey] { [] }
    func bookmarkData(for url: URL) throws -> Data { Data(url.path.utf8) }
}

private final class DestinationScopeCoding: SecurityScopedURLCoding, @unchecked Sendable {
    private let lock = NSLock()
    private var startCount = 0
    private var stopCount = 0

    var starts: Int { withLock { startCount } }
    var stops: Int { withLock { stopCount } }

    func bookmarkData(for url: URL) throws -> Data { Data(url.path.utf8) }
    func resolve(_ data: Data) throws -> ResolvedBookmark {
        throw CocoaError(.fileReadCorruptFile)
    }
    func startAccessing(_ url: URL) -> Bool {
        withLock { startCount += 1 }
        return true
    }
    func stopAccessing(_ url: URL) {
        withLock { stopCount += 1 }
    }

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }
}

private final class DestinationCapacityStorage: StorageManaging, @unchecked Sendable {
    struct Request: Equatable {
        let bytes: UInt64
        let headroom: UInt64
    }

    private(set) var capacityRequests: [Request] = []

    func volumes() throws -> [VolumeDescriptor] { [] }
    func describe(_ url: URL) -> StorageInfo { StorageInfo.describing(path: url.path) }
    func freeSpace(at url: URL) -> FreeSpace { FreeSpace(describe(url)) }
    func canHold(bytes: UInt64, at url: URL, headroomBytes: UInt64) -> SpaceVerdict {
        capacityRequests.append(Request(bytes: bytes, headroom: headroomBytes))
        let (_, overflowed) = bytes.addingReportingOverflow(headroomBytes)
        if overflowed {
            return .refused(
                .insufficientFreeSpace(
                    needBytes: bytes, headroomBytes: headroomBytes,
                    availableBytes: Int64.max, volume: "Test"))
        }
        return .fits(spareBytes: 0)
    }
    func scope(for url: URL) throws -> StorageScope { StorageScope(url: url) }
    func remember(_ url: URL, as key: StorageKey) throws -> StorageBookmark {
        StorageBookmark(key: key, createdAt: Date(), recordedPath: url.path, byteCount: 0)
    }
    func resolve(_ key: StorageKey) throws -> StorageScope { throw StorageError.noBookmark(key) }
    func bookmark(_ key: StorageKey) -> StorageBookmark? { nil }
    func forget(_ key: StorageKey) {}
    func knownLocations() -> [StorageKey] { [] }
    func bookmarkData(for url: URL) throws -> Data { Data(url.path.utf8) }
}

private final class ReauthorizingLocationStorage: StorageManaging, @unchecked Sendable {
    private let lock = NSLock()
    private var remembered: [StorageKey: URL] = [:]
    private var broken = Set<StorageKey>()
    private let failEveryResolve: Bool
    private(set) var forgetCount = 0
    private(set) var rememberCount = 0
    private(set) var pickerScope: StorageScope?

    init(existingBrokenURL: URL? = nil, failEveryResolve: Bool = false) {
        self.failEveryResolve = failEveryResolve
        if let existingBrokenURL {
            let key = StorageKey(InstalledModels.locationKeyPrefix + "broken")
            remembered[key] = existingBrokenURL
            broken.insert(key)
        }
    }

    func volumes() throws -> [VolumeDescriptor] { [] }
    func describe(_ url: URL) -> StorageInfo { StorageInfo.describing(path: url.path) }
    func freeSpace(at url: URL) -> FreeSpace { FreeSpace(describe(url)) }
    func canHold(bytes: UInt64, at url: URL, headroomBytes: UInt64) -> SpaceVerdict {
        .fits(spareBytes: 0)
    }
    func scope(for url: URL) throws -> StorageScope {
        let scope = StorageScope(url: url)
        pickerScope = scope
        return scope
    }
    func remember(_ url: URL, as key: StorageKey) throws -> StorageBookmark {
        lock.lock()
        rememberCount += 1
        remembered[key] = url
        lock.unlock()
        return StorageBookmark(
            key: key, createdAt: Date(), recordedPath: url.path, byteCount: 1)
    }
    func resolve(_ key: StorageKey) throws -> StorageScope {
        lock.lock()
        let url = remembered[key]
        let fails = failEveryResolve || broken.contains(key)
        lock.unlock()
        guard let url else { throw StorageError.noBookmark(key) }
        if fails {
            throw StorageError.bookmarkUnresolvable(key, reason: "test bookmark is broken")
        }
        return StorageScope(url: url)
    }
    func bookmark(_ key: StorageKey) -> StorageBookmark? {
        lock.lock()
        let url = remembered[key]
        lock.unlock()
        return url.map {
            StorageBookmark(key: key, createdAt: Date(), recordedPath: $0.path, byteCount: 1)
        }
    }
    func forget(_ key: StorageKey) {
        lock.lock()
        forgetCount += 1
        remembered[key] = nil
        broken.remove(key)
        lock.unlock()
    }
    func knownLocations() -> [StorageKey] {
        lock.lock()
        defer { lock.unlock() }
        return remembered.keys.sorted { $0.rawValue < $1.rawValue }
    }
    func bookmarkData(for url: URL) throws -> Data { Data(url.path.utf8) }
}

private actor RestoreGate {
    private var entered = false
    private var released = false
    private var entryWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

    func enterAndWait() async {
        entered = true
        let waiters = entryWaiters
        entryWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
        guard !released else { return }
        await withCheckedContinuation { releaseWaiters.append($0) }
    }

    func waitUntilEntered() async {
        guard !entered else { return }
        await withCheckedContinuation { entryWaiters.append($0) }
    }

    func release() {
        released = true
        let waiters = releaseWaiters
        releaseWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }
}

private struct SavedTransferCatalogSource: ModelCatalogSource {
    let snapshot: ModelCatalogSnapshot
    func fetch() async throws -> ModelCatalogSnapshot { snapshot }
}

private actor SequencedSavedTransferCatalogSource: ModelCatalogSource {
    private var snapshots: [ModelCatalogSnapshot]

    init(snapshots: [ModelCatalogSnapshot]) { self.snapshots = snapshots }

    func fetch() async throws -> ModelCatalogSnapshot {
        guard !snapshots.isEmpty else {
            throw CatalogError.malformedResponse("the test catalog sequence is empty")
        }
        return snapshots.removeFirst()
    }
}

private actor IgnoringCancellationVerificationGate {
    private var released = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        guard !released else { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func release() {
        released = true
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume() }
    }
}

private final class AppDownloadManagerSpy: DownloadManaging, @unchecked Sendable {
    private let lock = NSLock()
    private let resolvedPlan: DownloadPlan
    private let restoreReport: DownloadRestoreReport
    private let summaries: [DownloadJobSummary]
    private let restoreGate: RestoreGate?
    private let startJobID: DownloadJobID?
    private let eventsByJob: [DownloadJobID: [DownloadEvent]]
    private let verificationDelay: Duration?
    private let verificationError: DownloadError?
    private let verificationGate: IgnoringCancellationVerificationGate?
    private let verificationReport: VerificationReport?
    private var remainingRestoreFailures: Int
    private var restores = 0
    private var resumes = 0
    private var verifications = 0

    var startError: DownloadError?
    var releaseScopeBeforeStartError = false

    init(
        plan: DownloadPlan,
        restoreReport: DownloadRestoreReport = DownloadRestoreReport(),
        summaries: [DownloadJobSummary] = [],
        restoreGate: RestoreGate? = nil,
        restoreFailuresRemaining: Int = 0,
        startJobID: DownloadJobID? = nil,
        events: [DownloadJobID: [DownloadEvent]] = [:],
        verificationDelay: Duration? = nil,
        verificationError: DownloadError? = nil,
        verificationGate: IgnoringCancellationVerificationGate? = nil,
        verificationReport: VerificationReport? = nil
    ) {
        resolvedPlan = plan
        self.restoreReport = restoreReport
        self.summaries = summaries
        self.restoreGate = restoreGate
        remainingRestoreFailures = restoreFailuresRemaining
        self.startJobID = startJobID
        eventsByJob = events
        self.verificationDelay = verificationDelay
        self.verificationError = verificationError
        self.verificationGate = verificationGate
        self.verificationReport = verificationReport
    }

    var restoreCount: Int { withLock { restores } }
    var resumeCount: Int { withLock { resumes } }
    var verifyCount: Int { withLock { verifications } }

    func releaseVerification() async { await verificationGate?.release() }

    func plan(for model: ModelDescriptor) async throws -> DownloadPlan { resolvedPlan }

    func preflight(
        _ plan: DownloadPlan, into destination: DownloadDestination, options: DownloadOptions
    ) async throws -> DownloadPreflight {
        DownloadPreflight(
            destination: destination.directory.path,
            storage: StorageInfoSnapshot(StorageInfo.describing(path: destination.directory.path)),
            requiredBytes: plan.totalBytes, alreadyPresentBytes: 0,
            headroomBytes: options.freeSpaceHeadroomBytes, dependableFreeBytes: nil,
            fits: true, refusals: [], estimatedDuration: nil)
    }

    func start(
        _ plan: DownloadPlan, into destination: DownloadDestination, options: DownloadOptions
    ) async throws -> DownloadJobID {
        if let startError {
            if releaseScopeBeforeStartError { destination.scope?.release() }
            throw startError
        }
        destination.scope?.release()
        return startJobID ?? DownloadJobID()
    }

    func events(for job: DownloadJobID) -> AsyncStream<DownloadEvent> {
        let events = eventsByJob[job] ?? []
        return AsyncStream { continuation in
            for event in events { continuation.yield(event) }
            continuation.finish()
        }
    }

    func pause(_ job: DownloadJobID) async throws {}

    func resume(_ job: DownloadJobID, scope: StorageScope?) async throws {
        withLock { resumes += 1 }
    }

    func cancel(_ job: DownloadJobID, keepPartialFiles: Bool) async throws {}

    func verify(_ job: DownloadJobID, depth: VerificationDepth) async throws
        -> VerificationReport
    {
        withLock { verifications += 1 }
        if let verificationDelay { try await Task.sleep(for: verificationDelay) }
        if let verificationGate { await verificationGate.wait() }
        if let verificationError { throw verificationError }
        if let verificationReport { return verificationReport }
        return VerificationReport(
            job: job, model: resolvedPlan.model, depth: depth, checkedAt: Date(),
            ok: resolvedPlan.files.map(\.path), missing: [], wrongSize: [], wrongDigest: [],
            unreadable: [:], extraneous: [])
    }

    func repair(_ job: DownloadJobID, from report: VerificationReport) async throws
        -> DownloadJobID
    { job }

    func jobs() async -> [DownloadJobSummary] { summaries }

    func restorePersistedJobs() async throws -> DownloadRestoreReport {
        withLock { restores += 1 }
        if let restoreGate { await restoreGate.enterAndWait() }
        let shouldFail = withLock { () -> Bool in
            guard remainingRestoreFailures > 0 else { return false }
            remainingRestoreFailures -= 1
            return true
        }
        if shouldFail {
            throw DownloadError.statePersistenceFailed(
                job: DownloadJobID(), reason: "injected restore failure")
        }
        return restoreReport
    }

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }
}

private final class DuplicateDownloadStateStore: DownloadStateStore, @unchecked Sendable {
    private let lock = NSLock()
    private var states: [DownloadJobState]

    init(states: [DownloadJobState]) { self.states = states }

    func save(_ state: DownloadJobState) throws {
        withLock { states.append(state) }
    }

    func load(_ job: DownloadJobID) throws -> DownloadJobState? {
        withLock { states.first { $0.job == job } }
    }

    func all() throws -> [DownloadJobState] { withLock { states } }

    func remove(_ job: DownloadJobID) throws {
        withLock { states.removeAll { $0.job == job } }
    }

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }
}

private struct SequencedStoreFailure: Error {}

private final class SequencedDownloadStateStore: DownloadStateStore, @unchecked Sendable {
    private let lock = NSLock()
    private var states: [DownloadJobState]
    private let failingAllCalls: Set<Int>
    private var allCalls = 0

    init(states: [DownloadJobState], failingAllCalls: Set<Int>) {
        self.states = states
        self.failingAllCalls = failingAllCalls
    }

    func save(_ state: DownloadJobState) throws {
        withLock {
            states.removeAll { $0.job == state.job }
            states.append(state)
        }
    }

    func load(_ job: DownloadJobID) throws -> DownloadJobState? {
        withLock { states.first { $0.job == job } }
    }

    func all() throws -> [DownloadJobState] {
        let result = withLock { () -> (Int, [DownloadJobState]) in
            allCalls += 1
            return (allCalls, states)
        }
        if failingAllCalls.contains(result.0) { throw SequencedStoreFailure() }
        return result.1
    }

    func remove(_ job: DownloadJobID) throws {
        withLock { states.removeAll { $0.job == job } }
    }

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }
}

private final class ChangingDownloadStateStore: DownloadStateStore, @unchecked Sendable {
    private let lock = NSLock()
    private let before: [DownloadJobState]
    private var after: [DownloadJobState]
    private var allCalls = 0

    init(before: [DownloadJobState], after: [DownloadJobState]) {
        self.before = before
        self.after = after
    }

    func save(_ state: DownloadJobState) throws {
        withLock {
            after.removeAll { $0.job == state.job }
            after.append(state)
        }
    }

    func load(_ job: DownloadJobID) throws -> DownloadJobState? {
        withLock { after.first { $0.job == job } }
    }

    func all() throws -> [DownloadJobState] {
        withLock {
            allCalls += 1
            return allCalls == 1 ? before : after
        }
    }

    func remove(_ job: DownloadJobID) throws {
        withLock { after.removeAll { $0.job == job } }
    }

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }
}

private final class JobBookmarkStorage: StorageManaging, @unchecked Sendable {
    private let lock = NSLock()
    private let rawBookmarks: [Data: URL]
    private var remembered: [StorageKey: URL] = [:]
    private var rawResolutions: [Data] = []

    init(bookmarks: [Data: URL]) { rawBookmarks = bookmarks }

    var resolvedBookmarkData: [Data] { withLock { rawResolutions } }
    var registeredPaths: [String] { withLock { remembered.values.map(\.path) } }

    func volumes() throws -> [VolumeDescriptor] { [] }
    func describe(_ url: URL) -> StorageInfo { StorageInfo.describing(path: url.path) }
    func freeSpace(at url: URL) -> FreeSpace { FreeSpace(describe(url)) }
    func canHold(bytes: UInt64, at url: URL, headroomBytes: UInt64) -> SpaceVerdict {
        .fits(spareBytes: 0)
    }
    func scope(for url: URL) throws -> StorageScope { StorageScope(url: url) }

    func remember(_ url: URL, as key: StorageKey) throws -> StorageBookmark {
        withLock { remembered[key] = url.standardizedFileURL }
        return StorageBookmark(
            key: key, createdAt: Date(), recordedPath: url.standardizedFileURL.path,
            byteCount: 1)
    }

    func resolve(_ key: StorageKey) throws -> StorageScope {
        guard let url = withLock({ remembered[key] }) else {
            throw StorageError.noBookmark(key)
        }
        return StorageScope(url: url)
    }

    func bookmark(_ key: StorageKey) -> StorageBookmark? {
        withLock { remembered[key] }.map {
            StorageBookmark(
                key: key, createdAt: Date(), recordedPath: $0.path, byteCount: 1)
        }
    }

    func forget(_ key: StorageKey) { withLock { remembered[key] = nil } }

    func knownLocations() -> [StorageKey] {
        withLock { remembered.keys.sorted { $0.rawValue < $1.rawValue } }
    }

    func bookmarkData(for url: URL) throws -> Data { Data(url.standardizedFileURL.path.utf8) }

    func resolveBookmarkData(_ data: Data) throws -> ResolvedStorageBookmark {
        guard let url = rawBookmarks[data] else {
            throw StorageError.bookmarkDataUnresolvable(reason: "unknown test bookmark")
        }
        withLock { rawResolutions.append(data) }
        return ResolvedStorageBookmark(
            scope: StorageScope(url: url), bookmarkData: data, wasStale: false)
    }

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }
}
