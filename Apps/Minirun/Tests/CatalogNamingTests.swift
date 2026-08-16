import Foundation
import MinirunKit
import XCTest

@testable import MinirunApp

/// What the screens are allowed to call a model.
///
/// The review that produced these tests was blunt: the app was showing
/// `K3-flagship` where it meant Kimi K3. Internal identifiers are for code, and
/// a person reading a model list is entitled to the name the model is published
/// under. These pin that, in both the app's preview catalog and the kit's
/// bundled one, so a future edit to either cannot quietly reintroduce a
/// codename.
final class CatalogNamingTests: XCTestCase {

    /// The names the product uses, by identity.
    private static let expected: [ModelID: String] = [
        .kimiK3: "Kimi K3",
        .deepseekV4Flash: "DeepSeek V4 Flash",
        .minimaxH3: "MiniMax H3",
    ]

    func testEveryModelIsShownUnderItsRealName() throws {
        for (id, name) in Self.expected {
            let entry = try XCTUnwrap(CatalogFixtures.entry(id), "no fixture for \(id.rawValue)")
            XCTAssertEqual(entry.descriptor.displayName, name)
        }
        XCTAssertEqual(CatalogFixtures.all.count, Self.expected.count)
    }

    /// The kit ships its own catalog and the app ships a preview one. They are
    /// two files, and a name that disagreed between them would show one thing
    /// in the model list and another on a discovered artifact's pill.
    func testTheBundledCatalogAgreesWithThePreviewCatalog() throws {
        for (id, name) in Self.expected {
            let descriptor = try XCTUnwrap(
                ModelCatalog.bundled.descriptor(id), "no bundled descriptor for \(id.rawValue)")
            XCTAssertEqual(descriptor.displayName, name)
        }
    }

    /// A codename is a codename wherever it appears. `K3-flagship` is the one
    /// that shipped; the id spellings are the ones that could next.
    func testNoDisplayNameIsAnInternalIdentifier() {
        for entry in CatalogFixtures.all {
            let name = entry.descriptor.displayName
            XCTAssertFalse(
                name.lowercased().contains("flagship"),
                "\(name) is a codename, not a model name")
            XCTAssertNotEqual(name, entry.id.rawValue)
            XCTAssertFalse(name.contains("-minirun"), "\(name) names a repack, not a model")
        }
    }

    func testCatalogPublisherMarksUseKnownModelIdentityFirst() {
        XCTAssertEqual(
            ModelPublisherIdentity.resolve(
                modelID: .kimiK3, displayName: "unrelated", repositoryID: nil),
            .kimi)
        XCTAssertEqual(
            ModelPublisherIdentity.resolve(
                modelID: .deepseekV4Flash, displayName: "unrelated", repositoryID: nil),
            .deepSeek)
        XCTAssertEqual(
            ModelPublisherIdentity.resolve(
                modelID: .minimaxH3, displayName: "unrelated", repositoryID: nil),
            .miniMax)
    }

    func testNewPublishedModelsUseOnlyUnambiguousBrandTokens() {
        let discovered = ModelID.discoveredHuggingFaceRepository(
            "nanguoyu/Qwen3.8-2.4T-A95B-minirun")
        XCTAssertEqual(
            ModelPublisherIdentity.resolve(
                modelID: discovered,
                displayName: "Qwen3.8-2.4T-A95B",
                repositoryID: "nanguoyu/Qwen3.8-2.4T-A95B-minirun"),
            .qwen)
        XCTAssertEqual(
            ModelPublisherIdentity.resolve(
                modelID: ModelID("future-model"),
                displayName: "Future Model",
                repositoryID: "nanguoyu/future-model-minirun"),
            .unknown)
        XCTAssertEqual(ModelPublisherIdentity.kimi.publisherName, "Moonshot AI")
        XCTAssertNil(ModelPublisherIdentity.unknown.assetName)
    }

    /// One compact subtitle, and it says which build of the model this is —
    /// which is the fact a name alone cannot carry.
    func testTheSubtitleNamesWhereTheBytesComeFrom() throws {
        let k3 = try XCTUnwrap(CatalogFixtures.entry(.kimiK3))
        XCTAssertEqual(k3.provenanceSubtitle, "nanguoyu/Kimi-K3-minirun")

        // No published model is built locally today, but the branch still
        // exists and must name the tool rather than invent a repository line.
        let locallyBuilt = CatalogEntry(
            descriptor: locallyBuiltDescriptor(),
            index: k3.index, memory: k3.memory, terms: k3.terms,
            architectureSubtitle: k3.architectureSubtitle)
        XCTAssertEqual(
            locallyBuilt.provenanceSubtitle, "built locally by Tools/local_containers")
    }

    /// A bundled fixture list is not a product ceiling. The catalog
    /// header reports the open-ended scan result and never renders "x of 4".
    func testTheCatalogSummaryReportsWhatTheScanFoundWithoutAClosedTotal() {
        let summary = ModelCatalogPresentation.scanSummary(
            installedCount: 1, mountedLocationCount: 1, unavailableLocationCount: 2)
        XCTAssertEqual(summary, "1 local model · 1 location mounted · 2 unavailable")
        XCTAssertFalse(summary.contains("of 4"))

        XCTAssertEqual(
            ModelCatalogPresentation.scanSummary(
                installedCount: 0, mountedLocationCount: 2, unavailableLocationCount: 0),
            "0 local models · 2 locations mounted")
    }

    func testModelRowPrefersAnActualLastRunOverAStorageProjection() {
        let fitness = PlatformFitness(
            verdict: .runnable, reason: "supported", requiredFreeBytes: 1,
            headroomBytes: 1)

        XCTAssertEqual(
            ModelRowPresentation.trailingArgument(
                fitness: fitness, lastRunSecondsPerToken: 265),
            "Last run 4:25/token")
    }

    func testModelRowCallsAnAbsentProductRunWhatItIs() {
        let fitness = PlatformFitness(
            verdict: .runnable, reason: "supported", requiredFreeBytes: 1,
            headroomBytes: 1)

        XCTAssertNil(
            ModelRowPresentation.trailingArgument(
                fitness: fitness, lastRunSecondsPerToken: nil))
    }

    @MainActor
    func testPersistedTurnRateRemainsAttributedAfterAConversationChangesModel() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunPersistedRate-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ConversationStore(directory: directory)
        var conversation = Conversation(
            settings: ConversationSettings(
                model: .deepseekV4Flash, memoryBudgetBytes: 8_000_000_000))
        conversation.append(
            Message(
                role: .assistant, text: "measured",
                telemetry: MessageTelemetry(
                    model: .kimiK3, wallSeconds: 530, tokensPerSecond: 1 / 265,
                    bytesPerToken: nil, peakFootprintBytes: 1,
                    declaredBudgetBytes: 8_000_000_000, budgetRespected: true,
                    logitsDigest: nil, thermalState: nil)))
        try store.save(conversation)

        let entries = CatalogFixtures.productPreview
        let model = AppModel(
            entries: entries, catalogSnapshot: CatalogFixtures.productPreviewSnapshot,
            runtimes: .preview(entries: entries), store: store,
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: entries), startDiscovery: false)

        XCTAssertEqual(
            try XCTUnwrap(model.lastRunSecondsPerToken(for: .kimiK3)),
            265, accuracy: 0.001)
        XCTAssertNil(model.lastRunSecondsPerToken(for: .deepseekV4Flash))
    }

    func testCatalogSourceCopyDistinguishesLiveCacheFallbackAndReviewData() {
        XCTAssertEqual(
            ModelCatalogPresentation.sourceStatus(
                origin: .live, isRefreshing: false, hasRefreshError: false,
                isReviewPreview: false),
            "Up to date")
        XCTAssertEqual(
            ModelCatalogPresentation.sourceStatus(
                origin: .cached, isRefreshing: false, hasRefreshError: false,
                isReviewPreview: false),
            "Saved model list")
        XCTAssertEqual(
            ModelCatalogPresentation.sourceStatus(
                origin: .bundled, isRefreshing: false, hasRefreshError: true,
                isReviewPreview: false),
            "Update unavailable")
        XCTAssertEqual(
            ModelCatalogPresentation.sourceStatus(
                origin: .bundled, isRefreshing: true, hasRefreshError: false,
                isReviewPreview: true),
            "Sample model list")
    }

    func testModelFitnessUsesProductLanguageInsteadOfRuntimeCodes() {
        XCTAssertEqual(ModelFitnessPresentation.status(.runnable), "Ready")
        XCTAssertEqual(
            ModelFitnessPresentation.status(.runnableWithCaveats), "Runs with limits")
        XCTAssertEqual(
            ModelFitnessPresentation.status(.refused), "Can't run on this device")
        XCTAssertEqual(
            ModelFitnessPresentation.status(.noRunner), "Can't run in this version")
    }

    func testModelDetailSeparatesAppCompatibilityFromFileIntegrity() {
        let fitness = PlatformFitness(
            verdict: .runnable,
            reason: "A verified runtime is available and fits this device's reported budget.",
            requiredFreeBytes: 1,
            headroomBytes: 1)
        let artifact = DiscoveredArtifact(
            rootPath: "/Volumes/Test/K3", locationPath: "/Volumes/Test",
            index: .unreadable, model: .kimiK3, displayName: "Kimi K3",
            bytesOnDisk: 10, fileCount: 2, expectedFileCount: 2,
            expectedBytes: 10, verification: .unverified, verifiedAt: nil,
            scannedAt: Date())

        let compatibility = ModelCompatibilityPresentation.resolve(
            modelName: "Kimi K3", fitness: fitness,
            availability: .unverifiedArtifact)
        XCTAssertEqual(compatibility.status, "Supported")
        XCTAssertEqual(compatibility.reason, "This version supports Kimi K3 on this device.")
        XCTAssertEqual(compatibility.tone, .ok)

        let integrity = ModelIntegrityPresentation.resolve(
            installations: [artifact], mounted: [artifact])
        XCTAssertEqual(integrity.status, "Full verification required")
        XCTAssertEqual(
            integrity.reason, "Verify all files before using this local copy in a chat.")
        XCTAssertEqual(integrity.tone, .verify)
    }

    func testModelDetailShowsMissingRunnerAndSpotCheckAsIndependentFacts() {
        let fitness = PlatformFitness(
            verdict: .noRunner,
            reason: "This version can find the model, but cannot run it yet.",
            requiredFreeBytes: 1,
            headroomBytes: nil)
        let artifact = DiscoveredArtifact(
            rootPath: "/Volumes/Test/H3", locationPath: "/Volumes/Test",
            index: .unreadable, model: .minimaxH3, displayName: "MiniMax H3",
            bytesOnDisk: 10, fileCount: 2, expectedFileCount: 2,
            expectedBytes: 10, verification: .spotChecked, verifiedAt: Date(),
            scannedAt: Date())

        let compatibility = ModelCompatibilityPresentation.resolve(
            modelName: "MiniMax H3", fitness: fitness,
            availability: .unverifiedArtifact)
        XCTAssertEqual(compatibility.status, "Not supported by this version")
        XCTAssertEqual(
            compatibility.reason,
            "This version can manage MiniMax H3, but cannot use it in a chat.")
        XCTAssertEqual(compatibility.tone, .neutral)

        let integrity = ModelIntegrityPresentation.resolve(
            installations: [artifact], mounted: [artifact])
        XCTAssertEqual(integrity.status, "Spot-check only")
        XCTAssertTrue(integrity.reason.contains("Verify all files"))
        XCTAssertEqual(integrity.tone, .verify)
    }

    func testModelsSeparatesLocalInventoryFromRemoteDiscovery() throws {
        let appRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: appRoot.appendingPathComponent("Sources/Screens/ModelCatalogView.swift"),
            encoding: .utf8)
        let toolbarStart = try XCTUnwrap(source.range(of: ".toolbar {"))
        let toolbarEnd = try XCTUnwrap(
            source.range(of: ".task {", range: toolbarStart.upperBound..<source.endIndex))
        let toolbar = String(source[toolbarStart.lowerBound..<toolbarEnd.lowerBound])

        XCTAssertTrue(source.contains(".mrWorkspaceSurface()"))
        XCTAssertTrue(toolbar.contains("FindModelsButton()"))
        XCTAssertTrue(source.contains("Button(\"Find Models\")"))
        XCTAssertTrue(source.contains("RemoteModelBrowserView"))
        XCTAssertTrue(source.contains("ForEach(model.localEntriesForPresentation)"))
        XCTAssertTrue(source.contains("model.publishedModelsForPresentation"))
        XCTAssertEqual(toolbar.components(separatedBy: "ToolbarItem").count - 1, 1)
        XCTAssertTrue(source.contains("model.installed.refresh()"))
        XCTAssertTrue(source.contains("await model.refreshPublishedModels()"))
        XCTAssertFalse(source.contains("ModelCatalogRefreshButton"))

        let browserStart = try XCTUnwrap(
            source.range(of: "private struct RemoteModelBrowserView"))
        let browserEnd = try XCTUnwrap(
            source.range(of: "private struct RemoteModelDetailLoader"))
        let browser = String(source[browserStart.lowerBound..<browserEnd.lowerBound])
        XCTAssertTrue(browser.contains("refreshPublishedModelsIfNeeded"))
        XCTAssertFalse(browser.contains("refreshCatalog()"))
    }

    func testRemoteModelBrowserSearchAndActionsAreExplicit() {
        let k3 = CatalogFixtures.kimiK3
        let published = PublishedModelReference(
            id: k3.id, displayName: k3.descriptor.displayName,
            repository: k3.descriptor.source.repo!)

        XCTAssertTrue(RemoteModelBrowserPresentation.matches(published, query: "kimi"))
        XCTAssertTrue(RemoteModelBrowserPresentation.matches(published, query: "nanguoyu"))
        XCTAssertFalse(RemoteModelBrowserPresentation.matches(published, query: "minimax"))
        XCTAssertEqual(RemoteModelBrowserPresentation.summary(count: 3), "3 published models")
    }

    func testRemoteModelBrowserMakesTheWholeRowTheNavigationTarget() throws {
        let appRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: appRoot.appendingPathComponent("Sources/Screens/ModelCatalogView.swift"),
            encoding: .utf8)
        let rowStart = try XCTUnwrap(source.range(of: "private func remoteRow"))
        let rowEnd = try XCTUnwrap(
            source.range(of: "private var remoteEmptyState", range: rowStart.upperBound..<source.endIndex))
        let row = String(source[rowStart.lowerBound..<rowEnd.lowerBound])

        XCTAssertTrue(row.contains(".frame(maxWidth: .infinity, alignment: .leading)"))
        XCTAssertTrue(row.contains(".contentShape(Rectangle())"))
        XCTAssertTrue(row.contains("Image(systemName: \"chevron.right\")"))
        XCTAssertFalse(row.contains("hasLocalCopy ? \"On this device\" : \"View\""))
    }

    func testRemoteBrowserKeepsTheCurrentListUsableWhileAnExplicitUpdateRuns() {
        XCTAssertEqual(
            ModelCatalogPresentation.sourceStatus(
                origin: .cached, isRefreshing: true, hasRefreshError: false,
                isReviewPreview: false),
            "Current list · updating")
    }

    func testModelDetailDoesNotRenderEmptyTimingOrHistorySections() throws {
        let appRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let detail = try String(
            contentsOf: appRoot.appendingPathComponent("Sources/Screens/ModelDetailView.swift"),
            encoding: .utf8)
        let dial = try String(
            contentsOf: appRoot.appendingPathComponent("Sources/Screens/MemoryDialView.swift"),
            encoding: .utf8)

        XCTAssertTrue(detail.contains("if !records.isEmpty { history(records) }"))
        XCTAssertFalse(detail.contains("Download another copy"))
        XCTAssertFalse(detail.contains("No run of this model has been made"))
        XCTAssertFalse(dial.contains("Projected speed"))
        XCTAssertFalse(dial.contains("ProjectionStrip("))
    }

    /// Product rows are keyed by the kit snapshot, not by the preview
    /// fixtures. K3 makes the distinction observable: the bundled descriptor
    /// carries the repository's own licence tag, while the preview fixture
    /// carries the App's shorter product wording for the same model.
    func testTheProductAdapterTakesItsSetAndDescriptorsFromTheKitSnapshot() {
        let snapshot = ModelCatalog.bundled
        let entries = AppCatalogAdapter.entries(from: snapshot)

        XCTAssertEqual(entries.map(\.descriptor), snapshot.models)
        XCTAssertEqual(
            entries.first { $0.id == .kimiK3 }?.descriptor.licenseName,
            "other (repository tag license:other; see the repository LICENSE)")
        XCTAssertNotEqual(
            entries.first { $0.id == .kimiK3 }?.descriptor.licenseName,
            CatalogFixtures.kimiK3.descriptor.licenseName)
    }

    /// A future descriptor must become a row without being added to
    /// `CatalogFixtures.all`. App-only geometry may be unknown; catalog facts
    /// must still survive field for field.
    func testTheProductAdapterAcceptsADescriptorWithNoPreviewFixture() throws {
        let id = ModelID("future-model")
        let descriptor = ModelDescriptor(
            id: id,
            displayName: "Future Model",
            architecture: .deepseekV4FlashMoE,
            layout: .v4FlashUnitBundle,
            source: .locallyBuilt(tool: "future-tool"),
            payloadBytes: 123,
            metadataBytes: 7,
            payloadFileCount: 3,
            metadataFileCount: 1,
            largestFileBytes: 100,
            minimumBudgetBytes: nil,
            runner: .none,
            licenseName: "test",
            licenseAcknowledgementRequired: false,
            notes: [])
        let snapshot = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .cached, models: [descriptor])

        let entry = try XCTUnwrap(AppCatalogAdapter.entries(from: snapshot).only)
        XCTAssertEqual(entry.descriptor, descriptor)
        XCTAssertEqual(entry.index.files, 3)
        XCTAssertEqual(entry.index.bytes, 123)
        XCTAssertNil(CatalogFixtures.entry(id))
    }

    func testProductCatalogKeepsEveryOwnedHFContainerIncludingUnsupportedH3() throws {
        let futureID = ModelID.discoveredHuggingFaceRepository(
            "nanguoyu/Future-Chat-minirun")
        let future = ModelDescriptor(
            id: futureID,
            displayName: "Future Chat",
            architecture: .unrecognized,
            layout: .unrecognized,
            source: .huggingFaceRepo(
                HuggingFaceRepoRef(
                    repoID: "nanguoyu/Future-Chat-minirun",
                    revision: String(repeating: "f", count: 40))),
            payloadBytes: 123,
            metadataBytes: 7,
            payloadFileCount: 3,
            metadataFileCount: 1,
            largestFileBytes: 100,
            minimumBudgetBytes: nil,
            runner: .none,
            licenseName: "test",
            licenseAcknowledgementRequired: false,
            notes: [])
        let raw = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .live,
            models: [
                CatalogFixtures.kimiK3.descriptor,
                CatalogFixtures.deepseekV4Flash.descriptor,
                CatalogFixtures.minimaxH3.descriptor,
                locallyBuiltDescriptor(),
                future,
            ])

        let product = ProductCatalogPolicy.applying(to: raw)

        XCTAssertEqual(
            product.models.map(\.id),
            [.kimiK3, .deepseekV4Flash, .minimaxH3, futureID])
        XCTAssertNotNil(product.descriptor(.deepseekV4Flash))
        XCTAssertNotNil(product.descriptor(.minimaxH3))
        XCTAssertNotNil(product.descriptor(futureID))
    }

    #if DEBUG
        @MainActor
        func testReviewPreviewMatchesTheCurrentChatProductScope() {
            let model = AppModel.preview()

            XCTAssertEqual(
                model.entries.map(\.id), [.kimiK3, .deepseekV4Flash, .minimaxH3])
            XCTAssertNotNil(model.entry(.minimaxH3))
            XCTAssertNil(model.capabilities(for: .minimaxH3))
        }
    #endif

    func testTheProductAdapterKeepsCompatibleGeometryAcrossRepositoryUpdates() throws {
        let known = CatalogFixtures.kimiK3.descriptor
        let moved = ModelDescriptor(
            id: known.id, displayName: known.displayName,
            architecture: known.architecture, layout: known.layout,
            source: .huggingFaceRepo(
                HuggingFaceRepoRef(
                    repoID: try XCTUnwrap(known.source.repo).repoID,
                    revision: String(repeating: "b", count: 40))),
            payloadBytes: known.payloadBytes, metadataBytes: known.metadataBytes,
            payloadFileCount: known.payloadFileCount,
            metadataFileCount: known.metadataFileCount,
            largestFileBytes: known.largestFileBytes,
            minimumBudgetBytes: known.minimumBudgetBytes,
            runner: known.runner, licenseName: known.licenseName,
            licenseAcknowledgementRequired: known.licenseAcknowledgementRequired,
            notes: known.notes)

        let adapted = AppCatalogAdapter.entry(from: moved)

        XCTAssertEqual(adapted.descriptor, moved)
        XCTAssertEqual(adapted.memory.census, CatalogFixtures.kimiK3.memory.census)
        XCTAssertEqual(
            adapted.memory.expertPoolBytes, CatalogFixtures.kimiK3.memory.expertPoolBytes)
        XCTAssertEqual(
            adapted.memory.workingSetReserveBytes,
            CatalogFixtures.kimiK3.memory.workingSetReserveBytes)
        XCTAssertEqual(
            adapted.memory.onRecordMinimumBudgetBytes,
            CatalogFixtures.kimiK3.memory.onRecordMinimumBudgetBytes)
        XCTAssertEqual(adapted.memory.provenance, .declaredByIndex)
        XCTAssertEqual(adapted.terms, CatalogFixtures.kimiK3.terms)
        XCTAssertFalse(adapted.terms.isMeasuredHere)
    }

    @MainActor
    func testTheDefaultAppModelStartsWithNoVisiblePresetCatalog() {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunCatalogTest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: []),
            seedRecordedRuns: false,
            startDiscovery: false)

        XCTAssertEqual(model.snapshot.origin, .bundled)
        XCTAssertTrue(model.snapshot.models.isEmpty)
        XCTAssertTrue(model.entries.isEmpty)
        XCTAssertTrue(model.catalogEntriesForPresentation.isEmpty)
        XCTAssertTrue(model.isAwaitingInitialCatalog)
        XCTAssertTrue(model.assessments.catalogModelIDs.isEmpty)
    }

    @MainActor
    func testCatalogRunnerMetadataCannotMakeAnyUnlinkedHuggingFaceRuntimeLookRunnable() {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunFitnessTest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let entries = CatalogFixtures.all.filter { $0.descriptor.source.repo != nil }
        let snapshot = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .live,
            models: entries.map(\.descriptor))
        let model = AppModel(
            entries: entries, catalogSnapshot: snapshot,
            runtimes: ModelRuntimeRegistry(),
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: entries),
            seedRecordedRuns: false, startDiscovery: false)

        XCTAssertEqual(entries.count, 3)
        XCTAssertEqual(CatalogFixtures.kimiK3.descriptor.runner, .decodeRunner)
        for entry in entries {
            let fitness = model.effectiveProductFitness(for: entry)
            XCTAssertFalse(model.hasVerifiedRuntime(for: entry.id))
            XCTAssertEqual(fitness.verdict, .noRunner)
            XCTAssertTrue(fitness.reason.contains("cannot run it yet"))
            XCTAssertEqual(fitness.requiredFreeBytes, entry.descriptor.totalBytes)
            XCTAssertNil(fitness.headroomBytes)
            XCTAssertNil(model.projection(for: entry))
            XCTAssertEqual(
                ModelRowPresentation.trailingArgument(
                    fitness: fitness,
                    lastRunSecondsPerToken: nil),
                "Storage and verification only")
        }
    }

    @MainActor
    func testReviewCatalogCannotFetchOrWriteTheProductCache() async {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunReviewCatalog-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = CatalogFetchSpy(snapshot: CatalogFixtures.snapshot.labelled(.live))
        let cache = CatalogCacheSpy()
        let model = AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            catalogService: ModelCatalog(
                live: source, bundled: CatalogFixtures.snapshot, cache: cache),
            catalogCache: cache,
            allowsLiveCatalogRefresh: false,
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: CatalogFixtures.all),
            seedRecordedRuns: false,
            startDiscovery: false)

        await model.refreshCatalog()

        XCTAssertEqual(source.fetchCount, 0)
        XCTAssertEqual(cache.saveCount, 0)
        XCTAssertEqual(model.snapshot, CatalogFixtures.snapshot)
    }

    @MainActor
    func testLiveLaunchFetchesCatalogOnceAndConcurrentModelsAppearanceDoesNotDuplicateIt()
        async throws
    {
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let gate = CatalogFetchGate()
        let source = BlockingCatalogSource(
            snapshot: CatalogFixtures.snapshot.labelled(.live), gate: gate)
        let defaults = UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!
        let model = AppModel.live(
            userDefaults: defaults,
            catalogService: ModelCatalog(live: source, bundled: empty, cache: nil),
            conversationFactory: { .ephemeral() },
            downloadServicesFactory: { _ in .unavailable("test download service") },
            storageFactory: {
                StorageManager(bookmarkLedger: InMemoryBookmarkLedger())
            },
            startDiscovery: false)

        await gate.waitUntilEntered()
        XCTAssertEqual(source.listCount, 1)
        XCTAssertEqual(source.fetchCount, 0)
        XCTAssertTrue(model.isAwaitingInitialCatalog)
        await model.refreshPublishedModels()
        XCTAssertEqual(source.listCount, 1)

        await gate.release()
        for _ in 0..<200 {
            if !model.isAwaitingInitialCatalog { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(source.listCount, 1)
        XCTAssertEqual(source.fetchCount, 0)
        XCTAssertEqual(model.snapshot.origin, .bundled)
        XCTAssertFalse(model.isAwaitingInitialCatalog)
        // `entries` is the recognition table and is bundled until a live
        // response lands. What must stay empty is the storefront: a bundled
        // descriptor can name an artifact on the disk, and cannot claim that a
        // repository is published.
        XCTAssertTrue(model.catalogEntriesForPresentation.isEmpty)
        XCTAssertEqual(
            model.publishedModelsForPresentation.map(\.id),
            CatalogFixtures.snapshot.models.compactMap { descriptor in
                descriptor.source.repo == nil ? nil : descriptor.id
            }.sorted { $0.rawValue < $1.rawValue })
    }

    @MainActor
    func testLiveLaunchUsesARecentSavedCatalogWithoutTouchingTheNetwork() async throws {
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let cached = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .live,
            models: CatalogFixtures.snapshot.models)
        let cache = StaticCatalogCache(snapshot: cached)
        let source = CatalogFetchSpy(snapshot: CatalogFixtures.snapshot.labelled(.live))
        let defaults = UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!
        let model = AppModel.live(
            userDefaults: defaults,
            catalogService: ModelCatalog(live: source, bundled: empty, cache: cache),
            catalogCache: cache,
            conversationFactory: { .ephemeral() },
            downloadServicesFactory: { _ in .unavailable("test download service") },
            storageFactory: {
                StorageManager(bookmarkLedger: InMemoryBookmarkLedger())
            },
            startDiscovery: false)

        XCTAssertEqual(model.snapshot.origin, .cached)
        XCTAssertEqual(
            model.entries.map(\.id),
            AppCatalogAdapter.entries(from: ProductCatalogPolicy.applying(to: cached)).map(\.id))
        XCTAssertFalse(model.isAwaitingInitialCatalog)
        for _ in 0..<20 { await Task.yield() }
        XCTAssertEqual(source.fetchCount, 0)
    }

    @MainActor
    func testFindModelsRefreshesOnlyTheLightweightListOncePerLaunch() async throws {
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let cached = CatalogFixtures.snapshot.labelled(.live)
        let cache = StaticCatalogCache(snapshot: cached)
        let source = CatalogFetchSpy(snapshot: cached)
        let model = AppModel.live(
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            catalogService: ModelCatalog(live: source, bundled: empty, cache: cache),
            catalogCache: cache,
            conversationFactory: { .ephemeral() },
            downloadServicesFactory: { _ in .unavailable("test download service") },
            storageFactory: {
                StorageManager(bookmarkLedger: InMemoryBookmarkLedger())
            },
            startDiscovery: false)

        await model.refreshPublishedModelsIfNeeded()
        await model.refreshPublishedModelsIfNeeded()

        XCTAssertEqual(source.listCount, 1)
        XCTAssertEqual(source.fetchCount, 0)
        XCTAssertEqual(model.publishedModelsOrigin, .live)
        XCTAssertEqual(model.publishedModelsForPresentation.count, 3)
        XCTAssertEqual(model.snapshot.origin, .cached)
    }

    @MainActor
    func testLiveLaunchKeepsAnOldSavedCatalogOfflineUntilExplicitRefresh() async throws {
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let cached = ModelCatalogSnapshot(
            generatedAt: Date(timeIntervalSince1970: 1),
            origin: .live,
            models: CatalogFixtures.snapshot.models)
        let cache = StaticCatalogCache(snapshot: cached)
        let source = CatalogFetchSpy(snapshot: CatalogFixtures.snapshot.labelled(.live))
        let defaults = UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!
        let model = AppModel.live(
            userDefaults: defaults,
            catalogService: ModelCatalog(live: source, bundled: empty, cache: cache),
            catalogCache: cache,
            conversationFactory: { .ephemeral() },
            downloadServicesFactory: { _ in .unavailable("test download service") },
            storageFactory: {
                StorageManager(bookmarkLedger: InMemoryBookmarkLedger())
            },
            startDiscovery: false)

        XCTAssertEqual(model.snapshot.origin, .cached)
        XCTAssertFalse(model.entries.isEmpty)
        XCTAssertFalse(model.isAwaitingInitialCatalog)
        for _ in 0..<20 { await Task.yield() }
        XCTAssertEqual(source.fetchCount, 0)
        XCTAssertEqual(model.snapshot.origin, .cached)
    }

    @MainActor
    func testARecognizedLocalCopyThatOutgrewTheSavedTreeRequestsMetadataRefresh() throws {
        let descriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let cached = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .cached, models: [descriptor])
        let artifact = DiscoveredArtifact(
            rootPath: "/Volumes/Fixture/K3", locationPath: "/Volumes/Fixture",
            index: .unreadable, model: .kimiK3, displayName: "Kimi K3",
            bytesOnDisk: descriptor.totalBytes + 1,
            fileCount: descriptor.totalFileCount + 1,
            expectedFileCount: descriptor.totalFileCount,
            expectedBytes: descriptor.totalBytes,
            verification: .unverified, verifiedAt: nil, scannedAt: Date())
        let report = DiscoveryReport(
            locations: [
                LocationScan(
                    rootPath: "/Volumes/Fixture", displayName: "Fixture",
                    isMounted: true, storageKey: nil, artifacts: [artifact], unreadable: [:],
                    scannedAt: Date())
            ],
            scannedAt: Date())

        XCTAssertTrue(
            AppModel.cachedCatalogNeedsRefresh(snapshot: cached, report: report))
        XCTAssertFalse(
            AppModel.cachedCatalogNeedsRefresh(
                snapshot: cached.labelled(.live), report: report),
            "a current live response must not loop merely because a folder has an extra file")

        let incomplete = DiscoveredArtifact(
            rootPath: artifact.rootPath, locationPath: artifact.locationPath,
            index: artifact.index, model: artifact.model, displayName: artifact.displayName,
            bytesOnDisk: descriptor.totalBytes - 1,
            fileCount: descriptor.totalFileCount - 1,
            expectedFileCount: descriptor.totalFileCount,
            expectedBytes: descriptor.totalBytes,
            verification: .unverified, verifiedAt: nil, scannedAt: Date())
        let incompleteReport = DiscoveryReport(
            locations: [
                LocationScan(
                    rootPath: "/Volumes/Fixture", displayName: "Fixture",
                    isMounted: true, storageKey: nil, artifacts: [incomplete], unreadable: [:],
                    scannedAt: Date())
            ],
            scannedAt: Date())
        XCTAssertFalse(
            AppModel.cachedCatalogNeedsRefresh(snapshot: cached, report: incompleteReport),
            "a partial transfer is not evidence that the remote catalog changed")
    }

    @MainActor
    func testAListedNewCommitDoesNotReplaceMatchingCachedLocalAuthority() throws {
        let descriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let cached = ModelCatalogSnapshot(
            generatedAt: .distantPast, origin: .cached, models: [descriptor])
        let latest = PublishedModelReference(
            id: .kimiK3, displayName: descriptor.displayName,
            repository: HuggingFaceRepoRef(
                repoID: try XCTUnwrap(descriptor.source.repo?.repoID),
                revision: String(repeating: "f", count: 40)))
        XCTAssertNotEqual(latest.repository, descriptor.source.repo)
        let artifact = DiscoveredArtifact(
            rootPath: "/Volumes/Fixture/K3", locationPath: "/Volumes/Fixture",
            index: .unreadable, model: .kimiK3, displayName: descriptor.displayName,
            bytesOnDisk: descriptor.totalBytes, fileCount: descriptor.totalFileCount,
            expectedFileCount: descriptor.totalFileCount,
            expectedBytes: descriptor.totalBytes,
            verification: .fullyVerified, verifiedAt: Date(), scannedAt: Date())
        let report = DiscoveryReport(
            locations: [
                LocationScan(
                    rootPath: "/Volumes/Fixture", displayName: "Fixture",
                    isMounted: true, storageKey: nil, artifacts: [artifact], unreadable: [:],
                    scannedAt: Date())
            ],
            scannedAt: Date())

        XCTAssertFalse(
            AppModel.publishedReferenceNeedsResolution(
                latest, snapshot: cached, report: report),
            "a list refresh must not turn a verified offline copy back into unverified")

        let empty = ModelCatalogSnapshot(
            generatedAt: .distantPast, origin: .bundled, models: [])
        XCTAssertTrue(
            AppModel.publishedReferenceNeedsResolution(
                latest, snapshot: empty, report: report),
            "a local artifact needs one tree resolution when no descriptor is cached")
    }

    /// 2026-08-15: a launch with no saved catalogue used an empty snapshot and
    /// scanned under it. An empty snapshot has a descriptor for nothing, so
    /// every verification lookup reached the ledger with no repository — which
    /// the ledger read as a mismatch and answered by deleting complete evidence
    /// for 1.7 TB of artifacts. The bundled catalogue is the recognition table
    /// this build was compiled with, and is what a first launch must scan under.
    @MainActor
    func testLiveLaunchWithoutASavedCatalogScansUnderTheBundledRecognitionTable() throws {
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let cache = EmptyCatalogCache()
        let source = CatalogFetchSpy(snapshot: CatalogFixtures.snapshot.labelled(.live))
        let model = AppModel.live(
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            catalogService: ModelCatalog(live: source, bundled: empty, cache: cache),
            catalogCache: cache,
            conversationFactory: { .ephemeral() },
            downloadServicesFactory: { _ in .unavailable("test download service") },
            storageFactory: {
                StorageManager(bookmarkLedger: InMemoryBookmarkLedger())
            },
            startDiscovery: false)

        XCTAssertEqual(model.snapshot.origin, .bundled)
        XCTAssertFalse(
            model.snapshot.models.isEmpty,
            "a launch must never hold a catalogue that can recognize nothing")
        XCTAssertEqual(
            model.entries.map(\.id),
            AppCatalogAdapter.entries(
                from: ProductCatalogPolicy.applying(to: ModelCatalog.bundled)).map(\.id))
        XCTAssertEqual(model.assessments.catalogModelIDs, model.snapshot.models.map(\.id))

        // Recognition is not publication. Neither surface may show a row until
        // a live response has actually arrived.
        XCTAssertTrue(model.catalogEntriesForPresentation.isEmpty)
        XCTAssertTrue(model.publishedModelsForPresentation.isEmpty)
        XCTAssertEqual(model.publishedModelsOrigin, .bundled)
        XCTAssertTrue(model.isAwaitingInitialCatalog)
    }

    /// The invariant behind the fix above, at the seam that would have to break
    /// for it to regress: no scan may run against a snapshot with zero models.
    @MainActor
    func testDiscoveryNeverStartsUnderACatalogThatCanRecognizeNothing() async throws {
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let bundled = ProductCatalogPolicy.applying(to: ModelCatalog.bundled)

        let underEmpty = ScanCatalogRecorder()
        let emptyModel = makeDiscoveryLaunchModel(snapshot: empty, recorder: underEmpty)
        for _ in 0..<50 { await Task.yield() }
        XCTAssertEqual(
            underEmpty.count, 0,
            "a scan under an empty catalogue asks the ledger about every artifact "
                + "on the volume with no repository at all")
        XCTAssertEqual(emptyModel.installed.scanCount, 0)

        let underBundled = ScanCatalogRecorder()
        let bundledModel = makeDiscoveryLaunchModel(snapshot: bundled, recorder: underBundled)
        for _ in 0..<200 where underBundled.count == 0 {
            try await Task.sleep(for: .milliseconds(10))
        }
        // A mount event can legitimately add a pass; one is the floor.
        XCTAssertGreaterThanOrEqual(underBundled.count, 1)
        XCTAssertEqual(underBundled.catalogs.first?.models.map(\.id), bundled.models.map(\.id))
        XCTAssertGreaterThanOrEqual(bundledModel.installed.scanCount, 1)
    }

    @MainActor
    private func makeDiscoveryLaunchModel(
        snapshot: ModelCatalogSnapshot, recorder: ScanCatalogRecorder
    ) -> AppModel {
        let storage = StorageManager(bookmarkLedger: InMemoryBookmarkLedger())
        let installed = InstalledModels(
            storage: storage, catalog: snapshot, ledger: InMemoryVerificationLedger(),
            scanOperation: { _, catalog, _ in
                recorder.record(catalog)
                return .empty
            })
        return AppModel(
            catalogSnapshot: snapshot,
            catalogService: ModelCatalog(bundled: snapshot, cache: nil),
            allowsLiveCatalogRefresh: false,
            store: .ephemeral(),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            persistsDefaults: false,
            storage: storage,
            downloadServices: .unavailable("test download service"),
            installed: installed,
            assessments: VolumeAssessments(
                storage: storage, catalog: snapshot, ledger: InMemoryAssessmentLedger()),
            seedRecordedRuns: false, startDiscovery: true)
    }

    @MainActor
    func testARefreshFailureShowsNoBundledRowsWhenNoCacheExists() async {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunCatalogFailure-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let failure = CatalogError.transport(status: 503, url: "https://huggingface.co/api/models")
        let model = AppModel(
            catalogSnapshot: empty,
            catalogService: ModelCatalog(
                live: FailingCatalogSource(failure: failure), bundled: empty, cache: nil),
            catalogCache: EmptyCatalogCache(),
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: []),
            seedRecordedRuns: false,
            startDiscovery: false)

        await model.refreshCatalog()

        XCTAssertEqual(model.catalogError, failure)
        XCTAssertFalse(model.isAwaitingInitialCatalog)
        XCTAssertTrue(model.entries.isEmpty)
        XCTAssertTrue(model.catalogEntriesForPresentation.isEmpty)
    }

    @MainActor
    func testARefreshFailureRejectsAnOldBundledSnapshotInTheCacheSlot() async {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunBundledCacheRefusal-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let failure = CatalogError.transport(status: 503, url: "https://huggingface.co/api/models")
        let model = AppModel(
            catalogSnapshot: empty,
            catalogService: ModelCatalog(
                live: FailingCatalogSource(failure: failure), bundled: empty, cache: nil),
            catalogCache: StaticCatalogCache(snapshot: ModelCatalog.bundled),
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: []),
            seedRecordedRuns: false,
            startDiscovery: false)

        await model.refreshCatalog()

        XCTAssertEqual(model.catalogError, failure)
        XCTAssertEqual(model.snapshot, empty)
        XCTAssertTrue(model.entries.isEmpty)
        XCTAssertTrue(model.catalogEntriesForPresentation.isEmpty)
    }

    @MainActor
    func testARefreshFailureUsesOnlyAPreviouslySuccessfulHuggingFaceCache() async throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunCatalogCache-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let cachedDescriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let cached = ModelCatalogSnapshot(
            generatedAt: Date(timeIntervalSince1970: 123), origin: .live,
            models: [cachedDescriptor])
        let failure = CatalogError.transport(status: 503, url: "https://huggingface.co/api/models")
        let model = AppModel(
            catalogSnapshot: empty,
            catalogService: ModelCatalog(
                live: FailingCatalogSource(failure: failure), bundled: empty, cache: nil),
            catalogCache: StaticCatalogCache(snapshot: cached),
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: []),
            seedRecordedRuns: false,
            startDiscovery: false)

        await model.refreshCatalog()

        XCTAssertEqual(model.catalogError, failure)
        XCTAssertEqual(model.snapshot.origin, .cached)
        XCTAssertEqual(model.catalogEntriesForPresentation.map(\.id), [.kimiK3])
    }

    @MainActor
    func testLiveCatalogRemainsVisibleWhenOfflineCacheSaveFails() async {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunCatalogSaveFailure-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let live = CatalogFixtures.snapshot.labelled(.live)
        let cache = FailingSaveCatalogCache()
        let model = AppModel(
            catalogSnapshot: empty,
            catalogService: ModelCatalog(
                live: StaticCatalogSource(snapshot: live), bundled: empty, cache: cache),
            catalogCache: cache,
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: []),
            seedRecordedRuns: false, startDiscovery: false)

        await model.refreshCatalog()

        XCTAssertEqual(model.snapshot.origin, .live)
        XCTAssertFalse(model.catalogEntriesForPresentation.isEmpty)
        XCTAssertNil(model.catalogError)
        XCTAssertTrue(model.catalogCacheWarning?.contains("offline cache refused") == true)
    }

    @MainActor
    func testSuccessfulOfflineCacheSaveClearsAnEarlierWarning() async {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(
                "MinirunCatalogSaveRecovery-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let live = CatalogFixtures.snapshot.labelled(.live)
        let cache = ToggleCatalogCache(failuresRemaining: 1)
        let model = AppModel(
            catalogSnapshot: empty,
            catalogService: ModelCatalog(
                live: StaticCatalogSource(snapshot: live), bundled: empty, cache: cache),
            catalogCache: cache,
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: []),
            seedRecordedRuns: false, startDiscovery: false)

        await model.refreshCatalog()
        XCTAssertNotNil(model.catalogCacheWarning)

        await model.refreshCatalog()

        XCTAssertNil(model.catalogCacheWarning)
        XCTAssertEqual(model.snapshot.origin, .live)
        XCTAssertFalse(model.catalogEntriesForPresentation.isEmpty)
    }

    @MainActor
    func testLiveFailureClearsAStaleCacheSaveWarning() async {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(
                "MinirunCatalogWarningThenFailure-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let cache = FailingSaveCatalogCache()
        let source = ToggleCatalogSource(snapshot: CatalogFixtures.snapshot.labelled(.live))
        let model = AppModel(
            catalogSnapshot: empty,
            catalogService: ModelCatalog(live: source, bundled: empty, cache: cache),
            catalogCache: cache,
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: []),
            seedRecordedRuns: false, startDiscovery: false)

        await model.refreshCatalog()
        XCTAssertNotNil(model.catalogCacheWarning)
        await source.failNextFetch()

        await model.refreshCatalog()

        XCTAssertNil(model.catalogCacheWarning)
        XCTAssertNotNil(model.catalogError)
    }

    @MainActor
    func testCachedCatalogDropsLocallyBuiltModels() async {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunCatalogHFOnly-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let local = locallyBuiltDescriptor()
        // A cache written by a build that still carried a locally built entry.
        let cached = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .live,
            models: ModelCatalog.bundled.models + [local])
        let failure = CatalogError.transport(status: 503, url: "https://huggingface.co/api/models")
        let model = AppModel(
            catalogSnapshot: empty,
            catalogService: ModelCatalog(
                live: FailingCatalogSource(failure: failure), bundled: empty, cache: nil),
            catalogCache: StaticCatalogCache(snapshot: cached),
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: []),
            seedRecordedRuns: false,
            startDiscovery: false)

        await model.refreshCatalog()

        XCTAssertFalse(model.entries.isEmpty, "the cached Hugging Face rows must survive")
        XCTAssertFalse(model.entries.contains { $0.descriptor.source.repo == nil })
        XCTAssertFalse(model.catalogEntriesForPresentation.contains { $0.id == local.id })
    }

    @MainActor
    func testCatalogRefreshPreservesAnExistingDownloadControllerAndItsState() async throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunRefreshTest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let service = ModelCatalog(
            live: MockCatalogSource(latency: .zero),
            bundled: CatalogFixtures.snapshot,
            cache: nil)
        let model = AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            catalogService: service,
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: CatalogFixtures.all),
            installed: AppModel.previewInstalledModels(
                model: .kimiK3, snapshot: CatalogFixtures.snapshot),
            seedRecordedRuns: false,
            startDiscovery: false)
        let before = try XCTUnwrap(model.download(.kimiK3))
        before.markReadyForPreview()

        await model.refreshCatalog()

        let after = try XCTUnwrap(model.download(.kimiK3))
        XCTAssertTrue(before === after)
        XCTAssertTrue(after.state.isReady)
        XCTAssertFalse(after.catalogEntryIsStale)
    }

    @MainActor
    func testARemovedCatalogRowKeepsItsPlannedControllerAndMarksItStale() async throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunRemovedCatalogTest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let withoutK3 = ModelCatalogSnapshot(
            generatedAt: Date(),
            origin: .live,
            models: CatalogFixtures.snapshot.models.filter { $0.id != .kimiK3 })
        let service = ModelCatalog(
            live: StaticCatalogSource(snapshot: withoutK3),
            bundled: CatalogFixtures.snapshot,
            cache: nil)
        let model = AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            catalogService: service,
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: CatalogFixtures.all),
            installed: AppModel.previewInstalledModels(
                model: .kimiK3, snapshot: CatalogFixtures.snapshot),
            seedRecordedRuns: false,
            startDiscovery: false)
        let before = try XCTUnwrap(model.download(.kimiK3))
        before.markReadyForPreview()

        await model.refreshCatalog()

        let after = try XCTUnwrap(model.download(.kimiK3))
        XCTAssertTrue(before === after)
        XCTAssertTrue(after.state.isReady)
        XCTAssertTrue(after.catalogEntryIsStale)
        XCTAssertNil(model.entry(.kimiK3))
    }

    @MainActor
    func testARemovedUntouchedCatalogRowDoesNotLeaveAStartableStaleController() async {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunRemovedIdleCatalogTest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let withoutK3 = ModelCatalogSnapshot(
            generatedAt: Date(),
            origin: .live,
            models: CatalogFixtures.snapshot.models.filter { $0.id != .kimiK3 })
        let model = AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            catalogService: ModelCatalog(
                live: StaticCatalogSource(snapshot: withoutK3),
                bundled: CatalogFixtures.snapshot,
                cache: nil),
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: CatalogFixtures.all),
            seedRecordedRuns: false,
            startDiscovery: false)
        XCTAssertNotNil(model.download(.kimiK3))

        await model.refreshCatalog()

        XCTAssertNil(model.download(.kimiK3))
    }

    /// A descriptor with no Hugging Face repository behind it. No published
    /// model is built this way today, but the storefront policy that drops one
    /// still needs something to drop.
    private func locallyBuiltDescriptor(
        id: ModelID = ModelID("locally-built-model")
    ) -> ModelDescriptor {
        ModelDescriptor(
            id: id,
            displayName: "Locally Built",
            architecture: .unrecognized,
            layout: .unrecognized,
            source: .locallyBuilt(tool: "Tools/local_containers"),
            payloadBytes: 0, metadataBytes: 0, payloadFileCount: 0,
            metadataFileCount: 0, largestFileBytes: 0, minimumBudgetBytes: nil,
            runner: .decodeRunner, licenseName: "test",
            licenseAcknowledgementRequired: false, notes: [])
    }
}

private extension Array {
    var only: Element? { count == 1 ? first : nil }
}

private struct StaticCatalogSource: ModelCatalogSource {
    let snapshot: ModelCatalogSnapshot
    func fetch() async throws -> ModelCatalogSnapshot { snapshot }
}

private struct FailingCatalogSource: ModelCatalogSource {
    let failure: CatalogError
    func fetch() async throws -> ModelCatalogSnapshot { throw failure }
}

/// Captures the catalogue each discovery scan actually ran against.
private final class ScanCatalogRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [ModelCatalogSnapshot] = []

    func record(_ snapshot: ModelCatalogSnapshot) {
        lock.lock()
        stored.append(snapshot)
        lock.unlock()
    }

    var catalogs: [ModelCatalogSnapshot] {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }

    var count: Int { catalogs.count }
}

private struct EmptyCatalogCache: CatalogCache {
    func load() throws -> ModelCatalogSnapshot? { nil }
    func save(_ snapshot: ModelCatalogSnapshot) throws {}
}

private struct StaticCatalogCache: CatalogCache {
    let snapshot: ModelCatalogSnapshot
    func load() throws -> ModelCatalogSnapshot? { snapshot }
    func save(_ snapshot: ModelCatalogSnapshot) throws {}
}

private actor ToggleCatalogSource: ModelCatalogSource {
    let snapshot: ModelCatalogSnapshot
    private var shouldFail = false

    init(snapshot: ModelCatalogSnapshot) { self.snapshot = snapshot }
    func failNextFetch() { shouldFail = true }
    func fetch() async throws -> ModelCatalogSnapshot {
        if shouldFail {
            shouldFail = false
            throw CatalogError.transport(status: 503, url: "https://catalog.invalid")
        }
        return snapshot
    }
}

private struct CatalogCacheSaveFailure: Error, CustomStringConvertible {
    var description: String { "offline cache refused" }
}

private struct FailingSaveCatalogCache: CatalogCache {
    func load() throws -> ModelCatalogSnapshot? { nil }
    func save(_ snapshot: ModelCatalogSnapshot) throws { throw CatalogCacheSaveFailure() }
}

private final class ToggleCatalogCache: CatalogCache, @unchecked Sendable {
    private let lock = NSLock()
    private var failuresRemaining: Int
    private var snapshot: ModelCatalogSnapshot?

    init(failuresRemaining: Int) { self.failuresRemaining = failuresRemaining }

    func load() throws -> ModelCatalogSnapshot? {
        lock.lock()
        defer { lock.unlock() }
        return snapshot
    }

    func save(_ snapshot: ModelCatalogSnapshot) throws {
        lock.lock()
        defer { lock.unlock() }
        if failuresRemaining > 0 {
            failuresRemaining -= 1
            throw CatalogCacheSaveFailure()
        }
        self.snapshot = snapshot
    }
}

private final class CatalogFetchSpy: PublishedModelSource, @unchecked Sendable {
    private let lock = NSLock()
    private let snapshot: ModelCatalogSnapshot
    private var count = 0
    private var lists = 0

    init(snapshot: ModelCatalogSnapshot) { self.snapshot = snapshot }

    var fetchCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    var listCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return lists
    }

    func fetch() async throws -> ModelCatalogSnapshot {
        incrementFetch()
        return snapshot
    }

    func listPublishedModels() async throws -> [PublishedModelReference] {
        incrementList()
        return snapshot.models.compactMap { descriptor in
            guard let repository = descriptor.source.repo else { return nil }
            return PublishedModelReference(
                id: descriptor.id, displayName: descriptor.displayName,
                repository: repository)
        }
    }

    func resolvePublishedModel(_ reference: PublishedModelReference) async throws
        -> ModelDescriptor
    {
        guard let descriptor = snapshot.models.first(where: {
            $0.id == reference.id && $0.source.repo == reference.repository
        }) else { throw CatalogError.unknownModel(reference.id) }
        return descriptor
    }

    private func incrementFetch() {
        lock.lock()
        count += 1
        lock.unlock()
    }

    private func incrementList() {
        lock.lock()
        lists += 1
        lock.unlock()
    }
}

private final class CatalogCacheSpy: CatalogCache, @unchecked Sendable {
    private let lock = NSLock()
    private var saves = 0

    var saveCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return saves
    }

    func load() throws -> ModelCatalogSnapshot? { nil }
    func save(_ snapshot: ModelCatalogSnapshot) throws {
        lock.lock()
        saves += 1
        lock.unlock()
    }
}

private actor CatalogFetchGate {
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

private final class BlockingCatalogSource: PublishedModelSource, @unchecked Sendable {
    private let lock = NSLock()
    private let snapshot: ModelCatalogSnapshot
    private let gate: CatalogFetchGate
    private var count = 0
    private var lists = 0

    init(snapshot: ModelCatalogSnapshot, gate: CatalogFetchGate) {
        self.snapshot = snapshot
        self.gate = gate
    }

    var fetchCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    var listCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return lists
    }

    func fetch() async throws -> ModelCatalogSnapshot {
        increment()
        return snapshot
    }

    func listPublishedModels() async throws -> [PublishedModelReference] {
        incrementList()
        await gate.enterAndWait()
        return snapshot.models.compactMap { descriptor in
            guard let repository = descriptor.source.repo else { return nil }
            return PublishedModelReference(
                id: descriptor.id, displayName: descriptor.displayName,
                repository: repository)
        }
    }

    func resolvePublishedModel(_ reference: PublishedModelReference) async throws
        -> ModelDescriptor
    {
        guard let descriptor = snapshot.models.first(where: {
            $0.id == reference.id && $0.source.repo == reference.repository
        }) else { throw CatalogError.unknownModel(reference.id) }
        return descriptor
    }

    private func increment() {
        lock.lock()
        count += 1
        lock.unlock()
    }

    private func incrementList() {
        lock.lock()
        lists += 1
        lock.unlock()
    }
}
