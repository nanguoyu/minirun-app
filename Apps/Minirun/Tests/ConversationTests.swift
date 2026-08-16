import Darwin
import Foundation
import MinirunRunners
import StorageCore
import XCTest

@testable import MinirunKit
@testable import MinirunApp

/// The conversation layer: what is written to disk, what a title is derived
/// from, and — the one that keeps the settings split honest — that a default is
/// copied once and never consulted again.
final class ConversationStoreTests: XCTestCase {

    private var directory: URL!
    private var store: ConversationStore!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunStoreTests-\(UUID().uuidString)", isDirectory: true)
        store = ConversationStore(directory: directory)
        try store.createDirectoryIfNeeded()
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func sample(title: String = "A chat") -> Conversation {
        var conversation = Conversation(
            title: title,
            settings: ConversationSettings(
                model: .kimiK3, memoryBudgetBytes: 5_800_000_000, maximumNewTokens: 1,
                deterministicReadAheadLayers: 1, telemetryDensity: .cockpit))
        conversation.append(Message(role: .user, text: "北京是中国的首都。请补全下一个词。"))
        conversation.append(
            Message(
                role: .assistant, text: "北京",
                telemetry: MessageTelemetry(
                    wallSeconds: 357.4, tokensPerSecond: 0.0028,
                    bytesPerToken: 188_600_000_000, peakFootprintBytes: 5_261_334_528,
                    declaredBudgetBytes: 5_800_000_000, budgetRespected: true,
                    logitsDigest: CatalogFixtures.deviceLogitsDigest, thermalState: "nominal"),
                tokenIDs: [3372]))
        return conversation
    }

    func testARoundTripKeepsEveryFieldTheTranscriptClaims() throws {
        let original = sample()
        try store.save(original)

        let loaded = store.load()
        XCTAssertTrue(loaded.failures.isEmpty, "nothing should have failed to decode")
        XCTAssertEqual(loaded.conversations.count, 1)

        let restored = try XCTUnwrap(loaded.conversations.first)
        XCTAssertEqual(restored.id, original.id)
        XCTAssertEqual(restored.title, original.title)
        XCTAssertEqual(restored.messages.count, 2)
        XCTAssertEqual(restored.messages[1].tokenIDs, [3372])
        XCTAssertEqual(restored.settings, original.settings)
        XCTAssertEqual(
            restored.messages[1].telemetry?.peakFootprintBytes, 5_261_334_528,
            "a byte count that changed in a round trip would make every receipt suspect")
        XCTAssertEqual(
            restored.updatedAt.timeIntervalSince1970,
            original.updatedAt.timeIntervalSince1970, accuracy: 1)
    }

    /// A turn's attribution is computed on the device that ran it and is gone
    /// when the run is. If the transcript does not keep it, the only way back to
    /// it is another 220-second token on the phone.
    func testATurnKeepsWhereItsTimeWentAcrossARoundTrip() throws {
        let prefill = RunPhaseSummary(
            passKind: .prefill, passIndex: 0, passSeconds: 220,
            terms: [
                RunPhaseTerm(name: RunPhaseTermName.expertIOWait, seconds: 118, count: 13_248),
                RunPhaseTerm(name: RunPhaseTermName.expertGatherCompute, seconds: 31),
                RunPhaseTerm(
                    name: RunPhaseTermName.stagedRead, seconds: 96, isInsidePass: false),
                RunPhaseTerm(name: "aTermThisBuildDoesNotKnow", seconds: 4),
            ])
        let decode = RunPhaseSummary(
            passKind: .decode, passIndex: 1, passSeconds: 214,
            terms: [RunPhaseTerm(name: RunPhaseTermName.expertIOWait, seconds: 110)])
        var conversation = sample()
        conversation.messages[1].telemetry?.phases = [prefill, decode]
        conversation.messages[1].telemetry?.prefillPhase = prefill
        conversation.messages[1].telemetry?.decodePhaseAggregate =
            RunPhaseSummary.aggregate([decode])
        try store.save(conversation)

        let restored = try XCTUnwrap(store.load().conversations.first)
        let telemetry = try XCTUnwrap(restored.messages[1].telemetry)
        XCTAssertEqual(telemetry.phases, [prefill, decode])
        XCTAssertEqual(telemetry.prefillPhase, prefill)
        XCTAssertEqual(telemetry.decodePhaseAggregate?.passCount, 1)
        XCTAssertEqual(
            telemetry.prefillPhase?.seconds(of: "aTermThisBuildDoesNotKnow"), 4,
            "a term this build cannot name is still this run's measurement")
        XCTAssertEqual(
            try XCTUnwrap(telemetry.prefillPhase).attributedSeconds, 153, accuracy: 1e-9,
            "the staged read overlaps the pass and is not part of it")
        XCTAssertEqual(
            try XCTUnwrap(telemetry.prefillPhase).unattributedSeconds, 67, accuracy: 1e-9)
    }

    /// Every turn recorded before the engines could publish a decomposition.
    func testATurnWrittenBeforeThePhaseSplitDecodesWithNone() throws {
        let current = Message(
            role: .assistant, text: "北京",
            telemetry: MessageTelemetry(
                wallSeconds: 357.4, tokensPerSecond: nil, bytesPerToken: nil,
                peakFootprintBytes: 5_261_334_528, declaredBudgetBytes: 5_800_000_000,
                budgetRespected: true))
        let encoded = try JSONEncoder().encode(current)
        var legacy = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        var telemetry = try XCTUnwrap(legacy["telemetry"] as? [String: Any])
        for key in ["phases", "prefillPhase", "decodePhaseAggregate"] {
            telemetry.removeValue(forKey: key)
        }
        legacy["telemetry"] = telemetry

        let decoded = try JSONDecoder().decode(
            Message.self, from: JSONSerialization.data(withJSONObject: legacy))
        XCTAssertNil(decoded.telemetry?.phases)
        XCTAssertNil(decoded.telemetry?.prefillPhase)
        XCTAssertNil(decoded.telemetry?.decodePhaseAggregate)
        XCTAssertEqual(decoded.telemetry?.peakFootprintBytes, 5_261_334_528)
    }

    func testAMessageWrittenBeforeStructuredRecoveryStillDecodes() throws {
        let current = Message(
            role: .assistant, text: "", namedError: "old exact error",
            failure: .verifyModelFiles)
        let encoded = try JSONEncoder().encode(current)
        var legacy = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        legacy.removeValue(forKey: "failure")

        let decoded = try JSONDecoder().decode(
            Message.self, from: JSONSerialization.data(withJSONObject: legacy))

        XCTAssertEqual(decoded.namedError, "old exact error")
        XCTAssertNil(decoded.failure)
    }

    func testEphemeralStoreNeverCreatesOrReadsAConversationFile() throws {
        let ephemeral = ConversationStore.ephemeral()
        let conversation = sample(title: "visual review only")

        XCTAssertFalse(ephemeral.isPersistent)
        XCTAssertFalse(FileManager.default.fileExists(atPath: ephemeral.directory.path))

        try ephemeral.createDirectoryIfNeeded()
        try ephemeral.save(conversation)
        XCTAssertFalse(FileManager.default.fileExists(atPath: ephemeral.directory.path))
        XCTAssertEqual(ephemeral.load(), .init(conversations: [], failures: []))

        try ephemeral.delete(conversation.id)
        XCTAssertFalse(FileManager.default.fileExists(atPath: ephemeral.directory.path))
    }

    func testTheDocumentIsReadableJSONWithTheBudgetInIt() throws {
        let conversation = sample()
        try store.save(conversation)
        let text = try String(contentsOf: store.fileURL(for: conversation.id), encoding: .utf8)
        XCTAssertTrue(text.contains("5800000000"), "the stated budget is in the file, in plain")
        XCTAssertTrue(text.contains("\"model\""))
    }

    func testConversationsComeBackNewestFirst() throws {
        var older = sample()
        older.rename(to: "older")
        older.updatedAt = Date(timeIntervalSince1970: 1_000_000)
        var newer = sample()
        newer.rename(to: "newer")
        newer.updatedAt = Date(timeIntervalSince1970: 2_000_000)
        try store.save(older)
        try store.save(newer)
        XCTAssertEqual(store.load().conversations.map { $0.title }, ["newer", "older"])
    }

    /// A document that will not decode is REPORTED. Skipping it silently would
    /// lose a transcript without anybody being told.
    func testAnUnreadableDocumentIsReportedRatherThanSwallowed() throws {
        try store.save(sample())
        try "{ not json".write(
            to: directory.appendingPathComponent("broken.json"), atomically: true, encoding: .utf8)
        let loaded = store.load()
        XCTAssertEqual(loaded.conversations.count, 1)
        XCTAssertEqual(loaded.failures.map { $0.filename }, ["broken.json"])
        XCTAssertFalse(loaded.failures[0].reason.isEmpty)
    }

    func testDeleteRemovesTheDocumentAndIsIdempotent() throws {
        let conversation = sample()
        try store.save(conversation)
        try store.delete(conversation.id)
        XCTAssertTrue(store.load().conversations.isEmpty)
        XCTAssertNoThrow(try store.delete(conversation.id))
    }
}

/// Titles: derived from what was actually said, never invented, and never
/// overwritten once a human has named the thing.
final class ConversationTitleTests: XCTestCase {

    func testTheTitleComesFromTheFirstUserMessage() {
        var conversation = Conversation(
            settings: ConversationSettings(model: .kimiK3, memoryBudgetBytes: 5_800_000_000))
        XCTAssertEqual(conversation.title, Conversation.untitled)
        conversation.append(Message(role: .user, text: "What does a stated budget mean?"))
        XCTAssertEqual(conversation.title, "What does a stated budget mean?")
    }

    func testALongTitleIsCutAtAWordBoundaryAndSaysItWasCut() {
        let text =
            "Explain exactly why widening the widest deterministic layer to float32 doubles it"
        let title = Conversation.derivedTitle(from: text)
        XCTAssertTrue(title.hasSuffix("…"))
        XCTAssertLessThanOrEqual(title.count, 43)
        XCTAssertTrue(text.hasPrefix(title.dropLast()))
    }

    func testWhitespaceAndNewlinesCollapse() {
        XCTAssertEqual(
            Conversation.derivedTitle(from: "  a  question\nover\ttwo lines "),
            "a question over two lines")
        XCTAssertEqual(Conversation.derivedTitle(from: "   \n "), Conversation.untitled)
    }

    func testTextWithNoSpacesIsStillCutRatherThanRunOn() {
        let title = Conversation.derivedTitle(from: String(repeating: "北", count: 90))
        XCTAssertTrue(title.hasSuffix("…"))
        XCTAssertEqual(title.count, 43)
    }

    func testARenamedConversationStopsDerivingItsTitle() {
        var conversation = Conversation(
            settings: ConversationSettings(model: .kimiK3, memoryBudgetBytes: 5_800_000_000))
        conversation.rename(to: "K3 receipts")
        conversation.append(Message(role: .user, text: "something else entirely"))
        XCTAssertEqual(conversation.title, "K3 receipts")
    }

    func testRegenerateDropsTheLastAssistantTurnAndKeepsTheQuestion() {
        var conversation = Conversation(
            settings: ConversationSettings(model: .kimiK3, memoryBudgetBytes: 5_800_000_000))
        conversation.append(Message(role: .user, text: "first"))
        conversation.append(Message(role: .assistant, text: "北京"))
        let question = conversation.removeLastAssistantTurn()
        XCTAssertEqual(question?.text, "first")
        XCTAssertEqual(conversation.messages.count, 1)
        XCTAssertTrue(conversation.awaitsAnswer)
    }

    func testRelaunchInterruptionAddsNoOutputAndIsRetryableExactlyOnce() throws {
        let recoveredAt = Date(timeIntervalSince1970: 2_000_000)
        var conversation = Conversation(
            settings: ConversationSettings(
                model: .kimiK3, memoryBudgetBytes: 8_000_000_000))
        conversation.append(Message(role: .user, text: "answer me"))

        XCTAssertTrue(conversation.recordInterruptedReplyIfNeeded(at: recoveredAt))
        XCTAssertFalse(conversation.recordInterruptedReplyIfNeeded(at: recoveredAt))
        XCTAssertEqual(conversation.messages.map(\.role), [.user, .assistant])

        let interruption = try XCTUnwrap(conversation.messages.last)
        XCTAssertEqual(interruption.text, "")
        XCTAssertTrue(interruption.tokenIDs.isEmpty)
        XCTAssertNil(interruption.telemetry)
        XCTAssertEqual(interruption.failure, .interrupted)
        XCTAssertTrue(try XCTUnwrap(interruption.failure).allowsRetry)
        XCTAssertEqual(interruption.createdAt, recoveredAt)
        XCTAssertTrue(try XCTUnwrap(interruption.namedError).contains("previous app session"))
    }
}

final class ConversationPresentationTests: XCTestCase {
    func testChatSettingHelpUsesProductLanguage() {
        for setting in [ReadAheadSetting.auto, .off, .explicit(2)] {
            let copy = setting.explanation.lowercased()
            XCTAssertFalse(copy.contains("spec v"))
            XCTAssertFalse(copy.contains("measurement"))
            XCTAssertFalse(copy.contains("runner"))
            XCTAssertFalse(copy.contains("usb4"))
        }
        XCTAssertTrue(ReadAheadSetting.explicit(2).explanation.contains("Choose Auto or Off"))
    }

    func testInstrumentPreferenceExplainsWhetherThePanelOpens() {
        XCTAssertEqual(TelemetryDensity.strip.label, "Manual")
        XCTAssertEqual(TelemetryDensity.cockpit.label, "Automatic")
        XCTAssertTrue(TelemetryDensity.cockpit.explanation.contains("generation begins"))
    }

    func testLayerCensusFailureUsesProductCopyAndKeepsTechnicalDetailSeparate() {
        let raw =
            "the memory plan's layer census old does not match the opened artifact's layer census new"
        let summary = ConversationErrorCopy.summary(for: raw)

        XCTAssertEqual(
            summary, "The model files changed and this reply could not start.")
        XCTAssertFalse(summary.contains("census"))
        XCTAssertFalse(summary.contains("old"))
    }

    func testStoppedGenerationHasADirectChatMessage() {
        XCTAssertEqual(
            ConversationErrorCopy.summary(for: "cancelled — stopped at a layer boundary"),
            "Generation stopped.")
    }

    func testStructuredRecoveryNeverParsesTheNamedErrorForItsAction() {
        let misleading = "cancelled layer census storage vanished"
        XCTAssertEqual(
            ConversationErrorCopy.summary(for: misleading, failure: .verifyModelFiles),
            "The local model files need to be checked.")
        XCTAssertFalse(MessageFailure.verifyModelFiles.allowsRetry)
        XCTAssertTrue(MessageFailure.retryAfterStorageReturns.allowsRetry)
    }
}

/// The settings split, which is the point of this milestone: a global default
/// is a starting value, and a conversation owns what it runs at.
@MainActor
final class ConversationSettingsInheritanceTests: XCTestCase {

    func testSettingsReturnsToTheConversationThatOpenedIt() throws {
        let model = AppModel.freshForTests()
        let first = try XCTUnwrap(model.newConversation())
        _ = try XCTUnwrap(model.newConversation())
        model.destination = .conversation(first)

        model.openSettings(.models)
        model.backToApp()

        XCTAssertEqual(model.destination, .conversation(first))
    }

    func testAppModelOwnsSettingsAndConversationNavigationPaths() throws {
        let model = AppModel.freshForTests()
        let id = try XCTUnwrap(model.newConversation())

        model.openSettingsRoot()
        XCTAssertEqual(model.destination, .settings)
        XCTAssertTrue(model.settingsNavigationPath.isEmpty)

        model.setSettingsNavigationPath([.models])
        XCTAssertEqual(model.settingsSection, .models)
        XCTAssertEqual(model.settingsNavigationPath, [.models])

        model.backToApp()
        XCTAssertEqual(model.destination, .conversation(id))

        model.setConversationNavigationPath([])
        XCTAssertEqual(model.destination, .chats)
        model.setConversationNavigationPath([id])
        XCTAssertEqual(model.destination, .conversation(id))

        let conversationMirror = model.sceneNavigationMirror
        model.setConversationNavigationPath([])
        model.restoreNavigation(from: conversationMirror)
        XCTAssertEqual(model.destination, .conversation(id))

        model.restoreNavigation(
            from: AppModel.SceneNavigationMirror(
                place: .conversation, conversationID: UUID(), settingsSection: nil))
        XCTAssertEqual(
            model.destination, .chats,
            "a scene record cannot resurrect a conversation that is no longer durable")
    }

    func testFirstSettingsStackInitializationCannotEraseAHomeShortcut() {
        let model = AppModel.freshForTests()

        model.openSettings(.models)
        XCTAssertEqual(model.settingsNavigationPath, [.models])

        // A lazily created NavigationStack writes its empty initial value back
        // before the destination view appears. This is not a user pressing Back.
        model.setSettingsNavigationPath([])
        XCTAssertEqual(model.settingsNavigationPath, [.models])
        XCTAssertEqual(model.sceneNavigationMirror.settingsSection, .models)

        model.activatePendingSettingsNavigation()
        model.setSettingsNavigationPath([])
        XCTAssertTrue(
            model.settingsNavigationPath.isEmpty,
            "after activation, an ordinary Back gesture must still pop to Settings")
    }

    func testFirstStorageShortcutSurvivesLazyRegularWidthSelection() {
        let model = AppModel.freshForTests()

        model.openSettings(.storage)
        model.setSettingsNavigationPath([])
        model.activatePendingSettingsNavigation()

        XCTAssertEqual(model.destination, .settings)
        XCTAssertEqual(model.settingsNavigationPath, [.storage])
        XCTAssertEqual(model.settingsSection, .storage)
    }

    func testReadyNewChatUsesFallbackWithoutChangingTheDefault() throws {
        let model = AppModel.freshForTests(installedModels: [.deepseekV4Flash])
        model.defaults.model = .kimiK3

        XCTAssertEqual(model.chatReadiness, .ready(model: .deepseekV4Flash))
        XCTAssertNil(model.newConversation(), "the defaults-only API stays strict")
        let id = try XCTUnwrap(model.newReadyConversation())

        XCTAssertEqual(model.conversation(id)?.settings.model, .deepseekV4Flash)
        XCTAssertEqual(model.defaults.model, .kimiK3)
    }

    func testRelaunchAtomicallyRecordsAnUnansweredDurableTurnOnce() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(
                "MinirunInterruptedConversation-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ConversationStore(directory: directory)
        var pending = Conversation(
            settings: ConversationSettings(
                model: .kimiK3, memoryBudgetBytes: 8_000_000_000))
        pending.append(Message(role: .user, text: "persisted before the runner started"))
        try store.save(pending)
        let recoveryDate = Date(timeIntervalSince1970: 3_000_000)

        let firstLaunch = AppModel.freshForTests(store: store, now: recoveryDate)
        let recovered = try XCTUnwrap(firstLaunch.conversation(pending.id))
        XCTAssertEqual(recovered.messages.map(\.role), [.user, .assistant])
        XCTAssertEqual(recovered.messages.last?.failure, .interrupted)
        XCTAssertEqual(recovered.messages.last?.createdAt, recoveryDate)
        XCTAssertEqual(store.load().conversations.first?.messages, recovered.messages)

        let secondLaunch = AppModel.freshForTests(
            store: store, now: recoveryDate.addingTimeInterval(100))
        XCTAssertEqual(secondLaunch.conversation(pending.id)?.messages.count, 2)
        XCTAssertEqual(
            secondLaunch.conversation(pending.id)?.messages.last?.createdAt,
            recoveryDate)
    }

    func testLeavingTheActiveSceneRequestsSafeGenerationStop() async throws {
        let model = AppModel.freshForTests()
        model.transitionScene(to: .active)
        let id = try XCTUnwrap(model.newConversation())
        model.send("keep the process safe", in: id)
        XCTAssertTrue(model.run.isRunning)

        model.transitionScene(to: .inactive)

        XCTAssertEqual(model.sceneState, .inactive)
        XCTAssertTrue(
            model.run.isStopping || !model.run.isRunning,
            "a very fast runner may already have reached its safe cancellation boundary")
        model.transitionScene(to: .background)
        XCTAssertEqual(model.sceneState, .background)

        for _ in 0..<400 where model.hasActiveRun {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertFalse(model.hasActiveRun)
        XCTAssertEqual(model.conversation(id)?.messages.last?.failure, .stopped)
    }

    func testSettingsCanReturnToAnHonestEmptyChatWorkspaceWithoutAModel() {
        let model = AppModel.freshForTests(installedModel: nil)
        XCTAssertFalse(model.canCreateConversation)
        XCTAssertTrue(model.conversations.isEmpty)

        model.openSettings()
        model.backToApp()

        XCTAssertEqual(model.destination, .chats)
        XCTAssertTrue(model.conversations.isEmpty)
        XCTAssertTrue(model.conversationStore.load().conversations.isEmpty)
    }

    func testDeletingTheLastChatReturnsToTheChatsWorkspace() {
        let model = AppModel.freshForTests()
        let id = model.newConversation()!
        model.delete(id)

        XCTAssertEqual(model.destination, .chats)
        XCTAssertTrue(model.conversations.isEmpty)
    }

    func testANewChatCopiesTheDefaultsInForceAtItsCreation() {
        let model = AppModel.freshForTests()
        model.defaults.model = .kimiK3
        model.setDefaultBudget(6_100_000_000, for: .kimiK3)
        model.defaults.telemetryDensity = .cockpit

        let id = model.newConversation()!
        let conversation = model.conversation(id)!
        XCTAssertEqual(conversation.settings.model, .kimiK3)
        XCTAssertEqual(conversation.settings.memoryBudgetBytes, 6_100_000_000)
        XCTAssertEqual(conversation.settings.telemetryDensity, .cockpit)
    }

    func testStartingAChatForOneModelDoesNotChangeTheGlobalDefault() throws {
        let model = AppModel.freshForTests()
        model.defaults.model = .deepseekV4Flash

        let id = try XCTUnwrap(model.newConversation(modelID: .kimiK3))

        XCTAssertEqual(model.conversation(id)?.settings.model, .kimiK3)
        XCTAssertEqual(model.defaults.model, .deepseekV4Flash)
    }

    func testGettingReadyStateIsDerivedFromCurrentProductFacts() {
        let ready = AppModel.freshForTests()
        XCTAssertEqual(ready.chatReadiness, .ready(model: .kimiK3))

        let noStorage = AppModel.freshForTests(installedModel: nil)
        XCTAssertEqual(noStorage.chatReadiness, .needsStorage)

        let needsVerification = AppModel.freshForTests(verification: .unverified)
        XCTAssertEqual(
            needsVerification.chatReadiness,
            .needsVerification(model: .kimiK3))
    }

    func testInstrumentsOpenAutomaticallyByDefaultForNewChats() {
        let model = AppModel.freshForTests()

        XCTAssertEqual(model.defaults.telemetryDensity, .cockpit)
        let id = model.newConversation()!
        XCTAssertEqual(model.conversation(id)?.settings.telemetryDensity, .cockpit)
    }

    /// The invariant this whole milestone rests on: changing a default cannot
    /// change what an existing conversation runs at.
    func testChangingADefaultLeavesAnExistingChatAlone() {
        let model = AppModel.freshForTests()
        model.setDefaultBudget(5_800_000_000, for: .kimiK3)
        let id = model.newConversation()!

        model.setDefaultBudget(9_000_000_000, for: .kimiK3)

        XCTAssertEqual(model.conversation(id)?.settings.memoryBudgetBytes, 5_800_000_000)
        XCTAssertEqual(
            model.makeRequest(for: model.conversation(id)!)?.memoryBudgetBytes, 5_800_000_000)
    }

    /// And the other direction: editing one chat never edits the defaults or
    /// any other chat.
    func testEditingOneChatTouchesNeitherTheDefaultsNorAnotherChat() {
        let model = AppModel.freshForTests()
        model.setDefaultBudget(5_800_000_000, for: .kimiK3)
        let first = model.newConversation()!
        let second = model.newConversation()!

        model.setBudget(7_000_000_000, in: first)

        XCTAssertEqual(model.defaults.budget(for: .kimiK3), 5_800_000_000)
        XCTAssertEqual(model.conversation(second)?.settings.memoryBudgetBytes, 5_800_000_000)
        XCTAssertEqual(model.conversation(first)?.settings.memoryBudgetBytes, 7_000_000_000)
    }

    func testLegacyAutomaticK3BudgetMigratesOnceWithoutRewritingOtherChoices() throws {
        func loadedModel(legacyBudget: UInt64) throws -> (AppModel, UserDefaults) {
            let suite = "minirun.tests.\(UUID().uuidString)"
            let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
            let legacy = NewChatDefaults(
                budgetBytes: [ModelID.kimiK3.rawValue: legacyBudget],
                budgetSchemaVersion: nil)
            defaults.set(
                try JSONEncoder().encode(legacy),
                forKey: "minirun.newChatDefaults")
            return (AppModel.freshForTests(userDefaults: defaults), defaults)
        }

        let (automatic, automaticStore) = try loadedModel(legacyBudget: 5_800_000_000)
        XCTAssertEqual(
            automatic.defaults.budget(for: .kimiK3),
            K3ProductMemoryBudget.minimumBudgetBytes)
        let persisted = try XCTUnwrap(
            automaticStore.data(forKey: "minirun.newChatDefaults"))
        XCTAssertEqual(
            try JSONDecoder().decode(NewChatDefaults.self, from: persisted).budgetSchemaVersion,
            NewChatDefaults.currentBudgetSchemaVersion)

        let (operatorChoice, _) = try loadedModel(legacyBudget: 6_100_000_000)
        XCTAssertEqual(operatorChoice.defaults.budget(for: .kimiK3), 6_100_000_000)
        XCTAssertEqual(
            operatorChoice.defaults.budgetSchemaVersion,
            NewChatDefaults.currentBudgetSchemaVersion)
    }

    func testIOSExperimentalK3PolicyMigratesOnlyTheFormerAutomaticConversationBoundary() {
        var automatic = Conversation(
            settings: ConversationSettings(
                model: .kimiK3, memoryBudgetBytes: 8_000_000_000,
                maximumNewTokens: 64))
        XCTAssertTrue(
            AppModel.migrateK3Conversation(
                &automatic, to: K3ProductMemoryBudget.iOSExperimentalPolicy))
        XCTAssertEqual(automatic.settings.memoryBudgetBytes, 5_800_000_000)
        XCTAssertEqual(automatic.settings.maximumNewTokens, 2)

        var operatorChoice = Conversation(
            settings: ConversationSettings(
                model: .kimiK3, memoryBudgetBytes: 6_100_000_000,
                maximumNewTokens: 64))
        XCTAssertTrue(
            AppModel.migrateK3Conversation(
                &operatorChoice, to: K3ProductMemoryBudget.iOSExperimentalPolicy))
        XCTAssertEqual(operatorChoice.settings.memoryBudgetBytes, 6_100_000_000)
        XCTAssertEqual(operatorChoice.settings.maximumNewTokens, 2)

        var v4 = Conversation(
            settings: ConversationSettings(
                model: .deepseekV4Flash, memoryBudgetBytes: 8_000_000_000,
                maximumNewTokens: 64))
        XCTAssertFalse(
            AppModel.migrateK3Conversation(
                &v4, to: K3ProductMemoryBudget.iOSExperimentalPolicy))
        XCTAssertEqual(v4.settings.memoryBudgetBytes, 8_000_000_000)
        XCTAssertEqual(v4.settings.maximumNewTokens, 64)
    }

    func testLegacyManualInstrumentDefaultMigratesOnceThenPreservesAnExplicitOff() throws {
        let suite = "minirun.instrument-default-tests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }

        let legacy = NewChatDefaults(
            telemetryDensity: .strip,
            instrumentPanelSchemaVersion: nil)
        defaults.set(
            try JSONEncoder().encode(legacy),
            forKey: "minirun.newChatDefaults")

        let migrated = AppModel.freshForTests(userDefaults: defaults)
        XCTAssertEqual(migrated.defaults.telemetryDensity, .cockpit)
        XCTAssertEqual(
            migrated.defaults.instrumentPanelSchemaVersion,
            NewChatDefaults.currentInstrumentPanelSchemaVersion)

        migrated.defaults.telemetryDensity = .strip
        let relaunched = AppModel.freshForTests(userDefaults: defaults)
        XCTAssertEqual(relaunched.defaults.telemetryDensity, .strip)
        XCTAssertEqual(
            relaunched.defaults.instrumentPanelSchemaVersion,
            NewChatDefaults.currentInstrumentPanelSchemaVersion)
    }

    func testAppearanceDefaultsToSystemAndPersistsAnExplicitChoice() throws {
        let suite = "minirun.appearance-tests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }

        let first = AppModel.freshForTests(userDefaults: defaults)
        XCTAssertEqual(first.appearance, .system)

        first.appearance = .dark
        XCTAssertEqual(defaults.string(forKey: "minirun.appearance"), "Dark")

        let relaunched = AppModel.freshForTests(userDefaults: defaults)
        XCTAssertEqual(relaunched.appearance, .dark)
    }

    #if DEBUG
        func testReviewAppearanceCannotReadOrRewriteProductDefaults() throws {
            let suite = "minirun.appearance-review-tests.\(UUID().uuidString)"
            let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
            defer { defaults.removePersistentDomain(forName: suite) }
            defaults.set("Dark", forKey: "minirun.appearance")

            let review = AppModel.preview(userDefaults: defaults)
            XCTAssertEqual(review.appearance, .system)
            review.appearance = .light
            XCTAssertEqual(defaults.string(forKey: "minirun.appearance"), "Dark")
        }
    #endif

    func testPromptSpecificFloorRefusesBeforeMutatingTheConversation() throws {
        let model = AppModel.freshForTests(
            runtimeWorkingSetReserveBytes: { _, _ in 3_500_000_000 })
        let id = try XCTUnwrap(model.newConversation())
        model.setBudget(K3ProductMemoryBudget.minimumBudgetBytes, in: id)

        model.send("a prompt whose prepared state is larger", in: id)

        XCTAssertTrue(try XCTUnwrap(model.conversation(id)).messages.isEmpty)
        XCTAssertFalse(model.run.isRunning)
        let reason = try XCTUnwrap(model.runPreparationError(for: id))
        XCTAssertEqual(reason, "This prompt needs 8.23 GB; this chat states 8.00 GB.")
        XCTAssertEqual(model.runPreparationBudgetTarget(for: id), 8_229_307_136)
        let saved = try XCTUnwrap(model.conversation(id))
        let preparedPlan = try XCTUnwrap(model.budgetPlan(for: saved))
        XCTAssertFalse(preparedPlan.isRunnable)
        XCTAssertEqual(preparedPlan.refusal?.suggestedBudgetBytes, 8_229_307_136)

        model.setBudget(8_229_307_136, in: id)
        XCTAssertNil(model.runPreparationError(for: id))
        XCTAssertNil(model.runPreparationBudgetTarget(for: id))
        XCTAssertTrue(
            try XCTUnwrap(model.budgetPlan(for: XCTUnwrap(model.conversation(id)))).isRunnable)

        model.setDraft("different", in: id)
        XCTAssertEqual(
            try XCTUnwrap(
                model.budgetPlan(for: XCTUnwrap(model.conversation(id)))
            ).profile.workingSetReserveBytes,
            CatalogFixtures.workingSetReserve)
    }

    func testEveryEditIsOnDiskWithoutTheCallSiteRememberingTo() {
        let model = AppModel.freshForTests()
        let id = model.newConversation()!
        model.setBudget(6_500_000_000, in: id)
        model.rename(id, to: "budget experiment")

        let reloaded = model.conversationStore.load().conversations.first { $0.id == id }
        XCTAssertEqual(reloaded?.settings.memoryBudgetBytes, 6_500_000_000)
        XCTAssertEqual(reloaded?.title, "budget experiment")
    }

    func testDeletingAConversationRemovesItsDocument() {
        let model = AppModel.freshForTests()
        let id = model.newConversation()!
        model.delete(id)
        XCTAssertNil(model.conversation(id))
        XCTAssertTrue(model.conversationStore.load().conversations.contains { $0.id == id } == false)
    }

    func testConversationDeletionConfirmationNamesTheChatAndPermanentEffect() throws {
        let model = AppModel.freshForTests()
        let id = try XCTUnwrap(model.newConversation())
        model.rename(id, to: "A careful title")
        let conversation = try XCTUnwrap(model.conversation(id))

        let request = ConversationDeletionRequest(conversation, isRunning: false)

        XCTAssertEqual(request.id, id)
        XCTAssertTrue(request.message.contains("A careful title"))
        XCTAssertTrue(request.message.localizedCaseInsensitiveContains("permanently"))
        XCTAssertTrue(request.message.localizedCaseInsensitiveContains("cannot be undone"))
        XCTAssertTrue(request.message.contains("Model files and other conversations"))
        XCTAssertFalse(request.message.contains("current response"))
    }

    func testRunningConversationDeletionExplainsThatTheResponseStopsSafely() throws {
        let model = AppModel.freshForTests()
        let id = try XCTUnwrap(model.newConversation())
        let conversation = try XCTUnwrap(model.conversation(id))

        let request = ConversationDeletionRequest(conversation, isRunning: true)

        XCTAssertTrue(request.message.contains("current response"))
        XCTAssertTrue(request.message.contains("stopped safely before deletion"))
    }

    /// Only the knobs somebody actually stated are sent. An unset knob is a
    /// different statement from a zero, and the runner refuses by name either
    /// way rather than being handed a default it did not ask for.
    func testOnlyStatedKnobsReachTheRequest() {
        var settings = ConversationSettings(model: .kimiK3, memoryBudgetBytes: 5_800_000_000)
        XCTAssertTrue(settings.knobs.setKnobNames.isEmpty)
        settings.deterministicReadAheadLayers = 1
        XCTAssertEqual(settings.knobs.setKnobNames, ["deterministicReadAheadLayers"])
        XCTAssertTrue(settings.readAheadIsOn)
        settings.deterministicReadAheadLayers = 2
        XCTAssertFalse(settings.readAheadIsOn, "an unsupported saved depth is not a valid on state")
    }

    func testSwitchingModelsRemovesEveryKnobTheNewRunnerCannotHonour() throws {
        let model = AppModel.freshForTests(
            installedModels: [.kimiK3, .deepseekV4Flash])
        let id = try XCTUnwrap(model.newConversation())
        // The K3 ceiling is a platform policy — 64 on the supported Mac, 2 on
        // the bounded iPhone tier (ADR 0011) — so the value this test carries
        // across the two switches is that policy rather than a literal.
        let k3Ceiling = K3ProductMemoryBudget.currentPolicy.maximumNewTokens
        model.update(id) {
            $0.settings.maximumNewTokens = k3Ceiling
            $0.settings.deterministicReadAheadLayers = 1
            $0.settings.expertReadAhead = 2
            $0.settings.expertPoolSlots = 8
        }

        model.selectModel(.deepseekV4Flash, in: id)
        var settings = try XCTUnwrap(model.conversation(id)?.settings)
        XCTAssertEqual(settings.model, .deepseekV4Flash)
        XCTAssertNil(
            settings.deterministicReadAheadLayers,
            "V4 has no deterministic read-ahead knob to attach this to")
        XCTAssertEqual(settings.expertReadAhead, 2, "V4 honours this knob")
        XCTAssertEqual(settings.expertPoolSlots, 8, "V4 honours this knob")
        XCTAssertEqual(settings.maximumNewTokens, k3Ceiling)

        model.selectModel(.kimiK3, in: id)
        settings = try XCTUnwrap(model.conversation(id)?.settings)
        XCTAssertEqual(settings.model, .kimiK3)
        XCTAssertNil(settings.expertPoolSlots, "K3 has no expert-pool knob")
        XCTAssertEqual(settings.expertReadAhead, 2, "K3 honours this one too")
        XCTAssertEqual(
            settings.maximumNewTokens, k3Ceiling,
            "switching models preserves a value the K3 runner can honour")
        XCTAssertEqual(settings.knobs.setKnobNames, ["expertReadAhead"])
    }

    /// A budget nobody personally stated belongs to the model it was derived
    /// from, so changing the model re-derives it. V4's floor is not K3's, and
    /// carrying K3's number across would state a budget about the wrong
    /// artifact.
    func testADerivedBudgetIsReDerivedWhenTheModelChanges() throws {
        let model = AppModel.freshForTests(installedModels: [.kimiK3, .deepseekV4Flash])
        let id = try XCTUnwrap(model.newConversation())
        let k3Default = try XCTUnwrap(model.defaultBudgetPlan(for: .kimiK3)).budgetBytes
        XCTAssertEqual(model.conversation(id)?.settings.memoryBudgetBytes, k3Default)

        model.selectModel(.deepseekV4Flash, in: id)
        let v4Default = try XCTUnwrap(model.defaultBudgetPlan(for: .deepseekV4Flash)).budgetBytes
        XCTAssertEqual(model.conversation(id)?.settings.memoryBudgetBytes, v4Default)
        XCTAssertNotEqual(v4Default, k3Default, "the two models must not share a floor")

        // A preset of the current model counts as derived too.
        let conversation = try XCTUnwrap(model.conversation(id))
        let generous = try XCTUnwrap(model.budgetPlan(for: conversation)).budget(for: .generous)
        model.setBudget(generous, in: id)
        model.selectModel(.kimiK3, in: id)
        XCTAssertEqual(model.conversation(id)?.settings.memoryBudgetBytes, k3Default)
    }

    /// A budget somebody chose is theirs. Changing the model keeps it, and only
    /// moves it when the new model's floor makes the old number impossible to
    /// state.
    func testACustomizedBudgetSurvivesAModelChangeAndIsOnlyMovedWhenImpossible() throws {
        let model = AppModel.freshForTests(installedModels: [.kimiK3, .deepseekV4Flash])
        let id = try XCTUnwrap(model.newConversation())
        let k3Default = try XCTUnwrap(model.defaultBudgetPlan(for: .kimiK3)).budgetBytes
        let customized = k3Default + 3_000_000_000
        model.setBudget(customized, in: id)

        model.selectModel(.deepseekV4Flash, in: id)
        XCTAssertEqual(
            model.conversation(id)?.settings.memoryBudgetBytes, customized,
            "a stated budget the new model can hold is not silently restated")

        // Below the new model's floor it cannot be stated at all, so it is
        // raised to that floor rather than left as a configuration that only
        // ever produces a refusal.
        model.setBudget(1_000_000, in: id)
        model.selectModel(.kimiK3, in: id)
        let k3Floor = try XCTUnwrap(model.defaultBudgetPlan(for: .kimiK3)).profile
            .refusalThresholdBytes
        XCTAssertEqual(model.conversation(id)?.settings.memoryBudgetBytes, k3Floor)
    }

    func testANewChatConstrainsEvenAStalePersistedDefaultToItsRunner() throws {
        let model = AppModel.freshForTests(
            installedModels: [.kimiK3, .deepseekV4Flash])
        // Models from an older build could persist this impossible pair. The
        // creation boundary is the final guard even when a UI setter was not.
        model.defaults.model = .deepseekV4Flash
        model.defaults.deterministicReadAheadLayers = 1

        let id = try XCTUnwrap(model.newConversation())
        let settings = try XCTUnwrap(model.conversation(id)?.settings)
        XCTAssertEqual(settings.model, .deepseekV4Flash)
        XCTAssertNil(settings.deterministicReadAheadLayers)
        XCTAssertTrue(settings.knobs.setKnobNames.isEmpty)
    }

    func testRunSnapshotAndFaultBelongOnlyToTheConversationThatStartedThem() async throws {
        let model = AppModel.freshForTests(runFault: .shortRead(atLayer: 0))
        let first = try XCTUnwrap(model.newConversation())
        let second = try XCTUnwrap(model.newConversation())
        model.rename(first, to: "owner chat")

        model.send("first turn", in: first)
        XCTAssertTrue(model.isRunning(first))
        XCTAssertFalse(model.isRunning(second))
        XCTAssertEqual(model.runSnapshot(for: first).state, .running)
        XCTAssertEqual(model.runSnapshot(for: second), RunSnapshot())
        XCTAssertEqual(model.runBusyReason(for: second), "A turn is already running in “owner chat”.")

        model.drafts[second] = "keep this draft"
        model.send("must not append", in: second)
        XCTAssertTrue(try XCTUnwrap(model.conversation(second)).messages.isEmpty)
        XCTAssertEqual(model.drafts[second], "keep this draft")

        for _ in 0..<400 where model.hasActiveRun {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertFalse(model.hasActiveRun, "the injected short read never reached the controller")
        XCTAssertNotNil(model.runSnapshot(for: first).fault)
        XCTAssertNil(model.runSnapshot(for: second).fault)
        XCTAssertEqual(model.runSnapshot(for: second), RunSnapshot())

        let failed = try XCTUnwrap(model.conversation(first))
        XCTAssertEqual(failed.messages.map(\.role), [.user, .assistant])
        XCTAssertEqual(failed.messages.first?.text, "first turn")
        XCTAssertEqual(failed.messages.last?.failure, .verifyModelFiles)
        XCTAssertFalse(failed.messages.last?.namedError?.isEmpty ?? true)
    }

    func testANewChatsTokenCeilingIsTheRunnersOwn() {
        let model = AppModel.freshForTests()
        model.defaults.model = .kimiK3
        let id = model.newConversation()!
        let ceiling = K3ProductMemoryBudget.currentPolicy.maximumNewTokens
        XCTAssertEqual(model.conversation(id)?.settings.maximumNewTokens, ceiling)
        XCTAssertEqual(
            model.capabilities(for: .kimiK3)?.maximumNewTokens, ceiling,
            "a normal chat runs until response framing or the runner's explicit safety ceiling")
    }

    func testAnUnavailablePersistedDefaultIsNotSilentlyReplacedOrOfferedForANewChat() throws {
        let model = AppModel.freshForTests(installedModel: nil)
        model.defaults.model = .deepseekV4Flash

        XCTAssertNil(model.newConversation())
        XCTAssertTrue(model.conversations.isEmpty)
        XCTAssertEqual(model.defaults.model, .deepseekV4Flash)
        XCTAssertFalse(
            model.modelPickerChoices(current: .deepseekV4Flash).contains {
                $0.modelID == .deepseekV4Flash
            })
    }

    func testDownloadsNavigationDoesNotAdvertiseAnEmptyOperationalSurface() {
        let model = AppModel.freshForTests()

        XCTAssertTrue(model.transfersForPresentation.isEmpty)
        XCTAssertFalse(model.showsDownloadsSection)
    }

    func testARequestUsesTheArtifactRootTheScanActuallyFound() throws {
        let model = AppModel.freshForTests()
        let id = try XCTUnwrap(model.newConversation())
        let conversation = try XCTUnwrap(model.conversation(id))
        let scanned = try XCTUnwrap(model.installed.runnableArtifact(for: .kimiK3))

        XCTAssertEqual(model.makeRequest(for: conversation)?.artifact.root.path, scanned.rootPath)
    }

    func testSendDoesNotAppendAUserTurnWhenTheSelectedModelIsUnavailable() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunUnavailableSend-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ConversationStore(directory: directory)
        let conversation = Conversation(
            settings: ConversationSettings(
                model: .kimiK3,
                memoryBudgetBytes: CatalogFixtures.k3OnRecordMinimumBudget))
        try store.save(conversation)
        let model = AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            store: store,
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: CatalogFixtures.all),
            installed: InstalledModels(
                storage: StorageManager(),
                catalog: CatalogFixtures.snapshot,
                ledger: InMemoryVerificationLedger()),
            seedRecordedRuns: false,
            startDiscovery: false)

        model.send("must not be recorded", in: conversation.id)

        XCTAssertTrue(try XCTUnwrap(model.conversation(conversation.id)).messages.isEmpty)
        XCTAssertNil(model.makeRequest(for: conversation))
    }

    func testAStorageGrantFailureChangesNeitherSendNorRegenerateTranscript() throws {
        let model = makeModelWhoseRunGrantFails()
        let id = try XCTUnwrap(model.newConversation())
        model.drafts[id] = "draft survives"

        model.send("must not be recorded", in: id)
        XCTAssertTrue(try XCTUnwrap(model.conversation(id)).messages.isEmpty)
        XCTAssertEqual(model.drafts[id], "draft survives")
        XCTAssertNotNil(model.runPreparationError(for: id))

        model.update(id) {
            $0.append(Message(role: .user, text: "existing question"))
            $0.append(Message(role: .assistant, text: "existing answer"))
        }
        let before = try XCTUnwrap(model.conversation(id)).messages
        model.regenerateLastTurn(in: id)
        XCTAssertEqual(try XCTUnwrap(model.conversation(id)).messages, before)
        XCTAssertNotNil(model.runPreparationError(for: id))
    }

    private func makeModelWhoseRunGrantFails() -> AppModel {
        let location = "/Volumes/UNPLUGGED"
        let descriptor = CatalogFixtures.kimiK3.descriptor
        let now = Date()
        let artifact = DiscoveredArtifact(
            rootPath: "\(location)/k3-artifact", locationPath: location,
            index: .unreadable, model: .kimiK3, displayName: descriptor.displayName,
            bytesOnDisk: descriptor.totalBytes, fileCount: descriptor.totalFileCount,
            expectedFileCount: descriptor.totalFileCount, expectedBytes: descriptor.totalBytes,
            verification: .fullyVerified, verifiedAt: now, scannedAt: now)
        let report = DiscoveryReport(
            locations: [
                LocationScan(
                    rootPath: location, displayName: "UNPLUGGED", isMounted: true,
                    storageKey: FailingRunScopeStorage.key, artifacts: [artifact],
                    unreadable: [:], scannedAt: now)
            ],
            scannedAt: now)
        let installed = InstalledModels(
            storage: FailingRunScopeStorage(location: URL(fileURLWithPath: location)),
            catalog: CatalogFixtures.snapshot,
            ledger: InMemoryVerificationLedger(),
            initialReport: report)
        return AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            runtimes: .preview(entries: CatalogFixtures.all),
            store: ConversationStore(
                directory: URL(fileURLWithPath: NSTemporaryDirectory())
                    .appendingPathComponent("MinirunGrantGuard-\(UUID().uuidString)")),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: CatalogFixtures.all),
            installed: installed,
            seedRecordedRuns: false,
            startDiscovery: false)
    }
}

/// The product boundary is stricter than the preview boundary. Finding and
/// fully verifying an artifact is necessary, but it cannot make a stand-in
/// tokenizer or an invented runner become a product runtime.
@MainActor
final class ProductRuntimeTruthGateTests: XCTestCase {

    func testProductRuntimeCompositionContainsOnlyCompletedProviders() {
        #if arch(arm64)
            XCTAssertEqual(
                ModelRuntimeRegistry.product.registeredModelIDs,
                [.kimiK3, .deepseekV4Flash])
            XCTAssertNotNil(ModelRuntimeRegistry.product.runtime(for: .deepseekV4Flash))
        #else
            XCTAssertEqual(ModelRuntimeRegistry.product.registeredModelIDs, [])
            XCTAssertNil(ModelRuntimeRegistry.product.runtime(for: .kimiK3))
            XCTAssertNil(ModelRuntimeRegistry.product.runtime(for: .deepseekV4Flash))
        #endif
        XCTAssertNil(ModelRuntimeRegistry.product.runtime(for: .minimaxH3))
    }

    func testProductTokenizerUsesTheVerifiedCurrentArtifactIdentity() throws {
        let location = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunProductTokenizer-\(UUID().uuidString)")
        let artifact = location.appendingPathComponent("artifact", isDirectory: true)
        try FileManager.default.createDirectory(at: artifact, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: location) }

        let vocabulary = try syntheticK3Vocabulary()
        let tokenizerURL = artifact.appendingPathComponent("tiktoken.model")
        try vocabulary.write(to: tokenizerURL)
        let tokenizerDigest = FileDigestComputer.sha256(of: vocabulary)
        let sourceRevision = String(repeating: "d", count: 40)
        let indexData = Data(
            """
            {
              "format": "quantized_tile_container",
              "relationship": "byte_preserving_repack",
              "source_repo": "moonshotai/Kimi-K3",
              "source_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "files": 1,
              "bytes": \(vocabulary.count),
              "tokenizer": {
                "file": "tiktoken.model",
                "bytes": \(vocabulary.count),
                "sha256": "\(tokenizerDigest)",
                "source_repo": "moonshotai/Kimi-K3",
                "source_revision": "\(sourceRevision)",
                "license": "LICENSE"
              }
            }
            """.utf8)
        let indexURL = artifact.appendingPathComponent("index.json")
        try indexData.write(to: indexURL)
        let licenseData = Data("synthetic tokenizer license".utf8)
        let licenseURL = artifact.appendingPathComponent("LICENSE")
        try licenseData.write(to: licenseURL)
        let index = ArtifactIndexIdentity.parse(indexData)
        let rooted = try ArtifactVerificationRoot.open(
            relativeComponents: ["artifact"], beneath: location)
        let files = try [
            verifiedFile(
                path: "tiktoken.model", url: tokenizerURL, isPayload: true),
            verifiedFile(path: "index.json", url: indexURL, isPayload: false),
            verifiedFile(path: "LICENSE", url: licenseURL, isPayload: false),
        ]
        let repoFiles = files.map {
            RepoFile(
                path: $0.path, sizeBytes: $0.expectedSizeBytes, digest: $0.digest,
                isPayload: $0.isPayload)
        }
        let identity = ArtifactDigestPlanIdentity.compute(repoFiles)
        let total = files.reduce(UInt64(0)) { $0 + $1.expectedSizeBytes }
        let evidence = ArtifactVerificationEvidence(
            model: .kimiK3,
            repository: HuggingFaceRepoRef(
                repoID: "nanguoyu/Kimi-K3-minirun",
                revision: String(repeating: "e", count: 40)),
            treeIdentity: identity,
            completenessAuthorityIdentity: identity,
            selectedPlanIdentity: identity,
            treeFileCount: files.count,
            treePayloadFileCount: 1,
            treeMetadataFileCount: 2,
            treeBytes: total,
            treePayloadBytes: UInt64(vocabulary.count),
            treeMetadataBytes: UInt64(indexData.count + licenseData.count),
            selectedBytes: total,
            index: index,
            root: rooted.artifactIdentity,
            files: files)
        let authority = try ArtifactRuntimeAuthority(root: rooted, evidence: evidence)

        let tokenizer = try K3ProductTokenizer(
            artifact: ArtifactReference(root: artifact, runtimeAuthority: authority))
        let ids = try tokenizer.encode(turns: [PromptTurn(role: .user, text: "A")])

        XCTAssertFalse(ids.isEmpty)
        XCTAssertEqual(
            tokenizer.verifiedIdentity.artifactRevision,
            String(repeating: "e", count: 40))
        XCTAssertEqual(tokenizer.verifiedIdentity.sourceRevision, sourceRevision)
        XCTAssertEqual(tokenizer.verifiedIdentity.sha256, tokenizerDigest)
        XCTAssertTrue(tokenizer.appliesChatTemplate)
        XCTAssertTrue(tokenizer.provenance.contains(String(repeating: "e", count: 12)))
        XCTAssertTrue(tokenizer.provenance.contains(sourceRevision.prefix(12)))
        // K3's first reserved token is 163,584; close is reserved offset 4.
        // The package-level boundary test separately binds that offset to the
        // typed K3SpecialToken enum, so the App test needs no transitive import.
        let terminal = 163_584 + 4
        XCTAssertEqual(
            tokenizer.decode(tokenIDs: [65, terminal, 66]), "A",
            "chat protocol closing tags must not appear as assistant prose")
    }

    func testProductK3RuntimeRequiresCurrentRuntimeFilesButNotACompiledArtifactCommit() throws {
        let withoutTokenizer = ArtifactIndexIdentity.parse(
            Data(
                """
                {
                  "format": "quantized_tile_container",
                  "relationship": "byte_preserving_repack",
                  "source_repo": "moonshotai/Kimi-K3",
                  "source_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                  "files": 372,
                  "bytes": 1559976181760
                }
                """.utf8))
        let withTokenizerFromAnotherSourceCommit = ArtifactIndexIdentity.parse(
            Data(
                """
                {
                  "format": "quantized_tile_container",
                  "relationship": "byte_preserving_repack",
                  "source_repo": "moonshotai/Kimi-K3",
                  "source_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                  "files": 372,
                  "bytes": 1559976181760,
                  "tokenizer": {
                    "file": "tiktoken.model",
                    "bytes": 2795286,
                    "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                    "source_repo": "moonshotai/Kimi-K3",
                    "source_revision": "dddddddddddddddddddddddddddddddddddddddd",
                    "license": "LICENSE"
                  }
                }
                """.utf8))
        let withUnsupportedFormat = ArtifactIndexIdentity.parse(
            Data(
                """
                {
                  "format": "future_container_v2",
                  "relationship": "byte_preserving_repack",
                  "source_repo": "moonshotai/Kimi-K3",
                  "source_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                  "files": 372,
                  "bytes": 1559976181760,
                  "tokenizer": {
                    "file": "tiktoken.model",
                    "bytes": 2795286,
                    "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                    "source_repo": "moonshotai/Kimi-K3",
                    "source_revision": "dddddddddddddddddddddddddddddddddddddddd",
                    "license": "LICENSE"
                  }
                }
                """.utf8))

        let oldCopy = makeProductK3Model(index: withoutTokenizer)
        XCTAssertEqual(
            oldCopy.modelAvailability(for: .kimiK3), .runtimeFilesUnavailable)
        XCTAssertEqual(
            oldCopy.modelAvailability(for: .kimiK3).productReason,
            "Update this model copy, then verify it again")

        let updatedCopy = makeProductK3Model(index: withTokenizerFromAnotherSourceCommit)
        XCTAssertEqual(updatedCopy.modelAvailability(for: .kimiK3), .available)

        let incompatibleCopy = makeProductK3Model(index: withUnsupportedFormat)
        XCTAssertEqual(
            incompatibleCopy.modelAvailability(for: .kimiK3), .runtimeFilesUnavailable)
    }

    func testProductDefaultsRefuseAFoundArtifactWithoutAVerifiedRuntimeAndWriteNothing() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunProductRuntimeGate-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ConversationStore(directory: directory)
        let conversation = Conversation(
            settings: ConversationSettings(
                model: .kimiK3,
                memoryBudgetBytes: CatalogFixtures.k3OnRecordMinimumBudget))
        try store.save(conversation)

        // No runtime registry is injected. This is the product-safe default,
        // even though the fixture below is complete and fully verified.
        let model = AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            store: store,
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: CatalogFixtures.all),
            installed: AppModel.previewInstalledModels(
                model: .kimiK3, snapshot: CatalogFixtures.snapshot),
            startDiscovery: false)
        let loaded = try XCTUnwrap(model.conversation(conversation.id))
        let document = store.fileURL(for: conversation.id)
        let beforeDocument = try Data(contentsOf: document)
        let beforeFiles = try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
        let beforeDestination = model.destination
        model.drafts[conversation.id] = "keep this draft"

        XCTAssertEqual(model.modelAvailability(for: .kimiK3), .runtimeUnavailable)
        XCTAssertTrue(
            model.modelAvailability(for: .kimiK3).reason.contains("verified runner and tokenizer"))
        XCTAssertNil(model.capabilities(for: .kimiK3))
        XCTAssertNil(model.assembledPrompt(for: loaded), "the product must not fall back to preview ids")
        XCTAssertNil(model.makeRequest(for: loaded))
        XCTAssertNil(model.newConversation())
        model.send("must not become a turn", in: conversation.id)

        XCTAssertEqual(try Data(contentsOf: document), beforeDocument)
        XCTAssertEqual(
            try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted(), beforeFiles)
        XCTAssertEqual(model.conversation(conversation.id)?.messages, [])
        XCTAssertEqual(model.drafts[conversation.id], "keep this draft")
        XCTAssertEqual(model.destination, beforeDestination)
        XCTAssertNil(model.runPreparationError(for: conversation.id))
        XCTAssertFalse(model.hasActiveRun)
        XCTAssertTrue(model.history.isEmpty, "recorded preview runs must be opt-in, never the default")
        XCTAssertTrue(
            model.calibrations.isEmpty,
            "recorded link measurements must be opt-in, never the product default")
    }

    /// The same gate, for a published container this build ships no runner for.
    /// A catalogue row plus fully verified bytes on disk still cannot become an
    /// executable runtime, and the attempt must not mutate a persisted byte.
    func testAPublishedModelWithNoRegisteredRuntimeCannotBeRun() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunH3RuntimeGate-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ConversationStore(directory: directory)
        let conversation = Conversation(
            settings: ConversationSettings(
                model: .minimaxH3,
                memoryBudgetBytes: 4_000_000_000))
        try store.save(conversation)

        let model = AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            store: store,
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: CatalogFixtures.all),
            installed: AppModel.previewInstalledModels(
                model: .minimaxH3, snapshot: CatalogFixtures.snapshot),
            startDiscovery: false)
        model.defaults.model = .minimaxH3
        let loaded = try XCTUnwrap(model.conversation(conversation.id))
        let document = store.fileURL(for: conversation.id)
        let beforeDocument = try Data(contentsOf: document)
        let beforeFiles = try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
        let beforeDestination = model.destination
        model.drafts[conversation.id] = "preserve this draft"

        XCTAssertEqual(model.modelAvailability(for: .minimaxH3), .runtimeUnavailable)
        XCTAssertNil(model.capabilities(for: .minimaxH3))
        XCTAssertNil(model.assembledPrompt(for: loaded))
        XCTAssertNil(model.makeRequest(for: loaded))
        XCTAssertNil(model.newConversation())
        model.send("must not be tokenized by a preview stand-in", in: conversation.id)

        XCTAssertEqual(try Data(contentsOf: document), beforeDocument)
        XCTAssertEqual(
            try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted(), beforeFiles)
        XCTAssertEqual(model.conversation(conversation.id)?.messages, [])
        XCTAssertEqual(model.drafts[conversation.id], "preserve this draft")
        XCTAssertEqual(model.destination, beforeDestination)
        XCTAssertNil(model.runPreparationError(for: conversation.id))
        XCTAssertFalse(model.hasActiveRun)
        XCTAssertTrue(model.history.isEmpty)
        XCTAssertTrue(model.calibrations.isEmpty)
    }

    func testAnExplicitPreviewRegistryIsTheOnlyPathToTheStandIns() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunExplicitPreviewRuntime-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            runtimes: .preview(entries: CatalogFixtures.all),
            store: ConversationStore(directory: directory),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            downloadServices: .preview(entries: CatalogFixtures.all),
            installed: AppModel.previewInstalledModels(
                model: .kimiK3, snapshot: CatalogFixtures.snapshot),
            seedRecordedRuns: true,
            startDiscovery: false)

        XCTAssertEqual(model.modelAvailability(for: .kimiK3), .available)
        XCTAssertNotNil(model.capabilities(for: .kimiK3))
        XCTAssertFalse(model.history.isEmpty, "only the explicit preview path asks for archived records")
        XCTAssertFalse(model.calibrations.isEmpty)
        let id = try XCTUnwrap(model.newConversation())
        let conversation = try XCTUnwrap(model.conversation(id))
        let prompt = try XCTUnwrap(model.assembledPrompt(for: conversation))
        XCTAssertTrue(prompt.provenance.contains("preview stand-in"))
        XCTAssertFalse(prompt.appliesChatTemplate)
    }

    private func makeProductK3Model(index: ArtifactIndexIdentity) -> AppModel {
        let location = "/Volumes/PRODUCT-K3"
        let descriptor = CatalogFixtures.kimiK3.descriptor
        let now = Date()
        let artifact = DiscoveredArtifact(
            rootPath: "\(location)/k3-artifact", locationPath: location,
            index: index, model: .kimiK3, displayName: descriptor.displayName,
            bytesOnDisk: descriptor.totalBytes, fileCount: descriptor.totalFileCount,
            expectedFileCount: descriptor.totalFileCount,
            expectedBytes: descriptor.totalBytes,
            verification: .fullyVerified, verifiedAt: now, scannedAt: now)
        let installed = InstalledModels(
            storage: StorageManager(), catalog: CatalogFixtures.snapshot,
            ledger: InMemoryVerificationLedger(),
            initialReport: DiscoveryReport(
                locations: [
                    LocationScan(
                        rootPath: location, displayName: "PRODUCT-K3", isMounted: true,
                        storageKey: nil, artifacts: [artifact], unreadable: [:],
                        scannedAt: now)
                ],
                scannedAt: now))
        return AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            runtimes: .product,
            store: .ephemeral(),
            userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
            persistsDefaults: false,
            downloadServices: .preview(entries: CatalogFixtures.all),
            installed: installed,
            startDiscovery: false)
    }

    private func syntheticK3Vocabulary() throws -> Data {
        var out = Data()
        out.reserveCapacity(3_000_000)
        for rank in 0..<163_584 {
            let token: Data
            if rank < 256 {
                token = Data([UInt8(rank)])
            } else {
                let value = UInt32(rank).bigEndian
                token = withUnsafeBytes(of: value) { buffer in
                    var bytes: [UInt8] = [0xFE]
                    bytes.append(contentsOf: buffer)
                    return Data(bytes)
                }
            }
            out.append(Data(token.base64EncodedString().utf8))
            out.append(0x20)
            out.append(Data(String(rank).utf8))
            out.append(0x0A)
        }
        return out
    }

    private func verifiedFile(
        path: String, url: URL, isPayload: Bool
    ) throws -> ArtifactVerifiedFile {
        let descriptor = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        guard descriptor >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        defer { _ = Darwin.close(descriptor) }
        let filesystem = try ArtifactFilesystemEvidence.identity(
            of: descriptor, path: path, expectedDirectory: false)
        return ArtifactVerifiedFile(
            path: path, expectedSizeBytes: filesystem.sizeBytes,
            digest: .sha256(hex: try FileDigestComputer.sha256(ofFileAt: url)),
            isPayload: isPayload,
            filesystem: filesystem)
    }
}

/// The prompt seam. It orders turns and says who tokenized them; it does not
/// invent a chat template.
final class PromptAssemblerTests: XCTestCase {

    func testEveryTurnReachesTheAssembledPromptInOrder() throws {
        var conversation = Conversation(
            settings: ConversationSettings(model: .kimiK3, memoryBudgetBytes: 5_800_000_000))
        conversation.append(Message(role: .user, text: "one"))
        conversation.append(Message(role: .assistant, text: "two"))
        conversation.append(Message(role: .user, text: "three"))

        let assembled = try PromptAssembler.assemble(
            conversation: conversation, tokenizer: PreviewTokenizer())
        XCTAssertEqual(assembled.turns.map { $0.text }, ["one", "two", "three"])
        XCTAssertEqual(assembled.turns.map { $0.role }, [.user, .assistant, .user])
        XCTAssertFalse(
            assembled.appliesChatTemplate,
            "this build applies no chat template and must not claim to")
        XCTAssertFalse(assembled.provenance.isEmpty)
    }

    func testTheStandInTokenizerIsDeterministic() throws {
        var conversation = Conversation(
            settings: ConversationSettings(model: .kimiK3, memoryBudgetBytes: 5_800_000_000))
        conversation.append(Message(role: .user, text: "北京是中国的首都"))
        let first = try PromptAssembler.assemble(
            conversation: conversation, tokenizer: PreviewTokenizer()).tokenIDs
        let second = try PromptAssembler.assemble(
            conversation: conversation, tokenizer: PreviewTokenizer()).tokenIDs
        XCTAssertEqual(first, second)
        XCTAssertFalse(first.isEmpty)
    }

    /// The runner produces the archived token on a fresh turn, and continues
    /// rather than repeating on the next one.
    func testThePreviewLadderContinuesInsteadOfRepeating() {
        XCTAssertEqual(
            PreviewReplies.tokens(model: .kimiK3, turnIndex: 0, count: 1).map { $0.tokenID },
            [3372])
        XCTAssertNotEqual(
            PreviewReplies.tokens(model: .kimiK3, turnIndex: 1, count: 1).map { $0.tokenID },
            [3372])
    }
}

// MARK: - Test support

extension AppModel {
    /// An app model over a scratch directory and a scratch defaults domain, so a
    /// test never reads or writes the reviewer's own conversations.
    static func freshForTests(
        installedModel: ModelID? = .kimiK3,
        verification: ArtifactVerification = .fullyVerified,
        runFault: MockDecodeRunner.Fault? = nil,
        runtimeCensus: ArtifactCensus? = nil,
        runtimeWorkingSetReserveBytes: (@Sendable (Int, Int) throws -> UInt64)? = nil,
        userDefaults: UserDefaults? = nil,
        store: ConversationStore? = nil,
        now: Date = Date()
    ) -> AppModel {
        freshForTests(
            installedModels: installedModel.map { [$0] } ?? [],
            verification: verification,
            runFault: runFault,
            runtimeCensus: runtimeCensus,
            runtimeWorkingSetReserveBytes: runtimeWorkingSetReserveBytes,
            userDefaults: userDefaults,
            store: store,
            now: now)
    }

    static func freshForTests(
        installedModels: [ModelID],
        verification: ArtifactVerification = .fullyVerified,
        runFault: MockDecodeRunner.Fault? = nil,
        runtimeCensus: ArtifactCensus? = nil,
        runtimeWorkingSetReserveBytes: (@Sendable (Int, Int) throws -> UInt64)? = nil,
        userDefaults injectedDefaults: UserDefaults? = nil,
        store injectedStore: ConversationStore? = nil,
        now: Date = Date()
    ) -> AppModel {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunTests-\(UUID().uuidString)", isDirectory: true)
        let defaults = injectedDefaults
            ?? UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!
        let storage = StorageManager(bookmarkLedger: InMemoryBookmarkLedger())
        let installed = installedModels.isEmpty
            ? InstalledModels(
                storage: storage,
                catalog: CatalogFixtures.snapshot,
                ledger: InMemoryVerificationLedger())
            : AppModel.previewInstalledModels(
                models: installedModels,
                snapshot: CatalogFixtures.snapshot,
                verification: verification,
                storage: storage)
        return AppModel(
            entries: CatalogFixtures.all,
            catalogSnapshot: CatalogFixtures.snapshot,
            runtimes: .preview(
                entries: CatalogFixtures.all, fault: runFault,
                artifactCensus: runtimeCensus,
                workingSetReserveBytes: runtimeWorkingSetReserveBytes),
            store: injectedStore ?? ConversationStore(directory: directory),
            userDefaults: defaults,
            storage: storage,
            downloadServices: .preview(entries: CatalogFixtures.all),
            now: now,
            installed: installed,
            seedRecordedRuns: false,
            startDiscovery: false)
    }
}

private final class FailingRunScopeStorage: StorageManaging, @unchecked Sendable {
    static let key = StorageKey("artifact-location.failing-run-scope")
    let location: URL

    init(location: URL) { self.location = location }

    func volumes() throws -> [VolumeDescriptor] { [] }
    func describe(_ url: URL) -> StorageInfo { StorageInfo.describing(path: url.path) }
    func freeSpace(at url: URL) -> FreeSpace { FreeSpace(describe(url)) }
    func canHold(bytes: UInt64, at url: URL, headroomBytes: UInt64) -> SpaceVerdict {
        .fits(spareBytes: 0)
    }
    func scope(for url: URL) throws -> StorageScope { StorageScope(url: url) }
    func remember(_ url: URL, as key: StorageKey) throws -> StorageBookmark {
        StorageBookmark(key: key, createdAt: Date(), recordedPath: url.path, byteCount: 0)
    }
    func resolve(_ key: StorageKey) throws -> StorageScope {
        throw StorageError.volumeUnavailable(path: location.path)
    }
    func bookmark(_ key: StorageKey) -> StorageBookmark? {
        guard key == Self.key else { return nil }
        return StorageBookmark(
            key: key, createdAt: Date(), recordedPath: location.path, byteCount: 0)
    }
    func forget(_ key: StorageKey) {}
    func knownLocations() -> [StorageKey] { [Self.key] }
    func bookmarkData(for url: URL) throws -> Data { Data(url.path.utf8) }
}
