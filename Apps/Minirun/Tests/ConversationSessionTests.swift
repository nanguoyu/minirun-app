import Foundation
import MinirunKit
import MinirunRunners
import XCTest

@testable import MinirunApp

/// A conversation's saved generation state: where it lives, which changes keep
/// it, and which discard it so the next turn replays (ADR 0023).
@MainActor
final class ConversationSessionTests: XCTestCase {
    private var root: URL!
    private var store: ConversationStore!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MinirunSessionTests-\(UUID().uuidString)", isDirectory: true)
        store = ConversationStore(
            directory: root.appendingPathComponent("Conversations", isDirectory: true))
        try store.createDirectoryIfNeeded()
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private var sessions: ConversationSessionStore {
        ConversationSessionStore(besideConversationsIn: store)
    }

    private func writeSessionFile(for id: UUID) throws {
        let url = try XCTUnwrap(sessions.fileURL(for: id))
        XCTAssertTrue(ConversationSessionStore.ensureDirectory(for: url))
        try Data("state".utf8).write(to: url)
    }

    private func conversation(_ messages: [Message], model: ModelID = .kimiK3) -> Conversation {
        Conversation(
            settings: ConversationSettings(model: model, memoryBudgetBytes: 8 << 30),
            messages: messages)
    }

    // MARK: - The store

    func testAFileLivesBesideTheConversationsAndOnlyWhenTheyAreKept() throws {
        let id = UUID()
        XCTAssertEqual(
            sessions.fileURL(for: id),
            root.appendingPathComponent("Sessions", isDirectory: true)
                .appendingPathComponent("\(id.uuidString).v41session"))
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: root.appendingPathComponent("Sessions").path),
            "naming a file creates nothing")
        let ephemeral = ConversationSessionStore(besideConversationsIn: .ephemeral())
        XCTAssertFalse(ephemeral.isPersistent)
        XCTAssertNil(ephemeral.fileURL(for: id))
    }

    func testOrphansGoAndAConversationsOwnFileStays() throws {
        let kept = UUID()
        let gone = UUID()
        try writeSessionFile(for: kept)
        try writeSessionFile(for: gone)
        sessions.removeOrphans(keeping: [kept])
        XCTAssertTrue(sessions.exists(for: kept))
        XCTAssertFalse(sessions.exists(for: gone))
    }

    // MARK: - The rule

    func testOnlyAnAppendKeepsTheState() {
        let first = Message(role: .user, text: "what is the capital of austria")
        let answer = Message(role: .assistant, text: "The capital of Austria is Vienna.")
        let second = Message(role: .user, text: "and of hungary?")
        let before = conversation([first, answer])

        XCTAssertFalse(
            ConversationSessionStore.invalidates(before: before, after: before),
            "a rename or a budget leaves the transcript as it was")
        XCTAssertFalse(
            ConversationSessionStore.invalidates(
                before: before, after: conversation([first, answer, second])))

        var regenerated = before
        _ = regenerated.removeLastAssistantTurn()
        XCTAssertTrue(ConversationSessionStore.invalidates(before: before, after: regenerated))

        var edited = before
        edited.messages[0].text = "what is the capital of hungary"
        XCTAssertTrue(ConversationSessionStore.invalidates(before: before, after: edited))

        var removed = before
        removed.messages.remove(at: 0)
        XCTAssertTrue(ConversationSessionStore.invalidates(before: before, after: removed))

        var otherModel = before
        otherModel.settings.model = .deepseekV41Flash
        XCTAssertTrue(ConversationSessionStore.invalidates(before: before, after: otherModel))

        var otherBudget = before
        otherBudget.settings.memoryBudgetBytes = 15 << 30
        XCTAssertFalse(
            ConversationSessionStore.invalidates(before: before, after: otherBudget),
            "state does not depend on residency, so a budget keeps it")
    }

    // MARK: - The lifecycle, through the app model

    func testTheAppModelKeepsTheStateThroughAppendsAndBudgetsAndDropsItOnEverythingElse()
        throws
    {
        let model = AppModel.freshForTests(store: store)
        let id = try XCTUnwrap(model.newConversation())
        model.update(id) {
            $0.append(Message(role: .user, text: "what is the capital of austria"))
            $0.append(Message(role: .assistant, text: "Vienna."))
        }
        try writeSessionFile(for: id)

        model.update(id) { $0.append(Message(role: .user, text: "and of hungary?")) }
        XCTAssertTrue(sessions.exists(for: id), "a new turn keeps it")
        model.update(id) { $0.append(Message(role: .assistant, text: "Budapest.")) }
        XCTAssertTrue(sessions.exists(for: id), "a reply keeps it")
        model.setBudget(9 << 30, in: id)
        XCTAssertTrue(sessions.exists(for: id), "a budget keeps it")
        model.rename(id, to: "capitals")
        XCTAssertTrue(sessions.exists(for: id), "a title keeps it")

        // Regenerate drops the last reply before its turn starts; the turn
        // after it replays the transcript up to the turn being regenerated.
        model.update(id) { _ = $0.removeLastAssistantTurn() }
        XCTAssertFalse(sessions.exists(for: id), "regenerate discards it")

        try writeSessionFile(for: id)
        model.update(id) { $0.messages[0].text = "what is the capital of hungary" }
        XCTAssertFalse(sessions.exists(for: id), "an edited earlier message discards it")

        try writeSessionFile(for: id)
        model.update(id) { $0.settings.model = .deepseekV41Flash }
        XCTAssertFalse(sessions.exists(for: id), "another model discards it")

        try writeSessionFile(for: id)
        model.delete(id)
        XCTAssertFalse(sessions.exists(for: id), "deleting the conversation deletes it")
    }

    func testRegenerateThroughTheAppModelDiscardsTheState() async throws {
        let model = AppModel.freshForTests(store: store)
        let id = try XCTUnwrap(model.newConversation())
        model.send("first turn", in: id)
        // A stopped turn is still a recorded reply, and that is all this needs.
        model.stopTurn()
        for _ in 0..<1_000 where model.hasActiveRun {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertFalse(model.hasActiveRun)
        XCTAssertEqual(model.conversation(id)?.messages.last?.role, .assistant)
        try writeSessionFile(for: id)
        model.regenerateLastTurn(in: id)
        XCTAssertFalse(sessions.exists(for: id))
        model.stopTurn()
        for _ in 0..<1_000 where model.hasActiveRun {
            try await Task.sleep(for: .milliseconds(10))
        }
    }

    /// A reply the previous process never finished is recorded at launch as an
    /// interrupted turn, which appends, so its state stays for the next turn.
    /// A file whose conversation is gone does not.
    func testAnInterruptedReplyKeepsItsStateAndAnOrphanGoes() throws {
        let interrupted = conversation([Message(role: .user, text: "a reply that never ended")])
        try store.save(interrupted)
        try writeSessionFile(for: interrupted.id)
        let orphan = UUID()
        try writeSessionFile(for: orphan)

        let model = AppModel.freshForTests(store: store)
        let loaded = try XCTUnwrap(model.conversation(interrupted.id))
        XCTAssertEqual(loaded.messages.last?.failure, .interrupted)
        XCTAssertTrue(sessions.exists(for: interrupted.id))
        XCTAssertFalse(sessions.exists(for: orphan))
    }

    // MARK: - The ceiling

    /// The whole conversation is the prompt. Past the runtime's ceiling the
    /// next turn is refused in a sentence, before the transcript or the draft
    /// changes, and never answered by cutting earlier turns.
    func testAConversationPastThePromptCeilingIsRefusedInASentence() throws {
        let model = AppModel.freshForTests(runtimePromptTokenCeiling: 6, store: store)
        let id = try XCTUnwrap(model.newConversation())
        model.drafts[id] = "a question long enough to pass six preview ids"
        model.send("a question long enough to pass six preview ids", in: id)

        XCTAssertTrue(try XCTUnwrap(model.conversation(id)).messages.isEmpty)
        XCTAssertFalse(model.run.isRunning)
        XCTAssertEqual(model.drafts[id], "a question long enough to pass six preview ids")
        XCTAssertTrue(model.runPreparationReachedLength(for: id))
        let sentence = try XCTUnwrap(model.runPreparationError(for: id))
        XCTAssertTrue(sentence.hasPrefix("With this message the chat would be "), sentence)
        XCTAssertTrue(
            sentence.hasSuffix(
                "reads at most 6 in one conversation. Start a new chat to keep going."),
            sentence)

        // A turn that fits is not refused for its length.
        let roomy = AppModel.freshForTests(runtimePromptTokenCeiling: 512, store: store)
        let other = try XCTUnwrap(roomy.newConversation())
        roomy.send("short", in: other)
        XCTAssertFalse(roomy.runPreparationReachedLength(for: other))
        XCTAssertNil(roomy.runPreparationError(for: other))
        roomy.stopTurn()
    }

    func testTheSentenceNamesTheCountTheModelAndTheCeiling() {
        XCTAssertEqual(
            ConversationLengthRefusal(
                modelName: "DeepSeek V4.1 Flash", promptTokens: 530, ceiling: 512
            ).description,
            "With this message the chat would be 530 tokens, and DeepSeek V4.1 Flash reads "
                + "at most 512 in one conversation. Start a new chat to keep going.")
    }

    // MARK: - V4.1's options

    func testAV41TurnStatesItsFirstTurnAndItsFile() throws {
        let tokenizer = PreviewTokenizer()
        let first = "what is the capital of austria"
        let chat = conversation(
            [
                Message(role: .user, text: first),
                Message(role: .assistant, text: "Vienna."),
                Message(role: .user, text: "and of hungary?"),
            ], model: .deepseekV41Flash)
        let file = try XCTUnwrap(sessions.fileURL(for: chat.id))
        let options = try XCTUnwrap(
            DeepSeekV41ProductRuntimeProvider.sessionOptions(
                for: chat, tokenizer: tokenizer, sessionFile: file))
        XCTAssertEqual(
            options.firstTurnTokenCount,
            try tokenizer.encode(turns: [PromptTurn(role: .user, text: first)]).count)
        XCTAssertEqual(options.restoreFrom, file)
        XCTAssertEqual(options.writeTo, file)
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: file.deletingLastPathComponent().path),
            "the runner's atomic replace needs the directory")
        #if os(iOS)
            XCTAssertEqual(options.checkpoint, .everyPass)
        #else
            XCTAssertEqual(options.checkpoint, .atRelease)
        #endif

        // No file: the first-turn length still travels, so the turn replays.
        let unsaved = try XCTUnwrap(
            DeepSeekV41ProductRuntimeProvider.sessionOptions(
                for: chat, tokenizer: tokenizer, sessionFile: nil))
        XCTAssertNil(unsaved.restoreFrom)
        XCTAssertNil(unsaved.writeTo)
        XCTAssertEqual(unsaved.firstTurnTokenCount, options.firstTurnTokenCount)

        XCTAssertNil(
            try DeepSeekV41ProductRuntimeProvider.sessionOptions(
                for: conversation([], model: .deepseekV41Flash), tokenizer: tokenizer,
                sessionFile: file))
    }
}
