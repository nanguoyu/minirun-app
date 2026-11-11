import Foundation

/// Where each conversation's saved generation state lives, and when it stops
/// being that conversation's.
///
/// One file per conversation, beside `Conversations/` in Application Support
/// rather than in a cache directory the system may purge in the middle of a
/// chat:
///
/// ```text
/// ~/Library/Application Support/wang.wangdongdong.minirun/Sessions/<uuid>.v41session
/// ```
///
/// A file is an optimisation and never the answer. A turn whose file is
/// missing, refused or stale is replayed from the transcript in the same shape
/// and computes the same ids and logits (ADR 0023), so deleting one costs time
/// and nothing else. That is why every rule below deletes when in doubt.
struct ConversationSessionStore: Sendable {
    static let fileExtension = "v41session"

    let directory: URL
    /// False for visual review, refusal and test models: no file is ever
    /// named, created or removed, and every turn replays.
    let isPersistent: Bool
    private let fileManager: FileManager

    init(directory: URL, fileManager: FileManager = .default, isPersistent: Bool = true) {
        self.directory = directory
        self.fileManager = fileManager
        self.isPersistent = isPersistent
    }

    /// `Sessions/`, beside the store's own `Conversations/`, persistent exactly
    /// when the conversations are.
    init(besideConversationsIn store: ConversationStore, fileManager: FileManager = .default) {
        self.init(
            directory: store.directory.deletingLastPathComponent()
                .appendingPathComponent("Sessions", isDirectory: true),
            fileManager: fileManager,
            isPersistent: store.isPersistent)
    }

    /// Where `id`'s state is read from and written to. Naming it creates
    /// nothing: the runtime that writes one makes the directory
    /// (``ensureDirectory(for:)``), so a chat with a model that keeps no state
    /// leaves no trace here. Nil when this store keeps no files.
    func fileURL(for id: UUID) -> URL? {
        isPersistent ? url(for: id) : nil
    }

    /// Makes the directory a session file lives in, because the runner's
    /// atomic replace needs it to exist. False when it cannot be made.
    static func ensureDirectory(for file: URL, fileManager: FileManager = .default) -> Bool {
        do {
            try fileManager.createDirectory(
                at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            return true
        } catch {
            return false
        }
    }

    func exists(for id: UUID) -> Bool {
        isPersistent && fileManager.fileExists(atPath: url(for: id).path)
    }

    func delete(_ id: UUID) {
        guard isPersistent else { return }
        let file = url(for: id)
        guard fileManager.fileExists(atPath: file.path) else { return }
        try? fileManager.removeItem(at: file)
    }

    /// Removes every file whose conversation is gone: one deleted while its
    /// turn was still writing, or by an earlier build.
    func removeOrphans(keeping ids: Set<UUID>) {
        guard isPersistent,
            let names = try? fileManager.contentsOfDirectory(atPath: directory.path)
        else { return }
        for name in names where name.hasSuffix("." + Self.fileExtension) {
            let stem = String(name.dropLast(Self.fileExtension.count + 1))
            if let id = UUID(uuidString: stem), ids.contains(id) { continue }
            try? fileManager.removeItem(at: directory.appendingPathComponent(name))
        }
    }

    private func url(for id: UUID) -> URL {
        directory.appendingPathComponent(
            "\(id.uuidString).\(Self.fileExtension)", isDirectory: false)
    }

    /// Whether a change to a conversation leaves its saved state a prefix of
    /// what it will ask for next.
    ///
    /// Kept only when the change appends: a new user turn, a reply, an
    /// interrupted reply recorded at launch, or nothing at all to the
    /// transcript (a rename, a budget). Discarded when the model changes, or
    /// when any earlier message is edited or removed, which includes the
    /// assistant turn a regenerate drops: the next turn replays the transcript
    /// as it now stands. A budget does not appear here because state does not
    /// depend on residency: a block pinned or streamed holds the same bytes.
    static func invalidates(before: Conversation, after: Conversation) -> Bool {
        guard before.settings.model == after.settings.model else { return true }
        guard after.messages.count >= before.messages.count else { return true }
        return zip(before.messages, after.messages).contains { old, new in
            old.id != new.id || old.role != new.role || old.text != new.text
        }
    }
}
