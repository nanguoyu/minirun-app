import Foundation

/// Conversations on disk. Not a mock — this one keeps real files.
///
/// One JSON document per conversation under Application Support, because the
/// alternative (one big file) makes every keystroke rewrite every conversation
/// and makes a single corrupt document lose all of them. Writes are atomic; a
/// document that fails to decode is reported by name and skipped rather than
/// silently dropping the rest of the directory on the floor.
///
/// The layout is deliberately boring and greppable:
///
/// ```text
/// ~/Library/Application Support/wang.wangdongdong.minirun/Conversations/<uuid>.json
/// ```
struct ConversationStore: Sendable {

    /// A document that could not be read. Surfaced, never swallowed.
    struct LoadFailure: Equatable, Sendable {
        let filename: String
        let reason: String
    }

    struct LoadResult: Equatable, Sendable {
        var conversations: [Conversation]
        var failures: [LoadFailure]
    }

    let directory: URL
    private let fileManager: FileManager
    /// False only for explicit visual-review, refusal and test models. Their
    /// conversation graph still behaves normally in memory, but no directory
    /// or JSON file is ever created.
    let isPersistent: Bool

    init(
        directory: URL,
        fileManager: FileManager = .default,
        isPersistent: Bool = true
    ) {
        self.directory = directory
        self.fileManager = fileManager
        self.isPersistent = isPersistent
    }

    /// A store-shaped sink for a visual review. The unique URL makes the state
    /// auditable in tests, but `isPersistent == false` means it is never
    /// created, read, written or removed.
    static func ephemeral() -> ConversationStore {
        ConversationStore(
            directory: FileManager.default.temporaryDirectory.appendingPathComponent(
                "MinirunVisualReview-\(UUID().uuidString)", isDirectory: true),
            isPersistent: false)
    }

    /// The real location. `bundleIdentifier` is nil in a unit-test host process
    /// on some configurations, so the fallback is spelled out rather than
    /// force-unwrapped into a crash nobody can reproduce.
    static func applicationSupport(fileManager: FileManager = .default) throws
        -> ConversationStore
    {
        let base = try fileManager.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil,
            create: true)
        let root = base
            .appendingPathComponent(Bundle.main.bundleIdentifier ?? "wang.wangdongdong.minirun", isDirectory: true)
            .appendingPathComponent("Conversations", isDirectory: true)
        return ConversationStore(directory: root, fileManager: fileManager)
    }

    // MARK: Codec

    private var encoder: JSONEncoder {
        let encoder = JSONEncoder()
        // Sorted and pretty because these files are meant to be readable: a
        // conversation whose budget you cannot check in a text editor is a
        // conversation you have to take the app's word for.
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    private var decoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }

    private func url(for id: UUID) -> URL {
        directory.appendingPathComponent("\(id.uuidString).json", isDirectory: false)
    }

    // MARK: Reading and writing

    func createDirectoryIfNeeded() throws {
        guard isPersistent else { return }
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    /// Newest first — the order the sidebar wants and the order a person thinks
    /// in.
    func load() -> LoadResult {
        guard isPersistent else { return LoadResult(conversations: [], failures: []) }
        guard let names = try? fileManager.contentsOfDirectory(atPath: directory.path) else {
            return LoadResult(conversations: [], failures: [])
        }
        var conversations: [Conversation] = []
        var failures: [LoadFailure] = []
        for name in names.sorted() where name.hasSuffix(".json") {
            let file = directory.appendingPathComponent(name, isDirectory: false)
            do {
                let data = try Data(contentsOf: file)
                conversations.append(try decoder.decode(Conversation.self, from: data))
            } catch {
                failures.append(LoadFailure(filename: name, reason: "\(error)"))
            }
        }
        conversations.sort { $0.updatedAt > $1.updatedAt }
        return LoadResult(conversations: conversations, failures: failures)
    }

    func save(_ conversation: Conversation) throws {
        guard isPersistent else { return }
        try createDirectoryIfNeeded()
        let data = try encoder.encode(conversation)
        try data.write(to: url(for: conversation.id), options: .atomic)
    }

    func delete(_ id: UUID) throws {
        guard isPersistent else { return }
        let file = url(for: id)
        guard fileManager.fileExists(atPath: file.path) else { return }
        try fileManager.removeItem(at: file)
    }

    func fileURL(for id: UUID) -> URL { url(for: id) }
}
