import Foundation
import MinirunKit

/// A conversation, and everything that decides how its turns run.
///
/// The settings live **here** rather than in the app's global settings on
/// purpose: a budget, a model and a token ceiling are statements about a run,
/// and a run belongs to a conversation. Global settings may only supply the
/// values a *new* conversation starts from — they are copied at creation and
/// never consulted again, so changing a default cannot retroactively change what
/// an existing chat was run under. That is the same invariant the memory dial
/// enforces inside one run (§5.3, "stated at start"), one level up.
struct Conversation: Codable, Equatable, Sendable, Identifiable {

    let id: UUID
    /// Derived from the first user message unless the operator renamed it.
    var title: String
    /// True once a human has typed a title; derivation stops touching it.
    var titleIsCustom: Bool
    let createdAt: Date
    var updatedAt: Date
    var settings: ConversationSettings
    var messages: [Message]

    init(
        id: UUID = UUID(), title: String = Conversation.untitled, titleIsCustom: Bool = false,
        createdAt: Date = Date(), updatedAt: Date = Date(),
        settings: ConversationSettings, messages: [Message] = []
    ) {
        self.id = id
        self.title = title
        self.titleIsCustom = titleIsCustom
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.settings = settings
        self.messages = messages
    }

    static let untitled = "New chat"

    // MARK: Title

    /// The title a conversation takes from its first user message.
    ///
    /// One line, collapsed whitespace, cut at a word boundary. A cut is marked
    /// with an ellipsis so a truncated title never reads as the whole sentence
    /// somebody typed.
    static func derivedTitle(from text: String, limit: Int = 42) -> String {
        let collapsed = text
            .split(whereSeparator: { $0.isWhitespace || $0.isNewline })
            .joined(separator: " ")
        guard !collapsed.isEmpty else { return untitled }
        guard collapsed.count > limit else { return collapsed }
        let clipped = String(collapsed.prefix(limit))
        // Cut at the last space when there is one to cut at; CJK text has none,
        // and clipping mid-run is correct there.
        if let space = clipped.lastIndex(of: " "), clipped.distance(
            from: clipped.startIndex, to: space) > limit / 2
        {
            return String(clipped[clipped.startIndex..<space]) + "…"
        }
        return clipped + "…"
    }

    /// Called after a user message is appended. Does nothing once a human has
    /// named the conversation.
    mutating func deriveTitleIfNeeded() {
        guard !titleIsCustom else { return }
        guard let first = messages.first(where: { $0.role == .user }) else { return }
        title = Conversation.derivedTitle(from: first.text)
    }

    mutating func rename(to newTitle: String) {
        let trimmed = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        title = trimmed
        titleIsCustom = true
        updatedAt = Date()
    }

    // MARK: Turns

    mutating func append(_ message: Message) {
        messages.append(message)
        if message.role == .user { deriveTitleIfNeeded() }
        updatedAt = message.createdAt
    }

    /// The last assistant turn, dropped so it can be run again. Returns the user
    /// message the regenerated turn answers.
    mutating func removeLastAssistantTurn() -> Message? {
        if let index = messages.lastIndex(where: { $0.role == .assistant }) {
            messages.remove(at: index)
        }
        updatedAt = Date()
        return messages.last(where: { $0.role == .user })
    }

    var lastUserMessage: Message? { messages.last(where: { $0.role == .user }) }

    var isEmpty: Bool { messages.isEmpty }

    /// True when the last thing in the transcript is a user message with no
    /// answer — which is what a stopped or failed turn leaves behind.
    var awaitsAnswer: Bool { messages.last?.role == .user }

    /// Closes a turn that was durable before the process disappeared, but
    /// never received a durable terminal assistant message.
    ///
    /// Product code persists the user message only after run preparation has
    /// succeeded and immediately before starting the runner. A user message at
    /// the end of a document on the next launch therefore means one observable
    /// thing: the previous process did not persist a terminal result. It does
    /// not prove why. Record exactly that, make the turn retryable, and never
    /// manufacture output or telemetry.
    @discardableResult
    mutating func recordInterruptedReplyIfNeeded(at date: Date) -> Bool {
        guard awaitsAnswer else { return false }
        append(
            Message(
                role: .assistant,
                text: "",
                createdAt: date,
                namedError:
                    "interrupted — the previous app session ended before this reply "
                    + "reached a terminal result",
                failure: .interrupted))
        return true
    }
}

// MARK: - Messages

struct Message: Codable, Equatable, Sendable, Identifiable {

    enum Role: String, Codable, Sendable { case user, assistant }

    let id: UUID
    let role: Role
    var text: String
    let createdAt: Date
    /// Assistant turns only: the compact record of what the run cost.
    var telemetry: MessageTelemetry?
    /// Assistant turns only: the named error, verbatim, when the turn did not
    /// complete. A turn that failed is kept in the transcript rather than
    /// deleted — the run is not discarded, and neither is its record.
    var namedError: String?
    /// Product-safe recovery semantics captured from the structured terminal
    /// error while it still exists. The raw error remains available for exact
    /// diagnosis, but a future launch never has to parse prose to decide which
    /// action is safe.
    var failure: MessageFailure?
    /// The token ids the runner actually produced, kept because this build has
    /// no vocabulary and the ids are the honest output.
    var tokenIDs: [Int]

    init(
        id: UUID = UUID(), role: Role, text: String, createdAt: Date = Date(),
        telemetry: MessageTelemetry? = nil, namedError: String? = nil,
        failure: MessageFailure? = nil, tokenIDs: [Int] = []
    ) {
        self.id = id
        self.role = role
        self.text = text
        self.createdAt = createdAt
        self.telemetry = telemetry
        self.namedError = namedError
        self.failure = failure
        self.tokenIDs = tokenIDs
    }
}

/// What a failed assistant turn lets a person do next.
///
/// This is deliberately smaller than `RunError`: it is persisted product
/// state, not a second error hierarchy. Each case names only an action that is
/// safe after that class of terminal event. In particular, a short read cannot
/// become a resumable run and a stopped turn cannot become an integrity alarm.
enum MessageFailure: String, Codable, Equatable, Sendable {
    case stopped
    case interrupted
    case retryAfterStorageReturns
    case verifyModelFiles
    case reviewModel
    case adjustChatSettings
    case informational
    case unknown

    var summary: String {
        switch self {
        case .stopped:
            return "Generation stopped."
        case .interrupted:
            return "The previous reply was interrupted."
        case .retryAfterStorageReturns:
            return "The model's storage became unavailable."
        case .verifyModelFiles:
            return "The local model files need to be checked."
        case .reviewModel:
            return "This model could not start the reply."
        case .adjustChatSettings:
            return "This chat's settings could not be used."
        case .informational:
            return "The run finished with a warning."
        case .unknown:
            return "Minirun could not complete this reply."
        }
    }

    var allowsRetry: Bool {
        switch self {
        case .interrupted, .retryAfterStorageReturns: return true
        case .stopped, .verifyModelFiles, .reviewModel, .adjustChatSettings,
            .informational, .unknown:
            return false
        }
    }
}

/// What an assistant turn cost, in the four numbers this product argues with.
///
/// Deliberately small: the full telemetry belongs to the run and is gone when
/// the run is, but a conversation that cannot say what a turn cost would be a
/// chat app with an oscilloscope bolted on.
struct MessageTelemetry: Codable, Equatable, Sendable {
    /// The runtime that produced this turn. Optional so conversations written
    /// before model identity was persisted remain readable; new turns always
    /// record it and therefore survive a later model switch without being
    /// misattributed.
    var model: ModelID? = nil
    var wallSeconds: Double
    var tokensPerSecond: Double?
    var bytesPerToken: UInt64?
    var peakFootprintBytes: UInt64
    var declaredBudgetBytes: UInt64
    var budgetRespected: Bool
    var logitsDigest: String?
    var thermalState: String?
    /// Where every pass of this turn spent its wall time, prefill first.
    ///
    /// Optional because conversations written before the engines published a
    /// decomposition have none, and because a runner that cannot attribute a
    /// pass publishes nothing rather than zeros. A 64-token turn is 65 entries
    /// of ~15 terms — a few tens of kilobytes, which is what a turn's
    /// attribution is worth against having to rerun the phone to get it back.
    var phases: [RunPhaseSummary]? = nil
    /// The prompt-ingestion pass, lifted out of ``phases`` because it is the
    /// longest single pass of a turn and is never decode throughput.
    var prefillPhase: RunPhaseSummary? = nil
    /// Every decode pass of this turn summed into one. Kept beside the array
    /// rather than derived at read time so a record answers "where did a decode
    /// token go" without a reader re-deriving an aggregate a later version
    /// might derive differently.
    var decodePhaseAggregate: RunPhaseSummary? = nil

    /// One line, in the order a reader asks the questions: how long, how fast,
    /// how many bytes it cost, and whether the budget held.
    var line: String {
        var parts = [MRFormat.clock(wallSeconds)]
        if let tokensPerSecond { parts.append(MRFormat.tokenRate(tokensPerSecond)) }
        if let bytesPerToken { parts.append("\(MRFormat.bytesDecimal(bytesPerToken))/token") }
        parts.append(
            "peak \(MRFormat.bytesDecimal(peakFootprintBytes)) of "
                + "\(MRFormat.bytesDecimal(declaredBudgetBytes))")
        return parts.joined(separator: " · ")
    }

    var spokenLine: String {
        "took \(MRFormat.clock(wallSeconds)), peak footprint "
            + "\(MRFormat.bytesDecimal(peakFootprintBytes)) against a stated "
            + "\(MRFormat.bytesDecimal(declaredBudgetBytes)), budget "
            + (budgetRespected ? "respected" : "exceeded")
    }
}

// MARK: - Per-conversation settings

/// Everything that affects how this conversation's turns run.
///
/// If a control changes what a run does, it belongs in this struct. The rule has
/// one consequence worth stating: nothing here may be read from a global store
/// at run time. `NewChatDefaults` seeds it once, at creation.
struct ConversationSettings: Codable, Equatable, Sendable {

    /// Which model answers. Changing it mid-conversation is allowed and is
    /// recorded per turn by the telemetry each message carries.
    var model: ModelID
    /// The stated budget for this conversation's runs. Never clamped.
    var memoryBudgetBytes: UInt64
    /// Tokens one turn may produce, clamped by the runner's own ceiling at
    /// validation time — refused by name, not truncated.
    var maximumNewTokens: Int
    /// The prefetch knobs, all optional: nil means "the runner's default",
    /// which is a different statement from a number the operator chose.
    var deterministicReadAheadLayers: Int?
    var expertReadAhead: Int?
    var expertPoolSlots: Int?
    var telemetryDensity: TelemetryDensity

    init(
        model: ModelID, memoryBudgetBytes: UInt64, maximumNewTokens: Int = 1,
        deterministicReadAheadLayers: Int? = nil, expertReadAhead: Int? = nil,
        expertPoolSlots: Int? = nil, telemetryDensity: TelemetryDensity = .strip
    ) {
        self.model = model
        self.memoryBudgetBytes = memoryBudgetBytes
        self.maximumNewTokens = maximumNewTokens
        self.deterministicReadAheadLayers = deterministicReadAheadLayers
        self.expertReadAhead = expertReadAhead
        self.expertPoolSlots = expertPoolSlots
        self.telemetryDensity = telemetryDensity
    }

    /// The knobs, as the runner's own type. Only the ones that were set appear,
    /// so a runner that cannot honour one refuses it BY NAME rather than never
    /// hearing about it.
    var knobs: RunKnobs {
        var knobs = RunKnobs()
        knobs.deterministicReadAheadLayers = deterministicReadAheadLayers
        knobs.expertReadAhead = expertReadAhead
        knobs.expertPoolSlots = expertPoolSlots
        return knobs
    }

    /// Whether the run will prefetch. An unset knob prefetches — that is the
    /// runner's default since spec v0.6.19 — so this is not `?? 0`.
    var readAheadIsOn: Bool {
        let setting = ReadAheadSetting(deterministicReadAheadLayers)
        return setting.isProductSupported && setting.effectiveDepth == 1
    }
}

/// Whether the live instruments panel opens automatically for a new run.
///
/// The raw values remain stable for conversations written by older versions.
enum TelemetryDensity: String, Codable, Sendable, CaseIterable, Identifiable {
    /// Keep the transcript in place until the user opens Instruments.
    case strip
    /// Open Instruments when generation begins.
    case cockpit

    var id: String { rawValue }

    var label: String {
        switch self {
        case .strip: return "Manual"
        case .cockpit: return "Automatic"
        }
    }

    var explanation: String {
        switch self {
        case .strip:
            return "Open live performance metrics when you want to inspect them."
        case .cockpit:
            return "Open live performance metrics when generation begins."
        }
    }
}

// MARK: - Defaults for new conversations

/// The only run-shaped values global settings is allowed to hold, and they are
/// held as *defaults*: copied onto a conversation when it is created and never
/// read again. Changing one changes the next chat, not this one.
struct NewChatDefaults: Codable, Equatable, Sendable {
    /// 4 retires the first-launch seed for models whose new-chat default is a
    /// position on a ladder rather than a fixed floor. The seeded number is
    /// dropped so it can be re-derived; anything else stays.
    static let currentBudgetSchemaVersion = 4
    static let currentInstrumentPanelSchemaVersion = 1

    var model: ModelID
    /// Per model, because the floor of one model is a refusal for another.
    var budgetBytes: [String: UInt64]
    var telemetryDensity: TelemetryDensity
    var deterministicReadAheadLayers: Int?
    /// Missing in defaults written before product chat accounted for the K3
    /// rolling state retained by the second token pass.
    var budgetSchemaVersion: Int?
    /// Missing while the product default was manual. Versioning this separately
    /// lets one launch adopt automatic live instruments without rewriting an
    /// operator choice made after that migration.
    var instrumentPanelSchemaVersion: Int?

    init(
        model: ModelID = .kimiK3, budgetBytes: [String: UInt64] = [:],
        telemetryDensity: TelemetryDensity = .cockpit,
        deterministicReadAheadLayers: Int? = nil,
        budgetSchemaVersion: Int? = Self.currentBudgetSchemaVersion,
        instrumentPanelSchemaVersion: Int? = Self.currentInstrumentPanelSchemaVersion
    ) {
        self.model = model
        self.budgetBytes = budgetBytes
        self.telemetryDensity = telemetryDensity
        self.deterministicReadAheadLayers = deterministicReadAheadLayers
        self.budgetSchemaVersion = budgetSchemaVersion
        self.instrumentPanelSchemaVersion = instrumentPanelSchemaVersion
    }

    func budget(for id: ModelID) -> UInt64? { budgetBytes[id.rawValue] }

    mutating func setBudget(_ bytes: UInt64, for id: ModelID) {
        budgetBytes[id.rawValue] = bytes
    }

    /// Forget a stored budget so the app derives one again. Used only by the
    /// schema migration that retired an automatic seed; an operator's own value
    /// is never removed.
    mutating func clearBudget(for id: ModelID) {
        budgetBytes.removeValue(forKey: id.rawValue)
    }

    /// The settings a new conversation starts life with. This is the whole of
    /// the inheritance: after this call the conversation owns its values.
    func settings(for id: ModelID, fallbackBudget: UInt64, maximumNewTokens: Int)
        -> ConversationSettings
    {
        ConversationSettings(
            model: id,
            memoryBudgetBytes: budget(for: id) ?? fallbackBudget,
            maximumNewTokens: maximumNewTokens,
            deterministicReadAheadLayers: deterministicReadAheadLayers,
            telemetryDensity: telemetryDensity)
    }
}
