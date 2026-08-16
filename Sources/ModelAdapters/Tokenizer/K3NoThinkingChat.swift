import Foundation

/// A K3 special token whose position is known from the measured tokenizer
/// configuration.
///
/// Raw values are offsets from ``TikTokenVocabulary/baseTokenCount``, not
/// checkpoint-specific absolute ids. The M9C tiny and flagship vocabularies
/// have the same 163,584 mergeable ranks, but using offsets here also makes the
/// special-token layer testable with a small synthetic vocabulary.
public enum K3SpecialToken: Int, CaseIterable, Sendable {
    case beginningOfSequence = 0
    case endOfSequence = 1
    case endOfMessage = 2
    case open = 3
    case close = 4
    case separator = 5
    case startHeader = 6
    case endHeader = 7
    case endOfTurn = 9
    case mediaBegin = 18
    case mediaContent = 19
    case mediaEnd = 20
    case mediaPad = 21
    case osAgentMode = 65
    case unknown = 254
    case padding = 255

    /// The spelling recorded by the K3 tokenizer configuration.
    public var name: String {
        switch self {
        case .beginningOfSequence: return "[BOS]"
        case .endOfSequence: return "[EOS]"
        case .endOfMessage: return "<|end_of_msg|>"
        case .open: return "<|open|>"
        case .close: return "<|close|>"
        case .separator: return "<|sep|>"
        case .startHeader: return "[start_header_id]"
        case .endHeader: return "[end_header_id]"
        case .endOfTurn: return "[EOT]"
        case .mediaBegin: return "<|media_begin|>"
        case .mediaContent: return "<|media_content|>"
        case .mediaEnd: return "<|media_end|>"
        case .mediaPad: return "<|media_pad|>"
        case .osAgentMode: return "<osagent_mode>"
        case .unknown: return "[UNK]"
        case .padding: return "[PAD]"
        }
    }
}

/// The text-only roles supported by K3's measured chat renderer.
///
/// Tool and multimodal messages are deliberately absent. Supporting them
/// requires additional framing rules and is not a harmless extension of the
/// no-thinking text template.
public enum K3ChatRole: String, CaseIterable, Sendable {
    case system
    case user
    case assistant

    /// Validates an untyped role at an API boundary instead of letting the
    /// reference renderer's "unknown role means skip it" behaviour leak into
    /// the runtime.
    public static func parse(_ value: String) throws -> Self {
        guard let role = Self(rawValue: value) else {
            let supported = allCases.map(\.rawValue).joined(separator: ", ")
            throw TinyK3Error.tokenizer(
                "unsupported K3 chat role '\(value)'; supported roles: \(supported)")
        }
        return role
    }
}

/// One completed text turn in a K3 no-thinking conversation.
///
/// Content and names remain ordinary BPE text. They may not smuggle an
/// `<|...|>` marker into the structural stream; the prompt builder inserts
/// control ids only through ``K3SpecialToken``.
public struct K3ChatMessage: Equatable, Sendable {
    public let role: K3ChatRole
    public let content: String
    public let name: String?

    public init(role: K3ChatRole, content: String, name: String? = nil) throws {
        try Self.refuseControlToken(in: content, field: "content")
        if let name {
            try Self.refuseControlToken(in: name, field: "name")
        }
        self.role = role
        self.content = content
        self.name = name
    }

    /// A validating boundary for dictionary/JSON callers that do not yet have
    /// a ``K3ChatRole``. Invalid roles throw; no message is silently omitted.
    public init(role: String, content: String, name: String? = nil) throws {
        try self.init(role: K3ChatRole.parse(role), content: content, name: name)
    }

    private static func refuseControlToken(in value: String, field: String) throws {
        guard let marker = TikTokenVocabulary.firstControlTokenMarker(in: value) else { return }
        throw TinyK3Error.tokenizer(
            "K3 chat \(field) contains control-token marker '\(marker)'; control tokens "
                + "may only be inserted by the typed chat template")
    }
}

enum K3ChatEncodingSegment: Equatable {
    case text(String)
    case control(K3SpecialToken)

    var rendered: String {
        switch self {
        case .text(let text): return text
        case .control(let token): return token.name
        }
    }
}

/// The measured K3 text chat format with thinking disabled.
///
/// This is intentionally a separate type rather than a `thinking: Bool`
/// option. The thinking template has not been ported or validated here, so the
/// API cannot accidentally claim to support it. Completed assistant turns use
/// the reference renderer's `<response>` wrapper; a generation prefix opens
/// the assistant message and response and leaves both open for decoding.
public struct K3NoThinkingChatPrompt: Equatable, Sendable {
    public let messages: [K3ChatMessage]
    public let addAssistantGenerationPrefix: Bool

    public init(
        messages: [K3ChatMessage],
        addAssistantGenerationPrefix: Bool = true
    ) {
        self.messages = messages
        self.addAssistantGenerationPrefix = addAssistantGenerationPrefix
    }

    /// The byte-visible prompt, useful for audit logs and reference comparison.
    /// Encoding does not feed this string back through ordinary BPE: it consumes
    /// the typed segments below, so structural markers become special-token ids.
    public var rendered: String {
        segments.map(\.rendered).joined()
    }

    /// Encode ordinary text with BPE and structural controls with their K3 ids.
    public func encode(using vocabulary: TikTokenVocabulary) throws -> [Int] {
        try vocabulary.encode(self)
    }

    var segments: [K3ChatEncodingSegment] {
        var result: [K3ChatEncodingSegment] = []

        for message in messages {
            Self.appendOpenTag(
                "message",
                attributes: [
                    ("role", message.role.rawValue),
                    message.name.flatMap { $0.isEmpty ? nil : ("name", $0) },
                ].compactMap { $0 },
                to: &result)

            if message.role == .assistant {
                Self.appendOpenTag("response", to: &result)
                Self.appendText(message.content, to: &result)
                Self.appendCloseTag("response", to: &result)
            } else {
                Self.appendText(message.content, to: &result)
            }

            Self.appendCloseTag("message", to: &result)
            result.append(.control(.endOfMessage))
        }

        if addAssistantGenerationPrefix {
            Self.appendOpenTag("message", attributes: [("role", "assistant")], to: &result)
            Self.appendOpenTag("response", to: &result)
        }

        return result
    }

    private static func appendText(_ text: String, to segments: inout [K3ChatEncodingSegment]) {
        guard !text.isEmpty else { return }
        segments.append(.text(text))
    }

    private static func appendOpenTag(
        _ tag: String,
        attributes: [(String, String)] = [],
        to segments: inout [K3ChatEncodingSegment]
    ) {
        segments.append(.control(.open))
        segments.append(.text(tag))
        for (key, value) in attributes {
            segments.append(.text(" \(key)=\"\(escapeAttribute(value))\""))
        }
        segments.append(.control(.separator))
    }

    private static func appendCloseTag(
        _ tag: String,
        to segments: inout [K3ChatEncodingSegment]
    ) {
        segments.append(.control(.close))
        segments.append(.text(tag))
        segments.append(.control(.separator))
    }

    /// Matches `encoding_k3.py` at the measured M9C revision: ampersands are
    /// escaped before quotes so a pre-existing entity is not left ambiguous.
    private static func escapeAttribute(_ value: String) -> String {
        value
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "\"", with: "&quot;")
    }
}

extension TikTokenVocabulary {
    /// The absolute vocabulary id for one known K3 special token.
    public func id(for token: K3SpecialToken) -> Int {
        baseTokenCount + token.rawValue
    }

    /// Encodes a typed no-thinking prompt without ever treating control
    /// markers as ordinary bytes.
    public func encode(_ prompt: K3NoThinkingChatPrompt) throws -> [Int] {
        var ids: [Int] = []
        for segment in prompt.segments {
            switch segment {
            case .text(let text):
                ids += try encode(text)
            case .control(let token):
                ids.append(id(for: token))
            }
        }
        return ids
    }
}
