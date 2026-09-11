import Foundation

/// The V4.1 text-chat controls this runner uses.
///
/// Taken from the container's own `encoding.py` at
/// `deepseek-ai/DeepSeek-V4.1-Flash@df42c109`, which the publication carries
/// beside the weights. The set is V4's four plus `<｜System｜>`: V4.1 supports a
/// *mid-conversation* system message, which is rendered with its own token and
/// which — this is the part a port gets wrong — then behaves like a user message
/// for the purpose of appending the assistant header.
///
/// Everything else `encoding.py` can render is deliberately absent: DSML tool
/// calls, the numeric reasoning-effort prefix, the six task tokens, the latest
/// reminder, and every image form have additional framing and parsing rules that
/// cannot be inferred from this safe subset. `DeepSeekV4ChatControl` made the
/// same choice for the same reason.
public enum DeepSeekV41ChatControl: String, CaseIterable, Sendable {
    case beginningOfSentence = "<｜begin▁of▁sentence｜>"
    case endOfSentence = "<｜end▁of▁sentence｜>"
    case system = "<｜System｜>"
    case user = "<｜User｜>"
    case assistant = "<｜Assistant｜>"
    case thinkingEnd = "</think>"
}

/// DeepSeek V4.1's official text-only chat rendering with thinking disabled.
///
/// `encode_messages(messages, thinking_mode="chat")` for the text path, read off
/// `encoding.py`'s `_encode_messages_text` and `render_message`:
///
/// ```text
/// bos
/// for each message:
///     system  at index 0 : content                     (no token: the file
///                                                       emits <｜System｜> only
///                                                       for a reasoning-effort
///                                                       prefix or a *mid*
///                                                       conversation system)
///     system  later      : <｜System｜> content
///     user               : <｜User｜> content
///     assistant          : content + eos
///     after a user (or a mid-conversation system), when an assistant turn
///     follows or one is being asked for:  <｜Assistant｜> </think>
/// ```
///
/// which, for a single user turn, is exactly what V4's renderer produces —
/// `bos`, `<｜User｜>`, the text, `<｜Assistant｜>`, `</think>`. That the two
/// agree at one turn is a fact about the templates and not an assumption: this
/// type states V4.1's rules independently, and
/// `DeepSeekV41ChatPromptTests` holds the rendering to the container's own
/// `encoding-test_output_*.txt` fixtures where they cover this subset.
public struct DeepSeekV41NoThinkingChatPrompt: Equatable, Sendable {
    public enum Role: String, Sendable {
        case system
        case user
        case assistant
    }

    public struct Message: Equatable, Sendable {
        public let role: Role
        public let content: String

        public init(role: Role, content: String) {
            self.role = role
            self.content = content
        }
    }

    public let messages: [Message]
    public let addAssistantGenerationPrefix: Bool

    public init(messages: [Message], addAssistantGenerationPrefix: Bool = true) {
        self.messages = messages
        self.addAssistantGenerationPrefix = addAssistantGenerationPrefix
    }

    private enum Segment {
        case text(String)
        case control(DeepSeekV41ChatControl)
    }

    public var rendered: String {
        segments.map {
            switch $0 {
            case .text(let value): return value
            case .control(let value): return value.rawValue
            }
        }.joined()
    }

    public func encode(using vocabulary: DeepSeekV4Vocabulary) throws -> [Int] {
        var ids: [Int] = []
        for segment in segments {
            switch segment {
            case .text(let value): ids.append(contentsOf: try vocabulary.encode(value))
            case .control(let value): ids.append(try vocabulary.id(for: value.rawValue))
            }
        }
        return ids
    }

    private var segments: [Segment] {
        var result: [Segment] = [.control(.beginningOfSentence)]
        for (index, message) in messages.enumerated() {
            switch message.role {
            case .system:
                // `render_message` emits the token for a mid-conversation
                // system message and *not* for one that opens the
                // conversation — the leading `<｜System｜>` there is the
                // reasoning-effort prefix's, which no-thinking mode never has.
                if index > 0 { result.append(.control(.system)) }
                if !message.content.isEmpty { result.append(.text(message.content)) }
            case .user:
                result.append(.control(.user))
                if !message.content.isEmpty { result.append(.text(message.content)) }
            case .assistant:
                if !message.content.isEmpty { result.append(.text(message.content)) }
                result.append(.control(.endOfSentence))
            }

            let nextRole = index + 1 < messages.count ? messages[index + 1].role : nil
            let opensAssistant = message.role == .user
                || (message.role == .system && index > 0)
            if opensAssistant,
                nextRole == .assistant || (nextRole == nil && addAssistantGenerationPrefix)
            {
                result.append(.control(.assistant))
                result.append(.control(.thinkingEnd))
            }
        }
        return result
    }
}
