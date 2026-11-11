import Foundation

/// The V4.1 text-chat controls this runner uses.
///
/// Taken from the container's own `encoding.py` at
/// `deepseek-ai/DeepSeek-V4.1-Flash@df42c109`, which the publication carries
/// beside the weights, and unchanged for this subset at `314113f8`
/// (`Tests/Fixtures/deepseek-v41/chat-encoding.json`). The set is V4's four plus `<｜System｜>`: V4.1 supports a
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
/// consecutive user messages are first merged into one, their texts joined
/// by a blank line ("\n\n") -- `merge_tool_messages`, which runs on every
/// conversation, not only on one with tools
/// for each message:
///     system  at index 0 : <｜System｜> content        (`render_message` puts
///                                                       the token first when
///                                                       the opening message is
///                                                       a system one)
///     system  later      : <｜System｜> content
///     user               : <｜User｜> content
///     assistant          : content + eos
///     after a user (or a mid-conversation system), when an assistant turn
///     follows or one is being asked for:  <｜Assistant｜> </think>
/// ```
///
/// which, for a single user turn, is exactly what V4's renderer produces:
/// `bos`, `<｜User｜>`, the text, `<｜Assistant｜>`, `</think>`. That the two
/// agree at one turn is a fact about the templates and not an assumption: this
/// type states V4.1's rules independently, and `DeepSeekV41ChatPromptTests`
/// holds the rendering to what upstream's own `encoding.py` renders, at both
/// `df42c109` and `314113f8`, for every conversation in
/// `Tests/Fixtures/deepseek-v41/chat-encoding.json`
/// (`Tools/v41_flash/make_chat_encoding_fixture.py`).
///
/// Message text is never allowed to carry a control. `encoding.py` renders a
/// typed `<｜User｜>` into its prompt string as text, and the tokenizer that
/// reads that string would turn it into the control; ``encode(using:)``
/// refuses instead, through ``DeepSeekV4Vocabulary/encode(_:)``.
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

    /// `merge_tool_messages`: a user message that follows a user message is
    /// folded into it, the texts joined by a blank line. It runs before
    /// rendering at both upstream revisions, for every conversation.
    private var merged: [Message] {
        var result: [Message] = []
        for message in messages {
            if message.role == .user, let last = result.last, last.role == .user {
                result[result.count - 1] = Message(
                    role: .user, content: last.content + "\n\n" + message.content)
            } else {
                result.append(message)
            }
        }
        return result
    }

    private var segments: [Segment] {
        var result: [Segment] = [.control(.beginningOfSentence)]
        let messages = merged
        for (index, message) in messages.enumerated() {
            switch message.role {
            case .system:
                // `render_message` emits the token for every system message:
                // at index 0 as `SYSTEM_SP_TOKEN if index == 0 and (... or
                // role == "system")`, later as the mid-conversation marker.
                result.append(.control(.system))
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
