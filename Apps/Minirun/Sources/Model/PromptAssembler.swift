import Foundation
import MinirunKit

/// **The seam where the real chat template mounts.**
///
/// A chat template is a model's own property — its role markers, its special
/// token ids, its system-prompt convention. Inventing one here would produce a
/// transcript that looks right and tokenizes to something the model never saw,
/// which is precisely the class of quiet wrongness this product exists to
/// refuse. So this type does exactly two things and names what it is not doing:
///
/// 1. it orders the turns and hands them to a `PromptTokenizing` in the order
///    the model will see them;
/// 2. it records which tokenizer produced the ids, so a message can say where
///    its ids came from.
///
/// The product runtime registry supplies the model's verified template and
/// vocabulary. The preview registry supplies `PreviewTokenizer` explicitly;
/// the product has no fallback. Nothing else in the app knows how a prompt is
/// built.
enum PromptAssembler {

    /// The turns a request carries, in order, as the runner's own `Prompt`.
    static func assemble(
        conversation: Conversation, upToUserMessage messageID: UUID? = nil,
        tokenizer: any PromptTokenizing
    ) throws -> AssembledPrompt {
        var turns: [PromptTurn] = []
        for message in conversation.messages {
            turns.append(PromptTurn(role: message.role, text: message.text))
            if let messageID, message.id == messageID { break }
        }
        let ids = try tokenizer.encode(turns: turns)
        return AssembledPrompt(
            turns: turns, tokenIDs: ids, provenance: tokenizer.provenance,
            appliesChatTemplate: tokenizer.appliesChatTemplate)
    }
}

struct PromptTurn: Equatable, Sendable {
    let role: Message.Role
    let text: String
}

struct AssembledPrompt: Equatable, Sendable {
    let turns: [PromptTurn]
    let tokenIDs: [Int]
    /// One sentence naming what produced these ids. Shown to the operator,
    /// because "which tokenizer" is not a detail when the ids are the output.
    let provenance: String
    /// False until a real template is mounted here.
    let appliesChatTemplate: Bool

    var prompt: Prompt { .tokenIDs(tokenIDs) }
}

/// The tokenizer seam. A stand-in is compiled for previews and tests, but the
/// launched product cannot reach it: product registration requires a verified
/// runner and tokenizer as one binding.
protocol PromptTokenizing: Sendable {
    func encode(turns: [PromptTurn]) throws -> [Int]
    func decode(tokenIDs: [Int]) -> String
    var provenance: String { get }
    var appliesChatTemplate: Bool { get }
}
