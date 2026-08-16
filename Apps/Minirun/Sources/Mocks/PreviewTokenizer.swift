#if DEBUG
import Foundation
import MinirunKit

/// **Not a tokenizer.** A deterministic stand-in, so the multi-turn plumbing can
/// be exercised in a build that carries no vocabulary.
///
/// The rule is written down here rather than hidden: each turn contributes one
/// id per Unicode scalar cluster of four characters, folded into the range the
/// preview fixtures use. It is reproducible, it is stable across launches, and
/// it is meaningless — no id here corresponds to a word in any model's
/// vocabulary. Every screen that shows ids produced this way says where they
/// came from, and `appliesChatTemplate` is `false` because no template was
/// applied.
///
/// It is reachable only through `ModelRuntimeRegistry.preview`, which the app's
/// live factory never installs. A product runtime instead supplies a verified
/// vocabulary and the model's own chat template as one binding with its runner.
struct PreviewTokenizer: PromptTokenizing {

    let provenance =
        "ids from the preview stand-in — this build carries no vocabulary and applies no chat "
        + "template"

    let appliesChatTemplate = false

    func encode(turns: [PromptTurn]) throws -> [Int] {
        var ids: [Int] = []
        for turn in turns {
            // A role marker that is *labelled* a placeholder rather than
            // borrowed from some model's special-token table.
            ids.append(turn.role == .user ? 1 : 2)
            var accumulator = 0
            for (offset, scalar) in turn.text.unicodeScalars.enumerated() {
                accumulator = (accumulator &* 31 &+ Int(scalar.value)) % 120_000
                if offset % 4 == 3 {
                    ids.append(max(3, accumulator))
                    accumulator = 0
                }
            }
            if accumulator != 0 { ids.append(max(3, accumulator)) }
        }
        return ids.isEmpty ? [1] : ids
    }

    func decode(tokenIDs: [Int]) -> String {
        tokenIDs.map { "‹\($0)›" }.joined()
    }
}

/// The only construction path that binds preview ids to an invented runner.
/// `AppModel.live()` never calls this; SwiftUI previews and tests do so by name.
extension ModelRuntimeRegistry {
    static func preview(
        entries: [CatalogEntry], fault: MockDecodeRunner.Fault? = nil,
        artifactCensus: ArtifactCensus? = nil,
        workingSetReserveBytes: (@Sendable (Int, Int) throws -> UInt64)? = nil
    ) -> ModelRuntimeRegistry {
        let runtimes = entries.compactMap { entry -> ModelRuntime? in
            // This is fixture routing, not product capability. The product
            // ignores this descriptor field and consults its verified runtime
            // registry instead.
            guard entry.descriptor.runner == .decodeRunner else { return nil }
            let capabilities = MockDecodeRunner(entry: entry).capabilities
            return ModelRuntime.preview(
                capabilities: capabilities,
                tokenizer: PreviewTokenizer(),
                inspectArtifact: { _ in
                    PreparedArtifactRuntimeInspection(
                        census: artifactCensus,
                        workingSetReserveBytes: workingSetReserveBytes)
                }
            ) { conversation in
                let runner = MockDecodeRunner(entry: entry)
                runner.fault = fault
                runner.turnIndex = conversation.messages.filter { $0.role == .assistant }.count
                return runner
            }
        }
        return ModelRuntimeRegistry(preview: runtimes)
    }
}

/// The replies the preview runner produces, turn by turn.
///
/// PLACEHOLDER, all of it, and the reason it is shaped this way: K3 is on record
/// producing **one** new token per pass, so a K3 conversation genuinely advances
/// one token per turn. Rather than pretend otherwise, the fixture makes those
/// single tokens continue each other, so a multi-turn chat assembles a sentence
/// the way the real thing would — slowly, one token at a time. Token 3372
/// (`北京`) is first because that is the token the archived device record
/// produced, and the first turn of a preview conversation should be the
/// demonstration the product was built on.
enum PreviewReplies {

    struct Continuation: Equatable, Sendable {
        let tokenID: Int
        let text: String
    }

    /// The K3 ladder: transcribed first token, invented afterwards.
    static let deterministic: [Continuation] = [
        Continuation(tokenID: 3372, text: "北京"),
        Continuation(tokenID: 3221, text: "是"),
        Continuation(tokenID: 15235, text: "中国"),
        Continuation(tokenID: 17529, text: "的"),
        Continuation(tokenID: 9042, text: "首都"),
        Continuation(tokenID: 511, text: "。"),
    ]

    /// A longer, plainer ladder for runners whose ceiling is more than one
    /// token, so streaming is visible as streaming.
    static let streamed: [Continuation] = [
        Continuation(tokenID: 785, text: "The"),
        Continuation(tokenID: 4128, text: " weights"),
        Continuation(tokenID: 2299, text: " stay"),
        Continuation(tokenID: 6710, text: " on"),
        Continuation(tokenID: 8871, text: " disk"),
        Continuation(tokenID: 11, text: ","),
        Continuation(tokenID: 323, text: " and"),
        Continuation(tokenID: 1172, text: " only"),
        Continuation(tokenID: 279, text: " the"),
        Continuation(tokenID: 6324, text: " layer"),
        Continuation(tokenID: 1660, text: " being"),
        Continuation(tokenID: 25157, text: " computed"),
        Continuation(tokenID: 374, text: " is"),
        Continuation(tokenID: 19995, text: " resident"),
        Continuation(tokenID: 13, text: "."),
    ]

    /// `count` tokens for the given turn, walking the ladder so turn 2 continues
    /// where turn 1 stopped.
    static func tokens(model: ModelID, turnIndex: Int, count: Int) -> [Continuation] {
        let ladder = model == .kimiK3 ? deterministic : streamed
        guard count > 0, !ladder.isEmpty else { return [] }
        let start = turnIndex * max(1, count)
        return (0..<count).map { ladder[(start + $0) % ladder.count] }
    }
}
#endif
