import Foundation
import MinirunKit
import MinirunRunners
import ModelAdapters

/// The product-side half of DeepSeek V4's runtime binding.
///
/// Repository identity discovers a possible V4 artifact; it never authorizes
/// one compiled revision. The current live catalog resolves an immutable
/// publication, complete verification binds that exact tree, and this loader
/// accepts only the configuration, tokenizer and licence declared by the
/// verified index. The real V4 runner repeats the same source and rooted-file
/// checks before it yields `accepted`.
final class DeepSeekV4ProductTokenizer: VerifiedPromptTokenizing, @unchecked Sendable {
    let verifiedIdentity: VerifiedTokenizerIdentity
    let provenance: String
    let appliesChatTemplate = true

    private let vocabulary: DeepSeekV4Vocabulary
    private let terminalTokenIDs: Set<Int>

    init(artifact: ArtifactReference) throws {
        guard let authority = artifact.runtimeAuthority else {
            throw RunError.artifactNotReady(
                "DeepSeek V4 tokenizer loading requires complete rooted verification authority")
        }
        let evidence = authority.evidence
        guard evidence.model == .deepseekV4Flash,
            evidence.repository.repoID.caseInsensitiveCompare(
                DeepSeekV4ProductArtifact.publicationRepository) == .orderedSame,
            DeepSeekV4ProductArtifact.isImmutableRevision(evidence.repository.revision),
            DeepSeekV4ProductArtifact.accepts(evidence.index),
            let source = evidence.index.repositories.first,
            let sourceRevision = source.revision,
            let tokenizerIdentity = evidence.index.tokenizer
        else {
            throw RunError.artifactNotReady(
                "the verified artifact does not publish a supported DeepSeek V4 chat identity")
        }

        let configuration = try DeepSeekV4ProductArtifact.loadConfiguration(from: artifact)
        let tokenizerData = try DeepSeekV4ProductArtifact.verifiedData(
            file: tokenizerIdentity.file,
            expectedBytes: tokenizerIdentity.bytes,
            expectedSHA256: tokenizerIdentity.sha256,
            beneath: authority,
            maximumBytes: 32 << 20)
        let licence = try authority.openFile(tokenizerIdentity.license)
        try licence.validateCurrentIdentity()

        let vocabulary = try DeepSeekV4Vocabulary(data: tokenizerData)
        guard vocabulary.vocabularySize == configuration.vocabularySize else {
            throw RunError.artifactNotReady(
                "the V4 tokenizer and model configuration declare different vocabulary sizes")
        }

        var controlIDs = Set<Int>()
        for control in DeepSeekV4ChatControl.allCases {
            controlIDs.insert(try vocabulary.id(for: control.rawValue))
        }
        let beginning = try vocabulary.id(
            for: DeepSeekV4ChatControl.beginningOfSentence.rawValue)
        let end = try vocabulary.id(
            for: DeepSeekV4ChatControl.endOfSentence.rawValue)
        guard beginning == configuration.bosTokenID,
            end == configuration.eosTokenID
        else {
            throw RunError.artifactNotReady(
                "the V4 chat controls do not match config.json bos/eos token ids")
        }
        try authority.validateCurrentBinding()

        self.vocabulary = vocabulary
        self.terminalTokenIDs = controlIDs
        self.verifiedIdentity = VerifiedTokenizerIdentity(
            model: .deepseekV4Flash,
            artifactRepository: evidence.repository.repoID,
            artifactRevision: evidence.repository.revision,
            sourceRepository: source.repoID,
            sourceRevision: sourceRevision,
            sha256: tokenizerIdentity.sha256)
        self.provenance =
            "DeepSeek V4 no-thinking chat · artifact \(evidence.repository.repoID)@"
            + "\(evidence.repository.revision.prefix(12)) · tokenizer \(source.repoID)@"
            + "\(sourceRevision.prefix(12)) · SHA-256 "
            + "\(tokenizerIdentity.sha256.prefix(12))…"
    }

    func encode(turns: [PromptTurn]) throws -> [Int] {
        let messages = turns.map { turn in
            DeepSeekV4ChatMessage(
                role: turn.role == .user ? .user : .assistant,
                content: turn.text)
        }
        return try DeepSeekV4NoThinkingChatPrompt(
            messages: messages, addAssistantGenerationPrefix: true
        ).encode(using: vocabulary)
    }

    func decode(tokenIDs: [Int]) -> String {
        // Generated structural controls are framing, never assistant prose.
        // Stop at the first one rather than rendering a tokenizer marker into
        // the transcript or accepting a forged second role.
        vocabulary.decode(Array(tokenIDs.prefix { !terminalTokenIDs.contains($0) }))
    }

}

/// Cheap structural compatibility used before expensive preparation.
///
/// Counts and revisions are deliberately not compiled constants. A later
/// owner publication can move to a new immutable artifact and source commit;
/// it remains eligible when it still publishes the same supported V4 contract.
enum DeepSeekV4ProductArtifact {
    static let publicationRepository = "nanguoyu/DeepSeek-V4-Flash-0731-minirun"
    static let sourceRepository = "deepseek-ai/DeepSeek-V4-Flash-0731"

    static func accepts(_ index: ArtifactIndexIdentity) -> Bool {
        guard index.shape == .singleSourceUnits,
            index.relationship == "byte-preserving repack",
            (index.declaredFileCount ?? 0) > 0,
            (index.declaredBytes ?? 0) > 0,
            index.repositories.count == 1,
            let source = index.repositories.first,
            source.repoID.caseInsensitiveCompare(sourceRepository) == .orderedSame,
            let sourceRevision = source.revision,
            isImmutableRevision(sourceRevision),
            let configuration = index.configuration,
            let tokenizer = index.tokenizer
        else { return false }
        return configuration.sourceRepo.caseInsensitiveCompare(source.repoID) == .orderedSame
            && configuration.sourceRevision == sourceRevision
            && tokenizer.sourceRepo.caseInsensitiveCompare(source.repoID) == .orderedSame
            && tokenizer.sourceRevision == sourceRevision
    }

    /// Read the exact configuration already covered by complete verification.
    /// Its layer count drives progress only; it does not manufacture the
    /// per-layer byte census required by a memory-residency plan.
    static func loadConfiguration(from artifact: ArtifactReference) throws -> DeepSeekV4Config {
        guard let authority = artifact.runtimeAuthority else {
            throw RunError.artifactNotReady(
                "DeepSeek V4 configuration loading requires rooted verification authority")
        }
        let evidence = authority.evidence
        guard evidence.model == .deepseekV4Flash,
            evidence.repository.repoID.caseInsensitiveCompare(publicationRepository)
                == .orderedSame,
            isImmutableRevision(evidence.repository.revision),
            accepts(evidence.index),
            let identity = evidence.index.configuration
        else {
            throw RunError.artifactNotReady(
                "the verified artifact does not publish a supported DeepSeek V4 configuration")
        }
        let data = try verifiedData(
            file: identity.file,
            expectedBytes: identity.bytes,
            expectedSHA256: identity.sha256,
            beneath: authority,
            maximumBytes: 1 << 20)
        return try DeepSeekV4Config(json: data)
    }

    static func verifiedData(
        file: String, expectedBytes: UInt64, expectedSHA256: String,
        beneath authority: ArtifactRuntimeAuthority, maximumBytes: UInt64
    ) throws -> Data {
        let verified = try authority.openFile(file)
        guard verified.expectedSizeBytes == expectedBytes else {
            throw RunError.artifactNotReady(
                "\(file) has a different size from the identity published by index.json")
        }
        let data = try verified.readAll(maximumBytes: maximumBytes)
        guard UInt64(data.count) == expectedBytes,
            FileDigestComputer.sha256(of: data) == expectedSHA256
        else {
            throw RunError.artifactNotReady(
                "\(file) does not match the size and SHA-256 published by index.json")
        }
        return data
    }

    static func isImmutableRevision(_ value: String) -> Bool {
        value.count == 40 && value.utf8.allSatisfy {
            (48...57).contains($0) || (97...102).contains($0)
        }
    }
}

// This marker closes the runner half of the typed product boundary. Product
// registration remains conditional on the same fully verified artifact also
// satisfying the tokenizer binding below; either half on its own is not
// product availability.
extension DeepSeekV4DecodeRunner: ProductRunnerFacade {}

enum DeepSeekV4ProductRuntimeProvider {
    /// **The runner scale follows the conversation's stated budget.**
    ///
    /// The memory dial in Chat settings is a statement about memory, and until
    /// this decision existed it reached the *request* but never the *runner*:
    /// every App run was built at `.product`, whose MLX cache default is a
    /// fixed 0. So the budget-linked cache added with the stated scale —
    /// `min(budget / 10, 512 MiB)`, with boundary reclaim waiting for 90 % of
    /// the budget instead of firing at every boundary — was unreachable from
    /// the App, and a 2 GB chat and an 8 GB chat ran byte-identically.
    ///
    /// * **budget == the product floor** — the dial's "Floor" preset, which is
    ///   `DeepSeekV4ProductMemoryBudget.minimumBudgetBytes` for V4 and is what
    ///   a chat that never touched the dial carries → `.product`: the fully
    ///   gated conservative path, the 512-token prompt limit, the complete
    ///   product memory plan, and a zero cache. Unchanged behaviour.
    /// * **budget > the floor** → `.stated`, carrying that budget. The run then
    ///   gets the stated-scale admission — the admitted routed-expert pool plus
    ///   the derived MLX cache must fit inside the declared budget — and the
    ///   budget-linked cache and reclaim policy above.
    /// * **budget < the floor** → still `.product`. The dial already refuses
    ///   that region, and a persisted conversation carrying a sub-floor budget
    ///   must meet that existing refusal. Passing the sub-floor number in as a
    ///   stated minimum would be the runner being talked into admitting a
    ///   budget by being handed it.
    ///
    /// The token ceiling is the product ceiling at every scale: the dial states
    /// memory, and it is not an authorization to generate past the boundary the
    /// V4 product path has evidence for.
    static func runnerScale(statedBudgetBytes: UInt64) -> DeepSeekV4DecodeRunner.Scale {
        guard statedBudgetBytes > DeepSeekV4ProductMemoryBudget.minimumBudgetBytes else {
            return .product
        }
        return .stated(
            minimumBudgetBytes: statedBudgetBytes,
            maximumNewTokens: DeepSeekV4ProductMemoryBudget.maximumNewTokens)
    }

    static let registration = ProductModelRuntimeProvider(model: .deepseekV4Flash) {
        // The registered capabilities stay the product-scale ones. "The minimum
        // this model requires" and "the most tokens it will produce" are facts
        // about the model, not about whichever conversation is open, so the
        // settings panel and the availability gate keep reading them from
        // `.product`. Only the per-run runner below follows the dial.
        let capabilities = DeepSeekV4DecodeRunner(scale: .product).capabilities
        return ModelRuntime.verified(
            capabilities: capabilities,
            acceptsArtifact: { DeepSeekV4ProductArtifact.accepts($0) },
            inspectArtifact: { artifact in
                let configuration = try DeepSeekV4ProductArtifact.loadConfiguration(
                    from: artifact)
                return PreparedArtifactRuntimeInspection(
                    // Still nil, and still for the reason ADR/design record
                    // §3.1 gives: V4's residency is not derivable from one
                    // stored-byte column, so it does not have an
                    // `ArtifactCensus` and must not be given a misleading one.
                    census: nil,
                    // The ladder the memory dial ranks, read from the layer
                    // manifests this artifact publishes. A failure here is not
                    // a failure to run: V4 runs at the product floor with
                    // nothing pinned, and a dial with no ladder simply offers
                    // no pinned tier.
                    deepSeekV4Ladder: try? DeepSeekV4MemoryDialInputs.inspect(artifact),
                    progressLayerCount: configuration.numberOfLayers)
            },
            makeTokenizer: { try DeepSeekV4ProductTokenizer(artifact: $0) },
            makeRunner: { conversation in
                DeepSeekV4DecodeRunner(
                    scale: runnerScale(
                        statedBudgetBytes: conversation.settings.memoryBudgetBytes))
            })
    }
}
