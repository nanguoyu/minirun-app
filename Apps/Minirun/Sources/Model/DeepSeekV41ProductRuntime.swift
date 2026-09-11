import Foundation
import MinirunKit
import MinirunRunners
import ModelAdapters
import StorageCore

/// The product-side half of DeepSeek V4.1's runtime binding.
///
/// ``DeepSeekV4ProductRuntime`` for V4.1, with the same rule and the same order:
/// repository identity discovers a possible artifact and never authorizes a
/// compiled revision; the live catalog resolves an immutable publication;
/// complete verification binds that exact tree; and this loader accepts only
/// the configuration, tokenizer and licence the verified index declares. The
/// runner repeats every one of those checks before it yields `accepted`.
///
/// Two things are V4.1's own. The configuration is a **pair** of documents —
/// `inference-config.json` is the authority the reference runner reads and
/// `config.json` is the cross-check (ADR 0020) — so this loader opens both and
/// a disagreement between them is a refusal. And the chat controls are V4's
/// four plus `<｜System｜>`, which V4.1's `encoding.py` uses for a
/// mid-conversation system message.
final class DeepSeekV41ProductTokenizer: VerifiedPromptTokenizing, @unchecked Sendable {
    let verifiedIdentity: VerifiedTokenizerIdentity
    let provenance: String
    let appliesChatTemplate = true

    private let vocabulary: DeepSeekV4Vocabulary
    private let terminalTokenIDs: Set<Int>

    init(artifact: ArtifactReference) throws {
        guard let authority = artifact.runtimeAuthority else {
            throw RunError.artifactNotReady(
                "DeepSeek V4.1 tokenizer loading requires complete rooted verification "
                    + "authority")
        }
        let evidence = authority.evidence
        guard evidence.model == .deepseekV41Flash,
            evidence.repository.repoID.caseInsensitiveCompare(
                DeepSeekV41ProductArtifact.publicationRepository) == .orderedSame,
            DeepSeekV41ProductArtifact.isImmutableRevision(evidence.repository.revision),
            DeepSeekV41ProductArtifact.accepts(evidence.index),
            let source = evidence.index.repositories.first,
            let sourceRevision = source.revision,
            let tokenizerIdentity = evidence.index.tokenizer
        else {
            throw RunError.artifactNotReady(
                "the verified artifact does not publish a supported DeepSeek V4.1 chat "
                    + "identity")
        }

        let configuration = try DeepSeekV41ProductArtifact.loadConfiguration(from: artifact)
        let tokenizerData = try DeepSeekV41ProductArtifact.verifiedData(
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
                "the V4.1 tokenizer and model configuration declare different vocabulary "
                    + "sizes")
        }

        var controlIDs = Set<Int>()
        for control in DeepSeekV41ChatControl.allCases {
            controlIDs.insert(try vocabulary.id(for: control.rawValue))
        }
        let beginning = try vocabulary.id(
            for: DeepSeekV41ChatControl.beginningOfSentence.rawValue)
        let end = try vocabulary.id(for: DeepSeekV41ChatControl.endOfSentence.rawValue)
        guard beginning == configuration.bosTokenID, end == configuration.eosTokenID else {
            throw RunError.artifactNotReady(
                "the V4.1 chat controls do not match the configuration's bos/eos token ids")
        }
        try authority.validateCurrentBinding()

        self.vocabulary = vocabulary
        self.terminalTokenIDs = controlIDs
        self.verifiedIdentity = VerifiedTokenizerIdentity(
            model: .deepseekV41Flash,
            artifactRepository: evidence.repository.repoID,
            artifactRevision: evidence.repository.revision,
            sourceRepository: source.repoID,
            sourceRevision: sourceRevision,
            sha256: tokenizerIdentity.sha256)
        self.provenance =
            "DeepSeek V4.1 no-thinking chat · artifact \(evidence.repository.repoID)@"
            + "\(evidence.repository.revision.prefix(12)) · tokenizer \(source.repoID)@"
            + "\(sourceRevision.prefix(12)) · SHA-256 "
            + "\(tokenizerIdentity.sha256.prefix(12))…"
    }

    func encode(turns: [PromptTurn]) throws -> [Int] {
        let messages = turns.map { turn in
            DeepSeekV41NoThinkingChatPrompt.Message(
                role: turn.role == .user ? .user : .assistant, content: turn.text)
        }
        return try DeepSeekV41NoThinkingChatPrompt(
            messages: messages, addAssistantGenerationPrefix: true
        ).encode(using: vocabulary)
    }

    func decode(tokenIDs: [Int]) -> String {
        vocabulary.decode(Array(tokenIDs.prefix { !terminalTokenIDs.contains($0) }))
    }
}

/// Cheap structural compatibility, checked before expensive preparation.
enum DeepSeekV41ProductArtifact {
    static let publicationRepository = "nanguoyu/DeepSeek-V4.1-Flash-minirun"
    static let sourceRepository = "deepseek-ai/DeepSeek-V4.1-Flash"

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

    /// Both configuration documents, decoded together.
    ///
    /// `index.json` names the Transformers `config.json` under `configuration`
    /// and the flat `inference-config.json` under `arguments`. ADR 0020 made
    /// the second the authority — it is the file DeepSeek's own runner reads —
    /// and every value both state is compared, so a publication whose two files
    /// disagree is refused here rather than run.
    static func loadConfiguration(
        from artifact: ArtifactReference
    ) throws -> DeepSeekV41Config {
        guard let authority = artifact.runtimeAuthority else {
            throw RunError.artifactNotReady(
                "DeepSeek V4.1 configuration loading requires rooted verification authority")
        }
        let evidence = authority.evidence
        guard evidence.model == .deepseekV41Flash,
            evidence.repository.repoID.caseInsensitiveCompare(publicationRepository)
                == .orderedSame,
            isImmutableRevision(evidence.repository.revision),
            accepts(evidence.index),
            let identity = evidence.index.configuration
        else {
            throw RunError.artifactNotReady(
                "the verified artifact does not publish a supported DeepSeek V4.1 "
                    + "configuration")
        }
        let huggingFace = try verifiedData(
            file: identity.file, expectedBytes: identity.bytes,
            expectedSHA256: identity.sha256, beneath: authority, maximumBytes: 1 << 20)
        let index = try DeepSeekV41ArtifactIndex(
            json: try authority.openFile("index.json").readAll(maximumBytes: 8 << 20))
        let inference = try authority.openFile(index.argumentsFile).readAll(
            maximumBytes: 1 << 20)
        return try DeepSeekV41Config(
            inferenceJSON: inference, huggingFaceJSON: huggingFace)
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

extension DeepSeekV41DecodeRunner: ProductRunnerFacade {}

enum DeepSeekV41ProductRuntimeProvider {
    /// The runner scale follows the conversation's stated budget, exactly as
    /// V4's does and for the same reason: the dial is a statement about memory
    /// and it has to reach the *runner*, not only the request.
    ///
    /// The token ceiling stays the product ceiling at either scale. The dial
    /// states memory; it is not an authorization to generate past the boundary
    /// this product path has evidence for.
    ///
    /// ## What a stated scale gives up, and why a phone may not give it up
    ///
    /// `.stated` is not "the same run with a bigger number". It turns the
    /// product memory policy **off**: `DeepSeekV41RunEngine` prices its
    /// requirement as the pool plus the MLX cache plus whatever the ladder
    /// pinned, the 512/64 prompt and reply limits stop being enforced, and the
    /// budget sentry's ceiling becomes the stated number rather than the
    /// platform floor. On a Mac that is the whole point: a stated budget buys
    /// resident blocks and the envelope the plan would have charged is
    /// measured, not guessed.
    ///
    /// On the owner's iPhone 16 Pro it was the kill. A conversation carrying
    /// the **Mac** floor of 3.4 GB — seeded by a build from before the iPhone
    /// policy existed — was above the iOS floor, so this function promoted it;
    /// the run then admitted itself against a 465 MB requirement, asked the
    /// device nothing it could refuse, and let the sentry allow 3.4 GB *above*
    /// a post-verification entry footprint. Jetsam took the app in prefill at
    /// 5,234 MB resident, `vm-pageshortage`.
    ///
    /// So where the platform states a per-process limit, a stated budget has to
    /// clear two gates rather than one:
    ///
    /// 1. **it must buy a rung.** Below ``DeepSeekV41MemoryDialInputs/smallestPinningBudgetBytes(policy:)``
    ///    nothing can be pinned at any census, so the stated scale would buy
    ///    nothing at all and cost the product policy;
    /// 2. **the device must be able to give it.** `os_proc_available_memory()`
    ///    is what the runner's own pre-flight refusal consults, and the promise
    ///    a stated budget makes is "this run may add this much, plus the stated
    ///    overshoot". A device that cannot fund that cannot fund the promise.
    ///
    /// A phone's entire process allowance — about 5.0 GB on the owner's
    /// iPhone 16 Pro — is a gigabyte short of the first gate's 6.0 GB, so every
    /// iPhone budget resolves to `.product` — the fully gated path, with
    /// the 1.9 GB floor, the MLX cache at zero and the plan enforced. On macOS
    /// `availableMemoryBytes` is nil, both gates are absent by construction, and
    /// nothing about the Mac's behaviour changes.
    static func runnerScale(
        statedBudgetBytes: UInt64,
        policy: DeepSeekV41ProductMemoryBudget.Policy = DeepSeekV41ProductMemoryBudget
            .currentPolicy,
        availableMemoryBytes: UInt64? = ProcessFootprint.availableMemory(),
        smallestPinningBudgetBytes: UInt64? = nil
    ) -> DeepSeekV41DecodeRunner.Scale {
        guard statedBudgetBytes > policy.minimumBudgetBytes else { return .product }
        if let available = availableMemoryBytes {
            let pinFloor = smallestPinningBudgetBytes
                ?? DeepSeekV41MemoryDialInputs.smallestPinningBudgetBytes(policy: policy)
            let needed = statedBudgetBytes.addingReportingOverflow(
                policy.budgetOvershootAllowanceBytes)
            guard statedBudgetBytes >= pinFloor, !needed.overflow,
                needed.partialValue <= available
            else { return .product }
        }
        return .stated(
            minimumBudgetBytes: statedBudgetBytes,
            maximumNewTokens: policy.maximumNewTokens)
    }

    static let registration = ProductModelRuntimeProvider(model: .deepseekV41Flash) {
        let capabilities = DeepSeekV41DecodeRunner(scale: .product).capabilities
        return ModelRuntime.verified(
            capabilities: capabilities,
            acceptsArtifact: { DeepSeekV41ProductArtifact.accepts($0) },
            inspectArtifact: { artifact in
                let configuration = try DeepSeekV41ProductArtifact.loadConfiguration(
                    from: artifact)
                return PreparedArtifactRuntimeInspection(
                    // Nil for the reason V4's is nil: V4.1's residency is not
                    // derivable from one stored-byte column either — a pinned
                    // block holds the loaded `BlockFP8Weights` form, whose
                    // scale grid is expanded — so it has no `ArtifactCensus`
                    // and must not be given a misleading one.
                    census: nil,
                    // The ladder the memory dial ranks, read from the block
                    // manifests this artifact publishes. It was nil until the
                    // dial reached V4.1, and nil here is not "no ladder yet" to
                    // a plan — it is K3's residency rules applied to a census of
                    // one zero-sized layer, which is how every V4.1 preset came
                    // out at the 3.4 GB floor. A failure here is still not a
                    // failure to run: V4.1 runs at that floor with nothing
                    // pinned, and a dial with no ladder offers no pinned tier.
                    deepSeekLadder: try? DeepSeekV41MemoryDialInputs.inspect(artifact),
                    progressLayerCount: configuration.numberOfLayers)
            },
            makeTokenizer: { try DeepSeekV41ProductTokenizer(artifact: $0) },
            makeRunner: { conversation in
                DeepSeekV41DecodeRunner(
                    scale: runnerScale(
                        statedBudgetBytes: conversation.settings.memoryBudgetBytes))
            })
    }
}
