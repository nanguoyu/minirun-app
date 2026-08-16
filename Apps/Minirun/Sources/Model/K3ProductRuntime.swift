import Foundation
import MinirunKit
import MinirunRunners
import ModelAdapters

/// The K3 tokenizer loaded from the same fully verified artifact the runner
/// will consume.
///
/// No App release is tied to one artifact commit. A live catalog resolves the
/// repository to an immutable commit, full verification binds every file at
/// that commit, and this initializer accepts the tokenizer declared by that
/// verified index only when its bytes, SHA-256 and K3 vocabulary geometry all
/// agree. Updating the owned HF repository therefore needs no App update while
/// a mutable branch can never become runtime authority by itself.
final class K3ProductTokenizer: VerifiedPromptTokenizing, @unchecked Sendable {
    let verifiedIdentity: VerifiedTokenizerIdentity
    let provenance: String
    let appliesChatTemplate = true

    private let vocabulary: TikTokenVocabulary

    private var terminalTokenIDs: Set<Int> {
        let base = vocabulary.baseTokenCount
        return [
            base + K3SpecialToken.endOfSequence.rawValue,
            base + K3SpecialToken.endOfMessage.rawValue,
            base + K3SpecialToken.close.rawValue,
        ]
    }

    init(artifact: ArtifactReference) throws {
        guard let authority = artifact.runtimeAuthority else {
            throw RunError.artifactNotReady(
                "K3 tokenizer loading requires complete rooted verification authority")
        }
        let evidence = authority.evidence
        guard evidence.model == .kimiK3,
            evidence.repository.repoID == "nanguoyu/Kimi-K3-minirun",
            evidence.index.format == "quantized_tile_container",
            evidence.index.relationship == "byte_preserving_repack",
            let tokenizer = evidence.index.tokenizer,
            tokenizer.sourceRepo.caseInsensitiveCompare("moonshotai/Kimi-K3") == .orderedSame
        else {
            throw RunError.artifactNotReady(
                "the verified artifact does not publish a Kimi K3 tokenizer identity")
        }

        let file = try authority.openFile(tokenizer.file)
        let licenseFile = try authority.openFile(tokenizer.license)
        try licenseFile.validateCurrentIdentity()
        guard file.expectedSizeBytes == tokenizer.bytes else {
            throw RunError.artifactNotReady(
                "the tokenizer index declares \(tokenizer.bytes) bytes, but the verified file has \(file.expectedSizeBytes)")
        }
        let data = try file.readAll(maximumBytes: 16 << 20)
        let digest = FileDigestComputer.sha256(of: data)
        guard digest == tokenizer.sha256 else {
            throw RunError.artifactNotReady(
                "the tokenizer SHA-256 does not match the identity published by index.json")
        }
        let vocabulary = try TikTokenVocabulary(data: data)
        guard vocabulary.baseTokenCount == 163_584,
            vocabulary.vocabularySize == 163_840,
            vocabulary.canEncode
        else {
            throw RunError.artifactNotReady(
                "the verified tokenizer does not have Kimi K3's 163,584 + 256 vocabulary")
        }

        self.vocabulary = vocabulary
        self.verifiedIdentity = VerifiedTokenizerIdentity(
            model: .kimiK3,
            artifactRepository: evidence.repository.repoID,
            artifactRevision: evidence.repository.revision,
            sourceRepository: tokenizer.sourceRepo,
            sourceRevision: tokenizer.sourceRevision,
            sha256: tokenizer.sha256)
        self.provenance =
            "Kimi K3 no-thinking chat · artifact \(evidence.repository.repoID)@"
            + "\(evidence.repository.revision.prefix(12)) · tokenizer \(tokenizer.sourceRepo)@"
            + "\(tokenizer.sourceRevision.prefix(12)) · SHA-256 \(tokenizer.sha256.prefix(12))…"
    }

    func encode(turns: [PromptTurn]) throws -> [Int] {
        let messages = try turns.map { turn in
            let role: K3ChatRole = turn.role == .user ? .user : .assistant
            return try K3ChatMessage(role: role, content: turn.text)
        }
        return try K3NoThinkingChatPrompt(
            messages: messages, addAssistantGenerationPrefix: true
        ).encode(using: vocabulary)
    }

    func decode(tokenIDs: [Int]) -> String {
        // The terminal token remains in the runner's summary so its final
        // logits digest names the token those logits actually selected. It is
        // protocol framing rather than assistant prose, so presentation stops
        // immediately before it.
        vocabulary.decode(Array(tokenIDs.prefix { !terminalTokenIDs.contains($0) }))
    }
}

extension K3DecodeRunner: ProductRunnerFacade {}

private enum K3ProductRuntimePaths {
    static var preparationDirectory: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("minirun/runtime-preparation", isDirectory: true)
    }
}

private enum K3ProductRuntimeProvider {
    static let registration = ProductModelRuntimeProvider(model: .kimiK3) {
        let runner = K3DecodeRunner()
        return ModelRuntime.verified(
            capabilities: runner.capabilities,
            acceptsArtifact: { index in
                guard index.shape == .singleSourceTotals,
                    index.format == "quantized_tile_container",
                    index.relationship == "byte_preserving_repack",
                    (index.declaredFileCount ?? 0) > 0,
                    (index.declaredBytes ?? 0) > 0,
                    index.repositories.contains(where: {
                        $0.repoID.caseInsensitiveCompare("moonshotai/Kimi-K3") == .orderedSame
                    }),
                    let tokenizer = index.tokenizer
                else { return false }
                return tokenizer.sourceRepo.caseInsensitiveCompare("moonshotai/Kimi-K3")
                        == .orderedSame
                    && tokenizer.file == "tiktoken.model"
            },
            inspectArtifact: {
                let profile = try K3DecodeRunner.inspectArtifactRuntimeProfile(
                    $0, workingDirectory: K3ProductRuntimePaths.preparationDirectory)
                return PreparedArtifactRuntimeInspection(
                    census: profile.census,
                    workingSetReserveBytes: { promptTokens, maximumNewTokens in
                        try K3ProductMemoryBudget.workingReserveBytes(
                            state: profile.generationState,
                            promptTokenCount: promptTokens,
                            maximumNewTokens: maximumNewTokens)
                    })
            },
            makeTokenizer: { try K3ProductTokenizer(artifact: $0) },
            // **K3's runner scale does not follow the dial, and that is the
            // decision, not an oversight.** V4 needed the conversation's budget
            // at the runner because nothing else carried it there. K3's dial
            // already arrives twice over: the request states the budget, and
            // `MemoryDialPlanner` turns that same budget into the `PinPlan`
            // this runner validates and consumes, so a wider budget already
            // pins more layers.
            //
            // Restating the budget as `.stated` here would buy nothing and
            // spend something. K3's effective knobs are resolved from `RunKnobs`
            // alone, so there is no budget-linked cache to unlock; what
            // `.stated` does do is clear `usesProductMemoryBudget`, which drops
            // the exact KDA/MLA product state reserve and the fail-closed
            // working-floor gate. A chat that raised its budget would then run
            // under a weaker gate than one that left it alone. Every K3 turn
            // therefore stays on the flagship product boundary.
            makeRunner: { _ in K3DecodeRunner() })
    }
}

/// The single composition point for shipping model families. Providers own
/// their complete tokenizer/runner/evidence contract; this list only composes
/// them. H3 is intentionally not a Minirun chat product target.
private enum ProductRuntimeProviders {
    static var all: [ProductModelRuntimeProvider] {
        #if arch(arm64)
            return [
                K3ProductRuntimeProvider.registration,
                DeepSeekV4ProductRuntimeProvider.registration,
            ]
        #else
            // Both shipping runners require MLX. Keep the universal macOS
            // binary launchable on Intel, but never advertise an executable
            // runtime on an architecture MLX does not support.
            return []
        #endif
    }
}

extension ModelRuntimeRegistry {
    static var product: ModelRuntimeRegistry {
        ModelRuntimeRegistry(providers: ProductRuntimeProviders.all)
    }
}
