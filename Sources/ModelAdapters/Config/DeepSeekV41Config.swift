import Foundation

/// Named failures at the DeepSeek V4.1 Flash architecture boundary.
///
/// V4.1 is a different architecture from V4-Flash-0731, not a later revision of
/// it: 40 layers instead of 43, 384 routed experts instead of 256, a [32, 32]
/// FP8 block instead of [128, 128], `compress_ratios` in {0, 1, 2} instead of
/// {0, 4, 128}, an Engram conditional-memory module, and no hash routing at
/// all. ``DeepSeekV4Config`` refuses every one of those by name, which is why
/// this is a second decoder rather than a widened first one.
public enum DeepSeekV41Error: Error, CustomStringConvertible, Equatable {
    case configuration(String)
    case unsupportedArchitecture(String)
    case disagreement(String)
    case artifact(String)
    case experts(String)
    case engram(String)

    public var description: String {
        switch self {
        case .configuration(let detail):
            return "invalid DeepSeek V4.1 configuration: \(detail)"
        case .unsupportedArchitecture(let detail):
            return "unsupported DeepSeek V4.1 architecture: \(detail)"
        case .disagreement(let detail):
            return "DeepSeek V4.1 configuration files disagree: \(detail)"
        case .artifact(let detail):
            return "invalid DeepSeek V4.1 unit artifact: \(detail)"
        case .experts(let detail):
            return "invalid DeepSeek V4.1 expert artifact: \(detail)"
        case .engram(let detail):
            return "invalid DeepSeek V4.1 engram artifact: \(detail)"
        }
    }
}

/// What one block does about global (main-KV) attention.
///
/// CSA2 gives a block one of four jobs, decided entirely by `compress_ratios`,
/// `kv_source_layers` and `index_source_layers` — never by a layer number
/// compiled in here.
public enum DeepSeekV41AttentionMode: String, Sendable, Equatable {
    /// `compress_ratio == 0`: sliding-window attention only. No main KV, no
    /// indexer. Blocks 0 and 1 of the backbone and every DSpark block.
    case windowOnly = "window-only"
    /// A `kv_source_layers` block: computes main KV, derives the indexer key
    /// from it, runs its own indexer and selects its own Top-K.
    case full
    /// An `index_source_layers` block that is not a KV source: reuses the most
    /// recent owner's main KV and indexer key, re-scores with its own indexer
    /// query, and selects its own Top-K.
    case reindex
    /// Everything else: reuses both the main KV and the most recent Top-K, and
    /// has no indexer at all.
    case reuse
}

/// The product-relevant DeepSeek V4.1 Flash architecture, decoded from verified
/// bytes.
///
/// ## Which file is the authority
///
/// `inference/config.json` is, because it is the file DeepSeek's own reference
/// `inference/model.py` loads into `ModelArgs`. The repository's top-level
/// `config.json` is the Transformers `deepseek_v41` shape, with the same model
/// split across `text_config` and `vision_config` and a handful of fields that
/// exist in neither the other file nor `model.py` (`bos_token_id`,
/// `eos_token_id`, `pad_token_id`, `max_position_embeddings`, `topk_method`,
/// `norm_topk_prob`, `attention_bias`, `use_cache`, `tie_word_embeddings`).
///
/// So this decoder reads **both** and does two different things with them:
/// every value the reference runner uses comes from `inference/config.json`,
/// and every value both files state is compared. A disagreement is a refusal —
/// see ``DeepSeekV41Error/disagreement(_:)`` — because a publication whose two
/// configurations describe different models is one where neither can be
/// trusted to describe the weights.
///
/// The container republishes `inference/config.json` at its root as
/// `inference-config.json` (a repository root file is metadata; anything else
/// is payload — `RepoFileClassification`), and its `index.json` names it under
/// `arguments`. ``init(containerRoot:)`` follows exactly that.
///
/// ## What is refused
///
/// Everything this adapter cannot execute, by name, in the manner of
/// ``DeepSeekV4Config``. A later V4.1 revision that changes a routing function,
/// a quantization block, an attention mode set or an Engram geometry stops
/// here rather than being silently interpreted as today's model.
public struct DeepSeekV41Config: Sendable, Equatable {
    // MARK: Identity

    public let modelType: String
    public let architecture: String
    public let textModelType: String
    public let visionModelType: String

    // MARK: Backbone

    public let hiddenSize: Int
    public let vocabularySize: Int
    public let numberOfLayers: Int
    public let numberOfDraftLayers: Int

    // MARK: Attention

    public let numberOfAttentionHeads: Int
    public let numberOfKeyValueHeads: Int
    public let attentionHeadDimension: Int
    public let ropeHeadDimension: Int
    public let queryLowRank: Int
    public let outputGroups: Int
    public let outputLowRank: Int
    public let slidingWindow: Int
    /// One entry per published block: `numberOfLayers` backbone blocks followed
    /// by `numberOfDraftLayers` DSpark blocks. 0 = no global attention,
    /// 2 = encoder (two tokens compress to one main-KV entry), 1 = decoder
    /// (uncompressed).
    public let compressionRatios: [Int]
    public let compressionRopeTheta: Float
    public let keyValueSourceLayers: [Int]
    public let indexSourceLayers: [Int]

    // MARK: Indexer and candidate pool

    public let indexHeadCount: Int
    public let indexHeadDimension: Int
    public let indexTopK: Int
    public let candidateSourceLayer: Int
    public let candidateTopKBlocks: Int
    public let candidateBlockSize: Int

    // MARK: Mixture of experts

    public let routedExpertCount: Int
    public let expertsPerToken: Int
    public let sharedExpertCount: Int
    public let expertIntermediateSize: Int
    public let routingScoreFunction: String
    public let topKMethod: String
    public let normalizedTopK: Bool
    public let routingScale: Float
    public let swiGLULimit: Float

    // MARK: Hyper-connections

    public let hyperConnectionMultiplicity: Int
    public let hyperConnectionSinkhornIterations: Int
    public let hyperConnectionEpsilon: Float

    // MARK: Normalization and rope

    public let rmsNormEpsilon: Float
    public let ropeTheta: Float
    public let ropeFactor: Float
    public let ropeOriginalLength: Int
    public let ropeBetaFast: Int
    public let ropeBetaSlow: Int
    public let maximumPositionCount: Int

    // MARK: Engram

    public let engramLayerIDs: [Int]
    /// Rows of each Engram table, in the order of ``engramLayerIDs``. The two
    /// published tables genuinely differ (384,006,168 and 384,016,682), so this
    /// is never one number.
    public let engramRowCounts: [UInt64]
    public let engramMaximumNGramSize: Int
    public let engramVocabularySize: Int
    public let engramCompressedVocabularySize: Int
    public let engramPadTokenID: Int
    public let engramHeadCount: Int
    public let engramHeadDimension: Int

    // MARK: DSpark

    public let draftBlockSize: Int
    public let draftNoiseTokenID: Int
    public let draftTargetLayerIDs: [Int]
    public let draftMarkovRank: Int
    public let draftRoutedExpertCount: Int
    public let draftExpertsPerToken: Int

    // MARK: Vision (decoded, not executed by the text path)

    public let visionLayerCount: Int
    public let visionHiddenSize: Int
    public let visionHeadCount: Int
    public let visionIntermediateSize: Int
    public let visionPatchSize: Int
    public let visionDownsampleRatio: Int
    public let visionMaximumTokenCount: Int
    public let visionMinimumPixels: Int
    /// `null` in the published configuration; carried as `nil` rather than
    /// defaulted, because a number here would be a different model.
    public let visionMaximumWidthHeightRatio: Float?
    public let visionRopeTheta: Float
    public let imageTokenID: Int

    // MARK: Storage formats and tokens

    public let tensorDType: String
    public let denseDType: String
    public let expertDType: String
    public let quantizationMethod: String
    public let quantizationActivationScheme: String
    public let quantizedScaleFormat: String
    public let quantizedWeightBlock: [Int]
    public let bosTokenID: Int
    public let eosTokenID: Int
    public let padTokenID: Int
    public let tiesWordEmbeddings: Bool
    public let hiddenActivation: String

    // MARK: - Derived

    public var nonRopeHeadDimension: Int {
        attentionHeadDimension - ropeHeadDimension
    }

    /// Rows one Engram module reads per token: one per (n-gram order, hash
    /// head) pair, orders 2 through `engramMaximumNGramSize`.
    public var engramRowsPerToken: Int {
        (engramMaximumNGramSize - 1) * engramHeadCount
    }

    /// Bytes of E8M0 scale that travel with one Engram row —
    /// `engramHeadDimension / 32`, and the reason a row and its scales fit in
    /// one 4 KiB page fifteen at a time.
    public var engramRowScaleBytes: Int {
        engramHeadDimension / quantizedWeightBlock[1]
    }

    /// Blocks published in total: the backbone plus the DSpark draft stack.
    public var publishedBlockCount: Int {
        numberOfLayers + numberOfDraftLayers
    }

    /// The first block of the causal *decoder* half — the first backbone block
    /// whose compression ratio is 1.
    ///
    /// Derived from `compress_ratios` rather than stated, and the constructor
    /// refuses a schedule that is not one encoder run followed by one decoder
    /// run, so this cannot be an average of a shape nobody checked.
    public let firstDecoderBlock: Int

    public func compressionRatio(block: Int) throws -> Int {
        guard block >= 0, block < compressionRatios.count else {
            throw DeepSeekV41Error.configuration(
                "block \(block) is outside 0..<\(compressionRatios.count)")
        }
        return compressionRatios[block]
    }

    public func attentionMode(block: Int) throws -> DeepSeekV41AttentionMode {
        guard try compressionRatio(block: block) != 0 else { return .windowOnly }
        if keyValueSourceLayers.contains(block) { return .full }
        if indexSourceLayers.contains(block) { return .reindex }
        return .reuse
    }

    /// The block whose main KV a block reuses — itself when it is a KV source.
    public func keyValueSourceBlock(for block: Int) throws -> Int? {
        guard try compressionRatio(block: block) != 0 else { return nil }
        return keyValueSourceLayers.last { $0 <= block }
    }

    public func hasEngram(block: Int) -> Bool {
        engramLayerIDs.contains(block)
    }

    /// Rows of the Engram table this block owns, or `nil` when it owns none.
    public func engramRowCount(block: Int) -> UInt64? {
        engramLayerIDs.firstIndex(of: block).map { engramRowCounts[$0] }
    }

    /// Routed experts in one block: 384 in the backbone, 128 in a DSpark block.
    public func routedExpertCount(block: Int) throws -> Int {
        guard block >= 0, block < publishedBlockCount else {
            throw DeepSeekV41Error.configuration(
                "block \(block) is outside 0..<\(publishedBlockCount)")
        }
        return block < numberOfLayers ? routedExpertCount : draftRoutedExpertCount
    }

    public func expertsPerToken(block: Int) throws -> Int {
        guard block >= 0, block < publishedBlockCount else {
            throw DeepSeekV41Error.configuration(
                "block \(block) is outside 0..<\(publishedBlockCount)")
        }
        return block < numberOfLayers ? expertsPerToken : draftExpertsPerToken
    }

    // MARK: - Construction

    /// Decode a published container root: `inference-config.json` beside
    /// `config.json`, exactly as `index.json` names them.
    public init(containerRoot: URL) throws {
        try self.init(
            inferenceConfigURL: containerRoot
                .appendingPathComponent("inference-config.json"),
            huggingFaceConfigURL: containerRoot
                .appendingPathComponent("config.json"))
    }

    public init(inferenceConfigURL: URL, huggingFaceConfigURL: URL) throws {
        try self.init(
            inferenceJSON: Data(contentsOf: inferenceConfigURL),
            huggingFaceJSON: Data(contentsOf: huggingFaceConfigURL))
    }

    public init(inferenceJSON: Data, huggingFaceJSON: Data) throws {
        let inference: InferenceDocument
        do {
            inference = try JSONDecoder().decode(
                InferenceDocument.self, from: inferenceJSON)
        } catch {
            throw DeepSeekV41Error.configuration(
                "inference/config.json could not be decoded: \(error)")
        }
        let hugging: HuggingFaceDocument
        do {
            hugging = try JSONDecoder().decode(
                HuggingFaceDocument.self, from: huggingFaceJSON)
        } catch {
            throw DeepSeekV41Error.configuration(
                "config.json could not be decoded: \(error)")
        }
        let text = hugging.text
        let vision = hugging.vision

        // MARK: Identity, refused by name before anything is believed

        guard hugging.modelType == "deepseek_v41" else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "model_type '\(hugging.modelType)' is not 'deepseek_v41'")
        }
        guard hugging.architectures.count == 1,
            hugging.architectures[0] == "DeepseekV41ForCausalLM"
        else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "architectures \(hugging.architectures) do not name DeepseekV41ForCausalLM")
        }
        guard text.modelType == "deepseek_v41_text" else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "text_config.model_type '\(text.modelType)' is not 'deepseek_v41_text'")
        }
        guard vision.modelType == "deepseek_v41_vision" else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "vision_config.model_type '\(vision.modelType)' is not 'deepseek_v41_vision'")
        }
        guard text.hiddenActivation == "silu" else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "hidden_act '\(text.hiddenActivation)' is not silu")
        }
        guard hugging.tensorDType == "bfloat16" else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "dtype '\(hugging.tensorDType)' is not bfloat16")
        }
        guard text.attentionBias == false, text.attentionDropout == 0 else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "attention bias or nonzero attention dropout is not implemented")
        }
        guard text.useCache else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "use_cache is false; the product decoder requires incremental cache semantics")
        }
        guard !text.tiesWordEmbeddings else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "tie_word_embeddings is true; the artifact carries a distinct output head")
        }
        guard inference.routingScoreFunction == "sqrtsoftplus",
            text.topKMethod == "noaux_tc", text.normalizedTopK
        else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "routing requires score_func sqrtsoftplus, topk_method noaux_tc "
                    + "and norm_topk_prob true")
        }
        guard inference.denseDType == "fp8" else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "dtype '\(inference.denseDType)' is not fp8; the dense path is block-FP8")
        }
        guard inference.expertDType == "fp4" else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "expert_dtype '\(inference.expertDType)' is not fp4")
        }
        guard hugging.quantization.method == "fp8",
            hugging.quantization.activationScheme == "dynamic",
            hugging.quantization.scaleFormat == "ue8m0",
            hugging.quantization.weightBlockSize == [32, 32]
        else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "the non-expert quantization is not dynamic ue8m0 block-FP8 [32, 32]")
        }
        guard text.ropeScaling.type.lowercased() == "yarn" else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "rope_scaling.rope_type '\(text.ropeScaling.type)' is not yarn")
        }

        // MARK: Positive extents

        func requirePositive(_ value: Int, _ name: String) throws {
            guard value > 0 else {
                throw DeepSeekV41Error.configuration(
                    "\(name) must be positive, got \(value)")
            }
        }
        try requirePositive(inference.hiddenSize, "dim")
        try requirePositive(inference.vocabularySize, "vocab_size")
        try requirePositive(inference.numberOfLayers, "n_layers")
        try requirePositive(inference.numberOfAttentionHeads, "n_heads")
        try requirePositive(inference.attentionHeadDimension, "head_dim")
        try requirePositive(inference.ropeHeadDimension, "rope_head_dim")
        try requirePositive(inference.queryLowRank, "q_lora_rank")
        try requirePositive(inference.outputGroups, "o_groups")
        try requirePositive(inference.outputLowRank, "o_lora_rank")
        try requirePositive(inference.slidingWindow, "window_size")
        try requirePositive(inference.indexHeadCount, "index_n_heads")
        try requirePositive(inference.indexHeadDimension, "index_head_dim")
        try requirePositive(inference.indexTopK, "index_topk")
        try requirePositive(inference.candidateTopKBlocks, "candidate_topk_blocks")
        try requirePositive(inference.candidateBlockSize, "candidate_block_size")
        try requirePositive(inference.routedExpertCount, "n_routed_experts")
        try requirePositive(inference.expertsPerToken, "n_activated_experts")
        try requirePositive(inference.expertIntermediateSize, "moe_inter_dim")
        try requirePositive(inference.hyperConnectionMultiplicity, "hc_mult")
        try requirePositive(
            inference.hyperConnectionSinkhornIterations, "hc_sinkhorn_iters")
        try requirePositive(inference.ropeOriginalLength, "original_seq_len")
        try requirePositive(inference.ropeBetaFast, "beta_fast")
        try requirePositive(inference.ropeBetaSlow, "beta_slow")
        try requirePositive(inference.engramMaximumNGramSize, "engram_max_ngram_size")
        try requirePositive(inference.engramVocabularySize, "engram_vocab_size")
        try requirePositive(
            inference.engramCompressedVocabularySize, "engram_compressed_vocab_size")
        try requirePositive(inference.engramHeadCount, "engram_n_heads")
        try requirePositive(inference.engramHeadDimension, "engram_head_dim")
        try requirePositive(inference.draftBlockSize, "dspark_block_size")
        try requirePositive(inference.draftMarkovRank, "dspark_markov_rank")
        try requirePositive(inference.visionLayerCount, "vision_n_layers")
        try requirePositive(inference.visionHiddenSize, "vision_dim")
        try requirePositive(inference.visionHeadCount, "vision_n_heads")
        try requirePositive(inference.visionIntermediateSize, "vision_inter_dim")
        try requirePositive(inference.visionPatchSize, "vision_patch_size")
        try requirePositive(inference.visionDownsampleRatio, "vision_downsample_ratio")
        try requirePositive(inference.visionMaximumTokenCount, "vision_max_n_token")
        try requirePositive(inference.visionMinimumPixels, "vision_min_pixels")
        try requirePositive(text.maximumPositionCount, "max_position_embeddings")

        // MARK: Structure this adapter can execute

        guard inference.sharedExpertCount == 1 else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "n_shared_experts \(inference.sharedExpertCount); "
                    + "this adapter implements exactly one")
        }
        guard text.numberOfKeyValueHeads == 1 else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "num_key_value_heads \(text.numberOfKeyValueHeads); "
                    + "compressed MQA requires one")
        }
        guard inference.ropeHeadDimension < inference.attentionHeadDimension else {
            throw DeepSeekV41Error.configuration(
                "rope_head_dim must be smaller than head_dim")
        }
        guard inference.outputGroups <= inference.numberOfAttentionHeads,
            inference.numberOfAttentionHeads % inference.outputGroups == 0
        else {
            throw DeepSeekV41Error.configuration(
                "n_heads is not divisible by o_groups")
        }
        guard inference.expertsPerToken <= inference.routedExpertCount else {
            throw DeepSeekV41Error.configuration(
                "n_activated_experts exceeds n_routed_experts")
        }
        guard inference.draftExpertsPerToken <= inference.draftRoutedExpertCount,
            inference.draftExpertsPerToken > 0, inference.draftRoutedExpertCount > 0
        else {
            throw DeepSeekV41Error.configuration(
                "dspark_n_activated_experts exceeds dspark_n_routed_experts")
        }
        guard inference.indexTopK <= text.maximumPositionCount else {
            throw DeepSeekV41Error.configuration(
                "index_topk exceeds max_position_embeddings")
        }
        guard inference.numberOfDraftLayers >= 0 else {
            throw DeepSeekV41Error.configuration("n_mtp_layers cannot be negative")
        }
        guard inference.draftTargetLayerIDs.count == inference.numberOfDraftLayers,
            inference.draftTargetLayerIDs.allSatisfy({
                $0 >= 0 && $0 < inference.numberOfLayers
            })
        else {
            throw DeepSeekV41Error.configuration(
                "dspark_target_layer_ids \(inference.draftTargetLayerIDs) does not name "
                    + "\(inference.numberOfDraftLayers) backbone layers")
        }
        guard inference.draftNoiseTokenID >= 0,
            inference.draftNoiseTokenID < inference.vocabularySize,
            inference.imageTokenID >= 0,
            inference.imageTokenID < inference.vocabularySize
        else {
            throw DeepSeekV41Error.configuration(
                "dspark_noise_token_id and image_token_id must be inside the vocabulary")
        }

        // MARK: The CSA2 schedule

        let blockCount = inference.numberOfLayers + inference.numberOfDraftLayers
        guard inference.compressionRatios.count == blockCount else {
            throw DeepSeekV41Error.configuration(
                "compress_ratios has \(inference.compressionRatios.count) entries; "
                    + "n_layers + n_mtp_layers is \(blockCount)")
        }
        guard inference.compressionRatios.allSatisfy({ $0 == 0 || $0 == 1 || $0 == 2 })
        else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "compress_ratios contains a value other than 0, 1, or 2")
        }
        guard inference.compressionRatios
            .suffix(inference.numberOfDraftLayers).allSatisfy({ $0 == 0 })
        else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "a DSpark block with global attention is not implemented")
        }
        let backboneRatios = Array(
            inference.compressionRatios.prefix(inference.numberOfLayers))
        // One encoder run then one decoder run, and nothing else. `firstDecoder`
        // is derived from the schedule; the two guards below are what stop it
        // from being a summary of a shape nobody checked.
        guard let firstDecoder = backboneRatios.firstIndex(of: 1) else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "compress_ratios names no decoder block (ratio 1)")
        }
        guard backboneRatios.prefix(firstDecoder).allSatisfy({ $0 == 0 || $0 == 2 }),
            backboneRatios.dropFirst(firstDecoder).allSatisfy({ $0 == 1 })
        else {
            throw DeepSeekV41Error.unsupportedArchitecture(
                "compress_ratios is not one encoder run followed by one decoder run")
        }

        func requireAscendingBlocks(_ values: [Int], _ name: String) throws {
            guard !values.isEmpty,
                values == values.sorted(),
                Set(values).count == values.count,
                values.allSatisfy({ $0 >= 0 && $0 < inference.numberOfLayers })
            else {
                throw DeepSeekV41Error.configuration(
                    "\(name) \(values) is not a strictly ascending set of backbone blocks")
            }
        }
        try requireAscendingBlocks(inference.keyValueSourceLayers, "kv_source_layers")
        try requireAscendingBlocks(inference.indexSourceLayers, "index_source_layers")
        guard Set(inference.keyValueSourceLayers)
            .isSubset(of: Set(inference.indexSourceLayers))
        else {
            throw DeepSeekV41Error.configuration(
                "kv_source_layers is not a subset of index_source_layers; a block that "
                    + "owns main KV must own an indexer")
        }
        guard inference.indexSourceLayers
            .allSatisfy({ backboneRatios[$0] != 0 })
        else {
            throw DeepSeekV41Error.configuration(
                "an index or KV source block has compress_ratio 0, so it has no main KV "
                    + "to index")
        }
        guard let firstKeyValueSource = inference.keyValueSourceLayers.first,
            backboneRatios.enumerated().allSatisfy({ block, ratio in
                ratio == 0 || block >= firstKeyValueSource
            })
        else {
            throw DeepSeekV41Error.configuration(
                "a block with global attention precedes the first kv_source_layer")
        }
        guard inference.keyValueSourceLayers.contains(inference.candidateSourceLayer)
        else {
            throw DeepSeekV41Error.configuration(
                "candidate_source_layer \(inference.candidateSourceLayer) is not a "
                    + "kv_source_layer")
        }
        guard backboneRatios[inference.candidateSourceLayer] == 1 else {
            throw DeepSeekV41Error.configuration(
                "candidate_source_layer \(inference.candidateSourceLayer) is not a "
                    + "decoder block")
        }

        // MARK: Engram

        guard !inference.engramLayerIDs.isEmpty,
            inference.engramLayerIDs == inference.engramLayerIDs.sorted(),
            Set(inference.engramLayerIDs).count == inference.engramLayerIDs.count,
            inference.engramLayerIDs.allSatisfy({
                $0 >= 0 && $0 < inference.numberOfLayers
            })
        else {
            throw DeepSeekV41Error.engram(
                "engram_layer_ids \(inference.engramLayerIDs) is not a strictly "
                    + "ascending set of backbone blocks")
        }
        guard inference.engramRowCounts.count == inference.engramLayerIDs.count,
            inference.engramRowCounts.allSatisfy({ $0 > 0 })
        else {
            throw DeepSeekV41Error.engram(
                "engram_num_embeddings does not give one positive row count per "
                    + "engram layer")
        }
        guard inference.engramMaximumNGramSize >= 2 else {
            throw DeepSeekV41Error.engram(
                "engram_max_ngram_size \(inference.engramMaximumNGramSize) leaves no "
                    + "n-gram order to hash")
        }
        guard inference.engramPadTokenID >= 0,
            inference.engramPadTokenID < inference.vocabularySize
        else {
            throw DeepSeekV41Error.engram(
                "engram_pad_id \(inference.engramPadTokenID) is outside the vocabulary")
        }
        guard inference.engramCompressedVocabularySize <= inference.vocabularySize else {
            throw DeepSeekV41Error.engram(
                "engram_compressed_vocab_size exceeds vocab_size")
        }
        // One E8M0 exponent per quantization group of the row: this is what
        // makes a row 256 + 8 bytes and fifteen of them one 4 KiB page.
        let scaleGroup = hugging.quantization.weightBlockSize[1]
        guard inference.engramHeadDimension % scaleGroup == 0 else {
            throw DeepSeekV41Error.engram(
                "engram_head_dim \(inference.engramHeadDimension) is not a whole number "
                    + "of \(scaleGroup)-element scale groups")
        }
        guard inference.engramRowCounts.allSatisfy({
            $0 <= UInt64(inference.engramVocabularySize)
                * UInt64(inference.engramHeadCount)
                * UInt64(inference.engramMaximumNGramSize)
        }) else {
            throw DeepSeekV41Error.engram(
                "an engram table declares more rows than its hash space can address")
        }

        // MARK: Scalars

        guard inference.rmsNormEpsilon.isFinite, inference.rmsNormEpsilon > 0,
            inference.hyperConnectionEpsilon.isFinite,
            inference.hyperConnectionEpsilon > 0,
            inference.routingScale.isFinite, inference.routingScale > 0,
            inference.swiGLULimit.isFinite, inference.swiGLULimit >= 0,
            inference.ropeTheta.isFinite, inference.ropeTheta > 0,
            inference.compressionRopeTheta.isFinite,
            inference.compressionRopeTheta > 0,
            inference.ropeFactor.isFinite, inference.ropeFactor > 0,
            inference.visionRopeTheta.isFinite, inference.visionRopeTheta > 0
        else {
            throw DeepSeekV41Error.configuration(
                "normalization, hyper-connection, routing, SwiGLU and rope scalars "
                    + "must be finite and positive")
        }
        if let ratio = inference.visionMaximumWidthHeightRatio {
            guard ratio.isFinite, ratio > 0 else {
                throw DeepSeekV41Error.configuration(
                    "vision_max_wh_ratio must be null or a positive finite number")
            }
        }
        guard inference.ropeBetaFast >= inference.ropeBetaSlow else {
            throw DeepSeekV41Error.configuration(
                "beta_fast must not be smaller than beta_slow")
        }
        guard inference.ropeOriginalLength <= text.maximumPositionCount else {
            throw DeepSeekV41Error.configuration(
                "original_seq_len exceeds max_position_embeddings")
        }
        guard hugging.bosTokenID >= 0, hugging.bosTokenID < inference.vocabularySize,
            hugging.eosTokenID >= 0, hugging.eosTokenID < inference.vocabularySize,
            hugging.padTokenID >= 0, hugging.padTokenID < inference.vocabularySize,
            hugging.bosTokenID != hugging.eosTokenID
        else {
            throw DeepSeekV41Error.configuration(
                "bos_token_id and eos_token_id must be distinct ids inside the "
                    + "vocabulary, and pad_token_id inside it")
        }
        guard hugging.padTokenID == inference.engramPadTokenID else {
            throw DeepSeekV41Error.disagreement(
                "pad_token_id \(hugging.padTokenID) is not engram_pad_id "
                    + "\(inference.engramPadTokenID); Engram pads its n-gram history "
                    + "with the tokenizer's pad id")
        }

        // MARK: Every value both files state

        func agree<Value: Equatable>(
            _ inferenceValue: Value, _ inferenceName: String,
            _ huggingValue: Value, _ huggingName: String
        ) throws {
            guard inferenceValue == huggingValue else {
                throw DeepSeekV41Error.disagreement(
                    "inference/config.json \(inferenceName) is \(inferenceValue), "
                        + "config.json \(huggingName) is \(huggingValue)")
            }
        }
        try agree(inference.vocabularySize, "vocab_size", text.vocabularySize, "text_config.vocab_size")
        try agree(inference.hiddenSize, "dim", text.hiddenSize, "text_config.hidden_size")
        try agree(inference.expertIntermediateSize, "moe_inter_dim", text.expertIntermediateSize, "text_config.moe_intermediate_size")
        try agree(inference.numberOfLayers, "n_layers", text.numberOfLayers, "text_config.num_hidden_layers")
        try agree(inference.numberOfDraftLayers, "n_mtp_layers", text.numberOfDraftLayers, "text_config.num_nextn_predict_layers")
        try agree(inference.numberOfAttentionHeads, "n_heads", text.numberOfAttentionHeads, "text_config.num_attention_heads")
        try agree(inference.attentionHeadDimension, "head_dim", text.attentionHeadDimension, "text_config.head_dim")
        try agree(inference.ropeHeadDimension, "rope_head_dim", text.ropeHeadDimension, "text_config.qk_rope_head_dim")
        try agree(inference.queryLowRank, "q_lora_rank", text.queryLowRank, "text_config.q_lora_rank")
        try agree(inference.outputGroups, "o_groups", text.outputGroups, "text_config.o_groups")
        try agree(inference.outputLowRank, "o_lora_rank", text.outputLowRank, "text_config.o_lora_rank")
        try agree(inference.slidingWindow, "window_size", text.slidingWindow, "text_config.sliding_window")
        try agree(inference.compressionRatios, "compress_ratios", text.compressionRatios, "text_config.compress_ratios")
        try agree(inference.compressionRopeTheta, "compress_rope_theta", text.compressionRopeTheta, "text_config.compress_rope_theta")
        try agree(inference.keyValueSourceLayers, "kv_source_layers", text.keyValueSourceLayers, "text_config.kv_source_layer_ids")
        try agree(inference.indexSourceLayers, "index_source_layers", text.indexSourceLayers, "text_config.index_source_layer_ids")
        try agree(inference.indexHeadCount, "index_n_heads", text.indexHeadCount, "text_config.index_n_heads")
        try agree(inference.indexHeadDimension, "index_head_dim", text.indexHeadDimension, "text_config.index_head_dim")
        try agree(inference.indexTopK, "index_topk", text.indexTopK, "text_config.index_topk")
        try agree(inference.candidateSourceLayer, "candidate_source_layer", text.candidateSourceLayer, "text_config.candidate_source_layer_id")
        try agree(inference.candidateTopKBlocks, "candidate_topk_blocks", text.candidateTopKBlocks, "text_config.candidate_topk_blocks")
        try agree(inference.candidateBlockSize, "candidate_block_size", text.candidateBlockSize, "text_config.candidate_block_size")
        try agree(inference.routedExpertCount, "n_routed_experts", text.routedExpertCount, "text_config.n_routed_experts")
        try agree(inference.sharedExpertCount, "n_shared_experts", text.sharedExpertCount, "text_config.n_shared_experts")
        try agree(inference.expertsPerToken, "n_activated_experts", text.expertsPerToken, "text_config.num_experts_per_tok")
        try agree(inference.routingScoreFunction, "score_func", text.routingScoreFunction, "text_config.scoring_func")
        try agree(inference.routingScale, "route_scale", text.routingScale, "text_config.routed_scaling_factor")
        try agree(inference.swiGLULimit, "swiglu_limit", text.swiGLULimit, "text_config.swiglu_limit")
        try agree(inference.rmsNormEpsilon, "norm_eps", text.rmsNormEpsilon, "text_config.rms_norm_eps")
        try agree(inference.ropeTheta, "rope_theta", text.ropeTheta, "text_config.rope_theta")
        try agree(inference.ropeFactor, "rope_factor", text.ropeScaling.factor, "text_config.rope_scaling.factor")
        try agree(inference.ropeOriginalLength, "original_seq_len", text.ropeScaling.originalMaximumPositions, "text_config.rope_scaling.original_max_position_embeddings")
        try agree(inference.ropeBetaFast, "beta_fast", text.ropeScaling.betaFast, "text_config.rope_scaling.beta_fast")
        try agree(inference.ropeBetaSlow, "beta_slow", text.ropeScaling.betaSlow, "text_config.rope_scaling.beta_slow")
        try agree(inference.hyperConnectionMultiplicity, "hc_mult", text.hyperConnectionMultiplicity, "text_config.hc_mult")
        try agree(inference.hyperConnectionSinkhornIterations, "hc_sinkhorn_iters", text.hyperConnectionSinkhornIterations, "text_config.hc_sinkhorn_iters")
        try agree(inference.hyperConnectionEpsilon, "hc_eps", text.hyperConnectionEpsilon, "text_config.hc_eps")
        try agree(inference.engramLayerIDs, "engram_layer_ids", text.engramLayerIDs, "text_config.engram_layer_ids")
        try agree(inference.engramRowCounts, "engram_num_embeddings", text.engramRowCounts, "text_config.engram_num_embeddings")
        try agree(inference.engramMaximumNGramSize, "engram_max_ngram_size", text.engramMaximumNGramSize, "text_config.engram_max_ngram_size")
        try agree(inference.engramVocabularySize, "engram_vocab_size", text.engramVocabularySize, "text_config.engram_vocab_size")
        try agree(inference.engramHeadCount, "engram_n_heads", text.engramHeadCount, "text_config.engram_n_heads")
        try agree(inference.engramHeadDimension, "engram_head_dim", text.engramHeadDimension, "text_config.engram_head_dim")
        try agree(inference.engramPadTokenID, "engram_pad_id", text.engramPadTokenID, "text_config.engram_pad_token_id")
        try agree(inference.engramCompressedVocabularySize, "engram_compressed_vocab_size", text.engramCompressedVocabularySize, "text_config.engram_compressed_vocab_size")
        try agree(inference.draftBlockSize, "dspark_block_size", text.draftBlockSize, "text_config.dspark_block_size")
        try agree(inference.draftNoiseTokenID, "dspark_noise_token_id", text.draftNoiseTokenID, "text_config.dspark_noise_token_id")
        try agree(inference.draftTargetLayerIDs, "dspark_target_layer_ids", text.draftTargetLayerIDs, "text_config.dspark_target_layer_ids")
        try agree(inference.draftMarkovRank, "dspark_markov_rank", text.draftMarkovRank, "text_config.dspark_markov_rank")
        try agree(inference.draftRoutedExpertCount, "dspark_n_routed_experts", text.draftRoutedExpertCount, "text_config.dspark_n_routed_experts")
        try agree(inference.draftExpertsPerToken, "dspark_n_activated_experts", text.draftExpertsPerToken, "text_config.dspark_num_experts_per_tok")
        try agree(inference.expertDType, "expert_dtype", hugging.quantization.expertDType, "quantization_config.expert_dtype")
        try agree(inference.imageTokenID, "image_token_id", hugging.imageTokenID, "image_token_id")
        try agree(inference.visionLayerCount, "vision_n_layers", vision.numberOfLayers, "vision_config.num_hidden_layers")
        try agree(inference.visionHiddenSize, "vision_dim", vision.hiddenSize, "vision_config.hidden_size")
        try agree(inference.visionHeadCount, "vision_n_heads", vision.numberOfAttentionHeads, "vision_config.num_attention_heads")
        try agree(inference.visionIntermediateSize, "vision_inter_dim", vision.intermediateSize, "vision_config.intermediate_size")
        try agree(inference.visionPatchSize, "vision_patch_size", vision.patchSize, "vision_config.patch_size")
        try agree(inference.visionDownsampleRatio, "vision_downsample_ratio", vision.downsampleRatio, "vision_config.downsample_ratio")
        try agree(inference.visionMaximumTokenCount, "vision_max_n_token", vision.maximumImageTokens, "vision_config.max_image_tokens")
        try agree(inference.visionMinimumPixels, "vision_min_pixels", vision.minimumPixels, "vision_config.min_pixels")
        try agree(inference.visionMaximumWidthHeightRatio, "vision_max_wh_ratio", vision.maximumWidthHeightRatio, "vision_config.max_wh_ratio")
        try agree(inference.visionRopeTheta, "vision_rope_theta", vision.ropeTheta, "vision_config.rope_theta")

        // MARK: Stored

        modelType = hugging.modelType
        architecture = hugging.architectures[0]
        textModelType = text.modelType
        visionModelType = vision.modelType
        hiddenSize = inference.hiddenSize
        vocabularySize = inference.vocabularySize
        numberOfLayers = inference.numberOfLayers
        numberOfDraftLayers = inference.numberOfDraftLayers
        numberOfAttentionHeads = inference.numberOfAttentionHeads
        numberOfKeyValueHeads = text.numberOfKeyValueHeads
        attentionHeadDimension = inference.attentionHeadDimension
        ropeHeadDimension = inference.ropeHeadDimension
        queryLowRank = inference.queryLowRank
        outputGroups = inference.outputGroups
        outputLowRank = inference.outputLowRank
        slidingWindow = inference.slidingWindow
        compressionRatios = inference.compressionRatios
        compressionRopeTheta = inference.compressionRopeTheta
        keyValueSourceLayers = inference.keyValueSourceLayers
        indexSourceLayers = inference.indexSourceLayers
        indexHeadCount = inference.indexHeadCount
        indexHeadDimension = inference.indexHeadDimension
        indexTopK = inference.indexTopK
        candidateSourceLayer = inference.candidateSourceLayer
        candidateTopKBlocks = inference.candidateTopKBlocks
        candidateBlockSize = inference.candidateBlockSize
        routedExpertCount = inference.routedExpertCount
        expertsPerToken = inference.expertsPerToken
        sharedExpertCount = inference.sharedExpertCount
        expertIntermediateSize = inference.expertIntermediateSize
        routingScoreFunction = inference.routingScoreFunction
        topKMethod = text.topKMethod
        normalizedTopK = text.normalizedTopK
        routingScale = inference.routingScale
        swiGLULimit = inference.swiGLULimit
        hyperConnectionMultiplicity = inference.hyperConnectionMultiplicity
        hyperConnectionSinkhornIterations = inference.hyperConnectionSinkhornIterations
        hyperConnectionEpsilon = inference.hyperConnectionEpsilon
        rmsNormEpsilon = inference.rmsNormEpsilon
        ropeTheta = inference.ropeTheta
        ropeFactor = inference.ropeFactor
        ropeOriginalLength = inference.ropeOriginalLength
        ropeBetaFast = inference.ropeBetaFast
        ropeBetaSlow = inference.ropeBetaSlow
        maximumPositionCount = text.maximumPositionCount
        engramLayerIDs = inference.engramLayerIDs
        engramRowCounts = inference.engramRowCounts
        engramMaximumNGramSize = inference.engramMaximumNGramSize
        engramVocabularySize = inference.engramVocabularySize
        engramCompressedVocabularySize = inference.engramCompressedVocabularySize
        engramPadTokenID = inference.engramPadTokenID
        engramHeadCount = inference.engramHeadCount
        engramHeadDimension = inference.engramHeadDimension
        draftBlockSize = inference.draftBlockSize
        draftNoiseTokenID = inference.draftNoiseTokenID
        draftTargetLayerIDs = inference.draftTargetLayerIDs
        draftMarkovRank = inference.draftMarkovRank
        draftRoutedExpertCount = inference.draftRoutedExpertCount
        draftExpertsPerToken = inference.draftExpertsPerToken
        visionLayerCount = inference.visionLayerCount
        visionHiddenSize = inference.visionHiddenSize
        visionHeadCount = inference.visionHeadCount
        visionIntermediateSize = inference.visionIntermediateSize
        visionPatchSize = inference.visionPatchSize
        visionDownsampleRatio = inference.visionDownsampleRatio
        visionMaximumTokenCount = inference.visionMaximumTokenCount
        visionMinimumPixels = inference.visionMinimumPixels
        visionMaximumWidthHeightRatio = inference.visionMaximumWidthHeightRatio
        visionRopeTheta = inference.visionRopeTheta
        imageTokenID = inference.imageTokenID
        tensorDType = hugging.tensorDType
        denseDType = inference.denseDType
        expertDType = inference.expertDType
        quantizationMethod = hugging.quantization.method
        quantizationActivationScheme = hugging.quantization.activationScheme
        quantizedScaleFormat = hugging.quantization.scaleFormat
        quantizedWeightBlock = hugging.quantization.weightBlockSize
        bosTokenID = hugging.bosTokenID
        eosTokenID = hugging.eosTokenID
        padTokenID = hugging.padTokenID
        tiesWordEmbeddings = text.tiesWordEmbeddings
        hiddenActivation = text.hiddenActivation
        firstDecoderBlock = firstDecoder
    }

    // MARK: - Wire shapes

    /// `inference/config.json` — the flat `ModelArgs` DeepSeek's own reference
    /// runner loads.
    private struct InferenceDocument: Decodable {
        let vocabularySize: Int
        let hiddenSize: Int
        let expertIntermediateSize: Int
        let numberOfLayers: Int
        let numberOfDraftLayers: Int
        let draftBlockSize: Int
        let draftNoiseTokenID: Int
        let draftTargetLayerIDs: [Int]
        let draftMarkovRank: Int
        let draftRoutedExpertCount: Int
        let draftExpertsPerToken: Int
        let numberOfAttentionHeads: Int
        let routedExpertCount: Int
        let sharedExpertCount: Int
        let expertsPerToken: Int
        let routingScoreFunction: String
        let routingScale: Float
        let swiGLULimit: Float
        let queryLowRank: Int
        let attentionHeadDimension: Int
        let ropeHeadDimension: Int
        let rmsNormEpsilon: Float
        let outputGroups: Int
        let outputLowRank: Int
        let slidingWindow: Int
        let keyValueSourceLayers: [Int]
        let indexSourceLayers: [Int]
        let ropeOriginalLength: Int
        let ropeTheta: Float
        let ropeFactor: Float
        let ropeBetaFast: Int
        let ropeBetaSlow: Int
        let indexHeadCount: Int
        let indexHeadDimension: Int
        let indexTopK: Int
        let candidateSourceLayer: Int
        let candidateTopKBlocks: Int
        let candidateBlockSize: Int
        let hyperConnectionMultiplicity: Int
        let hyperConnectionSinkhornIterations: Int
        let hyperConnectionEpsilon: Float
        let engramLayerIDs: [Int]
        let engramVocabularySize: Int
        let engramRowCounts: [UInt64]
        let engramMaximumNGramSize: Int
        let engramPadTokenID: Int
        let engramCompressedVocabularySize: Int
        let engramHeadCount: Int
        let engramHeadDimension: Int
        let denseDType: String
        let expertDType: String
        let compressionRopeTheta: Float
        let compressionRatios: [Int]
        let visionLayerCount: Int
        let visionHiddenSize: Int
        let visionHeadCount: Int
        let visionIntermediateSize: Int
        let visionPatchSize: Int
        let visionDownsampleRatio: Int
        let visionMaximumTokenCount: Int
        let visionMinimumPixels: Int
        let visionMaximumWidthHeightRatio: Float?
        let visionRopeTheta: Float
        let imageTokenID: Int

        enum CodingKeys: String, CodingKey {
            case vocabularySize = "vocab_size"
            case hiddenSize = "dim"
            case expertIntermediateSize = "moe_inter_dim"
            case numberOfLayers = "n_layers"
            case numberOfDraftLayers = "n_mtp_layers"
            case draftBlockSize = "dspark_block_size"
            case draftNoiseTokenID = "dspark_noise_token_id"
            case draftTargetLayerIDs = "dspark_target_layer_ids"
            case draftMarkovRank = "dspark_markov_rank"
            case draftRoutedExpertCount = "dspark_n_routed_experts"
            case draftExpertsPerToken = "dspark_n_activated_experts"
            case numberOfAttentionHeads = "n_heads"
            case routedExpertCount = "n_routed_experts"
            case sharedExpertCount = "n_shared_experts"
            case expertsPerToken = "n_activated_experts"
            case routingScoreFunction = "score_func"
            case routingScale = "route_scale"
            case swiGLULimit = "swiglu_limit"
            case queryLowRank = "q_lora_rank"
            case attentionHeadDimension = "head_dim"
            case ropeHeadDimension = "rope_head_dim"
            case rmsNormEpsilon = "norm_eps"
            case outputGroups = "o_groups"
            case outputLowRank = "o_lora_rank"
            case slidingWindow = "window_size"
            case keyValueSourceLayers = "kv_source_layers"
            case indexSourceLayers = "index_source_layers"
            case ropeOriginalLength = "original_seq_len"
            case ropeTheta = "rope_theta"
            case ropeFactor = "rope_factor"
            case ropeBetaFast = "beta_fast"
            case ropeBetaSlow = "beta_slow"
            case indexHeadCount = "index_n_heads"
            case indexHeadDimension = "index_head_dim"
            case indexTopK = "index_topk"
            case candidateSourceLayer = "candidate_source_layer"
            case candidateTopKBlocks = "candidate_topk_blocks"
            case candidateBlockSize = "candidate_block_size"
            case hyperConnectionMultiplicity = "hc_mult"
            case hyperConnectionSinkhornIterations = "hc_sinkhorn_iters"
            case hyperConnectionEpsilon = "hc_eps"
            case engramLayerIDs = "engram_layer_ids"
            case engramVocabularySize = "engram_vocab_size"
            case engramRowCounts = "engram_num_embeddings"
            case engramMaximumNGramSize = "engram_max_ngram_size"
            case engramPadTokenID = "engram_pad_id"
            case engramCompressedVocabularySize = "engram_compressed_vocab_size"
            case engramHeadCount = "engram_n_heads"
            case engramHeadDimension = "engram_head_dim"
            case denseDType = "dtype"
            case expertDType = "expert_dtype"
            case compressionRopeTheta = "compress_rope_theta"
            case compressionRatios = "compress_ratios"
            case visionLayerCount = "vision_n_layers"
            case visionHiddenSize = "vision_dim"
            case visionHeadCount = "vision_n_heads"
            case visionIntermediateSize = "vision_inter_dim"
            case visionPatchSize = "vision_patch_size"
            case visionDownsampleRatio = "vision_downsample_ratio"
            case visionMaximumTokenCount = "vision_max_n_token"
            case visionMinimumPixels = "vision_min_pixels"
            case visionMaximumWidthHeightRatio = "vision_max_wh_ratio"
            case visionRopeTheta = "vision_rope_theta"
            case imageTokenID = "image_token_id"
        }
    }

    /// The repository's top-level `config.json` — the Transformers
    /// `deepseek_v41` shape.
    private struct HuggingFaceDocument: Decodable {
        let architectures: [String]
        let modelType: String
        let tensorDType: String
        let bosTokenID: Int
        let eosTokenID: Int
        let padTokenID: Int
        let imageTokenID: Int
        let quantization: Quantization
        let text: TextDocument
        let vision: VisionDocument

        enum CodingKeys: String, CodingKey {
            case architectures
            case modelType = "model_type"
            case tensorDType = "dtype"
            case bosTokenID = "bos_token_id"
            case eosTokenID = "eos_token_id"
            case padTokenID = "pad_token_id"
            case imageTokenID = "image_token_id"
            case quantization = "quantization_config"
            case text = "text_config"
            case vision = "vision_config"
        }
    }

    private struct TextDocument: Decodable {
        let modelType: String
        let vocabularySize: Int
        let hiddenSize: Int
        let expertIntermediateSize: Int
        let numberOfLayers: Int
        let numberOfAttentionHeads: Int
        let numberOfKeyValueHeads: Int
        let attentionHeadDimension: Int
        let ropeHeadDimension: Int
        let queryLowRank: Int
        let outputLowRank: Int
        let outputGroups: Int
        let hiddenActivation: String
        let swiGLULimit: Float
        let rmsNormEpsilon: Float
        let attentionBias: Bool
        let attentionDropout: Float
        let useCache: Bool
        let tiesWordEmbeddings: Bool
        let maximumPositionCount: Int
        let ropeTheta: Float
        let ropeScaling: RopeScaling
        let routedExpertCount: Int
        let sharedExpertCount: Int
        let expertsPerToken: Int
        let routingScoreFunction: String
        let topKMethod: String
        let normalizedTopK: Bool
        let routingScale: Float
        let slidingWindow: Int
        let compressionRatios: [Int]
        let compressionRopeTheta: Float
        let keyValueSourceLayers: [Int]
        let indexSourceLayers: [Int]
        let indexHeadCount: Int
        let indexHeadDimension: Int
        let indexTopK: Int
        let candidateSourceLayer: Int
        let candidateTopKBlocks: Int
        let candidateBlockSize: Int
        let hyperConnectionMultiplicity: Int
        let hyperConnectionSinkhornIterations: Int
        let hyperConnectionEpsilon: Float
        let engramLayerIDs: [Int]
        let engramRowCounts: [UInt64]
        let engramMaximumNGramSize: Int
        let engramVocabularySize: Int
        let engramHeadCount: Int
        let engramHeadDimension: Int
        let engramPadTokenID: Int
        let engramCompressedVocabularySize: Int
        let numberOfDraftLayers: Int
        let draftBlockSize: Int
        let draftNoiseTokenID: Int
        let draftTargetLayerIDs: [Int]
        let draftMarkovRank: Int
        let draftRoutedExpertCount: Int
        let draftExpertsPerToken: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case expertIntermediateSize = "moe_intermediate_size"
            case numberOfLayers = "num_hidden_layers"
            case numberOfAttentionHeads = "num_attention_heads"
            case numberOfKeyValueHeads = "num_key_value_heads"
            case attentionHeadDimension = "head_dim"
            case ropeHeadDimension = "qk_rope_head_dim"
            case queryLowRank = "q_lora_rank"
            case outputLowRank = "o_lora_rank"
            case outputGroups = "o_groups"
            case hiddenActivation = "hidden_act"
            case swiGLULimit = "swiglu_limit"
            case rmsNormEpsilon = "rms_norm_eps"
            case attentionBias = "attention_bias"
            case attentionDropout = "attention_dropout"
            case useCache = "use_cache"
            case tiesWordEmbeddings = "tie_word_embeddings"
            case maximumPositionCount = "max_position_embeddings"
            case ropeTheta = "rope_theta"
            case ropeScaling = "rope_scaling"
            case routedExpertCount = "n_routed_experts"
            case sharedExpertCount = "n_shared_experts"
            case expertsPerToken = "num_experts_per_tok"
            case routingScoreFunction = "scoring_func"
            case topKMethod = "topk_method"
            case normalizedTopK = "norm_topk_prob"
            case routingScale = "routed_scaling_factor"
            case slidingWindow = "sliding_window"
            case compressionRatios = "compress_ratios"
            case compressionRopeTheta = "compress_rope_theta"
            case keyValueSourceLayers = "kv_source_layer_ids"
            case indexSourceLayers = "index_source_layer_ids"
            case indexHeadCount = "index_n_heads"
            case indexHeadDimension = "index_head_dim"
            case indexTopK = "index_topk"
            case candidateSourceLayer = "candidate_source_layer_id"
            case candidateTopKBlocks = "candidate_topk_blocks"
            case candidateBlockSize = "candidate_block_size"
            case hyperConnectionMultiplicity = "hc_mult"
            case hyperConnectionSinkhornIterations = "hc_sinkhorn_iters"
            case hyperConnectionEpsilon = "hc_eps"
            case engramLayerIDs = "engram_layer_ids"
            case engramRowCounts = "engram_num_embeddings"
            case engramMaximumNGramSize = "engram_max_ngram_size"
            case engramVocabularySize = "engram_vocab_size"
            case engramHeadCount = "engram_n_heads"
            case engramHeadDimension = "engram_head_dim"
            case engramPadTokenID = "engram_pad_token_id"
            case engramCompressedVocabularySize = "engram_compressed_vocab_size"
            case numberOfDraftLayers = "num_nextn_predict_layers"
            case draftBlockSize = "dspark_block_size"
            case draftNoiseTokenID = "dspark_noise_token_id"
            case draftTargetLayerIDs = "dspark_target_layer_ids"
            case draftMarkovRank = "dspark_markov_rank"
            case draftRoutedExpertCount = "dspark_n_routed_experts"
            case draftExpertsPerToken = "dspark_num_experts_per_tok"
        }
    }

    private struct VisionDocument: Decodable {
        let modelType: String
        let numberOfLayers: Int
        let hiddenSize: Int
        let numberOfAttentionHeads: Int
        let intermediateSize: Int
        let patchSize: Int
        let ropeTheta: Float
        let downsampleRatio: Int
        let maximumImageTokens: Int
        let minimumPixels: Int
        let maximumWidthHeightRatio: Float?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case numberOfLayers = "num_hidden_layers"
            case hiddenSize = "hidden_size"
            case numberOfAttentionHeads = "num_attention_heads"
            case intermediateSize = "intermediate_size"
            case patchSize = "patch_size"
            case ropeTheta = "rope_theta"
            case downsampleRatio = "downsample_ratio"
            case maximumImageTokens = "max_image_tokens"
            case minimumPixels = "min_pixels"
            case maximumWidthHeightRatio = "max_wh_ratio"
        }
    }

    private struct Quantization: Decodable {
        let method: String
        let activationScheme: String
        let scaleFormat: String
        let expertDType: String
        let weightBlockSize: [Int]

        enum CodingKeys: String, CodingKey {
            case method = "quant_method"
            case activationScheme = "activation_scheme"
            case scaleFormat = "scale_fmt"
            case expertDType = "expert_dtype"
            case weightBlockSize = "weight_block_size"
        }
    }

    private struct RopeScaling: Decodable {
        let type: String
        let factor: Float
        let originalMaximumPositions: Int
        let betaFast: Int
        let betaSlow: Int

        enum CodingKeys: String, CodingKey {
            case type = "rope_type"
            case factor
            case originalMaximumPositions = "original_max_position_embeddings"
            case betaFast = "beta_fast"
            case betaSlow = "beta_slow"
        }
    }
}
