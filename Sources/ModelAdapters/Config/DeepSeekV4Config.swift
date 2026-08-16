import Foundation

/// Named failures at the DeepSeek V4 architecture boundary.
///
/// A V4 repository may move to a new immutable commit without an App release.
/// That does not make every future architecture change compatible: the current
/// verified `config.json` is decoded and checked at run preparation, and a
/// changed field is either understood here or refused by name.
public enum DeepSeekV4Error: Error, CustomStringConvertible, Equatable {
    case configuration(String)
    case unsupportedArchitecture(String)
    case routing(String)
    case hyperConnection(String)
    case attention(String)
    case compression(String)
    case indexing(String)
    case artifact(String)
    case experts(String)

    public var description: String {
        switch self {
        case .configuration(let detail):
            return "invalid DeepSeek V4 configuration: \(detail)"
        case .unsupportedArchitecture(let detail):
            return "unsupported DeepSeek V4 architecture: \(detail)"
        case .routing(let detail):
            return "invalid DeepSeek V4 routing input: \(detail)"
        case .hyperConnection(let detail):
            return "invalid DeepSeek V4 hyper-connection input: \(detail)"
        case .attention(let detail):
            return "invalid DeepSeek V4 attention input: \(detail)"
        case .compression(let detail):
            return "invalid DeepSeek V4 compression input: \(detail)"
        case .indexing(let detail):
            return "invalid DeepSeek V4 index input: \(detail)"
        case .artifact(let detail):
            return "invalid DeepSeek V4 layer artifact: \(detail)"
        case .experts(let detail):
            return "invalid DeepSeek V4 expert artifact: \(detail)"
        }
    }
}

/// The product-relevant DeepSeek V4 architecture, decoded from verified bytes.
///
/// This type intentionally carries no repository revision. Repository identity
/// discovers a possible V4 artifact; the verified config at that revision says
/// whether this adapter can execute it. The strict checks below prevent a later
/// V4 architecture change from being silently interpreted as today's model.
public struct DeepSeekV4Config: Sendable, Equatable {
    public let modelType: String
    public let architecture: String

    public let hiddenSize: Int
    public let vocabularySize: Int
    public let numberOfLayers: Int
    public let numberOfMTPPredictors: Int
    public let numberOfHashLayers: Int

    public let numberOfAttentionHeads: Int
    public let numberOfKeyValueHeads: Int
    public let attentionHeadDimension: Int
    public let queryLowRank: Int
    public let ropeHeadDimension: Int
    public let outputGroups: Int
    public let outputLowRank: Int
    public let slidingWindow: Int
    public let compressionRatios: [Int]
    public let compressionRopeTheta: Float

    public let indexHeadCount: Int
    public let indexHeadDimension: Int
    public let indexTopK: Int

    public let routedExpertCount: Int
    public let expertsPerToken: Int
    public let sharedExpertCount: Int
    public let expertIntermediateSize: Int
    public let normalizedTopK: Bool
    public let routingScale: Float
    public let routingScoreFunction: String
    public let topKMethod: String
    public let swiGLULimit: Float

    public let hyperConnectionMultiplicity: Int
    public let hyperConnectionSinkhornIterations: Int
    public let hyperConnectionEpsilon: Float

    public let rmsNormEpsilon: Float
    public let ropeTheta: Float
    public let ropeFactor: Float
    public let ropeOriginalLength: Int
    public let ropeBetaFast: Int
    public let ropeBetaSlow: Int
    public let maximumPositionCount: Int
    public let bosTokenID: Int
    public let eosTokenID: Int
    public let tiesWordEmbeddings: Bool

    public let expertDType: String
    public let quantizationMethod: String
    public let quantizedWeightFormat: String
    public let quantizedScaleFormat: String
    public let quantizedWeightBlock: [Int]

    public var nonRopeHeadDimension: Int {
        attentionHeadDimension - ropeHeadDimension
    }

    public func compressionRatio(layer: Int) throws -> Int {
        guard layer >= 0, layer < numberOfLayers else {
            throw DeepSeekV4Error.configuration(
                "layer \(layer) is outside 0..<\(numberOfLayers)")
        }
        return compressionRatios[layer]
    }

    public func usesHashRouting(layer: Int) -> Bool {
        layer >= 0 && layer < numberOfHashLayers
    }

    public init(contentsOf url: URL) throws {
        try self.init(json: Data(contentsOf: url))
    }

    public init(json data: Data) throws {
        let decoded: Document
        do {
            decoded = try JSONDecoder().decode(Document.self, from: data)
        } catch {
            throw DeepSeekV4Error.configuration("config.json could not be decoded: \(error)")
        }

        guard decoded.modelType == "deepseek_v4" else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "model_type '\(decoded.modelType)' is not 'deepseek_v4'")
        }
        guard decoded.architectures.count == 1,
            decoded.architectures[0] == "DeepseekV4ForCausalLM"
        else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "architectures \(decoded.architectures) do not name DeepseekV4ForCausalLM")
        }
        guard decoded.hiddenActivation == "silu" else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "hidden_act '\(decoded.hiddenActivation)' is not silu")
        }
        guard decoded.attentionBias == false, decoded.attentionDropout == 0 else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "attention bias or nonzero attention dropout is not implemented")
        }
        guard decoded.tensorDType == "bfloat16" else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "torch_dtype '\(decoded.tensorDType)' is not bfloat16")
        }
        guard decoded.useCache else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "use_cache is false; the product decoder requires incremental cache semantics")
        }
        guard !decoded.tiesWordEmbeddings else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "tie_word_embeddings is true; the artifact carries a distinct output head")
        }
        guard decoded.routingScoreFunction == "sqrtsoftplus",
            decoded.topKMethod == "noaux_tc"
        else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "routing requires scoring_func sqrtsoftplus and topk_method noaux_tc")
        }
        guard decoded.expertDType == "fp4" else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "expert_dtype '\(decoded.expertDType)' is not fp4")
        }
        guard decoded.quantization.method == "fp8",
            decoded.quantization.format == "e4m3",
            decoded.quantization.scaleFormat == "ue8m0",
            decoded.quantization.activationScheme == "dynamic",
            decoded.quantization.weightBlockSize == [128, 128]
        else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "the non-expert quantization is not dynamic e4m3/ue8m0 block-FP8 [128, 128]")
        }
        guard decoded.ropeScaling.type.lowercased() == "yarn" else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "rope_scaling.type '\(decoded.ropeScaling.type)' is not yarn")
        }

        func requirePositive(_ value: Int, _ name: String) throws {
            guard value > 0 else {
                throw DeepSeekV4Error.configuration("\(name) must be positive, got \(value)")
            }
        }
        try requirePositive(decoded.hiddenSize, "hidden_size")
        try requirePositive(decoded.vocabularySize, "vocab_size")
        try requirePositive(decoded.numberOfLayers, "num_hidden_layers")
        try requirePositive(decoded.numberOfAttentionHeads, "num_attention_heads")
        try requirePositive(decoded.numberOfKeyValueHeads, "num_key_value_heads")
        try requirePositive(decoded.attentionHeadDimension, "head_dim")
        try requirePositive(decoded.queryLowRank, "q_lora_rank")
        try requirePositive(decoded.ropeHeadDimension, "qk_rope_head_dim")
        try requirePositive(decoded.outputGroups, "o_groups")
        try requirePositive(decoded.outputLowRank, "o_lora_rank")
        try requirePositive(decoded.slidingWindow, "sliding_window")
        try requirePositive(decoded.indexHeadCount, "index_n_heads")
        try requirePositive(decoded.indexHeadDimension, "index_head_dim")
        try requirePositive(decoded.indexTopK, "index_topk")
        try requirePositive(decoded.routedExpertCount, "n_routed_experts")
        try requirePositive(decoded.expertsPerToken, "num_experts_per_tok")
        try requirePositive(decoded.sharedExpertCount, "n_shared_experts")
        try requirePositive(decoded.expertIntermediateSize, "moe_intermediate_size")
        try requirePositive(decoded.hyperConnectionMultiplicity, "hc_mult")
        try requirePositive(decoded.hyperConnectionSinkhornIterations, "hc_sinkhorn_iters")
        try requirePositive(decoded.ropeScaling.originalMaximumPositions,
            "rope_scaling.original_max_position_embeddings")
        try requirePositive(decoded.ropeScaling.betaFast, "rope_scaling.beta_fast")
        try requirePositive(decoded.ropeScaling.betaSlow, "rope_scaling.beta_slow")
        try requirePositive(decoded.maximumPositionCount, "max_position_embeddings")

        guard decoded.numberOfHashLayers >= 0,
            decoded.numberOfHashLayers <= decoded.numberOfLayers
        else {
            throw DeepSeekV4Error.configuration(
                "num_hash_layers \(decoded.numberOfHashLayers) is outside 0...\(decoded.numberOfLayers)")
        }
        guard decoded.numberOfMTPPredictors >= 0 else {
            throw DeepSeekV4Error.configuration("num_nextn_predict_layers cannot be negative")
        }
        // The official DeepSeek V4 configuration converts every legacy
        // `compress_ratios` value to a layer type, then takes the prefix for
        // `num_hidden_layers`. Current owned artifacts also carry trailing
        // dSpark schedule entries; `num_nextn_predict_layers` is not the array
        // length contract. Refuse a missing main-layer decision while retaining
        // and validating the complete published schedule.
        guard decoded.compressionRatios.count >= decoded.numberOfLayers else {
            throw DeepSeekV4Error.configuration(
                "compress_ratios has \(decoded.compressionRatios.count) entries; "
                    + "num_hidden_layers requires at least \(decoded.numberOfLayers)")
        }
        guard decoded.compressionRatios.allSatisfy({ $0 == 0 || $0 == 4 || $0 == 128 }) else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "compress_ratios contains a value other than 0, 4, or 128")
        }
        guard decoded.ropeHeadDimension < decoded.attentionHeadDimension else {
            throw DeepSeekV4Error.configuration(
                "qk_rope_head_dim must be smaller than head_dim")
        }
        guard decoded.outputGroups <= decoded.numberOfAttentionHeads,
            decoded.numberOfAttentionHeads % decoded.outputGroups == 0
        else {
            throw DeepSeekV4Error.configuration(
                "num_attention_heads is not divisible by o_groups")
        }
        guard decoded.indexHeadCount == decoded.numberOfAttentionHeads else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "index_n_heads must equal num_attention_heads")
        }
        guard decoded.indexTopK <= decoded.maximumPositionCount else {
            throw DeepSeekV4Error.configuration(
                "index_topk exceeds max_position_embeddings")
        }
        guard decoded.expertsPerToken <= decoded.routedExpertCount else {
            throw DeepSeekV4Error.configuration(
                "num_experts_per_tok exceeds n_routed_experts")
        }
        guard decoded.sharedExpertCount == 1 else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "n_shared_experts \(decoded.sharedExpertCount); this adapter implements exactly one")
        }
        guard decoded.numberOfKeyValueHeads == 1 else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "num_key_value_heads \(decoded.numberOfKeyValueHeads); compressed MQA requires one")
        }
        guard decoded.rmsNormEpsilon.isFinite, decoded.rmsNormEpsilon > 0,
            decoded.hyperConnectionEpsilon.isFinite, decoded.hyperConnectionEpsilon > 0,
            decoded.routingScale.isFinite, decoded.routingScale > 0,
            decoded.swiGLULimit.isFinite, decoded.swiGLULimit >= 0,
            decoded.ropeTheta.isFinite, decoded.ropeTheta > 0,
            decoded.compressionRopeTheta.isFinite, decoded.compressionRopeTheta > 0,
            decoded.ropeScaling.factor.isFinite, decoded.ropeScaling.factor > 0
        else {
            throw DeepSeekV4Error.configuration(
                "normalization, hyper-connection, routing, SwiGLU, and rope scalars must be finite")
        }
        guard decoded.ropeScaling.betaFast >= decoded.ropeScaling.betaSlow else {
            throw DeepSeekV4Error.configuration(
                "rope_scaling.beta_fast must not be smaller than beta_slow")
        }
        guard decoded.ropeScaling.originalMaximumPositions <= decoded.maximumPositionCount else {
            throw DeepSeekV4Error.configuration(
                "rope_scaling.original_max_position_embeddings exceeds max_position_embeddings")
        }
        guard decoded.bosTokenID >= 0, decoded.bosTokenID < decoded.vocabularySize,
            decoded.eosTokenID >= 0, decoded.eosTokenID < decoded.vocabularySize,
            decoded.bosTokenID != decoded.eosTokenID
        else {
            throw DeepSeekV4Error.configuration(
                "bos_token_id and eos_token_id must be distinct ids inside the vocabulary")
        }

        modelType = decoded.modelType
        architecture = decoded.architectures[0]
        hiddenSize = decoded.hiddenSize
        vocabularySize = decoded.vocabularySize
        numberOfLayers = decoded.numberOfLayers
        numberOfMTPPredictors = decoded.numberOfMTPPredictors
        numberOfHashLayers = decoded.numberOfHashLayers
        numberOfAttentionHeads = decoded.numberOfAttentionHeads
        numberOfKeyValueHeads = decoded.numberOfKeyValueHeads
        attentionHeadDimension = decoded.attentionHeadDimension
        queryLowRank = decoded.queryLowRank
        ropeHeadDimension = decoded.ropeHeadDimension
        outputGroups = decoded.outputGroups
        outputLowRank = decoded.outputLowRank
        slidingWindow = decoded.slidingWindow
        compressionRatios = decoded.compressionRatios
        compressionRopeTheta = decoded.compressionRopeTheta
        indexHeadCount = decoded.indexHeadCount
        indexHeadDimension = decoded.indexHeadDimension
        indexTopK = decoded.indexTopK
        routedExpertCount = decoded.routedExpertCount
        expertsPerToken = decoded.expertsPerToken
        sharedExpertCount = decoded.sharedExpertCount
        expertIntermediateSize = decoded.expertIntermediateSize
        normalizedTopK = decoded.normalizedTopK
        routingScale = decoded.routingScale
        routingScoreFunction = decoded.routingScoreFunction
        topKMethod = decoded.topKMethod
        swiGLULimit = decoded.swiGLULimit
        hyperConnectionMultiplicity = decoded.hyperConnectionMultiplicity
        hyperConnectionSinkhornIterations = decoded.hyperConnectionSinkhornIterations
        hyperConnectionEpsilon = decoded.hyperConnectionEpsilon
        rmsNormEpsilon = decoded.rmsNormEpsilon
        ropeTheta = decoded.ropeTheta
        ropeFactor = decoded.ropeScaling.factor
        ropeOriginalLength = decoded.ropeScaling.originalMaximumPositions
        ropeBetaFast = decoded.ropeScaling.betaFast
        ropeBetaSlow = decoded.ropeScaling.betaSlow
        maximumPositionCount = decoded.maximumPositionCount
        bosTokenID = decoded.bosTokenID
        eosTokenID = decoded.eosTokenID
        tiesWordEmbeddings = decoded.tiesWordEmbeddings
        expertDType = decoded.expertDType
        quantizationMethod = decoded.quantization.method
        quantizedWeightFormat = decoded.quantization.format
        quantizedScaleFormat = decoded.quantization.scaleFormat
        quantizedWeightBlock = decoded.quantization.weightBlockSize
    }

    private struct Document: Decodable {
        let architectures: [String]
        let modelType: String
        let attentionBias: Bool
        let attentionDropout: Float
        let hiddenActivation: String
        let hiddenSize: Int
        let vocabularySize: Int
        let numberOfLayers: Int
        let numberOfMTPPredictors: Int
        let numberOfHashLayers: Int
        let numberOfAttentionHeads: Int
        let numberOfKeyValueHeads: Int
        let attentionHeadDimension: Int
        let queryLowRank: Int
        let ropeHeadDimension: Int
        let outputGroups: Int
        let outputLowRank: Int
        let slidingWindow: Int
        let compressionRatios: [Int]
        let compressionRopeTheta: Float
        let indexHeadCount: Int
        let indexHeadDimension: Int
        let indexTopK: Int
        let routedExpertCount: Int
        let expertsPerToken: Int
        let sharedExpertCount: Int
        let expertIntermediateSize: Int
        let normalizedTopK: Bool
        let routingScale: Float
        let routingScoreFunction: String
        let topKMethod: String
        let swiGLULimit: Float
        let hyperConnectionMultiplicity: Int
        let hyperConnectionSinkhornIterations: Int
        let hyperConnectionEpsilon: Float
        let rmsNormEpsilon: Float
        let ropeTheta: Float
        let maximumPositionCount: Int
        let bosTokenID: Int
        let eosTokenID: Int
        let tiesWordEmbeddings: Bool
        let tensorDType: String
        let useCache: Bool
        let expertDType: String
        let quantization: Quantization
        let ropeScaling: RopeScaling

        enum CodingKeys: String, CodingKey {
            case architectures
            case modelType = "model_type"
            case attentionBias = "attention_bias"
            case attentionDropout = "attention_dropout"
            case hiddenActivation = "hidden_act"
            case hiddenSize = "hidden_size"
            case vocabularySize = "vocab_size"
            case numberOfLayers = "num_hidden_layers"
            case numberOfMTPPredictors = "num_nextn_predict_layers"
            case numberOfHashLayers = "num_hash_layers"
            case numberOfAttentionHeads = "num_attention_heads"
            case numberOfKeyValueHeads = "num_key_value_heads"
            case attentionHeadDimension = "head_dim"
            case queryLowRank = "q_lora_rank"
            case ropeHeadDimension = "qk_rope_head_dim"
            case outputGroups = "o_groups"
            case outputLowRank = "o_lora_rank"
            case slidingWindow = "sliding_window"
            case compressionRatios = "compress_ratios"
            case compressionRopeTheta = "compress_rope_theta"
            case indexHeadCount = "index_n_heads"
            case indexHeadDimension = "index_head_dim"
            case indexTopK = "index_topk"
            case routedExpertCount = "n_routed_experts"
            case expertsPerToken = "num_experts_per_tok"
            case sharedExpertCount = "n_shared_experts"
            case expertIntermediateSize = "moe_intermediate_size"
            case normalizedTopK = "norm_topk_prob"
            case routingScale = "routed_scaling_factor"
            case routingScoreFunction = "scoring_func"
            case topKMethod = "topk_method"
            case swiGLULimit = "swiglu_limit"
            case hyperConnectionMultiplicity = "hc_mult"
            case hyperConnectionSinkhornIterations = "hc_sinkhorn_iters"
            case hyperConnectionEpsilon = "hc_eps"
            case rmsNormEpsilon = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case maximumPositionCount = "max_position_embeddings"
            case bosTokenID = "bos_token_id"
            case eosTokenID = "eos_token_id"
            case tiesWordEmbeddings = "tie_word_embeddings"
            case tensorDType = "torch_dtype"
            case useCache = "use_cache"
            case expertDType = "expert_dtype"
            case quantization = "quantization_config"
            case ropeScaling = "rope_scaling"
        }
    }

    private struct Quantization: Decodable {
        let activationScheme: String
        let format: String
        let method: String
        let scaleFormat: String
        let weightBlockSize: [Int]

        enum CodingKeys: String, CodingKey {
            case activationScheme = "activation_scheme"
            case format = "fmt"
            case method = "quant_method"
            case scaleFormat = "scale_fmt"
            case weightBlockSize = "weight_block_size"
        }
    }

    private struct RopeScaling: Decodable {
        let betaFast: Int
        let betaSlow: Int
        let factor: Float
        let originalMaximumPositions: Int
        let type: String

        enum CodingKeys: String, CodingKey {
            case betaFast = "beta_fast"
            case betaSlow = "beta_slow"
            case factor
            case originalMaximumPositions = "original_max_position_embeddings"
            case type
        }
    }
}
