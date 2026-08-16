import Foundation

/// The Qwen3-MoE architecture, read from `config.json`.
///
/// The second model adapter (spec §15.4 deliverable): its existence is what
/// forces the 5.5 separation between the storage runtime and model-specific
/// logic to be real rather than asserted. Nothing here is shared with the K3
/// adapter, and nothing here reaches below `ModelAdapters`.
///
/// ## What this architecture is *not*
///
/// It is worth saying plainly, because the K3 adapter sitting next to it does
/// all of these and a reader will pattern-match:
///
/// - no KDA and no MLA — 48 identical GQA layers;
/// - no `situ` — the experts are ordinary SwiGLU;
/// - no AttnRes, no latent MoE projection, no routed-expert norm;
/// - **no shared expert**, so the resident set is attention, norms and the
///   router only, and every FFN byte a token needs is routing-dependent;
/// - no router correction bias, no routed scaling factor, and the router is
///   **softmax before top-k**, not sigmoid.
///
/// ## Router semantics
///
/// Transcribed from `mlx_lm/models/qwen3_moe.py` in mlx-lm 0.31.3, which is the
/// reference these fixtures are generated from:
///
/// ```python
/// gates = self.gate(x)
/// gates = mx.softmax(gates, axis=-1, precise=True)
/// k = self.top_k
/// inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
/// scores = mx.take_along_axis(gates, inds, axis=-1)
/// if self.norm_topk_prob:
///     scores /= mx.sum(scores, axis=-1, keepdims=True)
/// y = self.switch_mlp(x, inds)
/// y = (y * scores[..., None]).sum(axis=-2)
/// ```
///
/// Three consequences that are easy to get wrong in the same direction:
///
/// - the softmax is over **all** experts and happens **before** the top-k, so
///   the renormalised weights are a slice of a full-vocabulary distribution and
///   not a softmax of the selected logits;
/// - `argpartition` promises no order among the selected `k`, so ids are a
///   **set** — the same conclusion M6A reached about `torch.topk(sorted=False)`;
/// - the routing margin must be measured in the space that ranks, which here is
///   the post-softmax `gates` and not the raw logits.
public struct QwenMoEConfig: Codable, Sendable, Equatable {

    public var modelType: String
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var vocabSize: Int
    public var tieWordEmbeddings: Bool

    /// Dense-MLP width, used only by layers listed in ``mlpOnlyLayers``.
    public var intermediateSize: Int
    public var moeIntermediateSize: Int
    public var numExperts: Int
    public var numExpertsPerToken: Int
    public var normTopKProb: Bool
    public var decoderSparseStep: Int
    public var mlpOnlyLayers: [Int]

    /// Quantization of everything except the per-layer router.
    public var quantGroupSize: Int
    public var quantBits: Int
    /// The router's own quantization. 8-bit in this checkpoint while the rest is
    /// 4-bit, declared per tensor in `config.json` — spec §10.2's mixed
    /// quantization, in a checkpoint rather than in theory.
    public var routerGroupSize: Int
    public var routerBits: Int

    /// True when layer `index` routes to experts rather than a dense MLP.
    ///
    /// `decoder_sparse_step` is 1 and `mlp_only_layers` is empty for
    /// Qwen3-30B-A3B, so every layer is sparse — but the rule is transcribed
    /// rather than collapsed, because it is the checkpoint's rule and a sibling
    /// model changes it.
    public func isSparse(layer index: Int) -> Bool {
        !mlpOnlyLayers.contains(index) && numExperts > 0
            && (index + 1) % max(1, decoderSparseStep) == 0
    }

    /// Queries per key/value head.
    public var groupedQueryFactor: Int { numAttentionHeads / numKeyValueHeads }

    /// `head_dim ** -0.5`, the attention scale.
    public var attentionScale: Float { 1.0 / Foundation.sqrt(Float(headDim)) }

    public var expertsPerLayerBytesNote: String {
        "one expert projection is \(moeIntermediateSize)x\(hiddenSize)"
    }

    // MARK: - Loading

    private enum TopLevel: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case vocabSize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case normTopKProb = "norm_topk_prob"
        case decoderSparseStep = "decoder_sparse_step"
        case mlpOnlyLayers = "mlp_only_layers"
        case ropeScaling = "rope_scaling"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: TopLevel.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (hiddenSize / numAttentionHeads)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? false
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        numExperts = try container.decode(Int.self, forKey: .numExperts)
        numExpertsPerToken = try container.decode(Int.self, forKey: .numExpertsPerToken)
        normTopKProb = try container.decodeIfPresent(Bool.self, forKey: .normTopKProb) ?? true
        decoderSparseStep = try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        mlpOnlyLayers = try container.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []

        // `rope_scaling` is null in this checkpoint. It is *rejected* rather
        // than ignored when present: a scaled RoPE silently unapplied would move
        // every position and still produce plausible text (spec §5.8).
        let scaling = try? container.decodeIfPresent([String: JSONAny].self, forKey: .ropeScaling)
        if let scaling, !scaling.isEmpty {
            throw TinyK3Error.configuration(
                "rope_scaling \(scaling.keys.sorted()) is set; this adapter implements plain "
                    + "RoPE only and will not silently ignore a scaling rule")
        }

        // Quantization lives in a sibling object whose keys are a mixture of two
        // scalars and one entry per per-tensor override, so it is not a struct.
        quantGroupSize = 64
        quantBits = 4
        routerGroupSize = 64
        routerBits = 8
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: TopLevel.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try container.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try container.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(vocabSize, forKey: .vocabSize)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try container.encode(numExperts, forKey: .numExperts)
        try container.encode(numExpertsPerToken, forKey: .numExpertsPerToken)
        try container.encode(normTopKProb, forKey: .normTopKProb)
        try container.encode(decoderSparseStep, forKey: .decoderSparseStep)
        try container.encode(mlpOnlyLayers, forKey: .mlpOnlyLayers)
    }

    /// Read `config.json`, including the quantization block.
    public static func load(directory: String) throws -> QwenMoEConfig {
        let url = URL(fileURLWithPath: directory).appendingPathComponent("config.json")
        let data = try Data(contentsOf: url)
        var config = try JSONDecoder().decode(QwenMoEConfig.self, from: data)

        guard config.modelType == "qwen3_moe" else {
            throw TinyK3Error.configuration(
                "\(url.path) declares model_type '\(config.modelType)'; this adapter implements "
                    + "qwen3_moe")
        }

        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let quantization = root["quantization"] as? [String: Any],
            let groupSize = quantization["group_size"] as? Int,
            let bits = quantization["bits"] as? Int
        else {
            throw TinyK3Error.configuration("\(url.path) has no usable 'quantization' block")
        }
        config.quantGroupSize = groupSize
        config.quantBits = bits

        // Every `model.layers.N.mlp.gate` carries its own override. They are read
        // and required to agree with each other, rather than one being sampled:
        // a checkpoint that quantized one router differently would otherwise be
        // loaded with 47 correct routers and one silently wrong one.
        var routerOverrides: Set<[Int]> = []
        for index in 0..<config.numHiddenLayers {
            let key = "model.layers.\(index).mlp.gate"
            guard let entry = quantization[key] as? [String: Any],
                let groupSize = entry["group_size"] as? Int,
                let bits = entry["bits"] as? Int
            else {
                throw TinyK3Error.configuration(
                    "\(url.path) has no quantization entry for \(key); this checkpoint declares "
                        + "the router's 8-bit override per layer and a missing one would be read "
                        + "as 4-bit")
            }
            routerOverrides.insert([groupSize, bits])
        }
        guard routerOverrides.count == 1, let router = routerOverrides.first else {
            throw TinyK3Error.configuration(
                "the per-layer router quantization overrides disagree: \(routerOverrides.sorted { $0.lexicographicallyPrecedes($1) })"
            )
        }
        config.routerGroupSize = router[0]
        config.routerBits = router[1]
        return config
    }
}

/// Minimal `Any`-shaped decoder, used only to notice that `rope_scaling` is
/// present without committing to a schema this adapter does not implement.
struct JSONAny: Codable {
    init(from decoder: Decoder) throws {
        _ = try decoder.singleValueContainer()
    }
    func encode(to encoder: Encoder) throws {}
}

/// Tensor names, in one place.
///
/// The checkpoint's routed experts are *stacked* under `switch_mlp` — mlx-lm's
/// `sanitize` joins the per-expert tensors at conversion time — so the names
/// below are what is actually in the file, not what the HF modelling code uses.
public enum QwenTensors {
    public static let embedding = "model.embed_tokens"
    public static let finalNorm = "model.norm.weight"
    public static let lmHead = "lm_head"

    public static func layer(_ index: Int) -> String { "model.layers.\(index)" }
    public static func inputNorm(_ index: Int) -> String { "\(layer(index)).input_layernorm.weight" }
    public static func postAttentionNorm(_ index: Int) -> String {
        "\(layer(index)).post_attention_layernorm.weight"
    }
    public static func attention(_ index: Int, _ projection: String) -> String {
        "\(layer(index)).self_attn.\(projection)"
    }
    public static func queryNorm(_ index: Int) -> String {
        "\(layer(index)).self_attn.q_norm.weight"
    }
    public static func keyNorm(_ index: Int) -> String { "\(layer(index)).self_attn.k_norm.weight" }
    public static func router(_ index: Int) -> String { "\(layer(index)).mlp.gate" }
    public static func experts(_ index: Int, _ projection: String) -> String {
        "\(layer(index)).mlp.switch_mlp.\(projection)"
    }
}

/// Which of an expert's three projections is being fetched, in the checkpoint's
/// own vocabulary.
///
/// Deliberately a different type from the K3 adapter's ``ExpertProjection``:
/// K3's `w1/w2/w3` name a `situ` gate/down/up triple, and reusing that spelling
/// here would suggest the two activations are interchangeable. They are not —
/// this is a plain SwiGLU.
public enum QwenExpertProjection: String, CaseIterable, Sendable {
    case gateProj = "gate_proj"
    case upProj = "up_proj"
    case downProj = "down_proj"

    /// The container-file slot this projection is written to. Matches the
    /// converter's `PROJECTION_SLOTS`.
    public var slot: String {
        switch self {
        case .gateProj: return "w1"
        case .upProj: return "w3"
        case .downProj: return "w2"
        }
    }
}
