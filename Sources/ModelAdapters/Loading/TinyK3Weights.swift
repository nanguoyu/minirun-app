import Foundation
import MLX
import MLXBridge

/// The half of a layer's weights the **attention** phase consumes.
///
/// The split is by consumer, read off ``K3Model/forward(tokenIds:state:)``'s
/// layer body rather than off the tensor names: everything here is dead by the
/// time the layer's post-attention AttnRes runs, so a source that streams a
/// layer can return these bytes before it widens the other half. Nothing is
/// consumed by both phases; if a tensor ever were, it would belong to
/// ``TinyK3FeedForwardWeights`` — the half that is released later — because a
/// weight released before its last reader is a correctness bug and not an
/// optimisation.
public struct TinyK3AttentionWeights {
    public var inputLayerNorm: MLXArray
    /// `self_attention_res_norm.weight * self_attention_res_proj.weight[0]`.
    /// The two are only ever used as this product, so the product is what is
    /// kept — it is also the form the `subop-attnres` bundle records. Consumed
    /// by the AttnRes stage that runs *before* attention.
    public var selfAttentionResScore: MLXArray
    public var kda: K3KDA.Weights?
    public var mla: K3MLA.Weights?

    public init(
        inputLayerNorm: MLXArray, selfAttentionResScore: MLXArray,
        kda: K3KDA.Weights? = nil, mla: K3MLA.Weights? = nil
    ) {
        self.inputLayerNorm = inputLayerNorm
        self.selfAttentionResScore = selfAttentionResScore
        self.kda = kda
        self.mla = mla
    }

    /// Exactly the arrays this half keeps alive. Enumerated rather than
    /// derived, so a residency figure computed from it is the real one.
    public var residentArrays: [MLXArray] {
        var list = [inputLayerNorm, selfAttentionResScore]
        if let kda {
            list += [
                kda.qProj, kda.kProj, kda.vProj, kda.qConv, kda.kConv, kda.vConv,
                kda.fAProj, kda.fBProj, kda.bProj, kda.gProj, kda.aLog, kda.dtBias,
                kda.oNormWeight, kda.oProj,
            ]
        }
        if let mla {
            list += [
                mla.qAProj, mla.qALayerNorm, mla.qBProj, mla.kvAProjWithMQA,
                mla.kvALayerNorm, mla.kvBProj, mla.gProj, mla.oProj,
            ]
        }
        return list
    }
}

/// The half of a layer's weights the **MLP / MoE** phase consumes.
///
/// `mlpResScore` is in this half although the AttnRes stage that reads it runs
/// before the MLP: that stage is still after attention has finished, which is
/// the boundary that matters. Grouping it with the attention half would mean
/// widening it a phase early for no saving.
public struct TinyK3FeedForwardWeights {
    public var postAttentionLayerNorm: MLXArray
    /// `mlp_res_norm.weight * mlp_res_proj.weight[0]`.
    public var mlpResScore: MLXArray
    public var dense: K3MLP.DenseWeights?
    public var moe: K3MoE.Weights?

    public init(
        postAttentionLayerNorm: MLXArray, mlpResScore: MLXArray,
        dense: K3MLP.DenseWeights? = nil, moe: K3MoE.Weights? = nil
    ) {
        self.postAttentionLayerNorm = postAttentionLayerNorm
        self.mlpResScore = mlpResScore
        self.dense = dense
        self.moe = moe
    }

    public var residentArrays: [MLXArray] {
        var list = [postAttentionLayerNorm, mlpResScore]
        if let dense {
            list += [dense.gateProj, dense.upProj, dense.downProj]
        }
        if let moe {
            list += [moe.gateWeight, moe.correctionBias, moe.routedNorm]
            list += [moe.routedDownProj, moe.routedUpProj].flatMap(\.residentArrays)
            list += [moe.shared.gateProj, moe.shared.upProj, moe.shared.downProj]
                .flatMap(\.residentArrays)
        }
        return list
    }
}

/// Every weight one decoder layer needs, resolved once at load time.
///
/// The two halves are the layer's real structure — a streaming source hands
/// them over separately so the first can be released before the second is
/// widened — and this is the whole-layer view, for a caller that wants the
/// layer at once and for the resident checkpoint that never released anything.
public struct TinyK3LayerWeights {
    public var attention: TinyK3AttentionWeights
    public var feedForward: TinyK3FeedForwardWeights

    public init(
        attention: TinyK3AttentionWeights, feedForward: TinyK3FeedForwardWeights
    ) {
        self.attention = attention
        self.feedForward = feedForward
    }

    public init(
        inputLayerNorm: MLXArray, postAttentionLayerNorm: MLXArray,
        selfAttentionResScore: MLXArray, mlpResScore: MLXArray,
        kda: K3KDA.Weights? = nil, mla: K3MLA.Weights? = nil,
        dense: K3MLP.DenseWeights? = nil, moe: K3MoE.Weights? = nil
    ) {
        self.attention = TinyK3AttentionWeights(
            inputLayerNorm: inputLayerNorm, selfAttentionResScore: selfAttentionResScore,
            kda: kda, mla: mla)
        self.feedForward = TinyK3FeedForwardWeights(
            postAttentionLayerNorm: postAttentionLayerNorm, mlpResScore: mlpResScore,
            dense: dense, moe: moe)
    }

    public var inputLayerNorm: MLXArray { attention.inputLayerNorm }
    public var selfAttentionResScore: MLXArray { attention.selfAttentionResScore }
    public var kda: K3KDA.Weights? { attention.kda }
    public var mla: K3MLA.Weights? { attention.mla }
    public var postAttentionLayerNorm: MLXArray { feedForward.postAttentionLayerNorm }
    public var mlpResScore: MLXArray { feedForward.mlpResScore }
    public var dense: K3MLP.DenseWeights? { feedForward.dense }
    public var moe: K3MoE.Weights? { feedForward.moe }
}

/// The resident half of the model: everything that is not a routed expert.
///
/// ## What is resident and what is not
///
/// - **Routed experts** stream through the pager. 12.04 MB across 21 container
///   files, held four-experts-to-a-tile.
/// - **`embed_tokens`** (640 MiB) is never materialised. Token rows are read
///   individually out of the mapped safetensors, which is what spec §16.4 asks
///   for and what keeps a 163,840-row vocabulary off the heap.
/// - **`lm_head`** (640 MiB) is never materialised either: the logits are
///   computed by walking its rows in chunks. See ``TinyK3Model/logits(for:)``.
/// - **Everything else** is resident: the norms, the KDA and MLA projections,
///   layer 0's dense MLP, and the MXFP4 latent projections and shared experts.
///   That is roughly 60 MB, and none of it is what the milestone is about.
///
/// ## The dtype the checkpoint actually uses
///
/// `config.dtype` says `bfloat16`; **every non-MoE tensor in this file is F32**
/// (M6A record, "Inputs"). Weights are therefore loaded at whatever the
/// safetensors header declares and kept in float32, not cast down to bf16 — a
/// bf16 round-trip carries 8 mantissa bits, i.e. ~4e-3 relative, which is 40×
/// the per-layer tolerance this port is gated on. Reading the header rather
/// than the config is what makes that a non-issue instead of a mystery.
public final class TinyK3Weights {

    public let file: SafetensorsFile
    public let config: TinyK3Config
    public let layers: [TinyK3LayerWeights]
    /// `output_attn_res_norm.weight * output_attn_res_proj.weight[0]`.
    public let outputAttnResScore: MLXArray
    public let finalNormWeight: MLXArray

    public static let embeddingTensor = "language_model.model.embed_tokens.weight"
    public static let lmHeadTensor = "language_model.lm_head.weight"

    public init(directory: String, config: TinyK3Config) throws {
        let path = URL(fileURLWithPath: directory).appendingPathComponent("model.safetensors").path
        // Bound locally rather than reached through `self`: the nested helper
        // below would otherwise capture a partially initialised `self`.
        let file = try SafetensorsFile(path: path)
        self.file = file
        self.config = config

        let root = "language_model.model"
        func weight(_ name: String) throws -> MLXArray { try file.float32(name) }

        // The vision tower and multimodal projector are present in this
        // checkpoint and are deliberately not loaded. Spec §15.6: an adapter
        // must reject or clearly ignore unsupported modalities rather than
        // produce ambiguous behaviour — this is the "clearly ignore" half, and
        // it is stated here rather than left to the reader of the tensor list.
        var layers: [TinyK3LayerWeights] = []
        for index in 0..<config.numHiddenLayers {
            let prefix = "\(root).layers.\(index)"
            let attention = "\(prefix).self_attn"

            let kda: K3KDA.Weights? = config.isLinearAttention(layer: index)
                ? K3KDA.Weights(
                    qProj: try weight("\(attention).q_proj.weight"),
                    kProj: try weight("\(attention).k_proj.weight"),
                    vProj: try weight("\(attention).v_proj.weight"),
                    // [channels, 1, W] in the file; the singleton group axis is
                    // dropped because the convolution is depthwise.
                    qConv: try weight("\(attention).q_conv1d.weight").reshaped(
                        [-1, config.kdaShortConvKernelSize]),
                    kConv: try weight("\(attention).k_conv1d.weight").reshaped(
                        [-1, config.kdaShortConvKernelSize]),
                    vConv: try weight("\(attention).v_conv1d.weight").reshaped(
                        [-1, config.kdaShortConvKernelSize]),
                    fAProj: try weight("\(attention).f_a_proj.weight"),
                    fBProj: try weight("\(attention).f_b_proj.weight"),
                    bProj: try weight("\(attention).b_proj.weight"),
                    gProj: try weight("\(attention).g_proj.weight"),
                    // Stored per-head, but padded to `head_dim` at flagship
                    // scale; `resolveALog` slices it back and refuses anything
                    // it cannot prove is padding.
                    aLog: try K3KDA.resolveALog(
                        try weight("\(attention).A_log"),
                        heads: config.kdaNumHeads, headDim: config.kdaHeadDim),
                    dtBias: try weight("\(attention).dt_bias"),
                    oNormWeight: try weight("\(attention).o_norm.weight"),
                    oProj: try weight("\(attention).o_proj.weight"))
                : nil

            let mla: K3MLA.Weights? = config.isLinearAttention(layer: index)
                ? nil
                : K3MLA.Weights(
                    qAProj: try weight("\(attention).q_a_proj.weight"),
                    qALayerNorm: try weight("\(attention).q_a_layernorm.weight"),
                    qBProj: try weight("\(attention).q_b_proj.weight"),
                    kvAProjWithMQA: try weight("\(attention).kv_a_proj_with_mqa.weight"),
                    kvALayerNorm: try weight("\(attention).kv_a_layernorm.weight"),
                    kvBProj: try weight("\(attention).kv_b_proj.weight"),
                    gProj: try weight("\(attention).g_proj.weight"),
                    oProj: try weight("\(attention).o_proj.weight"))

            var dense: K3MLP.DenseWeights?
            var moe: K3MoE.Weights?
            if config.isDense(layer: index) {
                dense = K3MLP.DenseWeights(
                    gateProj: try weight("\(prefix).mlp.gate_proj.weight"),
                    upProj: try weight("\(prefix).mlp.up_proj.weight"),
                    downProj: try weight("\(prefix).mlp.down_proj.weight"))
            } else {
                let block = "\(prefix).block_sparse_moe"
                moe = K3MoE.Weights(
                    gateWeight: try weight("\(block).gate.weight"),
                    correctionBias: try weight("\(block).gate.e_score_correction_bias"),
                    routedDownProj: try file.mxfp4("\(block).routed_expert_down_proj"),
                    routedUpProj: try file.mxfp4("\(block).routed_expert_up_proj"),
                    routedNorm: try weight("\(block).routed_expert_norm.weight"),
                    shared: K3MLP.QuantizedWeights(
                        gateProj: try file.mxfp4("\(block).shared_experts.gate_proj"),
                        upProj: try file.mxfp4("\(block).shared_experts.up_proj"),
                        downProj: try file.mxfp4("\(block).shared_experts.down_proj")))
            }

            // `norm.weight * proj.weight[0]` is precomputed: the two vectors
            // are only ever used as this elementwise product, and it is the
            // form `subop-attnres-output-stage` records as `score_weight`.
            let selfNorm = try weight("\(prefix).self_attention_res_norm.weight")
            let selfProj = try weight("\(prefix).self_attention_res_proj.weight").reshaped([-1])
            let mlpNorm = try weight("\(prefix).mlp_res_norm.weight")
            let mlpProj = try weight("\(prefix).mlp_res_proj.weight").reshaped([-1])

            layers.append(
                TinyK3LayerWeights(
                    inputLayerNorm: try weight("\(prefix).input_layernorm.weight"),
                    postAttentionLayerNorm: try weight("\(prefix).post_attention_layernorm.weight"),
                    selfAttentionResScore: selfNorm * selfProj,
                    mlpResScore: mlpNorm * mlpProj,
                    kda: kda, mla: mla, dense: dense, moe: moe))
        }
        self.layers = layers
        let outputNorm = try weight("\(root).output_attn_res_norm.weight")
        let outputProj = try weight("\(root).output_attn_res_proj.weight").reshaped([-1])
        self.outputAttnResScore = outputNorm * outputProj
        self.finalNormWeight = try weight("\(root).norm.weight")

        guard file.contains(Self.embeddingTensor), file.contains(Self.lmHeadTensor) else {
            throw TinyK3Error.missingTensor("\(Self.embeddingTensor) / \(Self.lmHeadTensor)")
        }
    }
}
