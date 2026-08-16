import Foundation

/// The tiny Kimi K3 text configuration, typed.
///
/// Read from the checkpoint's own `config.json` (nested under `text_config`,
/// with the KDA parameters nested again under `linear_attn_config`) rather than
/// hard-coded, because the numbers here are the ones every shape in the forward
/// pass is derived from and a hard-coded copy would silently disagree with a
/// different revision.
///
/// ## Two fields that mislead if taken at face value
///
/// - `config.dtype` says `bfloat16`. Every non-MoE tensor in this checkpoint is
///   **F32**. The dtype is therefore not read from here at all; the safetensors
///   header is authoritative.
/// - `text_config.head_dim` is 74 and is used by nothing. MLA's real dimensions
///   are `qk_nope_head_dim 64`, `qk_rope_head_dim 32`, `v_head_dim 64`, and this
///   type does not expose `head_dim` so it cannot be reached by accident.
public struct TinyK3Config: Sendable, Equatable {

    // Model shape.
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let intermediateSize: Int
    public let firstKDenseReplace: Int

    // situ.
    public let situBeta: Float
    public let situLinearBeta: Float?

    // AttnRes.
    public let attnResBlockSize: Int

    // KDA (`linear_attn_config`). Layer lists are 1-indexed in the file.
    public let kdaLayers: Set<Int>
    public let fullAttentionLayers: Set<Int>
    public let kdaNumHeads: Int
    public let kdaHeadDim: Int
    public let kdaShortConvKernelSize: Int
    public let kdaUseFullRankGate: Bool
    /// `linear_attn_config.gate_lower_bound`, which *selects the gate function*
    /// rather than clamping one: present (flagship, `-5.0`) means
    /// `L * sigmoid(exp(A_log) * g)`, absent (tiny) means
    /// `-exp(A_log) * softplus(g)`. See ``K3KDA/gate(_:aLog:dtBias:lowerBound:)``.
    public let kdaGateLowerBound: Float?

    // MLA.
    public let qLoraRank: Int
    public let kvLoraRank: Int
    public let numAttentionHeads: Int
    public let qkNopeHeadDim: Int
    public let qkRopeHeadDim: Int
    public let vHeadDim: Int
    public let mlaUseNope: Bool
    public let mlaUseOutputGate: Bool

    // MoE.
    public let numExperts: Int
    public let numExpertsPerToken: Int
    public let numSharedExperts: Int
    public let routedExpertHiddenSize: Int
    public let moeIntermediateSize: Int
    public let moeRenormalize: Bool
    public let routedScalingFactor: Float
    public let latentMoEUseNorm: Bool

    /// `qk_nope_head_dim + qk_rope_head_dim`, the full query width. The
    /// attention scale is `1/sqrt` of *this*, not of the nope part.
    public var mlaQueryHeadDim: Int { qkNopeHeadDim + qkRopeHeadDim }

    /// A layer is a KDA layer when its 1-indexed position is in `kda_layers`.
    public func isLinearAttention(layer index: Int) -> Bool { kdaLayers.contains(index + 1) }

    /// Layers below `first_k_dense_replace` carry a dense MLP; the rest carry a
    /// sparse MoE block. For this model that is layer 0 alone.
    public func isDense(layer index: Int) -> Bool { index < firstKDenseReplace }

    /// 0-indexed layers at which the current prefix sum is appended to the
    /// AttnRes checkpoint list.
    public var attnResCheckpointLayers: [Int] {
        (0..<numHiddenLayers).filter { $0 % attnResBlockSize == 0 }
    }

    public init(contentsOfDirectory directory: String) throws {
        let url = URL(fileURLWithPath: directory).appendingPathComponent("config.json")
        try self.init(json: try Data(contentsOf: url))
    }

    public init(json data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw TinyK3Error.configuration("top level is not a JSON object")
        }
        guard let text = root["text_config"] as? [String: Any] else {
            throw TinyK3Error.configuration("no 'text_config' block")
        }
        guard let linear = text["linear_attn_config"] as? [String: Any] else {
            throw TinyK3Error.configuration("no 'text_config.linear_attn_config' block")
        }

        func int(_ key: String, _ block: [String: Any] = text) throws -> Int {
            guard let value = block[key] as? Int else {
                throw TinyK3Error.configuration("'\(key)' is missing or not an integer")
            }
            return value
        }
        func float(_ key: String, _ block: [String: Any] = text) throws -> Float {
            guard let value = block[key] as? Double else {
                throw TinyK3Error.configuration("'\(key)' is missing or not a number")
            }
            return Float(value)
        }
        func bool(_ key: String, _ block: [String: Any] = text) throws -> Bool {
            guard let value = block[key] as? Bool else {
                throw TinyK3Error.configuration("'\(key)' is missing or not a boolean")
            }
            return value
        }

        hiddenSize = try int("hidden_size")
        numHiddenLayers = try int("num_hidden_layers")
        vocabSize = try int("vocab_size")
        rmsNormEps = try float("rms_norm_eps")
        intermediateSize = try int("intermediate_size")
        firstKDenseReplace = try int("first_k_dense_replace")

        guard let activation = text["hidden_act"] as? String else {
            throw TinyK3Error.configuration("'hidden_act' is missing")
        }
        guard activation == "situ" else {
            throw TinyK3Error.unsupportedArchitecture(
                "hidden_act '\(activation)'; only 'situ' is implemented")
        }
        situBeta = try float("activation_situ_beta")
        situLinearBeta = (text["activation_situ_linear_beta"] as? Double).map(Float.init)

        attnResBlockSize = try int("attn_res_block_size")

        guard let kda = linear["kda_layers"] as? [Int],
            let full = linear["full_attn_layers"] as? [Int]
        else {
            throw TinyK3Error.configuration("linear_attn_config layer lists are missing")
        }
        kdaLayers = Set(kda)
        fullAttentionLayers = Set(full)
        kdaNumHeads = try int("num_heads", linear)
        kdaHeadDim = try int("head_dim", linear)
        kdaShortConvKernelSize = try int("short_conv_kernel_size", linear)
        kdaUseFullRankGate = (linear["use_full_rank_gate"] as? Bool) ?? false

        // `gate_lower_bound` selects a *different* gate function
        // (`lower_bound * sigmoid(exp(A_log) * g)`). Both forms are now
        // transcribed from fla's `kda_gate_fwd_kernel` (M9B); this reads which
        // one the checkpoint asks for. A non-numeric value is still refused
        // rather than coerced, because the two forms do not degrade into each
        // other and a misparse would pick the wrong function silently.
        switch linear["gate_lower_bound"] {
        case nil, is NSNull:
            kdaGateLowerBound = nil
        case let value as Double:
            kdaGateLowerBound = Float(value)
        case let value as Int:
            kdaGateLowerBound = Float(value)
        case let other?:
            throw TinyK3Error.configuration(
                "linear_attn_config.gate_lower_bound is present but not a number (\(other))")
        }
        guard kdaUseFullRankGate else {
            throw TinyK3Error.unsupportedArchitecture(
                "use_full_rank_gate is false; the low-rank g_a_proj/g_b_proj output gate is not "
                    + "exercised by any fixture and is not implemented")
        }

        qLoraRank = try int("q_lora_rank")
        kvLoraRank = try int("kv_lora_rank")
        numAttentionHeads = try int("num_attention_heads")
        qkNopeHeadDim = try int("qk_nope_head_dim")
        qkRopeHeadDim = try int("qk_rope_head_dim")
        vHeadDim = try int("v_head_dim")
        mlaUseNope = try bool("mla_use_nope")
        mlaUseOutputGate = try bool("mla_use_output_gate")
        guard mlaUseNope else {
            throw TinyK3Error.unsupportedArchitecture(
                "mla_use_nope is false; this model has no rotary embedding and none is implemented")
        }

        numExperts = try int("num_experts")
        numExpertsPerToken = try int("num_experts_per_token")
        numSharedExperts = try int("num_shared_experts")
        routedExpertHiddenSize = try int("routed_expert_hidden_size")
        moeIntermediateSize = try int("moe_intermediate_size")
        moeRenormalize = try bool("moe_renormalize")
        routedScalingFactor = try float("routed_scaling_factor")
        latentMoEUseNorm = try bool("latent_moe_use_norm")

        guard let router = text["moe_router_activation_func"] as? String, router == "sigmoid" else {
            throw TinyK3Error.unsupportedArchitecture(
                "moe_router_activation_func must be 'sigmoid'; softmax routing is not implemented")
        }
        let groups = (text["num_expert_group"] as? Int) ?? 1
        let topkGroup = (text["topk_group"] as? Int) ?? 1
        guard groups <= topkGroup else {
            throw TinyK3Error.unsupportedArchitecture(
                "grouped top-k routing (num_expert_group \(groups) > topk_group \(topkGroup)) is "
                    + "unreachable for this config and is not transcribed")
        }
    }
}
