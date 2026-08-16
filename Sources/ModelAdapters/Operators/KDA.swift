import Foundation
import MLX

/// Kimi Delta Attention: a gated delta-rule recurrence with a short causal
/// convolution in front.
///
/// ## Provenance of every contested detail
///
/// KDA's numerics are the one part of the M6A reference that is a
/// *transcription* of Triton kernels rather than stock PyTorch or the model's
/// own code. Eight semantics were contested, and all eight were settled on a
/// DGX A100 against fla's shipped `fused_recurrent_kda` by flipping each one and
/// re-running (M6A record, addendum). This implementation follows the
/// transcription as confirmed, and does not re-derive:
///
/// | # | semantic | transcription | flipped |
/// | --- | --- | ---: | ---: |
/// | 1 | l2norm epsilon **inside** the sqrt | 1.558e-07 | 4.541e-05 |
/// | 2 | `dt_bias` reshaped `[HV*K]` → `[HV, K]` | 1.558e-07 | 3.288e-03 |
/// | 3 | gate `-exp(A_log) * softplus(...)` | 1.558e-07 | 7.446e+10 |
/// | 4 | `beta` through sigmoid | 1.558e-07 | 1.993e+00 |
/// | 5 | scale applied to `q` **after** the l2norm | 1.558e-07 | 4.656e+00 |
/// | 6 | state layout V-first `[HV, V, K]` across the cache | 1.914e-07 | 1.049e+00 |
/// | 7 | conv state roll, oldest-first, newest at `W-1` | 9.817e-08 | 1.512e+00 |
/// | 8 | gated norm: normalise, **then** gate | 1.325e-07 | 5.388e-01 |
///
/// Item 1 is the dangerous one: the wrong placement diverges by 4.5e-05, which
/// is *under* the 1e-04 per-layer tolerance, so no committed fixture fails it.
/// `KDAEpsilonPlacementTests` exists for that alone.
///
/// ## Which formulation
///
/// The reference uses the sequential recurrence for prefill and decode alike.
/// fla uses a 64-token chunked reformulation for prefill; the A100 run measured
/// the two at 6.6e-06 of each other at 160 tokens, comfortably inside the 1e-04
/// tolerance, so phase B may use either. This uses the sequential one: prompts
/// here are at most 8 tokens, so the chunked path would never cross a block
/// boundary anyway, and the sequential form is the one the fixtures'
/// per-position state captures can check step by step.
public enum K3KDA {

    /// The rolling state a KDA layer carries across the prefill/decode boundary.
    public struct State {
        /// `[channels, W]` each: the last `W` convolution **inputs**, oldest at
        /// index 0. Entry 0 is dead weight — it falls off on the next shift —
        /// and is kept because that is the shape the kernel hands over.
        public var convQ: MLXArray
        public var convK: MLXArray
        public var convV: MLXArray
        /// `[heads, valueDim, keyDim]` — **V-first**, the layout fla returns and
        /// consumes under `transpose_state_layout=True`, and the layout the
        /// committed `recurrent_state_final_v_first` capture is in.
        public var recurrent: MLXArray

        public init(convQ: MLXArray, convK: MLXArray, convV: MLXArray, recurrent: MLXArray) {
            self.convQ = convQ
            self.convK = convK
            self.convV = convV
            self.recurrent = recurrent
        }
    }

    // MARK: - Short convolution

    /// Depthwise causal 1-D convolution over the last `W` inputs, then SiLU.
    ///
    /// Prefill and decode are the same expression. fla splits them and
    /// dispatches to a `step` kernel when `B * T == N`; with the state defined
    /// as "the last `W` conv inputs", the single-token case is the general case
    /// at `T = 1`, and writing it once removes a place where the two could
    /// drift.
    ///
    /// - Parameters:
    ///   - x: `[tokens, channels]`, the projection output (pre-convolution).
    ///   - weight: `[channels, W]`, from the checkpoint's `[channels, 1, W]`.
    ///   - state: `[channels, W]` or `nil` to start from zeros.
    /// - Returns: the activated output `[tokens, channels]` and the new state.
    public static func shortConvolution(
        _ x: MLXArray, weight: MLXArray, state: MLXArray?
    ) -> (output: MLXArray, state: MLXArray) {
        let tokens = x.shape[0]
        let channels = x.shape[1]
        let width = weight.shape[1]

        // The prefix is the previous inputs, oldest first. `state[..., 1:]`
        // drops the dead entry; transposing puts time on the row axis.
        let prefix: MLXArray
        if let state {
            prefix = state[0..., 1..<width].transposed(1, 0)
        } else {
            prefix = MLXArray.zeros([width - 1, channels], dtype: .float32)
        }
        let padded = concatenated([prefix, x.asType(.float32)], axis: 0)

        // Ascending tap order, matching `tl.static_range(-W + 1, 1)` in fla's
        // kernel. Written as an explicit sum so the accumulation order is the
        // kernel's and not whatever a conv primitive chooses.
        var accumulator = MLXArray.zeros([tokens, channels], dtype: .float32)
        for tap in 0..<width {
            accumulator = accumulator + padded[tap..<(tap + tokens), 0...] * weight[0..., tap]
        }
        let newState = padded[(padded.shape[0] - width)..., 0...].transposed(1, 0)
        return (K3Activation.silu(accumulator), newState)
    }

    // MARK: - Gate

    /// The per-dimension log-decay gate, in whichever of its two forms the
    /// configuration selects.
    ///
    /// ```text
    /// lowerBound == nil:  g = -exp(A_log)[:, None] * softplus(gRaw + dtBias.view(heads, dim))
    /// lowerBound == L:    g =  L * sigmoid(exp(A_log)[:, None] * (gRaw + dtBias.view(heads, dim)))
    /// ```
    ///
    /// `dt_bias` is a flat `[heads * dim]` parameter viewed as `[heads, dim]`.
    /// fla's docstring calls it `[heads]`; the kernel indexes it as
    /// `dt_bias + i_hv * K + o_k`, and the kernel is authoritative — confirmed
    /// on the A100 at 3.288e-03 for the alternative.
    ///
    /// ## The two forms (M9B, 2026-08-10)
    ///
    /// These are different functions, not a clamp applied to one function. The
    /// bounded form has no `softplus` and no negated `exp`; it is a scaled
    /// sigmoid, so its range is exactly `(L, 0)` and `A_log` enters as a
    /// *slope* rather than as a magnitude. Selecting the wrong one does not
    /// degrade gracefully.
    ///
    /// Both are transcribed from one place — fla `ops/kda/gate.py`, the
    /// `USE_LOWER_BOUND` branch of `kda_gate_fwd_kernel`, which agrees
    /// expression-for-expression with `naive_kda_gate` /
    /// `naive_kda_lowerbound_gate` beside it and with the identical branch in
    /// `ops/kda/fused_recurrent.py`. `modeling_kimi_linear.py` passes
    /// `lower_bound=self.gate_lower_bound` on both the chunk and the recurrent
    /// path, so the selection is `linear_attn_config.gate_lower_bound` being
    /// present — `-5.0` at flagship, absent at tiny.
    ///
    /// - Parameters:
    ///   - gRaw: `[tokens, heads, dim]`.
    ///   - aLog: `[heads]` — the *resolved* vector. See ``resolveALog(_:heads:headDim:)``,
    ///     which is what turns a checkpoint's stored `A_log` into this.
    ///   - dtBias: `[heads * dim]`.
    ///   - lowerBound: `linear_attn_config.gate_lower_bound`, or `nil` for the
    ///     softplus form.
    public static func gate(
        _ gRaw: MLXArray, aLog: MLXArray, dtBias: MLXArray, lowerBound: Float?
    ) -> MLXArray {
        let heads = gRaw.shape[gRaw.ndim - 2]
        let dim = gRaw.shape[gRaw.ndim - 1]
        let biased = gRaw.asType(.float32) + dtBias.asType(.float32).reshaped([heads, dim])
        let decay = exp(aLog.asType(.float32).reshaped([heads, 1]))
        guard let lowerBound else {
            return -decay * K3Activation.softplus(biased)
        }
        return MLXArray(lowerBound) * sigmoid(decay * biased)
    }

    /// The `[heads]` decay vector the gate indexes, read out of whatever the
    /// checkpoint stored.
    ///
    /// ## Why this is not just the tensor (M9B, 2026-08-10)
    ///
    /// The flagship checkpoint stores `A_log` with **`head_dim` = 128 entries
    /// while the layer has 96 heads**, which is what M9A found and refused to
    /// bind. Three independent sources say the parameter is nevertheless
    /// per-head, and the storage is padded:
    ///
    /// 1. `modeling_kimi_linear.py:520` constructs it as
    ///    `Parameter(log(empty(self.num_heads)))` — `[96]`.
    /// 2. fla's kernels index it by the head id alone — `tl.load(A_log + i_h)`
    ///    in `kda_gate_fwd_kernel`, `tl.load(A_log + i_hv)` in
    ///    `fused_recurrent_kda_fwd_kernel` — and the torch reference beside them
    ///    writes `A_log.view(H, 1)`. Nothing reads past index `H - 1`, which is
    ///    why the padded tensor works unmodified in PyTorch and why the padding
    ///    was never noticed upstream.
    /// 3. The bytes: entries `96..<128` are **exactly zero** in every flagship
    ///    KDA layer sampled (1, 2, 50), while `0..<96` carry trained values
    ///    spanning `[-0.43, 1.23]`. A trained per-channel parameter does not
    ///    have a quarter of its channels pinned at exactly 0.0.
    ///
    /// So the correct reading is per-head, and the correct binding is a slice.
    /// The zero tail is *checked* rather than assumed: if a future checkpoint
    /// ever stores something meaningful past `heads`, the padding
    /// interpretation is wrong and this throws instead of silently discarding
    /// it. That check is the whole reason this is a function and not a
    /// subscript at the call site.
    ///
    /// - Returns: `aLog` when it is already `[heads]`; its first `heads`
    ///   entries when it is longer and the remainder is identically zero.
    public static func resolveALog(_ aLog: MLXArray, heads: Int, headDim: Int) throws -> MLXArray {
        let count = aLog.size
        if count == heads { return aLog }
        guard count > heads else {
            throw TinyK3Error.unsupportedArchitecture(
                "A_log has \(count) entries, fewer than the \(heads) heads the gate indexes; "
                    + "the gate cannot be bound.")
        }
        guard count == headDim else {
            throw TinyK3Error.unsupportedArchitecture(
                "A_log has \(count) entries, which is neither the head count (\(heads)) nor the "
                    + "head dimension (\(headDim)); the gate cannot be bound.")
        }
        let tail = aLog[heads..<count].asType(.float32)
        let residue = sum(abs(tail)).item(Float.self)
        guard residue == 0 else {
            throw TinyK3Error.unsupportedArchitecture(
                "A_log has \(count) entries against \(heads) heads, and entries \(heads)..<\(count) "
                    + "are not zero (sum |tail| = \(residue)). The per-head reading assumes those "
                    + "are storage padding; they are not, so the decay's semantics are unresolved "
                    + "and binding is refused (K3KDA.resolveALog).")
        }
        return aLog[0..<heads]
    }

    // MARK: - Recurrence

    /// One pass of the delta-rule recurrence, one token at a time.
    ///
    /// ```text
    /// S = S * exp(g_t)
    /// S = S + (beta_t * k_t) ⊗ (v_t - k_t · S)
    /// o_t = q_t · S
    /// ```
    ///
    /// The output uses the state **after** the update, not before.
    ///
    /// Inputs are post-transform: `q`/`k` already L2-normalised, `q` already
    /// scaled, `beta` already through sigmoid, `g` already the log-decay.
    ///
    /// - Parameters:
    ///   - q, k: `[tokens, heads, keyDim]`, scaled and normalised.
    ///   - v: `[tokens, heads, valueDim]`.
    ///   - g: `[tokens, heads, keyDim]` log-decay.
    ///   - beta: `[tokens, heads]`.
    ///   - initialState: `[heads, keyDim, valueDim]`, K-first, or `nil`.
    /// - Returns: the output `[tokens, heads, valueDim]`, the final K-first
    ///   state, and the per-position states when `recordStates` is set.
    public static func recurrence(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray,
        initialState: MLXArray?, recordStates: Bool = false
    ) -> (output: MLXArray, finalState: MLXArray, trace: [MLXArray]) {
        let tokens = q.shape[0]
        let heads = v.shape[1]
        let keyDim = q.shape[2]
        let valueDim = v.shape[2]

        var state = initialState ?? MLXArray.zeros([heads, keyDim, valueDim], dtype: .float32)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(tokens)
        var trace: [MLXArray] = []

        for position in 0..<tokens {
            let qStep = q[position]  // [heads, keyDim]
            let kStep = k[position]
            let vStep = v[position]  // [heads, valueDim]
            let gStep = g[position]
            let betaStep = beta[position]  // [heads]

            state = state * expandedDimensions(exp(gStep), axis: -1)
            let predicted = sum(expandedDimensions(kStep, axis: -1) * state, axis: -2)
            let delta = vStep - predicted
            let scaledKey = expandedDimensions(betaStep, axis: -1) * kStep
            state =
                state
                + expandedDimensions(scaledKey, axis: -1) * expandedDimensions(delta, axis: -2)
            outputs.append(sum(expandedDimensions(qStep, axis: -1) * state, axis: -2))
            if recordStates { trace.append(state) }
        }
        return (stacked(outputs, axis: 0), state, trace)
    }

    // MARK: - Layer

    /// Every weight one KDA layer needs, in checkpoint orientation.
    public struct Weights {
        public var qProj: MLXArray
        public var kProj: MLXArray
        public var vProj: MLXArray
        /// `[channels, W]`, flattened from the checkpoint's `[channels, 1, W]`.
        public var qConv: MLXArray
        public var kConv: MLXArray
        public var vConv: MLXArray
        public var fAProj: MLXArray
        public var fBProj: MLXArray
        public var bProj: MLXArray
        public var gProj: MLXArray
        public var aLog: MLXArray
        public var dtBias: MLXArray
        public var oNormWeight: MLXArray
        public var oProj: MLXArray
    }

    /// Every intermediate the `subop-kda-layer02` bundle records, so a
    /// mismatch lands on a named stage rather than on "layer 2 is wrong".
    public struct Trace {
        public var qProj: MLXArray
        public var kProj: MLXArray
        public var vProj: MLXArray
        public var qConv: MLXArray
        public var kConv: MLXArray
        public var vConv: MLXArray
        public var convStateQ: MLXArray
        public var convStateK: MLXArray
        public var convStateV: MLXArray
        public var gateRaw: MLXArray
        public var betaRaw: MLXArray
        public var qL2Norm: MLXArray
        public var kL2Norm: MLXArray
        public var betaSigmoid: MLXArray
        public var gateLogDecay: MLXArray
        public var recurrentStatePerPosition: [MLXArray]
        public var recurrentStateFinalVFirst: MLXArray
        public var recurrenceOutput: MLXArray
        public var outputGateRaw: MLXArray
        public var oNorm: MLXArray
        public var output: MLXArray
    }

    /// `KimiDeltaAttention.forward`, `modeling_kimi_k3_linear.py:609-735`.
    ///
    /// - Parameters:
    ///   - x: `[tokens, hidden]`, already through `input_layernorm`.
    ///   - state: the rolling state, or `nil` for a fresh sequence.
    /// - Returns: the layer output `[tokens, hidden]`, the new state, and the
    ///   full stage-by-stage trace.
    public static func forward(
        _ x: MLXArray, weights: Weights, heads: Int, headDim: Int, eps: Float,
        gateLowerBound: Float?, state: State?, recordStates: Bool = true
    ) -> (output: MLXArray, state: State, trace: Trace) {
        let tokens = x.shape[0]

        let qProj = k3Linear(x, weights.qProj)
        let kProj = k3Linear(x, weights.kProj)
        let vProj = k3Linear(x, weights.vProj)

        let (qConv, convStateQ) = shortConvolution(qProj, weight: weights.qConv, state: state?.convQ)
        let (kConv, convStateK) = shortConvolution(kProj, weight: weights.kConv, state: state?.convK)
        let (vConv, convStateV) = shortConvolution(vProj, weight: weights.vConv, state: state?.convV)

        let gateRaw = k3Linear(k3Linear(x, weights.fAProj), weights.fBProj)
            .reshaped([tokens, heads, headDim])
        // `b_proj(x).float()` is pinned to float32 inline in the modeling code;
        // this is that pin. It is also the one operation the reference's
        // float64 grounding could not promote, which is part of why the
        // per-layer tolerance carries 60x headroom.
        let betaRaw = k3Linear(x, weights.bProj).asType(.float32)

        let q = qConv.reshaped([tokens, heads, headDim])
        let k = kConv.reshaped([tokens, heads, headDim])
        let v = vConv.reshaped([tokens, heads, headDim])

        // Transform order, from fla/ops/kda/fused_recurrent.py:154-191:
        // l2norm -> sigmoid(beta) -> gate -> q *= scale.
        let qNorm = K3Norm.l2(q)
        let kNorm = K3Norm.l2(k)
        let beta = sigmoid(betaRaw)
        let logDecay = gate(
            gateRaw, aLog: weights.aLog, dtBias: weights.dtBias, lowerBound: gateLowerBound)

        let scale = MLXArray(Float(pow(Double(headDim), -0.5)))
        let (recurrenceOutput, finalState, trace) = recurrence(
            q: qNorm * scale, k: kNorm, v: v, g: logDecay, beta: beta,
            // The state crosses the boundary V-first and is used K-first.
            initialState: state.map { $0.recurrent.transposed(0, 2, 1) },
            recordStates: recordStates)

        let outputGateRaw = k3Linear(x, weights.gProj).reshaped([tokens, heads, headDim])
        let normed = K3Norm.gatedRMS(
            recurrenceOutput, gate: outputGateRaw, weight: weights.oNormWeight, eps: eps)
        let output = k3Linear(normed.reshaped([tokens, heads * headDim]), weights.oProj)

        let finalVFirst = finalState.transposed(0, 2, 1)
        return (
            output,
            State(convQ: convStateQ, convK: convStateK, convV: convStateV, recurrent: finalVFirst),
            Trace(
                qProj: qProj, kProj: kProj, vProj: vProj,
                qConv: qConv, kConv: kConv, vConv: vConv,
                convStateQ: convStateQ, convStateK: convStateK, convStateV: convStateV,
                gateRaw: gateRaw, betaRaw: betaRaw,
                qL2Norm: qNorm, kL2Norm: kNorm, betaSigmoid: beta, gateLogDecay: logDecay,
                recurrentStatePerPosition: trace, recurrentStateFinalVFirst: finalVFirst,
                recurrenceOutput: recurrenceOutput, outputGateRaw: outputGateRaw,
                oNorm: normed, output: output)
        )
    }
}
