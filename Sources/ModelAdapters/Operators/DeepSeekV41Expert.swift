import Foundation
import MLX
import MLXBridge

/// Whether the activation is rounded the way the reference's `linear()` rounds
/// it before a quantized GEMM.
///
/// This is not a tuning knob; it is a statement about which of two arithmetics
/// is being computed, and the two differ by percent, not by ulps.
///
/// `inference/model.py`'s `linear()` quantizes the *left* operand of every FP4
/// and FP8 GEMM to E4M3 in blocks of 32 with a power-of-two scale before
/// multiplying — `act_quant(x, fp8_block_size=32, scale_fmt="ue8m0")`. E4M3
/// carries three mantissa bits, so each element is rounded to about 6%, and a
/// dot product of random-sign terms keeps that relative error rather than
/// averaging it away: the fixture measures the whole MoE output moving by 5% to
/// 10% between the two settings. So a port that skips the rounding is not
/// "slightly more accurate", it is computing a different function, and any
/// digest gate has to say which one it took.
public enum DeepSeekV41ExpertActivation: Sendable, Equatable {
    /// What the reference does: E4M3 round trip in blocks of 32 on every GEMM's
    /// left operand.
    case referenceFP8
    /// No activation rounding. The dequantized weights, multiplied in float32.
    /// The arithmetic V4's expert path computes, kept so Phase 3 can hold the
    /// two against one reference digest and say which the checkpoint wants.
    case exact

    /// `act_quant`'s block size, and also the weights' scale-block width.
    public static let blockSize = 32

    func applied(
        _ input: MLXArray, diagnostics: DeepSeekV4Diagnostics,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil
    ) throws -> MLXArray {
        switch self {
        case .exact:
            return input
        case .referenceFP8:
            // The accounting is *not* handed down to the reference, and that is
            // the point: `quantizeDequantize` would charge its host round trip
            // to `activationScaleSyncSeconds`, a pass-level term, while this
            // call is inside `expertPhaseSeconds`. The same seconds under two
            // names is what `isAccountingBalanced` refuses. So the round trip —
            // including the sync it blocks on — is bracketed here instead.
            return try measuringPhase(
                phaseAccounting,
                excludingGPUBoundaryFrom: phaseAccounting?
                    .recordExpertActivationRound(nanoseconds:)
            ) {
                try DeepSeekV4FP8ActivationReference.quantizeDequantize(
                    input, blockSize: Self.blockSize, diagnostics: diagnostics)
            }
        }
    }
}

/// DeepSeek V4.1's routed-expert arithmetic, from `Expert.forward`.
///
/// ```python
/// gate = self.w1(x).float()
/// up   = self.w3(x).float()
/// if self.swiglu_limit > 0:
///     up   = torch.clamp(up, min=-limit, max=limit)   # both sides
///     gate = torch.clamp(gate, max=limit)             # from above only
/// x = F.silu(gate) * up
/// if weights is not None: x = weights * x             # routing weight, before w2
/// return self.w2(x.to(dtype))                         # input dtype, then w2
/// ```
///
/// Three things it is easy to get wrong, all of which produce finite numbers of
/// the right shape:
///
/// - the clamps are **not symmetric**. The gate branch is clamped from above
///   only, because `silu` is bounded below anyway and training only ever needed
///   the top. A port that clamped both would differ exactly where the fixture
///   says the clamp bites, which is 15% of gate elements and 31% of up ones.
/// - the routing weight is applied **between the SwiGLU and `w2`**, not after
///   `w2`. The two agree because `w2` is linear — until the operand of `w2` is
///   rounded, at which point the weight decides which values survive E4M3.
/// - `x.to(dtype)` casts back to the *input's* dtype before `w2`. On a bf16
///   decode that is a real rounding; this port carries it explicitly rather
///   than staying in float32 by accident.
///
/// The whole is the same shape as ``DeepSeekV4RoutedExperts/forward(_:layer:expertIDs:routingWeights:swiGLULimit:backend:)``
/// with the expert stack now coming from half-tiles.
public enum DeepSeekV41RoutedExperts {

    /// `out[t, :] = sum_j weights[t, j] * expert(ids[t][j])(x[t, :])`.
    ///
    /// One token at a time, because a V4.1 token's expert set is its own and
    /// the stack read for it is its own: two tokens share a tile only by
    /// coincidence, and pretending otherwise would hide the read the second
    /// token actually costs. Decode is one token; prefill is a loop over this,
    /// which is also how `MoE.forward` is written.
    public static func forward(
        _ input: MLXArray,
        expertIDs: [[Int]],
        routingWeights: MLXArray,
        swiGLULimit: Float,
        source: DeepSeekV41RoutedExpertSource,
        activation: DeepSeekV41ExpertActivation = .referenceFP8,
        boundsLiveOperands: Bool = false,
        diagnostics: DeepSeekV4Diagnostics = .validating,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        stream: StreamOrDevice = .default
    ) throws -> MLXArray {
        let perExpert = try perExpertOutputs(
            input, expertIDs: expertIDs, routingWeights: routingWeights,
            swiGLULimit: swiGLULimit, source: source, activation: activation,
            boundsLiveOperands: boundsLiveOperands,
            diagnostics: diagnostics, phaseAccounting: phaseAccounting, stream: stream)
        // `MoE.forward` accumulates into a float32 zero tensor; summing the
        // slots is the same arithmetic with the same dtype and one fewer
        // temporary.
        return sum(perExpert, axis: 1, stream: stream)
    }

    /// Every routed expert's weighted contribution, `[tokens, slots, dim]`.
    ///
    /// Exposed because a slot's own output is what a disagreement is localised
    /// to, and because the fixture states them one by one.
    public static func perExpertOutputs(
        _ input: MLXArray,
        expertIDs: [[Int]],
        routingWeights: MLXArray,
        swiGLULimit: Float,
        source: DeepSeekV41RoutedExpertSource,
        activation: DeepSeekV41ExpertActivation = .referenceFP8,
        /// Evaluate each token's contribution before building the next one's
        /// operands.
        ///
        /// Off, this loop's operands are all alive at once. `stack` materializes
        /// three arrays a token — `expertsPerToken` half-tiles each — and the
        /// lazy graph that consumes them is not evaluated until the *next*
        /// block's router calls `MLX.eval`, so an N-token prefill holds
        /// `3 x N x expertsPerToken` half-tiles of one block simultaneously. At
        /// the published geometry that is 112,803,840 B a token: 1.24 GB over
        /// the eleven-token arm phase 3 measured its floor on, and 57.8 GB at
        /// the 512-token product prompt limit the same floor claims to cover.
        ///
        /// On, the peak is one token's three operands whatever the prompt
        /// length. It costs one GPU synchronisation a token and changes no
        /// arithmetic: `eval` forces the computation the concatenation would
        /// force anyway, in the same order.
        boundsLiveOperands: Bool = false,
        diagnostics: DeepSeekV4Diagnostics = .validating,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        stream: StreamOrDevice = .default
    ) throws -> MLXArray {
        guard input.ndim == 2, input.shape[0] > 0, input.shape[1] > 0 else {
            throw DeepSeekV41Error.experts(
                "routed experts: input must be a non-empty [tokens, dim] matrix, got "
                    + "\(input.shape)")
        }
        let tokens = input.shape[0]
        guard expertIDs.count == tokens, let slots = expertIDs.first?.count, slots > 0,
            expertIDs.allSatisfy({ $0.count == slots })
        else {
            throw DeepSeekV41Error.experts(
                "routed experts: expert ids must be a rectangular row per input token")
        }
        guard routingWeights.shape == [tokens, slots] else {
            throw DeepSeekV41Error.experts(
                "routed experts: routing weights must be [\(tokens), \(slots)], got "
                    + "\(routingWeights.shape)")
        }
        guard swiGLULimit.isFinite, swiGLULimit >= 0 else {
            throw DeepSeekV41Error.experts(
                "routed experts: SwiGLU limit must be finite and nonnegative")
        }

        let dtype = input.dtype
        let weights = routingWeights.asType(.float32)
        var perToken = [MLXArray]()
        perToken.reserveCapacity(tokens)

        for token in 0..<tokens {
            try measuringExpertExpression(phaseAccounting) {
            let ids = expertIDs[token]
            // Every byte the three projections need is known here, before the
            // first of them is asked for: `w1`, `w3` and `w2` gather the same
            // experts, so a read-ahead has its whole list at this point.
            try source.prefetch(ids)

            let row = input[token..<(token + 1), 0...]
            let rounded = try activation.applied(
                row, diagnostics: diagnostics, phaseAccounting: phaseAccounting)
            let slotIndices = MLXArray(
                (0..<slots).map { Int32($0) }, [1, slots])

            let gateWeights = try source.stack(ids, projection: .gate)
            let upWeights = try source.stack(ids, projection: .up)
            var gate = mxfp4GatherMM(
                rounded, gateWeights, rhsIndices: slotIndices, stream: stream)
                .asType(.float32)
            var up = mxfp4GatherMM(
                rounded, upWeights, rhsIndices: slotIndices, stream: stream)
                .asType(.float32)
            let intermediate = gateWeights.outFeatures
            guard gate.shape == [1, slots, intermediate], up.shape == gate.shape else {
                throw DeepSeekV41Error.experts(
                    "routed experts: w1/w3 gather must be [1, \(slots), \(intermediate)], "
                        + "got \(gate.shape) and \(up.shape)")
            }
            if swiGLULimit > 0 {
                let upper = MLXArray(swiGLULimit)
                gate = minimum(gate, upper, stream: stream)
                up = minimum(maximum(up, MLXArray(-swiGLULimit), stream: stream), upper,
                             stream: stream)
            }
            let hidden = K3Activation.silu(gate) * up
                * expandedDimensions(
                    weights[token..<(token + 1), 0...], axis: -1)

            // `x.to(dtype)` before `w2`, then the same activation rounding the
            // reference applies to every quantized GEMM's left operand.
            let downInput = try activation.applied(
                hidden.asType(dtype).reshaped([slots, intermediate]),
                diagnostics: diagnostics, phaseAccounting: phaseAccounting)
            let downWeights = try source.stack(ids, projection: .down)
            let down = mxfp4GatherMM(
                downInput, downWeights,
                rhsIndices: MLXArray((0..<slots).map { Int32($0) }, [slots, 1]),
                stream: stream)
            guard down.shape == [slots, 1, input.shape[1]] else {
                throw DeepSeekV41Error.experts(
                    "routed experts: w2 gather must be [\(slots), 1, \(input.shape[1])], "
                        + "got \(down.shape)")
            }
            perToken.append(
                down.reshaped([1, slots, input.shape[1]]).asType(.float32))
            if boundsLiveOperands, let last = perToken.last { MLX.eval(last) }
            }
        }
        return concatenated(perToken, axis: 0, stream: stream)
    }
}
