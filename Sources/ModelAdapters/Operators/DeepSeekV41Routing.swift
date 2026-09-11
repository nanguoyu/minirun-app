import Foundation
import MLX

/// DeepSeek V4.1's `sqrtsoftplus` / `noaux_tc` router, from `Gate.forward`.
///
/// Same three traps as V4, plus three of its own.
///
/// Shared with V4:
///
/// - the router sees the **full hidden state**, before any latent projection;
/// - the correction bias steers **selection only** and never enters the routing
///   weights, which are gathered from the *unbiased* scores;
/// - `torch.topk` promises nothing about order, so the ids are a set.
///
/// New in V4.1:
///
/// - `gate_temp` and `norm_topk_prob` are absent from the published
///   `inference/config.json`, so they are `ModelArgs`' defaults, 1.0 and true.
///   A port that guessed differently would be wrong in a way no configuration
///   file mentions. ``DeepSeekV41Config`` decodes `normalizedTopK` from the
///   Transformers document; the temperature has no field at all and is 1.
/// - the normalisation denominator carries `+ 1e-20`. It is *not* `norm_eps`
///   (which happens to be 1e-20 as well) and it is not V4's normalisation,
///   which has no epsilon. The comment in the reference says so explicitly:
///   "not norm_eps, matches training".
/// - `bias_vl` exists on every published block, because `vision_n_layers` is
///   32 and so `vision_enabled` is true. `Gate.forward` substitutes it per
///   token wherever `image_mask` is set. The text path passes no mask, so this
///   router never reads it — deliberately, and stated here rather than left to
///   be inferred from an absence. ``DeepSeekV41BlockArtifact`` still publishes
///   it, so a later image path has it without a reload.
///
/// ## What is reproducible, and to what
///
/// `sqrt(softplus(x))` is **not** bit-reproducible across math libraries.
/// numpy's `log1p(exp(x))` and torch's fused softplus disagree by one float32
/// step on a few percent of values, and `sqrt` absorbs about half of those; a
/// third implementation is entitled to its own ulp. Neither is the
/// normalisation denominator: summing six float32 values has an order, and
/// torch's CPU reduction is not numpy's and is not MLX's — measured at two ulps
/// on the six-way sum.
///
/// So this router is written so that the parts which *can* be exact are:
/// selection, the gather, the division, the epsilon and the scale all happen
/// on the host in plain IEEE float32 over a handful of values per token. The
/// only inexact steps are the two named above, and
/// ``Selection/boundaryGapULPs`` reports whether a one-ulp score wobble could
/// have changed the selection — which, on the fixture, it never can.
///
/// ## Ties
///
/// `torch.topk` has no tie rule. On CPU, scores `[5, 3, 3, 3, 1, 3, 4, 3]` at
/// k=5 select indices `[0, 6, 2, 3, 7]`, skipping the equal values at 1 and 5.
/// So when the k-th boundary is an exact tie the reference's *choice of expert*
/// is a property of its partial sort and not of the values, and no port can
/// reproduce it from the numbers. This router breaks ties by the lower expert
/// id, exactly as ``DeepSeekV4Router`` does, and reports the tie in
/// ``Selection/boundaryTies`` so a caller can say so rather than discover it.
/// The routing *weights* are unaffected: tied experts have equal scores.
public enum DeepSeekV41Router {

    /// `F.softplus`'s threshold: above `beta * x`, torch returns the identity.
    /// Below it, `log1p(exp(x))` — which overflows to infinity by x = 89, so
    /// the branch is load-bearing and not an optimisation.
    public static let softplusThreshold: Float = 20

    /// `Gate.forward`'s literal, and not `norm_eps`.
    public static let normalizationEpsilon: Float = 1e-20

    /// No published field sets it; `ModelArgs.gate_temp` defaults to 1.
    public static let defaultGateTemperature: Float = 1

    /// Not `Sendable`: it carries `MLXArray`s, as ``DeepSeekV4Router/Selection``
    /// does, and an MLX array is not safe to hand across a boundary.
    public struct Selection {
        /// `[tokens][expertsPerToken]`, in descending ranked-score order with
        /// ties broken by the lower expert id.
        public let ids: [[Int]]
        /// `[tokens, expertsPerToken]`, aligned with ``ids``: the unbiased
        /// scores, normalised if the block normalises, times `route_scale`.
        public let weights: MLXArray
        /// The same values on the host, in the order they were computed.
        public let weightValues: [[Float]]
        /// The gathered unbiased scores, before normalisation and scaling.
        public let gathered: [[Float]]
        /// The normalisation denominator, before the epsilon. One per token,
        /// and the only inexact step downstream of the score.
        public let denominators: [Float]
        /// `sqrt(softplus(logits))` over every expert.
        public let scores: MLXArray
        /// `scores + bias` — what chose the experts, and never a weight.
        public let ranked: MLXArray
        /// `(kth selected − first rejected) / max(|kth|, |first rejected|)`
        /// per token, over the *ranked* scores.
        public let relativeMargins: [Double]
        /// Float32 steps between the k-th and (k+1)-th ranked score. 0 means an
        /// exact tie, where the reference's own choice is not a function of the
        /// values. A small non-zero value means a differently-rounded score
        /// could have selected a different expert.
        public let boundaryGapULPs: [Int]
        /// `boundaryGapULPs[token] == 0`.
        public var boundaryTies: [Bool] { boundaryGapULPs.map { $0 == 0 } }
    }

    /// `linear(x.float(), self.weight.float()) / self.gate_temp`.
    ///
    /// The published gate weight is BF16 `[routedExperts, dim]`; widening it to
    /// float32 is exact, and the reference does exactly that before the matmul
    /// rather than after it.
    public static func logits(
        hidden: MLXArray,
        gateWeight: MLXArray,
        gateTemperature: Float = defaultGateTemperature,
        stream: StreamOrDevice = .default
    ) throws -> MLXArray {
        guard hidden.ndim == 2, hidden.shape[0] > 0, hidden.shape[1] > 0 else {
            throw DeepSeekV41Error.experts(
                "router: hidden states must be [tokens, dim], got \(hidden.shape)")
        }
        guard gateWeight.ndim == 2, gateWeight.shape[1] == hidden.shape[1],
            gateWeight.shape[0] > 1
        else {
            throw DeepSeekV41Error.experts(
                "router: gate weight must be [experts, \(hidden.shape[1])], got "
                    + "\(gateWeight.shape)")
        }
        guard gateTemperature.isFinite, gateTemperature > 0 else {
            throw DeepSeekV41Error.experts(
                "router: gate temperature must be finite and positive")
        }
        let product = matmul(
            hidden.asType(.float32), gateWeight.asType(.float32).transposed(1, 0),
            stream: stream)
        return gateTemperature == 1 ? product : product / MLXArray(gateTemperature)
    }

    /// `F.softplus(scores).sqrt()`, with torch's threshold branch.
    ///
    /// The `minimum` inside the false arm is not redundant. MLX evaluates both
    /// arms of a `which`, so without it `exp(100)` would be computed, produce
    /// an infinity, and — because infinity times zero is NaN in some fused
    /// forms — risk contaminating the selected arm. Clamping the operand keeps
    /// the discarded arm finite while changing nothing where it is kept.
    public static func sqrtSoftplus(
        _ logits: MLXArray, stream: StreamOrDevice = .default
    ) -> MLXArray {
        let value = logits.asType(.float32)
        let threshold = MLXArray(softplusThreshold)
        let clamped = minimum(value, threshold, stream: stream)
        let below = log1p(exp(clamped, stream: stream), stream: stream)
        return sqrt(which(value .> threshold, value, below, stream: stream), stream: stream)
    }

    /// The whole of `Gate.forward` for the text path.
    public static func route(
        hidden: MLXArray,
        gateWeight: MLXArray,
        bias: MLXArray,
        expertCount: Int,
        expertsPerToken: Int,
        normalizeSelectedWeights: Bool,
        routingScale: Float,
        gateTemperature: Float = defaultGateTemperature,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        stream: StreamOrDevice = .default
    ) throws -> Selection {
        let logits = try logits(
            hidden: hidden, gateWeight: gateWeight, gateTemperature: gateTemperature,
            stream: stream)
        return try select(
            logits: logits, bias: bias, expertCount: expertCount,
            expertsPerToken: expertsPerToken,
            normalizeSelectedWeights: normalizeSelectedWeights,
            routingScale: routingScale, phaseAccounting: phaseAccounting,
            stream: stream)
    }

    /// Score, rank, select, gather, normalise and scale.
    public static func select(
        logits: MLXArray,
        bias: MLXArray,
        expertCount: Int,
        expertsPerToken: Int,
        normalizeSelectedWeights: Bool,
        routingScale: Float,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        stream: StreamOrDevice = .default
    ) throws -> Selection {
        guard logits.ndim == 2, logits.shape[1] == expertCount, logits.shape[0] > 0 else {
            throw DeepSeekV41Error.experts(
                "router: logits must be [tokens, \(expertCount)], got \(logits.shape)")
        }
        guard expertCount > 1, expertsPerToken > 0, expertsPerToken < expertCount else {
            throw DeepSeekV41Error.experts(
                "router: experts per token must be in 1..<\(expertCount), got "
                    + "\(expertsPerToken)")
        }
        guard bias.ndim == 1, bias.shape[0] == expertCount else {
            throw DeepSeekV41Error.experts(
                "router: correction bias must be [\(expertCount)], got \(bias.shape)")
        }
        guard routingScale.isFinite, routingScale > 0 else {
            throw DeepSeekV41Error.experts(
                "router: route scale must be finite and positive")
        }
        let tokenCount = logits.shape[0]
        guard !tokenCount.multipliedReportingOverflow(by: expertsPerToken).overflow else {
            throw DeepSeekV41Error.experts("router: token-by-slot work count overflows Int")
        }

        // Both host arrays are built before either is pulled, so the round trip
        // that delivers them is one — the same reason `DeepSeekV4Router` orders
        // it this way. The pager cannot name a tile until these ids exist.
        let scores = sqrtSoftplus(logits, stream: stream)
        let ranked = scores + bias.asType(.float32)
        // Bracketed, because this is the *only* blocking sync in a V4.1 decode
        // block and everything the block queued behind it drains here. Left
        // unbracketed it charged the whole block's GPU work to whichever term
        // enclosed the router — which is exactly the mis-labelling the
        // 2026-08-17 GPU-boundary terms were added to stop
        // (`DeepSeekV4PhaseMetrics`, "Waiting for the GPU").
        waitingForGPU(phaseAccounting, .routingSelect) {
            MLX.eval([scores, ranked])
        }

        let scoreValues = scores.asArray(Float.self)
        let rankedValues = ranked.asArray(Float.self)
        guard scoreValues.allSatisfy(\.isFinite) else {
            throw DeepSeekV41Error.experts("router: sqrt-softplus produced a non-finite score")
        }
        guard rankedValues.allSatisfy(\.isFinite) else {
            throw DeepSeekV41Error.experts("router: the biased selection score is non-finite")
        }

        var ids = [[Int]]()
        var gathered = [[Float]]()
        var denominators = [Float]()
        var weightValues = [[Float]]()
        var margins = [Double]()
        var gaps = [Int]()
        ids.reserveCapacity(tokenCount)
        gathered.reserveCapacity(tokenCount)
        denominators.reserveCapacity(tokenCount)
        weightValues.reserveCapacity(tokenCount)
        margins.reserveCapacity(tokenCount)
        gaps.reserveCapacity(tokenCount)

        for token in 0..<tokenCount {
            let start = token * expertCount
            // Descending ranked score; ties to the lower expert id. The rule is
            // the port's, not the reference's, because the reference has none.
            let order = (0..<expertCount).sorted { left, right in
                let lhs = rankedValues[start + left]
                let rhs = rankedValues[start + right]
                return lhs == rhs ? left < right : lhs > rhs
            }
            let chosen = Array(order.prefix(expertsPerToken))
            let selectedBoundary = rankedValues[start + order[expertsPerToken - 1]]
            let rejectedBoundary = rankedValues[start + order[expertsPerToken]]
            let scale = max(abs(Double(selectedBoundary)), abs(Double(rejectedBoundary)))
            margins.append(
                scale > 0 ? (Double(selectedBoundary) - Double(rejectedBoundary)) / scale : 0)
            gaps.append(floatSteps(selectedBoundary, rejectedBoundary))

            let row = chosen.map { scoreValues[start + $0] }
            // Sequential float32 accumulation, stated rather than delegated:
            // this is the step no two libraries agree on, so the port owns its
            // order instead of inheriting a reduction's.
            var denominator: Float = 0
            for value in row { denominator += value }
            denominators.append(denominator)

            var weights = row
            if normalizeSelectedWeights, expertsPerToken > 1 {
                let divisor = denominator + normalizationEpsilon
                guard divisor.isFinite, divisor > 0 else {
                    throw DeepSeekV41Error.experts(
                        "router: selected scores for token \(token) cannot be normalized")
                }
                weights = weights.map { $0 / divisor }
            }
            weights = weights.map { $0 * routingScale }

            ids.append(chosen)
            gathered.append(row)
            weightValues.append(weights)
        }

        return Selection(
            ids: ids,
            weights: MLXArray(weightValues.flatMap { $0 }, [tokenCount, expertsPerToken]),
            weightValues: weightValues,
            gathered: gathered,
            denominators: denominators,
            scores: scores,
            ranked: ranked,
            relativeMargins: margins,
            boundaryGapULPs: gaps)
    }

    /// Representable float32 values between two finite numbers, or `Int.max`
    /// when they straddle zero (where "one step" is not a useful measure).
    static func floatSteps(_ left: Float, _ right: Float) -> Int {
        if left == right { return 0 }
        guard left.isFinite, right.isFinite else { return Int.max }
        guard (left >= 0) == (right >= 0) else { return Int.max }
        let lhs = Int64(left.bitPattern & 0x7FFF_FFFF)
        let rhs = Int64(right.bitPattern & 0x7FFF_FFFF)
        return Int(abs(lhs - rhs))
    }
}
