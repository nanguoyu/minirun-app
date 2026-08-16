import Foundation
import MLX
import MLXBridge

/// Which of an expert's three projections is being fetched.
public enum ExpertProjection: String, CaseIterable, Sendable {
    /// `w1`, the situ **gate** half.
    case w1
    /// `w2`, the down projection.
    case w2
    /// `w3`, the situ **up** half.
    case w3
}

/// Supplies routed-expert weights to the MoE block.
///
/// The interface is a *gather*, not a weight handle, because the two backends
/// this milestone compares — a pager streaming container tiles and the same
/// containers held resident — differ in where the bytes come from and in
/// nothing else. Making the gather the boundary is what lets the bit-identity
/// gate be fair: both sides run the same tile loop with the same operands.
public protocol RoutedExpertBackend: AnyObject {
    /// `out[t, j, :] = W[expertIds[t, j]] · x[t, :]`
    ///
    /// - Parameters:
    ///   - x: `[tokens, inFeatures]`.
    ///   - expertIds: `[tokens, expertsPerToken]`, global expert indices.
    /// - Returns: `[tokens, expertsPerToken, outFeatures]`.
    func gather(
        _ x: MLXArray, layer: Int, projection: ExpertProjection, expertIds: [[Int]]
    ) throws -> MLXArray
}

/// The MoE router.
///
/// Three traps, all recorded in the `subop-moe-layer01` manifest and all easy
/// to get wrong in the same direction:
///
/// - the router sees the **full hidden state**, before the latent projection;
/// - the correction bias steers **selection only** and never enters the routing
///   weights, which are gathered from the *unbiased* scores;
/// - `torch.topk(sorted=False)` promises no order, so ids are a set.
public enum K3Router {

    public struct Selection {
        /// `[tokens, expertsPerToken]`, in descending score order.
        public var ids: [[Int]]
        /// `[tokens, expertsPerToken]`, aligned with `ids`.
        public var weights: MLXArray
        public var logits: MLXArray
        public var scoresForChoice: MLXArray
        /// `(kth selected − first rejected) / max(|kth|, |first rejected|)` per
        /// token — the statistic M6A screened prompts on. Reported so a port
        /// can say whether a disagreement sits near a tie.
        public var relativeMargins: [Double]
    }

    public static func select(
        _ x: MLXArray, gateWeight: MLXArray, correctionBias: MLXArray, config: TinyK3Config
    ) -> Selection {
        // `F.linear(hidden.type(torch.float32), self.weight.type(torch.float32))` —
        // the router is one of the five operations pinned to float32.
        return select(
            logits: k3Linear(x.asType(.float32), gateWeight.asType(.float32)),
            correctionBias: correctionBias,
            numExperts: config.numExperts,
            topK: config.numExpertsPerToken,
            renormalize: config.moeRenormalize,
            scalingFactor: config.routedScalingFactor)
    }

    /// The same selection, from logits produced somewhere else.
    ///
    /// A streaming layer multiplies the router weight in row bands as the pager
    /// delivers them, so it arrives here holding logits rather than a resident
    /// `[experts, hidden]` matrix. Splitting the function keeps the three traps
    /// above — full hidden state, bias steers selection only, ids are a set —
    /// stated in one place rather than two.
    public static func select(
        logits: MLXArray,
        correctionBias: MLXArray,
        numExperts experts: Int,
        topK: Int,
        renormalize: Bool,
        scalingFactor: Float
    ) -> Selection {
        let scores = sigmoid(logits)
        let scoresForChoice = scores + correctionBias.asType(.float32)

        let tokens = logits.shape[0]
        // The selection is made on the host: the chosen ids decide which
        // container tiles the pager has to fetch, so they have to be concrete
        // values before the expert gather can be expressed at all.
        scoresForChoice.eval()
        scores.eval()
        let choiceValues = scoresForChoice.asArray(Float.self)
        let scoreValues = scores.asArray(Float.self)

        var ids: [[Int]] = []
        var gathered = [Float](repeating: 0, count: tokens * topK)
        var margins: [Double] = []
        ids.reserveCapacity(tokens)

        for token in 0..<tokens {
            let row = Array(choiceValues[(token * experts)..<((token + 1) * experts)])
            // Descending by score, ties broken by index. torch.topk(sorted:false)
            // makes no order promise, so the reference's ids are compared as a
            // set; this port picks one deterministic order and states it.
            let order = (0..<experts).sorted { (row[$0], -$1) > (row[$1], -$0) }
            let selected = Array(order.prefix(topK))
            ids.append(selected)

            for (slot, expert) in selected.enumerated() {
                gathered[token * topK + slot] = scoreValues[token * experts + expert]
            }
            let kth = Double(row[order[topK - 1]])
            let rejected = Double(row[order[topK]])
            let scale = max(abs(kth), abs(rejected))
            margins.append(scale > 0 ? (kth - rejected) / scale : 0)
        }

        var weights = MLXArray(gathered, [tokens, topK])
        if topK > 1 && renormalize {
            weights = weights / (sum(weights, axis: -1, keepDims: true) + Float(1e-20))
        }
        weights = weights * scalingFactor

        return Selection(
            ids: ids, weights: weights, logits: logits, scoresForChoice: scoresForChoice,
            relativeMargins: margins)
    }
}

/// The latent MoE block, `KimiSparseMoeBlock.forward`
/// (`modeling_kimi_k3_linear.py:904-982`).
///
/// ```text
/// 1. topk_idx, topk_weight = gate(h)          // on the FULL hidden state
/// 2. z = routed_expert_down_proj(h)           // hidden -> latent 512
/// 3. y = Σ_j topk_weight[j] · expert_j(z)     // in the latent
/// 4. y = routed_expert_norm(y)                // RMSNorm on the SUM, latent width
/// 5. y = routed_expert_up_proj(y)             // latent -> hidden
/// 6. y = y + shared_experts(h)                // ORIGINAL hidden, bypasses the latent
/// ```
///
/// The shared expert runs at model width (1024 → 256 → 1024) while the routed
/// experts run in the 512-wide latent (512 → 256 → 512). They are not
/// interchangeable and their weights are differently shaped.
public enum K3MoE {

    public struct Weights {
        public var gateWeight: MLXArray
        public var correctionBias: MLXArray
        /// Quantized in the tiny checkpoint, BF16 in the flagship one — see
        /// ``K3Projection``.
        public var routedDownProj: K3Projection
        public var routedUpProj: K3Projection
        public var routedNorm: MLXArray
        public var shared: K3MLP.QuantizedWeights

        public init(
            gateWeight: MLXArray, correctionBias: MLXArray,
            routedDownProj: K3Projection, routedUpProj: K3Projection,
            routedNorm: MLXArray, shared: K3MLP.QuantizedWeights
        ) {
            self.gateWeight = gateWeight
            self.correctionBias = correctionBias
            self.routedDownProj = routedDownProj
            self.routedUpProj = routedUpProj
            self.routedNorm = routedNorm
            self.shared = shared
        }

        public init(
            gateWeight: MLXArray, correctionBias: MLXArray,
            routedDownProj: MXFP4Weights, routedUpProj: MXFP4Weights,
            routedNorm: MLXArray, shared: K3MLP.QuantizedWeights
        ) {
            self.init(
                gateWeight: gateWeight, correctionBias: correctionBias,
                routedDownProj: .mxfp4(routedDownProj), routedUpProj: .mxfp4(routedUpProj),
                routedNorm: routedNorm, shared: shared)
        }
    }

    public struct Trace {
        public var selection: K3Router.Selection
        public var routedDownLatent: MLXArray
        public var expertOutputs: MLXArray
        public var expertGate: MLXArray
        public var expertUp: MLXArray
        public var expertActivation: MLXArray
        public var routedWeightedSum: MLXArray
        public var routedNorm: MLXArray
        public var routedUp: MLXArray
        public var sharedExpertOutput: MLXArray
        public var output: MLXArray
    }

    public static func forward(
        _ x: MLXArray, weights: Weights, backend: RoutedExpertBackend, layer: Int,
        config: TinyK3Config
    ) throws -> (output: MLXArray, trace: Trace) {
        let tokens = x.shape[0]
        let topK = config.numExpertsPerToken

        let selection = K3Router.select(
            x, gateWeight: weights.gateWeight, correctionBias: weights.correctionBias,
            config: config)

        let latent = weights.routedDownProj(x)

        // Both situ halves come from the same routed activation, so both are
        // gathered at [tokens, expertsPerToken, moeIntermediate].
        let gateHalf = try backend.gather(latent, layer: layer, projection: .w1, expertIds: selection.ids)
        let upHalf = try backend.gather(latent, layer: layer, projection: .w3, expertIds: selection.ids)
        let activation = K3Activation.situ(
            concatenated([gateHalf, upHalf], axis: -1),
            beta: config.situBeta, linearBeta: config.situLinearBeta)

        // Each (token, slot) row now has its own activation, so the down
        // projection is a gather over `tokens * expertsPerToken` rows with one
        // expert each.
        let flatIds = selection.ids.flatMap { $0 }.map { [$0] }
        let expertOutputs = try backend.gather(
            activation.reshaped([tokens * topK, config.moeIntermediateSize]),
            layer: layer, projection: .w2, expertIds: flatIds
        ).reshaped([tokens, topK, config.routedExpertHiddenSize])

        // Σ_j weight[j] · out[j], summed over the slot axis in float32.
        let weighted = expertOutputs * expandedDimensions(selection.weights, axis: -1)
        let routedSum = sum(weighted, axis: 1)

        // The norm applies to the weighted SUM, in the latent, before the
        // up-projection — not to each expert output.
        let normed =
            config.latentMoEUseNorm
            ? K3Norm.rms(routedSum, weight: weights.routedNorm, eps: config.rmsNormEps)
            : routedSum
        let routedUp = weights.routedUpProj(normed)

        let shared = K3MLP.quantized(x, weights: weights.shared, config: config)
        let output = routedUp + shared

        return (
            output,
            Trace(
                selection: selection, routedDownLatent: latent, expertOutputs: expertOutputs,
                expertGate: gateHalf, expertUp: upHalf, expertActivation: activation,
                routedWeightedSum: routedSum, routedNorm: normed, routedUp: routedUp,
                sharedExpertOutput: shared, output: output)
        )
    }
}

/// The two feed-forward shapes that are not routed: layer 0's dense MLP
/// (float32 in this checkpoint) and the shared expert (MXFP4).
///
/// Both compute `down(situ(concat(gate(x), up(x))))`. The gate/up split is on
/// the *concatenated* pair, which is what makes `w1` the situ gate half and
/// `w3` the up half.
public enum K3MLP {

    public struct DenseWeights {
        public var gateProj: MLXArray
        public var upProj: MLXArray
        public var downProj: MLXArray
    }

    /// The shared expert's three projections, whichever way they are stored.
    ///
    /// The name is M6's and is now only half true — at flagship scale these are
    /// BF16 — but it is kept so the tiny suites keep reading as written. Both
    /// storages are accepted.
    public struct QuantizedWeights {
        public var gateProj: K3Projection
        public var upProj: K3Projection
        public var downProj: K3Projection

        public init(gateProj: K3Projection, upProj: K3Projection, downProj: K3Projection) {
            self.gateProj = gateProj
            self.upProj = upProj
            self.downProj = downProj
        }

        public init(gateProj: MXFP4Weights, upProj: MXFP4Weights, downProj: MXFP4Weights) {
            self.init(
                gateProj: .mxfp4(gateProj), upProj: .mxfp4(upProj), downProj: .mxfp4(downProj))
        }

        public init(gateProj: MLXArray, upProj: MLXArray, downProj: MLXArray) {
            self.init(
                gateProj: .dense(gateProj), upProj: .dense(upProj), downProj: .dense(downProj))
        }
    }

    public static func dense(
        _ x: MLXArray, weights: DenseWeights, config: TinyK3Config
    ) -> MLXArray {
        let gateUp = concatenated(
            [k3Linear(x, weights.gateProj), k3Linear(x, weights.upProj)], axis: -1)
        let activated = K3Activation.situ(
            gateUp, beta: config.situBeta, linearBeta: config.situLinearBeta)
        return k3Linear(activated, weights.downProj)
    }

    public static func quantized(
        _ x: MLXArray, weights: QuantizedWeights, config: TinyK3Config
    ) -> MLXArray {
        let gateUp = concatenated(
            [weights.gateProj(x), weights.upProj(x)], axis: -1)
        let activated = K3Activation.situ(
            gateUp, beta: config.situBeta, linearBeta: config.situLinearBeta)
        return weights.downProj(activated)
    }
}
