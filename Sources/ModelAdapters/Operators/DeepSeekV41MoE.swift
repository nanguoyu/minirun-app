import Foundation
import MLX
import MLXBridge

/// One V4.1 block's feed-forward: 384 routed experts, six of them, plus one
/// shared expert every token goes through.
///
/// `MoE.forward` in full:
///
/// ```python
/// weights, indices = self.gate(x, image_mask)
/// y = torch.zeros_like(x, dtype=torch.float32)
/// for i in ...: y[idx] += expert(x[idx], weights[idx, top, None])
/// y += self.shared_experts(x)
/// return y.type_as(x).view(shape)
/// ```
///
/// The accumulator is float32 whatever `x` is, and the cast back to `x`'s dtype
/// is the *last* thing that happens — after the shared expert has been added,
/// not before. On a bf16 decode that ordering is worth about a bf16 ulp of the
/// larger of the two terms, and it is free to get right.
///
/// Every routed expert's contribution already carries its routing weight
/// (``DeepSeekV41RoutedExperts``); the shared expert carries none. There is
/// exactly one shared expert — `assert args.n_shared_experts == 1` in the
/// reference — so this takes three matrices and not a list.
public enum DeepSeekV41MoE {

    public struct Output {
        /// `[tokens, dim]`, float32: the weighted routed sum alone.
        public let routed: MLXArray
        /// `[tokens, dim]`, float32: the shared expert alone.
        public let shared: MLXArray
        /// `routed + shared`, cast back to the hidden state's own dtype.
        public let combined: MLXArray
        /// `[tokens, slots, dim]`, float32: each routed slot's contribution,
        /// so a disagreement can be localised to one expert.
        public let perExpert: MLXArray
        /// What the router decided.
        public let selection: DeepSeekV41Router.Selection
    }

    /// Route, evaluate, and add the shared expert.
    public static func forward(
        _ input: MLXArray,
        gateWeight: MLXArray,
        gateBias: MLXArray,
        expertCount: Int,
        expertsPerToken: Int,
        normalizeSelectedWeights: Bool,
        routingScale: Float,
        swiGLULimit: Float,
        source: DeepSeekV41RoutedExpertSource,
        sharedGate: BlockFP8Weights,
        sharedDown: BlockFP8Weights,
        sharedUp: BlockFP8Weights,
        activation: DeepSeekV41ExpertActivation = .referenceFP8,
        boundsLiveOperands: Bool = false,
        diagnostics: DeepSeekV4Diagnostics = .validating,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        stream: StreamOrDevice = .default
    ) throws -> Output {
        let selection = try DeepSeekV41Router.route(
            hidden: input, gateWeight: gateWeight, bias: gateBias,
            expertCount: expertCount, expertsPerToken: expertsPerToken,
            normalizeSelectedWeights: normalizeSelectedWeights,
            routingScale: routingScale, stream: stream)
        return try evaluate(
            input, selection: selection, swiGLULimit: swiGLULimit, source: source,
            sharedGate: sharedGate, sharedDown: sharedDown, sharedUp: sharedUp,
            activation: activation, boundsLiveOperands: boundsLiveOperands,
            diagnostics: diagnostics,
            phaseAccounting: phaseAccounting, stream: stream)
    }

    /// The same, over a selection somebody else made — the shape a reference
    /// trace is replayed in, and what lets the router and the experts be
    /// tested apart.
    public static func evaluate(
        _ input: MLXArray,
        selection: DeepSeekV41Router.Selection,
        swiGLULimit: Float,
        source: DeepSeekV41RoutedExpertSource,
        sharedGate: BlockFP8Weights,
        sharedDown: BlockFP8Weights,
        sharedUp: BlockFP8Weights,
        activation: DeepSeekV41ExpertActivation = .referenceFP8,
        boundsLiveOperands: Bool = false,
        diagnostics: DeepSeekV4Diagnostics = .validating,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        stream: StreamOrDevice = .default
    ) throws -> Output {
        let perExpert = try DeepSeekV41RoutedExperts.perExpertOutputs(
            input, expertIDs: selection.ids, routingWeights: selection.weights,
            swiGLULimit: swiGLULimit, source: source, activation: activation,
            boundsLiveOperands: boundsLiveOperands,
            diagnostics: diagnostics, phaseAccounting: phaseAccounting, stream: stream)
        let routed = sum(perExpert, axis: 1, stream: stream)
        let shared = try DeepSeekV41SharedExpert.forward(
            input, gate: sharedGate, down: sharedDown, up: sharedUp,
            swiGLULimit: swiGLULimit, activation: activation,
            diagnostics: diagnostics, stream: stream)
        return Output(
            routed: routed,
            shared: shared,
            // `y.type_as(x)` last: after the shared expert, not before it.
            combined: (routed + shared).asType(input.dtype),
            perExpert: perExpert,
            selection: selection)
    }
}
