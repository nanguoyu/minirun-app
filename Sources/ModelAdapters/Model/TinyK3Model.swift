import Foundation
import MLX
import MLXBridge

/// Everything one forward pass captured, named as the reference's trace bundle
/// names it.
///
/// The point of carrying all of this is spec §23 M6's third exit criterion:
/// "failure localization is possible by layer/operator". A divergence lands on
/// `layer03_output` or `prefill_attn_res08_probs`, not on "the answer is wrong".
public struct TinyK3Capture {
    public var embedding: MLXArray
    /// One per decoder layer: the prefix sum the layer returns.
    public var layerOutputs: [MLXArray]
    /// The output-stage AttnRes result, i.e. the final norm's input.
    public var outputAttnRes: MLXArray
    public var finalNorm: MLXArray
    /// In call order: layer 0's mlp call, then (self_attn, mlp) per later
    /// layer, then the output stage. 16 calls for an 8-layer model.
    public var attnResProbabilities: [MLXArray]
    /// One entry per MoE layer, in layer order.
    public var routerLogits: [MLXArray]
    public var routerScoresForChoice: [MLXArray]
    public var routerTopKIds: [[[Int]]]
    public var routerTopKWeights: [MLXArray]
    /// Smallest relative routing margin over every (MoE layer, position) in
    /// this pass. M6A screens prompts on this; a port reports it so a
    /// disagreement near a tie can be told apart from a bug.
    public var minimumRelativeRoutingMargin: Double
    /// Every relative routing margin, one inner array per MoE layer in layer
    /// order, one entry per position.
    ///
    /// The minimum above is an extreme-value statistic: at flagship shape it is
    /// the smallest of `92 × positions` draws, so it falls as the prompt gets
    /// longer even when nothing about the routing is unusual. Screening a
    /// prompt on the minimum alone therefore cannot distinguish "this prompt
    /// routes near a tie" from "this prompt had more chances to". The
    /// distribution can, so it is reported.
    public var routerRelativeMargins: [[Double]]
    /// Last-position logits over the whole vocabulary.
    public var logits: [Float]
    public var greedyTokenId: Int
}

/// The rolling state a sequence carries between forward passes.
public struct TinyK3State {
    public var kda: [Int: K3KDA.State] = [:]
    public var mla: [Int: K3MLA.State] = [:]
    public init() {}
}

/// The tiny Kimi K3 text model: ``K3Model`` bound to a resident checkpoint.
///
/// The layer schedule and every numeric decision live in ``K3Model``. This type
/// remains because M6's suites are written against it and because "the whole
/// checkpoint is one mapped file" is a genuinely different binding from M9's,
/// worth keeping nameable rather than collapsing into a general case.
///
/// With 8 layers at block size 4, checkpoints are appended at layers 0 and 4,
/// so `blockResidual` holds 1 block for layers 0–3 and 2 for layers 4–7 — which
/// is exactly the `[T, 2]` then `[T, 3]` shape progression the committed
/// `attn_res*_probs` captures have.
public final class TinyK3Model {

    public let config: TinyK3Config
    public let weights: TinyK3Weights
    public let expertBackend: RoutedExpertBackend
    private let model: K3Model

    /// Rows of `lm_head` materialised at a time. 16,384 rows × 1,024 float32 is
    /// 64 MiB, against 640 MiB for the whole projection.
    public var logitChunkRows: Int {
        get { model.logitChunkRows }
        set { model.logitChunkRows = newValue }
    }

    public init(config: TinyK3Config, weights: TinyK3Weights, expertBackend: RoutedExpertBackend) {
        self.config = config
        self.weights = weights
        self.expertBackend = expertBackend
        self.model = K3Model(config: config, source: weights, expertBackend: expertBackend)
    }

    /// One forward pass over `tokenIds`, extending `state`.
    public func forward(tokenIds: [Int], state: inout TinyK3State) throws -> TinyK3Capture {
        try model.forward(tokenIds: tokenIds, state: &state)
    }

    /// The vocabulary projection, walked in row chunks.
    public func logits(for hidden: MLXArray) throws -> [Float] {
        try model.logits(for: hidden)
    }

    /// Prefill followed by greedy decode steps, threading the state through.
    ///
    /// - Returns: every capture in order — the prefill pass first, then one per
    ///   decode step — and the generated ids.
    public func generate(
        tokenIds: [Int], decodeSteps: Int
    ) throws -> (captures: [TinyK3Capture], generated: [Int]) {
        try model.generate(tokenIds: tokenIds, decodeSteps: decodeSteps)
    }
}
