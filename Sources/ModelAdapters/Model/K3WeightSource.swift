import Foundation
import MLX
import MLXBridge

/// Where a decode step gets its weights from.
///
/// M6 bound the model to one memory-resident `model.safetensors`; M9's model is
/// 1.56 TB spread over per-layer containers on an external volume. The layer
/// schedule, the operators and the numerics are the same in both cases — only
/// the provenance of the tensors differs — so the schedule is written against
/// this protocol and the provenance is a parameter.
///
/// ## The contract that keeps memory bounded
///
/// `layerWeights(_:)` may materialise the layer it is asked for. It must not
/// retain the previous one: the caller pairs every `layerWeights(i)` with a
/// `releaseLayer(i)` after that layer's output has been `eval()`d, and a
/// conforming source is entitled to free the layer's buffers at that point.
/// The memory budget of a full-model run is therefore
///
/// ```text
/// max over layers (one layer's weights) + expert pool + MLX working set
/// ```
///
/// which is a stated number (spec 5.3), not an emergent one. A source that
/// ignored `releaseLayer` would still be correct and would still be 1.56 TB.
///
/// ## The layer is handed over in two halves
///
/// A layer is not consumed all at once: attention reads its own projections and
/// norms and is finished with them before the MLP/MoE half is touched. The
/// caller therefore asks for ``layerAttentionWeights(_:)``, computes attention,
/// evaluates the result, calls ``releaseLayerAttention(_:)``, and only then asks
/// for ``layerFeedForwardWeights(_:)``. The first term above becomes
///
/// ```text
/// max over layers (max of that layer's two halves, widened)
/// ```
///
/// which at flagship shape is roughly half of what it was, because the widest
/// layer — layer 0, dense, 4.68 GB widened — is close to an even split. The
/// whole-layer ``layerWeights(_:)`` is still the truthful view for a caller that
/// wants the layer at once, and it is what a resident source serves either way.
///
/// ## Why the whole vocabulary never appears
///
/// `embeddingRows` and `lmHeadRows` are row-addressed rather than returning the
/// table, because at flagship shape each of those tensors is 2.35 GB — larger
/// than every other tensor in the model put together (spec 16.4, 16.5).
public protocol K3WeightSource: AnyObject {

    var config: TinyK3Config { get }

    /// The embedding rows for these token ids, in order. Never the table.
    func embeddingRows(_ tokenIds: [Int]) throws -> MLXArray

    /// Everything layer `index` needs, for the duration of that layer.
    func layerWeights(_ index: Int) throws -> TinyK3LayerWeights

    /// The half of layer `index` the attention phase consumes.
    ///
    /// Requested first, and released — through ``releaseLayerAttention(_:)`` —
    /// before the other half is asked for, so a streaming source never has to
    /// hold the whole layer widened at once.
    func layerAttentionWeights(_ index: Int) throws -> TinyK3AttentionWeights

    /// The half of layer `index` the MLP/MoE phase consumes.
    func layerFeedForwardWeights(_ index: Int) throws -> TinyK3FeedForwardWeights

    /// Signals that layer `index`'s **attention** weights have been consumed.
    ///
    /// The caller has already evaluated everything computed from them, so the
    /// arrays are dead the moment the source drops its own references. A source
    /// that ignores this is correct and holds the layer's widest moment for the
    /// layer's whole duration, which is what this exists to avoid.
    func releaseLayerAttention(_ index: Int)

    /// Signals that layer `index` is finished and its buffers may be released.
    func releaseLayer(_ index: Int)

    /// `output_attn_res_norm.weight * output_attn_res_proj.weight[0]`, precomputed.
    var outputAttnResScore: MLXArray { get }

    var finalNormWeight: MLXArray { get }

    /// `count` rows of `lm_head` starting at `first`.
    func lmHeadRows(from first: Int, count: Int) throws -> MLXArray
}

extension K3WeightSource {
    /// Most sources hold nothing per layer and need no hook.
    public func releaseLayer(_ index: Int) {}

    /// A source whose layer is already resident — a mapped checkpoint, a
    /// generated fixture — has nothing to release early, so the default is the
    /// whole layer sliced in two. It costs nothing and it keeps the two-phase
    /// call sequence the only sequence the model has to know about.
    public func layerAttentionWeights(_ index: Int) throws -> TinyK3AttentionWeights {
        try layerWeights(index).attention
    }

    public func layerFeedForwardWeights(_ index: Int) throws -> TinyK3FeedForwardWeights {
        try layerWeights(index).feedForward
    }

    public func releaseLayerAttention(_ index: Int) {}
}

/// The M6 resident checkpoint, seen through the M9 seam.
///
/// Nothing is copied and nothing is released: every layer already points into
/// one memory-mapped `model.safetensors`, which is exactly why this source is
/// only usable at tiny scale.
extension TinyK3Weights: K3WeightSource {

    public func embeddingRows(_ tokenIds: [Int]) throws -> MLXArray {
        try file.rows(TinyK3Weights.embeddingTensor, at: tokenIds)
    }

    public func layerWeights(_ index: Int) throws -> TinyK3LayerWeights {
        guard layers.indices.contains(index) else {
            throw TinyK3Error.configuration(
                "layer \(index) requested but the checkpoint has \(layers.count)")
        }
        return layers[index]
    }

    public func lmHeadRows(from first: Int, count: Int) throws -> MLXArray {
        try file.rowRange(TinyK3Weights.lmHeadTensor, from: first, count: count)
    }
}
