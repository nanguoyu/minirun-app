import Foundation
import MLX
import MLXBridge

/// The per-block numbers `Attention` reads out of the configuration.
///
/// Every one of them comes from ``DeepSeekV41Config``; none is written down
/// here. ``init(block:config:)`` is the only way to build one that a product
/// path uses, and a test builds one directly so it can shrink the model without
/// shrinking the arithmetic.
public struct DeepSeekV41AttentionGeometry: Sendable, Equatable {
    public let block: Int
    public let mode: DeepSeekV41AttentionMode
    public let hiddenSize: Int
    public let heads: Int
    public let headDimension: Int
    public let ropeHeadDimension: Int
    public let queryLowRank: Int
    public let outputGroups: Int
    public let outputLowRank: Int
    public let windowSize: Int
    public let compressionRatio: Int
    public let indexHeads: Int
    public let indexHeadDimension: Int
    public let indexTopK: Int
    public let candidateTopKBlocks: Int
    public let candidateBlockSize: Int
    /// Whether this block publishes the candidate pool the decoder's Reindex
    /// layers are then constrained by.
    public let isCandidateSource: Bool
    /// Whether this block is *after* the candidate source and must mask.
    public let usesCandidates: Bool
    public let normEpsilon: Float

    public var softmaxScale: Float {
        Float(1 / Foundation.sqrt(Double(headDimension)))
    }

    public init(block: Int, config: DeepSeekV41Config) throws {
        self.block = block
        self.mode = try config.attentionMode(block: block)
        self.hiddenSize = config.hiddenSize
        self.heads = config.numberOfAttentionHeads
        self.headDimension = config.attentionHeadDimension
        self.ropeHeadDimension = config.ropeHeadDimension
        self.queryLowRank = config.queryLowRank
        self.outputGroups = config.outputGroups
        self.outputLowRank = config.outputLowRank
        self.windowSize = config.slidingWindow
        self.compressionRatio = try config.compressionRatio(block: block)
        self.indexHeads = config.indexHeadCount
        self.indexHeadDimension = config.indexHeadDimension
        self.indexTopK = config.indexTopK
        self.candidateTopKBlocks = config.candidateTopKBlocks
        self.candidateBlockSize = config.candidateBlockSize
        self.isCandidateSource = block == config.candidateSourceLayer
        self.usesCandidates =
            config.candidateSourceLayer >= 0 && config.candidateSourceLayer < block
        self.normEpsilon = config.rmsNormEpsilon
    }

    public init(
        block: Int, mode: DeepSeekV41AttentionMode, hiddenSize: Int, heads: Int,
        headDimension: Int, ropeHeadDimension: Int, queryLowRank: Int, outputGroups: Int,
        outputLowRank: Int, windowSize: Int, compressionRatio: Int, indexHeads: Int,
        indexHeadDimension: Int, indexTopK: Int, candidateTopKBlocks: Int,
        candidateBlockSize: Int, isCandidateSource: Bool, usesCandidates: Bool,
        normEpsilon: Float
    ) {
        self.block = block
        self.mode = mode
        self.hiddenSize = hiddenSize
        self.heads = heads
        self.headDimension = headDimension
        self.ropeHeadDimension = ropeHeadDimension
        self.queryLowRank = queryLowRank
        self.outputGroups = outputGroups
        self.outputLowRank = outputLowRank
        self.windowSize = windowSize
        self.compressionRatio = compressionRatio
        self.indexHeads = indexHeads
        self.indexHeadDimension = indexHeadDimension
        self.indexTopK = indexTopK
        self.candidateTopKBlocks = candidateTopKBlocks
        self.candidateBlockSize = candidateBlockSize
        self.isCandidateSource = isCandidateSource
        self.usesCandidates = usesCandidates
        self.normEpsilon = normEpsilon
    }
}

/// One block's attention tensors, already read from its unit.
///
/// The optional members are optional *by design*, not by damage: a Reuse block
/// has no indexer and a window-only block has neither indexer nor compressor.
/// ``DeepSeekV41BlockArtifact/contains(_:)`` is what answers "published or
/// absent", and ``DeepSeekV41AttentionForward`` refuses a geometry whose mode
/// disagrees with the tensors it was handed rather than quietly taking another
/// path.
///
/// `outputDown` is `attn.wo_a` **already dequantized to bfloat16**. Our
/// container carries the published FP8 [32, 32] bytes; DeepSeek's `convert.py`
/// dequantizes that matrix, so the reference holds bfloat16 and reads it through
/// an `einsum` with no activation quantization at all. See
/// ``DeepSeekV41FP8Linear/dequantizedToBFloat16(_:stream:)``.
public struct DeepSeekV41AttentionWeights {
    public let sink: MLXArray
    public let queryDown: BlockFP8Weights
    public let queryNorm: MLXArray
    public let queryUp: BlockFP8Weights
    public let keyValue: BlockFP8Weights
    public let keyValueNorm: MLXArray
    public let outputDown: MLXArray
    public let outputUp: BlockFP8Weights
    public let compressorKeyValue: MLXArray?
    public let compressorGate: MLXArray?
    public let compressorNorm: MLXArray?
    public let indexerKey: MLXArray?
    public let indexerKeyNorm: MLXArray?
    public let indexerQueryUp: BlockFP8Weights?
    public let indexerWeights: MLXArray?

    public init(
        sink: MLXArray, queryDown: BlockFP8Weights, queryNorm: MLXArray,
        queryUp: BlockFP8Weights, keyValue: BlockFP8Weights, keyValueNorm: MLXArray,
        outputDown: MLXArray, outputUp: BlockFP8Weights,
        compressorKeyValue: MLXArray? = nil, compressorGate: MLXArray? = nil,
        compressorNorm: MLXArray? = nil, indexerKey: MLXArray? = nil,
        indexerKeyNorm: MLXArray? = nil, indexerQueryUp: BlockFP8Weights? = nil,
        indexerWeights: MLXArray? = nil
    ) {
        self.sink = sink
        self.queryDown = queryDown
        self.queryNorm = queryNorm
        self.queryUp = queryUp
        self.keyValue = keyValue
        self.keyValueNorm = keyValueNorm
        self.outputDown = outputDown
        self.outputUp = outputUp
        self.compressorKeyValue = compressorKeyValue
        self.compressorGate = compressorGate
        self.compressorNorm = compressorNorm
        self.indexerKey = indexerKey
        self.indexerKeyNorm = indexerKeyNorm
        self.indexerQueryUp = indexerQueryUp
        self.indexerWeights = indexerWeights
    }
}

/// What attention layers hand down the stack instead of recomputing it —
/// `SharedAttentionRuntime` in the reference.
///
/// Layers run in order and every source writes before its consumers read, so one
/// slot each is enough and nothing needs resetting between forwards. A class,
/// not a struct, because "the same object the whole stack writes into" is the
/// contract: a value type would give every block its own copy and a Reuse block
/// would silently attend to nothing.
public final class DeepSeekV41SharedAttentionRuntime {
    /// `[rows, headDimension]`, published by a KV source.
    public var compressedKeyValues: MLXArray?
    /// `[rows, indexHeadDimension]`, published by a KV source.
    public var indexKeys: MLXArray?
    /// Published by an index source; read by every Reuse block after it.
    public var topKIndices: [[Int]]?
    /// Published by the candidate source; read by every Reindex block after it.
    public var candidateMask: [[Bool]]?
    /// The last index score matrix a Top-K was taken from — after the
    /// reachability mask and after the candidate mask, which is the matrix
    /// `index_score.topk(...)` sees in the reference.
    ///
    /// Nothing in the model reads it. It is here because a Top-K that disagrees
    /// with the reference says only that *something* upstream did, and the
    /// scores are where the two candidate explanations — a wrong score and a
    /// wrong selection rule — separate.
    public var lastIndexScoreRows: [[Float]]?

    public init() {}
}

/// One block's own caches: the sliding-window ring, and — for a KV source —
/// the compressed KV, the index keys and the compressor's partial group.
public struct DeepSeekV41AttentionCaches {
    /// `[windowSize, headDimension]`, a ring. Slot `p % windowSize` holds
    /// token `p`, and a slot that has never been written is zero.
    public var window: MLXArray
    /// `[rows, headDimension]`, append-only in position order.
    public var compressed: MLXArray?
    /// `[rows, indexHeadDimension]`, append-only in position order.
    public var indexKeys: MLXArray?
    public var compressorState: DeepSeekV41Compressor.State?

    public init(geometry: DeepSeekV41AttentionGeometry) throws {
        self.window = MLXArray.zeros(
            [geometry.windowSize, geometry.headDimension], dtype: .bfloat16)
        if geometry.mode == .full, geometry.compressionRatio > 1 {
            self.compressorState = try DeepSeekV41Compressor.State(
                ratio: geometry.compressionRatio, headDimension: geometry.headDimension)
        }
    }
}

/// `Attention.forward`: latent attention over two KV sources at once.
///
/// A sliding window of raw KV, plus — when `compress_ratio > 0` — `index_topk`
/// compressed positions reaching further back, concatenated into **one**
/// `sparse_attn` call. Q and the output projection are both low-rank, the latter
/// grouped.
///
/// `compress_ratio > 0` does not mean the block computes its own main KV: only
/// `kv_source_layers` do, and everything after one reads that same cache. Which
/// of the four jobs a block has is ``DeepSeekV41AttentionMode``, derived by the
/// configuration from `compress_ratios`, `kv_source_layers` and
/// `index_source_layers` — never from a block number written down here.
///
/// ## The order inside is not free
///
/// The indexer must see the compressor's latent **before** RoPE, because the
/// index keys are derived from the unrotated form; the reference is explicit
/// that it runs the indexer before the cache write for exactly that reason. And
/// the compressed cache is read *after* the write, so a block that compresses
/// attends to the entry it just produced. Both are reproduced literally below.
public enum DeepSeekV41AttentionForward {
    public struct Result {
        /// `[tokens, hiddenSize]`, bfloat16.
        public let output: MLXArray
        public let caches: DeepSeekV41AttentionCaches
    }

    public static func forward(
        hidden: MLXArray,
        geometry: DeepSeekV41AttentionGeometry,
        weights: DeepSeekV41AttentionWeights,
        caches: DeepSeekV41AttentionCaches,
        shared: DeepSeekV41SharedAttentionRuntime,
        table: DeepSeekV41RotaryTable,
        startPosition: Int,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> Result {
        guard hidden.ndim == 2, hidden.shape[0] > 0,
            hidden.shape[1] == geometry.hiddenSize, startPosition >= 0
        else {
            throw DeepSeekV41Error.configuration(
                "attention input must be [tokens, \(geometry.hiddenSize)] with a nonnegative "
                    + "start position; got \(hidden.shape) and \(startPosition)")
        }
        let tokens = hidden.shape[0]
        guard startPosition == 0 || tokens == 1 else {
            throw DeepSeekV41Error.configuration(
                "a decode step carries exactly one token, got \(tokens)")
        }
        var caches = caches
        let positions = Array(startPosition..<(startPosition + tokens))

        // Q: wq_a -> q_norm -> wq_b -> heads -> rotary tail.
        let queryLatent = try DeepSeekV41RMSNorm.apply(
            try DeepSeekV41FP8Linear.project(
                hidden, weights: weights.queryDown,
                phaseAccounting: phaseAccounting, diagnostics: diagnostics),
            weight: weights.queryNorm, epsilon: geometry.normEpsilon)
        let queries = try DeepSeekV41Rotary.applyToTail(
            try DeepSeekV41FP8Linear.project(
                queryLatent, weights: weights.queryUp,
                phaseAccounting: phaseAccounting, diagnostics: diagnostics)
                .reshaped([tokens, geometry.heads, geometry.headDimension]),
            positions: positions, table: table)

        // The sliding window: this block's own K, quantized over the whole
        // post-RoPE vector, rotary tail included.
        let windowEntries = try DeepSeekV41FP8Linear.quantizeDequantizeActivation(
            try DeepSeekV41Rotary.applyToTail(
                try DeepSeekV41RMSNorm.apply(
                    try DeepSeekV41FP8Linear.project(
                        hidden, weights: weights.keyValue,
                        phaseAccounting: phaseAccounting, diagnostics: diagnostics),
                    weight: weights.keyValueNorm, epsilon: geometry.normEpsilon),
                positions: positions, table: table),
            phaseAccounting: phaseAccounting, diagnostics: diagnostics)
        var keysAndValues: MLXArray
        if startPosition == 0 {
            caches.window = try seedWindowRing(
                caches.window, with: windowEntries, windowSize: geometry.windowSize)
            // Prefill attends over the chunk itself, not over the ring: the ring
            // is only being seeded for the decode steps that follow.
            keysAndValues = windowEntries
        } else {
            caches.window = replacingRow(
                caches.window, at: startPosition % geometry.windowSize,
                with: windowEntries[0])
            keysAndValues = caches.window
        }
        var indices = try DeepSeekV41WindowIndices.rows(
            windowSize: geometry.windowSize, sequenceLength: tokens,
            startPosition: startPosition)

        if geometry.compressionRatio > 0 {
            let offset = keysAndValues.shape[0]
            let ratio = geometry.compressionRatio
            let compressedLength = (startPosition + tokens) / ratio

            var latent: MLXArray?
            if geometry.mode == .full {
                latent = try compressorLatent(
                    hidden: hidden, geometry: geometry, weights: weights,
                    caches: &caches, startPosition: startPosition)
            }

            let compressedIndices: [[Int]]
            if geometry.mode == .full || geometry.mode == .reindex {
                compressedIndices = try index(
                    hidden: hidden, queryLatent: queryLatent, latent: latent,
                    geometry: geometry, weights: weights, caches: &caches,
                    shared: shared, table: table, startPosition: startPosition,
                    tokens: tokens, offset: offset, compressedLength: compressedLength,
                    phaseAccounting: phaseAccounting, diagnostics: diagnostics)
                shared.topKIndices = compressedIndices
            } else {
                guard let published = shared.topKIndices else {
                    throw DeepSeekV41Error.configuration(
                        "block \(geometry.block) reuses a Top-K nobody published")
                }
                compressedIndices = published
            }

            if let latent {
                // A latent stands for the first token of its group, so group j
                // takes position j * ratio. Rotated *after* the indexer read it.
                let latentPositions = try compressedPositions(
                    count: latent.shape[0], ratio: ratio, startPosition: startPosition,
                    tokens: tokens)
                let rotated = try DeepSeekV41Rotary.applyToTail(
                    latent, positions: latentPositions, table: table)
                // Compressed KV uses groups of 16 with E4M3 scales; the indexer
                // uses 32 with E8M0. Two different rules, one function upstream.
                let quantized =
                    try DeepSeekV41FP4Activation.compressedKeyValueQuantizeDequantize(rotated)
                caches.compressed = append(caches.compressed, quantized)
            }
            // `shared_attn.compress_kv = self.compress_kv_cache` is set by every
            // KV source **unconditionally**, not only when it produced a latent:
            // the reference assigns the cache *object* before the indexer runs
            // and reads a slice of it afterwards. So a ratio-2 source whose
            // group is still filling still publishes, and the blocks after it
            // read its cache rather than whatever the previous *token* left
            // behind. Getting this wrong is invisible in prefill and in every
            // decode step where the group happens to complete.
            //
            // `index_k` is the opposite and deliberately so: the reference
            // assigns it *inside* `if self.owns_k and latent is not None`, so on
            // a step where no group completes, an index-key owner scores against
            // the keys the last owner that did publish left there. That is the
            // reference's behaviour, quirk and all, and it is reproduced above
            // by publishing only alongside a latent.
            if geometry.mode == .full, caches.compressed != nil {
                shared.compressedKeyValues = caches.compressed
            }
            guard let publishedCompressed = shared.compressedKeyValues,
                publishedCompressed.shape[0] >= compressedLength
            else {
                throw DeepSeekV41Error.configuration(
                    "block \(geometry.block) needs \(compressedLength) compressed rows and "
                        + "nobody published them")
            }
            if compressedLength > 0 {
                keysAndValues = concatenated(
                    [keysAndValues, publishedCompressed[0..<compressedLength, 0...]], axis: 0)
            }
            guard compressedIndices.count == tokens else {
                throw DeepSeekV41Error.configuration(
                    "the compressed index list has \(compressedIndices.count) rows against "
                        + "\(tokens) tokens")
            }
            indices = zip(indices, compressedIndices).map { $0 + $1 }
        }

        let attended = try DeepSeekV41SparseAttention.forward(
            queries: queries, keysAndValues: keysAndValues, sinks: weights.sink,
            indices: indices, scale: geometry.softmaxScale,
            phaseAccounting: phaseAccounting)
        // `sparse_attn` returns in the query's dtype, and the inverse rotation
        // then happens in that dtype. Both are bfloat16.
        let unrotated = try DeepSeekV41Rotary.applyToTail(
            attended.asType(.bfloat16), positions: positions, table: table, inverse: true)
        let output = try DeepSeekV41GroupedOutput.project(
            unrotated, groupWeight: weights.outputDown,
            outputWeight: weights.outputUp, groups: geometry.outputGroups,
            phaseAccounting: phaseAccounting, diagnostics: diagnostics)
        return Result(output: output, caches: caches)
    }

    // MARK: - Pieces

    private static func compressorLatent(
        hidden: MLXArray,
        geometry: DeepSeekV41AttentionGeometry,
        weights: DeepSeekV41AttentionWeights,
        caches: inout DeepSeekV41AttentionCaches,
        startPosition: Int
    ) throws -> MLXArray? {
        guard let keyValue = weights.compressorKeyValue,
            let norm = weights.compressorNorm
        else {
            throw DeepSeekV41Error.configuration(
                "block \(geometry.block) is a KV source with no compressor")
        }
        let result = try DeepSeekV41Compressor.forward(
            hidden: hidden,
            ratio: geometry.compressionRatio,
            keyValueWeight: keyValue,
            gateWeight: weights.compressorGate,
            normWeight: norm,
            normEpsilon: geometry.normEpsilon,
            startPosition: startPosition,
            state: caches.compressorState)
        if geometry.compressionRatio > 1 { caches.compressorState = result.state }
        return result.latent
    }

    private static func index(
        hidden: MLXArray,
        queryLatent: MLXArray,
        latent: MLXArray?,
        geometry: DeepSeekV41AttentionGeometry,
        weights: DeepSeekV41AttentionWeights,
        caches: inout DeepSeekV41AttentionCaches,
        shared: DeepSeekV41SharedAttentionRuntime,
        table: DeepSeekV41RotaryTable,
        startPosition: Int,
        tokens: Int,
        offset: Int,
        compressedLength: Int,
        phaseAccounting: DeepSeekV4PhaseAccounting?,
        diagnostics: DeepSeekV4Diagnostics
    ) throws -> [[Int]] {
        // Nothing compressed yet: the reference builds an empty index list
        // rather than running an indexer over zero keys.
        guard compressedLength > 0 else {
            return [[Int]](repeating: [], count: tokens)
        }
        guard let queryUp = weights.indexerQueryUp,
            let weightsProjection = weights.indexerWeights
        else {
            throw DeepSeekV41Error.configuration(
                "block \(geometry.block) indexes with no indexer weights")
        }
        let ratio = geometry.compressionRatio

        // An index-key owner turns its latent into keys here, before Attention
        // overwrites that same value with the RoPE'd, quantized one.
        if geometry.mode == .full, let latent {
            guard let keyWeight = weights.indexerKey, let keyNorm = weights.indexerKeyNorm
            else {
                throw DeepSeekV41Error.configuration(
                    "block \(geometry.block) owns index keys with no wk / k_norm")
            }
            let latentPositions = try compressedPositions(
                count: latent.shape[0], ratio: ratio, startPosition: startPosition,
                tokens: tokens)
            let keys = try DeepSeekV41Indexer.keys(
                latent: latent, weight: keyWeight, normWeight: keyNorm,
                normEpsilon: geometry.normEpsilon, positions: latentPositions,
                table: table, diagnostics: diagnostics)
            caches.indexKeys = append(caches.indexKeys, keys)
            shared.indexKeys = caches.indexKeys
        }
        guard let publishedKeys = shared.indexKeys,
            publishedKeys.shape[0] >= compressedLength
        else {
            throw DeepSeekV41Error.configuration(
                "block \(geometry.block) needs \(compressedLength) index keys and nobody "
                    + "published them")
        }

        let positions = Array(startPosition..<(startPosition + tokens))
        let queries = try DeepSeekV41Indexer.queries(
            queryLatent: queryLatent, weight: queryUp, heads: geometry.indexHeads,
            headDimension: geometry.indexHeadDimension, positions: positions,
            table: table, phaseAccounting: phaseAccounting, diagnostics: diagnostics)
        let scores = try DeepSeekV41Indexer.scores(
            hidden: hidden, queries: queries,
            indexKeys: publishedKeys[0..<compressedLength, 0...],
            weightsProjection: weightsProjection,
            headDimension: geometry.indexHeadDimension)
        let reachable = try DeepSeekV41Indexer.reachableCounts(
            ratio: ratio, tokens: tokens, startPosition: startPosition)
        let rows = try DeepSeekV41Indexer.scoreRows(
            scores, reachableCounts: reachable, phaseAccounting: phaseAccounting)

        var mask: [[Bool]]?
        if geometry.isCandidateSource {
            shared.candidateMask = try DeepSeekV41Indexer.candidateBlocks(
                scoreRows: rows, reachableCounts: reachable,
                topKBlocks: geometry.candidateTopKBlocks,
                blockSize: geometry.candidateBlockSize)
        } else if geometry.usesCandidates {
            guard let published = shared.candidateMask else {
                throw DeepSeekV41Error.configuration(
                    "block \(geometry.block) is after the candidate source and no pool was "
                        + "published")
            }
            mask = published
        }
        shared.lastIndexScoreRows = DeepSeekV41Indexer.masked(rows, by: mask)
        return try DeepSeekV41Indexer.select(
            scoreRows: rows, reachableCounts: reachable, candidateMask: mask,
            topK: Swift.min(geometry.indexTopK, compressedLength), offset: offset)
    }

    /// A latent stands for the first token of its group: group *j* takes
    /// position `j * ratio`. Prefill produces one per completed group; a decode
    /// step that completes a group produces one, at `start_pos + 1 - ratio`.
    static func compressedPositions(
        count: Int, ratio: Int, startPosition: Int, tokens: Int
    ) throws -> [Int] {
        guard count > 0, ratio > 0 else {
            throw DeepSeekV41Error.configuration(
                "compressed positions need a positive count and ratio")
        }
        if startPosition == 0 {
            let groups = (tokens - tokens % ratio) / ratio
            guard count == groups else {
                throw DeepSeekV41Error.configuration(
                    "\(count) latents against \(groups) completed groups of \(ratio)")
            }
            return (0..<groups).map { $0 * ratio }
        }
        guard count == 1 else {
            throw DeepSeekV41Error.configuration(
                "a decode step completes at most one group, got \(count) latents")
        }
        return [startPosition + 1 - ratio]
    }

    /// Seed the ring from a prefill chunk, exactly as the reference splits it.
    ///
    /// A chunk no longer than the window lands at slots `0..<seqlen`. A longer
    /// one keeps only its last `window` tokens, and they land where their own
    /// positions put them: `cutoff = seqlen % window` is where the newest token
    /// wraps to, so the last `cutoff` tokens go to `0..<cutoff` and the ones
    /// before them to `cutoff..<window`.
    static func seedWindowRing(
        _ ring: MLXArray, with entries: MLXArray, windowSize: Int
    ) throws -> MLXArray {
        let tokens = entries.shape[0]
        guard windowSize > 0, ring.shape[0] == windowSize else {
            throw DeepSeekV41Error.configuration(
                "the window ring is \(ring.shape) against a window of \(windowSize)")
        }
        if tokens <= windowSize {
            if tokens == windowSize { return entries }
            return concatenated([entries, ring[tokens...]], axis: 0)
        }
        let cutoff = tokens % windowSize
        let last = entries[(tokens - windowSize)..., 0...]
        if cutoff == 0 { return last }
        return concatenated(
            [last[(windowSize - cutoff)..., 0...], last[0..<(windowSize - cutoff), 0...]],
            axis: 0)
    }

    private static func replacingRow(
        _ table: MLXArray, at row: Int, with values: MLXArray
    ) -> MLXArray {
        var rows = [MLXArray]()
        rows.reserveCapacity(table.shape[0])
        for index in 0..<table.shape[0] {
            rows.append(index == row ? values.asType(table.dtype) : table[index])
        }
        return stacked(rows, axis: 0)
    }

    private static func append(_ table: MLXArray?, _ rows: MLXArray) -> MLXArray {
        guard let table else { return rows }
        return concatenated([table, rows], axis: 0)
    }
}
