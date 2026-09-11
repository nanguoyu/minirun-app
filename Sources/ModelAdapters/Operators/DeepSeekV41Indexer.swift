import Foundation
import MLX
import MLXBridge

/// `Indexer` and `select_candidate_blocks`: which compressed positions a query
/// attends to.
///
/// A small side attention. FP4 query heads against **one** shared key per
/// compressed position, the scores rectified and then combined across heads by a
/// learned `weights_proj` — not a softmax, and not a max: a rectified weighted
/// sum, which is why a negative score contributes exactly nothing rather than a
/// little.
///
/// ## Which blocks run which part
///
/// - **Full** (`kv_source_layers`): derives the index keys from its own
///   compressor latent — before `Attention` overwrites that storage with the
///   RoPE'd, quantized value — publishes them, runs the query side, and selects.
/// - **Reindex** (`index_source_layers` minus the KV sources): reads the keys
///   the source published, runs *its own* query and `weights_proj`, and selects
///   its own Top-K.
/// - **Reuse**: no indexer at all. It reads the last published Top-K.
///
/// ## The two-level pool
///
/// Block 20 — the first decoder KV source, and the only
/// `candidate_source_layer` — scores every compressed position and then keeps
/// the `candidate_topk_blocks` (2048) highest-scoring blocks of
/// `candidate_block_size` (8), by each block's *best* position. Every later
/// decoder Reindex layer scores with its own weights but only inside that mask.
///
/// At our prompt lengths the pool is the whole context and the mask changes
/// nothing — 16,384 candidate positions against at most a few hundred — so it
/// is implemented for parity rather than for the saving. Two details of it are
/// still load-bearing at any length:
///
/// - the block holding the query's newest position is **pinned in** with `+inf`,
///   because it is partly filled and would otherwise lose to an older full one;
/// - a block whose best score is `-inf` is unreachable and is dropped even when
///   there are fewer reachable blocks than `topk_blocks`, which is what
///   `top.values > -inf` in the scatter is doing.
///
/// ## Why the selection is on the host
///
/// The Top-K result is a gather list, and the gather it feeds is expressed with
/// host indices (``DeepSeekV41SparseAttention``). Pulling the `[tokens, keys]`
/// score matrix once and finishing on the host is one round trip per indexer
/// call; selecting on the device and then pulling the indices is also one, with
/// a less inspectable tie rule. V4 made the same choice for the same reason.
///
/// **Ties.** `torch.topk(..., sorted=False)` defines the selected *set* and not
/// its order, and its behaviour on equal scores is unspecified. Measured, it
/// returns the lowest indices on an all-equal row, so that is the rule here:
/// descending score, ascending position on a tie. It matters because a query
/// whose reachable or in-pool positions number fewer than `index_topk` selects
/// `-inf` entries to fill the list, and which ones it picks decides what the
/// attention reads. The fixture is what checks it.
public enum DeepSeekV41Indexer {
    /// The index keys one Full block derives from its compressor latent.
    ///
    /// `k_norm(wk(latent))`, the rotary tail at the group's first position, and
    /// then the FP4/E8M0 round trip over groups of 32. The result is what goes
    /// into the shared key cache; nothing downstream sees the unquantized form.
    public static func keys(
        latent: MLXArray,
        weight: MLXArray,
        normWeight: MLXArray,
        normEpsilon: Float,
        positions: [Int],
        table: DeepSeekV41RotaryTable,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        guard latent.ndim == 2, latent.shape[0] == positions.count else {
            throw DeepSeekV41Error.configuration(
                "index keys need [latents, headDimension] and one position per latent; got "
                    + "\(latent.shape) and \(positions.count) positions")
        }
        let projected = k3Linear(
            latent.asType(.float32), weight.asType(.float32)).asType(latent.dtype)
        let normalized = try DeepSeekV41RMSNorm.apply(
            projected, weight: normWeight, epsilon: normEpsilon)
        let rotated = try DeepSeekV41Rotary.applyToTail(
            normalized, positions: positions, table: table)
        return try DeepSeekV41FP4Activation.indexerQuantizeDequantize(
            rotated, diagnostics: diagnostics)
    }

    /// The query heads: `wq_b(qr)`, split into heads, rotary tail, FP4.
    public static func queries(
        queryLatent: MLXArray,
        weight: BlockFP8Weights,
        heads: Int,
        headDimension: Int,
        positions: [Int],
        table: DeepSeekV41RotaryTable,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        guard queryLatent.ndim == 2, queryLatent.shape[0] == positions.count,
            heads > 0, headDimension > 0
        else {
            throw DeepSeekV41Error.configuration(
                "index queries need [tokens, q_lora_rank] and one position per token; got "
                    + "\(queryLatent.shape) and \(positions.count) positions")
        }
        let projected = try DeepSeekV41FP8Linear.project(
            queryLatent, weights: weight,
            phaseAccounting: phaseAccounting, diagnostics: diagnostics)
        let split = projected.reshaped([positions.count, heads, headDimension])
        let rotated = try DeepSeekV41Rotary.applyToTail(
            split, positions: positions, table: table)
        return try DeepSeekV41FP4Activation.indexerQuantizeDequantize(
            rotated, diagnostics: diagnostics)
    }

    /// `[tokens, keys]` index scores, before any masking.
    ///
    /// ```python
    /// weights = weights_proj(x) * (index_head_dim**-0.5 * index_n_heads**-0.5)
    /// score = einsum("bshd,btd->bsht", q, index_k)
    /// score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    /// ```
    ///
    /// Both the einsum and the head sum are bfloat16 in the reference — the
    /// operands are bfloat16 and torch rounds the result — so the float32
    /// reductions below are cast back at exactly those two points.
    public static func scores(
        hidden: MLXArray,
        queries: MLXArray,
        indexKeys: MLXArray,
        weightsProjection: MLXArray,
        headDimension: Int
    ) throws -> MLXArray {
        guard queries.ndim == 3, indexKeys.ndim == 2,
            indexKeys.shape[1] == queries.shape[2],
            hidden.ndim == 2, hidden.shape[0] == queries.shape[0],
            weightsProjection.ndim == 2,
            weightsProjection.shape[0] == queries.shape[1],
            weightsProjection.shape[1] == hidden.shape[1],
            headDimension > 0
        else {
            throw DeepSeekV41Error.configuration(
                "index scoring shapes do not agree: hidden \(hidden.shape), queries "
                    + "\(queries.shape), keys \(indexKeys.shape), weights_proj "
                    + "\(weightsProjection.shape)")
        }
        let heads = queries.shape[1]
        let scale =
            Float(1 / Foundation.sqrt(Double(headDimension)))
            * Float(1 / Foundation.sqrt(Double(heads)))
        // Every `.asType(.bfloat16)` below is a rounding the reference takes,
        // in the order it takes it. `weights_proj(x)` is a bfloat16 GEMM, so it
        // rounds; the scalar multiply that follows rounds again; the einsum
        // rounds; the head-weighted product rounds; the sum over heads rounds.
        // Five roundings, not one — collapsing them into a single float32 chain
        // would be a different, and slightly better, model.
        let headWeights = k3Linear(
            hidden.asType(.float32), weightsProjection.asType(.float32))
            .asType(.bfloat16)
        let scaled = (headWeights.asType(.float32) * scale).asType(.bfloat16)
        // [tokens, heads, keys]
        let raw = einsum(
            "thd,kd->thk", queries.asType(.float32), indexKeys.asType(.float32))
            .asType(.bfloat16)
        let rectified = maximum(raw.asType(.float32), MLXArray(0 as Float))
        let weighted =
            (rectified * expandedDimensions(scaled.asType(.float32), axis: -1))
            .asType(.bfloat16)
        return sum(weighted.asType(.float32), axis: 1).asType(.bfloat16)
    }

    /// How many compressed positions each query can see.
    ///
    /// A compressed block becomes visible once the query has passed its last
    /// token, which is `(position + 1) / ratio` counted from the start of the
    /// sequence. One query per decode step, so there it is a single number.
    public static func reachableCounts(
        ratio: Int, tokens: Int, startPosition: Int
    ) throws -> [Int] {
        guard ratio > 0, tokens > 0, startPosition >= 0 else {
            throw DeepSeekV41Error.configuration(
                "reachable counts need a positive ratio and token count; got \(ratio), "
                    + "\(tokens), \(startPosition)")
        }
        if startPosition == 0 {
            return (1...tokens).map { $0 / ratio }
        }
        guard tokens == 1 else {
            throw DeepSeekV41Error.configuration(
                "a decode step indexes exactly one token, got \(tokens)")
        }
        return [(startPosition + tokens) / ratio]
    }

    /// The score matrix pulled to the host with the reachability mask applied.
    ///
    /// `-inf` for a position the query cannot see yet, which is what makes a
    /// block's `-inf` maximum mean "not reachable" to
    /// ``candidateBlocks(scoreRows:reachableCounts:topKBlocks:blockSize:)``.
    public static func scoreRows(
        _ scores: MLXArray,
        reachableCounts: [Int],
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil
    ) throws -> [[Float]] {
        guard scores.ndim == 2, scores.shape[0] == reachableCounts.count else {
            throw DeepSeekV41Error.configuration(
                "index scores \(scores.shape) do not match \(reachableCounts.count) queries")
        }
        let keys = scores.shape[1]
        let flat = measuringPhase(
            phaseAccounting,
            excludingGPUBoundaryFrom: phaseAccounting?.recordLightningIndexer(nanoseconds:)
        ) {
            waitingForGPU(phaseAccounting, .indexer) {
                scores.asType(.float32).asArray(Float.self)
            }
        }
        return reachableCounts.enumerated().map { token, reachable in
            (0..<keys).map { key in
                key >= reachable ? -Float.infinity : flat[token * keys + key]
            }
        }
    }

    /// `select_candidate_blocks`: level one of the two-level Top-K.
    ///
    /// Returns one boolean per compressed position per query, so the layers
    /// consuming it just mask and never think about blocks again.
    public static func candidateBlocks(
        scoreRows: [[Float]],
        reachableCounts: [Int],
        topKBlocks: Int,
        blockSize: Int
    ) throws -> [[Bool]] {
        guard topKBlocks > 0, blockSize > 0, scoreRows.count == reachableCounts.count else {
            throw DeepSeekV41Error.configuration(
                "candidate selection needs positive block parameters and one reachable "
                    + "count per query; got \(topKBlocks), \(blockSize), "
                    + "\(scoreRows.count) rows and \(reachableCounts.count) counts")
        }
        return zip(scoreRows, reachableCounts).map { row, reachable in
            let width = row.count
            let blocks = (width + blockSize - 1) / blockSize
            guard blocks > 0 else { return [Bool](repeating: false, count: width) }
            // Each block scores as its best position; the padding past `width`
            // is `-inf`, so a ragged last block scores as its real entries.
            var blockScores = [Float](repeating: -.infinity, count: blocks)
            for (position, score) in row.enumerated() {
                let block = position / blockSize
                blockScores[block] = Swift.max(blockScores[block], score)
            }
            // The block holding this query's newest position is pinned in.
            let newest = (reachable - 1) / blockSize
            if newest >= 0, newest < blocks { blockScores[newest] = .infinity }

            let keep = topBlocks(blockScores, count: Swift.min(topKBlocks, blocks))
            var mask = [Bool](repeating: false, count: width)
            for block in keep where blockScores[block] > -.infinity {
                for position in (block * blockSize)..<Swift.min((block + 1) * blockSize, width) {
                    mask[position] = true
                }
            }
            return mask
        }
    }

    /// Level two, and the whole of a Reuse-free block's answer: the `topk`
    /// highest-scoring positions per query, in position order, offset into the
    /// concatenated KV, with `-1` for a position the query cannot reach.
    ///
    /// `topk` is `min(index_topk, end_pos / ratio)` — a **sequence-wide** bound,
    /// not a per-query one, so an early prefill query does select entries it
    /// cannot reach and they come back as `-1`. That is the reference's
    /// behaviour and not a rounding of it.
    public static func select(
        scoreRows: [[Float]],
        reachableCounts: [Int],
        candidateMask: [[Bool]]?,
        topK: Int,
        offset: Int
    ) throws -> [[Int]] {
        guard topK >= 0, offset >= 0, scoreRows.count == reachableCounts.count else {
            throw DeepSeekV41Error.configuration(
                "index selection needs a nonnegative top-k and offset and one reachable "
                    + "count per query")
        }
        if let candidateMask {
            guard candidateMask.count == scoreRows.count else {
                throw DeepSeekV41Error.configuration(
                    "the candidate mask has \(candidateMask.count) rows against "
                        + "\(scoreRows.count) queries")
            }
        }
        return try scoreRows.indices.map { token in
            var row = scoreRows[token]
            if let candidateMask {
                guard candidateMask[token].count == row.count else {
                    throw DeepSeekV41Error.configuration(
                        "candidate mask row \(token) is \(candidateMask[token].count) wide "
                            + "against \(row.count) positions")
                }
                for position in row.indices where !candidateMask[token][position] {
                    row[position] = -.infinity
                }
            }
            let selected = topBlocks(row, count: Swift.min(topK, row.count)).sorted()
            let reachable = reachableCounts[token]
            return try selected.map { position in
                guard position < reachable else { return -1 }
                let index = offset.addingReportingOverflow(position)
                guard !index.overflow else {
                    throw DeepSeekV41Error.configuration("compressed cache index overflows Int")
                }
                return index.partialValue
            }
        }
    }

    /// The candidate mask applied, or the rows unchanged when there is none.
    ///
    /// This is the matrix `torch.topk` is handed in the reference, and it is
    /// what ``DeepSeekV41SharedAttentionRuntime/lastIndexScoreRows`` carries so
    /// a test can separate a scoring disagreement from a selection one.
    public static func masked(_ rows: [[Float]], by mask: [[Bool]]?) -> [[Float]] {
        guard let mask else { return rows }
        return rows.indices.map { token in
            rows[token].indices.map { position in
                mask[token][position] ? rows[token][position] : -.infinity
            }
        }
    }

    /// Descending score, ascending index on a tie. See the type's note on ties.
    ///
    /// `-inf` sorts below everything and still gets selected when there is
    /// nothing else to take, which is exactly what `torch.topk` does and what
    /// the `-1` mapping above then depends on.
    static func topBlocks(_ scores: [Float], count: Int) -> [Int] {
        guard count > 0 else { return [] }
        let ordered = scores.indices.sorted { left, right in
            let a = scores[left]
            let b = scores[right]
            if a == b { return left < right }
            // NaN cannot arise here — the scores are a rectified weighted sum of
            // finite values, masked with -inf — but an ordering predicate has to
            // be total, so an unordered pair falls back to the index.
            if a.isNaN { return false }
            if b.isNaN { return true }
            return a > b
        }
        return Array(ordered.prefix(count))
    }
}
