import Foundation
import MLX
import MLXBridge

/// V4.1's rotary frequency table: `precompute_freqs_cis`, with YaRN.
///
/// ## Why this is not ``DeepSeekV4RotaryParameters``
///
/// The equations agree. The *arithmetic* does not, and the difference is
/// visible: the reference builds the whole table in float32 —
/// `1.0 / (base ** (arange(0, dim, 2, float32) / dim))`, the YaRN ramp in
/// float32, then `torch.outer(arange(seqlen), freqs)` in float32 — while V4's
/// Swift table carries float64 frequencies and multiplies by a float64 position.
/// Rounding the frequency to float32 *before* multiplying by the position is a
/// different number, and at position 65,536 (V4.1's `original_seq_len`) it is a
/// different angle by more than one ulp.
///
/// So this table rounds where the reference rounds. It is a separate type
/// rather than a flag on V4's because "which precision the table is built in" is
/// a property of a published model, not a caller's preference.
///
/// ## Which parameters a block uses
///
/// `Attention.__init__` chooses per block: a block with `compress_ratio > 0`
/// gets YaRN over `original_seq_len` at `compress_rope_theta`; a pure
/// sliding-window block (`compress_ratio == 0`) gets **no** YaRN at all —
/// `original_seq_len` 0 — at the plain `rope_theta`. Both are 10,000 in the
/// published configuration, so only the YaRN half actually differs, but the
/// choice is stated here rather than assumed.
public struct DeepSeekV41RotaryTable: Sendable {
    public let dimension: Int
    /// `[positions, dimension / 2]`, row-major.
    public let cosines: [Float]
    public let sines: [Float]
    public let positionCount: Int

    /// The per-pair frequencies, before the position multiply. Exposed because
    /// a test that disagrees about the table should say so before it disagrees
    /// about a rotation.
    public let frequencies: [Float]

    public init(
        dimension: Int,
        positionCount: Int,
        originalSequenceLength: Int,
        base: Float,
        factor: Float,
        betaFast: Float,
        betaSlow: Float
    ) throws {
        guard dimension > 0, dimension.isMultiple(of: 2), positionCount > 0 else {
            throw DeepSeekV41Error.configuration(
                "rotary dimension must be positive and even and the position count positive; "
                    + "got \(dimension) and \(positionCount)")
        }
        guard base.isFinite, base > 1, factor.isFinite, factor > 0,
            betaFast.isFinite, betaFast > 0, betaSlow.isFinite, betaSlow > 0,
            originalSequenceLength >= 0
        else {
            throw DeepSeekV41Error.configuration(
                "rotary base, factor and correction rotations must be finite and positive")
        }
        let half = dimension / 2
        var frequencies = [Float](repeating: 0, count: half)
        for pair in 0..<half {
            // `base ** (arange(0, dim, 2, float32) / dim)`, then the reciprocal.
            let exponent = Float(pair * 2) / Float(dimension)
            frequencies[pair] = 1 / Float(Foundation.pow(Double(base), Double(exponent)))
        }
        if originalSequenceLength > 0 {
            func correctedDimension(_ rotations: Float) -> Double {
                Double(dimension)
                    * Foundation.log(
                        Double(originalSequenceLength) / (Double(rotations) * 2 * .pi))
                    / (2 * Foundation.log(Double(base)))
            }
            let low = max(Int(Foundation.floor(correctedDimension(betaFast))), 0)
            let high = min(Int(Foundation.ceil(correctedDimension(betaSlow))), dimension - 1)
            // `max(high - low, 1e-3)`, exactly: a degenerate band divides by the
            // floor rather than by zero, and a *negative* one divides by the
            // floor too, which `high < low` would otherwise sign-flip.
            let denominator = Swift.max(Float(high - low), 1e-3)
            for pair in 0..<half {
                let ramp = Swift.min(Swift.max((Float(pair) - Float(low)) / denominator, 0), 1)
                let smooth = 1 - ramp
                frequencies[pair] =
                    frequencies[pair] / factor * (1 - smooth) + frequencies[pair] * smooth
            }
        }
        guard frequencies.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw DeepSeekV41Error.configuration(
                "rotary frequency table is not finite and positive")
        }

        var cosines = [Float](repeating: 0, count: positionCount * half)
        var sines = [Float](repeating: 0, count: positionCount * half)
        for position in 0..<positionCount {
            for pair in 0..<half {
                // `torch.outer(arange(seqlen), freqs)` is a float32 product, so
                // the angle is rounded to float32 before the cosine is taken.
                let angle = Float(position) * frequencies[pair]
                cosines[position * half + pair] = Float(Foundation.cos(Double(angle)))
                sines[position * half + pair] = Float(Foundation.sin(Double(angle)))
            }
        }
        self.dimension = dimension
        self.positionCount = positionCount
        self.frequencies = frequencies
        self.cosines = cosines
        self.sines = sines
    }

    /// The table one block uses, per `Attention.__init__`.
    public static func forBlock(
        _ block: Int, config: DeepSeekV41Config, positionCount: Int
    ) throws -> DeepSeekV41RotaryTable {
        let compresses = try config.compressionRatio(block: block) != 0
        return try DeepSeekV41RotaryTable(
            dimension: config.ropeHeadDimension,
            positionCount: positionCount,
            originalSequenceLength: compresses ? config.ropeOriginalLength : 0,
            base: compresses ? config.compressionRopeTheta : config.ropeTheta,
            factor: config.ropeFactor,
            betaFast: Float(config.ropeBetaFast),
            betaSlow: Float(config.ropeBetaSlow))
    }
}

/// `apply_rotary_emb`: adjacent element pairs as complex numbers.
///
/// The accepted layout is `[tokens, ..., rotaryDimension]` — batch one, which
/// is the product's chat contract. Callers rotate only the trailing
/// `rope_head_dim` channels, by slicing before the call, exactly as
/// `q[..., -rd:]` does upstream.
///
/// `inverse` conjugates the rotation. That is not a convenience: it is how the
/// attention *output* gets the query's rotation taken back off, so the shared
/// KV cache can stay in one rotated form and every query that reads it can undo
/// its own. `Attention.forward` calls it exactly once, on `o[..., -rd:]`, and a
/// runner that skipped it would be reading a cache in the wrong frame.
public enum DeepSeekV41Rotary {
    public static func apply(
        _ input: MLXArray,
        positions: [Int],
        table: DeepSeekV41RotaryTable,
        inverse: Bool = false
    ) throws -> MLXArray {
        guard input.ndim >= 2, input.shape[0] == positions.count,
            input.shape[input.ndim - 1] == table.dimension
        else {
            throw DeepSeekV41Error.configuration(
                "rotary input must be [tokens, ..., \(table.dimension)] with one position per "
                    + "token; got \(input.shape) and \(positions.count) positions")
        }
        guard input.dtype.isFloatingPoint, !input.dtype.isComplex else {
            throw DeepSeekV41Error.configuration(
                "rotary input must have a real floating-point dtype, got \(input.dtype)")
        }
        let half = table.dimension / 2
        var cosine = [Float]()
        var sine = [Float]()
        cosine.reserveCapacity(positions.count * half)
        sine.reserveCapacity(positions.count * half)
        for position in positions {
            guard position >= 0, position < table.positionCount else {
                throw DeepSeekV41Error.configuration(
                    "rotary position \(position) is outside the \(table.positionCount)-row table")
            }
            let start = position * half
            cosine.append(contentsOf: table.cosines[start..<(start + half)])
            if inverse {
                sine.append(contentsOf: table.sines[start..<(start + half)].map(-))
            } else {
                sine.append(contentsOf: table.sines[start..<(start + half)])
            }
        }

        var frequencyShape = [positions.count]
        frequencyShape.append(contentsOf: repeatElement(1, count: input.ndim - 2))
        frequencyShape.append(half)
        let cosines = MLXArray(cosine, frequencyShape)
        let sines = MLXArray(sine, frequencyShape)
        let even = input[.ellipsis, .stride(from: 0, to: table.dimension, by: 2)]
            .asType(.float32)
        let odd = input[.ellipsis, .stride(from: 1, to: table.dimension, by: 2)]
            .asType(.float32)
        let real = even * cosines - odd * sines
        let imaginary = even * sines + odd * cosines
        // `y.copy_(x)`: the reference writes the float32 result back into the
        // tensor it was handed, so the rotation ends in that tensor's dtype.
        return stacked([real, imaginary], axis: -1)
            .reshaped(input.shape)
            .asType(input.dtype)
    }

    /// Rotate only the trailing `rotaryDimension` channels of a wider vector,
    /// which is every call site upstream.
    public static func applyToTail(
        _ input: MLXArray,
        positions: [Int],
        table: DeepSeekV41RotaryTable,
        inverse: Bool = false
    ) throws -> MLXArray {
        let width = input.shape[input.ndim - 1]
        guard width >= table.dimension else {
            throw DeepSeekV41Error.configuration(
                "cannot rotate the trailing \(table.dimension) channels of a \(width)-wide vector")
        }
        if width == table.dimension {
            return try apply(input, positions: positions, table: table, inverse: inverse)
        }
        let head = input[.ellipsis, 0..<(width - table.dimension)]
        let tail = try apply(
            input[.ellipsis, (width - table.dimension)...],
            positions: positions, table: table, inverse: inverse)
        return concatenated([head, tail], axis: -1)
    }
}

/// `get_window_topk_idxs`: which sliding-window slots a query may attend to.
///
/// The cache is a ring of `windowSize` slots and `-1` marks a slot holding
/// nothing. Prefill produces one row per query, each seeing its own causal
/// window into the *chunk*; a decode step produces one row listing the whole
/// ring, oldest first.
///
/// The order within a row does not matter to the attention kernel, which
/// handles every slot independently — but it is reproduced anyway, because a
/// float32 sum is not associative and a test that compared against a differently
/// ordered gather would be measuring the order rather than the model.
public enum DeepSeekV41WindowIndices {
    public static func rows(
        windowSize: Int, sequenceLength: Int, startPosition: Int
    ) throws -> [[Int]] {
        guard windowSize > 0, sequenceLength > 0, startPosition >= 0 else {
            throw DeepSeekV41Error.configuration(
                "window size and sequence length must be positive and the start position "
                    + "nonnegative; got \(windowSize), \(sequenceLength), \(startPosition)")
        }
        if startPosition == 0 {
            let width = Swift.min(sequenceLength, windowSize)
            return (0..<sequenceLength).map { end in
                let first = Swift.max(end - windowSize + 1, 0)
                return (0..<width).map { column in
                    let candidate = first + column
                    return candidate > end ? -1 : candidate
                }
            }
        }
        guard sequenceLength == 1 else {
            throw DeepSeekV41Error.configuration(
                "incremental window indexing accepts exactly one token, got \(sequenceLength)")
        }
        let oldest = startPosition % windowSize + 1
        let ring = Array(oldest..<windowSize) + Array(0..<oldest)
        return [ring.map { $0 > startPosition ? -1 : $0 }]
    }
}

/// `sparse_attn` from `inference/kernel.py`, with the learned sink.
///
/// Two details a plain softmax gets wrong, and both are in the kernel rather
/// than in the paper:
///
/// - **The running maximum starts at a finite `-1e30`, and the sink is not in
///   it.** `T.fill(scores_max, -1e30)` then `reduce_max(..., clear=False)` over
///   the score tiles; the sink is added to the denominator afterwards as
///   `exp(attn_sink[h] - scores_max[h])`. So the sink can be *larger* than every
///   score and the exponential is taken anyway — it is not folded into the
///   maximum the way a numerically-defensive implementation would fold it. V4's
///   ``DeepSeekV4SparseAttention`` does fold it in; V4.1's kernel does not, and
///   this is the difference.
/// - **A query whose whole index row is `-1` produces exactly zero**, not NaN:
///   with the finite bound every `exp(-inf - (-1e30))` is 0, the sink term makes
///   the denominator `+inf`, and `0 / inf` is 0. The kernel's own comment says
///   this matches the training kernel's convention.
///
/// The sink has no value vector: it absorbs probability mass and contributes
/// nothing to the numerator.
///
/// ## Batched heads
///
/// The head loop is expressed as a batch axis, which is ADR 0018's decision for
/// V4 and is taken here from the start rather than switched on later. There is
/// no per-head form to fall back to and therefore no environment switch: a
/// V4.1 digest has never been recorded against one, so there is nothing for a
/// control arm to reproduce.
public enum DeepSeekV41SparseAttention {
    /// The kernel's finite lower bound. Not `-Float.infinity`.
    static let runningMaximumFloor: Float = -1e30

    /// - Parameters:
    ///   - queries: `[tokens, heads, dimension]`.
    ///   - keysAndValues: `[rows, dimension]` — the window ring and the
    ///     compressed cache already concatenated, which is the single
    ///     `sparse_attn` call the reference makes.
    ///   - sinks: `[heads]`, float32.
    ///   - indices: one row per token, `-1` for an empty slot.
    public static func forward(
        queries: MLXArray,
        keysAndValues: MLXArray,
        sinks: MLXArray,
        indices: [[Int]],
        scale: Float,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil
    ) throws -> MLXArray {
        guard queries.ndim == 3, queries.shape[0] > 0, queries.shape[1] > 0,
            queries.shape[2] > 0,
            keysAndValues.ndim == 2, keysAndValues.shape[0] > 0,
            keysAndValues.shape[1] == queries.shape[2],
            sinks.shape == [queries.shape[1]], indices.count == queries.shape[0],
            scale.isFinite, scale > 0
        else {
            throw DeepSeekV41Error.configuration(
                "sparse attention queries \(queries.shape), KV \(keysAndValues.shape), sinks "
                    + "\(sinks.shape), \(indices.count) index rows or scale \(scale) are invalid")
        }
        return try measuringPhase(
            phaseAccounting,
            excludingGPUBoundaryFrom: phaseAccounting?.recordSparseAttention(nanoseconds:)
        ) {
            let heads = queries.shape[1]
            let dimension = queries.shape[2]
            let sink = sinks.asType(.float32)[0..., .newAxis]
            var tokenOutputs = [MLXArray]()
            tokenOutputs.reserveCapacity(queries.shape[0])
            for token in 0..<queries.shape[0] {
                var seen = Set<Int>()
                let valid = try indices[token].compactMap { index -> Int? in
                    if index == -1 { return nil }
                    guard index >= 0, index < keysAndValues.shape[0] else {
                        throw DeepSeekV41Error.configuration(
                            "token \(token) names KV row \(index), outside the cache")
                    }
                    guard seen.insert(index).inserted else {
                        throw DeepSeekV41Error.configuration(
                            "token \(token) names KV row \(index) more than once")
                    }
                    return index
                }
                guard !valid.isEmpty else {
                    // `0 / inf`. Stated rather than computed, because computing
                    // it would mean forming an empty reduction.
                    tokenOutputs.append(
                        MLXArray.zeros([heads, dimension], dtype: .float32))
                    continue
                }
                let kv = keysAndValues.take(MLXArray(valid), axis: 0).asType(.float32)
                let query = queries[token, 0..., 0...].asType(.float32)
                let scores =
                    sum(
                        kv[.newAxis, 0..., 0...] * query[0..., .newAxis, 0...],
                        axis: -1) * scale
                let maximumScore = maximum(
                    scores.max(axis: -1, keepDims: true),
                    MLXArray(runningMaximumFloor))
                let exponentials = exp(scores - maximumScore)
                let denominator =
                    sum(exponentials, axis: -1, keepDims: true)
                    + exp(sink - maximumScore)
                tokenOutputs.append(
                    sum(
                        kv[.newAxis, 0..., 0...] * exponentials[0..., 0..., .newAxis],
                        axis: 1) / denominator)
            }
            return stacked(tokenOutputs, axis: 0)
        }
    }
}

/// The grouped low-rank output projection, `wo_a` then `wo_b`.
///
/// `wo_a` is block-diagonal over `o_groups`: group *g*'s rows see only group
/// *g*'s heads, which is why the reference writes an `einsum` and not a
/// `Linear`. It is bfloat16 there — DeepSeek's `convert.py` dequantizes it —
/// while our container carries the published FP8 bytes, so the caller passes the
/// dequantized matrix (``DeepSeekV41FP8Linear/dequantizedToBFloat16(_:stream:)``).
/// `wo_b` stays FP8 and goes through the ordinary projection.
public enum DeepSeekV41GroupedOutput {
    public static func project(
        _ attention: MLXArray,
        groupWeight: MLXArray,
        outputWeight: BlockFP8Weights,
        groups: Int,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        guard attention.ndim == 3, attention.shape[0] > 0, attention.shape[1] > 0,
            attention.shape[2] > 0, groups > 0, attention.shape[1].isMultiple(of: groups)
        else {
            throw DeepSeekV41Error.configuration(
                "grouped output needs [tokens, heads, dimension] with heads divisible by "
                    + "\(groups); got \(attention.shape)")
        }
        let perGroup = attention.shape[1] / groups
        let groupInput = perGroup.multipliedReportingOverflow(by: attention.shape[2])
        guard !groupInput.overflow, groupWeight.ndim == 2,
            groupWeight.shape[0] > 0, groupWeight.shape[0].isMultiple(of: groups),
            groupWeight.shape[1] == groupInput.partialValue
        else {
            throw DeepSeekV41Error.configuration(
                "wo_a must be [groups * o_lora_rank, \(groupInput.partialValue)]; got "
                    + "\(groupWeight.shape)")
        }
        let rank = groupWeight.shape[0] / groups
        let flattenedRank = groups.multipliedReportingOverflow(by: rank)
        guard !flattenedRank.overflow,
            outputWeight.inFeatures == flattenedRank.partialValue
        else {
            throw DeepSeekV41Error.configuration(
                "wo_b must take \(flattenedRank.partialValue) inputs; got "
                    + "\(outputWeight.inFeatures)")
        }
        // The heads are flattened and re-split into groups exactly as
        // `o.view(bsz, seqlen, n_local_groups, -1)` does: group g takes the
        // consecutive heads [g * heads/groups, (g+1) * heads/groups).
        let grouped = attention.asType(.float32)
            .reshaped([attention.shape[0], groups, groupInput.partialValue])
        let first = groupWeight.asType(.float32)
            .reshaped([groups, rank, groupInput.partialValue])
        let latent = einsum("tgd,grd->tgr", grouped, first)
            // The reference's einsum is bfloat16 in and bfloat16 out; the
            // float32 accumulation above is the same reduction, rounded once.
            .asType(.bfloat16)
            .reshaped([attention.shape[0], flattenedRank.partialValue])
        return try DeepSeekV41FP8Linear.project(
            latent, weights: outputWeight,
            phaseAccounting: phaseAccounting, diagnostics: diagnostics)
    }
}
