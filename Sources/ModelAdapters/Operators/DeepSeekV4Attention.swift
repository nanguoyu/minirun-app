import Foundation
import MLX

/// YaRN rotary parameters used by DeepSeek V4 attention and both compressors.
///
/// Positions are deliberately supplied by the caller. The main KV stream and
/// the compressed stream use different bases, while compressed entries are
/// rotated at the first position of the group that produced them. Keeping that
/// choice outside this value prevents a cache from silently inventing one.
public struct DeepSeekV4RotaryParameters: Sendable, Hashable {
    public let dimension: Int
    public let originalSequenceLength: Int
    public let base: Double
    public let factor: Double
    public let betaFast: Double
    public let betaSlow: Double

    /// The pair frequencies these parameters imply, including the YaRN ramp.
    ///
    /// Derived state, computed once here rather than per ``DeepSeekV4Rotary``
    /// call: the table depends on nothing but the six values above, while the
    /// call sites re-enter rotary three to five times per layer per token with
    /// the geometry's single run-scoped parameter value. Keeping it in the
    /// value that already lives for the run avoids both a process-global cache
    /// and any question of which call computed it.
    let frequencies: [Double]

    public init(
        dimension: Int,
        originalSequenceLength: Int = 0,
        base: Double,
        factor: Double = 1,
        betaFast: Double = 32,
        betaSlow: Double = 1
    ) throws {
        guard dimension > 0, dimension.isMultiple(of: 2) else {
            throw DeepSeekV4Error.attention(
                "rotary dimension must be positive and even, got \(dimension)")
        }
        guard originalSequenceLength >= 0 else {
            throw DeepSeekV4Error.attention(
                "original rotary sequence length cannot be negative")
        }
        guard base.isFinite, base > 1, factor.isFinite, factor > 0,
            betaFast.isFinite, betaFast > 0, betaSlow.isFinite, betaSlow > 0
        else {
            throw DeepSeekV4Error.attention(
                "rotary base, factor, and correction rotations must be finite and positive")
        }
        self.dimension = dimension
        self.originalSequenceLength = originalSequenceLength
        self.base = base
        self.factor = factor
        self.betaFast = betaFast
        self.betaSlow = betaSlow
        self.frequencies = try Self.baseFrequencies(
            dimension: dimension,
            originalSequenceLength: originalSequenceLength,
            base: base,
            factor: factor,
            betaFast: betaFast,
            betaSlow: betaSlow)
    }

    private static func baseFrequencies(
        dimension: Int,
        originalSequenceLength: Int,
        base: Double,
        factor: Double,
        betaFast: Double,
        betaSlow: Double
    ) throws -> [Double] {
        let half = dimension / 2
        var frequencies = (0..<half).map { pair in
            1 / Foundation.pow(base, Double(pair * 2) / Double(dimension))
        }
        if originalSequenceLength > 0 {
            func correctionDimension(_ rotations: Double) -> Double {
                Double(dimension)
                    * Foundation.log(
                        Double(originalSequenceLength) / (rotations * 2 * .pi))
                    / (2 * Foundation.log(base))
            }
            let low = max(Int(Foundation.floor(correctionDimension(betaFast))), 0)
            let high = min(
                Int(Foundation.ceil(correctionDimension(betaSlow))), dimension - 1)
            let denominator = low == high ? 0.001 : Double(high - low)
            for pair in 0..<half {
                let ramp = min(max((Double(pair) - Double(low)) / denominator, 0), 1)
                let smooth = 1 - ramp
                frequencies[pair] = frequencies[pair] / factor * (1 - smooth)
                    + frequencies[pair] * smooth
            }
        }
        guard frequencies.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw DeepSeekV4Error.attention("rotary frequency table is not finite and positive")
        }
        return frequencies
    }
}

/// DeepSeek V4's complex-pair rotary embedding.
///
/// The accepted layout is `[sequence, ..., rotaryDimension]`; this is the
/// batch-one product layout, not an ambiguous arbitrary-axis helper. Pairing is
/// adjacent (`[0,1]`, `[2,3]`, ...), exactly like PyTorch's
/// `view_as_complex(unflatten(-1, (-1, 2)))` in the official implementation.
public enum DeepSeekV4Rotary {
    public static func apply(
        _ input: MLXArray,
        positions: [Int],
        parameters: DeepSeekV4RotaryParameters,
        inverse: Bool = false
    ) throws -> MLXArray {
        guard input.ndim >= 2, input.shape[0] == positions.count,
            input.shape[input.ndim - 1] == parameters.dimension
        else {
            throw DeepSeekV4Error.attention(
                "rotary input must be [sequence, ..., \(parameters.dimension)] with one "
                    + "position per sequence row; got \(input.shape) and \(positions.count) positions")
        }
        guard input.dtype.isFloatingPoint, !input.dtype.isComplex else {
            throw DeepSeekV4Error.attention(
                "rotary input must have a real floating-point dtype, got \(input.dtype)")
        }
        guard positions.allSatisfy({ $0 >= 0 }) else {
            throw DeepSeekV4Error.attention("rotary positions cannot be negative")
        }

        let frequencies = parameters.frequencies
        let angleCount = positions.count.multipliedReportingOverflow(by: frequencies.count)
        guard !angleCount.overflow else {
            throw DeepSeekV4Error.attention("rotary angle table overflows Int")
        }
        var cosine = [Float]()
        var sine = [Float]()
        cosine.reserveCapacity(angleCount.partialValue)
        sine.reserveCapacity(angleCount.partialValue)
        let direction = inverse ? -1.0 : 1.0
        for position in positions {
            for frequency in frequencies {
                let angle = direction * Double(position) * frequency
                guard angle.isFinite else {
                    throw DeepSeekV4Error.attention("rotary angle is not finite")
                }
                cosine.append(Float(Foundation.cos(angle)))
                sine.append(Float(Foundation.sin(angle)))
            }
        }

        let half = parameters.dimension / 2
        var frequencyShape = [positions.count]
        frequencyShape.append(contentsOf: repeatElement(1, count: input.ndim - 2))
        frequencyShape.append(half)
        let cosines = MLXArray(cosine, frequencyShape)
        let sines = MLXArray(sine, frequencyShape)
        let even = input[.ellipsis, .stride(from: 0, to: parameters.dimension, by: 2)]
            .asType(.float32)
        let odd = input[.ellipsis, .stride(from: 1, to: parameters.dimension, by: 2)]
            .asType(.float32)
        let real = even * cosines - odd * sines
        let imaginary = even * sines + odd * cosines
        return stacked([real, imaginary], axis: -1).reshaped(input.shape).asType(input.dtype)
    }
}

/// Exact logical cache indices used by the official sparse-attention kernel.
/// `-1` is the kernel's invalid-entry sentinel.
public enum DeepSeekV4AttentionIndices {
    public static func window(
        windowSize: Int, sequenceLength: Int, startPosition: Int
    ) throws -> [[Int]] {
        guard windowSize > 0, sequenceLength > 0, startPosition >= 0 else {
            throw DeepSeekV4Error.indexing(
                "window size and sequence length must be positive and start position nonnegative")
        }
        if startPosition > 0 {
            guard sequenceLength == 1 else {
                throw DeepSeekV4Error.indexing(
                    "incremental window indexing accepts exactly one token")
            }
            if startPosition >= windowSize - 1 {
                let cursor = startPosition % windowSize
                return [Array((cursor + 1)..<windowSize) + Array(0...cursor)]
            }
            return [Array(0...startPosition)
                + [Int](repeating: -1, count: windowSize - startPosition - 1)]
        }

        let width = min(sequenceLength, windowSize)
        return (0..<sequenceLength).map { position in
            let first = max(0, position - windowSize + 1)
            return (0..<width).map { column in
                let candidate = first + column
                return candidate <= position ? candidate : -1
            }
        }
    }

    public static func compressed(
        ratio: Int, sequenceLength: Int, startPosition: Int, offset: Int
    ) throws -> [[Int]] {
        guard ratio > 0, sequenceLength > 0, startPosition >= 0, offset >= 0 else {
            throw DeepSeekV4Error.indexing(
                "compression ratio and sequence length must be positive; position and offset "
                    + "must be nonnegative")
        }
        if startPosition > 0 {
            guard sequenceLength == 1 else {
                throw DeepSeekV4Error.indexing(
                    "incremental compressed indexing accepts exactly one token")
            }
            let end = startPosition.addingReportingOverflow(1)
            guard !end.overflow else {
                throw DeepSeekV4Error.indexing("compressed position overflows Int")
            }
            let count = end.partialValue / ratio
            return [try (0..<count).map { try checkedOffset(offset, $0) }]
        }

        let width = sequenceLength / ratio
        return try (0..<sequenceLength).map { position in
            let visible = (position + 1) / ratio
            return try (0..<width).map { key in
                key < visible ? try checkedOffset(offset, key) : -1
            }
        }
    }

    private static func checkedOffset(_ offset: Int, _ index: Int) throws -> Int {
        let result = offset.addingReportingOverflow(index)
        guard !result.overflow else {
            throw DeepSeekV4Error.indexing("cache index overflows Int")
        }
        return result.partialValue
    }
}

/// The learned gated pooling at the heart of DeepSeek V4's KV compressor.
///
/// Inputs are the already-projected `wkv(x)` and `wgate(x)` values. This keeps
/// storage-specific linear weights outside the state machine and makes prompt
/// versus decode equivalence independently testable. Batch size is one, which
/// is the product chat contract.
public enum DeepSeekV4CompressorPool {
    public struct State {
        let ratio: Int
        let overlap: Bool
        let headDimension: Int
        let values: MLXArray
        let scores: MLXArray
    }

    public struct PrefillResult {
        public let compressed: MLXArray
        public let state: State
    }

    public struct DecodeResult {
        public let compressed: MLXArray?
        public let state: State
    }

    public static func prefill(
        values: MLXArray,
        scores: MLXArray,
        positionalBias: MLXArray,
        ratio: Int,
        overlap: Bool
    ) throws -> PrefillResult {
        let geometry = try validate(
            values: values, scores: scores, positionalBias: positionalBias,
            ratio: ratio, overlap: overlap)
        let tokens = values.shape[0]
        let remainder = tokens % ratio
        let cutoff = tokens - remainder
        let groups = cutoff / ratio
        var outputs = [MLXArray]()
        outputs.reserveCapacity(groups)

        if groups > 0 {
            let groupedValues = values[0..<cutoff, 0...]
                .asType(.float32)
                .reshaped([groups, ratio, geometry.projectedDimension])
            let groupedScores = scores[0..<cutoff, 0...]
                .asType(.float32)
                .reshaped([groups, ratio, geometry.projectedDimension])
                + positionalBias.asType(.float32).reshaped(
                    [1, ratio, geometry.projectedDimension])
            for group in 0..<groups {
                let pooledValues: MLXArray
                let pooledScores: MLXArray
                if overlap {
                    let previousValues = group == 0
                        ? MLXArray.zeros([ratio, geometry.headDimension], dtype: .float32)
                        : groupedValues[group - 1, 0..., 0..<geometry.headDimension]
                    let previousScores = group == 0
                        ? MLXArray.full(
                            [ratio, geometry.headDimension],
                            values: MLXArray(-Float.infinity),
                            dtype: .float32)
                        : groupedScores[group - 1, 0..., 0..<geometry.headDimension]
                    pooledValues = concatenated([
                        previousValues,
                        groupedValues[group, 0..., geometry.headDimension...],
                    ], axis: 0)
                    pooledScores = concatenated([
                        previousScores,
                        groupedScores[group, 0..., geometry.headDimension...],
                    ], axis: 0)
                } else {
                    pooledValues = groupedValues[group, 0..., 0...]
                    pooledScores = groupedScores[group, 0..., 0...]
                }
                outputs.append(weightedPool(values: pooledValues, scores: pooledScores))
            }
        }

        let emptyValue = MLXArray.zeros(
            [ratio, geometry.projectedDimension], dtype: .float32)
        let emptyScore = MLXArray.full(
            [ratio, geometry.projectedDimension],
            values: MLXArray(-Float.infinity), dtype: .float32)
        let trailingValues: MLXArray
        let trailingScores: MLXArray
        if remainder > 0 {
            trailingValues = concatenated([
                values[cutoff..., 0...].asType(.float32),
                emptyValue[remainder..., 0...],
            ], axis: 0)
            trailingScores = concatenated([
                scores[cutoff..., 0...].asType(.float32)
                    + positionalBias[0..<remainder, 0...].asType(.float32),
                emptyScore[remainder..., 0...],
            ], axis: 0)
        } else {
            trailingValues = emptyValue
            trailingScores = emptyScore
        }

        let stateValues: MLXArray
        let stateScores: MLXArray
        if overlap {
            if cutoff >= ratio {
                stateValues = concatenated([
                    values[(cutoff - ratio)..<cutoff, 0...].asType(.float32),
                    trailingValues,
                ], axis: 0)
                stateScores = concatenated([
                    scores[(cutoff - ratio)..<cutoff, 0...].asType(.float32)
                        + positionalBias.asType(.float32),
                    trailingScores,
                ], axis: 0)
            } else {
                stateValues = concatenated([emptyValue, trailingValues], axis: 0)
                stateScores = concatenated([emptyScore, trailingScores], axis: 0)
            }
        } else {
            stateValues = trailingValues
            stateScores = trailingScores
        }
        let compressed = outputs.isEmpty
            ? MLXArray.zeros([0, geometry.headDimension], dtype: .float32)
            : stacked(outputs, axis: 0)
        // Compressor state survives the prompt call. Materialise it together
        // with the output so a tiny decode cache cannot retain the full prompt
        // projection graph through MLX's lazy slices.
        MLX.eval([compressed, stateValues, stateScores])
        return PrefillResult(
            compressed: compressed,
            state: State(
                ratio: ratio, overlap: overlap, headDimension: geometry.headDimension,
                values: stateValues, scores: stateScores))
    }

    public static func decode(
        value: MLXArray,
        score: MLXArray,
        positionalBias: MLXArray,
        startPosition: Int,
        state: State,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil
    ) throws -> DecodeResult {
        guard startPosition > 0 else {
            throw DeepSeekV4Error.compression(
                "decode start position must be positive; use prefill at position zero")
        }
        let geometry = try validate(
            values: value, scores: score, positionalBias: positionalBias,
            ratio: state.ratio, overlap: state.overlap)
        guard value.shape[0] == 1, geometry.headDimension == state.headDimension else {
            throw DeepSeekV4Error.compression(
                "decode accepts one row matching the compressor state")
        }
        let expectedRows = state.overlap ? state.ratio * 2 : state.ratio
        guard state.values.shape == [expectedRows, geometry.projectedDimension],
            state.scores.shape == [expectedRows, geometry.projectedDimension]
        else {
            throw DeepSeekV4Error.compression("compressor state geometry is inconsistent")
        }

        let position = startPosition % state.ratio
        let row = state.overlap ? state.ratio + position : position
        let adjustedScore = score.asType(.float32)
            + positionalBias[position..<(position + 1), 0...].asType(.float32)
        let updatedValues = replacingRow(
            state.values, at: row, with: value.asType(.float32))
        let updatedScores = replacingRow(state.scores, at: row, with: adjustedScore)
        let end = startPosition.addingReportingOverflow(1)
        guard !end.overflow else {
            throw DeepSeekV4Error.compression("decode position overflows Int")
        }
        guard end.partialValue.isMultiple(of: state.ratio) else {
            // Decode replaces one row per token. Realising the fixed-size state
            // here prevents the replacement graph from growing with the turn.
            //
            // Deferred: `asyncEval` walks, encodes and commits the same graph a
            // blocking `eval` would, and detaches every array on the tape, so
            // the growth this exists to stop is stopped just as completely. No
            // host value is read from either array, so nothing needs the wait.
            phaseAccounting?.recordAsyncEval(.compressedState)
            MLX.asyncEval([updatedValues, updatedScores])
            return DecodeResult(
                compressed: nil,
                state: State(
                    ratio: state.ratio, overlap: state.overlap,
                    headDimension: state.headDimension,
                    values: updatedValues, scores: updatedScores))
        }

        let poolingValues: MLXArray
        let poolingScores: MLXArray
        let nextValues: MLXArray
        let nextScores: MLXArray
        if state.overlap {
            poolingValues = concatenated([
                updatedValues[0..<state.ratio, 0..<state.headDimension],
                updatedValues[state.ratio..., state.headDimension...],
            ], axis: 0)
            poolingScores = concatenated([
                updatedScores[0..<state.ratio, 0..<state.headDimension],
                updatedScores[state.ratio..., state.headDimension...],
            ], axis: 0)
            let currentValues = updatedValues[state.ratio..., 0...]
            let currentScores = updatedScores[state.ratio..., 0...]
            nextValues = concatenated([currentValues, currentValues], axis: 0)
            nextScores = concatenated([currentScores, currentScores], axis: 0)
        } else {
            poolingValues = updatedValues
            poolingScores = updatedScores
            nextValues = updatedValues
            nextScores = updatedScores
        }
        let compressed = weightedPool(values: poolingValues, scores: poolingScores)
            .reshaped([1, state.headDimension])
        // The boundary-token branch. Mutually exclusive with the one above, so
        // a compressor contributes exactly one forcing point per decoded token
        // whichever side of the ratio it lands on. Deferred for the reason
        // above.
        phaseAccounting?.recordAsyncEval(.compressedState)
        MLX.asyncEval([compressed, nextValues, nextScores])
        return DecodeResult(
            compressed: compressed,
            state: State(
                ratio: state.ratio, overlap: state.overlap,
                headDimension: state.headDimension,
                values: nextValues, scores: nextScores))
    }

    private struct Geometry {
        let headDimension: Int
        let projectedDimension: Int
    }

    private static func validate(
        values: MLXArray,
        scores: MLXArray,
        positionalBias: MLXArray,
        ratio: Int,
        overlap: Bool
    ) throws -> Geometry {
        guard ratio > 0 else {
            throw DeepSeekV4Error.compression("compression ratio must be positive")
        }
        guard !overlap || ratio == 4 else {
            throw DeepSeekV4Error.compression(
                "overlapping compression is defined only for ratio 4")
        }
        guard values.ndim == 2, values.shape[0] > 0, scores.shape == values.shape else {
            throw DeepSeekV4Error.compression(
                "values and scores must be matching nonempty [sequence, projectedDimension] arrays")
        }
        let coefficient = overlap ? 2 : 1
        guard values.shape[1].isMultiple(of: coefficient) else {
            throw DeepSeekV4Error.compression(
                "overlap projection width must contain two equal halves")
        }
        let headDimension = values.shape[1] / coefficient
        guard headDimension > 0,
            positionalBias.shape == [ratio, values.shape[1]]
        else {
            throw DeepSeekV4Error.compression(
                "positional bias must be [\(ratio), \(values.shape[1])]")
        }
        return Geometry(headDimension: headDimension, projectedDimension: values.shape[1])
    }

    private static func weightedPool(values: MLXArray, scores: MLXArray) -> MLXArray {
        sum(values * softmax(scores, axis: 0, precise: true), axis: 0)
    }

    private static func replacingRow(
        _ array: MLXArray, at row: Int, with replacement: MLXArray
    ) -> MLXArray {
        var pieces = [MLXArray]()
        if row > 0 { pieces.append(array[0..<row, 0...]) }
        pieces.append(replacement)
        if row + 1 < array.shape[0] { pieces.append(array[(row + 1)..., 0...]) }
        return concatenated(pieces, axis: 0)
    }
}

/// Lightning index scoring and selection for ratio-four compressed attention.
public enum DeepSeekV4LightningIndexer {
    /// `queries` is `[sequence, heads, dimension]`, compressed keys are
    /// `[keys, dimension]`, and weights are `[sequence, heads]` after the
    /// official `headDim^-0.5 * headCount^-0.5` scale.
    public static func scores(
        queries: MLXArray, compressedKeys: MLXArray, headWeights: MLXArray
    ) throws -> MLXArray {
        guard queries.ndim == 3, queries.shape[0] > 0, queries.shape[1] > 0,
            queries.shape[2] > 0,
            compressedKeys.ndim == 2,
            compressedKeys.shape[1] == queries.shape[2],
            headWeights.shape == [queries.shape[0], queries.shape[1]]
        else {
            throw DeepSeekV4Error.indexing(
                "index queries, keys, and weights have incompatible geometry")
        }
        guard compressedKeys.shape[0] > 0 else {
            return MLXArray.zeros([queries.shape[0], 0], dtype: .float32)
        }
        let perHead = einsum(
            "shd,td->sht", queries.asType(.float32), compressedKeys.asType(.float32))
        return sum(maximum(perHead, 0) * headWeights.asType(.float32)[.ellipsis, .newAxis],
            axis: 1)
    }

    /// Deterministic host selection from the small `[sequence, compressedKeys]`
    /// score matrix. PyTorch's unsorted top-k defines the selected set but not
    /// its order; descending score with key-id tie-break is our stable order.
    public static func select(
        scores: MLXArray,
        topK: Int,
        ratio: Int,
        startPosition: Int,
        offset: Int,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil
    ) throws -> [[Int]] {
        try measuringPhase(phaseAccounting?.recordLightningIndexer(nanoseconds:)) {
            try selecting(
                scores: scores, topK: topK, ratio: ratio,
                startPosition: startPosition, offset: offset,
                phaseAccounting: phaseAccounting)
        }
    }

    /// The selection itself. It is a separate function so the bracket above is
    /// exactly its scope: the `eval`/`asArray` sync and the host top-k, and
    /// therefore also whatever deferred index graph that sync forces.
    private static func selecting(
        scores: MLXArray,
        topK: Int,
        ratio: Int,
        startPosition: Int,
        offset: Int,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil
    ) throws -> [[Int]] {
        guard scores.ndim == 2, scores.shape[0] > 0,
            topK > 0, ratio > 0, startPosition >= 0, offset >= 0
        else {
            throw DeepSeekV4Error.indexing("invalid lightning-index selection geometry")
        }
        if startPosition > 0, scores.shape[0] != 1 {
            throw DeepSeekV4Error.indexing(
                "incremental lightning indexing accepts exactly one token")
        }
        let end = startPosition.addingReportingOverflow(scores.shape[0])
        guard !end.overflow else {
            throw DeepSeekV4Error.indexing("lightning-index end position overflows Int")
        }
        let available = end.partialValue / ratio
        guard available <= scores.shape[1] else {
            throw DeepSeekV4Error.indexing(
                "score matrix has fewer compressed keys than the sequence position requires")
        }
        let selectedCount = min(topK, available)
        if selectedCount == 0 {
            return [[Int]](
                repeating: [], count: scores.shape[0])
        }
        let hostScores = scores.asType(.float32)
        // Counted after the early returns above, so the census states the
        // selections that actually pulled rather than the calls that entered.
        phaseAccounting?.recordEval(.indexer)
        hostScores.eval()
        let values = hostScores.asArray(Float.self)
        guard values.allSatisfy({ !$0.isNaN }) else {
            throw DeepSeekV4Error.indexing("lightning-index score contains NaN")
        }
        var result = [[Int]]()
        result.reserveCapacity(scores.shape[0])
        for row in 0..<scores.shape[0] {
            let visible = startPosition == 0 ? (row + 1) / ratio : available
            let start = row * scores.shape[1]
            let order = (0..<available).sorted { left, right in
                let lhs = left < visible ? values[start + left] : -Float.infinity
                let rhs = right < visible ? values[start + right] : -Float.infinity
                return lhs == rhs ? left < right : lhs > rhs
            }
            result.append(try order.prefix(selectedCount).map { key in
                guard key < visible else { return -1 }
                let indexed = offset.addingReportingOverflow(key)
                guard !indexed.overflow else {
                    throw DeepSeekV4Error.indexing("lightning cache index overflows Int")
                }
                return indexed.partialValue
            })
        }
        return result
    }
}

/// Sparse MQA attention with the learned sink in the softmax denominator.
///
/// The sink has no value vector: it can absorb probability mass but contributes
/// zero to the numerator. Treating it as another KV row is therefore wrong.
public enum DeepSeekV4SparseAttention {
    /// The bracket around this call states what it costs to *express* the
    /// per-head graph — the head loop below forces nothing, so the arithmetic
    /// it describes is charged to whichever later sync evaluates it.
    public static func forward(
        queries: MLXArray,
        keysAndValues: MLXArray,
        sinks: MLXArray,
        indices: [[Int]],
        scale: Float,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil
    ) throws -> MLXArray {
        try measuringPhase(phaseAccounting?.recordSparseAttention(nanoseconds:)) {
            try expressing(
                queries: queries, keysAndValues: keysAndValues, sinks: sinks,
                indices: indices, scale: scale)
        }
    }

    private static func expressing(
        queries: MLXArray,
        keysAndValues: MLXArray,
        sinks: MLXArray,
        indices: [[Int]],
        scale: Float
    ) throws -> MLXArray {
        guard queries.ndim == 3, queries.shape[0] > 0, queries.shape[1] > 0,
            queries.shape[2] > 0,
            keysAndValues.ndim == 2, keysAndValues.shape[0] > 0,
            keysAndValues.shape[1] == queries.shape[2],
            sinks.shape == [queries.shape[1]], indices.count == queries.shape[0],
            scale.isFinite, scale > 0
        else {
            throw DeepSeekV4Error.attention(
                "sparse attention queries, shared KV, sinks, indices, or scale are invalid")
        }
        var tokenOutputs = [MLXArray]()
        tokenOutputs.reserveCapacity(queries.shape[0])
        for token in 0..<queries.shape[0] {
            var seen = Set<Int>()
            let valid = try indices[token].compactMap { index -> Int? in
                if index == -1 { return nil }
                guard index >= 0, index < keysAndValues.shape[0] else {
                    throw DeepSeekV4Error.attention(
                        "token \(token) names KV row \(index), outside the cache")
                }
                guard seen.insert(index).inserted else {
                    throw DeepSeekV4Error.attention(
                        "token \(token) names KV row \(index) more than once")
                }
                return index
            }
            var headOutputs = [MLXArray]()
            headOutputs.reserveCapacity(queries.shape[1])
            for head in 0..<queries.shape[1] {
                guard !valid.isEmpty else {
                    headOutputs.append(
                        MLXArray.zeros([queries.shape[2]], dtype: .float32))
                    continue
                }
                let kv = keysAndValues.take(MLXArray(valid), axis: 0).asType(.float32)
                let query = queries[token, head, 0...].asType(.float32)
                let attentionScores = sum(kv * query, axis: -1) * scale
                let sink = sinks[head].asType(.float32)
                let maximumScore = max(concatenated([
                    attentionScores, sink.reshaped([1]),
                ]))
                let exponentials = exp(attentionScores - maximumScore)
                let denominator = sum(exponentials) + exp(sink - maximumScore)
                headOutputs.append(
                    sum(kv * exponentials[.ellipsis, .newAxis], axis: 0) / denominator)
            }
            tokenOutputs.append(stacked(headOutputs, axis: 0))
        }
        return stacked(tokenOutputs, axis: 0)
    }
}

/// The grouped low-rank output projection after sparse attention.
public enum DeepSeekV4GroupedOutput {
    public static func project(
        _ attention: MLXArray,
        groupWeight: MLXArray,
        outputWeight: MLXArray,
        groups: Int
    ) throws -> MLXArray {
        guard attention.ndim == 3, attention.shape[0] > 0,
            attention.shape[1] > 0, attention.shape[2] > 0,
            groups > 0, attention.shape[1].isMultiple(of: groups)
        else {
            throw DeepSeekV4Error.attention(
                "grouped output requires nonempty [sequence, heads, dimension] attention "
                    + "with heads divisible by groups")
        }
        let perGroup = attention.shape[1] / groups
        let groupInput = perGroup.multipliedReportingOverflow(by: attention.shape[2])
        guard !groupInput.overflow, groupWeight.ndim == 2,
            groupWeight.shape[0] > 0,
            groupWeight.shape[0].isMultiple(of: groups),
            groupWeight.shape[1] == groupInput.partialValue
        else {
            throw DeepSeekV4Error.attention("first output projection has invalid group geometry")
        }
        let rank = groupWeight.shape[0] / groups
        let flattenedRank = groups.multipliedReportingOverflow(by: rank)
        guard !flattenedRank.overflow, outputWeight.ndim == 2,
            outputWeight.shape[0] > 0,
            outputWeight.shape[1] == flattenedRank.partialValue
        else {
            throw DeepSeekV4Error.attention("second output projection has invalid rank geometry")
        }
        let grouped = attention.asType(.float32)
            .reshaped([attention.shape[0], groups, groupInput.partialValue])
        let firstWeight = groupWeight.asType(.float32)
            .reshaped([groups, rank, groupInput.partialValue])
        let latent = einsum("tgd,grd->tgr", grouped, firstWeight)
        return k3Linear(
            latent.reshaped([attention.shape[0], flattenedRank.partialValue]),
            outputWeight.asType(.float32))
    }
}
