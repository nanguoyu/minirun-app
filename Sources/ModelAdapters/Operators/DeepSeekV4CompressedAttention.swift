import Foundation
import MLX

/// The batch-one prefill equation for DeepSeek V4's ratio-four compressed
/// attention path.
///
/// Storage-backed linear projections are supplied by the caller. This value
/// owns only the architecture-specific composition after those projections:
/// window KV normalization, both learned compressors, indexer QAT and causal
/// selection, sparse attention, and inverse rotary. Keeping the large matrices
/// outside this type lets an artifact runner materialize one bounded projection
/// at a time instead of constructing a dense layer-sized weight object.
public enum DeepSeekV4CompressedAttention {
    public struct Geometry: Sendable, Hashable {
        public let attentionHeads: Int
        public let headDimension: Int
        public let ropeDimension: Int
        public let slidingWindow: Int
        public let compressionRatio: Int
        public let maximumPositionCount: Int
        public let indexHeads: Int
        public let indexHeadDimension: Int
        public let indexTopK: Int
        public let rmsNormEpsilon: Float
        public let rotary: DeepSeekV4RotaryParameters

        public init(
            attentionHeads: Int,
            headDimension: Int,
            ropeDimension: Int,
            slidingWindow: Int,
            compressionRatio: Int,
            maximumPositionCount: Int,
            indexHeads: Int,
            indexHeadDimension: Int,
            indexTopK: Int,
            rmsNormEpsilon: Float,
            rotary: DeepSeekV4RotaryParameters
        ) throws {
            let nonRotary = headDimension - ropeDimension
            guard attentionHeads > 0, headDimension > 0,
                ropeDimension > 0, ropeDimension <= headDimension,
                ropeDimension.isMultiple(of: 2),
                nonRotary.isMultiple(of: 64), slidingWindow > 0,
                compressionRatio == 4,
                maximumPositionCount >= slidingWindow,
                maximumPositionCount <= Int(Int32.max),
                indexHeads > 0, indexHeadDimension >= ropeDimension,
                indexHeadDimension.isMultiple(of: DeepSeekV4FP4ActivationReference.blockSize),
                indexTopK > 0,
                rmsNormEpsilon.isFinite, rmsNormEpsilon > 0,
                rotary.dimension == ropeDimension
            else {
                throw DeepSeekV4Error.unsupportedArchitecture(
                    "ratio-four compressed attention geometry is invalid")
            }
            self.attentionHeads = attentionHeads
            self.headDimension = headDimension
            self.ropeDimension = ropeDimension
            self.slidingWindow = slidingWindow
            self.compressionRatio = compressionRatio
            self.maximumPositionCount = maximumPositionCount
            self.indexHeads = indexHeads
            self.indexHeadDimension = indexHeadDimension
            self.indexTopK = indexTopK
            self.rmsNormEpsilon = rmsNormEpsilon
            self.rotary = rotary
        }

        public init(config: DeepSeekV4Config, layer: Int) throws {
            let ratio = try config.compressionRatio(layer: layer)
            guard ratio == 4 else {
                throw DeepSeekV4Error.unsupportedArchitecture(
                    "layer \(layer) uses compression ratio \(ratio), not the lightning-index path")
            }
            guard config.numberOfKeyValueHeads == 1 else {
                throw DeepSeekV4Error.unsupportedArchitecture(
                    "compressed attention requires one shared KV head")
            }
            try self.init(
                attentionHeads: config.numberOfAttentionHeads,
                headDimension: config.attentionHeadDimension,
                ropeDimension: config.ropeHeadDimension,
                slidingWindow: config.slidingWindow,
                compressionRatio: ratio,
                maximumPositionCount: config.maximumPositionCount,
                indexHeads: config.indexHeadCount,
                indexHeadDimension: config.indexHeadDimension,
                indexTopK: config.indexTopK,
                rmsNormEpsilon: config.rmsNormEpsilon,
                rotary: DeepSeekV4RotaryParameters(
                    dimension: config.ropeHeadDimension,
                    originalSequenceLength: config.ropeOriginalLength,
                    base: Double(config.compressionRopeTheta),
                    factor: Double(config.ropeFactor),
                    betaFast: Double(config.ropeBetaFast),
                    betaSlow: Double(config.ropeBetaSlow)))
        }
    }

    /// Results of the caller's storage-backed projections for one prompt.
    public struct ProjectedInputs {
        /// `wq_b(q_norm(wq_a(x)))`, flattened across attention heads.
        public let queries: MLXArray
        /// `wkv(x)` for the sliding-window KV rows.
        public let windowKeyValues: MLXArray
        /// Main compressor `wkv(x)` and `wgate(x)` projections.
        public let compressorValues: MLXArray
        public let compressorScores: MLXArray
        /// Indexer `wq_b(q_norm(wq_a(x)))`, flattened across index heads.
        public let indexQueries: MLXArray
        /// Unscaled BF16 `weights_proj(x)` values.
        public let indexHeadWeights: MLXArray
        /// Indexer compressor `wkv(x)` and `wgate(x)` projections.
        public let indexCompressorValues: MLXArray
        public let indexCompressorScores: MLXArray

        public init(
            queries: MLXArray,
            windowKeyValues: MLXArray,
            compressorValues: MLXArray,
            compressorScores: MLXArray,
            indexQueries: MLXArray,
            indexHeadWeights: MLXArray,
            indexCompressorValues: MLXArray,
            indexCompressorScores: MLXArray
        ) {
            self.queries = queries
            self.windowKeyValues = windowKeyValues
            self.compressorValues = compressorValues
            self.compressorScores = compressorScores
            self.indexQueries = indexQueries
            self.indexHeadWeights = indexHeadWeights
            self.indexCompressorValues = indexCompressorValues
            self.indexCompressorScores = indexCompressorScores
        }
    }

    /// Small vectors and positional tables consumed by the attention equation.
    public struct Parameters {
        public let keyValueNorm: MLXArray
        public let sinks: MLXArray
        public let compressorPositionalBias: MLXArray
        public let compressorNorm: MLXArray
        public let indexCompressorPositionalBias: MLXArray
        public let indexCompressorNorm: MLXArray

        public init(
            keyValueNorm: MLXArray,
            sinks: MLXArray,
            compressorPositionalBias: MLXArray,
            compressorNorm: MLXArray,
            indexCompressorPositionalBias: MLXArray,
            indexCompressorNorm: MLXArray
        ) {
            self.keyValueNorm = keyValueNorm
            self.sinks = sinks
            self.compressorPositionalBias = compressorPositionalBias
            self.compressorNorm = compressorNorm
            self.indexCompressorPositionalBias = indexCompressorPositionalBias
            self.indexCompressorNorm = indexCompressorNorm
        }
    }

    /// Materialized batch-one cache after a prompt or decode step.
    ///
    /// The window cache is always exactly `slidingWindow` rows. Compressed
    /// caches are allocated once from the run-scoped `decodePositionLimit`,
    /// never the model's potentially enormous context ceiling;
    /// `completedCompressedCount` says which prefix is valid. The compressor
    /// states themselves are fixed-size. Keeping the next position in the
    /// state rejects duplicate or skipped decode calls instead of silently
    /// corrupting the circular cache.
    public struct State {
        fileprivate let nextPosition: Int
        fileprivate let positionLimit: Int
        fileprivate let completedCompressedCount: Int
        fileprivate let windowKeyValues: MLXArray
        fileprivate let compressedKeyValues: MLXArray
        fileprivate let indexKeys: MLXArray
        fileprivate let compressorState: DeepSeekV4CompressorPool.State
        fileprivate let indexCompressorState: DeepSeekV4CompressorPool.State

        public var position: Int { nextPosition }
        public var decodePositionLimit: Int { positionLimit }
        public var compressedCount: Int { completedCompressedCount }
        public var compressedCapacity: Int { compressedKeyValues.shape[0] }
    }

    public struct PrefillResult {
        /// `[sequence, attentionHeads, headDimension]`, after inverse rotary and
        /// before the grouped low-rank output projection.
        public let attention: MLXArray
        /// Exact sparse-cache rows supplied to every prompt position.
        public let indices: [[Int]]
        public let compressorState: DeepSeekV4CompressorPool.State
        public let indexCompressorState: DeepSeekV4CompressorPool.State
        /// Present only when the caller declares a run-scoped position limit.
        /// Plain prefill does not reserve a model-maximum decode cache.
        public let state: State?
    }

    public struct DecodeResult {
        public let attention: MLXArray
        public let indices: [Int]
        public let state: State
    }

    public static func prefill(
        projected: ProjectedInputs,
        parameters: Parameters,
        geometry: Geometry,
        decodePositionLimit: Int? = nil,
        cancellationCheck: () throws -> Void = {},
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> PrefillResult {
        let sequence = try validate(
            projected: projected, parameters: parameters, geometry: geometry,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        guard sequence <= geometry.maximumPositionCount else {
            throw DeepSeekV4Error.attention(
                "prompt length \(sequence) exceeds the configured maximum "
                    + "\(geometry.maximumPositionCount)")
        }
        try cancellationCheck()
        let positions = Array(0..<sequence)

        var queries = projected.queries.asType(.float32).reshaped([
            sequence, geometry.attentionHeads, geometry.headDimension,
        ])
        queries = queries * rsqrt(
            mean(queries * queries, axis: -1, keepDims: true)
                + geometry.rmsNormEpsilon)
        queries = try rotateTail(
            queries, positions: positions, geometry: geometry, inverse: false)

        var windowKeyValues = K3Norm.rms(
            projected.windowKeyValues,
            weight: parameters.keyValueNorm,
            eps: geometry.rmsNormEpsilon)
        windowKeyValues = try rotateTail(
            windowKeyValues, positions: positions, geometry: geometry, inverse: false)
        windowKeyValues = try quantizeNonRotaryPrefix(
            windowKeyValues, geometry: geometry, phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        try cancellationCheck()

        let compressed = try DeepSeekV4CompressorPool.prefill(
            values: projected.compressorValues,
            scores: projected.compressorScores,
            positionalBias: parameters.compressorPositionalBias,
            ratio: geometry.compressionRatio,
            overlap: true)
        var compressedKeyValues: MLXArray
        if compressed.compressed.shape[0] == 0 {
            compressedKeyValues = MLXArray.zeros(
                [0, geometry.headDimension], dtype: .float32)
        } else {
            compressedKeyValues = K3Norm.rms(
                compressed.compressed,
                weight: parameters.compressorNorm,
                eps: geometry.rmsNormEpsilon)
        }
        let compressedPositions = (0..<(sequence / geometry.compressionRatio)).map {
            $0 * geometry.compressionRatio
        }
        if !compressedPositions.isEmpty {
            compressedKeyValues = try rotateTail(
                compressedKeyValues,
                positions: compressedPositions,
                geometry: geometry,
                inverse: false)
            compressedKeyValues = try quantizeNonRotaryPrefix(
                compressedKeyValues, geometry: geometry,
                phaseAccounting: phaseAccounting,
                diagnostics: diagnostics)
        }

        var indexQueries = projected.indexQueries.asType(.float32).reshaped([
            sequence, geometry.indexHeads, geometry.indexHeadDimension,
        ])
        indexQueries = try rotateTail(
            indexQueries, positions: positions, geometry: geometry, inverse: false)
        indexQueries = try DeepSeekV4FP4ActivationReference.rotateQuantizeDequantize(
            indexQueries, phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)

        let indexCompressed = try DeepSeekV4CompressorPool.prefill(
            values: projected.indexCompressorValues,
            scores: projected.indexCompressorScores,
            positionalBias: parameters.indexCompressorPositionalBias,
            ratio: geometry.compressionRatio,
            overlap: true)
        var indexKeys: MLXArray
        if indexCompressed.compressed.shape[0] == 0 {
            indexKeys = MLXArray.zeros(
                [0, geometry.indexHeadDimension], dtype: .float32)
        } else {
            indexKeys = K3Norm.rms(
                indexCompressed.compressed,
                weight: parameters.indexCompressorNorm,
                eps: geometry.rmsNormEpsilon)
        }
        if !compressedPositions.isEmpty {
            indexKeys = try rotateTail(
                indexKeys, positions: compressedPositions,
                geometry: geometry, inverse: false)
            indexKeys = try DeepSeekV4FP4ActivationReference.rotateQuantizeDequantize(
                indexKeys, phaseAccounting: phaseAccounting,
                diagnostics: diagnostics)
        }
        try cancellationCheck()

        let indexScale = Float(
            1 / Foundation.sqrt(
                Double(geometry.indexHeadDimension * geometry.indexHeads)))
        let indexScores = try DeepSeekV4LightningIndexer.scores(
            queries: indexQueries,
            compressedKeys: indexKeys,
            headWeights: projected.indexHeadWeights.asType(.float32) * indexScale)
        let offset = sequence
        let selected = try DeepSeekV4LightningIndexer.select(
            scores: indexScores,
            topK: geometry.indexTopK,
            ratio: geometry.compressionRatio,
            startPosition: 0,
            offset: offset,
            phaseAccounting: phaseAccounting)
        let window = try DeepSeekV4AttentionIndices.window(
            windowSize: geometry.slidingWindow,
            sequenceLength: sequence,
            startPosition: 0)
        let indices = zip(window, selected).map(+)
        let keyValues = concatenated([windowKeyValues, compressedKeyValues], axis: 0)
        var attention = try DeepSeekV4SparseAttention.forward(
            queries: queries,
            keysAndValues: keyValues,
            sinks: parameters.sinks,
            indices: indices,
            scale: Float(1 / Foundation.sqrt(Double(geometry.headDimension))),
            phaseAccounting: phaseAccounting)
        attention = try rotateTail(
            attention, positions: positions, geometry: geometry, inverse: true)
        let state: State?
        if let decodePositionLimit {
            guard decodePositionLimit >= sequence,
                decodePositionLimit <= geometry.maximumPositionCount
            else {
                throw DeepSeekV4Error.attention(
                    "decode position limit \(decodePositionLimit) must contain the prompt "
                        + "and not exceed \(geometry.maximumPositionCount)")
            }
            let windowCache = makeWindowCache(
                windowKeyValues, windowSize: geometry.slidingWindow)
            let compressedCapacity =
                decodePositionLimit / geometry.compressionRatio
            let fixedCompressedCache = makeFixedCache(
                compressedKeyValues,
                capacity: compressedCapacity,
                width: geometry.headDimension)
            let fixedIndexCache = makeFixedCache(
                indexKeys,
                capacity: compressedCapacity,
                width: geometry.indexHeadDimension)
            submittingToGPU(phaseAccounting, .blockCache) {
                MLX.asyncEval([attention, windowCache, fixedCompressedCache, fixedIndexCache])
            }
            state = State(
                nextPosition: sequence,
                positionLimit: decodePositionLimit,
                completedCompressedCount: compressedKeyValues.shape[0],
                windowKeyValues: windowCache,
                compressedKeyValues: fixedCompressedCache,
                indexKeys: fixedIndexCache,
                compressorState: compressed.state,
                indexCompressorState: indexCompressed.state)
        } else {
            attention.eval()
            state = nil
        }
        try cancellationCheck()
        return PrefillResult(
            attention: attention,
            indices: indices,
            compressorState: compressed.state,
            indexCompressorState: indexCompressed.state,
            state: state)
    }

    /// Execute one token against the materialized ratio-four cache.
    ///
    /// All storage-backed projections are supplied by the caller for exactly
    /// one row. This method owns cache mutation, compressor boundaries,
    /// lightning selection, sparse attention, and inverse rotary. Returned
    /// state is evaluated before it escapes, so repeated decode does not retain
    /// an ever-growing lazy replacement graph.
    public static func decode(
        projected: ProjectedInputs,
        parameters: Parameters,
        geometry: Geometry,
        startPosition: Int,
        state: State,
        cancellationCheck: () throws -> Void = {},
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> DecodeResult {
        let sequence = try validate(
            projected: projected, parameters: parameters, geometry: geometry,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        guard sequence == 1 else {
            throw DeepSeekV4Error.attention(
                "incremental compressed attention accepts exactly one token")
        }
        guard startPosition == state.nextPosition, startPosition > 0 else {
            throw DeepSeekV4Error.attention(
                "decode position \(startPosition) does not follow cache position "
                    + "\(state.nextPosition)")
        }
        guard startPosition < state.positionLimit else {
            throw DeepSeekV4Error.attention(
                "decode position \(startPosition) reaches this run's limit "
                    + "\(state.positionLimit)")
        }
        try validate(state: state, geometry: geometry)
        try cancellationCheck()

        let positions = [startPosition]
        var queries = projected.queries.asType(.float32).reshaped([
            1, geometry.attentionHeads, geometry.headDimension,
        ])
        queries = queries * rsqrt(
            mean(queries * queries, axis: -1, keepDims: true)
                + geometry.rmsNormEpsilon)
        queries = try rotateTail(
            queries, positions: positions, geometry: geometry, inverse: false)

        var windowRow = K3Norm.rms(
            projected.windowKeyValues,
            weight: parameters.keyValueNorm,
            eps: geometry.rmsNormEpsilon)
        windowRow = try rotateTail(
            windowRow, positions: positions, geometry: geometry, inverse: false)
        windowRow = try quantizeNonRotaryPrefix(
            windowRow, geometry: geometry, phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        let windowCache = replacingRow(
            state.windowKeyValues,
            at: startPosition % geometry.slidingWindow,
            with: windowRow)
        try cancellationCheck()

        let compressedStep = try DeepSeekV4CompressorPool.decode(
            value: projected.compressorValues,
            score: projected.compressorScores,
            positionalBias: parameters.compressorPositionalBias,
            startPosition: startPosition,
            state: state.compressorState,
            phaseAccounting: phaseAccounting)
        var compressedCache = state.compressedKeyValues
        var compressedCount = state.completedCompressedCount
        if var row = compressedStep.compressed {
            row = K3Norm.rms(
                row, weight: parameters.compressorNorm,
                eps: geometry.rmsNormEpsilon)
            row = try rotateTail(
                row,
                positions: [startPosition + 1 - geometry.compressionRatio],
                geometry: geometry,
                inverse: false)
            row = try quantizeNonRotaryPrefix(
                row, geometry: geometry, phaseAccounting: phaseAccounting,
                diagnostics: diagnostics)
            compressedCache = replacingRow(
                compressedCache, at: compressedCount, with: row)
            compressedCount += 1
        }

        var indexQueries = projected.indexQueries.asType(.float32).reshaped([
            1, geometry.indexHeads, geometry.indexHeadDimension,
        ])
        indexQueries = try rotateTail(
            indexQueries, positions: positions, geometry: geometry, inverse: false)
        indexQueries = try DeepSeekV4FP4ActivationReference.rotateQuantizeDequantize(
            indexQueries, phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        let indexStep = try DeepSeekV4CompressorPool.decode(
            value: projected.indexCompressorValues,
            score: projected.indexCompressorScores,
            positionalBias: parameters.indexCompressorPositionalBias,
            startPosition: startPosition,
            state: state.indexCompressorState,
            phaseAccounting: phaseAccounting)
        var indexCache = state.indexKeys
        var indexCount = state.completedCompressedCount
        if var row = indexStep.compressed {
            row = K3Norm.rms(
                row, weight: parameters.indexCompressorNorm,
                eps: geometry.rmsNormEpsilon)
            row = try rotateTail(
                row,
                positions: [startPosition + 1 - geometry.compressionRatio],
                geometry: geometry,
                inverse: false)
            row = try DeepSeekV4FP4ActivationReference.rotateQuantizeDequantize(
                row, phaseAccounting: phaseAccounting,
                diagnostics: diagnostics)
            indexCache = replacingRow(indexCache, at: indexCount, with: row)
            indexCount += 1
        }
        let expectedCompressed = (startPosition + 1) / geometry.compressionRatio
        guard compressedCount == expectedCompressed,
            indexCount == expectedCompressed
        else {
            throw DeepSeekV4Error.attention(
                "compressed caches do not match the decode position")
        }
        try cancellationCheck()

        let indexScale = Float(
            1 / Foundation.sqrt(
                Double(geometry.indexHeadDimension * geometry.indexHeads)))
        let indexScores = try DeepSeekV4LightningIndexer.scores(
            queries: indexQueries,
            compressedKeys: validPrefix(
                indexCache, count: indexCount, width: geometry.indexHeadDimension),
            headWeights: projected.indexHeadWeights.asType(.float32) * indexScale)
        let selected = try DeepSeekV4LightningIndexer.select(
            scores: indexScores,
            topK: geometry.indexTopK,
            ratio: geometry.compressionRatio,
            startPosition: startPosition,
            offset: geometry.slidingWindow,
            phaseAccounting: phaseAccounting)[0]
        let window = try DeepSeekV4AttentionIndices.window(
            windowSize: geometry.slidingWindow,
            sequenceLength: 1,
            startPosition: startPosition)[0]
        let indices = window + selected
        let keyValues = concatenated([
            windowCache,
            validPrefix(
                compressedCache, count: compressedCount, width: geometry.headDimension),
        ], axis: 0)
        var attention = try DeepSeekV4SparseAttention.forward(
            queries: queries,
            keysAndValues: keyValues,
            sinks: parameters.sinks,
            indices: [indices],
            scale: Float(1 / Foundation.sqrt(Double(geometry.headDimension))),
            phaseAccounting: phaseAccounting)
        attention = try rotateTail(
            attention, positions: positions, geometry: geometry, inverse: true)
        let next = State(
            nextPosition: startPosition + 1,
            positionLimit: state.positionLimit,
            completedCompressedCount: compressedCount,
            windowKeyValues: windowCache,
            compressedKeyValues: compressedCache,
            indexKeys: indexCache,
            compressorState: compressedStep.state,
            indexCompressorState: indexStep.state)
        // The cache boundary, deferred. These four arrays persist into the next
        // token and must be realised before this pass's graph is discarded —
        // which `asyncEval` does, because it runs the same `eval_impl` and
        // detaches the same tape. What it does not do is stop the decode thread
        // for the GPU round trip, and nothing here reads a host value.
        submittingToGPU(phaseAccounting, .blockCache) {
            MLX.asyncEval([attention, windowCache, compressedCache, indexCache])
        }
        try cancellationCheck()
        return DecodeResult(attention: attention, indices: indices, state: next)
    }

    private static func validate(
        projected: ProjectedInputs,
        parameters: Parameters,
        geometry: Geometry,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> Int {
        guard projected.queries.ndim == 2, projected.queries.shape[0] > 0 else {
            throw DeepSeekV4Error.attention(
                "compressed attention queries must be nonempty [sequence, heads * dimension]")
        }
        let sequence = projected.queries.shape[0]
        let queryWidth = geometry.attentionHeads.multipliedReportingOverflow(
            by: geometry.headDimension)
        let indexWidth = geometry.indexHeads.multipliedReportingOverflow(
            by: geometry.indexHeadDimension)
        let mainProjectionWidth = geometry.headDimension.multipliedReportingOverflow(by: 2)
        let indexProjectionWidth = geometry.indexHeadDimension.multipliedReportingOverflow(by: 2)
        guard !queryWidth.overflow, !indexWidth.overflow,
            !mainProjectionWidth.overflow, !indexProjectionWidth.overflow,
            projected.queries.shape == [sequence, queryWidth.partialValue],
            projected.windowKeyValues.shape == [sequence, geometry.headDimension],
            projected.compressorValues.shape == [sequence, mainProjectionWidth.partialValue],
            projected.compressorScores.shape == projected.compressorValues.shape,
            projected.indexQueries.shape == [sequence, indexWidth.partialValue],
            projected.indexHeadWeights.shape == [sequence, geometry.indexHeads],
            projected.indexCompressorValues.shape
                == [sequence, indexProjectionWidth.partialValue],
            projected.indexCompressorScores.shape
                == projected.indexCompressorValues.shape,
            parameters.keyValueNorm.shape == [geometry.headDimension],
            parameters.sinks.shape == [geometry.attentionHeads],
            parameters.compressorPositionalBias.shape
                == [geometry.compressionRatio, mainProjectionWidth.partialValue],
            parameters.compressorNorm.shape == [geometry.headDimension],
            parameters.indexCompressorPositionalBias.shape
                == [geometry.compressionRatio, indexProjectionWidth.partialValue],
            parameters.indexCompressorNorm.shape == [geometry.indexHeadDimension]
        else {
            throw DeepSeekV4Error.attention(
                "compressed attention projections or parameter tables have incompatible geometry")
        }
        let arrays = [
            projected.queries,
            projected.windowKeyValues,
            projected.compressorValues,
            projected.compressorScores,
            projected.indexQueries,
            projected.indexHeadWeights,
            projected.indexCompressorValues,
            projected.indexCompressorScores,
            parameters.keyValueNorm,
            parameters.sinks,
            parameters.compressorPositionalBias,
            parameters.compressorNorm,
            parameters.indexCompressorPositionalBias,
            parameters.indexCompressorNorm,
        ]
        guard arrays.allSatisfy({ $0.dtype.isFloatingPoint && !$0.dtype.isComplex }) else {
            throw DeepSeekV4Error.attention(
                "compressed attention projections and parameters must be real floating point")
        }
        // One bracket for the whole sweep, not one per array: fourteen syncs
        // are one guard, and a bracket per array would measure the clock. One
        // switch too, for the same reason — the fourteen syncs are one
        // diagnostic and are skipped together.
        if diagnostics.validateFiniteness {
            try measuringPhase(
                phaseAccounting,
                excludingGPUBoundaryFrom: phaseAccounting?
                    .recordFinitenessSweep(nanoseconds:)
            ) {
                try waitingForGPU(phaseAccounting, .finitenessSweep) {
                    for array in arrays where !isFinite(array).all().item(Bool.self) {
                        throw DeepSeekV4Error.attention(
                            "compressed attention projection or parameter contains "
                                + "a non-finite value")
                    }
                }
            }
        }
        return sequence
    }

    private static func quantizeNonRotaryPrefix(
        _ input: MLXArray,
        geometry: Geometry,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        let nonRotary = geometry.headDimension - geometry.ropeDimension
        guard nonRotary > 0 else { return input }
        let prefix = try DeepSeekV4FP8ActivationReference.quantizeDequantize(
            input[.ellipsis, 0..<nonRotary], blockSize: 64,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        return concatenated([prefix, input[.ellipsis, nonRotary...]], axis: -1)
    }

    private static func rotateTail(
        _ input: MLXArray,
        positions: [Int],
        geometry: Geometry,
        inverse: Bool
    ) throws -> MLXArray {
        let prefixWidth = input.shape[input.ndim - 1] - geometry.ropeDimension
        guard prefixWidth >= 0 else {
            throw DeepSeekV4Error.attention(
                "rotary tail exceeds the projected dimension")
        }
        let tail = try DeepSeekV4Rotary.apply(
            input[.ellipsis, prefixWidth...],
            positions: positions,
            parameters: geometry.rotary,
            inverse: inverse)
        guard prefixWidth > 0 else { return tail }
        return concatenated([input[.ellipsis, 0..<prefixWidth], tail], axis: -1)
    }

    private static func validate(state: State, geometry: Geometry) throws {
        let expectedCompressed = state.nextPosition / geometry.compressionRatio
        let compressedCapacity = state.positionLimit / geometry.compressionRatio
        let mainWidth = geometry.headDimension.multipliedReportingOverflow(by: 2)
        let indexWidth = geometry.indexHeadDimension.multipliedReportingOverflow(by: 2)
        guard !mainWidth.overflow, !indexWidth.overflow,
            state.nextPosition > 0,
            state.positionLimit > 0,
            state.positionLimit <= geometry.maximumPositionCount,
            state.nextPosition <= state.positionLimit,
            state.completedCompressedCount == expectedCompressed,
            state.windowKeyValues.shape
                == [geometry.slidingWindow, geometry.headDimension],
            state.compressedKeyValues.shape
                == [compressedCapacity, geometry.headDimension],
            state.indexKeys.shape
                == [compressedCapacity, geometry.indexHeadDimension],
            state.compressorState.ratio == geometry.compressionRatio,
            state.compressorState.overlap,
            state.compressorState.headDimension == geometry.headDimension,
            state.compressorState.values.shape
                == [geometry.compressionRatio * 2, mainWidth.partialValue],
            state.compressorState.scores.shape == state.compressorState.values.shape,
            state.indexCompressorState.ratio == geometry.compressionRatio,
            state.indexCompressorState.overlap,
            state.indexCompressorState.headDimension == geometry.indexHeadDimension,
            state.indexCompressorState.values.shape
                == [geometry.compressionRatio * 2, indexWidth.partialValue],
            state.indexCompressorState.scores.shape
                == state.indexCompressorState.values.shape
        else {
            throw DeepSeekV4Error.attention(
                "ratio-four decode state has incompatible geometry")
        }
    }

    private static func makeWindowCache(
        _ prompt: MLXArray, windowSize: Int
    ) -> MLXArray {
        let sequence = prompt.shape[0]
        let width = prompt.shape[1]
        if sequence >= windowSize {
            let recent = prompt[(sequence - windowSize)..., 0...]
            let cursor = sequence % windowSize
            guard cursor > 0 else { return recent }
            return concatenated([
                recent[(windowSize - cursor)..., 0...],
                recent[0..<(windowSize - cursor), 0...],
            ], axis: 0)
        }
        return concatenated([
            prompt,
            MLXArray.zeros([windowSize - sequence, width], dtype: prompt.dtype),
        ], axis: 0)
    }

    private static func replacingRow(
        _ array: MLXArray, at row: Int, with replacement: MLXArray
    ) -> MLXArray {
        var pieces = [MLXArray]()
        if row > 0 { pieces.append(array[0..<row, 0...]) }
        pieces.append(replacement)
        if row + 1 < array.shape[0] {
            pieces.append(array[(row + 1)..., 0...])
        }
        return concatenated(pieces, axis: 0)
    }

    private static func makeFixedCache(
        _ valid: MLXArray, capacity: Int, width: Int
    ) -> MLXArray {
        guard valid.shape[0] < capacity else { return valid }
        return concatenated([
            valid,
            MLXArray.zeros([capacity - valid.shape[0], width], dtype: valid.dtype),
        ], axis: 0)
    }

    private static func validPrefix(
        _ cache: MLXArray, count: Int, width: Int
    ) -> MLXArray {
        guard count > 0 else {
            return MLXArray.zeros([0, width], dtype: cache.dtype)
        }
        return cache[0..<count, 0...]
    }
}

/// DeepSeek V4's non-overlapping ratio-128 compressed-attention path.
///
/// Unlike ratio-four attention, this path has no learned lightning indexer.
/// Every completed 128-token compressor group is visible causally, exactly as
/// `get_compress_topk_idxs` in the published reference implementation defines.
/// Storage-backed projections remain outside this value so an artifact runner
/// can materialize one weight matrix at a time.
public enum DeepSeekV4CausalCompressedAttention {
    public struct Geometry: Sendable, Hashable {
        public let attentionHeads: Int
        public let headDimension: Int
        public let ropeDimension: Int
        public let slidingWindow: Int
        public let compressionRatio: Int
        public let maximumPositionCount: Int
        public let rmsNormEpsilon: Float
        public let rotary: DeepSeekV4RotaryParameters

        public init(
            attentionHeads: Int,
            headDimension: Int,
            ropeDimension: Int,
            slidingWindow: Int,
            compressionRatio: Int,
            maximumPositionCount: Int,
            rmsNormEpsilon: Float,
            rotary: DeepSeekV4RotaryParameters
        ) throws {
            let nonRotary = headDimension - ropeDimension
            guard attentionHeads > 0,
                headDimension > 0,
                ropeDimension > 0,
                ropeDimension <= headDimension,
                ropeDimension.isMultiple(of: 2),
                nonRotary.isMultiple(of: 64),
                slidingWindow > 0,
                compressionRatio == 128,
                maximumPositionCount >= slidingWindow,
                maximumPositionCount <= Int(Int32.max),
                rmsNormEpsilon.isFinite,
                rmsNormEpsilon > 0,
                rotary.dimension == ropeDimension
            else {
                throw DeepSeekV4Error.unsupportedArchitecture(
                    "ratio-128 causal compressed attention geometry is invalid")
            }
            self.attentionHeads = attentionHeads
            self.headDimension = headDimension
            self.ropeDimension = ropeDimension
            self.slidingWindow = slidingWindow
            self.compressionRatio = compressionRatio
            self.maximumPositionCount = maximumPositionCount
            self.rmsNormEpsilon = rmsNormEpsilon
            self.rotary = rotary
        }

        public init(config: DeepSeekV4Config, layer: Int) throws {
            let ratio = try config.compressionRatio(layer: layer)
            guard ratio == 128 else {
                throw DeepSeekV4Error.unsupportedArchitecture(
                    "layer \(layer) uses compression ratio \(ratio), not the causal ratio-128 path")
            }
            guard config.numberOfKeyValueHeads == 1 else {
                throw DeepSeekV4Error.unsupportedArchitecture(
                    "compressed attention requires one shared KV head")
            }
            try self.init(
                attentionHeads: config.numberOfAttentionHeads,
                headDimension: config.attentionHeadDimension,
                ropeDimension: config.ropeHeadDimension,
                slidingWindow: config.slidingWindow,
                compressionRatio: ratio,
                maximumPositionCount: config.maximumPositionCount,
                rmsNormEpsilon: config.rmsNormEpsilon,
                rotary: DeepSeekV4RotaryParameters(
                    dimension: config.ropeHeadDimension,
                    originalSequenceLength: config.ropeOriginalLength,
                    base: Double(config.compressionRopeTheta),
                    factor: Double(config.ropeFactor),
                    betaFast: Double(config.ropeBetaFast),
                    betaSlow: Double(config.ropeBetaSlow)))
        }
    }

    public struct ProjectedInputs {
        /// `wq_b(q_norm(wq_a(x)))`, flattened across attention heads.
        public let queries: MLXArray
        public let windowKeyValues: MLXArray
        public let compressorValues: MLXArray
        public let compressorScores: MLXArray

        public init(
            queries: MLXArray,
            windowKeyValues: MLXArray,
            compressorValues: MLXArray,
            compressorScores: MLXArray
        ) {
            self.queries = queries
            self.windowKeyValues = windowKeyValues
            self.compressorValues = compressorValues
            self.compressorScores = compressorScores
        }
    }

    public struct Parameters {
        public let keyValueNorm: MLXArray
        public let sinks: MLXArray
        public let compressorPositionalBias: MLXArray
        public let compressorNorm: MLXArray

        public init(
            keyValueNorm: MLXArray,
            sinks: MLXArray,
            compressorPositionalBias: MLXArray,
            compressorNorm: MLXArray
        ) {
            self.keyValueNorm = keyValueNorm
            self.sinks = sinks
            self.compressorPositionalBias = compressorPositionalBias
            self.compressorNorm = compressorNorm
        }
    }

    public struct State {
        fileprivate let nextPosition: Int
        fileprivate let positionLimit: Int
        fileprivate let completedCompressedCount: Int
        fileprivate let windowKeyValues: MLXArray
        fileprivate let compressedKeyValues: MLXArray
        fileprivate let compressorState: DeepSeekV4CompressorPool.State

        public var position: Int { nextPosition }
        public var decodePositionLimit: Int { positionLimit }
        public var compressedCount: Int { completedCompressedCount }
        public var compressedCapacity: Int { compressedKeyValues.shape[0] }
    }

    public struct PrefillResult {
        public let attention: MLXArray
        public let indices: [[Int]]
        public let compressorState: DeepSeekV4CompressorPool.State
        /// Present only when the caller declares a run-scoped position limit.
        /// Plain prefill does not reserve a model-maximum decode cache.
        public let state: State?
    }

    public struct DecodeResult {
        public let attention: MLXArray
        public let indices: [Int]
        public let state: State
    }

    public static func prefill(
        projected: ProjectedInputs,
        parameters: Parameters,
        geometry: Geometry,
        decodePositionLimit: Int? = nil,
        cancellationCheck: () throws -> Void = {},
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> PrefillResult {
        let sequence = try validate(
            projected: projected, parameters: parameters, geometry: geometry,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        guard sequence <= geometry.maximumPositionCount else {
            throw DeepSeekV4Error.attention(
                "prompt length \(sequence) exceeds the configured maximum "
                    + "\(geometry.maximumPositionCount)")
        }
        try cancellationCheck()
        let positions = Array(0..<sequence)

        var queries = projected.queries.asType(.float32).reshaped([
            sequence, geometry.attentionHeads, geometry.headDimension,
        ])
        queries = queries * rsqrt(
            mean(queries * queries, axis: -1, keepDims: true)
                + geometry.rmsNormEpsilon)
        queries = try rotateTail(
            queries, positions: positions, geometry: geometry, inverse: false)

        var windowKeyValues = K3Norm.rms(
            projected.windowKeyValues,
            weight: parameters.keyValueNorm,
            eps: geometry.rmsNormEpsilon)
        windowKeyValues = try rotateTail(
            windowKeyValues, positions: positions, geometry: geometry, inverse: false)
        windowKeyValues = try quantizeNonRotaryPrefix(
            windowKeyValues, geometry: geometry, phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        try cancellationCheck()

        let compressed = try DeepSeekV4CompressorPool.prefill(
            values: projected.compressorValues,
            scores: projected.compressorScores,
            positionalBias: parameters.compressorPositionalBias,
            ratio: geometry.compressionRatio,
            overlap: false)
        var compressedKeyValues: MLXArray
        if compressed.compressed.shape[0] == 0 {
            compressedKeyValues = MLXArray.zeros(
                [0, geometry.headDimension], dtype: .float32)
        } else {
            compressedKeyValues = K3Norm.rms(
                compressed.compressed,
                weight: parameters.compressorNorm,
                eps: geometry.rmsNormEpsilon)
        }
        let compressedPositions = (0..<(sequence / geometry.compressionRatio)).map {
            $0 * geometry.compressionRatio
        }
        if !compressedPositions.isEmpty {
            compressedKeyValues = try rotateTail(
                compressedKeyValues,
                positions: compressedPositions,
                geometry: geometry,
                inverse: false)
            compressedKeyValues = try quantizeNonRotaryPrefix(
                compressedKeyValues, geometry: geometry,
                phaseAccounting: phaseAccounting,
                diagnostics: diagnostics)
        }
        try cancellationCheck()

        let window = try DeepSeekV4AttentionIndices.window(
            windowSize: geometry.slidingWindow,
            sequenceLength: sequence,
            startPosition: 0)
        let causalCompressed = try DeepSeekV4AttentionIndices.compressed(
            ratio: geometry.compressionRatio,
            sequenceLength: sequence,
            startPosition: 0,
            offset: sequence)
        let indices = zip(window, causalCompressed).map(+)
        let keyValues = concatenated([windowKeyValues, compressedKeyValues], axis: 0)
        var attention = try DeepSeekV4SparseAttention.forward(
            queries: queries,
            keysAndValues: keyValues,
            sinks: parameters.sinks,
            indices: indices,
            scale: Float(1 / Foundation.sqrt(Double(geometry.headDimension))),
            phaseAccounting: phaseAccounting)
        attention = try rotateTail(
            attention, positions: positions, geometry: geometry, inverse: true)
        let state: State?
        if let decodePositionLimit {
            guard decodePositionLimit >= sequence,
                decodePositionLimit <= geometry.maximumPositionCount
            else {
                throw DeepSeekV4Error.attention(
                    "decode position limit \(decodePositionLimit) must contain the prompt "
                        + "and not exceed \(geometry.maximumPositionCount)")
            }
            let windowCache = makeWindowCache(
                windowKeyValues, windowSize: geometry.slidingWindow)
            let compressedCapacity =
                decodePositionLimit / geometry.compressionRatio
            let fixedCompressedCache = makeFixedCache(
                compressedKeyValues,
                capacity: compressedCapacity,
                width: geometry.headDimension)
            submittingToGPU(phaseAccounting, .blockCache) {
                MLX.asyncEval([attention, windowCache, fixedCompressedCache])
            }
            state = State(
                nextPosition: sequence,
                positionLimit: decodePositionLimit,
                completedCompressedCount: compressedKeyValues.shape[0],
                windowKeyValues: windowCache,
                compressedKeyValues: fixedCompressedCache,
                compressorState: compressed.state)
        } else {
            attention.eval()
            state = nil
        }
        try cancellationCheck()
        return PrefillResult(
            attention: attention,
            indices: indices,
            compressorState: compressed.state,
            state: state)
    }

    public static func decode(
        projected: ProjectedInputs,
        parameters: Parameters,
        geometry: Geometry,
        startPosition: Int,
        state: State,
        cancellationCheck: () throws -> Void = {},
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> DecodeResult {
        let sequence = try validate(
            projected: projected, parameters: parameters, geometry: geometry,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        guard sequence == 1 else {
            throw DeepSeekV4Error.attention(
                "incremental causal compressed attention accepts exactly one token")
        }
        guard startPosition == state.nextPosition, startPosition > 0 else {
            throw DeepSeekV4Error.attention(
                "decode position \(startPosition) does not follow cache position "
                    + "\(state.nextPosition)")
        }
        guard startPosition < state.positionLimit else {
            throw DeepSeekV4Error.attention(
                "decode position \(startPosition) reaches this run's limit "
                    + "\(state.positionLimit)")
        }
        try validate(state: state, geometry: geometry)
        try cancellationCheck()

        let positions = [startPosition]
        var queries = projected.queries.asType(.float32).reshaped([
            1, geometry.attentionHeads, geometry.headDimension,
        ])
        queries = queries * rsqrt(
            mean(queries * queries, axis: -1, keepDims: true)
                + geometry.rmsNormEpsilon)
        queries = try rotateTail(
            queries, positions: positions, geometry: geometry, inverse: false)

        var windowRow = K3Norm.rms(
            projected.windowKeyValues,
            weight: parameters.keyValueNorm,
            eps: geometry.rmsNormEpsilon)
        windowRow = try rotateTail(
            windowRow, positions: positions, geometry: geometry, inverse: false)
        windowRow = try quantizeNonRotaryPrefix(
            windowRow, geometry: geometry, phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        let windowCache = replacingRow(
            state.windowKeyValues,
            at: startPosition % geometry.slidingWindow,
            with: windowRow)
        try cancellationCheck()

        let compressedStep = try DeepSeekV4CompressorPool.decode(
            value: projected.compressorValues,
            score: projected.compressorScores,
            positionalBias: parameters.compressorPositionalBias,
            startPosition: startPosition,
            state: state.compressorState,
            phaseAccounting: phaseAccounting)
        var compressedCache = state.compressedKeyValues
        var compressedCount = state.completedCompressedCount
        if var row = compressedStep.compressed {
            row = K3Norm.rms(
                row, weight: parameters.compressorNorm,
                eps: geometry.rmsNormEpsilon)
            row = try rotateTail(
                row,
                positions: [startPosition + 1 - geometry.compressionRatio],
                geometry: geometry,
                inverse: false)
            row = try quantizeNonRotaryPrefix(
                row, geometry: geometry, phaseAccounting: phaseAccounting,
                diagnostics: diagnostics)
            compressedCache = replacingRow(
                compressedCache, at: compressedCount, with: row)
            compressedCount += 1
        }
        let expectedCompressed = (startPosition + 1) / geometry.compressionRatio
        guard compressedCount == expectedCompressed else {
            throw DeepSeekV4Error.attention(
                "causal compressed cache does not match the decode position")
        }
        try cancellationCheck()

        let window = try DeepSeekV4AttentionIndices.window(
            windowSize: geometry.slidingWindow,
            sequenceLength: 1,
            startPosition: startPosition)[0]
        let compressed = try DeepSeekV4AttentionIndices.compressed(
            ratio: geometry.compressionRatio,
            sequenceLength: 1,
            startPosition: startPosition,
            offset: geometry.slidingWindow)[0]
        let indices = window + compressed
        let keyValues = concatenated([
            windowCache,
            validPrefix(
                compressedCache, count: compressedCount, width: geometry.headDimension),
        ], axis: 0)
        var attention = try DeepSeekV4SparseAttention.forward(
            queries: queries,
            keysAndValues: keyValues,
            sinks: parameters.sinks,
            indices: [indices],
            scale: Float(1 / Foundation.sqrt(Double(geometry.headDimension))),
            phaseAccounting: phaseAccounting)
        attention = try rotateTail(
            attention, positions: positions, geometry: geometry, inverse: true)
        let next = State(
            nextPosition: startPosition + 1,
            positionLimit: state.positionLimit,
            completedCompressedCount: compressedCount,
            windowKeyValues: windowCache,
            compressedKeyValues: compressedCache,
            compressorState: compressedStep.state)
        submittingToGPU(phaseAccounting, .blockCache) {
            MLX.asyncEval([attention, windowCache, compressedCache])
        }
        try cancellationCheck()
        return DecodeResult(attention: attention, indices: indices, state: next)
    }

    private static func validate(
        projected: ProjectedInputs,
        parameters: Parameters,
        geometry: Geometry,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> Int {
        guard projected.queries.ndim == 2, projected.queries.shape[0] > 0 else {
            throw DeepSeekV4Error.attention(
                "causal compressed attention queries must be nonempty [sequence, heads * dimension]")
        }
        let sequence = projected.queries.shape[0]
        let queryWidth = geometry.attentionHeads.multipliedReportingOverflow(
            by: geometry.headDimension)
        guard !queryWidth.overflow,
            projected.queries.shape == [sequence, queryWidth.partialValue],
            projected.windowKeyValues.shape == [sequence, geometry.headDimension],
            projected.compressorValues.shape == [sequence, geometry.headDimension],
            projected.compressorScores.shape == projected.compressorValues.shape,
            parameters.keyValueNorm.shape == [geometry.headDimension],
            parameters.sinks.shape == [geometry.attentionHeads],
            parameters.compressorPositionalBias.shape
                == [geometry.compressionRatio, geometry.headDimension],
            parameters.compressorNorm.shape == [geometry.headDimension]
        else {
            throw DeepSeekV4Error.attention(
                "causal compressed attention projections or parameters have incompatible geometry")
        }
        let arrays = [
            projected.queries,
            projected.windowKeyValues,
            projected.compressorValues,
            projected.compressorScores,
            parameters.keyValueNorm,
            parameters.sinks,
            parameters.compressorPositionalBias,
            parameters.compressorNorm,
        ]
        guard arrays.allSatisfy({ $0.dtype.isFloatingPoint && !$0.dtype.isComplex }) else {
            throw DeepSeekV4Error.attention(
                "causal compressed attention inputs must be real floating point")
        }
        if diagnostics.validateFiniteness {
            try measuringPhase(
                phaseAccounting,
                excludingGPUBoundaryFrom: phaseAccounting?
                    .recordFinitenessSweep(nanoseconds:)
            ) {
                try waitingForGPU(phaseAccounting, .finitenessSweep) {
                    for array in arrays where !isFinite(array).all().item(Bool.self) {
                        throw DeepSeekV4Error.attention(
                            "causal compressed attention input contains a non-finite value")
                    }
                }
            }
        }
        return sequence
    }

    private static func quantizeNonRotaryPrefix(
        _ input: MLXArray,
        geometry: Geometry,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        let nonRotary = geometry.headDimension - geometry.ropeDimension
        guard nonRotary > 0 else { return input }
        let prefix = try DeepSeekV4FP8ActivationReference.quantizeDequantize(
            input[.ellipsis, 0..<nonRotary], blockSize: 64,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        return concatenated([prefix, input[.ellipsis, nonRotary...]], axis: -1)
    }

    private static func rotateTail(
        _ input: MLXArray,
        positions: [Int],
        geometry: Geometry,
        inverse: Bool
    ) throws -> MLXArray {
        let prefixWidth = input.shape[input.ndim - 1] - geometry.ropeDimension
        guard prefixWidth >= 0 else {
            throw DeepSeekV4Error.attention(
                "rotary tail exceeds the projected dimension")
        }
        let tail = try DeepSeekV4Rotary.apply(
            input[.ellipsis, prefixWidth...],
            positions: positions,
            parameters: geometry.rotary,
            inverse: inverse)
        guard prefixWidth > 0 else { return tail }
        return concatenated([input[.ellipsis, 0..<prefixWidth], tail], axis: -1)
    }

    private static func validate(state: State, geometry: Geometry) throws {
        let expectedCompressed = state.nextPosition / geometry.compressionRatio
        let compressedCapacity = state.positionLimit / geometry.compressionRatio
        guard state.nextPosition > 0,
            state.positionLimit > 0,
            state.positionLimit <= geometry.maximumPositionCount,
            state.nextPosition <= state.positionLimit,
            state.completedCompressedCount == expectedCompressed,
            state.windowKeyValues.shape
                == [geometry.slidingWindow, geometry.headDimension],
            state.compressedKeyValues.shape
                == [compressedCapacity, geometry.headDimension],
            state.compressorState.ratio == geometry.compressionRatio,
            !state.compressorState.overlap,
            state.compressorState.headDimension == geometry.headDimension,
            state.compressorState.values.shape
                == [geometry.compressionRatio, geometry.headDimension],
            state.compressorState.scores.shape == state.compressorState.values.shape
        else {
            throw DeepSeekV4Error.attention(
                "ratio-128 decode state has incompatible geometry")
        }
    }

    private static func makeWindowCache(
        _ prompt: MLXArray, windowSize: Int
    ) -> MLXArray {
        let sequence = prompt.shape[0]
        let width = prompt.shape[1]
        if sequence >= windowSize {
            let recent = prompt[(sequence - windowSize)..., 0...]
            let cursor = sequence % windowSize
            guard cursor > 0 else { return recent }
            return concatenated([
                recent[(windowSize - cursor)..., 0...],
                recent[0..<(windowSize - cursor), 0...],
            ], axis: 0)
        }
        return concatenated([
            prompt,
            MLXArray.zeros([windowSize - sequence, width], dtype: prompt.dtype),
        ], axis: 0)
    }

    private static func replacingRow(
        _ array: MLXArray, at row: Int, with replacement: MLXArray
    ) -> MLXArray {
        var pieces = [MLXArray]()
        if row > 0 { pieces.append(array[0..<row, 0...]) }
        pieces.append(replacement)
        if row + 1 < array.shape[0] {
            pieces.append(array[(row + 1)..., 0...])
        }
        return concatenated(pieces, axis: 0)
    }

    private static func makeFixedCache(
        _ valid: MLXArray, capacity: Int, width: Int
    ) -> MLXArray {
        guard valid.shape[0] < capacity else { return valid }
        return concatenated([
            valid,
            MLXArray.zeros([capacity - valid.shape[0], width], dtype: valid.dtype),
        ], axis: 0)
    }

    private static func validPrefix(
        _ cache: MLXArray, count: Int, width: Int
    ) -> MLXArray {
        guard count > 0 else {
            return MLXArray.zeros([0, width], dtype: cache.dtype)
        }
        return cache[0..<count, 0...]
    }
}
