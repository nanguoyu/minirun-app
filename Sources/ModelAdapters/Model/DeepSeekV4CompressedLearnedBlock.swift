import Foundation
import MLX

/// One DeepSeek V4 learned-routing layer with ratio-128 compressed attention.
///
/// Admission is derived from the verified `config.json` and layer manifest.
/// No repository revision is compiled into this adapter. Dense/equation-backed
/// and artifact-backed execution can retain a run-scoped attention state
/// across prompt prefill and one-token decode. The product runner remains a
/// separate, whole-model gate.
public enum DeepSeekV4CompressedLearnedBlock {
    public struct Geometry: Sendable, Hashable {
        public let layer: Int
        public let hiddenSize: Int
        public let hyperConnectionMultiplicity: Int
        public let hyperConnectionSinkhornIterations: Int
        public let hyperConnectionEpsilon: Float
        public let rmsNormEpsilon: Float

        public let queryRank: Int
        public let outputGroups: Int
        public let outputRank: Int
        public let attention: DeepSeekV4CausalCompressedAttention.Geometry

        public let expertCount: Int
        public let expertsPerToken: Int
        public let sharedIntermediateSize: Int
        public let normalizeRoutingWeights: Bool
        public let routingScale: Float
        public let swiGLULimit: Float

        public init(
            layer: Int,
            hiddenSize: Int,
            hyperConnectionMultiplicity: Int,
            hyperConnectionSinkhornIterations: Int,
            hyperConnectionEpsilon: Float,
            rmsNormEpsilon: Float,
            queryRank: Int,
            outputGroups: Int,
            outputRank: Int,
            attention: DeepSeekV4CausalCompressedAttention.Geometry,
            expertCount: Int,
            expertsPerToken: Int,
            sharedIntermediateSize: Int,
            normalizeRoutingWeights: Bool,
            routingScale: Float,
            swiGLULimit: Float
        ) throws {
            let headsPerGroup = attention.attentionHeads.quotientAndRemainder(
                dividingBy: outputGroups)
            let groupInput = headsPerGroup.quotient.multipliedReportingOverflow(
                by: attention.headDimension)
            let groupRows = outputGroups.multipliedReportingOverflow(by: outputRank)
            guard layer >= 0,
                hiddenSize > 0,
                hyperConnectionMultiplicity > 0,
                hyperConnectionSinkhornIterations > 0,
                hyperConnectionEpsilon.isFinite,
                hyperConnectionEpsilon > 0,
                rmsNormEpsilon.isFinite,
                rmsNormEpsilon > 0,
                queryRank > 0,
                outputGroups > 0,
                headsPerGroup.remainder == 0,
                outputRank > 0,
                !groupInput.overflow,
                !groupRows.overflow,
                hiddenSize.isMultiple(of: DeepSeekV4BlockFP8LinearReference.weightBlockSize),
                queryRank.isMultiple(of: DeepSeekV4BlockFP8LinearReference.weightBlockSize),
                groupInput.partialValue.isMultiple(
                    of: DeepSeekV4BlockFP8LinearReference.weightBlockSize),
                groupRows.partialValue.isMultiple(
                    of: DeepSeekV4BlockFP8LinearReference.weightBlockSize),
                sharedIntermediateSize.isMultiple(
                    of: DeepSeekV4BlockFP8LinearReference.weightBlockSize),
                expertCount > 1,
                expertsPerToken > 0,
                expertsPerToken < expertCount,
                routingScale.isFinite,
                routingScale > 0,
                swiGLULimit.isFinite,
                swiGLULimit >= 0
            else {
                throw DeepSeekV4Error.configuration(
                    "compressed learned-block geometry is invalid or cannot use 128x128 block-FP8")
            }
            self.layer = layer
            self.hiddenSize = hiddenSize
            self.hyperConnectionMultiplicity = hyperConnectionMultiplicity
            self.hyperConnectionSinkhornIterations = hyperConnectionSinkhornIterations
            self.hyperConnectionEpsilon = hyperConnectionEpsilon
            self.rmsNormEpsilon = rmsNormEpsilon
            self.queryRank = queryRank
            self.outputGroups = outputGroups
            self.outputRank = outputRank
            self.attention = attention
            self.expertCount = expertCount
            self.expertsPerToken = expertsPerToken
            self.sharedIntermediateSize = sharedIntermediateSize
            self.normalizeRoutingWeights = normalizeRoutingWeights
            self.routingScale = routingScale
            self.swiGLULimit = swiGLULimit
        }

        public init(config: DeepSeekV4Config, layer: Int) throws {
            guard !config.usesHashRouting(layer: layer) else {
                throw DeepSeekV4Error.unsupportedArchitecture(
                    "layer \(layer) uses token-hash routing, not learned routing")
            }
            guard config.sharedExpertCount == 1 else {
                throw DeepSeekV4Error.unsupportedArchitecture(
                    "compressed learned block requires exactly one shared expert")
            }
            let attention = try DeepSeekV4CausalCompressedAttention.Geometry(
                config: config, layer: layer)
            try self.init(
                layer: layer,
                hiddenSize: config.hiddenSize,
                hyperConnectionMultiplicity: config.hyperConnectionMultiplicity,
                hyperConnectionSinkhornIterations: config.hyperConnectionSinkhornIterations,
                hyperConnectionEpsilon: config.hyperConnectionEpsilon,
                rmsNormEpsilon: config.rmsNormEpsilon,
                queryRank: config.queryLowRank,
                outputGroups: config.outputGroups,
                outputRank: config.outputLowRank,
                attention: attention,
                expertCount: config.routedExpertCount,
                expertsPerToken: config.expertsPerToken,
                sharedIntermediateSize: config.expertIntermediateSize,
                normalizeRoutingWeights: config.normalizedTopK,
                routingScale: config.routingScale,
                swiGLULimit: config.swiGLULimit)
        }
    }

    /// Dense decoded weights used only by the equation-level reference path.
    public struct Weights {
        public let attentionNorm: MLXArray
        public let queryA: MLXArray
        public let queryNorm: MLXArray
        public let queryB: MLXArray
        public let windowKeyValue: MLXArray
        public let keyValueNorm: MLXArray
        public let sinks: MLXArray
        public let compressorValue: MLXArray
        public let compressorGate: MLXArray
        public let compressorPositionalBias: MLXArray
        public let compressorNorm: MLXArray
        public let outputGroup: MLXArray
        public let output: MLXArray
        public let attentionHyperFunction: MLXArray
        public let attentionHyperScale: MLXArray
        public let attentionHyperBase: MLXArray

        public let ffnNorm: MLXArray
        public let router: MLXArray
        public let correctionBias: MLXArray
        public let sharedGate: MLXArray
        public let sharedDown: MLXArray
        public let sharedUp: MLXArray
        public let ffnHyperFunction: MLXArray
        public let ffnHyperScale: MLXArray
        public let ffnHyperBase: MLXArray

        public init(
            attentionNorm: MLXArray,
            queryA: MLXArray,
            queryNorm: MLXArray,
            queryB: MLXArray,
            windowKeyValue: MLXArray,
            keyValueNorm: MLXArray,
            sinks: MLXArray,
            compressorValue: MLXArray,
            compressorGate: MLXArray,
            compressorPositionalBias: MLXArray,
            compressorNorm: MLXArray,
            outputGroup: MLXArray,
            output: MLXArray,
            attentionHyperFunction: MLXArray,
            attentionHyperScale: MLXArray,
            attentionHyperBase: MLXArray,
            ffnNorm: MLXArray,
            router: MLXArray,
            correctionBias: MLXArray,
            sharedGate: MLXArray,
            sharedDown: MLXArray,
            sharedUp: MLXArray,
            ffnHyperFunction: MLXArray,
            ffnHyperScale: MLXArray,
            ffnHyperBase: MLXArray
        ) {
            self.attentionNorm = attentionNorm
            self.queryA = queryA
            self.queryNorm = queryNorm
            self.queryB = queryB
            self.windowKeyValue = windowKeyValue
            self.keyValueNorm = keyValueNorm
            self.sinks = sinks
            self.compressorValue = compressorValue
            self.compressorGate = compressorGate
            self.compressorPositionalBias = compressorPositionalBias
            self.compressorNorm = compressorNorm
            self.outputGroup = outputGroup
            self.output = output
            self.attentionHyperFunction = attentionHyperFunction
            self.attentionHyperScale = attentionHyperScale
            self.attentionHyperBase = attentionHyperBase
            self.ffnNorm = ffnNorm
            self.router = router
            self.correctionBias = correctionBias
            self.sharedGate = sharedGate
            self.sharedDown = sharedDown
            self.sharedUp = sharedUp
            self.ffnHyperFunction = ffnHyperFunction
            self.ffnHyperScale = ffnHyperScale
            self.ffnHyperBase = ffnHyperBase
        }
    }

    public struct State {
        fileprivate let attention: DeepSeekV4CausalCompressedAttention.State
        fileprivate let geometry: Geometry
        fileprivate let artifactIdentity: ArtifactExecutionIdentity?

        public var position: Int { attention.position }
        public var decodePositionLimit: Int { attention.decodePositionLimit }
        public var compressedCount: Int { attention.compressedCount }
        public var compressedCapacity: Int { attention.compressedCapacity }
    }

    /// Run state is bound to the exact opened layer and expert containers that
    /// produced it. Provenance strings alone are not execution authority, and
    /// neither retained object contains decoded layer matrices.
    fileprivate struct ArtifactExecutionIdentity: Equatable {
        let artifact: DeepSeekV4LayerArtifact
        let expertContainers: RoutedExpertContainers

        init(
            artifact: DeepSeekV4LayerArtifact,
            expertUnit: DeepSeekV4ExpertUnit
        ) {
            self.artifact = artifact
            self.expertContainers = expertUnit.containers
        }

        static func == (
            lhs: ArtifactExecutionIdentity,
            rhs: ArtifactExecutionIdentity
        ) -> Bool {
            lhs.artifact === rhs.artifact
                && lhs.expertContainers === rhs.expertContainers
        }
    }

    public struct PrefillResult {
        public let residual: MLXArray
        public let routing: DeepSeekV4Router.Selection
        public let attentionIndices: [[Int]]
        /// Present only when the caller declares a run-scoped decode limit.
        public let state: State?
    }

    public struct DecodeResult {
        public let residual: MLXArray
        public let routing: DeepSeekV4Router.Selection
        public let state: State
    }

    private struct PlainWeights {
        let attentionNorm: MLXArray
        let queryNorm: MLXArray
        let keyValueNorm: MLXArray
        let sinks: MLXArray
        let compressorPositionalBias: MLXArray
        let compressorNorm: MLXArray
        let attentionHyperFunction: MLXArray
        let attentionHyperScale: MLXArray
        let attentionHyperBase: MLXArray
        let ffnNorm: MLXArray
        let router: MLXArray
        let correctionBias: MLXArray
        let ffnHyperFunction: MLXArray
        let ffnHyperScale: MLXArray
        let ffnHyperBase: MLXArray
    }

    private struct Projections {
        let queryA: (MLXArray) throws -> MLXArray
        let queryB: (MLXArray) throws -> MLXArray
        let windowKeyValue: (MLXArray) throws -> MLXArray
        let compressorValue: (MLXArray) throws -> MLXArray
        let compressorGate: (MLXArray) throws -> MLXArray
        let attentionOutput: (MLXArray) throws -> MLXArray
        let sharedGate: (MLXArray) throws -> MLXArray
        let sharedDown: (MLXArray) throws -> MLXArray
        let sharedUp: (MLXArray) throws -> MLXArray
    }

    private struct TensorShapes {
        let headsWidth: Int
        let compressorWidth: Int
        let groupInput: Int
        let groupRows: Int
        let flattenedResidual: Int
        let hyperWidth: Int

        init(_ geometry: Geometry) throws {
            func multiply(_ left: Int, _ right: Int, _ name: String) throws -> Int {
                let result = left.multipliedReportingOverflow(by: right)
                guard !result.overflow else {
                    throw DeepSeekV4Error.configuration(
                        "compressed learned-block \(name) overflows Int")
                }
                return result.partialValue
            }
            headsWidth = try multiply(
                geometry.attention.attentionHeads,
                geometry.attention.headDimension,
                "attention width")
            compressorWidth = geometry.attention.headDimension
            groupInput = try multiply(
                geometry.attention.attentionHeads / geometry.outputGroups,
                geometry.attention.headDimension,
                "grouped output input")
            groupRows = try multiply(
                geometry.outputGroups, geometry.outputRank,
                "grouped output rows")
            flattenedResidual = try multiply(
                geometry.hyperConnectionMultiplicity,
                geometry.hiddenSize,
                "flattened residual")
            let widened = geometry.hyperConnectionMultiplicity.addingReportingOverflow(2)
            guard !widened.overflow else {
                throw DeepSeekV4Error.configuration(
                    "compressed learned-block hyper width overflows Int")
            }
            hyperWidth = try multiply(
                geometry.hyperConnectionMultiplicity,
                widened.partialValue,
                "hyper width")
        }
    }

    public static func prefill(
        residual: MLXArray,
        weights: Weights,
        geometry: Geometry,
        routedExperts: RoutedExpertBackend,
        decodePositionLimit: Int? = nil,
        cancellationCheck: () throws -> Void = {}
    ) throws -> PrefillResult {
        let shapes = try TensorShapes(geometry)
        try validateDense(
            residual: residual,
            weights: weights,
            geometry: geometry,
            shapes: shapes)

        return try execute(
            residual: residual,
            plain: plainWeights(weights),
            projections: try denseProjections(
                weights: weights, geometry: geometry, shapes: shapes),
            geometry: geometry,
            routedExperts: routedExperts,
            decodePositionLimit: decodePositionLimit,
            artifactIdentity: nil,
            cancellationCheck: cancellationCheck)
    }

    /// Advance one token through a ratio-128 learned-routing block.
    public static func decode(
        residual: MLXArray,
        weights: Weights,
        geometry: Geometry,
        routedExperts: RoutedExpertBackend,
        startPosition: Int,
        state: State,
        cancellationCheck: () throws -> Void = {}
    ) throws -> DecodeResult {
        let shapes = try TensorShapes(geometry)
        try validateDense(
            residual: residual, weights: weights, geometry: geometry, shapes: shapes)
        return try executeDecode(
            residual: residual,
            plain: plainWeights(weights),
            projections: try denseProjections(
                weights: weights, geometry: geometry, shapes: shapes),
            geometry: geometry,
            routedExperts: routedExperts,
            startPosition: startPosition,
            state: state,
            artifactIdentity: nil,
            cancellationCheck: cancellationCheck)
    }

    private static func plainWeights(_ weights: Weights) -> PlainWeights {
        PlainWeights(
            attentionNorm: weights.attentionNorm,
            queryNorm: weights.queryNorm,
            keyValueNorm: weights.keyValueNorm,
            sinks: weights.sinks,
            compressorPositionalBias: weights.compressorPositionalBias,
            compressorNorm: weights.compressorNorm,
            attentionHyperFunction: weights.attentionHyperFunction,
            attentionHyperScale: weights.attentionHyperScale,
            attentionHyperBase: weights.attentionHyperBase,
            ffnNorm: weights.ffnNorm,
            router: weights.router,
            correctionBias: weights.correctionBias,
            ffnHyperFunction: weights.ffnHyperFunction,
            ffnHyperScale: weights.ffnHyperScale,
            ffnHyperBase: weights.ffnHyperBase)
    }

    private static func denseProjections(
        weights: Weights,
        geometry: Geometry,
        shapes: TensorShapes
    ) throws -> Projections {
        func fp8(_ input: MLXArray, _ weight: MLXArray) throws -> MLXArray {
            let rounded = try DeepSeekV4FP8ActivationReference.quantizeDequantize(
                input, blockSize: DeepSeekV4BlockFP8LinearReference.activationBlockSize)
            return k3Linear(rounded, weight)
        }
        return Projections(
            queryA: { try fp8($0, weights.queryA) },
            queryB: { try fp8($0, weights.queryB) },
            windowKeyValue: { try fp8($0, weights.windowKeyValue) },
            compressorValue: {
                k3Linear($0.asType(.float32), weights.compressorValue.asType(.float32))
            },
            compressorGate: {
                k3Linear($0.asType(.float32), weights.compressorGate.asType(.float32))
            },
            attentionOutput: { attention in
                let grouped = attention.asType(.float32).reshaped([
                    attention.shape[0], geometry.outputGroups, shapes.groupInput,
                ])
                let rounded = try DeepSeekV4FP8ActivationReference.quantizeDequantize(
                    grouped,
                    blockSize: DeepSeekV4BlockFP8LinearReference.activationBlockSize)
                var groups = [MLXArray]()
                groups.reserveCapacity(geometry.outputGroups)
                for group in 0..<geometry.outputGroups {
                    let rowStart = group * geometry.outputRank
                    groups.append(k3Linear(
                        rounded[0..., group, 0...],
                        weights.outputGroup[
                            rowStart..<(rowStart + geometry.outputRank), 0...]))
                }
                return try fp8(concatenated(groups, axis: -1), weights.output)
            },
            sharedGate: { try fp8($0, weights.sharedGate) },
            sharedDown: { try fp8($0, weights.sharedDown) },
            sharedUp: { try fp8($0, weights.sharedUp) })
    }

    /// Execute one verified ratio-128 learned-routing layer directly from its
    /// published layer and expert units. The complete tensor contract is
    /// reconciled before the first payload or expert read.
    public static func prefill(
        residual: MLXArray,
        artifact: DeepSeekV4LayerArtifact,
        expertUnit: DeepSeekV4ExpertUnit,
        geometry: Geometry,
        routedExperts: PagedRoutedExpertBackend,
        decodePositionLimit: Int? = nil,
        stream: StreamOrDevice = .default,
        cancellationCheck: @escaping () throws -> Void = {}
    ) throws -> PrefillResult {
        try withArtifactExecution(
            residual: residual,
            artifact: artifact,
            expertUnit: expertUnit,
            geometry: geometry,
            routedExperts: routedExperts,
            stream: stream,
            cancellationCheck: cancellationCheck,
            beforePayload: { _ in
                try validateDecodePositionLimit(
                    promptCount: residual.shape[0],
                    decodePositionLimit: decodePositionLimit,
                    maximumPositionCount: geometry.attention.maximumPositionCount)
            },
            body: { plain, projections, identity in
                try execute(
                    residual: residual,
                    plain: plain,
                    projections: projections,
                    geometry: geometry,
                    routedExperts: routedExperts,
                    decodePositionLimit: decodePositionLimit,
                    artifactIdentity: identity,
                    cancellationCheck: cancellationCheck,
                    phaseAccounting: artifact.phaseAccounting,
                    diagnostics: artifact.diagnostics)
            })
    }

    /// Advance one token through the same verified ratio-128 layer capability
    /// that created the compressed-attention cache. Large projections are
    /// read, evaluated, and released once for this layer step.
    public static func decode(
        residual: MLXArray,
        artifact: DeepSeekV4LayerArtifact,
        expertUnit: DeepSeekV4ExpertUnit,
        geometry: Geometry,
        routedExperts: PagedRoutedExpertBackend,
        startPosition: Int,
        state: State,
        stream: StreamOrDevice = .default,
        cancellationCheck: @escaping () throws -> Void = {}
    ) throws -> DecodeResult {
        try withArtifactExecution(
            residual: residual,
            artifact: artifact,
            expertUnit: expertUnit,
            geometry: geometry,
            routedExperts: routedExperts,
            stream: stream,
            cancellationCheck: cancellationCheck,
            beforePayload: { identity in
                try validateDecodeRequest(
                    residual: residual,
                    startPosition: startPosition,
                    state: state,
                    geometry: geometry,
                    artifactIdentity: identity)
            },
            body: { plain, projections, identity in
                try executeDecode(
                    residual: residual,
                    plain: plain,
                    projections: projections,
                    geometry: geometry,
                    routedExperts: routedExperts,
                    startPosition: startPosition,
                    state: state,
                    artifactIdentity: identity,
                    cancellationCheck: cancellationCheck,
                    phaseAccounting: artifact.phaseAccounting,
                    diagnostics: artifact.diagnostics)
            })
    }

    private static func withArtifactExecution<Result>(
        residual: MLXArray,
        artifact: DeepSeekV4LayerArtifact,
        expertUnit: DeepSeekV4ExpertUnit,
        geometry: Geometry,
        routedExperts: PagedRoutedExpertBackend,
        stream: StreamOrDevice,
        cancellationCheck: @escaping () throws -> Void,
        beforePayload: (ArtifactExecutionIdentity) throws -> Void,
        body: (
            PlainWeights,
            Projections,
            ArtifactExecutionIdentity
        ) throws -> Result
    ) throws -> Result {
        let shapes = try TensorShapes(geometry)
        try validateInput(residual: residual, geometry: geometry)
        try validateArtifactContract(
            artifact: artifact, expertUnit: expertUnit, geometry: geometry)
        try validateBackendIdentity(
            expertUnit: expertUnit, routedExperts: routedExperts)
        let identity = ArtifactExecutionIdentity(
            artifact: artifact, expertUnit: expertUnit)
        try beforePayload(identity)

        let loaded = try measuringLayerArtifactSetup(artifact.phaseAccounting) {
            try loadArtifactExecution(
                artifact: artifact,
                geometry: geometry,
                shapes: shapes,
                stream: stream,
                cancellationCheck: cancellationCheck)
        }
        return try body(loaded.plain, loaded.projections, identity)
    }

    /// One layer's plain tensors, `PlainWeights` and projection closures.
    ///
    /// Extracted from ``withArtifactExecution`` so the setup has a scope of
    /// its own to bracket. ``measuringLayerArtifactSetup(_:_:)`` subtracts the
    /// deterministic reads and tile digests performed in here, which already
    /// have terms of their own, and reports what remains: manifest lookups,
    /// the token table's validation, MLX array construction from the read
    /// bytes, and the closure allocation below.
    private static func loadArtifactExecution(
        artifact: DeepSeekV4LayerArtifact,
        geometry: Geometry,
        shapes: TensorShapes,
        stream: StreamOrDevice,
        cancellationCheck: @escaping () throws -> Void
    ) throws -> (plain: PlainWeights, projections: Projections) {
        let prefix = "layers.\(geometry.layer)."
        try cancellationCheck()

        let plain = try PlainWeights(
            attentionNorm: artifact.loadFloatingTensor(
                tensor: "\(prefix)attn_norm.weight",
                expectedDType: .bfloat16,
                expectedShape: [geometry.hiddenSize],
                cancellationCheck: cancellationCheck),
            queryNorm: artifact.loadFloatingTensor(
                tensor: "\(prefix)attn.q_norm.weight",
                expectedDType: .bfloat16,
                expectedShape: [geometry.queryRank],
                cancellationCheck: cancellationCheck),
            keyValueNorm: artifact.loadFloatingTensor(
                tensor: "\(prefix)attn.kv_norm.weight",
                expectedDType: .bfloat16,
                expectedShape: [geometry.attention.headDimension],
                cancellationCheck: cancellationCheck),
            sinks: artifact.loadFloatingTensor(
                tensor: "\(prefix)attn.attn_sink",
                expectedDType: .float32,
                expectedShape: [geometry.attention.attentionHeads],
                cancellationCheck: cancellationCheck),
            compressorPositionalBias: artifact.loadFloatingTensor(
                tensor: "\(prefix)attn.compressor.ape",
                expectedDType: .float32,
                expectedShape: [
                    geometry.attention.compressionRatio, shapes.compressorWidth,
                ],
                cancellationCheck: cancellationCheck),
            compressorNorm: artifact.loadFloatingTensor(
                tensor: "\(prefix)attn.compressor.norm.weight",
                expectedDType: .bfloat16,
                expectedShape: [geometry.attention.headDimension],
                cancellationCheck: cancellationCheck),
            attentionHyperFunction: artifact.loadFloatingTensor(
                tensor: "\(prefix)hc_attn_fn",
                expectedDType: .float32,
                expectedShape: [shapes.hyperWidth, shapes.flattenedResidual],
                cancellationCheck: cancellationCheck),
            attentionHyperScale: artifact.loadFloatingTensor(
                tensor: "\(prefix)hc_attn_scale",
                expectedDType: .float32,
                expectedShape: [3],
                cancellationCheck: cancellationCheck),
            attentionHyperBase: artifact.loadFloatingTensor(
                tensor: "\(prefix)hc_attn_base",
                expectedDType: .float32,
                expectedShape: [shapes.hyperWidth],
                cancellationCheck: cancellationCheck),
            ffnNorm: artifact.loadFloatingTensor(
                tensor: "\(prefix)ffn_norm.weight",
                expectedDType: .bfloat16,
                expectedShape: [geometry.hiddenSize],
                cancellationCheck: cancellationCheck),
            router: artifact.loadFloatingTensor(
                tensor: "\(prefix)ffn.gate.weight",
                expectedDType: .bfloat16,
                expectedShape: [geometry.expertCount, geometry.hiddenSize],
                cancellationCheck: cancellationCheck),
            correctionBias: artifact.loadFloatingTensor(
                tensor: "\(prefix)ffn.gate.bias",
                expectedDType: .float32,
                expectedShape: [geometry.expertCount],
                cancellationCheck: cancellationCheck),
            ffnHyperFunction: artifact.loadFloatingTensor(
                tensor: "\(prefix)hc_ffn_fn",
                expectedDType: .float32,
                expectedShape: [shapes.hyperWidth, shapes.flattenedResidual],
                cancellationCheck: cancellationCheck),
            ffnHyperScale: artifact.loadFloatingTensor(
                tensor: "\(prefix)hc_ffn_scale",
                expectedDType: .float32,
                expectedShape: [3],
                cancellationCheck: cancellationCheck),
            ffnHyperBase: artifact.loadFloatingTensor(
                tensor: "\(prefix)hc_ffn_base",
                expectedDType: .float32,
                expectedShape: [shapes.hyperWidth],
                cancellationCheck: cancellationCheck))

        func fp8Projection(
            _ tensor: String, outFeatures: Int, inFeatures: Int
        ) -> (MLXArray) throws -> MLXArray {
            { input in
                try artifact.projectBlockFP8Reference(
                    input,
                    tensor: tensor,
                    outFeatures: outFeatures,
                    inFeatures: inFeatures,
                    stream: stream,
                    cancellationCheck: cancellationCheck)
            }
        }
        func bf16Projection(
            _ tensor: String,
            outFeatures: Int,
            inFeatures: Int
        ) -> (MLXArray) throws -> MLXArray {
            { input in
                try cancellationCheck()
                let output = try autoreleasepool { () throws -> MLXArray in
                    let weights = try artifact.loadFloatingTensor(
                        tensor: tensor,
                        expectedDType: .bfloat16,
                        expectedShape: [outFeatures, inFeatures],
                        cancellationCheck: cancellationCheck)
                    let projected = k3Linear(
                        input.asType(.float32), weights.asType(.float32))
                    projected.eval()
                    return projected
                }
                try cancellationCheck()
                return output
            }
        }
        let projections = Projections(
            queryA: fp8Projection(
                "\(prefix)attn.wq_a.weight",
                outFeatures: geometry.queryRank,
                inFeatures: geometry.hiddenSize),
            queryB: fp8Projection(
                "\(prefix)attn.wq_b.weight",
                outFeatures: shapes.headsWidth,
                inFeatures: geometry.queryRank),
            windowKeyValue: fp8Projection(
                "\(prefix)attn.wkv.weight",
                outFeatures: geometry.attention.headDimension,
                inFeatures: geometry.hiddenSize),
            compressorValue: bf16Projection(
                "\(prefix)attn.compressor.wkv.weight",
                outFeatures: shapes.compressorWidth,
                inFeatures: geometry.hiddenSize),
            compressorGate: bf16Projection(
                "\(prefix)attn.compressor.wgate.weight",
                outFeatures: shapes.compressorWidth,
                inFeatures: geometry.hiddenSize),
            attentionOutput: { attention in
                let grouped = attention.asType(.float32).reshaped([
                    attention.shape[0], geometry.outputGroups, shapes.groupInput,
                ])
                let latent = try artifact.projectGroupedBlockFP8Reference(
                    grouped,
                    tensor: "\(prefix)attn.wo_a.weight",
                    outFeatures: shapes.groupRows,
                    inFeatures: shapes.groupInput,
                    groups: geometry.outputGroups,
                    stream: stream,
                    cancellationCheck: cancellationCheck)
                return try artifact.projectBlockFP8Reference(
                    latent,
                    tensor: "\(prefix)attn.wo_b.weight",
                    outFeatures: geometry.hiddenSize,
                    inFeatures: shapes.groupRows,
                    stream: stream,
                    cancellationCheck: cancellationCheck)
            },
            sharedGate: fp8Projection(
                "\(prefix)ffn.shared_experts.w1.weight",
                outFeatures: geometry.sharedIntermediateSize,
                inFeatures: geometry.hiddenSize),
            sharedDown: fp8Projection(
                "\(prefix)ffn.shared_experts.w2.weight",
                outFeatures: geometry.hiddenSize,
                inFeatures: geometry.sharedIntermediateSize),
            sharedUp: fp8Projection(
                "\(prefix)ffn.shared_experts.w3.weight",
                outFeatures: geometry.sharedIntermediateSize,
                inFeatures: geometry.hiddenSize))
        return (plain, projections)
    }

    private static func execute(
        residual: MLXArray,
        plain: PlainWeights,
        projections: Projections,
        geometry: Geometry,
        routedExperts: RoutedExpertBackend,
        decodePositionLimit: Int? = nil,
        artifactIdentity: ArtifactExecutionIdentity?,
        cancellationCheck: () throws -> Void,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> PrefillResult {
        try cancellationCheck()
        let attentionPrepared = try DeepSeekV4HyperConnections.prepare(
            residual: residual,
            function: plain.attentionHyperFunction,
            scale: plain.attentionHyperScale,
            base: plain.attentionHyperBase,
            multiplicity: geometry.hyperConnectionMultiplicity,
            iterations: geometry.hyperConnectionSinkhornIterations,
            epsilon: geometry.hyperConnectionEpsilon,
            normEpsilon: geometry.rmsNormEpsilon)
        let attentionInput = K3Norm.rms(
            attentionPrepared.input,
            weight: plain.attentionNorm,
            eps: geometry.rmsNormEpsilon)

        let queryLatent = K3Norm.rms(
            try projections.queryA(attentionInput),
            weight: plain.queryNorm,
            eps: geometry.rmsNormEpsilon)
        try cancellationCheck()
        let attended = try DeepSeekV4CausalCompressedAttention.prefill(
            projected: .init(
                queries: try projections.queryB(queryLatent),
                windowKeyValues: try projections.windowKeyValue(attentionInput),
                compressorValues: try projections.compressorValue(attentionInput),
                compressorScores: try projections.compressorGate(attentionInput)),
            parameters: .init(
                keyValueNorm: plain.keyValueNorm,
                sinks: plain.sinks,
                compressorPositionalBias: plain.compressorPositionalBias,
                compressorNorm: plain.compressorNorm),
            geometry: geometry.attention,
            decodePositionLimit: decodePositionLimit,
            cancellationCheck: cancellationCheck,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        try cancellationCheck()
        let attentionBranch = try projections.attentionOutput(attended.attention)
        let afterAttention = try DeepSeekV4HyperConnections.combine(
            branch: attentionBranch,
            residual: residual,
            split: attentionPrepared.split)

        let ffn = try DeepSeekV4MoEFFN.execute(
            afterAttention: afterAttention,
            routingChoice: .learned(correctionBias: plain.correctionBias),
            ffnNorm: plain.ffnNorm,
            router: plain.router,
            hyperFunction: plain.ffnHyperFunction,
            hyperScale: plain.ffnHyperScale,
            hyperBase: plain.ffnHyperBase,
            sharedGate: projections.sharedGate,
            sharedDown: projections.sharedDown,
            sharedUp: projections.sharedUp,
            geometry: .init(
                layer: geometry.layer,
                hiddenSize: geometry.hiddenSize,
                hyperConnectionMultiplicity: geometry.hyperConnectionMultiplicity,
                hyperConnectionSinkhornIterations: geometry.hyperConnectionSinkhornIterations,
                hyperConnectionEpsilon: geometry.hyperConnectionEpsilon,
                rmsNormEpsilon: geometry.rmsNormEpsilon,
                expertCount: geometry.expertCount,
                expertsPerToken: geometry.expertsPerToken,
                sharedIntermediateSize: geometry.sharedIntermediateSize,
                normalizeRoutingWeights: geometry.normalizeRoutingWeights,
                routingScale: geometry.routingScale,
                swiGLULimit: geometry.swiGLULimit),
            routedExperts: routedExperts,
            cancellationCheck: cancellationCheck,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        return PrefillResult(
            residual: ffn.residual,
            routing: ffn.routing,
            attentionIndices: attended.indices,
            state: attended.state.map {
                State(
                    attention: $0,
                    geometry: geometry,
                    artifactIdentity: artifactIdentity)
            })
    }

    private static func executeDecode(
        residual: MLXArray,
        plain: PlainWeights,
        projections: Projections,
        geometry: Geometry,
        routedExperts: RoutedExpertBackend,
        startPosition: Int,
        state: State,
        artifactIdentity: ArtifactExecutionIdentity?,
        cancellationCheck: () throws -> Void,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> DecodeResult {
        try validateDecodeRequest(
            residual: residual,
            startPosition: startPosition,
            state: state,
            geometry: geometry,
            artifactIdentity: artifactIdentity)
        try cancellationCheck()
        let attentionPrepared = try DeepSeekV4HyperConnections.prepare(
            residual: residual,
            function: plain.attentionHyperFunction,
            scale: plain.attentionHyperScale,
            base: plain.attentionHyperBase,
            multiplicity: geometry.hyperConnectionMultiplicity,
            iterations: geometry.hyperConnectionSinkhornIterations,
            epsilon: geometry.hyperConnectionEpsilon,
            normEpsilon: geometry.rmsNormEpsilon)
        let attentionInput = K3Norm.rms(
            attentionPrepared.input,
            weight: plain.attentionNorm,
            eps: geometry.rmsNormEpsilon)
        let queryLatent = K3Norm.rms(
            try projections.queryA(attentionInput),
            weight: plain.queryNorm,
            eps: geometry.rmsNormEpsilon)
        try cancellationCheck()
        let attended = try DeepSeekV4CausalCompressedAttention.decode(
            projected: .init(
                queries: try projections.queryB(queryLatent),
                windowKeyValues: try projections.windowKeyValue(attentionInput),
                compressorValues: try projections.compressorValue(attentionInput),
                compressorScores: try projections.compressorGate(attentionInput)),
            parameters: .init(
                keyValueNorm: plain.keyValueNorm,
                sinks: plain.sinks,
                compressorPositionalBias: plain.compressorPositionalBias,
                compressorNorm: plain.compressorNorm),
            geometry: geometry.attention,
            startPosition: startPosition,
            state: state.attention,
            cancellationCheck: cancellationCheck,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        try cancellationCheck()
        let attentionBranch = try projections.attentionOutput(attended.attention)
        let afterAttention = try DeepSeekV4HyperConnections.combine(
            branch: attentionBranch,
            residual: residual,
            split: attentionPrepared.split)
        let ffn = try DeepSeekV4MoEFFN.execute(
            afterAttention: afterAttention,
            routingChoice: .learned(correctionBias: plain.correctionBias),
            ffnNorm: plain.ffnNorm,
            router: plain.router,
            hyperFunction: plain.ffnHyperFunction,
            hyperScale: plain.ffnHyperScale,
            hyperBase: plain.ffnHyperBase,
            sharedGate: projections.sharedGate,
            sharedDown: projections.sharedDown,
            sharedUp: projections.sharedUp,
            geometry: .init(
                layer: geometry.layer,
                hiddenSize: geometry.hiddenSize,
                hyperConnectionMultiplicity: geometry.hyperConnectionMultiplicity,
                hyperConnectionSinkhornIterations: geometry.hyperConnectionSinkhornIterations,
                hyperConnectionEpsilon: geometry.hyperConnectionEpsilon,
                rmsNormEpsilon: geometry.rmsNormEpsilon,
                expertCount: geometry.expertCount,
                expertsPerToken: geometry.expertsPerToken,
                sharedIntermediateSize: geometry.sharedIntermediateSize,
                normalizeRoutingWeights: geometry.normalizeRoutingWeights,
                routingScale: geometry.routingScale,
                swiGLULimit: geometry.swiGLULimit),
            routedExperts: routedExperts,
            cancellationCheck: cancellationCheck,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        return DecodeResult(
            residual: ffn.residual,
            routing: ffn.routing,
            state: State(
                attention: attended.state,
                geometry: geometry,
                artifactIdentity: artifactIdentity))
    }

    private static func validateDecodeRequest(
        residual: MLXArray,
        startPosition: Int,
        state: State,
        geometry: Geometry,
        artifactIdentity: ArtifactExecutionIdentity?
    ) throws {
        guard residual.ndim == 3,
            residual.shape[0] == 1,
            residual.shape[1] == geometry.hyperConnectionMultiplicity,
            residual.shape[2] == geometry.hiddenSize
        else {
            throw DeepSeekV4Error.configuration(
                "incremental ratio-128 execution accepts exactly one compatible residual row")
        }
        guard startPosition == state.position, startPosition > 0 else {
            throw DeepSeekV4Error.attention(
                "decode position \(startPosition) does not follow cache position "
                    + "\(state.position)")
        }
        guard startPosition < state.decodePositionLimit else {
            throw DeepSeekV4Error.attention(
                "decode position \(startPosition) reaches this run's limit "
                    + "\(state.decodePositionLimit)")
        }
        guard state.geometry == geometry else {
            throw DeepSeekV4Error.attention(
                "ratio-128 decode state has incompatible geometry")
        }
        guard state.artifactIdentity == artifactIdentity else {
            throw DeepSeekV4Error.attention(
                "ratio-128 decode state has incompatible artifact identity")
        }
    }

    private static func validateDecodePositionLimit(
        promptCount: Int,
        decodePositionLimit: Int?,
        maximumPositionCount: Int
    ) throws {
        guard let decodePositionLimit else { return }
        guard decodePositionLimit >= promptCount,
            decodePositionLimit <= maximumPositionCount
        else {
            throw DeepSeekV4Error.attention(
                "decode position limit \(decodePositionLimit) must contain the prompt "
                    + "and not exceed \(maximumPositionCount)")
        }
    }

    private static func validateInput(
        residual: MLXArray,
        geometry: Geometry
    ) throws {
        guard residual.ndim == 3,
            residual.shape[0] > 0,
            residual.shape[1] == geometry.hyperConnectionMultiplicity,
            residual.shape[2] == geometry.hiddenSize,
            residual.dtype.isFloatingPoint,
            !residual.dtype.isComplex
        else {
            throw DeepSeekV4Error.configuration(
                "compressed learned-block residual must be nonempty "
                    + "[sequence, multiplicity, hidden]")
        }
    }

    private static func validateDense(
        residual: MLXArray,
        weights: Weights,
        geometry: Geometry,
        shapes: TensorShapes
    ) throws {
        try validateInput(residual: residual, geometry: geometry)
        let expected: [(String, MLXArray, [Int])] = [
            ("attention norm", weights.attentionNorm, [geometry.hiddenSize]),
            ("query A", weights.queryA, [geometry.queryRank, geometry.hiddenSize]),
            ("query norm", weights.queryNorm, [geometry.queryRank]),
            ("query B", weights.queryB, [shapes.headsWidth, geometry.queryRank]),
            ("window key/value", weights.windowKeyValue,
                [geometry.attention.headDimension, geometry.hiddenSize]),
            ("key/value norm", weights.keyValueNorm,
                [geometry.attention.headDimension]),
            ("attention sinks", weights.sinks,
                [geometry.attention.attentionHeads]),
            ("compressor value", weights.compressorValue,
                [shapes.compressorWidth, geometry.hiddenSize]),
            ("compressor gate", weights.compressorGate,
                [shapes.compressorWidth, geometry.hiddenSize]),
            ("compressor positional bias", weights.compressorPositionalBias,
                [geometry.attention.compressionRatio, shapes.compressorWidth]),
            ("compressor norm", weights.compressorNorm,
                [geometry.attention.headDimension]),
            ("output group", weights.outputGroup,
                [shapes.groupRows, shapes.groupInput]),
            ("output", weights.output,
                [geometry.hiddenSize, shapes.groupRows]),
            ("attention hyper function", weights.attentionHyperFunction,
                [shapes.hyperWidth, shapes.flattenedResidual]),
            ("attention hyper scale", weights.attentionHyperScale, [3]),
            ("attention hyper base", weights.attentionHyperBase,
                [shapes.hyperWidth]),
            ("FFN norm", weights.ffnNorm, [geometry.hiddenSize]),
            ("router", weights.router,
                [geometry.expertCount, geometry.hiddenSize]),
            ("router correction bias", weights.correctionBias,
                [geometry.expertCount]),
            ("shared gate", weights.sharedGate,
                [geometry.sharedIntermediateSize, geometry.hiddenSize]),
            ("shared down", weights.sharedDown,
                [geometry.hiddenSize, geometry.sharedIntermediateSize]),
            ("shared up", weights.sharedUp,
                [geometry.sharedIntermediateSize, geometry.hiddenSize]),
            ("FFN hyper function", weights.ffnHyperFunction,
                [shapes.hyperWidth, shapes.flattenedResidual]),
            ("FFN hyper scale", weights.ffnHyperScale, [3]),
            ("FFN hyper base", weights.ffnHyperBase,
                [shapes.hyperWidth]),
        ]
        for (name, value, shape) in expected {
            guard value.shape == shape,
                value.dtype.isFloatingPoint,
                !value.dtype.isComplex
            else {
                throw DeepSeekV4Error.configuration(
                    "\(name) must be real floating point \(shape), got "
                        + "\(value.shape) \(value.dtype)")
            }
        }
    }

    private struct ContractKey: Hashable {
        let expertUnit: ObjectIdentifier
        let geometry: Geometry
    }

    /// Reconcile one learned ratio-128 layer without reading payload bytes.
    /// Full-model admission and execution deliberately share this function.
    ///
    /// The artifact records one admission per (expert unit, geometry) request:
    /// the check reads only this artifact's immutable manifest and the
    /// caller's geometry, so repeating it per token cannot reach a different
    /// answer.
    static func validateArtifactContract(
        artifact: DeepSeekV4LayerArtifact,
        expertUnit: DeepSeekV4ExpertUnit,
        geometry: Geometry
    ) throws {
        try artifact.admitContractOnce(
            key: ContractKey(
                expertUnit: ObjectIdentifier(expertUnit), geometry: geometry),
            participants: [expertUnit]
        ) {
            try admitArtifactContract(
                artifact: artifact, expertUnit: expertUnit, geometry: geometry)
        }
    }

    private static func admitArtifactContract(
        artifact: DeepSeekV4LayerArtifact,
        expertUnit: DeepSeekV4ExpertUnit,
        geometry: Geometry
    ) throws {
        let shapes = try TensorShapes(geometry)
        let expectedExpertGeometry = try DeepSeekV4ExpertGeometry(
            hiddenSize: geometry.hiddenSize,
            intermediateSize: geometry.sharedIntermediateSize,
            expertCount: geometry.expertCount)
        guard artifact.layer == geometry.layer,
            expertUnit.layer == geometry.layer,
            artifact.unitID == expertUnit.unitID,
            artifact.sourceRepository == expertUnit.sourceRepository,
            artifact.sourceRevision == expertUnit.sourceRevision,
            expertUnit.geometry == expectedExpertGeometry
        else {
            throw DeepSeekV4Error.artifact(
                "layer manifest, expert unit, and compressed learned geometry "
                    + "do not share one identity")
        }
        let prefix = "layers.\(geometry.layer)."
        try artifact.validateTensorContract(
            blockFP8: [
                "\(prefix)attn.wq_a.weight": [geometry.queryRank, geometry.hiddenSize],
                "\(prefix)attn.wq_b.weight": [shapes.headsWidth, geometry.queryRank],
                "\(prefix)attn.wkv.weight": [
                    geometry.attention.headDimension, geometry.hiddenSize,
                ],
                "\(prefix)attn.wo_a.weight": [shapes.groupRows, shapes.groupInput],
                "\(prefix)attn.wo_b.weight": [geometry.hiddenSize, shapes.groupRows],
                "\(prefix)ffn.shared_experts.w1.weight": [
                    geometry.sharedIntermediateSize, geometry.hiddenSize,
                ],
                "\(prefix)ffn.shared_experts.w2.weight": [
                    geometry.hiddenSize, geometry.sharedIntermediateSize,
                ],
                "\(prefix)ffn.shared_experts.w3.weight": [
                    geometry.sharedIntermediateSize, geometry.hiddenSize,
                ],
            ],
            plain: [
                "\(prefix)attn_norm.weight": .init(
                    dtype: .bfloat16, shape: [geometry.hiddenSize]),
                "\(prefix)attn.q_norm.weight": .init(
                    dtype: .bfloat16, shape: [geometry.queryRank]),
                "\(prefix)attn.kv_norm.weight": .init(
                    dtype: .bfloat16, shape: [geometry.attention.headDimension]),
                "\(prefix)attn.attn_sink": .init(
                    dtype: .float32, shape: [geometry.attention.attentionHeads]),
                "\(prefix)attn.compressor.ape": .init(
                    dtype: .float32,
                    shape: [
                        geometry.attention.compressionRatio, shapes.compressorWidth,
                    ]),
                "\(prefix)attn.compressor.norm.weight": .init(
                    dtype: .bfloat16, shape: [geometry.attention.headDimension]),
                "\(prefix)attn.compressor.wkv.weight": .init(
                    dtype: .bfloat16,
                    shape: [shapes.compressorWidth, geometry.hiddenSize]),
                "\(prefix)attn.compressor.wgate.weight": .init(
                    dtype: .bfloat16,
                    shape: [shapes.compressorWidth, geometry.hiddenSize]),
                "\(prefix)hc_attn_fn": .init(
                    dtype: .float32,
                    shape: [shapes.hyperWidth, shapes.flattenedResidual]),
                "\(prefix)hc_attn_scale": .init(dtype: .float32, shape: [3]),
                "\(prefix)hc_attn_base": .init(
                    dtype: .float32, shape: [shapes.hyperWidth]),
                "\(prefix)ffn_norm.weight": .init(
                    dtype: .bfloat16, shape: [geometry.hiddenSize]),
                "\(prefix)ffn.gate.weight": .init(
                    dtype: .bfloat16,
                    shape: [geometry.expertCount, geometry.hiddenSize]),
                "\(prefix)ffn.gate.bias": .init(
                    dtype: .float32, shape: [geometry.expertCount]),
                "\(prefix)hc_ffn_fn": .init(
                    dtype: .float32,
                    shape: [shapes.hyperWidth, shapes.flattenedResidual]),
                "\(prefix)hc_ffn_scale": .init(dtype: .float32, shape: [3]),
                "\(prefix)hc_ffn_base": .init(
                    dtype: .float32, shape: [shapes.hyperWidth]),
            ])
    }

    private static func validateBackendIdentity(
        expertUnit: DeepSeekV4ExpertUnit,
        routedExperts: PagedRoutedExpertBackend
    ) throws {
        guard routedExperts.containers === expertUnit.containers else {
            throw DeepSeekV4Error.experts(
                "paged backend was not constructed from this learned layer's expert unit")
        }
    }
}
