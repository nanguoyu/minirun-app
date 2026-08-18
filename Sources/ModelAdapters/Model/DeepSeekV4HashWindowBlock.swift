import Foundation
import MLX

/// The complete prefill equation for a DeepSeek V4 layer that uses token-hash
/// routing and the uncompressed sliding-window attention path.
///
/// This is deliberately narrower than "a V4 block". The published model uses
/// compressed attention from layer 2 onward and learned routing after the hash
/// prefix. Accepting either here would turn missing architecture into a silent
/// approximation. Today this type is the correctness composition for layers 0
/// and 1; the artifact-backed runner must compare its optimized implementation
/// with this equation before either layer can enter the product.
public enum DeepSeekV4HashWindowBlock {
    public struct Geometry: Sendable, Hashable {
        public let layer: Int
        public let hiddenSize: Int
        public let hyperConnectionMultiplicity: Int
        public let hyperConnectionSinkhornIterations: Int
        public let hyperConnectionEpsilon: Float
        public let rmsNormEpsilon: Float

        public let attentionHeads: Int
        public let headDimension: Int
        public let ropeDimension: Int
        public let queryRank: Int
        public let outputGroups: Int
        public let outputRank: Int
        public let slidingWindow: Int
        public let maximumPositionCount: Int
        public let rotary: DeepSeekV4RotaryParameters

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
            attentionHeads: Int,
            headDimension: Int,
            ropeDimension: Int,
            queryRank: Int,
            outputGroups: Int,
            outputRank: Int,
            slidingWindow: Int,
            maximumPositionCount: Int = .max,
            rotary: DeepSeekV4RotaryParameters,
            expertCount: Int,
            expertsPerToken: Int,
            sharedIntermediateSize: Int,
            normalizeRoutingWeights: Bool,
            routingScale: Float,
            swiGLULimit: Float
        ) throws {
            guard layer >= 0 else {
                throw DeepSeekV4Error.configuration("hash-window layer cannot be negative")
            }
            guard hiddenSize > 0, hyperConnectionMultiplicity > 0,
                hyperConnectionSinkhornIterations > 0,
                hyperConnectionEpsilon.isFinite, hyperConnectionEpsilon > 0,
                rmsNormEpsilon.isFinite, rmsNormEpsilon > 0
            else {
                throw DeepSeekV4Error.configuration(
                    "hash-window hidden and hyper-connection geometry is invalid")
            }
            guard attentionHeads > 0, headDimension > 0,
                ropeDimension > 0, ropeDimension <= headDimension,
                ropeDimension.isMultiple(of: 2),
                (headDimension - ropeDimension).isMultiple(of: 64),
                queryRank > 0, outputGroups > 0, outputRank > 0,
                attentionHeads.isMultiple(of: outputGroups), slidingWindow > 0,
                maximumPositionCount > 0,
                rotary.dimension == ropeDimension
            else {
                throw DeepSeekV4Error.configuration(
                    "hash-window attention geometry is invalid")
            }
            guard expertCount > 1, expertsPerToken > 0,
                expertsPerToken < expertCount, sharedIntermediateSize > 0,
                routingScale.isFinite, routingScale > 0,
                swiGLULimit.isFinite, swiGLULimit >= 0
            else {
                throw DeepSeekV4Error.configuration(
                    "hash-window expert geometry is invalid")
            }
            self.layer = layer
            self.hiddenSize = hiddenSize
            self.hyperConnectionMultiplicity = hyperConnectionMultiplicity
            self.hyperConnectionSinkhornIterations = hyperConnectionSinkhornIterations
            self.hyperConnectionEpsilon = hyperConnectionEpsilon
            self.rmsNormEpsilon = rmsNormEpsilon
            self.attentionHeads = attentionHeads
            self.headDimension = headDimension
            self.ropeDimension = ropeDimension
            self.queryRank = queryRank
            self.outputGroups = outputGroups
            self.outputRank = outputRank
            self.slidingWindow = slidingWindow
            self.maximumPositionCount = maximumPositionCount
            self.rotary = rotary
            self.expertCount = expertCount
            self.expertsPerToken = expertsPerToken
            self.sharedIntermediateSize = sharedIntermediateSize
            self.normalizeRoutingWeights = normalizeRoutingWeights
            self.routingScale = routingScale
            self.swiGLULimit = swiGLULimit
        }

        public init(config: DeepSeekV4Config, layer: Int) throws {
            guard config.usesHashRouting(layer: layer) else {
                throw DeepSeekV4Error.unsupportedArchitecture(
                    "layer \(layer) does not use token-hash routing")
            }
            let compression = try config.compressionRatio(layer: layer)
            guard compression == 0 else {
                throw DeepSeekV4Error.unsupportedArchitecture(
                    "layer \(layer) uses compression ratio \(compression), not the hash-window path")
            }
            guard config.numberOfKeyValueHeads == 1 else {
                throw DeepSeekV4Error.unsupportedArchitecture(
                    "hash-window attention requires one shared KV head")
            }
            guard config.sharedExpertCount == 1 else {
                throw DeepSeekV4Error.unsupportedArchitecture(
                    "hash-window composition implements exactly one shared expert")
            }
            try self.init(
                layer: layer,
                hiddenSize: config.hiddenSize,
                hyperConnectionMultiplicity: config.hyperConnectionMultiplicity,
                hyperConnectionSinkhornIterations: config.hyperConnectionSinkhornIterations,
                hyperConnectionEpsilon: config.hyperConnectionEpsilon,
                rmsNormEpsilon: config.rmsNormEpsilon,
                attentionHeads: config.numberOfAttentionHeads,
                headDimension: config.attentionHeadDimension,
                ropeDimension: config.ropeHeadDimension,
                queryRank: config.queryLowRank,
                outputGroups: config.outputGroups,
                outputRank: config.outputLowRank,
                slidingWindow: config.slidingWindow,
                maximumPositionCount: config.maximumPositionCount,
                rotary: DeepSeekV4RotaryParameters(
                    dimension: config.ropeHeadDimension,
                    // The official compression-ratio-zero path explicitly
                    // disables YaRN and uses the base rotary frequencies.
                    originalSequenceLength: 0,
                    base: Double(config.ropeTheta),
                    factor: Double(config.ropeFactor),
                    betaFast: Double(config.ropeBetaFast),
                    betaSlow: Double(config.ropeBetaSlow)),
                expertCount: config.routedExpertCount,
                expertsPerToken: config.expertsPerToken,
                sharedIntermediateSize: config.expertIntermediateSize,
                normalizeRoutingWeights: config.normalizedTopK,
                routingScale: config.routingScale,
                swiGLULimit: config.swiGLULimit)
        }
    }

    /// Dense weights for the reference composition. Artifact execution may
    /// supply the same projections through block-FP8 kernels, but it must not
    /// make those full matrices resident merely to satisfy this value.
    public struct Weights {
        public let attentionNorm: MLXArray
        public let queryA: MLXArray
        public let queryNorm: MLXArray
        public let queryB: MLXArray
        public let keyValue: MLXArray
        public let keyValueNorm: MLXArray
        public let sinks: MLXArray
        public let outputGroup: MLXArray
        public let output: MLXArray
        public let attentionHyperFunction: MLXArray
        public let attentionHyperScale: MLXArray
        public let attentionHyperBase: MLXArray

        public let ffnNorm: MLXArray
        public let router: MLXArray
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
            keyValue: MLXArray,
            keyValueNorm: MLXArray,
            sinks: MLXArray,
            outputGroup: MLXArray,
            output: MLXArray,
            attentionHyperFunction: MLXArray,
            attentionHyperScale: MLXArray,
            attentionHyperBase: MLXArray,
            ffnNorm: MLXArray,
            router: MLXArray,
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
            self.keyValue = keyValue
            self.keyValueNorm = keyValueNorm
            self.sinks = sinks
            self.outputGroup = outputGroup
            self.output = output
            self.attentionHyperFunction = attentionHyperFunction
            self.attentionHyperScale = attentionHyperScale
            self.attentionHyperBase = attentionHyperBase
            self.ffnNorm = ffnNorm
            self.router = router
            self.sharedGate = sharedGate
            self.sharedDown = sharedDown
            self.sharedUp = sharedUp
            self.ffnHyperFunction = ffnHyperFunction
            self.ffnHyperScale = ffnHyperScale
            self.ffnHyperBase = ffnHyperBase
        }
    }

    /// Materialized batch-one circular KV state for layers 0 and 1.
    ///
    /// The cache is always exactly `slidingWindow` rows. `nextPosition`
    /// rejects duplicate or skipped decode calls, while `positionLimit` binds
    /// it to the response limit declared for this run. State has no public
    /// initializer: callers can only obtain it from a validated prefill.
    public struct State {
        fileprivate let nextPosition: Int
        fileprivate let positionLimit: Int
        fileprivate let geometry: Geometry
        fileprivate let artifactIdentity: ArtifactExecutionIdentity?
        fileprivate let windowKeyValues: MLXArray

        public var position: Int { nextPosition }
        public var decodePositionLimit: Int { positionLimit }
        public var cachedPositions: Int { windowKeyValues.shape[0] }
    }

    /// Artifact state is intentionally bound to the live, reconciled layer
    /// objects that produced it. Repository/revision strings are provenance,
    /// not an execution capability: two separately opened artifacts may name
    /// the same source while addressing different filesystem objects. Strong
    /// references prevent object-identifier reuse while the state is alive;
    /// these objects retain metadata and rooted openers, never layer weights.
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
        /// `[sequence, multiplicity, hidden]`, ready for the next layer.
        public let residual: MLXArray
        public let routing: DeepSeekV4Router.Selection
        public let attentionIndices: [[Int]]
        /// Present only when the caller declares a run-scoped decode limit.
        public let state: State?
    }

    public struct DecodeResult {
        public let residual: MLXArray
        public let routing: DeepSeekV4Router.Selection
        public let attentionIndices: [Int]
        public let state: State
    }

    private struct PlainWeights {
        let attentionNorm: MLXArray
        let queryNorm: MLXArray
        let keyValueNorm: MLXArray
        let sinks: MLXArray
        let attentionHyperFunction: MLXArray
        let attentionHyperScale: MLXArray
        let attentionHyperBase: MLXArray
        let ffnNorm: MLXArray
        let router: MLXArray
        let ffnHyperFunction: MLXArray
        let ffnHyperScale: MLXArray
        let ffnHyperBase: MLXArray
    }

    private struct Projections {
        let queryA: (MLXArray) throws -> MLXArray
        let queryB: (MLXArray) throws -> MLXArray
        let keyValue: (MLXArray) throws -> MLXArray
        let attentionOutput: (MLXArray) throws -> MLXArray
        let sharedGate: (MLXArray) throws -> MLXArray
        let sharedDown: (MLXArray) throws -> MLXArray
        let sharedUp: (MLXArray) throws -> MLXArray
    }

    private struct TensorShapes {
        let headsWidth: Int
        let groupInput: Int
        let groupRows: Int
        let flattenedResidual: Int
        let hyperWidth: Int

        init(_ geometry: Geometry) throws {
            let headsWidth = geometry.attentionHeads.multipliedReportingOverflow(
                by: geometry.headDimension)
            let headsPerGroup = geometry.attentionHeads / geometry.outputGroups
            let groupInput = headsPerGroup.multipliedReportingOverflow(
                by: geometry.headDimension)
            let groupRows = geometry.outputGroups.multipliedReportingOverflow(
                by: geometry.outputRank)
            let flattenedResidual = geometry.hyperConnectionMultiplicity
                .multipliedReportingOverflow(by: geometry.hiddenSize)
            let widenedMultiplicity = geometry.hyperConnectionMultiplicity
                .addingReportingOverflow(2)
            guard !widenedMultiplicity.overflow else {
                throw DeepSeekV4Error.configuration(
                    "hash-window weight geometry overflows Int")
            }
            let hyperWidth = geometry.hyperConnectionMultiplicity
                .multipliedReportingOverflow(by: widenedMultiplicity.partialValue)
            guard !headsWidth.overflow, !groupInput.overflow, !groupRows.overflow,
                !flattenedResidual.overflow, !hyperWidth.overflow
            else {
                throw DeepSeekV4Error.configuration(
                    "hash-window weight geometry overflows Int")
            }
            self.headsWidth = headsWidth.partialValue
            self.groupInput = groupInput.partialValue
            self.groupRows = groupRows.partialValue
            self.flattenedResidual = flattenedResidual.partialValue
            self.hyperWidth = hyperWidth.partialValue
        }
    }

    /// Batch-one prompt execution at position zero. Decode/cache state is not
    /// invented here; the later artifact runner owns that separate lifecycle.
    public static func prefill(
        residual: MLXArray,
        tokenIDs: [Int],
        tokenExpertMap: DeepSeekV4TokenExpertMap,
        weights: Weights,
        geometry: Geometry,
        routedExperts: RoutedExpertBackend,
        decodePositionLimit: Int? = nil,
        cancellationCheck: () throws -> Void = {}
    ) throws -> PrefillResult {
        let shapes = try TensorShapes(geometry)
        try validateDense(
            residual: residual, tokenIDs: tokenIDs, tokenExpertMap: tokenExpertMap,
            weights: weights, geometry: geometry, shapes: shapes)
        let projections = Projections(
            queryA: { k3Linear($0, weights.queryA) },
            queryB: { k3Linear($0, weights.queryB) },
            keyValue: { k3Linear($0, weights.keyValue) },
            attentionOutput: {
                try DeepSeekV4GroupedOutput.project(
                    $0, groupWeight: weights.outputGroup,
                    outputWeight: weights.output, groups: geometry.outputGroups)
            },
            sharedGate: { k3Linear($0, weights.sharedGate) },
            sharedDown: { k3Linear($0, weights.sharedDown.asType(.float32)) },
            sharedUp: { k3Linear($0, weights.sharedUp) })
        return try execute(
            residual: residual,
            tokenIDs: tokenIDs,
            tokenExpertMap: tokenExpertMap,
            plain: PlainWeights(
                attentionNorm: weights.attentionNorm,
                queryNorm: weights.queryNorm,
                keyValueNorm: weights.keyValueNorm,
                sinks: weights.sinks,
                attentionHyperFunction: weights.attentionHyperFunction,
                attentionHyperScale: weights.attentionHyperScale,
                attentionHyperBase: weights.attentionHyperBase,
                ffnNorm: weights.ffnNorm,
                router: weights.router,
                ffnHyperFunction: weights.ffnHyperFunction,
                ffnHyperScale: weights.ffnHyperScale,
                ffnHyperBase: weights.ffnHyperBase),
            projections: projections,
            geometry: geometry,
            routedExperts: routedExperts,
            decodePositionLimit: decodePositionLimit,
            artifactIdentity: nil,
            cancellationCheck: cancellationCheck)
    }

    /// Advance one token through an uncompressed hash-window block.
    public static func decode(
        residual: MLXArray,
        tokenID: Int,
        tokenExpertMap: DeepSeekV4TokenExpertMap,
        weights: Weights,
        geometry: Geometry,
        routedExperts: RoutedExpertBackend,
        startPosition: Int,
        state: State,
        cancellationCheck: () throws -> Void = {}
    ) throws -> DecodeResult {
        let shapes = try TensorShapes(geometry)
        try validateDense(
            residual: residual,
            tokenIDs: [tokenID],
            tokenExpertMap: tokenExpertMap,
            weights: weights,
            geometry: geometry,
            shapes: shapes)
        return try executeDecode(
            residual: residual,
            tokenID: tokenID,
            tokenExpertMap: tokenExpertMap,
            plain: PlainWeights(
                attentionNorm: weights.attentionNorm,
                queryNorm: weights.queryNorm,
                keyValueNorm: weights.keyValueNorm,
                sinks: weights.sinks,
                attentionHyperFunction: weights.attentionHyperFunction,
                attentionHyperScale: weights.attentionHyperScale,
                attentionHyperBase: weights.attentionHyperBase,
                ffnNorm: weights.ffnNorm,
                router: weights.router,
                ffnHyperFunction: weights.ffnHyperFunction,
                ffnHyperScale: weights.ffnHyperScale,
                ffnHyperBase: weights.ffnHyperBase),
            projections: Projections(
                queryA: { k3Linear($0, weights.queryA) },
                queryB: { k3Linear($0, weights.queryB) },
                keyValue: { k3Linear($0, weights.keyValue) },
                attentionOutput: {
                    try DeepSeekV4GroupedOutput.project(
                        $0, groupWeight: weights.outputGroup,
                        outputWeight: weights.output, groups: geometry.outputGroups)
                },
                sharedGate: { k3Linear($0, weights.sharedGate) },
                sharedDown: { k3Linear($0, weights.sharedDown.asType(.float32)) },
                sharedUp: { k3Linear($0, weights.sharedUp) }),
            geometry: geometry,
            routedExperts: routedExperts,
            startPosition: startPosition,
            state: state,
            artifactIdentity: nil,
            cancellationCheck: cancellationCheck)
    }

    /// Execute one layer-0/1 block directly from a fully verified artifact.
    ///
    /// Every manifest requirement and the layer/expert identity are checked
    /// before any tensor payload or expert tile is read. Non-expert matrices
    /// are then read, digested, consumed, and released one at a time. The
    /// supplied paged backend must be built from `expertUnit.containers`; this
    /// prevents a caller from pairing a valid layer manifest with expert bytes
    /// from another unit. User cancellation should both make
    /// `cancellationCheck` throw and call `routedExperts.cancel(reason:)` so a
    /// blocked expert read is interrupted as well as the between-stage work.
    public static func prefill(
        residual: MLXArray,
        tokenIDs: [Int],
        vocabularySize: Int,
        artifact: DeepSeekV4LayerArtifact,
        expertUnit: DeepSeekV4ExpertUnit,
        geometry: Geometry,
        routedExperts: PagedRoutedExpertBackend,
        decodePositionLimit: Int? = nil,
        stream: StreamOrDevice = .default,
        cancellationCheck: @escaping () throws -> Void = {}
    ) throws -> PrefillResult {
        let shapes = try TensorShapes(geometry)
        try validateInput(
            residual: residual, tokenIDs: tokenIDs, geometry: geometry)
        try validateArtifactContract(
            artifact: artifact, expertUnit: expertUnit,
            geometry: geometry, vocabularySize: vocabularySize)
        try validateBackendIdentity(
            expertUnit: expertUnit, routedExperts: routedExperts)
        guard tokenIDs.allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
            throw DeepSeekV4Error.routing(
                "hash-window token ids must be inside 0..<\(vocabularySize)")
        }
        try cancellationCheck()
        let loaded = try measuringLayerArtifactSetup(artifact.phaseAccounting) {
            try loadArtifactExecution(
                artifact: artifact,
                geometry: geometry,
                shapes: shapes,
                vocabularySize: vocabularySize,
                stream: stream,
                cancellationCheck: cancellationCheck)
        }
        return try execute(
            residual: residual,
            tokenIDs: tokenIDs,
            tokenExpertMap: loaded.tokenExpertMap,
            plain: loaded.plain,
            projections: loaded.projections,
            geometry: geometry,
            routedExperts: routedExperts,
            decodePositionLimit: decodePositionLimit,
            artifactIdentity: ArtifactExecutionIdentity(
                artifact: artifact, expertUnit: expertUnit),
            cancellationCheck: cancellationCheck,
            phaseAccounting: artifact.phaseAccounting,
            diagnostics: artifact.diagnostics)
    }

    /// Advance one token from the same verified artifact that produced the
    /// run-scoped cache. Large projections are opened, digested, evaluated,
    /// and released exactly once for this layer step; none are retained in
    /// ``State``. The artifact and expert-container identities are checked
    /// before the first tensor payload is read.
    public static func decode(
        residual: MLXArray,
        tokenID: Int,
        vocabularySize: Int,
        artifact: DeepSeekV4LayerArtifact,
        expertUnit: DeepSeekV4ExpertUnit,
        geometry: Geometry,
        routedExperts: PagedRoutedExpertBackend,
        startPosition: Int,
        state: State,
        stream: StreamOrDevice = .default,
        cancellationCheck: @escaping () throws -> Void = {}
    ) throws -> DecodeResult {
        let shapes = try TensorShapes(geometry)
        try validateInput(
            residual: residual, tokenIDs: [tokenID], geometry: geometry)
        try validateArtifactContract(
            artifact: artifact, expertUnit: expertUnit,
            geometry: geometry, vocabularySize: vocabularySize)
        try validateBackendIdentity(
            expertUnit: expertUnit, routedExperts: routedExperts)
        guard tokenID >= 0, tokenID < vocabularySize else {
            throw DeepSeekV4Error.routing(
                "hash-window token id must be inside 0..<\(vocabularySize)")
        }
        let identity = ArtifactExecutionIdentity(
            artifact: artifact, expertUnit: expertUnit)
        try validateDecodeRequest(
            residual: residual,
            startPosition: startPosition,
            state: state,
            geometry: geometry,
            artifactIdentity: identity)
        try cancellationCheck()
        let loaded = try measuringLayerArtifactSetup(artifact.phaseAccounting) {
            try loadArtifactExecution(
                artifact: artifact,
                geometry: geometry,
                shapes: shapes,
                vocabularySize: vocabularySize,
                stream: stream,
                cancellationCheck: cancellationCheck)
        }
        return try executeDecode(
            residual: residual,
            tokenID: tokenID,
            tokenExpertMap: loaded.tokenExpertMap,
            plain: loaded.plain,
            projections: loaded.projections,
            geometry: geometry,
            routedExperts: routedExperts,
            startPosition: startPosition,
            state: state,
            artifactIdentity: identity,
            cancellationCheck: cancellationCheck,
            phaseAccounting: artifact.phaseAccounting,
            diagnostics: artifact.diagnostics)
    }

    private struct LoadedArtifactExecution {
        let tokenExpertMap: DeepSeekV4TokenExpertMap
        let plain: PlainWeights
        let projections: Projections
    }

    private static func loadArtifactExecution(
        artifact: DeepSeekV4LayerArtifact,
        geometry: Geometry,
        shapes: TensorShapes,
        vocabularySize: Int,
        stream: StreamOrDevice,
        cancellationCheck: @escaping () throws -> Void
    ) throws -> LoadedArtifactExecution {
        let prefix = "layers.\(geometry.layer)."
        let tokenExpertMap = try artifact.loadTokenExpertMap(
            tensor: "\(prefix)ffn.gate.tid2eid",
            vocabularySize: vocabularySize,
            expertsPerToken: geometry.expertsPerToken,
            expertCount: geometry.expertCount,
            cancellationCheck: cancellationCheck)
        let plain = try PlainWeights(
            attentionNorm: artifact.loadFloatingTensor(
                tensor: "\(prefix)attn_norm.weight", expectedDType: .bfloat16,
                expectedShape: [geometry.hiddenSize], cancellationCheck: cancellationCheck),
            queryNorm: artifact.loadFloatingTensor(
                tensor: "\(prefix)attn.q_norm.weight", expectedDType: .bfloat16,
                expectedShape: [geometry.queryRank], cancellationCheck: cancellationCheck),
            keyValueNorm: artifact.loadFloatingTensor(
                tensor: "\(prefix)attn.kv_norm.weight", expectedDType: .bfloat16,
                expectedShape: [geometry.headDimension], cancellationCheck: cancellationCheck),
            sinks: artifact.loadFloatingTensor(
                tensor: "\(prefix)attn.attn_sink", expectedDType: .float32,
                expectedShape: [geometry.attentionHeads], cancellationCheck: cancellationCheck),
            attentionHyperFunction: artifact.loadFloatingTensor(
                tensor: "\(prefix)hc_attn_fn", expectedDType: .float32,
                expectedShape: [shapes.hyperWidth, shapes.flattenedResidual],
                cancellationCheck: cancellationCheck),
            attentionHyperScale: artifact.loadFloatingTensor(
                tensor: "\(prefix)hc_attn_scale", expectedDType: .float32,
                expectedShape: [3], cancellationCheck: cancellationCheck),
            attentionHyperBase: artifact.loadFloatingTensor(
                tensor: "\(prefix)hc_attn_base", expectedDType: .float32,
                expectedShape: [shapes.hyperWidth], cancellationCheck: cancellationCheck),
            ffnNorm: artifact.loadFloatingTensor(
                tensor: "\(prefix)ffn_norm.weight", expectedDType: .bfloat16,
                expectedShape: [geometry.hiddenSize], cancellationCheck: cancellationCheck),
            router: artifact.loadFloatingTensor(
                tensor: "\(prefix)ffn.gate.weight", expectedDType: .bfloat16,
                expectedShape: [geometry.expertCount, geometry.hiddenSize],
                cancellationCheck: cancellationCheck),
            ffnHyperFunction: artifact.loadFloatingTensor(
                tensor: "\(prefix)hc_ffn_fn", expectedDType: .float32,
                expectedShape: [shapes.hyperWidth, shapes.flattenedResidual],
                cancellationCheck: cancellationCheck),
            ffnHyperScale: artifact.loadFloatingTensor(
                tensor: "\(prefix)hc_ffn_scale", expectedDType: .float32,
                expectedShape: [3], cancellationCheck: cancellationCheck),
            ffnHyperBase: artifact.loadFloatingTensor(
                tensor: "\(prefix)hc_ffn_base", expectedDType: .float32,
                expectedShape: [shapes.hyperWidth], cancellationCheck: cancellationCheck))

        func projection(
            _ tensor: String, outFeatures: Int, inFeatures: Int
        ) -> (MLXArray) throws -> MLXArray {
            { input in
                try artifact.projectBlockFP8Reference(
                    input, tensor: tensor, outFeatures: outFeatures,
                    inFeatures: inFeatures, stream: stream,
                    cancellationCheck: cancellationCheck)
            }
        }
        let projections = Projections(
            queryA: projection(
                "\(prefix)attn.wq_a.weight",
                outFeatures: geometry.queryRank, inFeatures: geometry.hiddenSize),
            queryB: projection(
                "\(prefix)attn.wq_b.weight",
                outFeatures: shapes.headsWidth, inFeatures: geometry.queryRank),
            keyValue: projection(
                "\(prefix)attn.wkv.weight",
                outFeatures: geometry.headDimension, inFeatures: geometry.hiddenSize),
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
            sharedGate: projection(
                "\(prefix)ffn.shared_experts.w1.weight",
                outFeatures: geometry.sharedIntermediateSize,
                inFeatures: geometry.hiddenSize),
            sharedDown: projection(
                "\(prefix)ffn.shared_experts.w2.weight",
                outFeatures: geometry.hiddenSize,
                inFeatures: geometry.sharedIntermediateSize),
            sharedUp: projection(
                "\(prefix)ffn.shared_experts.w3.weight",
                outFeatures: geometry.sharedIntermediateSize,
                inFeatures: geometry.hiddenSize))
        return LoadedArtifactExecution(
            tokenExpertMap: tokenExpertMap,
            plain: plain,
            projections: projections)
    }

    private static func execute(
        residual: MLXArray,
        tokenIDs: [Int],
        tokenExpertMap: DeepSeekV4TokenExpertMap,
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
        let sequence = residual.shape[0]
        if let decodePositionLimit {
            guard decodePositionLimit >= sequence,
                decodePositionLimit <= geometry.maximumPositionCount
            else {
                throw DeepSeekV4Error.attention(
                    "decode position limit \(decodePositionLimit) must contain the prompt "
                        + "and not exceed \(geometry.maximumPositionCount)")
            }
        }
        let positions = Array(0..<sequence)

        let attentionPrepared = try DeepSeekV4HyperConnections.prepare(
            residual: residual,
            function: plain.attentionHyperFunction,
            scale: plain.attentionHyperScale,
            base: plain.attentionHyperBase,
            multiplicity: geometry.hyperConnectionMultiplicity,
            iterations: geometry.hyperConnectionSinkhornIterations,
            epsilon: geometry.hyperConnectionEpsilon,
            normEpsilon: geometry.rmsNormEpsilon,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        let attentionInput = K3Norm.rms(
            attentionPrepared.input, weight: plain.attentionNorm,
            eps: geometry.rmsNormEpsilon)

        let queryLatent = K3Norm.rms(
            try projections.queryA(attentionInput),
            weight: plain.queryNorm,
            eps: geometry.rmsNormEpsilon)
        try cancellationCheck()
        var queries = try projections.queryB(queryLatent)
            .asType(.float32)
            .reshaped([sequence, geometry.attentionHeads, geometry.headDimension])
        queries = queries * rsqrt(
            mean(queries * queries, axis: -1, keepDims: true)
                + geometry.rmsNormEpsilon)
        queries = try rotateTail(
            queries, positions: positions, geometry: geometry, inverse: false)

        var keyValues = K3Norm.rms(
            try projections.keyValue(attentionInput),
            weight: plain.keyValueNorm,
            eps: geometry.rmsNormEpsilon)
        keyValues = try rotateTail(
            keyValues, positions: positions, geometry: geometry, inverse: false)
        keyValues = try quantizeNonRotaryPrefix(
            keyValues, geometry: geometry, phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        let indices = try DeepSeekV4AttentionIndices.window(
            windowSize: geometry.slidingWindow,
            sequenceLength: sequence,
            startPosition: 0)
        var attended = try DeepSeekV4SparseAttention.forward(
            queries: queries,
            keysAndValues: keyValues,
            sinks: plain.sinks,
            indices: indices,
            scale: Float(1 / Foundation.sqrt(Double(geometry.headDimension))),
            phaseAccounting: phaseAccounting)
        attended = try rotateTail(
            attended, positions: positions, geometry: geometry, inverse: true)
        try cancellationCheck()
        let attentionBranch = try projections.attentionOutput(attended)
        let afterAttention = try DeepSeekV4HyperConnections.combine(
            branch: attentionBranch,
            residual: residual,
            split: attentionPrepared.split)

        let ffn = try DeepSeekV4MoEFFN.execute(
            afterAttention: afterAttention,
            routingChoice: .tokenHash(tokenIDs: tokenIDs, map: tokenExpertMap),
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
        let state: State?
        if let decodePositionLimit {
            let cache = makeWindowCache(
                keyValues, windowSize: geometry.slidingWindow)
            submittingToGPU(phaseAccounting, .blockCache) {
                MLX.asyncEval([ffn.residual, cache])
            }
            state = State(
                nextPosition: sequence,
                positionLimit: decodePositionLimit,
                geometry: geometry,
                artifactIdentity: artifactIdentity,
                windowKeyValues: cache)
        } else {
            state = nil
        }
        try cancellationCheck()
        return PrefillResult(
            residual: ffn.residual,
            routing: ffn.routing,
            attentionIndices: indices,
            state: state)
    }

    private static func executeDecode(
        residual: MLXArray,
        tokenID: Int,
        tokenExpertMap: DeepSeekV4TokenExpertMap,
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

        let positions = [startPosition]
        let attentionPrepared = try DeepSeekV4HyperConnections.prepare(
            residual: residual,
            function: plain.attentionHyperFunction,
            scale: plain.attentionHyperScale,
            base: plain.attentionHyperBase,
            multiplicity: geometry.hyperConnectionMultiplicity,
            iterations: geometry.hyperConnectionSinkhornIterations,
            epsilon: geometry.hyperConnectionEpsilon,
            normEpsilon: geometry.rmsNormEpsilon,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        let attentionInput = K3Norm.rms(
            attentionPrepared.input,
            weight: plain.attentionNorm,
            eps: geometry.rmsNormEpsilon)

        let queryLatent = K3Norm.rms(
            try projections.queryA(attentionInput),
            weight: plain.queryNorm,
            eps: geometry.rmsNormEpsilon)
        try cancellationCheck()
        var queries = try projections.queryB(queryLatent)
            .asType(.float32)
            .reshaped([1, geometry.attentionHeads, geometry.headDimension])
        queries = queries * rsqrt(
            mean(queries * queries, axis: -1, keepDims: true)
                + geometry.rmsNormEpsilon)
        queries = try rotateTail(
            queries, positions: positions, geometry: geometry, inverse: false)

        var keyValue = K3Norm.rms(
            try projections.keyValue(attentionInput),
            weight: plain.keyValueNorm,
            eps: geometry.rmsNormEpsilon)
        keyValue = try rotateTail(
            keyValue, positions: positions, geometry: geometry, inverse: false)
        keyValue = try quantizeNonRotaryPrefix(
            keyValue, geometry: geometry, phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        let windowCache = replacingRow(
            state.windowKeyValues,
            at: startPosition % geometry.slidingWindow,
            with: keyValue)
        try cancellationCheck()

        let indices = try DeepSeekV4AttentionIndices.window(
            windowSize: geometry.slidingWindow,
            sequenceLength: 1,
            startPosition: startPosition)[0]
        var attended = try DeepSeekV4SparseAttention.forward(
            queries: queries,
            keysAndValues: windowCache,
            sinks: plain.sinks,
            indices: [indices],
            scale: Float(1 / Foundation.sqrt(Double(geometry.headDimension))),
            phaseAccounting: phaseAccounting)
        attended = try rotateTail(
            attended, positions: positions, geometry: geometry, inverse: true)
        try cancellationCheck()
        let attentionBranch = try projections.attentionOutput(attended)
        let afterAttention = try DeepSeekV4HyperConnections.combine(
            branch: attentionBranch,
            residual: residual,
            split: attentionPrepared.split)

        let ffn = try DeepSeekV4MoEFFN.execute(
            afterAttention: afterAttention,
            routingChoice: .tokenHash(tokenIDs: [tokenID], map: tokenExpertMap),
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
        let next = State(
            nextPosition: startPosition + 1,
            positionLimit: state.positionLimit,
            geometry: geometry,
            artifactIdentity: artifactIdentity,
            windowKeyValues: windowCache)
        submittingToGPU(phaseAccounting, .blockCache) {
            MLX.asyncEval([ffn.residual, windowCache])
        }
        try cancellationCheck()
        return DecodeResult(
            residual: ffn.residual,
            routing: ffn.routing,
            attentionIndices: indices,
            state: next)
    }

    private static func validateDecodeRequest(
        residual: MLXArray,
        startPosition: Int,
        state: State,
        geometry: Geometry,
        artifactIdentity: ArtifactExecutionIdentity?
    ) throws {
        guard residual.shape[0] == 1 else {
            throw DeepSeekV4Error.configuration(
                "incremental hash-window execution accepts exactly one residual row")
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
        guard state.nextPosition > 0,
            state.positionLimit > 0,
            state.nextPosition <= state.positionLimit,
            state.positionLimit <= geometry.maximumPositionCount,
            state.geometry == geometry,
            state.artifactIdentity == artifactIdentity,
            state.windowKeyValues.shape
                == [geometry.slidingWindow, geometry.headDimension],
            state.windowKeyValues.dtype.isFloatingPoint,
            !state.windowKeyValues.dtype.isComplex
        else {
            throw DeepSeekV4Error.attention(
                "hash-window decode state has incompatible geometry or artifact identity")
        }
    }

    private static func quantizeNonRotaryPrefix(
        _ input: MLXArray,
        geometry: Geometry,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        let nonRotary = geometry.headDimension - geometry.ropeDimension
        guard nonRotary > 0 else { return input }
        guard nonRotary.isMultiple(of: 64) else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "sliding-window non-rotary KV width \(nonRotary) is not divisible by 64")
        }
        let prefix = try DeepSeekV4FP8ActivationReference.quantizeDequantize(
            input[.ellipsis, 0..<nonRotary], blockSize: 64,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        return concatenated([prefix, input[.ellipsis, nonRotary...]], axis: -1)
    }

    private static func makeWindowCache(
        _ prompt: MLXArray,
        windowSize: Int
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
        _ array: MLXArray,
        at row: Int,
        with replacement: MLXArray
    ) -> MLXArray {
        var pieces = [MLXArray]()
        if row > 0 { pieces.append(array[0..<row, 0...]) }
        pieces.append(replacement)
        if row + 1 < array.shape[0] {
            pieces.append(array[(row + 1)..., 0...])
        }
        return concatenated(pieces, axis: 0)
    }

    private static func rotateTail(
        _ input: MLXArray,
        positions: [Int],
        geometry: Geometry,
        inverse: Bool
    ) throws -> MLXArray {
        guard input.ndim >= 2, input.shape[input.ndim - 1] == geometry.headDimension else {
            throw DeepSeekV4Error.attention(
                "rotary input does not end in the configured head dimension")
        }
        let nonRotary = geometry.headDimension - geometry.ropeDimension
        let tail = input[.ellipsis, nonRotary...]
        let rotated = try DeepSeekV4Rotary.apply(
            tail, positions: positions, parameters: geometry.rotary, inverse: inverse)
        guard nonRotary > 0 else { return rotated }
        return concatenated([input[.ellipsis, 0..<nonRotary], rotated], axis: -1)
    }

    private static func validateInput(
        residual: MLXArray,
        tokenIDs: [Int],
        geometry: Geometry
    ) throws {
        guard residual.ndim == 3, residual.shape[0] > 0,
            residual.shape[1] == geometry.hyperConnectionMultiplicity,
            residual.shape[2] == geometry.hiddenSize,
            residual.dtype.isFloatingPoint, !residual.dtype.isComplex,
            tokenIDs.count == residual.shape[0]
        else {
            throw DeepSeekV4Error.configuration(
                "hash-window residual must be nonempty [sequence, multiplicity, hidden] "
                    + "with one token id per row")
        }
    }

    private static func validateDense(
        residual: MLXArray,
        tokenIDs: [Int],
        tokenExpertMap: DeepSeekV4TokenExpertMap,
        weights: Weights,
        geometry: Geometry,
        shapes: TensorShapes
    ) throws {
        try validateInput(residual: residual, tokenIDs: tokenIDs, geometry: geometry)
        guard tokenExpertMap.expertCount == geometry.expertCount,
            tokenExpertMap.expertsPerToken == geometry.expertsPerToken
        else {
            throw DeepSeekV4Error.routing(
                "token-to-expert table does not match the block geometry")
        }

        let expected: [(String, MLXArray, [Int])] = [
            ("attention norm", weights.attentionNorm, [geometry.hiddenSize]),
            ("query A", weights.queryA, [geometry.queryRank, geometry.hiddenSize]),
            ("query norm", weights.queryNorm, [geometry.queryRank]),
            ("query B", weights.queryB, [shapes.headsWidth, geometry.queryRank]),
            ("key/value", weights.keyValue, [geometry.headDimension, geometry.hiddenSize]),
            ("key/value norm", weights.keyValueNorm, [geometry.headDimension]),
            ("attention sinks", weights.sinks, [geometry.attentionHeads]),
            ("output group", weights.outputGroup, [shapes.groupRows, shapes.groupInput]),
            ("output", weights.output, [geometry.hiddenSize, shapes.groupRows]),
            ("attention hyper function", weights.attentionHyperFunction,
                [shapes.hyperWidth, shapes.flattenedResidual]),
            ("attention hyper scale", weights.attentionHyperScale, [3]),
            ("attention hyper base", weights.attentionHyperBase, [shapes.hyperWidth]),
            ("FFN norm", weights.ffnNorm, [geometry.hiddenSize]),
            ("router", weights.router, [geometry.expertCount, geometry.hiddenSize]),
            ("shared gate", weights.sharedGate,
                [geometry.sharedIntermediateSize, geometry.hiddenSize]),
            ("shared down", weights.sharedDown,
                [geometry.hiddenSize, geometry.sharedIntermediateSize]),
            ("shared up", weights.sharedUp,
                [geometry.sharedIntermediateSize, geometry.hiddenSize]),
            ("FFN hyper function", weights.ffnHyperFunction,
                [shapes.hyperWidth, shapes.flattenedResidual]),
            ("FFN hyper scale", weights.ffnHyperScale, [3]),
            ("FFN hyper base", weights.ffnHyperBase, [shapes.hyperWidth]),
        ]
        for (name, value, shape) in expected {
            guard value.shape == shape, value.dtype.isFloatingPoint, !value.dtype.isComplex else {
                throw DeepSeekV4Error.configuration(
                    "\(name) must be floating point \(shape), got \(value.shape) \(value.dtype)")
            }
        }
    }

    private struct ContractKey: Hashable {
        let expertUnit: ObjectIdentifier
        let geometry: Geometry
        let vocabularySize: Int
    }

    /// Validate every metadata and shape requirement without reading a tensor
    /// payload. Full-model admission calls this for every layer before the
    /// first large weight read; `prefill` and `decode` call the same boundary
    /// again so the execution path cannot drift from admission.
    ///
    /// The artifact records one admission per (expert unit, geometry,
    /// vocabulary) request, so the per-token repetition of a metadata check
    /// that already passed against the same immutable manifest costs a
    /// dictionary lookup instead of rebuilding twenty-odd expectations.
    static func validateArtifactContract(
        artifact: DeepSeekV4LayerArtifact,
        expertUnit: DeepSeekV4ExpertUnit,
        geometry: Geometry,
        vocabularySize: Int
    ) throws {
        try artifact.admitContractOnce(
            key: ContractKey(
                expertUnit: ObjectIdentifier(expertUnit),
                geometry: geometry,
                vocabularySize: vocabularySize),
            participants: [expertUnit]
        ) {
            try admitArtifactContract(
                artifact: artifact,
                expertUnit: expertUnit,
                geometry: geometry,
                vocabularySize: vocabularySize)
        }
    }

    private static func admitArtifactContract(
        artifact: DeepSeekV4LayerArtifact,
        expertUnit: DeepSeekV4ExpertUnit,
        geometry: Geometry,
        vocabularySize: Int
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
                "layer manifest, expert unit, and hash-window geometry do not share one identity")
        }
        guard vocabularySize > 0 else {
            throw DeepSeekV4Error.configuration(
                "hash-window vocabulary size must be positive")
        }
        let prefix = "layers.\(geometry.layer)."
        try artifact.validateTensorContract(
            blockFP8: [
                "\(prefix)attn.wq_a.weight": [geometry.queryRank, geometry.hiddenSize],
                "\(prefix)attn.wq_b.weight": [shapes.headsWidth, geometry.queryRank],
                "\(prefix)attn.wkv.weight": [geometry.headDimension, geometry.hiddenSize],
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
                    dtype: .bfloat16, shape: [geometry.headDimension]),
                "\(prefix)attn.attn_sink": .init(
                    dtype: .float32, shape: [geometry.attentionHeads]),
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
                "\(prefix)ffn.gate.tid2eid": .init(
                    dtype: .int64,
                    shape: [vocabularySize, geometry.expertsPerToken]),
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
                "paged backend was not constructed from this layer's validated expert unit")
        }
    }
}
