import Foundation
import MLX

/// Shared MoE tail for DeepSeek V4 transformer layers.
///
/// Layers 0 and 1 use window-only attention while layer 2 adds ratio-four
/// compressed attention. Later layers replace only the hash expert choice with
/// the learned correction-bias choice; the router weights, shared expert,
/// hyper-connection, and residual equation remain identical.
enum DeepSeekV4MoEFFN {
    struct Geometry: Sendable, Equatable {
        let layer: Int
        let hiddenSize: Int
        let hyperConnectionMultiplicity: Int
        let hyperConnectionSinkhornIterations: Int
        let hyperConnectionEpsilon: Float
        let rmsNormEpsilon: Float
        let expertCount: Int
        let expertsPerToken: Int
        let sharedIntermediateSize: Int
        let normalizeRoutingWeights: Bool
        let routingScale: Float
        let swiGLULimit: Float
    }

    struct Result {
        let residual: MLXArray
        let routing: DeepSeekV4Router.Selection
    }

    static func execute(
        afterAttention: MLXArray,
        routingChoice: DeepSeekV4Router.Choice,
        ffnNorm: MLXArray,
        router: MLXArray,
        hyperFunction: MLXArray,
        hyperScale: MLXArray,
        hyperBase: MLXArray,
        sharedGate: (MLXArray) throws -> MLXArray,
        sharedDown: (MLXArray) throws -> MLXArray,
        sharedUp: (MLXArray) throws -> MLXArray,
        geometry: Geometry,
        routedExperts: RoutedExpertBackend,
        cancellationCheck: () throws -> Void,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> Result {
        try validate(
            afterAttention: afterAttention,
            routingChoice: routingChoice,
            ffnNorm: ffnNorm,
            router: router,
            hyperFunction: hyperFunction,
            hyperScale: hyperScale,
            hyperBase: hyperBase,
            geometry: geometry,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        try cancellationCheck()

        let prepared = try DeepSeekV4HyperConnections.prepare(
            residual: afterAttention,
            function: hyperFunction,
            scale: hyperScale,
            base: hyperBase,
            multiplicity: geometry.hyperConnectionMultiplicity,
            iterations: geometry.hyperConnectionSinkhornIterations,
            epsilon: geometry.hyperConnectionEpsilon,
            normEpsilon: geometry.rmsNormEpsilon,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        let input = K3Norm.rms(
            prepared.input, weight: ffnNorm,
            eps: geometry.rmsNormEpsilon)
        let routing = try DeepSeekV4Router.select(
            logits: k3Linear(input.asType(.float32), router.asType(.float32)),
            choice: routingChoice,
            expertCount: geometry.expertCount,
            expertsPerToken: geometry.expertsPerToken,
            normalizeSelectedWeights: geometry.normalizeRoutingWeights,
            routingScale: geometry.routingScale,
            phaseAccounting: phaseAccounting)
        try cancellationCheck()
        let routed = try DeepSeekV4RoutedExperts.forward(
            input,
            layer: geometry.layer,
            expertIDs: routing.ids,
            routingWeights: routing.weights,
            swiGLULimit: geometry.swiGLULimit,
            backend: routedExperts)
        let shared = try sharedExpert(
            input,
            gate: sharedGate,
            down: sharedDown,
            up: sharedUp,
            intermediateSize: geometry.sharedIntermediateSize,
            limit: geometry.swiGLULimit,
            cancellationCheck: cancellationCheck)
        try cancellationCheck()
        let residual = try DeepSeekV4HyperConnections.combine(
            branch: routed + shared,
            residual: afterAttention,
            split: prepared.split)
        // The MoE tail's own boundary: one per layer per pass, and the point
        // that submits the routed and shared expert arithmetic together.
        // Deferred — it bounds the graph, and the next thing the decode thread
        // does is build the next layer, not read a number out of this one.
        phaseAccounting?.recordAsyncEval(.ffnResidual)
        MLX.asyncEval([residual])
        try cancellationCheck()
        return Result(residual: residual, routing: routing)
    }

    private static func validate(
        afterAttention: MLXArray,
        routingChoice: DeepSeekV4Router.Choice,
        ffnNorm: MLXArray,
        router: MLXArray,
        hyperFunction: MLXArray,
        hyperScale: MLXArray,
        hyperBase: MLXArray,
        geometry: Geometry,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws {
        guard geometry.layer >= 0,
            geometry.hiddenSize > 0,
            geometry.hyperConnectionMultiplicity > 0,
            geometry.hyperConnectionSinkhornIterations > 0,
            geometry.hyperConnectionEpsilon.isFinite,
            geometry.hyperConnectionEpsilon > 0,
            geometry.rmsNormEpsilon.isFinite,
            geometry.rmsNormEpsilon > 0,
            geometry.expertCount > 1,
            geometry.expertsPerToken > 0,
            geometry.expertsPerToken < geometry.expertCount,
            geometry.sharedIntermediateSize > 0,
            geometry.routingScale.isFinite,
            geometry.routingScale > 0,
            geometry.swiGLULimit.isFinite,
            geometry.swiGLULimit >= 0
        else {
            throw DeepSeekV4Error.configuration(
                "MoE FFN geometry is invalid")
        }
        guard afterAttention.ndim == 3,
            afterAttention.shape[0] > 0,
            afterAttention.shape[1] == geometry.hyperConnectionMultiplicity,
            afterAttention.shape[2] == geometry.hiddenSize,
            afterAttention.dtype.isFloatingPoint,
            !afterAttention.dtype.isComplex
        else {
            throw DeepSeekV4Error.configuration(
                "MoE FFN residual does not match its geometry")
        }
        switch routingChoice {
        case .tokenHash(let tokenIDs, let tokenExpertMap):
            guard tokenIDs.count == afterAttention.shape[0],
                tokenExpertMap.expertCount == geometry.expertCount,
                tokenExpertMap.expertsPerToken == geometry.expertsPerToken
            else {
                throw DeepSeekV4Error.configuration(
                    "hash route token ids or token map do not match the MoE geometry")
            }
        case .learned(let correctionBias):
            // Two guards where there was one, so the sweep can be bracketed —
            // and now switched — on its own. The shape and dtype checks are
            // structural and always run; only the sync below is diagnostic.
            // The thrown message and the short-circuit order are unchanged.
            guard correctionBias.shape == [geometry.expertCount],
                correctionBias.dtype.isFloatingPoint,
                !correctionBias.dtype.isComplex
            else {
                throw DeepSeekV4Error.configuration(
                    "learned route correction bias does not match the MoE geometry")
            }
            if diagnostics.validateFiniteness {
                phaseAccounting?.recordEval(.finitenessSweep)
                guard measuringPhase(
                    phaseAccounting?.recordFinitenessSweep(nanoseconds:),
                    { isFinite(correctionBias).all().item(Bool.self) })
                else {
                    throw DeepSeekV4Error.configuration(
                        "learned route correction bias does not match the MoE geometry")
                }
            }
        }
        let flattened = geometry.hyperConnectionMultiplicity
            .multipliedReportingOverflow(by: geometry.hiddenSize)
        let widened = geometry.hyperConnectionMultiplicity.addingReportingOverflow(2)
        let hyperWidth = widened.overflow
            ? (partialValue: 0, overflow: true)
            : geometry.hyperConnectionMultiplicity.multipliedReportingOverflow(
                by: widened.partialValue)
        guard !flattened.overflow, !hyperWidth.overflow,
            ffnNorm.shape == [geometry.hiddenSize],
            router.shape == [geometry.expertCount, geometry.hiddenSize],
            hyperFunction.shape == [hyperWidth.partialValue, flattened.partialValue],
            hyperScale.shape == [3],
            hyperBase.shape == [hyperWidth.partialValue]
        else {
            throw DeepSeekV4Error.configuration(
                "MoE FFN norm, router, or hyper-connection tensors have incompatible geometry")
        }
        let arrays = [ffnNorm, router, hyperFunction, hyperScale, hyperBase]
        guard arrays.allSatisfy({ $0.dtype.isFloatingPoint && !$0.dtype.isComplex }) else {
            throw DeepSeekV4Error.configuration(
                "MoE FFN parameters must be real floating point")
        }
    }

    private static func sharedExpert(
        _ input: MLXArray,
        gate: (MLXArray) throws -> MLXArray,
        down: (MLXArray) throws -> MLXArray,
        up: (MLXArray) throws -> MLXArray,
        intermediateSize: Int,
        limit: Float,
        cancellationCheck: () throws -> Void
    ) throws -> MLXArray {
        var gateValue = try gate(input).asType(.float32)
        try cancellationCheck()
        var upValue = try up(input).asType(.float32)
        guard gateValue.shape == upValue.shape,
            gateValue.ndim == 2,
            gateValue.shape[0] == input.shape[0],
            gateValue.shape[1] == intermediateSize
        else {
            throw DeepSeekV4Error.configuration(
                "shared expert gate and up projections have incompatible geometry")
        }
        if limit > 0 {
            let upper = MLXArray(limit)
            gateValue = minimum(gateValue, upper)
            upValue = minimum(maximum(upValue, MLXArray(-limit)), upper)
        }
        try cancellationCheck()
        let output = try down(K3Activation.silu(gateValue) * upValue)
        guard output.shape == input.shape else {
            throw DeepSeekV4Error.configuration(
                "shared expert down projection has incompatible geometry")
        }
        return output
    }
}
