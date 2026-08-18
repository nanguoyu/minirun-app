import Foundation
import MLX

/// The official V4 mHC split, Sinkhorn normalization, contraction and expansion.
///
/// This is intentionally an ordinary MLX graph, not a fused kernel. Correctness
/// is established before optimization; a fused version must compare every
/// returned tensor against this implementation.
public enum DeepSeekV4HyperConnections {
    public struct Split {
        /// `[items, multiplicity]`, contraction coefficients.
        public let pre: MLXArray
        /// `[items, multiplicity]`, new-branch expansion coefficients.
        public let post: MLXArray
        /// `[items, multiplicity, multiplicity]`, residual mixing matrix.
        public let combination: MLXArray
    }

    public struct PreparedBranch {
        /// `[items, hidden]`, input to attention or MoE after contraction.
        public let input: MLXArray
        public let split: Split
    }

    /// Contract the final residual streams with the model's hyper-head.
    ///
    /// This is the `hc_head` equation from the verified V4 architecture: the
    /// flattened residual is RMS-scaled only for the coefficient projection,
    /// then the learned sigmoid coefficients contract the original residual
    /// streams. Final model RMS normalization is deliberately a separate step.
    public static func collapseHead(
        residual: MLXArray,
        function: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        multiplicity: Int,
        epsilon: Float,
        normEpsilon: Float,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        guard multiplicity > 0, epsilon.isFinite, epsilon > 0,
            normEpsilon.isFinite, normEpsilon > 0
        else {
            throw DeepSeekV4Error.hyperConnection(
                "head multiplicity and epsilons must be finite and positive")
        }
        guard residual.ndim == 3, residual.shape[1] == multiplicity else {
            throw DeepSeekV4Error.hyperConnection(
                "head residual must be [items, \(multiplicity), hidden], got \(residual.shape)")
        }
        let flattenedWidth = multiplicity.multipliedReportingOverflow(
            by: residual.shape[2])
        guard !flattenedWidth.overflow,
            function.ndim == 2,
            function.shape == [multiplicity, flattenedWidth.partialValue],
            scale.ndim == 1, scale.shape == [1],
            base.ndim == 1, base.shape == [multiplicity]
        else {
            throw DeepSeekV4Error.hyperConnection(
                "head function, scale, base, and residual shapes do not agree")
        }
        // The one-element `scale` pull used to be load-bearing, because
        // `scaleValues[0]` was a Swift multiplier below. It is the device scalar
        // now, for the reason `splitSinkhorn` gives at length, so all three
        // reads here — scale, base, function — exist only for the guard. That
        // makes this statement group a finiteness sweep: bracketed as one, and
        // skipped as one. The head tables it sweeps are also swept once on
        // admission, in `DeepSeekV4GlobalArtifact.readHeadParameters`.
        let scale32 = scale.asType(.float32)
        if diagnostics.validateFiniteness {
            phaseAccounting?.recordEval(.hyperConnectionHead)
            try measuringPhase(
                phaseAccounting,
                excludingGPUBoundaryFrom: phaseAccounting?
                    .recordFinitenessSweep(nanoseconds:)
            ) {
                // The three `asArray` calls are the force, so the wait is the
                // whole guard and the sweep keeps only the comparisons.
                try waitingForGPU(phaseAccounting, .finitenessSweep) {
                    let scaleValues = scale32.asArray(Float.self)
                    let baseValues = base.asType(.float32).asArray(Float.self)
                    let functionValues = function.asType(.float32).asArray(Float.self)
                    guard scaleValues.allSatisfy(\.isFinite),
                        baseValues.allSatisfy(\.isFinite),
                        functionValues.allSatisfy(\.isFinite)
                    else {
                        throw DeepSeekV4Error.hyperConnection(
                            "head parameters contain a non-finite value")
                    }
                }
            }
        }

        let widened = residual.asType(.float32)
        let flattened = widened.reshaped([-1, flattenedWidth.partialValue])
        let inverseRMS = rsqrt(
            mean(flattened * flattened, axis: -1, keepDims: true) + normEpsilon)
        let mixes = k3Linear(flattened, function.asType(.float32)) * inverseRMS
        let coefficients = sigmoid(
            mixes * scale32[0] + base.asType(.float32)) + epsilon
        return sum(
            expandedDimensions(coefficients, axis: -1) * widened,
            axis: 1)
    }

    public static func splitSinkhorn(
        mixes: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        multiplicity: Int,
        iterations: Int,
        epsilon: Float,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating,
        compiling: Bool = DeepSeekV4Compile.compilesSinkhorn
    ) throws -> Split {
        guard multiplicity > 0, iterations > 0, epsilon.isFinite, epsilon > 0 else {
            throw DeepSeekV4Error.hyperConnection(
                "multiplicity, iterations, and epsilon must be positive")
        }
        let widenedMultiplicity = multiplicity.addingReportingOverflow(2)
        guard !widenedMultiplicity.overflow else {
            throw DeepSeekV4Error.hyperConnection("mixed width overflows Int")
        }
        let mixedWidth = multiplicity.multipliedReportingOverflow(
            by: widenedMultiplicity.partialValue)
        guard !mixedWidth.overflow else {
            throw DeepSeekV4Error.hyperConnection("mixed width overflows Int")
        }
        guard mixes.ndim == 2, mixes.shape[1] == mixedWidth.partialValue else {
            throw DeepSeekV4Error.hyperConnection(
                "mixes must be [items, \(mixedWidth.partialValue)], got \(mixes.shape)")
        }
        guard scale.ndim == 1, scale.shape[0] == 3 else {
            throw DeepSeekV4Error.hyperConnection("scale must be [3], got \(scale.shape)")
        }
        guard base.ndim == 1, base.shape[0] == mixedWidth.partialValue else {
            throw DeepSeekV4Error.hyperConnection(
                "base must be [\(mixedWidth.partialValue)], got \(base.shape)")
        }

        // These three floats used to be pulled to the host unconditionally,
        // because `scaleValues[0..2]` were Swift multipliers in the coefficients
        // below. That made the pull load-bearing rather than validation, and it
        // was the census's second-largest site: 86 per pass, because `prepare`
        // runs twice per layer — once before attention, once before the MoE tail
        // — and each pull drains a fresh graph over the whole flattened
        // residual.
        //
        // The multipliers are now the device scalars themselves. `scale` is a
        // float32 `[3]` checkpoint constant and each coefficient multiplies a
        // float32 tensor by one of its elements, so `mixParts[i] * scale32[i]`
        // is the same float32 broadcast multiply against the same value that
        // `mixParts[i] * scaleValues[i]` was — the arithmetic is unchanged and
        // only the host round trip is gone.
        //
        // What remains of the old statement is the guard, and a guard is all it
        // is: nothing below reads a host float, so this is a finiteness sweep
        // like every other. It is bracketed and skipped as one. `.off` — what
        // both product scales run — therefore makes this site zero, and
        // `.validating` still exercises it in every fixture suite.
        let scale32 = scale.asType(.float32)
        if diagnostics.validateFiniteness {
            phaseAccounting?.recordEval(.hyperConnectionSplit)
            try measuringPhase(
                phaseAccounting,
                excludingGPUBoundaryFrom: phaseAccounting?
                    .recordFinitenessSweep(nanoseconds:)
            ) {
                try waitingForGPU(phaseAccounting, .finitenessSweep) {
                    let scaleValues = scale32.asArray(Float.self)
                    guard scaleValues.count == 3, scaleValues.allSatisfy(\.isFinite) else {
                        throw DeepSeekV4Error.hyperConnection(
                            "scale contains a non-finite value")
                    }
                }
            }
        }
        let doubledMultiplicity = multiplicity.multipliedReportingOverflow(by: 2)
        guard !doubledMultiplicity.overflow else {
            throw DeepSeekV4Error.hyperConnection("split boundary overflows Int")
        }

        // Everything from here down is a pure function of `(mixes, scale,
        // base)`: no host read, no diagnostic, no accounting. That is the
        // precondition `MLX.compile` needs, and the 20-iteration loop below is
        // the reason it is worth meeting — 153 of the operator's 188 primitives
        // are iterations 2…20 over a `[1, 4, 4]` matrix
        // (`docs/experiments/2026-08-17-v4-mlx-dispatch.md` §3).
        let outputs =
            compiling
            ? DeepSeekV4CompiledSubgraphs.sinkhornTail(
                multiplicity: multiplicity, iterations: iterations, epsilon: epsilon)(
                    [mixes, scale, base])
            : sinkhornTail(
                mixes: mixes, scale: scale, base: base,
                multiplicity: multiplicity, iterations: iterations, epsilon: epsilon)
        return Split(pre: outputs[0], post: outputs[1], combination: outputs[2])
    }

    /// The split, the Sinkhorn iterations, and nothing else.
    ///
    /// Named and separated from ``splitSinkhorn(mixes:scale:base:multiplicity:iterations:epsilon:phaseAccounting:diagnostics:compiling:)``'s
    /// guards for the same reason
    /// ``DeepSeekV4FP8ActivationReference/roundAndDequantize(blocked:scale:)``
    /// is named: it is the exact subgraph the compiled kernel has to reproduce,
    /// so a test can hold the two forms against each other's bits. The result
    /// order is `[pre, post, combination]` because `MLX.compile`'s multi-output
    /// form speaks in arrays, not in ``Split``.
    static func sinkhornTail(
        mixes: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        multiplicity: Int,
        iterations: Int,
        epsilon: Float
    ) -> [MLXArray] {
        let scale32 = scale.asType(.float32)
        let boundaries = [multiplicity, multiplicity * 2]
        let mixParts = split(mixes.asType(.float32), indices: boundaries, axis: -1)
        let baseParts = split(base.asType(.float32), indices: boundaries, axis: -1)

        let pre = sigmoid(mixParts[0] * scale32[0] + baseParts[0]) + epsilon
        let post = sigmoid(mixParts[1] * scale32[1] + baseParts[1]) * Float(2)
        var combination = (
            mixParts[2] * scale32[2] + baseParts[2]
        ).reshaped([-1, multiplicity, multiplicity])

        // Official kernel: row softmax + eps, then a column normalization;
        // subsequent iterations alternate bare row and column normalizations.
        combination = softmax(combination, axis: -1, precise: true) + epsilon
        combination = combination / (sum(combination, axis: -2, keepDims: true) + epsilon)
        if iterations > 1 {
            for _ in 1..<iterations {
                combination = combination
                    / (sum(combination, axis: -1, keepDims: true) + epsilon)
                combination = combination
                    / (sum(combination, axis: -2, keepDims: true) + epsilon)
            }
        }
        return [pre, post, combination]
    }

    /// Contract `multiplicity` residual streams into one branch input.
    public static func prepare(
        residual: MLXArray,
        function: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        multiplicity: Int,
        iterations: Int,
        epsilon: Float,
        normEpsilon: Float,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating,
        compiling: Bool = DeepSeekV4Compile.compilesSinkhorn
    ) throws -> PreparedBranch {
        guard residual.ndim == 3, residual.shape[1] == multiplicity else {
            throw DeepSeekV4Error.hyperConnection(
                "residual must be [items, \(multiplicity), hidden], got \(residual.shape)")
        }
        guard normEpsilon.isFinite, normEpsilon > 0 else {
            throw DeepSeekV4Error.hyperConnection("norm epsilon must be finite and positive")
        }
        let flattenedWidth = multiplicity.multipliedReportingOverflow(by: residual.shape[2])
        guard !flattenedWidth.overflow else {
            throw DeepSeekV4Error.hyperConnection("flattened residual width overflows Int")
        }
        let widenedMultiplicity = multiplicity.addingReportingOverflow(2)
        guard !widenedMultiplicity.overflow else {
            throw DeepSeekV4Error.hyperConnection("mixed width overflows Int")
        }
        let mixedWidth = multiplicity.multipliedReportingOverflow(
            by: widenedMultiplicity.partialValue)
        guard !mixedWidth.overflow,
            function.ndim == 2,
            function.shape == [mixedWidth.partialValue, flattenedWidth.partialValue]
        else {
            throw DeepSeekV4Error.hyperConnection(
                "function must be [\(mixedWidth.partialValue), \(flattenedWidth.partialValue)], "
                    + "got \(function.shape)")
        }

        let flattened = residual.asType(.float32).reshaped([-1, flattenedWidth.partialValue])
        let inverseRMS = rsqrt(
            mean(flattened * flattened, axis: -1, keepDims: true) + normEpsilon)
        let mixes = k3Linear(flattened, function.asType(.float32)) * inverseRMS
        let split = try splitSinkhorn(
            mixes: mixes,
            scale: scale,
            base: base,
            multiplicity: multiplicity,
            iterations: iterations,
            epsilon: epsilon,
            phaseAccounting: phaseAccounting,
            diagnostics: diagnostics,
            compiling: compiling)
        let contracted = sum(
            expandedDimensions(split.pre, axis: -1) * residual.asType(.float32),
            axis: 1)
        return PreparedBranch(input: contracted.asType(residual.dtype), split: split)
    }

    /// Expand a branch result and mix it with the held residual streams.
    public static func combine(
        branch: MLXArray,
        residual: MLXArray,
        split: Split
    ) throws -> MLXArray {
        guard branch.ndim == 2, residual.ndim == 3,
            branch.shape[0] == residual.shape[0],
            branch.shape[1] == residual.shape[2],
            split.post.shape == [residual.shape[0], residual.shape[1]],
            split.combination.shape == [
                residual.shape[0], residual.shape[1], residual.shape[1],
            ]
        else {
            throw DeepSeekV4Error.hyperConnection(
                "branch, residual, post, and combination shapes do not agree")
        }
        let newBranch = expandedDimensions(split.post, axis: -1)
            * expandedDimensions(branch.asType(.float32), axis: 1)
        let mixedResidual = sum(
            expandedDimensions(split.combination, axis: -1)
                * expandedDimensions(residual.asType(.float32), axis: 1),
            axis: 2)
        return (newBranch + mixedResidual).asType(branch.dtype)
    }
}
