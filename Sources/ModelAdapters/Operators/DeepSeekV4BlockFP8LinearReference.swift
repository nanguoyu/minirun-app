import MLX
import MLXBridge

/// Equation-level bridge for one DeepSeek V4 dynamic block-FP8 projection.
///
/// The published V4 linear path first quantizes each activation row in groups
/// of 128 E4M3 values with an E8M0 power-of-two scale. Its checkpoint weights
/// use a two-dimensional 128×128 E4M3/E8M0 scale grid. MLX does not currently
/// expose that exact FP8×FP8 operation on Metal, so this reference preserves
/// the activation rounding explicitly and then consumes the unchanged weight
/// bytes through ``blockFP8MM``. It is a correctness bridge, not the eventual
/// fused product kernel.
public enum DeepSeekV4BlockFP8LinearReference {
    public static let activationBlockSize = 128
    public static let weightBlockSize = 128

    public static func project(
        _ input: MLXArray,
        weights: BlockFP8Weights,
        stream: StreamOrDevice = .default,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        guard weights.scaleBlockRows == weightBlockSize,
            weights.scaleBlockColumns == weightBlockSize
        else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "block-FP8 linear weights must use a 128x128 scale grid; got "
                    + "\(weights.scaleBlockRows)x\(weights.scaleBlockColumns)")
        }
        let roundedActivation = try DeepSeekV4FP8ActivationReference.quantizeDequantize(
            input, blockSize: activationBlockSize, phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        return try blockFP8MM(roundedActivation, weights, stream: stream)
    }

    /// Grouped `wo_a` projection over one unchanged checkpoint matrix.
    ///
    /// The official attention path reshapes heads into output groups. Each
    /// activation group is multiplied by the matching consecutive row band in
    /// `wo_a`; applying every row to one flattened activation would be a
    /// different program. Row bands are zero-copy views over the validated
    /// parent matrix.
    public static func projectGroupedOutput(
        _ input: MLXArray,
        weights: BlockFP8Weights,
        groups: Int,
        stream: StreamOrDevice = .default,
        cancellationCheck: () throws -> Void = {},
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        guard weights.scaleBlockRows == weightBlockSize,
            weights.scaleBlockColumns == weightBlockSize
        else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "grouped block-FP8 weights must use a 128x128 scale grid; got "
                    + "\(weights.scaleBlockRows)x\(weights.scaleBlockColumns)")
        }
        guard input.ndim == 3, input.shape[0] > 0, groups > 0,
            input.shape[1] == groups, input.shape[2] == weights.inFeatures,
            weights.outFeatures.isMultiple(of: groups)
        else {
            throw DeepSeekV4Error.configuration(
                "grouped block-FP8 input must be [tokens, groups, \(weights.inFeatures)] "
                    + "and output rows must divide by groups; got \(input.shape), "
                    + "\(weights.outFeatures) rows, and \(groups) groups")
        }
        try cancellationCheck()
        let rounded = try DeepSeekV4FP8ActivationReference.quantizeDequantize(
            input, blockSize: activationBlockSize, phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        let rowsPerGroup = weights.outFeatures / groups
        var projected = [MLXArray]()
        projected.reserveCapacity(groups)
        for group in 0..<groups {
            try cancellationCheck()
            let rowStart = group * rowsPerGroup
            let groupWeights = try weights.rowSlice(
                rowStart..<(rowStart + rowsPerGroup))
            let groupInput = rounded[0..., group, 0...]
            projected.append(try blockFP8MM(groupInput, groupWeights, stream: stream))
        }
        try cancellationCheck()
        return concatenated(projected, axis: -1)
    }
}
