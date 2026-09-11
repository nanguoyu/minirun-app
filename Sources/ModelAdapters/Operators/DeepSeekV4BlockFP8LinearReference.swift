import MLX
import MLXBridge

/// Equation-level bridge for one dynamic block-FP8 projection.
///
/// The published linear path first quantizes each activation row in groups of
/// `activationBlockSize` E4M3 values with an E8M0 power-of-two scale. Its
/// checkpoint weights use a two-dimensional E4M3/E8M0 scale grid. MLX does not
/// currently expose that exact FP8×FP8 operation on Metal, so this reference
/// preserves the activation rounding explicitly and then consumes the unchanged
/// weight bytes through ``blockFP8MM``. It is a correctness bridge, not the
/// eventual fused product kernel.
///
/// ## Why the block size is a parameter
///
/// V4 quantizes in groups of 128 and carries a 128×128 weight scale grid. V4.1
/// Flash uses 32 for all three — `fp8_block_size = 32` in its `inference/model.py`
/// and `[32, 32]` in its `quantization_config` — which is sixteen times as many
/// weight scales per matrix and a different activation grouping, and nothing
/// else. So the sizes are arguments with V4's values as defaults: one operator,
/// two producers' choices, and no second copy of the rounding rule to drift.
/// ``DeepSeekV41FP8Linear`` is the V4.1 face of it.
///
/// The rounding rule itself does not vary with the size. ``BlockFP8Weights``
/// already carried both scale-grid dimensions and expands any multiple of MLX's
/// 32-wide MX group; ``DeepSeekV4FP8ActivationReference`` already took its group
/// width as an argument. What was hard-coded was only this type's refusal.
public enum DeepSeekV4BlockFP8LinearReference {
    /// V4's own grouping. A caller that means V4 says nothing; a caller that
    /// means V4.1 passes 32.
    public static let activationBlockSize = 128
    public static let weightBlockSize = 128
    /// V4.1 Flash keeps the same equation and shrinks the block to 32 in both
    /// directions — sixteen times the scales per matrix. The arithmetic below
    /// is unchanged by that, so the sizes are parameters with V4's values as
    /// defaults rather than a second copy of this file.
    public static let v41BlockSize = 32

    public static func project(
        _ input: MLXArray,
        weights: BlockFP8Weights,
        activationBlockSize: Int = activationBlockSize,
        weightBlockRows: Int = weightBlockSize,
        weightBlockColumns: Int = weightBlockSize,
        stream: StreamOrDevice = .default,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        guard weights.scaleBlockRows == weightBlockRows,
            weights.scaleBlockColumns == weightBlockColumns
        else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "block-FP8 linear weights must use a "
                    + "\(weightBlockRows)x\(weightBlockColumns) scale grid; got "
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
        activationBlockSize: Int = activationBlockSize,
        weightBlockRows: Int = weightBlockSize,
        weightBlockColumns: Int = weightBlockSize,
        stream: StreamOrDevice = .default,
        cancellationCheck: () throws -> Void = {},
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        guard weights.scaleBlockRows == weightBlockRows,
            weights.scaleBlockColumns == weightBlockColumns
        else {
            throw DeepSeekV4Error.unsupportedArchitecture(
                "grouped block-FP8 weights must use a "
                    + "\(weightBlockRows)x\(weightBlockColumns) scale grid; got "
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
