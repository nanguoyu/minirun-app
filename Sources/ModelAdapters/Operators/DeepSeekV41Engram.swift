import Foundation
import MLX
import MLXBridge

/// Where the 24 rows a token addresses actually come from.
///
/// The rows are 4 KiB apart in a 98 GB file and are chosen by the token ids
/// alone, so their addresses are known before block 0 runs — which is the
/// whole reason Engram is affordable on storage: the reads for both blocks can
/// be issued at the start of the token and overlapped with the first block's
/// compute, exactly as DeepSeek overlaps them with an RDMA fetch. That
/// scheduling belongs to the container reader, not to this operator, so the
/// operator asks for rows through this protocol and the reader decides whether
/// they come from a `.engrampage` file, a page cache, or a test fixture.
///
/// Both returns are the checkpoint's own bytes, never converted:
///
/// - `values`: `uint8[n, 256]`, E4M3FN codes.
/// - `scales`: `uint8[n, 8]`, E8M0 codes, one per 32 values of the row.
///
/// `n` is `indices.count`, in the order given, and a row must appear as many
/// times as it is asked for — a reader that deduplicated would change the
/// shape this operator flattens.
public protocol DeepSeekV41EngramRowProvider {
    func rows(_ indices: [Int]) throws -> (values: MLXArray, scales: MLXArray)
}

/// DeepSeek V4.1 Flash's Engram: an n-gram lookup written into the residual
/// stream, gated by how well it matches that stream.
///
/// ```text
/// rows   = table[hash_ids]                     [tokens, 24, 256] fp8 -> bf16
/// kv     = wkv(rows.flatten())                 [tokens, dim * (hc_mult + 1)]
/// key, value = kv.split                        [tokens, hc_mult, dim], [tokens, dim]
/// rstd   = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)   per (token, stream)
/// dot    = sum(h * q_weight * k_weight * key) * rstd * dim^-0.5
/// gate   = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))
/// h'     = h + gate * value
/// ```
///
/// Three details are not decoration:
///
/// - **The normalization is per (token, residual stream), not joint.** `h` is
///   `[tokens, hc_mult, dim]` and the mean runs over `dim` only, so the four
///   mHC copies each get their own gate.
/// - **The value is shared across the streams, the key is not.** `wkv` emits
///   `hc_mult` keys and one value; that one value is added to every stream,
///   scaled by that stream's own gate.
/// - **`copysign`, not `sign`.** At `dot == 0` the reference's signed square
///   root is `+sqrt(1e-6)`, and at `-0.0` it is `-sqrt(1e-6)`; a `sign()`
///   spelling gives zero for both. The sign *bit* is read below for that
///   reason.
///
/// ## What is bit-exact and what is not
///
/// The row dequantization and the activation rounding are exact: an E4M3 value
/// times a power of two is exact in fp32 and in bf16, and the activation's
/// scale search and round-to-nearest-even are the same integer arithmetic
/// ``DeepSeekV4FP8ActivationReference`` already reproduces. `DeepSeekV41EngramTests`
/// asserts those two against the reference bit for bit.
///
/// Past them the `wkv` GEMM and the reductions over `dim` are floating-point
/// sums whose *order* differs between MLX's kernels and torch's, so the test
/// asserts a relative-error bound there instead. That is the same line
/// ``DeepSeekV4BlockFP8LinearReference`` is held to.
public enum DeepSeekV41Engram {
    /// The reference's `clamp_value`: the floor under `|dot|` before the sqrt,
    /// which keeps the gradient finite at zero in training and is kept here
    /// because it moves the value.
    public static let clampValue: Float = 1e-6

    /// Apply one Engram block to a residual stream.
    ///
    /// - Parameters:
    ///   - h: `[tokens, multiplicity, dimension]`; one token for decode, the
    ///     whole prompt for prefill. The dtype is preserved on return.
    ///   - rowIndices: `[token][column]` from ``DeepSeekV41EngramHasher``, for
    ///     *this block's* table. Indices for the other block address a
    ///     different file and are not interchangeable.
    ///   - rowProvider: the reader for those rows.
    ///   - wkv: the block's `engram.wkv` matrix, `[dimension * (multiplicity +
    ///     1), columns * 256]`, with V4.1's 32x32 scale grid.
    ///   - queryWeight: `engram.q_weight`, `[multiplicity, dimension]`.
    ///   - keyWeight: `engram.k_weight`, same shape. Only the product of the
    ///     two is ever used, and it is formed in fp32 as the reference does.
    ///   - epsilon: `norm_eps` from the checkpoint's config — 1e-20 for V4.1,
    ///     not V4's 1e-6, so it is a parameter and never a default.
    ///   - alive: false for positions that take no part in Engram (an image
    ///     span), whose gate is forced to zero and which therefore pass
    ///     through untouched. Text-only callers pass nil.
    public static func apply(
        _ h: MLXArray,
        rowIndices: [[Int]],
        rowProvider: DeepSeekV41EngramRowProvider,
        wkv: BlockFP8Weights,
        queryWeight: MLXArray,
        keyWeight: MLXArray,
        dimension: Int,
        multiplicity: Int,
        epsilon: Float,
        alive: [Bool]? = nil,
        headDimension: Int = DeepSeekV41EngramConstants.headDimension,
        projectionDType: DType = .bfloat16,
        stream: StreamOrDevice = .default,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        diagnostics: DeepSeekV4Diagnostics = .validating
    ) throws -> MLXArray {
        let tokens = try validate(
            h, rowIndices: rowIndices, wkv: wkv, queryWeight: queryWeight,
            keyWeight: keyWeight, dimension: dimension,
            multiplicity: multiplicity, epsilon: epsilon, alive: alive,
            headDimension: headDimension)
        let columns = rowIndices.first?.count ?? 0

        let lookup = try lookupRows(
            rowIndices: rowIndices, rowProvider: rowProvider, tokens: tokens,
            columns: columns, headDimension: headDimension,
            diagnostics: diagnostics)
        let kv = try project(
            lookup.reshaped([tokens, columns * headDimension]),
            wkv: wkv, projectionDType: projectionDType, stream: stream,
            phaseAccounting: phaseAccounting, diagnostics: diagnostics)

        return gated(
            h, kv: kv, queryWeight: queryWeight, keyWeight: keyWeight,
            dimension: dimension, multiplicity: multiplicity, epsilon: epsilon,
            alive: alive
        ).output.asType(h.dtype)
    }

    /// The `wkv` projection: FP8 activation, FP8 weights, and a narrow return.
    ///
    /// The rows arrive as bf16 and are widened to fp32 *before* the projection,
    /// which is not a liberty: the reference's `fp8_gemm` accumulates each
    /// 32-wide K block in fp32 and adds it into a second fp32 accumulator, and
    /// only the *result* is written in `torch.get_default_dtype()`. Handing MLX
    /// a bf16 activation instead lets it accumulate 6,144 terms narrow, which
    /// measured two orders of magnitude worse against the reference. The
    /// widening is free of information: an E4M3 value times a power of two is
    /// exact in both formats.
    ///
    /// `projectionDType` is that default dtype — bf16 for the released run —
    /// and it is deliberately not the residual stream's, because the reference
    /// narrows here whatever the stream is carrying.
    static func project(
        _ activation: MLXArray,
        wkv: BlockFP8Weights,
        projectionDType: DType,
        stream: StreamOrDevice,
        phaseAccounting: DeepSeekV4PhaseAccounting?,
        diagnostics: DeepSeekV4Diagnostics
    ) throws -> MLXArray {
        let widened = try DeepSeekV4BlockFP8LinearReference.project(
            activation.asType(.float32),
            weights: wkv,
            activationBlockSize: DeepSeekV4BlockFP8LinearReference.v41BlockSize,
            weightBlockRows: DeepSeekV4BlockFP8LinearReference.v41BlockSize,
            weightBlockColumns: DeepSeekV4BlockFP8LinearReference.v41BlockSize,
            stream: stream, phaseAccounting: phaseAccounting,
            diagnostics: diagnostics)
        return widened.asType(projectionDType)
    }

    /// Everything after the `wkv` projection, kept behind its own seam.
    ///
    /// Not a decomposition for its own sake: `wkv` returns bf16, so the two
    /// implementations' `kv` can differ by an ulp before this stage even
    /// starts, and a test that only compared the final output could not tell
    /// that apart from an error in the gate. Handing this function the
    /// reference's own `kv` isolates the gate to fp32 arithmetic, where the
    /// comparison is worth six digits instead of two.
    ///
    /// `output` is fp32, before the cast back to the residual stream's dtype.
    static func gated(
        _ h: MLXArray,
        kv: MLXArray,
        queryWeight: MLXArray,
        keyWeight: MLXArray,
        dimension: Int,
        multiplicity: Int,
        epsilon: Float,
        alive: [Bool]? = nil
    ) -> (dot: MLXArray, gate: MLXArray, output: MLXArray) {
        let tokens = h.shape[0]
        let keyWidth = multiplicity * dimension
        let key = kv[0..., 0..<keyWidth]
            .asType(.float32)
            .reshaped([tokens, multiplicity, dimension])
        let value = kv[0..., keyWidth..<(keyWidth + dimension)].asType(.float32)

        let widened = h.asType(.float32)
        let weight = queryWeight.asType(.float32) * keyWeight.asType(.float32)
        // Per (token, stream) over `dimension`, and never jointly over the
        // streams: two separate RMS terms multiplied, not one normalization of
        // a concatenation.
        let inverseDeviation =
            rsqrt(mean(widened * widened, axis: -1) + epsilon)
            * rsqrt(mean(key * key, axis: -1) + epsilon)
        let dot =
            sum(widened * weight * key, axis: -1)
            * inverseDeviation
            * MLXArray(Float(1.0 / Double(dimension).squareRoot()))

        var gate = signedSquareRootSigmoid(dot)
        if let alive {
            let mask = MLXArray(alive.map { $0 ? Float(1) : Float(0) }, [tokens, 1])
            gate = gate * mask
        }
        let output =
            widened
            + expandedDimensions(gate, axis: -1)
            * expandedDimensions(value, axis: 1)
        return (dot, gate, output)
    }

    /// `sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))`.
    ///
    /// The sign is taken from the bit, not from a comparison: `copysign`
    /// propagates the sign of `-0.0`, and `dot .< 0` does not.
    static func signedSquareRootSigmoid(_ dot: MLXArray) -> MLXArray {
        let magnitude = MLX.sqrt(maximum(MLX.abs(dot), MLXArray(clampValue)))
        let negative =
            (dot.view(dtype: .uint32) & MLXArray(UInt32(0x8000_0000)))
            .!= MLXArray(UInt32(0))
        return sigmoid(which(negative, -magnitude, magnitude))
    }

    /// Fetch the token's rows and dequantize them to bf16, as the reference's
    /// `ParallelEngramEmbedding` does on a single rank.
    ///
    /// The multiplication is by a power of two, so it is exact in fp32 and
    /// still exact after the bf16 cast: an E4M3 significand is four bits and
    /// bf16 carries eight.
    static func lookupRows(
        rowIndices: [[Int]],
        rowProvider: DeepSeekV41EngramRowProvider,
        tokens: Int,
        columns: Int,
        headDimension: Int,
        diagnostics: DeepSeekV4Diagnostics
    ) throws -> MLXArray {
        var flattened = [Int]()
        flattened.reserveCapacity(tokens * columns)
        for token in rowIndices { flattened.append(contentsOf: token) }

        let fetched = try rowProvider.rows(flattened)
        guard fetched.values.dtype == .uint8, fetched.scales.dtype == .uint8 else {
            throw DeepSeekV4Error.configuration(
                "Engram rows must arrive as the checkpoint's own bytes; got "
                    + "\(fetched.values.dtype) values and \(fetched.scales.dtype) scales")
        }
        let groups = headDimension / DeepSeekV41EngramConstants.scaleGroupSize
        guard fetched.values.shape == [flattened.count, headDimension],
            fetched.scales.shape == [flattened.count, groups]
        else {
            throw DeepSeekV4Error.configuration(
                "Engram rows must be [\(flattened.count), \(headDimension)] values "
                    + "and [\(flattened.count), \(groups)] scales; got "
                    + "\(fetched.values.shape) and \(fetched.scales.shape)")
        }
        if diagnostics.validateFiniteness {
            // MLX's E4M3 decoder saturates 0x7f/0xff instead of preserving the
            // checkpoint format's NaN, so those codes must not reach it: the
            // result would be a plausible number rather than a refusal. Both
            // are the bytes whose low seven bits are all set.
            let nonFinite = (fetched.values & MLXArray(UInt8(0x7F)))
                .== MLXArray(UInt8(0x7F))
            if nonFinite.any().item(Bool.self) {
                throw DeepSeekV4Error.configuration(
                    "an Engram row contains a non-finite E4M3FN code")
            }
        }

        let packed = fetched.values.view(dtype: .uint32)
        return dequantized(
            packed,
            scales: fetched.scales,
            biases: nil,
            groupSize: DeepSeekV41EngramConstants.scaleGroupSize,
            bits: 8,
            mode: .mxfp8,
            dtype: .bfloat16)
    }

    private static func validate(
        _ h: MLXArray,
        rowIndices: [[Int]],
        wkv: BlockFP8Weights,
        queryWeight: MLXArray,
        keyWeight: MLXArray,
        dimension: Int,
        multiplicity: Int,
        epsilon: Float,
        alive: [Bool]?,
        headDimension: Int
    ) throws -> Int {
        guard dimension > 0, multiplicity > 0, headDimension > 0,
            headDimension.isMultiple(of: DeepSeekV41EngramConstants.scaleGroupSize),
            epsilon.isFinite, epsilon > 0
        else {
            throw DeepSeekV4Error.configuration(
                "Engram geometry must be positive and epsilon finite; got "
                    + "dimension \(dimension), multiplicity \(multiplicity), "
                    + "head dimension \(headDimension), epsilon \(epsilon)")
        }
        guard h.ndim == 3, h.shape[1] == multiplicity, h.shape[2] == dimension,
            h.shape[0] > 0
        else {
            throw DeepSeekV4Error.configuration(
                "Engram input must be [tokens, \(multiplicity), \(dimension)]; got "
                    + "\(h.shape)")
        }
        let tokens = h.shape[0]
        guard rowIndices.count == tokens else {
            throw DeepSeekV4Error.configuration(
                "Engram row indices cover \(rowIndices.count) tokens for a "
                    + "\(tokens)-token input")
        }
        guard let columns = rowIndices.first?.count, columns > 0,
            rowIndices.allSatisfy({ $0.count == columns })
        else {
            throw DeepSeekV4Error.configuration(
                "every Engram token must address the same positive number of rows")
        }
        let outFeatures = dimension.multipliedReportingOverflow(by: multiplicity + 1)
        let inFeatures = columns.multipliedReportingOverflow(by: headDimension)
        guard !outFeatures.overflow, !inFeatures.overflow,
            wkv.outFeatures == outFeatures.partialValue,
            wkv.inFeatures == inFeatures.partialValue
        else {
            throw DeepSeekV4Error.configuration(
                "Engram wkv must be [\(dimension) * (\(multiplicity) + 1), "
                    + "\(columns) * \(headDimension)]; got "
                    + "[\(wkv.outFeatures), \(wkv.inFeatures)]")
        }
        guard queryWeight.shape == [multiplicity, dimension],
            keyWeight.shape == [multiplicity, dimension]
        else {
            throw DeepSeekV4Error.configuration(
                "Engram q_weight and k_weight must be [\(multiplicity), "
                    + "\(dimension)]; got \(queryWeight.shape) and \(keyWeight.shape)")
        }
        if let alive, alive.count != tokens {
            throw DeepSeekV4Error.configuration(
                "the Engram liveness mask has \(alive.count) entries for "
                    + "\(tokens) tokens")
        }
        return tokens
    }
}
