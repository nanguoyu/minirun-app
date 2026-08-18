import Foundation
import MLX

/// Counts the MLX primitives one expression builds, by asking MLX itself.
///
/// ## Why this exists
///
/// Every V4 record that has quoted a primitive count so far quoted a *hand*
/// count from the source — `docs/experiments/2026-08-17-v4-cpu-profile.md`'s
/// ~39,600 per pass and `…-v4-fp8-fusion.md`'s 90 → 41 per call — with the same
/// caveat attached each time: "mlx 0.31.4 exposes no primitive counter". That
/// caveat is true of the *public* counter and false of the file format. mlx's
/// `export_function` traces the closure, runs `compile_dfs` to a tape, and
/// serialises
///
/// ```text
///   uint64  tape.size()
///   per node:  uint64 id, bool has_primitive, …, string primitive name, …
/// ```
///
/// so a `.mlxfn` written by ``census(of:inputs:)`` *is* a primitive census, and
/// the names in it are the primitives MLX would have walked. This turns a hand
/// count into a measurement, and it is the only instrument here that can be
/// wrong in a way a test catches: ``MLXGraphCensus/isConsistent`` restates the
/// same file two ways and refuses to agree with itself unless both readings do.
///
/// ## What the number means, exactly
///
/// mlx runs `compile_simplify(tape, …, passes: 3)` **before** it serialises, so
/// the count is the tape *after* common-subexpression elimination — the same
/// simplification `MLX.compile` performs before it fuses. That makes it the
/// right number for two questions and the wrong number for a third:
///
/// - **right** for "how many primitives would a compiled version of this
///   subgraph contain", which is what an `MLX.compile` arm needs to predict;
/// - **right**, to within CSE, for "how many primitives does `eval_impl` walk",
///   because the uncompiled path walks the *unsimplified* graph and CSE only
///   ever removes duplicates the source really wrote twice;
/// - **wrong** as a restatement of a hand count. Where the two disagree the
///   difference is what CSE removed, and that difference is a result, not an
///   error. ``MLXGraphCensus/leaves`` is reported beside it so the reader can
///   see how much of the tape is input and constant rather than arithmetic.
///
/// ## This is an instrument, not a decode path
///
/// Nothing here runs during a decode. It writes a file, it traces the closure
/// an extra time, and it is called only from tests and benches. It lives in
/// `MLXBridge` rather than in a test target because two targets measure with
/// it and because it is MLX integration, which is what this module is for.
public struct MLXGraphCensus: Sendable, Equatable {

    /// Every node in the traced tape, arithmetic and leaf alike, as mlx
    /// serialised it.
    public let tapeNodes: Int

    /// Primitive nodes, by mlx's own primitive name (`Add`, `Multiply`,
    /// `Select`, `GatherQMM`, `View`, …). Summing this is the primitive count.
    public let primitives: [String: Int]

    /// Tape nodes that carry no primitive: the traced inputs, plus every
    /// constant the expression materialised — an `MLXArray(448)` written inside
    /// an operator is one of these, not a primitive.
    public var leaves: Int { tapeNodes - primitiveTotal }

    /// The primitive count. This is the number a record quotes.
    public var primitiveTotal: Int { primitives.values.reduce(0, +) }

    /// The primitive names in a stable order — commonest first, then
    /// alphabetically, so two runs print the same table.
    public var ranked: [(name: String, count: Int)] {
        primitives
            .map { (name: $0.key, count: $0.value) }
            .sorted { $0.count == $1.count ? $0.name < $1.name : $0.count > $1.count }
    }

    /// Whether the two readings of the same file agree.
    ///
    /// The tape size is read by *parsing* the header; the primitive names are
    /// found by *scanning* for length-prefixed known names. Those are
    /// independent readings of the same bytes, and a scan that picked up a
    /// false positive inside constant data, or missed a name, breaks the
    /// identity `tapeNodes >= primitiveTotal`. A census that fails this must
    /// not be quoted.
    public var isConsistent: Bool { tapeNodes >= primitiveTotal && tapeNodes > 0 }

    /// One line for a table.
    public var summaryLine: String {
        "\(primitiveTotal) primitives, \(leaves) leaves, \(tapeNodes) tape nodes"
            + (ranked.isEmpty
                ? ""
                : "; " + ranked.map { "\($0.name) \($0.count)" }.joined(separator: ", "))
    }

    /// Trace `body` at the given inputs and count the primitives it builds.
    ///
    /// The inputs are the arrays the expression is *traced over*. Anything the
    /// closure closes over instead — a weight, a scalar — becomes a constant in
    /// the tape and is counted in ``leaves``, not in ``primitives``. Passing
    /// weights as inputs therefore keeps the file free of bulk data, which is
    /// worth doing: it keeps the export small and it removes the only place the
    /// name scan could see a false positive.
    ///
    /// - Throws: whatever `body` throws, and ``MLXGraphCensusError`` if mlx
    ///   refuses to serialise a primitive or the file cannot be read back.
    public static func census(
        of body: @escaping ([MLXArray]) -> [MLXArray],
        inputs: [MLXArray]
    ) throws -> MLXGraphCensus {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-graph-census-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("graph.mlxfn")

        let exporter = exportFunction(to: url, body)
        // `@dynamicCallable`'s `KeyValuePairs` has no runtime initialiser, so
        // the argument list has to be spelled out at each arity. Eight covers
        // every V4 family; a wider one should add a case rather than close the
        // extra arrays over the closure, because a closed-over array is a
        // constant in the tape and lands in `leaves` instead of being traced.
        switch inputs.count {
        case 1: try exporter(inputs[0])
        case 2: try exporter(inputs[0], inputs[1])
        case 3: try exporter(inputs[0], inputs[1], inputs[2])
        case 4: try exporter(inputs[0], inputs[1], inputs[2], inputs[3])
        case 5: try exporter(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
        case 6: try exporter(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5])
        case 7:
            try exporter(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6])
        case 8:
            try exporter(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6],
                inputs[7])
        default: throw MLXGraphCensusError.unsupportedArity(inputs.count)
        }
        return try read(url)
    }

    /// The same, for the common single-array case.
    public static func census(
        of body: @escaping (MLXArray) -> MLXArray,
        input: MLXArray
    ) throws -> MLXGraphCensus {
        try census(of: { [body($0[0])] }, inputs: [input])
    }

    // MARK: Reading the file

    /// Parse a `.mlxfn` written by mlx 0.31.4.
    ///
    /// Two independent readings, deliberately:
    ///
    /// 1. **the header**, parsed field by field to the `uint64` tape size that
    ///    `export.cpp` writes immediately before the nodes;
    /// 2. **the names**, scanned for as length-prefixed ASCII matching mlx's
    ///    own primitive table.
    ///
    /// Neither can check the other's arithmetic, but together they bound it:
    /// see ``isConsistent``.
    static func read(_ url: URL) throws -> MLXGraphCensus {
        let data = try Data(contentsOf: url)
        var reader = ByteReader(data)

        _ = try reader.string()  // mlx version
        let traces = try reader.int32()
        guard traces >= 1 else { throw MLXGraphCensusError.emptyExport }
        _ = try reader.bool()  // shapeless

        // One trace only: `census` records exactly one call. A file with more
        // would need the node walk this parser deliberately does not have.
        guard traces == 1 else { throw MLXGraphCensusError.multipleTraces(Int(traces)) }

        _ = try reader.strings()  // kwarg keys
        _ = try reader.uint64s()  // trace input ids
        let inputCount = try reader.uint64()
        for _ in 0..<inputCount {
            _ = try reader.int32s()  // shape
            _ = try reader.int32()  // dtype val
            _ = try reader.uint8()  // dtype size
        }
        _ = try reader.uint64s()  // trace output ids
        let tapeNodes = Int(try reader.uint64())

        return MLXGraphCensus(
            tapeNodes: tapeNodes,
            primitives: Self.scanForPrimitiveNames(data, from: reader.offset))
    }

    /// Count length-prefixed occurrences of mlx's own primitive names.
    ///
    /// `PrimitiveFactory::save` writes `stream, name, state` for every
    /// primitive node, and a name is serialised the way every string is: a
    /// `uint64` length then the bytes. Scanning for exactly that shape, against
    /// the closed set of names mlx can serialise at all, is what makes this
    /// readable without a per-primitive state parser — mlx's state layout is
    /// per primitive type and there is no generic way to skip it.
    private static func scanForPrimitiveNames(_ data: Data, from start: Int) -> [String: Int] {
        var counts = [String: Int]()
        let bytes = [UInt8](data)
        var index = start
        while index + 8 < bytes.count {
            var length = 0
            var wide = false
            for byte in 0..<8 {
                let value = Int(bytes[index + byte])
                if byte >= 2, value != 0 { wide = true }
                length |= value << (8 * byte)
            }
            guard !wide, length >= Self.shortestName, length <= Self.longestName,
                index + 8 + length <= bytes.count
            else {
                index += 1
                continue
            }
            let candidate = String(
                decoding: bytes[(index + 8)..<(index + 8 + length)], as: UTF8.self)
            guard Self.primitiveNames.contains(candidate) else {
                index += 1
                continue
            }
            counts[candidate, default: 0] += 1
            index += 8 + length
        }
        return counts
    }

    private static let shortestName = 3
    private static let longestName = 32

    /// mlx 0.31.4's serialisable primitive names, from `PrimitiveFactory`'s own
    /// table in `mlx/export.cpp`. A primitive absent from that table cannot be
    /// exported at all — mlx throws — so this set is complete by construction
    /// for anything ``census(of:inputs:)`` returns rather than throws.
    static let primitiveNames: Set<String> = [
        "Abs", "Add", "AddMM", "Arange", "ArcCos", "ArcCosh", "ArcSin", "ArcSinh",
        "ArcTan", "ArcTan2", "ArcTanh", "ArgPartition", "ArgReduce", "ArgSort",
        "AsType", "AsStrided", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "LeftShift",
        "RightShift", "BitwiseBinary", "BlockMaskedMM", "Broadcast", "BroadcastAxes",
        "Ceil", "Concatenate", "Conjugate", "Convolution", "Copy", "Cos", "Cosh",
        "Depends", "Divide", "DivMod", "DynamicSlice", "DynamicSliceUpdate", "Equal",
        "NaNEqual", "Erf", "ErfInv", "Exp", "Expm1", "ExpandDims", "FFT", "Flatten",
        "Floor", "Full", "Gather", "GatherAxis", "GatherMM", "Greater", "GreaterEqual",
        "Hadamard", "Imag", "Less", "LessEqual", "Log", "Log2", "Log10", "Log1p",
        "LogicalNot", "LogicalAnd", "LogicalOr", "LogAddExp", "LogSumExp",
        "MaskedScatter", "Matmul", "Maximum", "Minimum", "Multiply", "Negative",
        "NotEqual", "Reshape", "NumberOfElements", "Pad", "Partition", "Power",
        "QuantizedMatmul", "GatherQMM", "RandomBits", "Real", "Remainder", "Reduce",
        "And", "Or", "Sum", "Prod", "Min", "Max", "Round", "Scan", "Scatter",
        "ScatterAxis", "Select", "Sigmoid", "Sign", "Sin", "Sinh", "Slice",
        "SliceUpdate", "Softmax", "Sort", "Split", "Square", "Squeeze", "Sqrt",
        "Rsqrt", "StopGradient", "Subtract", "Tan", "Tanh", "View", "Transpose",
        "Unflatten", "QRF", "SVD", "Inverse", "Cholesky", "Eig", "Eigh", "Quantize",
        "RMSNorm", "RMSNormVJP", "LayerNorm", "LayerNormVJP", "RoPE",
        "ScaledDotProductAttention", "CustomKernel",
    ]
}

/// What can go wrong reading a `.mlxfn`.
public enum MLXGraphCensusError: Error, CustomStringConvertible {
    case truncated(at: Int)
    case emptyExport
    case multipleTraces(Int)
    case unsupportedArity(Int)

    public var description: String {
        switch self {
        case .truncated(let offset):
            "mlxfn export ends inside the header at byte \(offset)"
        case .emptyExport:
            "mlxfn export recorded no trace"
        case .multipleTraces(let count):
            "mlxfn export recorded \(count) traces; the census reads exactly one"
        case .unsupportedArity(let count):
            "graph census traces 1...8 inputs; asked for \(count)"
        }
    }
}

/// A little-endian cursor over the export's header.
///
/// Only the shapes the header uses. It is not a general `.mlxfn` reader and it
/// stops at the tape, which is the last field it can parse without a
/// per-primitive state layout.
private struct ByteReader {
    private let bytes: [UInt8]
    private(set) var offset = 0

    init(_ data: Data) { bytes = [UInt8](data) }

    mutating func take(_ count: Int) throws -> ArraySlice<UInt8> {
        guard offset + count <= bytes.count else { throw MLXGraphCensusError.truncated(at: offset) }
        defer { offset += count }
        return bytes[offset..<(offset + count)]
    }

    mutating func uint8() throws -> UInt8 { try take(1).first! }
    mutating func bool() throws -> Bool { try uint8() != 0 }

    mutating func uint64() throws -> UInt64 {
        var value: UInt64 = 0
        for (index, byte) in try take(8).enumerated() { value |= UInt64(byte) << (8 * index) }
        return value
    }

    mutating func int32() throws -> Int32 {
        var value: UInt32 = 0
        for (index, byte) in try take(4).enumerated() { value |= UInt32(byte) << (8 * index) }
        return Int32(bitPattern: value)
    }

    mutating func string() throws -> String {
        let length = Int(try uint64())
        return String(decoding: try take(length), as: UTF8.self)
    }

    mutating func strings() throws -> [String] {
        try (0..<Int(uint64())).map { _ in try string() }
    }

    mutating func uint64s() throws -> [UInt64] {
        try (0..<Int(uint64())).map { _ in try uint64() }
    }

    mutating func int32s() throws -> [Int32] {
        try (0..<Int(uint64())).map { _ in try int32() }
    }
}
