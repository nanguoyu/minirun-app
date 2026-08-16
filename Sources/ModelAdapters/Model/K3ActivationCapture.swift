import Foundation
import MLX
import StorageCore

/// Where ``K3Model/forward(tokenIds:state:)`` sends the activations criterion
/// (b) recomputes.
///
/// M9's exit criterion (b) is a *stratified per-layer oracle spot check*: take
/// one layer's real inputs, recompute that layer on a different machine with
/// different code against the original checkpoint, and compare. That needs the
/// inputs, and until now nothing in this package could write an activation to
/// disk — M9C's record says so plainly, and it is why (b) went undone.
///
/// ## The one property this protocol must have
///
/// **Capturing must not change the run.** A hook that perturbs the arithmetic
/// invalidates the thing it was built to check, and a per-layer comparison
/// against a capture from a *different* forward pass proves nothing about the
/// pass that produced the token. Two design choices serve that:
///
/// - Every value offered here is one the forward pass already computed. The
///   operators build their `Trace` structs unconditionally and ``K3Model``
///   discards them; capture keeps them instead. No expression is added to the
///   graph and no operand changes.
/// - ``selects(layer:)`` is asked once per layer, and for an unselected layer
///   nothing else is called. A 93-layer run with six layers selected does the
///   same work in the other 87 as an uncaptured run.
///
/// That argument is not offered as proof. The captured run re-emits the logits,
/// and the gate is that they still hash to the value the uncaptured runs
/// produced. If forcing a subgraph to evaluate early ever changed a result, the
/// digest would move and the capture would be thrown away, not explained.
public protocol K3ActivationSink: AnyObject {

    /// Whether this layer's activations are wanted. Must be cheap: it is called
    /// once per layer on every run that installs a sink.
    func selects(layer: Int) -> Bool

    /// Record one tensor. `name` is the checkpoint-side stage name, without a
    /// layer prefix; the sink adds one.
    func record(layer: Int, _ name: String, _ value: MLXArray)

    /// Record the router's chosen expert ids, which are host-side integers
    /// rather than a tensor.
    func record(layer: Int, _ name: String, ids: [[Int]])
}

extension K3ActivationSink {
    /// Record a tensor only when it exists, so call sites stay one line.
    public func record(layer: Int, _ name: String, _ value: MLXArray?) {
        if let value { record(layer: layer, name, value) }
    }
}

/// Writes captured activations as NumPy `.npy` files with a JSON index.
///
/// ## Why `.npy` and not the metrics-JSON shape the rest of the package uses
///
/// The consumer is a Python oracle on another machine. `.npy` is read by
/// `numpy.load` with the shape and dtype carried in the file, so a transcription
/// mistake in a shape becomes an exception on load rather than a silently
/// mis-strided comparison — which is exactly the failure mode a spot check
/// exists to catch. Everything is written as `float32` little-endian: the
/// forward pass computes in float32, the widened BF16 weights are exact in it,
/// and a single dtype means the oracle has no branch to get wrong.
///
/// The index carries a SHA-256 per file, so the archived copy in
/// `docs/experiments/data/` can be shown to be the bytes the run produced.
public final class K3ActivationCapture: K3ActivationSink {

    /// Layers whose activations are wanted.
    public let layers: Set<Int>
    /// Where the `.npy` files and `index.json` land.
    public let directory: URL

    public struct Entry: Codable {
        public var layer: Int
        public var name: String
        public var file: String
        public var shape: [Int]
        public var dtype: String
        public var bytes: Int
        public var sha256: String
    }

    public private(set) var entries: [Entry] = []
    /// Host-side integer records, e.g. the routed expert ids.
    public private(set) var identifiers: [String: [[Int]]] = [:]
    /// Free-form facts the record needs beside the tensors.
    public var notes: [String: String] = [:]

    public init(directory: URL, layers: Set<Int>) throws {
        self.directory = directory
        self.layers = layers
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
    }

    public func selects(layer: Int) -> Bool { layers.contains(layer) }

    public func record(layer: Int, _ name: String, _ value: MLXArray) {
        guard layers.contains(layer) else { return }
        let float = value.asType(.float32)
        float.eval()
        let shape = float.shape
        let payload = Self.npy(float.asArray(Float.self), shape: shape)
        let file = String(format: "layer%02d.%@.npy", layer, name)
        do {
            try payload.write(to: directory.appendingPathComponent(file))
        } catch {
            // A capture that cannot write is a failed capture, and the run it
            // is attached to is a 112-second one. Say so loudly rather than
            // discovering an absent file at comparison time.
            fatalError("activation capture could not write \(file): \(error)")
        }
        entries.append(
            Entry(
                layer: layer, name: name, file: file, shape: shape, dtype: "float32",
                bytes: payload.count,
                sha256: SHA256.hash(payload).map { String(format: "%02x", $0) }.joined()))
    }

    public func record(layer: Int, _ name: String, ids: [[Int]]) {
        guard layers.contains(layer) else { return }
        identifiers[String(format: "layer%02d.%@", layer, name)] = ids
    }

    /// Write `index.json` beside the tensors.
    public func finish(extra: [String: Any] = [:]) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let tensorData = try encoder.encode(entries)
        var payload: [String: Any] = [
            "schema_version": 1,
            "layers": layers.sorted(),
            "tensors": try JSONSerialization.jsonObject(with: tensorData),
            "expert_ids": identifiers,
            "notes": notes,
        ]
        for (key, value) in extra { payload[key] = value }
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: directory.appendingPathComponent("index.json"))
    }

    /// Total bytes written, for the record's cost line.
    public var bytesWritten: Int { entries.reduce(0) { $0 + $1.bytes } }

    // MARK: - NPY

    /// NPY format v1.0, the subset numpy writes for a C-contiguous float32
    /// array: magic, version, a 2-byte little-endian header length, then an
    /// ASCII dict padded with spaces so the header ends on a 64-byte boundary.
    static func npy(_ values: [Float], shape: [Int]) -> Data {
        let dimensions = shape.isEmpty ? "()" : "(" + shape.map { "\($0)," }.joined() + ")"
        var header = "{'descr': '<f4', 'fortran_order': False, 'shape': \(dimensions), }"
        let prefix = 6 + 2 + 2  // magic + version + header length field
        // The trailing newline is part of the header numpy expects.
        while (prefix + header.count + 1) % 64 != 0 { header += " " }
        header += "\n"

        var data = Data([0x93])
        data.append(contentsOf: Array("NUMPY".utf8))
        data.append(contentsOf: [0x01, 0x00])
        let length = UInt16(header.utf8.count)
        data.append(contentsOf: [UInt8(length & 0xff), UInt8(length >> 8)])
        data.append(contentsOf: Array(header.utf8))
        data.reserveCapacity(data.count + values.count * 4)
        for value in values {
            withUnsafeBytes(of: value.bitPattern.littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }
}
