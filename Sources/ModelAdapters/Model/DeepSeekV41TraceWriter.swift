import Foundation
import MLX

/// Writes a V4.1 forward trace to disk in the layout
/// `Tools/v41_flash/v41_trace.py` reads, so a Minirun pass and a pass of
/// DeepSeek's own `inference/model.py` can be compared quantity by quantity.
///
/// ## Why it exists
///
/// The Mac runner and the cluster reference produce the same eighteen ids for
/// the standing prompt and logits that differ by 0.45 rms per position
/// (`docs/experiments/2026-11-10-v41-mac-vs-reference-divergence.md`). A logits
/// comparison cannot say where that difference starts, and the Phase 1 to 3
/// operator agreements were measured against a transcription at a 32-wide mini
/// checkpoint, not against DeepSeek's kernels at the published geometry. This
/// writes the same quantities the reference tracer writes, in the same packing,
/// so the first block and branch at which the two part company can be named.
///
/// ## Layout
///
/// ```text
/// <dir>/trace.json                        format, producer, how many passes carry tensors
/// <dir>/pass-<ppp>/pass.json              start position, token ids, greedy token
/// <dir>/pass-<ppp>/<quantity>-<bb>.bin    float32, little-endian, row-major, no header
/// <dir>/pass-<ppp>/tensors.json           the sidecar: per file its quantity, block, shape, dtype
/// <dir>/pass-<ppp>/routing.json           per block: selected ids, weights, top 8 by score
/// <dir>/pass-<ppp>/index.json             per selecting block: Top-K lists, candidate mask
/// <dir>/pass-<ppp>/engram.json            per Engram block: the table rows addressed
/// ```
///
/// Quantities, all `float32` widened exactly from what the runner held:
///
/// | quantity | shape | what |
/// | --- | --- | --- |
/// | `stream` | `[tokens, hc, hidden]` | entering block `bb`, before its Engram; `bb == 40` is what the final collapse reads |
/// | `premix` | `[tokens, hc]` | the collapse coefficients handed to block `bb` |
/// | `attn` | `[tokens, hidden]` | the attention branch's output, before `hc_post` |
/// | `ffn` | `[tokens, hidden]` | the MoE branch's output, before `hc_post` |
/// | `engram` | `[tokens, hc, hidden]` | the stream Engram returned at block `bb` |
///
/// Tensors are written for the first ``tensorPasses`` passes only; routing,
/// index selections and Engram rows are small and are written for every pass.
///
/// Writing is synchronous and reads each array back to the host, which forces
/// evaluation of values the pass would have evaluated anyway. Nothing about the
/// arithmetic changes; the run's logits digest is the check that this is so.
public final class DeepSeekV41TraceWriter: DeepSeekV41ForwardTrace {
    public static let formatName = "minirun-v41-trace"
    public static let formatVersion = 1
    /// How many of the highest biased routing scores a token's record keeps:
    /// more than the six the router selects, so a near-tie at the selection
    /// boundary is visible on both sides of it.
    public static let routingTopCount = 8

    public let directory: URL
    public let tensorPasses: Int
    /// Messages for writes that failed. A trace is an instrument attached to a
    /// run, and a full disk must not turn the run into a refusal.
    public private(set) var failures: [String] = []
    public private(set) var passCount = 0

    private var passIndex = -1
    private var passDirectory: URL?
    private var startPosition = 0
    private var tokenIDs: [Int] = []
    private var routing: [String: RoutingRecord] = [:]
    private var index: [String: IndexRecord] = [:]
    private var engram: [String: EngramRecord] = [:]
    private var tensors: [String: TensorRecord] = [:]
    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()

    /// - Parameters:
    ///   - producer: who computed the values, recorded in `trace.json`.
    ///   - metadata: anything else that identifies the run (revisions, knobs).
    public init(
        directory: URL, tensorPasses: Int, producer: String,
        metadata: [String: String] = [:]
    ) throws {
        self.directory = directory
        self.tensorPasses = max(0, tensorPasses)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let header = Header(
            format: Self.formatName, version: Self.formatVersion, producer: producer,
            tensorPasses: self.tensorPasses, routingTop: Self.routingTopCount,
            metadata: metadata)
        try encoder.encode(header).write(
            to: directory.appendingPathComponent("trace.json"), options: .atomic)
    }

    // MARK: - DeepSeekV41ForwardTrace

    public func beginPass(startPosition: Int, tokenIDs: [Int]) {
        passIndex += 1
        self.startPosition = startPosition
        self.tokenIDs = tokenIDs
        routing = [:]
        index = [:]
        engram = [:]
        tensors = [:]
        let url = directory.appendingPathComponent(Self.passName(passIndex))
        do {
            try FileManager.default.createDirectory(
                at: url, withIntermediateDirectories: true)
            passDirectory = url
        } catch {
            fail("pass \(passIndex): \(error.localizedDescription)")
            passDirectory = nil
        }
    }

    public func recordStreamState(block: Int, stream: MLXArray, preMix: MLXArray) {
        writeTensor("stream", block: block, stream)
        writeTensor("premix", block: block, preMix)
    }

    public func recordAttentionBranch(block: Int, output: MLXArray) {
        writeTensor("attn", block: block, output)
    }

    public func recordFeedForwardBranch(block: Int, output: MLXArray) {
        writeTensor("ffn", block: block, output)
    }

    public func recordEngram(block: Int, rows: [[Int]], output: MLXArray) {
        engram[String(block)] = EngramRecord(rows: rows)
        writeTensor("engram", block: block, output)
    }

    public func recordRouting(block: Int, selection: DeepSeekV41Router.Selection) {
        let ranked = selection.ranked.asType(.float32)
        let scores = selection.scores.asType(.float32)
        guard ranked.ndim == 2, scores.shape == ranked.shape else {
            fail("pass \(passIndex) block \(block): routing scores are \(ranked.shape)")
            return
        }
        let experts = ranked.shape[1]
        let rankedValues = ranked.asArray(Float.self)
        let scoreValues = scores.asArray(Float.self)
        let keep = Swift.min(Self.routingTopCount, experts)
        var topIDs = [[Int]]()
        var topRanked = [[Double]]()
        var topScores = [[Double]]()
        for token in 0..<ranked.shape[0] {
            let start = token * experts
            // The router's own order: descending biased score, lower id first
            // at a tie (``DeepSeekV41Router``).
            let order = (0..<experts).sorted { left, right in
                let lhs = rankedValues[start + left]
                let rhs = rankedValues[start + right]
                return lhs == rhs ? left < right : lhs > rhs
            }.prefix(keep)
            topIDs.append(Array(order))
            topRanked.append(order.map { Double(rankedValues[start + $0]) })
            topScores.append(order.map { Double(scoreValues[start + $0]) })
        }
        routing[String(block)] = RoutingRecord(
            selected: selection.ids,
            weights: selection.weightValues.map { $0.map(Double.init) },
            topIDs: topIDs, topRanked: topRanked, topScores: topScores)
    }

    public func recordIndexSelection(
        block: Int, indices: [[Int]], offset: Int, candidateMask: [[Bool]]?
    ) {
        index[String(block)] = IndexRecord(
            indices: indices, offset: offset,
            candidates: candidateMask.map { mask in
                mask.map { row in row.indices.filter { row[$0] } }
            })
    }

    public func endPass(greedyTokenID: Int) {
        defer { passCount = passIndex + 1 }
        guard let passDirectory else { return }
        let summary = PassRecord(
            pass: passIndex, startPosition: startPosition, tokenIDs: tokenIDs,
            greedyTokenID: greedyTokenID, tensors: passIndex < tensorPasses)
        write(summary, to: passDirectory.appendingPathComponent("pass.json"))
        write(
            RoutingFile(top: Self.routingTopCount, blocks: routing),
            to: passDirectory.appendingPathComponent("routing.json"))
        write(IndexFile(blocks: index), to: passDirectory.appendingPathComponent("index.json"))
        write(EngramFile(blocks: engram), to: passDirectory.appendingPathComponent("engram.json"))
        write(tensors, to: passDirectory.appendingPathComponent("tensors.json"))
    }

    // MARK: - Files

    public static func passName(_ pass: Int) -> String {
        "pass-" + String(format: "%03d", pass)
    }

    public static func tensorName(_ quantity: String, block: Int) -> String {
        quantity + "-" + String(format: "%02d", block)
    }

    /// The bytes a tensor is written as: its float32 values, little-endian,
    /// row-major.
    public static func rawBytes(_ values: [Float]) -> Data {
        var data = Data(capacity: values.count * 4)
        for value in values {
            var word = value.bitPattern.littleEndian
            withUnsafeBytes(of: &word) { data.append(contentsOf: $0) }
        }
        return data
    }

    private func writeTensor(_ quantity: String, block: Int, _ array: MLXArray) {
        guard passIndex >= 0, passIndex < tensorPasses, let passDirectory else { return }
        let values = array.asType(.float32).asArray(Float.self)
        let raw = Self.rawBytes(values)
        let name = Self.tensorName(quantity, block: block)
        do {
            try raw.write(
                to: passDirectory.appendingPathComponent(name + ".bin"), options: .atomic)
        } catch {
            fail("\(name): \(error.localizedDescription)")
            return
        }
        tensors[name + ".bin"] = TensorRecord(
            quantity: quantity, pass: passIndex, block: block, shape: array.shape,
            dtype: "float32", byteOrder: "little", byteCount: raw.count,
            sourceDtype: String(describing: array.dtype))
    }

    private func write<T: Encodable>(_ value: T, to url: URL) {
        do {
            try encoder.encode(value).write(to: url, options: .atomic)
        } catch {
            fail("\(url.lastPathComponent): \(error.localizedDescription)")
        }
    }

    private func fail(_ message: String) {
        failures.append(message)
        FileHandle.standardError.write(Data("v41 trace: \(message)\n".utf8))
    }

    // MARK: - Records

    private struct Header: Encodable {
        let format: String
        let version: Int
        let producer: String
        let tensorPasses: Int
        let routingTop: Int
        let metadata: [String: String]

        enum CodingKeys: String, CodingKey {
            case format, version, producer, metadata
            case tensorPasses = "tensor_passes"
            case routingTop = "routing_top"
        }
    }

    private struct PassRecord: Encodable {
        let pass: Int
        let startPosition: Int
        let tokenIDs: [Int]
        let greedyTokenID: Int
        let tensors: Bool

        enum CodingKeys: String, CodingKey {
            case pass, tensors
            case startPosition = "start_position"
            case tokenIDs = "token_ids"
            case greedyTokenID = "greedy_token_id"
        }
    }

    private struct TensorRecord: Encodable {
        let quantity: String
        let pass: Int
        let block: Int
        let shape: [Int]
        let dtype: String
        let byteOrder: String
        let byteCount: Int
        let sourceDtype: String

        enum CodingKeys: String, CodingKey {
            case quantity, pass, block, shape, dtype
            case byteOrder = "byte_order"
            case byteCount = "byte_count"
            case sourceDtype = "source_dtype"
        }
    }

    private struct RoutingRecord: Encodable {
        let selected: [[Int]]
        let weights: [[Double]]
        let topIDs: [[Int]]
        let topRanked: [[Double]]
        let topScores: [[Double]]

        enum CodingKeys: String, CodingKey {
            case selected, weights
            case topIDs = "top_ids"
            case topRanked = "top_ranked"
            case topScores = "top_scores"
        }
    }

    private struct RoutingFile: Encodable {
        let top: Int
        let blocks: [String: RoutingRecord]
    }

    private struct IndexRecord: Encodable {
        let indices: [[Int]]
        let offset: Int
        let candidates: [[Int]]?
    }

    private struct IndexFile: Encodable {
        let blocks: [String: IndexRecord]
    }

    private struct EngramRecord: Encodable {
        let rows: [[Int]]
    }

    private struct EngramFile: Encodable {
        let blocks: [String: EngramRecord]
    }
}
