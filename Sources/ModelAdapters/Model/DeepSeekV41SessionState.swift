import Foundation
import MLX
import StorageCore

/// An Engram row-address source whose n-gram state can leave the process.
///
/// The addresses a token gets depend on the tokens before it, so a generation
/// that is put down and picked up again has to carry that history. It is a
/// separate protocol rather than two more requirements on
/// ``DeepSeekV41EngramIndexSource`` because a source that cannot be carried
/// should be refused by name when a snapshot is asked for, not be made to
/// pretend.
public protocol DeepSeekV41RestorableEngramIndexSource: DeepSeekV41EngramIndexSource {
    /// Everything the source would need to produce the same next rows.
    var sessionStateWords: [Int64] { get }
    /// Put back what ``sessionStateWords`` returned, on a freshly built source.
    func restoreSessionState(words: [Int64]) throws
}

extension DeepSeekV41PublishedEngramIndexSource: DeepSeekV41RestorableEngramIndexSource {
    public var sessionStateWords: [Int64] { hasher.stateWords }

    public func restoreSessionState(words: [Int64]) throws {
        try hasher.restore(stateWords: words)
    }
}

/// The two facts about the weights a snapshot is bound to that the model
/// cannot see for itself: which publication was opened, and the digest of its
/// `index.json`. The caller that opened the container supplies them.
public struct DeepSeekV41SessionBinding: Codable, Equatable, Sendable {
    public let publicationRevision: String
    public let indexSHA256: String

    public init(publicationRevision: String, indexSHA256: String) {
        self.publicationRevision = publicationRevision
        self.indexSHA256 = indexSHA256
    }
}

/// Why a snapshot was not written or not restored, naming the field.
public struct DeepSeekV41SessionRefusal: Error, CustomStringConvertible, Equatable {
    public let field: String
    public let detail: String

    public init(field: String, detail: String) {
        self.field = field
        self.detail = detail
    }

    public var description: String { "V4.1 session state refused at \(field): \(detail)" }
}

/// What a written snapshot holds.
public struct DeepSeekV41SessionSnapshotSummary: Sendable, Equatable {
    public let fileBytes: UInt64
    public let tensorBytes: UInt64
    public let positions: Int
    public let tokenSHA256: String

    public init(fileBytes: UInt64, tensorBytes: UInt64, positions: Int, tokenSHA256: String) {
        self.fileBytes = fileBytes
        self.tensorBytes = tensorBytes
        self.positions = positions
        self.tokenSHA256 = tokenSHA256
    }
}

/// A complete V4.1 generation state on disk: everything a decode step reads
/// that is neither a weight nor its own input token.
///
/// ## What a step reads
///
/// | entry | where | carried as |
/// | --- | --- | --- |
/// | window ring, `[window, head_dim]` bfloat16 | every block | `block.<b>.window` |
/// | ring write cursor | every block | not stored: it is `position % window` |
/// | main KV latents | KV sources (`mode == .full`) | `block.<b>.compressed` |
/// | index keys | KV sources | `block.<b>.indexKeys` |
/// | compressor partial group, `[ratio, head_dim]` float32 values and scores | ratio-2 KV sources | `block.<b>.compressor.values`, `.scores` |
/// | shared runtime: published main KV and index keys | ``DeepSeekV41SharedAttentionRuntime`` | `shared.compressedKeyValues`, `shared.indexKeys` |
/// | shared runtime: Top-K for Reuse blocks, candidate pool | same | `shared.topK.*`, `shared.candidateMask.*` |
/// | shared runtime: last index score rows | same (diagnostic, never read) | `shared.lastIndexScoreRows.*` |
/// | Engram n-gram history | the generation's index source | header `engramIndexState` |
/// | positions consumed, token ids consumed | ``DeepSeekV41GenerationState`` | header `positions`, `tokens`, and `tokens` |
/// | the length of the position-0 prefill block | same | header `firstBlockTokens` |
///
/// The hyper-connection carry is **not** state: every pass starts from
/// `make_identity_pre_mix` and threads `pre_mix` through its own blocks only.
///
/// The shared runtime is carried because the reference carries it (see
/// ``DeepSeekV41GenerationState``). At the configurations this runner reads,
/// the first compressing block is both a KV source and an index source, so every
/// one of its fields is overwritten before any block of the next pass reads it;
/// the snapshot holds them anyway, because "never read" is a property of a
/// configuration and the file is not allowed to depend on one.
///
/// ## The file
///
/// ```text
/// "MRV41SS\n"                  8 bytes
/// header length                UInt64, little-endian
/// header                       JSON, sorted keys
/// payload                      every tensor, contiguous, little-endian, in
///                              the dtype it had in memory
/// ```
///
/// The header binds the payload to the publication revision and `index.json`
/// digest (``DeepSeekV41SessionBinding``), ``DeepSeekV41UpstreamInference/revision``,
/// the configuration's geometry, the exact token ids consumed (count and
/// SHA-256 of their little-endian `UInt32` images), the length of the block the
/// position-0 prefill took, the payload's own length and SHA-256, and a format
/// version. ``restore(from:model:binding:decodePositionLimit:firstBlockTokens:)``
/// refuses, naming the field, on any disagreement and on a file of any other
/// length.
///
/// Version 2 added `firstBlockTokens`. ADR 0023 makes a conversation one
/// generation of a fixed shape (the first turn's prompt prefilled as one block,
/// every later id one pass each), so the block length is part of what the state
/// *is*: two states over the same ids with different blocks are different
/// computations. A version 1 file does not say, and is refused by version.
public enum DeepSeekV41SessionState {
    public static let format = "minirun.deepseek-v41.session-state"
    public static let version = 2
    static let magic: [UInt8] = Array("MRV41SS\n".utf8)
    static let prefixBytes = 16

    /// The configuration numbers the stored tensors' meaning depends on.
    public struct Geometry: Codable, Equatable, Sendable {
        public let layers: Int
        public let hiddenSize: Int
        public let headDimension: Int
        public let ropeHeadDimension: Int
        public let window: Int
        public let compressionRatios: [Int]
        public let keyValueSourceLayers: [Int]
        public let indexSourceLayers: [Int]
        public let candidateSourceLayer: Int
        public let indexHeadDimension: Int
        public let indexTopK: Int
        public let hyperConnectionMultiplicity: Int
        public let vocabularySize: Int
        public let engramLayerIDs: [Int]
        public let engramMaximumNGramSize: Int
        public let maximumPositionCount: Int

        public init(config: DeepSeekV41Config) {
            layers = config.numberOfLayers
            hiddenSize = config.hiddenSize
            headDimension = config.attentionHeadDimension
            ropeHeadDimension = config.ropeHeadDimension
            window = config.slidingWindow
            compressionRatios = config.compressionRatios
            keyValueSourceLayers = config.keyValueSourceLayers
            indexSourceLayers = config.indexSourceLayers
            candidateSourceLayer = config.candidateSourceLayer
            indexHeadDimension = config.indexHeadDimension
            indexTopK = config.indexTopK
            hyperConnectionMultiplicity = config.hyperConnectionMultiplicity
            vocabularySize = config.vocabularySize
            engramLayerIDs = config.engramLayerIDs
            engramMaximumNGramSize = config.engramMaximumNGramSize
            maximumPositionCount = config.maximumPositionCount
        }
    }

    struct TokenDigest: Codable, Equatable {
        let count: Int
        let sha256: String
    }

    struct TensorEntry: Codable, Equatable {
        let name: String
        let dtype: String
        let shape: [Int]
        let offset: UInt64
        let bytes: UInt64
    }

    struct Header: Codable {
        let format: String
        let version: Int
        let binding: DeepSeekV41SessionBinding
        let inferenceRevision: String
        let geometry: Geometry
        let positions: Int
        let positionLimit: Int
        let tokens: TokenDigest
        /// Optional only so that an older file reaches the version check and
        /// is refused there by name, rather than failing to decode.
        let firstBlockTokens: Int?
        let engramIndexState: [Int64]?
        let tensors: [TensorEntry]
        let payloadBytes: UInt64
        let payloadSHA256: String
    }

    /// SHA-256 over the ids as little-endian `UInt32`, the digest the header
    /// carries.
    public static func tokenSHA256(_ tokenIDs: [Int]) -> String {
        var bytes = [UInt8]()
        bytes.reserveCapacity(tokenIDs.count * 4)
        for id in tokenIDs {
            let word = UInt32(truncatingIfNeeded: id).littleEndian
            withUnsafeBytes(of: word) { bytes.append(contentsOf: $0) }
        }
        return SHA256.hexString(SHA256.hash(bytes))
    }

    // MARK: - Snapshot

    /// Write `state` to `url` in one atomic replace.
    ///
    /// Evaluates every array the state holds, which a decode step would do
    /// anyway; it changes no value.
    @discardableResult
    public static func snapshot(
        _ state: DeepSeekV41GenerationState,
        model: DeepSeekV41Model,
        binding: DeepSeekV41SessionBinding,
        to url: URL
    ) throws -> DeepSeekV41SessionSnapshotSummary {
        try requireLittleEndian()
        let config = model.config
        guard state.caches.count == config.numberOfLayers, state.nextPosition > 0 else {
            throw DeepSeekV41SessionRefusal(
                field: "state",
                detail: "a released or never-prefilled generation has nothing to snapshot")
        }
        guard state.consumedTokenIDs.count == state.nextPosition else {
            throw DeepSeekV41SessionRefusal(
                field: "tokens",
                detail: "\(state.consumedTokenIDs.count) ids recorded against "
                    + "\(state.nextPosition) positions consumed")
        }
        guard state.firstBlockTokenCount > 0,
            state.firstBlockTokenCount <= state.nextPosition
        else {
            throw DeepSeekV41SessionRefusal(
                field: "firstBlockTokens",
                detail: "a first block of \(state.firstBlockTokenCount) ids cannot open "
                    + "\(state.nextPosition) positions")
        }
        var engramWords: [Int64]?
        if !config.engramLayerIDs.isEmpty {
            guard let source = state.engramIndices as? any DeepSeekV41RestorableEngramIndexSource
            else {
                throw DeepSeekV41SessionRefusal(
                    field: "engramIndexState",
                    detail: "this generation's Engram row-address source cannot be carried")
            }
            engramWords = source.sessionStateWords
        }

        var writer = PayloadWriter()
        for (block, caches) in state.caches.enumerated() {
            try writer.append("block.\(block).window", caches.window)
            if let compressed = caches.compressed {
                try writer.append("block.\(block).compressed", compressed)
            }
            if let keys = caches.indexKeys {
                try writer.append("block.\(block).indexKeys", keys)
            }
            if let compressor = caches.compressorState {
                try writer.append("block.\(block).compressor.values", compressor.values)
                try writer.append("block.\(block).compressor.scores", compressor.scores)
            }
        }
        let shared = state.shared
        if let compressed = shared.compressedKeyValues {
            try writer.append("shared.compressedKeyValues", compressed)
        }
        if let keys = shared.indexKeys {
            try writer.append("shared.indexKeys", keys)
        }
        if let rows = shared.topKIndices {
            writer.appendRows("shared.topK", rows, dtype: "int64") { value, bytes in
                withUnsafeBytes(of: Int64(value).littleEndian) { bytes.append(contentsOf: $0) }
            }
        }
        if let rows = shared.candidateMask {
            writer.appendRows("shared.candidateMask", rows, dtype: "uint8") { value, bytes in
                bytes.append(value ? 1 : 0)
            }
        }
        if let rows = shared.lastIndexScoreRows {
            writer.appendRows("shared.lastIndexScoreRows", rows, dtype: "float32") {
                value, bytes in
                withUnsafeBytes(of: value.bitPattern.littleEndian) {
                    bytes.append(contentsOf: $0)
                }
            }
        }
        var tokenBytes = Data()
        tokenBytes.reserveCapacity(state.consumedTokenIDs.count * 4)
        for id in state.consumedTokenIDs {
            guard let word = UInt32(exactly: id) else {
                throw DeepSeekV41SessionRefusal(
                    field: "tokens", detail: "token id \(id) does not fit a UInt32")
            }
            withUnsafeBytes(of: word.littleEndian) { tokenBytes.append(contentsOf: $0) }
        }
        writer.appendRaw(
            "tokens", dtype: "uint32", shape: [state.consumedTokenIDs.count], bytes: tokenBytes)

        let tokenDigest = tokenSHA256(state.consumedTokenIDs)
        let header = Header(
            format: format,
            version: version,
            binding: binding,
            inferenceRevision: DeepSeekV41UpstreamInference.revision,
            geometry: Geometry(config: config),
            positions: state.nextPosition,
            positionLimit: state.positionLimit,
            tokens: TokenDigest(count: state.consumedTokenIDs.count, sha256: tokenDigest),
            firstBlockTokens: state.firstBlockTokenCount,
            engramIndexState: engramWords,
            tensors: writer.entries,
            payloadBytes: UInt64(writer.payload.count),
            payloadSHA256: SHA256.hexString(SHA256.hash(writer.payload)))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let headerData = try encoder.encode(header)

        var file = Data()
        file.reserveCapacity(prefixBytes + headerData.count + writer.payload.count)
        file.append(contentsOf: magic)
        withUnsafeBytes(of: UInt64(headerData.count).littleEndian) {
            file.append(contentsOf: $0)
        }
        file.append(headerData)
        file.append(writer.payload)
        try file.write(to: url, options: .atomic)
        return DeepSeekV41SessionSnapshotSummary(
            fileBytes: UInt64(file.count),
            tensorBytes: UInt64(writer.payload.count),
            positions: state.nextPosition,
            tokenSHA256: tokenDigest)
    }

    // MARK: - Restore

    /// Rebuild a generation state for `model` from `url`.
    ///
    /// - Parameter decodePositionLimit: the ceiling for the run that continues
    ///   it. It must cover the positions already consumed and lie inside the
    ///   positions `model` was built for.
    /// - Parameter firstBlockTokens: the first-turn block the caller's
    ///   conversation was split at. A state opened with a different block is a
    ///   different computation of the same ids, and is refused by name. Nil
    ///   accepts whatever block the file records.
    public static func restore(
        from url: URL,
        model: DeepSeekV41Model,
        binding: DeepSeekV41SessionBinding,
        decodePositionLimit: Int,
        firstBlockTokens expectedFirstBlock: Int? = nil
    ) throws -> DeepSeekV41GenerationState {
        try requireLittleEndian()
        let file: Data
        do {
            file = try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            throw DeepSeekV41SessionRefusal(
                field: "file", detail: "\(url.path) cannot be read: \(error)")
        }
        guard file.count >= prefixBytes else {
            throw DeepSeekV41SessionRefusal(
                field: "file", detail: "truncated: \(file.count) B is shorter than the prefix")
        }
        guard Array(file.prefix(magic.count)) == magic else {
            throw DeepSeekV41SessionRefusal(
                field: "magic", detail: "this is not a V4.1 session state file")
        }
        let headerLength = file.withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: magic.count, as: UInt64.self))
        }
        guard headerLength <= UInt64(file.count - prefixBytes) else {
            throw DeepSeekV41SessionRefusal(
                field: "header",
                detail: "truncated: the header declares \(headerLength) B and the file holds "
                    + "\(file.count - prefixBytes) B after the prefix")
        }
        let headerEnd = prefixBytes + Int(headerLength)
        let header: Header
        do {
            header = try JSONDecoder().decode(
                Header.self, from: file.subdata(in: prefixBytes..<headerEnd))
        } catch {
            throw DeepSeekV41SessionRefusal(
                field: "header", detail: "the header does not decode: \(error)")
        }

        func require(_ condition: Bool, _ field: String, _ detail: @autoclosure () -> String)
            throws
        {
            if !condition { throw DeepSeekV41SessionRefusal(field: field, detail: detail()) }
        }
        try require(
            header.format == format, "format", "\(header.format) is not \(format)")
        try require(
            header.version == version, "version",
            "format version \(header.version); this build reads \(version)")
        try require(
            header.binding.publicationRevision == binding.publicationRevision,
            "binding.publicationRevision",
            "written against \(header.binding.publicationRevision), opened "
                + binding.publicationRevision)
        try require(
            header.binding.indexSHA256 == binding.indexSHA256, "binding.indexSHA256",
            "written against index.json \(header.binding.indexSHA256), opened "
                + binding.indexSHA256)
        try require(
            header.inferenceRevision == DeepSeekV41UpstreamInference.revision,
            "inferenceRevision",
            "written by the arithmetic of \(header.inferenceRevision); this runner "
                + "implements \(DeepSeekV41UpstreamInference.revision)")
        let config = model.config
        let geometry = Geometry(config: config)
        if header.geometry != geometry {
            throw DeepSeekV41SessionRefusal(
                field: "geometry." + (try firstDifference(header.geometry, geometry)),
                detail: "the snapshot's configuration is not this model's")
        }

        let payloadLength = UInt64(file.count - headerEnd)
        try require(
            payloadLength == header.payloadBytes, "payload",
            "truncated or extended: the header declares \(header.payloadBytes) B and the "
                + "file holds \(payloadLength) B")
        let payload = file.subdata(in: headerEnd..<file.count)
        try require(
            SHA256.hexString(SHA256.hash(payload)) == header.payloadSHA256, "payloadSHA256",
            "the payload does not hash to the header's digest")

        let reader = try PayloadReader(entries: header.tensors, payload: payload)
        let tokenBytes = try reader.raw("tokens", dtype: "uint32")
        try require(
            tokenBytes.shape == [header.tokens.count], "tokens.count",
            "the header declares \(header.tokens.count) ids and the payload holds "
                + "\(tokenBytes.shape)")
        let tokens: [Int] = tokenBytes.data.withUnsafeBytes { raw in
            (0..<header.tokens.count).map {
                Int(UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self)))
            }
        }
        try require(
            tokenSHA256(tokens) == header.tokens.sha256, "tokens.sha256",
            "the stored ids do not hash to the header's digest")
        try require(
            header.positions == tokens.count && header.positions > 0, "positions",
            "\(header.positions) positions against \(tokens.count) ids")
        try require(
            tokens.allSatisfy { $0 < config.vocabularySize }, "tokens",
            "an id lies outside the \(config.vocabularySize)-entry vocabulary")
        guard let firstBlock = header.firstBlockTokens, firstBlock > 0,
            firstBlock <= header.positions
        else {
            throw DeepSeekV41SessionRefusal(
                field: "firstBlockTokens",
                detail: "the header records a first block of "
                    + "\(header.firstBlockTokens.map(String.init) ?? "no") ids for "
                    + "\(header.positions) positions")
        }
        if let expectedFirstBlock {
            try require(
                firstBlock == expectedFirstBlock, "firstBlockTokens",
                "the state was opened with a \(firstBlock)-id first block and this "
                    + "conversation's first turn is \(expectedFirstBlock) ids")
        }
        try require(
            decodePositionLimit >= header.positions
                && decodePositionLimit <= model.positionCount,
            "decodePositionLimit",
            "\(decodePositionLimit) must cover the \(header.positions) positions consumed "
                + "and lie inside the \(model.positionCount) this model was built for")

        var caches = [DeepSeekV41AttentionCaches]()
        caches.reserveCapacity(config.numberOfLayers)
        for block in 0..<config.numberOfLayers {
            let blockGeometry = model.geometries[block]
            var cache = try DeepSeekV41AttentionCaches(geometry: blockGeometry)
            cache.window = try reader.array(
                "block.\(block).window", dtype: "bfloat16",
                shape: [blockGeometry.windowSize, blockGeometry.headDimension])
            let ownsKeyValues = blockGeometry.mode == .full && blockGeometry.compressionRatio > 0
            let rows = ownsKeyValues ? header.positions / blockGeometry.compressionRatio : 0
            if rows > 0 {
                cache.compressed = try reader.array(
                    "block.\(block).compressed", shape: [rows, blockGeometry.headDimension])
                cache.indexKeys = try reader.array(
                    "block.\(block).indexKeys",
                    shape: [rows, blockGeometry.indexHeadDimension])
            } else {
                try reader.requireAbsent("block.\(block).compressed")
                try reader.requireAbsent("block.\(block).indexKeys")
            }
            if cache.compressorState != nil {
                let shape = [blockGeometry.compressionRatio, blockGeometry.headDimension]
                cache.compressorState = DeepSeekV41Compressor.State(
                    values: try reader.array(
                        "block.\(block).compressor.values", dtype: "float32", shape: shape),
                    scores: try reader.array(
                        "block.\(block).compressor.scores", dtype: "float32", shape: shape))
            } else {
                try reader.requireAbsent("block.\(block).compressor.values")
                try reader.requireAbsent("block.\(block).compressor.scores")
            }
            caches.append(cache)
        }

        let shared = DeepSeekV41SharedAttentionRuntime()
        shared.compressedKeyValues = try reader.optionalArray("shared.compressedKeyValues")
        shared.indexKeys = try reader.optionalArray("shared.indexKeys")
        shared.topKIndices = try reader.rows("shared.topK", dtype: "int64", width: 8) {
            raw, offset in Int(Int64(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: Int64.self)))
        }
        shared.candidateMask = try reader.rows("shared.candidateMask", dtype: "uint8", width: 1) {
            raw, offset in raw[offset] != 0
        }
        shared.lastIndexScoreRows = try reader.rows(
            "shared.lastIndexScoreRows", dtype: "float32", width: 4
        ) { raw, offset in
            Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
        }
        try reader.requireAllConsumed()

        var engramIndices: (any DeepSeekV41EngramIndexSource)?
        if !config.engramLayerIDs.isEmpty {
            guard let words = header.engramIndexState else {
                throw DeepSeekV41SessionRefusal(
                    field: "engramIndexState",
                    detail: "this configuration has Engram blocks and the snapshot carries no "
                        + "n-gram state")
            }
            guard let fresh = try model.makeEngramIndices?(),
                let restorable = fresh as? any DeepSeekV41RestorableEngramIndexSource
            else {
                throw DeepSeekV41SessionRefusal(
                    field: "engramIndexState",
                    detail: "this model's Engram row-address source cannot be restored")
            }
            do {
                try restorable.restoreSessionState(words: words)
            } catch {
                throw DeepSeekV41SessionRefusal(
                    field: "engramIndexState", detail: "\(error)")
            }
            engramIndices = restorable
        } else {
            try require(
                header.engramIndexState == nil, "engramIndexState",
                "the snapshot carries Engram state and this configuration has no Engram block")
        }

        return DeepSeekV41GenerationState(
            caches: caches,
            shared: shared,
            engramIndices: engramIndices,
            nextPosition: header.positions,
            positionLimit: decodePositionLimit,
            consumedTokenIDs: tokens,
            firstBlockTokenCount: firstBlock)
    }

    // MARK: - Pieces

    private static func requireLittleEndian() throws {
        guard 1.littleEndian == 1 else {
            throw DeepSeekV41SessionRefusal(
                field: "byteOrder", detail: "session state files are little-endian")
        }
    }

    private static func firstDifference(_ stored: Geometry, _ current: Geometry) throws
        -> String
    {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let left = try JSONSerialization.jsonObject(with: encoder.encode(stored))
            as? [String: Any] ?? [:]
        let right = try JSONSerialization.jsonObject(with: encoder.encode(current))
            as? [String: Any] ?? [:]
        for key in left.keys.sorted() where !(left[key] as AnyObject).isEqual(right[key]) {
            return "\(key) (stored \(left[key] ?? "none"), model \(right[key] ?? "none"))"
        }
        return "unknown"
    }

    static let dtypeNames: [String: DType] = [
        "bool": .bool, "uint8": .uint8, "uint32": .uint32, "int32": .int32,
        "int64": .int64, "float16": .float16, "bfloat16": .bfloat16, "float32": .float32,
    ]

    static func name(of dtype: DType) throws -> String {
        guard let name = dtypeNames.first(where: { $0.value == dtype })?.key else {
            throw DeepSeekV41SessionRefusal(
                field: "dtype", detail: "\(dtype) has no session-state name")
        }
        return name
    }

    struct PayloadWriter {
        var payload = Data()
        var entries: [TensorEntry] = []

        mutating func appendRaw(_ name: String, dtype: String, shape: [Int], bytes: Data) {
            entries.append(
                TensorEntry(
                    name: name, dtype: dtype, shape: shape,
                    offset: UInt64(payload.count), bytes: UInt64(bytes.count)))
            payload.append(bytes)
        }

        mutating func append(_ name: String, _ array: MLXArray) throws {
            let data = array.asData(access: .copy)
            appendRaw(
                name, dtype: try DeepSeekV41SessionState.name(of: data.dType),
                shape: data.shape, bytes: data.data)
        }

        /// A ragged list as two entries: `<name>.values`, every row
        /// concatenated, and `<name>.rowLengths`, int64.
        mutating func appendRows<Element>(
            _ name: String, _ rows: [[Element]], dtype: String,
            encode: (Element, inout Data) -> Void
        ) {
            var values = Data()
            var lengths = Data()
            var count = 0
            for row in rows {
                withUnsafeBytes(of: Int64(row.count).littleEndian) {
                    lengths.append(contentsOf: $0)
                }
                for value in row { encode(value, &values) }
                count += row.count
            }
            appendRaw(name + ".values", dtype: dtype, shape: [count], bytes: values)
            appendRaw(name + ".rowLengths", dtype: "int64", shape: [rows.count], bytes: lengths)
        }
    }

    final class PayloadReader {
        private let payload: Data
        private var entries: [String: TensorEntry]
        private var consumed = Set<String>()

        init(entries list: [TensorEntry], payload: Data) throws {
            self.payload = payload
            var entries = [String: TensorEntry]()
            for entry in list {
                guard entries[entry.name] == nil else {
                    throw DeepSeekV41SessionRefusal(
                        field: entry.name, detail: "appears twice in the tensor table")
                }
                guard let width = DeepSeekV41SessionState.width(entry.dtype) else {
                    throw DeepSeekV41SessionRefusal(
                        field: entry.name, detail: "unknown dtype \(entry.dtype)")
                }
                let elements = entry.shape.reduce(1) { $0 * $1 }
                guard entry.shape.allSatisfy({ $0 >= 0 }),
                    UInt64(elements * width) == entry.bytes,
                    entry.offset.addingReportingOverflow(entry.bytes).partialValue
                        <= UInt64(payload.count)
                else {
                    throw DeepSeekV41SessionRefusal(
                        field: entry.name,
                        detail: "shape \(entry.shape) of \(entry.dtype) does not match its "
                            + "\(entry.bytes) B at offset \(entry.offset) in a "
                            + "\(payload.count) B payload")
                }
                entries[entry.name] = entry
            }
            self.entries = entries
        }

        func raw(_ name: String, dtype: String) throws -> (shape: [Int], data: Data) {
            guard let entry = entries[name] else {
                throw DeepSeekV41SessionRefusal(field: name, detail: "missing")
            }
            guard entry.dtype == dtype else {
                throw DeepSeekV41SessionRefusal(
                    field: name, detail: "stored as \(entry.dtype), expected \(dtype)")
            }
            consumed.insert(name)
            let start = Int(entry.offset)
            return (entry.shape, payload.subdata(in: start..<(start + Int(entry.bytes))))
        }

        func array(_ name: String, dtype: String? = nil, shape: [Int]) throws -> MLXArray {
            guard let entry = entries[name] else {
                throw DeepSeekV41SessionRefusal(field: name, detail: "missing")
            }
            guard entry.shape == shape else {
                throw DeepSeekV41SessionRefusal(
                    field: name, detail: "stored shape \(entry.shape), expected \(shape)")
            }
            return try optionalArray(name, dtype: dtype)!
        }

        func optionalArray(_ name: String, dtype: String? = nil) throws -> MLXArray? {
            guard let entry = entries[name] else { return nil }
            if let dtype, entry.dtype != dtype {
                throw DeepSeekV41SessionRefusal(
                    field: name, detail: "stored as \(entry.dtype), expected \(dtype)")
            }
            guard let type = DeepSeekV41SessionState.dtypeNames[entry.dtype],
                !entry.shape.isEmpty, entry.shape.allSatisfy({ $0 > 0 })
            else {
                throw DeepSeekV41SessionRefusal(
                    field: name, detail: "\(entry.dtype) \(entry.shape) is not an array")
            }
            let bytes = try raw(name, dtype: entry.dtype).data
            return MLXArray(bytes, entry.shape, dtype: type)
        }

        func requireAbsent(_ name: String) throws {
            guard entries[name] == nil else {
                throw DeepSeekV41SessionRefusal(
                    field: name, detail: "present where this block's mode holds none")
            }
        }

        func rows<Element>(
            _ name: String, dtype: String, width: Int,
            decode: (UnsafeRawBufferPointer, Int) -> Element
        ) throws -> [[Element]]? {
            let valuesName = name + ".values"
            let lengthsName = name + ".rowLengths"
            switch (entries[valuesName] != nil, entries[lengthsName] != nil) {
            case (false, false): return nil
            case (true, true): break
            default:
                throw DeepSeekV41SessionRefusal(
                    field: name, detail: "values and row lengths must both be present")
            }
            let values = try raw(valuesName, dtype: dtype)
            let lengths = try raw(lengthsName, dtype: "int64")
            let counts: [Int] = lengths.data.withUnsafeBytes { raw in
                (0..<lengths.shape[0]).map {
                    Int(Int64(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 8, as: Int64.self)))
                }
            }
            guard counts.allSatisfy({ $0 >= 0 }), counts.reduce(0, +) == values.shape[0] else {
                throw DeepSeekV41SessionRefusal(
                    field: name, detail: "row lengths do not add up to the stored values")
            }
            return values.data.withUnsafeBytes { raw in
                var result = [[Element]]()
                result.reserveCapacity(counts.count)
                var index = 0
                for count in counts {
                    result.append((0..<count).map { decode(raw, (index + $0) * width) })
                    index += count
                }
                return result
            }
        }

        func requireAllConsumed() throws {
            if let stray = entries.keys.sorted().first(where: { !consumed.contains($0) }) {
                throw DeepSeekV41SessionRefusal(
                    field: stray, detail: "stored and not part of any state this model holds")
            }
        }
    }

    static func width(_ dtype: String) -> Int? {
        switch dtype {
        case "bool", "uint8": return 1
        case "float16", "bfloat16": return 2
        case "uint32", "int32", "float32": return 4
        case "int64": return 8
        default: return nil
        }
    }
}
