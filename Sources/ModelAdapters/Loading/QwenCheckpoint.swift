import Foundation
import MLX
import MLXBridge
import StorageCore

/// The sharded Qwen checkpoint, seen as one namespace.
///
/// `SafetensorsFile` already does the hard part — bounds-checked mmap, row
/// addressing, BF16 widening — so this is only the shard index on top of it. It
/// is a separate type rather than a mode of `SafetensorsFile` because sharding
/// is a property of *this* checkpoint, and pushing it down would make the tiny
/// K3 reader carry a concept it has no use for.
///
/// Mapping rather than reading matters more here than at tiny scale: the four
/// shards are 17.2 GB, of which the 15.2 GB of routed experts must never be
/// touched through this path at all. They are streamed from containers instead,
/// and if a resident run ever faulted them in, the milestone's memory claim
/// would be about nothing.
public final class QwenCheckpoint {

    public let directory: String
    public let config: QwenMoEConfig
    /// `tensor name -> shard file`.
    public let weightMap: [String: String]

    /// Widen the checkpoint's bfloat16 scales and biases to float32 on load.
    ///
    /// Off by default, because the point of this project is to consume the
    /// checkpoint's bytes as they lie: a pager slot is adopted as three
    /// zero-copy `MLXArray`s and the scales go to the kernel in their stored
    /// dtype. But MLX's affine kernels carry the *scales'* dtype into their
    /// arithmetic, so that choice costs bfloat16 rounding — measured at 1.1e-03
    /// relative on the embedding, against **exactly 0.0** when the scales are
    /// widened first.
    ///
    /// The mlx-lm reference widens (`model.set_dtype(mx.float32)`), so the
    /// per-layer trace comparison has to widen too or it would be measuring a
    /// dtype choice rather than the transcription. Every other gate — the greedy
    /// tokens, the router selections, streamed-versus-resident — is unaffected,
    /// which is itself the useful result: the rounding is real and it is not
    /// large enough to move a discrete decision on a screened prompt.
    public var widenQuantizationScales: Bool = false

    private var shards: [String: SafetensorsFile] = [:]
    private let lock = NSLock()

    public init(directory: String) throws {
        self.directory = directory
        self.config = try QwenMoEConfig.load(directory: directory)

        let indexURL = URL(fileURLWithPath: directory)
            .appendingPathComponent("model.safetensors.index.json")
        guard
            let root = try JSONSerialization.jsonObject(with: try Data(contentsOf: indexURL))
                as? [String: Any],
            let map = root["weight_map"] as? [String: String]
        else {
            throw TinyK3Error.configuration("\(indexURL.path) has no 'weight_map'")
        }
        self.weightMap = map
    }

    /// Shards are opened on first use and kept. Mapping is cheap; re-mapping per
    /// tensor is not, and a layer's tensors usually share a shard.
    private func shard(for name: String) throws -> SafetensorsFile {
        guard let file = weightMap[name] else {
            throw TinyK3Error.missingTensor(name)
        }
        lock.lock()
        defer { lock.unlock() }
        if let open = shards[file] { return open }
        let opened = try SafetensorsFile(
            path: URL(fileURLWithPath: directory).appendingPathComponent(file).path)
        shards[file] = opened
        return opened
    }

    public func contains(_ name: String) -> Bool { weightMap[name] != nil }

    public func info(_ name: String) throws -> SafetensorsFile.TensorInfo {
        try shard(for: name).info(name)
    }

    /// A float32 view of a BF16 or F32 tensor. Norm weights, and nothing else in
    /// this checkpoint.
    public func float32(_ name: String) throws -> MLXArray {
        try shard(for: name).float32(name)
    }

    /// An affine-quantized linear's three tensors, wrapped without repacking.
    ///
    /// - Parameter stacked: `true` for the routed experts, whose leading axis is
    ///   the expert index.
    public func affine(
        _ base: String, groupSize: Int, bits: Int, stacked: Bool = false
    ) throws -> AffineWeights {
        // Resolved per tensor, not per base name. A quantized tensor's three
        // parts are three independent entries in the index and the shard
        // boundary can fall between them — `model.layers.29.mlp.switch_mlp
        // .down_proj` has its weight in one shard and its biases in the next.
        // Looking all three up in the weight's shard worked for 28 layers.
        let packedInfo = try info("\(base).weight")
        let scalesInfo = try info("\(base).scales")
        let biasesInfo = try info("\(base).biases")

        guard packedInfo.dtype == "U32" else {
            throw TinyK3Error.unsupportedDType(tensor: "\(base).weight", dtype: packedInfo.dtype)
        }
        guard scalesInfo.dtype == biasesInfo.dtype else {
            throw TinyK3Error.configuration(
                "\(base): scales are \(scalesInfo.dtype) but biases are \(biasesInfo.dtype)")
        }
        let scaleDType: DType
        switch scalesInfo.dtype {
        case "BF16": scaleDType = .bfloat16
        case "F16": scaleDType = .float16
        case "F32": scaleDType = .float32
        default:
            throw TinyK3Error.unsupportedDType(tensor: "\(base).scales", dtype: scalesInfo.dtype)
        }

        let expectedRank = stacked ? 3 : 2
        guard packedInfo.shape.count == expectedRank else {
            throw TinyK3Error.shapeMismatch(
                tensor: "\(base).weight",
                expected: [Int](repeating: -1, count: expectedRank), actual: packedInfo.shape)
        }

        // The bytes cross into MLX as they lie in the file. `U32` is already the
        // operand type, and BF16 scales are handed over as `.bfloat16` rather
        // than widened: MLX's affine kernel takes them in their own dtype, and
        // widening here would both cost a pass and change the arithmetic.
        let packed = try shard(for: "\(base).weight").bytes("\(base).weight")
        let scales = try shard(for: "\(base).scales").bytes("\(base).scales")
        let biases = try shard(for: "\(base).biases").bytes("\(base).biases")

        return try packed.withUnsafeBytes { packedBytes in
            try scales.withUnsafeBytes { scaleBytes in
                try biases.withUnsafeBytes { biasBytes in
                    let packedArray = packedBytes.withMemoryRebound(to: UInt32.self) {
                        MLXArray($0, packedInfo.shape)
                    }
                    // `view` and not `asType`: these are 16 bits that already
                    // *are* a bfloat16, so the operation is a reinterpretation.
                    // `asType` would read them as integers and convert, turning
                    // a scale of 0.0038 into 15,437.
                    var scalesArray = MLXArray(
                        scaleBytes.bindMemory(to: UInt16.self), scalesInfo.shape)
                        .view(dtype: scaleDType)
                    var biasesArray = MLXArray(
                        biasBytes.bindMemory(to: UInt16.self), biasesInfo.shape)
                        .view(dtype: scaleDType)
                    if widenQuantizationScales {
                        // Lossless: every bfloat16 is exactly representable in
                        // float32. It changes the kernel's working precision,
                        // not the values it starts from.
                        scalesArray = scalesArray.asType(.float32)
                        biasesArray = biasesArray.asType(.float32)
                    }
                    return try AffineWeights(
                        packed: packedArray, scales: scalesArray, biases: biasesArray,
                        groupSize: groupSize, bits: bits)
                }
            }
        }
    }

    /// Embedding rows for these token ids, dequantized. Never the table.
    ///
    /// The table is `[151936, 2048]` quantized — 175 MB with its scales — and a
    /// five-token prompt needs five rows of it (spec §16.4). This gathers the
    /// packed, scale and bias rows and dequantizes only those, which is what
    /// `mlx.nn.QuantizedEmbedding` does and the only way to get the same answer:
    /// dequantizing the table and then indexing would be arithmetically
    /// identical but 175 MB more expensive, and dequantizing is exactly what the
    /// project exists to avoid doing wholesale.
    public func embeddingRows(_ tokenIds: [Int]) throws -> MLXArray {
        let base = QwenTensors.embedding
        let weights = try affine(base, groupSize: config.quantGroupSize, bits: config.quantBits)
        let index = MLXArray(tokenIds.map { Int32($0) })
        return dequantized(
            weights.packed[index],
            scales: weights.scales[index],
            biases: weights.biases[index],
            groupSize: weights.groupSize,
            bits: weights.bits,
            mode: .affine,
            dtype: .float32)
    }
}

/// One layer's resident weights: everything except the routed experts.
///
/// The split is the milestone's whole claim. Attention, both norms and the
/// router are resident and total roughly 8.3 MB per layer; the 128 experts are
/// 340 MB per layer and are streamed. Qwen3-30B has **no shared expert**, so
/// there is no always-hot FFN component to blur the line — every feed-forward
/// byte a token touches came through the pager.
public struct QwenLayerWeights {
    public var inputNorm: MLXArray
    public var postAttentionNorm: MLXArray
    public var queryNorm: MLXArray
    public var keyNorm: MLXArray
    public var queryProj: AffineWeights
    public var keyProj: AffineWeights
    public var valueProj: AffineWeights
    public var outputProj: AffineWeights
    public var router: AffineWeights

    /// Bytes the resident half of one layer occupies, for the budget report.
    public var residentBytes: Int {
        func bytes(_ weights: AffineWeights) -> Int {
            weights.packed.size * 4 + weights.scales.size * 2 + weights.biases.size * 2
        }
        let norms = (inputNorm.size + postAttentionNorm.size + queryNorm.size + keyNorm.size) * 4
        return norms + bytes(queryProj) + bytes(keyProj) + bytes(valueProj) + bytes(outputProj)
            + bytes(router)
    }
}

/// Where a Qwen decode step gets its non-expert weights.
///
/// The same seam as ``K3WeightSource`` and for the same reason (spec §5.5): the
/// layer schedule, the operators and the numerics do not depend on whether a
/// tensor came from a mapped checkpoint or a pager slot. Only the routed experts
/// have a second implementation, because only they are large enough to matter.
public protocol QwenWeightSource: AnyObject {
    var config: QwenMoEConfig { get }
    func embeddingRows(_ tokenIds: [Int]) throws -> MLXArray
    func layerWeights(_ index: Int) throws -> QwenLayerWeights
    func releaseLayer(_ index: Int)
    var finalNormWeight: MLXArray { get }
    /// `count` rows of `lm_head` starting at `first`, as affine weights.
    func lmHead() throws -> AffineWeights
}

extension QwenWeightSource {
    public func releaseLayer(_ index: Int) {}
}

/// Resident non-expert weights, mapped from the checkpoint.
///
/// Holds every layer's attention, norms and router — about 760 MB including the
/// embedding table and LM head, against the checkpoint's 17.2 GB. Layers are
/// materialised on first use and kept, because that is what "resident" means
/// here; the streaming that this milestone is about happens on the expert side.
public final class QwenResidentWeights: QwenWeightSource {

    public let checkpoint: QwenCheckpoint
    public var config: QwenMoEConfig { checkpoint.config }

    private var layers: [Int: QwenLayerWeights] = [:]
    private var cachedFinalNorm: MLXArray?
    private var cachedLMHead: AffineWeights?
    private let lock = NSLock()

    public init(checkpoint: QwenCheckpoint) throws {
        self.checkpoint = checkpoint
    }

    public convenience init(directory: String) throws {
        try self.init(checkpoint: try QwenCheckpoint(directory: directory))
    }

    public func embeddingRows(_ tokenIds: [Int]) throws -> MLXArray {
        try checkpoint.embeddingRows(tokenIds)
    }

    public var finalNormWeight: MLXArray {
        lock.lock()
        defer { lock.unlock() }
        if let cachedFinalNorm { return cachedFinalNorm }
        // Force-try is confined to this one accessor because the protocol makes
        // it a property; a missing final norm is a corrupt checkpoint, caught by
        // `preflight()` long before a decode step asks for it.
        let value = try! checkpoint.float32(QwenTensors.finalNorm)
        cachedFinalNorm = value
        return value
    }

    public func lmHead() throws -> AffineWeights {
        lock.lock()
        defer { lock.unlock() }
        if let cachedLMHead { return cachedLMHead }
        let weights = try checkpoint.affine(
            QwenTensors.lmHead, groupSize: config.quantGroupSize, bits: config.quantBits)
        cachedLMHead = weights
        return weights
    }

    public func layerWeights(_ index: Int) throws -> QwenLayerWeights {
        lock.lock()
        defer { lock.unlock() }
        if let existing = layers[index] { return existing }
        guard index >= 0, index < config.numHiddenLayers else {
            throw TinyK3Error.configuration(
                "layer \(index) requested but the model has \(config.numHiddenLayers)")
        }
        func attention(_ projection: String) throws -> AffineWeights {
            try checkpoint.affine(
                QwenTensors.attention(index, projection),
                groupSize: config.quantGroupSize, bits: config.quantBits)
        }
        let weights = QwenLayerWeights(
            inputNorm: try checkpoint.float32(QwenTensors.inputNorm(index)),
            postAttentionNorm: try checkpoint.float32(QwenTensors.postAttentionNorm(index)),
            queryNorm: try checkpoint.float32(QwenTensors.queryNorm(index)),
            keyNorm: try checkpoint.float32(QwenTensors.keyNorm(index)),
            queryProj: try attention("q_proj"),
            keyProj: try attention("k_proj"),
            valueProj: try attention("v_proj"),
            outputProj: try attention("o_proj"),
            router: try checkpoint.affine(
                QwenTensors.router(index),
                groupSize: config.routerGroupSize, bits: config.routerBits))
        layers[index] = weights
        return weights
    }

    /// Total resident bytes, so the milestone can state a number rather than
    /// imply one (spec §5.3).
    public func residentBytes() throws -> Int {
        var total = 0
        for index in 0..<config.numHiddenLayers {
            total += try layerWeights(index).residentBytes
        }
        let head = try lmHead()
        total += head.packed.size * 4 + head.scales.size * 2 + head.biases.size * 2
        let embedding = try checkpoint.affine(
            QwenTensors.embedding, groupSize: config.quantGroupSize, bits: config.quantBits)
        total += embedding.packed.size * 4 + embedding.scales.size * 2 + embedding.biases.size * 2
        total += finalNormWeight.size * 4
        return total
    }
}
