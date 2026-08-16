import Foundation
import StorageCore

@testable import ModelAdapters

/// A whole K3 artifact — every layer, both attention archetypes, the dense
/// prefix, the routed experts and the three global tables — at a size that fits
/// in a temporary directory.
///
/// ``SyntheticLayerStreams`` already builds a bundle the *engine* can stream;
/// this builds one the **model** can decode from, which is a different and
/// larger claim: ``ArtifactWeightSource`` materialises a complete
/// ``TinyK3LayerWeights`` per layer, so every tensor `layerWeights` names has to
/// be present, at a shape the operators accept. Without it the artifact path is
/// only reachable with the 1.56 TB drive attached, and a read-ahead that changes
/// when bytes arrive would have nowhere to prove it does not change what they
/// are.
///
/// ## The shapes are the checkpoint's, scaled
///
/// Every relationship below is transcribed from a real layer manifest
/// (`docs/experiments/data/2026-08-08-m8-partial-k3/layer01-manifest.json` for
/// KDA, `layer47` for MLA) and then divided down: `f_b_proj` is
/// `[heads·headDim, fanRank]` because it is `[12288, 128]` there, `A_log` is
/// `[1, headDim]` with the entries past `heads` exactly zero because the
/// flagship stores it padded that way (M9B), `q_conv1d` is F32 `[channels, W]`,
/// and the row bands are 16 KiB-aligned with padding between them. A fixture
/// that got these wrong would test a layout the artifact does not have.
enum SyntheticArtifact {

    struct Shape {
        var layers = 4
        var hiddenSize = 128
        var vocabSize = 256
        /// The dense layer's MLP width.
        var intermediateSize = 256
        var attnResBlockSize = 2
        /// Layers below this are dense. One, as in the flagship.
        var firstKDenseReplace = 1

        // KDA. `inner = heads · headDim`.
        var kdaHeads = 4
        var kdaHeadDim = 16
        var kdaKernel = 4
        /// `f_a_proj`'s rank — 128 against 12,288 channels in the flagship.
        var kdaFanRank = 16

        // MLA.
        var mlaHeads = 4
        var qLoraRank = 32
        var kvLoraRank = 16
        var qkNopeHeadDim = 8
        var qkRopeHeadDim = 4
        var vHeadDim = 8

        // MoE. The container geometry is what fixes these: a tile needs
        // `rows · cols` divisible by 524,288 (one scale per 32 elements, 16 KiB
        // aligned), so one band of `w1` is 1024 × 512 and one band of `w2` is
        // 512 × 1024. Smaller would not be a legal container.
        var experts = 12
        var expertsPerToken = 2
        var latentSize = 512
        var moeIntermediateSize = 1024
        var sharedExperts = 1

        /// 1-indexed, exactly as `linear_attn_config` writes them.
        var mlaLayers: [Int] { [4] }
        var kdaLayerNumbers: [Int] {
            (1...layers).filter { !mlaLayers.contains($0) }
        }

        var kdaInner: Int { kdaHeads * kdaHeadDim }
        var mlaQueryHeadDim: Int { qkNopeHeadDim + qkRopeHeadDim }
        var sharedIntermediate: Int { moeIntermediateSize * sharedExperts }

        func isDense(layer index: Int) -> Bool { index < firstKDenseReplace }
        func isLinearAttention(layer index: Int) -> Bool {
            !mlaLayers.contains(index + 1)
        }
    }

    struct Bundle {
        let root: URL
        /// What ``ArtifactWeightSource`` opens: flat `layerNN-manifest.json`.
        let translated: URL
        let globals: URL
        let config: TinyK3Config
        let shape: Shape
    }

    /// Write the whole artifact under `root` and return it, parsed.
    @discardableResult
    static func write(root: URL, shape: Shape = Shape(), seed: UInt64 = 0x4B33_0001) throws
        -> Bundle
    {
        let translated = root.appendingPathComponent("translated")
        let globals = root.appendingPathComponent("global")
        try FileManager.default.createDirectory(at: translated, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: globals, withIntermediateDirectories: true)

        var generator = SplitMix64(seed: seed)
        for layer in 0..<shape.layers {
            try writeLayer(
                directory: translated, layer: layer, shape: shape, generator: &generator,
                seed: seed)
        }
        try writeGlobals(directory: globals, shape: shape, generator: &generator)

        let configURL = root.appendingPathComponent("config.json")
        try JSONSerialization
            .data(withJSONObject: configJSON(shape), options: [.prettyPrinted, .sortedKeys])
            .write(to: configURL)
        let config = try TinyK3Config(json: try Data(contentsOf: configURL))

        return Bundle(
            root: root, translated: translated, globals: globals, config: config, shape: shape)
    }

    /// Convert the complete synthetic model to the repository's published K3
    /// shape: one directory per layer, a reduced `manifest.json`, containers
    /// beside it, and the dense global tables at the root. This is used by the
    /// product-runner tests which must exercise translation through rooted
    /// verification authority rather than the older flat fixture seam.
    @discardableResult
    static func writePublished(from bundle: Bundle, at root: URL) throws -> URL {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let global = root.appendingPathComponent("global", isDirectory: true)
        try FileManager.default.copyItem(at: bundle.globals, to: global)
        try FileManager.default.copyItem(
            at: bundle.root.appendingPathComponent("config.json"),
            to: root.appendingPathComponent("config.json"))

        for layer in 0..<bundle.shape.layers {
            let stem = String(format: "layer%02d", layer)
            let flatManifest = bundle.translated.appendingPathComponent("\(stem)-manifest.json")
            guard
                let object = try JSONSerialization.jsonObject(
                    with: Data(contentsOf: flatManifest)) as? [String: Any],
                let deterministic = object["deterministic"] as? [String: Any],
                let deterministicName = deterministic["file"] as? String,
                let containers = object["containers"] as? [[String: Any]]
            else {
                throw TinyK3Error.configuration("malformed synthetic manifest \(flatManifest.path)")
            }
            let layerDirectory = root.appendingPathComponent(stem, isDirectory: true)
            try FileManager.default.createDirectory(
                at: layerDirectory, withIntermediateDirectories: true)
            try FileManager.default.copyItem(
                at: bundle.translated.appendingPathComponent(deterministicName),
                to: layerDirectory.appendingPathComponent(deterministicName))
            for container in containers {
                guard let name = container["file"] as? String else {
                    throw TinyK3Error.configuration(
                        "synthetic layer \(layer) has a container without a file")
                }
                try FileManager.default.copyItem(
                    at: bundle.translated.appendingPathComponent(name),
                    to: layerDirectory.appendingPathComponent(name))
            }
            let published: [String: Any] = [
                "format": "quantized_tile_container",
                "format_version": 1,
                "source_repo": "moonshotai/Kimi-K3",
                "layer": layer,
                "dense": true,
                "nonexpert": deterministic,
            ]
            try JSONSerialization.data(withJSONObject: published, options: [.sortedKeys])
                .write(to: layerDirectory.appendingPathComponent("manifest.json"))
        }
        let index: [String: Any] = [
            "source_repo": "moonshotai/Kimi-K3",
            "source_revision": String(repeating: "a", count: 40),
            "files": bundle.shape.layers * 4 + 3,
            "bytes": 1,
        ]
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: root.appendingPathComponent("index.json"))
        return root
    }

    // MARK: - One layer

    private static func writeLayer(
        directory: URL, layer: Int, shape: Shape, generator: inout SplitMix64, seed: UInt64
    ) throws {
        let stem = String(format: "layer%02d", layer)
        var containers: [[String: Any]] = []
        var slotToExpert: [Int] = []

        if !shape.isDense(layer: layer) {
            // Lexicographic by index string, like the checkpoint: slot 2 is
            // expert 10, so a loader that ignored the permutation would read
            // the wrong expert.
            slotToExpert = (0..<shape.experts).sorted { String($0) < String($1) }
            for (projection, rows, cols) in [
                ("w1", shape.moeIntermediateSize, shape.latentSize),
                ("w2", shape.latentSize, shape.moeIntermediateSize),
                ("w3", shape.moeIntermediateSize, shape.latentSize),
            ] {
                let name = "\(stem)-\(projection).mxfp4tile"
                let layout = try MXFP4TileContainer.write(
                    path: directory.appendingPathComponent(name).path,
                    geometry: TileGeometry(rows: rows, cols: cols),
                    tileCount: shape.experts,
                    seed: seed &+ UInt64(layer) << 8 &+ UInt64(projection.utf8.last ?? 0))
                containers.append([
                    "projection": projection,
                    "file": name,
                    "tiles_per_expert": 1,
                    "tile_count": layout.tileCount,
                    "tile_stride": layout.tileStride,
                    "out_features_per_expert": rows,
                    "in_features": cols,
                    "experts_per_tile": 1,
                ])
            }
        }

        let deterministic = try writeDeterministic(
            directory: directory, stem: stem, layer: layer, shape: shape, generator: &generator)

        let manifest: [String: Any] = [
            "schema_version": 1,
            "bundle_kind": "k3_flagship_layer_streams",
            "layer": layer,
            "experts": shape.isDense(layer: layer) ? 0 : shape.experts,
            "slot_to_expert": slotToExpert,
            "containers": containers,
            "deterministic": deterministic,
            "source": ["kind": "synthetic", "repo_id": "none", "revision": "none"],
        ]
        try JSONSerialization
            .data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
            .write(to: directory.appendingPathComponent("\(stem)-manifest.json"))
    }

    /// One tensor of one layer's deterministic blob.
    private struct Tensor {
        let name: String
        let role: String
        let rows: Int
        let cols: Int
        let dtype: String
        /// Trailing elements forced to exactly zero — `A_log`'s padding, which
        /// `K3KDA.resolveALog` refuses to slice unless it is provably padding.
        var zeroTail = 0
    }

    /// Exactly the tensors ``ArtifactWeightSource/layerWeights(_:)`` reads for
    /// this layer, and nothing else. The blob of a real layer is the same set —
    /// 28 tensors for a KDA MoE layer, checked against `layer01-manifest.json`.
    private static func tensors(layer index: Int, shape: Shape) -> [Tensor] {
        let prefix = "language_model.model.layers.\(index)"
        let attention = "\(prefix).self_attn"
        let inner = shape.kdaInner
        var list: [Tensor] = []

        if shape.isLinearAttention(layer: index) {
            list += [
                Tensor(
                    name: "\(attention).q_proj.weight", role: "attention", rows: inner,
                    cols: shape.hiddenSize, dtype: "BF16"),
                Tensor(
                    name: "\(attention).k_proj.weight", role: "attention", rows: inner,
                    cols: shape.hiddenSize, dtype: "BF16"),
                Tensor(
                    name: "\(attention).v_proj.weight", role: "attention", rows: inner,
                    cols: shape.hiddenSize, dtype: "BF16"),
                Tensor(
                    name: "\(attention).q_conv1d.weight", role: "attention", rows: inner,
                    cols: shape.kdaKernel, dtype: "F32"),
                Tensor(
                    name: "\(attention).k_conv1d.weight", role: "attention", rows: inner,
                    cols: shape.kdaKernel, dtype: "F32"),
                Tensor(
                    name: "\(attention).v_conv1d.weight", role: "attention", rows: inner,
                    cols: shape.kdaKernel, dtype: "F32"),
                Tensor(
                    name: "\(attention).f_a_proj.weight", role: "attention",
                    rows: shape.kdaFanRank, cols: shape.hiddenSize, dtype: "BF16"),
                Tensor(
                    name: "\(attention).f_b_proj.weight", role: "attention", rows: inner,
                    cols: shape.kdaFanRank, dtype: "BF16"),
                Tensor(
                    name: "\(attention).b_proj.weight", role: "attention", rows: shape.kdaHeads,
                    cols: shape.hiddenSize, dtype: "BF16"),
                Tensor(
                    name: "\(attention).g_proj.weight", role: "attention", rows: inner,
                    cols: shape.hiddenSize, dtype: "BF16"),
                // Stored `[1, head_dim]` with the entries past `heads` zero,
                // exactly as the flagship stores it.
                Tensor(
                    name: "\(attention).A_log", role: "attention", rows: 1,
                    cols: shape.kdaHeadDim, dtype: "F32",
                    zeroTail: shape.kdaHeadDim - shape.kdaHeads),
                Tensor(
                    name: "\(attention).dt_bias", role: "attention", rows: 1, cols: inner,
                    dtype: "F32"),
                Tensor(
                    name: "\(attention).o_norm.weight", role: "attention", rows: 1,
                    cols: shape.kdaHeadDim, dtype: "F32"),
                Tensor(
                    name: "\(attention).o_proj.weight", role: "attention",
                    rows: shape.hiddenSize, cols: inner, dtype: "BF16"),
            ]
        } else {
            let queryWidth = shape.mlaHeads * shape.mlaQueryHeadDim
            let kvWidth = shape.mlaHeads * (shape.qkNopeHeadDim + shape.vHeadDim)
            let gateWidth = shape.mlaHeads * shape.vHeadDim
            list += [
                Tensor(
                    name: "\(attention).q_a_proj.weight", role: "attention",
                    rows: shape.qLoraRank, cols: shape.hiddenSize, dtype: "BF16"),
                Tensor(
                    name: "\(attention).q_a_layernorm.weight", role: "attention", rows: 1,
                    cols: shape.qLoraRank, dtype: "BF16"),
                Tensor(
                    name: "\(attention).q_b_proj.weight", role: "attention", rows: queryWidth,
                    cols: shape.qLoraRank, dtype: "BF16"),
                Tensor(
                    name: "\(attention).kv_a_proj_with_mqa.weight", role: "attention",
                    rows: shape.kvLoraRank + shape.qkRopeHeadDim, cols: shape.hiddenSize,
                    dtype: "BF16"),
                Tensor(
                    name: "\(attention).kv_a_layernorm.weight", role: "attention", rows: 1,
                    cols: shape.kvLoraRank, dtype: "BF16"),
                Tensor(
                    name: "\(attention).kv_b_proj.weight", role: "attention", rows: kvWidth,
                    cols: shape.kvLoraRank, dtype: "BF16"),
                Tensor(
                    name: "\(attention).g_proj.weight", role: "attention", rows: gateWidth,
                    cols: shape.hiddenSize, dtype: "BF16"),
                Tensor(
                    name: "\(attention).o_proj.weight", role: "attention",
                    rows: shape.hiddenSize, cols: gateWidth, dtype: "BF16"),
            ]
        }

        if shape.isDense(layer: index) {
            list += [
                Tensor(
                    name: "\(prefix).mlp.gate_proj.weight", role: "dense_mlp",
                    rows: shape.intermediateSize, cols: shape.hiddenSize, dtype: "BF16"),
                Tensor(
                    name: "\(prefix).mlp.up_proj.weight", role: "dense_mlp",
                    rows: shape.intermediateSize, cols: shape.hiddenSize, dtype: "BF16"),
                Tensor(
                    name: "\(prefix).mlp.down_proj.weight", role: "dense_mlp",
                    rows: shape.hiddenSize, cols: shape.intermediateSize, dtype: "BF16"),
            ]
        } else {
            let block = "\(prefix).block_sparse_moe"
            list += [
                Tensor(
                    name: "\(block).gate.weight", role: "router", rows: shape.experts,
                    cols: shape.hiddenSize, dtype: "BF16"),
                Tensor(
                    name: "\(block).gate.e_score_correction_bias", role: "router", rows: 1,
                    cols: shape.experts, dtype: "F32"),
                Tensor(
                    name: "\(block).routed_expert_down_proj.weight", role: "latent",
                    rows: shape.latentSize, cols: shape.hiddenSize, dtype: "BF16"),
                Tensor(
                    name: "\(block).routed_expert_up_proj.weight", role: "latent",
                    rows: shape.hiddenSize, cols: shape.latentSize, dtype: "BF16"),
                Tensor(
                    name: "\(block).routed_expert_norm.weight", role: "latent_norm", rows: 1,
                    cols: shape.latentSize, dtype: "BF16"),
                Tensor(
                    name: "\(block).shared_experts.gate_proj.weight", role: "shared_expert",
                    rows: shape.sharedIntermediate, cols: shape.hiddenSize, dtype: "BF16"),
                Tensor(
                    name: "\(block).shared_experts.up_proj.weight", role: "shared_expert",
                    rows: shape.sharedIntermediate, cols: shape.hiddenSize, dtype: "BF16"),
                Tensor(
                    name: "\(block).shared_experts.down_proj.weight", role: "shared_expert",
                    rows: shape.hiddenSize, cols: shape.sharedIntermediate, dtype: "BF16"),
            ]
        }

        list += [
            Tensor(
                name: "\(prefix).self_attention_res_norm.weight", role: "norm", rows: 1,
                cols: shape.hiddenSize, dtype: "BF16"),
            Tensor(
                name: "\(prefix).self_attention_res_proj.weight", role: "norm", rows: 1,
                cols: shape.hiddenSize, dtype: "BF16"),
            Tensor(
                name: "\(prefix).mlp_res_norm.weight", role: "norm", rows: 1,
                cols: shape.hiddenSize, dtype: "BF16"),
            Tensor(
                name: "\(prefix).mlp_res_proj.weight", role: "norm", rows: 1,
                cols: shape.hiddenSize, dtype: "BF16"),
            Tensor(
                name: "\(prefix).input_layernorm.weight", role: "norm", rows: 1,
                cols: shape.hiddenSize, dtype: "BF16"),
            Tensor(
                name: "\(prefix).post_attention_layernorm.weight", role: "norm", rows: 1,
                cols: shape.hiddenSize, dtype: "BF16"),
        ]
        return list
    }

    /// The blob, row-banded onto 16 KiB boundaries with padding between bands —
    /// the layout the fetcher writes and the reader reassembles.
    private static func writeDeterministic(
        directory: URL, stem: String, layer: Int, shape: Shape, generator: inout SplitMix64
    ) throws -> [String: Any] {
        let alignment = MXFP4TileContainer.regionAlignment
        var units: [[String: Any]] = []
        var bytesByRole: [String: Int] = [:]
        var payload = Data()

        for tensor in tensors(layer: layer, shape: shape) {
            let itemSize = tensor.dtype == "F32" ? 4 : 2
            // Two bands where there is room for two, so the staged path and the
            // serial path both have to reassemble a tensor from pieces.
            let bandCount = tensor.rows >= 8 && tensor.rows % 2 == 0 ? 2 : 1
            let bandRows = tensor.rows / bandCount
            let total = tensor.rows * tensor.cols
            var produced = 0

            for band in 0..<bandCount {
                let elements = bandRows * tensor.cols
                var bytes = Data(capacity: elements * itemSize)
                for _ in 0..<elements {
                    let zero = produced >= total - tensor.zeroTail
                    // Values in [-1, 1). Building BF16 from the top half of a
                    // float32 keeps every value finite: the NaN encoding is
                    // unreachable from this range.
                    let value = zero ? 0 : Float(generator.next() >> 40) / Float(1 << 23) - 1
                    let pattern = value.bitPattern
                    if tensor.dtype == "F32" {
                        bytes.append(contentsOf: [
                            UInt8(pattern & 0xFF), UInt8((pattern >> 8) & 0xFF),
                            UInt8((pattern >> 16) & 0xFF), UInt8((pattern >> 24) & 0xFF),
                        ])
                    } else {
                        bytes.append(contentsOf: [
                            UInt8((pattern >> 16) & 0xFF), UInt8((pattern >> 24) & 0xFF),
                        ])
                    }
                    produced += 1
                }
                let payloadLength = elements * itemSize
                let length = (payloadLength + alignment - 1) / alignment * alignment
                bytes.append(Data(repeating: 0, count: length - payloadLength))

                units.append([
                    "index": units.count,
                    "tensor": tensor.name,
                    "role": tensor.role,
                    "dtype": tensor.dtype,
                    "row_start": band * bandRows,
                    "rows": bandRows,
                    "cols": tensor.cols,
                    "offset": payload.count,
                    "length": length,
                    "payload_length": payloadLength,
                    "whole_tensor": bandCount == 1,
                ])
                bytesByRole[tensor.role, default: 0] += length
                payload.append(bytes)
            }
        }

        let name = "\(stem)-deterministic.bin"
        try payload.write(to: directory.appendingPathComponent(name))
        return [
            "file": name,
            "bytes": payload.count,
            "unit_count": units.count,
            "max_unit_bytes": units.map { $0["length"] as! Int }.max() ?? 0,
            "bytes_by_role": bytesByRole,
            "units": units,
        ]
    }

    // MARK: - Globals

    /// `embed_tokens`, `lm_head` and `final.bin`, in the layouts
    /// ``ArtifactWeightSource`` recovers from geometry and then checks: two
    /// dense `[vocab, hidden]` BF16 tables with no padding at all, and three
    /// 16 KiB units holding one `[hidden]` BF16 vector each.
    private static func writeGlobals(
        directory: URL, shape: Shape, generator: inout SplitMix64
    ) throws {
        func table(_ rows: Int) -> Data {
            var data = Data(capacity: rows * shape.hiddenSize * 2)
            for _ in 0..<(rows * shape.hiddenSize) {
                let pattern = (Float(generator.next() >> 40) / Float(1 << 23) - 1).bitPattern
                data.append(contentsOf: [
                    UInt8((pattern >> 16) & 0xFF), UInt8((pattern >> 24) & 0xFF),
                ])
            }
            return data
        }
        try table(shape.vocabSize).write(to: directory.appendingPathComponent("embed_tokens.bin"))
        try table(shape.vocabSize).write(to: directory.appendingPathComponent("lm_head.bin"))

        let unit = 16384
        var finals = Data()
        for _ in 0..<3 {
            var vector = table(1)
            vector.append(Data(repeating: 0, count: unit - vector.count))
            finals.append(vector)
        }
        try finals.write(to: directory.appendingPathComponent("final.bin"))
    }

    // MARK: - Config

    private static func configJSON(_ shape: Shape) -> [String: Any] {
        [
            "model_type": "kimi_k3",
            "text_config": [
                "hidden_size": shape.hiddenSize,
                "num_hidden_layers": shape.layers,
                "vocab_size": shape.vocabSize,
                "rms_norm_eps": 1e-5,
                "intermediate_size": shape.intermediateSize,
                "first_k_dense_replace": shape.firstKDenseReplace,
                "hidden_act": "situ",
                "activation_situ_beta": 4.0,
                "activation_situ_linear_beta": 25.0,
                "attn_res_block_size": shape.attnResBlockSize,
                "linear_attn_config": [
                    "kda_layers": shape.kdaLayerNumbers,
                    "full_attn_layers": shape.mlaLayers,
                    "num_heads": shape.kdaHeads,
                    "head_dim": shape.kdaHeadDim,
                    "short_conv_kernel_size": shape.kdaKernel,
                    "use_full_rank_gate": true,
                    "gate_lower_bound": -5.0,
                ] as [String: Any],
                "q_lora_rank": shape.qLoraRank,
                "kv_lora_rank": shape.kvLoraRank,
                "num_attention_heads": shape.mlaHeads,
                "qk_nope_head_dim": shape.qkNopeHeadDim,
                "qk_rope_head_dim": shape.qkRopeHeadDim,
                "v_head_dim": shape.vHeadDim,
                "mla_use_nope": true,
                "mla_use_output_gate": true,
                "num_experts": shape.experts,
                "num_experts_per_token": shape.expertsPerToken,
                "num_shared_experts": shape.sharedExperts,
                "routed_expert_hidden_size": shape.latentSize,
                "moe_intermediate_size": shape.moeIntermediateSize,
                "moe_renormalize": true,
                "moe_router_activation_func": "sigmoid",
                "routed_scaling_factor": 1.0,
                "latent_moe_use_norm": true,
                "num_expert_group": 1,
                "topk_group": 1,
            ] as [String: Any],
        ]
    }
}
