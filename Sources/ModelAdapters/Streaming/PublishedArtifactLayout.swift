import Foundation
import StorageCore

/// Translates the *published* K3 artifact into the layout ``FlagshipLayerStreams``
/// reads.
///
/// ## Why a translation exists at all
///
/// The artifact on the Hub (`nanguoyu/Kimi-K3-minirun`) is the output of
/// `Tools/k3_flagship/build_full_artifact.py` run through
/// `cluster/publish_to_hub.py`, and the publisher changed the shape in two ways
/// that make the runtime loader unable to open it:
///
/// 1. **Layout.** The loader globs `layer*-manifest.json` from one flat
///    directory. The artifact has `layerNN/manifest.json` in 93 subdirectories,
///    so the glob matches nothing.
/// 2. **Schema.** The publisher's `sanitise_layer_manifest` keeps only the keys
///    it recognises, and it does not recognise `containers`, `experts` or
///    `slot_to_expert`. Worse, it emits every layer's *dense* manifest variant
///    last, so all 93 published manifests carry `"dense": true` and the
///    non-expert half only — including the 92 layers that ship three
///    `.mxfp4tile` containers next to the manifest that denies they exist.
///
/// Nothing was lost that cannot be recovered locally, which is why this is a
/// translation and not a re-download:
///
/// - `nonexpert.units[]` already carries exactly ``FlagshipLayerStreams/DeterministicUnit``'s
///   coding keys, so the deterministic half passes through verbatim.
/// - The container descriptors are recoverable because the three `.mxfp4tile`
///   headers are **jointly** self-describing (see ``ExpertGeometry``).
/// - `slot_to_expert` is the checkpoint's expert permutation, reconstructed by
///   ``slotToExpert(experts:)`` and verified against the checkpoint separately.
///
/// ## Build step, not adapter
///
/// This materialises a directory rather than constructing ``FlagshipLayerStreams``
/// in memory, and that is deliberate. An in-memory adapter would need a second
/// initialiser on ``FlagshipLayerStreams`` — a new, less-checked path into the
/// type every later reader has to know about. Materialising instead means the
/// loader keeps exactly one entry point, its validation runs on the translated
/// manifests unchanged (it re-opens every container header and re-derives the
/// geometry, so a translation bug is caught by the *existing* checks rather than
/// by new ones), and the translated directory is an inspectable artifact that a
/// human can diff against `docs/experiments/data/2026-08-08-m8-partial-k3/`.
/// ``FlagshipLayerStreams`` is not modified by this file in any way.
///
/// The cost is ~16 MB of rewritten JSON and 368 symlinks, both of which live in
/// scratch on the internal disk. The artifact volume is never written.
public enum PublishedArtifactLayout {

    /// How a translated manifest names the blob it describes.
    ///
    /// The symlink farm is the original scheme and stays the default: it keeps
    /// every path in a translated bundle relative, so the directory can be moved
    /// or diffed as a unit.
    ///
    /// It does not survive iOS. An external volume mounts under
    /// `/private/var/mobile/Library/LiveFiles/com.apple.filesystems.userfsd/<UUID>/`,
    /// and that **UUID is per-mount** — a farm of absolute symlinks written on
    /// internal storage names a path that is correct only until the drive is
    /// unplugged, and the next run would resolve 369 links to nothing. Naming
    /// the blob in the manifest instead moves that string into the file the
    /// translation rewrites on every run anyway, so it is re-derived from the
    /// live security-scoped URL each time and cannot go stale.
    ///
    /// Both modes produce manifests the *same* loader reads with the *same*
    /// validation; only the spelling of `file` differs (see
    /// ``FlagshipLayerStreams``'s path resolution).
    public enum BlobReference: Sendable {
        /// A symlink beside the manifest, named `layerNN-<role>`; `file` is
        /// relative.
        case symlink
        /// No links at all; `file` is the artifact's own absolute path.
        case absolutePath
        /// `file` is the canonical repository-relative identity consumed by a
        /// rooted ``ModelFileAccess``. No payload path or symlink is trusted.
        case artifactReference
    }

    /// What the published artifact's per-layer `manifest.json` still carries.
    ///
    /// Only the fields the translation needs are decoded. `dense` is read but
    /// deliberately **not** trusted — see ``LayerSource/hasContainers``.
    struct PublishedLayerManifest: Decodable {
        struct NonExpert: Decodable {
            let file: String
            let bytes: Int
            let units: [FlagshipLayerStreams.DeterministicUnit]
        }
        let layer: Int
        let nonexpert: NonExpert
    }

    /// One layer as found on disk.
    public struct LayerSource {
        public let layer: Int
        public let directory: URL
        public let manifestPath: URL
        public let deterministicFile: String
        /// `projection -> file name`, empty for the dense layer.
        public let containerFiles: [ExpertProjection: String]

        /// Whether this layer routes. Taken from the **presence of container
        /// files**, never from the manifest's `dense` flag: that flag is `true`
        /// on all 93 published layers including the 92 that are demonstrably
        /// MoE, so believing it would make the whole model dense.
        public var hasContainers: Bool { !containerFiles.isEmpty }
    }

    /// Geometry recovered from the three container headers together.
    ///
    /// A single header does not say how many experts it holds: it reports
    /// `rows`, `cols` and `tileCount`, and `experts = tileCount / (outFeatures /
    /// rows)` needs `outFeatures`, which is not in it. The triple supplies it,
    /// because a K3 expert is `w2 . act(w1 x) * (w3 x)` and therefore
    ///
    /// ```text
    /// outFeatures(w1) == outFeatures(w3) == cols(w2)   // the intermediate width
    /// outFeatures(w2) == cols(w1) == cols(w3)          // the latent width
    /// ```
    ///
    /// Both readings then have to produce the *same* expert count, which is the
    /// check that makes this a derivation rather than a guess: at flagship shape
    /// w1 gives `2688 / (3072/1024) = 896` and w2 gives `6272 / (3584/512) =
    /// 896`. A transposed, mis-tiled or truncated container disagrees.
    public struct ExpertGeometry {
        public let experts: Int
        /// Latent width — the expert block's input and output width.
        public let latentSize: Int
        /// Per-expert intermediate width.
        public let intermediateSize: Int
        public let tilesPerExpert: [ExpertProjection: Int]
        public let outFeaturesPerExpert: [ExpertProjection: Int]
        public let layouts: [ExpertProjection: MXFP4TileContainer.Layout]
    }

    // MARK: - The expert permutation

    /// The physical slot order of experts in a container.
    ///
    /// The producing code (`Tools/k3_flagship/fetch_layer_streams.py:plan_layer`)
    /// sorts experts by the byte offset of their first tensor in the shard, and
    /// records the result. The published artifact dropped that record, so this
    /// reconstructs it from the property the checkpoint's writer happened to
    /// have: it emitted experts in lexicographic order of the **index string**,
    /// so slot 2 is expert 10.
    ///
    /// This is a reconstruction of a measurement, so it is verified rather than
    /// assumed. `docs/experiments/data/2026-08-10-m9c-first-token/verify_slot_order.py`
    /// re-derives the authoritative byte-offset sort from every MoE layer's
    /// shard header and compares: 92 of 92 agree, where M8 and M9B had checked
    /// the 2 layers they happened to record. A run using the wrong permutation would
    /// read correctly-shaped bytes from the wrong expert and produce no error at
    /// all, which is exactly why it is checked externally rather than trusted.
    public static func slotToExpert(experts: Int) -> [Int] {
        (0..<experts).sorted { String($0) < String($1) }
    }

    // MARK: - Discovery

    /// Find every `layerNN/` in the published artifact, lowest layer first.
    public static func layers(inArtifact artifact: String) throws -> [LayerSource] {
        let root = URL(fileURLWithPath: artifact)
        let entries = try FileManager.default.contentsOfDirectory(atPath: artifact)
            .filter { $0.hasPrefix("layer") && $0.count == 7 }
            .sorted()
        guard !entries.isEmpty else {
            throw TinyK3Error.configuration(
                "no layerNN/ directory in \(artifact); this does not look like the published "
                    + "K3 artifact")
        }
        return try entries.map { name in
            let directory = root.appendingPathComponent(name)
            let manifest = directory.appendingPathComponent("manifest.json")
            guard FileManager.default.fileExists(atPath: manifest.path) else {
                throw TinyK3Error.configuration("\(name) has no manifest.json")
            }
            guard let layer = Int(name.dropFirst(5)) else {
                throw TinyK3Error.configuration("cannot read a layer number out of '\(name)'")
            }
            let files = try FileManager.default.contentsOfDirectory(atPath: directory.path)
            var containers: [ExpertProjection: String] = [:]
            for projection in ExpertProjection.allCases {
                let suffix = "-\(projection.rawValue).mxfp4tile"
                if let file = files.first(where: { $0.hasSuffix(suffix) }) {
                    containers[projection] = file
                }
            }
            guard containers.isEmpty || containers.count == ExpertProjection.allCases.count else {
                throw TinyK3Error.configuration(
                    "\(name) has \(containers.count) of \(ExpertProjection.allCases.count) "
                        + "expert containers; a partially-downloaded layer is not usable")
            }
            guard let deterministic = files.first(where: { $0.hasSuffix("-deterministic.bin") })
            else {
                throw TinyK3Error.configuration("\(name) has no deterministic blob")
            }
            return LayerSource(
                layer: layer, directory: directory, manifestPath: manifest,
                deterministicFile: deterministic, containerFiles: containers)
        }
        .sorted { $0.layer < $1.layer }
    }

    /// Recover one layer's expert geometry from its container headers.
    public static func geometry(
        of source: LayerSource, fileAccess: ModelFileAccess? = nil
    ) throws -> ExpertGeometry? {
        guard source.hasContainers else { return nil }
        var layouts: [ExpertProjection: MXFP4TileContainer.Layout] = [:]
        for (projection, file) in source.containerFiles {
            let reference = fileAccess == nil
                ? source.directory.appendingPathComponent(file).path
                : source.directory.lastPathComponent + "/" + file
            if let fileAccess {
                layouts[projection] = try fileAccess.withDescriptor(reference) {
                    try MXFP4TileContainer.open(fileDescriptor: $0, path: reference)
                }
            } else {
                layouts[projection] = try MXFP4TileContainer.open(path: reference)
            }
        }
        guard let gate = layouts[.w1], let down = layouts[.w2], let up = layouts[.w3] else {
            throw TinyK3Error.configuration("layer \(source.layer): missing a projection")
        }
        // `w1` and `w3` are the same matrix shape by construction. If they are
        // not, the pair cannot both be gate/up of one expert block and the
        // joint derivation below has no meaning.
        guard gate.geometry.rows == up.geometry.rows,
            gate.geometry.cols == up.geometry.cols,
            gate.tileCount == up.tileCount
        else {
            throw TinyK3Error.configuration(
                "layer \(source.layer): w1 (\(gate.geometry.rows)x\(gate.geometry.cols), "
                    + "\(gate.tileCount) tiles) and w3 (\(up.geometry.rows)x\(up.geometry.cols), "
                    + "\(up.tileCount) tiles) are not the same shape")
        }
        let latent = gate.geometry.cols
        let intermediate = down.geometry.cols
        guard down.geometry.rows > 0, gate.geometry.rows > 0 else {
            throw TinyK3Error.configuration("layer \(source.layer): a container has zero rows")
        }

        var tilesPerExpert: [ExpertProjection: Int] = [:]
        var outFeatures: [ExpertProjection: Int] = [:]
        var expertCounts: [ExpertProjection: Int] = [:]
        for (projection, layout) in layouts {
            let out = projection == .w2 ? latent : intermediate
            guard out % layout.geometry.rows == 0 else {
                throw TinyK3Error.configuration(
                    "layer \(source.layer) \(projection.rawValue): \(out) output features is not a "
                        + "whole number of \(layout.geometry.rows)-row bands")
            }
            let bands = out / layout.geometry.rows
            guard bands > 0, layout.tileCount % bands == 0 else {
                throw TinyK3Error.configuration(
                    "layer \(source.layer) \(projection.rawValue): \(layout.tileCount) tiles is not "
                        + "a whole number of \(bands)-band experts")
            }
            tilesPerExpert[projection] = bands
            outFeatures[projection] = out
            expertCounts[projection] = layout.tileCount / bands
        }
        let counts = Set(expertCounts.values)
        guard counts.count == 1, let experts = counts.first, experts > 0 else {
            throw TinyK3Error.configuration(
                "layer \(source.layer): the containers disagree on the expert count "
                    + "(\(expertCounts.map { "\($0.key.rawValue)=\($0.value)" }.sorted().joined(separator: ", ")))")
        }
        return ExpertGeometry(
            experts: experts, latentSize: latent, intermediateSize: intermediate,
            tilesPerExpert: tilesPerExpert, outFeaturesPerExpert: outFeatures, layouts: layouts)
    }

    // MARK: - Translation

    public struct Report: Sendable, Equatable {
        public var layers = 0
        public var moeLayers = 0
        public var denseLayers = 0
        public var symlinks = 0
        public var experts = 0
        public var manifestBytes = 0
    }

    /// Write a flat `layerNN-manifest.json` set plus a symlink farm into
    /// `scratch`, so that `FlagshipLayerStreams.loadAll(directory: scratch)`
    /// opens the published artifact unchanged.
    ///
    /// - Parameters:
    ///   - config: when supplied, the recovered geometry is cross-checked
    ///     against it. This is the difference between "the containers agree with
    ///     each other" and "the containers agree with the model we are about to
    ///     run", and only the second one catches an artifact for a different
    ///     checkpoint.
    ///   - blobReference: how the manifests name their blobs. See
    ///     ``BlobReference``; `.absolutePath` is what an iOS run needs, because
    ///     the artifact's mount point changes between mounts.
    ///
    /// Idempotent: re-running replaces the manifests and relinks.
    @discardableResult
    public static func translate(
        artifact: String,
        into scratch: String,
        config: TinyK3Config? = nil,
        blobReference: BlobReference = .symlink,
        fileAccess: ModelFileAccess? = nil
    ) throws -> Report {
        let sources = try layers(inArtifact: artifact)
        if let config, sources.count != config.numHiddenLayers {
            throw TinyK3Error.configuration(
                "the artifact has \(sources.count) layers, the config declares "
                    + "\(config.numHiddenLayers)")
        }
        let scratchURL = URL(fileURLWithPath: scratch)
        try FileManager.default.createDirectory(
            at: scratchURL, withIntermediateDirectories: true)

        var report = Report()
        for source in sources {
            let published = try JSONDecoder().decode(
                PublishedLayerManifest.self,
                from: try Data(contentsOf: source.manifestPath))
            guard published.layer == source.layer else {
                throw TinyK3Error.configuration(
                    "layer\(source.layer)/manifest.json declares layer \(published.layer)")
            }

            let stem = String(format: "layer%02d", source.layer)
            var containers: [[String: Any]] = []
            var experts = 0

            if let geometry = try geometry(of: source, fileAccess: fileAccess) {
                experts = geometry.experts
                if let config {
                    guard geometry.experts == config.numExperts,
                        geometry.latentSize == config.routedExpertHiddenSize,
                        geometry.intermediateSize == config.moeIntermediateSize
                    else {
                        throw TinyK3Error.configuration(
                            """
                            layer \(source.layer): containers describe \(geometry.experts) experts \
                            of \(geometry.latentSize)->\(geometry.intermediateSize), the config \
                            declares \(config.numExperts) of \(config.routedExpertHiddenSize)->\
                            \(config.moeIntermediateSize)
                            """)
                    }
                }
                for projection in ExpertProjection.allCases {
                    guard let layout = geometry.layouts[projection],
                        let file = source.containerFiles[projection],
                        let bands = geometry.tilesPerExpert[projection],
                        let out = geometry.outFeaturesPerExpert[projection]
                    else { continue }
                    let origin = source.directory.appendingPathComponent(file)
                    let linked = "\(stem)-\(projection.rawValue).mxfp4tile"
                    let named: String
                    switch blobReference {
                    case .symlink:
                        try link(origin, to: scratchURL.appendingPathComponent(linked))
                        report.symlinks += 1
                        named = linked
                    case .absolutePath:
                        named = origin.path
                    case .artifactReference:
                        named = source.directory.lastPathComponent + "/" + file
                    }
                    containers.append([
                        "projection": projection.rawValue,
                        "file": named,
                        "tiles_per_expert": bands,
                        "tile_count": layout.tileCount,
                        "tile_stride": layout.tileStride,
                        "out_features_per_expert": out,
                        "in_features": layout.geometry.cols,
                        "experts_per_tile": 1,
                    ])
                }
                report.moeLayers += 1
            } else {
                // The dense layer routes nothing. `FlagshipLayerStreams` accepts
                // an empty container set and an empty permutation, so the dense
                // layer goes through the same loader as every other one rather
                // than becoming a second code path that only it exercises.
                report.denseLayers += 1
            }

            let blobOrigin = source.directory.appendingPathComponent(source.deterministicFile)
            let linkedBlob: String
            switch blobReference {
            case .symlink:
                let name = "\(stem)-deterministic.bin"
                try link(blobOrigin, to: scratchURL.appendingPathComponent(name))
                report.symlinks += 1
                linkedBlob = name
            case .absolutePath:
                linkedBlob = blobOrigin.path
            case .artifactReference:
                linkedBlob = source.directory.lastPathComponent + "/" + source.deterministicFile
            }

            var bytesByRole: [String: Int] = [:]
            for unit in published.nonexpert.units {
                bytesByRole[unit.role, default: 0] += unit.length
            }

            let manifest: [String: Any] = [
                "schema_version": 1,
                "bundle_kind": "k3_flagship_layer_streams",
                "translated_from": source.manifestPath.path,
                "layer": source.layer,
                "experts": experts,
                "slot_to_expert": slotToExpert(experts: experts),
                "containers": containers,
                "deterministic": [
                    "file": linkedBlob,
                    "bytes": published.nonexpert.bytes,
                    "units": published.nonexpert.units.map(encode),
                    "bytes_by_role": bytesByRole,
                ] as [String: Any],
            ]
            let data = try JSONSerialization.data(
                withJSONObject: manifest, options: [.sortedKeys])
            let out = scratchURL.appendingPathComponent("\(stem)-manifest.json")
            try data.write(to: out)
            report.manifestBytes += data.count
            report.layers += 1
            report.experts = max(report.experts, experts)
        }
        return report
    }

    /// Round-trip a unit back to the manifest's own key spelling.
    ///
    /// Written out longhand rather than re-encoding with `JSONEncoder`, because
    /// the decoder that reads it back is `Decodable` on the same type: a
    /// mismatch here would be caught, but only after the translation had
    /// produced a file that looks right.
    private static func encode(_ unit: FlagshipLayerStreams.DeterministicUnit) -> [String: Any] {
        [
            "index": unit.index,
            "tensor": unit.tensor,
            "role": unit.role,
            "dtype": unit.dtype,
            "row_start": unit.rowStart,
            "rows": unit.rows,
            "cols": unit.cols,
            "offset": unit.offset,
            "length": unit.length,
            "payload_length": unit.payloadLength,
        ]
    }

    /// Point `destination` at `source`, replacing whatever was there.
    ///
    /// A symlink rather than a copy: the target is 1.56 TB and the volume it
    /// lives on is read-only by instruction, so the only writable end is this
    /// one.
    private static func link(_ source: URL, to destination: URL) throws {
        let manager = FileManager.default
        if let existing = try? manager.destinationOfSymbolicLink(atPath: destination.path) {
            if existing == source.path { return }
            try manager.removeItem(at: destination)
        } else if manager.fileExists(atPath: destination.path) {
            try manager.removeItem(at: destination)
        }
        try manager.createSymbolicLink(at: destination, withDestinationURL: source)
    }
}
