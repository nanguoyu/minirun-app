import Darwin
import Foundation
import StorageCore

/// The two byte streams one real K3 MoE layer produces, as written by
/// `Tools/k3_flagship/fetch_layer_streams.py`.
///
/// ## Why there are two
///
/// A routed MoE layer reads two very different things per token. The **expert
/// stream** is routing-dependent: 16 of 896 experts, a different 16 next token.
/// The **deterministic stream** is everything else — attention, the shared
/// expert, the latent projections, the router itself — and it is read in full
/// every token no matter what the router decides. Prior milestones measured
/// only the first, which is the smaller half: spec §15.5 puts the ratio at
/// roughly 4:1 the other way. Sizing a runtime on the expert stream alone would
/// size it for a quarter of the problem.
///
/// ## Expert containers are expert-major, in *physical slot* order
///
/// The checkpoint stores experts in lexicographic order of the index string, so
/// expert 10 sits at physical slot 2. `slotForExpert` carries that permutation
/// rather than recomputing it, for the same reason the tiny adapter reads its
/// tile map from a manifest: it is data, and a future checkpoint that orders
/// experts differently must show up as a different manifest, not as a silently
/// wrong offset.
public struct FlagshipLayerStreams {

    /// One `(layer, projection)` container file.
    public struct ExpertContainer {
        public let projection: ExpertProjection
        public let path: String
        public let layout: MXFP4TileContainer.Layout
        /// Row bands one expert is split into: 3 for `w1`/`w3`, 7 for `w2`.
        public let tilesPerExpert: Int
        public let outFeaturesPerExpert: Int
        public let inFeatures: Int
        public let sourceID: UInt64
        let fileAccess: ModelFileAccess

        /// Rows in one band. Every band of a projection is the same height.
        public var rowsPerTile: Int { layout.geometry.rows }
        /// Bytes of one expert, whole: `tilesPerExpert` tiles back to back.
        public var bytesPerExpert: Int { tilesPerExpert * layout.tileStride }

        /// First tile of an expert, given its physical slot.
        public func firstTile(slot: Int) -> Int { slot * tilesPerExpert }

        /// One read per band (13 per expert across the three projections).
        public func tileRegion(slot: Int, band: Int) -> RegionDescriptor {
            layout.region(firstTile(slot: slot) + band, sourceID: sourceID)
        }

        /// One read spanning the whole expert (3 per expert across the three
        /// projections). Legal only because tiles are expert-major and the
        /// stride is uniform, so an expert's bands are physically adjacent.
        public func expertRegion(slot: Int) -> RegionDescriptor {
            RegionDescriptor(
                identity: RegionIdentity(
                    sourceID: sourceID, index: Self.contiguousNamespace | UInt64(slot)),
                offset: layout.tileOffset(firstTile(slot: slot)),
                length: bytesPerExpert)
        }

        /// Keeps whole-expert identities from colliding with tile identities,
        /// so a metrics record from one granularity can never be read as the
        /// other's.
        static let contiguousNamespace: UInt64 = 1 << 63
    }

    /// One 16 KiB-aligned read unit of the deterministic stream.
    public struct DeterministicUnit: Codable {
        public let index: Int
        public let tensor: String
        public let role: String
        public let dtype: String
        public let rowStart: Int
        public let rows: Int
        public let cols: Int
        public let offset: UInt64
        public let length: Int
        /// Bytes of real content; the rest of `length` is alignment padding.
        public let payloadLength: Int

        enum CodingKeys: String, CodingKey {
            case index, tensor, role, dtype, rows, cols, offset, length
            case rowStart = "row_start"
            case payloadLength = "payload_length"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            index = try container.decode(Int.self, forKey: .index)
            tensor = try container.decode(String.self, forKey: .tensor)
            role = try container.decode(String.self, forKey: .role)
            dtype = try container.decode(String.self, forKey: .dtype)
            rowStart = try container.decode(Int.self, forKey: .rowStart)
            rows = try container.decode(Int.self, forKey: .rows)
            cols = try container.decode(Int.self, forKey: .cols)
            do {
                offset = try container.decode(UInt64.self, forKey: .offset)
            } catch {
                throw DecodingError.dataCorruptedError(
                    forKey: .offset, in: container,
                    debugDescription: "offset must be nonnegative and fit UInt64")
            }
            length = try container.decode(Int.self, forKey: .length)
            payloadLength = try container.decode(Int.self, forKey: .payloadLength)
        }
    }

    public struct DeterministicStream {
        public let path: String
        public let bytes: Int
        public let units: [DeterministicUnit]
        public let sourceID: UInt64
        public let bytesByRole: [String: Int]
        let fileAccess: ModelFileAccess

        public func region(_ unit: DeterministicUnit) -> RegionDescriptor {
            RegionDescriptor(
                identity: RegionIdentity(sourceID: sourceID, index: UInt64(unit.index)),
                offset: unit.offset,
                length: unit.length)
        }

        public func units(role: String) -> [DeterministicUnit] {
            units.filter { $0.role == role }
        }

        public func units(tensorSuffix: String) -> [DeterministicUnit] {
            units.filter { $0.tensor.hasSuffix(tensorSuffix) }
        }
    }

    public let layer: Int
    public let experts: Int
    public let containers: [ExpertProjection: ExpertContainer]
    public let deterministic: DeterministicStream
    /// Expert id -> physical slot in the containers.
    public let slotForExpert: [Int: Int]
    public let manifestPath: String

    public var tilesPerExpert: Int {
        ExpertProjection.allCases.reduce(0) { $0 + (containers[$1]?.tilesPerExpert ?? 0) }
    }
    public var bytesPerExpert: Int {
        ExpertProjection.allCases.reduce(0) { $0 + (containers[$1]?.bytesPerExpert ?? 0) }
    }
    public var expertContainerBytes: Int {
        containers.values.reduce(0) { $0 + Int($1.layout.totalBytes) }
    }

    public func container(_ projection: ExpertProjection) throws -> ExpertContainer {
        guard let container = containers[projection] else {
            throw TinyK3Error.configuration(
                "layer \(layer) has no \(projection.rawValue) container")
        }
        return container
    }

    public func slot(forExpert expert: Int) throws -> Int {
        guard let slot = slotForExpert[expert] else {
            throw TinyK3Error.configuration("layer \(layer) has no expert \(expert)")
        }
        return slot
    }

    /// Largest single read either granularity can ask for. This is the pool's
    /// slot size, and therefore a budget line.
    public func maximumRegionBytes(granularity: FetchGranularity) -> Int {
        containers.values.reduce(0) { longest, container in
            switch granularity {
            case .perTile: return max(longest, container.layout.tileStride)
            case .perExpert: return max(longest, container.bytesPerExpert)
            }
        }
    }

    // MARK: - Loading

    private struct Manifest: Decodable {
        struct Container: Decodable {
            let projection: String
            let file: String
            let tilesPerExpert: Int
            let tileCount: Int
            let tileStride: Int
            let outFeaturesPerExpert: Int
            let inFeatures: Int
            let expertsPerTile: Int

            enum CodingKeys: String, CodingKey {
                case projection, file
                case tilesPerExpert = "tiles_per_expert"
                case tileCount = "tile_count"
                case tileStride = "tile_stride"
                case outFeaturesPerExpert = "out_features_per_expert"
                case inFeatures = "in_features"
                case expertsPerTile = "experts_per_tile"
            }
        }
        struct Deterministic: Decodable {
            let file: String
            let bytes: Int
            let units: [DeterministicUnit]
            let bytesByRole: [String: Int]

            enum CodingKeys: String, CodingKey {
                case file, bytes, units
                case bytesByRole = "bytes_by_role"
            }
        }
        let layer: Int
        let experts: Int
        let slotToExpert: [Int]
        let containers: [Container]
        let deterministic: Deterministic

        enum CodingKeys: String, CodingKey {
            case layer, experts, containers, deterministic
            case slotToExpert = "slot_to_expert"
        }
    }

    /// Load one layer from its manifest. Container headers are re-opened and
    /// re-derived rather than trusted, exactly as the tiny adapter does.
    public init(
        manifestPath: String, fileAccess rootedAccess: ModelFileAccess? = nil
    ) throws {
        let directory = URL(fileURLWithPath: manifestPath).deletingLastPathComponent()
        let fileAccess = rootedAccess ?? .filesystem
        let manifest = try JSONDecoder().decode(
            Manifest.self, from: try Data(contentsOf: URL(fileURLWithPath: manifestPath)))

        guard manifest.layer >= 0, UInt64(manifest.layer) <= UInt64.max >> 8 else {
            throw TinyK3Error.configuration(
                "manifest layer \(manifest.layer) cannot form a stable source identity")
        }
        guard manifest.experts >= 0 else {
            throw TinyK3Error.configuration(
                "layer \(manifest.layer) declares \(manifest.experts) experts")
        }

        var containers: [ExpertProjection: ExpertContainer] = [:]
        for record in manifest.containers {
            guard let projection = ExpertProjection(rawValue: record.projection) else {
                throw TinyK3Error.configuration("unknown projection '\(record.projection)'")
            }
            guard record.expertsPerTile == 1 else {
                throw TinyK3Error.configuration(
                    """
                    layer \(manifest.layer) \(record.projection) packs \
                    \(record.expertsPerTile) experts per tile. At flagship shape a tile holds \
                    a row band of ONE expert; a container that stacks experts belongs to the \
                    tiny adapter's loader, not this one.
                    """)
            }
            guard record.tilesPerExpert > 0, record.tileCount > 0,
                record.tileStride > 0, record.outFeaturesPerExpert > 0,
                record.inFeatures > 0
            else {
                throw TinyK3Error.configuration(
                    "layer \(manifest.layer) \(record.projection) carries a non-positive "
                        + "container count, stride, or shape")
            }
            guard containers[projection] == nil else {
                throw TinyK3Error.configuration(
                    "layer \(manifest.layer) repeats the \(record.projection) container")
            }
            let path = rootedAccess == nil
                ? Self.resolve(record.file, relativeTo: directory) : record.file
            let layout = try fileAccess.withDescriptor(path) {
                try MXFP4TileContainer.open(fileDescriptor: $0, path: path)
            }
            let manifestRows = layout.geometry.rows.multipliedReportingOverflow(
                by: record.tilesPerExpert)
            guard layout.tileCount == record.tileCount,
                layout.tileStride == record.tileStride,
                layout.geometry.cols == record.inFeatures,
                !manifestRows.overflow,
                manifestRows.partialValue == record.outFeaturesPerExpert
            else {
                throw TinyK3Error.configuration(
                    """
                    \(record.file): header (\(layout.tileCount) tiles of \
                    \(layout.geometry.rows)x\(layout.geometry.cols), stride \
                    \(layout.tileStride)) contradicts the manifest (\(record.tileCount) tiles, \
                    \(record.tilesPerExpert) bands of \(record.outFeaturesPerExpert) rows, \
                    stride \(record.tileStride))
                    """)
            }
            let expectedTiles = record.tilesPerExpert.multipliedReportingOverflow(
                by: manifest.experts)
            guard !expectedTiles.overflow, layout.tileCount == expectedTiles.partialValue else {
                throw TinyK3Error.configuration(
                    "\(record.file): \(layout.tileCount) tiles is not \(record.tilesPerExpert) "
                        + "x \(manifest.experts) experts")
            }
            containers[projection] = ExpertContainer(
                projection: projection,
                path: path,
                layout: layout,
                tilesPerExpert: record.tilesPerExpert,
                outFeaturesPerExpert: record.outFeaturesPerExpert,
                inFeatures: record.inFeatures,
                sourceID: Self.sourceID(layer: manifest.layer, projection: projection),
                fileAccess: fileAccess)
        }
        guard containers.isEmpty || containers.count == ExpertProjection.allCases.count else {
            throw TinyK3Error.configuration(
                "layer \(manifest.layer) has \(containers.count) of "
                    + "\(ExpertProjection.allCases.count) expert containers")
        }
        guard containers.isEmpty || manifest.experts > 0 else {
            throw TinyK3Error.configuration(
                "layer \(manifest.layer) has expert containers but declares no experts")
        }

        var slotForExpert: [Int: Int] = [:]
        for (slot, expert) in manifest.slotToExpert.enumerated() { slotForExpert[expert] = slot }
        guard manifest.slotToExpert.count == manifest.experts,
            slotForExpert.count == manifest.slotToExpert.count,
            Set(manifest.slotToExpert) == Set(0..<manifest.experts)
        else {
            throw TinyK3Error.configuration(
                "slot_to_expert must list every id in 0..<\(manifest.experts) exactly once")
        }

        let blobPath = rootedAccess == nil
            ? Self.resolve(manifest.deterministic.file, relativeTo: directory)
            : manifest.deterministic.file
        // `stat(2)`, not `attributesOfItem`: the latter does not traverse
        // symbolic links, and every path that actually *reads* this blob —
        // `TileReader`'s `open(2)`, and `MXFP4TileContainer.open`'s `stat` for
        // the containers next to it — does. A symlinked blob would therefore
        // report the length of the link's target string (133 bytes) and be
        // rejected here while reading perfectly well, which is how a manifest
        // whose files are linked rather than copied gets refused for no reason.
        let actual = try fileAccess.withDescriptor(blobPath) { descriptor in
            var status = stat()
            guard fstat(descriptor, &status) == 0, status.st_size >= 0 else {
                throw StorageCoreError.posix(operation: "fstat", path: blobPath, code: errno)
            }
            return Int(status.st_size)
        }
        guard manifest.deterministic.bytes >= 0,
            actual >= manifest.deterministic.bytes
        else {
            throw TinyK3Error.configuration(
                "\(manifest.deterministic.file) is \(actual), "
                    + "manifest declares \(manifest.deterministic.bytes)")
        }
        try Self.validateDeterministic(
            manifest.deterministic, layer: manifest.layer, actualBytes: actual)

        self.layer = manifest.layer
        self.experts = manifest.experts
        self.containers = containers
        self.slotForExpert = slotForExpert
        self.manifestPath = manifestPath
        self.deterministic = DeterministicStream(
            path: blobPath,
            bytes: manifest.deterministic.bytes,
            units: manifest.deterministic.units,
            sourceID: Self.sourceID(layer: manifest.layer, projection: nil),
            bytesByRole: manifest.deterministic.bytesByRole,
            fileAccess: fileAccess)
    }

    private static func validateDeterministic(
        _ stream: Manifest.Deterministic, layer: Int, actualBytes: Int
    ) throws {
        guard !stream.units.isEmpty else {
            throw TinyK3Error.configuration("layer \(layer) deterministic stream has no units")
        }
        guard stream.bytesByRole.values.allSatisfy({ $0 >= 0 }) else {
            throw TinyK3Error.configuration(
                "layer \(layer) deterministic bytes_by_role contains a negative length")
        }

        var indexes = Set<Int>()
        var ranges: [(unit: DeterministicUnit, end: UInt64)] = []
        var lengthSum = UInt64Accounting.SaturatingSum()
        var derivedByRole: [String: UInt64Accounting.SaturatingSum] = [:]
        for unit in stream.units {
            guard unit.index >= 0, indexes.insert(unit.index).inserted else {
                throw TinyK3Error.configuration(
                    "layer \(layer) deterministic unit index \(unit.index) is negative or repeated")
            }
            guard !unit.tensor.isEmpty, !unit.role.isEmpty else {
                throw TinyK3Error.configuration(
                    "layer \(layer) deterministic unit \(unit.index) has no tensor or role")
            }
            guard unit.dtype == "F32" || unit.dtype == "BF16" else {
                throw TinyK3Error.configuration(
                    "layer \(layer) deterministic unit \(unit.index) has unsupported dtype \(unit.dtype)")
            }
            guard unit.rowStart >= 0, unit.rows > 0, unit.cols > 0 else {
                throw TinyK3Error.configuration(
                    "layer \(layer) deterministic unit \(unit.index) carries a negative start or "
                        + "non-positive shape")
            }
            guard unit.length > 0, unit.payloadLength > 0,
                unit.payloadLength <= unit.length
            else {
                throw TinyK3Error.configuration(
                    "layer \(layer) deterministic unit \(unit.index) has length \(unit.length) "
                        + "and payload_length \(unit.payloadLength); both must be positive and "
                        + "payload must fit the range")
            }
            guard unit.offset % UInt64(MXFP4TileContainer.regionAlignment) == 0,
                unit.length % MXFP4TileContainer.regionAlignment == 0
            else {
                throw TinyK3Error.configuration(
                    "layer \(layer) deterministic unit \(unit.index) range \(unit.offset) "
                        + "+\(unit.length) is not \(MXFP4TileContainer.regionAlignment)-byte aligned")
            }
            let end = unit.offset.addingReportingOverflow(UInt64(unit.length))
            guard !end.overflow else {
                throw TinyK3Error.configuration(
                    "layer \(layer) deterministic unit \(unit.index) offset + length exceeds UInt64.max")
            }
            guard end.partialValue <= UInt64(stream.bytes),
                end.partialValue <= UInt64(actualBytes),
                end.partialValue <= UInt64(Int64.max)
            else {
                throw TinyK3Error.configuration(
                    "layer \(layer) deterministic unit \(unit.index) ends at \(end.partialValue), "
                        + "outside the declared \(stream.bytes)-byte stream")
            }

            let elementBytes = unit.dtype == "F32" ? 4 : 2
            let elements = unit.rows.multipliedReportingOverflow(by: unit.cols)
            let payload = elements.partialValue.multipliedReportingOverflow(by: elementBytes)
            guard !elements.overflow, !payload.overflow,
                payload.partialValue == unit.payloadLength
            else {
                throw TinyK3Error.configuration(
                    "layer \(layer) deterministic unit \(unit.index) shape "
                        + "\(unit.rows)x\(unit.cols) \(unit.dtype) does not fit its "
                        + "\(unit.payloadLength)-byte payload")
            }
            ranges.append((unit, end.partialValue))
            lengthSum.add(UInt64(unit.length))
            var role = derivedByRole[unit.role] ?? UInt64Accounting.SaturatingSum()
            role.add(UInt64(unit.length))
            derivedByRole[unit.role] = role
        }
        guard indexes == Set(0..<stream.units.count) else {
            throw TinyK3Error.configuration(
                "layer \(layer) deterministic unit indexes must be exactly 0..<\(stream.units.count)")
        }
        guard !lengthSum.didOverflow, lengthSum.value == UInt64(stream.bytes) else {
            throw TinyK3Error.configuration(
                "layer \(layer) deterministic unit lengths do not account for "
                    + "the declared \(stream.bytes) bytes")
        }

        let ordered = ranges.sorted { $0.unit.offset < $1.unit.offset }
        for (previous, current) in zip(ordered, ordered.dropFirst())
        where current.unit.offset < previous.end {
            throw TinyK3Error.configuration(
                "layer \(layer) deterministic unit \(current.unit.index) range overlaps "
                    + "unit \(previous.unit.index)")
        }

        let stated = stream.bytesByRole.mapValues(UInt64.init)
        let derived = derivedByRole.mapValues(\.value)
        guard derivedByRole.values.allSatisfy({ !$0.didOverflow }), stated == derived else {
            throw TinyK3Error.configuration(
                "layer \(layer) deterministic bytes_by_role does not match its unit ranges")
        }

        var grouped: [String: [DeterministicUnit]] = [:]
        for unit in stream.units { grouped[unit.tensor, default: []].append(unit) }
        for (tensor, units) in grouped {
            var nextRow = 0
            for unit in units.sorted(by: { $0.rowStart < $1.rowStart }) {
                guard unit.rowStart == nextRow else {
                    throw TinyK3Error.configuration(
                        "layer \(layer) tensor \(tensor) band \(unit.index) starts at row "
                            + "\(unit.rowStart), expected \(nextRow)")
                }
                let end = nextRow.addingReportingOverflow(unit.rows)
                guard !end.overflow else {
                    throw TinyK3Error.configuration(
                        "layer \(layer) tensor \(tensor) row ranges exceed Int.max")
                }
                nextRow = end.partialValue
            }
        }
    }

    /// Where a manifest's `file` actually is.
    ///
    /// A relative name is joined against the manifest's own directory — the
    /// symlink-farm layout, unchanged and still the default. An **absolute**
    /// path is taken as written, which is how a translation avoids materialising
    /// links at all (``PublishedArtifactLayout/BlobReference``): on iOS the
    /// blobs sit on a security-scoped external volume whose mount point carries
    /// a per-mount UUID, so a link written on internal storage names a path that
    /// stops being true the moment the drive is replugged.
    ///
    /// One function, used by both the container path and the deterministic blob
    /// path, because the `stat(2)` guard below and the reads must resolve a name
    /// the same way — the bug this file already carries a comment about.
    private static func resolve(_ file: String, relativeTo directory: URL) -> String {
        file.hasPrefix("/") ? file : directory.appendingPathComponent(file).path
    }

    /// Distinct per file, so the pool's cache map cannot confuse tile 4 of one
    /// container with tile 4 of another.
    static func sourceID(layer: Int, projection: ExpertProjection?) -> UInt64 {
        let code: UInt64
        switch projection {
        case .w1: code = 1
        case .w2: code = 2
        case .w3: code = 3
        case nil: code = 4
        }
        return UInt64(layer) << 8 | code
    }

    /// Load every `layer*-manifest.json` in a directory, lowest layer first.
    public static func loadAll(
        directory: String, fileAccess: ModelFileAccess? = nil
    ) throws -> [FlagshipLayerStreams] {
        let names = try FileManager.default.contentsOfDirectory(atPath: directory)
            .filter { $0.hasPrefix("layer") && $0.hasSuffix("-manifest.json") }
            .sorted()
        guard !names.isEmpty else {
            throw TinyK3Error.configuration("no layer manifest in \(directory)")
        }
        return try names
            .map { try FlagshipLayerStreams(
                manifestPath: URL(fileURLWithPath: directory).appendingPathComponent($0).path,
                fileAccess: fileAccess)
            }
            .sorted { $0.layer < $1.layer }
    }
}

/// How many reads one expert costs.
///
/// The whole question M8 asks of the read plan. `perTile` is the container's
/// natural unit — 13 reads per expert at flagship shape (spec §16.2). `perExpert`
/// exploits the expert-major layout to span an expert's bands in one read per
/// projection, 3 per expert. Same bytes either way; the difference is request
/// count, request size, and how much of a fixed-size slot each one wastes.
public enum FetchGranularity: String, Codable, Sendable, CaseIterable {
    case perTile = "per-tile"
    case perExpert = "per-expert-contiguous"

    public var readsPerExpert: String {
        switch self {
        case .perTile: return "13 (3 + 7 + 3)"
        case .perExpert: return "3 (one per projection)"
        }
    }
}
