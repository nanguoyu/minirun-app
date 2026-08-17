import Darwin
import Foundation
import MLX
import MLXBridge
import StorageCore

/// The part of V4 expert geometry that is load-bearing for one routed layer.
///
/// Product code derives this from the verified `config.json`; small fixtures
/// state it directly so correctness and memory tests do not need a 284B model.
public struct DeepSeekV4ExpertGeometry: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let expertCount: Int

    public init(hiddenSize: Int, intermediateSize: Int, expertCount: Int) throws {
        guard hiddenSize > 0, intermediateSize > 0, expertCount > 0 else {
            throw DeepSeekV4Error.experts(
                "hidden, intermediate, and expert dimensions must be positive")
        }
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.expertCount = expertCount
    }

    public init(config: DeepSeekV4Config) throws {
        try self.init(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.expertIntermediateSize,
            expertCount: config.routedExpertCount)
    }
}

/// One published `layersNN` V4 expert unit, reconciled against its three
/// self-describing checkpoint containers.
///
/// No repository commit is compiled into this loader. A mutable repository is
/// first resolved to a complete immutable tree by MinirunKit, full verification
/// supplies a rooted opener for those exact bytes, and this type checks whether
/// the architecture at that commit is one it understands.
public final class DeepSeekV4ExpertUnit {
    public let unitID: String
    public let layer: Int
    public let sourceRepository: String
    public let sourceRevision: String
    public let geometry: DeepSeekV4ExpertGeometry
    public let containers: RoutedExpertContainers

    private struct Manifest: Decodable {
        struct File: Decodable {
            let name: String
            let kind: String
            let bytes: UInt64
            let rows: Int?
            let cols: Int?
            let count: Int?
            let elementBits: Int?
            let group: Int?
            let layout: String?

            enum CodingKeys: String, CodingKey {
                case name, kind, bytes, rows, cols, count, group, layout
                case elementBits = "element_bits"
            }
        }

        let unit: String
        let sourceRepository: String
        let sourceRevision: String
        let files: [File]
        let expertOrder: [Int]

        enum CodingKeys: String, CodingKey {
            case unit, files
            case sourceRepository = "source_repo"
            case sourceRevision = "source_revision"
            case expertOrder = "expert_order"
        }
    }

    /// Filesystem convenience for tests and command-line tools. A sandboxed
    /// product should read the manifest through its runtime authority and call
    /// the data initializer below.
    public convenience init(
        manifestURL: URL,
        geometry: DeepSeekV4ExpertGeometry,
        expectedSourceRepository: String? = nil,
        expectedSourceRevision: String? = nil
    ) throws {
        let data = try Data(contentsOf: manifestURL)
        let directory = manifestURL.deletingLastPathComponent()
        try self.init(
            manifestData: data,
            unitReference: directory.lastPathComponent,
            filesystemDirectory: directory,
            fileAccess: .filesystem,
            geometry: geometry,
            expectedSourceRepository: expectedSourceRepository,
            expectedSourceRevision: expectedSourceRevision)
    }

    /// Rooted entry point used by a product runner. `unitReference` is a single
    /// repository directory such as `layers03`; payloads are opened by that
    /// identity, never by a path recovered from the manifest.
    public convenience init(
        manifestData: Data,
        unitReference: String,
        fileAccess: ModelFileAccess,
        geometry: DeepSeekV4ExpertGeometry,
        expectedSourceRepository: String? = nil,
        expectedSourceRevision: String? = nil
    ) throws {
        try self.init(
            manifestData: manifestData,
            unitReference: unitReference,
            filesystemDirectory: nil,
            fileAccess: fileAccess,
            geometry: geometry,
            expectedSourceRepository: expectedSourceRepository,
            expectedSourceRevision: expectedSourceRevision)
    }

    private init(
        manifestData: Data,
        unitReference: String,
        filesystemDirectory: URL?,
        fileAccess: ModelFileAccess,
        geometry: DeepSeekV4ExpertGeometry,
        expectedSourceRepository: String?,
        expectedSourceRevision: String?
    ) throws {
        let manifest: Manifest
        do {
            manifest = try JSONDecoder().decode(Manifest.self, from: manifestData)
        } catch {
            throw DeepSeekV4Error.experts("manifest.json could not be decoded: \(error)")
        }
        guard Self.isSafeComponent(unitReference), manifest.unit == unitReference else {
            throw DeepSeekV4Error.experts(
                "manifest unit '\(manifest.unit)' does not match '\(unitReference)'")
        }
        guard let parsedLayer = Self.layerNumber(manifest.unit) else {
            throw DeepSeekV4Error.experts(
                "unit '\(manifest.unit)' is not a layersNN model layer")
        }
        guard !manifest.sourceRepository.isEmpty, !manifest.sourceRevision.isEmpty else {
            throw DeepSeekV4Error.experts("manifest source repository and revision must be non-empty")
        }
        if let expectedSourceRepository,
            manifest.sourceRepository != expectedSourceRepository
        {
            throw DeepSeekV4Error.experts(
                "source repository '\(manifest.sourceRepository)' does not match the artifact index")
        }
        if let expectedSourceRevision, manifest.sourceRevision != expectedSourceRevision {
            throw DeepSeekV4Error.experts(
                "source revision '\(manifest.sourceRevision)' does not match the artifact index")
        }
        guard manifest.expertOrder.count == geometry.expertCount,
            Set(manifest.expertOrder) == Set(0..<geometry.expertCount)
        else {
            throw DeepSeekV4Error.experts(
                "expert_order must list every id in 0..<\(geometry.expertCount) exactly once")
        }
        var filesByName: [String: Manifest.File] = [:]
        for file in manifest.files {
            guard Self.isSafeComponent(file.name), filesByName[file.name] == nil else {
                throw DeepSeekV4Error.experts(
                    "manifest contains an unsafe or repeated file name '\(file.name)'")
            }
            filesByName[file.name] = file
        }

        var entries: [RoutedExpertContainers.Entry] = []
        for projection in ExpertProjection.allCases {
            let name = "\(manifest.unit)-\(projection.rawValue).mxfp4tile"
            guard let record = filesByName[name], record.kind == "tile-container" else {
                throw DeepSeekV4Error.experts("manifest has no expert container '\(name)'")
            }
            guard record.rows != nil, record.cols != nil, record.count != nil,
                record.elementBits != nil, record.group != nil, record.layout != nil
            else {
                throw DeepSeekV4Error.experts("\(name) omits its container geometry")
            }
            let reference = filesystemDirectory?.appendingPathComponent(name).path
                ?? "\(unitReference)/\(name)"
            let layout = try fileAccess.withDescriptor(reference) { descriptor in
                let opened = try MXFP4TileContainer.open(
                    fileDescriptor: descriptor, path: reference)
                var status = stat()
                guard fstat(descriptor, &status) == 0, status.st_size >= 0 else {
                    throw StorageCoreError.posix(
                        operation: "fstat", path: reference, code: errno)
                }
                guard UInt64(status.st_size) == opened.totalBytes,
                    UInt64(status.st_size) == record.bytes
                else {
                    throw DeepSeekV4Error.experts(
                        "\(name) byte length disagrees with its header or manifest")
                }
                return opened
            }
            let expectedRows = projection == .w2
                ? geometry.hiddenSize : geometry.intermediateSize
            let expectedCols = projection == .w2
                ? geometry.intermediateSize : geometry.hiddenSize
            guard layout.geometry.quantMode == .mxfp4,
                layout.geometry.bits == MXFP4Weights.bits,
                layout.geometry.groupSize == MXFP4Weights.groupSize,
                layout.contentKind == .checkpointDerived,
                layout.tileDigests?.count == geometry.expertCount,
                layout.tileCount == geometry.expertCount,
                layout.geometry.rows == expectedRows,
                layout.geometry.cols == expectedCols,
                record.rows == expectedRows,
                record.cols == expectedCols,
                record.count == geometry.expertCount,
                record.elementBits == MXFP4Weights.bits,
                record.group == MXFP4Weights.groupSize,
                record.layout == TileQuantMode.mxfp4.name
            else {
                throw DeepSeekV4Error.experts(
                    "\(name) is not the expected checkpoint-derived MXFP4 "
                        + "\(expectedRows)x\(expectedCols) expert stack")
            }
            entries.append(
                RoutedExpertContainers.Entry(
                    layer: parsedLayer,
                    projection: projection,
                    path: reference,
                    layout: layout,
                    expertsPerTile: 1,
                    rowsPerExpert: expectedRows,
                    inFeatures: expectedCols,
                    tileExperts: manifest.expertOrder.map { [$0] },
                    fileAccess: fileAccess))
        }

        self.unitID = manifest.unit
        self.layer = parsedLayer
        self.sourceRepository = manifest.sourceRepository
        self.sourceRevision = manifest.sourceRevision
        self.geometry = geometry
        self.containers = try RoutedExpertContainers(
            directory: unitReference, validatedEntries: entries)
    }

    private static func isSafeComponent(_ value: String) -> Bool {
        !value.isEmpty && value != "." && value != ".."
            && !value.contains("/") && !value.contains("\\") && !value.contains("\0")
    }

    private static func layerNumber(_ unit: String) -> Int? {
        guard unit.hasPrefix("layers") else { return nil }
        let suffix = unit.dropFirst("layers".count)
        guard !suffix.isEmpty, suffix.allSatisfy(\.isNumber) else { return nil }
        return Int(suffix)
    }
}

/// Exact V4 routed-expert arithmetic over any backend that implements the
/// shared gather contract.
///
/// The routing weight is applied between SwiGLU and `w2`, matching the official
/// ExpertMLP. This returns the routed sum only; the one shared expert is a
/// separate deterministic branch in the full V4 block.
public enum DeepSeekV4RoutedExperts {
    public static func forward(
        _ input: MLXArray,
        layer: Int,
        expertIDs: [[Int]],
        routingWeights: MLXArray,
        swiGLULimit: Float,
        backend: RoutedExpertBackend
    ) throws -> MLXArray {
        guard input.ndim == 2, input.shape[0] > 0, input.shape[1] > 0 else {
            throw DeepSeekV4Error.experts(
                "input must be a non-empty [tokens, hidden] matrix, got \(input.shape)")
        }
        guard expertIDs.count == input.shape[0],
            let slots = expertIDs.first?.count, slots > 0,
            expertIDs.allSatisfy({ $0.count == slots })
        else {
            throw DeepSeekV4Error.experts(
                "expert ids must be a rectangular row per input token")
        }
        guard routingWeights.shape == [input.shape[0], slots] else {
            throw DeepSeekV4Error.experts(
                "routing weights must be [\(input.shape[0]), \(slots)], got "
                    + "\(routingWeights.shape)")
        }
        guard swiGLULimit.isFinite, swiGLULimit >= 0 else {
            throw DeepSeekV4Error.experts("SwiGLU limit must be finite and nonnegative")
        }
        let flattenedCount = input.shape[0].multipliedReportingOverflow(by: slots)
        guard !flattenedCount.overflow else {
            throw DeepSeekV4Error.experts("token-by-expert work count overflows Int")
        }

        // The earliest point at which every byte this layer's experts need is
        // known: the router has already named the ids on the host, and `w1`,
        // `w3` and `w2` are gathered over that same selection. Queuing all
        // three schedules here, before the first gather is expressed, is what
        // lets the reads for `w3` and `w2` proceed underneath the `w1`
        // dispatch, the SwiGLU, and the shared expert instead of after them.
        // A backend that holds its bytes, or a pager not configured for it,
        // does nothing here.
        try backend.prefetch(
            layer: layer, projections: ExpertProjection.allCases, expertIds: expertIDs)

        var gate = try backend.gather(
            input, layer: layer, projection: .w1, expertIds: expertIDs
        ).asType(.float32)
        var up = try backend.gather(
            input, layer: layer, projection: .w3, expertIds: expertIDs
        ).asType(.float32)
        guard gate.ndim == 3, up.ndim == 3,
            gate.shape[0] == input.shape[0], gate.shape[1] == slots,
            up.shape == gate.shape, gate.shape[2] > 0
        else {
            throw DeepSeekV4Error.experts(
                "w1/w3 gather output must be matching [tokens, routes, intermediate] tensors")
        }
        if swiGLULimit > 0 {
            let upper = MLXArray(swiGLULimit)
            gate = minimum(gate, upper)
            up = minimum(maximum(up, MLXArray(-swiGLULimit)), upper)
        }
        let weighted = K3Activation.silu(gate) * up
            * expandedDimensions(routingWeights.asType(.float32), axis: -1)
        let flattened = weighted.asType(input.dtype).reshaped([
            flattenedCount.partialValue, weighted.shape[2],
        ])
        let flattenedIDs = expertIDs.flatMap { $0 }.map { [$0] }
        let down = try backend.gather(
            flattened, layer: layer, projection: .w2, expertIds: flattenedIDs)
        guard down.shape == [flattenedCount.partialValue, 1, input.shape[1]] else {
            throw DeepSeekV4Error.experts(
                "w2 gather output must be [token-routes, 1, hidden], got \(down.shape)")
        }
        return sum(
            down.reshaped([input.shape[0], slots, input.shape[1]]), axis: 1)
    }
}
