import Darwin
import Foundation
import MLX
import MLXBridge
import StorageCore

/// Supplies one block's routed-expert weights, a half-tile at a time.
///
/// The seam is a *stack*, not a weight handle, for the same reason V4's
/// ``RoutedExpertBackend`` is a gather: the backends this will eventually
/// compare — a pager streaming pair tiles, the same containers held resident —
/// differ in where the bytes come from and in nothing else.
///
/// It is deliberately narrower than V4's protocol. There is no `layer`
/// parameter, because a V4.1 reader is opened on one unit; and the unit of
/// prefetch is a *tile*, because that is what a read costs.
public protocol DeepSeekV41RoutedExpertSource: AnyObject {
    /// `[experts.count, rows, cols]` for one projection, in the order asked
    /// for; `[rows, cols]` when one expert is asked for.
    func stack(
        _ experts: [Int], projection: DeepSeekV41ExpertProjection
    ) throws -> MXFP4Weights

    /// Begin reading the tiles these experts need, without waiting for any.
    ///
    /// All three projections of a block read the *same* experts, so every byte
    /// this queues is a byte the block is certain to ask for. The default does
    /// nothing, which is right for a source that holds its bytes already.
    func prefetch(_ experts: [Int]) throws
}

extension DeepSeekV41RoutedExpertSource {
    public func prefetch(_ experts: [Int]) throws {}
}

/// Reads routed-expert halves out of an opened V4.1 block unit.
///
/// Every read is one whole tile, because that is what the container's recorded
/// SHA-256 covers and because the pair is the format's own granularity. A tile
/// that serves two of a token's routed experts is read once —
/// ``DeepSeekV41ExpertStaging`` says when, and ``tileReads`` counts what
/// actually happened, so a caller can hold the plan against the I/O rather than
/// trust it.
///
/// ### Why this takes a `ModelFileAccess`
///
/// ``DeepSeekV41BlockArtifact`` keeps its own access private, and a reader that
/// opened the placement's `fileReference` by path would silently escape a
/// rooted, identity-checked opener the moment the product used one. So the
/// caller passes the same access it opened the unit with, and the pairing is
/// its responsibility. Phase 3 should replace this with an accessor on the
/// artifact so the pairing cannot be got wrong; the open item is recorded in
/// `docs/experiments/2026-09-11-v41-phase2-moe.md`.
public final class DeepSeekV41ExpertReader: DeepSeekV41RoutedExpertSource, @unchecked Sendable {
    public let artifact: DeepSeekV41BlockArtifact
    private let fileAccess: ModelFileAccess
    private let verifiesTileDigests: Bool

    /// Whole tiles read since this reader was opened. A pair collision shows
    /// up here as a read that did not happen.
    public private(set) var tileReads = 0
    /// Expert-projections asked for since this reader was opened.
    public private(set) var expertRequests = 0

    public init(
        artifact: DeepSeekV41BlockArtifact,
        fileAccess: ModelFileAccess = .filesystem,
        verifiesTileDigests: Bool = true
    ) {
        self.artifact = artifact
        self.fileAccess = fileAccess
        self.verifiesTileDigests = verifiesTileDigests
    }

    /// The pair-aware plan for these experts, from the unit's `expert_order`.
    public func staging(for experts: [Int]) throws -> DeepSeekV41ExpertStaging {
        try DeepSeekV41ExpertStaging.plan(experts: experts, in: artifact)
    }

    /// One expert's own `[rows, cols]` matrix, reading its pair tile once.
    public func weights(
        expert: Int, projection: DeepSeekV41ExpertProjection
    ) throws -> MXFP4Weights {
        try stack([expert], projection: projection)
    }

    public func stack(
        _ experts: [Int], projection: DeepSeekV41ExpertProjection
    ) throws -> MXFP4Weights {
        guard !experts.isEmpty else {
            throw DeepSeekV41Error.experts("expert reader: asked for no experts")
        }
        let placements = try experts.map { try artifact.expert($0, projection: projection) }
        let first = placements[0]
        // One container, one geometry: the placements come from one unit and
        // one projection, so this is an invariant being stated rather than a
        // condition being handled.
        guard placements.allSatisfy({
            $0.fileReference == first.fileReference && $0.rows == first.rows
                && $0.columns == first.columns && $0.packedBytes == first.packedBytes
                && $0.scaleBytes == first.scaleBytes && $0.tileStride == first.tileStride
        }) else {
            throw DeepSeekV41Error.experts(
                "expert reader: one projection's experts do not share a container geometry")
        }

        // Request indices grouped by the tile that serves them: a tile holding
        // two of the requested experts appears once, and both halves are cut
        // out of the one read.
        var requestsByTile = [Int: [Int]]()
        for (request, placement) in placements.enumerated() {
            requestsByTile[placement.tile, default: []].append(request)
        }

        var packed = [UInt8](repeating: 0, count: experts.count * first.packedBytes)
        var scales = [UInt8](repeating: 0, count: experts.count * first.scaleBytes)

        try fileAccess.withDescriptor(first.fileReference) { descriptor in
            let layout = try QuantizedTileContainer.open(
                fileDescriptor: descriptor, path: first.fileReference)
            guard layout.tileStride == first.tileStride,
                layout.geometry.cols == first.columns
            else {
                throw DeepSeekV41Error.experts(
                    "expert reader: \(first.fileReference) no longer describes the tiles "
                        + "the manifest was reconciled against")
            }
            var tile = [UInt8](repeating: 0, count: layout.tileStride)
            // Ascending, so the reads walk the file forward whatever order the
            // router named the experts in.
            for tileIndex in requestsByTile.keys.sorted() {
                try tile.withUnsafeMutableBytes { destination in
                    try Self.readExactly(
                        descriptor: descriptor, reference: first.fileReference,
                        offset: layout.tileOffset(tileIndex), length: layout.tileStride,
                        destination: destination.baseAddress!)
                }
                tileReads += 1
                if verifiesTileDigests {
                    try tile.withUnsafeBytes { bytes in
                        try QuantizedTileContainer.verifyTileDigest(
                            bytes, layout: layout, tile: tileIndex)
                    }
                }
                for request in requestsByTile[tileIndex] ?? [] {
                    let placement = placements[request]
                    packed.replaceSubrange(
                        (request * first.packedBytes)..<((request + 1) * first.packedBytes),
                        with: tile[placement.packedOffsetInTile
                            ..< (placement.packedOffsetInTile + placement.packedBytes)])
                    scales.replaceSubrange(
                        (request * first.scaleBytes)..<((request + 1) * first.scaleBytes),
                        with: tile[placement.scaleOffsetInTile
                            ..< (placement.scaleOffsetInTile + placement.scaleBytes)])
                }
            }
        }
        expertRequests += experts.count

        return try packed.withUnsafeBytes { packedBytes in
            try scales.withUnsafeBytes { scaleBytes in
                try MXFP4Weights(
                    packedBytes: packedBytes, scaleBytes: scaleBytes,
                    experts: experts.count == 1 ? nil : experts.count,
                    outFeatures: first.rows, inFeatures: first.columns)
            }
        }
    }

    public func prefetch(_ experts: [Int]) throws {
        // A synchronous reader has nothing to queue: the tiles it will read are
        // read when `stack` asks for them. The plan is still derived, so a
        // caller can see what a read-ahead *would* have to issue and a Phase 3
        // backend can override this one method.
        _ = try staging(for: experts)
    }

    private static func readExactly(
        descriptor: Int32, reference: String, offset: UInt64, length: Int,
        destination: UnsafeMutableRawPointer
    ) throws {
        guard length > 0, let start = off_t(exactly: offset) else {
            throw DeepSeekV41Error.experts(
                "expert reader: \(reference) has an unrepresentable read range")
        }
        var completed = 0
        while completed < length {
            let result = pread(
                descriptor, destination + completed, length - completed,
                start + off_t(completed))
            if result < 0 {
                if errno == EINTR { continue }
                throw StorageCoreError.posix(
                    operation: "pread", path: reference, code: errno)
            }
            guard result > 0 else {
                throw StorageCoreError.shortTransfer(
                    operation: "pread", path: reference,
                    offset: offset + UInt64(completed),
                    expected: length - completed, actual: 0)
            }
            completed += result
        }
    }
}
