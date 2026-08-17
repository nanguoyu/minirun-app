import Foundation
import MLX
import MLXBridge
import StorageCore

/// The expert containers written by `Tools/k3_reference/convert_tiny_experts.py`,
/// indexed by `(layer, projection)`.
///
/// ## One tile holds four experts
///
/// The container format requires a tile's packed *and* scale sub-region each to
/// be a whole number of 16 KiB units, so each can be adopted as its own
/// zero-copy `MLXArray`. The scale side binds: `rows * cols / 32` divisible by
/// 16384 forces `rows * cols` divisible by 524,288. A tiny-K3 expert projection
/// is 256×512 or 512×256 — 131,072 elements, exactly a quarter of the minimum.
///
/// So a tile spans four experts, and "one tile is one expert" is the wrong
/// mental model to design a pager request around at this scale. At flagship K3
/// the arithmetic inverts: 3584×3072 is 21 minimum tiles per projection.
///
/// The mapping is read from the converter's manifest rather than recomputed
/// here — M6A decided it is data, not a convention in code, precisely so a
/// future geometry change shows up as a different manifest and not as a
/// silently wrong row offset.
public final class RoutedExpertContainers {

    public struct Entry: Sendable {
        public let layer: Int
        public let projection: ExpertProjection
        public let path: String
        public let layout: MXFP4TileContainer.Layout
        /// Experts stacked into one tile, in row order.
        public let expertsPerTile: Int
        /// Output features of one expert, i.e. the tile's rows divided by
        /// ``expertsPerTile``.
        public let rowsPerExpert: Int
        public let inFeatures: Int
        /// `tile -> global expert ids`, in the order their rows appear.
        public let tileExperts: [[Int]]
        /// Opens the container from its owning authority. Filesystem fixtures
        /// use ``ModelFileAccess/filesystem``; a product runtime supplies an
        /// opener backed by the fully verified artifact root.
        let fileAccess: ModelFileAccess

        init(
            layer: Int, projection: ExpertProjection, path: String,
            layout: MXFP4TileContainer.Layout, expertsPerTile: Int,
            rowsPerExpert: Int, inFeatures: Int, tileExperts: [[Int]],
            fileAccess: ModelFileAccess = .filesystem
        ) {
            self.layer = layer
            self.projection = projection
            self.path = path
            self.layout = layout
            self.expertsPerTile = expertsPerTile
            self.rowsPerExpert = rowsPerExpert
            self.inFeatures = inFeatures
            self.tileExperts = tileExperts
            self.fileAccess = fileAccess
        }

        /// Which tile holds a given global expert, or `nil`.
        public func tile(holding expert: Int) -> Int? {
            tileExperts.firstIndex { $0.contains(expert) }
        }
    }

    public let directory: String
    public let entries: [String: Entry]
    /// Sum of every container file's size — the number the paged budget has to
    /// come in under for the milestone's memory claim to mean anything.
    public let totalBytes: Int

    private static func key(layer: Int, projection: ExpertProjection) -> String {
        "\(layer).\(projection.rawValue)"
    }

    public func entry(layer: Int, projection: ExpertProjection) throws -> Entry {
        guard let entry = entries[Self.key(layer: layer, projection: projection)] else {
            throw TinyK3Error.configuration(
                "no expert container for layer \(layer) projection \(projection.rawValue) in \(directory)")
        }
        return entry
    }

    public init(directory: String) throws {
        self.directory = directory
        let manifestURL = URL(fileURLWithPath: directory).appendingPathComponent("manifest.json")
        guard
            let root = try JSONSerialization.jsonObject(with: try Data(contentsOf: manifestURL))
                as? [String: Any],
            let containers = root["containers"] as? [[String: Any]]
        else {
            throw TinyK3Error.configuration("\(manifestURL.path) has no 'containers' list")
        }

        var entries: [String: Entry] = [:]
        var totalBytes = 0
        for record in containers {
            guard
                let layer = record["layer"] as? Int,
                let projectionName = record["projection"] as? String,
                let projection = ExpertProjection(rawValue: projectionName),
                let file = record["file"] as? String,
                let expertsPerTile = record["experts_per_tile"] as? Int,
                let rowsPerExpert = record["out_features_per_expert"] as? Int,
                let inFeatures = record["in_features"] as? Int,
                let tiles = record["tiles"] as? [[String: Any]]
            else {
                throw TinyK3Error.configuration("malformed container record: \(record)")
            }
            let path = URL(fileURLWithPath: directory).appendingPathComponent(file).path
            // The header is re-read and re-derived rather than trusted from the
            // manifest: `MXFP4TileContainer.open` checks the file is as long as
            // it claims and that every derivable field agrees.
            let layout = try MXFP4TileContainer.open(path: path)
            guard layout.geometry.rows == expertsPerTile * rowsPerExpert,
                layout.geometry.cols == inFeatures
            else {
                throw TinyK3Error.configuration(
                    "\(file): header geometry \(layout.geometry.rows)x\(layout.geometry.cols) "
                        + "contradicts the manifest's \(expertsPerTile)x\(rowsPerExpert)x\(inFeatures)")
            }

            var tileExperts = [[Int]](repeating: [], count: layout.tileCount)
            for tile in tiles {
                guard let index = tile["tile"] as? Int, let members = tile["experts"] as? [Int],
                    index >= 0, index < layout.tileCount
                else {
                    throw TinyK3Error.configuration("malformed tile record in \(file): \(tile)")
                }
                tileExperts[index] = members
            }
            guard tileExperts.allSatisfy({ $0.count == expertsPerTile }) else {
                throw TinyK3Error.configuration("\(file): a tile does not list \(expertsPerTile) experts")
            }

            entries[Self.key(layer: layer, projection: projection)] = Entry(
                layer: layer, projection: projection, path: path, layout: layout,
                expertsPerTile: expertsPerTile, rowsPerExpert: rowsPerExpert,
                inFeatures: inFeatures, tileExperts: tileExperts,
                fileAccess: .filesystem)
            guard let bytes = Int(exactly: layout.totalBytes) else {
                throw TinyK3Error.configuration("\(file): byte length does not fit Int")
            }
            let sum = totalBytes.addingReportingOverflow(bytes)
            guard !sum.overflow else {
                throw TinyK3Error.configuration("expert container byte total overflows Int")
            }
            totalBytes = sum.partialValue
        }
        guard !entries.isEmpty else {
            throw TinyK3Error.configuration("\(manifestURL.path) lists no containers")
        }
        self.entries = entries
        self.totalBytes = totalBytes
    }

    /// Build the same generic `(layer, projection)` index from another model's
    /// already-validated public manifest.
    ///
    /// The K3 JSON decoder above remains K3-specific. V4 validates its own
    /// manifest and self-describing headers, then enters the shared pager only
    /// through this checked constructor.
    init(directory: String, validatedEntries: [Entry]) throws {
        guard !validatedEntries.isEmpty else {
            throw DeepSeekV4Error.experts("the unit contains no expert containers")
        }
        var indexed: [String: Entry] = [:]
        var total = 0
        for entry in validatedEntries {
            let key = Self.key(layer: entry.layer, projection: entry.projection)
            guard indexed[key] == nil else {
                throw DeepSeekV4Error.experts(
                    "layer \(entry.layer) repeats projection \(entry.projection.rawValue)")
            }
            guard let bytes = Int(exactly: entry.layout.totalBytes) else {
                throw DeepSeekV4Error.experts(
                    "\(entry.path) byte length does not fit Int")
            }
            let sum = total.addingReportingOverflow(bytes)
            guard !sum.overflow else {
                throw DeepSeekV4Error.experts("expert container byte total overflows Int")
            }
            total = sum.partialValue
            indexed[key] = entry
        }
        self.directory = directory
        self.entries = indexed
        self.totalBytes = total
    }
}

// MARK: - The shared tile loop

/// Which tiles a routing decision actually needs, and where each `(token, slot)`
/// lands inside them.
struct TilePlan {
    /// Tiles that hold at least one selected expert, ascending.
    let tiles: [Int]
    /// `tile -> [tokens * slots]` local expert index within that tile. Entries
    /// whose expert is elsewhere carry 0 and are masked out.
    let localIndices: [Int: [Int32]]
    /// `tile -> [tokens * slots]` selection mask.
    let masks: [Int: [Bool]]

    init(entry: RoutedExpertContainers.Entry, expertIds: [[Int]]) throws {
        var needed: Set<Int> = []
        var owner: [Int] = []
        owner.reserveCapacity(expertIds.count * (expertIds.first?.count ?? 0))
        for row in expertIds {
            for expert in row {
                guard let tile = entry.tile(holding: expert) else {
                    throw TinyK3Error.configuration(
                        "expert \(expert) is in no tile of layer \(entry.layer) \(entry.projection.rawValue)")
                }
                needed.insert(tile)
                owner.append(tile)
            }
        }
        var localIndices: [Int: [Int32]] = [:]
        var masks: [Int: [Bool]] = [:]
        for tile in needed.sorted() {
            var local = [Int32](repeating: 0, count: owner.count)
            var mask = [Bool](repeating: false, count: owner.count)
            var flat = 0
            for row in expertIds {
                for expert in row {
                    if owner[flat] == tile {
                        // Position within the tile, from the manifest's row
                        // order rather than from `expert % expertsPerTile`.
                        local[flat] = Int32(entry.tileExperts[tile].firstIndex(of: expert)!)
                        mask[flat] = true
                    }
                    flat += 1
                }
            }
            localIndices[tile] = local
            masks[tile] = mask
        }
        self.tiles = needed.sorted()
        self.localIndices = localIndices
        self.masks = masks
    }
}

/// The one place the routed-expert tile loop is written.
///
/// Both backends go through it, so the paged and resident runs cannot drift
/// apart in tile order, gather geometry, merge order or evaluation cadence —
/// which is what makes "bit-identical" a statement about the byte source and
/// nothing else. It is the same discipline `PagedMXFP4Linear.executeTiles`
/// applies to the dense case, for the same reason.
///
/// ## Per tile, or batched
///
/// `tilesPerDispatch` selects between two programs and both backends are given
/// the same value by their caller, for the reason above.
///
/// - `1` is the program M6 measured: one `gatherQuantizedMM` per tile that
///   holds at least one selected expert, merged with `which`.
/// - `k > 1` concatenates up to `k` tiles' expert stacks into one and issues
///   **one** dispatch for them. A tile is already a stack of `expertsPerTile`,
///   so this lengthens an axis that is there rather than inventing one.
///
/// The doc comment on `ResidentRoutedExpertBackend` used to argue against this,
/// on the grounds that a wider stack "would additionally change the kernel's
/// launch geometry, and a mismatch would no longer point at the pager". That is
/// a real hazard and the answer to it is not to avoid the width — it is to give
/// *both* arms the same width, which is what the shared parameter here does.
/// Whether the width itself moves a bit is a separate question, measured in
/// `MLXBridgeTests.BatchedGatherEquivalenceTests` and again arm-against-arm in
/// `PagedExpertGatherTests`.
/// `evalObserver` receives the nanoseconds spent forcing each dispatch's graph.
/// That `eval` is the one place this loop waits on Metal, which is what makes
/// it the whole of a `gatherCompute` measurement; passing nil measures nothing
/// and changes nothing.
func gatherOverTiles(
    _ x: MLXArray,
    entry: RoutedExpertContainers.Entry,
    expertIds: [[Int]],
    tilesPerDispatch: Int = 1,
    evalObserver: ((UInt64) -> Void)? = nil,
    weightsForTile: (Int) throws -> MXFP4Weights
) throws -> MLXArray {
    let tokens = expertIds.count
    let slots = expertIds.first?.count ?? 0
    let plan = try TilePlan(entry: entry, expertIds: expertIds)
    guard tilesPerDispatch >= 1 else {
        throw TinyK3Error.configuration(
            "tilesPerDispatch must be at least 1, got \(tilesPerDispatch)")
    }
    let flatX = x.reshaped([tokens, entry.inFeatures])

    let chunks = stride(from: 0, to: plan.tiles.count, by: tilesPerDispatch).map {
        Array(plan.tiles[$0..<min($0 + tilesPerDispatch, plan.tiles.count)])
    }
    // One chunk covers every selected expert, so the gather's own output *is*
    // the answer: `which` against an all-true mask selects `produced`
    // everywhere, which is what makes skipping it a removal of work rather than
    // a change of arithmetic.
    let single = chunks.count == 1
    var accumulator: MLXArray? =
        single ? nil : MLXArray.zeros([tokens, slots, entry.rowsPerExpert], dtype: .float32)

    for chunk in chunks {
        var stacks: [MXFP4Weights] = []
        stacks.reserveCapacity(chunk.count)
        for tile in chunk { stacks.append(try weightsForTile(tile)) }

        // A tile's local index becomes an index into the concatenated stack, so
        // it is shifted by the tiles ahead of it in this chunk. Rows whose tile
        // is not in this chunk keep index 0 and are discarded by the mask —
        // arithmetic that is computed and thrown away, never added.
        var indexValues = [Int32](repeating: 0, count: tokens * slots)
        var maskValues = [Bool](repeating: false, count: tokens * slots)
        var base = 0
        for (position, tile) in chunk.enumerated() {
            let local = plan.localIndices[tile]!
            let mask = plan.masks[tile]!
            for flat in 0..<(tokens * slots) where mask[flat] {
                indexValues[flat] = Int32(base) + local[flat]
                maskValues[flat] = true
            }
            base += stacks[position].experts ?? 1
        }

        let weights = try stackTiles(stacks)
        let produced = mxfp4GatherMM(flatX, weights, rhsIndices: MLXArray(indexValues, [tokens, slots]))
        let merged: MLXArray
        if let current = accumulator {
            merged = which(MLXArray(maskValues, [tokens, slots, 1]), produced, current)
        } else {
            merged = produced
        }
        // Evaluating here is what releases a paged tile's slot. Without it the
        // loop would hold every tile it has ever expressed, and the budget
        // would describe the graph rather than the working set.
        //
        // **And the wait stays, even under `transferMode: .copy` where nothing
        // depends on it for correctness.** `copyOut` returns each lease before
        // the dispatch is expressed and the operands are then MLX-owned, so an
        // `asyncEval` here would keep every guarantee this point exists for —
        // it runs the same `eval_impl` and detaches the same tape. It was
        // implemented, armed on the real 166.9 GB artifact, and **refused by
        // the memory gate**: with the three projections' dispatches left in
        // flight, `mlxActiveBytes` ran +53.5 MB above the first decode pass on
        // an ordinary token and +157.6 MB on the worst of forty, against the
        // 64 MB `DeepSeekV4ProductGateTests` allows before it calls a pass
        // "another generation of V4 state". +53.5 MB is exactly one gather's
        // transient — `expertsPerToken` copied tiles plus the concatenated
        // stack — so the depth that fits the stated envelope is *zero*.
        //
        // Deferring it needs `transientExecutionBytes` re-derived first
        // (ADR 0015), which is a budget change and not a scheduling one.
        // Measured, not assumed: `docs/experiments/2026-08-17-v4-async-forcing.md`
        // arm T1 carries the numbers and the digest that says the arithmetic
        // was never the problem.
        if let evalObserver {
            let evaluate = MonotonicClock.now()
            merged.eval()
            evalObserver(MonotonicClock.now() &- evaluate)
        } else {
            merged.eval()
        }
        accumulator = merged
    }

    guard let result = accumulator else {
        throw TinyK3Error.configuration("a routing decision selected no experts")
    }
    return result
}

/// One gather operand out of several tiles' expert stacks.
///
/// `gatherQuantizedMM` indexes a contiguous expert axis, and each tile is its
/// own allocation — a pager slot, or an MLX-owned array — so a batch spanning
/// tiles has to be assembled. Both backends assemble it identically and from
/// the same shape, because an assembly performed by only one of them would be a
/// difference in the program rather than in the byte source.
private func stackTiles(_ parts: [MXFP4Weights]) throws -> MXFP4Weights {
    guard let first = parts.first else {
        throw TinyK3Error.configuration("a batched gather needs at least one tile")
    }
    guard parts.count > 1 else { return first }
    return try MXFP4Weights(
        packed: concatenated(parts.map(\.packed), axis: 0),
        scales: concatenated(parts.map(\.scales), axis: 0))
}
