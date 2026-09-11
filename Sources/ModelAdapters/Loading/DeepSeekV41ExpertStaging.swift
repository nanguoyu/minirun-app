import Foundation

/// What a token's routed experts cost to *read*, given that a tile is a pair.
///
/// V4's pool slot is an expert, because a V4 tile is an expert. A V4.1 tile is
/// two experts, so the read granularity and the arithmetic granularity have
/// come apart: six routed experts are at most six tiles and sometimes five,
/// and a pool slot has to be a tile or it will hold half a tile it paid for.
///
/// This type is the whole of that difference, and it is deliberately not part
/// of any pool. It answers two questions — which distinct tiles a token needs,
/// and for each expert which tile and half to evaluate — from ``expertOrder``
/// alone, with no I/O and no residency. Phase 3 can hang a read-ahead of the
/// ``tiles`` list off it in the shape of `DeepSeekV4RunExpertBackends` without
/// this file learning anything about pools.
///
/// ## How often a pair collides
///
/// Two of a token's routed experts land in one tile only when they are the two
/// halves of that tile. For a uniform choice of `k` experts from `N` with `m`
/// to a tile — the published backbone block is `N = 384`, `k = 6`, `m = 2`,
/// `T = 192` tiles — a given tile is untouched with probability
/// `C(N - m, k) / C(N, k)`, so
///
/// ```text
/// E[distinct tiles] = T * (1 - prod_{i<m} (N - k - i) / (N - i))
/// ```
///
/// which is 5.96084 for the backbone block and 2.97638 for a DSpark block
/// (`N = 128`, `k = 3`). A collision therefore saves one tile read on 3.92% of
/// (token, block) draws in the backbone and 2.36% in DSpark: 0.65% and 0.79% of
/// expert bytes. That is small enough that the pair layout is a correctness
/// problem and not a bandwidth lever, and stating the number is what stops it
/// being mistaken for one.
///
/// The routing is of course not uniform, so ``DeepSeekV41ExpertStaging/collisions``
/// is measured on every plan and the expectation above is what the measurement
/// is compared against, not a substitute for it.
public struct DeepSeekV41ExpertStaging: Sendable, Equatable {

    /// One routed expert, and where in the pair layout it is evaluated.
    public struct Slot: Sendable, Equatable {
        public let expert: Int
        public let tile: Int
        /// 0 or 1 — which expert of the pair.
        public let half: Int
        /// Index into ``DeepSeekV41ExpertStaging/tiles``, so a caller that
        /// staged the tiles in order can find this slot's bytes without a
        /// search.
        public let tileSlot: Int
    }

    /// Distinct tiles this token needs, ascending. At most one per expert.
    public let tiles: [Int]
    /// One per requested expert, in the order the router named them.
    public let slots: [Slot]
    /// Experts asked for, minus tiles that must be read. Zero when no two
    /// routed experts share a tile.
    public var collisions: Int { slots.count - tiles.count }

    /// Plan from an opened unit. The unit owns `expert_order`; this owns
    /// nothing but the arithmetic over it.
    public static func plan(
        experts: [Int], in artifact: DeepSeekV41BlockArtifact
    ) throws -> DeepSeekV41ExpertStaging {
        try plan(experts: experts) { try artifact.expert($0) }
    }

    /// Plan from any placement function, so the arithmetic is testable without
    /// a container on disk.
    public static func plan(
        experts: [Int],
        placement: (Int) throws -> (tile: Int, half: Int)
    ) rethrows -> DeepSeekV41ExpertStaging {
        var order = [Int]()
        var slotOfTile = [Int: Int]()
        var slots = [Slot]()
        slots.reserveCapacity(experts.count)
        for expert in experts {
            let (tile, half) = try placement(expert)
            let tileSlot: Int
            if let existing = slotOfTile[tile] {
                tileSlot = existing
            } else {
                tileSlot = order.count
                slotOfTile[tile] = tileSlot
                order.append(tile)
            }
            slots.append(Slot(expert: expert, tile: tile, half: half, tileSlot: tileSlot))
        }
        // `tiles` is ascending so a read-ahead issues them in file order; the
        // slots keep the router's order and carry the index into it, so
        // neither has to be re-derived from the other.
        let ascending = order.sorted()
        var rank = [Int: Int]()
        for (index, tile) in ascending.enumerated() { rank[tile] = index }
        return DeepSeekV41ExpertStaging(
            tiles: ascending,
            slots: slots.map {
                Slot(expert: $0.expert, tile: $0.tile, half: $0.half,
                     tileSlot: rank[$0.tile] ?? 0)
            })
    }

    /// `E[distinct tiles]` for a uniform choice, as derived above.
    ///
    /// Returned as the expectation rather than a probability because it is the
    /// number a read-ahead depth is set from.
    public static func expectedDistinctTiles(
        routedExperts: Int, expertsPerToken: Int, expertsPerTile: Int
    ) throws -> Double {
        guard routedExperts > 0, expertsPerTile > 0,
            routedExperts % expertsPerTile == 0,
            expertsPerToken > 0, expertsPerToken <= routedExperts - expertsPerTile
        else {
            throw DeepSeekV41Error.experts(
                "staging: \(expertsPerToken) of \(routedExperts) experts in tiles of "
                    + "\(expertsPerTile) is not a geometry an expectation is defined for")
        }
        var untouched = 1.0
        for offset in 0..<expertsPerTile {
            untouched *= Double(routedExperts - expertsPerToken - offset)
                / Double(routedExperts - offset)
        }
        return Double(routedExperts / expertsPerTile) * (1 - untouched)
    }

    /// `expertsPerToken - E[distinct tiles]`: tile reads a pair collision saves
    /// per token, on average.
    public static func expectedTileSavings(
        routedExperts: Int, expertsPerToken: Int, expertsPerTile: Int
    ) throws -> Double {
        Double(expertsPerToken) - (try expectedDistinctTiles(
            routedExperts: routedExperts, expertsPerToken: expertsPerToken,
            expertsPerTile: expertsPerTile))
    }
}
