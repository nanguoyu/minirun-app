import Foundation

/// The budget of a paged execution, checked for deadlock-freedom at
/// construction.
///
/// ## The deadlock this exists to prevent
///
/// A lazy compute graph holds its operands. When a tile's bytes are adopted
/// into graph nodes, the slot underneath them stays leased until the graph is
/// evaluated — not until the tile's matmul is *expressed*. So a loop that
/// expresses many tiles before evaluating any of them accumulates held slots,
/// and a loop that expresses more tiles than the pool has slots stops dead: the
/// reader cannot get a slot for the next tile, the consumer is blocked waiting
/// for that tile, and nothing will ever evaluate the graph that would release
/// the slots. It is a true deadlock, not a slowdown, and no timeout fixes it.
///
/// ## The invariant
///
/// ```text
/// evalEveryK + readAhead <= slotCount
/// ```
///
/// Argument that it suffices, given ``TileReader``'s in-order slot assignment.
/// At the moment the consumer blocks waiting for tile *i*:
///
/// - tiles expressed since the last evaluation, and therefore graph-held, number
///   at most `evalEveryK - 1` (tile *i* is not among them; it has not been
///   leased yet);
/// - slots held by the reader are for tiles at index *i* or later, and number at
///   most `readAhead`;
/// - so slots unavailable to tile *i* number at most
///   `evalEveryK - 1 + readAhead - 1 < slotCount`, leaving at least one.
///
/// Tile *i*'s slot request is at the head of the dispatcher's queue, because
/// slots are assigned in schedule order and an out-of-order demand is promoted
/// to the head. It therefore gets that slot, the read completes, the consumer
/// unblocks, and the loop reaches its next evaluation point. Progress is
/// guaranteed at every step, so the loop terminates.
///
/// The check throws rather than clamping. A silently clamped `evalEveryK` would
/// change the measured evaluation cadence without the caller knowing, and a
/// silently raised `slotCount` would change the memory budget — which is the
/// one property this milestone exists to bound (spec §5.3).
public struct PagedReadPlan: Sendable, Equatable {
    /// Slots in the pool. `slotCount * slotBytes` is the whole weight budget.
    public let slotCount: Int
    /// Regions the reader may hold slots for at once.
    public let readAhead: Int
    /// Evaluate the compute graph after this many tiles.
    public let evalEveryK: Int

    public init(slotCount: Int, readAhead: Int, evalEveryK: Int) throws {
        guard slotCount >= 1 else {
            throw PagerError.configuration("slotCount must be at least 1")
        }
        guard readAhead >= 1 else {
            throw PagerError.configuration("readAhead must be at least 1")
        }
        guard evalEveryK >= 1 else {
            throw PagerError.configuration("evalEveryK must be at least 1")
        }
        guard evalEveryK + readAhead <= slotCount else {
            throw PagerError.configuration(
                """
                evalEveryK (\(evalEveryK)) + readAhead (\(readAhead)) exceeds slotCount \
                (\(slotCount)). A compute graph holds its operand slots until it is \
                evaluated, so the pool must have room for both the unevaluated tiles and \
                the read-ahead window at the same time; otherwise the reader waits for a \
                slot that only an evaluation can free, and the evaluation waits for the \
                reader. Raise slotCount to at least \(evalEveryK + readAhead), or lower \
                evalEveryK to \(slotCount - readAhead), or lower readAhead to \
                \(slotCount - evalEveryK).
                """)
        }
        self.slotCount = slotCount
        self.readAhead = readAhead
        self.evalEveryK = evalEveryK
    }

    /// The largest read-ahead this plan could use without violating the
    /// invariant. Useful when sizing a run from a fixed byte budget.
    public var maximumReadAhead: Int { slotCount - evalEveryK }
}
