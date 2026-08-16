import Foundation

/// Lifecycle states of a buffer slot (spec §8.5, WP-008).
///
/// WP-008 names the sequence `free -> filling -> ready -> gpu_in_flight ->
/// cached/free`. `leased` is this implementation's name for `gpu_in_flight`,
/// widened for a reason that turned out to matter: a slot is not held by "the
/// GPU" but by *n* independent consumers, and in this milestone n is routinely
/// two — one slot is adopted as two `MLXArray`s (packed weights and scales),
/// each with its own finalizer. A boolean "in flight" flag would return the
/// slot to the pool when the first of the two finalizers ran, and the second
/// consumer would then be reading recycled memory.
public enum SlotState: String, Sendable, Codable, CaseIterable {
    /// Owned by the pool, contents meaningless (poisoned in debug builds).
    case free
    /// Assigned to a read that has not completed. Contents are partial and
    /// MUST NOT be exposed to a consumer (spec §12.4).
    case filling
    /// Completely filled and verified against the requested length; not yet
    /// handed to a consumer.
    case ready
    /// One or more live references. The refcount is carried alongside.
    case leased
    /// Refcount reached zero but the contents are still valid and retained
    /// under the cache policy. Reusable by identity; evictable on demand.
    case cached
}

/// Failures of the pager itself, as distinct from the storage under it.
///
/// Spec §5.8: none of these is recoverable by guessing. An illegal transition
/// is a programming error and says which slot and which edge; a short read says
/// which bytes are missing; cancellation says why.
public enum PagerError: Error, CustomStringConvertible, LocalizedError {
    /// A state-machine edge that does not exist was attempted.
    case illegalTransition(slot: Int, from: SlotState, to: String)
    /// `release` was called on a slot whose refcount was already zero.
    case refcountUnderflow(slot: Int, state: SlotState)
    /// The pool or reader was cancelled while a caller was waiting.
    case cancelled(reason: String)
    /// A waiter was woken by ``BufferPool/interrupt()`` so it could re-check
    /// its own state. Not a failure of the pool, and not sticky: a caller that
    /// still wants a slot simply asks again.
    case interrupted
    /// A bounded wait expired. Carries the wait so a too-short timeout is
    /// distinguishable from a genuine stall.
    case timedOut(operation: String, waitedSeconds: Double)
    /// A configuration is internally inconsistent. Rejected at construction
    /// rather than clamped, because a silently clamped budget stops being a
    /// budget (spec §5.3).
    case configuration(String)
    /// A region does not fit the pool's slot size.
    case regionTooLarge(identity: RegionIdentity, length: Int, slotBytes: Int)
    /// A public byte-range request is not representable or lies outside the
    /// source as it existed when the reader opened it.
    case invalidRegion(
        identity: RegionIdentity, offset: UInt64, length: Int, sourceBytes: UInt64,
        reason: String)
    /// `acquire` was asked for a region that was never scheduled and could not
    /// be scheduled (the reader was already finished).
    case regionNotScheduled(identity: RegionIdentity)
    /// A read for this region failed; carries the underlying storage error.
    case readFailed(identity: RegionIdentity, underlying: String)

    public var description: String {
        switch self {
        case .illegalTransition(let slot, let from, let to):
            return "slot \(slot): illegal transition \(from.rawValue) -> \(to)"
        case .refcountUnderflow(let slot, let state):
            return "slot \(slot): release with refcount already 0 (state \(state.rawValue))"
        case .cancelled(let reason):
            return "pager cancelled: \(reason)"
        case .interrupted:
            return "slot wait interrupted"
        case .timedOut(let operation, let waited):
            return String(format: "%@ timed out after %.3f s", operation, waited)
        case .configuration(let detail):
            return "pager configuration rejected: \(detail)"
        case .regionTooLarge(let identity, let length, let slotBytes):
            return "region \(identity) is \(length) bytes, slot size is \(slotBytes)"
        case .invalidRegion(let identity, let offset, let length, let sourceBytes, let reason):
            return "region \(identity) at \(offset) +\(length) is invalid for a \(sourceBytes)-byte source: \(reason)"
        case .regionNotScheduled(let identity):
            return "region \(identity) was never scheduled for reading"
        case .readFailed(let identity, let underlying):
            return "read of region \(identity) failed: \(underlying)"
        }
    }

    public var errorDescription: String? { description }
}
