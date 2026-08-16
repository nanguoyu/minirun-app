import Foundation

/// Stable name for a byte range, independent of where it currently lives.
///
/// The pager keys everything on identity rather than on file offset so that a
/// cache lookup, a lease, and a read request all talk about the same thing. The
/// two fields are deliberately opaque integers: this layer must not know
/// whether `index` counts row tiles, experts, or physical chunks (spec §5.5,
/// §8.4 — the read unit is not fixed globally).
public struct RegionIdentity: Hashable, Sendable, CustomStringConvertible, Codable {
    /// Which source the region belongs to. One container, one shard, one file.
    public let sourceID: UInt64
    /// Which region within that source.
    public let index: UInt64

    public init(sourceID: UInt64, index: UInt64) {
        self.sourceID = sourceID
        self.index = index
    }

    public var description: String { "\(sourceID)#\(index)" }
}

/// The pager's request unit: "these bytes, called this" (spec §8.3, §8.4).
///
/// This is the whole input side of the pager API. It is intentionally *not* a
/// cursor — there is no `nextTile()` anywhere in this module. A dense linear
/// walks tiles in order and a routed MoE layer asks for the experts its router
/// selected, in router order; both are the same call with different
/// descriptors, which is what keeps M8 from needing a redesign.
public struct RegionDescriptor: Hashable, Sendable, CustomStringConvertible {
    public let identity: RegionIdentity
    /// Byte offset in the source. Should be page-aligned for a direct read path.
    public let offset: UInt64
    /// Byte length. Must not exceed the pool's slot size.
    public let length: Int

    public init(identity: RegionIdentity, offset: UInt64, length: Int) {
        self.identity = identity
        self.offset = offset
        self.length = length
    }

    public var description: String {
        "region \(identity) at \(offset) +\(length)"
    }
}
