import Foundation

/// File content defined as a pure function of `(seed, byte offset)`.
///
/// The file is a sequence of 8-byte lanes; lane *i* holds
/// `SplitMix64.value(seed:index: i)` in little-endian order. Two consequences
/// matter:
///
/// - The same seed and size always produce byte-identical files, on any
///   machine, regardless of the chunk size used to write them.
/// - The expected content of *any* range can be recomputed from the seed and
///   the offset alone. A benchmark reading random 512 KiB blocks can therefore
///   check what it read without holding a reference copy, which is what makes
///   `bench --verify` affordable (spec §17.1 wants correctness checked
///   alongside measurement, not after it).
public enum DeterministicContent {
    /// Bytes per generated lane.
    public static let laneSize = 8

    /// Content of the lane covering `laneIndex`, as it appears in the file.
    @inline(__always)
    public static func lane(seed: UInt64, index: UInt64) -> UInt64 {
        SplitMix64.value(seed: seed, index: index)
    }

    /// Writes the expected content of `[fileOffset, fileOffset + count)` into
    /// `destination`. `fileOffset` need not be lane-aligned.
    public static func fill(
        _ destination: UnsafeMutableRawBufferPointer,
        seed: UInt64,
        fileOffset: UInt64
    ) {
        guard let base = destination.baseAddress, !destination.isEmpty else { return }
        let count = destination.count
        let lanes = UInt64(laneSize)
        var written = 0
        var offset = fileOffset

        // Leading partial lane, when the range does not start on a lane boundary.
        let skew = Int(offset % lanes)
        if skew != 0 {
            var value = lane(seed: seed, index: offset / lanes).littleEndian
            let take = min(laneSize - skew, count)
            withUnsafeBytes(of: &value) { source in
                base.copyMemory(from: source.baseAddress! + skew, byteCount: take)
            }
            written += take
            offset += UInt64(take)
        }

        // Whole lanes.
        while count - written >= laneSize {
            var value = lane(seed: seed, index: offset / lanes).littleEndian
            withUnsafeBytes(of: &value) { source in
                (base + written).copyMemory(from: source.baseAddress!, byteCount: laneSize)
            }
            written += laneSize
            offset += lanes
        }

        // Trailing partial lane.
        if written < count {
            var value = lane(seed: seed, index: offset / lanes).littleEndian
            let take = count - written
            withUnsafeBytes(of: &value) { source in
                (base + written).copyMemory(from: source.baseAddress!, byteCount: take)
            }
        }
    }

    /// Convenience allocation-returning form. For tests and small ranges; the
    /// hot paths use `fill(_:seed:fileOffset:)` into a reused buffer.
    public static func bytes(seed: UInt64, fileOffset: UInt64, count: Int) -> [UInt8] {
        guard count > 0 else { return [] }
        var output = [UInt8](repeating: 0, count: count)
        output.withUnsafeMutableBytes { fill($0, seed: seed, fileOffset: fileOffset) }
        return output
    }

    /// Outcome of comparing observed bytes against the expected content.
    public struct Verification: Equatable {
        public var comparedBytes: Int
        public var mismatchedBytes: Int
        /// Absolute file offset of the first differing byte, if any.
        public var firstMismatchOffset: UInt64?

        public var isClean: Bool { mismatchedBytes == 0 }

        public static let clean = Verification(
            comparedBytes: 0, mismatchedBytes: 0, firstMismatchOffset: nil)

        public mutating func merge(_ other: Verification) {
            comparedBytes += other.comparedBytes
            mismatchedBytes += other.mismatchedBytes
            if let offset = other.firstMismatchOffset {
                firstMismatchOffset = min(firstMismatchOffset ?? offset, offset)
            }
        }
    }

    /// Compares `source` against the expected content at `fileOffset`.
    ///
    /// A whole aligned lane is checked with one unaligned load and one integer
    /// compare; per-byte work happens only for the ragged ends of an unaligned
    /// range and for lanes that actually differ. A clean verification of a
    /// 512 KiB read therefore costs about 65k compares, which is cheap enough
    /// to sample inside a running benchmark.
    public static func verify(
        _ source: UnsafeRawBufferPointer,
        seed: UInt64,
        fileOffset: UInt64
    ) -> Verification {
        guard let base = source.baseAddress, !source.isEmpty else { return .clean }
        let count = source.count
        let lanes = UInt64(laneSize)
        var result = Verification(
            comparedBytes: count, mismatchedBytes: 0, firstMismatchOffset: nil)

        var index = 0
        while index < count {
            let offset = fileOffset &+ UInt64(index)
            let skew = Int(offset % lanes)
            let span = min(laneSize - skew, count - index)
            let expected = lane(seed: seed, index: offset / lanes)

            if span == laneSize {
                let observed = UInt64(
                    littleEndian: base.loadUnaligned(fromByteOffset: index, as: UInt64.self))
                if observed != expected {
                    recordByteMismatches(
                        observed: base + index, expected: expected, skew: 0, span: span,
                        fileOffset: offset, into: &result)
                }
            } else {
                recordByteMismatches(
                    observed: base + index, expected: expected, skew: skew, span: span,
                    fileOffset: offset, into: &result)
            }
            index += span
        }
        return result
    }

    /// Per-byte comparison of `span` bytes starting at lane byte `skew`.
    private static func recordByteMismatches(
        observed: UnsafeRawPointer,
        expected: UInt64,
        skew: Int,
        span: Int,
        fileOffset: UInt64,
        into result: inout Verification
    ) {
        var value = expected.littleEndian
        withUnsafeBytes(of: &value) { expectedBytes in
            for byte in 0..<span
            where observed.load(fromByteOffset: byte, as: UInt8.self) != expectedBytes[skew + byte] {
                result.mismatchedBytes += 1
                let offset = fileOffset &+ UInt64(byte)
                result.firstMismatchOffset = min(result.firstMismatchOffset ?? offset, offset)
            }
        }
    }
}
