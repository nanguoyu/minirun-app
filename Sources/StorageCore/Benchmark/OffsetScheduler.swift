import Foundation

/// Chooses the offset of the next read.
///
/// Two invariants hold for every offset it produces, and the tests assert both
/// exhaustively:
///
/// - It is a multiple of the read size. Unaligned reads would be measuring the
///   kernel's read-modify-write path rather than the storage path.
/// - `offset + readSize <= fileSize`. The file is treated as a whole number of
///   blocks and the remainder is never read, so a run can never produce a short
///   read by walking off the end — which keeps a short read meaningful as a
///   signal of something actually going wrong, such as a drive disconnect
///   (spec §12.4).
public struct OffsetScheduler {
    public let readSizeBytes: Int
    /// Whole blocks in the file. The trailing partial block is unused.
    public let blockCount: UInt64
    /// First block of this worker's slice (sequential only).
    public let rangeStartBlock: UInt64
    /// Blocks in this worker's slice (sequential only).
    public let rangeBlockCount: UInt64

    private let pattern: AccessPattern
    private var random: SplitMix64
    private var cursor: UInt64 = 0

    public static func blockCount(fileSizeBytes: UInt64, readSizeBytes: Int) -> UInt64 {
        guard readSizeBytes > 0 else { return 0 }
        return fileSizeBytes / UInt64(readSizeBytes)
    }

    /// - Parameters:
    ///   - workerIndex: which worker this scheduler belongs to, `0 ..< workerCount`.
    ///   - seed: the run seed. Offset selection uses a stream derived from it,
    ///     independent of the stream that defines file content, so that the two
    ///     can share one user-visible `--seed` without correlating.
    public init(
        fileSizeBytes: UInt64,
        readSizeBytes: Int,
        pattern: AccessPattern,
        workerIndex: Int,
        workerCount: Int,
        seed: UInt64
    ) {
        precondition(readSizeBytes > 0, "read size must be positive")
        precondition(workerCount >= 1, "worker count must be positive")
        precondition(workerIndex >= 0 && workerIndex < workerCount, "worker index out of range")

        self.readSizeBytes = readSizeBytes
        self.pattern = pattern
        self.blockCount = Self.blockCount(
            fileSizeBytes: fileSizeBytes, readSizeBytes: readSizeBytes)
        self.random = SplitMix64(
            seed: SplitMix64.derive(seed: seed, label: UInt64(workerIndex)))

        // Sequential workers get disjoint contiguous slices, so each one sees
        // real sequential locality instead of interleaving with its peers and
        // producing a random pattern by accident.
        let start = blockCount * UInt64(workerIndex) / UInt64(workerCount)
        let end = blockCount * UInt64(workerIndex + 1) / UInt64(workerCount)
        if end > start {
            self.rangeStartBlock = start
            self.rangeBlockCount = end - start
        } else {
            // More workers than blocks: the slice would be empty, so this
            // worker shares the whole file rather than stalling.
            self.rangeStartBlock = 0
            self.rangeBlockCount = blockCount
        }
    }

    /// Next offset, or `nil` when the file holds no complete block.
    public mutating func nextOffset() -> UInt64? {
        guard blockCount > 0 else { return nil }
        let block: UInt64
        switch pattern {
        case .random:
            // Modulo bias here is on the order of blockCount / 2^64 and is not
            // worth a rejection loop in the read path.
            block = random.next() % blockCount
        case .sequential:
            block = rangeStartBlock + (cursor % rangeBlockCount)
            cursor &+= 1
        }
        return block * UInt64(readSizeBytes)
    }
}
