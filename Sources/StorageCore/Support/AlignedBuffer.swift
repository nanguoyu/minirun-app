import Darwin
import Foundation

/// A page-aligned heap buffer, reused across reads.
///
/// Alignment matters twice over. `F_NOCACHE` reads take a direct path only when
/// the buffer is suitably aligned, and an unaligned destination otherwise
/// forces the kernel to bounce through an intermediate copy — which would show
/// up as storage latency in a benchmark that is trying to measure storage. The
/// buffer is allocated once per worker and reused so that allocation cost never
/// lands inside a measured read.
public final class AlignedBuffer {
    public let pointer: UnsafeMutableRawPointer
    public let count: Int
    public let alignment: Int

    public static var pageSize: Int { Int(getpagesize()) }

    public init(count: Int, alignment: Int = AlignedBuffer.pageSize) throws {
        precondition(count > 0, "buffer size must be positive")
        // posix_memalign requires a power-of-two alignment that is a multiple
        // of sizeof(void *).
        var effective = max(alignment, MemoryLayout<UnsafeRawPointer>.size)
        if effective & (effective - 1) != 0 {
            effective = Int(UInt64(effective).nextPowerOfTwo)
        }

        var raw: UnsafeMutableRawPointer?
        let result = posix_memalign(&raw, effective, count)
        guard result == 0, let allocated = raw else {
            throw StorageCoreError.posix(
                operation: "posix_memalign(\(count) bytes)", path: "", code: result)
        }
        self.pointer = allocated
        self.count = count
        self.alignment = effective
    }

    deinit {
        free(pointer)
    }

    public var mutableBytes: UnsafeMutableRawBufferPointer {
        UnsafeMutableRawBufferPointer(start: pointer, count: count)
    }

    public var bytes: UnsafeRawBufferPointer {
        UnsafeRawBufferPointer(start: pointer, count: count)
    }
}

extension UInt64 {
    fileprivate var nextPowerOfTwo: UInt64 {
        guard self > 1 else { return 1 }
        return 1 << (64 - (self - 1).leadingZeroBitCount)
    }
}
