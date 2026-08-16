import Darwin
import Foundation

/// Parameters for generating a reproducible benchmark file (WP-003).
public struct FileGenerationRequest: Equatable {
    public var path: String
    public var sizeBytes: UInt64
    public var seed: UInt64
    public var chunkBytes: Int

    /// Large enough to amortize the write syscall, small enough that the
    /// working set of the generator itself stays irrelevant.
    public static let defaultChunkBytes = 4 << 20
    public static let maximumChunkBytes = 1 << 30

    public init(
        path: String,
        sizeBytes: UInt64,
        seed: UInt64,
        chunkBytes: Int = FileGenerationRequest.defaultChunkBytes
    ) {
        self.path = path
        self.sizeBytes = sizeBytes
        self.seed = seed
        self.chunkBytes = chunkBytes
    }

    public func validate() throws {
        guard !path.isEmpty else {
            throw StorageCoreError.invalidArgument("--path is required")
        }
        guard sizeBytes > 0 else {
            throw StorageCoreError.invalidArgument("--size must be greater than zero")
        }
        guard chunkBytes > 0 else {
            throw StorageCoreError.invalidArgument("--chunk must be greater than zero")
        }
        guard chunkBytes <= Self.maximumChunkBytes else {
            throw StorageCoreError.invalidArgument(
                "--chunk of \(ByteSize.format(UInt64(chunkBytes))) exceeds the "
                    + "\(ByteSize.format(UInt64(Self.maximumChunkBytes))) limit")
        }
    }
}

public struct FileGenerationResult: Equatable {
    public let path: String
    public let sizeBytes: UInt64
    public let seed: UInt64
    public let chunkBytes: Int
    public let elapsedSeconds: Double

    public var bytesPerSecond: Double {
        elapsedSeconds > 0 ? Double(sizeBytes) / elapsedSeconds : 0
    }
}

/// Writes files whose content is a pure function of `(seed, offset)`.
///
/// The chunk size is a throughput knob only: content is addressed by absolute
/// file offset, so the same seed and size produce identical bytes whatever
/// chunking is used. `FileGeneratorTests` pins that.
public enum FileGenerator {
    public static func generate(
        _ request: FileGenerationRequest,
        progress: ((_ written: UInt64, _ total: UInt64) -> Void)? = nil
    ) throws -> FileGenerationResult {
        try request.validate()

        let descriptor = open(request.path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
        guard descriptor >= 0 else {
            throw StorageCoreError.posix(operation: "open", path: request.path, code: errno)
        }
        defer { close(descriptor) }

        // Write around the page cache. A file that is still resident when the
        // benchmark starts would measure memcpy, and spec §5.7 is explicit that
        // a warm cache must not be the only case that is ever measured. It also
        // keeps a multi-gigabyte fixture from evicting everything else.
        _ = fcntl(descriptor, F_NOCACHE, 1)

        let chunkBytes = min(request.chunkBytes, Int(min(request.sizeBytes, UInt64(Int.max))))
        let buffer = try AlignedBuffer(count: chunkBytes)

        let started = MonotonicClock.now()
        var written: UInt64 = 0
        while written < request.sizeBytes {
            let remaining = request.sizeBytes - written
            let span = Int(min(UInt64(chunkBytes), remaining))
            let slice = UnsafeMutableRawBufferPointer(start: buffer.pointer, count: span)
            DeterministicContent.fill(slice, seed: request.seed, fileOffset: written)

            try writeFully(
                descriptor: descriptor, path: request.path, buffer: buffer.pointer, count: span,
                fileOffset: written)
            written += UInt64(span)
            progress?(written, request.sizeBytes)
        }

        // Push the file to media rather than to the volatile write cache, so a
        // benchmark run immediately afterwards reads storage and not a pending
        // write buffer. fsync() alone does not guarantee this on macOS.
        if fcntl(descriptor, F_FULLFSYNC) == -1 {
            // Some filesystems (notably some exFAT and network mounts) reject
            // F_FULLFSYNC; a plain fsync is the honest best effort there.
            guard fsync(descriptor) == 0 else {
                throw StorageCoreError.posix(
                    operation: "fsync", path: request.path, code: errno)
            }
        }

        let elapsed = MonotonicClock.seconds(since: started)
        return FileGenerationResult(
            path: request.path,
            sizeBytes: written,
            seed: request.seed,
            chunkBytes: chunkBytes,
            elapsedSeconds: elapsed
        )
    }

    private static func writeFully(
        descriptor: Int32,
        path: String,
        buffer: UnsafeMutableRawPointer,
        count: Int,
        fileOffset: UInt64
    ) throws {
        var done = 0
        while done < count {
            let result = pwrite(
                descriptor, buffer + done, count - done, off_t(fileOffset + UInt64(done)))
            if result < 0 {
                if errno == EINTR { continue }
                throw StorageCoreError.posix(operation: "pwrite", path: path, code: errno)
            }
            if result == 0 {
                throw StorageCoreError.shortTransfer(
                    operation: "pwrite", path: path, offset: fileOffset + UInt64(done),
                    expected: count - done, actual: 0)
            }
            done += result
        }
    }
}
