import CryptoKit
import Darwin
import Foundation

/// Which function produced a digest.
///
/// Two, and not one, because the repos need two. A `*-minirun` repo stores its
/// tiles as LFS objects, which carry a SHA-256; it also stores plain JSON and
/// text as ordinary git blobs, which carry a git-blob SHA-1 and no SHA-256 at
/// all. MiniMax-H3 has 16 such payload files. A verifier that knows only
/// SHA-256 does not fail on them — it finds no digest and skips them, which is
/// the quiet version of not checking.
public enum DigestAlgorithm: String, Codable, Sendable, CaseIterable {
    case sha256
    case gitBlobSHA1
}

/// A digest, tagged with the function that made it.
public enum FileDigest: Codable, Sendable, Equatable, Hashable {
    /// LFS-backed: the tree's `lfs.oid`, and the same value the resolve 302
    /// puts in `x-linked-etag`.
    case sha256(hex: String)
    /// Git-blob-backed: the tree's `oid`, which is
    /// `sha1("blob " + byteCount + "\0" + content)`.
    case gitBlobSHA1(hex: String)

    public var algorithm: DigestAlgorithm {
        switch self {
        case .sha256: return .sha256
        case .gitBlobSHA1: return .gitBlobSHA1
        }
    }

    public var hex: String {
        switch self {
        case .sha256(let hex), .gitBlobSHA1(let hex): return hex.lowercased()
        }
    }

    public init(algorithm: DigestAlgorithm, hex: String) {
        switch algorithm {
        case .sha256: self = .sha256(hex: hex.lowercased())
        case .gitBlobSHA1: self = .gitBlobSHA1(hex: hex.lowercased())
        }
    }

    public func matches(_ other: String) -> Bool {
        hex == other.lowercased().trimmingCharacters(in: .whitespaces)
    }

    private enum CodingKeys: String, CodingKey { case algorithm, hex }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let algorithm = try container.decode(DigestAlgorithm.self, forKey: .algorithm)
        let hex = try container.decode(String.self, forKey: .hex)
        self.init(algorithm: algorithm, hex: hex)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(algorithm, forKey: .algorithm)
        try container.encode(hex, forKey: .hex)
    }
}

/// Computes both digests without ever holding a whole file.
///
/// A K3 tile is five gigabytes. `Data(contentsOf:)` on one is not a slow
/// verification, it is a dead process, so every path here streams in 4 MiB
/// pieces and keeps only the hash state.
public enum FileDigestComputer {
    /// Read size. Large enough that the syscall overhead disappears, small
    /// enough that eight concurrent verifications cost 32 MiB, not 32 GiB.
    public static let chunkBytes = 4 << 20

    public static func digest(
        ofFileAt url: URL, algorithm: DigestAlgorithm,
        cancellationRequested: @escaping @Sendable () -> Bool = { false }
    ) throws -> FileDigest {
        let descriptor = try openForReading(url)
        defer { _ = close(descriptor) }
        return try digest(
            ofOpenFileDescriptor: descriptor, algorithm: algorithm,
            cancellationRequested: cancellationRequested)
    }

    /// Hash an already-open regular file without taking ownership of its
    /// descriptor. This is internal so filesystem-authority code can bind the
    /// bytes it verifies to an `openat` result instead of reopening a path
    /// between validation and hashing.
    static func digest(
        ofOpenFileDescriptor descriptor: Int32, algorithm: DigestAlgorithm,
        cancellationRequested: @escaping @Sendable () -> Bool = { false }
    ) throws -> FileDigest {
        switch algorithm {
        case .sha256:
            return .sha256(
                hex: try sha256(
                    ofOpenFileDescriptor: descriptor,
                    cancellationRequested: cancellationRequested))
        case .gitBlobSHA1:
            return .gitBlobSHA1(
                hex: try gitBlobSHA1(
                    ofOpenFileDescriptor: descriptor,
                    cancellationRequested: cancellationRequested))
        }
    }

    public static func sha256(
        ofFileAt url: URL,
        cancellationRequested: @escaping @Sendable () -> Bool = { false }
    ) throws -> String {
        let descriptor = try openForReading(url)
        defer { _ = close(descriptor) }
        return try sha256(
            ofOpenFileDescriptor: descriptor,
            cancellationRequested: cancellationRequested)
    }

    private static func sha256(
        ofOpenFileDescriptor descriptor: Int32,
        cancellationRequested: @escaping @Sendable () -> Bool
    ) throws -> String {
        var hasher = SHA256()
        try stream(descriptor, cancellationRequested: cancellationRequested) {
            hasher.update(bufferPointer: $0)
        }
        return hexadecimal(hasher.finalize())
    }

    /// Git's object id: SHA-1 over the header `blob <byteCount>\0` followed by
    /// the content. The header is why a git-blob id is not "the SHA-1 of the
    /// file" and why recomputing it needs the size first.
    public static func gitBlobSHA1(
        ofFileAt url: URL,
        cancellationRequested: @escaping @Sendable () -> Bool = { false }
    ) throws -> String {
        let descriptor = try openForReading(url)
        defer { _ = close(descriptor) }
        return try gitBlobSHA1(
            ofOpenFileDescriptor: descriptor,
            cancellationRequested: cancellationRequested)
    }

    private static func gitBlobSHA1(
        ofOpenFileDescriptor descriptor: Int32,
        cancellationRequested: @escaping @Sendable () -> Bool
    ) throws -> String {
        let size = try byteCount(ofOpenFileDescriptor: descriptor)
        var hasher = Insecure.SHA1()
        hasher.update(data: Data("blob \(size)\0".utf8))
        try stream(descriptor, cancellationRequested: cancellationRequested) {
            hasher.update(bufferPointer: $0)
        }
        return hexadecimal(hasher.finalize())
    }

    public static func gitBlobSHA1(of content: Data) -> String {
        var hasher = Insecure.SHA1()
        hasher.update(data: Data("blob \(content.count)\0".utf8))
        hasher.update(data: content)
        return hexadecimal(hasher.finalize())
    }

    public static func sha256(of content: Data) -> String {
        hexadecimal(SHA256.hash(data: content))
    }

    /// The file's length right now.
    ///
    /// `stat(2)` and not `URL.resourceValues(forKeys:)`. Foundation caches
    /// resource values on the `URL` instance, so a URL held across a loop keeps
    /// answering with the size it had the first time it was asked — which, for
    /// a part file being appended to chunk by chunk, means the resume offset
    /// never moves and the same range is fetched forever. The whole resume
    /// contract rests on this number being the file's, not a remembered one.
    public static func byteCount(ofFileAt url: URL) throws -> UInt64 {
        var status = stat()
        guard stat(url.path, &status) == 0 else {
            throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: url.path])
        }
        return UInt64(max(0, status.st_size))
    }

    private static func openForReading(_ url: URL) throws -> Int32 {
        try url.withUnsafeFileSystemRepresentation { path in
            guard let path else { throw CocoaError(.fileReadInvalidFileName) }
            let value = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
            guard value >= 0 else { throw posixError() }
            return value
        }
    }

    private static func byteCount(ofOpenFileDescriptor descriptor: Int32) throws -> UInt64 {
        var status = stat()
        guard fstat(descriptor, &status) == 0 else { throw posixError() }
        guard status.st_mode & S_IFMT == S_IFREG, status.st_size >= 0 else {
            throw POSIXError(.EFTYPE)
        }
        return UInt64(status.st_size)
    }

    private static func stream(
        _ descriptor: Int32,
        cancellationRequested: @escaping @Sendable () -> Bool,
        _ consume: (UnsafeRawBufferPointer) -> Void
    ) throws {
        // FileHandle.read(upToCount:) returns a fresh Foundation Data object on
        // every iteration. On very large artifacts those backing NSDatas can
        // remain in the surrounding autorelease pool until the verification
        // operation returns, so a logically streaming digest still grows with
        // the file. One POSIX buffer makes the memory bound structural: there
        // is exactly one 4 MiB allocation no matter whether the file is 4 MiB
        // or 1.5 TB.
        let storage = UnsafeMutableRawBufferPointer.allocate(
            byteCount: chunkBytes, alignment: MemoryLayout<UInt64>.alignment)
        defer { storage.deallocate() }

        var offset: off_t = 0
        while true {
            if cancellationRequested() { throw CancellationError() }
            let count = pread(descriptor, storage.baseAddress, storage.count, offset)
            if count > 0 {
                consume(UnsafeRawBufferPointer(start: storage.baseAddress, count: count))
                let (next, overflow) = offset.addingReportingOverflow(off_t(count))
                guard !overflow else { throw POSIXError(.EOVERFLOW) }
                offset = next
                continue
            }
            if count == 0 { return }
            if errno == EINTR { continue }
            throw posixError()
        }
    }

    private static func posixError() -> POSIXError {
        POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }

    private static func hexadecimal<D: Sequence>(_ bytes: D) -> String where D.Element == UInt8 {
        bytes.map { String(format: "%02x", $0) }.joined()
    }
}
