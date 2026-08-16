import Darwin
import Foundation
import StorageCore

/// Opens one model file by the reference carried in a translated manifest.
///
/// Normal package callers use ``filesystem`` and therefore keep the existing
/// path-based behaviour. A sandboxed product can instead supply an opener
/// backed by an already-authorized directory descriptor. Each call returns an
/// owned descriptor; readers close it when their bounded lifetime ends.
public struct ModelFileAccess: @unchecked Sendable {
    public typealias Opener = @Sendable (String) throws -> Int32

    /// What an opener promises about the object behind each descriptor it
    /// returns. Stated by whoever writes the opener, because only that code
    /// knows what it does; a reader may act on the promise but must never
    /// assume it.
    public enum IdentityAssurance: Sendable, Equatable {
        /// A pathname is resolved on every open and nothing rechecks that the
        /// object opened is the object some earlier pass inspected. The
        /// filesystem opener, tests, and command-line probes are all this.
        case pathReopen
        /// Every open is serviced beneath an already-open registered root with
        /// `O_NOFOLLOW` on every component, and the complete recorded file
        /// identity — device or persistent volume, object id, size, and the
        /// nanosecond modification and status-change times — plus the
        /// registered-root binding are rechecked before the descriptor is
        /// returned. This is the rooted runtime I/O of spec §30 v0.6.22, with
        /// the durable identity of v0.6.26/27.
        case rootedDescriptorIdentity
    }

    /// What this access promises about descriptor identity. See
    /// ``IdentityAssurance``.
    public let identityAssurance: IdentityAssurance
    private let opener: Opener

    public init(
        identityAssurance: IdentityAssurance = .pathReopen,
        opener: @escaping Opener
    ) {
        self.identityAssurance = identityAssurance
        self.opener = opener
    }

    public static let filesystem = ModelFileAccess { path in
        let descriptor = Darwin.open(path, O_RDONLY | O_CLOEXEC)
        guard descriptor >= 0 else {
            throw StorageCoreError.posix(operation: "open", path: path, code: errno)
        }
        return descriptor
    }

    /// Return an owned descriptor. The caller must close it.
    func open(_ reference: String) throws -> Int32 {
        try opener(reference)
    }

    func withDescriptor<T>(
        _ reference: String, _ body: (Int32) throws -> T
    ) throws -> T {
        let descriptor = try open(reference)
        defer { _ = Darwin.close(descriptor) }
        return try body(descriptor)
    }
}
