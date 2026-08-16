import Foundation

/// Whether a tile reader recomputes the SHA-256 a container recorded for a tile
/// every time it loads that tile, or reads it under verification authority the
/// caller already holds for the exact artifact revision.
///
/// The default at every declaration site is ``verifyEveryLoad``. A library
/// cannot invent trust: only the caller that ran the complete verification pass
/// and still holds its rooted authority knows that the bytes about to be read
/// are the bytes that pass covered, and it says so by naming that authority
/// here. This mirrors the K3 rooted-I/O model of spec §30 v0.6.22 — its
/// deterministic, global, expert and tile-container readers open bounded
/// descriptors from held authority and do not re-hash per read; descriptor
/// identity and the registered-root binding are what they recheck instead.
///
/// What ``trustHeldAuthority`` removes is exactly one thing: the SHA-256 over
/// the tile's bytes. Every other per-load check stays — the file's recorded
/// physical identity is rechecked by the rooted opener on every open, the file
/// length is rechecked against the manifest and the container header, the
/// requested geometry is rechecked against the header, and the packed bytes are
/// still handled by ``BlockFP8Weights`` as before.
public enum TileDigestPolicy: Sendable {
    /// Recompute and check the recorded digest on every load. The only policy a
    /// reader may assume for itself.
    case verifyEveryLoad
    /// Read under the caller's held full-verification authority for this exact
    /// artifact revision, without re-hashing tile bytes.
    case trustHeldAuthority(HeldVerificationAuthority)

    public var recomputesDigestOnEveryLoad: Bool {
        if case .verifyEveryLoad = self { return true }
        return false
    }
}

/// The evidence a caller states when it asks a reader to trust bytes it already
/// verified: which artifact revision the pass covered, how much of it, and the
/// live authority object itself.
///
/// `holder` is the authority — retained for this value's lifetime so the rooted
/// descriptors, the registered-root binding and the recorded file identities
/// that make the trust meaningful cannot be released while a reader is trusting
/// them. `ModelAdapters` deliberately does not know that type: the verification
/// machinery lives above it in the package graph (spec §7.1), and the reader
/// only needs the revision it names and the guarantee that it is still alive.
public struct HeldVerificationAuthority: @unchecked Sendable {
    /// The source repository the complete verification pass covered.
    public let sourceRepository: String
    /// The immutable source revision it covered. A reader refuses the policy if
    /// this does not name the revision its own manifest declares.
    public let sourceRevision: String
    /// Files included in that pass.
    public let verifiedFileCount: Int
    /// Bytes hashed by that pass.
    public let verifiedBytes: UInt64
    private let holder: AnyObject

    public init(
        sourceRepository: String,
        sourceRevision: String,
        verifiedFileCount: Int,
        verifiedBytes: UInt64,
        heldBy holder: AnyObject
    ) {
        self.sourceRepository = sourceRepository
        self.sourceRevision = sourceRevision
        self.verifiedFileCount = verifiedFileCount
        self.verifiedBytes = verifiedBytes
        self.holder = holder
    }

    /// One line for a run's log, so a record says what the run trusted.
    public var summary: String {
        "\(sourceRepository)@\(sourceRevision), \(verifiedFileCount) files / "
            + "\(verifiedBytes) B verified"
    }
}
