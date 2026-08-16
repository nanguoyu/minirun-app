import Foundation

/// Build identity for the running binary.
///
/// Spec 17.4 requires every documented benchmark result to record the code
/// revision that produced it. The revision is embedded at build time by the
/// `EmbedGitRevision` plugin, which writes `minirunEmbeddedGitRevision` into a
/// generated source file. Nothing here shells out to `git` at run time: a
/// benchmark binary may be copied to another machine, and a revision read at
/// run time would then describe whatever repository happened to be nearby.
public enum BuildInfo {
    /// Package version. Bumped by hand; the git revision is the precise identity.
    public static let version = "0.1"

    /// Short git revision the binary was built from, with a `-dirty` suffix when
    /// the working tree had uncommitted changes. `"unknown"` when the build did
    /// not happen inside a git checkout.
    public static var gitRevision: String { minirunEmbeddedGitRevision }

    /// Whether the revision is precise enough to reproduce the build.
    public static var isRevisionKnown: Bool {
        gitRevision != "unknown" && !gitRevision.hasSuffix("-dirty")
    }

    public static var versionString: String { "\(version) (\(gitRevision))" }
}
