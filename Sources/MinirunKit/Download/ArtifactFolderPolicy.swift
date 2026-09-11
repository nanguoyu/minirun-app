import Darwin
import Foundation

/// What is already at the artifact folder a transfer wants to write.
///
/// Four states and no fifth, because every one of them decides something
/// different: nothing there is created, an empty folder and this artifact's own
/// folder are reused, and anything else is refused by name. There is no
/// "probably fine".
public enum ArtifactFolderState: Sendable, Equatable {
    /// No object exists at the path.
    case absent
    /// A directory holding nothing this policy has to preserve.
    case empty
    /// A directory that already belongs to this artifact — the resume case.
    case reusable
    /// Something else lives there. The reason is the sentence shown verbatim.
    case occupied(reason: String)

    public var allowsWriting: Bool {
        switch self {
        case .absent, .empty, .reusable: return true
        case .occupied: return false
        }
    }

    public var refusal: String? {
        if case .occupied(let reason) = self { return reason }
        return nil
    }
}

/// The folder a user picked, the folder the bytes go in, and what is there now.
///
/// The two are never the same folder unless the user pointed straight at the
/// artifact folder itself. `parent` is what the capacity verdict is taken
/// against — free space is a fact about the volume, and the child does not
/// exist yet — and `directory` is what the download manager is handed.
public struct DownloadDestinationLayout: Sendable, Equatable {
    /// The folder the operator confirmed.
    public let parent: URL
    /// `parent/<subfolderName>`; where the repository files are written.
    public let directory: URL
    /// The published repository name, exactly as the repository spells it.
    public let subfolderName: String
    public let state: ArtifactFolderState

    public init(
        parent: URL, directory: URL, subfolderName: String, state: ArtifactFolderState
    ) {
        self.parent = parent
        self.directory = directory
        self.subfolderName = subfolderName
        self.state = state
    }

    /// The sentence the picker shows before Continue.
    public var previewSentence: String {
        ArtifactFolderPolicy.previewPrefix + directory.path
    }

    /// The sentence the folder chooser puts in its own message.
    public var chooserMessage: String {
        "Choose where to keep this model. Minirun creates a folder named "
            + "\(subfolderName) inside it."
    }
}

/// A model's bytes live in a folder named after the repository that publishes
/// them, inside the folder the operator picked.
///
/// This exists because the opposite rule shipped: the confirmed folder *was*
/// the artifact folder, so picking the volume `/Volumes/K3NVME` sprayed
/// `index.json`, `LICENSE` and `layer00/` across the drive's root beside the
/// operator's own directories, and they had to be deleted by hand. The name is
/// the repository's own second path component, unchanged — no lowercasing, no
/// slug, no id mangling — because that is the name the Hub shows, the name the
/// README quotes and the name a person recognises on the drive.
///
/// Everything here is a pure function of a URL, a repository and the
/// filesystem, so it is decided once, in one place, and asserted by tests
/// rather than re-derived at each call site.
public enum ArtifactFolderPolicy {
    /// The lead-in of the picker's preview line. A constant so the view and its
    /// test agree on one spelling.
    public static let previewPrefix = "Will be saved to "

    /// Entries that are not content: the Finder writes them into any folder a
    /// window has been opened on, and refusing a drive because of one would be
    /// refusing it because it was looked at.
    static let ignoredEntries: Set<String> = [".DS_Store", ".localized"]

    /// The largest `index.json` this policy will read before deciding. A
    /// document larger than this is not one of the published indexes, and a
    /// classification must not become a several-gigabyte read.
    static let maximumIndexBytes: Int64 = 8 << 20

    // MARK: - The name

    /// `owner/Name-minirun` → `Name-minirun`. Nil when the id is not the
    /// canonical two-component Hugging Face spelling, because a name derived
    /// from an id this package refuses to put in a URL has no business being
    /// put in a path either.
    public static func subfolderName(forRepositoryID repoID: String) -> String? {
        guard HuggingFaceRepoRef.isCanonicalRepositoryID(repoID) else { return nil }
        let components = repoID.split(separator: "/")
        guard components.count == 2 else { return nil }
        return String(components[1])
    }

    public static func subfolderName(for repo: HuggingFaceRepoRef) -> String? {
        subfolderName(forRepositoryID: repo.repoID)
    }

    /// The sentence a refusal shows when the folder holds something else.
    public static func occupiedRefusal(_ url: URL) -> String {
        "\(url.standardizedFileURL.path) already exists and holds something else"
    }

    // MARK: - Where the bytes go

    /// Resolve the folder a picked location implies for `repository`.
    ///
    /// `picked` is treated as the parent, except when it already *is* the
    /// artifact folder — a job whose recorded destination is
    /// `…/Kimi-K3-minirun` re-enters this flow on *Continue with kept files*,
    /// and nesting a second `Kimi-K3-minirun` inside it would strand the files
    /// it was supposed to reuse.
    ///
    /// Returns nil only when the repository publishes no usable folder name.
    public static func resolve(
        picked: URL, repository: HuggingFaceRepoRef, model: ModelID? = nil,
        plannedPaths: [String]? = nil
    ) -> DownloadDestinationLayout? {
        guard let name = subfolderName(for: repository) else { return nil }
        let standardized = picked.standardizedFileURL
        let parent: URL
        let directory: URL
        if standardized.lastPathComponent == name {
            parent = standardized.deletingLastPathComponent()
            directory = standardized
        } else {
            parent = standardized
            directory = standardized.appendingPathComponent(name, isDirectory: true)
        }
        return DownloadDestinationLayout(
            parent: parent, directory: directory, subfolderName: name,
            state: classify(
                directory, repository: repository, model: model,
                plannedPaths: plannedPaths))
    }

    /// The same resolution for a plan that has already been fetched, which is
    /// the authority the controller uses at Continue time: a plan knows every
    /// path the transfer will write, so a half-finished folder is recognised
    /// exactly instead of by resemblance.
    public static func resolve(picked: URL, plan: DownloadPlan) -> DownloadDestinationLayout? {
        resolve(
            picked: picked, repository: plan.repo, model: plan.model,
            plannedPaths: plan.files.map(\.path))
    }

    // MARK: - What is there now

    public static func classify(_ url: URL, plan: DownloadPlan) -> ArtifactFolderState {
        classify(
            url, repository: plan.repo, model: plan.model,
            plannedPaths: plan.files.map(\.path))
    }

    /// - Parameters:
    ///   - plannedPaths: every repository-relative path the transfer will
    ///     write, when they are known. Nil before a plan has been fetched; the
    ///     folder is then recognised by its `index.json` alone, plus the
    ///     narrower rule that a folder of nothing but this app's own part files
    ///     is this app's own interrupted work.
    public static func classify(
        _ url: URL, repository: HuggingFaceRepoRef, model: ModelID? = nil,
        plannedPaths: [String]? = nil
    ) -> ArtifactFolderState {
        let directory = url.standardizedFileURL
        let path = directory.path

        var metadata = stat()
        guard lstat(path, &metadata) == 0 else {
            if errno == ENOENT { return .absent }
            return .occupied(
                reason: "\(path) could not be examined: \(String(cString: strerror(errno)))")
        }
        // `lstat`, so a symlink is a symlink. Following one would let a link
        // planted in the parent redirect a terabyte outside the granted tree.
        guard metadata.st_mode & S_IFMT == S_IFDIR else {
            if metadata.st_mode & S_IFMT == S_IFLNK {
                return .occupied(
                    reason: "\(path) already exists and is a symbolic link, not a folder")
            }
            return .occupied(reason: "\(path) already exists and is not a folder")
        }

        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: path) else {
            return .occupied(reason: "\(path) already exists and could not be read")
        }
        let content = entries.filter { !ignoredEntries.contains($0) }
        if content.isEmpty { return .empty }

        switch indexVerdict(in: directory, repository: repository, model: model) {
        case .matches: return .reusable
        case .names(let other):
            return .occupied(
                reason: "\(path) already exists and holds another artifact "
                    + "(its index.json names \(other))")
        case .absent, .unreadable: break
        }

        if let plannedPaths {
            let allowed = allowedTopLevelNames(for: plannedPaths)
            if content.allSatisfy({ allowed.contains($0) }) { return .reusable }
            return .occupied(reason: occupiedRefusal(directory))
        }
        // No plan yet. A folder of nothing but part files carries no verified
        // file to identify it, and nothing but this app writes that suffix.
        if content.allSatisfy({ $0.hasSuffix(DownloadManager.partSuffix) }) { return .reusable }
        return .occupied(reason: occupiedRefusal(directory))
    }

    // MARK: - Internals

    enum IndexVerdict: Equatable {
        case absent
        case unreadable
        case matches
        case names(String)
    }

    static func indexVerdict(
        in directory: URL, repository: HuggingFaceRepoRef, model: ModelID?
    ) -> IndexVerdict {
        let indexURL = directory.appendingPathComponent("index.json", isDirectory: false)
        var metadata = stat()
        guard lstat(indexURL.path, &metadata) == 0 else { return .absent }
        guard metadata.st_mode & S_IFMT == S_IFREG,
            metadata.st_size > 0, metadata.st_size <= maximumIndexBytes,
            let data = try? Data(contentsOf: indexURL)
        else { return .unreadable }

        let identity = ArtifactIndexIdentity.parse(data)
        guard !identity.repositories.isEmpty else { return .unreadable }
        let wanted = repository.repoID.lowercased()
        for source in identity.repositories {
            let named = source.repoID.lowercased()
            // The publishing repository, when the document names it at all.
            if named == wanted { return .matches }
            // Otherwise the upstream the repack was made from: a directory
            // fetched from `nanguoyu/Kimi-K3-minirun` carries an index that
            // says `moonshotai/Kimi-K3` and nothing else.
            if let model, ArtifactIdentityMatcher.upstreamRepositories[named] == model {
                return .matches
            }
        }
        return .names(identity.repositories.map(\.repoID).joined(separator: ", "))
    }

    /// Top-level names a planned transfer may legitimately have created:
    /// each planned path's first component, plus the part file of a root file.
    /// A nested path's part file lives under its own directory, which is
    /// already covered by the first component.
    static func allowedTopLevelNames(for plannedPaths: [String]) -> Set<String> {
        var allowed = Set<String>()
        for planned in plannedPaths {
            let components = planned.split(separator: "/")
            guard let first = components.first.map(String.init) else { continue }
            allowed.insert(first)
            if components.count == 1 {
                allowed.insert(first + DownloadManager.partSuffix)
            }
        }
        return allowed
    }
}
