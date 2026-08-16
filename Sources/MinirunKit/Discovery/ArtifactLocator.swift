import Foundation
import StorageCore

/// Finds minirun artifacts in the places the operator registered.
///
/// The whole module exists to answer one question from the disk rather than
/// from a registry: **is this model installed?** A registry records what was
/// once downloaded. A volume records what is there now — and the two disagree
/// the moment a drive is unplugged, a directory is moved, or an artifact is
/// copied in by hand rather than fetched by this app. All three happen, and the
/// third is how the artifacts on this machine's drive got there.
///
/// What a scan does: read `index.json`, sum file sizes, and count files. For a
/// remembered positive verification it also opens all evidenced files without following
/// symlinks and compares `fstat` identity; it never reads their contents. This
/// remains bounded metadata work, so scanning a 1.56 TB artifact costs a
/// directory walk rather than 1.56 TB of I/O. What a scan never does: digest a
/// payload or write anything at all. Verification is a separate, operator-
/// triggered act (``ArtifactVerification``), because a screen appearing is not
/// consent to hash a terabyte.
public struct ArtifactLocator: Sendable {
    /// The file whose presence makes a directory an artifact root.
    public static let indexFilename = "index.json"

    /// Directory names that are bookkeeping rather than artifact.
    ///
    /// `.cache` is the Hugging Face downloader's tree and part-file store. It
    /// sits *inside* the artifact root and holds hundreds of files; counting it
    /// would put the file count above the catalog's expectation for a directory
    /// that is in fact incomplete, which is the exact wrong direction to be
    /// wrong in.
    public static let ignoredDirectories: Set<String> = [".cache"]

    /// Files that belong to the operating system rather than to anybody.
    public static let ignoredFiles: Set<String> = [".DS_Store"]

    /// How far below a registered location to look for artifact roots.
    ///
    /// Two, because that is what the real layout needs: a location is a volume
    /// like `/Volumes/K3NVME`, and the artifacts are directories directly
    /// inside it. One extra level is allowed for an operator who filed them
    /// under a folder. Deeper is refused rather than made configurable upwards:
    /// an unbounded walk of a volume root is a scan that can take minutes and
    /// find somebody's photo library.
    public var maximumDepth: Int

    private let catalog: ModelCatalogSnapshot
    private let matcher: ArtifactIdentityMatcher
    private let ledger: any ArtifactVerificationLedger
    private let fileManager: FileManager

    public init(
        catalog: ModelCatalogSnapshot,
        verificationLedger: any ArtifactVerificationLedger = InMemoryVerificationLedger(),
        maximumDepth: Int = 2,
        fileManager: FileManager = .default
    ) {
        self.catalog = catalog
        self.matcher = ArtifactIdentityMatcher(catalog: catalog)
        self.ledger = verificationLedger
        self.maximumDepth = maximumDepth
        self.fileManager = fileManager
    }

    // MARK: - Scanning

    /// Scan one registered location.
    ///
    /// Never throws. A location that is not mounted, not readable, or not there
    /// at all is a `LocationScan` that says so — an operator who registered a
    /// removable drive is owed the row even while the drive is in a drawer.
    public func scan(_ url: URL, storageKey: StorageKey? = nil) -> LocationScan {
        let now = Date()
        let root = url.standardizedFileURL
        let info = StorageInfo.describing(path: root.path)
        let displayName = info.volumeName ?? root.lastPathComponent

        guard info.pathExists, info.pathIsDirectory, info.pathIsReadable else {
            return LocationScan(
                rootPath: root.path, displayName: displayName, isMounted: false,
                storageKey: storageKey, artifacts: [], unreadable: [:], scannedAt: now)
        }

        var unreadable: [String: String] = [:]
        var roots: [URL] = []
        collectArtifactRoots(under: root, depth: 0, into: &roots, unreadable: &unreadable)

        let artifacts =
            roots
            .map { describe(artifactAt: $0, locationPath: root.path, at: now, unreadable: &unreadable) }
            .sorted { $0.rootPath < $1.rootPath }

        return LocationScan(
            rootPath: root.path, displayName: displayName, isMounted: true,
            storageKey: storageKey, artifacts: artifacts, unreadable: unreadable, scannedAt: now)
    }

    public func scan(_ urls: [URL]) -> DiscoveryReport {
        DiscoveryReport(locations: urls.map { scan($0) }, scannedAt: Date())
    }

    /// Scan every location a storage manager remembers.
    ///
    /// Each bookmark is resolved into a live scope and released as soon as the
    /// walk is done: the grant is needed for the walk and for nothing after it,
    /// and a grant left open on a removable volume is what stops an ejection
    /// from completing.
    public func scanRegisteredLocations(of storage: any StorageManaging) -> DiscoveryReport {
        var scans: [LocationScan] = []
        for key in storage.knownLocations() {
            if let scope = try? storage.resolve(key) {
                scans.append(scan(scope.url, storageKey: key))
                scope.release()
            } else if let bookmark = storage.bookmark(key) {
                // The bookmark exists and would not resolve — the volume is
                // away. Report the remembered path rather than dropping the
                // location, so the operator sees why the model went quiet.
                scans.append(scan(URL(fileURLWithPath: bookmark.recordedPath), storageKey: key))
            }
        }
        return DiscoveryReport(locations: scans, scannedAt: Date())
    }

    // MARK: - Walking

    private func collectArtifactRoots(
        under directory: URL, depth: Int, into roots: inout [URL],
        unreadable: inout [String: String]
    ) {
        if fileManager.fileExists(atPath: directory.appendingPathComponent(Self.indexFilename).path) {
            // An artifact root is a leaf for this walk: a nested `index.json`
            // inside an artifact belongs to the artifact, not to a second one.
            roots.append(directory)
            return
        }
        guard depth < maximumDepth else { return }

        let children: [URL]
        do {
            children = try fileManager.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsPackageDescendants])
        } catch {
            unreadable[directory.path] = String(describing: error)
            return
        }
        for child in children.sorted(by: { $0.path < $1.path }) {
            let name = child.lastPathComponent
            guard !Self.ignoredDirectories.contains(name), !name.hasPrefix(".") else { continue }
            let isDirectory =
                (try? child.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
            guard isDirectory else { continue }
            collectArtifactRoots(under: child, depth: depth + 1, into: &roots, unreadable: &unreadable)
        }
    }

    private func describe(
        artifactAt root: URL, locationPath: String, at now: Date,
        unreadable: inout [String: String]
    ) -> DiscoveredArtifact {
        let indexURL = root.appendingPathComponent(Self.indexFilename)
        let identity: ArtifactIndexIdentity
        if let data = try? Data(contentsOf: indexURL) {
            identity = ArtifactIndexIdentity.parse(data)
        } else {
            unreadable[indexURL.path] = "index.json could not be read"
            identity = .unreadable
        }

        let model = matcher.match(identity)
        let descriptor = model.flatMap { catalog.descriptor($0) }
        let tally = tally(under: root, unreadable: &unreadable)
        let record = ledger.record(
            matching: ArtifactVerificationLookup(
                rootPath: root.path, model: model, repository: descriptor?.source.repo,
                index: identity))

        return DiscoveredArtifact(
            rootPath: root.path,
            locationPath: locationPath,
            index: identity,
            model: model,
            displayName: descriptor?.displayName ?? Self.unidentifiedName(identity, root: root),
            bytesOnDisk: tally.bytes,
            fileCount: tally.files,
            expectedFileCount: descriptor?.totalFileCount,
            expectedBytes: descriptor?.totalBytes,
            verification: record?.state ?? .unverified,
            verifiedAt: record?.state == .unverified ? nil : record?.checkedAt,
            scannedAt: now,
            metadataOverflowed: tally.overflowed)
    }

    /// A name for an artifact the catalog does not know.
    ///
    /// The repository the index names, when it names one, because that is the
    /// most specific true thing available. The directory name otherwise —
    /// never a generic "Unknown model", which tells an operator staring at 60 GB
    /// nothing they did not already know.
    private static func unidentifiedName(_ identity: ArtifactIndexIdentity, root: URL) -> String {
        if let first = identity.repositories.first { return first.repoID }
        return root.lastPathComponent
    }

    // MARK: - Counting

    struct Tally {
        var bytes: UInt64 = 0
        var files: Int = 0
        private(set) var overflowed = false

        /// Count one regular file without allowing adversarial filesystem
        /// metadata to wrap the artifact total back to a plausible small size.
        /// The total saturates for reporting, while `overflowed` makes the scan
        /// explicitly unusable as a completeness claim.
        @discardableResult
        mutating func add(fileSize: UInt64) -> Bool {
            let (nextBytes, bytesOverflowed) = bytes.addingReportingOverflow(fileSize)
            if bytesOverflowed {
                bytes = .max
                overflowed = true
            } else {
                bytes = nextBytes
            }

            if files == .max {
                overflowed = true
            } else {
                files += 1
            }
            return overflowed
        }
    }

    /// Sum of file sizes and file count under a root.
    ///
    /// `.fileSizeKey` off the enumerator, which is a `stat` per entry and never
    /// opens anything. Allocated size is deliberately not used: the question is
    /// how many bytes the artifact *is*, which a sparse or compressed file would
    /// answer differently from how many blocks it occupies, and the catalog's
    /// expectation is a byte count.
    func tally(under root: URL, unreadable: inout [String: String]) -> Tally {
        var tally = Tally()
        guard
            let enumerator = fileManager.enumerator(
                at: root, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
                options: [.skipsPackageDescendants])
        else {
            unreadable[root.path] = "the directory could not be enumerated"
            return tally
        }
        for case let url as URL in enumerator {
            let name = url.lastPathComponent
            if Self.ignoredDirectories.contains(name) {
                enumerator.skipDescendants()
                continue
            }
            if Self.ignoredFiles.contains(name) { continue }
            guard let values = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]),
                values.isRegularFile == true
            else { continue }
            guard let fileSize = values.fileSize, fileSize >= 0 else {
                unreadable[url.path] = "the file reported an invalid size"
                continue
            }
            if tally.add(fileSize: UInt64(fileSize)) {
                unreadable[root.path] =
                    "the artifact's file count or byte total exceeds what this build can represent"
            }
        }
        return tally
    }
}
