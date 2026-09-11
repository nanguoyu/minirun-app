import Darwin
import Foundation
import StorageCore

/// What a stopped transfer actually left on the drive, looked at now.
///
/// A cancelled or interrupted job carries the progress snapshot it had when it
/// stopped, and that snapshot is a memory: it says "74.2 MB of 517 GB kept"
/// whether or not anybody has since deleted the directory. The card that reads
/// it therefore states a fact about a drive it never re-checked, which is the
/// one thing this product refuses to do everywhere else.
///
/// This is the re-check, and it is deliberately the *cheap* one: sizes only,
/// exactly the rule ``DownloadManager`` already applies before a transfer
/// starts. No digest is computed and no file body is read, so reconciling a
/// 517 GB plan costs two `lstat` calls per planned file and nothing else. A
/// file present at its planned size is *kept*; a `.minirun-part` file is
/// *partial*; anything else is absent. Whether those kept bytes are the right
/// bytes remains a question only verification answers.
public struct KeptFiles: Equatable, Sendable {

    /// How the destination directory looked when it was examined. Absence and
    /// unreachability are different answers and are never collapsed: a drive in
    /// a drawer has not lost anything.
    public enum Destination: Equatable, Sendable {
        /// The directory is there and was read.
        case present
        /// The path names a volume that is not mounted. Nothing can be said
        /// about the files, and nothing about them has been lost.
        case volumeNotMounted(volume: String)
        /// The volume is there; this directory is not.
        case missing
        /// The directory exists but could not be examined, for the named
        /// reason.
        case unreadable(reason: String)
    }

    public let destination: Destination
    public let plannedFileCount: Int
    public let plannedBytes: UInt64
    /// Planned files present at exactly their planned size.
    public let completeFileCount: Int
    public let completeBytes: UInt64
    /// `.minirun-part` files for planned files, and the bytes in them.
    public let partialFileCount: Int
    public let partialBytes: UInt64

    public init(
        destination: Destination, plannedFileCount: Int, plannedBytes: UInt64,
        completeFileCount: Int, completeBytes: UInt64,
        partialFileCount: Int, partialBytes: UInt64
    ) {
        self.destination = destination
        self.plannedFileCount = plannedFileCount
        self.plannedBytes = plannedBytes
        self.completeFileCount = completeFileCount
        self.completeBytes = completeBytes
        self.partialFileCount = partialFileCount
        self.partialBytes = partialBytes
    }

    public var fileCount: Int { completeFileCount + partialFileCount }

    /// Saturating, like every other byte total in this package: an absurd
    /// on-disk size may not wrap into a small, believable one.
    public var bytes: UInt64 {
        let (sum, overflow) = completeBytes.addingReportingOverflow(partialBytes)
        return overflow ? .max : sum
    }

    public var isEmpty: Bool { fileCount == 0 }

    /// Whether the directory itself was actually looked at. False means the
    /// counts below are zero for want of a look, not for want of files.
    public var wasExamined: Bool {
        switch destination {
        case .present, .missing: return true
        case .volumeNotMounted, .unreadable: return false
        }
    }

    /// A report for a destination that could not be examined, or that holds
    /// nothing at all.
    public static func nothing(destination: Destination, of plan: DownloadPlan) -> KeptFiles {
        KeptFiles(
            destination: destination, plannedFileCount: plan.files.count,
            plannedBytes: plan.totalBytes, completeFileCount: 0, completeBytes: 0,
            partialFileCount: 0, partialBytes: 0)
    }

    /// Size-only reconciliation of one plan against one directory.
    public static func measuring(_ plan: DownloadPlan, in directory: URL) -> KeptFiles {
        let root = directory.standardizedFileURL
        let destination = destinationStatus(of: root)
        guard case .present = destination else {
            return nothing(destination: destination, of: plan)
        }

        var completeCount = 0
        var completeBytes: UInt64 = 0
        var partialCount = 0
        var partialBytes: UInt64 = 0
        for file in plan.files {
            // A plan path that is not repo-relative was never written by this
            // package and is not looked for outside the destination.
            guard HuggingFaceTreeClient.isSafeRepositoryPath(file.path) else { continue }
            if let size = regularFileSize(root.appendingPathComponent(file.path)),
                size == file.sizeBytes
            {
                completeCount += 1
                completeBytes = saturatingSum(completeBytes, size)
                continue
            }
            let part = root.appendingPathComponent(file.path + DownloadManager.partSuffix)
            if let size = regularFileSize(part), size > 0 {
                partialCount += 1
                partialBytes = saturatingSum(partialBytes, size)
            }
        }

        return KeptFiles(
            destination: .present, plannedFileCount: plan.files.count,
            plannedBytes: plan.totalBytes, completeFileCount: completeCount,
            completeBytes: completeBytes, partialFileCount: partialCount,
            partialBytes: partialBytes)
    }

    /// Whether the destination directory can be read at all, and if not, why.
    public static func destinationStatus(of directory: URL) -> Destination {
        let info = StorageInfo.describing(path: directory.standardizedFileURL.path)
        if info.pathExists {
            guard info.pathIsDirectory else {
                return .unreadable(reason: "the destination path is not a directory")
            }
            guard info.pathIsReadable else {
                return .unreadable(reason: "the destination directory is not readable")
            }
            return .present
        }
        if let volume = unmountedVolumeName(for: directory) {
            return .volumeNotMounted(volume: volume)
        }
        return .missing
    }

    /// The volume a missing destination belongs to, when that volume is itself
    /// absent from the mount table.
    ///
    /// Only `/Volumes/<name>/…` is answered, because that is the only spelling
    /// on these platforms whose second component *is* a mount point. Anything
    /// else stays unanswered rather than guessing which ancestor was a drive.
    static func unmountedVolumeName(for directory: URL) -> String? {
        let components = directory.standardizedFileURL.pathComponents
        guard components.count >= 3, components[0] == "/", components[1] == "Volumes" else {
            return nil
        }
        let name = components[2]
        let mountPath = "/Volumes/\(name)"
        let mounted =
            FileManager.default.mountedVolumeURLs(
                includingResourceValuesForKeys: nil, options: [])?
            .map { $0.standardizedFileURL.path } ?? []
        return mounted.contains(mountPath) ? nil : name
    }

    /// `lstat`, and a size only for a regular file: a directory has an
    /// `st_size` too, and counting one as a downloaded file would be the same
    /// class of lie this type exists to remove.
    static func regularFileSize(_ url: URL) -> UInt64? {
        var metadata = stat()
        guard lstat(url.standardizedFileURL.path, &metadata) == 0 else { return nil }
        guard metadata.st_mode & S_IFMT == S_IFREG, metadata.st_size >= 0 else { return nil }
        return UInt64(metadata.st_size)
    }

    private static func saturatingSum(_ total: UInt64, _ addition: UInt64) -> UInt64 {
        let (sum, overflow) = total.addingReportingOverflow(addition)
        return overflow ? .max : sum
    }
}
