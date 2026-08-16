import Foundation

/// The volume a measured path lives on.
///
/// Populates the `storage` object of the benchmark schema (spec §18.3). Spec
/// §18.2 requires results to be comparable across internal SSD, external NVMe,
/// and iPhone storage, so a result that does not say which volume it measured
/// is not interpretable — hence volume name, filesystem type, and the
/// removable/internal flags travel with every run.
public struct StorageInfo: Codable, Equatable {
    /// Path as given on the command line.
    public let requestedPath: String
    /// Absolute, symlink-resolved path.
    public let resolvedPath: String
    /// Whether that path exists. When it does not, volume fields below describe
    /// the nearest existing ancestor directory — which is what a run that is
    /// about to create the file needs to know.
    public let pathExists: Bool
    /// Whether the path exists and is a directory.
    public let pathIsDirectory: Bool
    public let pathIsReadable: Bool
    public let pathIsWritable: Bool
    /// Size of the file at the path, when it is a regular file.
    public let fileSizeBytes: UInt64?
    /// Volume the path resolves onto. Null when the path could not be probed.
    public let volumeName: String?
    /// Mount point, e.g. `/` or `/Volumes/NVMe`.
    public let volumeMountPath: String?
    /// Filesystem type, e.g. `apfs`, `exfat`.
    public let filesystemType: String?
    public let volumeTotalBytes: Int64?
    /// Free space as reported for ordinary use.
    public let volumeAvailableBytes: Int64?
    /// Free space including space the system would purge if pressed. On APFS
    /// this can be much larger than the previous field; a benchmark file that
    /// only fits into purgeable space is not a safe test fixture.
    public let volumeAvailableBytesForImportantUsage: Int64?
    public let volumeIsInternal: Bool?
    public let volumeIsRemovable: Bool?
    public let volumeIsEjectable: Bool?
    /// Whether the volume is a read-only mount.
    public let volumeIsReadOnly: Bool?

    private static let volumeKeys: Set<URLResourceKey> = [
        .volumeNameKey,
        .volumeURLKey,
        .volumeTypeNameKey,
        .volumeTotalCapacityKey,
        .volumeAvailableCapacityKey,
        .volumeAvailableCapacityForImportantUsageKey,
        .volumeIsInternalKey,
        .volumeIsRemovableKey,
        .volumeIsEjectableKey,
        .volumeIsReadOnlyKey,
    ]

    /// Describes `path`. Never throws: a path that cannot be probed yields
    /// nulls, because an environment report that fails outright is less useful
    /// than a partial one.
    public static func describing(path: String) -> StorageInfo {
        let fileManager = FileManager.default
        let requested = path.isEmpty ? fileManager.currentDirectoryPath : path
        let url = URL(fileURLWithPath: requested).standardizedFileURL
        let resolved = url.resolvingSymlinksInPath()

        var isDirectory: ObjCBool = false
        let exists = fileManager.fileExists(atPath: resolved.path, isDirectory: &isDirectory)

        var fileSize: UInt64?
        if exists, !isDirectory.boolValue,
            let attributes = try? fileManager.attributesOfItem(atPath: resolved.path),
            let size = attributes[.size] as? NSNumber
        {
            fileSize = size.uint64Value
        }

        // Volume metadata comes from the nearest existing ancestor when the
        // path itself does not exist yet, so `minirun env --path <file to be
        // written>` still reports the volume it would be written to.
        let values = resourceValues(forNearestExistingAncestorOf: resolved)

        return StorageInfo(
            requestedPath: requested,
            resolvedPath: resolved.path,
            pathExists: exists,
            pathIsDirectory: exists && isDirectory.boolValue,
            pathIsReadable: fileManager.isReadableFile(atPath: resolved.path),
            pathIsWritable: fileManager.isWritableFile(atPath: resolved.path),
            fileSizeBytes: fileSize,
            volumeName: values?.volumeName,
            volumeMountPath: values?.volume?.path,
            filesystemType: values?.volumeTypeName,
            volumeTotalBytes: values?.volumeTotalCapacity.map(Int64.init),
            volumeAvailableBytes: values?.volumeAvailableCapacity.map(Int64.init),
            volumeAvailableBytesForImportantUsage: values?
                .volumeAvailableCapacityForImportantUsage,
            volumeIsInternal: values?.volumeIsInternal,
            volumeIsRemovable: values?.volumeIsRemovable,
            volumeIsEjectable: values?.volumeIsEjectable,
            volumeIsReadOnly: values?.volumeIsReadOnly
        )
    }

    private static func resourceValues(
        forNearestExistingAncestorOf url: URL
    ) -> URLResourceValues? {
        var candidate = url
        while true {
            if let values = try? candidate.resourceValues(forKeys: volumeKeys) {
                return values
            }
            let parent = candidate.deletingLastPathComponent().standardizedFileURL
            if parent.path == candidate.path { return nil }
            candidate = parent
        }
    }

    private enum CodingKeys: String, CodingKey {
        case requestedPath, resolvedPath, pathExists, pathIsDirectory
        case pathIsReadable, pathIsWritable, fileSizeBytes
        case volumeName, volumeMountPath, filesystemType
        case volumeTotalBytes, volumeAvailableBytes
        case volumeAvailableBytesForImportantUsage
        case volumeIsInternal, volumeIsRemovable, volumeIsEjectable, volumeIsReadOnly
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(requestedPath, forKey: .requestedPath)
        try container.encode(resolvedPath, forKey: .resolvedPath)
        try container.encode(pathExists, forKey: .pathExists)
        try container.encode(pathIsDirectory, forKey: .pathIsDirectory)
        try container.encode(pathIsReadable, forKey: .pathIsReadable)
        try container.encode(pathIsWritable, forKey: .pathIsWritable)
        try container.encodeAlways(fileSizeBytes, forKey: .fileSizeBytes)
        try container.encodeAlways(volumeName, forKey: .volumeName)
        try container.encodeAlways(volumeMountPath, forKey: .volumeMountPath)
        try container.encodeAlways(filesystemType, forKey: .filesystemType)
        try container.encodeAlways(volumeTotalBytes, forKey: .volumeTotalBytes)
        try container.encodeAlways(volumeAvailableBytes, forKey: .volumeAvailableBytes)
        try container.encodeAlways(
            volumeAvailableBytesForImportantUsage,
            forKey: .volumeAvailableBytesForImportantUsage)
        try container.encodeAlways(volumeIsInternal, forKey: .volumeIsInternal)
        try container.encodeAlways(volumeIsRemovable, forKey: .volumeIsRemovable)
        try container.encodeAlways(volumeIsEjectable, forKey: .volumeIsEjectable)
        try container.encodeAlways(volumeIsReadOnly, forKey: .volumeIsReadOnly)
    }
}
