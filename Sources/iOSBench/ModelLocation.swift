import Foundation
import StorageCore

/// Which artifact layout a directory holds, if any.
///
/// Two manifest shapes exist in this repository and the resolver has to tell
/// them apart without linking `ModelAdapters` (spec §7.1 — the phone-side
/// picker sits beside the storage layer, not above the model adapter):
///
/// - `containers`: the tiny-K3 expert container set written by
///   `Tools/k3_reference/convert_tiny_experts.py` and read by
///   `RoutedExpertContainers`. Each record names a `file` in the same
///   directory.
/// - `units`: the flagship artifact written by
///   `Tools/k3_flagship/build_full_artifact.py`, which groups files under a
///   per-unit `directory` and carries a `complete` flag because it is built
///   incrementally across sessions.
///
/// Anything else is reported as ``unrecognized`` rather than guessed at.
public enum ArtifactLayoutKind: String, Equatable, Codable {
    case tinyContainerSet
    case fullArtifactBundle
    case unrecognized
}

/// A file the manifest promised, and what is actually on disk.
public struct ArtifactFileStatus: Equatable, Codable {
    /// Path relative to the scanned directory, as the manifest spells it.
    public let relativePath: String
    public let exists: Bool
    /// Size on disk, or nil when the file is missing.
    public let sizeBytes: UInt64?

    public init(relativePath: String, exists: Bool, sizeBytes: UInt64?) {
        self.relativePath = relativePath
        self.exists = exists
        self.sizeBytes = sizeBytes
    }
}

/// What a picked directory turned out to be.
///
/// This is the report the on-device harness shows before it reads a single
/// weight byte. Spec §5.8 wants the unsupported case named, so a directory
/// that is nearly right (manifest present, three files missing) has to be
/// distinguishable from one that is not an artifact at all — hence
/// ``missingFiles`` rather than a bare bool.
public struct ModelLocationReport: Equatable, Codable {
    public let directoryPath: String
    /// Nil when the directory holds no `manifest.json`.
    public let layout: ArtifactLayoutKind?
    /// Set when a manifest was present but could not be read or understood.
    /// The layout is then ``ArtifactLayoutKind/unrecognized``.
    public let manifestProblem: String?
    public let files: [ArtifactFileStatus]
    /// Sum of the sizes of the files that exist.
    public let presentBytes: UInt64
    /// What the flagship builder's own `complete` flag says. Nil for layouts
    /// that do not carry one — absence is not incompleteness.
    public let builderReportsComplete: Bool?
    /// Volume facts, straight from ``StorageCore/StorageInfo``. Not
    /// re-derived here: §18.3's storage object is already exactly this, and
    /// two implementations of "how much space is left" would eventually
    /// disagree.
    public let storage: StorageInfo

    /// Every file the manifest named is on disk.
    public var isComplete: Bool {
        layout != nil && layout != .unrecognized && !files.isEmpty
            && files.allSatisfy(\.exists)
    }

    public var missingFiles: [String] {
        files.filter { !$0.exists }.map(\.relativePath)
    }

    /// Free space excluding what the system would only surrender under
    /// pressure. `volumeAvailableCapacityForImportantUsage` can be far larger
    /// on APFS because it counts purgeable space, and a streaming run that
    /// depends on the system evicting someone else's cache is not a
    /// measurement anyone can reproduce (spec §17.4).
    public var freeBytes: Int64? { storage.volumeAvailableBytes }
}

/// Answers the two questions the phone has to ask about a user-picked
/// directory before anything else can happen: is the artifact here, and does
/// this volume have room.
public enum ModelLocationResolver {

    /// Describe `directory`. Never throws — a directory that cannot be read is
    /// itself the answer, and the caller shows it to the user.
    ///
    /// The caller is responsible for holding a security scope over
    /// `directory` across this call; see
    /// ``SecurityScopedLocationStore/withAccess(to:_:)``.
    public static func describe(directory: URL) -> ModelLocationReport {
        let storage = StorageInfo.describing(path: directory.path)
        let manifestURL = directory.appendingPathComponent("manifest.json")

        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return ModelLocationReport(
                directoryPath: directory.path, layout: nil, manifestProblem: nil,
                files: [], presentBytes: 0, builderReportsComplete: nil, storage: storage)
        }

        let root: [String: Any]
        do {
            let data = try Data(contentsOf: manifestURL)
            guard let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw CocoaError(.propertyListReadCorrupt)
            }
            root = parsed
        } catch {
            return ModelLocationReport(
                directoryPath: directory.path, layout: .unrecognized,
                manifestProblem: "manifest.json could not be read: \(error)",
                files: [], presentBytes: 0, builderReportsComplete: nil, storage: storage)
        }

        let named: [String]
        let kind: ArtifactLayoutKind
        var complete: Bool?
        if let containers = root["containers"] as? [[String: Any]] {
            kind = .tinyContainerSet
            named = containers.compactMap { $0["file"] as? String }
        } else if let units = root["units"] as? [[String: Any]] {
            kind = .fullArtifactBundle
            complete = root["complete"] as? Bool
            named = units.flatMap { unit -> [String] in
                let subdirectory = unit["directory"] as? String
                // `files` is a map from a role to a filename; the builder
                // fills it in as each unit verifies, so a pending unit
                // legitimately has none yet.
                let files = (unit["files"] as? [String: Any])?.values.compactMap { $0 as? String }
                return (files ?? []).map { name in
                    subdirectory.map { "\($0)/\(name)" } ?? name
                }
            }
        } else {
            return ModelLocationReport(
                directoryPath: directory.path, layout: .unrecognized,
                manifestProblem: "manifest.json has neither a 'containers' nor a 'units' list",
                files: [], presentBytes: 0, builderReportsComplete: nil, storage: storage)
        }

        var files: [ArtifactFileStatus] = []
        var presentBytes: UInt64 = 0
        for relative in named {
            let url = directory.appendingPathComponent(relative)
            let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
            let size = (attributes?[.size] as? NSNumber)?.uint64Value
            if let size { presentBytes += size }
            files.append(
                ArtifactFileStatus(
                    relativePath: relative, exists: attributes != nil, sizeBytes: size))
        }

        let problem = files.isEmpty ? "manifest.json names no files" : nil
        return ModelLocationReport(
            directoryPath: directory.path, layout: kind, manifestProblem: problem,
            files: files, presentBytes: presentBytes, builderReportsComplete: complete,
            storage: storage)
    }
}
