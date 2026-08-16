import Foundation
import MinirunKit
import ModelAdapters

/// A private, per-run metadata view of a fully verified K3 artifact.
///
/// The external volume is never reopened by path. Published manifests are
/// copied from rooted descriptors into private scratch, while every payload
/// reference in the translated manifests remains a repository-relative key
/// serviced by ``fileAccess``. This keeps descriptor use bounded: deterministic
/// streams and globals keep the descriptors they actively read, while expert
/// containers continue through the existing six-reader LRU.
final class K3RootedArtifactMirror {
    let reference: ArtifactReference
    let fileAccess: ModelFileAccess

    private let authority: ArtifactRuntimeAuthority
    private let directory: URL

    init(reference: ArtifactReference, workingDirectory: URL) throws {
        guard let authority = reference.runtimeAuthority else {
            throw RunError.artifactNotReady(
                "the live K3 runner requires complete rooted verification authority")
        }
        try authority.validateCurrentBinding()

        let directory = workingDirectory.appendingPathComponent(
            "k3-rooted-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])

        let runtimePaths = authority.verifiedRepositoryPaths.filter(Self.isRuntimePath)
        let available = Set(runtimePaths)
        let required = [
            "global/embed_tokens.bin", "global/lm_head.bin", "global/final.bin",
            "layer00/manifest.json", "layer00/layer00-deterministic.bin",
        ]
        guard required.allSatisfy(available.contains) else {
            let missing = required.filter { !available.contains($0) }.joined(separator: ", ")
            try? FileManager.default.removeItem(at: directory)
            throw RunError.artifactNotReady(
                "the verified K3 artifact is missing required runtime files: \(missing)")
        }

        do {
            for path in runtimePaths {
                let destination = directory.appendingPathComponent(path)
                try FileManager.default.createDirectory(
                    at: destination.deletingLastPathComponent(),
                    withIntermediateDirectories: true)
                if path == "config.json" || path.hasSuffix("/manifest.json") {
                    let file = try authority.openFile(path)
                    let data = try file.readAll(maximumBytes: 64 << 20)
                    try data.write(to: destination, options: .atomic)
                } else if !FileManager.default.createFile(
                    atPath: destination.path, contents: Data())
                {
                    throw RunError.artifactNotReady(
                        "could not prepare the private K3 metadata view for \(path)")
                }
            }
            self.fileAccess = ModelFileAccess { path in
                try authority.openFileDescriptor(path)
            }
            self.reference = ArtifactReference(
                root: directory, runtimeAuthority: authority)
            self.authority = authority
            self.directory = directory
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    deinit { try? FileManager.default.removeItem(at: directory) }

    func validateCurrentBinding() throws {
        try authority.validateAllFilesCurrent()
    }

    private static func isRuntimePath(_ path: String) -> Bool {
        if path == "config.json" { return true }
        if path.hasPrefix("global/") { return true }
        guard path.count > 8, path.hasPrefix("layer") else { return false }
        let prefix = path.prefix(7)
        guard prefix.dropFirst(5).allSatisfy(\.isNumber),
            path[path.index(path.startIndex, offsetBy: 7)] == "/"
        else { return false }
        return true
    }
}
