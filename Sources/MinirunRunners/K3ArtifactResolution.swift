import BenchScenarios
import Foundation
import MinirunKit
import ModelAdapters

/// Where a K3 run's inputs are, resolved from an ``ArtifactReference`` before
/// any payload byte is read.
///
/// Three shapes reach this runner and they are told apart by what is on disk,
/// never by a flag the caller passes:
///
/// | on disk | meaning |
/// | --- | --- |
/// | `layerNN-manifest.json` in the root, in `translated/`, or in `auxiliaryRoots["translated"]` | already translated; open it as it is |
/// | `layerNN/` directories, each with a `manifest.json` | the published artifact; translate first |
/// | neither | refused — `RunError.artifactLayoutMismatch` |
///
/// Files win over metadata (spec §5.10): a caller that names a translated
/// directory which does not contain manifests is refused rather than believed.
struct K3ArtifactResolution {

    enum Shape {
        /// Flat `layerNN-manifest.json`, ready for ``ArtifactWeightSource``.
        case translated(URL)
        /// `layerNN/` published layout; needs
        /// ``PublishedArtifactLayout/translate(artifact:into:config:blobReference:)``.
        case published(URL)
    }

    let shape: Shape
    let globals: URL
    let globalFiles: ArtifactWeightSource.GlobalFiles
    let config: TinyK3Config
    /// Where the config came from, for the run's own record. The published
    /// artifact carries none — it is a repack of weights, not of a checkpoint —
    /// so the bundled flagship config is the only statement of the model's
    /// shape a phone has, and which one was used is not a detail.
    let configOrigin: String

    static func resolve(
        _ reference: ArtifactReference, fileAccess: ModelFileAccess? = nil
    ) throws -> K3ArtifactResolution {
        let root = reference.root
        guard isDirectory(root) else {
            throw RunError.artifactNotReady(
                "\(root.path) is not a directory — is the drive mounted and powered?")
        }
        if let scope = reference.scope, scope.isReleased {
            throw RunError.scopeRefused(path: root.path)
        }

        let globals = root.appendingPathComponent("global")
        guard isDirectory(globals) else {
            throw RunError.artifactLayoutMismatch(
                expected: .k3FlagshipLayerStreams, atPath: root.path)
        }

        let (config, origin) = try configuration(for: reference)
        let globalFiles = ArtifactWeightSource.GlobalFiles(
            embedTokens: fileAccess == nil
                ? globals.appendingPathComponent("embed_tokens.bin").path
                : "global/embed_tokens.bin",
            lmHead: fileAccess == nil
                ? globals.appendingPathComponent("lm_head.bin").path
                : "global/lm_head.bin",
            final: fileAccess == nil
                ? globals.appendingPathComponent("final.bin").path
                : "global/final.bin")

        var candidates: [URL] = []
        if let named = reference.auxiliaryRoots["translated"] { candidates.append(named) }
        candidates.append(root)
        candidates.append(root.appendingPathComponent("translated"))
        if let manifests = candidates.first(where: { hasFlatManifests($0) }) {
            return K3ArtifactResolution(
                shape: .translated(manifests), globals: globals, globalFiles: globalFiles,
                config: config,
                configOrigin: origin)
        }
        // A named translated directory that holds no manifests is a mistake
        // worth naming, rather than one to paper over by translating 1.56 TB
        // into somewhere else.
        if let named = reference.auxiliaryRoots["translated"] {
            throw RunError.artifactLayoutMismatch(
                expected: .k3FlagshipLayerStreams, atPath: named.path)
        }
        guard (try? PublishedArtifactLayout.layers(inArtifact: root.path))?.isEmpty == false else {
            throw RunError.artifactLayoutMismatch(
                expected: .k3FlagshipLayerStreams, atPath: root.path)
        }
        return K3ArtifactResolution(
            shape: .published(root), globals: globals, globalFiles: globalFiles,
            config: config, configOrigin: origin)
    }

    /// The config, in the order a run should prefer it: the caller's, then the
    /// artifact's own, then the flagship config bundled with `BenchScenarios`.
    ///
    /// The last of those is not a fallback in the "probably fine" sense — it is
    /// the file `K3DecodeScenario` decodes the flagship with, checked
    /// byte-for-byte against the test fixture by `K3DecodeScenarioTests`.
    private static func configuration(for reference: ArtifactReference) throws
        -> (TinyK3Config, String)
    {
        if let named = reference.auxiliaryRoots["config"] {
            do {
                return (try TinyK3Config(json: try Data(contentsOf: named)), named.path)
            } catch {
                throw RunError.artifactNotReady("\(named.path): \(error)")
            }
        }
        let inArtifact = reference.root.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: inArtifact.path) {
            do {
                return (try TinyK3Config(json: try Data(contentsOf: inArtifact)), inArtifact.path)
            } catch {
                throw RunError.artifactNotReady("\(inArtifact.path): \(error)")
            }
        }
        do {
            return (try K3DecodeScenario.flagshipConfig(), "BenchScenarios/k3-flagship-config.json")
        } catch {
            throw RunError.artifactNotReady("\(error)")
        }
    }

    private static func hasFlatManifests(_ directory: URL) -> Bool {
        guard isDirectory(directory),
            let entries = try? FileManager.default.contentsOfDirectory(atPath: directory.path)
        else { return false }
        return entries.contains { $0.hasPrefix("layer") && $0.hasSuffix("-manifest.json") }
    }

    private static func isDirectory(_ url: URL) -> Bool {
        var directory: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: url.path, isDirectory: &directory)
        return exists && directory.boolValue
    }
}
