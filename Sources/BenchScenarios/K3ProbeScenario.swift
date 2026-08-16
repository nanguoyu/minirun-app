import Foundation
import MLX
import ModelAdapters
import StorageCore

/// Everything about a K3 device run that can be checked in seconds instead of
/// in minutes.
///
/// A flagship token is ~188 GB and minutes of wall time, and it depends on a
/// chain of assumptions — the scope holds, the artifact opens, the manifests
/// translate, the budget survives the widest layer — any one of which fails the
/// whole run at the end. Each of those is cheap to test on its own, and this
/// scenario tests them in the order they would fail, so a broken assumption
/// costs seconds and one line of the log rather than a rig move.
///
/// Two of its questions are recorded findings in their own right.
///
/// **Does a symlink into a security-scoped volume resolve?** The Mac
/// translation writes a farm of them and iOS may or may not follow one from
/// internal storage into `userfsd`. The decode scenario does not depend on the
/// answer — it uses ``PublishedArtifactLayout/BlobReference/absolutePath`` — but
/// the answer is worth having written down, because the next person to reach
/// for the farm will otherwise have to find out the expensive way.
///
/// **Does layer 0 fit?** Its deterministic blob is 2.34 GB of BF16 widened to
/// **4.68 GB** of float32, which is the entire memory story of a K3 forward pass
/// and the one number that decides whether iOS kills the run. Materialising that
/// single layer costs a few seconds of reading and answers it outright.
public struct K3ProbeScenario: HarnessScenario {

    public let id = "k3-probe"
    public let title = "K3 pre-flight probe"
    public let summary =
        "Seconds: scope, artifact census, symlink resolution, translation, "
        + "and layer 0's 4.68 GB against the budget."

    public let resolveArtifact: @Sendable () throws -> URL
    public let accessBracket: (@Sendable () -> Void)?
    public let declaredBudgetBytes: UInt64
    /// Materialise layer 0 — the widest — and report the peak. The expensive
    /// part of the probe, and still only seconds.
    public let materialiseWidestLayer: Bool

    public init(
        resolveArtifact: @escaping @Sendable () throws -> URL,
        declaredBudgetBytes: UInt64 = K3DecodeScenario.defaultBudgetBytes,
        materialiseWidestLayer: Bool = true,
        accessBracket: (@Sendable () -> Void)? = nil
    ) {
        self.resolveArtifact = resolveArtifact
        self.declaredBudgetBytes = declaredBudgetBytes
        self.materialiseWidestLayer = materialiseWidestLayer
        self.accessBracket = accessBracket
    }

    public func run(_ context: ScenarioContext) throws -> ScenarioResult {
        let artifact = try resolveArtifact()
        defer { accessBracket?() }
        var checks: [Check] = []
        func note(_ name: String, _ ok: Bool, _ detail: String) {
            checks.append(Check(name: name, ok: ok, detail: detail))
            context.log("\(ok ? "ok  " : "FAIL") \(name): \(detail)")
        }

        context.log("artifact: \(artifact.path)")
        context.report(ScenarioProgress(phase: "probing", detail: "scope"))

        // 1. The volume, as the pager will see it.
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(
            atPath: artifact.path, isDirectory: &isDirectory)
        note("artifact root", exists && isDirectory.boolValue, artifact.path)
        guard exists && isDirectory.boolValue else {
            throw ScenarioError.notReady(
                "\(artifact.path) is not a directory — is the drive mounted and powered?")
        }
        let storage = StorageInfo.describing(path: artifact.path)
        note(
            "volume", true,
            "\(storage.filesystemType ?? "?"), "
                + "removable \(storage.volumeIsRemovable.map(String.init) ?? "?"), "
                + "\(ByteSize.format(UInt64(storage.volumeAvailableBytes ?? 0))) free")

        // 2. Census. `layers(inArtifact:)` is the same enumeration the
        //    translation does, so a missing container is found here.
        context.report(ScenarioProgress(phase: "probing", detail: "census"))
        let sources = try PublishedArtifactLayout.layers(inArtifact: artifact.path)
        let withContainers = sources.filter(\.hasContainers).count
        note(
            "layer census", sources.count == 93 && withContainers == 92,
            "\(sources.count) layers, \(withContainers) route, "
                + "\(sources.count - withContainers) dense")

        // 3. Symlink resolution into a security-scoped volume — the recorded
        //    question. Tried against a real blob so a false pass is not
        //    possible: the link is opened and a byte is read through it.
        context.report(ScenarioProgress(phase: "probing", detail: "symlink"))
        let symlink = try probeSymlink(
            into: context.workingDirectory,
            target: sources[0].directory.appendingPathComponent(sources[0].deterministicFile))
        note("symlink into scoped volume", symlink.ok, symlink.detail)

        // 4. Translation, in the mode the decode will use.
        context.report(ScenarioProgress(phase: "probing", detail: "translating"))
        let config = try K3DecodeScenario.flagshipConfig()
        let translated = context.workingDirectory.appendingPathComponent("k3-translated")
        try FileManager.default.createDirectory(
            at: translated, withIntermediateDirectories: true)
        let started = Date()
        let report = try PublishedArtifactLayout.translate(
            artifact: artifact.path, into: translated.path, config: config,
            blobReference: .absolutePath)
        note(
            "translate (absolutePath)",
            report.layers == 93 && report.experts == 896 && report.symlinks == 0,
            "\(report.layers) layers, \(report.experts) experts, \(report.symlinks) links, "
                + String(format: "%.2f s", Date().timeIntervalSince(started)))

        // 5. The loader's own validation, which re-opens every container header
        //    and re-derives the geometry. If path indirection were wrong, this
        //    is where it would say so.
        context.report(ScenarioProgress(phase: "probing", detail: "opening"))
        MLX.GPU.set(cacheLimit: 64 << 20)
        MLX.GPU.clearCache()
        let source = try ArtifactWeightSource(
            translated: translated.path,
            globals: artifact.appendingPathComponent("global").path,
            config: config)
        note("open 93 streams", source.streams.count == 93, "\(source.streams.count) streams")

        // 6. The embedding rows the prompt needs — three rows out of 163,840,
        //    which is the whole of spec §16.4's claim in one call.
        let rows = try source.embeddingRows(K3DecodeScenario.zhCapitalPromptIds)
        let finite = rows.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite }
        note(
            "embedding rows", rows.shape == [3, config.hiddenSize] && finite,
            "\(rows.shape) finite=\(finite)")

        // 7. Layer 0, the widest, against the budget.
        var peak: UInt64 = 0
        var layerZeroSeconds = 0.0
        var layerZeroBytes: UInt64 = 0
        if materialiseWidestLayer {
            context.report(
                ScenarioProgress(phase: "probing", detail: "layer 0 — 2.34 GB read, 4.68 GB live"))
            let before = source.deterministicBytesRead
            let layerStarted = Date()
            _ = try source.layerWeights(0)
            layerZeroSeconds = Date().timeIntervalSince(layerStarted)
            layerZeroBytes = source.deterministicBytesRead &- before
            let telemetry = HarnessTelemetry.sample(declaredBudgetBytes: declaredBudgetBytes)
            peak = telemetry.peakFootprintBytes
            note(
                "layer 0 within budget", peak <= declaredBudgetBytes,
                "read \(ByteSize.format(layerZeroBytes)) in "
                    + String(format: "%.1f s", layerZeroSeconds)
                    + ", peak \(ByteSize.format(peak)) of "
                    + "\(ByteSize.format(declaredBudgetBytes)) declared, "
                    + "available \(ByteSize.format(telemetry.availableBytes ?? 0)), "
                    + "MLX peak \(ByteSize.format(UInt64(MLX.GPU.peakMemory)))")
            source.releaseLayer(0)
        }

        let telemetry = HarnessTelemetry.sample(declaredBudgetBytes: declaredBudgetBytes)
        let failures = checks.filter { !$0.ok }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let payload = Payload(
            scenario: id,
            revision: BuildInfo.gitRevision,
            environment: EnvironmentReport.current(path: artifact.path),
            artifactDirectory: artifact.path,
            checks: checks,
            symlinkResolves: symlink.ok,
            symlinkDetail: symlink.detail,
            layerZeroSeconds: layerZeroSeconds,
            layerZeroBytesRead: layerZeroBytes,
            declaredBudgetBytes: declaredBudgetBytes,
            peakFootprintBytes: max(peak, telemetry.peakFootprintBytes),
            mlxPeakBytes: MLX.GPU.peakMemory,
            telemetry: telemetry)

        let headline =
            failures.isEmpty
            ? "\(checks.count)/\(checks.count) ok, peak "
                + ByteSize.format(max(peak, telemetry.peakFootprintBytes))
            : "\(failures.count) of \(checks.count) FAILED: "
                + failures.map(\.name).joined(separator: ", ")

        return ScenarioResult(
            headline: headline,
            lines: checks.map { "\($0.ok ? "ok" : "FAIL") \($0.name): \($0.detail)" },
            json: try? encoder.encode(payload), jsonFilename: "\(id).json")
    }

    /// Write a symlink on internal storage pointing at a blob on the scoped
    /// volume, then `stat` it and read a byte **through** it.
    ///
    /// Both halves are needed to call it a pass: `stat(2)` follows links and
    /// `open(2)` enforces the sandbox, so a link can size correctly and still
    /// refuse to open. Cleaned up either way — the probe leaves nothing behind.
    private func probeSymlink(into directory: URL, target: URL) throws -> (
        ok: Bool, detail: String
    ) {
        let link = directory.appendingPathComponent("k3-symlink-probe")
        let manager = FileManager.default
        try? manager.removeItem(at: link)
        defer { try? manager.removeItem(at: link) }
        do {
            try manager.createSymbolicLink(at: link, withDestinationURL: target)
        } catch {
            return (false, "could not create the link: \(error)")
        }
        var status = stat()
        guard stat(link.path, &status) == 0 else {
            return (false, "stat through the link failed: errno \(errno)")
        }
        let descriptor = Darwin.open(link.path, O_RDONLY)
        guard descriptor >= 0 else {
            return (
                false,
                "stat saw \(status.st_size) B but open(2) refused: errno \(errno) "
                    + "— the sandbox does not follow a link into the scoped volume"
            )
        }
        defer { Darwin.close(descriptor) }
        var byte: UInt8 = 0
        let got = pread(descriptor, &byte, 1, 0)
        return (
            got == 1,
            got == 1
                ? "stat \(status.st_size) B, open and read through the link both succeeded"
                : "opened but pread returned \(got), errno \(errno)"
        )
    }

    struct Check: Encodable {
        let name: String
        let ok: Bool
        let detail: String
    }

    struct Payload: Encodable {
        let scenario: String
        let revision: String
        let environment: EnvironmentReport
        let artifactDirectory: String
        let checks: [Check]
        let symlinkResolves: Bool
        let symlinkDetail: String
        let layerZeroSeconds: Double
        let layerZeroBytesRead: UInt64
        let declaredBudgetBytes: UInt64
        let peakFootprintBytes: UInt64
        let mlxPeakBytes: Int
        let telemetry: HarnessTelemetry
    }
}
