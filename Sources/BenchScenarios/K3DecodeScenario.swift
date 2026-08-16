import Foundation
import MLX
import ModelAdapters
import StorageCore

#if canImport(UIKit)
    import UIKit
#endif

/// One text-only Kimi K3 token, decoded on the device from the published 1.56 TB
/// artifact on an external NVMe.
///
/// This is the device arm of `ArtifactFirstTokenTests.testFirstFullModelToken`,
/// and deliberately assembles the model the same way that test does — translate,
/// ``ArtifactWeightSource``, ``FlagshipRoutedExpertBackend``, ``K3Model`` — so
/// the two platforms differ in hardware and in nothing else. Where the test
/// asserts, this reports: a scenario that threw on a wrong-looking token would
/// destroy the run's evidence at exactly the moment the evidence became
/// interesting.
///
/// ## What bounds the memory, and what does not
///
/// The routed experts are not the problem. Their pool is a stated budget of tens
/// of megabytes against a 1,560 GB model, and M9C measured it at 44.6 MB.
///
/// The binding constraint is **layer 0**. Its deterministic blob is
/// 2,341,257,216 B of BF16, every tensor of which
/// ``ArtifactWeightSource/layerWeights(_:)`` widens to float32 — exactly, and by
/// design, because that widening is what makes the device's arithmetic the Mac's
/// arithmetic — so one resident layer is **4.68 GB**. That is the whole of the
/// Mac's 5.17 GB MLX peak, it cannot be reduced without changing the numbers the
/// run exists to reproduce, and it is why this scenario's declared budget is
/// large and its headroom thin. Every other knob here is chosen to give that one
/// allocation as much room as possible.
public struct K3DecodeScenario: HarnessScenario {

    /// `中国的首都是` under the flagship tiktoken vocabulary.
    ///
    /// M9C measured the flagship and tiny `tiktoken.model` to be byte-identical
    /// (same SHA-256), so M6A's screened ids carry over unchanged and these are
    /// the same three integers the Mac decoded.
    public static let zhCapitalPromptIds = [8448, 1997, 1924]

    /// What the Mac produced, for the screen to compare against — **reporting
    /// targets, not a gate.**
    ///
    /// Spec §10.4 (v0.6.12) is explicit that a flagship token cannot be
    /// bit-gated across implementations: at 896-expert top-16 over 92 layers
    /// roughly 19% of routing decisions sit below the 1e-3 margin the tiny model
    /// was calibrated on, so a different-but-correct implementation may route
    /// differently and emit a different id for reasons that are not a defect.
    /// A different GPU is such an implementation. The device's reproducibility
    /// claim is therefore made *within* the device, by the logits digest.
    public static let macGreedyTokenId = 3372
    public static let macTop1Top2Gap = 0.8024177551269531

    /// The budget this milestone runs against, stated before the run rather than
    /// discovered by it (spec §5.3).
    ///
    /// 5.8 GB of the 6.0 GiB (6.442 GB) `os_proc_available_memory` reports for
    /// this entitled process on this phone — 90%, which is far less headroom
    /// than any earlier milestone claimed and is stated as a risk rather than
    /// dressed up. The number is not a choice about how much to allow; it is
    /// layer 0's 4.68 GB plus the ~0.5 GB of working set the Mac measured
    /// around it, rounded up. Declaring 4.0 GB "to be safe" would just
    /// guarantee a breach that means nothing.
    public static let defaultBudgetBytes: UInt64 = 5_800_000_000

    /// The part of ``defaultBudgetBytes`` that is not the resident layer.
    ///
    /// The budget above is not an allowance, it is arithmetic: layer 0's
    /// 4.68 GB plus the ~0.5 GB of working set the Mac measured around it —
    /// the expert pool (44.6 MB, M9C), MLX's own cache, and the process
    /// overhead. The read-ahead schedule has to hold that half back rather than
    /// spend it on staged bytes, so it is stated here instead of being
    /// rediscovered as a breach.
    public static let workingSetReserveBytes: UInt64 = 500_000_000

    public let id = "k3-decode"
    public let title = "K3 flagship token from external NVMe"
    public var summary: String {
        "Prefill \(promptTokenIds.count) ids + \(cappedDecodeSteps) decode step"
            + (cappedDecodeSteps == 1 ? "" : "s")
            + " over 93 layers. ~188 GB read, minutes per token."
    }

    /// The milestone's two-token ceiling, applied once and read everywhere.
    ///
    /// The prefill already produces a token, so at most one decode step may
    /// follow. This is a computed property rather than a clamp inside `run` so
    /// that the summary on the button cannot promise a number the run will not
    /// honour — which is exactly what it did before a test caught it.
    public var cappedDecodeSteps: Int { max(0, min(decodeSteps, 1)) }

    /// Resolves the artifact root and takes the security scope. Held for the
    /// whole run.
    public let resolveArtifact: @Sendable () throws -> URL
    /// Drops the scope. Called from this scenario's own `defer`, so it runs on
    /// the failure paths too.
    public let accessBracket: (@Sendable () -> Void)?

    public let promptTokenIds: [Int]
    /// Single-token forwards *after* the prefill. The prefill itself produces
    /// one token, so the milestone's two-token ceiling is `decodeSteps <= 1`.
    public let decodeSteps: Int
    public let declaredBudgetBytes: UInt64

    // Pool and evaluation configuration — see `validate()`.
    public let granularity: FetchGranularity
    public let readAhead: Int
    /// Queue a layer's expert regions when the router names them, so the
    /// ``readAhead`` window above is actually used. On by default since the
    /// owner's decision of 2026-08-11; off is what M9C and M10 measured, where
    /// every expert read was demanded one at a time.
    public let prefetchExpertReads: Bool
    /// The expert pool's slot count, stated rather than inferred. `nil` takes
    /// the floor the read-ahead window implies.
    public let expertSlotCount: Int?
    public let expertsPerDispatch: Int
    public let queueDepth: Int
    public let mlxCacheLimitBytes: Int
    public let logitChunkRows: Int
    /// Layers of the **deterministic** stream staged ahead of the one being
    /// computed. One — the default since the owner's decision of 2026-08-11 —
    /// stages layer *i+1* while layer *i* computes, inside
    /// ``declaredBudgetBytes`` less ``readAheadReserveBytes``; zero is the
    /// serial behaviour every number in the M9C and M10 records was measured
    /// with. This scenario can default it on where the library type cannot,
    /// because it declares the budget depth 1 requires. At the phone's 5.8 GB
    /// the schedule refuses exactly the layer-0 pair and stages the other 91 —
    /// a refusal is the feature working, not a case to special-case away.
    /// Unrelated to ``readAhead``, which is the expert pager's window.
    public let deterministicReadAhead: Int
    /// Held back from ``declaredBudgetBytes`` when the read-ahead schedule
    /// decides whether a pair fits. See ``workingSetReserveBytes``.
    public let readAheadReserveBytes: UInt64
    /// The memory dial's residency plan, or `nil` for the streamed-everything
    /// behaviour every earlier record measured.
    ///
    /// When present it states every memory term at once — pinned layers, pair
    /// budget, reserve — so this run cannot be configured with one residency and
    /// reported with another. At the phone's 5.8 GB the planner pins **nothing**
    /// and says why (`docs/design/memory-dial.md` §7): the headroom above the
    /// floor is 570,692,864 B and the smallest pinnable layer is 844,398,592 B.
    /// So this defaults to `nil` on the device not as a hedge but because that
    /// is the plan the device's own budget produces.
    public let pinPlan: PinPlan?
    /// Where the translated manifests go. `nil` means
    /// `context.workingDirectory/k3-translated`.
    public let translationDirectory: URL?

    public init(
        resolveArtifact: @escaping @Sendable () throws -> URL,
        promptTokenIds: [Int] = K3DecodeScenario.zhCapitalPromptIds,
        decodeSteps: Int = 0,
        declaredBudgetBytes: UInt64 = K3DecodeScenario.defaultBudgetBytes,
        granularity: FetchGranularity = .perExpert,
        readAhead: Int = 2,
        prefetchExpertReads: Bool = true,
        expertSlotCount: Int? = nil,
        expertsPerDispatch: Int = 1,
        queueDepth: Int = 3,
        mlxCacheLimitBytes: Int = 64 << 20,
        logitChunkRows: Int = 8192,
        deterministicReadAhead: Int = 1,
        readAheadReserveBytes: UInt64 = K3DecodeScenario.workingSetReserveBytes,
        pinPlan: PinPlan? = nil,
        translationDirectory: URL? = nil,
        accessBracket: (@Sendable () -> Void)? = nil
    ) {
        self.resolveArtifact = resolveArtifact
        self.promptTokenIds = promptTokenIds
        self.decodeSteps = decodeSteps
        self.declaredBudgetBytes = declaredBudgetBytes
        self.granularity = granularity
        self.readAhead = readAhead
        self.prefetchExpertReads = prefetchExpertReads
        self.expertSlotCount = expertSlotCount
        self.expertsPerDispatch = expertsPerDispatch
        self.queueDepth = queueDepth
        self.mlxCacheLimitBytes = mlxCacheLimitBytes
        self.logitChunkRows = logitChunkRows
        self.deterministicReadAhead = deterministicReadAhead
        self.readAheadReserveBytes = readAheadReserveBytes
        self.pinPlan = pinPlan
        self.translationDirectory = translationDirectory
        self.accessBracket = accessBracket
    }

    /// The flagship `config.json`, from the app bundle.
    ///
    /// Byte-identical to `Tests/Fixtures/k3-flagship/flagship-config/config.json`
    /// and checked to be so by `K3DecodeScenarioTests`. The published artifact
    /// does not ship a config — it is a repack of weights — so this file is the
    /// only statement of the model's shape the phone has, and a drifted copy
    /// would change the device's logits with nothing in the record to say why.
    public static func flagshipConfig() throws -> TinyK3Config {
        guard let url = Bundle.module.url(
            forResource: "k3-flagship-config", withExtension: "json")
        else {
            throw ScenarioError.failed("k3-flagship-config.json is missing from the app bundle")
        }
        return try TinyK3Config(json: try Data(contentsOf: url))
    }

    public func run(_ context: ScenarioContext) throws -> ScenarioResult {
        let artifact = try resolveArtifact()
        // Before anything else, and on every exit path: the scope this run reads
        // through is the operator's grant, and leaking it past the run would
        // leave the volume pinned open with nothing using it.
        defer { accessBracket?() }

        let started = Date()
        var thermalTrail: [ThermalSample] = []
        let battery = BatteryReading.begin()

        context.log("artifact: \(artifact.path)")
        context.log(
            "budget: \(ByteSize.format(declaredBudgetBytes)) declared, "
                + "MLX cache \(ByteSize.format(UInt64(mlxCacheLimitBytes))), "
                + "lm_head chunk \(logitChunkRows) rows")

        // A read that fails because the drive went away must arrive as a named
        // error, not as a mystery: check the root is really there before 188 GB
        // of reads start depending on it (spec §12.4 / §19.4).
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: artifact.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw ScenarioError.notReady(
                "\(artifact.path) is not a directory — is the drive mounted and powered?")
        }

        // MLX's cache is a second memory pool, and on this run every megabyte it
        // holds is a megabyte layer 0's 4.68 GB does not have. Tighter than the
        // Qwen scenario's 256 MiB deliberately: that run peaked at 1.4 GiB
        // against 4 GB, this one has no such slack.
        MLX.GPU.set(cacheLimit: mlxCacheLimitBytes)
        MLX.GPU.clearCache()

        let config = try Self.flagshipConfig()
        context.log(
            "config: \(config.numHiddenLayers) layers, \(config.numExperts) experts "
                + "top-\(config.numExpertsPerToken), hidden \(config.hiddenSize)")

        // MARK: Translate

        let translated = translationDirectory
            ?? context.workingDirectory.appendingPathComponent("k3-translated")
        try FileManager.default.createDirectory(
            at: translated, withIntermediateDirectories: true)
        context.report(ScenarioProgress(phase: "translating", detail: "93 manifests"))

        // `.absolutePath`, not the Mac's symlink farm. An iOS external volume
        // mounts under a per-mount UUID, so links written on internal storage
        // name a path that stops being true when the drive is replugged. The
        // manifests are rewritten from the live URL on every run instead.
        let report = try PublishedArtifactLayout.translate(
            artifact: artifact.path, into: translated.path, config: config,
            blobReference: .absolutePath)
        context.log(
            "translated: \(report.layers) layers (\(report.denseLayers) dense, "
                + "\(report.moeLayers) MoE), \(report.experts) experts, "
                + "\(report.symlinks) symlinks, \(ByteSize.format(UInt64(report.manifestBytes))) JSON")
        try context.checkCancellation()

        // MARK: Open

        context.report(ScenarioProgress(phase: "opening", detail: "93 layer streams"))
        // Rejects rather than clamps an undefined depth, and refuses depth 1
        // without a budget: staged bytes are resident bytes (spec §5.3).
        let source = try ArtifactWeightSource(
            translated: translated.path,
            globals: artifact.appendingPathComponent("global").path,
            config: config,
            configuration: pinPlan.map {
                try ArtifactWeightSource.Configuration(
                    plan: $0, deterministicReadAhead: deterministicReadAhead)
            } ?? ArtifactWeightSource.Configuration(
                deterministicReadAhead: deterministicReadAhead,
                budgetBytes: deterministicReadAhead > 0 ? declaredBudgetBytes : nil,
                reserveBytes: readAheadReserveBytes))
        defer { source.shutdown() }
        if let pinPlan {
            context.log(
                "memory dial: \(MemoryDialPlanner.summary(pinPlan).headline), "
                    + "\(pinPlan.pinnedBytes) B pinned of \(pinPlan.budgetBytes) B declared")
        }
        if deterministicReadAhead > 0 {
            let schedule = source.readAheadSchedule()
            let refused = schedule.filter { !$0.isAllowed }.map(\.stagedLayer)
            context.log(
                "deterministic read-ahead: depth \(deterministicReadAhead), "
                    + "reserve \(ByteSize.format(readAheadReserveBytes)), "
                    + "\(schedule.count - refused.count)/\(schedule.count) pairs fit"
                    + (refused.isEmpty ? "" : ", skipping layers \(refused)"))
        }

        let configuration = FlagshipRoutedExpertBackend.Configuration(
            granularity: granularity,
            queueDepth: queueDepth,
            readAhead: readAhead,
            prefetchExpertReads: prefetchExpertReads,
            expertsPerDispatch: expertsPerDispatch,
            slotCount: expertSlotCount)
        // Rejects rather than clamps if the pool cannot serve one dispatch plus
        // read-ahead; a clamped pool deadlocks on `acquire` instead of running
        // slowly, so the throw is the feature.
        let backend = try FlagshipRoutedExpertBackend(
            layers: source.streams, configuration: configuration)
        defer { backend.shutdown() }
        context.log(
            "expert pool: \(ByteSize.format(UInt64(backend.budgetBytes))) "
                + "(granularity \(granularity), readAhead \(readAhead), "
                + "\(expertsPerDispatch)/dispatch, "
                + (prefetchExpertReads ? "prefetching" : "on demand") + ")")

        let model = K3Model(config: config, source: source, expertBackend: backend)
        model.logitChunkRows = logitChunkRows

        // MARK: Decode

        let layerCount = config.numHiddenLayers
        var layerSamples: [LayerSample] = []
        layerSamples.reserveCapacity(layerCount * max(1, decodeSteps + 1))
        var pass = 0
        var passLabel = "prefill"

        model.onLayerCompleted = { index, seconds in
            try context.checkCancellation()
            let telemetry = HarnessTelemetry.sample(declaredBudgetBytes: declaredBudgetBytes)
            layerSamples.append(
                LayerSample(
                    pass: pass, layer: index, seconds: seconds,
                    deterministicBytesRead: source.deterministicBytesRead,
                    expertBytesRequested: backend.expertBytesRequested,
                    footprintBytes: telemetry.footprintBytes,
                    availableBytes: telemetry.availableBytes,
                    mlxActiveBytes: MLX.GPU.activeMemory,
                    mlxCacheBytes: MLX.GPU.cacheMemory,
                    thermalState: telemetry.thermalState,
                    // Where this layer's bytes came from, and what waiting for
                    // them cost. A stall localised to a layer is the difference
                    // between "the overlap worked" and "the overlap worked
                    // except at the pair the budget refused".
                    prefetch: source.lastLayerOutcome.rawValue,
                    prefetchWaitSeconds: source.lastLayerWaitSeconds))
            if thermalTrail.last?.state != telemetry.thermalState {
                thermalTrail.append(
                    ThermalSample(
                        state: telemetry.thermalState,
                        atSeconds: Date().timeIntervalSince(started),
                        pass: pass, layer: index))
                context.log(
                    "thermal -> \(telemetry.thermalState) at \(passLabel) layer \(index)")
            }
            let total = source.deterministicBytesRead &+ backend.expertBytesRequested
            let elapsed = Date().timeIntervalSince(started)
            context.report(
                ScenarioProgress(
                    phase: passLabel,
                    fraction: Double(index + 1) / Double(layerCount),
                    detail: "layer \(index + 1)/\(layerCount) — "
                        + String(format: "%.2f s", seconds),
                    bytesRead: total,
                    bytesPerSecond: elapsed > 0 ? Double(total) / elapsed : nil,
                    telemetry: telemetry))
        }

        thermalTrail.append(
            ThermalSample(
                state: ThermalState.currentName, atSeconds: 0, pass: 0, layer: -1))

        var state = TinyK3State()
        var captures: [TinyK3Capture] = []
        var generated: [Int] = []

        let prefillStarted = Date()
        var capture = try model.forward(tokenIds: promptTokenIds, state: &state)
        var passSeconds: [Double] = [Date().timeIntervalSince(prefillStarted)]
        captures.append(capture)
        generated.append(capture.greedyTokenId)
        context.log(
            "prefill: token \(capture.greedyTokenId) in "
                + String(format: "%.1f s", passSeconds[0]))

        // The milestone caps a run at two tokens. `decodeSteps` defaults to 0,
        // so the ordinary run is prefill only and the cap cannot be exceeded by
        // forgetting a flag.
        for step in 0..<cappedDecodeSteps {
            try context.checkCancellation()
            pass = step + 1
            passLabel = "decode \(pass)"
            let stepStarted = Date()
            capture = try model.forward(tokenIds: [capture.greedyTokenId], state: &state)
            passSeconds.append(Date().timeIntervalSince(stepStarted))
            captures.append(capture)
            generated.append(capture.greedyTokenId)
            context.log(
                "decode \(pass): token \(capture.greedyTokenId) in "
                    + String(format: "%.1f s", passSeconds[pass]))
        }
        model.onLayerCompleted = nil
        let wall = Date().timeIntervalSince(started)

        // MARK: Report

        let first = captures[0]
        guard first.logits.count == config.vocabSize else {
            throw ScenarioError.failed(
                "\(first.logits.count) logits, config declares \(config.vocabSize)")
        }
        let nonFinite = first.logits.filter { !$0.isFinite }.count
        let sorted = first.logits.enumerated().sorted { $0.element > $1.element }
        let gap = Double(sorted[0].element - sorted[1].element)

        // Criterion (a) is a comparison of 163,840 float32 between two cold runs
        // on this device, so the vector itself is written, not a summary that
        // could agree while the logits differ. 655 KB.
        var logitBytes = Data(capacity: first.logits.count * 4)
        for value in first.logits {
            withUnsafeBytes(of: value.bitPattern.littleEndian) { logitBytes.append(contentsOf: $0) }
        }
        let digest = SHA256.hash([UInt8](logitBytes)).map { String(format: "%02x", $0) }.joined()
        let stamp = Self.fileStamp()
        let logitName = "k3-logits-\(stamp).bin"
        try logitBytes.write(
            to: context.workingDirectory.appendingPathComponent(logitName), options: .atomic)
        context.log("logits: \(logitName), sha256 \(digest)")

        // Criterion (b), computed here rather than after a file transfer that
        // may not exist. Non-fatal on purpose: a comparison that cannot be made
        // must not destroy a token that was.
        var comparison: MacReferenceLogits.Comparison?
        do {
            let made = try MacReferenceLogits.compare(
                device: first.logits, deviceGreedy: first.greedyTokenId, deviceGap: gap,
                reductionLength: config.hiddenSize)
            comparison = made
            context.log(
                String(
                    format: "vs Mac: relative %.3e, tolerance %.3e, margin %.2fx, "
                        + "%d of %d elements differ",
                    made.relative, made.tolerance, made.margin, made.differingElements,
                    made.elements))
        } catch {
            context.log("vs Mac: comparison unavailable — \(error)")
        }

        let telemetry = HarnessTelemetry.sample(declaredBudgetBytes: declaredBudgetBytes)
        let peakFootprint = layerSamples.map(\.footprintBytes).max() ?? telemetry.footprintBytes
        let overBudget = telemetry.peakFootprintBytes > declaredBudgetBytes
        let readAheadMetrics = source.readAheadMetrics
        // Before the `defer`red shutdown: the pool census describes this run
        // only while the run still owns its slots.
        let expertPhase = backend.expertPhaseMetrics

        let payload = Payload(
            scenario: id,
            revision: BuildInfo.gitRevision,
            environment: EnvironmentReport.current(path: artifact.path),
            artifactDirectory: artifact.path,
            translatedDirectory: translated.path,
            promptTokenIds: promptTokenIds,
            greedyTokenId: first.greedyTokenId,
            generatedTokenIds: generated,
            logitsFile: logitName,
            logitsSHA256: digest,
            nonFiniteLogits: nonFinite,
            top8: sorted.prefix(8).map { Top(id: $0.offset, logit: Double($0.element)) },
            top1Top2Gap: gap,
            macGreedyTokenId: Self.macGreedyTokenId,
            macTop1Top2Gap: Self.macTop1Top2Gap,
            macComparison: comparison,
            minimumRelativeRoutingMargin: first.minimumRelativeRoutingMargin,
            routerRelativeMargins: first.routerRelativeMargins,
            routingDecisions: first.routerRelativeMargins.reduce(0) { $0 + $1.count },
            wallSeconds: wall,
            passSeconds: passSeconds,
            layerSeconds: model.lastLayerSeconds,
            layerReclaimSeconds: model.lastLayerReclaimSeconds,
            layerAttentionBarrierSeconds: model.lastLayerAttentionBarrierSeconds,
            deterministicBytesRead: source.deterministicBytesRead,
            expertBytesRequested: backend.expertBytesRequested,
            expertReadCount: backend.expertReadCount,
            gatherCount: backend.gatherCount,
            expertPoolBudgetBytes: backend.budgetBytes,
            mlxPeakBytes: MLX.GPU.peakMemory,
            deterministicReadAhead: readAheadMetrics,
            pinnedTier: source.pinnedTierMetrics,
            pinPlan: pinPlan,
            expertPhase: expertPhase,
            readAheadSkippedLayers: source.readAheadSchedule()
                .filter { !$0.isAllowed }.map(\.stagedLayer),
            declaredBudgetBytes: declaredBudgetBytes,
            peakFootprintBytes: max(peakFootprint, telemetry.peakFootprintBytes),
            budgetRespected: !overBudget,
            telemetry: telemetry,
            thermalTrail: thermalTrail,
            battery: battery.finish(),
            configuration: [
                "granularity": "\(granularity)",
                "readAhead": "\(readAhead)",
                "prefetchExpertReads": "\(prefetchExpertReads)",
                "expertPoolSlots": "\(expertPhase.poolSlotCount)",
                "expertsPerDispatch": "\(expertsPerDispatch)",
                "queueDepth": "\(queueDepth)",
                "mlxCacheLimitBytes": "\(mlxCacheLimitBytes)",
                "logitChunkRows": "\(logitChunkRows)",
                "blobReference": "absolutePath",
                "deterministicReadAhead": "\(deterministicReadAhead)",
                "readAheadReserveBytes": "\(readAheadReserveBytes)",
            ],
            translation: TranslationReport(
                layers: report.layers, denseLayers: report.denseLayers,
                moeLayers: report.moeLayers, experts: report.experts,
                symlinks: report.symlinks, manifestBytes: report.manifestBytes),
            layers: layerSamples)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        let minutes = wall / 60
        var headline = String(
            format: "token %d in %.1f min, peak %@", first.greedyTokenId, minutes,
            ByteSize.format(max(peakFootprint, telemetry.peakFootprintBytes)))
        if first.greedyTokenId != Self.macGreedyTokenId {
            // Not "MISMATCH". Spec §10.4 says a flagship token is not
            // cross-implementation gate-able, so a different id here is a
            // near-tie observation until the margins say otherwise.
            headline += " (Mac \(Self.macGreedyTokenId) — see routing margin)"
        }
        if overBudget { headline += ", OVER BUDGET" }

        var lines = [
            String(
                format: "read %@ deterministic + %@ expert",
                ByteSize.format(source.deterministicBytesRead),
                ByteSize.format(backend.expertBytesRequested)),
            "expert pool \(ByteSize.format(UInt64(backend.budgetBytes))), "
                + "MLX peak \(ByteSize.format(UInt64(MLX.GPU.peakMemory)))",
            "logits sha256 \(digest)",
            comparison.map {
                String(
                    format: "vs Mac: rel %.3e vs tol %.3e (margin %.2fx), %d/%d differ",
                    $0.relative, $0.tolerance, $0.margin, $0.differingElements, $0.elements)
            } ?? "vs Mac: comparison unavailable",
            String(
                format: "top1-top2 %.4f (Mac %.4f), min routing margin %.3e", gap,
                Self.macTop1Top2Gap, first.minimumRelativeRoutingMargin),
            "thermal " + thermalTrail.map(\.state).joined(separator: " -> "),
        ]
        if deterministicReadAhead > 0 {
            lines.append(
                String(
                    format: "read-ahead: %d staged / %d skipped / %d missed, waited %.1f s of "
                        + "%.1f s staged (%.0f%% hidden), peak staged %@",
                    readAheadMetrics.stagedLayerCount, readAheadMetrics.skippedLayerCount,
                    readAheadMetrics.missedLayerCount, readAheadMetrics.prefetchWaitSeconds,
                    readAheadMetrics.stageReadSeconds,
                    readAheadMetrics.overlapEfficiency * 100,
                    ByteSize.format(readAheadMetrics.peakStagedBytes)))
        }
        if pinPlan != nil { lines.append(source.pinnedTierMetrics.summaryLine) }
        lines.append(expertPhase.summaryLine)
        if nonFinite > 0 { lines.append("\(nonFinite) NON-FINITE LOGITS") }

        return ScenarioResult(
            headline: headline, lines: lines,
            json: try? encoder.encode(payload), jsonFilename: "\(id).json")
    }

    private static func fileStamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        formatter.timeZone = TimeZone(identifier: "UTC")
        return formatter.string(from: Date())
    }

    // MARK: - Records

    /// One layer, on one pass. 93 of these per token is the trace spec §18.1
    /// asks for and the only evidence that would localise a stall or a thermal
    /// cliff to a place in the model rather than to the run as a whole.
    struct LayerSample: Encodable {
        let pass: Int
        let layer: Int
        let seconds: Double
        let deterministicBytesRead: UInt64
        let expertBytesRequested: UInt64
        let footprintBytes: UInt64
        let availableBytes: UInt64?
        let mlxActiveBytes: Int
        let mlxCacheBytes: Int
        let thermalState: String
        /// `serial`, `staged`, `skipped` or `missed` — how this layer's
        /// deterministic bytes arrived. Always `serial` at depth 0.
        let prefetch: String
        let prefetchWaitSeconds: Double
    }

    /// Recorded on change rather than on a timer, so the trail says *where* the
    /// device got hot — M8.5 reached `serious` on sustained external reads, and
    /// which layer that happens at is the useful form of the fact.
    struct ThermalSample: Encodable {
        let state: String
        let atSeconds: Double
        let pass: Int
        let layer: Int
    }

    struct Top: Encodable {
        let id: Int
        let logit: Double
    }

    struct TranslationReport: Encodable {
        let layers: Int
        let denseLayers: Int
        let moeLayers: Int
        let experts: Int
        let symlinks: Int
        let manifestBytes: Int
    }

    struct Payload: Encodable {
        let scenario: String
        let revision: String
        let environment: EnvironmentReport
        let artifactDirectory: String
        let translatedDirectory: String
        let promptTokenIds: [Int]
        let greedyTokenId: Int
        let generatedTokenIds: [Int]
        let logitsFile: String
        let logitsSHA256: String
        let nonFiniteLogits: Int
        let top8: [Top]
        let top1Top2Gap: Double
        let macGreedyTokenId: Int
        let macTop1Top2Gap: Double
        /// Criterion (b): the output-scale relative metric against the Mac's
        /// archived logits, computed on device. Nil only if the bundled
        /// reference could not be loaded.
        let macComparison: MacReferenceLogits.Comparison?
        let minimumRelativeRoutingMargin: Double
        /// Every routing decision, not only the tightest: the minimum is an
        /// extreme-value statistic over `92 x positions` draws, and spec §10.4
        /// requires the distribution rather than the minimum.
        let routerRelativeMargins: [[Double]]
        let routingDecisions: Int
        let wallSeconds: Double
        let passSeconds: [Double]
        let layerSeconds: [Double]
        let layerReclaimSeconds: [Double]
        /// The mid-layer evaluation barrier that lets the attention half be
        /// released before the MLP half is widened. Inside `layerSeconds`, and
        /// reported separately because it is a synchronisation bought for a
        /// memory saving — the same trade `layerReclaimSeconds` records.
        let layerAttentionBarrierSeconds: [Double]
        let deterministicBytesRead: UInt64
        let expertBytesRequested: UInt64
        let expertReadCount: Int
        let gatherCount: Int
        let expertPoolBudgetBytes: Int
        let mlxPeakBytes: Int
        /// What the deterministic read-ahead did. All zeros at depth 0, which
        /// is the arm every earlier record was measured on.
        let deterministicReadAhead: DeterministicReadAheadMetrics
        /// What the memory dial held in RAM, and what that saved. All zeros
        /// when nothing is pinned — the arm every earlier record measured, and
        /// the plan the phone's own 5.8 GB budget produces.
        let pinnedTier: PinnedTierMetrics
        /// The plan that produced the tier, recorded beside it so a run says
        /// which residency it was asked for as well as which it achieved.
        let pinPlan: PinPlan?
        /// Where the routed-expert phase's seconds went: storage wait, Metal,
        /// and the remainder. The half of the token deterministic read-ahead
        /// cannot touch, because the router decides inside the layer.
        let expertPhase: ExpertPhaseMetrics
        /// Pairs the budget refused, by staged layer. Empty at depth 0.
        let readAheadSkippedLayers: [Int]
        let declaredBudgetBytes: UInt64
        let peakFootprintBytes: UInt64
        let budgetRespected: Bool
        let telemetry: HarnessTelemetry
        let thermalTrail: [ThermalSample]
        let battery: BatteryReading.Delta
        let configuration: [String: String]
        let translation: TranslationReport
        let layers: [LayerSample]
    }
}

/// Battery level at the start and end of a run.
///
/// Spec §23 M10 asks for "energy/thermal observations", and on a device with no
/// power meter the battery gauge is the only energy instrument available. It is
/// coarse — iOS reports it in 5% steps — so a single token may not move it at
/// all, and the record says `nil` rather than `0` when monitoring is
/// unavailable, because "unmeasured" and "unchanged" are different claims.
public struct BatteryReading: Sendable {

    public struct Delta: Encodable, Sendable {
        public let startLevel: Float?
        public let endLevel: Float?
        public let startState: String?
        public let endState: String?
        /// Nil when either end is unavailable.
        public let drop: Float?
    }

    private let startLevel: Float?
    private let startState: String?

    public static func begin() -> BatteryReading {
        #if canImport(UIKit)
            let device = UIDevice.current
            device.isBatteryMonitoringEnabled = true
            let level = device.batteryLevel
            return BatteryReading(
                startLevel: level < 0 ? nil : level, startState: Self.name(device.batteryState))
        #else
            return BatteryReading(startLevel: nil, startState: nil)
        #endif
    }

    public func finish() -> Delta {
        #if canImport(UIKit)
            let device = UIDevice.current
            let level = device.batteryLevel
            let end: Float? = level < 0 ? nil : level
            var drop: Float?
            if let startLevel, let end { drop = startLevel - end }
            return Delta(
                startLevel: startLevel, endLevel: end, startState: startState,
                endState: Self.name(device.batteryState), drop: drop)
        #else
            return Delta(
                startLevel: nil, endLevel: nil, startState: nil, endState: nil, drop: nil)
        #endif
    }

    #if canImport(UIKit)
        private static func name(_ state: UIDevice.BatteryState) -> String {
            switch state {
            case .charging: return "charging"
            case .full: return "full"
            case .unplugged: return "unplugged"
            default: return "unknown"
            }
        }
    #endif
}
