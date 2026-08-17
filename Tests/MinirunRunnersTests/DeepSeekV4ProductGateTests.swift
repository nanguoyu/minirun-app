import Foundation
import MinirunKit
import ModelAdapters
import XCTest

@testable import MinirunRunners

/// The opt-in, real-artifact exit gate for DeepSeek V4 product registration.
///
/// Ordinary test runs skip before touching the network or artifact. An armed
/// run deliberately performs complete repository verification, then makes the
/// same rooted authority feed the tokenizer and production runner. Reference
/// output is supplied alongside the arm instead of being compiled into the
/// App: the owner may publish a newer compatible artifact, but an observation
/// from one immutable source/publication pair cannot silently authorize the
/// next pair.
///
/// The gate runs at either runner scale. `MINIRUN_V4_SCALE=product` is the
/// default and is this gate unchanged — the 2 GB floor, the complete product
/// memory plan, and no pinned weights. `MINIRUN_V4_SCALE=stated` runs the same
/// four lifecycle arms against a declared budget above that floor, so the
/// memory dial's residency plan and the per-pass phase decomposition are
/// recorded for a run that actually pins. The dial's first in-app arm moved a
/// decode pass from ~10 s to ~6 s at 8 GB and could not say where the
/// remaining ~6 s went, because the only instrumented gate ran product scale
/// (`docs/experiments/2026-08-15-v4-memory-dial-first-arm.md`); this is the
/// arm that can say.
///
/// Build the Release test bundle, then invoke this test through `xcrun xctest`
/// so the environment reaches the process. Required variables are documented
/// by ``DeepSeekV4ProductGateConfiguration/load()``.
final class DeepSeekV4ProductGateTests: XCTestCase {
    func testCurrentVerifiedArtifactMatchesReferenceAndStopsSafely() async throws {
        #if !arch(arm64)
            throw XCTSkip("the V4 MLX runtime requires Apple silicon")
        #else
            let configuration = try DeepSeekV4ProductGateConfiguration.load()
            let tokenProvider = configuration.huggingFaceToken.map { token in
                { @Sendable in token as String? }
            }
            let transport = URLSessionTransport()
            let catalog = try await HuggingFaceCatalogSource(
                transport: transport, token: tokenProvider
            ).fetch()
            let descriptor = try XCTUnwrap(catalog.descriptor(.deepseekV4Flash))
            let publication = try XCTUnwrap(descriptor.source.repo)
            XCTAssertEqual(
                publication.repoID.caseInsensitiveCompare(
                    "nanguoyu/DeepSeek-V4-Flash-0731-minirun"),
                .orderedSame)
            XCTAssertEqual(publication.revision, configuration.publicationRevision)

            let ledger = InMemoryVerificationLedger()
            let location = ArtifactLocator(
                catalog: catalog, verificationLedger: ledger,
                // The gate names the artifact root itself; no parent walk is
                // needed and an accidentally supplied volume root must not
                // turn into a broad scan.
                maximumDepth: 0
            ).scan(configuration.artifactRoot)
            let artifact = try XCTUnwrap(
                location.artifacts.first(where: { $0.model == .deepseekV4Flash }),
                "the armed path does not contain a catalog-matched V4 artifact")
            XCTAssertEqual(
                artifact.rootPath, configuration.artifactRoot.standardizedFileURL.path)
            XCTAssertEqual(
                artifact.index.repositories.first?.revision,
                configuration.sourceRevision)
            XCTAssertEqual(
                artifact.isComplete, true,
                "the real gate refuses a short or metadata-stale local tree")

            let rooted = try ArtifactVerificationRoot.open(
                relativeComponents: [], beneath: configuration.artifactRoot)
            let verifier = ArtifactVerifier(
                tree: HuggingFaceTreeClient(
                    transport: transport, token: tokenProvider),
                catalog: catalog, ledger: ledger)
            let report = try await verifier.verify(
                artifact, .full, rootedAt: rooted,
                onProgress: { progress in
                    guard progress.filesChecked == progress.filesToCheck
                        || progress.filesChecked.isMultiple(of: 16)
                    else { return }
                    print(
                        "[v4-product-gate] verified \(progress.filesChecked)/"
                            + "\(progress.filesToCheck) files, \(progress.bytesChecked)/"
                            + "\(progress.bytesToCheck) bytes")
                })
            XCTAssertTrue(report.isComplete, "complete verification returned a refusal report")
            XCTAssertEqual(report.depth, .digestEverything)

            let record = try XCTUnwrap(
                ledger.record(
                    matching: ArtifactVerificationLookup(
                        rootPath: artifact.rootPath, model: artifact.model,
                        repository: publication, index: artifact.index)),
                "the complete pass did not leave applicable evidence")
            XCTAssertEqual(record.state, .fullyVerified)
            let evidence = try XCTUnwrap(record.evidence)
            let authority = try ArtifactRuntimeAuthority(root: rooted, evidence: evidence)

            let vocabulary = try DeepSeekV4Vocabulary(
                data: authority.openFile(
                    try XCTUnwrap(evidence.index.tokenizer).file
                ).readAll(maximumBytes: 32 << 20))
            let promptIDs = try DeepSeekV4NoThinkingChatPrompt(
                messages: [
                    DeepSeekV4ChatMessage(
                        role: .user, content: configuration.prompt)
                ]
            ).encode(using: vocabulary)

            let completed = try await runToCompletion(
                authority: authority, promptIDs: promptIDs,
                configuration: configuration)
            XCTAssertEqual(completed.summary.tokenIDs, configuration.expectedTokenIDs)
            XCTAssertEqual(
                productText(completed.summary.tokenIDs, vocabulary: vocabulary),
                configuration.expectedText)
            XCTAssertLessThanOrEqual(
                completed.summary.peakFootprintBytes, configuration.memoryBudgetBytes)
            XCTAssertTrue(completed.summary.budgetRespected)
            try assertScaleMemoryContract(
                completed.resultJSON, arm: "completion", configuration: configuration)
            try configuration.write(
                completed.resultJSON,
                filename: configuration.resultFilename("completion"))

            if let memoryTokens = try configuration.optionalMemoryNewTokenCount() {
                let plateau = try await runToCompletion(
                    authority: authority, promptIDs: promptIDs,
                    maximumNewTokens: memoryTokens,
                    configuration: configuration)
                XCTAssertEqual(plateau.summary.tokenIDs.count, memoryTokens)
                XCTAssertEqual(
                    Array(plateau.summary.tokenIDs.prefix(configuration.expectedTokenIDs.count)),
                    configuration.expectedTokenIDs,
                    "the longer lifecycle arm diverged before the reference prefix")
                try assertDecodeMemoryPlateau(
                    plateau.telemetry,
                    expectedDecodePasses: memoryTokens - 1,
                    declaredBudgetBytes: configuration.memoryBudgetBytes)
                try assertScaleMemoryContract(
                    plateau.resultJSON, arm: "memory-plateau",
                    configuration: configuration)
                try configuration.write(
                    plateau.resultJSON,
                    filename: configuration.resultFilename("memory-plateau"))
            }

            let stopped = try await runUntilFirstCompletedLayerThenStop(
                authority: authority, promptIDs: promptIDs,
                configuration: configuration)
            XCTAssertTrue(stopped.sawCompletedLayer)
            XCTAssertEqual(stopped.terminalEventCount, 1)
            XCTAssertNotNil(stopped.summary)
            let cancelled = try XCTUnwrap(stopped.summary)
            XCTAssertLessThanOrEqual(
                cancelled.peakFootprintBytes, configuration.memoryBudgetBytes)
            XCTAssertTrue(cancelled.budgetRespected)
            try assertReleasedBeforeTerminal(cancelled)
            let stopJSON = try XCTUnwrap(cancelled.resultJSON)
            try assertScaleMemoryContract(
                stopJSON, arm: "stop", configuration: configuration)
            try configuration.write(
                stopJSON, filename: configuration.resultFilename("stop"))

            if let promptCount = try configuration.optionalMemoryPromptTokenCount() {
                let ordinaryToken = try XCTUnwrap(configuration.expectedTokenIDs.first)
                let memoryPrompt = Array(repeating: ordinaryToken, count: promptCount)
                let memoryArm = try await runToCompletion(
                    authority: authority, promptIDs: memoryPrompt,
                    maximumNewTokens: 1, configuration: configuration)
                XCTAssertEqual(memoryArm.summary.tokenIDs.count, 1)
                XCTAssertLessThanOrEqual(
                    memoryArm.summary.peakFootprintBytes,
                    configuration.memoryBudgetBytes)
                XCTAssertTrue(memoryArm.summary.budgetRespected)
                try assertScaleMemoryContract(
                    memoryArm.resultJSON, arm: "memory-\(promptCount)",
                    configuration: configuration)
                try configuration.write(
                    memoryArm.resultJSON,
                    filename: configuration.resultFilename("memory-\(promptCount)"))
            }
        #endif
    }

    #if arch(arm64)
        private func runToCompletion(
            authority: ArtifactRuntimeAuthority, promptIDs: [Int],
            maximumNewTokens: Int? = nil,
            configuration: DeepSeekV4ProductGateConfiguration
        ) async throws -> (summary: RunSummary, resultJSON: Data, telemetry: [RunTelemetry]) {
            let runner = DeepSeekV4DecodeRunner(scale: configuration.runnerScale)
            let requestedTokens = maximumNewTokens
                ?? configuration.expectedTokenIDs.count
            let session = try runner.start(
                request(
                    authority: authority, promptIDs: promptIDs,
                    maximumNewTokens: requestedTokens,
                    configuration: configuration))
            var accepted = false
            var tokens: [Int] = []
            var finished: RunSummary?
            var terminalEvents = 0
            var telemetry: [RunTelemetry] = []
            for try await event in session.events {
                switch event {
                case .accepted:
                    XCTAssertFalse(accepted)
                    accepted = true
                case .token(let token):
                    tokens.append(token.tokenID)
                case .finished(let summary):
                    terminalEvents += 1
                    finished = summary
                case .cancelled:
                    terminalEvents += 1
                    XCTFail("the reference completion arm was cancelled")
                case .telemetry(let sample):
                    telemetry.append(sample)
                case .phase, .log, .phaseSummary:
                    break
                }
            }
            XCTAssertTrue(accepted)
            XCTAssertEqual(terminalEvents, 1)
            let summary = try XCTUnwrap(finished)
            XCTAssertEqual(tokens, summary.tokenIDs)
            try assertReleasedBeforeTerminal(summary)
            return (summary, try XCTUnwrap(summary.resultJSON), telemetry)
        }

        private func assertDecodeMemoryPlateau(
            _ samples: [RunTelemetry], expectedDecodePasses: Int,
            declaredBudgetBytes: UInt64
        ) throws {
            let grouped = Dictionary(grouping: samples) { $0.phase }
            let phases = grouped.keys.filter { $0.hasPrefix("decode token ") }.sorted {
                let left = Int($0.split(separator: " ").last ?? "") ?? 0
                let right = Int($1.split(separator: " ").last ?? "") ?? 0
                return left < right
            }
            XCTAssertEqual(phases.count, expectedDecodePasses)
            guard let first = phases.first, let firstSamples = grouped[first], !firstSamples.isEmpty
            else { throw ProductGateConfigurationError.invalid("decode telemetry", "empty") }

            let firstActivePeak = firstSamples.map(\.mlxActiveBytes).max() ?? 0
            let firstFootprintPeak = firstSamples.map(\.footprintBytes).max() ?? 0
            let activeAllowance: UInt64 = 64 << 20
            let footprintAllowance: UInt64 = 256 << 20
            for phase in phases {
                let pass = try XCTUnwrap(grouped[phase])
                XCTAssertFalse(pass.isEmpty)
                XCTAssertTrue(
                    pass.allSatisfy { $0.footprintBytes <= declaredBudgetBytes },
                    "\(phase) crossed the declared memory ceiling")
                if phase != first {
                    XCTAssertLessThanOrEqual(
                        pass.map(\.mlxActiveBytes).max() ?? 0,
                        firstActivePeak + activeAllowance,
                        "\(phase) retained another generation of V4 state")
                    XCTAssertLessThanOrEqual(
                        pass.map(\.footprintBytes).max() ?? 0,
                        firstFootprintPeak + footprintAllowance,
                        "\(phase) did not settle near the first decode pass")
                }
            }
            print(
                "[v4-product-gate] decode memory plateaus: "
                    + phases.map { phase in
                        let pass = grouped[phase] ?? []
                        return "\(phase)=active:\(pass.map(\.mlxActiveBytes).max() ?? 0),"
                            + "footprint:\(pass.map(\.footprintBytes).max() ?? 0)"
                    }.joined(separator: "; "))
        }

        private func runUntilFirstCompletedLayerThenStop(
            authority: ArtifactRuntimeAuthority, promptIDs: [Int],
            configuration: DeepSeekV4ProductGateConfiguration
        ) async throws -> (summary: RunSummary?, sawCompletedLayer: Bool, terminalEventCount: Int) {
            let requestedTokens = max(2, configuration.expectedTokenIDs.count)
            let runner = DeepSeekV4DecodeRunner(scale: configuration.runnerScale)
            let session = try runner.start(
                request(
                    authority: authority, promptIDs: promptIDs,
                    maximumNewTokens: requestedTokens,
                    configuration: configuration))
            var sawCompletedLayer = false
            var didRequestStop = false
            var cancelled: RunSummary?
            var terminalEvents = 0
            for try await event in session.events {
                switch event {
                case .phase(let phase):
                    if phase.fraction.map({ $0 > 0 }) == true {
                        sawCompletedLayer = true
                        if !didRequestStop {
                            didRequestStop = true
                            session.cancel()
                        }
                    }
                case .cancelled(let summary):
                    terminalEvents += 1
                    cancelled = summary
                case .finished:
                    terminalEvents += 1
                    XCTFail("Stop arrived too late; the real cancellation arm finished")
                case .accepted, .token, .telemetry, .log, .phaseSummary:
                    break
                }
            }
            XCTAssertTrue(didRequestStop)
            return (cancelled, sawCompletedLayer, terminalEvents)
        }

        private func request(
            authority: ArtifactRuntimeAuthority, promptIDs: [Int],
            maximumNewTokens: Int,
            configuration: DeepSeekV4ProductGateConfiguration
        ) -> RunRequest {
            RunRequest(
                model: .deepseekV4Flash,
                artifact: ArtifactReference(
                    root: configuration.artifactRoot,
                    runtimeAuthority: authority),
                prompt: .tokenIDs(promptIDs),
                memoryBudgetBytes: configuration.memoryBudgetBytes,
                maximumNewTokens: maximumNewTokens,
                knobs: configuration.expertKnobs,
                workingDirectory: configuration.outputDirectory)
        }

        private func productText(
            _ tokenIDs: [Int], vocabulary: DeepSeekV4Vocabulary
        ) -> String {
            let terminalIDs = Set(
                DeepSeekV4ChatControl.allCases.compactMap {
                    try? vocabulary.id(for: $0.rawValue)
                })
            return vocabulary.decode(
                Array(tokenIDs.prefix { !terminalIDs.contains($0) }))
        }

        /// The memory contract of whichever scale this arm ran, read back from
        /// the run's own JSON.
        ///
        /// The two scales make different promises and each is checked against
        /// its own. Product: the 2 GB floor, the complete bounded plan, and a
        /// zero pinned term — the arm the floor was accepted against, and
        /// nothing here may soften it. Stated: the dial's residency plan, whose
        /// identity must close, whose pinned bytes must fit the declared
        /// budget, and which must match what the tier actually held.
        private func assertScaleMemoryContract(
            _ data: Data, arm: String,
            configuration: DeepSeekV4ProductGateConfiguration
        ) throws {
            let record = try JSONDecoder().decode(GateRunRecord.self, from: data)
            let label = "[v4-\(configuration.scale.rawValue)-gate] \(arm)"
            XCTAssertEqual(
                record.declaredBudgetBytes, configuration.memoryBudgetBytes,
                "\(arm) recorded a budget other than the one it was given")
            XCTAssertTrue(record.budgetRespected, "\(arm) crossed its declared budget")
            XCTAssertLessThanOrEqual(
                record.peakFootprintBytes, record.declaredBudgetBytes,
                "\(arm) peaked above its declared budget")

            switch configuration.scale {
            case .product:
                XCTAssertNil(
                    record.pinPlan,
                    "the product scale must not reach the memory dial")
                XCTAssertNil(record.pinnedTier, "the product scale pins nothing")
                let plan = try XCTUnwrap(
                    record.productMemoryPlan,
                    "the product arm recorded no bounded memory plan")
                // The floor is the product scale's policy: it does not depend
                // on anything the conversation chose, and the exact terms stay
                // below it at every product request shape.
                XCTAssertEqual(
                    plan.requiredBudgetBytes,
                    DeepSeekV4ProductMemoryBudget.minimumBudgetBytes)
                XCTAssertEqual(plan.pinnedDeterministicBytes, 0)
                XCTAssertTrue(plan.isAdmitted)
                XCTAssertTrue(
                    plan.accountsForDeclaredBudget,
                    "the eight-term identity does not partition the declared budget")
                XCTAssertEqual(record.prefillPinnedServedBytes ?? 0, 0)
                XCTAssertEqual(record.decodePinnedServedBytes.reduce(0, +), 0)
            case .stated:
                // A stated run is admitted against its own reservation plus the
                // residency it planned. The 2 GB product floor and its plan are
                // deliberately absent, and their absence is the assertion.
                XCTAssertNil(
                    record.productMemoryPlan,
                    "a stated run must not be admitted against the product floor")
                let plan = try XCTUnwrap(
                    record.pinPlan, "\(arm) produced no residency plan")
                XCTAssertEqual(plan.budgetBytes, configuration.memoryBudgetBytes)
                XCTAssertEqual(plan.requestedMaximumNewTokens, record.requestedNewTokens)
                XCTAssertTrue(
                    plan.isResidencyBalanced,
                    "floor + pinned + hot set + unused does not equal the budget")
                XCTAssertTrue(
                    plan.isAccountingBalanced,
                    "the plan's per-token byte buckets do not balance")
                XCTAssertLessThanOrEqual(
                    plan.pinnedBytes, configuration.memoryBudgetBytes)
                XCTAssertEqual(
                    plan.expertHotSetBytes, 0,
                    "no V4 budget in the product range comes near pinning an expert")

                let servedPerPass = (record.prefillPinnedServedBytes ?? 0)
                    + record.decodePinnedServedBytes.reduce(0, +)
                if plan.pinnedBytes > 0 {
                    let tier = try XCTUnwrap(
                        record.pinnedTier,
                        "the plan pinned bytes but no tier reported holding them")
                    XCTAssertTrue(
                        tier.isAccountingBalanced,
                        "the pinned tier's residency or service identity does not hold")
                    XCTAssertEqual(tier.budgetBytes, plan.pinnedBytes)
                    XCTAssertEqual(tier.pinnedLayerCount, plan.pinnedLayers.count)
                    XCTAssertLessThanOrEqual(tier.residentBytes, plan.pinnedBytes)
                    // Bytes served inside a pass that never completed belong to
                    // no pass, so only a finished run's per-pass split accounts
                    // for the run-scoped counter exactly.
                    XCTAssertLessThanOrEqual(servedPerPass, tier.bytesServed)
                    if record.outcome == "finished" {
                        XCTAssertEqual(
                            servedPerPass, tier.bytesServed,
                            "the per-pass served bytes do not sum to what the tier served")
                    }
                    print("\(label): \(tier.summaryLine)")
                }
                print(
                    "\(label): dial pinned \(plan.pinnedLayers.count) of "
                        + "\(plan.layerCount.map(String.init) ?? "?") layers, "
                        + "\(plan.pinnedBytes) B resident over a "
                        + "\(plan.workingFloorBytes) B floor, "
                        + "\(plan.unusedBudgetBytes) B unused; served per pass "
                        + "\(record.decodePinnedServedBytes)")
            }

            // The decomposition is the reason this arm exists, so it is printed
            // and not only written: an arm whose output directory is lost still
            // leaves its attribution in the run log.
            if let prefill = record.prefillPhaseMetrics {
                print("\(label) prefill pass: \(prefill.summaryLine)")
            }
            for (index, metrics) in record.decodePhaseMetrics.enumerated() {
                print("\(label) decode pass \(index + 1): \(metrics.summaryLine)")
            }
        }

        /// The fields of the run's JSON this gate reads back. Decoding the
        /// published types rather than picking keys out of a dictionary is what
        /// lets the derived identities above be evaluated at all.
        private struct GateRunRecord: Decodable {
            let outcome: String
            let requestedNewTokens: Int
            let declaredBudgetBytes: UInt64
            let peakFootprintBytes: UInt64
            let budgetRespected: Bool
            let productMemoryPlan: DeepSeekV4ProductMemoryPlan?
            let pinPlan: PinPlan?
            let pinnedTier: DeepSeekV4PinnedTierMetrics?
            let prefillPhaseMetrics: DeepSeekV4PhaseMetrics?
            let decodePhaseMetrics: [DeepSeekV4PhaseMetrics]
            let prefillPinnedServedBytes: UInt64?
            let decodePinnedServedBytes: [UInt64]
        }

        private func assertReleasedBeforeTerminal(_ summary: RunSummary) throws {
            let data = try XCTUnwrap(summary.resultJSON)
            let object = try XCTUnwrap(
                JSONSerialization.jsonObject(with: data) as? [String: Any])
            let teardown = try XCTUnwrap(object["teardown"] as? [String: Any])
            XCTAssertEqual(teardown["workloadReleased"] as? Bool, true)
            XCTAssertEqual(teardown["mlxCacheBytesAfterClear"] as? UInt64, 0)
            XCTAssertEqual(summary.finalTelemetry.mlxCacheBytes, 0)
        }
    #endif
}

/// Which runner scale the armed gate exercises.
///
/// `product` is the default, and it is the gate as it stood: the 2 GB floor,
/// the complete product memory plan, the 512-token prompt ceiling, a zero MLX
/// cache and no pinned weights. `stated` declares the same budget as a stated
/// minimum, which is what makes the memory dial reachable.
private enum DeepSeekV4GateScale: String {
    case product
    case stated
}

private struct DeepSeekV4ProductGateConfiguration {
    let scale: DeepSeekV4GateScale
    let artifactRoot: URL
    let outputDirectory: URL
    let publicationRevision: String
    let sourceRevision: String
    let prompt: String
    let expectedTokenIDs: [Int]
    let expectedText: String
    let memoryBudgetBytes: UInt64
    let huggingFaceToken: String?
    /// Pager knobs stated by the arm, or an empty set when it states none.
    ///
    /// Unset is the runner's own default in every case, so an arm that states
    /// nothing runs exactly the configuration every previous record measured.
    /// A stated value that the runner refuses surfaces as the refusal, because
    /// these are passed through ``RunKnobs`` untouched — the harness does not
    /// clamp, and §5.3's reject-don't-clamp rule is the runner's to apply.
    ///
    /// | variable | knob |
    /// | --- | --- |
    /// | `MINIRUN_V4_EXPERT_READ_AHEAD` | `expertReadAhead`, default 2 |
    /// | `MINIRUN_V4_EXPERT_QUEUE_DEPTH` | `queueDepth`, default 2 |
    /// | `MINIRUN_V4_EXPERT_SLOTS` | `expertPoolSlots`, default 2 |
    /// | `MINIRUN_V4_EXPERT_PROJECTION_PREFETCH` | `expertProjectionPrefetch`, `0` or `1` |
    /// | `MINIRUN_V4_EXPERT_RUN_SCOPED` | `expertRunScopedBackends`, `0` or `1` |
    /// | `MINIRUN_V4_EXPERT_CROSS_LAYER` | `expertCrossLayerPrefetch`, tiles a not-yet-current layer may hold |
    /// | `MINIRUN_V4_EXPERT_ADOPT` | `expertTileAdoption`, `0` or `1` |
    ///
    /// Measured in `docs/experiments/2026-08-17-v4-expert-overlap.md`. Note
    /// that a wider pool is a *reserved* term — `expertPoolBudgetBytes` is
    /// subtracted before the memory dial plans — so at the 10.6 GB budget the
    /// earlier records used, which leaves 4,195,256 B unpinned, any arm that
    /// widens the pool also unpins a layer unless it raises its budget.
    let expertKnobs: RunKnobs

    static func load(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> DeepSeekV4ProductGateConfiguration {
        guard environment["MINIRUN_V4_PRODUCT_GATE"] == "1" else {
            throw XCTSkip(
                "set MINIRUN_V4_PRODUCT_GATE=1 only for the explicit 166 GB product gate")
        }
        func required(_ name: String) throws -> String {
            guard let value = environment[name], !value.isEmpty else {
                throw ProductGateConfigurationError.missing(name)
            }
            return value
        }
        let artifactRoot = URL(
            fileURLWithPath: try required("MINIRUN_V4_PRODUCT_ARTIFACT"),
            isDirectory: true).standardizedFileURL
        let outputDirectory = URL(
            fileURLWithPath: try required("MINIRUN_V4_PRODUCT_OUT"),
            isDirectory: true).standardizedFileURL
        let publicationRevision = try required(
            "MINIRUN_V4_PRODUCT_PUBLICATION_REVISION")
        let sourceRevision = try required("MINIRUN_V4_PRODUCT_SOURCE_REVISION")
        let prompt = try required("MINIRUN_V4_PRODUCT_PROMPT")
        let expectedText = try required("MINIRUN_V4_PRODUCT_EXPECTED_TEXT")
        let rawTokenIDs = try required("MINIRUN_V4_PRODUCT_EXPECTED_TOKEN_IDS")
        let expectedTokenIDs = rawTokenIDs.split(separator: ",").compactMap {
            Int($0.trimmingCharacters(in: .whitespaces))
        }
        guard !expectedTokenIDs.isEmpty,
            expectedTokenIDs.count == rawTokenIDs.split(separator: ",").count,
            expectedTokenIDs.allSatisfy({ $0 >= 0 })
        else {
            throw ProductGateConfigurationError.invalid(
                "MINIRUN_V4_PRODUCT_EXPECTED_TOKEN_IDS", rawTokenIDs)
        }
        let rawBudget = try required("MINIRUN_V4_PRODUCT_BUDGET_BYTES")
        guard let memoryBudgetBytes = UInt64(rawBudget), memoryBudgetBytes > 0 else {
            throw ProductGateConfigurationError.invalid(
                "MINIRUN_V4_PRODUCT_BUDGET_BYTES", rawBudget)
        }
        // Unset is `product`, so an arm that states nothing runs exactly the
        // gate that was recorded before this knob existed.
        let rawScale = environment["MINIRUN_V4_SCALE"] ?? DeepSeekV4GateScale.product.rawValue
        guard let scale = DeepSeekV4GateScale(rawValue: rawScale) else {
            throw ProductGateConfigurationError.invalid("MINIRUN_V4_SCALE", rawScale)
        }
        // The App refuses to turn a sub-floor budget into a stated minimum —
        // that would be the runner being talked into admitting a budget by
        // being handed it — and this harness derives its scale the same way.
        guard scale == .product
            || memoryBudgetBytes > DeepSeekV4ProductMemoryBudget.minimumBudgetBytes
        else {
            throw ProductGateConfigurationError.invalid(
                "MINIRUN_V4_SCALE=stated with MINIRUN_V4_PRODUCT_BUDGET_BYTES", rawBudget)
        }
        guard isImmutableRevision(publicationRevision) else {
            throw ProductGateConfigurationError.invalid(
                "MINIRUN_V4_PRODUCT_PUBLICATION_REVISION", publicationRevision)
        }
        guard isImmutableRevision(sourceRevision) else {
            throw ProductGateConfigurationError.invalid(
                "MINIRUN_V4_PRODUCT_SOURCE_REVISION", sourceRevision)
        }
        func optionalPositive(_ name: String) throws -> Int? {
            guard let raw = environment[name], !raw.isEmpty else { return nil }
            guard let value = Int(raw) else {
                throw ProductGateConfigurationError.invalid(name, raw)
            }
            return value
        }
        var expertKnobs = RunKnobs()
        expertKnobs.expertReadAhead = try optionalPositive("MINIRUN_V4_EXPERT_READ_AHEAD")
        expertKnobs.queueDepth = try optionalPositive("MINIRUN_V4_EXPERT_QUEUE_DEPTH")
        expertKnobs.expertPoolSlots = try optionalPositive("MINIRUN_V4_EXPERT_SLOTS")
        func optionalFlag(_ name: String) throws -> Bool? {
            guard let raw = environment[name], !raw.isEmpty else { return nil }
            guard raw == "0" || raw == "1" else {
                throw ProductGateConfigurationError.invalid(name, raw)
            }
            return raw == "1"
        }
        expertKnobs.expertProjectionPrefetch = try optionalFlag(
            "MINIRUN_V4_EXPERT_PROJECTION_PREFETCH")
        expertKnobs.expertRunScopedBackends = try optionalFlag(
            "MINIRUN_V4_EXPERT_RUN_SCOPED")
        expertKnobs.expertTileAdoption = try optionalFlag("MINIRUN_V4_EXPERT_ADOPT")
        if let raw = environment["MINIRUN_V4_EXPERT_CROSS_LAYER"], !raw.isEmpty {
            guard let value = Int(raw), value >= 0 else {
                throw ProductGateConfigurationError.invalid(
                    "MINIRUN_V4_EXPERT_CROSS_LAYER", raw)
            }
            expertKnobs.expertCrossLayerPrefetch = value
        }

        try FileManager.default.createDirectory(
            at: outputDirectory, withIntermediateDirectories: true)
        return DeepSeekV4ProductGateConfiguration(
            scale: scale,
            artifactRoot: artifactRoot, outputDirectory: outputDirectory,
            publicationRevision: publicationRevision,
            sourceRevision: sourceRevision, prompt: prompt,
            expectedTokenIDs: expectedTokenIDs, expectedText: expectedText,
            memoryBudgetBytes: memoryBudgetBytes,
            huggingFaceToken: environment["HF_TOKEN"],
            expertKnobs: expertKnobs)
    }

    /// The runner scale this arm constructs, derived exactly as the App's
    /// `DeepSeekV4ProductRuntimeProvider.runnerScale(statedBudgetBytes:)`
    /// derives it: the declared budget becomes the stated minimum, and the
    /// token ceiling stays the product ceiling at either scale, because the
    /// dial states memory and is not an authorization to generate past the
    /// boundary the V4 product path has evidence for.
    var runnerScale: DeepSeekV4DecodeRunner.Scale {
        switch scale {
        case .product:
            return .product
        case .stated:
            return .stated(
                minimumBudgetBytes: memoryBudgetBytes,
                maximumNewTokens: DeepSeekV4ProductMemoryBudget.maximumNewTokens)
        }
    }

    /// Result names carry the scale, so one before/after directory can hold a
    /// product arm and a stated arm without either overwriting the other.
    func resultFilename(_ arm: String) -> String {
        switch scale {
        case .product:
            return "deepseek-v4-product-\(arm).json"
        case .stated:
            return "deepseek-v4-product-\(arm)-stated.json"
        }
    }

    func write(_ data: Data, filename: String) throws {
        try data.write(
            to: outputDirectory.appendingPathComponent(filename),
            options: .atomic)
    }

    func optionalMemoryPromptTokenCount(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> Int? {
        guard let raw = environment["MINIRUN_V4_PRODUCT_MEMORY_PROMPT_TOKENS"] else {
            return nil
        }
        guard let count = Int(raw),
            (1...DeepSeekV4ProductMemoryBudget.maximumPromptTokens).contains(count)
        else {
            throw ProductGateConfigurationError.invalid(
                "MINIRUN_V4_PRODUCT_MEMORY_PROMPT_TOKENS", raw)
        }
        return count
    }

    func optionalMemoryNewTokenCount(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> Int? {
        guard let raw = environment["MINIRUN_V4_PRODUCT_MEMORY_NEW_TOKENS"] else {
            return nil
        }
        guard let count = Int(raw),
            (3...DeepSeekV4ProductMemoryBudget.maximumNewTokens).contains(count)
        else {
            throw ProductGateConfigurationError.invalid(
                "MINIRUN_V4_PRODUCT_MEMORY_NEW_TOKENS", raw)
        }
        return count
    }

    private static func isImmutableRevision(_ value: String) -> Bool {
        value.utf8.count == 40 && value.utf8.allSatisfy {
            (48...57).contains($0) || (97...102).contains($0)
        }
    }
}

/// The scale knob is parsed on every developer Mac, from a synthetic
/// environment and no artifact at all.
///
/// An arm that verifies 166 GB and only then discovers it was pointed at the
/// wrong scale has already spent half an hour, and the gate's own environment
/// is not something a run can be asked to check for itself.
final class DeepSeekV4GateScaleTests: XCTestCase {
    private var outputDirectory: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        outputDirectory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-v4-gate-scale-\(UUID().uuidString)")
    }

    override func tearDownWithError() throws {
        if let outputDirectory {
            try? FileManager.default.removeItem(at: outputDirectory)
        }
        try super.tearDownWithError()
    }

    func testAnUnstatedScaleIsProductAndAStatedOneCarriesTheDeclaredBudget() throws {
        let byDefault = try DeepSeekV4ProductGateConfiguration.load(
            environment: armedEnvironment())
        XCTAssertEqual(byDefault.scale, .product)
        XCTAssertEqual(byDefault.runnerScale, .product)
        XCTAssertEqual(
            byDefault.resultFilename("completion"),
            "deepseek-v4-product-completion.json",
            "the default arm must write the names the recorded gate wrote")

        let stated = try DeepSeekV4ProductGateConfiguration.load(
            environment: armedEnvironment(
                scale: "stated", budgetBytes: "8000000000"))
        XCTAssertEqual(stated.scale, .stated)
        XCTAssertEqual(stated.memoryBudgetBytes, 8_000_000_000)
        XCTAssertEqual(
            stated.runnerScale,
            .stated(
                minimumBudgetBytes: 8_000_000_000,
                maximumNewTokens: DeepSeekV4ProductMemoryBudget.maximumNewTokens))
        XCTAssertEqual(
            stated.resultFilename("completion"),
            "deepseek-v4-product-completion-stated.json")
        XCTAssertEqual(
            stated.resultFilename("memory-512"),
            "deepseek-v4-product-memory-512-stated.json")
    }

    func testAnExplicitProductScaleIsTheDefaultArmExactly() throws {
        let explicit = try DeepSeekV4ProductGateConfiguration.load(
            environment: armedEnvironment(scale: "product"))
        XCTAssertEqual(explicit.runnerScale, .product)
        XCTAssertEqual(
            explicit.resultFilename("stop"), "deepseek-v4-product-stop.json")
    }

    func testAnUnknownScaleIsRefusedByName() {
        XCTAssertThrowsError(
            try DeepSeekV4ProductGateConfiguration.load(
                environment: armedEnvironment(scale: "Stated"))
        ) { error in
            XCTAssertEqual(
                error as? ProductGateConfigurationError,
                .invalid("MINIRUN_V4_SCALE", "Stated"))
        }
    }

    func testASubFloorStatedBudgetIsRefusedRatherThanAdmitted() {
        let floor = DeepSeekV4ProductMemoryBudget.minimumBudgetBytes
        XCTAssertThrowsError(
            try DeepSeekV4ProductGateConfiguration.load(
                environment: armedEnvironment(
                    scale: "stated", budgetBytes: "\(floor)"))
        ) { error in
            XCTAssertEqual(
                error as? ProductGateConfigurationError,
                .invalid(
                    "MINIRUN_V4_SCALE=stated with MINIRUN_V4_PRODUCT_BUDGET_BYTES",
                    "\(floor)"))
        }
    }

    func testTheGateStaysOptInWhateverScaleIsStated() {
        var environment = armedEnvironment(scale: "stated", budgetBytes: "8000000000")
        environment["MINIRUN_V4_PRODUCT_GATE"] = nil
        XCTAssertThrowsError(
            try DeepSeekV4ProductGateConfiguration.load(environment: environment)
        ) { XCTAssertTrue($0 is XCTSkip) }
    }

    private func armedEnvironment(
        scale: String? = nil, budgetBytes: String = "2000000000"
    ) -> [String: String] {
        var environment = [
            "MINIRUN_V4_PRODUCT_GATE": "1",
            "MINIRUN_V4_PRODUCT_ARTIFACT": outputDirectory.appendingPathComponent(
                "artifact"
            ).path,
            "MINIRUN_V4_PRODUCT_OUT": outputDirectory.path,
            "MINIRUN_V4_PRODUCT_PUBLICATION_REVISION":
                "04c752c3fc26670992254c68bae45fb7face7859",
            "MINIRUN_V4_PRODUCT_SOURCE_REVISION":
                "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
            "MINIRUN_V4_PRODUCT_PROMPT": "中国的首都是",
            "MINIRUN_V4_PRODUCT_EXPECTED_TOKEN_IDS": "17607,2793",
            "MINIRUN_V4_PRODUCT_EXPECTED_TEXT": "中国的首",
            "MINIRUN_V4_PRODUCT_BUDGET_BYTES": budgetBytes,
        ]
        environment["MINIRUN_V4_SCALE"] = scale
        return environment
    }
}

private enum ProductGateConfigurationError: Error, LocalizedError, Equatable {
    case missing(String)
    case invalid(String, String)

    var errorDescription: String? {
        switch self {
        case .missing(let name):
            return "the armed V4 product gate requires \(name)"
        case .invalid(let name, let value):
            return "the armed V4 product gate received invalid \(name)=\(value)"
        }
    }
}
