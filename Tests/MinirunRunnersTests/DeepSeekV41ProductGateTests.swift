import Foundation
import MinirunKit
import ModelAdapters
import XCTest

@testable import MinirunRunners

/// The opt-in, real-artifact exit gate for DeepSeek V4.1 product registration.
///
/// Ordinary test runs skip before touching the network or the drive. An armed
/// run performs complete repository verification of all 517 GB, then makes the
/// same rooted authority feed the tokenizer and the production runner. Reference
/// output is supplied alongside the arm rather than compiled into the App, for
/// the reason ``DeepSeekV4ProductGateTests`` gives: the owner may publish a
/// newer compatible artifact, and an observation from one immutable
/// source/publication pair cannot silently authorize the next pair.
///
/// The gate runs at either runner scale. `MINIRUN_V41_SCALE=product` is the
/// default — the 2.9 GB floor, the complete bounded plan, and no pinned
/// weights. `MINIRUN_V41_SCALE=stated` runs the same arms against a declared
/// budget above that floor, so the memory dial's residency plan and the per-pass
/// phase decomposition are recorded for a run that actually pins.
///
/// Build the Release test bundle, then invoke this through `xcrun xctest` so the
/// environment reaches the process. `Tools/v41_flash/run_stated_arm.sh` is the
/// script that does it.
final class DeepSeekV41ProductGateTests: XCTestCase {
    func testCurrentVerifiedArtifactMatchesReferenceAndStopsSafely() async throws {
        #if !arch(arm64)
            throw XCTSkip("the V4.1 MLX runtime requires Apple silicon")
        #else
            let configuration = try DeepSeekV41ProductGateConfiguration.load()
            let tokenProvider = configuration.huggingFaceToken.map { token in
                { @Sendable in token as String? }
            }
            let transport = URLSessionTransport()
            let catalog = try await HuggingFaceCatalogSource(
                transport: transport, token: tokenProvider
            ).fetch()
            let descriptor = try XCTUnwrap(catalog.descriptor(.deepseekV41Flash))
            let publication = try XCTUnwrap(descriptor.source.repo)
            XCTAssertEqual(
                publication.repoID.caseInsensitiveCompare(
                    "nanguoyu/DeepSeek-V4.1-Flash-minirun"),
                .orderedSame)
            XCTAssertEqual(publication.revision, configuration.publicationRevision)

            let ledger = InMemoryVerificationLedger()
            let location = ArtifactLocator(
                catalog: catalog, verificationLedger: ledger, maximumDepth: 0
            ).scan(configuration.artifactRoot)
            let artifact = try XCTUnwrap(
                location.artifacts.first(where: { $0.model == .deepseekV41Flash }),
                "the armed path does not contain a catalog-matched V4.1 artifact")
            XCTAssertEqual(
                artifact.rootPath, configuration.artifactRoot.standardizedFileURL.path)
            XCTAssertEqual(
                artifact.index.repositories.first?.revision, configuration.sourceRevision)
            XCTAssertEqual(
                artifact.isComplete, true,
                "the real gate refuses a short or metadata-stale local tree")

            let rooted = try ArtifactVerificationRoot.open(
                relativeComponents: [], beneath: configuration.artifactRoot)
            let verifier = ArtifactVerifier(
                tree: HuggingFaceTreeClient(transport: transport, token: tokenProvider),
                catalog: catalog, ledger: ledger)
            let report = try await verifier.verify(
                artifact, configuration.verificationRequest, rootedAt: rooted,
                onProgress: { progress in
                    guard progress.filesChecked == progress.filesToCheck
                        || progress.filesChecked.isMultiple(of: 16)
                    else { return }
                    print(
                        "[v41-product-gate] verified \(progress.filesChecked)/"
                            + "\(progress.filesToCheck) files, \(progress.bytesChecked)/"
                            + "\(progress.bytesToCheck) bytes")
                })
            XCTAssertTrue(report.isComplete, "verification returned a refusal report")

            let record = try XCTUnwrap(
                ledger.record(
                    matching: ArtifactVerificationLookup(
                        rootPath: artifact.rootPath, model: artifact.model,
                        repository: publication, index: artifact.index)),
                "the verification pass did not leave applicable evidence")
            let evidence = try XCTUnwrap(record.evidence)
            let authority = try ArtifactRuntimeAuthority(root: rooted, evidence: evidence)

            let vocabulary = try DeepSeekV4Vocabulary(
                data: authority.openFile(
                    try XCTUnwrap(evidence.index.tokenizer).file
                ).readAll(maximumBytes: 32 << 20))
            let promptIDs = try DeepSeekV41NoThinkingChatPrompt(
                messages: [.init(role: .user, content: configuration.prompt)]
            ).encode(using: vocabulary)
            print("[v41-product-gate] prompt ids \(promptIDs)")

            let completed = try await runToCompletion(
                authority: authority, promptIDs: promptIDs,
                maximumNewTokens: configuration.newTokenCount,
                configuration: configuration)
            print(
                "[v41-product-gate] generated \(completed.summary.tokenIDs); text "
                    + "\"\(productText(completed.summary.tokenIDs, vocabulary: vocabulary))\"; "
                    + "digest \(completed.summary.logitsDigest?.hex ?? "none")")

            // The ids are the gate. An arm that states them asserts them; one
            // that does not is a *recording* arm and says so by refusing to
            // pretend otherwise — it still writes the run record, which is what
            // an auditor compares against the cluster reference.
            if let expected = configuration.expectedTokenIDs {
                XCTAssertEqual(
                    Array(completed.summary.tokenIDs.prefix(expected.count)), expected,
                    "the run diverged from the reference prefix")
            }
            if let expectedText = configuration.expectedText {
                XCTAssertEqual(
                    productText(completed.summary.tokenIDs, vocabulary: vocabulary),
                    expectedText)
            }
            let endOfSentence = try vocabulary.id(
                for: DeepSeekV41ChatControl.endOfSentence.rawValue)
            if completed.summary.tokenIDs.count < configuration.newTokenCount {
                XCTAssertEqual(
                    completed.summary.tokenIDs.last, endOfSentence,
                    "the arm stopped short without ending at end-of-sentence")
            }
            XCTAssertLessThanOrEqual(
                completed.summary.peakFootprintBytes, configuration.memoryBudgetBytes)
            XCTAssertTrue(completed.summary.budgetRespected)
            try assertScaleMemoryContract(
                completed.resultJSON, arm: "completion", configuration: configuration)
            try configuration.write(
                completed.resultJSON, filename: configuration.resultFilename("completion"))

            let stopped = try await runUntilFirstCompletedBlockThenStop(
                authority: authority, promptIDs: promptIDs, configuration: configuration)
            XCTAssertTrue(stopped.sawCompletedBlock)
            XCTAssertEqual(stopped.terminalEventCount, 1)
            let cancelled = try XCTUnwrap(stopped.summary)
            XCTAssertLessThanOrEqual(
                cancelled.peakFootprintBytes, configuration.memoryBudgetBytes)
            XCTAssertTrue(cancelled.budgetRespected)
            let stopJSON = try XCTUnwrap(cancelled.resultJSON)
            try assertReleasedBeforeTerminal(stopJSON)
            try assertScaleMemoryContract(
                stopJSON, arm: "stop", configuration: configuration)
            try configuration.write(
                stopJSON, filename: configuration.resultFilename("stop"))
        #endif
    }

    #if arch(arm64)
        private func runToCompletion(
            authority: ArtifactRuntimeAuthority, promptIDs: [Int],
            maximumNewTokens: Int,
            configuration: DeepSeekV41ProductGateConfiguration
        ) async throws -> (summary: RunSummary, resultJSON: Data, telemetry: [RunTelemetry]) {
            let runner = DeepSeekV41DecodeRunner(scale: configuration.runnerScale)
            let session = try runner.start(
                request(
                    authority: authority, promptIDs: promptIDs,
                    maximumNewTokens: maximumNewTokens, configuration: configuration))
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
                    XCTFail("the completion arm was cancelled")
                case .telemetry(let sample):
                    telemetry.append(sample)
                case .log(let line):
                    print("[v41-product-gate] \(line)")
                case .phase, .phaseSummary:
                    break
                }
            }
            XCTAssertTrue(accepted)
            XCTAssertEqual(terminalEvents, 1)
            let summary = try XCTUnwrap(finished)
            XCTAssertEqual(tokens, summary.tokenIDs)
            return (summary, try XCTUnwrap(summary.resultJSON), telemetry)
        }

        private func runUntilFirstCompletedBlockThenStop(
            authority: ArtifactRuntimeAuthority, promptIDs: [Int],
            configuration: DeepSeekV41ProductGateConfiguration
        ) async throws -> (summary: RunSummary?, sawCompletedBlock: Bool, terminalEventCount: Int)
        {
            let runner = DeepSeekV41DecodeRunner(scale: configuration.runnerScale)
            let session = try runner.start(
                request(
                    authority: authority, promptIDs: promptIDs,
                    maximumNewTokens: max(2, configuration.newTokenCount),
                    configuration: configuration))
            var sawCompletedBlock = false
            var didRequestStop = false
            var cancelled: RunSummary?
            var terminalEvents = 0
            for try await event in session.events {
                switch event {
                case .phase(let phase):
                    if phase.fraction.map({ $0 > 0 }) == true {
                        sawCompletedBlock = true
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
                    XCTFail("Stop arrived too late; the cancellation arm finished")
                case .accepted, .token, .telemetry, .log, .phaseSummary:
                    break
                }
            }
            XCTAssertTrue(didRequestStop)
            return (cancelled, sawCompletedBlock, terminalEvents)
        }

        private func request(
            authority: ArtifactRuntimeAuthority, promptIDs: [Int],
            maximumNewTokens: Int,
            configuration: DeepSeekV41ProductGateConfiguration
        ) -> RunRequest {
            RunRequest(
                model: .deepseekV41Flash,
                artifact: ArtifactReference(
                    root: configuration.artifactRoot, runtimeAuthority: authority),
                prompt: .tokenIDs(promptIDs),
                memoryBudgetBytes: configuration.memoryBudgetBytes,
                maximumNewTokens: maximumNewTokens,
                knobs: configuration.knobs,
                workingDirectory: configuration.outputDirectory)
        }

        private func productText(
            _ tokenIDs: [Int], vocabulary: DeepSeekV4Vocabulary
        ) -> String {
            let terminalIDs = Set(
                DeepSeekV41ChatControl.allCases.compactMap {
                    try? vocabulary.id(for: $0.rawValue)
                })
            return vocabulary.decode(Array(tokenIDs.prefix { !terminalIDs.contains($0) }))
        }

        /// The memory contract of whichever scale this arm ran, read back from
        /// the run's own JSON — each scale against its own promise.
        private func assertScaleMemoryContract(
            _ data: Data, arm: String,
            configuration: DeepSeekV41ProductGateConfiguration
        ) throws {
            let record = try JSONDecoder().decode(GateRunRecord.self, from: data)
            XCTAssertEqual(
                record.declaredBudgetBytes, configuration.memoryBudgetBytes,
                "\(arm) recorded a budget other than the one it was given")
            XCTAssertTrue(record.budgetRespected, "\(arm) crossed its declared budget")
            XCTAssertLessThanOrEqual(
                record.peakFootprintBytes, record.declaredBudgetBytes)

            switch configuration.scale {
            case .product:
                XCTAssertNil(
                    record.pinPlan, "the product scale must not reach the memory dial")
                XCTAssertEqual(record.pinnedBlockCount, 0, "the product scale pins nothing")
                let plan = try XCTUnwrap(
                    record.productMemoryPlan, "the product arm recorded no bounded plan")
                XCTAssertEqual(plan.pinnedDeterministicBytes, 0)
                XCTAssertTrue(plan.isAdmitted)
                XCTAssertTrue(
                    plan.accountsForDeclaredBudget,
                    "the eight-term identity does not partition the declared budget")
            case .stated:
                XCTAssertNil(
                    record.productMemoryPlan,
                    "a stated run must not be admitted against the product floor")
                let plan = try XCTUnwrap(record.pinPlan, "\(arm) produced no residency plan")
                XCTAssertEqual(plan.budgetBytes, configuration.memoryBudgetBytes)
                XCTAssertTrue(
                    plan.isResidencyBalanced,
                    "floor + pinned + hot set + unused does not equal the budget")
                XCTAssertEqual(
                    record.pinnedBlockCount, plan.pinnedLayers.count,
                    "the tier held a different number of blocks than the dial planned")
                XCTAssertEqual(
                    record.pinnedOutputHead, plan.pinnedGlobals?.unit == .outputHead)
            }
            print(
                "[v41-\(configuration.scale.rawValue)-gate] \(arm): peak "
                    + "\(record.peakFootprintBytes) B of \(record.declaredBudgetBytes) B; "
                    + "\(record.pinnedBlockCount) blocks pinned; tiles "
                    + "\(record.expertTileReads) read / \(record.expertTileHits) resident")
        }

        private func assertReleasedBeforeTerminal(_ data: Data) throws {
            let record = try JSONDecoder().decode(GateRunRecord.self, from: data)
            XCTAssertTrue(record.teardown.workloadWasPrepared)
            XCTAssertTrue(record.teardown.workloadReleased)
        }

        /// The subset of the run record this gate reads back. Deliberately not
        /// the engine's own `Payload` type: a gate that decoded the writer's
        /// struct would pass whatever the writer emitted.
        private struct GateRunRecord: Decodable {
            let declaredBudgetBytes: UInt64
            let peakFootprintBytes: UInt64
            let budgetRespected: Bool
            let requestedNewTokens: Int
            let generatedTokenIDs: [Int]
            let logitsSHA256: String?
            let productMemoryPlan: DeepSeekV41ProductMemoryPlan?
            let pinPlan: PinPlan?
            let pinnedBlockCount: Int
            let pinnedOutputHead: Bool
            let expertTileReads: Int
            let expertTileHits: Int
            let teardown: DeepSeekV41RunEngine.TeardownRecord
        }
    #endif
}

enum DeepSeekV41GateScale: String {
    case product
    case stated
}

enum V41ProductGateConfigurationError: Error, CustomStringConvertible {
    case missing(String)
    case invalid(String, String)

    var description: String {
        switch self {
        case .missing(let name): "the V4.1 product gate needs \(name)"
        case .invalid(let name, let value): "\(name)=\(value) is not usable"
        }
    }
}

/// Everything the armed gate reads from the environment.
///
/// | variable | meaning |
/// | --- | --- |
/// | `MINIRUN_V41_PRODUCT_GATE` | `1`, and nothing else arms the gate |
/// | `MINIRUN_V41_PRODUCT_ARTIFACT` | the container root |
/// | `MINIRUN_V41_PRODUCT_OUT` | where the run records are written |
/// | `MINIRUN_V41_PRODUCT_PUBLICATION_REVISION` | the immutable publication revision |
/// | `MINIRUN_V41_PRODUCT_SOURCE_REVISION` | the immutable upstream revision |
/// | `MINIRUN_V41_PRODUCT_PROMPT` | the user turn |
/// | `MINIRUN_V41_PRODUCT_BUDGET_BYTES` | the declared memory ceiling |
/// | `MINIRUN_V41_PRODUCT_NEW_TOKENS` | tokens to ask for, default 40 |
/// | `MINIRUN_V41_PRODUCT_EXPECTED_TOKEN_IDS` | optional; asserted when stated |
/// | `MINIRUN_V41_PRODUCT_EXPECTED_TEXT` | optional; asserted when stated |
/// | `MINIRUN_V41_SCALE` | `product` (default) or `stated` |
/// | `MINIRUN_V41_VERIFY_DEPTH` | `full` (default) or `metadata` |
/// | `MINIRUN_V41_EXPERT_READ_AHEAD` / `_QUEUE_DEPTH` / `_SLOTS` | pool knobs |
/// | `MINIRUN_V41_EXPERT_TILE_ADOPTION` | `1`/`0`; the gather's transfer mode |
/// | `MINIRUN_V41_LOGIT_CHUNK_ROWS` | output-head window |
///
/// The expected ids are **optional**, which V4's are not, and that is the whole
/// difference between a gate and the arm that produces one: the first V4.1 run
/// has nothing to be held to, and ADR 0021 says how the digest it produces
/// becomes the thing the next run is held to.
struct DeepSeekV41ProductGateConfiguration {
    let scale: DeepSeekV41GateScale
    let artifactRoot: URL
    let outputDirectory: URL
    let publicationRevision: String
    let sourceRevision: String
    let prompt: String
    let newTokenCount: Int
    let expectedTokenIDs: [Int]?
    let expectedText: String?
    let memoryBudgetBytes: UInt64
    let huggingFaceToken: String?
    let knobs: RunKnobs
    let verificationRequest: ArtifactVerificationRequest

    static func load(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> DeepSeekV41ProductGateConfiguration {
        guard environment["MINIRUN_V41_PRODUCT_GATE"] == "1" else {
            throw XCTSkip(
                "set MINIRUN_V41_PRODUCT_GATE=1 only for the explicit 517 GB product gate")
        }
        func required(_ name: String) throws -> String {
            guard let value = environment[name], !value.isEmpty else {
                throw V41ProductGateConfigurationError.missing(name)
            }
            return value
        }
        let artifactRoot = URL(
            fileURLWithPath: try required("MINIRUN_V41_PRODUCT_ARTIFACT"),
            isDirectory: true).standardizedFileURL
        let outputDirectory = URL(
            fileURLWithPath: try required("MINIRUN_V41_PRODUCT_OUT"),
            isDirectory: true).standardizedFileURL
        let publicationRevision = try required("MINIRUN_V41_PRODUCT_PUBLICATION_REVISION")
        let sourceRevision = try required("MINIRUN_V41_PRODUCT_SOURCE_REVISION")
        let prompt = try required("MINIRUN_V41_PRODUCT_PROMPT")
        let rawBudget = try required("MINIRUN_V41_PRODUCT_BUDGET_BYTES")
        guard let memoryBudgetBytes = UInt64(rawBudget), memoryBudgetBytes > 0 else {
            throw V41ProductGateConfigurationError.invalid(
                "MINIRUN_V41_PRODUCT_BUDGET_BYTES", rawBudget)
        }
        var newTokens = 40
        if let raw = environment["MINIRUN_V41_PRODUCT_NEW_TOKENS"], !raw.isEmpty {
            guard let value = Int(raw), value >= 1 else {
                throw V41ProductGateConfigurationError.invalid(
                    "MINIRUN_V41_PRODUCT_NEW_TOKENS", raw)
            }
            newTokens = value
        }
        var expectedTokenIDs: [Int]?
        if let raw = environment["MINIRUN_V41_PRODUCT_EXPECTED_TOKEN_IDS"], !raw.isEmpty {
            let parsed = raw.split(separator: ",").compactMap {
                Int($0.trimmingCharacters(in: .whitespaces))
            }
            guard parsed.count == raw.split(separator: ",").count,
                parsed.allSatisfy({ $0 >= 0 })
            else {
                throw V41ProductGateConfigurationError.invalid(
                    "MINIRUN_V41_PRODUCT_EXPECTED_TOKEN_IDS", raw)
            }
            expectedTokenIDs = parsed
        }
        let rawScale = environment["MINIRUN_V41_SCALE"] ?? DeepSeekV41GateScale.product.rawValue
        guard let scale = DeepSeekV41GateScale(rawValue: rawScale) else {
            throw V41ProductGateConfigurationError.invalid("MINIRUN_V41_SCALE", rawScale)
        }
        // The App refuses to turn a sub-floor budget into a stated minimum, and
        // this harness derives its scale the same way.
        guard scale == .product
            || memoryBudgetBytes > DeepSeekV41ProductMemoryBudget.minimumBudgetBytes
        else {
            throw V41ProductGateConfigurationError.invalid(
                "MINIRUN_V41_SCALE=stated with MINIRUN_V41_PRODUCT_BUDGET_BYTES", rawBudget)
        }
        guard isImmutableRevision(publicationRevision) else {
            throw V41ProductGateConfigurationError.invalid(
                "MINIRUN_V41_PRODUCT_PUBLICATION_REVISION", publicationRevision)
        }
        guard isImmutableRevision(sourceRevision) else {
            throw V41ProductGateConfigurationError.invalid(
                "MINIRUN_V41_PRODUCT_SOURCE_REVISION", sourceRevision)
        }
        func optionalPositive(_ name: String) throws -> Int? {
            guard let raw = environment[name], !raw.isEmpty else { return nil }
            guard let value = Int(raw) else {
                throw V41ProductGateConfigurationError.invalid(name, raw)
            }
            return value
        }
        // Refused rather than coerced, for the reason every other knob here is:
        // an arm recorded under a transfer mode it did not use is worse than an
        // arm that did not start.
        func optionalFlag(_ name: String) throws -> Bool? {
            guard let raw = environment[name], !raw.isEmpty else { return nil }
            switch raw {
            case "1", "true", "on": return true
            case "0", "false", "off": return false
            default: throw V41ProductGateConfigurationError.invalid(name, raw)
            }
        }
        var knobs = RunKnobs()
        knobs.expertReadAhead = try optionalPositive("MINIRUN_V41_EXPERT_READ_AHEAD")
        knobs.queueDepth = try optionalPositive("MINIRUN_V41_EXPERT_QUEUE_DEPTH")
        knobs.expertPoolSlots = try optionalPositive("MINIRUN_V41_EXPERT_SLOTS")
        knobs.logitChunkRows = try optionalPositive("MINIRUN_V41_LOGIT_CHUNK_ROWS")
        knobs.expertTileAdoption = try optionalFlag("MINIRUN_V41_EXPERT_TILE_ADOPTION")

        // Full is the default and the only request a *gate* may use: it digests
        // every published file, which for V4.1 is 517 GB. A `spot` arm reads a
        // couple of gigabytes and is for iterating on the runner, never for
        // producing a reference -- the record's name carries the depth it used,
        // so an arm that spot-checked cannot be mistaken for one that verified.
        let rawDepth = environment["MINIRUN_V41_VERIFY_DEPTH"] ?? "full"
        let request: ArtifactVerificationRequest
        switch rawDepth {
        case "full": request = .full
        case "spot": request = .spotCheck()
        default:
            throw V41ProductGateConfigurationError.invalid(
                "MINIRUN_V41_VERIFY_DEPTH", rawDepth)
        }

        try FileManager.default.createDirectory(
            at: outputDirectory, withIntermediateDirectories: true)
        return DeepSeekV41ProductGateConfiguration(
            scale: scale, artifactRoot: artifactRoot, outputDirectory: outputDirectory,
            publicationRevision: publicationRevision, sourceRevision: sourceRevision,
            prompt: prompt, newTokenCount: newTokens,
            expectedTokenIDs: expectedTokenIDs,
            // Empty is *unstated*, not "expect the empty string". A shell
            // passes an unset variable through as an empty one, and an arm that
            // read that as an expectation would fail every recording run
            // against the text it had just produced.
            expectedText: environment["MINIRUN_V41_PRODUCT_EXPECTED_TEXT"]
                .flatMap { $0.isEmpty ? nil : $0 },
            memoryBudgetBytes: memoryBudgetBytes,
            huggingFaceToken: environment["HF_TOKEN"],
            knobs: knobs, verificationRequest: request)
    }

    var runnerScale: DeepSeekV41DecodeRunner.Scale {
        switch scale {
        case .product:
            return .product
        case .stated:
            return .stated(
                minimumBudgetBytes: memoryBudgetBytes,
                maximumNewTokens: DeepSeekV41ProductMemoryBudget.maximumNewTokens)
        }
    }

    func resultFilename(_ arm: String) -> String {
        let depth: String
        switch verificationRequest {
        case .full: depth = ""
        case .spotCheck: depth = "-spotcheck"
        }
        return "deepseek-v41-product-\(arm)-\(scale.rawValue)\(depth).json"
    }

    func write(_ data: Data, filename: String) throws {
        try data.write(
            to: outputDirectory.appendingPathComponent(filename), options: .atomic)
    }

    /// A 40-hex-character commit. A branch name would let the gate's evidence
    /// describe a moving target.
    static func isImmutableRevision(_ value: String) -> Bool {
        value.count == 40
            && value.allSatisfy { $0.isHexDigit && ($0.isNumber || $0.isLowercase) }
    }
}
