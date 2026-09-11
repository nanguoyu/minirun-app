import Foundation
import MinirunKit
import ModelAdapters
import XCTest

@testable import MinirunRunners

/// **The product path the memory dial now opens, end to end, without launching
/// the app.**
///
/// `DeepSeekV41ProductGateTests` answers "does this artifact decode": it states
/// a budget and runs it. This answers the question the dial raised — *does the
/// budget the app would choose produce the same chat, and does it actually pin
/// what the plan promised* — by taking the budget from the same arithmetic the
/// app's Chat settings takes it from rather than from the environment:
///
/// 1. `DeepSeekV41MemoryDialInputs.inspect(_:)` over the rooted authority, which
///    is literally the call `DeepSeekV41ProductRuntimeProvider.registration`
///    makes in `inspectArtifact`;
/// 2. `budgetThatPinsEverything(census:floor:)`, which is what
///    `BudgetPlan.budget(for: .balanced)` returns whenever the whole ladder fits
///    under three-fifths of the device ceiling — 14.87 GB against this Mac's
///    20.6 GB, so it does;
/// 3. the scale `DeepSeekV41ProductRuntimeProvider.runnerScale(statedBudgetBytes:)`
///    yields for a budget above the product floor, which is `.stated` carrying
///    that budget and the product token ceiling.
///
/// **Why it is here and not in `Apps/Minirun/Tests`.** The app's test bundle is
/// hosted by `Minirun.app`, which is sandboxed to user-selected files; a test
/// process inside it cannot open `/Volumes/K3NVME` without a security-scoped
/// bookmark a test has no way to grant itself. So the app half of this pair is
/// `DeepSeekV41ProductRuntimeTests.testTheDialsBalancedBudgetIsTheScaleTheRunnerIsBuiltAt`,
/// which asserts the same two numbers from `BudgetPlan` itself, and the two
/// halves meet at the budget printed below.
///
/// Ordinary runs skip. Arm it the way the phase 3 arms are armed — Release
/// bundle, `xcrun xctest`, so the environment reaches the process:
///
/// | variable | meaning |
/// | --- | --- |
/// | `MINIRUN_V41_APP_CHAT` | `1`, and nothing else arms this |
/// | `MINIRUN_V41_APP_CHAT_ARTIFACT` | the container root |
/// | `MINIRUN_V41_APP_CHAT_OUT` | where the run record is written |
/// | `MINIRUN_V41_APP_CHAT_PUBLICATION_REVISION` | the immutable publication revision |
/// | `MINIRUN_V41_APP_CHAT_SOURCE_REVISION` | the immutable upstream revision |
/// | `HF_TOKEN` | optional; raises the tree API rate limit |
/// | `MINIRUN_V41_APP_CHAT_PROMPT` | the user turn; defaults to the phase 3 arms' |
///
/// There is no verification-depth knob, and phase 3's sixth open item is why: a
/// spot check does not produce complete evidence, so `ArtifactRuntimeAuthority`
/// refuses to be built from it — correctly, and the refusal is the product
/// behaviour. A run needs a full pass, which is about five minutes of the eight
/// this arm takes. That also makes this the *whole* product path: the same
/// authority verifies, inspects, tokenizes and decodes.
///
/// It is still not the reference gate. It asserts that the budget the app
/// chooses reproduces the answer phase 3 recorded at two other budgets; what the
/// checkpoint says is `DeepSeekV41ProductGateTests`' question and stays there.
final class DeepSeekV41AppChatGateTests: XCTestCase {

    /// The phase 3 arms' answer, identical at the 3.4 GB floor and at 15 GB
    /// (`docs/experiments/2026-09-11-v41-phase3-runner.md` §3).
    static let expectedTokenIDs = [
        671, 6102, 294, 33395, 344, 2619, 56, 35393, 666, 343, 43886, 28, 982, 57, 1594, 12,
        797, 1,
    ]
    static let expectedLogitsDigest =
        "dee6d63cbd476588f2d07f65a53b00fc1570bc25aff737c2de5954cbbbaeb52c"

    func testTheBudgetTheDialChoosesProducesThePhaseThreeAnswer() async throws {
        #if !arch(arm64)
            throw XCTSkip("the V4.1 MLX runtime requires Apple silicon")
        #else
            let environment = ProcessInfo.processInfo.environment
            guard environment["MINIRUN_V41_APP_CHAT"] == "1" else {
                throw XCTSkip(
                    "set MINIRUN_V41_APP_CHAT=1 only for the explicit app-path chat arm")
            }
            func required(_ name: String) throws -> String {
                guard let value = environment[name], !value.isEmpty else {
                    throw V41ProductGateConfigurationError.missing(name)
                }
                return value
            }
            let root = URL(
                fileURLWithPath: try required("MINIRUN_V41_APP_CHAT_ARTIFACT"),
                isDirectory: true)
            let outputDirectory = URL(
                fileURLWithPath: try required("MINIRUN_V41_APP_CHAT_OUT"), isDirectory: true)
            try FileManager.default.createDirectory(
                at: outputDirectory, withIntermediateDirectories: true)
            let publicationRevision = try required("MINIRUN_V41_APP_CHAT_PUBLICATION_REVISION")
            let sourceRevision = try required("MINIRUN_V41_APP_CHAT_SOURCE_REVISION")
            let prompt = environment["MINIRUN_V41_APP_CHAT_PROMPT"].flatMap {
                $0.isEmpty ? nil : $0
            } ?? "what is the capital of austria"
            // Full, and not a choice: see the note above.
            let request = ArtifactVerificationRequest.full

            // MARK: The artifact, verified and rooted

            let tokenProvider = environment["HF_TOKEN"].map { token in
                { @Sendable in token as String? }
            }
            let transport = URLSessionTransport()
            // The **bundled** catalogue, not a live fetch. `DeepSeekV41ProductGateTests`
            // fetches because it is the gate and has to see what is published
            // today; this arm is about a budget, and a live fetch here walks the
            // tree of every model in the storefront — which is how it earned an
            // HTTP 429 on its first attempt, from a repository that has nothing
            // to do with V4.1. What matters is still asserted: the revision the
            // shipped catalogue pins has to be the one the arm was told to run.
            let catalog = ModelCatalog.bundled
            let descriptor = try XCTUnwrap(catalog.descriptor(.deepseekV41Flash))
            let publication = try XCTUnwrap(descriptor.source.repo)
            XCTAssertEqual(
                publication.revision, publicationRevision,
                "the bundled catalogue pins a different publication than this arm names")

            let ledger = InMemoryVerificationLedger()
            let location = ArtifactLocator(
                catalog: catalog, verificationLedger: ledger, maximumDepth: 0
            ).scan(root)
            let discovered = try XCTUnwrap(
                location.artifacts.first(where: { $0.model == .deepseekV41Flash }),
                "the armed path does not contain a catalog-matched V4.1 artifact")
            XCTAssertEqual(discovered.index.repositories.first?.revision, sourceRevision)
            XCTAssertEqual(
                discovered.isComplete, true,
                "this arm refuses a short or metadata-stale local tree")

            let rooted = try ArtifactVerificationRoot.open(
                relativeComponents: [], beneath: root)
            let report = try await ArtifactVerifier(
                tree: HuggingFaceTreeClient(transport: transport, token: tokenProvider),
                catalog: catalog, ledger: ledger
            ).verify(
                discovered, request, rootedAt: rooted,
                onProgress: { progress in
                    guard progress.filesChecked == progress.filesToCheck
                        || progress.filesChecked.isMultiple(of: 64)
                    else { return }
                    print(
                        "[v41-app-chat] verified \(progress.filesChecked)/"
                            + "\(progress.filesToCheck) files")
                })
            XCTAssertTrue(report.isComplete, "verification returned a refusal report")
            let evidence = try XCTUnwrap(
                ledger.record(
                    matching: ArtifactVerificationLookup(
                        rootPath: discovered.rootPath, model: discovered.model,
                        repository: publication, index: discovered.index)
                )?.evidence)
            let authority = try ArtifactRuntimeAuthority(root: rooted, evidence: evidence)
            let artifact = ArtifactReference(root: root, runtimeAuthority: authority)

            // MARK: The budget, from the dial's own arithmetic

            let ladder = try DeepSeekV41MemoryDialInputs.inspect(artifact)
            XCTAssertEqual(ladder.layerCount, 40)
            let balanced = DeepSeekV41MemoryDialInputs.budgetThatPinsEverything(
                census: ladder.census, floor: ladder.floor)
            let everyUnit = ladder.census.layerResidentBytes.reduce(0, +)
                + ladder.census.globals.headResidentBytes
            print(
                "[v41-app-chat] ladder floor \(ladder.floor.totalBytes) B; Balanced "
                    + "\(balanced) B; every block + head \(everyUnit) B")
            XCTAssertGreaterThan(balanced, DeepSeekV41ProductMemoryBudget.minimumBudgetBytes)
            // The rule `DeepSeekV41ProductRuntimeProvider.runnerScale` applies.
            let scale = DeepSeekV41DecodeRunner.Scale.stated(
                minimumBudgetBytes: balanced,
                maximumNewTokens: DeepSeekV41ProductMemoryBudget.maximumNewTokens)

            // MARK: The chat

            let vocabulary = try DeepSeekV4Vocabulary(
                data: authority.openFile(try XCTUnwrap(evidence.index.tokenizer).file)
                    .readAll(maximumBytes: 32 << 20))
            let promptIDs = try DeepSeekV41NoThinkingChatPrompt(
                messages: [.init(role: .user, content: prompt)]
            ).encode(using: vocabulary)
            print("[v41-app-chat] prompt ids \(promptIDs)")

            let runner = DeepSeekV41DecodeRunner(scale: scale)
            let session = try runner.start(
                RunRequest(
                    model: .deepseekV41Flash,
                    artifact: artifact,
                    prompt: .tokenIDs(promptIDs),
                    memoryBudgetBytes: balanced,
                    maximumNewTokens: DeepSeekV41ProductMemoryBudget.maximumNewTokens,
                    workingDirectory: outputDirectory))
            var tokens: [Int] = []
            var finished: RunSummary?
            for try await event in session.events {
                switch event {
                case .token(let token): tokens.append(token.tokenID)
                case .finished(let summary): finished = summary
                case .cancelled: XCTFail("the app-path arm was cancelled")
                case .log(let line): print("[v41-app-chat] \(line)")
                case .accepted, .telemetry, .phase, .phaseSummary: break
                }
            }
            let summary = try XCTUnwrap(finished)
            XCTAssertEqual(tokens, summary.tokenIDs)

            // MARK: What it has to have produced

            XCTAssertEqual(summary.tokenIDs, Self.expectedTokenIDs)
            XCTAssertEqual(summary.logitsDigest?.hex, Self.expectedLogitsDigest)
            XCTAssertLessThanOrEqual(summary.peakFootprintBytes, balanced)
            XCTAssertTrue(summary.budgetRespected)

            let record = try JSONDecoder().decode(
                AppChatRunRecord.self, from: try XCTUnwrap(summary.resultJSON))
            XCTAssertEqual(record.declaredBudgetBytes, balanced)
            let plan = try XCTUnwrap(
                record.pinPlan, "the Balanced budget produced no residency plan")
            XCTAssertEqual(
                record.pinnedBlockCount, 40,
                "Balanced promises every dense block; the run held a different number")
            XCTAssertTrue(record.pinnedOutputHead, "Balanced promises the output head")
            XCTAssertEqual(plan.pinnedLayers.count, record.pinnedBlockCount)
            XCTAssertTrue(plan.isResidencyBalanced)
            let secondsPerToken = String(
                format: "%.2f", 1 / (summary.tokensPerSecond ?? .nan))
            print(
                "[v41-app-chat] \(record.pinnedBlockCount) blocks + head pinned; peak "
                    + "\(record.peakFootprintBytes) B of \(balanced) B; "
                    + "\(secondsPerToken) s/token; "
                    + "digest \(summary.logitsDigest?.hex ?? "none")")
            try XCTUnwrap(summary.resultJSON).write(
                to: outputDirectory.appendingPathComponent("deepseek-v41-app-chat.json"),
                options: .atomic)
        #endif
    }

    /// The subset of the run record this arm reads back. Deliberately not the
    /// engine's own payload type, for `DeepSeekV41ProductGateTests`' reason: a
    /// gate that decoded the writer's struct would pass whatever the writer
    /// emitted.
    private struct AppChatRunRecord: Decodable {
        let declaredBudgetBytes: UInt64
        let peakFootprintBytes: UInt64
        let pinPlan: PinPlan?
        let pinnedBlockCount: Int
        let pinnedOutputHead: Bool
    }
}
