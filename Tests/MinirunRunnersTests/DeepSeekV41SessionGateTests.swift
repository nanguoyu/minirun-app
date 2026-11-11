import Foundation
import MinirunKit
import ModelAdapters
import XCTest

@testable import MinirunRunners

/// A two-turn conversation over the real container, run the way the app runs
/// it, one arm per process.
///
/// Every arm except `scratch` passes the options the app's
/// `DeepSeekV41ProductRuntimeProvider` builds for a conversation: the first
/// turn's length (the chat encoding of the first user turn alone, with its
/// generation prefix), a restore file and a write file that are the same file,
/// and a checkpoint cadence. Each arm copies the state it starts from into its
/// own file first, so the first turn's state is never written over.
///
/// Chosen by `MINIRUN_V41_SESSION_ARM`:
///
/// | arm | what it does |
/// | --- | --- |
/// | `turn1` | runs the first user turn, writes the state to `MINIRUN_V41_SESSION_FILE` and the turn's ids beside it (`<file>.turn1.json`) |
/// | `scratch` | runs turn 2 over the whole conversation in one prefill from position 0, with no options: the path ADR 0023 rejected, kept for comparison |
/// | `continued` | runs turn 2 from a copy of the first turn's state |
/// | `replayed` | runs turn 2 with no state on disk, which replays: the first turn's block, then one pass per id |
/// | `interrupt` | runs turn 2 like `continued`, writing the state after every pass, and kills its own process with `SIGKILL` after `MINIRUN_V41_SESSION_INTERRUPT_AFTER` reply tokens (default 3), leaving `<file>.interrupted.json` and `<file>.interrupted.v41session` |
/// | `resume` | a new process: turn 2's prompt plus the interrupted reply's tokens, continued from what `interrupt` left, finishing the reply |
/// | `forced` | teacher-forced: a reference run's prompt as the first block, then its generated ids one pass each, so every pass's logits are the reference's position's |
///
/// The `forced` arm reads `MINIRUN_V41_SESSION_FORCED_IDS`, a reference run's
/// `generated.json` (`prompt_token_ids`, `generated_token_ids`), and checks
/// that the product encoding of `MINIRUN_V41_PRODUCT_PROMPT` is that prompt.
/// It runs the prompt and the first `MINIRUN_V41_SESSION_FORCED_COUNT`
/// generated ids but the last (default: all of them) as one replayed
/// conversation with no state on disk, asking for one token. With
/// `MINIRUN_V41_DUMP_LOGITS` the pass that consumed prompt id `p` writes
/// `prompt-<p>.bin` and the last pass `logits-0.bin`, so reference position
/// `n` is `prompt-<n + prompt count - 1>.bin`, and the last one `logits-0.bin`.
/// `MINIRUN_V41_PRODUCT_TOKEN_CEILINGS=off` lets it past the product's prompt
/// ceiling, which is what it exists to test.
///
/// `MINIRUN_V41_SESSION_SECOND_TURN` is the second user message.
/// `MINIRUN_V41_SESSION_CHECKPOINT` (`every-pass` or `at-release`) overrides
/// an arm's cadence. Everything else is
/// ``DeepSeekV41ProductGateConfiguration``'s, and the arm verifies the whole
/// container first exactly as the product gate does. Per-position logits come
/// from `MINIRUN_V41_DUMP_LOGITS`, one directory per arm;
/// `MINIRUN_V41_SESSION_COMPARE_WITH` names another arm's directory, and the
/// arm then prints the comparison, position `p` here against position
/// `p + MINIRUN_V41_SESSION_COMPARE_OFFSET` there.
/// `Tools/v41_flash/run_session_arm.sh` runs it.
final class DeepSeekV41SessionGateTests: XCTestCase {
    /// The two fields of a reference run's `generated.json` the forced arm reads.
    struct ForcedSequence: Decodable {
        let promptTokenIDs: [Int]
        let generatedTokenIDs: [Int]
        enum CodingKeys: String, CodingKey {
            case promptTokenIDs = "prompt_token_ids"
            case generatedTokenIDs = "generated_token_ids"
        }
    }

    struct TurnOne: Codable {
        let promptTokenIDs: [Int]
        let generatedTokenIDs: [Int]
        let text: String
        let logitsSHA256: String?
    }

    func testTwoTurnConversationFromScratchAndContinued() async throws {
        #if !arch(arm64)
            throw XCTSkip("the V4.1 MLX runtime requires Apple silicon")
        #else
            let environment = ProcessInfo.processInfo.environment
            guard let arm = environment["MINIRUN_V41_SESSION_ARM"], !arm.isEmpty else {
                throw XCTSkip("set MINIRUN_V41_SESSION_ARM to turn1, scratch or continued")
            }
            let arms = [
                "turn1", "scratch", "continued", "replayed", "interrupt", "resume", "forced",
            ]
            guard arms.contains(arm) else {
                throw V41ProductGateConfigurationError.invalid("MINIRUN_V41_SESSION_ARM", arm)
            }
            let configuration = try DeepSeekV41ProductGateConfiguration.load()
            guard let rawFile = environment["MINIRUN_V41_SESSION_FILE"], !rawFile.isEmpty else {
                throw V41ProductGateConfigurationError.missing("MINIRUN_V41_SESSION_FILE")
            }
            let stateFile = URL(fileURLWithPath: rawFile)
            let turnOneFile = URL(fileURLWithPath: rawFile + ".turn1.json")
            let interruptedIDsFile = URL(fileURLWithPath: rawFile + ".interrupted.json")
            let interruptedStateFile = URL(fileURLWithPath: rawFile + ".interrupted.v41session")
            let ownStateFile = configuration.outputDirectory.appendingPathComponent(
                "session-\(arm).v41session")
            let interruptAfter = environment["MINIRUN_V41_SESSION_INTERRUPT_AFTER"]
                .flatMap(Int.init) ?? 3
            let cadenceOverride = environment["MINIRUN_V41_SESSION_CHECKPOINT"].flatMap {
                DeepSeekV41SessionOptions.Checkpoint(rawValue: $0)
            }

            let (authority, vocabulary) = try await verifiedAuthority(configuration)
            let firstTurn = DeepSeekV41NoThinkingChatPrompt.Message(
                role: .user, content: configuration.prompt)
            // The app's own count: the first user turn alone, encoded with its
            // generation prefix (`DeepSeekV41ProductTokenizer.encode`).
            let firstTurnTokens = try DeepSeekV41NoThinkingChatPrompt(
                messages: [firstTurn], addAssistantGenerationPrefix: true
            ).encode(using: vocabulary).count
            var promptIDs: [Int]
            let options: DeepSeekV41SessionOptions?
            var newTokens = configuration.newTokenCount
            var resumedIDs: [Int] = []
            var forcedCount = 0
            let fileManager = FileManager.default
            func ownCopy(of source: URL) throws -> URL {
                try? fileManager.removeItem(at: ownStateFile)
                try fileManager.copyItem(at: source, to: ownStateFile)
                return ownStateFile
            }
            if arm == "turn1" {
                promptIDs = try DeepSeekV41NoThinkingChatPrompt(messages: [firstTurn])
                    .encode(using: vocabulary)
                XCTAssertEqual(promptIDs.count, firstTurnTokens)
                try? fileManager.removeItem(at: stateFile)
                options = DeepSeekV41SessionOptions(
                    restoreFrom: stateFile, writeTo: stateFile,
                    firstTurnTokenCount: firstTurnTokens,
                    checkpoint: cadenceOverride ?? .atRelease)
            } else if arm == "forced" {
                guard let rawForced = environment["MINIRUN_V41_SESSION_FORCED_IDS"],
                    !rawForced.isEmpty
                else {
                    throw V41ProductGateConfigurationError.missing(
                        "MINIRUN_V41_SESSION_FORCED_IDS")
                }
                let forced = try JSONDecoder().decode(
                    ForcedSequence.self, from: Data(contentsOf: URL(fileURLWithPath: rawForced)))
                let encoded = try DeepSeekV41NoThinkingChatPrompt(messages: [firstTurn])
                    .encode(using: vocabulary)
                XCTAssertEqual(
                    encoded, forced.promptTokenIDs,
                    "the product encoding of the prompt is not the reference's prompt")
                XCTAssertEqual(firstTurnTokens, forced.promptTokenIDs.count)
                forcedCount = forced.generatedTokenIDs.count
                if let raw = environment["MINIRUN_V41_SESSION_FORCED_COUNT"], !raw.isEmpty {
                    guard let value = Int(raw), (1...forcedCount).contains(value) else {
                        throw V41ProductGateConfigurationError.invalid(
                            "MINIRUN_V41_SESSION_FORCED_COUNT", raw)
                    }
                    forcedCount = value
                }
                // The last forced id is never consumed: the pass before it is
                // the one that predicts it.
                promptIDs =
                    forced.promptTokenIDs + forced.generatedTokenIDs.prefix(forcedCount - 1)
                newTokens = 1
                try? fileManager.removeItem(at: ownStateFile)
                options = DeepSeekV41SessionOptions(
                    restoreFrom: ownStateFile, writeTo: ownStateFile,
                    firstTurnTokenCount: forced.promptTokenIDs.count,
                    checkpoint: cadenceOverride ?? .atRelease)
                print(
                    "[v41-session-gate] forced: \(forced.promptTokenIDs.count)-id block, "
                        + "then \(forcedCount - 1) of the reference's "
                        + "\(forced.generatedTokenIDs.count) ids, one pass each")
            } else {
                guard let second = environment["MINIRUN_V41_SESSION_SECOND_TURN"],
                    !second.isEmpty
                else {
                    throw V41ProductGateConfigurationError.missing(
                        "MINIRUN_V41_SESSION_SECOND_TURN")
                }
                let turnOne = try JSONDecoder().decode(
                    TurnOne.self, from: Data(contentsOf: turnOneFile))
                // The product's own encoding of the whole conversation, with
                // the first answer as text, exactly as a chat would hold it.
                promptIDs = try DeepSeekV41NoThinkingChatPrompt(messages: [
                    firstTurn,
                    .init(role: .assistant, content: turnOne.text),
                    .init(role: .user, content: second),
                ]).encode(using: vocabulary)
                let carried = turnOne.promptTokenIDs + turnOne.generatedTokenIDs
                let agreeing = zip(carried, promptIDs).prefix { $0 == $1 }.count
                print(
                    "[v41-session-gate] conversation ids \(promptIDs); turn 1 carried "
                        + "\(carried.count) ids, of which the re-encoded conversation "
                        + "repeats the first \(agreeing)")
                switch arm {
                case "scratch":
                    options = nil
                case "continued":
                    let own = try ownCopy(of: stateFile)
                    options = DeepSeekV41SessionOptions(
                        restoreFrom: own, writeTo: own, firstTurnTokenCount: firstTurnTokens,
                        checkpoint: cadenceOverride ?? .atRelease)
                case "replayed":
                    try? fileManager.removeItem(at: ownStateFile)
                    options = DeepSeekV41SessionOptions(
                        restoreFrom: ownStateFile, writeTo: ownStateFile,
                        firstTurnTokenCount: firstTurnTokens,
                        checkpoint: cadenceOverride ?? .atRelease)
                case "interrupt":
                    try? fileManager.removeItem(at: interruptedStateFile)
                    try? fileManager.removeItem(at: interruptedIDsFile)
                    try fileManager.copyItem(at: stateFile, to: interruptedStateFile)
                    options = DeepSeekV41SessionOptions(
                        restoreFrom: interruptedStateFile, writeTo: interruptedStateFile,
                        firstTurnTokenCount: firstTurnTokens,
                        checkpoint: cadenceOverride ?? .everyPass)
                default:  // resume
                    resumedIDs = try JSONDecoder().decode(
                        [Int].self, from: Data(contentsOf: interruptedIDsFile))
                    promptIDs += resumedIDs
                    newTokens -= resumedIDs.count
                    let own = try ownCopy(of: interruptedStateFile)
                    options = DeepSeekV41SessionOptions(
                        restoreFrom: own, writeTo: own, firstTurnTokenCount: firstTurnTokens,
                        checkpoint: cadenceOverride ?? .everyPass)
                    print(
                        "[v41-session-gate] resume after \(resumedIDs) with "
                            + "\(newTokens) tokens to go")
                }
            }
            print("[v41-session-gate] \(arm) prompt ids \(promptIDs)")

            let runner = DeepSeekV41DecodeRunner(
                scale: configuration.runnerScale, session: options,
                enforcesProductTokenCeilings: configuration.enforcesTokenCeilings)
            let session = try runner.start(
                RunRequest(
                    model: .deepseekV41Flash,
                    artifact: ArtifactReference(
                        root: configuration.artifactRoot, runtimeAuthority: authority),
                    prompt: .tokenIDs(promptIDs),
                    memoryBudgetBytes: configuration.memoryBudgetBytes,
                    maximumNewTokens: newTokens,
                    knobs: configuration.knobs,
                    workingDirectory: configuration.outputDirectory))
            var finished: RunSummary?
            var received: [Int] = []
            var lastReplay: RunReplayProgress?
            for try await event in session.events {
                switch event {
                case .finished(let summary): finished = summary
                case .cancelled: XCTFail("the \(arm) arm was cancelled")
                case .log(let line): print("[v41-session-gate] \(line)")
                case .phase(let phase):
                    if let replay = phase.replay, replay != lastReplay {
                        lastReplay = replay
                        print("[v41-session-gate] replay token \(replay.token) of \(replay.of)")
                    }
                case .token(let token):
                    received.append(token.tokenID)
                    if arm == "interrupt", received.count == interruptAfter {
                        // A kill, not a stop: no release, no terminal event, no
                        // write at the end. What survives is what the last
                        // finished pass wrote.
                        try JSONEncoder().encode(received).write(
                            to: interruptedIDsFile, options: .atomic)
                        let positions = try? DeepSeekV41SessionStateProbe.positions(
                            interruptedStateFile)
                        print(
                            "[v41-session-gate] interrupt after \(received); state on disk "
                                + "holds \(positions.map(String.init) ?? "?") positions; "
                                + "SIGKILL")
                        fflush(stdout)
                        kill(getpid(), SIGKILL)
                    }
                default: break
                }
            }
            let summary = try XCTUnwrap(finished)
            let json = try XCTUnwrap(summary.resultJSON)
            try configuration.write(json, filename: configuration.resultFilename("session-\(arm)"))
            let record = try JSONDecoder().decode(Record.self, from: json)
            let text = productText(summary.tokenIDs, vocabulary: vocabulary)
            print(
                "[v41-session-gate] \(arm) generated \(summary.tokenIDs); text \"\(text)\"; "
                    + "digest \(record.logitsSHA256 ?? "none"); prefill "
                    + "\(record.telemetry.prefillSeconds.map { String(format: "%.1f s", $0) } ?? "-"), "
                    + "\(record.telemetry.prefillBytes?.totalBytesRead ?? 0) B read "
                    + "(\(record.telemetry.prefillBytes?.expertBytesRead ?? 0) B experts)")
            if let sessionRecord = record.session {
                print("[v41-session-gate] \(arm) session \(sessionRecord)")
                if let writes = sessionRecord.snapshotWriteSeconds, !writes.isEmpty {
                    print(
                        "[v41-session-gate] \(arm) \(writes.count) state writes, "
                            + String(
                                format: "mean %.4f s, max %.4f s, min %.4f s",
                                writes.reduce(0, +) / Double(writes.count),
                                writes.max() ?? 0, writes.min() ?? 0)
                            + ", \(sessionRecord.snapshotBytes ?? 0) B last")
                }
            }
            print(
                "[v41-session-gate] \(arm) whole-run bytes "
                    + "\(record.telemetry.bytes.totalBytesRead) "
                    + "(\(record.telemetry.bytes.expertBytesRead) B experts)")
            XCTAssertTrue(record.budgetRespected)

            switch arm {
            case "turn1":
                if let expected = configuration.expectedTokenIDs {
                    XCTAssertEqual(Array(summary.tokenIDs.prefix(expected.count)), expected)
                }
                let written = try XCTUnwrap(record.session?.snapshotWrittenTo)
                XCTAssertEqual(written, stateFile.path)
                XCTAssertEqual(
                    record.session?.snapshotPositions,
                    promptIDs.count + summary.tokenIDs.count - 1,
                    "the state holds the prompt and every generated token but the last")
                let turnOne = TurnOne(
                    promptTokenIDs: promptIDs,
                    // The last generated id was never fed back, so it is the
                    // first id a continued turn runs.
                    generatedTokenIDs: summary.tokenIDs,
                    text: text, logitsSHA256: record.logitsSHA256)
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(turnOne).write(to: turnOneFile, options: .atomic)
                XCTAssertEqual(record.session?.path, .firstTurn)
            case "continued", "resume":
                XCTAssertEqual(
                    record.session?.path, .continued,
                    "fell back: \(record.session?.fallbackReason ?? "no session record")")
                if arm == "resume" {
                    print(
                        "[v41-session-gate] resume whole reply "
                            + "\(resumedIDs + summary.tokenIDs)")
                }
            case "forced":
                XCTAssertEqual(record.session?.path, .replayed)
                XCTAssertEqual(lastReplay?.of, promptIDs.count)
                print(
                    "[v41-session-gate] forced over \(promptIDs.count) prompt ids; "
                        + "\(forcedCount) reference positions, the last one generated id "
                        + "\(summary.tokenIDs.first ?? -1)")
            case "replayed":
                XCTAssertEqual(record.session?.path, .replayed)
                XCTAssertEqual(record.session?.firstBlockTokens, firstTurnTokens)
                XCTAssertEqual(lastReplay?.of, promptIDs.count)
            case "interrupt":
                XCTFail("the interrupt arm reached the end of its reply without being killed")
            default:
                XCTAssertNil(record.session)
            }

            if let own = environment["MINIRUN_V41_DUMP_LOGITS"], !own.isEmpty,
                let other = environment["MINIRUN_V41_SESSION_COMPARE_WITH"], !other.isEmpty
            {
                try compareDumps(
                    URL(fileURLWithPath: own), URL(fileURLWithPath: other),
                    positions: summary.tokenIDs.count,
                    offset: environment["MINIRUN_V41_SESSION_COMPARE_OFFSET"]
                        .flatMap(Int.init) ?? 0)
            }
        #endif
    }

    #if arch(arm64)
        /// Max, RMS, and both over the other run's standard deviation, for
        /// every position both dumps hold: position `p` here against `p +
        /// offset` there.
        private func compareDumps(
            _ own: URL, _ other: URL, positions: Int, offset: Int
        ) throws {
            var identical = 0
            for position in 0..<positions {
                let name = "logits-\(position).bin"
                guard let a = try? Data(contentsOf: own.appendingPathComponent(name)),
                    let b = try? Data(
                        contentsOf: other.appendingPathComponent(
                            "logits-\(position + offset).bin")),
                    a.count == b.count, !a.isEmpty
                else {
                    print("[v41-session-compare] position \(position): not in both dumps")
                    continue
                }
                let x = Self.floats(a)
                let y = Self.floats(b)
                var worst = 0.0
                var squares = 0.0
                var mean = 0.0
                for (p, q) in zip(x, y) {
                    let difference = Double(p) - Double(q)
                    worst = max(worst, abs(difference))
                    squares += difference * difference
                    mean += Double(q)
                }
                mean /= Double(y.count)
                let std = (y.reduce(0.0) { $0 + (Double($1) - mean) * (Double($1) - mean) }
                    / Double(y.count)).squareRoot()
                let rms = (squares / Double(x.count)).squareRoot()
                if a == b { identical += 1 }
                print(
                    "[v41-session-compare] position \(position) against "
                        + "\(position + offset): "
                        + (a == b
                            ? "bit-identical"
                            : String(
                                format: "max %.4e rms %.4e std %.4e max/std %.4e rms/std %.4e",
                                worst, rms, std, worst / std, rms / std))
                        + " argmax \(Self.argmax(x))/\(Self.argmax(y))")
            }
            print("[v41-session-compare] \(identical) of \(positions) positions bit-identical")
        }

        private static func floats(_ data: Data) -> [Float] {
            data.withUnsafeBytes { raw in
                (0..<(data.count / 4)).map {
                    Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(
                        fromByteOffset: $0 * 4, as: UInt32.self)))
                }
            }
        }

        private static func argmax(_ values: [Float]) -> Int {
            values.indices.max { values[$0] < values[$1] } ?? -1
        }

        /// The product gate's verification, unchanged: the whole container,
        /// then the rooted authority and the verified tokenizer.
        private func verifiedAuthority(
            _ configuration: DeepSeekV41ProductGateConfiguration
        ) async throws -> (ArtifactRuntimeAuthority, DeepSeekV4Vocabulary) {
            let tokenProvider = configuration.huggingFaceToken.map { token in
                { @Sendable in token as String? }
            }
            let transport = URLSessionTransport()
            let catalog =
                configuration.usesBundledCatalog
                ? ModelCatalog.bundled
                : try await HuggingFaceCatalogSource(
                    transport: transport, token: tokenProvider
                ).fetch()
            let descriptor = try XCTUnwrap(catalog.descriptor(.deepseekV41Flash))
            let publication = try XCTUnwrap(descriptor.source.repo)
            XCTAssertEqual(publication.revision, configuration.publicationRevision)
            let ledger = InMemoryVerificationLedger()
            let location = ArtifactLocator(
                catalog: catalog, verificationLedger: ledger, maximumDepth: 0
            ).scan(configuration.artifactRoot)
            let artifact = try XCTUnwrap(
                location.artifacts.first(where: { $0.model == .deepseekV41Flash }))
            XCTAssertEqual(
                artifact.index.repositories.first?.revision, configuration.sourceRevision)
            let rooted = try ArtifactVerificationRoot.open(
                relativeComponents: [], beneath: configuration.artifactRoot)
            let verifier = ArtifactVerifier(
                tree: HuggingFaceTreeClient(transport: transport, token: tokenProvider),
                catalog: catalog, ledger: ledger)
            let report = try await verifier.verify(
                artifact, configuration.verificationRequest, rootedAt: rooted,
                onProgress: { progress in
                    guard progress.filesChecked == progress.filesToCheck else { return }
                    print(
                        "[v41-session-gate] verified \(progress.filesChecked) files, "
                            + "\(progress.bytesChecked) bytes")
                })
            XCTAssertTrue(report.isComplete)
            let record = try XCTUnwrap(
                ledger.record(
                    matching: ArtifactVerificationLookup(
                        rootPath: artifact.rootPath, model: artifact.model,
                        repository: publication, index: artifact.index)))
            let evidence = try XCTUnwrap(record.evidence)
            let authority = try ArtifactRuntimeAuthority(root: rooted, evidence: evidence)
            let vocabulary = try DeepSeekV4Vocabulary(
                data: authority.openFile(
                    try XCTUnwrap(evidence.index.tokenizer).file
                ).readAll(maximumBytes: 32 << 20))
            return (authority, vocabulary)
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

        private struct Record: Decodable {
            struct Telemetry: Decodable {
                let prefillSeconds: Double?
                let prefillBytes: ByteAccounting?
                let bytes: ByteAccounting
            }
            let budgetRespected: Bool
            let logitsSHA256: String?
            let telemetry: Telemetry
            let session: DeepSeekV41SessionRecord?
        }
    #endif
}

/// How many positions a state file on disk holds, read from its header alone,
/// for a log line printed in the instant before a kill.
enum DeepSeekV41SessionStateProbe {
    static func positions(_ url: URL) throws -> Int {
        let data = try Data(contentsOf: url)
        let length = data.withUnsafeBytes {
            Int(UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: 8, as: UInt64.self)))
        }
        let header = try JSONSerialization.jsonObject(with: data.subdata(in: 16..<(16 + length)))
        return (header as? [String: Any])?["positions"] as? Int ?? -1
    }
}
