import Foundation
import MinirunKit
import XCTest

@testable import MinirunRunners

/// **The one number that killed the phone.**
///
/// `DeepSeekV41ArtifactWorkload` builds its ``DeepSeekV41Model`` inside
/// `factory.prepare`, and the model builds one rotary table per backbone block
/// in its initializer. Until 2026-09-11 the count it was given was
/// `config.maximumPositionCount` — the checkpoint's declared
/// `max_position_embeddings`, 1,048,576 for DeepSeek V4.1 Flash — so a chat of
/// eleven prompt tokens and 64 new ones allocated **10,737,418,240 B of float32
/// cosines and sines**, exactly 10.0 GiB, before it had read a weight, taken a
/// budget floor, or emitted a single telemetry event.
///
/// On a Mac that vanished into the floor the budget is measured from. On the
/// owner's iPhone 16 Pro, Jetsam took the process about nineteen 268 MB blocks
/// in: 5,234 / 5,084 / ~5,100 MB, three times, with a flight-recorder trace
/// holding a start line and nothing else.
///
/// Every assertion below is about the count the workload now asks for.
final class DeepSeekV41RunPositionsTests: XCTestCase {

    /// The published ceiling, for scale.
    private let publishedCeiling = 1_048_576

    /// The chat in the trace: eleven prompt tokens, 64 new tokens. The engine
    /// passes `prompt + newTokens - 1` to `prefill` as the decode position
    /// limit, so that is the last position a run reaches and the exact number
    /// of table rows it needs.
    func testTheChatThatWasKilledNeedsSeventyFourPositions() {
        XCTAssertEqual(
            DeepSeekV41RunPositions.count(
                promptTokenCount: 11, maximumNewTokens: 64,
                configuredMaximum: publishedCeiling),
            74)
    }

    /// The product's widest chat is 512 prompt tokens and 64 new ones, and it
    /// is still four orders of magnitude below the checkpoint's ceiling.
    func testTheWidestProductChatIsStillFarBelowTheCheckpointsCeiling() {
        let widest = DeepSeekV41RunPositions.count(
            promptTokenCount: 512, maximumNewTokens: 64,
            configuredMaximum: publishedCeiling)
        XCTAssertEqual(widest, 575)
        XCTAssertLessThan(widest * 1_000, publishedCeiling)
    }

    /// A request for more positions than the checkpoint has is clamped here and
    /// refused by name in `DeepSeekV41RunEngine.execute()`, which has validated
    /// the prompt by then. Clamping decides only how many rows get built in the
    /// meantime, and building the ceiling's worth in order to then refuse the
    /// run would be the bug this exists to prevent.
    func testARequestPastTheCheckpointsCeilingIsClampedAndNotExpanded() {
        XCTAssertEqual(
            DeepSeekV41RunPositions.count(
                promptTokenCount: 4_000_000, maximumNewTokens: 64,
                configuredMaximum: publishedCeiling),
            publishedCeiling)
        XCTAssertEqual(
            DeepSeekV41RunPositions.count(
                promptTokenCount: .max, maximumNewTokens: .max,
                configuredMaximum: publishedCeiling),
            publishedCeiling)
    }

    /// A degenerate request never produces a zero count: the model refuses zero
    /// positions, and the refusal a caller deserves is the engine's sentence
    /// about the prompt rather than a configuration error from one line earlier.
    func testADegenerateRequestStillAsksForAtLeastOnePosition() {
        XCTAssertEqual(
            DeepSeekV41RunPositions.count(
                promptTokenCount: 0, maximumNewTokens: 0,
                configuredMaximum: publishedCeiling),
            1)
        XCTAssertEqual(
            DeepSeekV41RunPositions.count(
                promptTokenCount: -3, maximumNewTokens: 1, configuredMaximum: 0),
            1)
    }

    /// The same number, taken from a request, which is what the factory does.
    func testTheCountIsTakenFromTheRequestsOwnPromptAndTokenCeiling() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        let request = RunRequest(
            model: .deepseekV41Flash,
            artifact: ArtifactReference(root: root),
            prompt: .tokenIDs(Array(0..<11)),
            memoryBudgetBytes: 1_900_000_000,
            maximumNewTokens: 64,
            knobs: RunKnobs(),
            workingDirectory: root)
        XCTAssertEqual(
            DeepSeekV41RunPositions.count(
                for: request, configuredMaximum: publishedCeiling),
            74)

        // A text prompt never reaches here — `acceptsTextPrompts` is false and
        // `validate` has already refused — but a count of zero would turn that
        // refusal into a confusing configuration error, so it does not happen.
        let text = RunRequest(
            model: .deepseekV41Flash,
            artifact: ArtifactReference(root: root),
            prompt: .text("hello"),
            memoryBudgetBytes: 1_900_000_000,
            maximumNewTokens: 1,
            knobs: RunKnobs(),
            workingDirectory: root)
        XCTAssertEqual(
            DeepSeekV41RunPositions.count(for: text, configuredMaximum: publishedCeiling),
            1)
    }
}
