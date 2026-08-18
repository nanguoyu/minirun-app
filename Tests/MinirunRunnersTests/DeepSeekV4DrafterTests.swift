import XCTest

@testable import MinirunRunners

/// The drafter is a pure function of the ids already emitted, so its whole
/// contract is testable without a model, an artifact, or a GPU — which is the
/// point of choosing a free drafter for ADR 0017's first increment.
final class DeepSeekV4DrafterTests: XCTestCase {
    func testAnEmptyOrShortHistoryProposesNothing() {
        var drafter = DeepSeekV4PromptLookupDrafter()
        XCTAssertNil(drafter.proposal())
        drafter.append([7])
        XCTAssertNil(drafter.proposal())
        drafter.append([8, 9])
        // Three ids, no repetition: nothing has ever followed "8, 9" before.
        XCTAssertNil(drafter.proposal())
    }

    func testItProposesTheTokenThatFollowedTheMostRecentOccurrence() {
        var drafter = DeepSeekV4PromptLookupDrafter(orders: [3, 2])
        // "1 2 3" was followed by 4 the first time and by 5 the second; the
        // most recent occurrence wins.
        drafter.append([1, 2, 3, 4, 0, 1, 2, 3, 5, 0, 1, 2, 3])
        let proposal = try? XCTUnwrap(drafter.proposal())
        XCTAssertEqual(proposal?.tokenID, 5)
        XCTAssertEqual(proposal?.order, 3)
    }

    func testItFallsBackFromTheWiderOrderToTheNarrowerOne() {
        var drafter = DeepSeekV4PromptLookupDrafter(orders: [3, 2])
        // "9 3" has occurred before and was followed by 4; "1 9 3" has not.
        drafter.append([9, 3, 4, 7, 8, 1, 9, 3])
        let proposal = try? XCTUnwrap(drafter.proposal())
        XCTAssertEqual(proposal?.tokenID, 4)
        XCTAssertEqual(proposal?.order, 2)
    }

    func testTheQueryNGramIsNotItsOwnMostRecentOccurrence() {
        var drafter = DeepSeekV4PromptLookupDrafter(orders: [2])
        drafter.append([4, 5])
        // "4 5" occurs exactly once, at the end, and has no successor. A
        // drafter that indexed it would propose out of bounds or propose the
        // n-gram's own last id.
        XCTAssertNil(drafter.proposal())
    }

    func testTheIncrementalIndexAgreesWithARescanAtEveryPosition() {
        // The index is maintained per append; a full rescan is the definition.
        // They must agree at every prefix, or the drafter's alpha depends on
        // how the ids arrived.
        let ids = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        var drafter = DeepSeekV4PromptLookupDrafter(orders: [3, 2])
        for (index, id) in ids.enumerated() {
            drafter.append(id)
            let prefix = Array(ids[0...index])
            XCTAssertEqual(
                drafter.proposal()?.tokenID,
                Self.rescanProposal(prefix, orders: [3, 2]),
                "prefix of length \(index + 1) disagrees with a rescan")
        }
    }

    func testRepeatedTextIsWhereThisDrafterEarnsItsKeep() {
        var drafter = DeepSeekV4PromptLookupDrafter()
        let sentence = [11, 12, 13, 14, 15]
        drafter.append(sentence)
        drafter.append([99])
        // Re-emitting the same sentence: after two ids of it the drafter knows
        // the rest.
        drafter.append([11, 12])
        for expected in sentence.dropFirst(2) {
            XCTAssertEqual(drafter.proposal()?.tokenID, expected)
            drafter.append(expected)
        }
    }

    // MARK: - The policy knob

    func testAnUnsetKnobIsOffAndOffIsNotEnabled() throws {
        XCTAssertEqual(try DeepSeekV4DraftPolicy.fromEnvironment([:]), .off)
        XCTAssertEqual(
            try DeepSeekV4DraftPolicy.fromEnvironment(["MINIRUN_V4_DRAFT": "0"]), .off)
        XCTAssertFalse(try DeepSeekV4DraftPolicy.fromEnvironment([:]).isEnabled)
    }

    func testOneTurnsOnTheFreeDrafterAtTheStatedOrders() throws {
        XCTAssertEqual(
            try DeepSeekV4DraftPolicy.fromEnvironment(["MINIRUN_V4_DRAFT": "1"]),
            .promptLookup(orders: [3, 2]))
    }

    func testForceCarriesItsFallbackAndDefaultsItToZero() throws {
        XCTAssertEqual(
            try DeepSeekV4DraftPolicy.fromEnvironment(["MINIRUN_V4_DRAFT": "force"]),
            .forced(fallbackTokenID: 0, orders: [3, 2]))
        XCTAssertEqual(
            try DeepSeekV4DraftPolicy.fromEnvironment([
                "MINIRUN_V4_DRAFT": "force",
                "MINIRUN_V4_DRAFT_FALLBACK_TOKEN": "42",
            ]),
            .forced(fallbackTokenID: 42, orders: [3, 2]))
    }

    func testAnUnknownValueIsRefusedRatherThanTreatedAsOff() {
        XCTAssertThrowsError(
            try DeepSeekV4DraftPolicy.fromEnvironment(["MINIRUN_V4_DRAFT": "yes please"])
        ) { error in
            XCTAssertEqual(
                error as? DeepSeekV4DraftPolicyError,
                .invalid("MINIRUN_V4_DRAFT", "yes please"))
        }
    }

    // MARK: - The accounting

    func testProposalAndAcceptanceAreReportedSeparately() {
        var metrics = DeepSeekV4DraftMetrics()
        metrics.draftPositions = 10
        metrics.draftProposed = 4
        metrics.draftAccepted = 1
        metrics.passes = 9
        metrics.tokensEmitted = 10
        // Six positions had nothing to propose and cost an ordinary pass; three
        // proposals were wrong and cost a wide pass for one token. Only alpha
        // divides by positions, and only it is the term the wall-clock
        // arithmetic uses.
        XCTAssertEqual(metrics.alpha, 0.1, accuracy: 1e-12)
        XCTAssertEqual(metrics.acceptanceOverProposals, 0.25, accuracy: 1e-12)
        XCTAssertEqual(metrics.proposedRate, 0.4, accuracy: 1e-12)
        XCTAssertEqual(metrics.tokensPerPass, 10.0 / 9.0, accuracy: 1e-12)
    }

    func testAnEmptyRunReportsZeroRatherThanDividingByZero() {
        let metrics = DeepSeekV4DraftMetrics()
        XCTAssertEqual(metrics.alpha, 0)
        XCTAssertEqual(metrics.acceptanceOverProposals, 0)
        XCTAssertEqual(metrics.tokensPerPass, 0)
    }

    /// The measured finding this drafter has to be honest about: on the stated
    /// 40-token arm's own token stream it proposes once in 39 positions and is
    /// accepted none of them.
    ///
    /// The ids are the arm's, verbatim
    /// (`docs/experiments/data/2026-08-17-v4-gpu-wait-attribution/`
    /// `deepseek-v4-product-memory-plateau-stated-A.json.gz`). This test is a
    /// regression on the drafter, not on the model: if a later change to the
    /// n-gram rule made alpha nonzero here, the record's conclusion would have
    /// moved and the record has to move with it.
    func testTheStatedArmsOwnTokenStreamAcceptsNothing() {
        var drafter = DeepSeekV4PromptLookupDrafter()
        drafter.append(Self.statedArmPrompt)
        var metrics = DeepSeekV4DraftMetrics()
        let generated = Self.statedArmGenerated
        for (index, token) in generated.enumerated() {
            drafter.append(token)
            guard index + 1 < generated.count else { break }
            metrics.draftPositions += 1
            guard let proposal = drafter.proposal() else { continue }
            metrics.draftProposed += 1
            if proposal.tokenID == generated[index + 1] { metrics.draftAccepted += 1 }
        }
        XCTAssertEqual(metrics.draftPositions, 39)
        XCTAssertEqual(metrics.draftProposed, 1)
        XCTAssertEqual(metrics.draftAccepted, 0)
        XCTAssertEqual(metrics.alpha, 0)
    }

    private static let statedArmPrompt = [
        0, 128803, 9602, 344, 270, 6102, 294, 63473, 6309, 128804, 128822,
    ]
    private static let statedArmGenerated = [
        671, 6102, 294, 33395, 344, 2619, 56, 35393, 666, 343, 7825, 412, 982,
        57, 1594, 12, 295, 8640, 797, 983, 344, 990, 270, 3924, 734, 9152, 4593,
        305, 14455, 412, 1009, 7267, 14, 5389, 14, 305, 5218, 6354, 16, 1,
    ]

    /// The definition the incremental index has to match: scan the whole
    /// history for the latest earlier occurrence of the tail.
    private static func rescanProposal(_ ids: [Int], orders: [Int]) -> Int? {
        for order in orders where ids.count >= order + 1 {
            let tail = Array(ids.suffix(order))
            for start in stride(from: ids.count - order - 1, through: 0, by: -1)
            where Array(ids[start..<(start + order)]) == tail {
                return ids[start + order]
            }
        }
        return nil
    }
}
