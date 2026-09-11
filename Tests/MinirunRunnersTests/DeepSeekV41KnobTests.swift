import MinirunKit
import XCTest

@testable import MinirunRunners

/// What a V4.1 run gets when it states nothing, and what it gets when it
/// states something.
///
/// The transfer mode is here rather than inside the gate because its default is
/// a *decision with a measurement behind it*:
/// `docs/experiments/2026-09-11-v41-phase3-runner.md` §6 ran both modes from one
/// binary and adoption lost, 4.160 s a token against 3.046, while halving the
/// terms it removes. A default that quietly flipped would cost a third of a
/// pinned decode with nothing saying so, and the 517 GB gate is not the place to
/// find that out.
final class DeepSeekV41KnobTests: XCTestCase {

    private func resolve(
        _ knobs: RunKnobs = RunKnobs(),
        scale: DeepSeekV41DecodeRunner.Scale,
        budget: UInt64 = 15_000_000_000
    ) throws -> DeepSeekV41EffectiveKnobs {
        try DeepSeekV41EffectiveKnobs.resolve(
            knobs, scale: scale, declaredBudgetBytes: budget)
    }

    func testTheTransferModeDefaultsOffAtEitherScale() throws {
        XCTAssertFalse(
            try resolve(
                scale: .stated(minimumBudgetBytes: 3_400_000_000, maximumNewTokens: 64)
            ).expertTileAdoption,
            "adoption halves the gather's own terms and loses the pass; §6 measured it")
        XCTAssertFalse(
            try resolve(scale: .product, budget: 3_400_000_000).expertTileAdoption)
    }

    func testAStatedTransferModeIsHonouredAtEitherScale() throws {
        var on = RunKnobs()
        on.expertTileAdoption = true
        XCTAssertTrue(
            try resolve(
                on, scale: .stated(minimumBudgetBytes: 3_400_000_000, maximumNewTokens: 64)
            ).expertTileAdoption)

        // A product run may ask for it too. Nothing here refuses it; its own
        // budget does, at the peak — 4,163,426,176 B against 3,400,000,000 in
        // the arm that tried — which is the refusal working rather than a clamp.
        XCTAssertTrue(
            try resolve(on, scale: .product, budget: 3_400_000_000).expertTileAdoption)
    }

    func testTheModeSurvivesTheRoundTripARunRecordIsWrittenFrom() throws {
        let resolved = try resolve(
            scale: .stated(minimumBudgetBytes: 3_400_000_000, maximumNewTokens: 64))
        XCTAssertEqual(resolved.asRunKnobs.expertTileAdoption, false)
        XCTAssertEqual(resolved.asStrings["expertTileAdoption"], "false")

        var on = RunKnobs()
        on.expertTileAdoption = true
        let adopting = try resolve(
            on, scale: .stated(minimumBudgetBytes: 3_400_000_000, maximumNewTokens: 64))
        XCTAssertEqual(adopting.asRunKnobs.expertTileAdoption, true)
        XCTAssertEqual(adopting.asStrings["expertTileAdoption"], "true")
    }

    func testTheRunnerAdvertisesTheKnobItConsumes() {
        XCTAssertTrue(DeepSeekV41DecodeRunner.knobs.contains("expertTileAdoption"))
        // Still absent, and still not by oversight: one `prefetch` queues all
        // three projections of the block's routed set.
        XCTAssertFalse(DeepSeekV41DecodeRunner.knobs.contains("expertProjectionPrefetch"))
        XCTAssertFalse(DeepSeekV41DecodeRunner.knobs.contains("expertCrossLayerPrefetch"))
    }
}
