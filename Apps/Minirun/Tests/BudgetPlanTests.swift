import MinirunKit
import MinirunRunners
import XCTest

@testable import MinirunApp

/// The dial's arithmetic, checked against the numbers the record actually
/// carries. Every assertion here is a sentence the refusal card will say out
/// loud to an operator, so it had better be true.
final class BudgetPlanTests: XCTestCase {

    private var profile: MemoryProfile { CatalogFixtures.kimiK3.memory }

    private func plan(
        budget: UInt64, ceiling: UInt64 = 16 << 30, readAhead: Int = 0,
        maximumNewTokens: Int = 64
    )
        -> BudgetPlan
    {
        BudgetPlan(
            model: .kimiK3, modelName: "Kimi K3", profile: profile,
            budgetBytes: budget, maximumNewTokens: maximumNewTokens,
            deviceCeilingBytes: ceiling, readAheadDepth: readAhead)
    }

    private func v4Plan(budget: UInt64, ceiling: UInt64 = 16_000_000_000) -> BudgetPlan {
        let entry = CatalogFixtures.deepseekV4Flash
        return BudgetPlan(
            model: .deepseekV4Flash, modelName: entry.descriptor.displayName,
            profile: entry.memory, budgetBytes: budget, maximumNewTokens: 64,
            deviceCeilingBytes: ceiling, readAheadDepth: 1)
    }

    // MARK: The floor

    func testWideningDoublesTheStoredLayer() {
        XCTAssertEqual(profile.widestDeterministicLayerStoredBytes, 2_341_257_216)
        XCTAssertEqual(profile.widenedResidentBytes, 4_682_514_432)
    }

    /// The product-chat floor includes the complete second-pass rolling state
    /// and allocator/process headroom, not the one-token harness reserve.
    func testArithmeticFloorIsWidenedPlusPoolPlusReserve() {
        XCTAssertEqual(profile.workingSetReserveBytes, 2_581_886_208)
        XCTAssertEqual(profile.expertPoolBytes, 46_792_704)
        XCTAssertEqual(profile.arithmeticFloorBytes, 7_311_193_344)
        XCTAssertEqual(
            profile.arithmeticFloorBytes,
            profile.widenedResidentBytes + profile.expertPoolBytes
                + profile.workingSetReserveBytes)
    }

    /// The product card stays short while retaining every value the operator
    /// needs to understand and fix the refusal. Planner operand identities are
    /// tested above rather than repeated as engineering prose in Settings.
    func testRefusalNamesFinalRequiredStatedAndDeficit() throws {
        let candidate = plan(budget: 4_000_000_000)
        let refusal = try XCTUnwrap(candidate.refusal)
        let copy = RefusalCard.bodyLines(plan: candidate, refusal: refusal).joined(separator: " ")

        XCTAssertTrue(copy.contains(MRFormat.bytesDecimal(profile.refusalThresholdBytes)))
        XCTAssertTrue(copy.contains(MRFormat.bytesDecimal(candidate.budgetBytes)))
        XCTAssertTrue(copy.contains(MRFormat.bytesDecimal(refusal.deficitBytes)))
        XCTAssertFalse(copy.localizedCaseInsensitiveContains("float32"))
        XCTAssertFalse(copy.localizedCaseInsensitiveContains("expert pool"))
    }

    /// 5.8 GB remains the historical one-token record. Product chat uses a
    /// distinct conservative boundary above its exact second-pass floor, and
    /// the UI must not relabel that safety margin as another completed run.
    func testRefusalThresholdKeepsProductSafetySeparateFromTheOneTokenRecord() {
        XCTAssertEqual(profile.onRecordMinimumBudgetBytes, 5_800_000_000)
        XCTAssertEqual(profile.requiredMinimumBudgetBytes, 8_000_000_000)
        XCTAssertGreaterThan(profile.refusalThresholdBytes, profile.arithmeticFloorBytes)
        XCTAssertEqual(profile.refusalThresholdBytes, 8_000_000_000)
    }

    // MARK: Refusals

    func testBudgetBelowSeveralGatesIsOfferedTheFinalRunnableBoundaryOnce() {
        let refused = plan(budget: 4_000_000_000)
        guard case .belowRequiredMinimum(let required, let stated)? = refused.refusal else {
            return XCTFail("expected the highest refusal, got \(String(describing: refused.refusal))")
        }
        XCTAssertEqual(required, 8_000_000_000)
        XCTAssertEqual(stated, 4_000_000_000)
        XCTAssertEqual(refused.refusal?.deficitBytes, 4_000_000_000)
        XCTAssertEqual(refused.refusal?.suggestedBudgetBytes, 8_000_000_000)
        XCTAssertFalse(refused.isRunnable)
    }

    func testPureArithmeticProfileStillNamesItsExactFloor() {
        let arithmeticOnly = MemoryProfile(
            census: profile.census,
            expertPoolBytes: profile.expertPoolBytes,
            workingSetReserveBytes: profile.workingSetReserveBytes,
            requiredMinimumBudgetBytes: nil,
            onRecordMinimumBudgetBytes: nil,
            provenance: profile.provenance)
        let refused = BudgetPlan(
            model: .kimiK3, modelName: "Kimi K3", profile: arithmeticOnly,
            budgetBytes: 4_000_000_000, maximumNewTokens: 64,
            deviceCeilingBytes: 16 << 30,
            readAheadDepth: 0)

        guard case .belowArithmeticFloor(let required, let stated)? = refused.refusal else {
            return XCTFail("expected an arithmetic refusal")
        }
        XCTAssertEqual(required, 7_311_193_344)
        XCTAssertEqual(stated, 4_000_000_000)
    }

    func testBudgetAboveTheFloorButBelowProductSafetyIsStillRefused() {
        let refused = plan(budget: 7_995_000_000)
        guard case .belowRequiredMinimum(let minimum, _)? = refused.refusal else {
            return XCTFail("expected a product safety refusal")
        }
        XCTAssertEqual(minimum, 8_000_000_000)
        XCTAssertEqual(refused.refusal?.deficitBytes, 5_000_000)
    }

    /// Nothing clamps. A budget over the device ceiling is refused, not reduced
    /// to what fits.
    func testBudgetOverTheDeviceCeilingIsRefusedRatherThanClamped() {
        let refused = plan(budget: 8_000_000_000, ceiling: 6_442_450_944)
        guard case .aboveDeviceCeiling(let ceiling, let stated)? = refused.refusal else {
            return XCTFail("expected a ceiling refusal")
        }
        XCTAssertEqual(ceiling, 6_442_450_944)
        XCTAssertEqual(stated, 8_000_000_000)
        XCTAssertEqual(refused.budgetBytes, 8_000_000_000, "the stated budget is never rewritten")
    }

    func testTheProductSafetyBoundaryIsAccepted() {
        XCTAssertNil(plan(budget: 8_000_000_000, ceiling: 16 << 30).refusal)
        XCTAssertTrue(
            plan(budget: 8_000_000_000, ceiling: 6_442_450_944).deviceCannotHostModel)
    }

    // MARK: Tiers

    func testTiersSumToTheStatedBudgetAtEveryBudget() {
        for budget in stride(from: UInt64(0), through: 12_000_000_000, by: 97_000_000) {
            let candidate = plan(budget: budget, ceiling: 16 << 30, readAhead: 1)
            XCTAssertTrue(
                candidate.tiersAccountForBudget,
                "floor + staged + pinned + hot + free ≠ budget at \(budget)")
        }
    }

    // MARK: V4's ladder

    /// The dial's floor, from `docs/design/v4-memory-dial.md` §4's own terms:
    /// the 1.8 GB measured transient execution envelope, the routed-expert
    /// pool, the saturated MLX cache and the stated pin margin.
    private let v4LadderFloor: UInt64 = 2_623_132_416
    /// 43 layers at 136,495,488 B read and 33/32 of that resident.
    private let v4LayerResidentBytes: UInt64 = 140_760_972
    /// The ladder's last rung: the `[129280, 4096]` BF16 output head, held in
    /// its stored form because the walk keeps widening per window.
    private let v4OutputHeadBytes: UInt64 = 1_059_061_760
    private var v4EveryLayerBudget: UInt64 { v4LadderFloor + 43 * v4LayerResidentBytes }
    private var v4FullResidencyBudget: UInt64 { v4EveryLayerBudget + v4OutputHeadBytes }

    /// The dial's ladder is the kit's, not a second implementation of it.
    ///
    /// This is the same agreement `testTheAppsTiersAreThePlannersFieldsAtEvery‌InterestingBudget`
    /// asserts for K3, and it is why the V4 branch could be given real presets
    /// at all: `BudgetPlan` reads a `PinPlan` off `DeepSeekV4MemoryDial` rather
    /// than computing a composition of its own.
    func testTheV4TiersAreTheKitPlannersFields() throws {
        let ladder = try XCTUnwrap(CatalogFixtures.deepseekV4Flash.memory.deepSeekV4)
        for budget: UInt64 in [
            3_000_000_000, 4_000_000_000, 8_000_000_000, v4EveryLayerBudget,
            v4FullResidencyBudget, 12_000_000_000,
        ] {
            let candidate = v4Plan(budget: budget, ceiling: 34_400_000_000)
            let reference = try DeepSeekV4MemoryDial.plan(
                budgetBytes: budget, census: ladder.census, floor: ladder.floor,
                maximumNewTokens: candidate.maximumNewTokens)

            XCTAssertEqual(candidate.pinnedBytes, reference.pinnedBytes, "pinned at \(budget)")
            XCTAssertEqual(
                candidate.pinnedLayerCount, reference.pinnedLayers.count, "layers at \(budget)")
            XCTAssertEqual(candidate.floorBytes, reference.workingFloorBytes, "floor at \(budget)")
            XCTAssertEqual(candidate.freeBytes, reference.unusedBudgetBytes, "free at \(budget)")
            XCTAssertEqual(
                candidate.bytesSavedPerToken, reference.projectedBytesPerTokenSaved,
                "saved at \(budget)")
            XCTAssertEqual(candidate.stagedBytes, 0, "V4's read-ahead is named, not built")
            XCTAssertEqual(candidate.hotSetBytes, 0)
            XCTAssertTrue(candidate.tiersAccountForBudget, "tiers ≠ budget at \(budget)")
        }
    }

    /// Between the product floor and the dial's floor a V4 chat runs and pins
    /// nothing. That is not a refusal, and the bar draws the admitted envelope.
    func testAV4BudgetBelowTheDialFloorRunsAndPinsNothing() {
        let candidate = v4Plan(budget: 2_100_000_000)

        XCTAssertEqual(candidate.strategy, .boundedLayerStreaming)
        XCTAssertTrue(candidate.isRunnable)
        XCTAssertNil(candidate.pinPlan)
        XCTAssertEqual(
            candidate.floorBytes, DeepSeekV4ProductMemoryBudget.minimumBudgetBytes)
        XCTAssertEqual(
            candidate.freeBytes,
            2_100_000_000 - DeepSeekV4ProductMemoryBudget.minimumBudgetBytes)
        XCTAssertEqual(candidate.pinnedBytes, 0)
        XCTAssertTrue(candidate.tiersAccountForBudget)
        // It pins nothing, so it reads everything — which is the number the
        // panel showed as "0 MB" before this branch existed. Everything is the
        // 43 layers, the whole output-head walk, and one embedding row.
        XCTAssertEqual(
            candidate.bytesReadPerToken, 5_869_305_984 + 1_059_061_760 + 8_192)
        XCTAssertEqual(candidate.bytesSavedPerToken, 0)
    }

    /// Dragging a V4 budget up must buy resident layers, not empty space —
    /// the same property the K3 dial was rewritten for.
    func testRaisingAV4BudgetPinsStrictlyMoreLayers() {
        var previous: UInt64 = 0
        var sawAnIncrease = false
        for budget in stride(
            from: UInt64(2_000_000_000), through: 12_000_000_000, by: 211_000_000)
        {
            let candidate = v4Plan(budget: budget, ceiling: 34_400_000_000)
            XCTAssertGreaterThanOrEqual(
                candidate.pinnedBytes, previous, "the pinned tier shrank at \(budget)")
            if candidate.pinnedBytes > previous { sawAnIncrease = true }
            previous = candidate.pinnedBytes
            XCTAssertTrue(candidate.tiersAccountForBudget, "tiers ≠ budget at \(budget)")
        }
        XCTAssertTrue(sawAnIncrease)
    }

    /// **Balanced is the point at which the whole ladder is resident: every
    /// deterministic layer *and* the output head.**
    ///
    /// The preset it replaces was a flat three-fifths of the device ceiling —
    /// 20.6 GB on this 34.4 GB Mac — and there is nothing on the ladder after
    /// the head to spend the remaining ~11 GB on.
    func testV4BalancedHoldsEveryLayerAndTheHeadOnAMacThatCanReachIt() {
        let candidate = v4Plan(budget: 2_000_000_000, ceiling: 34_400_000_000)

        XCTAssertEqual(candidate.budget(for: .floor), 2_000_000_000)
        XCTAssertEqual(candidate.budget(for: .balanced), v4FullResidencyBudget)
        XCTAssertEqual(candidate.budget(for: .generous), 34_400_000_000)
        XCTAssertLessThanOrEqual(
            candidate.budget(for: .balanced), 34_400_000_000 * 3 / 5,
            "full residency has to be inside the 60 % rule to be chosen by it")

        let balanced = candidate.with(budgetBytes: candidate.budget(for: .balanced))
        XCTAssertEqual(balanced.pinnedLayerCount, 43)
        XCTAssertTrue(balanced.pinsOutputHead)
        XCTAssertEqual(balanced.residentUnitsLabel, "43 of 43 + output head")
        XCTAssertEqual(
            balanced.pinnedBytes, 43 * v4LayerResidentBytes + v4OutputHeadBytes)
        // What a token still reads at the top of the ladder is one embedding
        // row: the table it comes from is the same size as the head and is
        // never a rung.
        XCTAssertEqual(balanced.bytesReadPerToken, 8_192)
        XCTAssertEqual(balanced.freeBytes, 0, "the last snap point spends the whole ladder")
        XCTAssertEqual(
            candidate.explanation(for: .balanced),
            "All 43 layers + output head resident — about 6.93 GB fewer bytes per token.")
        XCTAssertFalse(
            candidate.explanation(for: .balanced).contains("moderate hard ceiling"))
    }

    /// One rung below Balanced: every layer is held and the head is not, and
    /// the panel says what the next 1.06 GB would buy.
    ///
    /// This is the ordering rule made visible in the product — the head is
    /// behind every layer on the ladder, so a budget that can fund the layers
    /// but not the head funds the layers.
    func testV4HoldsEveryLayerBeforeItHoldsTheHead() {
        let candidate = v4Plan(budget: v4EveryLayerBudget, ceiling: 34_400_000_000)

        XCTAssertEqual(candidate.pinnedLayerCount, 43)
        XCTAssertFalse(candidate.pinsOutputHead)
        XCTAssertEqual(candidate.residentUnitsLabel, "43 of 43")
        XCTAssertEqual(candidate.pinnedBytes, 43 * v4LayerResidentBytes)
        // The whole head walk and the embedding row are still read.
        XCTAssertEqual(candidate.bytesReadPerToken, 1_059_061_760 + 8_192)
        XCTAssertEqual(
            candidate.nextPinSentence, "+1.06 GB pins the output head.")
        XCTAssertTrue(candidate.tiersAccountForBudget)

        // One byte short of the head is the same plan; the head's own budget
        // pins it and nothing else changes.
        let short = candidate.with(budgetBytes: v4FullResidencyBudget - 1)
        XCTAssertFalse(short.pinsOutputHead)
        XCTAssertEqual(short.pinnedLayerCount, 43)
    }

    /// A 6 GB phone cannot reach full residency, so Balanced falls back to the
    /// largest boundary under three-fifths of its ceiling — K3's rule, applied
    /// to V4's ladder.
    func testV4BalancedFallsBackToTheLargestBoundaryASmallDeviceCanReach() {
        let candidate = v4Plan(budget: 2_000_000_000, ceiling: 6_000_000_000)
        let balanced = candidate.budget(for: .balanced)

        XCTAssertLessThan(balanced, v4FullResidencyBudget)
        XCTAssertLessThanOrEqual(balanced, 6_000_000_000 * 3 / 5)
        XCTAssertGreaterThan(balanced + v4LayerResidentBytes, 6_000_000_000 * 3 / 5)
        XCTAssertEqual(balanced, v4LadderFloor + 6 * v4LayerResidentBytes)
        XCTAssertEqual(candidate.with(budgetBytes: balanced).pinnedLayerCount, 6)
        XCTAssertEqual(
            candidate.explanation(for: .balanced),
            "6 of 43 layers resident — about 819 MB fewer bytes per token.")
        XCTAssertNil(candidate.presetDeficit(.balanced))
    }

    /// When not even the first pin fits, the preset states the first pin
    /// boundary anyway — over the ceiling, disabled, with its deficit. It never
    /// quietly becomes the floor and tells the operator this device can do
    /// something it cannot.
    func testV4BalancedIsShownWithItsDeficitWhenTheFirstPinIsOutOfReach() throws {
        let candidate = v4Plan(budget: 2_000_000_000, ceiling: 2_700_000_000)
        let balanced = candidate.budget(for: .balanced)

        XCTAssertEqual(balanced, v4LadderFloor + v4LayerResidentBytes)
        XCTAssertGreaterThan(balanced, candidate.budget(for: .floor))
        XCTAssertEqual(try XCTUnwrap(candidate.presetDeficit(.balanced)), balanced - 2_700_000_000)
    }

    func testV4FloorPinsNothingAndSaysSo() {
        let candidate = v4Plan(budget: 2_000_000_000, ceiling: 34_400_000_000)
        XCTAssertEqual(candidate.pinnedLayerCount, 0)
        XCTAssertEqual(
            candidate.explanation(for: .floor),
            "The minimum admitted V4 envelope. Nothing is held resident: every layer streams, "
                + "every token.")
    }

    func testRefusedV4BudgetAllocatesNoDisplayedTier() {
        let refused = v4Plan(
            budget: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes - 1)

        XCTAssertFalse(refused.isRunnable)
        XCTAssertEqual(refused.floorBytes, 0)
        XCTAssertEqual(refused.freeBytes, 0)
        XCTAssertTrue(refused.tiersAccountForBudget)
    }

    func testStagedTierAppearsOnlyWithReadAheadAndOnlyAboveTheFloor() {
        let noReadAhead = plan(budget: 8_000_000_000, ceiling: 12 << 30, readAhead: 0)
        XCTAssertEqual(noReadAhead.stagedBytes, 0)

        let belowFloor = plan(budget: 5_000_000_000, ceiling: 8 << 30, readAhead: 1)
        XCTAssertEqual(belowFloor.stagedBytes, 0)

        let funded = plan(budget: 9_000_000_000, ceiling: 12 << 30, readAhead: 1)
        XCTAssertEqual(funded.stagedBytes, 1_267_810_304)
        XCTAssertTrue(funded.stagedTierIsFunded)
    }

    // MARK: The pinned tier — the bug this file was rewritten for

    /// Dragging the budget UP must buy resident layers, not empty space.
    ///
    /// This is the owner's report, as an assertion. The dial's own arithmetic
    /// had no pinned tier at all: everything above the staged pair went to a hot
    /// set and then to free, so raising the budget grew the part of the bar that
    /// means "allocated to nothing". The derived ladder says the opposite — a
    /// deterministic layer returns 1.000 bytes saved per resident byte, the best
    /// thing after it returns 0.499, so every byte of surplus buys layers until
    /// all 93 are held.
    func testRaisingTheBudgetPinsStrictlyMoreLayers() {
        let ceiling: UInt64 = 32 << 30
        var previousBytes: UInt64 = 0
        var previousLayers = 0
        var sawAnIncrease = false

        for budget in stride(from: UInt64(6) << 30, through: UInt64(30) << 30, by: 1 << 30) {
            let candidate = plan(budget: budget, ceiling: ceiling, readAhead: 1)
            XCTAssertGreaterThanOrEqual(
                candidate.pinnedBytes, previousBytes,
                "the pinned tier shrank between budgets at \(budget)")
            if candidate.pinnedBytes > previousBytes { sawAnIncrease = true }
            previousBytes = candidate.pinnedBytes
            previousLayers = candidate.pinnedLayerCount
        }
        XCTAssertTrue(sawAnIncrease, "no budget in the whole domain bought a single layer")
        XCTAssertGreaterThanOrEqual(
            previousLayers, 20, "a 30 GiB budget should hold about twenty layers")
    }

    /// Free is what is left under the next layer's price, and nothing more.
    ///
    /// The failing behaviour was free growing without bound as the slider moved
    /// right. It now cannot reach the price of a layer, because anything that
    /// large would have been spent on one.
    func testFreeNeverExceedsTheLargestUnitTheLadderCouldStillBuy() {
        for budget in stride(from: UInt64(6) << 30, through: UInt64(30) << 30, by: 517_000_000) {
            let candidate = plan(budget: budget, ceiling: 32 << 30, readAhead: 1)
            guard candidate.isRunnable else { continue }
            XCTAssertLessThan(
                candidate.freeBytes, profile.widestDeterministicLayerStoredBytes,
                "free ran past the price of a layer at \(budget), so a layer went unbought")
        }
    }

    /// The pin set is not always a clean prefix, and that is not a bug.
    ///
    /// The ladder maximises bytes saved, not layers held. At a budget that
    /// cannot take the next 1,267,810,304 B KDA layer, an 844,398,592 B MLA
    /// layer further along still fits, so the ranges have a hole in them. The
    /// screen prints the ranges precisely so that hole reads as arithmetic.
    func testThePinSetMayReachPastALayerItCannotAfford() {
        // Preserve the same surplus over the floor as the archived 24 GiB
        // vector after adding the exact second-pass state term.
        let candidate = plan(
            budget: (24 << 30) + 2_452_295_680, ceiling: 32 << 30, readAhead: 1)
        XCTAssertEqual(candidate.pinnedLayerRanges, [0...15, 19...19])
        XCTAssertEqual(candidate.pinnedBytes, 20_509_163_520)
        XCTAssertEqual(candidate.pinnedLayerCount, 17)
    }

    func testPinnedTierNamesFirstFillAndTheConversationReuseHorizon() throws {
        let budget: UInt64 = (24 << 30) + 2_452_295_680
        let complete = plan(
            budget: budget, ceiling: 32 << 30, readAhead: 1,
            maximumNewTokens: 64)
        XCTAssertEqual(complete.maximumNewTokens, 64)
        XCTAssertEqual(
            complete.firstPassPinnedLayerFillBytes,
            try XCTUnwrap(complete.pinPlan?.projectedFirstPassPinnedLayerFillBytes))
        XCTAssertEqual(
            complete.pinnedLayerBytesServedAtTokenLimit,
            try XCTUnwrap(complete.pinPlan?.projectedPinnedLayerBytesServedAtTokenLimit))
        let completeCopy = try XCTUnwrap(complete.pinReuseSentence)
        XCTAssertTrue(completeCopy.contains("The first token fills"), completeCopy)
        XCTAssertTrue(completeCopy.contains("tokens 2–64"), completeCopy)

        let one = plan(
            budget: budget, ceiling: 32 << 30, readAhead: 1,
            maximumNewTokens: 1)
        XCTAssertEqual(one.maximumNewTokens, 1)
        XCTAssertEqual(one.pinnedLayerBytesServedAtTokenLimit, 0)
        let oneTokenCopy = try XCTUnwrap(one.pinReuseSentence)
        XCTAssertTrue(oneTokenCopy.contains("cannot reuse"), oneTokenCopy)
    }

    func testChangingBudgetOrReadAheadKeepsTheResponseHorizon() {
        let original = plan(
            budget: 24 << 30, ceiling: 32 << 30, readAhead: 1,
            maximumNewTokens: 37)
        XCTAssertEqual(original.with(budgetBytes: 16 << 30).maximumNewTokens, 37)
        XCTAssertEqual(original.with(readAheadDepth: 0).maximumNewTokens, 37)
    }

    /// The app's tiers and the kit's plan are the same numbers, at seven budgets
    /// including both snap points the design document names.
    ///
    /// This is the agreement the whole relocation was for. `BudgetPlan` no
    /// longer computes a composition; it reads one off a `PinPlan`. If that ever
    /// stops being true, this fails.
    func testTheAppsTiersAreThePlannersFieldsAtEveryInterestingBudget() throws {
        let census = CatalogFixtures.k3Census
        let floor = profile.floor
        for budget: UInt64 in [
            8_000_000_000, 8 << 30, 16 << 30, 24 << 30, 32 << 30,
            116_129_117_440, 120_826_787_072,
        ] {
            let candidate = plan(budget: budget, ceiling: 128 << 30, readAhead: 1)
            let reference = try MemoryDialPlanner.plan(
                budgetBytes: budget, census: census, floor: floor, readAhead: .depthOne,
                maximumNewTokens: candidate.maximumNewTokens)

            XCTAssertEqual(candidate.pinnedBytes, reference.pinnedBytes, "pinned at \(budget)")
            XCTAssertEqual(
                candidate.pinnedLayerCount, reference.pinnedLayers.count, "layers at \(budget)")
            XCTAssertEqual(
                candidate.hotSetBytes, reference.expertHotSetBytes, "hot set at \(budget)")
            XCTAssertEqual(candidate.freeBytes, reference.unusedBudgetBytes, "free at \(budget)")
            XCTAssertEqual(
                candidate.floorBytes + candidate.stagedBytes, reference.workingFloorBytes,
                "floor at \(budget)")
            XCTAssertEqual(
                candidate.bytesSavedPerToken, reference.projectedBytesPerTokenSaved,
                "saved per token at \(budget)")
            XCTAssertTrue(reference.isResidencyBalanced, "the plan itself at \(budget)")
            XCTAssertTrue(candidate.tiersAccountForBudget, "the app's tiers at \(budget)")
        }
    }

    /// 116,129,117,440 B holds all 93 layers and still pins no expert; the
    /// globals bundle is the next unit, not an expert. The derived ladder
    /// disagrees with spec §11.5's table here, and this is that disagreement
    /// asserted rather than argued.
    func testFullDeterministicResidencyIsTheLastPinBeforeAnythingElse() {
        let full = plan(budget: 116_129_117_440, ceiling: 128 << 30, readAhead: 1)
        XCTAssertEqual(full.pinnedLayerCount, 93)
        XCTAssertEqual(full.pinnedBytes, 108_817_924_096)
        XCTAssertEqual(full.hotSetBytes, 0)
        XCTAssertEqual(full.freeBytes, 0)

        let withGlobals = plan(budget: 120_826_787_072, ceiling: 128 << 30, readAhead: 1)
        XCTAssertEqual(withGlobals.pinnedBytes, 113_515_593_728)
        XCTAssertEqual(withGlobals.hotSetBytes, 0)
    }

    /// Free grows only once there is nothing left to buy.
    func testFreeOnlyGrowsPastFullResidency() {
        let full = plan(budget: 120_826_787_072, ceiling: 256 << 30, readAhead: 1)
        XCTAssertEqual(full.freeBytes, 0)
        let over = plan(budget: 130_000_000_000, ceiling: 256 << 30, readAhead: 1)
        XCTAssertEqual(over.freeBytes, 9_173_212_928)
        XCTAssertEqual(over.pinnedBytes, full.pinnedBytes, "there was nothing more to pin")
    }

    /// A refused configuration allocates nothing. The legend and the bar have to
    /// say the same thing.
    func testARefusedBudgetAllocatesNothing() {
        let refused = plan(budget: 2_329_473_787)
        XCTAssertEqual(refused.floorBytes, 0)
        XCTAssertEqual(refused.stagedBytes, 0)
        XCTAssertEqual(refused.pinnedBytes, 0)
        XCTAssertEqual(refused.hotSetBytes, 0)
        XCTAssertEqual(refused.freeBytes, 0)
        XCTAssertNil(refused.pinPlan)
        XCTAssertTrue(refused.tiersAccountForBudget)
        XCTAssertEqual(refused.budgetBytes, 2_329_473_787, "the stated budget stands")
    }

    // MARK: The decision the storage layer would print

    func testDecisionMirrorsTheStorageLayersArithmetic() {
        let decision = plan(budget: 7_000_000_000, readAhead: 1).decision
        XCTAssertEqual(decision.residentBytes, 4_682_514_432)
        XCTAssertEqual(decision.stagedBytes, 1_267_810_304)
        XCTAssertEqual(decision.reserveBytes, 2_581_886_208)
        XCTAssertEqual(decision.requiredBytes, 8_532_210_944)
        XCTAssertFalse(decision.isAllowed)
        XCTAssertEqual(decision.headroomBytes, -1_532_210_944)
        XCTAssertTrue(decision.refusalError.description.hasPrefix("invalid argument: "))
    }

    // MARK: Presets

    /// K3's presets did not move. The V4 work changed the strategy branch above
    /// them and nothing else, and these are the exact bytes the residency
    /// branch produced before it.
    func testK3PresetsAreByteIdenticalOnTheExistingFixtures() {
        let mac = plan(budget: 8_000_000_000, ceiling: 34_400_000_000, readAhead: 1)
        XCTAssertEqual(mac.budget(for: .floor), 8_000_000_000)
        XCTAssertEqual(mac.budget(for: .balanced), 21_483_730_176)
        XCTAssertEqual(mac.budget(for: .generous), 34_159_408_384)

        let phone = plan(budget: 8_000_000_000, ceiling: 6_000_000_000, readAhead: 1)
        XCTAssertEqual(phone.budget(for: .floor), 8_000_000_000)
        XCTAssertEqual(phone.budget(for: .balanced), 10_920_260_864)
        XCTAssertEqual(phone.budget(for: .generous), 14_300_280_064)
        XCTAssertNotNil(
            phone.presetDeficit(.balanced),
            "a 6 GB device is told by how much Balanced misses, not shown a smaller one")
    }

    // MARK: The default a new chat starts from

    /// Mac and phone start a V4 chat in different places, and the reason is
    /// evidence rather than taste: the Mac's ladder reaches full residency well
    /// inside the device, and no pinned V4 arm has ever been run on a phone.
    @MainActor
    func testTheDefaultPresetIsPlatformConditionalForV4Only() {
        #if os(macOS)
            XCTAssertEqual(AppModel.defaultPreset(for: .deepseekV4Flash), .balanced)
        #else
            XCTAssertEqual(AppModel.defaultPreset(for: .deepseekV4Flash), .floor)
        #endif
        XCTAssertEqual(AppModel.defaultPreset(for: .kimiK3), .floor)
        XCTAssertEqual(AppModel.defaultPreset(for: .minimaxH3), .floor)
        XCTAssertEqual(AppModel.defaultPreset(for: ModelID("future-model")), .floor)
    }

    func testPresetsClimbAndAreReportedRatherThanHiddenWhenTheyDoNotFit() {
        let candidate = plan(budget: 8_000_000_000, ceiling: 8_500_000_000)
        XCTAssertEqual(candidate.budget(for: .floor), 8_000_000_000)
        XCTAssertGreaterThan(candidate.budget(for: .balanced), candidate.budget(for: .floor))
        XCTAssertGreaterThan(candidate.budget(for: .generous), candidate.budget(for: .balanced))
        XCTAssertNotNil(
            candidate.presetDeficit(.generous),
            "Generous does not fit this 8.50 GB device and must state by how much")
    }

    // MARK: Read-ahead, after the v0.6.19 default flip

    /// Auto is on. The screen used to call an unset knob "the runner's own
    /// default" and then draw the dial as if that default were zero; since spec
    /// v0.6.19 the prefetchers default on wherever a budget is stated, and the
    /// dial has to draw the run that will actually happen.
    func testAutoMeansOnAndIsStillADifferentStatementFromZero() {
        XCTAssertEqual(ReadAheadSetting(nil), .auto)
        XCTAssertEqual(ReadAheadSetting(nil).effectiveDepth, 1)
        XCTAssertNil(ReadAheadSetting.auto.knobValue, "auto leaves the knob unset")

        XCTAssertEqual(ReadAheadSetting(0), .off)
        XCTAssertEqual(ReadAheadSetting.off.effectiveDepth, 0)
        XCTAssertEqual(ReadAheadSetting.off.knobValue, 0, "off states zero, and says so on the wire")

        XCTAssertTrue(ReadAheadSetting(1).isProductSupported)
        XCTAssertEqual(ReadAheadSetting(2), .explicit(2))
        XCTAssertEqual(ReadAheadSetting.explicit(2).effectiveDepth, 2)
        XCTAssertEqual(ReadAheadSetting.explicit(2).knobValue, 2)
        XCTAssertFalse(ReadAheadSetting.explicit(2).isProductSupported)
    }

    /// The knob round-trips. A control that could not recover the value it was
    /// given would silently rewrite a conversation's settings on every render.
    func testEveryKnobValueSurvivesTheRoundTrip() {
        for knob: Int? in [nil, 0, 1, 2, 4] {
            XCTAssertEqual(ReadAheadSetting(knob).knobValue, knob)
        }
    }

    /// Older builds offered depths above one. Preserve the document value, but
    /// do not silently project it as the measured depth-one plan.
    func testUnsupportedSavedDepthIsRefusedRatherThanClampedIntoAPlan() throws {
        let candidate = plan(budget: 8_000_000_000, readAhead: 2)
        let refusal = try XCTUnwrap(candidate.readAheadRefusal)

        XCTAssertEqual(refusal.requestedDepth, 2)
        XCTAssertTrue(refusal.message.contains("Auto (one layer) or Off"))
        XCTAssertTrue(refusal.namedError.contains("0...1"))
        XCTAssertNil(candidate.pinPlan)
        XCTAssertFalse(candidate.isRunnable)
        XCTAssertEqual(candidate.floorBytes, 0)
        XCTAssertEqual(candidate.freeBytes, 0)
    }

    /// The staged tier appears under auto, which is the visible consequence of
    /// the flip: at the same budget, the old reading drew no staged pair at all.
    func testTheStagedTierIsDrawnUnderAutoAndAbsentUnderOff() {
        let budget: UInt64 = 8_000_000_000
        let auto = plan(
            budget: budget, ceiling: 12 << 30,
            readAhead: ReadAheadSetting.auto.effectiveDepth)
        let off = plan(
            budget: budget, ceiling: 12 << 30,
            readAhead: ReadAheadSetting.off.effectiveDepth)

        XCTAssertGreaterThan(auto.stagedBytes, 0)
        XCTAssertEqual(off.stagedBytes, 0)
        // The staged pair is drawn inside the floor rather than out of the
        // surplus, because that is where its bytes actually come from: the floor
        // reserves the widest widened layer, and a narrower resident layer
        // leaves exactly that room. So turning read-ahead off moves the segment
        // back into the floor and changes nothing else — same pins, same free.
        XCTAssertEqual(off.floorBytes, auto.floorBytes + auto.stagedBytes)
        XCTAssertEqual(off.pinnedBytes, auto.pinnedBytes)
        XCTAssertEqual(off.freeBytes, auto.freeBytes)
    }

    // MARK: The composition bar's arithmetic

    /// The composition bar is normalised to the budget, so its segments have to
    /// add up to the budget exactly — at every read-ahead setting and at both
    /// ends of the domain. This is the identity the bar's own balance dot
    /// reports, asserted here so a drawing bug is a test failure.
    func testTheTiersAlwaysAccountForExactlyTheStatedBudget() {
        for depth in [0, 1] {
            for budget: UInt64 in [
                0, 1_000_000_000, profile.refusalThresholdBytes,
                profile.refusalThresholdBytes + 1, 8_000_000_000, 12 << 30,
            ] {
                let candidate = plan(budget: budget, ceiling: 12 << 30, readAhead: depth)
                XCTAssertTrue(
                    candidate.tiersAccountForBudget,
                    "depth \(depth), budget \(budget): tiers ≠ budget")
                let sum =
                    candidate.floorBytes + candidate.stagedBytes + candidate.pinnedBytes
                    + candidate.hotSetBytes + candidate.freeBytes
                XCTAssertEqual(sum, candidate.isRunnable ? budget : 0)
            }
        }
    }

    /// A refused budget allocates nothing, so the bar has nothing to apportion.
    /// Drawing a plausible split of a budget nobody accepted is the failure this
    /// guards.
    func testARefusedBudgetHasNoCompositionToDraw() {
        let refused = plan(budget: 1_000_000_000, ceiling: 12 << 30, readAhead: 1)
        XCTAssertFalse(refused.isRunnable)
        XCTAssertEqual(refused.floorBytes, 0)
        XCTAssertEqual(refused.stagedBytes, 0)
        XCTAssertEqual(refused.pinnedBytes, 0)
        XCTAssertEqual(refused.hotSetBytes, 0)
        XCTAssertEqual(refused.freeBytes, 0)
    }
}
