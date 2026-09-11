import MinirunKit
import MinirunRunners
import XCTest

@testable import MinirunApp

/// The dial's arithmetic, checked against the numbers the record actually
/// carries. Every assertion here is a sentence the refusal card will say out
/// loud to an operator, so it had better be true.
final class BudgetPlanTests: XCTestCase {

    /// **The Mac's K3 profile, named rather than inherited.**
    ///
    /// This used to be `CatalogFixtures.kimiK3.memory`, which is priced by
    /// whichever platform the suite is *hosted* on. Every number below — the
    /// 2.58 GB working reserve, the 7.31 GB arithmetic floor, the 8.00 GB
    /// product boundary, the preset ladder — is the Mac's, so on the iPhone
    /// simulator the whole file asserted one platform's arithmetic against the
    /// other's policy and thirteen of its K3 tests failed for being hosted
    /// somewhere else.
    ///
    /// Naming the policy is the same move the fixtures already made for V4.1:
    /// a suite that means the Mac's numbers says so, and is byte-identical
    /// wherever it runs. That the profile the dial *actually* draws follows
    /// this platform's policy is the separate claim, asserted on purpose at the
    /// end of
    /// ``testRefusalThresholdKeepsProductSafetySeparateFromTheOneTokenRecord``.
    private var profile: MemoryProfile {
        CatalogFixtures.k3Memory(policy: K3ProductMemoryBudget.macOSProductPolicy)
    }

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
    ///
    /// The last three assertions are the platform half, and they are the ones
    /// that would have caught a fixture stating one platform's floor on the
    /// other: whatever this build is, the K3 row the dial draws is priced by
    /// *this* build's policy, and its refusal threshold is that policy's
    /// boundary — 8.00 GB on the Mac, the bounded tier's 5.80 GB on a phone.
    func testRefusalThresholdKeepsProductSafetySeparateFromTheOneTokenRecord() {
        XCTAssertEqual(profile.onRecordMinimumBudgetBytes, 5_800_000_000)
        XCTAssertEqual(profile.requiredMinimumBudgetBytes, 8_000_000_000)
        XCTAssertGreaterThan(profile.refusalThresholdBytes, profile.arithmeticFloorBytes)
        XCTAssertEqual(profile.refusalThresholdBytes, 8_000_000_000)

        let drawn = CatalogFixtures.kimiK3.memory
        let policy = K3ProductMemoryBudget.currentPolicy
        XCTAssertEqual(drawn.requiredMinimumBudgetBytes, policy.minimumBudgetBytes)
        XCTAssertEqual(
            drawn.workingSetReserveBytes,
            K3ProductMemoryBudget.defaultWorkingReserveBytes(for: policy))
        XCTAssertEqual(
            drawn.refusalThresholdBytes,
            max(policy.minimumBudgetBytes, 5_800_000_000),
            "the dial refuses at this platform's boundary, never at the other's")
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
        let ladder = try XCTUnwrap(CatalogFixtures.deepseekV4Flash.memory.deepSeekLadder)
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

    /// The sentence says "this model" rather than "V4" because two models draw
    /// it, and the one it used to name is not always the one running.
    func testV4FloorPinsNothingAndSaysSo() {
        let candidate = v4Plan(budget: 2_000_000_000, ceiling: 34_400_000_000)
        XCTAssertEqual(candidate.pinnedLayerCount, 0)
        XCTAssertEqual(
            candidate.explanation(for: .floor),
            "The minimum envelope this model is admitted at. Nothing is held resident: every "
                + "layer streams, every token.")
        XCTAssertFalse(candidate.explanation(for: .floor).contains("V4"))
    }

    func testRefusedV4BudgetAllocatesNoDisplayedTier() {
        let refused = v4Plan(
            budget: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes - 1)

        XCTAssertFalse(refused.isRunnable)
        XCTAssertEqual(refused.floorBytes, 0)
        XCTAssertEqual(refused.freeBytes, 0)
        XCTAssertTrue(refused.tiersAccountForBudget)
    }

    // MARK: V4.1's ladder

    /// V4.1's dial floor: the 5.2 GB transient envelope of a run that **pins**
    /// (phase 3's stated arm, not the 3.2 GB the floor arm measured with nothing
    /// resident), the 20-slot routed-expert pool at one expert's half-tile, the
    /// saturated 512 MiB MLX cache and the stated 256 MiB pin margin.
    private let v41LadderFloor: UInt64 =
        5_200_000_000 + 20 * 6_266_880 + 536_870_912 + 268_435_456
    /// Every dense block, as the loaded `BlockFP8Weights` form — the column the
    /// dial spends budget on. The phase 3 record's 8,729,882,048 B of resident
    /// weights at 15 GB is this plus the head below.
    private let v41EveryBlockResidentBytes: UInt64 = 7_406_054_848
    /// The ladder's last rung: the `[129280, 5120]` BF16 output head.
    private let v41OutputHeadBytes: UInt64 = 1_323_827_200
    /// What a pass re-reads when nothing is pinned, blocks only.
    private let v41EveryBlockReadBytes: UInt64 = 7_206_792_640
    private var v41EveryBlockBudget: UInt64 { v41LadderFloor + v41EveryBlockResidentBytes }
    private var v41FullResidencyBudget: UInt64 { v41EveryBlockBudget + v41OutputHeadBytes }

    /// The Mac's V4.1 profile, named for the reason ``profile`` is: the floor
    /// these tests assert is 3.4 GB, which is the macOS policy's and not the
    /// iPhone's 1.9 GB. The phone's own dial is asserted at the byte in
    /// `DeepSeekV41ProductRuntimeTests`.
    private var v41Profile: MemoryProfile {
        CatalogFixtures.deepSeekV41Memory(
            policy: DeepSeekV41ProductMemoryBudget.macOSProductPolicy)
    }

    private func v41Plan(budget: UInt64, ceiling: UInt64 = 34_400_000_000) -> BudgetPlan {
        let entry = CatalogFixtures.deepseekV41Flash
        return BudgetPlan(
            model: .deepseekV41Flash, modelName: entry.descriptor.displayName,
            profile: v41Profile, budgetBytes: budget, maximumNewTokens: 64,
            deviceCeilingBytes: ceiling, readAheadDepth: 1)
    }

    /// **V4.1 plans like V4 and not like K3.**
    ///
    /// This is the whole defect in one assertion. The strategy used to be
    /// `.residency` for V4.1, so the plan went to `MemoryDialPlanner` over
    /// `unknownGeometryCensus` — one zero-sized layer — and every budget bought
    /// the same nothing.
    func testV41StreamsRatherThanPlanningResidencyOverAnEmptyCensus() {
        let candidate = v41Plan(budget: v41FullResidencyBudget)
        XCTAssertEqual(candidate.strategy, .boundedLayerStreaming)
        XCTAssertEqual(candidate.profile.layerCount, 40)
        XCTAssertGreaterThan(candidate.pinnedLayerCount, 0)
    }

    /// The snap points are V4.1's own rungs: the floor, then each of the forty
    /// blocks in schedule order, then the output head. Forty-two positions, and
    /// the last one is the whole ladder.
    func testV41SnapPointsAreFortyBlocksThenTheHead() throws {
        let ladder = try XCTUnwrap(v41Profile.deepSeekLadder)
        let points = v41Profile.snapPoints

        XCTAssertEqual(points.count, 42)
        XCTAssertEqual(points.first, v41LadderFloor)
        XCTAssertEqual(points[40], v41EveryBlockBudget)
        XCTAssertEqual(points.last, v41FullResidencyBudget)
        XCTAssertEqual(
            points.last,
            DeepSeekV41MemoryDialInputs.budgetThatPinsEverything(
                census: ladder.census, floor: ladder.floor),
            "the dial's last rung and the kit's own answer must be one number")
        XCTAssertEqual(points, points.sorted(), "a ladder that is not monotone is not a ladder")
    }

    /// The dial's tiers are the kit planner's fields for V4.1 as they are for
    /// V4 — `BudgetPlan` reads a `PinPlan` rather than computing a composition
    /// of its own.
    func testTheV41TiersAreTheKitPlannersFields() throws {
        let ladder = try XCTUnwrap(v41Profile.deepSeekLadder)
        for budget: UInt64 in [
            7_000_000_000, 10_000_000_000, v41EveryBlockBudget, v41FullResidencyBudget,
            20_000_000_000,
        ] {
            let candidate = v41Plan(budget: budget)
            let reference = try DeepSeekV4MemoryDial.plan(
                budgetBytes: budget, census: ladder.census, floor: ladder.floor,
                maximumNewTokens: candidate.maximumNewTokens)

            XCTAssertEqual(candidate.pinnedBytes, reference.pinnedBytes, "pinned at \(budget)")
            XCTAssertEqual(
                candidate.pinnedLayerCount, reference.pinnedLayers.count, "blocks at \(budget)")
            XCTAssertEqual(
                candidate.floorBytes, reference.workingFloorBytes, "floor at \(budget)")
            XCTAssertEqual(
                candidate.freeBytes, reference.unusedBudgetBytes, "free at \(budget)")
            XCTAssertEqual(
                candidate.stagedBytes, 0, "V4.1 replaces one block's state at a time")
            XCTAssertEqual(candidate.hotSetBytes, 0)
            XCTAssertTrue(candidate.tiersAccountForBudget, "tiers ≠ budget at \(budget)")
        }
    }

    /// **Balanced on this owner's Mac holds all forty blocks and the head.**
    ///
    /// The arms this is chosen against: at the 3.4 GB floor a decode token
    /// reads 11.72 GB and takes 7.49 s; at 15 GB, with all forty blocks and the
    /// head pinned, it reads 4.51 GB and takes 3.40 s
    /// (`docs/experiments/2026-09-11-v41-phase3-runner.md` §3). Before this,
    /// Balanced *was* the floor.
    func testV41BalancedHoldsEveryBlockAndTheHeadOnTheOwnersMac() {
        let candidate = v41Plan(budget: DeepSeekV41ProductMemoryBudget.minimumBudgetBytes)

        XCTAssertEqual(candidate.budget(for: .floor), 3_400_000_000)
        XCTAssertEqual(candidate.budget(for: .balanced), v41FullResidencyBudget)
        XCTAssertEqual(candidate.budget(for: .generous), 34_400_000_000)
        XCTAssertLessThanOrEqual(
            candidate.budget(for: .balanced), 34_400_000_000 * 3 / 5,
            "full residency has to be inside the 60 % rule to be chosen by it")

        let balanced = candidate.with(budgetBytes: candidate.budget(for: .balanced))
        XCTAssertEqual(balanced.pinnedLayerCount, 40)
        XCTAssertTrue(balanced.pinsOutputHead)
        XCTAssertEqual(balanced.residentUnitsLabel, "40 of 40 + output head")
        XCTAssertEqual(
            balanced.pinnedBytes, v41EveryBlockResidentBytes + v41OutputHeadBytes)
        XCTAssertEqual(
            balanced.pinnedBytes, 8_729_882_048,
            "the resident weights the 15 GB arm actually held")
        // One embedding row is what a token still reads at the top of the
        // ladder. The table it comes from is the same 1.32 GB as the head and
        // is never a rung.
        XCTAssertEqual(balanced.bytesReadPerToken, 10_240)
        XCTAssertEqual(balanced.freeBytes, 0, "the last snap point spends the whole ladder")
        XCTAssertEqual(
            balanced.bytesSavedPerToken, v41EveryBlockReadBytes + v41OutputHeadBytes)
        XCTAssertEqual(
            candidate.explanation(for: .balanced),
            "All 40 layers + output head resident — about 8.53 GB fewer bytes per token.")
    }

    /// The floor is the product floor, it pins nothing, and it reads the whole
    /// deterministic census every token — which is what makes Balanced worth
    /// stating.
    func testV41FloorIsTheProductFloorAndPinsNothing() {
        let candidate = v41Plan(budget: 3_400_000_000)

        XCTAssertEqual(candidate.budget(for: .floor), 3_400_000_000)
        XCTAssertTrue(candidate.isRunnable)
        XCTAssertNil(candidate.pinPlan, "3.4 GB is below the dial floor and pins nothing")
        XCTAssertEqual(candidate.pinnedLayerCount, 0)
        XCTAssertEqual(candidate.residentUnitsLabel, "None")
        XCTAssertEqual(
            candidate.bytesReadPerToken, v41EveryBlockReadBytes + v41OutputHeadBytes + 10_240)
        XCTAssertEqual(
            candidate.explanation(for: .floor),
            "The minimum envelope this model is admitted at. Nothing is held resident: every "
                + "layer streams, every token.")
        XCTAssertTrue(candidate.tiersAccountForBudget)
    }

    /// A budget below the product floor is refused in the dial rather than
    /// inside the run. The entry used to state no minimum at all.
    func testV41BelowTheProductFloorIsRefusedByTheDial() throws {
        let refused = v41Plan(budget: 3_399_999_999)

        XCTAssertFalse(refused.isRunnable)
        let refusal = try XCTUnwrap(refused.refusal)
        XCTAssertEqual(refusal.suggestedBudgetBytes, 3_400_000_000)
        XCTAssertEqual(refused.floorBytes, 0)
        XCTAssertTrue(refused.tiersAccountForBudget)
    }

    /// Every block is held before the head is, so a budget that can fund the
    /// blocks but not the head funds the blocks — and the panel says what the
    /// next 1.32 GB buys.
    func testV41HoldsEveryBlockBeforeItHoldsTheHead() {
        let candidate = v41Plan(budget: v41EveryBlockBudget)

        XCTAssertEqual(candidate.pinnedLayerCount, 40)
        XCTAssertFalse(candidate.pinsOutputHead)
        XCTAssertEqual(candidate.residentUnitsLabel, "40 of 40")
        XCTAssertEqual(candidate.bytesReadPerToken, v41OutputHeadBytes + 10_240)
        XCTAssertEqual(candidate.nextPinSentence, "+1.32 GB pins the output head.")
    }

    /// Dragging a V4.1 budget up buys resident blocks, not empty space.
    func testRaisingAV41BudgetPinsStrictlyMoreBlocks() {
        var previous: UInt64 = 0
        var sawAnIncrease = false
        for budget in stride(
            from: UInt64(3_400_000_000), through: 14_000_000_000, by: 199_000_000)
        {
            let candidate = v41Plan(budget: budget)
            XCTAssertGreaterThanOrEqual(
                candidate.pinnedBytes, previous, "the pinned tier shrank at \(budget)")
            if candidate.pinnedBytes > previous { sawAnIncrease = true }
            previous = candidate.pinnedBytes
            XCTAssertTrue(candidate.tiersAccountForBudget, "tiers ≠ budget at \(budget)")
        }
        XCTAssertTrue(sawAnIncrease)
    }

    /// **The phone.** A 4.50 GB process ceiling runs V4.1 at its floor and
    /// cannot pin a single block: the first rung is the 6.13 GB dial floor plus
    /// block 2, which is more than the whole device offers. Balanced is shown
    /// over the ceiling with its deficit rather than quietly becoming the floor,
    /// and the chat starts at Floor.
    // Main-actor: the iOS half asks `AppModel.defaultPreset(for:)`, which is
    // main-actor isolated, and a nonisolated test cannot call it on that platform.
    @MainActor
    func testV41OnAPhoneRunsAtTheFloorAndIsToldWhatBalancedWouldCost() throws {
        let candidate = v41Plan(budget: 3_400_000_000, ceiling: 4_500_000_000)
        let balanced = candidate.budget(for: .balanced)

        XCTAssertEqual(candidate.budget(for: .floor), 3_400_000_000)
        XCTAssertTrue(candidate.isRunnable, "the product floor still runs on a phone")
        XCTAssertEqual(candidate.pinnedLayerCount, 0)
        XCTAssertFalse(candidate.deviceCannotHostModel)
        // Not block 0: inside the block rung the ladder ranks by bytes saved per
        // resident byte, and block 2's KV-source geometry scores 0.9747 against
        // a plain block's 0.9730.
        XCTAssertEqual(balanced, v41LadderFloor + 191_312_088, "block 2's residency")
        XCTAssertLessThan(balanced, v41FullResidencyBudget)
        XCTAssertEqual(
            try XCTUnwrap(candidate.presetDeficit(.balanced)), balanced - 4_500_000_000)
        #if os(iOS)
            XCTAssertEqual(AppModel.defaultPreset(for: .deepseekV41Flash), .floor)
        #endif
    }

    /// **Generous on a phone is the first rung, not the ceiling.**
    ///
    /// It used to be the ceiling: a budget that runs, pins nothing, and whose
    /// own explanation called it "the first pin boundary this ladder has". Both
    /// halves were wrong. Every byte above the floor buys nothing on a device
    /// that cannot reach the first rung — the dial floor of a run that *pins* is
    /// 6.13 GB before a single block is held — so a preset that spent the whole
    /// device on it was the dial offering a residency it has no rung for, and
    /// on the phone that is the direction that gets an app killed.
    ///
    /// So it reports the rung, which the dial draws disabled with its deficit —
    /// the same thing Balanced does, because on such a device they are the same
    /// sentence. On any device that *can* reach the first rung, Generous is the
    /// ceiling exactly as before.
    func testV41GenerousOnAPhoneIsTheFirstRungItCannotReach() {
        let candidate = v41Plan(budget: 3_400_000_000, ceiling: 4_500_000_000)
        let generous = candidate.budget(for: .generous)

        XCTAssertEqual(generous, v41LadderFloor + 191_312_088)
        XCTAssertEqual(generous, candidate.budget(for: .balanced))
        XCTAssertEqual(
            try? XCTUnwrap(candidate.presetDeficit(.generous)), generous - 4_500_000_000)

        // A Mac reaches the first rung, so its Generous is still the ceiling.
        let mac = v41Plan(budget: 3_400_000_000, ceiling: 34_400_000_000)
        XCTAssertEqual(mac.budget(for: .generous), 34_400_000_000)
        XCTAssertNil(mac.presetDeficit(.generous))
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

    /// Mac and phone start a streamed DeepSeek chat in different places, and the
    /// reason is evidence rather than taste: the Mac's ladder reaches full
    /// residency well inside the device, and no pinned arm of either model has
    /// ever been run on a phone.
    @MainActor
    func testTheDefaultPresetIsPlatformConditionalForTheStreamedModels() {
        #if os(macOS)
            XCTAssertEqual(AppModel.defaultPreset(for: .deepseekV4Flash), .balanced)
            XCTAssertEqual(AppModel.defaultPreset(for: .deepseekV41Flash), .balanced)
        #else
            XCTAssertEqual(AppModel.defaultPreset(for: .deepseekV4Flash), .floor)
            XCTAssertEqual(AppModel.defaultPreset(for: .deepseekV41Flash), .floor)
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
