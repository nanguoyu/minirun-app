import Foundation
import MinirunKit
import ModelAdapters
import XCTest

@testable import MinirunRunners

/// The two V4.1 platform boundaries, checked from the Mac.
///
/// K3's reason, kept: a boundary is a contract, the policies are values, and a
/// macOS suite can hold the iPhone one to its own arithmetic without pretending
/// it ran on an iPhone.
final class DeepSeekV41PlatformPolicyTests: XCTestCase {

    // MARK: - What the published container's geometry costs

    /// One expert's half of a pair tile, from
    /// `docs/experiments/2026-09-10-v41-flash-container.md`: `w1` is
    /// `[2304, 5120]` FP4, so 2304 x 5120 x 17/32.
    private static let halfTileBytes: UInt64 = 6_266_880
    private static let expertsPerToken = 6
    private static let projections = 3

    func testTheMacPolicyIsThePhaseThreeArmsToTheByte() {
        let policy = DeepSeekV41ProductMemoryBudget.macOSProductPolicy
        XCTAssertEqual(policy.transientExecutionBytes, 3_200_000_000)
        XCTAssertEqual(policy.minimumBudgetBytes, 3_400_000_000)
        XCTAssertEqual(policy.maximumPromptTokens, 512)
        XCTAssertEqual(policy.maximumNewTokens, 64)
        XCTAssertEqual(policy.expertPoolSlots, 20)
        XCTAssertEqual(policy.expertReadAhead, 6)
        XCTAssertEqual(policy.headWindowRows, 4_096)
        XCTAssertFalse(policy.isExperimental)
        // Bounded execution, unpriced term: the envelope was measured on an arm
        // that held the whole prompt's operands, so pricing them again would
        // count them twice and move a floor four recorded arms rest on.
        XCTAssertTrue(policy.boundsLiveOperands)
        XCTAssertNil(policy.liveGatherOperandTokens)
    }

    /// The recorded floor arm's plan, reproduced exactly.
    ///
    /// `docs/experiments/data/2026-09-11-v41-phase3-runner/floor-default-deepseek-v41-product-completion-product.json`
    /// published `transientExecutionBytes` 3,200,000,000, `retainedStateBytes`
    /// 5,427,456, `replacementBlockStateBytes` 195,072, `expertPoolBytes`
    /// 125,337,600 and `productFloorPaddingBytes` 69,039,872 inside a declared
    /// 3,400,000,000. This is the arithmetic that produces those five numbers,
    /// stated here so that a future policy edit that moves the Mac floor fails
    /// a test rather than a merge.
    func testTheMacPlansArithmeticStillPartitionsTheRecordedFloor() {
        let policy = DeepSeekV41ProductMemoryBudget.macOSProductPolicy
        let state: UInt64 = 5_427_456
        let widestBlock: UInt64 = 195_072
        let pool: UInt64 = 125_337_600
        let exact = policy.transientExecutionBytes + state + widestBlock + pool
        XCTAssertEqual(exact, 3_330_960_128)
        XCTAssertEqual(policy.minimumBudgetBytes - exact, 69_039_872)
    }

    func testTheIPhonePolicyStatesASmallerFloorAndSaysItIsExperimental() {
        let policy = DeepSeekV41ProductMemoryBudget.iOSProductPolicy
        XCTAssertEqual(policy.transientExecutionBytes, 1_600_000_000)
        XCTAssertEqual(policy.minimumBudgetBytes, 1_900_000_000)
        XCTAssertTrue(policy.isExperimental)
        XCTAssertTrue(policy.boundsLiveOperands)
        XCTAssertEqual(policy.liveGatherOperandTokens, 1)
        XCTAssertEqual(policy.headWindowRows, 1_024)
        XCTAssertLessThan(
            policy.minimumBudgetBytes,
            DeepSeekV41ProductMemoryBudget.macOSProductPolicy.minimumBudgetBytes,
            "the iPhone floor exists because the Mac's did not fit on a phone")
    }

    /// The iPhone floor's arithmetic, term by term.
    ///
    /// The envelope is stated; everything else is derived, and the floor is
    /// their sum rounded up. Written out so the rounding is visible as rounding.
    func testTheIPhoneFloorIsItsTermsRoundedUp() {
        let policy = DeepSeekV41ProductMemoryBudget.iOSProductPolicy
        let gather =
            UInt64(Self.projections * Self.expertsPerToken) * Self.halfTileBytes
        XCTAssertEqual(gather, 112_803_840)
        let stateAtProductCeilings: UInt64 = 7_109_376
        let widestBlockReplacement: UInt64 = 867_072
        let pool = UInt64(policy.expertPoolSlots) * Self.halfTileBytes
        XCTAssertEqual(pool, 125_337_600)
        let exact =
            policy.transientExecutionBytes + gather + stateAtProductCeilings
            + widestBlockReplacement + pool
        XCTAssertEqual(exact, 1_846_117_888)
        XCTAssertGreaterThanOrEqual(policy.minimumBudgetBytes, exact)
        XCTAssertEqual(policy.minimumBudgetBytes - exact, 53_882_112)
    }

    /// The iPhone kill, arithmetically.
    ///
    /// The device was running at the Mac floor. What the run actually held at
    /// its widest was the eleven-token gather term plus the unbounded head
    /// walk, and a longer chat prompt scales the first of the two linearly —
    /// which is what puts a 3.4 GB budget past the ~5 GB one app gets.
    func testTheUnboundedGatherTermExplainsAPhoneSizedOvershoot() {
        let perToken = UInt64(Self.projections * Self.expertsPerToken) * Self.halfTileBytes
        XCTAssertEqual(perToken * 11, 1_240_842_240)
        // Twenty prompt tokens — a chat template plus a short turn — is already
        // a gigabyte more than the arm the floor was measured on.
        XCTAssertEqual(perToken * 20 - perToken * 11, 1_015_234_560)
        // And the product prompt limit the same floor claims to cover is not a
        // budget any device has.
        XCTAssertEqual(perToken * 512, 57_755_566_080)
    }

    /// Every policy this product ships must be self-consistent: a priced live
    /// set implies a bounded one, and a pool must hold the read-ahead window.
    func testBothPoliciesAreInternallyConsistent() {
        for policy in [
            DeepSeekV41ProductMemoryBudget.macOSProductPolicy,
            DeepSeekV41ProductMemoryBudget.iOSProductPolicy,
        ] {
            XCTAssertGreaterThanOrEqual(
                policy.expertPoolSlots, 3 * policy.expertReadAhead + 1, policy.name)
            if policy.liveGatherOperandTokens != nil {
                XCTAssertTrue(policy.boundsLiveOperands, policy.name)
            }
            XCTAssertGreaterThan(
                policy.minimumBudgetBytes, policy.transientExecutionBytes, policy.name)
            XCTAssertGreaterThan(policy.budgetOvershootAllowanceBytes, 0, policy.name)
        }
    }

    /// The Mac build gets the Mac policy. Spelled out because the whole point of
    /// the value is that the selection is a compile-time fact and not a device
    /// guess.
    func testTheCurrentPolicyFollowsThePlatform() {
        #if os(iOS)
            XCTAssertEqual(
                DeepSeekV41ProductMemoryBudget.currentPolicy,
                DeepSeekV41ProductMemoryBudget.iOSProductPolicy)
        #else
            XCTAssertEqual(
                DeepSeekV41ProductMemoryBudget.currentPolicy,
                DeepSeekV41ProductMemoryBudget.macOSProductPolicy)
        #endif
        XCTAssertEqual(
            DeepSeekV41ProductMemoryBudget.minimumBudgetBytes,
            DeepSeekV41ProductMemoryBudget.currentPolicy.minimumBudgetBytes)
        XCTAssertEqual(
            DeepSeekV41ProductMemoryBudget.maximumNewTokens,
            DeepSeekV41ProductMemoryBudget.currentPolicy.maximumNewTokens)
    }

    /// The knob defaults follow the policy, which is how a phone gets a smaller
    /// head window without the app stating one.
    func testTheKnobDefaultsFollowThePolicy() throws {
        let policy = DeepSeekV41ProductMemoryBudget.currentPolicy
        let knobs = try DeepSeekV41EffectiveKnobs.resolve(
            RunKnobs(), scale: .product, declaredBudgetBytes: policy.minimumBudgetBytes)
        XCTAssertEqual(knobs.expertReadAhead, policy.expertReadAhead)
        XCTAssertEqual(knobs.expertPoolSlots, policy.expertPoolSlots)
        XCTAssertEqual(knobs.logitChunkRows, policy.headWindowRows)
        XCTAssertEqual(knobs.boundsLiveOperands, policy.boundsLiveOperands)
        XCTAssertEqual(knobs.mlxCacheLimitBytes, 0)
        XCTAssertEqual(knobs.asStrings["boundsLiveOperands"], "\(policy.boundsLiveOperands)")
    }

    // MARK: - The dial's terms are the policy's

    /// **The pinned envelope is a policy term, and the floor reads it from
    /// there.**
    ///
    /// It was a constant on `DeepSeekV41MemoryDialInputs`, which is how a dial
    /// comes to price one platform's ladder with another's numbers. The Mac's
    /// value is unchanged to the byte — phase 3's stated arm measured
    /// 5,161,820,992 B of transient with all forty blocks and the head resident
    /// — and the iPhone states the same one, deliberately: no phone has ever
    /// pinned anything, so the Mac's measurement is the only number the ladder
    /// may be priced with, and scaling it down would open rungs on a device
    /// that was killed by Jetsam at a smaller budget than the first one.
    func testThePinnedTransientEnvelopeIsAPolicyTermOnBothPlatforms() {
        for policy in [
            DeepSeekV41ProductMemoryBudget.macOSProductPolicy,
            DeepSeekV41ProductMemoryBudget.iOSProductPolicy,
        ] {
            XCTAssertEqual(
                policy.pinnedTransientExecutionBytes, 5_200_000_000,
                "\(policy.name) prices a pinning run with the only arm that has ever "
                    + "pinned")
            XCTAssertGreaterThanOrEqual(
                policy.pinnedTransientExecutionBytes, policy.transientExecutionBytes)
        }
        XCTAssertEqual(
            DeepSeekV41MemoryDialInputs.pinnedTransientExecutionBytes,
            DeepSeekV41ProductMemoryBudget.currentPolicy.pinnedTransientExecutionBytes)
    }

    /// The ladder floor is built from the named policy, so the macOS suite can
    /// price the phone's ladder and see that it is out of any phone's reach.
    ///
    /// 6,130,643,968 B before a single block is held: 5.2 GB of pinned
    /// transient, the 20-slot pool, a saturated 512 MiB MLX cache and the
    /// 256 MiB pin margin. `os_proc_available_memory()` offers an iPhone 16 Pro
    /// about 5 GB, which is why the phone's Balanced and Generous are drawn
    /// disabled with a deficit rather than offered.
    func testTheLadderFloorIsPricedByTheNamedPolicy() throws {
        let iOS = DeepSeekV41ProductMemoryBudget.iOSProductPolicy
        let plan = DeepSeekV41ProductMemoryPlan(
            declaredBudgetBytes: .max,
            transientExecutionBytes: iOS.transientExecutionBytes,
            retainedStateBytes: 0, replacementBlockStateBytes: 0,
            expertPoolBytes: UInt64(iOS.expertPoolSlots) * Self.halfTileBytes,
            mlxCacheBytes: 536_870_912,
            pinnedDeterministicBytes: 0, liveGatherOperandBytes: 0,
            productFloorPaddingBytes: 0, requiredBudgetBytes: 0, headroomBytes: 0)
        let floor = try XCTUnwrap(
            DeepSeekV41MemoryDialInputs.floor(pricing: plan, policy: iOS))
        XCTAssertEqual(floor.widestResidentLayerBytes, 5_200_000_000)
        XCTAssertEqual(floor.totalBytes, 6_130_643_968)
        XCTAssertGreaterThan(
            floor.totalBytes, iOS.minimumBudgetBytes,
            "the floor of a run that pins is far above the floor of one that streams")
    }

    /// The lower bound the app's scale decision consults. Below it nothing can
    /// be pinned at any census, so a stated budget below it would turn the
    /// product policy off and buy nothing with it.
    func testTheSmallestPinningBudgetIsALowerBoundOnTheLadderFloor() throws {
        let iOS = DeepSeekV41ProductMemoryBudget.iOSProductPolicy
        let bound = DeepSeekV41MemoryDialInputs.smallestPinningBudgetBytes(policy: iOS)
        XCTAssertEqual(bound, 5_200_000_000 + 536_870_912 + 268_435_456)
        XCTAssertLessThan(
            bound, 6_130_643_968,
            "it omits the pool and the bounded state, which is what makes it a bound")
        XCTAssertGreaterThan(bound, iOS.minimumBudgetBytes)
    }

    /// A run record published before the gather term existed still decodes.
    func testAnOlderPlanRecordDecodesWithTheGatherTermAtZero() throws {
        let json = Data(
            """
            {"declaredBudgetBytes":3400000000,"expertPoolBytes":125337600,
             "headroomBytes":0,"mlxCacheBytes":0,"pinnedDeterministicBytes":0,
             "productFloorPaddingBytes":69039872,"replacementBlockStateBytes":195072,
             "requiredBudgetBytes":3400000000,"retainedStateBytes":5427456,
             "transientExecutionBytes":3200000000}
            """.utf8)
        let plan = try JSONDecoder().decode(DeepSeekV41ProductMemoryPlan.self, from: json)
        XCTAssertEqual(plan.liveGatherOperandBytes, 0)
        XCTAssertTrue(plan.isAdmitted)
        XCTAssertTrue(plan.accountsForDeclaredBudget)
    }
}
