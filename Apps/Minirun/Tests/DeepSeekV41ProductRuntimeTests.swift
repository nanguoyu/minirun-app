import Foundation
import MinirunKit
import MinirunRunners
import XCTest

@testable import MinirunApp

/// What the app says about DeepSeek V4.1, and what it will do about it.
///
/// The tokenizer and configuration halves need a synthetic 517 GB-shaped
/// artifact and are exercised by `DeepSeekV41ProductGateTests` against the real
/// one. What is checked here is everything the app decides *before* an artifact
/// exists: the catalog entry, the fitness verdict that entry produces, the
/// registration, and the scale the memory dial hands the runner.
@MainActor
final class DeepSeekV41ProductRuntimeTests: XCTestCase {

    /// **The catalog row names a runner; the policy names the floor.**
    ///
    /// This used to assert that the bundled row's `minimumBudgetBytes` *was*
    /// the runner's floor, and on the iPhone simulator it failed with the whole
    /// defect in one line: `Optional(3400000000)` against `Optional(1900000000)`.
    ///
    /// A published catalog row carries **one** `minimumBudgetBytes`, and this
    /// model's two platform policies do not share one — 3.4 GB on the Mac, and
    /// 1.9 GB on the phone that keeps being killed at the Mac's number. A
    /// single JSON field cannot express both, so it is not allowed to be the
    /// floor: the runtime capability is, and every floor the app actually
    /// consults comes from ``DeepSeekV41ProductMemoryBudget/currentPolicy``
    /// — `AppModel.effectiveProductFitness` reads
    /// `runtime.capabilities.minimumBudgetBytes`, and the dial reads the
    /// profile's `refusalThresholdBytes`. K3's row already worked this way: it
    /// states 5.80 GB while the supported Mac boundary is 8.00 GB.
    ///
    /// What the row still has to be is *consistent with the publication*, so it
    /// is checked against the macOS policy by name rather than against
    /// whichever platform is hosting the test.
    func testTheCatalogEntryNamesARunnerAndItsFloor() throws {
        let descriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(.deepseekV41Flash))
        XCTAssertEqual(descriptor.runner, .decodeRunner)
        XCTAssertEqual(descriptor.layout, .v41FlashUnitBundle)
        XCTAssertEqual(
            descriptor.minimumBudgetBytes,
            DeepSeekV41ProductMemoryBudget.macOSProductPolicy.minimumBudgetBytes,
            "the published row states the Mac's floor; it cannot state two")

        // The authority, on whichever platform this is hosted.
        let policy = DeepSeekV41ProductMemoryBudget.currentPolicy
        XCTAssertEqual(DeepSeekV41ProductMemoryBudget.minimumBudgetBytes, policy.minimumBudgetBytes)
        XCTAssertEqual(
            DeepSeekV41DecodeRunner(scale: .product).capabilities.minimumBudgetBytes,
            policy.minimumBudgetBytes,
            "the runner's own capability is what the fitness verdict quotes")
        XCTAssertEqual(
            CatalogFixtures.deepseekV41Flash.memory.refusalThresholdBytes,
            policy.minimumBudgetBytes,
            "and the dial refuses below the same number")
        #if os(iOS)
            XCTAssertEqual(policy.minimumBudgetBytes, 1_900_000_000)
            XCTAssertNotEqual(descriptor.minimumBudgetBytes, policy.minimumBudgetBytes)
        #else
            XCTAssertEqual(policy.minimumBudgetBytes, 3_400_000_000)
        #endif
    }

    func testTheEntryIsRunnableOnAMacWithRoomAndRefusedOnOneWithout() throws {
        let descriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(.deepseekV41Flash))
        let roomy = descriptor.fitness(
            on: DeviceProfile(
                platform: .macOS, processMemoryBudgetBytes: 32_000_000_000,
                hasMLXGPU: true, hasIncreasedMemoryLimitEntitlement: false))
        XCTAssertNotEqual(
            roomy.verdict, PlatformFitness.Verdict.noRunner,
            "the catalog now names a runner, so no verdict may say there is none")
        XCTAssertEqual(roomy.requiredFreeBytes, descriptor.totalBytes)

        // A phone whose whole process budget is under the product floor is
        // refused by name, and the reason is the number rather than a category.
        let phone = descriptor.fitness(
            on: DeviceProfile(
                platform: .iOS, processMemoryBudgetBytes: 2_000_000_000,
                hasMLXGPU: true, hasIncreasedMemoryLimitEntitlement: true))
        XCTAssertNotEqual(phone.verdict, PlatformFitness.Verdict.runnable)
    }

    /// The dial states memory and must reach the *runner*, not only the
    /// request — the decision V4's provider records in as many words.
    func testTheRunnerScaleFollowsTheStatedBudget() {
        let floor = DeepSeekV41ProductMemoryBudget.minimumBudgetBytes
        XCTAssertEqual(
            DeepSeekV41ProductRuntimeProvider.runnerScale(statedBudgetBytes: floor),
            .product)
        // Below the floor stays `.product`: the dial already refuses that
        // region, and passing a sub-floor number in as a stated minimum would
        // be the runner being talked into admitting a budget by being handed it.
        XCTAssertEqual(
            DeepSeekV41ProductRuntimeProvider.runnerScale(statedBudgetBytes: floor - 1),
            .product)
        XCTAssertEqual(
            DeepSeekV41ProductRuntimeProvider.runnerScale(
                statedBudgetBytes: 12_300_000_000),
            .stated(
                minimumBudgetBytes: 12_300_000_000,
                maximumNewTokens: DeepSeekV41ProductMemoryBudget.maximumNewTokens))
    }

    /// Registration is a runtime this build can actually execute, and its
    /// capabilities are the product-scale ones — "the least memory this model
    /// needs" and "the most tokens it will produce" are facts about the model
    /// and not about whichever conversation is open.
    func testTheProductRegistryOffersV41WithItsProductCapabilities() throws {
        #if arch(arm64)
            let registry = ModelRuntimeRegistry.product
            let runtime = try XCTUnwrap(registry.runtime(for: .deepseekV41Flash))
            XCTAssertTrue(runtime.trust.isVerified)
            XCTAssertEqual(runtime.capabilities.model, .deepseekV41Flash)
            XCTAssertEqual(
                runtime.capabilities.minimumBudgetBytes,
                DeepSeekV41ProductMemoryBudget.minimumBudgetBytes)
            XCTAssertEqual(
                runtime.capabilities.maximumNewTokens,
                DeepSeekV41ProductMemoryBudget.maximumNewTokens)
            XCTAssertFalse(
                runtime.capabilities.acceptsTextPrompts,
                "the runner takes ids from its own verified tokenizer")
            XCTAssertTrue(runtime.capabilities.requiresMLX)
        #else
            throw XCTSkip("the V4.1 runtime is registered only where MLX runs")
        #endif
    }

    /// The knobs the V4.1 pool consumes, and the two V4 ones it deliberately
    /// does not: a V4.1 prefetch names all three projections at once, so
    /// projection prefetch is not a switch, and the pool is already run-scoped,
    /// so a cross-block window has no per-block backend to be scoped against.
    ///
    /// `expertTileAdoption` is on the consumed side: the retained adoption mode
    /// reaches the engine as `adoptsExpertOperands`, so the runner advertises
    /// the knob it actually reads.
    func testTheAdvertisedKnobsAreTheOnesTheV41PoolConsumes() {
        let knobs = DeepSeekV41DecodeRunner.knobs
        XCTAssertTrue(knobs.contains("expertPoolSlots"))
        XCTAssertTrue(knobs.contains("expertReadAhead"))
        XCTAssertTrue(knobs.contains("queueDepth"))
        XCTAssertTrue(knobs.contains("expertTileAdoption"))
        XCTAssertFalse(knobs.contains("expertProjectionPrefetch"))
        XCTAssertFalse(knobs.contains("expertCrossLayerPrefetch"))
    }

    // MARK: The memory dial

    /// The catalog entry carries the ladder, so the dial has rungs to offer
    /// before any copy has been verified on this machine.
    ///
    /// It carried none until the dial reached V4.1, and "no ladder" is not a
    /// neutral state: a plan with no ladder and the streaming strategy has one
    /// rung, so Floor, Balanced and Generous were all the same 3.4 GB budget.
    func testTheCatalogEntryCarriesTheLadderAndTheProductFloor() throws {
        let memory = CatalogFixtures.deepseekV41Flash.memory
        let ladder = try XCTUnwrap(memory.deepSeekLadder)
        let policy = DeepSeekV41ProductMemoryBudget.currentPolicy

        XCTAssertEqual(ladder.layerCount, 40)
        XCTAssertEqual(memory.layerCount, 40)
        // The floor is this platform's policy, which is the whole point of the
        // entry being a function of one: the Mac's 3.4 GB and the phone's
        // 1.9 GB are the same expression evaluated on two platforms.
        XCTAssertEqual(memory.requiredMinimumBudgetBytes, policy.minimumBudgetBytes)
        XCTAssertEqual(memory.refusalThresholdBytes, policy.minimumBudgetBytes)
        // Only an arm that ran may claim a record. Phase 3's floor arm ran on
        // the Mac at exactly the macOS floor; no V4.1 arm has ever run on a
        // phone, so the experimental policy claims none and the dial refuses at
        // the product boundary without calling it a measurement.
        XCTAssertEqual(
            memory.onRecordMinimumBudgetBytes,
            policy.isExperimental ? nil : policy.minimumBudgetBytes,
            "the floor arm of phase 3 ran at exactly this budget")
        #if os(iOS)
            XCTAssertEqual(memory.refusalThresholdBytes, 1_900_000_000)
            XCTAssertNil(memory.onRecordMinimumBudgetBytes)
        #else
            XCTAssertEqual(memory.refusalThresholdBytes, 3_400_000_000)
        #endif
        // The ladder itself is not a platform term. Both policies state the
        // same 20-slot pool and the same 5.2 GB pinned envelope — the Mac's,
        // because it is the only measurement of a run that pins — so the rungs
        // are identical and only the floor beneath them moves.
        XCTAssertEqual(
            ladder.census.globals.headResidentBytes, 1_323_827_200,
            "the [129280, 5120] BF16 output head")
        XCTAssertEqual(
            memory.snapPoints,
            CatalogFixtures.deepSeekV41Memory(
                policy: DeepSeekV41ProductMemoryBudget.macOSProductPolicy
            ).snapPoints)
    }

    /// **The dial's Balanced reaches the runner as a stated budget.**
    ///
    /// The runner has honoured a stated budget since phase 3; what was missing
    /// was a dial that ever stated anything other than the floor. These are the
    /// two ends of the same wire: the plan's Balanced, and the scale the
    /// provider builds a runner at when a conversation carries it.
    func testTheDialsBalancedBudgetIsTheScaleTheRunnerIsBuiltAt() {
        let entry = CatalogFixtures.deepseekV41Flash
        let plan = BudgetPlan(
            model: .deepseekV41Flash, modelName: entry.descriptor.displayName,
            profile: entry.memory,
            budgetBytes: DeepSeekV41ProductMemoryBudget.minimumBudgetBytes,
            maximumNewTokens: DeepSeekV41ProductMemoryBudget.maximumNewTokens,
            deviceCeilingBytes: 34_400_000_000, readAheadDepth: 1)
        let balanced = plan.budget(for: .balanced)

        XCTAssertEqual(balanced, 14_860_526_016)
        XCTAssertEqual(plan.with(budgetBytes: balanced).pinnedLayerCount, 40)
        XCTAssertEqual(
            DeepSeekV41ProductRuntimeProvider.runnerScale(statedBudgetBytes: balanced),
            .stated(
                minimumBudgetBytes: balanced,
                maximumNewTokens: DeepSeekV41ProductMemoryBudget.maximumNewTokens),
            "Balanced must reach the runner, not only the request")
        // The floor still resolves to the fully gated product path, unchanged.
        XCTAssertEqual(
            DeepSeekV41ProductRuntimeProvider.runnerScale(
                statedBudgetBytes: plan.budget(for: .floor)),
            .product)
    }

    // MARK: The iPhone dial

    private var iOS: DeepSeekV41ProductMemoryBudget.Policy {
        DeepSeekV41ProductMemoryBudget.iOSProductPolicy
    }

    /// An iPhone-priced plan, at a ceiling of the shape a phone reports.
    ///
    /// 5.0 GB is about what `os_proc_available_memory()` gave the owner's
    /// iPhone 16 Pro on the launch that was killed; the exact number does not
    /// matter to these assertions, only that it is under the ladder's first
    /// rung, which every phone's is.
    private func iPhonePlan(budget: UInt64, ceiling: UInt64 = 5_000_000_000) -> BudgetPlan {
        BudgetPlan(
            model: .deepseekV41Flash, modelName: "DeepSeek V4.1 Flash",
            profile: CatalogFixtures.deepSeekV41Memory(policy: iOS),
            budgetBytes: budget,
            maximumNewTokens: iOS.maximumNewTokens,
            deviceCeilingBytes: ceiling, readAheadDepth: 1)
    }

    /// **The dial's floor on a phone is the phone's floor.**
    ///
    /// It was the Mac's. A chat pulled off the owner's iPhone after the second
    /// Jetsam kill said `memoryBudgetBytes = 3400000000` — the macOS policy's
    /// number — on a build whose own runner floor was 1.9 GB, because every
    /// floor term the dial priced was a constant rather than the platform
    /// policy's. This is that, at the byte.
    func testTheIPhoneDialFloorIsTheIPhoneFloor() {
        let plan = iPhonePlan(budget: iOS.minimumBudgetBytes)
        XCTAssertEqual(iOS.minimumBudgetBytes, 1_900_000_000)
        XCTAssertEqual(plan.budget(for: .floor), 1_900_000_000)
        XCTAssertEqual(plan.profile.refusalThresholdBytes, 1_900_000_000)
        XCTAssertNil(plan.refusal)
        XCTAssertTrue(plan.isRunnable)
        XCTAssertTrue(plan.tiersAccountForBudget)
        // Nothing is pinned at the floor — that is what the floor is, on both
        // platforms — so the ladder's rungs are all still ahead of it.
        XCTAssertEqual(plan.pinnedLayerCount, 0)
        XCTAssertEqual(plan.pinnedBytes, 0)

        // And the Mac's floor is untouched by the same expression.
        let mac = BudgetPlan(
            model: .deepseekV41Flash, modelName: "DeepSeek V4.1 Flash",
            profile: CatalogFixtures.deepSeekV41Memory(
                policy: DeepSeekV41ProductMemoryBudget.macOSProductPolicy),
            budgetBytes: 3_400_000_000, maximumNewTokens: 64,
            deviceCeilingBytes: 34_400_000_000, readAheadDepth: 1)
        XCTAssertEqual(mac.budget(for: .floor), 3_400_000_000)
    }

    /// Balanced and Generous are priced with the phone's terms, and both are
    /// out of reach — shown disabled with a deficit rather than collapsed onto
    /// the floor or, worse, quietly offered.
    ///
    /// The ladder's first rung is 6.13 GB of floor before a single block is
    /// held, because a run that **pins** holds 5.2 GB of transient and no phone
    /// arm has ever measured less. A phone offers about 5 GB.
    func testTheIPhoneDialOffersNoRungItCannotHold() {
        let plan = iPhonePlan(budget: iOS.minimumBudgetBytes)
        let points = plan.profile.snapPoints
        XCTAssertEqual(points.first, 6_130_643_968, "the pinned ladder's own floor")
        XCTAssertEqual(points.dropFirst().first, 6_321_956_056, "floor + the first block")

        for preset in [BudgetPlan.Preset.balanced, .generous] {
            let budget = plan.budget(for: preset)
            XCTAssertEqual(
                budget, 6_321_956_056,
                "\(preset.rawValue) must report the first rung it cannot reach")
            XCTAssertEqual(plan.presetDeficit(preset), 6_321_956_056 - 5_000_000_000)
            XCTAssertEqual(
                plan.explanation(for: preset),
                "The first pin boundary this ladder has. This device cannot reach it.")
        }
        XCTAssertNil(plan.presetDeficit(.floor))
    }

    /// **A stated budget a phone cannot hold does not reach the runner.**
    ///
    /// `.stated` turns the product memory policy off: the engine then prices
    /// its requirement as the pool plus the MLX cache, stops enforcing the
    /// 512/64 limits, and lets the sentry's ceiling become the stated number.
    /// The killed run took exactly that path at 3.4 GB. Where the platform
    /// states a per-process limit, a stated budget now has to buy a rung *and*
    /// be a promise the device can keep.
    func testAStatedBudgetIsRefusedTheRunnerWhenTheDeviceCannotHoldAPinPlan() {
        let pinFloor = DeepSeekV41MemoryDialInputs.smallestPinningBudgetBytes(policy: iOS)
        XCTAssertEqual(pinFloor, 5_200_000_000 + 536_870_912 + 268_435_456)

        // The budget the phone was killed at: above the iOS floor, so the old
        // rule promoted it, and nowhere near a rung.
        XCTAssertEqual(
            DeepSeekV41ProductRuntimeProvider.runnerScale(
                statedBudgetBytes: 3_400_000_000, policy: iOS,
                availableMemoryBytes: 5_000_000_000),
            .product)
        // Even at the whole of what the device offers.
        XCTAssertEqual(
            DeepSeekV41ProductRuntimeProvider.runnerScale(
                statedBudgetBytes: 5_000_000_000, policy: iOS,
                availableMemoryBytes: 5_000_000_000),
            .product)
        // A budget that does buy a rung, on a device that says it cannot fund
        // it plus the stated overshoot, is still refused the stated scale.
        XCTAssertEqual(
            DeepSeekV41ProductRuntimeProvider.runnerScale(
                statedBudgetBytes: 6_400_000_000, policy: iOS,
                availableMemoryBytes: 6_500_000_000),
            .product)
        // And a device that can fund both gets the stated scale it asked for.
        XCTAssertEqual(
            DeepSeekV41ProductRuntimeProvider.runnerScale(
                statedBudgetBytes: 6_400_000_000, policy: iOS,
                availableMemoryBytes: 16_000_000_000),
            .stated(minimumBudgetBytes: 6_400_000_000, maximumNewTokens: 64))
        // A platform that states no per-process limit — the Mac — is not gated
        // by either rule, which is why nothing about its behaviour moved.
        XCTAssertEqual(
            DeepSeekV41ProductRuntimeProvider.runnerScale(
                statedBudgetBytes: 12_300_000_000,
                policy: DeepSeekV41ProductMemoryBudget.macOSProductPolicy,
                availableMemoryBytes: nil),
            .stated(minimumBudgetBytes: 12_300_000_000, maximumNewTokens: 64))
    }

    /// Both policies state a pinned envelope, and the iPhone's is the Mac's
    /// measurement because it is the only one that exists. A policy that
    /// claimed a smaller one would open rungs on a device that has just been
    /// killed at a smaller budget.
    func testBothPoliciesPriceAPinnedEnvelopeAndNeitherInventsOne() {
        XCTAssertEqual(
            DeepSeekV41ProductMemoryBudget.macOSProductPolicy.pinnedTransientExecutionBytes,
            5_200_000_000)
        XCTAssertEqual(iOS.pinnedTransientExecutionBytes, 5_200_000_000)
        XCTAssertGreaterThan(
            iOS.pinnedTransientExecutionBytes, iOS.transientExecutionBytes,
            "a run that pins holds at least what a run that streams holds")
        XCTAssertEqual(
            DeepSeekV41MemoryDialInputs.pinnedTransientExecutionBytes,
            DeepSeekV41ProductMemoryBudget.currentPolicy.pinnedTransientExecutionBytes,
            "the dial's term is the platform policy's, not a second constant")
    }

    // MARK: The chat a phone actually opens

    #if os(iOS)
        /// **The whole chain, hosted on the phone: New chat → 1,900,000,000 B.**
        ///
        /// Every other assertion in this file names a policy and checks the
        /// arithmetic over it. This one names nothing on the way in: it builds
        /// the app on an iPhone profile, presses New chat, and reads the number
        /// the conversation was written with — which is exactly the field the
        /// killed chat was pulled out of the app container carrying, and it
        /// said `3400000000`.
        ///
        /// It runs only on iOS because that is the claim. The Mac's half of the
        /// same sentence — a new V4.1 chat there starts at Balanced — is
        /// `BudgetPlanTests.testTheDefaultPresetIsPlatformConditionalForTheStreamedModels`.
        ///
        /// The device is stated rather than read: there is no test-host profile
        /// on iOS, so a plan computed against whatever simulator hosts the
        /// bundle would be a different assertion on every machine. 5.0 GB is
        /// about what `os_proc_available_memory()` gave the owner's iPhone 16
        /// Pro on the launch that was killed.
        @MainActor
        func testANewV41ChatOnAPhoneIsOpenedAtThePhonesFloor() throws {
            let app = AppModel.freshForTests(
                installedModels: [.deepseekV41Flash],
                device: DeviceProfile(
                    platform: .iOS, processMemoryBudgetBytes: 5_000_000_000,
                    hasMLXGPU: true, hasIncreasedMemoryLimitEntitlement: true))
            app.defaults.model = .deepseekV41Flash

            let id = try XCTUnwrap(app.newConversation())
            let conversation = try XCTUnwrap(app.conversation(id))

            XCTAssertEqual(conversation.settings.model, .deepseekV41Flash)
            XCTAssertEqual(conversation.settings.memoryBudgetBytes, 1_900_000_000)
            XCTAssertEqual(
                conversation.settings.memoryBudgetBytes,
                DeepSeekV41ProductMemoryBudget.iOSProductPolicy.minimumBudgetBytes,
                "the chat is opened at the phone's floor, not the Mac's")

            // A chat that carries the phone's floor takes the fully gated
            // product path: the plan enforced, the 512/64 limits enforced, the
            // MLX cache at zero. `.stated` is the path the killed run took.
            XCTAssertEqual(
                DeepSeekV41ProductRuntimeProvider.runnerScale(
                    statedBudgetBytes: conversation.settings.memoryBudgetBytes),
                .product)

            // And the dial the chat opens onto says the same number out loud.
            let plan = try XCTUnwrap(app.budgetPlan(for: conversation))
            XCTAssertEqual(plan.budget(for: .floor), 1_900_000_000)
            XCTAssertEqual(MRFormat.bytesDecimal(plan.budget(for: .floor)), "1.90 GB")
            XCTAssertEqual(plan.profile.refusalThresholdBytes, 1_900_000_000)
            XCTAssertNil(plan.refusal)
            XCTAssertTrue(plan.isRunnable)
            XCTAssertEqual(AppModel.defaultPreset(for: .deepseekV41Flash), .floor)
        }
    #endif
}
