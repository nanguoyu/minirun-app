import Foundation
import MinirunKit
import MLXBridge
import ModelAdapters
import StorageCore
import XCTest

@testable import MinirunRunners

/// V4's product runner is exercised against an injected, bounded workload.
///
/// The 166 GB publication remains an explicit environment gate. These tests
/// instead prove the lifecycle claims that must hold on every developer Mac:
/// complete preflight precedes acceptance, Stop reaches a blocking pass,
/// invalid runtime evidence cannot look like a user cancellation, and every
/// terminal path releases the workload and storage scope first.
final class DeepSeekV4DecodeRunnerTests: XCTestCase {
    private var root: URL!
    private var working: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-v4-runner-\(UUID().uuidString)")
        working = root.appendingPathComponent("work")
        try FileManager.default.createDirectory(
            at: working, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let root { try? FileManager.default.removeItem(at: root) }
        try super.tearDownWithError()
    }

    func testProductMemoryPolicyCombinesMeasuredAndExactTermsWithCheckedArithmetic() throws {
        let configURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("../Fixtures/deepseek-v4/current-official-config.json")
            .standardizedFileURL
        let config = try DeepSeekV4Config(contentsOf: configURL)
        let expertPool: UInt64 = 17_826_048

        XCTAssertEqual(DeepSeekV4ProductMemoryBudget.minimumBudgetBytes, 2_000_000_000)
        XCTAssertEqual(DeepSeekV4ProductMemoryBudget.transientExecutionBytes, 1_800_000_000)
        XCTAssertEqual(
            DeepSeekV4ProductMemoryBudget.observedMaximumPromptFootprintBytes,
            1_446_793_408)
        XCTAssertEqual(
            DeepSeekV4ProductMemoryBudget.observedMaximumPromptMLXPeakBytes,
            1_547_716_524)
        XCTAssertGreaterThan(
            DeepSeekV4ProductMemoryBudget.transientExecutionBytes,
            DeepSeekV4ProductMemoryBudget.observedMaximumPromptPeakBytes)

        let floor = try DeepSeekV4ProductMemoryBudget.plan(
            declaredBudgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
            config: config, promptTokenCount: 7, maximumNewTokens: 2,
            expertPoolBytes: expertPool, mlxCacheBytes: 0)
        XCTAssertEqual(
            floor.requiredBudgetBytes,
            DeepSeekV4ProductMemoryBudget.minimumBudgetBytes)
        XCTAssertTrue(floor.isAdmitted)
        XCTAssertTrue(floor.accountsForDeclaredBudget)
        XCTAssertGreaterThan(floor.retainedStateBytes, floor.replacementLayerStateBytes)

        let maximumRequest = try DeepSeekV4ProductMemoryBudget.plan(
            declaredBudgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
            config: config,
            promptTokenCount: DeepSeekV4ProductMemoryBudget.maximumPromptTokens,
            maximumNewTokens: DeepSeekV4ProductMemoryBudget.maximumNewTokens,
            expertPoolBytes: 8_912_896, mlxCacheBytes: 0)
        XCTAssertEqual(
            maximumRequest.requiredBudgetBytes,
            DeepSeekV4ProductMemoryBudget.minimumBudgetBytes)
        XCTAssertTrue(maximumRequest.isAdmitted)
        XCTAssertTrue(maximumRequest.accountsForDeclaredBudget)

        let cached = try DeepSeekV4ProductMemoryBudget.plan(
            declaredBudgetBytes: 3_000_000_000,
            config: config, promptTokenCount: 7, maximumNewTokens: 2,
            expertPoolBytes: expertPool, mlxCacheBytes: 200_000_000)
        XCTAssertEqual(
            cached.requiredBudgetBytes,
            2_041_411_840 + cached.replacementLayerStateBytes)
        XCTAssertEqual(cached.headroomBytes, 3_000_000_000 - cached.requiredBudgetBytes)
        XCTAssertTrue(cached.accountsForDeclaredBudget)
        XCTAssertGreaterThan(
            DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
            DeepSeekV4ProductMemoryBudget.observedMaximumPromptPeakBytes)

        XCTAssertThrowsError(
            try DeepSeekV4ProductMemoryBudget.requiredBudgetBytes(
                config: config, promptTokenCount: 7, maximumNewTokens: 2,
                expertPoolBytes: .max, mlxCacheBytes: 0)) {
            guard case DeepSeekV4Error.configuration(let detail) = $0 else {
                return XCTFail("unexpected error: \($0)")
            }
            XCTAssertTrue(detail.contains("UInt64.max"), detail)
        }
    }

    /// The stated scale derives `min(declaredBudget / 10, 512 MiB)`; a stated
    /// knob always wins, including a stated 0. The product scale keeps 0, and
    /// with it the unconditional every-boundary reclaim.
    func testMLXCacheDefaultFollowsTheScaleAndTheDeclaredBudget() throws {
        let stated = DeepSeekV4DecodeRunner.Scale.stated(
            minimumBudgetBytes: 1 << 30, maximumNewTokens: 8)
        XCTAssertEqual(DeepSeekV4EffectiveKnobs.maximumDefaultMLXCacheBytes, 512 << 20)

        for (budget, expected) in [
            (UInt64(24) << 30, 512 << 20),
            (UInt64(8) << 30, 512 << 20),
            (UInt64(2_000_000_000), 200_000_000),
            (UInt64(32), 3),
            (UInt64(0), 0),
        ] as [(UInt64, Int)] {
            XCTAssertEqual(
                DeepSeekV4EffectiveKnobs.defaultMLXCacheLimitBytes(
                    scale: stated, declaredBudgetBytes: budget),
                expected, "stated budget \(budget)")
            XCTAssertEqual(
                DeepSeekV4EffectiveKnobs.defaultMLXCacheLimitBytes(
                    scale: .product, declaredBudgetBytes: budget),
                0, "the product scale keeps the measured floor's zero")
        }

        let derived = try DeepSeekV4EffectiveKnobs.resolve(
            .init(), scale: stated, declaredBudgetBytes: 24 << 30)
        XCTAssertEqual(derived.mlxCacheLimitBytes, 512 << 20)
        XCTAssertEqual(derived.asStrings["mlxCacheLimitBytes"], "\(512 << 20)")
        XCTAssertEqual(derived.asRunKnobs.mlxCacheLimitBytes, 512 << 20)

        var explicit = RunKnobs()
        explicit.mlxCacheLimitBytes = 64 << 20
        XCTAssertEqual(
            try DeepSeekV4EffectiveKnobs.resolve(
                explicit, scale: stated, declaredBudgetBytes: 24 << 30
            ).mlxCacheLimitBytes,
            64 << 20)

        var explicitZero = RunKnobs()
        explicitZero.mlxCacheLimitBytes = 0
        XCTAssertEqual(
            try DeepSeekV4EffectiveKnobs.resolve(
                explicitZero, scale: stated, declaredBudgetBytes: 24 << 30
            ).mlxCacheLimitBytes,
            0, "a stated 0 is a statement, not an absent knob")
        XCTAssertEqual(
            try DeepSeekV4EffectiveKnobs.resolve(
                .init(), scale: .product,
                declaredBudgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes
            ).mlxCacheLimitBytes,
            0)

        // The reclaim policy follows the effective limit, not the scale name:
        // any run that recycles nothing keeps the unconditional behaviour.
        XCTAssertTrue(
            try DeepSeekV4EffectiveKnobs.resolve(
                .init(), scale: .product,
                declaredBudgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes
            ).boundaryReclaim(
                declaredBudgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes
            ).reclaimsAtEveryBoundary)
        XCTAssertTrue(
            try DeepSeekV4EffectiveKnobs.resolve(
                explicitZero, scale: stated, declaredBudgetBytes: 24 << 30
            ).boundaryReclaim(declaredBudgetBytes: 24 << 30).reclaimsAtEveryBoundary)
        let bounded = derived.boundaryReclaim(declaredBudgetBytes: 24 << 30)
        XCTAssertFalse(bounded.reclaimsAtEveryBoundary)
        XCTAssertEqual(
            bounded.thresholdFootprintBytes,
            (24 << 30) - (24 << 30) / 10)
    }

    /// The derived cache is memory the plan names. It must appear in the
    /// `mlxCacheBytes` term and leave the seven-term identity intact, and the
    /// 2 GB floor — which derives 0 — must be byte-identical to before.
    /// The memory dial's residency is the eighth term of the identity, and it
    /// partitions the declared ceiling exactly like the other seven.
    func testPinnedDeterministicBytesAreTheEighthTermOfTheIdentity() throws {
        let config = try Self.officialConfig()
        let expertPool: UInt64 = 17_826_048
        let declared: UInt64 = 24 << 30
        let cache: UInt64 = 512 << 20
        let pinned: UInt64 = 6_000_000_000

        let plan = try DeepSeekV4ProductMemoryBudget.plan(
            declaredBudgetBytes: declared,
            config: config, promptTokenCount: 7, maximumNewTokens: 2,
            expertPoolBytes: expertPool, mlxCacheBytes: cache,
            pinnedDeterministicBytes: pinned)

        XCTAssertEqual(plan.pinnedDeterministicBytes, pinned)
        XCTAssertTrue(plan.isAdmitted)
        XCTAssertTrue(plan.accountsForDeclaredBudget)
        XCTAssertEqual(
            plan.transientExecutionBytes + plan.retainedStateBytes
                + plan.replacementLayerStateBytes + plan.expertPoolBytes
                + plan.mlxCacheBytes + plan.pinnedDeterministicBytes
                + plan.productFloorPaddingBytes + plan.headroomBytes,
            declared)

        // Pins are paid for out of headroom, byte for byte.
        let unpinned = try DeepSeekV4ProductMemoryBudget.plan(
            declaredBudgetBytes: declared,
            config: config, promptTokenCount: 7, maximumNewTokens: 2,
            expertPoolBytes: expertPool, mlxCacheBytes: cache)
        XCTAssertEqual(unpinned.pinnedDeterministicBytes, 0)
        XCTAssertEqual(unpinned.headroomBytes - plan.headroomBytes, pinned)
        XCTAssertEqual(plan.requiredBudgetBytes - unpinned.requiredBudgetBytes, pinned)
    }

    /// The product scale is byte-identical to what the 2 GB floor was accepted
    /// against: the eighth term exists but is zero, so nothing moves.
    func testTheProductScalePinsNothingAndItsPlanIsUnchanged() throws {
        let config = try Self.officialConfig()
        let plan = try DeepSeekV4ProductMemoryBudget.plan(
            declaredBudgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
            config: config, promptTokenCount: 512, maximumNewTokens: 64,
            expertPoolBytes: 17_826_048, mlxCacheBytes: 0)

        XCTAssertEqual(plan.pinnedDeterministicBytes, 0)
        XCTAssertEqual(
            plan.requiredBudgetBytes, DeepSeekV4ProductMemoryBudget.minimumBudgetBytes)
        XCTAssertTrue(plan.isAdmitted)
        XCTAssertTrue(plan.accountsForDeclaredBudget)
    }

    /// A pin set the budget cannot fund is refused by name, and the refusal
    /// names the larger requirement rather than trimming the set to fit.
    func testAPinSetTheBudgetCannotFundIsRefusedRatherThanClamped() throws {
        let config = try Self.officialConfig()
        let declared: UInt64 = 8 << 30
        let pinned: UInt64 = 60 << 30

        let plan = try DeepSeekV4ProductMemoryBudget.plan(
            declaredBudgetBytes: declared,
            config: config, promptTokenCount: 7, maximumNewTokens: 2,
            expertPoolBytes: 17_826_048, mlxCacheBytes: 512 << 20,
            pinnedDeterministicBytes: pinned)

        XCTAssertFalse(plan.isAdmitted)
        XCTAssertGreaterThan(plan.requiredBudgetBytes, declared)
        // The whole pin set is still stated. Nothing was silently dropped to
        // make the plan fit.
        XCTAssertEqual(plan.pinnedDeterministicBytes, pinned)
        XCTAssertEqual(plan.headroomBytes, 0)
        // A refused plan does not claim to partition a budget it exceeded.
        XCTAssertFalse(plan.accountsForDeclaredBudget)
    }

    /// The stated scale states its own horizon; the product limits are the
    /// product scale's policy and must not follow the dial.
    func testStatedPlanningSkipsTheProductPromptAndTokenCeilings() throws {
        let config = try Self.officialConfig()
        XCTAssertThrowsError(
            try DeepSeekV4ProductMemoryBudget.plan(
                declaredBudgetBytes: 64 << 30,
                config: config, promptTokenCount: 4_096, maximumNewTokens: 256,
                expertPoolBytes: 0, mlxCacheBytes: 0))

        let stated = try DeepSeekV4ProductMemoryBudget.plan(
            declaredBudgetBytes: 64 << 30,
            config: config, promptTokenCount: 4_096, maximumNewTokens: 256,
            expertPoolBytes: 0, mlxCacheBytes: 0,
            enforcesProductLimits: false)
        XCTAssertTrue(stated.isAdmitted)
        XCTAssertTrue(stated.accountsForDeclaredBudget)
    }

    private static func officialConfig() throws -> DeepSeekV4Config {
        let configURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("../Fixtures/deepseek-v4/current-official-config.json")
            .standardizedFileURL
        return try DeepSeekV4Config(contentsOf: configURL)
    }

    func testDerivedMLXCacheIsAccountedForInTheDeclaredBudget() throws {
        let configURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("../Fixtures/deepseek-v4/current-official-config.json")
            .standardizedFileURL
        let config = try DeepSeekV4Config(contentsOf: configURL)
        let expertPool: UInt64 = 17_826_048
        let declared: UInt64 = 24 << 30
        let derived = UInt64(
            DeepSeekV4EffectiveKnobs.defaultMLXCacheLimitBytes(
                scale: .stated(minimumBudgetBytes: 1 << 30, maximumNewTokens: 8),
                declaredBudgetBytes: declared))
        XCTAssertEqual(derived, 512 << 20)

        let plan = try DeepSeekV4ProductMemoryBudget.plan(
            declaredBudgetBytes: declared,
            config: config, promptTokenCount: 7, maximumNewTokens: 2,
            expertPoolBytes: expertPool, mlxCacheBytes: derived)
        XCTAssertEqual(plan.mlxCacheBytes, derived)
        XCTAssertTrue(plan.isAdmitted)
        XCTAssertTrue(plan.accountsForDeclaredBudget)
        XCTAssertEqual(
            plan.transientExecutionBytes + plan.retainedStateBytes
                + plan.replacementLayerStateBytes + plan.expertPoolBytes
                + plan.mlxCacheBytes + plan.productFloorPaddingBytes
                + plan.headroomBytes,
            declared)

        // The product floor derives 0, so the accepted plan is unchanged.
        let floor = try DeepSeekV4ProductMemoryBudget.plan(
            declaredBudgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
            config: config, promptTokenCount: 7, maximumNewTokens: 2,
            expertPoolBytes: expertPool,
            mlxCacheBytes: UInt64(
                DeepSeekV4EffectiveKnobs.defaultMLXCacheLimitBytes(
                    scale: .product,
                    declaredBudgetBytes:
                        DeepSeekV4ProductMemoryBudget.minimumBudgetBytes)))
        XCTAssertEqual(floor.mlxCacheBytes, 0)
        XCTAssertEqual(
            floor.requiredBudgetBytes,
            DeepSeekV4ProductMemoryBudget.minimumBudgetBytes)
        XCTAssertTrue(floor.accountsForDeclaredBudget)
    }

    func testDerivedMLXCacheIsRecordedAndReservedByAnAcceptedStatedRun() async throws {
        let state = V4FakeState()
        let run = await DrainedRun.drain(
            try makeRunner(state: state).start(
                request(budgetBytes: 24 << 30, maximumNewTokens: 1)))

        XCTAssertNil(run.failure)
        XCTAssertEqual(
            run.acceptance?.effectiveKnobs.mlxCacheLimitBytes, 512 << 20,
            "the accepted record states the derived cache, not a silent one")

        // The same derived term is reserved before acceptance: a stated budget
        // that cannot hold the pool plus the derived cache is refused by name.
        let crowded = V4FakeState(expertPoolBudgetBytes: 300)
        let refused = await DrainedRun.drain(
            try makeRunner(state: crowded, minimumBudgetBytes: 1).start(
                request(budgetBytes: 320)))
        XCTAssertEqual(
            refused.failure as? RunError,
            .budgetBelowMinimum(
                declared: 320, minimum: 332, model: .deepseekV4Flash))
    }

    func testCapabilitiesAndUnsupportedRequestsAreRefusedByName() throws {
        let state = V4FakeState()
        let runner = makeRunner(state: state, minimumBudgetBytes: 8, maximumNewTokens: 4)

        XCTAssertEqual(runner.capabilities.model, .deepseekV4Flash)
        XCTAssertEqual(runner.capabilities.layout, .v4FlashUnitBundle)
        XCTAssertEqual(runner.capabilities.tokenEventGranularity, .perPass)
        XCTAssertFalse(runner.capabilities.acceptsTextPrompts)
        XCTAssertTrue(runner.capabilities.requiresMLX)
        XCTAssertFalse(runner.capabilities.supportsPinPlan)
        XCTAssertNil(MinirunRunnerRegistry.runner(for: .deepseekV4Flash))
        let productKnobs = try DeepSeekV4EffectiveKnobs.resolve(
            .init(), scale: .product,
            declaredBudgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes)
        XCTAssertEqual(productKnobs.expertReadAhead, 2)
        XCTAssertEqual(productKnobs.expertPoolSlots, 2)
        XCTAssertEqual(productKnobs.expertConfiguration.transferMode, .copy)
        XCTAssertEqual(productKnobs.asStrings["expertTransferMode"], "copy")

        XCTAssertThrowsError(
            try runner.validate(request(prompt: .text("hello"), budgetBytes: 8))
        ) { error in
            guard case RunError.promptNotSupported = error else {
                return XCTFail("\(error)")
            }
        }

        var unsupported = RunKnobs()
        unsupported.expertsPerDispatch = 2
        XCTAssertThrowsError(
            try runner.validate(request(budgetBytes: 8, knobs: unsupported))
        ) { error in
            XCTAssertEqual(
                error as? RunError,
                .knobNotSupported(
                    name: "expertsPerDispatch", byModel: .deepseekV4Flash))
        }

        var serial = RunKnobs()
        serial.expertReadAhead = 0
        XCTAssertThrowsError(
            try runner.validate(request(budgetBytes: 8, knobs: serial))
        ) { error in
            guard case RunError.knobOutOfRange(let name, _, _) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertEqual(name, "expertReadAhead")
        }

        var deadlock = RunKnobs()
        deadlock.expertReadAhead = 3
        deadlock.expertPoolSlots = 2
        XCTAssertThrowsError(
            try runner.validate(request(budgetBytes: 8, knobs: deadlock))
        ) { error in
            guard case RunError.knobOutOfRange(let name, _, let allowed) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertEqual(name, "expertPoolSlots")
            XCTAssertTrue(allowed.contains(">= expertReadAhead (3)"), allowed)
            XCTAssertTrue(allowed.contains("copy transfer"), allowed)
        }

        var excessiveQueue = RunKnobs()
        excessiveQueue.queueDepth = 9
        XCTAssertThrowsError(
            try runner.validate(request(budgetBytes: 8, knobs: excessiveQueue))
        ) { error in
            XCTAssertEqual(
                error as? RunError,
                .knobOutOfRange(name: "queueDepth", value: "9", allowed: "1...8"))
        }

        let released = StorageScope(url: root)
        released.release()
        XCTAssertThrowsError(
            try runner.validate(request(scope: released, budgetBytes: 8))
        ) { error in
            XCTAssertEqual(error as? RunError, .scopeRefused(path: self.root.path))
        }
    }

    func testProductionRunnerRequiresRootedVerificationAuthority() throws {
        #if arch(arm64)
            let runner = DeepSeekV4DecodeRunner(
                scale: .stated(minimumBudgetBytes: 1, maximumNewTokens: 1))
            XCTAssertThrowsError(
                try runner.validate(request(budgetBytes: 1, maximumNewTokens: 1))
            ) { error in
                guard case RunError.artifactNotReady(let reason) = error else {
                    return XCTFail("\(error)")
                }
                XCTAssertTrue(reason.contains("rooted runtime authority"), reason)
            }
        #endif
    }

    func testProductScalePublishesAndEnforcesItsMeasuredLimits() throws {
        #if arch(arm64)
            let state = V4FakeState()
            let runner = DeepSeekV4DecodeRunner(
                scale: .product, factory: V4FakeFactory(state: state))
            XCTAssertEqual(
                runner.capabilities.minimumBudgetBytes,
                DeepSeekV4ProductMemoryBudget.minimumBudgetBytes)
            XCTAssertEqual(
                runner.capabilities.maximumNewTokens,
                DeepSeekV4ProductMemoryBudget.maximumNewTokens)

            XCTAssertThrowsError(
                try runner.validate(
                    request(
                        prompt: .tokenIDs(
                            Array(
                                repeating: 1,
                                count: DeepSeekV4ProductMemoryBudget.maximumPromptTokens + 1)),
                        budgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
                        maximumNewTokens: 1))) {
                guard case RunError.promptNotSupported(let reason) = $0 else {
                    return XCTFail("unexpected error: \($0)")
                }
                XCTAssertTrue(reason.contains("at most 512"), reason)
            }
        #endif
    }

    func testProductRequestDerivedBudgetRefusesBeforeAcceptance() async throws {
        #if arch(arm64)
            let required = DeepSeekV4ProductMemoryBudget.minimumBudgetBytes + 1
            let state = V4FakeState(productMemoryRequirementBytes: required)
            let runner = DeepSeekV4DecodeRunner(
                scale: .product, factory: V4FakeFactory(state: state))
            let session = try runner.start(
                request(
                    prompt: .tokenIDs([1]),
                    budgetBytes: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
                    maximumNewTokens: 1))
            let run = await DrainedRun.drain(session)

            XCTAssertNil(run.acceptance)
            XCTAssertNil(run.finished)
            XCTAssertEqual(
                run.failure as? RunError,
                .budgetBelowMinimum(
                    declared: DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
                    minimum: required,
                    model: .deepseekV4Flash))
            XCTAssertEqual(state.shutdownCount, 1)
        #endif
    }

    func testAcceptedProductRunRecordsTheExactBoundedMemoryPlan() async throws {
        #if arch(arm64)
            let required = DeepSeekV4ProductMemoryBudget.minimumBudgetBytes
            let declared = required + 500_000_000
            let state = V4FakeState(productMemoryRequirementBytes: required)
            let runner = DeepSeekV4DecodeRunner(
                scale: .product, factory: V4FakeFactory(state: state))
            let run = await DrainedRun.drain(
                try runner.start(
                    request(
                        prompt: .tokenIDs([1]), budgetBytes: declared,
                        maximumNewTokens: 1)))

            XCTAssertNil(run.failure)
            XCTAssertEqual(run.acceptance?.declaredBudgetBytes, declared)
            XCTAssertNil(run.acceptance?.pinPlan)
            let payload = try run.payload()
            let plan = try XCTUnwrap(payload["productMemoryPlan"] as? [String: Any])
            XCTAssertEqual(plan["declaredBudgetBytes"] as? UInt64, declared)
            XCTAssertEqual(plan["requiredBudgetBytes"] as? UInt64, required)
            XCTAssertEqual(plan["headroomBytes"] as? UInt64, declared - required)
            XCTAssertTrue(
                try XCTUnwrap(run.finished).lines.contains {
                    $0.contains("one retained generation + one layer replacement")
                })
        #endif
    }

    func testACompletePreflightIsAcceptedThenFinishesAndReleasesEverything() async throws {
        let state = V4FakeState(
            steps: [
                .init(tokenID: 1, logits: [0, 2, 1, -1], completedLayers: 3),
                .init(tokenID: 3, logits: [0, 1, 2, 4], completedLayers: 3),
            ])
        let scope = StorageScope(url: root)
        let runner = makeRunner(state: state)
        let session = try runner.start(request(scope: scope))

        var observedShutdownAtTerminal = false
        var run = DrainedRun()
        do {
            for try await event in session.events {
                run.events.append(event)
                switch event {
                case .accepted(let acceptance): run.acceptance = acceptance
                case .phase(let phase): run.phases.append(phase)
                case .token(let token): run.tokens.append(token)
                case .telemetry(let telemetry): run.telemetry.append(telemetry)
                case .phaseSummary(let summary): run.phaseSummaries.append(summary)
                case .log(let line): run.logs.append(line)
                case .finished(let summary):
                    observedShutdownAtTerminal = state.isShutdown
                    run.finished = summary
                case .cancelled(let summary): run.cancelled = summary
                }
            }
        } catch {
            run.failure = error
        }

        XCTAssertNil(run.failure)
        XCTAssertNil(run.cancelled)
        XCTAssertEqual(run.finished?.tokenIDs, [1, 3])
        XCTAssertEqual(run.tokens.map(\.tokenID), [1, 3])
        XCTAssertEqual(state.prepareCount, 1)
        XCTAssertEqual(state.terminalValidationCount, 1)
        XCTAssertEqual(state.shutdownCount, 1)
        XCTAssertTrue(observedShutdownAtTerminal)
        XCTAssertTrue(scope.isReleased)
        XCTAssertTrue(run.finished?.logitsDigest?.isValid == true)
        XCTAssertTrue(
            run.finished?.lines.contains {
                $0.contains("payload byte flow unavailable")
            } == true)
        XCTAssertTrue(run.telemetry.allSatisfy { $0.byteAccountingReported == false })
        XCTAssertTrue(run.telemetry.allSatisfy { !$0.reportsByteAccounting })
        XCTAssertEqual(run.finished?.finalTelemetry.byteAccountingReported, false)
        XCTAssertFalse(try XCTUnwrap(run.finished).finalTelemetry.reportsByteAccounting)
        guard case .accepted = run.events.first else {
            return XCTFail("acceptance must be the first visible event")
        }

        let payload = try run.payload()
        XCTAssertEqual(payload["outcome"] as? String, "finished")
        XCTAssertEqual(payload["byteAccountingReported"] as? Bool, false)
        let telemetry = try XCTUnwrap(payload["telemetry"] as? [String: Any])
        XCTAssertEqual(telemetry["byteAccountingReported"] as? Bool, false)
        let teardown = try XCTUnwrap(payload["teardown"] as? [String: Any])
        XCTAssertEqual(teardown["workloadReleased"] as? Bool, true)
        XCTAssertEqual(teardown["scopeReleased"] as? Bool, true)
    }

    func testCompleteObserverPublishesBalancedBytesAtEachTokenBoundary() async throws {
        let state = V4FakeState(
            steps: [
                .init(tokenID: 1, logits: [0, 2, 1, -1], completedLayers: 3),
                .init(tokenID: 3, logits: [0, 1, 2, 4], completedLayers: 3),
            ],
            readAccountingPerStep: [(100, 40), (70, 30)])
        let run = await DrainedRun.drain(
            try makeRunner(state: state).start(
                request(maximumNewTokens: 2)))

        XCTAssertNil(run.failure)
        let final = try XCTUnwrap(run.finished?.finalTelemetry)
        XCTAssertTrue(final.reportsByteAccounting)
        XCTAssertEqual(final.bytes.totalBytesRead, 240)
        XCTAssertEqual(final.bytes.deterministicBytesRead, 170)
        XCTAssertEqual(final.bytes.expertBytesRead, 70)
        XCTAssertTrue(final.bytes.isBalanced)
        XCTAssertEqual(final.completedTokenBytes, final.bytes)
        XCTAssertEqual(
            final.prefillBytes,
            ByteAccounting(
                totalBytesRead: 140, deterministicBytesRead: 100, expertBytesRead: 40))
        XCTAssertEqual(final.bytesPerToken, 100)
        XCTAssertEqual(final.decodeTokensCompleted, 1)
        XCTAssertGreaterThan(try XCTUnwrap(final.prefillSeconds), 0)
        XCTAssertGreaterThan(try XCTUnwrap(final.decodeSeconds), 0)
        XCTAssertEqual(final.generationStage, .terminal)
        XCTAssertNotNil(final.bytesPerSecond)
        let completedSamples = Dictionary(
            run.telemetry.compactMap { telemetry -> (UInt64, UInt64)? in
                guard let completed = telemetry.completedTokenBytes else { return nil }
                return (completed.totalBytesRead, telemetry.bytesPerToken ?? .max)
            },
            uniquingKeysWith: { _, newest in newest })
        XCTAssertEqual(completedSamples[140], .max)
        XCTAssertEqual(completedSamples[240], 100)
        XCTAssertTrue(
            try XCTUnwrap(run.finished).lines.contains {
                $0.contains("payload read") && $0.contains("deterministic")
                    && $0.contains("expert")
            })
        let payload = try run.payload()
        XCTAssertEqual(payload["byteAccountingReported"] as? Bool, true)
    }

    /// V4 had no phase decomposition at all, so the audit of 2026-08-14 could
    /// not say where a ~53 s decode pass goes. Prefill and decode are different
    /// workloads and are attributed apart, exactly as their seconds and bytes
    /// already are.
    func testEachPassIsAttributedApartAndSurvivesTheResultEncoding() async throws {
        let state = V4FakeState(
            steps: [
                .init(tokenID: 1, logits: [0, 2, 1, -1], completedLayers: 3),
                .init(tokenID: 2, logits: [0, 1, 2, -1], completedLayers: 3),
                .init(tokenID: 3, logits: [0, 1, 2, 4], completedLayers: 3),
            ],
            reportsPhaseMetrics: true)
        let run = await DrainedRun.drain(
            try makeRunner(state: state).start(request(maximumNewTokens: 3)))

        XCTAssertNil(run.failure)
        XCTAssertEqual(run.finished?.tokenIDs, [1, 2, 3])
        XCTAssertEqual(
            run.logs.filter { $0.hasPrefix("[v4 phase]") }.count, 3,
            "one split per pass, prefill included")
        XCTAssertTrue(
            try XCTUnwrap(run.finished).lines.contains { $0.hasPrefix("[v4 phase]") })

        let payload = try run.payload()
        let prefill = try XCTUnwrap(payload["prefillPhaseMetrics"] as? [String: Any])
        let decode = try XCTUnwrap(payload["decodePhaseMetrics"] as? [[String: Any]])
        XCTAssertEqual(decode.count, 2)

        for (label, pass) in [("prefill", prefill)] + decode.map({ ("decode", $0) }) {
            let wall = try XCTUnwrap(pass["passSeconds"] as? Double, label)
            let attributed = try XCTUnwrap(pass["attributedSeconds"] as? Double, label)
            let unattributed = try XCTUnwrap(pass["unattributedSeconds"] as? Double, label)
            let terms = try [
                "deterministicReadSeconds", "tileDigestSeconds", "expertPhaseSeconds",
                "outputHeadReadSeconds", "outputHeadComputeSeconds", "reclaimSeconds",
                "expertIOWaitSeconds", "expertGatherComputeSeconds",
            ].map { try XCTUnwrap(pass[$0] as? Double, "\(label) \($0)") }

            XCTAssertGreaterThan(wall, 0, label)
            XCTAssertTrue(terms.allSatisfy { $0 >= 0 }, "\(label) has a negative term")
            XCTAssertLessThanOrEqual(attributed, wall + 1e-9, label)
            XCTAssertEqual(unattributed, wall - attributed, accuracy: 1e-9)
            XCTAssertGreaterThan(
                try XCTUnwrap(pass["tileDigestSeconds"] as? Double), 0,
                "\(label) loaded tiles, so its digest bracket cannot be empty")
            XCTAssertGreaterThanOrEqual(
                try XCTUnwrap(pass["expertPhaseSeconds"] as? Double),
                try XCTUnwrap(pass["expertIOWaitSeconds"] as? Double)
                    + XCTUnwrap(pass["expertGatherComputeSeconds"] as? Double) - 1e-9,
                "\(label): the expert halves must lie inside their phase")
            XCTAssertEqual(pass["deterministicReadCount"] as? Int, 3, label)
            XCTAssertEqual(pass["expertAcquireCount"] as? Int, 9, label)
        }

        // Each pass is its own difference, not a running total.
        let prefillExpert = try XCTUnwrap(prefill["expertPhaseSeconds"] as? Double)
        let firstDecodeExpert = try XCTUnwrap(decode[0]["expertPhaseSeconds"] as? Double)
        let secondDecodeExpert = try XCTUnwrap(decode[1]["expertPhaseSeconds"] as? Double)
        XCTAssertLessThan(
            secondDecodeExpert, prefillExpert + firstDecodeExpert + secondDecodeExpert)
    }

    /// The same measurement the log line carries, in the shape the product can
    /// draw and persist. It travels on the run's own event stream, so a screen
    /// never has to parse `[v4 phase]` prose to recover fields that already
    /// exist.
    func testEachPassAlsoPublishesItsSplitAsAProductEvent() async throws {
        let state = V4FakeState(
            steps: [
                .init(tokenID: 1, logits: [0, 2, 1, -1], completedLayers: 3),
                .init(tokenID: 2, logits: [0, 1, 2, -1], completedLayers: 3),
                .init(tokenID: 3, logits: [0, 1, 2, 4], completedLayers: 3),
            ],
            reportsPhaseMetrics: true)
        let run = await DrainedRun.drain(
            try makeRunner(state: state).start(request(maximumNewTokens: 3)))

        XCTAssertNil(run.failure)
        XCTAssertEqual(run.phaseSummaries.count, 3, "one summary per pass, prefill included")
        XCTAssertEqual(
            run.phaseSummaries.map(\.passKind), [.prefill, .decode, .decode])
        XCTAssertEqual(run.phaseSummaries.map(\.passIndex), [0, 1, 2])
        XCTAssertEqual(
            run.phaseSummaries.count,
            run.logs.filter { $0.hasPrefix("[v4 phase]") }.count,
            "the event and the log line are one measurement, not two")

        for summary in run.phaseSummaries {
            XCTAssertGreaterThan(summary.passSeconds, 0)
            XCTAssertTrue(summary.isBalanced, "the terms do not fit inside the pass")
            XCTAssertLessThanOrEqual(
                summary.attributedSeconds, summary.passSeconds + 1e-9)
            XCTAssertTrue(summary.terms.allSatisfy { $0.isInsidePass })
            XCTAssertEqual(summary.terms.count, Set(summary.terms.map(\.name)).count)
            // The expert phase is published as its disjoint pieces; drawing the
            // container beside them would count the same seconds twice.
            XCTAssertNotNil(summary.seconds(of: RunPhaseTermName.expertIOWait))
            XCTAssertNotNil(summary.seconds(of: RunPhaseTermName.expertGatherCompute))
            XCTAssertNotNil(summary.seconds(of: RunPhaseTermName.expertOther))
            XCTAssertNil(summary.seconds(of: "expertPhase"))
            XCTAssertEqual(
                summary.terms.first { $0.name == RunPhaseTermName.deterministicRead }?.count, 3)
            XCTAssertEqual(
                summary.terms.first { $0.name == RunPhaseTermName.expertIOWait }?.count, 9)
        }

        // A pass is its own difference, not a running total.
        let waits = run.phaseSummaries.compactMap { $0.seconds(of: RunPhaseTermName.expertIOWait) }
        XCTAssertEqual(waits.count, 3)
        XCTAssertGreaterThan(waits.reduce(0, +), 0)

        let aggregate = try XCTUnwrap(
            RunPhaseSummary.aggregate(run.phaseSummaries.filter { $0.passKind == .decode }))
        XCTAssertEqual(aggregate.passCount, 2)
        XCTAssertEqual(
            aggregate.passSeconds,
            run.phaseSummaries.filter { $0.passKind == .decode }
                .reduce(0) { $0 + $1.passSeconds },
            accuracy: 1e-9)
    }

    func testAWorkloadThatMeasuresNothingPublishesNoProductSplitEither() async throws {
        let run = await DrainedRun.drain(
            try makeRunner(state: V4FakeState()).start(request(maximumNewTokens: 1)))

        XCTAssertNil(run.failure)
        XCTAssertTrue(
            run.phaseSummaries.isEmpty,
            "a split that cannot be trusted is worse than none, and zeros read as a measurement")
    }

    /// The pinned tier's counter is run-scoped, and a run-scoped total cannot
    /// answer the question a pinned arm is run to answer. The pass that fills
    /// the tier reads its bytes; every pass after it is served them. Attributing
    /// the counter per pass is what separates the two.
    func testPinnedServedBytesAreAttributedPerPassRatherThanPerRun() async throws {
        let state = V4FakeState(
            steps: [
                .init(tokenID: 1, logits: [0, 2, 1, -1], completedLayers: 3),
                .init(tokenID: 2, logits: [0, 1, 2, -1], completedLayers: 3),
                .init(tokenID: 3, logits: [0, 1, 2, 4], completedLayers: 3),
            ],
            readAccountingPerStep: [(4_096, 512), (0, 512), (0, 512)],
            pinnedServedPerStep: [0, 4_096, 4_096])
        let run = await DrainedRun.drain(
            try makeRunner(state: state).start(request(maximumNewTokens: 3)))

        XCTAssertNil(run.failure)
        XCTAssertEqual(run.finished?.tokenIDs, [1, 2, 3])
        let payload = try run.payload()
        XCTAssertEqual(payload["prefillPinnedServedBytes"] as? UInt64, 0)
        XCTAssertEqual(payload["decodePinnedServedBytes"] as? [UInt64], [4_096, 4_096])
        // Served bytes stay outside the read identity: the run read less
        // because the tier held more, and the total may not grow to hide it.
        XCTAssertEqual(
            (payload["telemetry"] as? [String: Any])
                .flatMap { ($0["bytes"] as? [String: Any])?["totalBytesRead"] as? UInt64 },
            4_096 + 512 * 3)
    }

    func testAWorkloadThatMeasuresNothingPublishesNoDecomposition() async throws {
        let run = await DrainedRun.drain(
            try makeRunner(state: V4FakeState()).start(request(maximumNewTokens: 1)))

        XCTAssertNil(run.failure)
        XCTAssertTrue(run.logs.allSatisfy { !$0.hasPrefix("[v4 phase]") })
        let payload = try run.payload()
        XCTAssertNil(payload["prefillPhaseMetrics"])
        XCTAssertEqual((payload["decodePhaseMetrics"] as? [[String: Any]])?.count, 0)
        // A workload that observes no bytes cannot say a pass was served zero
        // of them either, and a zero there would read as a measurement.
        XCTAssertNil(payload["prefillPinnedServedBytes"])
        XCTAssertEqual((payload["decodePinnedServedBytes"] as? [UInt64])?.count, 0)
    }

    func testOverflowedObserverCannotLookLikeReportedByteFlow() async throws {
        let state = V4FakeState(
            readAccountingPerStep: [(.max, 1)])
        let run = await DrainedRun.drain(
            try makeRunner(state: state).start(
                request(maximumNewTokens: 1)))

        XCTAssertNil(run.failure)
        let final = try XCTUnwrap(run.finished?.finalTelemetry)
        XCTAssertFalse(final.reportsByteAccounting)
        XCTAssertNil(final.completedTokenBytes)
        XCTAssertNil(final.bytesPerToken)
        XCTAssertNil(final.bytesPerSecond)
        XCTAssertTrue(
            try XCTUnwrap(run.finished).lines.contains {
                $0.contains("accounting overflowed")
            })
    }

    func testStopInterruptsABlockingPassAndPublishesOneCancelledTerminal() async throws {
        let state = V4FakeState(blockPrefill: true)
        let scope = StorageScope(url: root)
        let session = try makeRunner(state: state).start(request(scope: scope))
        let draining = Task { await DrainedRun.drain(session) }

        XCTAssertTrue(state.waitUntilPrefillEntered(timeout: 2))
        session.cancel()
        let run = await draining.value

        XCTAssertNil(run.failure)
        XCTAssertNil(run.finished)
        XCTAssertNotNil(run.cancelled)
        XCTAssertEqual(run.events.reduce(into: 0) { count, event in
            if case .cancelled = event { count += 1 }
        }, 1)
        XCTAssertEqual(state.shutdownCount, 1)
        XCTAssertTrue(scope.isReleased)
        XCTAssertEqual(run.cancelled?.tokenIDs, [])
    }

    func testCancellingTheEventConsumerClosesTheSameBlockingDoor() async throws {
        let state = V4FakeState(blockPrefill: true)
        let scope = StorageScope(url: root)
        let session = try makeRunner(state: state).start(request(scope: scope))
        let consumer = Task {
            do {
                for try await _ in session.events {}
            } catch {}
        }

        XCTAssertTrue(state.waitUntilPrefillEntered(timeout: 2))
        consumer.cancel()
        let closedByConsumer = state.waitUntilShutdown(timeout: 2)
        // Keep a failed regression from leaving its dedicated decode thread
        // behind after the test process moves on.
        session.cancel()
        _ = await consumer.result

        XCTAssertTrue(closedByConsumer)
        XCTAssertEqual(state.shutdownCount, 1)
        XCTAssertTrue(scope.isReleased)
    }

    func testPreflightFailureNeverAcceptsAndStillReleasesTheScope() async throws {
        let state = V4FakeState(prepareError: .artifactNotReady("preflight rejected"))
        let scope = StorageScope(url: root)
        let run = await DrainedRun.drain(
            try makeRunner(state: state).start(request(scope: scope)))

        XCTAssertEqual(
            run.failure as? RunError,
            .artifactNotReady("preflight rejected"))
        XCTAssertNil(run.acceptance)
        XCTAssertTrue(run.events.isEmpty)
        XCTAssertEqual(state.prepareCount, 1)
        XCTAssertEqual(state.shutdownCount, 0)
        XCTAssertTrue(scope.isReleased)
    }

    func testAdmittedExpertPoolMustFitBeforeAcceptance() async throws {
        let state = V4FakeState(expertPoolBudgetBytes: 33)
        let scope = StorageScope(url: root)
        // The pool is the only reservation under test, so the MLX cache is
        // stated at 0 rather than left to the budget-linked stated default.
        var knobs = RunKnobs()
        knobs.mlxCacheLimitBytes = 0
        let run = await DrainedRun.drain(
            try makeRunner(state: state, minimumBudgetBytes: 1).start(
                request(scope: scope, budgetBytes: 32, knobs: knobs)))

        XCTAssertEqual(
            run.failure as? RunError,
            .budgetBelowMinimum(
                declared: 32, minimum: 33, model: .deepseekV4Flash))
        XCTAssertNil(run.acceptance)
        XCTAssertEqual(state.shutdownCount, 1)
        XCTAssertTrue(scope.isReleased)
    }

    func testExplicitMLXCacheAndExpertPoolMustTogetherFitBeforeAcceptance() async throws {
        let state = V4FakeState(expertPoolBudgetBytes: 17)
        var knobs = RunKnobs()
        knobs.mlxCacheLimitBytes = 16
        let run = await DrainedRun.drain(
            try makeRunner(state: state, minimumBudgetBytes: 1).start(
                request(budgetBytes: 32, knobs: knobs)))

        XCTAssertEqual(
            run.failure as? RunError,
            .budgetBelowMinimum(
                declared: 32, minimum: 33, model: .deepseekV4Flash))
        XCTAssertNil(run.acceptance)
        XCTAssertEqual(state.shutdownCount, 1)
    }

    func testMeasuredFootprintBreakingTheBudgetFailsInsteadOfLookingStopped() async throws {
        let state = V4FakeState()
        let run = await DrainedRun.drain(
            try makeRunner(state: state, minimumBudgetBytes: 1).start(
                request(budgetBytes: 1)))

        guard let error = run.failure as? RunError,
            case .budgetExceeded(let peak, let declared) = error
        else {
            return XCTFail("expected budgetExceeded, got \(String(describing: run.failure))")
        }
        XCTAssertGreaterThan(peak, 1)
        XCTAssertEqual(declared, 1)
        XCTAssertNotNil(run.acceptance)
        XCTAssertNil(run.cancelled)
        XCTAssertEqual(state.shutdownCount, 1)
    }

    func testInvalidLayerProgressIsAnArtifactFailureNotAUserStop() async throws {
        let state = V4FakeState(invalidProgress: true)
        let run = await DrainedRun.drain(
            try makeRunner(state: state).start(request()))

        guard let error = run.failure as? RunError,
            case .artifactNotReady(let reason) = error
        else {
            return XCTFail("expected artifactNotReady, got \(String(describing: run.failure))")
        }
        XCTAssertTrue(reason.contains("layer progress"), reason)
        XCTAssertNil(run.cancelled)
        XCTAssertEqual(state.shutdownCount, 1)
    }

    func testInvalidLogitsCannotBecomeASuccessfulToken() async throws {
        let state = V4FakeState(
            steps: [
                .init(tokenID: 1, logits: [0, 2, 1], completedLayers: 3)
            ])
        let run = await DrainedRun.drain(
            try makeRunner(state: state).start(request()))

        guard let error = run.failure as? RunError,
            case .artifactNotReady(let reason) = error
        else {
            return XCTFail("expected artifactNotReady, got \(String(describing: run.failure))")
        }
        XCTAssertTrue(reason.contains("logits"), reason)
        XCTAssertNotNil(run.acceptance)
        XCTAssertTrue(run.tokens.isEmpty)
        XCTAssertNil(run.finished)
        XCTAssertNil(run.cancelled)
        XCTAssertEqual(state.shutdownCount, 1)
    }

    func testTerminalBindingFailureCannotPublishFinished() async throws {
        let state = V4FakeState(
            terminalError: .artifactNotReady("artifact changed during generation"))
        let scope = StorageScope(url: root)
        let run = await DrainedRun.drain(
            try makeRunner(state: state).start(
                request(scope: scope, maximumNewTokens: 1)))

        XCTAssertEqual(
            run.failure as? RunError,
            .artifactNotReady("artifact changed during generation"))
        XCTAssertNotNil(run.acceptance)
        XCTAssertNil(run.finished)
        XCTAssertNil(run.cancelled)
        XCTAssertEqual(state.terminalValidationCount, 1)
        XCTAssertEqual(state.shutdownCount, 1)
        XCTAssertTrue(scope.isReleased)
    }

    func testPositionLimitIsRefusedAfterPreflightButBeforeAcceptance() async throws {
        let state = V4FakeState(maximumPositionCount: 4)
        let run = await DrainedRun.drain(
            try makeRunner(state: state).start(
                request(prompt: .tokenIDs([1, 2]), maximumNewTokens: 4)))

        guard let error = run.failure as? RunError,
            case .promptNotSupported(let reason) = error
        else {
            return XCTFail("expected promptNotSupported, got \(String(describing: run.failure))")
        }
        XCTAssertTrue(reason.contains("position limit"), reason)
        XCTAssertNil(run.acceptance)
        XCTAssertEqual(state.prepareCount, 1)
        XCTAssertEqual(state.shutdownCount, 1)
    }

    private func makeRunner(
        state: V4FakeState,
        minimumBudgetBytes: UInt64 = 1 << 30,
        maximumNewTokens: Int = 8
    ) -> DeepSeekV4DecodeRunner {
        DeepSeekV4DecodeRunner(
            scale: .stated(
                minimumBudgetBytes: minimumBudgetBytes,
                maximumNewTokens: maximumNewTokens),
            factory: V4FakeFactory(state: state))
    }

    private func request(
        prompt: Prompt = .tokenIDs([1, 2]),
        scope: StorageScope? = nil,
        budgetBytes: UInt64 = 64 << 30,
        maximumNewTokens: Int = 4,
        knobs: RunKnobs = RunKnobs()
    ) -> RunRequest {
        RunRequest(
            model: .deepseekV4Flash,
            artifact: ArtifactReference(root: root, scope: scope),
            prompt: prompt,
            memoryBudgetBytes: budgetBytes,
            maximumNewTokens: maximumNewTokens,
            knobs: knobs,
            workingDirectory: working)
    }
}

private struct V4FakeFactory: DeepSeekV4RunWorkloadFactory {
    let state: V4FakeState
    let requiresRuntimeAuthority = false

    func prepare(
        request: RunRequest, knobs: DeepSeekV4EffectiveKnobs,
        cancellation: DeepSeekV4RunCancellation
    ) throws -> any DeepSeekV4RunWorkload {
        try cancellation.check()
        return try state.prepareWorkload()
    }
}

private final class V4FakeWorkload: DeepSeekV4RunWorkload {
    private let state: V4FakeState

    var layerCount: Int { state.layerCount }
    var vocabularySize: Int { state.vocabularySize }
    var eosTokenID: Int { state.eosTokenID }
    var maximumPositionCount: Int { state.maximumPositionCount }
    var sourceRepository: String { "deepseek-ai/DeepSeek-V4" }
    var sourceRevision: String { "0123456789abcdef" }
    var expertPoolBudgetBytes: UInt64 { state.expertPoolBudgetBytes }
    var readAccountingSnapshot: DeepSeekV4ReadAccountingSnapshot? {
        state.readAccountingSnapshot
    }
    var phaseMetricsSnapshot: DeepSeekV4PhaseMetrics? {
        state.phaseMetricsSnapshot
    }

    init(state: V4FakeState) { self.state = state }

    func productMemoryPlan(
        declaredBudgetBytes: UInt64,
        promptTokenCount: Int,
        maximumNewTokens: Int,
        mlxCacheBytes: UInt64,
        pinnedDeterministicBytes: UInt64,
        enforcesProductLimits: Bool
    ) throws -> DeepSeekV4ProductMemoryPlan {
        let exact = state.productMemoryRequirementBytes.addingReportingOverflow(
            pinnedDeterministicBytes)
        let required = exact.overflow ? .max : exact.partialValue
        return DeepSeekV4ProductMemoryPlan(
            declaredBudgetBytes: declaredBudgetBytes,
            transientExecutionBytes: 0,
            retainedStateBytes: 0,
            replacementLayerStateBytes: 0,
            expertPoolBytes: 0,
            mlxCacheBytes: 0,
            pinnedDeterministicBytes: pinnedDeterministicBytes,
            productFloorPaddingBytes: state.productMemoryRequirementBytes,
            requiredBudgetBytes: required,
            headroomBytes: declaredBudgetBytes >= required
                ? declaredBudgetBytes - required : 0)
    }

    func prefill(
        tokenIDs: [Int], decodePositionLimit: Int,
        cancellationCheck: () throws -> Void,
        onLayerCompleted: (Int, Int) -> Void
    ) throws -> DeepSeekV4WorkloadStep {
        state.markPrefillEntered()
        if state.blockPrefill {
            while true {
                try cancellationCheck()
                state.waitBriefly()
            }
        }
        return try perform(
            cancellationCheck: cancellationCheck,
            onLayerCompleted: onLayerCompleted)
    }

    func decode(
        tokenID: Int, cancellationCheck: () throws -> Void,
        onLayerCompleted: (Int, Int) -> Void
    ) throws -> DeepSeekV4WorkloadStep {
        try perform(
            cancellationCheck: cancellationCheck,
            onLayerCompleted: onLayerCompleted)
    }

    func validateTerminalBinding() throws { try state.validateTerminal() }

    func shutdown() { state.didShutdown() }

    private func perform(
        cancellationCheck: () throws -> Void,
        onLayerCompleted: (Int, Int) -> Void
    ) throws -> DeepSeekV4WorkloadStep {
        // The synthetic decomposition is charged against the time this call
        // actually took, which the engine's pass bracket strictly contains. A
        // fabricated constant would eventually exceed a fast pass and be
        // refused by the engine for the right reason at the wrong time.
        let entered = MonotonicClock.now()
        defer { state.recordSyntheticPhases(seconds: MonotonicClock.seconds(since: entered)) }
        try cancellationCheck()
        if state.invalidProgress {
            onLayerCompleted(0, layerCount)
            try cancellationCheck()
        }
        for layer in 1...layerCount {
            try cancellationCheck()
            onLayerCompleted(layer, layerCount)
            try cancellationCheck()
        }
        return try state.nextStep()
    }
}

private final class V4FakeState: @unchecked Sendable {
    let layerCount: Int
    let vocabularySize: Int
    let eosTokenID: Int
    let maximumPositionCount: Int
    let expertPoolBudgetBytes: UInt64
    let productMemoryRequirementBytes: UInt64
    let blockPrefill: Bool
    let invalidProgress: Bool

    private let condition = NSCondition()
    private var steps: [DeepSeekV4WorkloadStep]
    private let prepareError: RunError?
    private var terminalError: RunError?
    private var nextStepIndex = 0
    private var didEnterPrefill = false
    private var prepares = 0
    private var shutdowns = 0
    private var terminalValidations = 0
    private let readAccountingPerStep: [(deterministic: UInt64, expert: UInt64)]?
    private let pinnedServedPerStep: [UInt64]?
    private var deterministicBytesRead: UInt64 = 0
    private var expertBytesRead: UInt64 = 0
    private var pinnedServedBytes: UInt64 = 0
    private var byteAccountingOverflowed = false
    private let reportsPhaseMetrics: Bool
    private var phaseMetrics = DeepSeekV4PhaseMetrics()

    init(
        steps: [DeepSeekV4WorkloadStep] = [
            .init(tokenID: 1, logits: [0, 2, 1, -1], completedLayers: 3)
        ],
        prepareError: RunError? = nil,
        terminalError: RunError? = nil,
        layerCount: Int = 3,
        vocabularySize: Int = 4,
        eosTokenID: Int = 3,
        maximumPositionCount: Int = 128,
        expertPoolBudgetBytes: UInt64 = 1,
        productMemoryRequirementBytes: UInt64 =
            DeepSeekV4ProductMemoryBudget.minimumBudgetBytes,
        blockPrefill: Bool = false,
        invalidProgress: Bool = false,
        readAccountingPerStep: [(UInt64, UInt64)]? = nil,
        pinnedServedPerStep: [UInt64]? = nil,
        reportsPhaseMetrics: Bool = false
    ) {
        self.pinnedServedPerStep = pinnedServedPerStep
        self.reportsPhaseMetrics = reportsPhaseMetrics
        self.steps = steps
        self.prepareError = prepareError
        self.terminalError = terminalError
        self.layerCount = layerCount
        self.vocabularySize = vocabularySize
        self.eosTokenID = eosTokenID
        self.maximumPositionCount = maximumPositionCount
        self.expertPoolBudgetBytes = expertPoolBudgetBytes
        self.productMemoryRequirementBytes = productMemoryRequirementBytes
        self.blockPrefill = blockPrefill
        self.invalidProgress = invalidProgress
        self.readAccountingPerStep = readAccountingPerStep
    }

    var prepareCount: Int { locked { prepares } }
    var shutdownCount: Int { locked { shutdowns } }
    var terminalValidationCount: Int { locked { terminalValidations } }
    var isShutdown: Bool { shutdownCount > 0 }
    var readAccountingSnapshot: DeepSeekV4ReadAccountingSnapshot? {
        locked {
            guard readAccountingPerStep != nil else { return nil }
            return DeepSeekV4ReadAccountingSnapshot(
                deterministicBytesRead: deterministicBytesRead,
                expertBytesRead: expertBytesRead,
                pinnedServedBytes: pinnedServedBytes,
                didOverflow: byteAccountingOverflowed)
        }
    }

    var phaseMetricsSnapshot: DeepSeekV4PhaseMetrics? {
        locked { reportsPhaseMetrics ? phaseMetrics : nil }
    }

    /// Charge fixed fractions of one pass to the phases a real V4 pass moves
    /// through. The fractions sum to 0.70, so the residual is positive and the
    /// engine sees a decomposition it can accept for the same reason a real one
    /// is accepted: the named terms fit inside the measured pass.
    func recordSyntheticPhases(seconds: Double) {
        guard reportsPhaseMetrics, seconds >= 0 else { return }
        locked {
            phaseMetrics.deterministicReadSeconds += seconds * 0.20
            phaseMetrics.deterministicReadCount += layerCount
            phaseMetrics.tileDigestSeconds += seconds * 0.10
            phaseMetrics.tileDigestCount += layerCount
            phaseMetrics.expertPhaseSeconds += seconds * 0.30
            phaseMetrics.expertGatherCallCount += layerCount
            phaseMetrics.expertIOWaitSeconds += seconds * 0.20
            phaseMetrics.expertGatherComputeSeconds += seconds * 0.05
            phaseMetrics.expertAcquireCount += layerCount * 3
            phaseMetrics.outputHeadReadSeconds += seconds * 0.05
            phaseMetrics.outputHeadComputeSeconds += seconds * 0.04
            phaseMetrics.outputHeadWindowCount += 2
            phaseMetrics.reclaimSeconds += seconds * 0.01
            phaseMetrics.reclaimCount += layerCount + 2
        }
    }

    func prepareWorkload() throws -> any DeepSeekV4RunWorkload {
        let failure: RunError? = locked {
            prepares += 1
            return prepareError
        }
        if let failure { throw failure }
        return V4FakeWorkload(state: self)
    }

    func markPrefillEntered() {
        condition.lock()
        didEnterPrefill = true
        condition.broadcast()
        condition.unlock()
    }

    func waitUntilPrefillEntered(timeout: TimeInterval) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        let deadline = Date().addingTimeInterval(timeout)
        while !didEnterPrefill {
            if !condition.wait(until: deadline) { return false }
        }
        return true
    }

    func waitBriefly() {
        condition.lock()
        _ = condition.wait(until: Date().addingTimeInterval(0.005))
        condition.unlock()
    }

    func nextStep() throws -> DeepSeekV4WorkloadStep {
        try locked {
            guard nextStepIndex < steps.count else {
                throw RunError.artifactNotReady("the fake V4 workload ran out of steps")
            }
            if let bytes = readAccountingPerStep?[nextStepIndex] {
                let deterministic = deterministicBytesRead.addingReportingOverflow(
                    bytes.deterministic)
                deterministicBytesRead = deterministic.overflow
                    ? .max : deterministic.partialValue
                let expert = expertBytesRead.addingReportingOverflow(bytes.expert)
                expertBytesRead = expert.overflow ? .max : expert.partialValue
                byteAccountingOverflowed = byteAccountingOverflowed
                    || deterministic.overflow || expert.overflow
            }
            if let served = pinnedServedPerStep?[nextStepIndex] {
                let total = pinnedServedBytes.addingReportingOverflow(served)
                pinnedServedBytes = total.overflow ? .max : total.partialValue
                byteAccountingOverflowed = byteAccountingOverflowed || total.overflow
            }
            defer { nextStepIndex += 1 }
            return steps[nextStepIndex]
        }
    }

    func validateTerminal() throws {
        let failure: RunError? = locked {
            terminalValidations += 1
            return terminalError
        }
        if let failure { throw failure }
    }

    func waitUntilShutdown(timeout: TimeInterval) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        let deadline = Date().addingTimeInterval(timeout)
        while shutdowns == 0 {
            if !condition.wait(until: deadline) { return false }
        }
        return true
    }

    func didShutdown() {
        condition.lock()
        shutdowns += 1
        condition.broadcast()
        condition.unlock()
    }

    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        condition.lock()
        defer { condition.unlock() }
        return try body()
    }
}
