import BenchScenarios
import Foundation
import MinirunKit
import MinirunRunners
import StorageCore
import XCTest

@testable import ModelAdapters

/// What the runner refuses, and whether it refuses *before* a byte is read.
///
/// Every case here could have been a clamp instead — a budget rounded up to the
/// minimum, an unsupported knob dropped, a prompt tokenized by a guess — and
/// every one of those would produce a run whose record describes an experiment
/// nobody asked for. So the refusals are named, they carry the offending value,
/// and `validate` reaches all of them without opening the artifact.
final class K3DecodeRunnerRefusalTests: XCTestCase {

    private var root: URL!
    private var working: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-refusal-\(UUID().uuidString)")
        working = root.appendingPathComponent("work")
        try FileManager.default.createDirectory(at: working, withIntermediateDirectories: true)
        // Enough of an artifact to get past the layout check: the refusals below
        // are about the request, and a layout failure would mask them.
        try SyntheticArtifact.write(root: root, shape: SyntheticArtifact.Shape(layers: 1))
    }

    override func tearDownWithError() throws {
        if let root { try? FileManager.default.removeItem(at: root) }
        try super.tearDownWithError()
    }

    private func request(
        budgetBytes: UInt64 = K3RunFixture.declaredBudgetBytes, newTokens: Int = 1,
        prompt: Prompt = .tokenIDs([1, 2]), knobs: RunKnobs = RunKnobs(),
        model: ModelID = .kimiK3, rootOverride: URL? = nil, pinPlan: PinPlan? = nil
    ) -> RunRequest {
        RunRequest(
            model: model,
            artifact: ArtifactReference(root: rootOverride ?? root),
            prompt: prompt, memoryBudgetBytes: budgetBytes, pinPlan: pinPlan,
            maximumNewTokens: newTokens,
            knobs: knobs, workingDirectory: working)
    }

    private func flagshipPlan(budget: UInt64) throws -> PinPlan {
        let census = try ArtifactCensus.measuredFlagship()
        return try MemoryDialPlanner.plan(
            budgetBytes: budget, census: census,
            floor: WorkingSetFloor.measuredFlagship(census))
    }

    /// The refusal that matters most: a budget below the smallest one this model
    /// has ever run under. It is not clamped up to the minimum, because an
    /// unmeasured budget is not a statement (spec §5.3).
    func testABudgetBelowTheMinimumIsRefusedRatherThanClamped() throws {
        let runner = K3RunFixture.runner()
        XCTAssertThrowsError(try runner.validate(request(budgetBytes: 1 << 20))) { error in
            guard case RunError.budgetBelowMinimum(let declared, let minimum, let model) = error
            else { return XCTFail("\(error)") }
            XCTAssertEqual(declared, 1 << 20)
            XCTAssertEqual(minimum, K3RunFixture.minimumBudgetBytes)
            XCTAssertEqual(model, .kimiK3)
        }
        // And `start` refuses on the same path, so a caller who skipped
        // `validate` does not get a stream that fails later.
        XCTAssertThrowsError(try runner.start(request(budgetBytes: 1 << 20)))
    }

    /// The historical device scenario remains a one-token record. The product
    /// runner cannot borrow it for a 64-token chat whose second pass retains the
    /// full rolling state.
    func testTheFlagshipProductFloorIsSeparateFromTheOneTokenScenario() throws {
        let flagship = K3DecodeRunner()
        XCTAssertEqual(K3DecodeScenario.defaultBudgetBytes, 5_800_000_000)
        XCTAssertEqual(
            flagship.capabilities.minimumBudgetBytes,
            K3ProductMemoryBudget.minimumBudgetBytes)
        XCTAssertEqual(
            flagship.capabilities.maximumNewTokens,
            K3ProductMemoryBudget.maximumNewTokens)
        XCTAssertFalse(flagship.capabilities.acceptsTextPrompts)
        XCTAssertTrue(flagship.capabilities.requiresMLX)
        XCTAssertTrue(flagship.capabilities.supportsPinPlan)
        XCTAssertEqual(flagship.capabilities.layout, .k3FlagshipLayerStreams)
        XCTAssertEqual(flagship.capabilities.tokenEventGranularity, .perPass)
        XCTAssertThrowsError(
            try flagship.validate(
                request(budgetBytes: K3ProductMemoryBudget.minimumBudgetBytes - 1))
        ) { error in
            guard case RunError.budgetBelowMinimum = error else { return XCTFail("\(error)") }
        }
    }

    func testIOSExperimentalPolicyIsBoundedWithoutWeakeningTheMacPolicy() throws {
        let mac = K3ProductMemoryBudget.macOSProductPolicy
        XCTAssertEqual(mac.minimumBudgetBytes, 8_000_000_000)
        XCTAssertEqual(mac.runtimeAndAllocatorReserveBytes, 2_100_000_000)
        XCTAssertEqual(mac.maximumNewTokens, 64)
        XCTAssertFalse(mac.isExperimental)

        let phone = K3ProductMemoryBudget.iOSExperimentalPolicy
        XCTAssertEqual(phone.minimumBudgetBytes, 5_800_000_000)
        XCTAssertEqual(phone.runtimeAndAllocatorReserveBytes, 500_000_000)
        XCTAssertEqual(phone.maximumNewTokens, 2)
        XCTAssertTrue(phone.isExperimental)

        let state = try K3GenerationStateMemory(config: K3DecodeScenario.flagshipConfig())
        let reserve = try K3ProductMemoryBudget.defaultWorkingReserveBytes(
            state: state, policy: phone)
        XCTAssertEqual(reserve, 981_886_208)

        let census = try ArtifactCensus.measuredFlagship()
        let exactFloor = try XCTUnwrap(
            UInt64Accounting.checkedSum([
                census.layerResidentBytes.max() ?? 0,
                46_792_704,
                reserve,
            ]))
        XCTAssertEqual(exactFloor, 5_711_193_344)
        XCTAssertLessThan(exactFloor, phone.minimumBudgetBytes)

        let fixedStateBeforeAllocatorHeadroom = try XCTUnwrap(
            UInt64Accounting.checkedSum([
                census.layerResidentBytes.max() ?? 0,
                46_792_704,
                K3ProductMemoryBudget.flagshipDefaultGenerationStateBytes,
            ]))
        XCTAssertGreaterThan(fixedStateBeforeAllocatorHeadroom, 5_000_000_000)
    }

    func testFlagshipRollingStateMakesTheSecondPassAn8GBProductFloor() throws {
        let config = try K3DecodeScenario.flagshipConfig()
        let state = try K3GenerationStateMemory(config: config)
        XCTAssertEqual(state.kdaBytes, 474_808_320)
        XCTAssertEqual(state.mlaBytesPerCachedPosition, 55_296)
        XCTAssertEqual(
            try state.bytes(cachedPositions: 128),
            481_886_208)
        XCTAssertEqual(
            try K3ProductMemoryBudget.defaultWorkingReserveBytes(state: state),
            K3ProductMemoryBudget.flagshipDefaultWorkingReserveBytes)

        let census = try ArtifactCensus.measuredFlagship()
        let exactFloor = try XCTUnwrap(
            UInt64Accounting.checkedSum([
                census.layerResidentBytes.max() ?? 0,
                46_792_704,
                K3ProductMemoryBudget.flagshipDefaultWorkingReserveBytes,
            ]))
        XCTAssertEqual(exactFloor, 7_311_193_344)
        XCTAssertGreaterThanOrEqual(K3ProductMemoryBudget.minimumBudgetBytes, exactFloor)
    }

    func testLongerPromptsGrowTheProductReserveWithoutOverflowOrClamping() throws {
        let state = try K3GenerationStateMemory(config: K3DecodeScenario.flagshipConfig())
        let baseline = try K3ProductMemoryBudget.workingReserveBytes(
            state: state, promptTokenCount: 1, maximumNewTokens: 64)
        let long = try K3ProductMemoryBudget.workingReserveBytes(
            state: state, promptTokenCount: 512, maximumNewTokens: 64)
        XCTAssertEqual(baseline, K3ProductMemoryBudget.flagshipDefaultWorkingReserveBytes)
        XCTAssertEqual(long - baseline, 447 * state.mlaBytesPerCachedPosition)
    }

    func testProjectedGenerationStateOverflowIsRefusedByName() throws {
        let state = try K3GenerationStateMemory(config: K3DecodeScenario.flagshipConfig())
        XCTAssertThrowsError(
            try K3ProductMemoryBudget.cachedPositions(
                promptTokenCount: Int.max, maximumNewTokens: 2))
        XCTAssertThrowsError(try state.bytes(cachedPositions: Int.max))
    }

    func testATextPromptIsRefusedRatherThanTokenizedByAGuess() throws {
        XCTAssertThrowsError(
            try K3RunFixture.runner().validate(request(prompt: .text("中国的首都是")))
        ) { error in
            guard case RunError.promptNotSupported(let reason) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertTrue(reason.contains("vocabulary"), reason)
        }
        XCTAssertThrowsError(
            try K3RunFixture.runner().validate(request(prompt: .tokenIDs([])))
        ) { error in
            guard case RunError.promptNotSupported = error else { return XCTFail("\(error)") }
        }
    }

    func testMoreTokensThanTheCeilingAreRefusedByCount() throws {
        let runner = K3RunFixture.runner(maximumNewTokens: 2)
        XCTAssertNoThrow(try runner.validate(request(newTokens: 2)))
        XCTAssertThrowsError(try runner.validate(request(newTokens: 3))) { error in
            guard case RunError.tooManyTokens(let requested, let maximum, _) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertEqual(requested, 3)
            XCTAssertEqual(maximum, 2)
        }
        XCTAssertThrowsError(try runner.validate(request(newTokens: 0))) { error in
            guard case RunError.knobOutOfRange(let name, _, _) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertEqual(name, "maximumNewTokens")
        }
    }

    /// A knob this runner cannot honour is refused **by name**. Dropping it
    /// would produce a run quoted in a record as though the knob had been
    /// honoured.
    func testAnUnsupportableKnobIsRefusedByName() throws {
        var knobs = RunKnobs()
        knobs.expertPoolSlots = 8
        XCTAssertThrowsError(try K3RunFixture.runner().validate(request(knobs: knobs))) { error in
            guard case RunError.knobNotSupported(let name, let model) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertEqual(name, "expertPoolSlots")
            XCTAssertEqual(model, .kimiK3)
        }

        var margins = RunKnobs()
        margins.captureRoutingMargins = true
        XCTAssertThrowsError(try K3RunFixture.runner().validate(request(knobs: margins)))
    }

    func testAKnobOutOfRangeCarriesTheValueAndTheRange() throws {
        var knobs = RunKnobs()
        knobs.logitChunkRows = 0
        XCTAssertThrowsError(try K3RunFixture.runner().validate(request(knobs: knobs))) { error in
            guard case RunError.knobOutOfRange(let name, let value, let allowed) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertEqual(name, "logitChunkRows")
            XCTAssertEqual(value, "0")
            XCTAssertFalse(allowed.isEmpty)
        }
    }

    func testDeterministicReadAheadAboveOneIsRefusedBeforeArtifactResolution() throws {
        var knobs = RunKnobs()
        knobs.deterministicReadAheadLayers = 2
        let missing = root.appendingPathComponent("not-an-artifact")

        XCTAssertThrowsError(
            try K3RunFixture.runner().validate(
                request(knobs: knobs, rootOverride: missing))
        ) { error in
            guard case RunError.knobOutOfRange(let name, let value, let allowed) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertEqual(name, "deterministicReadAheadLayers")
            XCTAssertEqual(value, "2")
            XCTAssertTrue(allowed.contains("0"))
            XCTAssertTrue(allowed.contains("1"))
        }
    }

    func testAPlanForAnotherBudgetIsRefusedBeforeArtifactResolution() throws {
        let planBudget: UInt64 = 6_000_000_000
        let plan = try flagshipPlan(budget: planBudget)
        let missing = root.appendingPathComponent("not-an-artifact")
        XCTAssertThrowsError(
            try K3RunFixture.runner().validate(
                request(
                    budgetBytes: K3RunFixture.declaredBudgetBytes,
                    rootOverride: missing, pinPlan: plan))
        ) { error in
            XCTAssertEqual(
                error as? RunError,
                .pinPlanBudgetMismatch(
                    planBytes: planBudget,
                    declaredBytes: K3RunFixture.declaredBudgetBytes))
        }
    }

    func testAPlanReserveCannotDisagreeWithTheRunKnob() throws {
        let plan = try flagshipPlan(budget: K3RunFixture.declaredBudgetBytes)
        var knobs = RunKnobs()
        knobs.readAheadReserveBytes = plan.readAheadReserveBytes + 1
        XCTAssertThrowsError(
            try K3RunFixture.runner().validate(request(knobs: knobs, pinPlan: plan))
        ) { error in
            guard case RunError.knobOutOfRange(let name, let value, let allowed) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertEqual(name, "readAheadReserveBytes")
            XCTAssertEqual(value, "\(plan.readAheadReserveBytes + 1)")
            XCTAssertTrue(allowed.contains("memory plan"), allowed)
        }
    }

    func testAPlanForAnotherResponseHorizonIsRefusedBeforeArtifactReads() throws {
        let plan = try flagshipPlan(budget: K3RunFixture.declaredBudgetBytes)
        XCTAssertEqual(plan.requestedMaximumNewTokens, 1)

        XCTAssertThrowsError(
            try K3RunFixture.runner().validate(request(newTokens: 2, pinPlan: plan))
        ) { error in
            guard case RunError.pinPlanInvalid(let reason) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertTrue(reason.contains("1 new tokens"), reason)
            XCTAssertTrue(reason.contains("requests 2"), reason)
        }
    }

    func testAPlanCannotBeRunUnderAnotherReadAheadArm() throws {
        let plan = try flagshipPlan(budget: K3RunFixture.declaredBudgetBytes)
        XCTAssertEqual(plan.readAheadDepth, 1)
        var knobs = RunKnobs()
        knobs.deterministicReadAheadLayers = 0

        XCTAssertThrowsError(
            try K3RunFixture.runner().start(request(knobs: knobs, pinPlan: plan))
        ) { error in
            guard case RunError.pinPlanInvalid(let reason) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertTrue(reason.contains("read-ahead depth"), reason)
        }
    }

    func testAPlanWithAnUnimplementedTierIsRefusedRatherThanPartlyHonoured() throws {
        let census = try ArtifactCensus.measuredFlagship()
        let floor = WorkingSetFloor.measuredFlagship(census)
        let globalsBudget = try XCTUnwrap(
            MemoryDialPlanner.snapPoints(census: census, floor: floor).last)
        let plan = try MemoryDialPlanner.plan(
            budgetBytes: globalsBudget, census: census, floor: floor)
        XCTAssertNotNil(plan.pinnedGlobals)

        XCTAssertThrowsError(
            try K3RunFixture.runner().validate(
                request(budgetBytes: globalsBudget, pinPlan: plan))
        ) { error in
            XCTAssertEqual(error as? RunError, .pinPlanUnitUnsupported("globals"))
        }
    }

    /// A root that is not this layout is named as such before anything opens it.
    func testARootThatIsNotAK3ArtifactIsRefusedByLayout() throws {
        let empty = root.appendingPathComponent("empty")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        XCTAssertThrowsError(
            try K3RunFixture.runner().validate(request(rootOverride: empty))
        ) { error in
            guard case RunError.artifactLayoutMismatch(let expected, let path) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertEqual(expected, .k3FlagshipLayerStreams)
            XCTAssertEqual(path, empty.path)
        }

        let missing = root.appendingPathComponent("nowhere")
        XCTAssertThrowsError(
            try K3RunFixture.runner().validate(request(rootOverride: missing))
        ) { error in
            guard case RunError.artifactNotReady = error else { return XCTFail("\(error)") }
        }
    }

    /// A released scope is refused rather than read through: the grant is the
    /// operator's, and a run that proceeded without it would be reading on
    /// borrowed access it no longer has.
    func testAReleasedScopeIsRefused() throws {
        let scope = StorageScope(url: root)
        scope.release()
        let request = RunRequest(
            model: .kimiK3, artifact: ArtifactReference(root: root, scope: scope),
            prompt: .tokenIDs([1]), memoryBudgetBytes: K3RunFixture.declaredBudgetBytes,
            maximumNewTokens: 1, workingDirectory: working)
        XCTAssertThrowsError(try K3RunFixture.runner().validate(request)) { error in
            guard case RunError.scopeRefused(let path) = error else { return XCTFail("\(error)") }
            XCTAssertEqual(path, root.path)
        }
    }

    func testAnotherModelsRequestIsRefusedRatherThanDecodedAnyway() throws {
        XCTAssertThrowsError(
            try K3RunFixture.runner().validate(request(model: .deepseekV4Flash))
        ) { error in
            guard case RunError.runnerUnavailable(let model, _) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertEqual(model, .deepseekV4Flash)
        }
    }

    /// The registry says what this build can decode. A model with no runner is
    /// absent, not a stub that fails at `start` — the app is expected to read
    /// this before it offers a Run button.
    func testTheRegistryOnlyClaimsWhatThisBuildCanDecode() {
        XCTAssertNotNil(MinirunRunnerRegistry.runner(for: .kimiK3))
        XCTAssertNil(MinirunRunnerRegistry.runner(for: .deepseekV4Flash))
        XCTAssertNil(MinirunRunnerRegistry.runner(for: .minimaxH3))
        XCTAssertNil(MinirunRunnerRegistry.runner(for: ModelID("future-model")))
        XCTAssertEqual(Array(MinirunRunnerRegistry.all.keys), [.kimiK3])
        XCTAssertEqual(
            MinirunRunnerRegistry.runner(for: .kimiK3)?.capabilities.minimumBudgetBytes,
            K3ProductMemoryBudget.minimumBudgetBytes)
    }
}
