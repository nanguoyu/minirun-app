import Foundation
import XCTest

@testable import MinirunKit

/// The memory dial, checked against the flagship's own numbers.
///
/// Every figure below is transcribed from a record — the layer census from M9C
/// §239 and `2026-08-10-m10-k3-iphone.md` §3, the per-token totals from M9C/M10,
/// the link rates from `2026-08-11-readahead-mac.md` — and the planner is a pure
/// function over them, so this file states what each budget buys without a
/// drive, a checkpoint, a phone or MLX. That is the point of the planner being
/// pure: the table in `docs/design/memory-dial.md` §7 is executable.
final class MemoryDialPlannerTests: XCTestCase {

    // MARK: - Fixtures

    static let census = try! ArtifactCensus.measuredFlagship()
    static let floor = WorkingSetFloor.measuredFlagship(MemoryDialPlannerTests.census)
    static let usb4 = LinkCalibration.usb4Rig2026_08_11

    /// The phone's declared budget (`K3DecodeScenario.defaultBudgetBytes`).
    static let phoneBudget: UInt64 = 5_800_000_000

    static func gib(_ count: UInt64) -> UInt64 { count * 1_073_741_824 }

    func plan(
        _ budget: UInt64,
        readAhead: ReadAheadOption = .depthOne,
        expertHotSet: ExpertHotSetOption = .none,
        calibration: LinkCalibration? = MemoryDialPlannerTests.usb4,
        maximumNewTokens: Int = 1
    ) throws -> PinPlan {
        try MemoryDialPlanner.plan(
            budgetBytes: budget, census: Self.census, floor: Self.floor,
            readAhead: readAhead, expertHotSet: expertHotSet, calibration: calibration,
            maximumNewTokens: maximumNewTokens)
    }

    func pinnedLayers(_ plan: PinPlan) -> [Int] {
        plan.pinnedLayers.compactMap {
            guard case .deterministicLayer(let layer) = $0.unit else { return nil }
            return layer
        }
    }

    // MARK: - The census is a measurement, not a model

    /// The check that makes the layer table a census: 93 blobs reproduce the
    /// published total exactly, and the KDA/MLA split is the one M9C §239
    /// states. Nothing downstream means anything if this line is off by a byte.
    func testLayerCensusReproducesThePublishedTotal() {
        let census = Self.census
        XCTAssertEqual(census.layerStoredBytes.count, 93)
        XCTAssertEqual(census.deterministicStoredBytes, 108_817_924_096)
        XCTAssertEqual(census.layerStoredBytes[0], 2_341_257_216)
        XCTAssertEqual(census.layerStoredBytes.filter { $0 == 1_267_810_304 }.count, 68)
        XCTAssertEqual(census.layerStoredBytes.filter { $0 == 844_398_592 }.count, 24)
        // MLA sits at 0-based {3, 7, …, 91} ∪ {92}; everything else in 1…92 is KDA.
        for layer in [3, 7, 11, 87, 91, 92] {
            XCTAssertEqual(census.layerStoredBytes[layer], 844_398_592, "layer \(layer)")
        }
        for layer in [1, 2, 4, 86, 90] {
            XCTAssertEqual(census.layerStoredBytes[layer], 1_267_810_304, "layer \(layer)")
        }
    }

    /// 92 × 16 × 17,547,264 = 25,829,572,608 per position, and three positions
    /// is exactly the 77,488,717,824 B the flagship prefill recorded.
    func testExpertBytesPerTokenMatchTheRecordedPrefill() {
        XCTAssertEqual(Self.census.expertBytesPerToken, 25_829_572_608)
        XCTAssertEqual(Self.census.expertBytesPerToken * 3, 77_488_717_824)
        XCTAssertEqual(Self.census.measuredBytesPerToken, 136_990_303_232)
    }

    /// The blobs plus the measured globals traffic are the measured
    /// deterministic total to the byte, so nothing is unattributed at flagship
    /// shape. (The 6,003,712 B residual the design document reports is a
    /// different quantity: globals *traffic* against `lm_head`'s file size.)
    func testFlagshipCensusLeavesNothingUnattributed() {
        XCTAssertEqual(Self.census.unattributedBytesPerToken, 0)
        XCTAssertEqual(
            Self.census.deterministicStoredBytes + Self.census.globalsBytesReadPerToken,
            Self.census.measuredDeterministicBytesPerToken)
    }

    /// A census that claims more bytes than the run measured describes a
    /// different artifact, and is refused by name rather than absorbed.
    func testCensusClaimingMoreThanTheRunMeasuredIsRefused() {
        XCTAssertThrowsError(
            try ArtifactCensus(
                layerStoredBytes: [1_000], layerResidentBytes: [2_000],
                globalsStoredBytes: 100, globalsBytesReadPerToken: 100,
                expertStrideBytes: 10, expertsPerLayer: 8, routedExpertsPerToken: 1,
                moeLayerCount: 1, measuredDeterministicBytesPerToken: 900)
        ) { error in
            guard case MemoryDialError.censusDisagreesWithMeasuredTotal(
                let census, let measured, let difference) = error
            else { return XCTFail("expected censusDisagreesWithMeasuredTotal, got \(error)") }
            XCTAssertEqual(census, 1_100)
            XCTAssertEqual(measured, 900)
            XCTAssertEqual(difference, -200)
        }
    }

    /// The other direction is *reported*, not refused: bytes the run read that
    /// no unit explains are real, and they land in the inline bucket because
    /// nothing can pin what it cannot name.
    func testUnexplainedMeasuredBytesAreReportedAndReadInline() throws {
        let census = try ArtifactCensus(
            layerStoredBytes: [1_000, 1_000], layerResidentBytes: [2_000, 2_000],
            globalsStoredBytes: 100, globalsBytesReadPerToken: 100,
            expertStrideBytes: 10, expertsPerLayer: 8, routedExpertsPerToken: 1,
            moeLayerCount: 1, measuredDeterministicBytesPerToken: 2_500)
        XCTAssertEqual(census.unattributedBytesPerToken, 400)
        let floor = WorkingSetFloor(
            widestResidentLayerBytes: 2_000, expertPoolBytes: 0, workingReserveBytes: 0)
        let plan = try MemoryDialPlanner.plan(
            budgetBytes: 2_000, census: census, floor: floor)
        XCTAssertTrue(plan.isAccountingBalanced)
        XCTAssertEqual(
            plan.pinnedBytesPerToken + plan.stagedBytesPerToken + plan.inlineBytesPerToken,
            2_510)
    }

    func testARaggedOrEmptyLayerTableIsRefused() {
        XCTAssertThrowsError(
            try ArtifactCensus(
                layerStoredBytes: [], layerResidentBytes: [],
                globalsStoredBytes: 0, globalsBytesReadPerToken: 0, expertStrideBytes: 0,
                expertsPerLayer: 0, routedExpertsPerToken: 0, moeLayerCount: 0,
                measuredDeterministicBytesPerToken: 0)
        ) { XCTAssertEqual($0 as? MemoryDialError, .layerTableEmpty) }

        XCTAssertThrowsError(
            try ArtifactCensus(
                layerStoredBytes: [1, 2], layerResidentBytes: [2],
                globalsStoredBytes: 0, globalsBytesReadPerToken: 0, expertStrideBytes: 0,
                expertsPerLayer: 0, routedExpertsPerToken: 0, moeLayerCount: 0,
                measuredDeterministicBytesPerToken: 3)
        ) {
            XCTAssertEqual(
                $0 as? MemoryDialError, .layerTableRagged(storedCount: 2, residentCount: 1))
        }
    }

    /// Widening never shrinks a tensor, so a resident size below the stored one
    /// is a broken census and not a very small layer.
    func testResidentSmallerThanStoredIsRefused() {
        XCTAssertThrowsError(
            try ArtifactCensus(
                layerStoredBytes: [1_000, 4_000], layerResidentBytes: [2_000, 3_000],
                globalsStoredBytes: 0, globalsBytesReadPerToken: 0, expertStrideBytes: 0,
                expertsPerLayer: 0, routedExpertsPerToken: 0, moeLayerCount: 0,
                measuredDeterministicBytesPerToken: 5_000)
        ) {
            XCTAssertEqual(
                $0 as? MemoryDialError,
                .residentSmallerThanStored(layer: 1, stored: 4_000, resident: 3_000))
        }
    }

    /// A census is also a wire format. Decoding must not bypass the same
    /// invariants as the public throwing initializer, especially when an
    /// aggregate would otherwise wrap back to a plausible small number.
    func testDecodedCensusCannotBypassValidationOrWrapItsLayerTotal() throws {
        let ragged = UncheckedCensus(
            layerStoredBytes: [1, 2], layerResidentBytes: [2],
            globalsStoredBytes: 0, globalsBytesReadPerToken: 0,
            expertStrideBytes: 0, expertsPerLayer: 0, routedExpertsPerToken: 0,
            moeLayerCount: 0, measuredDeterministicBytesPerToken: 3)
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                ArtifactCensus.self, from: JSONEncoder().encode(ragged))
        ) {
            XCTAssertEqual(
                $0 as? MemoryDialError,
                .layerTableRagged(storedCount: 2, residentCount: 1))
        }

        let overflowing = UncheckedCensus(
            layerStoredBytes: [.max, 1], layerResidentBytes: [.max, 1],
            globalsStoredBytes: 0, globalsBytesReadPerToken: 0,
            expertStrideBytes: 0, expertsPerLayer: 0, routedExpertsPerToken: 0,
            moeLayerCount: 0, measuredDeterministicBytesPerToken: .max)
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                ArtifactCensus.self, from: JSONEncoder().encode(overflowing))
        ) {
            XCTAssertEqual(
                $0 as? MemoryDialError,
                .byteCountUnrepresentable(context: "deterministic census bytes per token"))
        }
    }

    func testInvalidOrUnrepresentableExpertGeometryIsRefused() {
        XCTAssertThrowsError(
            try ArtifactCensus(
                layerStoredBytes: [0], layerResidentBytes: [0],
                globalsStoredBytes: 0, globalsBytesReadPerToken: 0,
                expertStrideBytes: 1, expertsPerLayer: 1, routedExpertsPerToken: 2,
                moeLayerCount: 1, measuredDeterministicBytesPerToken: 0)
        ) {
            XCTAssertEqual(
                $0 as? MemoryDialError,
                .invalidExpertGeometry(
                    expertsPerLayer: 1, routedExpertsPerToken: 2, moeLayerCount: 1))
        }

        XCTAssertThrowsError(
            try ArtifactCensus(
                layerStoredBytes: [0], layerResidentBytes: [0],
                globalsStoredBytes: 0, globalsBytesReadPerToken: 0,
                expertStrideBytes: .max, expertsPerLayer: 2, routedExpertsPerToken: 2,
                moeLayerCount: 1, measuredDeterministicBytesPerToken: 0)
        ) {
            XCTAssertEqual(
                $0 as? MemoryDialError,
                .byteCountUnrepresentable(context: "routed expert bytes per token"))
        }
    }

    /// The public residual remains Int64 for source compatibility, but the plan
    /// must retain the exact UInt64 residual internally or it will report a
    /// balanced set of buckets that is billions of bytes short.
    func testAnUnattributedResidualAboveInt64MaxStillBalancesExactly() throws {
        let residual = UInt64(Int64.max) + 10
        let census = try ArtifactCensus(
            layerStoredBytes: [0], layerResidentBytes: [0],
            globalsStoredBytes: 0, globalsBytesReadPerToken: 0,
            expertStrideBytes: 0, expertsPerLayer: 0, routedExpertsPerToken: 0,
            moeLayerCount: 0, measuredDeterministicBytesPerToken: residual)
        XCTAssertEqual(census.unattributedBytesPerToken, .max)

        let plan = try MemoryDialPlanner.plan(
            budgetBytes: 0, census: census,
            floor: WorkingSetFloor(
                widestResidentLayerBytes: 0, expertPoolBytes: 0, workingReserveBytes: 0),
            readAhead: .off)
        XCTAssertEqual(plan.inlineBytesPerToken, residual)
        XCTAssertEqual(plan.projectedBytesPerTokenRead, residual)
        XCTAssertTrue(plan.isAccountingBalanced)
    }

    // MARK: - The floor

    /// The floor is stated arithmetic — layer 0 widened, the expert pool, the
    /// working reserve — and it predicts the Mac's measured `mlx_peak_bytes`
    /// (5,166,220,288 B, M9C) 1.2% high. Over-predicting is the safe side; a
    /// floor below the measured peak would be a bug.
    func testWorkingFloorIsTheStatedFlagshipArithmetic() {
        XCTAssertEqual(Self.floor.widestResidentLayerBytes, 4_682_514_432)
        XCTAssertEqual(Self.floor.expertPoolBytes, 46_792_704)
        XCTAssertEqual(Self.floor.workingReserveBytes, 500_000_000)
        XCTAssertEqual(Self.floor.totalBytes, 5_229_307_136)
        let measuredPeak = 5_166_220_288.0
        XCTAssertEqual(Double(Self.floor.totalBytes) / measuredPeak, 1.012, accuracy: 0.001)
        XCTAssertGreaterThan(Double(Self.floor.totalBytes), measuredPeak)
    }

    /// `totalBytes` is a conservative display aggregate, but a planner cannot
    /// honour a floor whose exact terms exceed UInt64. Even a UInt64.max budget
    /// is therefore refused rather than accepted as spuriously balanced.
    func testAnOverflowingWorkingFloorSaturatesAndCannotProduceAPlan() {
        let floor = WorkingSetFloor(
            widestResidentLayerBytes: .max, expertPoolBytes: 1, workingReserveBytes: 0)
        XCTAssertEqual(floor.totalBytes, .max)
        XCTAssertThrowsError(
            try MemoryDialPlanner.plan(
                budgetBytes: .max, census: Self.census, floor: floor)
        ) {
            XCTAssertEqual(
                $0 as? MemoryDialError,
                .byteCountUnrepresentable(context: "working-set floor"))
        }
    }

    /// Below the floor the dial refuses, and the error carries all three terms
    /// so the message can say which one the budget failed against. It is never
    /// clamped up to the floor: a budget the user did not declare is a budget
    /// the run discovered.
    func testABudgetBelowTheFloorIsRefusedWithEveryTermNamed() {
        XCTAssertThrowsError(try plan(4_000_000_000)) { error in
            guard case MemoryDialError.budgetBelowWorkingFloor(
                let budget, let floor, let shortfall, let widest, let pool, let reserve) = error
            else { return XCTFail("expected budgetBelowWorkingFloor, got \(error)") }
            XCTAssertEqual(budget, 4_000_000_000)
            XCTAssertEqual(floor, 5_229_307_136)
            XCTAssertEqual(shortfall, 1_229_307_136)
            XCTAssertEqual(widest, 4_682_514_432)
            XCTAssertEqual(pool, 46_792_704)
            XCTAssertEqual(reserve, 500_000_000)
            let text = "\(error)"
            XCTAssertTrue(text.contains("4,682,514,432"), text)
            XCTAssertTrue(text.contains("refused rather than clamped"), text)
        }
        // One byte under is under; the floor itself is a legal budget.
        XCTAssertThrowsError(try plan(5_229_307_135))
        XCTAssertNoThrow(try plan(5_229_307_136))
    }

    // MARK: - The ratio ladder

    /// The tier order is derived from one column, not assumed: every
    /// deterministic layer returns 1.000 bytes saved per resident byte, the
    /// globals bundle 0.499, and the best expert hot set ever measured 0.230.
    /// That is why the globals land *after* all 93 layers and no expert is
    /// pinnable below 114 GB.
    func testTheRatioColumnRanksLayersAboveGlobalsAboveExperts() throws {
        let all = try plan(200_000_000_000, expertHotSet: .measured(
            expertsPerLayer: 8, hitRate: 0.115, traceLabel: "M8 zipf-1.4"))
        XCTAssertEqual(pinnedLayers(all).count, 93)
        for decision in all.pinnedLayers {
            XCTAssertEqual(decision.savedPerResidentByte, 1.0, accuracy: 1e-12)
        }
        let globals = try XCTUnwrap(all.pinnedGlobals)
        XCTAssertEqual(globals.rank, 93)
        XCTAssertEqual(globals.residentBytes, 4_697_669_632)
        XCTAssertEqual(globals.bytesSavedPerToken, 2_342_806_528)
        XCTAssertEqual(globals.savedPerResidentByte, 0.499, accuracy: 0.001)

        let hotSet = try XCTUnwrap(all.expertHotSet)
        XCTAssertEqual(hotSet.rank, 94)
        XCTAssertEqual(hotSet.residentBytes, 12_914_786_304)
        XCTAssertEqual(hotSet.bytesSavedPerToken, 2_970_400_849)
        XCTAssertEqual(hotSet.savedPerResidentByte, 0.230, accuracy: 0.001)
        // 4.3× worse than any layer — the reason expert pinning is capped by
        // measurement rather than by taste.
        XCTAssertEqual(1.0 / hotSet.savedPerResidentByte, 4.3, accuracy: 0.05)
    }

    /// The two other measured hot sets, for the same reason: the ladder's
    /// bottom rung is a measurement with a trace behind it.
    func testTheOtherMeasuredHotSetsScoreWhereTheRecordSaysTheyDo() throws {
        for (experts, hitRate, resident, saved, ratio) in [
            (32, 0.417, UInt64(51_659_145_216), UInt64(10_770_931_777), 0.209),
            (32, 0.263, UInt64(51_659_145_216), UInt64(6_793_177_595), 0.132),
        ] {
            let plan = try self.plan(
                250_000_000_000,
                expertHotSet: .measured(
                    expertsPerLayer: experts, hitRate: hitRate, traceLabel: "M8"))
            let hotSet = try XCTUnwrap(plan.expertHotSet)
            XCTAssertEqual(hotSet.residentBytes, resident)
            XCTAssertEqual(hotSet.bytesSavedPerToken, saved)
            XCTAssertEqual(hotSet.savedPerResidentByte, ratio, accuracy: 0.001)
        }
    }

    /// A hot set without a measured hit rate and a named trace is refused. M8
    /// measured LRU as worse than no cache under read-ahead scan pollution, so
    /// "some experts, probably" is not an admissible plan.
    func testAnUnmeasuredHotSetIsRefused() {
        for option: ExpertHotSetOption in [
            .measured(expertsPerLayer: 8, hitRate: 0.115, traceLabel: "   "),
            .measured(expertsPerLayer: 8, hitRate: 0, traceLabel: "M8"),
            .measured(expertsPerLayer: 8, hitRate: 1.5, traceLabel: "M8"),
            .measured(expertsPerLayer: 0, hitRate: 0.115, traceLabel: "M8"),
            .measured(expertsPerLayer: 897, hitRate: 0.115, traceLabel: "M8"),
        ] {
            XCTAssertThrowsError(try plan(Self.gib(200), expertHotSet: option)) { error in
                guard case MemoryDialError.expertHotSetWithoutMeasuredTrace = error else {
                    return XCTFail("expected expertHotSetWithoutMeasuredTrace, got \(error)")
                }
            }
        }
    }

    /// `Double(UInt64.max)` rounds to 2^64, so converting the product back for
    /// a 100% measured hit rate used to trap. The exact endpoint needs no
    /// floating-point conversion at all.
    func testAFullHitRateAtUInt64MaxDoesNotTrapOrLoseAccounting() throws {
        let census = try ArtifactCensus(
            layerStoredBytes: [0], layerResidentBytes: [0],
            globalsStoredBytes: 0, globalsBytesReadPerToken: 0,
            expertStrideBytes: .max, expertsPerLayer: 1, routedExpertsPerToken: 1,
            moeLayerCount: 1, measuredDeterministicBytesPerToken: 0)
        let plan = try MemoryDialPlanner.plan(
            budgetBytes: .max, census: census,
            floor: WorkingSetFloor(
                widestResidentLayerBytes: 0, expertPoolBytes: 0, workingReserveBytes: 0),
            readAhead: .off,
            expertHotSet: .measured(expertsPerLayer: 1, hitRate: 1, traceLabel: "edge"))
        XCTAssertEqual(plan.expertHotSetBytes, .max)
        XCTAssertEqual(plan.projectedBytesPerTokenSaved, .max)
        XCTAssertEqual(plan.projectedBytesPerTokenRead, 0)
        XCTAssertTrue(plan.isAccountingBalanced)
        XCTAssertTrue(plan.isResidencyBalanced)
    }

    func testAnUnrepresentableExpertHotSetIsRefusedInsteadOfWrappedSmall() throws {
        let census = try ArtifactCensus(
            layerStoredBytes: [0], layerResidentBytes: [0],
            globalsStoredBytes: 0, globalsBytesReadPerToken: 0,
            expertStrideBytes: .max, expertsPerLayer: 2, routedExpertsPerToken: 0,
            moeLayerCount: 2, measuredDeterministicBytesPerToken: 0)
        XCTAssertThrowsError(
            try MemoryDialPlanner.plan(
                budgetBytes: .max, census: census,
                floor: WorkingSetFloor(
                    widestResidentLayerBytes: 0, expertPoolBytes: 0,
                    workingReserveBytes: 0),
                expertHotSet: .measured(
                    expertsPerLayer: 2, hitRate: 1, traceLabel: "edge"))
        ) {
            XCTAssertEqual(
                $0 as? MemoryDialError,
                .byteCountUnrepresentable(context: "expert hot-set residency"))
        }
    }

    /// No budget below 114,047,231,232 B pins a single expert, because every
    /// deterministic layer outranks the best measured hot set by 4.3×.
    func testNoExpertIsPinnedBelowTheAllLayersBudget() throws {
        let hotSet = ExpertHotSetOption.measured(
            expertsPerLayer: 8, hitRate: 0.115, traceLabel: "M8 zipf-1.4")
        for budget in [Self.phoneBudget, Self.gib(8), Self.gib(24), Self.gib(64),
                       114_047_231_231] {
            let plan = try self.plan(budget, expertHotSet: hotSet)
            XCTAssertNil(plan.expertHotSet, "budget \(budget)")
            XCTAssertEqual(plan.expertHotSetBytes, 0)
        }
    }

    // MARK: - The §7 table, row by row

    /// The phone pins nothing, and says exactly why: 570,692,864 B of headroom
    /// against a smallest pinnable unit of 844,398,592 B. It also has no
    /// calibration, so it prints no seconds — not zero, not "unknown".
    func testPhoneBudgetPinsNothingAndStatesTheShortfall() throws {
        let plan = try self.plan(Self.phoneBudget, calibration: nil)
        XCTAssertTrue(plan.pinnedLayers.isEmpty)
        XCTAssertNil(plan.pinnedGlobals)
        XCTAssertEqual(plan.pinnedBytes, 0)
        XCTAssertEqual(plan.unusedBudgetBytes, 570_692_864)
        XCTAssertEqual(plan.projectedBytesPerTokenSaved, 0)
        XCTAssertEqual(plan.projectedBytesPerTokenRead, 136_990_303_232)
        XCTAssertEqual(plan.projectedStagedLayerCount, 91)
        XCTAssertEqual(plan.projectedInlineLayerCount, 2)
        XCTAssertNil(plan.projectedTokenSeconds)
        XCTAssertNil(plan.calibrationLabel)

        // The first pin is an MLA layer, short by 273,705,728 B, so it would
        // need a declared budget of 6,073,705,728 B — 94.3% of what
        // os_proc_available_memory reports. Offered as a warning, not advice.
        XCTAssertEqual(plan.nextPinBudgetBytes, 6_073_705_728)
        XCTAssertEqual(plan.nextPinUnit, .deterministicLayer(3))
        XCTAssertEqual(
            Double(6_073_705_728) / Double(6_442_450_944), 0.943, accuracy: 0.001)
    }

    /// The phone's one refused read-ahead pair, from the same budget: staging
    /// layer 1 behind layer 0 needs 697,117,440 B more than the phone declared,
    /// so layer 1 is read inline and 91 of 92 pairs stage.
    func testPhoneBudgetRefusesExactlyTheLayerZeroPair() throws {
        let plan = try self.plan(Self.phoneBudget, calibration: nil)
        XCTAssertEqual(plan.projectedRefusedPairCount, 1)
        let refused = try XCTUnwrap(plan.refusedPairs.first)
        XCTAssertEqual(refused.residentLayer, 0)
        XCTAssertEqual(refused.stagedLayer, 1)
        XCTAssertEqual(refused.requiredBytes, 6_497_117_440)
        XCTAssertEqual(refused.headroomBytes, -697_117_440)
        XCTAssertEqual(plan.readAheadReserveBytes, 546_792_704)
    }

    func testReadAheadRequirementSaturatesButStillRefusesAtUInt64Max() {
        let decision = ReadAheadPairSchedule.decide(
            residentLayer: 0,
            resident: .init(storedBytes: 0, residentBytes: .max),
            stagedLayer: 1,
            next: .init(storedBytes: 1, residentBytes: 1),
            budgetBytes: .max)
        XCTAssertEqual(decision.requiredBytes, .max)
        XCTAssertFalse(decision.isAllowed)
        XCTAssertEqual(decision.headroomBytes, .min)
    }

    /// The same 5.8 GB budget on the USB4 rig: 85.79 s against the 85.60 s
    /// baseline. 0.19 s is the whole price of refusing that one pair, and it is
    /// the only thing the phone's budget costs relative to an unbounded one.
    func testThePriceOfThePhonesRefusedPairIsATenthOfASecond() throws {
        let bounded = try XCTUnwrap(try plan(Self.phoneBudget).projectedTokenSeconds)
        XCTAssertEqual(bounded, 85.79, accuracy: 0.01)
        XCTAssertEqual(bounded - Self.usb4.baselineTokenSeconds, 0.19, accuracy: 0.01)
    }

    /// 8 GiB: the first budget that pins anything, and the first one that shows
    /// the fill is not a prefix — layer 0 then layer *3*, because two KDA layers
    /// in between do not fit and an MLA layer does.
    func testMacEightGibibytes() throws {
        let plan = try self.plan(Self.gib(8))
        XCTAssertEqual(pinnedLayers(plan), [0, 3])
        XCTAssertEqual(plan.pinnedBytes, 3_185_655_808)
        XCTAssertEqual(plan.unusedBudgetBytes, 174_971_648)
        XCTAssertEqual(plan.projectedBytesPerTokenSaved, 3_185_655_808)
        XCTAssertEqual(plan.projectedBytesPerTokenRead, 133_804_647_424)
        XCTAssertEqual(plan.projectedStagedLayerCount, 90)
        XCTAssertEqual(plan.projectedInlineLayerCount, 1)
        XCTAssertEqual(plan.projectedRefusedPairCount, 1)
        XCTAssertEqual(try XCTUnwrap(plan.projectedTokenSeconds), 84.80, accuracy: 0.01)
    }

    /// 16 GiB: the first budget at which every read-ahead pair is affordable.
    /// The layer-0 pair becomes payable once layer 0 is pinned, because the
    /// widened copy is then the only deterministic term the pair adds.
    func testMacSixteenGibibytesIsWhereEveryPairBecomesAffordable() throws {
        let plan = try self.plan(Self.gib(16))
        XCTAssertEqual(pinnedLayers(plan), Array(0...8))
        XCTAssertEqual(plan.pinnedBytes, 11_636_916_224)
        XCTAssertEqual(plan.unusedBudgetBytes, 313_645_824)
        XCTAssertEqual(plan.projectedBytesPerTokenRead, 125_353_387_008)
        XCTAssertEqual(plan.projectedStagedLayerCount, 84)
        XCTAssertEqual(plan.projectedInlineLayerCount, 0)
        XCTAssertEqual(plan.projectedRefusedPairCount, 0)
        XCTAssertEqual(try XCTUnwrap(plan.projectedTokenSeconds), 82.90, accuracy: 0.01)
        // One rung down still refuses a pair, which is what makes 16 GiB the
        // inflection rather than a round number.
        XCTAssertGreaterThan(try self.plan(Self.gib(8)).projectedRefusedPairCount, 0)
    }

    /// 24 GiB: the design document's worked example — 17 layers, 0–15 and 19.
    func testMacTwentyFourGibibytes() throws {
        let plan = try self.plan(Self.gib(24))
        XCTAssertEqual(pinnedLayers(plan), Array(0...15) + [19])
        XCTAssertEqual(plan.pinnedLayerRanges, [0...15, 19...19])
        XCTAssertEqual(plan.pinnedBytes, 20_509_163_520)
        XCTAssertEqual(plan.unusedBudgetBytes, 31_333_120)
        XCTAssertEqual(plan.projectedBytesPerTokenSaved, 20_509_163_520)
        XCTAssertEqual(plan.projectedBytesPerTokenRead, 116_481_139_712)
        XCTAssertEqual(plan.projectedStagedLayerCount, 76)
        XCTAssertEqual(plan.projectedInlineLayerCount, 0)
        XCTAssertEqual(try XCTUnwrap(plan.projectedTokenSeconds), 81.11, accuracy: 0.01)
        XCTAssertEqual(plan.requestedMaximumNewTokens, 1)
        XCTAssertEqual(plan.projectedFirstPassPinnedLayerFillBytes, 20_509_163_520)
        XCTAssertEqual(plan.projectedPinnedLayerBytesServedAtTokenLimit, 0)
    }

    /// Pinning is a fill-once/reuse-later policy. The planner records that
    /// horizon in bytes and does not relabel a steady-state projection as the
    /// first token's cost.
    func testResponseHorizonAccountsForFirstPassFillAndLaterReuse() throws {
        let one = try self.plan(Self.gib(24), maximumNewTokens: 1)
        XCTAssertEqual(one.projectedFirstPassPinnedLayerFillBytes, 20_509_163_520)
        XCTAssertEqual(one.projectedPinnedLayerBytesServedAtTokenLimit, 0)

        let completeChat = try self.plan(Self.gib(24), maximumNewTokens: 64)
        XCTAssertEqual(completeChat.requestedMaximumNewTokens, 64)
        XCTAssertEqual(
            completeChat.projectedFirstPassPinnedLayerFillBytes, 20_509_163_520)
        XCTAssertEqual(
            completeChat.projectedPinnedLayerBytesServedAtTokenLimit,
            1_292_077_301_760)
        XCTAssertEqual(completeChat.pinnedLayers, one.pinnedLayers)
        XCTAssertEqual(completeChat.projectedBytesPerTokenRead, one.projectedBytesPerTokenRead)
    }

    func testAPlanRequiresAPositiveResponseHorizon() {
        for count in [0, -1] {
            XCTAssertThrowsError(
                try self.plan(Self.gib(24), maximumNewTokens: count)
            ) {
                XCTAssertEqual($0 as? MemoryDialError, .invalidMaximumNewTokens(count))
            }
        }
    }

    func testAnUnrepresentableLaterReuseBoundIsRefused() throws {
        let census = try ArtifactCensus(
            layerStoredBytes: [.max], layerResidentBytes: [.max],
            globalsStoredBytes: 0, globalsBytesReadPerToken: 0,
            expertStrideBytes: 0, expertsPerLayer: 0, routedExpertsPerToken: 0,
            moeLayerCount: 0, measuredDeterministicBytesPerToken: .max)
        let floor = WorkingSetFloor(
            widestResidentLayerBytes: 0, expertPoolBytes: 0, workingReserveBytes: 0)
        XCTAssertThrowsError(
            try MemoryDialPlanner.plan(
                budgetBytes: .max, census: census, floor: floor, readAhead: .off,
                maximumNewTokens: 3)
        ) {
            XCTAssertEqual(
                $0 as? MemoryDialError,
                .byteCountUnrepresentable(
                    context: "pinned-layer bytes served at the response limit"))
        }
    }

    func testMacThirtyTwoGibibytes() throws {
        let plan = try self.plan(Self.gib(32))
        XCTAssertEqual(pinnedLayers(plan), Array(0...23))
        XCTAssertEqual(plan.pinnedBytes, 28_960_423_936)
        XCTAssertEqual(plan.unusedBudgetBytes, 170_007_296)
        XCTAssertEqual(plan.projectedBytesPerTokenRead, 108_029_879_296)
        XCTAssertEqual(plan.projectedStagedLayerCount, 69)
        XCTAssertEqual(plan.projectedInlineLayerCount, 0)
        XCTAssertEqual(try XCTUnwrap(plan.projectedTokenSeconds), 79.41, accuracy: 0.01)
    }

    func testMacSixtyFourGibibytes() throws {
        let plan = try self.plan(Self.gib(64))
        XCTAssertEqual(pinnedLayers(plan), Array(0...52))
        XCTAssertEqual(plan.pinnedBytes, 62_763_040_768)
        XCTAssertEqual(plan.unusedBudgetBytes, 727_128_832)
        XCTAssertEqual(plan.projectedBytesPerTokenRead, 74_227_262_464)
        XCTAssertEqual(plan.projectedStagedLayerCount, 40)
        XCTAssertEqual(plan.projectedInlineLayerCount, 0)
        XCTAssertEqual(try XCTUnwrap(plan.projectedTokenSeconds), 72.59, accuracy: 0.01)
    }

    /// 114,047,231,232 B holds every deterministic layer — 106.2 GiB, and *not*
    /// the "~150 GB+" spec §11.5 states. Nothing streams, and the projection
    /// lands on 63.30 s, which is the serial arm's independently measured
    /// compute+expert path rather than an extrapolation.
    func testEveryDeterministicLayerFitsAtOneHundredSixPointTwoGibibytes() throws {
        let plan = try self.plan(114_047_231_232)
        XCTAssertEqual(plan.pinnedLayers.count, 93)
        XCTAssertNil(plan.pinnedGlobals)
        XCTAssertEqual(plan.pinnedBytes, 108_817_924_096)
        XCTAssertEqual(plan.unusedBudgetBytes, 0)
        XCTAssertEqual(plan.projectedBytesPerTokenSaved, 108_817_924_096)
        XCTAssertEqual(plan.projectedBytesPerTokenRead, 28_172_379_136)
        XCTAssertEqual(plan.projectedStagedLayerCount, 0)
        XCTAssertEqual(plan.projectedInlineLayerCount, 0)
        XCTAssertEqual(try XCTUnwrap(plan.projectedTokenSeconds), 63.30, accuracy: 0.01)
        XCTAssertEqual(
            Double(114_047_231_232) / 1_073_741_824, 106.2, accuracy: 0.05)
    }

    /// One tier further, the globals bundle: 118,744,900,864 B. Read per token
    /// is then the routed experts alone.
    func testTheGlobalsBundleIsTheTierAfterAllNinetyThreeLayers() throws {
        let plan = try self.plan(118_744_900_864)
        XCTAssertEqual(plan.pinnedLayers.count, 93)
        XCTAssertNotNil(plan.pinnedGlobals)
        XCTAssertEqual(plan.pinnedBytes, 113_515_593_728)
        XCTAssertEqual(plan.unusedBudgetBytes, 0)
        XCTAssertEqual(plan.projectedBytesPerTokenSaved, 111_160_730_624)
        XCTAssertEqual(plan.projectedBytesPerTokenRead, 25_829_572_608)
        XCTAssertEqual(try XCTUnwrap(plan.projectedTokenSeconds), 62.48, accuracy: 0.01)
    }

    // MARK: - The accounting identities

    /// Every byte of the dial is spent on something named or it is slack, and
    /// every per-token byte is pinned, staged or inline and never two of those.
    /// Both identities, at every budget in the table.
    func testBothAccountingIdentitiesHoldAtEveryBudgetInTheTable() throws {
        let budgets: [UInt64] = [
            5_229_307_136, Self.phoneBudget, Self.gib(8), Self.gib(16), Self.gib(24),
            Self.gib(32), Self.gib(64), 114_047_231_232, 118_744_900_864, Self.gib(200),
        ]
        for budget in budgets {
            for readAhead: ReadAheadOption in [.off, .depthOne] {
                let plan = try self.plan(
                    budget, readAhead: readAhead,
                    expertHotSet: .measured(
                        expertsPerLayer: 8, hitRate: 0.115, traceLabel: "M8 zipf-1.4"))
                XCTAssertTrue(plan.isAccountingBalanced, "bytes at \(budget)/\(readAhead)")
                XCTAssertTrue(plan.isResidencyBalanced, "residency at \(budget)/\(readAhead)")
                XCTAssertEqual(
                    plan.workingFloorBytes + plan.pinnedBytes + plan.expertHotSetBytes
                        + plan.unusedBudgetBytes,
                    budget, "floor + pinned + hot set + free at \(budget)")
                XCTAssertEqual(
                    plan.pinnedBytesPerToken + plan.stagedBytesPerToken
                        + plan.inlineBytesPerToken,
                    136_990_303_232, "per-token buckets at \(budget)")
                XCTAssertEqual(
                    plan.pinnedLayers.count + plan.projectedStagedLayerCount
                        + plan.projectedInlineLayerCount,
                    93, "layer count at \(budget)")
                XCTAssertEqual(
                    plan.projectedPeakBytes,
                    plan.workingFloorBytes + plan.pinnedBytes + plan.expertHotSetBytes)
                XCTAssertLessThanOrEqual(plan.projectedPeakBytes, budget)
            }
        }
    }

    /// The read-ahead reserve is exactly what the plan makes permanently
    /// resident, and it is what `ReadAheadPairSchedule` must be handed
    /// so that its pair decisions and this plan's staged/inline split agree.
    func testTheReadAheadReserveIsWhatTheScheduleMustBeHanded() throws {
        let plan = try self.plan(Self.gib(24))
        XCTAssertEqual(
            plan.readAheadReserveBytes,
            Self.floor.expertPoolBytes + Self.floor.workingReserveBytes + plan.pinnedBytes
                + plan.expertHotSetBytes)
        XCTAssertEqual(plan.readAheadReserveBytes, 21_055_956_224)
        // Fed to the schedule with the pinned layers excluded, the pairs the
        // plan says stage are exactly the pairs the schedule allows.
        let pinned = Set(pinnedLayers(plan))
        for layer in 1..<93 where !pinned.contains(layer) {
            let decision = ReadAheadPairSchedule.decide(
                residentLayer: layer - 1,
                resident: .init(
                    storedBytes: Self.census.layerStoredBytes[layer - 1],
                    residentBytes: Self.census.layerResidentBytes[layer - 1]),
                stagedLayer: layer,
                next: .init(
                    storedBytes: Self.census.layerStoredBytes[layer],
                    residentBytes: Self.census.layerResidentBytes[layer]),
                budgetBytes: plan.budgetBytes,
                reserveBytes: plan.readAheadReserveBytes)
            XCTAssertTrue(decision.isAllowed, "pair \(layer - 1)→\(layer)")
        }
    }

    /// With read-ahead off nothing hides behind compute: every streamed layer is
    /// inline, no pair is refused because no pair is scheduled, and the
    /// projection rises by the difference between the two rates.
    func testReadAheadOffMovesEveryStreamedLayerInline() throws {
        let off = try plan(Self.gib(24), readAhead: .off)
        XCTAssertEqual(off.projectedStagedLayerCount, 0)
        XCTAssertEqual(off.projectedInlineLayerCount, 76)
        XCTAssertEqual(off.projectedRefusedPairCount, 0)
        XCTAssertEqual(off.stagedBytesPerToken, 0)
        XCTAssertTrue(off.isAccountingBalanced)
        let on = try plan(Self.gib(24))
        XCTAssertGreaterThan(
            try XCTUnwrap(off.projectedTokenSeconds),
            try XCTUnwrap(on.projectedTokenSeconds))
        // Pinning is unchanged: staging decides which bucket a byte lands in,
        // never whether a unit is pinnable.
        XCTAssertEqual(pinnedLayers(off), pinnedLayers(on))
    }

    // MARK: - The time model

    /// Both endpoints of the interpolation are measurements: the null plan
    /// returns the staged arm's 85.6 s and the all-pinned plan the serial arm's
    /// compute+expert path at 63.3 s. Everything between them is a stated model.
    func testTheTimeModelReproducesBothMeasuredEndpoints() throws {
        // The null plan is an unbounded budget's staging at zero pinning, which
        // is what the 85.6 s arm ran: layer 0 and the globals inline, layers
        // 1…92 staged, experts inline.
        let inlineNull: UInt64 = 2_341_257_216 + 2_342_806_528 + 25_829_572_608
        let stagedNull: UInt64 = 108_817_924_096 - 2_341_257_216
        let null = Self.usb4.computeSeconds
            + Self.usb4.inlineSecondsPerByte * Double(inlineNull)
            + Self.usb4.stagedSecondsPerByte * Double(stagedNull)
        XCTAssertEqual(null, 85.60, accuracy: 0.01)
        XCTAssertEqual(
            try XCTUnwrap(try plan(114_047_231_232).projectedTokenSeconds), 63.30,
            accuracy: 0.01)
    }

    /// A staged byte is worth 57% of an inline byte in wall time. That is the
    /// fact that makes the dial's value *rise* as the expert stream's own
    /// overlap lands.
    func testAStagedByteIsWorthFiftySevenPercentOfAnInlineByte() {
        XCTAssertEqual(
            Self.usb4.stagedSecondsPerByte / Self.usb4.inlineSecondsPerByte, 0.57,
            accuracy: 0.005)
    }

    /// The rig's own calibration, checked against the arms it was fitted to:
    /// 108,817,924,096 B over the serial arm's 38.3 s is 2.84 GB/s, and layer 0
    /// alone at that rate is the 0.8 s of inline deterministic residue the
    /// staged arm reported.
    func testTheCalibrationIsTheMeasuredArms() {
        XCTAssertEqual(
            108_817_924_096 * Self.usb4.inlineSecondsPerByte, 38.3, accuracy: 0.05)
        XCTAssertEqual(1 / Self.usb4.inlineSecondsPerByte / 1e9, 2.84, accuracy: 0.01)
        XCTAssertEqual(1 / Self.usb4.stagedSecondsPerByte / 1e9, 4.958, accuracy: 0.01)
        XCTAssertEqual(
            2_341_257_216 * Self.usb4.inlineSecondsPerByte, 0.82, accuracy: 0.01)
        XCTAssertEqual(Self.usb4.computeSeconds, 53.38, accuracy: 0.01)
    }

    /// No calibration, no second. The phone has no measured link, and a borrowed
    /// number would be a claim no record supports.
    func testWithoutACalibrationThereIsNoProjectedSecond() throws {
        let plan = try self.plan(Self.gib(24), calibration: nil)
        XCTAssertNil(plan.projectedTokenSeconds)
        XCTAssertNil(plan.calibrationLabel)
        XCTAssertNil(plan.baselineTokenSeconds)
        XCTAssertNil(MemoryDialPlanner.summary(plan).tokenSecondsLabel)
    }

    func testAnInvalidCalibrationDoesNotPublishAFictionalProjection() throws {
        let calibration = LinkCalibration(
            label: "invalid", inlineSecondsPerByte: .infinity,
            stagedSecondsPerByte: 0, computeSeconds: 0, baselineTokenSeconds: 0)
        let plan = try self.plan(Self.gib(24), calibration: calibration)
        XCTAssertNil(plan.projectedTokenSeconds)
        XCTAssertNil(plan.calibrationLabel)
        XCTAssertNil(plan.baselineTokenSeconds)
    }

    // MARK: - Fill rule and snap points

    /// The fill *skips* what does not fit rather than stopping there. At 24 GiB
    /// that is the difference between 16 layers and 17: layers 16, 17 and 18 are
    /// unaffordable, layer 19 is not, and reaching past them saves another
    /// 844,398,592 B per token.
    func testTheFillSkipsRatherThanStopsAtTheFirstMiss() throws {
        let plan = try self.plan(Self.gib(24))
        XCTAssertEqual(plan.pinnedLayers.count, 17)

        // A stop-at-first-miss fill over the same ranked ladder, for contrast.
        var remaining = Self.gib(24) - Self.floor.totalBytes
        var stopped: [Int] = []
        for (layer, stored) in Self.census.layerStoredBytes.enumerated() {
            guard stored <= remaining else { break }
            remaining -= stored
            stopped.append(layer)
        }
        XCTAssertEqual(stopped, Array(0...15))
        XCTAssertEqual(remaining, 875_731_712)
        XCTAssertEqual(
            plan.projectedBytesPerTokenSaved - stopped.reduce(UInt64(0)) {
                $0 + Self.census.layerStoredBytes[$1]
            }, 844_398_592)
    }

    /// The next-pin budget is the smallest one that pins strictly more, not this
    /// budget plus the next unit's size. At 24 GiB layer 16 needs 1,236,477,184 B
    /// more — and admitting it does not cost layer 19, because at that budget
    /// both fit.
    func testTheNextPinBudgetIsTheSmallestOneThatPinsMore() throws {
        let plan = try self.plan(Self.gib(24))
        XCTAssertEqual(plan.nextPinUnit, .deterministicLayer(16))
        XCTAssertEqual(plan.nextPinBudgetBytes, 27_006_280_960)
        let next = try self.plan(try XCTUnwrap(plan.nextPinBudgetBytes))
        XCTAssertEqual(next.pinnedLayers.count, 18)
        XCTAssertEqual(pinnedLayers(next), Array(0...16) + [19])
        // One byte less pins no more than the original plan did.
        let below = try self.plan(try XCTUnwrap(plan.nextPinBudgetBytes) - 1)
        XCTAssertEqual(below.pinnedLayers.count, 17)
    }

    /// Everything pinned means no next step at all — the dial has run out of
    /// units, not out of budget.
    func testAFullyPinnedPlanHasNoNextStep() throws {
        let plan = try self.plan(
            250_000_000_000,
            expertHotSet: .measured(expertsPerLayer: 8, hitRate: 0.115, traceLabel: "M8"))
        XCTAssertNil(plan.nextPinBudgetBytes)
        XCTAssertNil(plan.nextPinUnit)
        XCTAssertNil(MemoryDialPlanner.summary(plan).nextStepLabel)
    }

    /// The snap points are the floor, then the budget at which each successive
    /// prefix of the ladder is affordable in full. The last two are the two
    /// budgets the design document names.
    func testSnapPointsStartAtTheFloorAndEndAtTheWholeLadder() throws {
        let points = MemoryDialPlanner.snapPoints(census: Self.census, floor: Self.floor)
        XCTAssertEqual(points.count, 95)
        XCTAssertEqual(points.first, 5_229_307_136)
        XCTAssertEqual(points[1], 5_229_307_136 + 2_341_257_216)
        XCTAssertEqual(points[93], 114_047_231_232)
        XCTAssertEqual(points[94], 118_744_900_864)
        XCTAssertEqual(points, points.sorted())
        // At a snap point the prefix is affordable in full and nothing is
        // slack; one byte below, it is not affordable at all.
        for index in [1, 9, 40, 93] {
            let at = try MemoryDialPlanner.plan(
                budgetBytes: points[index], census: Self.census, floor: Self.floor)
            XCTAssertEqual(pinnedLayers(at), Array(0..<index))
            XCTAssertEqual(at.unusedBudgetBytes, 0)
            let below = try MemoryDialPlanner.plan(
                budgetBytes: points[index] - 1, census: Self.census, floor: Self.floor)
            XCTAssertNotEqual(pinnedLayers(below), Array(0..<index))
            XCTAssertLessThan(below.pinnedBytes, at.pinnedBytes)
        }
    }

    /// A byte below a snap point the fill does not simply lose a layer — it
    /// reaches *past* the one it can no longer afford and pins two smaller ones.
    /// More layers, fewer bytes saved: the ladder maximises bytes per resident
    /// byte, and the layer count is a consequence rather than the goal.
    func testOneByteBelowASnapPointTheFillReachesPast() throws {
        let points = MemoryDialPlanner.snapPoints(census: Self.census, floor: Self.floor)
        let below = try plan(points[1] - 1, calibration: nil)
        XCTAssertEqual(pinnedLayers(below), [1, 3])
        XCTAssertEqual(below.pinnedBytes, 2_112_208_896)
        let at = try plan(points[1], calibration: nil)
        XCTAssertEqual(pinnedLayers(at), [0])
        XCTAssertEqual(at.pinnedBytes, 2_341_257_216)
        XCTAssertGreaterThan(below.pinnedLayers.count, at.pinnedLayers.count)
        XCTAssertLessThan(below.projectedBytesPerTokenSaved, at.projectedBytesPerTokenSaved)
    }

    // MARK: - The screen

    /// The dial screen, at the design document's worked budget. Every figure is
    /// a `PinPlan` field formatted, so the screen and the run's JSON cannot
    /// disagree.
    func testSummaryAtTwentyFourGibibytes() throws {
        let summary = MemoryDialPlanner.summary(try plan(Self.gib(24)))
        XCTAssertEqual(summary.budgetLabel, "24 GiB")
        XCTAssertEqual(summary.headline, "17 of 93 layers held in RAM")
        XCTAssertEqual(summary.pinnedLayerRanges, [0...15, 19...19])
        XCTAssertEqual(summary.residencyLabel, "19.1 GiB pinned + 4.87 GiB floor")
        XCTAssertEqual(summary.perTokenLabel, "116.5 GB read per token, was 137.0 GB")
        XCTAssertEqual(summary.savedFraction, 0.150, accuracy: 0.001)
        XCTAssertEqual(
            summary.tokenSecondsLabel, "≈81.1 s/token (USB4 rig, baseline 85.6 s)")
        XCTAssertEqual(summary.readAheadLabel, "read-ahead: 76 of 76 streamed layers staged")
        XCTAssertEqual(summary.nextStepLabel, "+1.15 GiB pins layer 16")
        XCTAssertEqual(summary.nextStepBudgetBytes, 27_006_280_960)
        XCTAssertTrue(summary.floorLabel.hasPrefix("4.87 GiB minimum"), summary.floorLabel)
        XCTAssertEqual(summary.warnings, [])
    }

    /// The phone's screen: no seconds, and two warnings that name numbers
    /// rather than adjectives.
    func testSummaryAtThePhoneBudgetWarnsWithNumbers() throws {
        let summary = MemoryDialPlanner.summary(
            try plan(Self.phoneBudget, calibration: nil))
        XCTAssertEqual(summary.budgetLabel, "5.40 GiB")
        XCTAssertEqual(summary.headline, "no layers held in RAM — 93 stream")
        XCTAssertEqual(summary.pinnedLayerRanges, [])
        XCTAssertEqual(summary.savedFraction, 0, accuracy: 1e-12)
        XCTAssertNil(summary.tokenSecondsLabel)
        XCTAssertEqual(summary.readAheadLabel, "read-ahead: 91 of 93 streamed layers staged")
        XCTAssertEqual(summary.warnings.count, 2)
        XCTAssertTrue(
            summary.warnings[0].contains("first pin needs a declared budget of 6,073,705,728 B"),
            summary.warnings[0])
        XCTAssertEqual(
            summary.warnings[1],
            "layer 0 read-ahead pair refused (needs 697,117,440 B more)")
    }

    // MARK: - Purity and the record

    /// Same inputs, same plan — the property that lets a phone, a Mac and a
    /// unit test agree about what a budget buys.
    func testThePlannerIsAPureFunction() throws {
        for budget in [Self.phoneBudget, Self.gib(24), 118_744_900_864] {
            let first = try plan(budget)
            let second = try plan(budget)
            XCTAssertEqual(first, second)
        }
    }

    /// A plan serialises beside `DeterministicReadAheadMetrics` in a run's JSON,
    /// so a timing run records the plan that produced it. A record that cannot
    /// be read back is not a record (AGENTS.md §17.4).
    func testAPlanRoundTripsThroughJSON() throws {
        let plan = try self.plan(
            Self.gib(24),
            expertHotSet: .measured(expertsPerLayer: 8, hitRate: 0.115, traceLabel: "M8"),
            maximumNewTokens: 64)
        let data = try JSONEncoder().encode(plan)
        XCTAssertEqual(try JSONDecoder().decode(PinPlan.self, from: data), plan)

        let object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(
            (object["pinnedBytes"] as? NSNumber)?.uint64Value, 20_509_163_520)
        XCTAssertEqual(object["calibrationLabel"] as? String, "USB4 rig")
        XCTAssertEqual((object["requestedMaximumNewTokens"] as? NSNumber)?.intValue, 64)
        XCTAssertEqual(
            (object["projectedFirstPassPinnedLayerFillBytes"] as? NSNumber)?.uint64Value,
            20_509_163_520)
        XCTAssertEqual(
            (object["projectedPinnedLayerBytesServedAtTokenLimit"] as? NSNumber)?.uint64Value,
            1_292_077_301_760)
    }

    func testLegacyOrForgedResponseHorizonIsRefusedByRuntimeValidation() throws {
        let valid = try self.plan(Self.gib(24), calibration: nil, maximumNewTokens: 64)
        var legacy = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(valid)) as? [String: Any])
        legacy.removeValue(forKey: "requestedMaximumNewTokens")
        legacy.removeValue(forKey: "projectedFirstPassPinnedLayerFillBytes")
        legacy.removeValue(forKey: "projectedPinnedLayerBytesServedAtTokenLimit")
        let decodedLegacy = try JSONDecoder().decode(
            PinPlan.self, from: JSONSerialization.data(withJSONObject: legacy))
        XCTAssertTrue(
            try XCTUnwrap(decodedLegacy.runtimeValidationError).contains("response-token limit"))

        var forged = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(valid)) as? [String: Any])
        forged["projectedPinnedLayerBytesServedAtTokenLimit"] = 1
        let decodedForged = try JSONDecoder().decode(
            PinPlan.self, from: JSONSerialization.data(withJSONObject: forged))
        XCTAssertTrue(
            try XCTUnwrap(decodedForged.runtimeValidationError).contains("reuse bound"))
    }

    /// Archived plans are untrusted inputs too. Overflowed decoded buckets must
    /// report an imbalance, while hostile count fields must not trap summary
    /// construction.
    func testADecodedOverflowingPlanIsNotFalselyBalancedOrCrashable() throws {
        let valid = try self.plan(Self.gib(24), calibration: nil)
        var json = try XCTUnwrap(String(data: JSONEncoder().encode(valid), encoding: .utf8))
        json = json.replacingOccurrences(
            of: "\"pinnedBytesPerToken\":20509163520",
            with: "\"pinnedBytesPerToken\":18446744073709551615")
        json = json.replacingOccurrences(
            of: "\"projectedStagedLayerCount\":76",
            with: "\"projectedStagedLayerCount\":9223372036854775807")
        json = json.replacingOccurrences(
            of: "\"projectedInlineLayerCount\":0",
            with: "\"projectedInlineLayerCount\":9223372036854775807")
        let decoded = try JSONDecoder().decode(PinPlan.self, from: Data(json.utf8))
        XCTAssertFalse(decoded.isAccountingBalanced)
        XCTAssertEqual(decoded.streamedLayerCount, .max)
        let summary = MemoryDialPlanner.summary(decoded)
        XCTAssertTrue(
            summary.warnings.contains("per-token byte buckets do not sum to the measured total"))
    }

    /// Budgets far above anything real must produce a plan, not a trap: the
    /// arithmetic saturates rather than overflowing.
    func testAbsurdBudgetsSaturateRatherThanTrap() throws {
        let plan = try self.plan(.max, calibration: nil)
        XCTAssertEqual(plan.pinnedLayers.count, 93)
        XCTAssertNotNil(plan.pinnedGlobals)
        XCTAssertTrue(plan.isResidencyBalanced)
        XCTAssertTrue(plan.isAccountingBalanced)
    }
}

private struct UncheckedCensus: Encodable {
    let layerStoredBytes: [UInt64]
    let layerResidentBytes: [UInt64]
    let globalsStoredBytes: UInt64
    let globalsBytesReadPerToken: UInt64
    let expertStrideBytes: UInt64
    let expertsPerLayer: Int
    let routedExpertsPerToken: Int
    let moeLayerCount: Int
    let measuredDeterministicBytesPerToken: UInt64
}
