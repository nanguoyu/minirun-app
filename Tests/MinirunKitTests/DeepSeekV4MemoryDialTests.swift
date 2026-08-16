import XCTest

@testable import MinirunKit

/// The V4 dial is pure arithmetic, so every case here runs with no drive, no
/// artifact and no MLX — the same property that makes `MemoryDialPlannerTests`
/// a table of fixtures rather than a set of runs.
final class DeepSeekV4MemoryDialTests: XCTestCase {

    // MARK: Fixtures

    /// A census shaped like the published artifact: 43 layers whose block-FP8
    /// tiles sum to 5,869,305,856 B, and a resident column 1/32 larger for the
    /// expanded scale grid, which is the whole reason V4 needs its own census.
    private func census(
        layers: Int = 43,
        // A multiple of 32 so the resident column is exact and the ratio is
        // exactly 32/33 rather than a rounding of it.
        readBytes: UInt64 = 136_495_488,
        residentFactorNumerator: UInt64 = 33,
        residentFactorDenominator: UInt64 = 32,
        globals: UInt64 = 1_058_996_224,
        experts: UInt64 = 3_450_000_000
    ) throws -> DeepSeekV4MemoryDial.Census {
        let read = [UInt64](repeating: readBytes, count: layers)
        return try DeepSeekV4MemoryDial.Census(
            layerReadBytes: read,
            layerResidentBytes: read.map {
                $0 * residentFactorNumerator / residentFactorDenominator
            },
            globalsBytesReadPerToken: globals,
            expertBytesPerToken: experts)
    }

    /// The published globals rung: two `[129280, 4096]` BF16 tables. The head
    /// is read whole on every pass; the embedding is read one 8,192 B row at a
    /// time, which is what puts it five orders of magnitude down the ladder.
    private static let headTableBytes: UInt64 = 1_059_061_760
    private static let embeddingRowBytes: UInt64 = 8_192

    private func globals(
        head: UInt64 = headTableBytes,
        embeddingRow: UInt64 = embeddingRowBytes
    ) -> DeepSeekV4MemoryDial.Globals {
        DeepSeekV4MemoryDial.Globals(
            headResidentBytes: head,
            headReadBytesPerToken: head,
            embeddingResidentBytes: Self.headTableBytes,
            embeddingReadBytesPerToken: embeddingRow)
    }

    /// The 43-layer census with the globals rung the artifact actually has.
    private func pinnableCensus(layers: Int = 43) throws -> DeepSeekV4MemoryDial.Census {
        let read = [UInt64](repeating: 136_495_488, count: layers)
        return try DeepSeekV4MemoryDial.Census(
            layerReadBytes: read,
            layerResidentBytes: read.map { $0 * 33 / 32 },
            globals: globals(),
            expertBytesPerToken: 0)
    }

    private func floor(
        transient: UInt64 = 1_800_000_000,
        pool: UInt64 = 46_792_704,
        reserve: UInt64 = 400_000_000
    ) -> WorkingSetFloor {
        WorkingSetFloor(
            widestResidentLayerBytes: transient,
            expertPoolBytes: pool,
            workingReserveBytes: reserve)
    }

    // MARK: The census boundary

    func testACensusRefusesARaggedOrShrinkingTable() throws {
        XCTAssertThrowsError(
            try DeepSeekV4MemoryDial.Census(
                layerReadBytes: [], layerResidentBytes: [],
                globalsBytesReadPerToken: 0, expertBytesPerToken: 0)
        ) { XCTAssertEqual($0 as? MemoryDialError, .layerTableEmpty) }

        XCTAssertThrowsError(
            try DeepSeekV4MemoryDial.Census(
                layerReadBytes: [10, 20], layerResidentBytes: [10],
                globalsBytesReadPerToken: 0, expertBytesPerToken: 0)
        ) {
            XCTAssertEqual(
                $0 as? MemoryDialError,
                .layerTableRagged(storedCount: 2, residentCount: 1))
        }

        // A layer that costs less resident than it reads would describe a
        // compression the loader does not perform, and its ratio would outrank
        // every honest candidate.
        XCTAssertThrowsError(
            try DeepSeekV4MemoryDial.Census(
                layerReadBytes: [100], layerResidentBytes: [99],
                globalsBytesReadPerToken: 0, expertBytesPerToken: 0)
        ) {
            XCTAssertEqual(
                $0 as? MemoryDialError,
                .residentSmallerThanStored(layer: 0, stored: 100, resident: 99))
        }
    }

    // MARK: Refusal, not clamping

    func testABudgetBelowTheFloorIsRefusedByNameAndNamesEveryTerm() throws {
        let census = try census()
        let floor = floor()
        let short = floor.totalBytes - 1

        XCTAssertThrowsError(
            try DeepSeekV4MemoryDial.plan(
                budgetBytes: short, census: census, floor: floor, maximumNewTokens: 8)
        ) { error in
            guard case MemoryDialError.budgetBelowWorkingFloor(
                let budget, let floorBytes, let shortfall,
                let widest, let pool, let reserve) = error
            else { return XCTFail("expected budgetBelowWorkingFloor, got \(error)") }
            XCTAssertEqual(budget, short)
            XCTAssertEqual(floorBytes, floor.totalBytes)
            XCTAssertEqual(shortfall, 1)
            // The message can say *which* term the budget failed against.
            XCTAssertEqual(widest, 1_800_000_000)
            XCTAssertEqual(pool, 46_792_704)
            XCTAssertEqual(reserve, 400_000_000)
        }
    }

    func testExactlyTheFloorIsAdmittedAndPinsNothing() throws {
        let census = try census()
        let floor = floor()
        let plan = try DeepSeekV4MemoryDial.plan(
            budgetBytes: floor.totalBytes, census: census, floor: floor,
            maximumNewTokens: 8)

        XCTAssertTrue(plan.pinnedLayers.isEmpty)
        XCTAssertEqual(plan.pinnedBytes, 0)
        XCTAssertEqual(plan.unusedBudgetBytes, 0)
        XCTAssertEqual(plan.projectedBytesPerTokenSaved, 0)
        XCTAssertEqual(plan.projectedBytesPerTokenRead, census.bytesPerToken)
        XCTAssertNil(plan.runtimeValidationError)
    }

    // MARK: The fill

    func testTheFillPinsEveryLayerThatFitsAndNeverExceedsTheHeadroom() throws {
        let census = try census()
        let floor = floor()
        let resident = census.layerResidentBytes[0]

        // Room for exactly three layers, one byte short of a fourth.
        let budget = floor.totalBytes + resident * 4 - 1
        let plan = try DeepSeekV4MemoryDial.plan(
            budgetBytes: budget, census: census, floor: floor, maximumNewTokens: 8)

        XCTAssertEqual(plan.pinnedLayers.count, 3)
        XCTAssertEqual(plan.pinnedBytes, resident * 3)
        XCTAssertEqual(plan.unusedBudgetBytes, resident - 1)
        XCTAssertNil(plan.runtimeValidationError)
        // Ties break by layer order, so an equal-ratio table pins a prefix.
        XCTAssertEqual(plan.pinnedLayerRanges, [0...2])
        // `nextPinBudgetBytes` is the smallest budget that pins *more*, not the
        // size of the next unit: one more byte buys the fourth layer.
        XCTAssertEqual(plan.nextPinBudgetBytes, budget + 1)
        XCTAssertEqual(plan.nextPinUnit, .deterministicLayer(3))
        // Residency is spent, never overspent.
        XCTAssertTrue(plan.isResidencyBalanced)
        XCTAssertEqual(
            plan.workingFloorBytes + plan.pinnedBytes + plan.unusedBudgetBytes, budget)
    }

    func testTheFillSkipsAnUnaffordableCandidateAndReachesPastIt() throws {
        // One expensive layer first, then two cheap ones. A stop-at-first-miss
        // fill pins nothing; the skip rule pins both cheap layers. This is the
        // rule V4 inherits unchanged from K3's planner.
        let census = try DeepSeekV4MemoryDial.Census(
            layerReadBytes: [1_000, 100, 100],
            layerResidentBytes: [1_000, 100, 100],
            globalsBytesReadPerToken: 0,
            expertBytesPerToken: 0)
        let floor = floor(transient: 0, pool: 0, reserve: 0)
        let plan = try DeepSeekV4MemoryDial.plan(
            budgetBytes: 250, census: census, floor: floor, maximumNewTokens: 4)

        XCTAssertEqual(plan.pinnedLayers.count, 2)
        XCTAssertEqual(plan.pinnedLayerRanges, [1...2])
        XCTAssertEqual(plan.pinnedBytes, 200)
        XCTAssertEqual(plan.unusedBudgetBytes, 50)
        XCTAssertNil(plan.runtimeValidationError)
    }

    func testAHigherRatioOutranksLayerOrder() throws {
        // Layer 1 costs the same resident bytes but saves more, so it is ranked
        // first even though layer 0 comes earlier. The ladder maximises bytes
        // saved; layer order is only the tie-break.
        let census = try DeepSeekV4MemoryDial.Census(
            layerReadBytes: [50, 100],
            layerResidentBytes: [100, 100],
            globalsBytesReadPerToken: 0,
            expertBytesPerToken: 0)
        let floor = floor(transient: 0, pool: 0, reserve: 0)
        let plan = try DeepSeekV4MemoryDial.plan(
            budgetBytes: 100, census: census, floor: floor, maximumNewTokens: 2)

        XCTAssertEqual(plan.pinnedLayers.count, 1)
        XCTAssertEqual(plan.pinnedLayers.first?.unit, .deterministicLayer(1))
        XCTAssertEqual(plan.projectedBytesPerTokenSaved, 100)
    }

    // MARK: The identities a run validates before allocating

    func testAFullyPinnedPlanBalancesBothIdentitiesAndValidatesAtRuntime() throws {
        let census = try census()
        let floor = floor()
        let residency = census.layerResidentBytes.reduce(0, +)
        let plan = try DeepSeekV4MemoryDial.plan(
            budgetBytes: floor.totalBytes + residency, census: census, floor: floor,
            maximumNewTokens: 16)

        XCTAssertEqual(plan.pinnedLayers.count, 43)
        XCTAssertEqual(plan.pinnedBytes, residency)
        XCTAssertEqual(plan.unusedBudgetBytes, 0)
        // Every deterministic layer byte is now saved; the globals walk and the
        // routed experts are not pinnable and stay in the read total.
        XCTAssertEqual(
            plan.projectedBytesPerTokenSaved, census.deterministicLayerBytesPerToken)
        XCTAssertEqual(
            plan.projectedBytesPerTokenRead,
            census.globalsBytesReadPerToken + census.expertBytesPerToken)
        XCTAssertEqual(
            plan.inlineBytesPerToken,
            census.globalsBytesReadPerToken + census.expertBytesPerToken)
        XCTAssertEqual(plan.stagedBytesPerToken, 0)

        XCTAssertTrue(plan.isAccountingBalanced)
        XCTAssertTrue(plan.isResidencyBalanced)
        XCTAssertNil(plan.runtimeValidationError)
    }

    func testTheResidentColumnIsChargedAndTheReadColumnIsSaved() throws {
        // The single fact that stops `ArtifactCensus` from being reusable here:
        // a V4 pin costs more than it saves, and the plan must state both
        // numbers rather than deriving one from the other.
        let census = try census(layers: 1)
        let floor = floor(transient: 0, pool: 0, reserve: 0)
        let resident = census.layerResidentBytes[0]
        let read = census.layerReadBytes[0]
        XCTAssertGreaterThan(resident, read)

        let plan = try DeepSeekV4MemoryDial.plan(
            budgetBytes: resident, census: census, floor: floor, maximumNewTokens: 2)
        let decision = try XCTUnwrap(plan.pinnedLayers.first)

        XCTAssertEqual(decision.residentBytes, resident)
        XCTAssertEqual(decision.bytesSavedPerToken, read)
        XCTAssertEqual(plan.pinnedBytes, resident)
        XCTAssertEqual(plan.projectedBytesPerTokenSaved, read)
        // ~0.970 rather than K3's 1.000, and still far above the 0.230 ceiling
        // any measured expert hot set has reached.
        XCTAssertEqual(decision.savedPerResidentByte, 32.0 / 33.0, accuracy: 1e-9)
        XCTAssertNil(plan.runtimeValidationError)
    }

    // MARK: Purity and the response horizon

    func testThePlanIsAPureFunctionOfItsInputs() throws {
        let census = try census()
        let floor = floor()
        let budget = floor.totalBytes + 20_000_000_000

        let first = try DeepSeekV4MemoryDial.plan(
            budgetBytes: budget, census: census, floor: floor, maximumNewTokens: 8)
        let second = try DeepSeekV4MemoryDial.plan(
            budgetBytes: budget, census: census, floor: floor, maximumNewTokens: 8)
        XCTAssertEqual(first, second)
    }

    func testTheHorizonBindsTheReuseBoundAndAOneTokenRunReusesNothing() throws {
        let census = try census()
        let floor = floor()
        let budget = floor.totalBytes + census.layerResidentBytes[0] * 3

        let single = try DeepSeekV4MemoryDial.plan(
            budgetBytes: budget, census: census, floor: floor, maximumNewTokens: 1)
        XCTAssertEqual(single.requestedMaximumNewTokens, 1)
        XCTAssertEqual(single.projectedPinnedLayerBytesServedAtTokenLimit, 0)
        XCTAssertEqual(
            single.projectedFirstPassPinnedLayerFillBytes, single.pinnedBytes)
        XCTAssertNil(single.runtimeValidationError)

        let eight = try DeepSeekV4MemoryDial.plan(
            budgetBytes: budget, census: census, floor: floor, maximumNewTokens: 8)
        XCTAssertEqual(
            eight.projectedPinnedLayerBytesServedAtTokenLimit, eight.pinnedBytes * 7)

        XCTAssertThrowsError(
            try DeepSeekV4MemoryDial.plan(
                budgetBytes: budget, census: census, floor: floor, maximumNewTokens: 0)
        ) { XCTAssertEqual($0 as? MemoryDialError, .invalidMaximumNewTokens(0)) }
    }

    // MARK: The globals rung

    /// The arithmetic that made the head a rung and left the embedding out of
    /// one. Both numbers are the census's own, so a reader can check the
    /// decision rather than take it.
    func testTheGlobalsArithmeticRanksTheHeadAtOneAndTheEmbeddingAtNothing() throws {
        let census = try pinnableCensus()

        XCTAssertEqual(census.globals.headSavedPerResidentByte, 1.0, accuracy: 1e-12)
        XCTAssertEqual(
            census.globals.embeddingSavedPerResidentByte, 7.7351e-6, accuracy: 1e-10)
        // The head is read whole every pass; the embedding contributes one row.
        XCTAssertEqual(
            census.globalsBytesReadPerToken,
            Self.headTableBytes + Self.embeddingRowBytes)
        // A census that claims a table costs less to hold than it saves is
        // refused: no loader compresses a table by holding it.
        XCTAssertThrowsError(
            try DeepSeekV4MemoryDial.Census(
                layerReadBytes: [100], layerResidentBytes: [100],
                globals: DeepSeekV4MemoryDial.Globals(
                    headResidentBytes: 99, headReadBytesPerToken: 100,
                    embeddingResidentBytes: 0, embeddingReadBytesPerToken: 0),
                expertBytesPerToken: 0)
        ) {
            XCTAssertEqual(
                $0 as? MemoryDialError,
                .globalUnitResidentSmallerThanRead(
                    unit: "output head", read: 100, resident: 99))
        }
    }

    /// **The ordering rule: every layer, in schedule order, then the head.**
    ///
    /// A budget with room for all 43 layers and the head pins both, and the
    /// head's rank is last — behind every layer, even though its ratio (1.000)
    /// is above theirs (~0.970). The rung is stated rather than derived from the
    /// ratio because a layer's 0.970 undercounts: the 3% is the expanded scale
    /// grid, which is what the pin *buys*.
    func testTheHeadIsRankedAfterEveryLayerAndPinnedWithThem() throws {
        let census = try pinnableCensus()
        let floor = floor()
        let layerResidency = census.layerResidentBytes.reduce(0, +)
        let budget = floor.totalBytes + layerResidency + Self.headTableBytes

        let plan = try DeepSeekV4MemoryDial.plan(
            budgetBytes: budget, census: census, floor: floor, maximumNewTokens: 16)

        XCTAssertEqual(plan.pinnedLayers.count, 43)
        let head = try XCTUnwrap(plan.pinnedGlobals)
        XCTAssertEqual(head.unit, .outputHead)
        XCTAssertEqual(head.residentBytes, Self.headTableBytes)
        XCTAssertEqual(head.bytesSavedPerToken, Self.headTableBytes)
        XCTAssertEqual(head.savedPerResidentByte, 1.0, accuracy: 1e-12)
        XCTAssertEqual(head.rank, 43, "the head is the 44th rung, not the first")
        XCTAssertGreaterThan(
            head.savedPerResidentByte,
            try XCTUnwrap(plan.pinnedLayers.first).savedPerResidentByte,
            "the ordering is the stated rule, not a consequence of the ratio")

        // The pinned term absorbs the head: residency, the saved column and the
        // reserve all include it, and no new term was needed for it.
        XCTAssertEqual(plan.pinnedBytes, layerResidency + Self.headTableBytes)
        XCTAssertEqual(
            plan.projectedBytesPerTokenSaved,
            census.deterministicLayerBytesPerToken + Self.headTableBytes)
        // What is left is the embedding row, which is never a candidate.
        XCTAssertEqual(plan.inlineBytesPerToken, Self.embeddingRowBytes)
        XCTAssertEqual(plan.projectedBytesPerTokenRead, Self.embeddingRowBytes)
        XCTAssertEqual(plan.unusedBudgetBytes, 0)

        // The layer-scoped fields stay layer-scoped, which is what the shared
        // validator checks them against.
        XCTAssertEqual(plan.projectedFirstPassPinnedLayerFillBytes, layerResidency)
        XCTAssertEqual(
            plan.projectedPinnedLayerBytesServedAtTokenLimit, layerResidency * 15)
        XCTAssertEqual(plan.projectedInlineLayerCount, 0)
        XCTAssertTrue(plan.isAccountingBalanced)
        XCTAssertTrue(plan.isResidencyBalanced)
        XCTAssertNil(plan.runtimeValidationError)
    }

    /// A budget that fits every layer but not the head pins the layers only —
    /// and says the head is what the next budget buys.
    func testABudgetThatFitsEveryLayerButNotTheHeadPinsLayersOnly() throws {
        let census = try pinnableCensus()
        let floor = floor()
        let layerResidency = census.layerResidentBytes.reduce(0, +)
        let budget = floor.totalBytes + layerResidency + Self.headTableBytes - 1

        let plan = try DeepSeekV4MemoryDial.plan(
            budgetBytes: budget, census: census, floor: floor, maximumNewTokens: 8)

        XCTAssertEqual(plan.pinnedLayers.count, 43)
        XCTAssertNil(plan.pinnedGlobals)
        XCTAssertEqual(plan.pinnedBytes, layerResidency)
        XCTAssertEqual(plan.nextPinUnit, .outputHead)
        XCTAssertEqual(plan.nextPinBudgetBytes, budget + 1)
        // The whole globals read is still inline — head walk and embedding row.
        XCTAssertEqual(plan.inlineBytesPerToken, census.globalsBytesReadPerToken)
        XCTAssertNil(plan.runtimeValidationError)
    }

    /// The head never displaces a layer, and this is the budget where the rule
    /// is visible rather than incidental.
    ///
    /// With room for 42 layers plus one head table, a ladder ordered by the
    /// ratio alone would pin the head (1.000) first and then 42 layers. This one
    /// pins **all 43 layers** and skips the head, which saves fewer bytes on
    /// paper — 5.87 GB against 6.79 GB — and is the intended answer: the layer
    /// column undercounts by the per-pass host work a pin also removes, and 43
    /// layers is the residency the phone-sized budgets in this range can
    /// actually spend.
    func testTheHeadNeverDisplacesALayer() throws {
        let census = try pinnableCensus()
        let floor = floor()
        let resident = census.layerResidentBytes[0]
        let budget = floor.totalBytes + resident * 42 + Self.headTableBytes

        let plan = try DeepSeekV4MemoryDial.plan(
            budgetBytes: budget, census: census, floor: floor, maximumNewTokens: 8)

        XCTAssertEqual(plan.pinnedLayers.count, 43)
        XCTAssertEqual(plan.pinnedLayerRanges, [0...42])
        XCTAssertNil(plan.pinnedGlobals, "the head does not take a layer's budget")
        XCTAssertEqual(plan.nextPinUnit, .outputHead)
        XCTAssertNil(plan.runtimeValidationError)
    }

    /// A census with no resident column for its globals — every caller before
    /// the rung existed — states its read total and pins nothing of it.
    func testACensusWithoutAStatedHeadResidencyPinsNoGlobals() throws {
        let census = try census()
        let floor = floor()
        let plan = try DeepSeekV4MemoryDial.plan(
            budgetBytes: floor.totalBytes + 20_000_000_000, census: census, floor: floor,
            maximumNewTokens: 8)

        XCTAssertEqual(plan.pinnedLayers.count, 43)
        XCTAssertNil(plan.pinnedGlobals)
        XCTAssertEqual(
            plan.inlineBytesPerToken,
            census.globalsBytesReadPerToken + census.expertBytesPerToken)
        XCTAssertNil(plan.runtimeValidationError)
    }

    func testSnapPointsEndAtTheHeadOnceEveryLayerIsHeld() throws {
        let census = try pinnableCensus(layers: 3)
        let floor = floor()
        let points = DeepSeekV4MemoryDial.snapPoints(census: census, floor: floor)
        let resident = census.layerResidentBytes[0]

        XCTAssertEqual(
            points,
            [
                floor.totalBytes,
                floor.totalBytes + resident,
                floor.totalBytes + resident * 2,
                floor.totalBytes + resident * 3,
                floor.totalBytes + resident * 3 + Self.headTableBytes,
            ])
    }

    func testSnapPointsStartAtTheFloorAndRiseByOneCandidateEach() throws {
        let census = try census(layers: 3)
        let floor = floor()
        let points = DeepSeekV4MemoryDial.snapPoints(census: census, floor: floor)
        let resident = census.layerResidentBytes[0]

        XCTAssertEqual(
            points,
            [
                floor.totalBytes,
                floor.totalBytes + resident,
                floor.totalBytes + resident * 2,
                floor.totalBytes + resident * 3,
            ])
    }
}
