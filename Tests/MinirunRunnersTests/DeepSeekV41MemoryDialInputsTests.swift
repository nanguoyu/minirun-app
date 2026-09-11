import Foundation
import MinirunKit
import ModelAdapters
import XCTest

@testable import MinirunRunners

/// The V4.1 ladder's two columns, read from a manifest and nothing else.
///
/// `inspect(_:)` needs a rooted verification authority over 517 GB, so the
/// arithmetic it performs is exercised here directly on documents this file
/// writes — and, when the real container is mounted and named, on the published
/// manifests themselves. Both matter: the synthetic case pins the rules, and the
/// armed case is what says the rules produce the numbers the phase 3 arms
/// measured.
final class DeepSeekV41MemoryDialInputsTests: XCTestCase {

    /// `layers00`'s own `attn.wkv` container: FP8 `[512, 5120]` at group 32,
    /// one tile, 2,654,208 B on disk. The bytes are the manifest's; the
    /// arithmetic below is the dial's.
    func testADenseMatrixIsReadWholeAndPinnedWithItsExpandedScaleGrid() throws {
        let manifest = """
            {"unit": "layers00", "files": [
              {"name": "layers00-attn_wkv_weight.fp8tile", "kind": "tile-container",
               "bytes": 2654208, "rows": 512, "cols": 5120, "count": 1,
               "element_bits": 8, "group": 32, "block_rows": 32,
               "layout": "fp8-e4m3-block", "source_tensor": "layers.0.attn.wkv.weight"}
            ]}
            """
        let census = try DeepSeekV41MemoryDialInputs.blockCensus(
            manifestData: Data(manifest.utf8), unitReference: "layers00")

        // A pass reads the file. A pin holds the packed matrix plus one E8M0
        // exponent per row per 32 columns — 512 × 160 = 81,920 B more than the
        // 2,621,440 B of packed weights, and the reason resident and read are
        // two columns rather than one.
        XCTAssertEqual(census.readBytesPerPass, 2_654_208)
        XCTAssertEqual(census.pinnedResidentBytes, 2_621_440 + 81_920)
        XCTAssertEqual(census.widestExpertSlotBytes, 0)
    }

    /// A routed-expert container has no `source_tensor` — the same test the
    /// block constructor applies — and contributes to neither column. What it
    /// does state is the pool's granularity: one expert's half of a pair tile.
    func testRoutedExpertsArePricedAsPoolSlotsAndNotAsLadderRungs() throws {
        let manifest = """
            {"unit": "layers00", "files": [
              {"name": "layers00-w1.mxfp4tile", "kind": "tile-container",
               "bytes": 2406498304, "rows": 4608, "cols": 5120, "count": 192,
               "element_bits": 4, "group": 32, "layout": "mxfp4",
               "rows_per_member": 2304, "members_per_tile": 2, "members": 384},
              {"name": "layers00-plain.bin", "kind": "blob", "bytes": 7891928,
               "sha256": "ca7b1518cff2be2729ca232c072151d46407245919c363f816b8267a0929e9a3"},
              {"name": "layers00-engram-000.engrampage", "kind": "engram-part",
               "bytes": 104857600}
            ]}
            """
        let census = try DeepSeekV41MemoryDialInputs.blockCensus(
            manifestData: Data(manifest.utf8), unitReference: "layers00")

        // The blob and only the blob. The Engram part is a ~100 GB table a token
        // reads 24 pages of, which is neither a per-block quantity nor a
        // pinnable one.
        XCTAssertEqual(census.readBytesPerPass, 7_891_928)
        XCTAssertEqual(census.pinnedResidentBytes, 7_891_928)
        // (2,406,498,304 − 16,384) / 192 tiles = 12,533,760 B a pair;
        // 6,266,880 B an expert, which × 20 slots is the 125,337,600 B pool
        // ADR 0021 §5 prices.
        XCTAssertEqual(census.widestExpertSlotBytes, 6_266_880)
        XCTAssertEqual(
            census.widestExpertSlotBytes
                * UInt64(DeepSeekV41EffectiveKnobs.defaultExpertPoolSlots),
            125_337_600)
    }

    func testAManifestThatNamesAnotherUnitIsRefused() {
        let manifest = """
            {"unit": "layers07", "files": [{"name": "a", "kind": "blob", "bytes": 1}]}
            """
        XCTAssertThrowsError(
            try DeepSeekV41MemoryDialInputs.blockCensus(
                manifestData: Data(manifest.utf8), unitReference: "layers00"))
    }

    func testAnUnknownPayloadKindIsRefusedRatherThanIgnored() {
        let manifest = """
            {"unit": "layers00", "files": [
              {"name": "layers00-mystery.bin", "kind": "sparse-thing", "bytes": 4096}
            ]}
            """
        XCTAssertThrowsError(
            try DeepSeekV41MemoryDialInputs.blockCensus(
                manifestData: Data(manifest.utf8), unitReference: "layers00"))
    }

    /// The globals rung: the head costs what it saves, and the embedding table
    /// is censused at one row a token, which is what keeps it off the ladder.
    func testTheGlobalsRungPricesTheHeadWholeAndTheEmbeddingByTheRow() throws {
        let manifest = """
            {"unit": "global00", "files": [
              {"name": "global00-embed_weight.bin", "kind": "blob", "bytes": 1323827200,
               "source_tensor": "embed.weight", "dtype": "BF16", "shape": [129280, 5120]},
              {"name": "global00-head_weight.bin", "kind": "blob", "bytes": 1323827200,
               "source_tensor": "head.weight", "dtype": "BF16", "shape": [129280, 5120]}
            ]}
            """
        let globals = try DeepSeekV41MemoryDialInputs.globals(
            manifestData: Data(manifest.utf8), unitReference: "global00")

        XCTAssertEqual(globals.headResidentBytes, 1_323_827_200)
        XCTAssertEqual(globals.headReadBytesPerToken, 1_323_827_200)
        XCTAssertEqual(globals.embeddingResidentBytes, 1_323_827_200)
        XCTAssertEqual(globals.embeddingReadBytesPerToken, 10_240)
    }

    /// A bundled blob states its tensors as members over one file, which is how
    /// `global00` publishes everything that is not one of the two big tables.
    func testABundledBlobsMembersAreRead() throws {
        let manifest = """
            {"unit": "global00", "files": [
              {"name": "global00-plain.bin", "kind": "blob", "bytes": 2647664640,
               "members": [
                 {"source_tensor": "embed.weight", "dtype": "BF16",
                  "shape": [129280, 5120], "offset": 0, "bytes": 1323827200},
                 {"source_tensor": "head.weight", "dtype": "BF16",
                  "shape": [129280, 5120], "offset": 1323827200, "bytes": 1323827200}
               ]}
            ]}
            """
        let globals = try DeepSeekV41MemoryDialInputs.globals(
            manifestData: Data(manifest.utf8), unitReference: "global00")
        XCTAssertEqual(globals.headResidentBytes, 1_323_827_200)
        XCTAssertEqual(globals.embeddingReadBytesPerToken, 10_240)
    }

    /// **The published container's own ladder**, when it is mounted and named.
    ///
    /// Metadata only — `index.json`, forty block manifests and one globals
    /// manifest, no payload byte — so this is seconds rather than the gate's
    /// 517 GB. Arm it with the container root:
    ///
    ///     MINIRUN_V41_LADDER_ARTIFACT=/Volumes/K3NVME/DeepSeek-V4.1-Flash-minirun
    ///
    /// The totals are the phase 3 record's
    /// (`docs/experiments/2026-09-11-v41-phase3-runner.md` §3): 7.21 GB of dense
    /// weights re-read every token at the floor, and 8,729,882,048 B of resident
    /// weights held at 15 GB with the output head.
    func testThePublishedManifestsCensusToThePhaseThreeNumbers() throws {
        guard let root = ProcessInfo.processInfo.environment["MINIRUN_V41_LADDER_ARTIFACT"],
            !root.isEmpty
        else {
            throw XCTSkip("set MINIRUN_V41_LADDER_ARTIFACT to the mounted container root")
        }
        let container = URL(fileURLWithPath: root, isDirectory: true)
        let index = try DeepSeekV41ArtifactIndex(
            json: try Data(contentsOf: container.appendingPathComponent("index.json")))
        XCTAssertEqual(index.blockUnits.count, 40)

        var read = [UInt64]()
        var resident = [UInt64]()
        var widestSlot: UInt64 = 0
        for unit in index.blockUnits {
            let census = try DeepSeekV41MemoryDialInputs.blockCensus(
                manifestData: try Data(
                    contentsOf: container.appendingPathComponent(unit.id)
                        .appendingPathComponent("manifest.json")),
                unitReference: unit.id)
            read.append(census.readBytesPerPass)
            resident.append(census.pinnedResidentBytes)
            widestSlot = max(widestSlot, census.widestExpertSlotBytes)
        }
        let globals = try DeepSeekV41MemoryDialInputs.globals(
            manifestData: try Data(
                contentsOf: container.appendingPathComponent(index.globalUnit.id)
                    .appendingPathComponent("manifest.json")),
            unitReference: index.globalUnit.id)

        XCTAssertEqual(read.reduce(0, +), 7_206_792_640)
        XCTAssertEqual(resident.reduce(0, +), 7_406_054_848)
        XCTAssertEqual(
            resident.reduce(0, +) + globals.headResidentBytes, 8_729_882_048,
            "the resident weights the 15 GB arm held")
        XCTAssertEqual(globals.headResidentBytes, 1_323_827_200)
        XCTAssertEqual(globals.embeddingReadBytesPerToken, 10_240)
        XCTAssertEqual(widestSlot * 20, 125_337_600)

        // The catalog's ladder is the same artifact's, spelled as six block
        // shapes. If the published container ever moves, this is the assertion
        // that says the fixture has to move with it.
        let census = try DeepSeekV4MemoryDial.Census(
            layerReadBytes: read, layerResidentBytes: resident, globals: globals,
            expertBytesPerToken: 0)
        let floor = WorkingSetFloor(
            widestResidentLayerBytes: DeepSeekV41ProductMemoryBudget.transientExecutionBytes,
            expertPoolBytes: widestSlot * 20,
            workingReserveBytes: 536_870_912
                + DeepSeekV41MemoryDialInputs.statedPinMarginBytes)
        XCTAssertEqual(
            DeepSeekV41MemoryDialInputs.budgetThatPinsEverything(
                census: census, floor: floor),
            14_860_526_016,
            "Balanced on this owner's Mac, from the catalog entry's floor terms")

        // And the floor `inspect` actually prices, which differs from the one
        // above by the bounded generation state alone: the catalog entry omits
        // it because it is prompt-dependent and small, and the measured ladder
        // that replaces the entry prices it exactly.
        let config = try DeepSeekV41Config(
            inferenceJSON: try Data(
                contentsOf: container.appendingPathComponent(index.argumentsFile)),
            huggingFaceJSON: try Data(
                contentsOf: container.appendingPathComponent(index.configurationFile)))
        let priced = try DeepSeekV41ProductMemoryBudget.plan(
            declaredBudgetBytes: .max, config: config,
            promptTokenCount: DeepSeekV41ProductMemoryBudget.maximumPromptTokens,
            maximumNewTokens: DeepSeekV41ProductMemoryBudget.maximumNewTokens,
            expertPoolBytes: widestSlot * 20,
            mlxCacheBytes: DeepSeekV41EffectiveKnobs.maximumDefaultMLXCacheBytes,
            pinnedDeterministicBytes: 0, enforcesProductLimits: false)
        let measuredFloor = try XCTUnwrap(DeepSeekV41MemoryDialInputs.floor(pricing: priced))
        let measuredBalanced = DeepSeekV41MemoryDialInputs.budgetThatPinsEverything(
            census: census, floor: measuredFloor)
        print(
            "[v41-ladder] measured floor \(measuredFloor.totalBytes) B, state "
                + "\(priced.retainedStateBytes) + \(priced.replacementBlockStateBytes) B, "
                + "Balanced \(measuredBalanced) B")
        XCTAssertGreaterThan(measuredBalanced, 14_860_526_016)
        XCTAssertLessThan(
            measuredBalanced - 14_860_526_016, 32 << 20,
            "the entry's floor may omit the bounded state; it may not omit a tier")
    }
}
