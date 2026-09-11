import Foundation
import MinirunKit
import ModelAdapters
import StorageCore

/// The two operands the memory dial needs — a census and a floor — for V4.1.
///
/// The **rules** are not forked. `DeepSeekV4MemoryDial` is pure arithmetic over
/// a two-column census: rank by bytes saved per resident byte, break ties by
/// block order, skip a unit the budget cannot afford instead of stopping, refuse
/// a budget below the floor by name, clamp nothing. Those rules are about a
/// storage-native decode and not about V4, and V4.1's census has the same two
/// columns for the same reason V4's does — a pinned block holds the *loaded*
/// `BlockFP8Weights` form, whose expanded scale grid is larger than the stored
/// tile's, so resident and saved are different numbers and a census that derived
/// both from one array would be wrong in whichever direction it chose.
///
/// What is V4.1's own is the arithmetic of the columns:
///
/// | rung | resident | read per token | saved per resident byte |
/// | --- | --- | --- | --- |
/// | a dense block | ~176 MB | ~170 MB | ~0.969 |
/// | blocks 1 and 14 | ~338 / ~355 MB | 328 / 344 MB | ~0.969 |
/// | output head | 1.324 GB | 1.324 GB | 1.000 |
/// | embedding table | 1.324 GB | 10,240 B | 0.0000077 |
///
/// so the ladder is forty blocks, then the head, and the embedding table is
/// censused and never ranked — the same verdict V4 and K3 reached from the same
/// column.
public enum DeepSeekV41MemoryDialInputs {

    /// Held back from the pin budget on top of every named term, exactly as
    /// V4's is and for the same reason: the census is a manifest-derived upper
    /// bound on tensor bytes and does not account for what MLX's allocator adds
    /// around each resident array. 256 MiB over an 8.8 GB tier is ~3%.
    public static let statedPinMarginBytes: UInt64 = DeepSeekV4MemoryDialInputs
        .statedPinMarginBytes

    /// The transient envelope of a run that **pins**, which is the only kind of
    /// run this ladder is ever consulted about.
    ///
    /// `DeepSeekV41ProductMemoryBudget.transientExecutionBytes` is 3.2 GB, and
    /// it is correct for what it measures: the floor arm, which pins nothing and
    /// streams every block. A ladder priced with it is under-priced at **every
    /// rung**, because a snap point is the floor plus a prefix and therefore
    /// spends its budget exactly — so a floor that is short by any amount at all
    /// produces a preset the run is stopped by. It did, twice, at the two
    /// budgets `docs/experiments/2026-09-11-v41-app-dial.md` records: a
    /// 12,868,502,464 B budget peaked at 12,875,485,504 B and a 13,002,720,192 B
    /// budget peaked at 13,025,776,064 B, both after pinning all forty blocks
    /// and the head, both refused by their own budget. The overshoot **grew**
    /// with the budget, which is what says it is not a margin.
    ///
    /// **Measured, from the arm that passed.** Phase 3's stated arm ran the same
    /// residency inside 15 GB and added 14,561,888,000 B. Subtract what that run
    /// held and the terms this plan names — 8,729,882,048 B of resident weights,
    /// the 125,337,600 B pool, the 536,870,912 B saturated cache and 7,976,448 B
    /// of bounded state — and its transient was **5,161,820,992 B**, 1.78× the
    /// floor arm's 2,897,876,864 B. Pinning does not only move bytes from the
    /// drive into residency; the matmuls those resident weights feed have
    /// intermediates of their own, and MLX's allocator keeps more around them.
    ///
    /// 5.2 GB rounds above that observation exactly as the product floor's
    /// 3.2 GB rounds above its own. It is used **only by the ladder**: the
    /// product floor is a claim about a run that pins nothing and stays 3.4 GB,
    /// which phase 3 measured and this does not touch.
    ///
    /// Since the iPhone-dial work the number lives on the **platform policy**
    /// rather than here, because a ladder is a claim about this platform's run
    /// and a floor priced with another platform's terms offers rungs this
    /// device cannot hold. The Mac's value did not change; the iPhone states
    /// the same one and says why (``DeepSeekV41ProductMemoryBudget/iOSProductPolicy``).
    public static var pinnedTransientExecutionBytes: UInt64 {
        DeepSeekV41ProductMemoryBudget.currentPolicy.pinnedTransientExecutionBytes
    }

    /// A **lower bound** on the smallest budget at which this platform's ladder
    /// can pin anything at all, priced from the policy and the dial's own
    /// stated terms — the pinned envelope, the saturated MLX cache a stated run
    /// is allowed, and the pin margin.
    ///
    /// It is a lower bound and is documented as one: a real floor also carries
    /// the routed-expert pool and the bounded state, both of which are read
    /// from a published census this function has not been given. That makes it
    /// exactly the right shape for the one question it is asked — *can a stated
    /// budget buy a rung on this device at all?* — because below it the answer
    /// is no for certain, and above it the run's own plan still decides.
    public static func smallestPinningBudgetBytes(
        policy: DeepSeekV41ProductMemoryBudget.Policy = DeepSeekV41ProductMemoryBudget
            .currentPolicy
    ) -> UInt64 {
        let sum = UInt64Accounting.saturatingSum([
            policy.pinnedTransientExecutionBytes,
            DeepSeekV41EffectiveKnobs.maximumDefaultMLXCacheBytes,
            statedPinMarginBytes,
        ])
        return sum.didOverflow ? .max : sum.value
    }

    /// The floor in the three terms ``WorkingSetFloor`` names, from a product
    /// plan priced with an unbounded declared budget.
    ///
    /// The transient execution envelope takes the "widest resident layer" slot
    /// for the same reason it does in V4: it is the same quantity — the widest
    /// moment the run has evidence for — and it was measured rather than
    /// derived. The 3.4 GB product floor is deliberately absent; that is the
    /// product scale's policy, and a stated run declares its own.
    ///
    /// It is ``pinnedTransientExecutionBytes`` and not the plan's own, because a
    /// floor that is consulted to decide what to *pin* has to price a run that
    /// pins. The plan's term describes the floor arm, and using it produced two
    /// recorded chats that pinned everything the dial promised and were then
    /// stopped by the budget the dial had chosen.
    public static func floor(
        pricing plan: DeepSeekV41ProductMemoryPlan,
        policy: DeepSeekV41ProductMemoryBudget.Policy = DeepSeekV41ProductMemoryBudget
            .currentPolicy
    ) -> WorkingSetFloor? {
        let reserve = UInt64Accounting.saturatingSum([
            plan.retainedStateBytes, plan.replacementBlockStateBytes,
            plan.mlxCacheBytes, statedPinMarginBytes,
        ])
        guard !reserve.didOverflow else { return nil }
        return WorkingSetFloor(
            widestResidentLayerBytes: max(
                plan.transientExecutionBytes, policy.pinnedTransientExecutionBytes),
            expertPoolBytes: plan.expertPoolBytes,
            workingReserveBytes: reserve.value)
    }

    /// Everything a screen needs to draw V4.1's ladder for an artifact that is
    /// on disk and not running.
    ///
    /// The **same type** V4's inputs publish, and declared as an alias rather
    /// than as a second struct with the same three fields. The dial's operands
    /// are a census, a floor and a pool size; the census is already
    /// ``DeepSeekV4MemoryDial/Census`` here because the rules are shared, so a
    /// parallel struct would have added a conversion between two spellings of
    /// one thing and a second place for them to drift. The screens hold one
    /// field, and it is honest for both models.
    public typealias ArtifactProfile = DeepSeekV4MemoryDialInputs.ArtifactProfile

    /// The smallest budget that pins every dense block and the output head.
    ///
    /// Stated as a function rather than a constant because it is arithmetic over
    /// a census, and a census is a property of a published artifact: a revision
    /// that widened a block would move this number, and a constant would not
    /// notice.
    public static func budgetThatPinsEverything(
        census: DeepSeekV4MemoryDial.Census, floor: WorkingSetFloor
    ) -> UInt64 {
        let blocks = UInt64Accounting.saturatingSum(census.layerResidentBytes)
        let sum = UInt64Accounting.saturatingSum([
            floor.totalBytes, blocks.value, census.globals.headResidentBytes,
        ])
        return sum.didOverflow ? .max : sum.value
    }

    // MARK: - The ladder, before anything is opened

    /// Read the census and price the floor for a verified V4.1 artifact.
    ///
    /// ``DeepSeekV4MemoryDialInputs/inspect(_:)`` for V4.1, and it exists for
    /// the same reason: the memory dial has to draw a ladder *before* the run
    /// that would consume it, so the operand cannot come from an opened
    /// workload. Without it the app had no ladder for V4.1 at all — every preset
    /// collapsed onto the 3.4 GB floor and a Mac chat streamed all forty blocks
    /// every token while 30 GB sat unused.
    ///
    /// **No payload byte is read.** The reads are `index.json`, the two
    /// configuration documents ADR 0020 binds together, one `manifest.json` per
    /// block and one for `global00` — metadata the verification pass already
    /// covered. The per-block columns are recovered from the manifest's own
    /// published geometry rather than by opening a container, which is exact
    /// because the constructor asserts the identity that makes it exact: a
    /// manifest tile-container's `bytes` equals its header's `totalBytes`, which
    /// is `QuantizedTileContainer.headerBytes + tileCount × tileStride`. A
    /// manifest that disagrees with its header is refused there, when the run
    /// opens it, rather than smoothed over here.
    ///
    /// The floor is priced at the **product ceilings** — 512 prompt tokens, 64
    /// generated — and at the saturated MLX cache, for V4's reason: a ladder is
    /// a statement about every budget rather than about one request, and pricing
    /// it at the largest state this product will ever hold is the conservative
    /// direction. A preset chosen off this ladder is never short of what the run
    /// then reserves.
    public static func inspect(_ artifact: ArtifactReference) throws -> ArtifactProfile {
        try inspect(artifact, policy: DeepSeekV41ProductMemoryBudget.currentPolicy)
    }

    /// The same read, priced by a named platform policy.
    ///
    /// Spelled as a second entry point rather than as a defaulted parameter
    /// because the app holds ``inspect(_:)`` as a plain
    /// `(ArtifactReference) throws -> ArtifactProfile` in its per-model table,
    /// and a function with a defaulted argument cannot be referenced at that
    /// type. The macOS suite uses this one to price the iPhone ladder without
    /// pretending it ran on an iPhone.
    public static func inspect(
        _ artifact: ArtifactReference,
        policy: DeepSeekV41ProductMemoryBudget.Policy
    ) throws -> ArtifactProfile {
        guard let authority = artifact.runtimeAuthority else {
            throw RunError.artifactNotReady(
                "the V4.1 memory dial requires complete rooted verification authority")
        }
        let indexData = try authority.openFile("index.json").readAll(maximumBytes: 8 << 20)
        let index = try DeepSeekV41ArtifactIndex(json: indexData)
        let config = try DeepSeekV41Config(
            inferenceJSON: try authority.openFile(index.argumentsFile).readAll(
                maximumBytes: 1 << 20),
            huggingFaceJSON: try authority.openFile(index.configurationFile).readAll(
                maximumBytes: 1 << 20))

        var readBytes = [UInt64]()
        var residentBytes = [UInt64]()
        var widestExpertStride: UInt64 = 0
        readBytes.reserveCapacity(index.blockUnits.count)
        residentBytes.reserveCapacity(index.blockUnits.count)
        for unit in index.blockUnits {
            let manifest = try authority.openFile("\(unit.id)/manifest.json").readAll(
                maximumBytes: 8 << 20)
            let block = try blockCensus(manifestData: manifest, unitReference: unit.id)
            readBytes.append(block.readBytesPerPass)
            residentBytes.append(block.pinnedResidentBytes)
            widestExpertStride = max(widestExpertStride, block.widestExpertSlotBytes)
        }
        let globalUnit = index.globalUnit.id
        let globals = try self.globals(
            manifestData: try authority.openFile("\(globalUnit)/manifest.json").readAll(
                maximumBytes: 8 << 20),
            unitReference: globalUnit)
        try authority.validateCurrentBinding()

        let census = try DeepSeekV4MemoryDial.Census(
            layerReadBytes: readBytes,
            layerResidentBytes: residentBytes,
            globals: globals,
            // Routed-expert reads are not addressed by this tier — no budget
            // any device here can offer reaches an expert rung — and the run
            // reports its own expert accounting rather than a projection.
            expertBytesPerToken: 0)

        // Every floor term below is the **policy's**, not a constant: the pool
        // is its slot count, the prompt and reply ceilings are its own, and the
        // pinned envelope reaches the floor through `floor(pricing:policy:)`.
        // A ladder priced with another platform's terms is the defect this
        // threading exists to stop.
        let pool = widestExpertStride.multipliedReportingOverflow(
            by: UInt64(policy.expertPoolSlots))
        guard !pool.overflow else {
            throw RunError.artifactNotReady(
                "the V4.1 expert pool byte count exceeds UInt64.max")
        }
        let priced = try DeepSeekV41ProductMemoryBudget.plan(
            declaredBudgetBytes: .max,
            config: config,
            promptTokenCount: policy.maximumPromptTokens,
            maximumNewTokens: policy.maximumNewTokens,
            expertPoolBytes: pool.partialValue,
            mlxCacheBytes: DeepSeekV41EffectiveKnobs.maximumDefaultMLXCacheBytes,
            pinnedDeterministicBytes: 0,
            expertTileStrideBytes: widestExpertStride,
            policy: policy,
            enforcesProductLimits: false)
        guard let floor = floor(pricing: priced, policy: policy) else {
            throw RunError.artifactNotReady("the V4.1 working-set floor exceeds UInt64.max")
        }
        return ArtifactProfile(
            census: census, floor: floor, expertPoolBytes: pool.partialValue)
    }

    /// One block's two ladder columns, and the pool granularity beside them.
    struct BlockCensus: Equatable {
        /// Dense bytes a pass re-reads when the block is not pinned: every FP8
        /// matrix container and the plain-tensor blob. No routed-expert tile —
        /// those are chosen per token — and no Engram part, because a ~100 GB
        /// table of which a token reads 24 pages is neither a per-block quantity
        /// nor a pinnable one.
        let readBytesPerPass: UInt64
        /// What pinning those tensors costs. Larger than the column above by the
        /// FP8 scale expansion: a stored tile carries one E8M0 exponent per
        /// 32×32 block and the resident `BlockFP8Weights` form carries one per
        /// row per 32 columns.
        let pinnedResidentBytes: UInt64
        /// One pool slot: one expert's half of the widest published pair tile,
        /// which is what the product pool reserves (ADR 0021 §5). Not part of
        /// either column — the pool is its own floor term, and this is only the
        /// granularity it is priced in.
        let widestExpertSlotBytes: UInt64
    }

    static func blockCensus(
        manifestData: Data, unitReference: String
    ) throws -> BlockCensus {
        let document = try decode(manifestData, unitReference: unitReference)
        let headerBytes = UInt64(QuantizedTileContainer.headerBytes)
        var read = UInt64Accounting.SaturatingSum()
        var resident = UInt64Accounting.SaturatingSum()
        var widestSlot: UInt64 = 0

        for file in document.files {
            switch file.kind {
            case "tile-container":
                guard let tiles = file.count, tiles > 0, file.bytes > headerBytes else {
                    throw DeepSeekV41Error.artifact(
                        "\(file.name) declares no tile count or no payload beyond its header")
                }
                let stride = (file.bytes - headerBytes) / UInt64(tiles)
                guard let tensor = file.sourceTensor, !tensor.isEmpty else {
                    // No `source_tensor` is what makes a container routed
                    // experts rather than a dense matrix — the same test the
                    // block constructor applies. A slot holds one expert's half
                    // of the pair the tile publishes.
                    let members = UInt64(max(1, file.membersPerTile ?? 1))
                    widestSlot = max(widestSlot, stride / members)
                    continue
                }
                guard let rows = file.rows, let columns = file.cols, let group = file.group,
                    let bits = file.elementBits, rows > 0, columns > 0, group > 0, bits > 0
                else {
                    throw DeepSeekV41Error.artifact(
                        "\(file.name) names a source tensor without a declared geometry")
                }
                read.add(file.bytes)
                resident.add(UInt64(rows) * UInt64(columns) * UInt64(bits) / 8)
                // The expanded grid: one exponent per row per `group` columns.
                resident.add(UInt64(rows) * UInt64(columns / group))

            case "blob":
                // A bundled blob's members partition its bytes exactly — the
                // constructor refuses one that does not — and a standalone blob
                // is one tensor. Either way the file is what a pass reads and
                // what a pinned block holds.
                read.add(file.bytes)
                resident.add(file.bytes)

            case "engram-part":
                continue

            default:
                throw DeepSeekV41Error.artifact(
                    "\(file.name) uses unsupported payload kind '\(file.kind)'")
            }
        }
        guard !read.didOverflow, !resident.didOverflow, read.value > 0 else {
            throw DeepSeekV41Error.artifact(
                "\(unitReference) declares no deterministic bytes for a pass to read")
        }
        return BlockCensus(
            readBytesPerPass: read.value,
            pinnedResidentBytes: resident.value,
            widestExpertSlotBytes: widestSlot)
    }

    /// The globals rung, from `global00`'s own manifest.
    ///
    /// The head is read whole every pass, in `logitChunkRows` windows, so
    /// pinning it saves exactly what it costs and it ranks at 1.000. The
    /// embedding table is the same size and a decode token reads one row of it,
    /// so it is censused and never ranked.
    static func globals(
        manifestData: Data, unitReference: String
    ) throws -> DeepSeekV4MemoryDial.Globals {
        let document = try decode(manifestData, unitReference: unitReference)
        var tables = [String: (bytes: UInt64, rows: UInt64)]()
        for file in document.files where file.kind == "blob" {
            if let members = file.members {
                for member in members {
                    tables[member.sourceTensor] = (
                        member.bytes, UInt64(max(1, member.shape.first ?? 1))
                    )
                }
            } else if let tensor = file.sourceTensor {
                tables[tensor] = (file.bytes, UInt64(max(1, file.shape?.first ?? 1)))
            }
        }
        guard let head = tables[DeepSeekV41GlobalTensor.head.rawValue],
            let embedding = tables[DeepSeekV41GlobalTensor.embed.rawValue]
        else {
            throw DeepSeekV41Error.artifact(
                "\(unitReference) publishes no output head and embedding table")
        }
        return DeepSeekV4MemoryDial.Globals(
            headResidentBytes: head.bytes,
            headReadBytesPerToken: head.bytes,
            embeddingResidentBytes: embedding.bytes,
            embeddingReadBytesPerToken: embedding.bytes / embedding.rows)
    }

    private static func decode(
        _ manifestData: Data, unitReference: String
    ) throws -> Manifest {
        let document: Manifest
        do {
            document = try JSONDecoder().decode(Manifest.self, from: manifestData)
        } catch {
            throw DeepSeekV41Error.artifact("manifest.json could not be decoded: \(error)")
        }
        guard document.unit == unitReference else {
            throw DeepSeekV41Error.artifact(
                "manifest unit '\(document.unit)' does not name rooted unit "
                    + "'\(unitReference)'")
        }
        guard !document.files.isEmpty else {
            throw DeepSeekV41Error.artifact("\(unitReference) contains no payload files")
        }
        return document
    }

    /// The manifest fields this census needs, and no others.
    ///
    /// Deliberately not ``DeepSeekV41BlockArtifact``'s own wire type: that one
    /// is the *constructor's* reading, with every field the reconciliation
    /// checks, and it reaches the filesystem for each of them. This reads the
    /// published document and nothing else.
    struct Manifest: Decodable {
        struct Member: Decodable {
            let sourceTensor: String
            let shape: [Int]
            let bytes: UInt64

            enum CodingKeys: String, CodingKey {
                case shape, bytes
                case sourceTensor = "source_tensor"
            }
        }

        struct File: Decodable {
            let name: String
            let kind: String
            let bytes: UInt64
            let rows: Int?
            let cols: Int?
            let count: Int?
            let group: Int?
            let elementBits: Int?
            let membersPerTile: Int?
            let sourceTensor: String?
            let shape: [Int]?
            /// Present on a bundled blob. The same key carries an integer on a
            /// routed-expert container, which this census does not read, so the
            /// decode is attempted rather than required.
            let members: [Member]?

            enum CodingKeys: String, CodingKey {
                case name, kind, bytes, rows, cols, count, group, shape, members
                case elementBits = "element_bits"
                case membersPerTile = "members_per_tile"
                case sourceTensor = "source_tensor"
            }

            init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                name = try container.decode(String.self, forKey: .name)
                kind = try container.decode(String.self, forKey: .kind)
                bytes = try container.decode(UInt64.self, forKey: .bytes)
                rows = try container.decodeIfPresent(Int.self, forKey: .rows)
                cols = try container.decodeIfPresent(Int.self, forKey: .cols)
                count = try container.decodeIfPresent(Int.self, forKey: .count)
                group = try container.decodeIfPresent(Int.self, forKey: .group)
                elementBits = try container.decodeIfPresent(Int.self, forKey: .elementBits)
                membersPerTile = try container.decodeIfPresent(
                    Int.self, forKey: .membersPerTile)
                sourceTensor = try container.decodeIfPresent(String.self, forKey: .sourceTensor)
                shape = try container.decodeIfPresent([Int].self, forKey: .shape)
                members = try? container.decodeIfPresent([Member].self, forKey: .members)
            }
        }

        let unit: String
        let files: [File]
    }
}
