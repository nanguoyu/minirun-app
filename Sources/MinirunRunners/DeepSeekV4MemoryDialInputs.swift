import Foundation
import MinirunKit
import ModelAdapters
import StorageCore

/// The two operands ``DeepSeekV4MemoryDial`` needs — a census and a floor — in
/// the one place both the run and the screen that draws it can read them.
///
/// The dial's rules live in MinirunKit and are pure arithmetic over these two
/// values. What was missing was a shared way to *obtain* them: the run engine
/// built the floor inline from the priced product plan and read the census off
/// an opened workload, so a product screen could only draw a ladder by
/// re-deriving both. Re-derivation is exactly the failure the K3 dial was
/// rewritten to remove — a screen and a run that disagree about what the
/// artifact costs — so the derivation is here, once, and the engine calls it.
///
/// Nothing in this type reads a payload byte. ``inspect(_:)`` opens the
/// verified `index.json`, the verified config, and one `manifest.json` per
/// layer; the per-layer byte columns come from
/// ``DeepSeekV4LayerArtifact/manifestCensus(manifestData:unitReference:)``,
/// which recovers a tile stride from the container's own published header
/// arithmetic rather than by opening the container.
public enum DeepSeekV4MemoryDialInputs {

    /// Held back from the pin budget, on top of every named term.
    ///
    /// The census the plan is built from is a per-layer upper bound taken from
    /// the manifests, but it accounts for tensor bytes and not for what MLX's
    /// allocator adds around each resident array. 256 MiB over a 5.87 GB tier
    /// is ~4.4%, comfortably above per-allocation rounding, and — like every
    /// other term — it is refused against rather than clamped into.
    public static let statedPinMarginBytes: UInt64 = 256 << 20

    /// The output head's own walk: the whole `[vocabulary, hidden]` BF16
    /// matrix, re-read in `logitChunkRows` windows on every pass
    /// (`docs/2026-08-14-decode-performance-audit.md` §2 P0-2).
    ///
    /// Since the dial grew a globals rung this is both columns of that rung:
    /// what a pass reads, and what pinning it costs. They are equal, because
    /// the pinned table is the stored BF16 table and the walk keeps widening
    /// per window.
    public static func outputHeadWalkBytes(_ config: DeepSeekV4Config) -> UInt64 {
        let elements = UInt64(max(0, config.vocabularySize))
            .multipliedReportingOverflow(by: UInt64(max(0, config.hiddenSize)))
        guard !elements.overflow else { return 0 }
        // BF16, two bytes an element.
        return elements.partialValue.multipliedReportingOverflow(by: 2).partialValue
    }

    /// One embedding row: `hidden × 2` BF16 bytes, which is what a decode token
    /// reads of a table the same size as the head.
    ///
    /// Stated because the ladder's answer for the embedding table depends on
    /// it: 1.06 GB resident against this per token is ~0.0000077 bytes saved
    /// per resident byte, so the table is censused and never ranked.
    public static func embeddingRowBytes(_ config: DeepSeekV4Config) -> UInt64 {
        UInt64(max(0, config.hiddenSize)).multipliedReportingOverflow(by: 2).partialValue
    }

    /// The globals rung, from the two `global00` tables a run actually holds.
    ///
    /// `census` comes from the unit's manifest — file lengths and declared
    /// shapes, no payload — so the columns describe the artifact on disk rather
    /// than the config's idea of it.
    public static func globals(
        _ census: DeepSeekV4GlobalArtifact.ManifestCensus
    ) -> DeepSeekV4MemoryDial.Globals {
        DeepSeekV4MemoryDial.Globals(
            headResidentBytes: census.headTableBytes,
            headReadBytesPerToken: census.headTableBytes,
            embeddingResidentBytes: census.embeddingTableBytes,
            embeddingReadBytesPerToken: census.embeddingRowBytes)
    }

    /// The floor in the three terms ``WorkingSetFloor`` names, from a product
    /// memory plan priced with an unbounded declared budget.
    ///
    /// V4's shape is not K3's — nothing here widens a layer to float32, so
    /// there is no "widest resident layer" — and the transient execution
    /// envelope takes that slot: it is the same quantity, the widest moment the
    /// run has evidence for, and it was measured rather than derived.
    ///
    /// The 2 GB product floor is deliberately absent. That is the product
    /// scale's policy; a stated run declares its own, and is admitted against
    /// its reservation plus its pins.
    public static func floor(pricing plan: DeepSeekV4ProductMemoryPlan) -> WorkingSetFloor? {
        let reserve = UInt64Accounting.saturatingSum([
            plan.retainedStateBytes, plan.replacementLayerStateBytes,
            plan.mlxCacheBytes, statedPinMarginBytes,
        ])
        guard !reserve.didOverflow else { return nil }
        return WorkingSetFloor(
            widestResidentLayerBytes: plan.transientExecutionBytes,
            expertPoolBytes: plan.expertPoolBytes,
            workingReserveBytes: reserve.value)
    }

    /// Everything a screen needs to draw V4's ladder for an artifact that is on
    /// disk and not running.
    public struct ArtifactProfile: Sendable, Equatable {
        public let census: DeepSeekV4MemoryDial.Census
        public let floor: WorkingSetFloor
        public let expertPoolBytes: UInt64
        public var layerCount: Int { census.layerReadBytes.count }

        public init(
            census: DeepSeekV4MemoryDial.Census, floor: WorkingSetFloor,
            expertPoolBytes: UInt64
        ) {
            self.census = census
            self.floor = floor
            self.expertPoolBytes = expertPoolBytes
        }
    }

    /// Read the census and price the floor for a fully verified V4 artifact.
    ///
    /// The floor is priced at the **product ceilings** — 512 prompt tokens, 64
    /// generated — and at the saturated MLX cache, because a ladder is a
    /// statement about every budget rather than about one request. Pricing it
    /// at the largest state and cache this product will ever hold is the
    /// conservative direction: a preset chosen off this ladder is never short
    /// of what the run then reserves.
    public static func inspect(_ artifact: ArtifactReference) throws -> ArtifactProfile {
        guard let authority = artifact.runtimeAuthority else {
            throw RunError.artifactNotReady(
                "the V4 memory dial requires complete rooted verification authority")
        }
        let evidence = authority.evidence
        guard let configurationIdentity = evidence.index.configuration else {
            throw RunError.artifactNotReady(
                "index.json does not bind a verified DeepSeek V4 config")
        }
        let configData = try authority.openFile(configurationIdentity.file).readAll(
            maximumBytes: 1 << 20)
        let config = try DeepSeekV4Config(json: configData)
        let indexData = try authority.openFile("index.json").readAll(maximumBytes: 8 << 20)
        let index = try DeepSeekV4ArtifactIndex(json: indexData, config: config)

        var readBytes = [UInt64]()
        var residentBytes = [UInt64]()
        var widestExpertStride: UInt64 = 0
        readBytes.reserveCapacity(index.mainLayerUnits.count)
        residentBytes.reserveCapacity(index.mainLayerUnits.count)
        for unit in index.mainLayerUnits {
            let manifest = try authority.openFile("\(unit.id)/manifest.json").readAll(
                maximumBytes: 8 << 20)
            let layer = try DeepSeekV4LayerArtifact.manifestCensus(
                manifestData: manifest, unitReference: unit.id)
            readBytes.append(layer.deterministicReadBytesPerPass)
            residentBytes.append(layer.pinnedResidentBytes)
            widestExpertStride = max(widestExpertStride, layer.widestExpertTileStrideBytes)
        }
        // The globals rung, from `global00`'s own manifest: one more metadata
        // read, and the only source that states what the two tables cost rather
        // than what the config implies they should.
        let globalUnit = index.globalUnit.id
        let globalManifest = try authority.openFile("\(globalUnit)/manifest.json").readAll(
            maximumBytes: 8 << 20)
        let globalCensus = try DeepSeekV4GlobalArtifact.manifestCensus(
            manifestData: globalManifest, unitReference: globalUnit)
        try authority.validateCurrentBinding()

        let census = try DeepSeekV4MemoryDial.Census(
            layerReadBytes: readBytes,
            layerResidentBytes: residentBytes,
            globals: globals(globalCensus),
            // Routed-expert reads are not addressed by this tier and are
            // reported by the run's own accounting, not projected here.
            expertBytesPerToken: 0)

        let pool = widestExpertStride.multipliedReportingOverflow(
            by: UInt64(DeepSeekV4EffectiveKnobs.defaultExpertPoolSlots))
        guard !pool.overflow else {
            throw RunError.artifactNotReady(
                "the V4 expert pool byte count exceeds UInt64.max")
        }
        let priced = try DeepSeekV4ProductMemoryBudget.plan(
            declaredBudgetBytes: .max,
            config: config,
            promptTokenCount: DeepSeekV4ProductMemoryBudget.maximumPromptTokens,
            maximumNewTokens: DeepSeekV4ProductMemoryBudget.maximumNewTokens,
            expertPoolBytes: pool.partialValue,
            mlxCacheBytes: DeepSeekV4EffectiveKnobs.maximumDefaultMLXCacheBytes,
            pinnedDeterministicBytes: 0,
            enforcesProductLimits: false)
        guard let floor = floor(pricing: priced) else {
            throw RunError.artifactNotReady(
                "the V4 working-set floor exceeds UInt64.max")
        }
        return ArtifactProfile(
            census: census, floor: floor, expertPoolBytes: pool.partialValue)
    }
}
