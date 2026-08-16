import Foundation
import ModelAdapters

/// Product memory policy for the bounded DeepSeek V4 chat runtime.
///
/// The original adopting-buffer path reached 3,865,535,104 bytes at the
/// 512-token product limit. After the product switched to eager expert copies,
/// released pager leases before compute, and reclaimed evaluated transient
/// storage, the same real-artifact arm observed a 1,446,793,408-byte process
/// footprint and a 1,547,716,524-byte MLX peak. The 1.8 GB transient envelope
/// rounds above the larger observation. Exact request-derived generation
/// state, the admitted routed-expert pool and any explicit MLX cache are then
/// added again rather than being hidden inside it. This deliberate overlap
/// makes a future compatible artifact with larger state or pool geometry raise
/// the floor before acceptance.
public struct DeepSeekV4ProductMemoryPlan: Codable, Equatable, Sendable {
    public let declaredBudgetBytes: UInt64
    public let transientExecutionBytes: UInt64
    public let retainedStateBytes: UInt64
    public let replacementLayerStateBytes: UInt64
    public let expertPoolBytes: UInt64
    public let mlxCacheBytes: UInt64

    /// Deterministic layer weights the memory dial holds resident for the run.
    ///
    /// **Zero at the product scale**, which is what the 2 GB floor is defined
    /// against and what a chat that never touched the dial carries. Only a
    /// `.stated` run pins, and it pins from its own declared headroom: this
    /// term is exactly the residency ``DeepSeekV4MemoryDial`` planned, so a
    /// budget that cannot fund the pins is refused at acceptance by the same
    /// `isAdmitted` gate as any other term rather than being quietly trimmed.
    public let pinnedDeterministicBytes: UInt64
    public let productFloorPaddingBytes: UInt64
    public let requiredBudgetBytes: UInt64
    public let headroomBytes: UInt64

    public var isAdmitted: Bool { declaredBudgetBytes >= requiredBudgetBytes }

    /// Accepted plans partition the stated ceiling exactly. Refused plans do
    /// not pretend that a negative headroom exists.
    public var accountsForDeclaredBudget: Bool {
        guard isAdmitted else { return false }
        var total: UInt64 = 0
        for value in [
            transientExecutionBytes, retainedStateBytes,
            replacementLayerStateBytes, expertPoolBytes, mlxCacheBytes,
            pinnedDeterministicBytes,
            productFloorPaddingBytes, headroomBytes,
        ] {
            let next = total.addingReportingOverflow(value)
            guard !next.overflow else { return false }
            total = next.partialValue
        }
        return total == declaredBudgetBytes
    }
}

public enum DeepSeekV4ProductMemoryBudget {
    public static let observedMaximumPromptFootprintBytes: UInt64 = 1_446_793_408
    public static let observedMaximumPromptMLXPeakBytes: UInt64 = 1_547_716_524
    public static let observedMaximumPromptPeakBytes: UInt64 = max(
        observedMaximumPromptFootprintBytes, observedMaximumPromptMLXPeakBytes)
    public static let transientExecutionBytes: UInt64 = 1_800_000_000
    public static let minimumBudgetBytes: UInt64 = 2_000_000_000
    public static let maximumPromptTokens = 512
    public static let maximumNewTokens = 64

    public static func requiredBudgetBytes(
        config: DeepSeekV4Config,
        promptTokenCount: Int,
        maximumNewTokens: Int,
        expertPoolBytes: UInt64,
        mlxCacheBytes: UInt64,
        pinnedDeterministicBytes: UInt64 = 0
    ) throws -> UInt64 {
        try plan(
            declaredBudgetBytes: .max,
            config: config,
            promptTokenCount: promptTokenCount,
            maximumNewTokens: maximumNewTokens,
            expertPoolBytes: expertPoolBytes,
            mlxCacheBytes: mlxCacheBytes,
            pinnedDeterministicBytes: pinnedDeterministicBytes
        ).requiredBudgetBytes
    }

    /// Build the exact product memory contract before runner acceptance.
    ///
    /// `retainedStateBytes` is one complete causal state generation.
    /// `replacementLayerStateBytes` is the only old/new overlap permitted by
    /// the consuming session decoder. `pinnedDeterministicBytes` is the memory
    /// dial's resident weight tier and is zero at the product scale. The
    /// remainder is explicit headroom; no part of a V4 plan is an unnamed
    /// allocation.
    ///
    /// `enforcesProductLimits` is what distinguishes the two scales. The
    /// product path keeps the 512-prompt and 64-token ceilings it has evidence
    /// for; a `.stated` caller has declared its own budget and states its own
    /// horizon, and the seven-term identity is checked either way — the dial
    /// must never be able to buy a pin by loosening the accounting.
    public static func plan(
        declaredBudgetBytes: UInt64,
        config: DeepSeekV4Config,
        promptTokenCount: Int,
        maximumNewTokens: Int,
        expertPoolBytes: UInt64,
        mlxCacheBytes: UInt64,
        pinnedDeterministicBytes: UInt64 = 0,
        enforcesProductLimits: Bool = true
    ) throws -> DeepSeekV4ProductMemoryPlan {
        if enforcesProductLimits {
            try validateProductLimits(
                promptTokenCount: promptTokenCount, maximumNewTokens: maximumNewTokens)
        }
        return try build(
            declaredBudgetBytes: declaredBudgetBytes,
            config: config,
            promptTokenCount: promptTokenCount,
            maximumNewTokens: maximumNewTokens,
            expertPoolBytes: expertPoolBytes,
            mlxCacheBytes: mlxCacheBytes,
            pinnedDeterministicBytes: pinnedDeterministicBytes)
    }

    private static func validateProductLimits(
        promptTokenCount: Int, maximumNewTokens: Int
    ) throws {
        guard promptTokenCount <= maximumPromptTokens else {
            throw DeepSeekV4Error.configuration(
                "product prompt has \(promptTokenCount) tokens; the verified limit is "
                    + "\(maximumPromptTokens)")
        }
        guard maximumNewTokens <= Self.maximumNewTokens else {
            throw DeepSeekV4Error.configuration(
                "product response requests \(maximumNewTokens) tokens; the verified limit is "
                    + "\(Self.maximumNewTokens)")
        }
    }

    private static func build(
        declaredBudgetBytes: UInt64,
        config: DeepSeekV4Config,
        promptTokenCount: Int,
        maximumNewTokens: Int,
        expertPoolBytes: UInt64,
        mlxCacheBytes: UInt64,
        pinnedDeterministicBytes: UInt64
    ) throws -> DeepSeekV4ProductMemoryPlan {
        let stateBytes = try DeepSeekV4GenerationMemoryBudget.logicalStateBytes(
            config: config,
            promptTokenCount: promptTokenCount,
            maximumNewTokens: maximumNewTokens)
        let replacementBytes = try DeepSeekV4GenerationMemoryBudget.maximumLayerStateBytes(
            config: config,
            promptTokenCount: promptTokenCount,
            maximumNewTokens: maximumNewTokens)
        var exactTerms = transientExecutionBytes
        for (name, value) in [
            ("generation state", stateBytes),
            ("one-layer state replacement", replacementBytes),
            ("routed-expert pool", expertPoolBytes),
            ("MLX cache", mlxCacheBytes),
            ("pinned deterministic weights", pinnedDeterministicBytes),
        ] {
            let next = exactTerms.addingReportingOverflow(value)
            guard !next.overflow else {
                throw DeepSeekV4Error.configuration(
                    "V4 \(name) makes the product memory requirement exceed UInt64.max")
            }
            exactTerms = next.partialValue
        }
        let required = max(minimumBudgetBytes, exactTerms)
        let padding = required - exactTerms
        let headroom = declaredBudgetBytes >= required
            ? declaredBudgetBytes - required : 0
        let result = DeepSeekV4ProductMemoryPlan(
            declaredBudgetBytes: declaredBudgetBytes,
            transientExecutionBytes: transientExecutionBytes,
            retainedStateBytes: stateBytes,
            replacementLayerStateBytes: replacementBytes,
            expertPoolBytes: expertPoolBytes,
            mlxCacheBytes: mlxCacheBytes,
            pinnedDeterministicBytes: pinnedDeterministicBytes,
            productFloorPaddingBytes: padding,
            requiredBudgetBytes: required,
            headroomBytes: headroom)
        guard !result.isAdmitted || result.accountsForDeclaredBudget else {
            throw DeepSeekV4Error.configuration(
                "V4 product memory plan does not account for the declared budget")
        }
        return result
    }
}
