import Foundation
import MinirunKit
import ModelAdapters
import StorageCore

public struct K3ArtifactRuntimeProfile: Sendable, Equatable {
    public let census: ArtifactCensus
    public let generationState: K3GenerationStateMemory

    public init(census: ArtifactCensus, generationState: K3GenerationStateMemory) {
        self.census = census
        self.generationState = generationState
    }
}

/// The product-chat memory terms above the currently widened layer and routed
/// expert pool.
///
/// The supported Mac policy and bounded iPhone policy deliberately differ.
/// The Mac boundary carries the real multi-pass observations from 2026-08-12.
/// The iPhone tier is constrained to the previously completed 5.80 GB device
/// boundary and two output tokens while its post-MLA-cache product path is
/// validated. Both policies retain the exact config-derived generation state;
/// neither turns an observed process footprint into an unexplained universal
/// constant.
public enum K3ProductMemoryBudget {
    /// A platform product boundary is an explicit contract, not a device-RAM
    /// heuristic. Keeping the policies as values lets the macOS test suite
    /// verify the iPhone boundary without pretending it ran on an iPhone.
    public struct Policy: Sendable, Equatable {
        public let minimumBudgetBytes: UInt64
        public let runtimeAndAllocatorReserveBytes: UInt64
        public let maximumNewTokens: Int
        public let isExperimental: Bool

        public init(
            minimumBudgetBytes: UInt64,
            runtimeAndAllocatorReserveBytes: UInt64,
            maximumNewTokens: Int,
            isExperimental: Bool
        ) {
            precondition(minimumBudgetBytes > 0)
            precondition(maximumNewTokens > 0)
            self.minimumBudgetBytes = minimumBudgetBytes
            self.runtimeAndAllocatorReserveBytes = runtimeAndAllocatorReserveBytes
            self.maximumNewTokens = maximumNewTokens
            self.isExperimental = isExperimental
        }
    }

    /// The supported Mac chat boundary. Its 2.1 GB residual comes from the
    /// real multi-pass product observations documented above.
    public static let macOSProductPolicy = Policy(
        minimumBudgetBytes: 8_000_000_000,
        runtimeAndAllocatorReserveBytes: 2_100_000_000,
        maximumNewTokens: 64,
        isExperimental: false)

    /// The deliberately narrow iPhone tier authorized by the owner after the
    /// real 5.80 GB one-token device arm completed twice with matching logits.
    /// It is not a generic force-run switch: verification, the derived memory
    /// plan, and the runner's fail-closed budget gate remain mandatory.
    ///
    /// The stated 500 MB allocator/transient headroom plus the exact
    /// 128-position KDA/MLA state produces
    /// a 5,711,193,344 B arithmetic floor with the current widest layer and
    /// expert pool, leaving 88,806,656 B inside the stated 5.80 GB boundary.
    /// Two output tokens bound the first unmeasured retained-state transition.
    public static let iOSExperimentalPolicy = Policy(
        minimumBudgetBytes: 5_800_000_000,
        runtimeAndAllocatorReserveBytes: 500_000_000,
        maximumNewTokens: 2,
        isExperimental: true)

    public static var currentPolicy: Policy {
        #if os(iOS)
            iOSExperimentalPolicy
        #else
            macOSProductPolicy
        #endif
    }

    /// App/process baseline, MLX allocator headroom, globals and transient
    /// output work not represented by the layer or expert-pool terms.
    ///
    /// Each platform policy states this term explicitly. The Mac uses the
    /// conservative 2.1 GB residual of its multi-pass product observations;
    /// the bounded iPhone tier states 500 MB and limits generation to two
    /// tokens. Neither value is a speed claim.
    public static var runtimeAndAllocatorReserveBytes: UInt64 {
        currentPolicy.runtimeAndAllocatorReserveBytes
    }

    /// The default plan protects a modest chat without reallocating its promise
    /// for every short prompt. Longer histories use the exact linear MLA term
    /// and are refused by name until the user states the larger budget.
    public static let defaultCachedPositions = 128

    /// The supported Mac boundary rounds above its 7,311,193,344 B exact
    /// 128-position floor. The bounded iPhone policy has its own explicit
    /// reserve and a 5,711,193,344 B arithmetic floor. A longer prompt still
    /// raises the request-specific floor and is refused rather than clamped.
    public static var minimumBudgetBytes: UInt64 { currentPolicy.minimumBudgetBytes }

    public static var maximumNewTokens: Int { currentPolicy.maximumNewTokens }

    public static var isExperimental: Bool { currentPolicy.isExperimental }

    /// Current flagship config: 474,808,320 B fixed KDA state plus 55,296 B per
    /// MLA position, projected for 128 positions. The platform policy's stated
    /// runtime/allocator term is added separately.
    public static let flagshipDefaultGenerationStateBytes: UInt64 = 481_886_208

    public static var flagshipDefaultWorkingReserveBytes: UInt64 {
        defaultWorkingReserveBytes(for: currentPolicy)
    }

    public static func defaultWorkingReserveBytes(for policy: Policy) -> UInt64 {
        let sum = policy.runtimeAndAllocatorReserveBytes.addingReportingOverflow(
            flagshipDefaultGenerationStateBytes)
        precondition(!sum.overflow, "the K3 product working reserve must fit UInt64")
        return sum.partialValue
    }

    public static func cachedPositions(
        promptTokenCount: Int, maximumNewTokens: Int
    ) throws -> Int {
        guard promptTokenCount > 0 else {
            throw TinyK3Error.configuration("a product prompt has no token ids")
        }
        guard maximumNewTokens > 0 else {
            throw TinyK3Error.configuration("maximumNewTokens is not positive")
        }
        // The final generated token is not fed back into another pass. A turn
        // that may emit N tokens therefore caches prompt + N - 1 positions.
        let generated = maximumNewTokens - 1
        let sum = promptTokenCount.addingReportingOverflow(generated)
        guard !sum.overflow else {
            throw TinyK3Error.configuration("the projected cached position count overflows Int")
        }
        return max(defaultCachedPositions, sum.partialValue)
    }

    public static func workingReserveBytes(
        state: K3GenerationStateMemory, promptTokenCount: Int,
        maximumNewTokens: Int, policy: Policy = currentPolicy
    ) throws -> UInt64 {
        let positions = try cachedPositions(
            promptTokenCount: promptTokenCount, maximumNewTokens: maximumNewTokens)
        let generation = try state.bytes(cachedPositions: positions)
        guard let reserve = UInt64Accounting.checkedSum(
            [policy.runtimeAndAllocatorReserveBytes, generation])
        else {
            throw TinyK3Error.configuration("the product working reserve exceeds UInt64.max")
        }
        return reserve
    }

    public static func defaultWorkingReserveBytes(
        state: K3GenerationStateMemory, policy: Policy = currentPolicy
    ) throws -> UInt64 {
        try workingReserveBytes(
            state: state, promptTokenCount: 1,
            maximumNewTokens: defaultCachedPositions, policy: policy)
    }
}
