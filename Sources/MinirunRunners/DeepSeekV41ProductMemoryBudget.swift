import Foundation
import MinirunKit
import ModelAdapters

/// A saturating `UInt64` sum, spelled here because the byte arithmetic below
/// must refuse rather than trap: a nonsensical configuration is a run this
/// refuses to accept, not a crash.
struct PlanningUInt64Sum {
    private(set) var value: UInt64 = 0
    private(set) var didOverflow = false

    mutating func add(_ increment: UInt64) {
        let next = value.addingReportingOverflow(increment)
        value = next.overflow ? .max : next.partialValue
        didOverflow = didOverflow || next.overflow
    }
}

/// Product memory policy for the bounded DeepSeek V4.1 chat runtime.
///
/// The shape is V4's — a measured transient envelope, plus every exactly
/// derivable term added again rather than hidden inside it, plus explicit
/// headroom, and an identity that must partition the declared ceiling — because
/// the thing that policy protects against is the same: a future compatible
/// artifact with larger state or pool geometry must raise the floor *before*
/// acceptance rather than discover it during a chat.
///
/// What differs from V4 is every number, and one term.
public struct DeepSeekV41ProductMemoryPlan: Codable, Equatable, Sendable {
    public let declaredBudgetBytes: UInt64
    public let transientExecutionBytes: UInt64
    public let retainedStateBytes: UInt64
    public let replacementBlockStateBytes: UInt64
    public let expertPoolBytes: UInt64
    public let mlxCacheBytes: UInt64
    /// Dense block tensors and, when the plan reached the globals rung, the
    /// output head. Zero at the product scale, which pins nothing.
    public let pinnedDeterministicBytes: UInt64
    /// The routed gather's operands, held alive across a pass.
    ///
    /// Zero when the policy folds them into ``transientExecutionBytes`` — which
    /// is what the macOS policy does, because the arm that measured its
    /// envelope was an eleven-token prefill that held every one of them. A
    /// policy that bounds the live set states this term separately, because
    /// then it is *exactly derivable* rather than measured: three projections
    /// times the tokens whose operands are alive at once times
    /// `expertsPerToken` times one expert's half-tile.
    ///
    /// ADR 0015's rule, applied to the one term of a V4.1 prefill that grows
    /// with the prompt.
    public let liveGatherOperandBytes: UInt64
    public let productFloorPaddingBytes: UInt64
    public let requiredBudgetBytes: UInt64
    public let headroomBytes: UInt64

    public var isAdmitted: Bool { declaredBudgetBytes >= requiredBudgetBytes }

    /// Accepted plans partition the stated ceiling exactly.
    public var accountsForDeclaredBudget: Bool {
        guard isAdmitted else { return false }
        var total: UInt64 = 0
        for value in [
            transientExecutionBytes, retainedStateBytes, replacementBlockStateBytes,
            expertPoolBytes, mlxCacheBytes, pinnedDeterministicBytes,
            liveGatherOperandBytes, productFloorPaddingBytes, headroomBytes,
        ] {
            let next = total.addingReportingOverflow(value)
            guard !next.overflow else { return false }
            total = next.partialValue
        }
        return total == declaredBudgetBytes
    }

    /// Records written before the gather term existed decode with it at zero,
    /// which is what a macOS plan writes anyway. Spelled out rather than
    /// synthesized so a run record published by an earlier revision stays
    /// readable by the gate that compares against it.
    private enum CodingKeys: String, CodingKey {
        case declaredBudgetBytes, transientExecutionBytes, retainedStateBytes
        case replacementBlockStateBytes, expertPoolBytes, mlxCacheBytes
        case pinnedDeterministicBytes, liveGatherOperandBytes
        case productFloorPaddingBytes, requiredBudgetBytes, headroomBytes
    }

    public init(
        declaredBudgetBytes: UInt64, transientExecutionBytes: UInt64,
        retainedStateBytes: UInt64, replacementBlockStateBytes: UInt64,
        expertPoolBytes: UInt64, mlxCacheBytes: UInt64,
        pinnedDeterministicBytes: UInt64, liveGatherOperandBytes: UInt64,
        productFloorPaddingBytes: UInt64, requiredBudgetBytes: UInt64,
        headroomBytes: UInt64
    ) {
        self.declaredBudgetBytes = declaredBudgetBytes
        self.transientExecutionBytes = transientExecutionBytes
        self.retainedStateBytes = retainedStateBytes
        self.replacementBlockStateBytes = replacementBlockStateBytes
        self.expertPoolBytes = expertPoolBytes
        self.mlxCacheBytes = mlxCacheBytes
        self.pinnedDeterministicBytes = pinnedDeterministicBytes
        self.liveGatherOperandBytes = liveGatherOperandBytes
        self.productFloorPaddingBytes = productFloorPaddingBytes
        self.requiredBudgetBytes = requiredBudgetBytes
        self.headroomBytes = headroomBytes
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        declaredBudgetBytes = try container.decode(UInt64.self, forKey: .declaredBudgetBytes)
        transientExecutionBytes = try container.decode(
            UInt64.self, forKey: .transientExecutionBytes)
        retainedStateBytes = try container.decode(UInt64.self, forKey: .retainedStateBytes)
        replacementBlockStateBytes = try container.decode(
            UInt64.self, forKey: .replacementBlockStateBytes)
        expertPoolBytes = try container.decode(UInt64.self, forKey: .expertPoolBytes)
        mlxCacheBytes = try container.decode(UInt64.self, forKey: .mlxCacheBytes)
        pinnedDeterministicBytes = try container.decode(
            UInt64.self, forKey: .pinnedDeterministicBytes)
        liveGatherOperandBytes =
            try container.decodeIfPresent(UInt64.self, forKey: .liveGatherOperandBytes) ?? 0
        productFloorPaddingBytes = try container.decode(
            UInt64.self, forKey: .productFloorPaddingBytes)
        requiredBudgetBytes = try container.decode(UInt64.self, forKey: .requiredBudgetBytes)
        headroomBytes = try container.decode(UInt64.self, forKey: .headroomBytes)
    }
}

public enum DeepSeekV41ProductMemoryBudget {

    // MARK: - Platform policy

    /// A platform product boundary, kept as a value.
    ///
    /// K3's ``K3ProductMemoryBudget/Policy`` for V4.1, for the same two reasons
    /// it exists there: a boundary is a contract rather than a device-RAM
    /// heuristic, and keeping the policies as values lets the macOS suite
    /// verify the iPhone one without pretending it ran on an iPhone.
    ///
    /// What a V4.1 policy carries that K3's does not is a pair of decisions
    /// about the routed gather's operands, and they are **two** decisions:
    ///
    /// - ``boundsLiveOperands`` is about the *execution* — whether a token's
    ///   contribution is evaluated before the next token's operands are built;
    /// - ``liveGatherOperandTokens`` is about the *pricing* — whether the plan
    ///   adds the term explicitly or the measured envelope already contains it.
    ///
    /// The macOS policy bounds the execution and does **not** price the term,
    /// because its envelope was measured on an arm that held every operand of
    /// an eleven-token prefill; pricing it again would raise a floor that is
    /// already, after this change, a generous upper bound. The iPhone policy
    /// does both, because its envelope is stated bottom-up from measured terms
    /// and contains none of it.
    public struct Policy: Sendable, Equatable {
        /// What it is called in a record.
        public let name: String
        /// The stated envelope, measured on an arm and written down (ADR 0015).
        public let transientExecutionBytes: UInt64
        /// The transient envelope of a run that **pins**, which is the only
        /// kind of run the memory dial's ladder is ever consulted about.
        ///
        /// It is a policy term and not a dial constant because the dial's floor
        /// is a claim about *this platform's* run: a ladder priced with another
        /// platform's envelope offers rungs this device cannot hold, and a rung
        /// a phone cannot hold is the shape of the bug this field exists to
        /// stop. ``DeepSeekV41MemoryDialInputs/floor(pricing:policy:)`` reads
        /// it, and every screen reaches the ladder through that function.
        public let pinnedTransientExecutionBytes: UInt64
        public let minimumBudgetBytes: UInt64
        public let maximumPromptTokens: Int
        public let maximumNewTokens: Int
        /// Whether the routed gather and the output head's row windows are
        /// evaluated as they are built, so that at most
        /// ``liveGatherOperandTokens`` tokens' operands and one head window are
        /// alive at once.
        public let boundsLiveOperands: Bool
        /// How many tokens' gather operands the policy prices separately, or
        /// `nil` when the measured envelope already contains them.
        ///
        /// `nil` is not "no operands": it is "the arm that measured
        /// ``transientExecutionBytes`` held the whole prompt's, so pricing them
        /// again would count them twice".
        public let liveGatherOperandTokens: Int?
        public let expertReadAhead: Int
        public let expertPoolSlots: Int
        public let headWindowRows: Int
        /// The most a run may cross its declared budget before the refusal
        /// fires — the largest single allocation between two budget checks.
        public let budgetOvershootAllowanceBytes: UInt64
        public let isExperimental: Bool

        public init(
            name: String,
            transientExecutionBytes: UInt64,
            pinnedTransientExecutionBytes: UInt64,
            minimumBudgetBytes: UInt64,
            maximumPromptTokens: Int,
            maximumNewTokens: Int,
            boundsLiveOperands: Bool,
            liveGatherOperandTokens: Int?,
            expertReadAhead: Int,
            expertPoolSlots: Int,
            headWindowRows: Int,
            budgetOvershootAllowanceBytes: UInt64,
            isExperimental: Bool
        ) {
            precondition(transientExecutionBytes > 0)
            precondition(
                pinnedTransientExecutionBytes >= transientExecutionBytes,
                "a run that pins holds at least what a run that streams holds")
            precondition(minimumBudgetBytes > 0)
            precondition(maximumPromptTokens > 0 && maximumNewTokens > 0)
            precondition(expertReadAhead >= 1 && headWindowRows >= 1)
            precondition(expertPoolSlots >= 3 * expertReadAhead + 1)
            precondition((liveGatherOperandTokens ?? 1) >= 1)
            precondition(
                liveGatherOperandTokens == nil || boundsLiveOperands,
                "a policy that prices a bounded live set must bound it")
            self.name = name
            self.transientExecutionBytes = transientExecutionBytes
            self.pinnedTransientExecutionBytes = pinnedTransientExecutionBytes
            self.minimumBudgetBytes = minimumBudgetBytes
            self.maximumPromptTokens = maximumPromptTokens
            self.maximumNewTokens = maximumNewTokens
            self.boundsLiveOperands = boundsLiveOperands
            self.liveGatherOperandTokens = liveGatherOperandTokens
            self.expertReadAhead = expertReadAhead
            self.expertPoolSlots = expertPoolSlots
            self.headWindowRows = headWindowRows
            self.budgetOvershootAllowanceBytes = budgetOvershootAllowanceBytes
            self.isExperimental = isExperimental
        }
    }

    /// The Mac boundary. Every priced number is phase 3's, to the byte.
    ///
    /// `transientExecutionBytes` is still 3.2 GB, the floor is still 3.4 GB, the
    /// pool is still twenty slots and the head window is still 4,096 rows, so a
    /// plan built from this policy partitions the declared budget exactly as
    /// `docs/experiments/data/2026-09-11-v41-phase3-runner/` recorded it:
    /// 3,200,000,000 + 5,427,456 + 195,072 + 125,337,600 = 3,330,960,128 inside
    /// 3,400,000,000, padding 69,039,872.
    ///
    /// What **did** change is `boundsLiveOperands`, and it changes the execution
    /// rather than the arithmetic. The 2026-09-11 term measurements found that
    /// the whole of this arm's 3,082,370,944 B of transient is two lazy
    /// accumulations — 1,204,127,936 B of routed-gather operands over eleven
    /// prompt tokens and 2,010,333,184 B of output-head row windows — and that
    /// both grow with something a product prompt is allowed to make larger. At
    /// this policy's own 512-token prompt limit the gather term alone would be
    /// 57,755,566,080 B, which is not a budget any Mac has. Bounding them is
    /// bit-identical (`DeepSeekV41TransientTermTests`) and leaves 3.2 GB as a
    /// bound the run now comes nowhere near: conservative, not wrong. Making it
    /// *right* needs an arm, and an arm needs the container.
    ///
    /// `pinnedTransientExecutionBytes` is the dial's own term and is 5.2 GB
    /// here for the reason `DeepSeekV41MemoryDialInputs` records: phase 3's
    /// stated arm, which held all forty blocks and the head, had a transient of
    /// 5,161,820,992 B — 1.78x the floor arm's, because the matmuls resident
    /// weights feed have intermediates of their own. It moved from a constant
    /// on the dial into the policy and its Mac value did not change by a byte.
    public static let macOSProductPolicy = Policy(
        name: "macOS",
        transientExecutionBytes: 3_200_000_000,
        pinnedTransientExecutionBytes: 5_200_000_000,
        minimumBudgetBytes: 3_400_000_000,
        maximumPromptTokens: 512,
        maximumNewTokens: 64,
        boundsLiveOperands: true,
        liveGatherOperandTokens: nil,
        expertReadAhead: 6,
        expertPoolSlots: 20,
        headWindowRows: 4_096,
        budgetOvershootAllowanceBytes: 192_000_000,
        isExperimental: false)

    /// The iPhone boundary.
    ///
    /// Stated after the owner's iPhone 16 Pro was killed by Jetsam twice, in
    /// prefill, at the Mac floor — `vm-pageshortage`, 5,156 MB and 5,044 MB
    /// resident against a 3.4 GB declared budget, on a device that gives one
    /// app about 5 GB.
    ///
    /// Unlike the Mac's, this envelope is built **bottom-up** out of what a
    /// bounded pass holds at its widest moment, because subtracting the two
    /// bounded terms from the Mac arm's measurement leaves nothing to state:
    /// those two terms account for the whole of it.
    ///
    /// | term | bytes | where from |
    /// | --- | ---: | --- |
    /// | widest dense block, expanded (block 14) | 353,595,608 | published census |
    /// | one bounded head window set, 1,024 rows | 52,314,112 | measured |
    /// | **stated envelope** | **1,600,000,000** | 3.9x the two above |
    /// | one token's gather operands | 112,803,840 | priced, exactly |
    /// | bounded state at 512 + 64 positions | 7,976,448 | derived |
    /// | one-block state replacement | 867,072 | derived |
    /// | routed-expert pool, 20 slots | 125,337,600 | derived |
    /// | **floor** | **1,900,000,000** | 1,846,117,888 rounded up |
    ///
    /// The 3.9x is margin and is named as margin rather than buried in the
    /// envelope: no V4.1 arm has run on a phone, nothing has yet measured a
    /// 512-token prefill's attention intermediates or the block reader's own
    /// staging buffers, and a phone answers an overshoot with `SIGKILL` rather
    /// than with a refusal. The peak a short-prompt chat is predicted to add is
    /// about 0.61 GB; if the confirming arm finds that, this number should come
    /// **down** rather than stay a constant nobody can account for.
    ///
    /// The read-ahead and the pool are the Mac's deliberately: phase 3 measured
    /// 1.04 s of a 3.02 s expert phase still waiting on storage at a read-ahead
    /// of six, so the pool is not oversized, and its 125 MB is 6.6% of this
    /// floor. The head window is 1,024 rather than 4,096 because bounding makes
    /// the window *the* live set — 52,314,112 B against 167,919,616 B, measured
    /// — where before it was a slice of a table that stayed resident either way.
    ///
    /// **Experimental**: every number above is arithmetic and bench
    /// measurement. None of it is a device arm.
    ///
    /// ## The pinned envelope, and why it is the Mac's
    ///
    /// `pinnedTransientExecutionBytes` is 5.2 GB here too, and that is not a
    /// copied constant — it is the statement that **no phone has ever pinned
    /// anything**. The only measurement of what a pinning V4.1 run holds is the
    /// Mac's stated arm, so it is the only number the ladder may be priced
    /// with; scaling it down by the Mac's own 1.78x ratio would invent a phone
    /// measurement out of a Mac one and would make rungs *reachable* on a
    /// device that has just been killed by Jetsam at a smaller budget.
    ///
    /// The consequence is deliberate and is the behaviour the dial wants: the
    /// ladder's first rung sits above anything `os_proc_available_memory()`
    /// offers an iPhone, so Balanced and Generous are drawn **disabled with
    /// their deficit** and Floor — this policy's 1.9 GB, which pins nothing —
    /// is the one position a phone can take. When a phone arm eventually pins a
    /// block, this number comes down and the ladder opens by itself.
    public static let iOSProductPolicy = Policy(
        name: "iOS",
        transientExecutionBytes: 1_600_000_000,
        pinnedTransientExecutionBytes: 5_200_000_000,
        minimumBudgetBytes: 1_900_000_000,
        maximumPromptTokens: 512,
        maximumNewTokens: 64,
        boundsLiveOperands: true,
        liveGatherOperandTokens: 1,
        expertReadAhead: 6,
        expertPoolSlots: 20,
        headWindowRows: 1_024,
        budgetOvershootAllowanceBytes: 192_000_000,
        isExperimental: true)

    public static var currentPolicy: Policy {
        #if os(iOS)
            iOSProductPolicy
        #else
            macOSProductPolicy
        #endif
    }

    /// The widest moment a V4.1 pass has evidence for.
    ///
    /// ADR 0015 makes this a **stated** term: measured on a real arm and
    /// written down, never derived from the config. It is larger than V4's
    /// 1.8 GB for two reasons that are properties of this checkpoint rather
    /// than of this runner:
    ///
    /// - a V4.1 block's dense tensors are 170 MB, and the two Engram blocks are
    ///   328 and 344 MB because `engram.wkv` is `[25600, 6144]` FP8 — 157 MB in
    ///   one matrix, which is transient on a streamed block and resident on a
    ///   pinned one;
    /// - the routed gather stacks `expertsPerToken` half-tiles per projection
    ///   before the quantized matmul, which is the term ADR 0015 priced for V4
    ///   at a different tile size.
    ///
    /// **Measured**, at the product limits, on the arm this number is named
    /// after: a streaming run that pins nothing added 3,028,635,520 B above the
    /// floor it started from, of which 125,337,600 B is the pool and 5,421,056 B
    /// the bounded state, leaving **2,897,876,864 B** of transient. 3.2 GB
    /// rounds above that observation, as V4's 1.8 GB rounds above its own
    /// 1,547,716,524 B.
    ///
    /// Recorded in `docs/experiments/2026-09-11-v41-phase3-runner.md`.
    ///
    /// Since 2026-09-11 this is the **current platform policy's** term, and the
    /// macOS policy's value is the one this comment describes. What it does not
    /// contain, on a policy that bounds the live set, is the routed gather's
    /// operands: see ``Policy/liveGatherOperandTokens``.
    public static var transientExecutionBytes: UInt64 {
        currentPolicy.transientExecutionBytes
    }

    /// The current platform's pinned transient envelope. See
    /// ``Policy/pinnedTransientExecutionBytes``.
    public static var pinnedTransientExecutionBytes: UInt64 {
        currentPolicy.pinnedTransientExecutionBytes
    }

    /// The product floor.
    ///
    /// Derived the way V4's 2 GB was (ADR 0015/0016): the transient envelope
    /// plus the terms a product chat always reserves — the routed-expert pool
    /// at the default slot count and the bounded generation state — rounded up,
    /// with the arithmetic stated rather than the number chosen. At the default
    /// twenty slots a V4.1 pool is 20 x 6,266,880 = 125,337,600 B (one expert's
    /// half of a pair tile, per ADR 0021 §5) and the bounded state at 512 + 64
    /// positions is 5,421,056 B, so the floor is 3.20 + 0.125 + 0.005 =
    /// 3.33 GB, rounded to 3.4.
    ///
    /// The first stated number was 2.9 GB, from the pre-measurement estimate.
    /// The floor arm added 3.03 GB and was refused by its own budget, which is
    /// the refusal working: a floor is not a guess that a run may exceed.
    ///
    /// Unlike V4's, this floor does **not** buy a chat that pins anything: at
    /// 3.4 GB every block is streamed and the head is walked in windows, and a
    /// decode token reads 11.7 GB. What it buys is a chat that runs, which is
    /// what a floor is for.
    public static var minimumBudgetBytes: UInt64 { currentPolicy.minimumBudgetBytes }

    /// Prompt and reply ceilings.
    ///
    /// 512 and 64 are V4's, kept because the arithmetic that set them is the
    /// same here and not because they were copied: the exact prefill runs every
    /// block over every prompt token, and the KV a V4.1 run retains at 576
    /// positions is 5.6 MB — far from the binding constraint. What binds is
    /// *prefill time*, not memory: a token's routed set is its own, so a
    /// 512-token prompt reads its blocks' expert tiles 512 times over.
    /// `docs/experiments/2026-09-11-v41-phase3-runner.md` measures it.
    public static var maximumPromptTokens: Int { currentPolicy.maximumPromptTokens }
    public static var maximumNewTokens: Int { currentPolicy.maximumNewTokens }

    /// The routed gather's live operands, exactly.
    ///
    /// Three projections — `w1`, `w3`, `w2` — each stacking `expertsPerToken`
    /// half-tiles, for every token whose operands are alive at once.
    /// `stack` is called once per token per projection per block and nothing
    /// evaluates between the calls, so the count of live tokens is the count of
    /// tokens the pass has reached: the whole prompt when nothing bounds it,
    /// and ``Policy/liveGatherOperandTokens`` when something does.
    ///
    /// Returns zero when the policy folds the term into its measured envelope,
    /// which is what stops it being counted twice on macOS.
    public static func liveGatherOperandBytes(
        config: DeepSeekV41Config,
        promptTokenCount: Int,
        expertTileStrideBytes: UInt64,
        policy: Policy = currentPolicy
    ) throws -> UInt64 {
        guard let liveTokens = policy.liveGatherOperandTokens else { return 0 }
        guard expertTileStrideBytes > 0 else {
            throw DeepSeekV41Error.configuration(
                "the \(policy.name) V4.1 policy prices the routed gather separately and "
                    + "needs the artifact's expert tile stride to do it")
        }
        var widestExperts = 0
        for block in 0..<config.numberOfLayers {
            widestExperts = max(widestExperts, try config.expertsPerToken(block: block))
        }
        let tokens = UInt64(max(1, min(liveTokens, max(1, promptTokenCount))))
        let projections = UInt64(DeepSeekV41ExpertProjection.allCases.count)
        var product = PlanningUInt64Sum()
        product.add(projections * tokens * UInt64(widestExperts) * expertTileStrideBytes)
        guard !product.didOverflow else {
            throw DeepSeekV41Error.configuration(
                "the V4.1 routed gather's live operands exceed UInt64.max")
        }
        return product.value
    }

    public static func plan(
        declaredBudgetBytes: UInt64,
        config: DeepSeekV41Config,
        promptTokenCount: Int,
        maximumNewTokens: Int,
        expertPoolBytes: UInt64,
        mlxCacheBytes: UInt64,
        pinnedDeterministicBytes: UInt64 = 0,
        expertTileStrideBytes: UInt64 = 0,
        policy: Policy = currentPolicy,
        enforcesProductLimits: Bool = true
    ) throws -> DeepSeekV41ProductMemoryPlan {
        if enforcesProductLimits {
            guard promptTokenCount <= policy.maximumPromptTokens else {
                throw DeepSeekV41Error.configuration(
                    "product prompt has \(promptTokenCount) tokens; the verified limit is "
                        + "\(policy.maximumPromptTokens)")
            }
            guard maximumNewTokens <= policy.maximumNewTokens else {
                throw DeepSeekV41Error.configuration(
                    "product response requests \(maximumNewTokens) tokens; the verified "
                        + "limit is \(policy.maximumNewTokens)")
            }
        }
        let state = try stateBytes(
            config: config, promptTokenCount: promptTokenCount,
            maximumNewTokens: maximumNewTokens)
        let gather = try liveGatherOperandBytes(
            config: config, promptTokenCount: promptTokenCount,
            expertTileStrideBytes: expertTileStrideBytes, policy: policy)
        var exact = policy.transientExecutionBytes
        for (name, value) in [
            ("generation state", state.total),
            ("one-block state replacement", state.widestBlock),
            ("routed-expert pool", expertPoolBytes),
            ("MLX cache", mlxCacheBytes),
            ("pinned deterministic weights", pinnedDeterministicBytes),
            ("live routed-gather operands", gather),
        ] {
            let next = exact.addingReportingOverflow(value)
            guard !next.overflow else {
                throw DeepSeekV41Error.configuration(
                    "V4.1 \(name) makes the product memory requirement exceed UInt64.max")
            }
            exact = next.partialValue
        }
        let required = max(policy.minimumBudgetBytes, exact)
        let plan = DeepSeekV41ProductMemoryPlan(
            declaredBudgetBytes: declaredBudgetBytes,
            transientExecutionBytes: policy.transientExecutionBytes,
            retainedStateBytes: state.total,
            replacementBlockStateBytes: state.widestBlock,
            expertPoolBytes: expertPoolBytes,
            mlxCacheBytes: mlxCacheBytes,
            pinnedDeterministicBytes: pinnedDeterministicBytes,
            liveGatherOperandBytes: gather,
            productFloorPaddingBytes: required - exact,
            requiredBudgetBytes: required,
            headroomBytes: declaredBudgetBytes >= required
                ? declaredBudgetBytes - required : 0)
        guard !plan.isAdmitted || plan.accountsForDeclaredBudget else {
            throw DeepSeekV41Error.configuration(
                "the V4.1 product memory plan does not account for the declared budget")
        }
        return plan
    }

    /// Exactly what a bounded V4.1 run retains, from the configuration and the
    /// request — never from a measurement, so a compatible artifact with a
    /// wider window or a longer context raises the requirement before it runs.
    ///
    /// Four caches, and the reason each is the size it is:
    ///
    /// - the **sliding-window ring**, `window_size x head_dim` bfloat16, on
    ///   every block. It is a ring, so it does not grow with the context;
    /// - the **main KV** latents, on the `kv_source_layers` **only** — every
    ///   other block reads the most recent owner's cache and holds none of its
    ///   own. An encoder source at `compress_ratio` 2 holds one entry per two
    ///   positions; the decoder source at 1 holds one per position;
    /// - the **indexer keys** beside them, `index_head_dim` wide;
    /// - the compressor's **partial group**, `ratio x head_dim` float32 twice
    ///   (the latent and its score), on the ratio-2 sources.
    public static func stateBytes(
        config: DeepSeekV41Config,
        promptTokenCount: Int,
        maximumNewTokens: Int
    ) throws -> (total: UInt64, widestBlock: UInt64) {
        guard promptTokenCount >= 1, maximumNewTokens >= 1 else {
            throw DeepSeekV41Error.configuration(
                "a bounded V4.1 run needs at least one prompt token and one new token")
        }
        let positions = UInt64(promptTokenCount + maximumNewTokens - 1)
        guard positions <= UInt64(config.maximumPositionCount) else {
            throw DeepSeekV41Error.configuration(
                "\(positions) positions exceed the verified V4.1 position limit "
                    + "\(config.maximumPositionCount)")
        }
        let headDimension = UInt64(config.attentionHeadDimension)
        let indexDimension = UInt64(config.indexHeadDimension)
        var total = PlanningUInt64Sum()
        var widest: UInt64 = 0
        for block in 0..<config.numberOfLayers {
            var perBlock = PlanningUInt64Sum()
            // The ring, on every block: bounded by the window, not the context.
            perBlock.add(UInt64(config.slidingWindow) * headDimension * 2)
            let ratio = try config.compressionRatio(block: block)
            if ratio > 0, config.keyValueSourceLayers.contains(block) {
                let rows = (positions + UInt64(ratio) - 1) / UInt64(ratio)
                perBlock.add(rows * headDimension * 2)
                perBlock.add(rows * indexDimension * 2)
                if ratio > 1 {
                    // The partial group: the latent and its gate score, both
                    // float32, both `ratio` entries wide.
                    perBlock.add(2 * UInt64(ratio) * headDimension * 4)
                }
            }
            guard !perBlock.didOverflow else {
                throw DeepSeekV41Error.configuration(
                    "the V4.1 generation state of block \(block) exceeds UInt64.max")
            }
            widest = max(widest, perBlock.value)
            total.add(perBlock.value)
        }
        guard !total.didOverflow else {
            throw DeepSeekV41Error.configuration(
                "the V4.1 generation state exceeds UInt64.max")
        }
        return (total.value, widest)
    }
}
