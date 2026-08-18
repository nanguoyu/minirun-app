import Foundation

/// A drafter that costs nothing: it proposes the token that followed the most
/// recent earlier occurrence of the last `n` ids.
///
/// ADR 0017 chose this shape deliberately. The two halves of draft-and-verify
/// are separable and very unequally priced; the *verify* half is the reusable
/// one, and pairing it with a drafter that has no weights, no residency and no
/// budget consequence is what lets the verify half be measured on the real
/// artifact at an unchanged residency plan. Nothing here reads a model.
///
/// The proposal is a pure function of the ids emitted so far, which is why this
/// type is a value and why the acceptance rate of a finished run can be
/// computed from that run's own record without executing anything.
public struct DeepSeekV4PromptLookupDrafter: Equatable {
    /// The n-gram widths tried, in order. ADR 0017 asks for 3 then 2.
    public let orders: [Int]

    /// Ids seen so far: the prompt, then every emitted token.
    private var sequence: [Int] = []
    /// For each order, the last index at which each n-gram *started*. The value
    /// is the start, so the proposal is `sequence[start + n]`; keeping the start
    /// rather than the successor is what makes the final, incomplete occurrence
    /// harmless — it simply has no successor to read.
    private var lastOccurrence: [Int: [NGram: Int]] = [:]

    /// A hashable n-gram without an allocation per lookup for short widths.
    private struct NGram: Hashable {
        let ids: [Int]
    }

    public init(orders: [Int] = [3, 2]) {
        precondition(orders.allSatisfy { $0 >= 1 }, "an n-gram order must be positive")
        self.orders = orders
        for order in orders { lastOccurrence[order] = [:] }
    }

    /// Ids observed so far, prompt first.
    public var observedIDs: [Int] { sequence }

    /// Append ids the run has committed to — the prompt before the first pass,
    /// then each accepted token.
    public mutating func append(_ ids: [Int]) {
        for id in ids { append(id) }
    }

    public mutating func append(_ id: Int) {
        sequence.append(id)
        // Index every n-gram that *ends* one position before the new token, so
        // an n-gram is only recorded once its successor exists. The n-gram that
        // ends at the last id is deliberately not indexed: it is the query.
        for order in orders where sequence.count > order {
            let start = sequence.count - order - 1
            let gram = NGram(ids: Array(sequence[start..<(start + order)]))
            lastOccurrence[order]?[gram] = start
        }
    }

    /// The token this drafter proposes for the next position, or nil when it
    /// has nothing to say.
    ///
    /// A nil proposal is not a failure — it is a position that runs as an
    /// ordinary single-position pass, and the accounting keeps it separate from
    /// a proposal that was made and rejected.
    public func proposal() -> (tokenID: Int, order: Int)? {
        for order in orders where sequence.count >= order + 1 {
            let tail = NGram(ids: Array(sequence.suffix(order)))
            guard let start = lastOccurrence[order]?[tail],
                start + order < sequence.count
            else { continue }
            return (sequence[start + order], order)
        }
        return nil
    }

    public static func == (
        lhs: DeepSeekV4PromptLookupDrafter, rhs: DeepSeekV4PromptLookupDrafter
    ) -> Bool {
        lhs.orders == rhs.orders && lhs.sequence == rhs.sequence
    }
}

/// How a run draws its drafts, and therefore what an armed pass measures.
///
/// The three modes answer different questions and only one of them is a
/// candidate for shipping.
public enum DeepSeekV4DraftPolicy: Equatable, Sendable {
    /// No draft is ever proposed. Every pass is one position, and the run is
    /// byte-for-byte the run every record before draft-and-verify measured.
    case off
    /// The free drafter. Proposes only where the n-gram index has an answer.
    case promptLookup(orders: [Int])
    /// Always propose, falling back to `fallbackTokenID` where the index has
    /// nothing.
    ///
    /// This is not a shipping policy — its acceptance rate is whatever the
    /// fallback happens to score. It exists because the *cost* of a
    /// two-position pass is the other half of ADR 0017's arithmetic and is
    /// independent of who proposed the draft: a forced run makes every pass a
    /// K = 2 pass, so it measures `V(2)` and the realised union ratio `u`
    /// directly, while still being obliged to emit exactly the greedy ids
    /// because a rejected draft rolls the state back.
    case forced(fallbackTokenID: Int, orders: [Int])

    /// `MINIRUN_V4_DRAFT`: `0`/unset off, `1` the free drafter, `force`
    /// the always-propose cost arm.
    public static let environmentKey = "MINIRUN_V4_DRAFT"

    public static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> DeepSeekV4DraftPolicy {
        let raw = environment[environmentKey]?
            .trimmingCharacters(in: .whitespaces).lowercased() ?? "0"
        switch raw {
        case "", "0", "off", "false":
            return .off
        case "1", "on", "true":
            return .promptLookup(orders: [3, 2])
        case "force", "forced":
            let fallbackRaw = environment["MINIRUN_V4_DRAFT_FALLBACK_TOKEN"] ?? "0"
            guard let fallback = Int(fallbackRaw), fallback >= 0 else {
                throw DeepSeekV4DraftPolicyError.invalid(
                    "MINIRUN_V4_DRAFT_FALLBACK_TOKEN", fallbackRaw)
            }
            return .forced(fallbackTokenID: fallback, orders: [3, 2])
        default:
            // Refused rather than defaulted: a typo that silently disables the
            // thing being measured is the failure mode AGENTS.md's env-gated
            // arms section exists to prevent.
            throw DeepSeekV4DraftPolicyError.invalid(environmentKey, raw)
        }
    }

    public var isEnabled: Bool {
        if case .off = self { return false }
        return true
    }

    var orders: [Int] {
        switch self {
        case .off: return []
        case .promptLookup(let orders), .forced(_, let orders): return orders
        }
    }
}

public enum DeepSeekV4DraftPolicyError: Error, LocalizedError, Equatable {
    case invalid(String, String)

    public var errorDescription: String? {
        switch self {
        case .invalid(let name, let value):
            return "\(name)=\(value) is not a value this runner accepts"
        }
    }
}

/// What a draft-and-verify run reports about its drafting, separately from what
/// it reports about its speed.
///
/// Proposal and acceptance are kept apart on purpose. A position where the
/// drafter had nothing to say costs one ordinary pass and cannot emit two
/// tokens; a position where it proposed and was wrong costs a *wider* pass and
/// still emits one. Collapsing the two into a single "acceptance rate" hides
/// exactly the term that decides whether the knob is a win.
public struct DeepSeekV4DraftMetrics: Equatable, Codable, Sendable {
    /// Positions at which a draft was proposed.
    public var draftProposed: Int = 0
    /// Proposals the verify pass agreed with.
    public var draftAccepted: Int = 0
    /// Positions the run reached, proposal or not.
    public var draftPositions: Int = 0
    /// Model passes executed, whatever their width.
    public var passes: Int = 0
    /// Tokens emitted.
    public var tokensEmitted: Int = 0

    public init() {}

    /// Accepted over positions — the `alpha` ADR 0017's wall-clock and byte
    /// arithmetic uses.
    public var alpha: Double {
        draftPositions > 0 ? Double(draftAccepted) / Double(draftPositions) : 0
    }

    /// Accepted over proposals. Says how good the drafter is where it speaks,
    /// which is a different question from how often it speaks.
    public var acceptanceOverProposals: Double {
        draftProposed > 0 ? Double(draftAccepted) / Double(draftProposed) : 0
    }

    public var proposedRate: Double {
        draftPositions > 0 ? Double(draftProposed) / Double(draftPositions) : 0
    }

    public var tokensPerPass: Double {
        passes > 0 ? Double(tokensEmitted) / Double(passes) : 0
    }

    /// The one line an armed run prints, and the one the Instruments panel
    /// shows. Absent entirely when the knob is off.
    public var summaryLine: String {
        String(
            format:
                "Draft accepted %.0f%% (%d of %d proposed, %d positions), "
                + "%.2f tokens per pass",
            acceptanceOverProposals * 100, draftAccepted, draftProposed,
            draftPositions, tokensPerPass)
    }
}
