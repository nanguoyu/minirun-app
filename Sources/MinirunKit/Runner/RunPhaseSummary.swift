import Foundation

/// Which pass a phase decomposition describes.
///
/// Prefill and decode are different workloads — the audit of 2026-08-14 and
/// every record since keep their seconds, their bytes and their rates apart —
/// so a decomposition says which one it belongs to rather than being averaged
/// into a number no prompt length can be compared against.
public enum RunPassKind: String, Sendable, Codable, CaseIterable {
    case prefill
    case decode
}

/// One named piece of a pass.
///
/// The name is a `String` rather than an enum on purpose: `MinirunKit` links no
/// MLX and cannot name a model's internal brackets, and a runner that starts
/// measuring something new must be able to publish it without this file
/// changing. A consumer that does not recognise a name still carries it,
/// records it, and draws it — see ``RunPhaseTermName/displayName(for:)``.
public struct RunPhaseTerm: Sendable, Codable, Equatable {

    /// Stable machine name. ``RunPhaseTermName`` holds the ones both product
    /// engines currently publish; anything else is carried unchanged.
    public let name: String
    /// Wall seconds this term measured inside (or, when ``isInsidePass`` is
    /// false, beside) the pass.
    public let seconds: Double
    /// How many times the bracket ran, when the engine counts it. Nil means
    /// "not counted" and never a measured zero.
    public let count: Int?
    /// Whether this term is part of the pass's accounting identity.
    ///
    /// False for evidence that describes the *other* side of the pipe — a
    /// background stager's read time, the total a container bracket already
    /// contains — exactly as `ExpertPhaseMetrics` keeps `fetchBusySeconds`
    /// outside its three-way split. Such a term is reported beside the pass and
    /// never summed into ``RunPhaseSummary/attributedSeconds``, because adding
    /// overlapped work to a wall time would make the parts exceed the whole.
    public let isInsidePass: Bool

    public init(
        name: String, seconds: Double, count: Int? = nil, isInsidePass: Bool = true
    ) {
        self.name = name
        self.seconds = seconds
        self.count = count
        self.isInsidePass = isInsidePass
    }

    enum CodingKeys: String, CodingKey {
        case name, seconds, count, isInsidePass
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        seconds = try container.decode(Double.self, forKey: .seconds)
        count = try container.decodeIfPresent(Int.self, forKey: .count)
        // Records written before terms could sit beside a pass contained only
        // terms that were inside one, so a missing key is not unknown.
        isInsidePass = try container.decodeIfPresent(Bool.self, forKey: .isInsidePass) ?? true
    }
}

/// Where the wall time of one pass went, in terms a product can render without
/// knowing which model produced them.
///
/// ## Why this type exists
///
/// Both engines already decompose a pass. K3 has had ``ExpertPhaseMetrics``
/// since v0.6.18 plus the deterministic read-ahead ledger; V4 has fourteen
/// disjoint terms and a residual. Neither could reach the product: `MinirunKit`
/// links no MLX and therefore cannot name `ExpertPhaseMetrics` or
/// `DeepSeekV4PhaseMetrics`, so the App showed none of it and persisted none of
/// it. Tonight's iPhone runs — K3 at 220 s/token, V4 at 17 s/token — computed
/// their attribution on the phone and then threw it away.
///
/// ## The accounting identity
///
/// ```text
/// passSeconds == Σ terms where isInsidePass + unattributedSeconds
/// ```
///
/// ``unattributedSeconds`` is *derived*, never stored, so it cannot disagree
/// with the terms. It is not meant to look small: naming a bucket is what makes
/// it possible to move work out of the residual later, and pretending the sum
/// is complete is not.
public struct RunPhaseSummary: Sendable, Codable, Equatable {

    public let passKind: RunPassKind
    /// 0 is the prefill pass; decode passes count from 1, matching the token
    /// index each pass produced. Nil on an aggregate, which is not one pass.
    public let passIndex: Int?
    /// How many passes these terms sum. 1 for a measured pass; more for an
    /// aggregate, whose per-pass reading is ``secondsPerPass``.
    public let passCount: Int
    /// Wall time of the pass (or of every pass in an aggregate), stated by
    /// whoever bracketed it.
    public let passSeconds: Double
    public let terms: [RunPhaseTerm]

    public init(
        passKind: RunPassKind, passIndex: Int?, passCount: Int = 1,
        passSeconds: Double, terms: [RunPhaseTerm]
    ) {
        self.passKind = passKind
        self.passIndex = passIndex
        self.passCount = max(0, passCount)
        self.passSeconds = passSeconds
        self.terms = terms
    }

    // MARK: Derived

    /// The terms that are part of the pass's identity, largest first. Ties keep
    /// the engine's own order, so a redraw cannot shuffle two equal bars.
    public var insideTerms: [RunPhaseTerm] {
        terms.filter(\.isInsidePass)
    }

    /// Terms reported beside the pass rather than inside it.
    public var besideTerms: [RunPhaseTerm] {
        terms.filter { !$0.isInsidePass }
    }

    public var attributedSeconds: Double {
        insideTerms.reduce(0) { $0 + $1.seconds }
    }

    /// The pass minus everything named. Derived, so it can never disagree with
    /// the terms — and it can go negative if an engine ever publishes terms
    /// taken outside the pass it bracketed, which is what ``isBalanced``
    /// asserts against.
    public var unattributedSeconds: Double {
        passSeconds - attributedSeconds
    }

    /// Every inside term is non-negative and they fit inside the pass. A
    /// microsecond of tolerance absorbs the clock reads themselves.
    public var isBalanced: Bool {
        guard passSeconds.isFinite, passSeconds >= 0 else { return false }
        guard terms.allSatisfy({ $0.seconds.isFinite && $0.seconds >= 0 }) else { return false }
        return unattributedSeconds >= -1e-6
    }

    /// Wall seconds one pass of this kind took. The same as ``passSeconds`` for
    /// a measured pass; the mean for an aggregate.
    public var secondsPerPass: Double? {
        passCount > 0 ? passSeconds / Double(passCount) : nil
    }

    /// The inside terms ordered by cost, largest first, with the residual
    /// appended under ``RunPhaseTermName/unattributed``.
    ///
    /// This is what a bar draws: the residual is a slice like any other, and
    /// leaving it out would make the parts look like the whole.
    public var orderedTermsWithResidual: [RunPhaseTerm] {
        let sorted = insideTerms
            .enumerated()
            .sorted { left, right in
                left.element.seconds == right.element.seconds
                    ? left.offset < right.offset
                    : left.element.seconds > right.element.seconds
            }
            .map(\.element)
        let residual = unattributedSeconds
        guard residual > 0 else { return sorted }
        return sorted + [RunPhaseTerm(name: RunPhaseTermName.unattributed, seconds: residual)]
    }

    public func seconds(of name: String) -> Double? {
        terms.first { $0.name == name }?.seconds
    }

    public func fraction(of seconds: Double) -> Double? {
        guard passSeconds > 0 else { return nil }
        return seconds / passSeconds
    }

    // MARK: Aggregation

    /// One summary that sums a run's passes of a single kind.
    ///
    /// Terms are matched by name and summed; counts are summed only when every
    /// contributing pass counted that term, because a partial sum of counts is
    /// not a count. Term order follows first appearance, so an aggregate reads
    /// in the same order as the passes it came from. Nil for an empty sequence
    /// and for a mixture of pass kinds — a decode aggregate that quietly
    /// absorbed the prefill would misattribute the longest pass of the turn.
    public static func aggregate(_ summaries: [RunPhaseSummary]) -> RunPhaseSummary? {
        guard let first = summaries.first else { return nil }
        guard summaries.allSatisfy({ $0.passKind == first.passKind }) else { return nil }

        var order: [String] = []
        var seconds: [String: Double] = [:]
        var counts: [String: Int] = [:]
        var countedIn: [String: Int] = [:]
        var appearedIn: [String: Int] = [:]
        var inside: [String: Bool] = [:]

        for summary in summaries {
            for term in summary.terms {
                if seconds[term.name] == nil {
                    order.append(term.name)
                    inside[term.name] = term.isInsidePass
                }
                seconds[term.name, default: 0] += term.seconds
                appearedIn[term.name, default: 0] += 1
                if let count = term.count {
                    counts[term.name, default: 0] += count
                    countedIn[term.name, default: 0] += 1
                }
            }
        }

        let passCount = summaries.count
        return RunPhaseSummary(
            passKind: first.passKind,
            passIndex: nil,
            passCount: passCount,
            passSeconds: summaries.reduce(0) { $0 + $1.passSeconds },
            terms: order.map { name in
                RunPhaseTerm(
                    name: name,
                    seconds: seconds[name] ?? 0,
                    count: countedIn[name] == passCount && appearedIn[name] == passCount
                        ? counts[name] : nil,
                    isInsidePass: inside[name] ?? true)
            })
    }

    // MARK: Codable

    enum CodingKeys: String, CodingKey {
        case passKind, passIndex, passCount, passSeconds, terms
        case attributedSeconds, unattributedSeconds
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        passKind = try container.decode(RunPassKind.self, forKey: .passKind)
        passIndex = try container.decodeIfPresent(Int.self, forKey: .passIndex)
        passCount = try container.decodeIfPresent(Int.self, forKey: .passCount) ?? 1
        passSeconds = try container.decode(Double.self, forKey: .passSeconds)
        terms = try container.decode([RunPhaseTerm].self, forKey: .terms)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(passKind, forKey: .passKind)
        try container.encodeIfPresent(passIndex, forKey: .passIndex)
        try container.encode(passCount, forKey: .passCount)
        try container.encode(passSeconds, forKey: .passSeconds)
        try container.encode(terms, forKey: .terms)
        // Derived, written so a reader of the record does not have to redo the
        // arithmetic — and never read back, so the two cannot disagree.
        try container.encode(attributedSeconds, forKey: .attributedSeconds)
        try container.encode(unattributedSeconds, forKey: .unattributedSeconds)
    }
}

/// The term names both product engines publish today, and the words a screen
/// says them with.
///
/// A name that is not here is not an error: it is a term this build has not
/// heard of yet, and ``displayName(for:)`` still produces something readable
/// for it rather than dropping it.
public enum RunPhaseTermName {

    /// The residual. Never published by an engine — it is derived from the
    /// pass — but named here so a bar and a table say the same word for it.
    public static let unattributed = "unattributed"

    // Storage, both engines.

    /// The decode thread blocking on a deterministic weight read.
    public static let deterministicRead = "deterministicRead"
    /// K3 only: the decode loop waiting for bytes the stager had not finished.
    public static let stagerWait = "stagerWait"
    /// K3 only, beside the pass: the background stager's own read time. It
    /// overlaps compute by design, so it is not part of the identity.
    public static let stagedRead = "stagedRead"

    // The routed-expert phase, both engines, on the K3 rule.

    public static let expertIOWait = "expertIOWait"
    public static let expertGatherCompute = "expertGatherCompute"
    /// The gather's remainder: expression building, adoption, slot bookkeeping.
    public static let expertOther = "expertOther"

    // K3.

    /// The mid-layer evaluation barrier that lets attention weights be released
    /// before the MLP half is widened.
    public static let attentionBarrier = "attentionBarrier"
    /// Beside the pass: the layer loop's own total, which contains several of
    /// the terms above and is reported for scale rather than summed.
    public static let layerTotal = "layerTotal"

    // V4.

    public static let tileDigest = "tileDigest"
    public static let outputHeadRead = "outputHeadRead"
    public static let outputHeadCompute = "outputHeadCompute"
    public static let expertBackendLifecycle = "expertBackendLifecycle"
    public static let layerArtifactSetup = "layerArtifactSetup"
    public static let tileAdoption = "tileAdoption"
    public static let activationScaleSync = "activationScaleSync"
    public static let finitenessSweep = "finitenessSweep"
    public static let routingSelect = "routingSelect"
    public static let sparseAttention = "sparseAttention"
    public static let lightningIndexer = "lightningIndexer"

    /// Both engines: returning pages at a boundary.
    public static let reclaim = "reclaim"

    private static let names: [String: String] = [
        unattributed: "Unattributed",
        deterministicRead: "Model weight reads",
        stagerWait: "Waiting for staged reads",
        stagedRead: "Staged reads (background)",
        expertIOWait: "Waiting for expert weights",
        expertGatherCompute: "Expert gather",
        expertOther: "Expert bookkeeping",
        attentionBarrier: "Attention barrier",
        layerTotal: "Layers (contains the terms above)",
        tileDigest: "Tile digests",
        outputHeadRead: "Output head reads",
        outputHeadCompute: "Output head compute",
        expertBackendLifecycle: "Expert backend setup",
        layerArtifactSetup: "Layer setup",
        tileAdoption: "Tile adoption",
        activationScaleSync: "Activation scale sync",
        finitenessSweep: "Finiteness checks",
        routingSelect: "Routing",
        sparseAttention: "Sparse attention",
        lightningIndexer: "Indexer",
        reclaim: "Memory reclaim",
    ]

    /// A readable label for any term name, known or not.
    ///
    /// An unknown name is split at its camel-case humps and sentence-cased,
    /// which turns a future `kvCacheRebuild` into "Kv cache rebuild" rather
    /// than into nothing at all. A screen that silently dropped a term it did
    /// not recognise would under-report a pass while still summing to it.
    public static func displayName(for name: String) -> String {
        if let known = names[name] { return known }
        var words = ""
        for character in name {
            if character.isUppercase, !words.isEmpty {
                words.append(" ")
                words.append(Character(character.lowercased()))
            } else {
                words.append(character)
            }
        }
        return words.prefix(1).uppercased() + words.dropFirst()
    }
}
