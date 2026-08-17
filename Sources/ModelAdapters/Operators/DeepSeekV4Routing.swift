import Foundation
import MLX

/// The first V4 layers map token ids directly to experts. Keeping the map a
/// checked flat Int32 table avoids the allocation overhead of 129,280 small
/// Swift arrays while preserving the checkpoint's exact integer decisions.
public struct DeepSeekV4TokenExpertMap: Sendable, Equatable {
    public let vocabularySize: Int
    public let expertsPerToken: Int
    public let expertCount: Int
    private let ids: [Int32]

    public init(
        ids: [Int32], vocabularySize: Int, expertsPerToken: Int, expertCount: Int
    ) throws {
        guard vocabularySize > 0, expertsPerToken > 0, expertCount > 0 else {
            throw DeepSeekV4Error.routing(
                "vocabulary size, experts per token, and expert count must be positive")
        }
        let expected = vocabularySize.multipliedReportingOverflow(by: expertsPerToken)
        guard !expected.overflow, ids.count == expected.partialValue else {
            throw DeepSeekV4Error.routing(
                "token-to-expert table has \(ids.count) entries; expected "
                    + "\(expected.overflow ? -1 : expected.partialValue)")
        }
        // Validation order is part of the contract: within a row, slot `j`'s
        // range check precedes its duplicate check, and both precede slot
        // `j+1`. The duplicate check compares against the slots already
        // accepted in the same row instead of building a `Set` per row —
        // `expertsPerToken` is a handful of slots, while the per-row set cost
        // one allocation per vocabulary entry for the whole table.
        try ids.withUnsafeBufferPointer { buffer in
            for token in 0..<vocabularySize {
                let start = token * expertsPerToken
                for slot in 0..<expertsPerToken {
                    let id = buffer[start + slot]
                    guard id >= 0, id < expertCount else {
                        throw DeepSeekV4Error.routing(
                            "token \(token) slot \(slot) names expert \(id), outside 0..<\(expertCount)")
                    }
                    for earlier in 0..<slot where buffer[start + earlier] == id {
                        throw DeepSeekV4Error.routing(
                            "token \(token) names expert \(id) more than once")
                    }
                }
            }
        }
        self.ids = ids
        self.vocabularySize = vocabularySize
        self.expertsPerToken = expertsPerToken
        self.expertCount = expertCount
    }

    public func experts(for tokenIDs: [Int]) throws -> [[Int]] {
        try tokenIDs.map { token in
            guard token >= 0, token < vocabularySize else {
                throw DeepSeekV4Error.routing(
                    "token id \(token) is outside 0..<\(vocabularySize)")
            }
            let start = token * expertsPerToken
            return (0..<expertsPerToken).map { Int(ids[start + $0]) }
        }
    }
}

/// DeepSeek V4's `sqrtsoftplus` / `noaux_tc` router.
///
/// The selected expert set and the weights deliberately come from different
/// tensors: correction bias changes top-k choice only, while the gathered
/// unmodified sqrt-softplus scores are normalized and scaled. The first hash
/// layers replace only the expert-id choice with the token table; they still
/// use the learned router scores as weights.
public enum DeepSeekV4Router {
    public enum Choice {
        case learned(correctionBias: MLXArray)
        case tokenHash(tokenIDs: [Int], map: DeepSeekV4TokenExpertMap)
    }

    public struct Selection {
        public let ids: [[Int]]
        public let weights: MLXArray
        public let logits: MLXArray
        public let unmodifiedScores: MLXArray
        public let scoresForChoice: MLXArray?
        /// Nil for hash routing: no score boundary selected those experts.
        public let relativeMargins: [Double]?
    }

    /// The bracket covers the whole selection: the sqrt-softplus `eval` and
    /// `asArray`, the host sort over every expert, and the re-upload of the
    /// gathered weights. All three are one decision, and splitting them would
    /// name three fractions of a round trip rather than the round trip.
    public static func select(
        logits: MLXArray,
        choice: Choice,
        expertCount: Int,
        expertsPerToken: Int,
        normalizeSelectedWeights: Bool,
        routingScale: Float,
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil
    ) throws -> Selection {
        try measuringPhase(phaseAccounting?.recordRoutingSelect(nanoseconds:)) {
            try selecting(
                logits: logits,
                choice: choice,
                expertCount: expertCount,
                expertsPerToken: expertsPerToken,
                normalizeSelectedWeights: normalizeSelectedWeights,
                routingScale: routingScale,
                phaseAccounting: phaseAccounting)
        }
    }

    private static func selecting(
        logits: MLXArray,
        choice: Choice,
        expertCount: Int,
        expertsPerToken: Int,
        normalizeSelectedWeights: Bool,
        routingScale: Float,
        phaseAccounting: DeepSeekV4PhaseAccounting?
    ) throws -> Selection {
        guard logits.ndim == 2, logits.shape[1] == expertCount else {
            throw DeepSeekV4Error.routing(
                "router logits must be [tokens, \(expertCount)], got \(logits.shape)")
        }
        guard logits.shape[0] > 0 else {
            throw DeepSeekV4Error.routing("router logits must contain at least one token")
        }
        guard expertCount > 1, expertsPerToken > 0, expertsPerToken < expertCount else {
            throw DeepSeekV4Error.routing(
                "expertsPerToken must be in 1..<expertCount")
        }
        guard routingScale.isFinite, routingScale > 0 else {
            throw DeepSeekV4Error.routing("routing scale must be finite and positive")
        }

        let scores = sqrt(K3Activation.softplus(logits.asType(.float32)))

        // Both host values this selection needs are built *before* either is
        // pulled, so the round trip that delivers them is one.
        //
        // Learned routing reads two arrays: the sqrt-softplus scores become the
        // weights, and the bias-ranked scores decide the choice. Expressing the
        // ranked score after forcing the plain one made it a second graph, and
        // a second graph after a drain is a second GPU round trip — 40 of the
        // 43 layers paid one every pass for nothing but statement order. The
        // shape guard therefore moves up to here, which is the only observable
        // difference: a call whose bias has the wrong shape *and* whose scores
        // are non-finite now reports the shape first. Both are configuration
        // errors that abort the pass, and both messages are unchanged.
        let ranked: MLXArray?
        switch choice {
        case .learned(let correctionBias):
            guard correctionBias.ndim == 1, correctionBias.shape[0] == expertCount else {
                throw DeepSeekV4Error.routing(
                    "correction bias must be [\(expertCount)], got \(correctionBias.shape)")
            }
            ranked = scores + correctionBias.asType(.float32)
        case .tokenHash:
            ranked = nil
        }

        // One sync, however many arrays it delivers: the `asArray` calls below
        // copy arrays this `eval` has already materialised, so they wait for
        // nothing. This is the pass's irreducible per-layer stop — the pager
        // cannot name a tile until these ids exist on the host.
        phaseAccounting?.recordEval(.routingSelect)
        if let ranked {
            MLX.eval([scores, ranked])
        } else {
            scores.eval()
        }
        let values = scores.asArray(Float.self)
        guard values.allSatisfy(\.isFinite) else {
            throw DeepSeekV4Error.routing("sqrt-softplus produced a non-finite score")
        }
        let tokenCount = logits.shape[0]

        let ids: [[Int]]
        let scoresForChoice: MLXArray?
        let margins: [Double]?
        switch choice {
        case .learned:
            // Built from this same case a few statements above. The guard
            // states that invariant instead of forcing it.
            guard let ranked else {
                throw DeepSeekV4Error.routing(
                    "learned routing reached selection without a biased score")
            }
            let rankedValues = ranked.asArray(Float.self)
            guard rankedValues.allSatisfy(\.isFinite) else {
                throw DeepSeekV4Error.routing("biased selection score is non-finite")
            }
            var selected = [[Int]]()
            var measuredMargins = [Double]()
            selected.reserveCapacity(tokenCount)
            measuredMargins.reserveCapacity(tokenCount)
            for token in 0..<tokenCount {
                let start = token * expertCount
                let order = (0..<expertCount).sorted { left, right in
                    let lhs = rankedValues[start + left]
                    let rhs = rankedValues[start + right]
                    return lhs == rhs ? left < right : lhs > rhs
                }
                selected.append(Array(order.prefix(expertsPerToken)))
                let selectedBoundary = Double(rankedValues[start + order[expertsPerToken - 1]])
                let rejectedBoundary = Double(rankedValues[start + order[expertsPerToken]])
                let scale = max(abs(selectedBoundary), abs(rejectedBoundary))
                measuredMargins.append(
                    scale > 0 ? (selectedBoundary - rejectedBoundary) / scale : 0)
            }
            ids = selected
            scoresForChoice = ranked
            margins = measuredMargins

        case .tokenHash(let tokenIDs, let map):
            guard tokenIDs.count == tokenCount else {
                throw DeepSeekV4Error.routing(
                    "hash routing has \(tokenIDs.count) token ids for \(tokenCount) rows")
            }
            guard map.expertCount == expertCount,
                map.expertsPerToken == expertsPerToken
            else {
                throw DeepSeekV4Error.routing(
                    "token-to-expert table geometry does not match this router")
            }
            ids = try map.experts(for: tokenIDs)
            scoresForChoice = nil
            margins = nil
        }

        let gatheredCount = tokenCount.multipliedReportingOverflow(by: expertsPerToken)
        guard !gatheredCount.overflow else {
            throw DeepSeekV4Error.routing("selected router weights overflow Int")
        }
        var gathered = [Float](repeating: 0, count: gatheredCount.partialValue)
        for token in 0..<tokenCount {
            let start = token * expertCount
            for (slot, expert) in ids[token].enumerated() {
                gathered[token * expertsPerToken + slot] = values[start + expert]
            }
        }
        var weights = MLXArray(gathered, [tokenCount, expertsPerToken])
        if normalizeSelectedWeights {
            for token in 0..<tokenCount {
                let start = token * expertsPerToken
                let denominator = gathered[start..<(start + expertsPerToken)].reduce(0, +)
                guard denominator.isFinite, denominator > 0 else {
                    throw DeepSeekV4Error.routing(
                        "selected router scores for token \(token) cannot be normalized")
                }
            }
            let denominator = sum(weights, axis: -1, keepDims: true)
            weights = weights / denominator
        }
        weights = weights * routingScale

        return Selection(
            ids: ids,
            weights: weights,
            logits: logits,
            unmodifiedScores: scores,
            scoresForChoice: scoresForChoice,
            relativeMargins: margins)
    }
}
