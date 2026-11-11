import Foundation

/// How a turn's prompt reached the model: over a restored state, or from
/// position 0.
public enum DeepSeekV41SessionPath: Equatable, Sendable {
    /// The restored state's ids were a strict prefix of the prompt, and only
    /// the `newTokens` after them were run.
    case continued(reusedPositions: Int, newTokens: Int)
    /// The whole prompt was prefilled, for the reason given.
    case fullPrefill(reason: String)

    public var reason: String? {
        if case .fullPrefill(let reason) = self { return reason }
        return nil
    }
}

/// The passes a turn's prompt takes, in ADR 0023's fixed shape.
///
/// A V4.1 conversation is one generation: the first user turn's prompt is
/// prefilled as one block at position 0, and every later id goes through one
/// pass of its own. A turn therefore reaches its first generated token in one
/// of three ways, and each is that same shape:
///
/// - **first turn**: the prompt *is* the first block;
/// - **continued**: a restored state already holds a prefix of the prompt, and
///   the rest goes one pass each;
/// - **replayed**: no usable state, so the first block is prefilled and every
///   remaining id goes one pass each, which recomputes what the lost state held
///   and then continues. It is never one prefill over the whole conversation,
///   because that computes different logits (the 2026-11-10 session-state
///   record) and the answer would then depend on whether a file survived.
///
/// Pure: it reads ids and lengths and nothing else.
public struct DeepSeekV41PromptSchedule: Equatable, Sendable {
    public enum Path: Equatable, Sendable {
        case firstTurn
        case continued(reusedPositions: Int)
        case replayed
    }

    public let path: Path
    /// Ids prefilled as one block at position 0. Empty when continued.
    public let block: [Int]
    /// Ids that go one pass each, after the block or the restored state.
    public let passes: [Int]
    /// Why a restored state was not used, when one was offered or looked for.
    public let fallbackReason: String?

    /// Every id the prompt has, which is what replay progress counts against.
    public var promptTokens: Int {
        if case .continued(let reused) = path { return reused + passes.count }
        return block.count + passes.count
    }

    /// - Parameters:
    ///   - prompt: the whole conversation's ids, this turn's generation prefix
    ///     included.
    ///   - firstBlockTokens: the number of ids the chat encoding gives the
    ///     first user turn alone, with its generation prefix.
    ///   - restored: the ids a restored state consumed and the block it was
    ///     opened with, or nil when there is none.
    ///   - unusableReason: why a state that was looked for is not offered (a
    ///     refused or missing file), carried into the replay's reason.
    public static func make(
        prompt: [Int],
        firstBlockTokens: Int,
        restored: (tokenIDs: [Int], firstBlockTokens: Int)?,
        unusableReason: String? = nil
    ) throws -> DeepSeekV41PromptSchedule {
        guard !prompt.isEmpty else {
            throw DeepSeekV41Error.configuration("a turn needs at least one prompt token")
        }
        guard firstBlockTokens >= 1, firstBlockTokens <= prompt.count else {
            throw DeepSeekV41Error.configuration(
                "a first turn of \(firstBlockTokens) ids cannot open a "
                    + "\(prompt.count)-id conversation")
        }
        var reason = unusableReason
        if let restored {
            if restored.firstBlockTokens != firstBlockTokens {
                reason = "the session state was opened with a \(restored.firstBlockTokens)-id "
                    + "first block and this conversation's first turn is \(firstBlockTokens) ids"
            } else {
                switch DeepSeekV41Model.sessionPath(consumed: restored.tokenIDs, prompt: prompt) {
                case .continued(let reused, _):
                    return DeepSeekV41PromptSchedule(
                        path: .continued(reusedPositions: reused), block: [],
                        passes: Array(prompt[reused...]), fallbackReason: nil)
                case .fullPrefill(let why):
                    reason = why
                }
            }
        }
        let block = Array(prompt[..<firstBlockTokens])
        let passes = Array(prompt[firstBlockTokens...])
        return DeepSeekV41PromptSchedule(
            path: passes.isEmpty ? .firstTurn : .replayed, block: block, passes: passes,
            fallbackReason: reason)
    }
}

extension DeepSeekV41Model {
    /// Whether a state that consumed `consumed` can carry `prompt`, and if not,
    /// why. Pure: it reads ids and nothing else.
    public static func sessionPath(consumed: [Int]?, prompt: [Int]) -> DeepSeekV41SessionPath {
        guard let consumed else {
            return .fullPrefill(reason: "no session state was offered")
        }
        guard !consumed.isEmpty else {
            return .fullPrefill(reason: "the session state consumed no tokens")
        }
        guard prompt.count > consumed.count else {
            return .fullPrefill(
                reason: "the prompt has \(prompt.count) tokens and the session state consumed "
                    + "\(consumed.count); a continuation needs at least one new token")
        }
        if let first = zip(consumed, prompt).enumerated().first(where: { $0.element.0 != $0.element.1 }) {
            return .fullPrefill(
                reason: "the prompt departs from the session state's tokens at position "
                    + "\(first.offset) (\(first.element.1) against \(first.element.0))")
        }
        return .continued(reusedPositions: consumed.count, newTokens: prompt.count - consumed.count)
    }

    /// Run `tokenIDs` through `state`, one token a pass, and return the last
    /// pass's step.
    ///
    /// ## One token a pass, not a block at an offset
    ///
    /// DeepSeek's `inference/model.py` runs a multi-token pass only at
    /// `start_pos == 0`; at any other position every path it has (the window
    /// write, `get_window_topk_idxs`, the compressor's state, the indexer's
    /// reachable counts) takes exactly one token, and the operators here refuse
    /// anything else for the same reason. A block of new tokens at an offset
    /// would be arithmetic with no reference to hold it to, so the new tokens of
    /// a continued turn go through the reference's own cached path instead:
    /// the same one decode uses.
    public func advance(
        tokenIDs: [Int],
        state: inout DeepSeekV41GenerationState,
        cancellationCheck: () throws -> Void = {},
        onBlockCompleted: (_ completed: Int, _ total: Int) -> Void = { _, _ in }
    ) throws -> DeepSeekV41Step {
        guard !tokenIDs.isEmpty else {
            throw DeepSeekV41Error.configuration("an advance needs at least one token")
        }
        var last: DeepSeekV41Step?
        for token in tokenIDs {
            last = try decode(
                tokenID: token, state: &state,
                cancellationCheck: cancellationCheck, onBlockCompleted: onBlockCompleted)
        }
        return last!
    }

    /// A turn's prompt pass: continued over `restored` when its ids are a
    /// strict prefix of `tokenIDs`, a full prefill otherwise.
    ///
    /// `restored` is consumed either way. The caller learns which path ran from
    /// the returned ``DeepSeekV41SessionPath`` and is expected to say so in its
    /// run record: the two paths need not produce the same bits (the
    /// 2026-11-10 session-state record measures by how much).
    public func prefill(
        tokenIDs: [Int],
        continuing restored: DeepSeekV41GenerationState?,
        decodePositionLimit: Int,
        cancellationCheck: () throws -> Void = {},
        onBlockCompleted: (_ completed: Int, _ total: Int) -> Void = { _, _ in }
    ) throws -> (
        step: DeepSeekV41Step, state: DeepSeekV41GenerationState, path: DeepSeekV41SessionPath
    ) {
        var path = Self.sessionPath(consumed: restored?.consumedTokenIDs, prompt: tokenIDs)
        if case .continued = path, let restored {
            if restored.caches.count != config.numberOfLayers
                || restored.nextPosition != restored.consumedTokenIDs.count
            {
                path = .fullPrefill(reason: "the session state is released or inconsistent")
            } else if decodePositionLimit < tokenIDs.count || decodePositionLimit > positionCount {
                throw DeepSeekV41Error.configuration(
                    "decode position limit \(decodePositionLimit) cannot hold a "
                        + "\(tokenIDs.count)-token prompt inside the \(positionCount) "
                        + "positions this model was built for")
            }
        }
        guard case .continued(let reused, _) = path, var state = restored else {
            let result = try prefill(
                tokenIDs: tokenIDs, decodePositionLimit: decodePositionLimit,
                cancellationCheck: cancellationCheck, onBlockCompleted: onBlockCompleted)
            return (result.step, result.state, path)
        }
        state.positionLimit = decodePositionLimit
        do {
            let step = try advance(
                tokenIDs: Array(tokenIDs[reused...]), state: &state,
                cancellationCheck: cancellationCheck, onBlockCompleted: onBlockCompleted)
            return (step, state, path)
        } catch {
            state.release()
            throw error
        }
    }

    /// A turn's prompt in ADR 0023's shape: continued over `restored` when it
    /// holds a prefix of `tokenIDs` opened with the same first block, otherwise
    /// the first block prefilled and every later id one pass each.
    ///
    /// `restored` is consumed either way. `onPass` is told, after every pass,
    /// how many of the prompt's ids are now consumed and how many it has.
    public func prefill(
        tokenIDs: [Int],
        firstBlockTokens: Int,
        continuing restored: DeepSeekV41GenerationState?,
        decodePositionLimit: Int,
        cancellationCheck: () throws -> Void = {},
        onBlockCompleted: (_ completed: Int, _ total: Int) -> Void = { _, _ in },
        onPass: (_ consumed: Int, _ promptTokens: Int) -> Void = { _, _ in }
    ) throws -> (
        step: DeepSeekV41Step, state: DeepSeekV41GenerationState,
        schedule: DeepSeekV41PromptSchedule
    ) {
        var offered = restored
        if let candidate = offered,
            candidate.caches.count != config.numberOfLayers
                || candidate.nextPosition != candidate.consumedTokenIDs.count
        {
            offered?.release()
            offered = nil
        }
        let schedule = try DeepSeekV41PromptSchedule.make(
            prompt: tokenIDs, firstBlockTokens: firstBlockTokens,
            restored: offered.map { ($0.consumedTokenIDs, $0.firstBlockTokenCount) },
            unusableReason: restored != nil && offered == nil
                ? "the session state is released or inconsistent" : nil)
        var state: DeepSeekV41GenerationState
        var last: DeepSeekV41Step?
        if case .continued = schedule.path, var continuing = offered {
            guard decodePositionLimit >= tokenIDs.count, decodePositionLimit <= positionCount
            else {
                continuing.release()
                throw DeepSeekV41Error.configuration(
                    "decode position limit \(decodePositionLimit) cannot hold a "
                        + "\(tokenIDs.count)-token prompt inside the \(positionCount) "
                        + "positions this model was built for")
            }
            continuing.positionLimit = decodePositionLimit
            state = continuing
        } else {
            offered?.release()
            let opened = try prefill(
                tokenIDs: schedule.block, decodePositionLimit: decodePositionLimit,
                cancellationCheck: cancellationCheck, onBlockCompleted: onBlockCompleted)
            state = opened.state
            last = opened.step
            onPass(state.nextPosition, schedule.promptTokens)
        }
        do {
            for token in schedule.passes {
                last = try decode(
                    tokenID: token, state: &state,
                    cancellationCheck: cancellationCheck, onBlockCompleted: onBlockCompleted)
                onPass(state.nextPosition, schedule.promptTokens)
            }
        } catch {
            state.release()
            throw error
        }
        return (last!, state, schedule)
    }
}
