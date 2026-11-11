import Foundation
import MinirunKit
import ModelAdapters
import StorageCore

/// Where a V4.1 run picks a conversation up from, where it puts it down, and
/// the shape the conversation is computed in.
///
/// Both files are of ``DeepSeekV41SessionState``'s format. A run with no
/// options is today's run, and its record carries no session entry at all.
///
/// - `firstTurnTokenCount`: the number of ids the chat encoding gives the
///   conversation's first user turn alone, with its generation prefix. ADR 0023
///   makes a conversation one generation: that block is prefilled at position
///   0 and every later id goes one pass each. With it, a run that has no usable
///   state **replays** in that shape; without it (nil) the whole prompt is the
///   block, which is right only for a first turn and is what a run without this
///   field did before replay existed.
/// - `restoreFrom`: the state an earlier turn left. Used when its ids are a
///   strict prefix of this run's prompt and it was opened with the same first
///   block. A file that is missing, refused (a different publication,
///   inference revision, geometry or split, a truncated file) or stale (the
///   conversation was edited) is replayed past, and the record says why.
/// - `writeTo`: where the state goes, at the cadence `checkpoint` names. The
///   last generated token has not been fed back; the next turn's prompt
///   continues with it.
public struct DeepSeekV41SessionOptions: Sendable, Equatable {
    /// When the state is written.
    public enum Checkpoint: String, Codable, Sendable {
        /// After every completed pass, prompt passes included, so a process
        /// that is killed keeps everything up to its last finished token.
        case everyPass = "every-pass"
        /// Once, when the workload is released: at the end of the turn, or
        /// when it is stopped between passes.
        case atRelease = "at-release"

        /// A phone is where a process is killed under a reply (Jetsam, the
        /// app sent to the background), so it writes every pass. A Mac writes
        /// when the turn ends.
        public static var platformDefault: Checkpoint {
            #if os(iOS)
                return .everyPass
            #else
                return .atRelease
            #endif
        }
    }

    public var restoreFrom: URL?
    public var writeTo: URL?
    public var firstTurnTokenCount: Int?
    public var checkpoint: Checkpoint

    public init(
        restoreFrom: URL? = nil, writeTo: URL? = nil, firstTurnTokenCount: Int? = nil,
        checkpoint: Checkpoint = .platformDefault
    ) {
        self.restoreFrom = restoreFrom
        self.writeTo = writeTo
        self.firstTurnTokenCount = firstTurnTokenCount
        self.checkpoint = checkpoint
    }
}

/// The run record's account of the session: which path the prompt took, and
/// what was written.
public struct DeepSeekV41SessionRecord: Codable, Sendable, Equatable {
    public enum Path: String, Codable, Sendable {
        /// The prompt was the first turn's block.
        case firstTurn = "first-turn"
        /// A restored state held a prefix; the rest went one pass each.
        case continued
        /// No usable state: the first block, then every later id one pass each.
        case replayed
        /// The whole prompt in one prefill, for a run that stated no first-turn
        /// length. A conversation that states one never takes this path.
        case fullPrefill = "full-prefill"
    }

    public var restoredFrom: String?
    public var path: Path = .fullPrefill
    public var firstBlockTokens: Int?
    public var reusedPositions: Int?
    public var newTokens: Int?
    /// One-id passes a replay ran after its first block.
    public var replayedPasses: Int?
    public var fallbackReason: String?
    public var restoreSeconds: Double?
    public var checkpoint: DeepSeekV41SessionOptions.Checkpoint?
    public var snapshotWrittenTo: String?
    public var snapshotBytes: UInt64?
    public var snapshotPositions: Int?
    public var snapshotTokenSHA256: String?
    /// The last write's seconds.
    public var snapshotSeconds: Double?
    public var snapshotRefusal: String?
    /// Every write this run made, and the seconds each took, in order.
    public var snapshotWrites: Int?
    public var snapshotWriteSeconds: [Double]?

    public init() {}

    mutating func record(_ schedule: DeepSeekV41PromptSchedule, statedFirstTurn: Bool) {
        firstBlockTokens = schedule.block.isEmpty ? nil : schedule.block.count
        fallbackReason = schedule.fallbackReason
        switch schedule.path {
        case .continued(let reused):
            path = .continued
            reusedPositions = reused
            newTokens = schedule.passes.count
        case .replayed:
            path = .replayed
            replayedPasses = schedule.passes.count
        case .firstTurn:
            path = statedFirstTurn ? .firstTurn : .fullPrefill
        }
    }
}

/// Writes a run's generation state at the cadence the session asks for, and
/// keeps the account of every write.
///
/// Its own type, and not three lines in the workload, so that the cadence is
/// testable without a container: the writer is a closure.
final class DeepSeekV41SessionCheckpointer {
    typealias Write = (_ url: URL) throws -> DeepSeekV41SessionSnapshotSummary

    let url: URL
    let cadence: DeepSeekV41SessionOptions.Checkpoint
    private(set) var writtenPositions: Int?
    private(set) var seconds: [Double] = []
    private(set) var lastSummary: DeepSeekV41SessionSnapshotSummary?
    private(set) var lastRefusal: String?

    init(url: URL, cadence: DeepSeekV41SessionOptions.Checkpoint) {
        self.url = url
        self.cadence = cadence
    }

    /// After a pass completed. Writes only at `everyPass`.
    func passCompleted(positions: Int, write: Write) {
        guard cadence == .everyPass else { return }
        perform(positions: positions, write: write)
    }

    /// At release. Writes unless the state at these positions is already on
    /// disk, which it is after the last pass of an `everyPass` run.
    ///
    /// - Parameter positions: nil when there is no state at a pass boundary to
    ///   write: a pass threw and dropped it. Whatever an earlier pass wrote
    ///   stays, which is the point of writing every pass.
    func released(positions: Int?, write: Write) {
        guard let positions else {
            if writtenPositions == nil {
                lastRefusal = "no complete generation state to write"
            }
            return
        }
        guard writtenPositions != positions else { return }
        perform(positions: positions, write: write)
    }

    private func perform(positions: Int, write: Write) {
        let started = MonotonicClock.now()
        do {
            lastSummary = try write(url)
            writtenPositions = positions
            lastRefusal = nil
        } catch {
            lastRefusal = "\(error)"
        }
        seconds.append(MonotonicClock.seconds(since: started))
    }

    func fill(_ record: inout DeepSeekV41SessionRecord) {
        record.checkpoint = cadence
        if let summary = lastSummary {
            record.snapshotWrittenTo = url.path
            record.snapshotBytes = summary.fileBytes
            record.snapshotPositions = summary.positions
            record.snapshotTokenSHA256 = summary.tokenSHA256
        }
        record.snapshotRefusal = lastRefusal
        record.snapshotSeconds = seconds.last
        record.snapshotWrites = seconds.count
        record.snapshotWriteSeconds = seconds
    }
}

/// The publication facts a session file is bound to, from what the factory
/// already opened.
extension DeepSeekV41SessionBinding {
    init(authority: ArtifactRuntimeAuthority, indexData: Data) {
        self.init(
            publicationRevision: authority.evidence.repository.revision,
            indexSHA256: SHA256.hexString(SHA256.hash(indexData)))
    }
}
