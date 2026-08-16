import Foundation
import MinirunKit
import StorageCore

/// A download here can run for days across 470 files and 1.56 TB. Resumability
/// is not an error path; it is the normal shape of the feature, and the state
/// machine says so — `paused` and `interrupted` are ordinary states with the
/// same visual weight as `active`. Only `failed` is red.
enum DownloadState: Equatable, Sendable {
    case notStarted
    case resolvingIndex
    case awaitingDestination
    case active(ProgressSnapshot)
    /// The operator asked.
    case paused(ProgressSnapshot)
    /// The world intervened.
    case interrupted(ProgressSnapshot, reason: InterruptionReason)
    case verifying(VerificationPhase, ProgressSnapshot)
    case cancelled(ProgressSnapshot)
    case ready(measuredBytes: UInt64, fileCount: Int)
    case incomplete(missingFiles: Int, missingBytes: UInt64)
    case failed(StorageCoreError)

    var progress: ProgressSnapshot? {
        switch self {
        case .active(let snapshot), .paused(let snapshot), .verifying(_, let snapshot),
            .cancelled(let snapshot):
            return snapshot
        case .interrupted(let snapshot, _):
            return snapshot
        default:
            return nil
        }
    }

    /// A model whose bytes are on disk and checked. The only state a run may
    /// start from.
    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }

    var isBusy: Bool {
        switch self {
        case .resolvingIndex, .active, .verifying: return true
        default: return false
        }
    }

    /// One phrase, for a place that has room for a value and not for a card.
    /// Never a bare word: "not downloaded" and "incomplete" are different
    /// answers to "can this run", and a row that blurred them would be lying by
    /// omission.
    var phrase: String {
        switch self {
        case .notStarted: return "not downloaded"
        case .resolvingIndex: return "resolving the index"
        case .awaitingDestination: return "waiting for a destination"
        case .active: return "downloading"
        case .paused: return "paused"
        case .interrupted(_, let reason): return "interrupted — \(reason.sentence)"
        case .verifying: return "verifying"
        case .cancelled: return "cancelled — partial files kept"
        case .ready: return "ready"
        case .incomplete(let files, _): return "incomplete — \(files) files missing"
        case .failed(let error): return "failed — \(error.description)"
        }
    }

    /// The catalog row's leading glyph.
    var glyphName: String {
        switch self {
        case .notStarted, .awaitingDestination: return "cloud"
        case .resolvingIndex, .active: return "arrow.down.circle"
        case .paused, .interrupted: return "pause.circle"
        case .verifying: return "checkmark.shield"
        case .cancelled: return "xmark.circle"
        case .ready: return "checkmark.seal"
        case .incomplete, .failed: return "exclamationmark.triangle"
        }
    }

    /// Compact, user-facing status for catalog and global download rows. The
    /// full card still carries the named reason and exact byte accounting.
    var compactTransferStatus: String? {
        func percent(_ progress: ProgressSnapshot) -> String {
            "\(Int((progress.fraction * 100).rounded(.down)))%"
        }
        switch self {
        case .notStarted, .awaitingDestination:
            return nil
        case .resolvingIndex:
            return "Preparing download"
        case .active(let progress):
            return "Downloading · \(percent(progress))"
        case .paused(let progress):
            return "Paused · \(percent(progress))"
        case .interrupted(let progress, _):
            return "Interrupted · \(percent(progress)) kept"
        case .verifying:
            return "Checking downloaded files"
        case .cancelled(let progress):
            return "Cancelled · \(percent(progress)) kept"
        case .ready:
            return "Payload checked"
        case .incomplete(let files, _):
            return "Needs \(files) file\(files == 1 ? "" : "s")"
        case .failed:
            return "Download failed"
        }
    }
}

enum InterruptionReason: Equatable, Sendable {
    case volumeDisappeared(String)
    case network
    /// A read or transport ended, but its concrete cause was not proved to be
    /// the network. Preserve the manager's named reason instead of turning
    /// every unknown interruption into a connectivity claim.
    case transfer(String)
    case spaceExhausted(deficit: UInt64)
    case thermal

    var sentence: String {
        switch self {
        case .volumeDisappeared(let name):
            return "\(name) disconnected."
        case .network:
            return "The network went away."
        case .transfer(let reason):
            return reason
        case .spaceExhausted(let deficit):
            return "The volume ran out of room — short by \(MRFormat.bytesDecimal(deficit))."
        case .thermal:
            return "The device is too hot to keep transferring."
        }
    }

    var recoverySentence: String {
        switch self {
        case .volumeDisappeared:
            return "Reconnect the drive, then continue using the verified and partial files kept on disk."
        case .network:
            return "When the network returns, continue using the verified and partial files kept on disk."
        case .transfer:
            return "After resolving the reported condition, continue using the files kept on disk."
        case .spaceExhausted:
            return "Free space, or choose another volume."
        case .thermal:
            return "When the device has cooled, continue using the verified and partial files kept on disk."
        }
    }
}

enum VerificationPhase: Equatable, Sendable {
    /// `DownloadManaging.verify` currently reports only its final file report,
    /// not invented intermediate counts. The UI therefore uses an indeterminate
    /// indicator for exactly the evidence it has.
    case fileDigests

    var sentence: String {
        "Checking file digests against the exact repository version…"
    }
}

/// What the download card renders. A projection of `DownloadProgress` plus the
/// per-file position a resumable transfer has to show.
struct ProgressSnapshot: Equatable, Sendable {
    var totalBytes: UInt64
    var verifiedBytes: UInt64
    var fetchedUnverifiedBytes: UInt64
    var inFlightBytes: UInt64
    var filesTotal: Int
    var filesDone: Int
    var currentFilePath: String?
    var currentFileBytes: UInt64
    var currentFileOffset: UInt64
    var bytesPerSecond: Double
    var estimatedTimeRemaining: TimeInterval?
    var networkBytes: UInt64
    var wastedBytes: UInt64

    var remainingBytes: UInt64 {
        let done = completed
        return totalBytes > done.value ? totalBytes - done.value : 0
    }

    /// Every byte in exactly one bucket. The balance dot reads this.
    var accountsForEveryByte: Bool {
        let done = completed
        guard !done.didOverflow else { return false }
        let (all, overflowed) = done.value.addingReportingOverflow(remainingBytes)
        return !overflowed && all == totalBytes
    }

    var fraction: Double {
        guard totalBytes > 0 else { return accountsForEveryByte ? 1 : 0 }
        let done = completed
        guard !done.didOverflow else { return 0 }
        return min(1, Double(done.value) / Double(totalBytes))
    }

    private var completed: (value: UInt64, didOverflow: Bool) {
        let (first, firstOverflow) = verifiedBytes.addingReportingOverflow(
            fetchedUnverifiedBytes)
        let (second, secondOverflow) = first.addingReportingOverflow(inFlightBytes)
        return (
            firstOverflow || secondOverflow ? UInt64.max : second,
            firstOverflow || secondOverflow)
    }

    /// `file 388 of 470 · resuming at offset 1,073,741,824`
    var resumeSentence: String? {
        guard let path = currentFilePath else { return nil }
        _ = path
        return "file \(filesDone + 1) of \(filesTotal) · resuming at offset "
            + MRFormat.grouped(currentFileOffset)
    }

    static func empty(totalBytes: UInt64, filesTotal: Int) -> ProgressSnapshot {
        ProgressSnapshot(
            totalBytes: totalBytes, verifiedBytes: 0, fetchedUnverifiedBytes: 0,
            inFlightBytes: 0, filesTotal: filesTotal, filesDone: 0, currentFilePath: nil,
            currentFileBytes: 0, currentFileOffset: 0, bytesPerSecond: 0,
            estimatedTimeRemaining: nil, networkBytes: 0, wastedBytes: 0)
    }
}

/// What the app knows about one model's bytes on this machine.
struct InstalledArtifact: Equatable, Sendable, Identifiable {
    var id: ModelID { model }
    let model: ModelID
    var state: DownloadState
    var destinationPath: String?
    var volumeName: String?
    /// Bytes counted on disk after verification. Nil until then — before
    /// verification the catalog's number is *declared*, and this app does not
    /// present a declared number as a fact.
    var measuredBytes: UInt64?
    var verifiedAt: Date?
    var lastVerification: VerificationReport?
}
