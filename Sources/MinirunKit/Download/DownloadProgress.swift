import Foundation
import StorageCore

public struct DownloadJobID: Hashable, Sendable, Codable {
    public let rawValue: UUID
    public init(rawValue: UUID = UUID()) { self.rawValue = rawValue }
}

/// Where a job's bytes are, with every byte in exactly one bucket.
///
/// The four buckets are a partition, not four interesting numbers that happen
/// to be nearby: `verified + fetchedUnverified + inFlight + remaining` is the
/// plan total, and ``accountsForEveryByte`` says so out loud. A progress bar
/// that double-counts a resumed file reads as ahead of itself and then stalls
/// at the end; the identity is what makes that a test failure instead of a
/// support question.
public struct DownloadProgress: Sendable, Equatable, Codable {
    public let job: DownloadJobID
    public let planTotalBytes: UInt64
    /// On disk, digest checked.
    public let verifiedBytes: UInt64
    /// On disk, complete, digest not checked yet.
    public let fetchedUnverifiedBytes: UInt64
    /// In a `.minirun-part` file whose object is not complete.
    public let inFlightBytes: UInt64
    public let remainingBytes: UInt64
    public let filesTotal: Int
    public let filesVerified: Int
    /// Complete on disk at their planned size, digest not checked yet — the
    /// files a resumed job carried in rather than fetched. Counted so that a
    /// resume reports "file 566 of 624", not "file 3 of 624".
    public let filesFetchedUnverified: Int
    public let filesFailed: Int
    /// Bytes off the wire this job, *including* bytes a failed digest threw
    /// away. Not the same as progress, and kept separate so a retry loop that
    /// is burning bandwidth without advancing is visible.
    public let networkBytes: UInt64
    /// Of those, the discarded ones. `networkBytes - wastedBytes` is useful
    /// transfer.
    public let wastedBytes: UInt64
    public let instantaneousBytesPerSecond: Double
    /// 30-second exponentially weighted mean.
    public let smoothedBytesPerSecond: Double
    /// Nil until the smoothed rate is above zero. A time remaining computed
    /// from a rate of zero is infinity, and infinity displayed as "--" is a
    /// worse lie than showing nothing.
    public let estimatedTimeRemaining: TimeInterval?
    public let startedAt: Date
    public let elapsed: TimeInterval

    public init(
        job: DownloadJobID, planTotalBytes: UInt64, verifiedBytes: UInt64,
        fetchedUnverifiedBytes: UInt64, inFlightBytes: UInt64, remainingBytes: UInt64,
        filesTotal: Int, filesVerified: Int, filesFetchedUnverified: Int = 0, filesFailed: Int,
        networkBytes: UInt64, wastedBytes: UInt64,
        instantaneousBytesPerSecond: Double, smoothedBytesPerSecond: Double,
        estimatedTimeRemaining: TimeInterval?, startedAt: Date, elapsed: TimeInterval
    ) {
        self.job = job
        self.planTotalBytes = planTotalBytes
        self.verifiedBytes = verifiedBytes
        self.fetchedUnverifiedBytes = fetchedUnverifiedBytes
        self.inFlightBytes = inFlightBytes
        self.remainingBytes = remainingBytes
        self.filesTotal = filesTotal
        self.filesVerified = filesVerified
        self.filesFetchedUnverified = filesFetchedUnverified
        self.filesFailed = filesFailed
        self.networkBytes = networkBytes
        self.wastedBytes = wastedBytes
        self.instantaneousBytesPerSecond = instantaneousBytesPerSecond
        self.smoothedBytesPerSecond = smoothedBytesPerSecond
        self.estimatedTimeRemaining = estimatedTimeRemaining
        self.startedAt = startedAt
        self.elapsed = elapsed
    }

    /// The partition holds.
    public var accountsForEveryByte: Bool {
        UInt64Accounting.checkedSum([
            verifiedBytes, fetchedUnverifiedBytes, inFlightBytes, remainingBytes,
        ]) == planTotalBytes
    }

    public var fractionComplete: Double {
        guard planTotalBytes > 0 else { return accountsForEveryByte ? 1 : 0 }
        guard let complete = UInt64Accounting.checkedSum([
            verifiedBytes, fetchedUnverifiedBytes,
        ]) else { return 0 }
        return min(1, Double(complete) / Double(planTotalBytes))
    }
}

public struct DownloadSummary: Sendable, Codable, Equatable {
    public let job: DownloadJobID
    public let model: ModelID
    public let repo: HuggingFaceRepoRef
    public let destinationPath: String
    public let bytesWritten: UInt64
    public let networkBytes: UInt64
    public let wastedBytes: UInt64
    public let wallSeconds: TimeInterval
    public let meanBytesPerSecond: Double
    public let verification: VerificationReport
    /// One entry per planned file whose bytes this job read and matched against
    /// the digest the repository publishes — as it landed, in the closing pass,
    /// or both — carrying the filesystem identity the closing pass observed.
    ///
    /// This is what makes the closing verification re-usable instead of
    /// disposable: it is the same per-file shape ``ArtifactVerifier`` records,
    /// so a finished transfer can hand the ledger a full verification without a
    /// third read of the artifact (ADR 0019). A file the job only size-checked
    /// gets no entry, and the hand-off refuses rather than claiming it.
    public let filesVerified: [ArtifactVerifiedFile]

    public init(
        job: DownloadJobID, model: ModelID, repo: HuggingFaceRepoRef, destinationPath: String,
        bytesWritten: UInt64, networkBytes: UInt64, wastedBytes: UInt64,
        wallSeconds: TimeInterval, meanBytesPerSecond: Double, verification: VerificationReport,
        filesVerified: [ArtifactVerifiedFile] = []
    ) {
        self.job = job
        self.model = model
        self.repo = repo
        self.destinationPath = destinationPath
        self.bytesWritten = bytesWritten
        self.networkBytes = networkBytes
        self.wastedBytes = wastedBytes
        self.wallSeconds = wallSeconds
        self.meanBytesPerSecond = meanBytesPerSecond
        self.verification = verification
        self.filesVerified = filesVerified.sorted { $0.path < $1.path }
    }

    private enum CodingKeys: String, CodingKey {
        case job, model, repo, destinationPath, bytesWritten, networkBytes, wastedBytes
        case wallSeconds, meanBytesPerSecond, verification, filesVerified
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            job: try container.decode(DownloadJobID.self, forKey: .job),
            model: try container.decode(ModelID.self, forKey: .model),
            repo: try container.decode(HuggingFaceRepoRef.self, forKey: .repo),
            destinationPath: try container.decode(String.self, forKey: .destinationPath),
            bytesWritten: try container.decode(UInt64.self, forKey: .bytesWritten),
            networkBytes: try container.decode(UInt64.self, forKey: .networkBytes),
            wastedBytes: try container.decode(UInt64.self, forKey: .wastedBytes),
            wallSeconds: try container.decode(TimeInterval.self, forKey: .wallSeconds),
            meanBytesPerSecond: try container.decode(Double.self, forKey: .meanBytesPerSecond),
            verification: try container.decode(VerificationReport.self, forKey: .verification),
            // Absent in any summary written before the hand-off existed. Such a
            // summary proves nothing per file, and the recorder refuses it.
            filesVerified: try container.decodeIfPresent(
                [ArtifactVerifiedFile].self, forKey: .filesVerified) ?? [])
    }
}

public struct DownloadJobSummary: Sendable, Codable, Equatable {
    public enum JobState: String, Codable, Sendable {
        case planned, running, paused, verifying, finished, failed, cancelled
    }

    public let job: DownloadJobID
    public let model: ModelID
    public let state: JobState
    public let progress: DownloadProgress

    public init(
        job: DownloadJobID, model: ModelID, state: JobState, progress: DownloadProgress
    ) {
        self.job = job
        self.model = model
        self.state = state
        self.progress = progress
    }
}

/// One persisted job that could not be reconstructed safely.
public struct DownloadRestoreFailure: Sendable, Equatable {
    public let job: DownloadJobID
    public let destinationPath: String
    public let reason: String

    public init(job: DownloadJobID, destinationPath: String, reason: String) {
        self.job = job
        self.destinationPath = destinationPath
        self.reason = reason
    }
}

/// The explicit result of loading persisted jobs. Recovery never starts a
/// transfer: resumable jobs enter the manager as paused and need a separate
/// `resume` call. That includes a previously finished record, because file size
/// alone cannot carry its old digest verdict across a process boundary.
public struct DownloadRestoreReport: Sendable, Equatable {
    public let restoredJobs: [DownloadJobID]
    public let failures: [DownloadRestoreFailure]

    public init(
        restoredJobs: [DownloadJobID] = [], failures: [DownloadRestoreFailure] = []
    ) {
        self.restoredJobs = restoredJobs
        self.failures = failures
    }
}

/// What a destination directory is, and what a job needs from it.
public struct DownloadPreflight: Sendable, Codable, Equatable {
    public let destination: String
    /// `StorageCore`'s type, reused rather than re-derived. Two answers to "how
    /// much room is left" eventually disagree, and the one that disagrees is
    /// always the one a decision was made on.
    public let storage: StorageInfoSnapshot
    /// `plan.totalBytes` minus what is already verified on disk.
    public let requiredBytes: UInt64
    public let alreadyPresentBytes: UInt64
    public let headroomBytes: UInt64
    public let dependableFreeBytes: Int64?
    public let fits: Bool
    /// Empty if and only if `fits`.
    public let refusals: [DownloadError]
    public let estimatedDuration: TimeInterval?

    public init(
        destination: String, storage: StorageInfoSnapshot, requiredBytes: UInt64,
        alreadyPresentBytes: UInt64, headroomBytes: UInt64, dependableFreeBytes: Int64?,
        fits: Bool, refusals: [DownloadError], estimatedDuration: TimeInterval?
    ) {
        self.destination = destination
        self.storage = storage
        self.requiredBytes = requiredBytes
        self.alreadyPresentBytes = alreadyPresentBytes
        self.headroomBytes = headroomBytes
        self.dependableFreeBytes = dependableFreeBytes
        self.fits = fits
        self.refusals = refusals
        self.estimatedDuration = estimatedDuration
    }
}

/// `StorageCore.StorageInfo` is `Codable` but not `Equatable` across
/// module boundaries in a way that composes with `DownloadError`'s equality, so
/// the handful of fields a preflight quotes travel as a value of their own.
/// Nothing is re-derived: every field is copied straight out of `StorageInfo`.
public struct StorageInfoSnapshot: Sendable, Codable, Equatable {
    public let resolvedPath: String
    public let volumeName: String?
    public let volumeMountPath: String?
    public let filesystemType: String?
    public let volumeTotalBytes: Int64?
    public let volumeAvailableBytes: Int64?
    public let volumeAvailableBytesForImportantUsage: Int64?
    public let isReadOnly: Bool?
    public let isRemovable: Bool?
    public let pathIsWritable: Bool

    public init(_ info: StorageCore.StorageInfo) {
        self.resolvedPath = info.resolvedPath
        self.volumeName = info.volumeName
        self.volumeMountPath = info.volumeMountPath
        self.filesystemType = info.filesystemType
        self.volumeTotalBytes = info.volumeTotalBytes
        self.volumeAvailableBytes = info.volumeAvailableBytes
        self.volumeAvailableBytesForImportantUsage = info.volumeAvailableBytesForImportantUsage
        self.isReadOnly = info.volumeIsReadOnly
        self.isRemovable = info.volumeIsRemovable
        self.pathIsWritable = info.pathIsWritable
    }
}

public enum DownloadEvent: Sendable {
    case planned(DownloadPlan)
    case preflight(DownloadPreflight)
    case fileStarted(path: String, bytes: UInt64, resumedAtByte: UInt64)
    case progress(DownloadProgress)
    case fileVerified(path: String, digest: FileDigest, bytes: UInt64)
    case fileFailed(path: String, DownloadError)
    case paused(DownloadProgress)
    case resumed(DownloadProgress)
    /// A verification pass began — the closing pass after the last byte, or a
    /// pass the operator asked for. Until this existed, a job whose transfer
    /// had ended sat at "downloading, 100%" for the minutes the pass took.
    case verificationStarted(depth: VerificationDepth, filesTotal: Int, bytesTotal: UInt64)
    /// One more file examined by that pass. `bytesChecked` counts every file
    /// the pass has passed over, digested or size-checked.
    case verificationProgress(
        filesChecked: Int, filesTotal: Int, bytesChecked: UInt64, bytesTotal: UInt64)
    case verificationFinished(VerificationReport)
    case finished(DownloadSummary)
    case failed(DownloadError, DownloadProgress)
    case cancelled(DownloadProgress)

    /// Whether this event ends the stream.
    public var isTerminal: Bool {
        switch self {
        case .finished, .failed, .cancelled: return true
        default: return false
        }
    }
}
