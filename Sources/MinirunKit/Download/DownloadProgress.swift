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
        filesTotal: Int, filesVerified: Int, filesFailed: Int,
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

    public init(
        job: DownloadJobID, model: ModelID, repo: HuggingFaceRepoRef, destinationPath: String,
        bytesWritten: UInt64, networkBytes: UInt64, wastedBytes: UInt64,
        wallSeconds: TimeInterval, meanBytesPerSecond: Double, verification: VerificationReport
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
