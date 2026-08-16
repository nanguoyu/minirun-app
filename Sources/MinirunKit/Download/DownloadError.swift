import Foundation

public struct SizeMismatch: Codable, Sendable, Equatable {
    public let path: String
    public let expected: UInt64
    public let found: UInt64

    public init(path: String, expected: UInt64, found: UInt64) {
        self.path = path
        self.expected = expected
        self.found = found
    }
}

public struct DigestMismatch: Codable, Sendable, Equatable {
    public let path: String
    public let algorithm: DigestAlgorithm
    public let expected: String
    public let found: String

    public init(path: String, algorithm: DigestAlgorithm, expected: String, found: String) {
        self.path = path
        self.algorithm = algorithm
        self.expected = expected
        self.found = found
    }
}

public enum DownloadError: Error, Equatable, CustomStringConvertible {
    case invalidOption(name: String, value: String, allowed: String)
    case destinationNotWritable(path: String, reason: String)
    case destinationIsReadOnlyVolume(path: String)
    case insufficientFreeSpace(
        needBytes: UInt64, headroomBytes: UInt64, availableBytes: Int64, volume: String?)
    /// The origin served a different commit than the plan pinned. A repository
    /// that moved under a running transfer is a different artifact, and
    /// continuing would silently mix two.
    case revisionMoved(expected: String, served: String, path: String)
    case rangeNotSupported(path: String)
    case sizeMismatch(SizeMismatch)
    case digestMismatch(DigestMismatch)
    case digestUnavailable(path: String)
    case transport(status: Int, path: String, attempt: Int)
    case interrupted(path: String, atByte: UInt64, underlying: String)
    /// The volume went away mid-transfer.
    case scopeLost(path: String)
    case cancelled
    case unknownJob(DownloadJobID)
    case jobNotPaused(DownloadJobID)
    case jobAlreadyRunning(DownloadJobID)
    /// The bytes may be intact, but the durable job record is not. Continuing
    /// would turn a resumable transfer into one that silently forgets work.
    case statePersistenceFailed(job: DownloadJobID, reason: String)
    case spaceUnknown(path: String, reason: String)

    public var description: String {
        switch self {
        case .invalidOption(let name, let value, let allowed):
            return "\(name) is \(value); allowed: \(allowed)"
        case .destinationNotWritable(let path, let reason):
            return "'\(path)' is not writable: \(reason)"
        case .destinationIsReadOnlyVolume(let path):
            return "'\(path)' is on a volume this build will not write to"
        case .insufficientFreeSpace(let need, let headroom, let available, let volume):
            let where_ = volume.map { " on \($0)" } ?? ""
            return
                "needs \(need) bytes plus \(headroom) headroom\(where_); \(available) available"
        case .revisionMoved(let expected, let served, let path):
            return "'\(path)' was served from \(served); the plan is pinned to \(expected)"
        case .rangeNotSupported(let path):
            return "the server would not serve a byte range for '\(path)', so it cannot be resumed"
        case .sizeMismatch(let mismatch):
            return
                "'\(mismatch.path)' is \(mismatch.found) bytes; the tree says \(mismatch.expected)"
        case .digestMismatch(let mismatch):
            return
                "'\(mismatch.path)' hashes to \(mismatch.found) (\(mismatch.algorithm.rawValue)); the tree says \(mismatch.expected)"
        case .digestUnavailable(let path):
            return "'\(path)' has no digest in the tree, so it cannot be verified"
        case .transport(let status, let path, let attempt):
            return "HTTP \(status) for '\(path)' on attempt \(attempt)"
        case .interrupted(let path, let byte, let underlying):
            return "'\(path)' was interrupted at byte \(byte): \(underlying)"
        case .scopeLost(let path):
            return "access to '\(path)' was lost mid-transfer"
        case .cancelled:
            return "cancelled"
        case .unknownJob(let job):
            return "no job \(job.rawValue)"
        case .jobNotPaused(let job):
            return "job \(job.rawValue) is not paused"
        case .jobAlreadyRunning(let job):
            return "job \(job.rawValue) is already running"
        case .statePersistenceFailed(let job, let reason):
            return "could not persist job \(job.rawValue): \(reason)"
        case .spaceUnknown(let path, let reason):
            return "free space at '\(path)' could not be determined: \(reason)"
        }
    }
}

/// A refusal has to survive being written down.
///
/// ``DownloadPreflight`` is a document: it is persisted, shown, and sometimes
/// mailed to whoever is asked why a terabyte would not start. Encoding a
/// refusal as its `description` alone would make the reason readable and the
/// *cause* unrecoverable, so every case round-trips field for field. A caller
/// that reads a stored preflight gets back the same value it would have
/// received live, and can still switch on it.
extension DownloadError: Codable {
    private enum CodingKeys: String, CodingKey {
        case kind, name, value, allowed, path, reason, needBytes, headroomBytes, availableBytes
        case volume, expected, served, status, attempt, atByte, underlying, job, mismatch
        case digestMismatch
    }

    private enum Kind: String, Codable {
        case invalidOption, destinationNotWritable, destinationIsReadOnlyVolume
        case insufficientFreeSpace, revisionMoved, rangeNotSupported, sizeMismatch, digestMismatch
        case digestUnavailable, transport, interrupted, scopeLost, cancelled, unknownJob
        case jobNotPaused, jobAlreadyRunning, statePersistenceFailed, spaceUnknown
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .invalidOption(let name, let value, let allowed):
            try c.encode(Kind.invalidOption, forKey: .kind)
            try c.encode(name, forKey: .name)
            try c.encode(value, forKey: .value)
            try c.encode(allowed, forKey: .allowed)
        case .destinationNotWritable(let path, let reason):
            try c.encode(Kind.destinationNotWritable, forKey: .kind)
            try c.encode(path, forKey: .path)
            try c.encode(reason, forKey: .reason)
        case .destinationIsReadOnlyVolume(let path):
            try c.encode(Kind.destinationIsReadOnlyVolume, forKey: .kind)
            try c.encode(path, forKey: .path)
        case .insufficientFreeSpace(let need, let headroom, let available, let volume):
            try c.encode(Kind.insufficientFreeSpace, forKey: .kind)
            try c.encode(need, forKey: .needBytes)
            try c.encode(headroom, forKey: .headroomBytes)
            try c.encode(available, forKey: .availableBytes)
            try c.encodeIfPresent(volume, forKey: .volume)
        case .revisionMoved(let expected, let served, let path):
            try c.encode(Kind.revisionMoved, forKey: .kind)
            try c.encode(expected, forKey: .expected)
            try c.encode(served, forKey: .served)
            try c.encode(path, forKey: .path)
        case .rangeNotSupported(let path):
            try c.encode(Kind.rangeNotSupported, forKey: .kind)
            try c.encode(path, forKey: .path)
        case .sizeMismatch(let mismatch):
            try c.encode(Kind.sizeMismatch, forKey: .kind)
            try c.encode(mismatch, forKey: .mismatch)
        case .digestMismatch(let mismatch):
            try c.encode(Kind.digestMismatch, forKey: .kind)
            try c.encode(mismatch, forKey: .digestMismatch)
        case .digestUnavailable(let path):
            try c.encode(Kind.digestUnavailable, forKey: .kind)
            try c.encode(path, forKey: .path)
        case .transport(let status, let path, let attempt):
            try c.encode(Kind.transport, forKey: .kind)
            try c.encode(status, forKey: .status)
            try c.encode(path, forKey: .path)
            try c.encode(attempt, forKey: .attempt)
        case .interrupted(let path, let atByte, let underlying):
            try c.encode(Kind.interrupted, forKey: .kind)
            try c.encode(path, forKey: .path)
            try c.encode(atByte, forKey: .atByte)
            try c.encode(underlying, forKey: .underlying)
        case .scopeLost(let path):
            try c.encode(Kind.scopeLost, forKey: .kind)
            try c.encode(path, forKey: .path)
        case .cancelled:
            try c.encode(Kind.cancelled, forKey: .kind)
        case .unknownJob(let job):
            try c.encode(Kind.unknownJob, forKey: .kind)
            try c.encode(job, forKey: .job)
        case .jobNotPaused(let job):
            try c.encode(Kind.jobNotPaused, forKey: .kind)
            try c.encode(job, forKey: .job)
        case .jobAlreadyRunning(let job):
            try c.encode(Kind.jobAlreadyRunning, forKey: .kind)
            try c.encode(job, forKey: .job)
        case .statePersistenceFailed(let job, let reason):
            try c.encode(Kind.statePersistenceFailed, forKey: .kind)
            try c.encode(job, forKey: .job)
            try c.encode(reason, forKey: .reason)
        case .spaceUnknown(let path, let reason):
            try c.encode(Kind.spaceUnknown, forKey: .kind)
            try c.encode(path, forKey: .path)
            try c.encode(reason, forKey: .reason)
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        switch try c.decode(Kind.self, forKey: .kind) {
        case .invalidOption:
            self = .invalidOption(
                name: try c.decode(String.self, forKey: .name),
                value: try c.decode(String.self, forKey: .value),
                allowed: try c.decode(String.self, forKey: .allowed))
        case .destinationNotWritable:
            self = .destinationNotWritable(
                path: try c.decode(String.self, forKey: .path),
                reason: try c.decode(String.self, forKey: .reason))
        case .destinationIsReadOnlyVolume:
            self = .destinationIsReadOnlyVolume(path: try c.decode(String.self, forKey: .path))
        case .insufficientFreeSpace:
            self = .insufficientFreeSpace(
                needBytes: try c.decode(UInt64.self, forKey: .needBytes),
                headroomBytes: try c.decode(UInt64.self, forKey: .headroomBytes),
                availableBytes: try c.decode(Int64.self, forKey: .availableBytes),
                volume: try c.decodeIfPresent(String.self, forKey: .volume))
        case .revisionMoved:
            self = .revisionMoved(
                expected: try c.decode(String.self, forKey: .expected),
                served: try c.decode(String.self, forKey: .served),
                path: try c.decode(String.self, forKey: .path))
        case .rangeNotSupported:
            self = .rangeNotSupported(path: try c.decode(String.self, forKey: .path))
        case .sizeMismatch:
            self = .sizeMismatch(try c.decode(SizeMismatch.self, forKey: .mismatch))
        case .digestMismatch:
            self = .digestMismatch(try c.decode(DigestMismatch.self, forKey: .digestMismatch))
        case .digestUnavailable:
            self = .digestUnavailable(path: try c.decode(String.self, forKey: .path))
        case .transport:
            self = .transport(
                status: try c.decode(Int.self, forKey: .status),
                path: try c.decode(String.self, forKey: .path),
                attempt: try c.decode(Int.self, forKey: .attempt))
        case .interrupted:
            self = .interrupted(
                path: try c.decode(String.self, forKey: .path),
                atByte: try c.decode(UInt64.self, forKey: .atByte),
                underlying: try c.decode(String.self, forKey: .underlying))
        case .scopeLost:
            self = .scopeLost(path: try c.decode(String.self, forKey: .path))
        case .cancelled:
            self = .cancelled
        case .unknownJob:
            self = .unknownJob(try c.decode(DownloadJobID.self, forKey: .job))
        case .jobNotPaused:
            self = .jobNotPaused(try c.decode(DownloadJobID.self, forKey: .job))
        case .jobAlreadyRunning:
            self = .jobAlreadyRunning(try c.decode(DownloadJobID.self, forKey: .job))
        case .statePersistenceFailed:
            self = .statePersistenceFailed(
                job: try c.decode(DownloadJobID.self, forKey: .job),
                reason: try c.decode(String.self, forKey: .reason))
        case .spaceUnknown:
            self = .spaceUnknown(
                path: try c.decode(String.self, forKey: .path),
                reason: try c.decode(String.self, forKey: .reason))
        }
    }
}
