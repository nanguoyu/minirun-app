import Foundation

/// What is known about one file of one job.
public enum FileState: Codable, Sendable, Equatable {
    case pending
    /// A `.minirun-part` file exists with this many bytes in it.
    case partial(bytesOnDisk: UInt64)
    /// Complete on disk, digest not checked.
    case fetched(bytesOnDisk: UInt64)
    case verified(bytes: UInt64, at: Date)
    case failed(reason: String, attempts: Int)

    public var bytesOnDisk: UInt64 {
        switch self {
        case .pending, .failed: return 0
        case .partial(let bytes), .fetched(let bytes): return bytes
        case .verified(let bytes, _): return bytes
        }
    }

    private enum CodingKeys: String, CodingKey { case kind, bytes, at, reason, attempts }
    private enum Kind: String, Codable { case pending, partial, fetched, verified, failed }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(Kind.self, forKey: .kind) {
        case .pending:
            self = .pending
        case .partial:
            self = .partial(bytesOnDisk: try container.decode(UInt64.self, forKey: .bytes))
        case .fetched:
            self = .fetched(bytesOnDisk: try container.decode(UInt64.self, forKey: .bytes))
        case .verified:
            self = .verified(
                bytes: try container.decode(UInt64.self, forKey: .bytes),
                at: try container.decode(Date.self, forKey: .at))
        case .failed:
            self = .failed(
                reason: try container.decode(String.self, forKey: .reason),
                attempts: try container.decode(Int.self, forKey: .attempts))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .pending:
            try container.encode(Kind.pending, forKey: .kind)
        case .partial(let bytes):
            try container.encode(Kind.partial, forKey: .kind)
            try container.encode(bytes, forKey: .bytes)
        case .fetched(let bytes):
            try container.encode(Kind.fetched, forKey: .kind)
            try container.encode(bytes, forKey: .bytes)
        case .verified(let bytes, let at):
            try container.encode(Kind.verified, forKey: .kind)
            try container.encode(bytes, forKey: .bytes)
            try container.encode(at, forKey: .at)
        case .failed(let reason, let attempts):
            try container.encode(Kind.failed, forKey: .kind)
            try container.encode(reason, forKey: .reason)
            try container.encode(attempts, forKey: .attempts)
        }
    }
}

/// Everything needed to pick a job back up in a later process.
public struct DownloadJobState: Codable, Sendable, Equatable {
    public let job: DownloadJobID
    public let plan: DownloadPlan
    public let destinationPath: String
    /// Security-scoped bookmark for the destination.
    ///
    /// Carried per job because two jobs can be writing to two volumes. M10 recorded an iOS
    /// grant surviving both a reboot and a replug, so a job started before a
    /// restart is genuinely resumable without prompting the operator again —
    /// which is the only reason storing this is worth the bytes.
    public let destinationBookmark: Data?
    /// Physical directory identity captured before the first transfer. The
    /// spelling in `destinationPath` is not enough: a removable volume can be
    /// unmounted and a different directory created at the same path.
    public let artifactRootPath: String?
    public let artifactRootDevice: UInt64?
    public let artifactRootInode: UInt64?
    public let options: DownloadOptions
    public let fileStates: [String: FileState]
    /// Last durable lifecycle state. Older records decode this as `nil` and
    /// are recovered conservatively as paused, never auto-started.
    public let runState: DownloadJobSummary.JobState?
    public let createdAt: Date
    public let updatedAt: Date

    public init(
        job: DownloadJobID, plan: DownloadPlan, destinationPath: String,
        destinationBookmark: Data?, artifactRootPath: String? = nil,
        artifactRootDevice: UInt64? = nil, artifactRootInode: UInt64? = nil,
        options: DownloadOptions,
        fileStates: [String: FileState], runState: DownloadJobSummary.JobState? = nil,
        createdAt: Date, updatedAt: Date
    ) {
        self.job = job
        self.plan = plan
        self.destinationPath = destinationPath
        self.destinationBookmark = destinationBookmark
        self.artifactRootPath = artifactRootPath
        self.artifactRootDevice = artifactRootDevice
        self.artifactRootInode = artifactRootInode
        self.options = options
        self.fileStates = fileStates
        self.runState = runState
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

public protocol DownloadStateStore: Sendable {
    func save(_ state: DownloadJobState) throws
    func load(_ job: DownloadJobID) throws -> DownloadJobState?
    func all() throws -> [DownloadJobState]
    func remove(_ job: DownloadJobID) throws
}

/// One JSON file per job in a directory.
public struct FileDownloadStateStore: DownloadStateStore {
    public let directory: URL

    public init(directory: URL) { self.directory = directory }

    private func url(_ job: DownloadJobID) -> URL {
        directory.appendingPathComponent("\(job.rawValue.uuidString).json")
    }

    public func save(_ state: DownloadJobState) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let data = try MinirunKitJSON.encoder().encode(state)
        try data.write(to: url(state.job), options: .atomic)
    }

    public func load(_ job: DownloadJobID) throws -> DownloadJobState? {
        let file = url(job)
        guard FileManager.default.fileExists(atPath: file.path) else { return nil }
        return try MinirunKitJSON.decoder().decode(
            DownloadJobState.self, from: try Data(contentsOf: file))
    }

    public func all() throws -> [DownloadJobState] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: directory.path, isDirectory: &isDirectory)
        else { return [] }
        guard isDirectory.boolValue else {
            throw CocoaError(.fileReadUnknown)
        }
        let names = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        let decoder = MinirunKitJSON.decoder()
        return try names
            .filter { $0.pathExtension == "json" }
            .map { url in
                try decoder.decode(
                    DownloadJobState.self, from: Data(contentsOf: url))
            }
            .sorted { $0.createdAt < $1.createdAt }
    }

    public func remove(_ job: DownloadJobID) throws {
        let file = url(job)
        guard FileManager.default.fileExists(atPath: file.path) else { return }
        try FileManager.default.removeItem(at: file)
    }
}

/// A store that lives only as long as the process. Used by tests, and by any
/// caller that does not want a job to outlive a launch.
public final class InMemoryDownloadStateStore: DownloadStateStore, @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [DownloadJobID: DownloadJobState] = [:]

    public init() {}

    public func save(_ state: DownloadJobState) throws {
        lock.lock()
        defer { lock.unlock() }
        storage[state.job] = state
    }

    public func load(_ job: DownloadJobID) throws -> DownloadJobState? {
        lock.lock()
        defer { lock.unlock() }
        return storage[job]
    }

    public func all() throws -> [DownloadJobState] {
        lock.lock()
        defer { lock.unlock() }
        return storage.values.sorted { $0.createdAt < $1.createdAt }
    }

    public func remove(_ job: DownloadJobID) throws {
        lock.lock()
        defer { lock.unlock() }
        storage.removeValue(forKey: job)
    }
}
