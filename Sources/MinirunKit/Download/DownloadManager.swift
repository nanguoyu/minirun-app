import Darwin
import Foundation
import StorageCore

/// Where a job writes, and the grant that lets it.
public struct DownloadDestination: Sendable {
    public let directory: URL
    /// Non-nil on iOS for anything outside the app container. Held for the
    /// whole transfer and released when the job reaches a terminal state — a
    /// closure-shaped bracket cannot span hours of `await`.
    public let scope: StorageScope?

    public init(directory: URL, scope: StorageScope? = nil) {
        self.directory = directory
        self.scope = scope
    }
}

private extension DownloadError {
    var isStatePersistenceFailure: Bool {
        if case .statePersistenceFailed = self { return true }
        return false
    }
}

/// Revokes a borrowed HTTP sink when the Swift task that owns the request is
/// cancelled. `JobControl` covers user cancellation and persistence failure;
/// this separate latch covers task-group cancellation even when a custom
/// transport does not cooperate with Swift cancellation itself.
private final class HTTPDownloadCancellationGate: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }
}

public protocol DownloadManaging: Sendable {
    func plan(for model: ModelDescriptor) async throws -> DownloadPlan
    func preflight(
        _ plan: DownloadPlan, into destination: DownloadDestination, options: DownloadOptions
    ) async throws -> DownloadPreflight
    @discardableResult
    func start(
        _ plan: DownloadPlan, into destination: DownloadDestination, options: DownloadOptions
    ) async throws -> DownloadJobID
    func events(for job: DownloadJobID) -> AsyncStream<DownloadEvent>
    func pause(_ job: DownloadJobID) async throws
    func resume(_ job: DownloadJobID, scope: StorageScope?) async throws
    func cancel(_ job: DownloadJobID, keepPartialFiles: Bool) async throws
    func verify(_ job: DownloadJobID, depth: VerificationDepth) async throws -> VerificationReport
    /// Re-fetches exactly ``VerificationReport/pathsToRefetch``. Same job id.
    @discardableResult
    func repair(_ job: DownloadJobID, from report: VerificationReport) async throws -> DownloadJobID
    func jobs() async -> [DownloadJobSummary]
    /// Reconstruct saved jobs without starting network work. Resumable records
    /// return as paused and require an explicit `resume`; even a previously
    /// finished record must re-establish its digest evidence after a relaunch.
    func restorePersistedJobs() async throws -> DownloadRestoreReport
}

extension DownloadManaging {
    public func restorePersistedJobs() async throws -> DownloadRestoreReport {
        DownloadRestoreReport()
    }
}

/// Fetches an artifact, resumably, and checks every byte it wrote.
///
/// Three rules are structural rather than defensive, and each one exists
/// because its absence produces a green result over wrong bytes:
///
/// 1. **The part file's length is the resume point.** A recorded `FileState`
///    that disagrees is discarded and logged, never trusted (spec §5.10). The
///    record can be stale in ways the file cannot.
/// 2. **`x-repo-commit` is checked on every response.** A repository that moved
///    under a running transfer is a different artifact; continuing would blend
///    two of them into one directory that verifies against neither.
/// 3. **The CDN's `ETag` is never read as a digest.** It is the Xet content
///    hash, not the SHA-256. Comparing it against itself would pass forever.
public actor DownloadManager: DownloadManaging {
    typealias DigestOperation = @Sendable (
        Int32, DigestAlgorithm, @escaping @Sendable () -> Bool
    ) throws -> FileDigest

    /// Suffix for bytes that are on disk but not yet vouched for. A file is
    /// renamed out of this name only after its digest passes, so anything
    /// without the suffix in a destination has been checked.
    public static let partSuffix = ".minirun-part"

    private struct Job {
        var state: DownloadJobState
        var runState: DownloadJobSummary.JobState
        var destination: DownloadDestination
        /// The physical artifact root at registration time. Repair compares
        /// against it before deleting anything, so replacing a destination
        /// path with a symlink cannot retarget an old report at another tree.
        let artifactRoot: URL
        /// Path equality alone cannot detect `rename(oldRoot)` followed by a
        /// fresh directory at the same spelling. Device + inode binds later
        /// mutations to the directory object registration actually opened.
        let artifactRootIdentity: ArtifactRootIdentity
        let counters: JobCounters
        let control: JobControl
        var startedAt: Date
        var lastSampleAt: Date
        var lastSampleBytes: UInt64
        var smoothedBytesPerSecond: Double
        var lastInstantaneous: Double = 0
        var task: Task<Void, Never>?
    }

    private struct ArtifactPaths {
        let final: URL
        let part: URL
    }

    private struct ArtifactRootIdentity: Equatable {
        let device: UInt64
        let inode: UInt64
    }

    private struct ArtifactLeafIdentity: Equatable, Sendable {
        let device: UInt64
        let inode: UInt64
    }

    private struct OpenArtifactLeaf: Sendable {
        let descriptor: Int32
        let identity: ArtifactLeafIdentity
        let sizeBytes: UInt64
    }

    private enum AnchoredPathError: Error {
        case missingParent
    }

    /// A plan after both of its authority boundaries have been checked: its
    /// public file table, and the concrete destination tree those files would
    /// touch. Callers use the checked total rather than re-running a trapping
    /// `UInt64` reduction over a publicly constructible plan.
    private struct ValidatedPlan {
        let totalBytes: UInt64
        let lexicalRoot: URL
        let physicalRoot: URL
        let rootIdentity: ArtifactRootIdentity?
        let pathsByFile: [String: ArtifactPaths]
    }

    /// Time constant of the smoothed rate, in seconds.
    private static let rateTimeConstant: Double = 30

    private let transport: any HTTPTransport
    private let storage: any StorageManaging
    private let stateStore: any DownloadStateStore
    private let endpoint: URL
    private let token: (@Sendable () -> String?)?
    private let digestOperation: DigestOperation
    private nonisolated let hub = DownloadEventHub()

    private var jobs: [DownloadJobID: Job] = [:]

    public init(
        transport: any HTTPTransport = URLSessionTransport(),
        storage: any StorageManaging = StorageManager(),
        stateStore: any DownloadStateStore,
        catalogEndpoint: URL = HuggingFaceEndpoint.default,
        token: (@Sendable () -> String?)? = nil
    ) {
        self.transport = transport
        self.storage = storage
        self.stateStore = stateStore
        self.endpoint = catalogEndpoint
        self.token = token
        self.digestOperation = { descriptor, algorithm, cancellationRequested in
            try FileDigestComputer.digest(
                ofOpenFileDescriptor: descriptor, algorithm: algorithm,
                cancellationRequested: cancellationRequested)
        }
    }

    init(
        transport: any HTTPTransport, storage: any StorageManaging,
        stateStore: any DownloadStateStore, catalogEndpoint: URL,
        token: (@Sendable () -> String?)? = nil,
        digestOperation: @escaping DigestOperation
    ) {
        self.transport = transport
        self.storage = storage
        self.stateStore = stateStore
        self.endpoint = catalogEndpoint
        self.token = token
        self.digestOperation = digestOperation
    }

    // MARK: - Planning

    public func plan(for model: ModelDescriptor) async throws -> DownloadPlan {
        guard let repo = model.source.repo else {
            if case .locallyBuilt(let tool) = model.source {
                throw CatalogError.notDownloadable(model.id, builtBy: tool)
            }
            throw CatalogError.unknownModel(model.id)
        }
        let client = HuggingFaceTreeClient(transport: transport, endpoint: endpoint, token: token)
        let files = try await client.files(in: repo)
        guard !files.isEmpty else {
            throw CatalogError.malformedResponse("\(repo.repoID) at \(repo.revision) lists no files")
        }
        // A repository with no `index.json`, or one whose `index.json` will not
        // come down, still has a complete tree — and the tree is what gets
        // fetched. The claim is recorded when it is available and its absence
        // is noted in the reconciliation, but it never blocks a plan.
        let indexBody = try? await client.fetchSmallFile("index.json", in: repo)
        let claim = indexBody.flatMap { $0 }.flatMap(IndexClaim.parse)
        return DownloadPlan(
            model: model.id, repo: repo, files: files, index: claim,
            reconciliation: IndexReconciliation.between(files: files, claim: claim))
    }

    // MARK: - Preflight

    public func preflight(
        _ plan: DownloadPlan, into destination: DownloadDestination, options: DownloadOptions
    ) async throws -> DownloadPreflight {
        let options = try options.validated()
        let validated = try Self.validate(plan, at: destination.directory)
        return try makePreflight(
            plan, destination: destination, options: options, validated: validated)
    }

    private func makePreflight(
        _ plan: DownloadPlan, destination: DownloadDestination, options: DownloadOptions,
        validated: ValidatedPlan
    ) throws -> DownloadPreflight {
        let info = storage.describe(destination.directory)
        var refusals: [DownloadError] = []

        let present = alreadyPresentBytes(plan, pathsByFile: validated.pathsByFile)
        let required = validated.totalBytes - present
        try Self.validateCapacityArithmetic(
            requiredBytes: required, headroomBytes: options.freeSpaceHeadroomBytes)

        if info.pathExists && !info.pathIsWritable {
            refusals.append(
                .destinationNotWritable(
                    path: info.resolvedPath, reason: "the filesystem reports it as not writable"))
        }
        switch storage.canHold(
            bytes: required, at: destination.directory,
            headroomBytes: options.freeSpaceHeadroomBytes)
        {
        case .fits:
            break
        case .refused(let error):
            refusals.append(error)
        case .unknown(let reason):
            refusals.append(.spaceUnknown(path: info.resolvedPath, reason: reason))
        }

        return DownloadPreflight(
            destination: info.resolvedPath,
            storage: StorageInfoSnapshot(info),
            requiredBytes: required,
            alreadyPresentBytes: present,
            headroomBytes: options.freeSpaceHeadroomBytes,
            dependableFreeBytes: info.volumeAvailableBytes,
            fits: refusals.isEmpty,
            refusals: refusals,
            // No rate has been observed yet, and a duration extrapolated from a
            // nominal link speed is a number nobody measured.
            estimatedDuration: nil)
    }

    /// Bytes of the plan already on disk at the right size. Size only: this runs
    /// before a transfer starts and must not read a terabyte to answer.
    private func alreadyPresentBytes(
        _ plan: DownloadPlan, pathsByFile: [String: ArtifactPaths]
    ) -> UInt64 {
        var total: UInt64 = 0
        for file in plan.files {
            guard let url = pathsByFile[file.path]?.final else { continue }
            guard let size = try? FileDigestComputer.byteCount(ofFileAt: url) else { continue }
            if size == file.sizeBytes { total += size }
        }
        return total
    }

    // MARK: - Starting, pausing, resuming, cancelling

    @discardableResult
    public func start(
        _ plan: DownloadPlan, into destination: DownloadDestination, options: DownloadOptions
    ) async throws -> DownloadJobID {
        // Passing a scope transfers ownership immediately. Every throw before
        // registration releases it exactly once; success transfers it to the
        // installed Job, whose terminal path owns the release.
        var registered = false
        defer {
            if !registered { destination.scope?.release() }
        }
        let options = try options.validated()
        let initiallyValidated = try Self.validate(plan, at: destination.directory)

        let id = DownloadJobID()
        hub.emit(.planned(plan), for: id)

        let check = try makePreflight(
            plan, destination: destination, options: options, validated: initiallyValidated)
        hub.emit(.preflight(check), for: id)
        // Refused, not clamped and not attempted anyway. The first refusal is
        // thrown because it is the one the operator has to act on; the rest
        // travel on the preflight event, which was already emitted.
        if let refusal = check.refusals.first { throw refusal }

        // Re-resolve immediately before and after creation. The first check
        // prevents a destination alias from being switched after preflight;
        // the second binds the job to the directory that was actually made.
        _ = try Self.validate(
            plan, at: destination.directory,
            expectedPhysicalRoot: initiallyValidated.physicalRoot,
            expectedRootIdentity: initiallyValidated.rootIdentity)
        try FileManager.default.createDirectory(
            at: destination.directory, withIntermediateDirectories: true)
        let registeredPlan = try Self.validate(
            plan, at: destination.directory,
            expectedPhysicalRoot: initiallyValidated.physicalRoot,
            expectedRootIdentity: initiallyValidated.rootIdentity)
        try register(
            id: id, plan: plan, destination: destination, options: options,
            validated: registeredPlan)
        registered = true
        return id
    }

    private func register(
        id: DownloadJobID, plan: DownloadPlan, destination: DownloadDestination,
        options: DownloadOptions, validated: ValidatedPlan
    ) throws {
        guard let rootIdentity = validated.rootIdentity else {
            throw Self.planRefusal(
                "downloadDestination.root", validated.physicalRoot.path,
                "an existing artifact directory")
        }
        let now = Date()
        // A job that cannot preserve its destination authority cannot honestly
        // call itself resumable. Refuse before a task or network request exists
        // instead of saving a plain path as a silent fallback.
        let bookmark = try storage.bookmarkData(for: destination.directory)
        let rebound = try Self.validate(
            plan, at: destination.directory,
            expectedPhysicalRoot: validated.physicalRoot,
            expectedRootIdentity: rootIdentity)
        let state = DownloadJobState(
            job: id, plan: plan, destinationPath: destination.directory.path,
            destinationBookmark: bookmark,
            artifactRootPath: validated.physicalRoot.path,
            artifactRootDevice: rootIdentity.device,
            artifactRootInode: rootIdentity.inode,
            options: options,
            fileStates: adoptFileStates(plan: plan, pathsByFile: rebound.pathsByFile),
            runState: .running,
            createdAt: now, updatedAt: now)
        _ = try Self.validate(
            plan, at: destination.directory,
            expectedPhysicalRoot: validated.physicalRoot,
            expectedRootIdentity: rootIdentity)
        try saveState(state)

        var job = Job(
            state: state, runState: .running, destination: destination,
            artifactRoot: validated.physicalRoot,
            artifactRootIdentity: rootIdentity,
            counters: JobCounters(), control: JobControl(), startedAt: now,
            lastSampleAt: now, lastSampleBytes: 0, smoothedBytesPerSecond: 0, task: nil)
        job.task = Task { [weak self] in
            await self?.transfer(id, restrictedTo: nil)
        }
        jobs[id] = job
    }

    /// Reconcile recorded file states against what is actually on disk.
    ///
    /// The file wins, always. Recorded offsets are discarded in favour of the
    /// part file's length, and even a recorded `.verified` final returns as
    /// `.fetched`: size alone cannot carry a digest verdict across a pause or a
    /// process death. A record can be stale in every way a file cannot, and
    /// resuming from a remembered offset into a shorter file writes a hole that
    /// no size check would catch.
    private func adoptFileStates(
        plan: DownloadPlan, pathsByFile: [String: ArtifactPaths]
    ) -> [String: FileState] {
        var out: [String: FileState] = [:]
        for file in plan.files {
            guard let paths = pathsByFile[file.path] else { continue }
            let final = paths.final
            let part = paths.part
            if let size = try? FileDigestComputer.byteCount(ofFileAt: final), size == file.sizeBytes
            {
                // A durable record cannot prove that same-sized bytes were not
                // replaced while this process was away. Carry the file, not
                // the old digest verdict; the closing verification will read
                // it again before the job can finish.
                out[file.path] = .fetched(bytesOnDisk: size)
                continue
            }
            if let size = try? FileDigestComputer.byteCount(ofFileAt: part), size > 0 {
                out[file.path] = .partial(bytesOnDisk: min(size, file.sizeBytes))
                continue
            }
            out[file.path] = .pending
        }
        return out
    }

    public func restorePersistedJobs() async throws -> DownloadRestoreReport {
        let states = try stateStore.all()
        var restored: [DownloadJobID] = []
        var failures: [DownloadRestoreFailure] = []

        for saved in states {
            if jobs[saved.job] != nil {
                restored.append(saved.job)
                continue
            }

            do {
                let options = try saved.options.validated()
                guard let bookmark = saved.destinationBookmark else {
                    throw StorageError.bookmarkDataUnresolvable(
                        reason: "job \(saved.job.rawValue) has no saved destination bookmark")
                }
                let resolved = try storage.resolveBookmarkData(bookmark)
                defer { resolved.scope.release() }

                let recordedPath = URL(fileURLWithPath: saved.destinationPath)
                    .standardizedFileURL.path
                let resolvedPath = resolved.scope.url.standardizedFileURL.path
                guard recordedPath == resolvedPath else {
                    throw StorageError.bookmarkDataUnresolvable(
                        reason:
                            "job \(saved.job.rawValue) recorded '\(recordedPath)' but its bookmark resolved to '\(resolvedPath)'")
                }

                guard
                    let savedRootPath = saved.artifactRootPath,
                    let savedDevice = saved.artifactRootDevice,
                    let savedInode = saved.artifactRootInode
                else {
                    throw Self.planRefusal(
                        "downloadDestination.identity", "missing",
                        "the physical root path, device and inode saved when the job started")
                }
                let rootIdentity = ArtifactRootIdentity(
                    device: savedDevice, inode: savedInode)
                let savedPhysicalRoot = URL(fileURLWithPath: savedRootPath)
                    .standardizedFileURL
                let validated = try Self.validate(
                    saved.plan, at: resolved.scope.url,
                    expectedPhysicalRoot: savedPhysicalRoot,
                    expectedRootIdentity: rootIdentity)

                let durableState: DownloadJobSummary.JobState
                switch saved.runState {
                case .finished:
                    // File size survives a relaunch as evidence; an old digest
                    // verdict does not. Re-enter through `resume`, whose final
                    // verification re-reads every carried final file.
                    durableState = .paused
                case .failed:
                    durableState = .failed
                case .cancelled:
                    durableState = .cancelled
                case .none, .planned, .running, .paused, .verifying:
                    durableState = .paused
                }
                let now = Date()
                let restoredState = DownloadJobState(
                    job: saved.job, plan: saved.plan, destinationPath: resolvedPath,
                    destinationBookmark: resolved.bookmarkData,
                    artifactRootPath: savedPhysicalRoot.path,
                    artifactRootDevice: rootIdentity.device,
                    artifactRootInode: rootIdentity.inode,
                    options: options,
                    fileStates: adoptFileStates(
                        plan: saved.plan, pathsByFile: validated.pathsByFile),
                    runState: durableState, createdAt: saved.createdAt, updatedAt: now)
                // The rewritten bookmark, disk-derived file states and paused
                // lifecycle become durable before this process advertises the
                // restored job.
                try saveState(restoredState)

                let control = JobControl()
                control.pause()
                jobs[saved.job] = Job(
                    state: restoredState, runState: durableState,
                    destination: DownloadDestination(directory: resolved.scope.url),
                    artifactRoot: validated.physicalRoot,
                    artifactRootIdentity: rootIdentity,
                    counters: JobCounters(), control: control,
                    startedAt: saved.createdAt, lastSampleAt: now,
                    lastSampleBytes: 0, smoothedBytesPerSecond: 0, task: nil)
                if durableState == .paused {
                    hub.emit(.paused(progress(of: saved.job)), for: saved.job)
                } else if durableState == .cancelled {
                    hub.emit(.cancelled(progress(of: saved.job)), for: saved.job)
                } else if durableState == .failed {
                    hub.emit(
                        .failed(
                            .interrupted(
                                path: resolvedPath, atByte: 0,
                                underlying: "restored persisted job was already failed"),
                            progress(of: saved.job)),
                        for: saved.job)
                }
                restored.append(saved.job)
            } catch {
                failures.append(
                    DownloadRestoreFailure(
                        job: saved.job, destinationPath: saved.destinationPath,
                        reason: String(describing: error)))
            }
        }

        return DownloadRestoreReport(
            restoredJobs: restored.sorted {
                $0.rawValue.uuidString < $1.rawValue.uuidString
            },
            failures: failures.sorted {
                $0.job.rawValue.uuidString < $1.job.rawValue.uuidString
            })
    }

    public func pause(_ id: DownloadJobID) async throws {
        guard let job = jobs[id] else { throw DownloadError.unknownJob(id) }
        guard job.runState == .running else { return }
        job.control.pause()
    }

    public func resume(_ id: DownloadJobID, scope: StorageScope?) async throws {
        guard var job = jobs[id] else { throw DownloadError.unknownJob(id) }
        guard job.runState == .paused else { throw DownloadError.jobNotPaused(id) }
        let activeScope: StorageScope
        let bookmark: Data
        if let scope {
            activeScope = scope
            do {
                bookmark = try storage.bookmarkData(for: scope.url)
            } catch {
                scope.release()
                throw error
            }
        } else {
            guard let savedBookmark = job.state.destinationBookmark else {
                throw StorageError.bookmarkDataUnresolvable(
                    reason: "job \(id.rawValue) has no saved destination bookmark")
            }
            let resolved = try storage.resolveBookmarkData(savedBookmark)
            activeScope = resolved.scope
            bookmark = resolved.bookmarkData
        }

        do {
            guard !activeScope.isReleased else {
                throw DownloadError.scopeLost(path: activeScope.url.path)
            }
            let recordedPath = URL(fileURLWithPath: job.state.destinationPath)
                .standardizedFileURL.path
            let resolvedPath = activeScope.url.standardizedFileURL.path
            guard recordedPath == resolvedPath else {
                throw DownloadError.scopeLost(path: resolvedPath)
            }
            let validated = try Self.validate(
                job.state.plan, at: activeScope.url,
                expectedPhysicalRoot: job.artifactRoot,
                expectedRootIdentity: job.artifactRootIdentity)

            // The record is re-derived from disk here, not trusted forward:
            // the pause may have outlived the process, and part files may have
            // been touched by something else in between.
            let fileStates = adoptFileStates(
                plan: job.state.plan, pathsByFile: validated.pathsByFile)
            job.state = DownloadJobState(
                job: job.state.job, plan: job.state.plan,
                destinationPath: resolvedPath, destinationBookmark: bookmark,
                artifactRootPath: job.artifactRoot.path,
                artifactRootDevice: job.artifactRootIdentity.device,
                artifactRootInode: job.artifactRootIdentity.inode,
                options: job.state.options, fileStates: fileStates,
                runState: .running, createdAt: job.state.createdAt, updatedAt: Date())
            // Persist the refreshed bookmark and disk truth before opening the
            // network. If saving fails, the job remains paused and the grant is
            // released rather than creating an unresumable in-flight transfer.
            try saveState(job.state)

            job.destination.scope?.release()
            job.destination = DownloadDestination(directory: activeScope.url, scope: activeScope)
            job.control.unpause()
            job.runState = .running
            hub.restart(id)
            jobs[id] = job
            job.task = Task { [weak self] in
                await self?.transfer(id, restrictedTo: nil)
            }
            jobs[id] = job
            hub.emit(.resumed(progress(of: id)), for: id)
        } catch {
            activeScope.release()
            throw error
        }
    }

    public func cancel(_ id: DownloadJobID, keepPartialFiles: Bool) async throws {
        guard let job = jobs[id] else { throw DownloadError.unknownJob(id) }
        switch job.runState {
        case .finished, .failed, .cancelled:
            return
        default:
            break
        }
        job.control.cancel(keepPartialFiles: keepPartialFiles)
        if job.runState != .running, job.runState != .verifying {
            if let failure = finishCancelled(id, keepPartialFiles: keepPartialFiles) {
                throw failure
            }
        }
    }

    public func jobs() async -> [DownloadJobSummary] {
        jobs.keys
            .map { id in
                DownloadJobSummary(
                    job: id, model: jobs[id]!.state.plan.model, state: jobs[id]!.runState,
                    progress: progress(of: id))
            }
            .sorted { $0.job.rawValue.uuidString < $1.job.rawValue.uuidString }
    }

    public nonisolated func events(for job: DownloadJobID) -> AsyncStream<DownloadEvent> {
        hub.stream(for: job)
    }

    // MARK: - The transfer

    private func transfer(_ id: DownloadJobID, restrictedTo paths: Set<String>?) async {
        guard let job = jobs[id] else { return }
        let plan = job.state.plan
        let options = job.state.options
        let control = job.control

        do {
            _ = try Self.validate(
                plan, at: job.destination.directory,
                expectedPhysicalRoot: job.artifactRoot,
                expectedRootIdentity: job.artifactRootIdentity)
        } catch let error as DownloadError {
            finishFailed(id, error)
            return
        } catch {
            let refusal = DownloadError.destinationNotWritable(
                path: job.destination.directory.path, reason: String(describing: error))
            finishFailed(id, refusal)
            return
        }

        var pending = plan.files.filter { file in
            if let paths, !paths.contains(file.path) { return false }
            switch job.state.fileStates[file.path] {
            // Already on disk at the right size. Not re-fetched — the closing
            // verification is what decides whether it is also the right bytes.
            case .verified, .fetched: return false
            default: return true
            }
        }
        // Smallest first. A metadata file that fails fast is worth more than a
        // five-gigabyte tile that fails slowly, and the manifests are what tell
        // an operator whether the destination is even the right shape.
        pending.sort { ($0.sizeBytes, $0.path) < ($1.sizeBytes, $1.path) }

        var failure: DownloadError?
        var iterator = pending.makeIterator()
        let width = max(1, min(options.maximumConcurrentFiles, pending.count))

        await withTaskGroup(of: DownloadError?.self) { group in
            for _ in 0..<width {
                guard let file = iterator.next() else { break }
                group.addTask { [weak self] in await self?.fetch(file, in: id) ?? nil }
            }
            while let result = await group.next() {
                if let result, result != .cancelled {
                    // A durable-state failure is different from a bad remote
                    // file. Once we can no longer record what landed, every
                    // sibling is asked to stop and none may digest, promote or
                    // persist after it observes the abort latch. Content
                    // failures retain the established download contract:
                    // finish the remaining independent files so a later repair
                    // only has to fetch the paths that were actually bad.
                    if result.isStatePersistenceFailure {
                        // Persistence failure takes precedence even if a
                        // content failure happened first: the durable record
                        // is now the condition that makes continuation unsafe.
                        failure = result
                        group.cancelAll()
                    } else if failure == nil {
                        failure = result
                    }
                }
                if failure?.isStatePersistenceFailure == true
                    || control.mustAbortForPersistenceFailure
                    || control.isCancelled || control.isPaused
                {
                    continue
                }
                guard let file = iterator.next() else { continue }
                group.addTask { [weak self] in await self?.fetch(file, in: id) ?? nil }
            }
        }

        if control.isCancelled {
            _ = finishCancelled(id, keepPartialFiles: control.keepsPartialFiles)
            return
        }
        if let failure {
            finishFailed(id, failure)
            return
        }
        if control.isPaused {
            jobs[id]?.runState = .paused
            do {
                try persist(id)
            } catch {
                finishFailed(id, persistenceFailure(error, job: id))
                return
            }
            hub.emit(.paused(progress(of: id)), for: id)
            releaseScope(id)
            return
        }

        jobs[id]?.runState = .verifying
        do {
            try persist(id)
        } catch {
            finishFailed(id, persistenceFailure(error, job: id))
            return
        }
        // Files digested as they landed do not need digesting again; their
        // sizes are still re-read, because a file can be truncated after it was
        // checked. A file that was merely *adopted* off disk — present at the
        // right size when the job started — has never been digested by anyone,
        // so its presence pulls the whole pass up to a digest.
        let everyFileWasDigested =
            jobs[id]?.state.fileStates.values.allSatisfy {
                if case .verified = $0 { return true } else { return false }
            } ?? false
        let passDepth: VerificationDepth = everyFileWasDigested ? .sizeOnly : .digestPayload
        let passReport: VerificationReport
        do {
            passReport = try await verify(id, depth: passDepth)
        } catch let failure as DownloadError {
            if failure == .cancelled, control.isCancelled {
                _ = finishCancelled(id, keepPartialFiles: control.keepsPartialFiles)
                return
            }
            finishFailed(id, failure)
            return
        } catch is CancellationError {
            if control.isCancelled {
                _ = finishCancelled(id, keepPartialFiles: control.keepsPartialFiles)
                return
            }
            finishFailed(
                id,
                .interrupted(
                    path: job.destination.directory.path, atByte: 0,
                    underlying: "verification task was cancelled"))
            return
        } catch {
            finishFailed(
                id,
                .interrupted(
                    path: job.destination.directory.path, atByte: 0,
                    underlying: "verification failed: \(String(describing: error))"))
            return
        }
        // When every file was digested immediately after it landed, this final
        // size pass completes (rather than repeats) payload-digest coverage for
        // the immutable job. Advertise the cumulative achieved depth so a
        // product consumer can distinguish it from a standalone size-only
        // report without reading 1.5 TB twice.
        let report: VerificationReport
        if everyFileWasDigested, passReport.depth == .sizeOnly {
            report = VerificationReport(
                job: passReport.job, model: passReport.model, depth: .digestPayload,
                checkedAt: passReport.checkedAt, ok: passReport.ok,
                missing: passReport.missing, wrongSize: passReport.wrongSize,
                wrongDigest: passReport.wrongDigest, unreadable: passReport.unreadable,
                extraneous: passReport.extraneous,
                bytesToRefetch: passReport.bytesToRefetch)
        } else {
            report = passReport
        }
        hub.emit(.verificationFinished(report), for: id)
        if control.isCancelled {
            _ = finishCancelled(id, keepPartialFiles: control.keepsPartialFiles)
            return
        }

        let elapsed = Date().timeIntervalSince(jobs[id]?.startedAt ?? Date())
        let counters = jobs[id]?.counters
        let written = progress(of: id).verifiedBytes + progress(of: id).fetchedUnverifiedBytes
        jobs[id]?.runState = report.isComplete ? .finished : .failed
        do {
            try persist(id)
        } catch {
            finishFailed(id, persistenceFailure(error, job: id))
            return
        }
        let summary = DownloadSummary(
            job: id, model: plan.model, repo: plan.repo,
            destinationPath: job.destination.directory.path,
            bytesWritten: written,
            networkBytes: counters?.networkBytes ?? 0,
            wastedBytes: counters?.wastedBytes ?? 0,
            wallSeconds: elapsed,
            meanBytesPerSecond: elapsed > 0 ? Double(written) / elapsed : 0,
            verification: report)
        if report.isComplete {
            hub.emit(.finished(summary), for: id)
        } else {
            hub.emit(
                .failed(
                    report.wrongDigest.first.map(DownloadError.digestMismatch)
                        ?? report.wrongSize.first.map(DownloadError.sizeMismatch)
                        ?? .interrupted(
                            path: report.pathsToRefetch.first ?? "",
                            atByte: 0, underlying: "verification did not pass"),
                    progress(of: id)), for: id)
        }
        releaseScope(id)
    }

    /// Fetch one file to completion, or say why not.
    private func fetch(_ file: RepoFile, in id: DownloadJobID) async -> DownloadError? {
        guard let job = jobs[id] else { return .unknownJob(id) }
        let control = job.control
        let options = job.state.options
        let counters = job.counters
        let directory = job.destination.directory
        let repo = job.state.plan.repo

        if control.isCancelled || control.mustAbortForPersistenceFailure { return .cancelled }
        if control.isPaused { return nil }

        let initialPaths: ArtifactPaths
        do {
            initialPaths = try Self.validateArtifactPaths(
                file.path, at: directory, expectedPhysicalRoot: job.artifactRoot,
                expectedRootIdentity: job.artifactRootIdentity)
        } catch let error as DownloadError {
            setState(id, file.path, .failed(reason: error.description, attempts: 0))
            hub.emit(.fileFailed(path: file.path, error), for: id)
            return error
        } catch {
            let failure = DownloadError.destinationNotWritable(
                path: directory.path, reason: String(describing: error))
            setState(id, file.path, .failed(reason: failure.description, attempts: 0))
            hub.emit(.fileFailed(path: file.path, failure), for: id)
            return failure
        }
        var finalURL = initialPaths.final
        var partURL = initialPaths.part

        // Create/open the private part through an anchored parent before any
        // transport receives its borrowed sink. This is also the zero-byte
        // file path.
        var onDisk: UInt64
        do {
            let prepared = try Self.preparePartLeaf(
                for: file.path, root: job.artifactRoot,
                rootIdentity: job.artifactRootIdentity)
            onDisk = prepared.sizeBytes
            _ = Darwin.close(prepared.descriptor)
        } catch {
            let failure = DownloadError.destinationNotWritable(
                path: partURL.path, reason: String(describing: error))
            setState(id, file.path, .failed(reason: failure.description, attempts: 0))
            hub.emit(.fileFailed(path: file.path, failure), for: id)
            return failure
        }
        if onDisk > file.sizeBytes {
            hub.emit(
                .fileFailed(
                    path: file.path,
                    .sizeMismatch(
                        SizeMismatch(path: file.path, expected: file.sizeBytes, found: onDisk))),
                for: id)
            do {
                let current = try Self.validateArtifactPaths(
                    file.path, at: directory, expectedPhysicalRoot: job.artifactRoot,
                    expectedRootIdentity: job.artifactRootIdentity)
                partURL = current.part
                try Self.unlinkArtifactLeafIfPresent(
                    file.path + Self.partSuffix, root: job.artifactRoot,
                    rootIdentity: job.artifactRootIdentity, requireSingleLink: true)
            } catch let failure as DownloadError {
                setState(id, file.path, .failed(reason: failure.description, attempts: 0))
                hub.emit(.fileFailed(path: file.path, failure), for: id)
                return failure
            } catch {
                let failure = DownloadError.destinationNotWritable(
                    path: partURL.path, reason: String(describing: error))
                setState(id, file.path, .failed(reason: failure.description, attempts: 0))
                hub.emit(.fileFailed(path: file.path, failure), for: id)
                return failure
            }
            counters.addWasted(onDisk)
            onDisk = 0
        }
        hub.emit(.fileStarted(path: file.path, bytes: file.sizeBytes, resumedAtByte: onDisk), for: id)

        var attempt = 0
        var lastError: DownloadError?
        while true {
            // Re-read at the top of *every* attempt, not only after a
            // successful one. A request that wrote some bytes and then failed
            // leaves them on disk; asking again from a remembered offset would
            // append the same bytes a second time, and the file would end up
            // longer than the plan with a perfectly healthy-looking transfer
            // behind it. The file wins, on every iteration.
            do {
                let current = try Self.validateArtifactPaths(
                    file.path, at: directory, expectedPhysicalRoot: job.artifactRoot,
                    expectedRootIdentity: job.artifactRootIdentity)
                finalURL = current.final
                partURL = current.part
            } catch let error as DownloadError {
                setState(id, file.path, .failed(reason: error.description, attempts: attempt))
                hub.emit(.fileFailed(path: file.path, error), for: id)
                return error
            } catch {
                let failure = DownloadError.destinationNotWritable(
                    path: directory.path, reason: String(describing: error))
                setState(id, file.path, .failed(reason: failure.description, attempts: attempt))
                hub.emit(.fileFailed(path: file.path, failure), for: id)
                return failure
            }
            let prepared: OpenArtifactLeaf
            let onDisk: UInt64
            do {
                prepared = try Self.preparePartLeaf(
                    for: file.path, root: job.artifactRoot,
                    rootIdentity: job.artifactRootIdentity)
                onDisk = prepared.sizeBytes
            } catch {
                let failure = DownloadError.destinationNotWritable(
                    path: partURL.path, reason: String(describing: error))
                setState(id, file.path, .failed(reason: failure.description, attempts: attempt))
                hub.emit(.fileFailed(path: file.path, failure), for: id)
                return failure
            }
            // Keep the descriptor alive across the borrowed-sink transport
            // call. Besides binding the resume size to this leaf, this
            // prevents an unlinked inode from being recycled into a false
            // identity match.
            defer { _ = Darwin.close(prepared.descriptor) }
            if onDisk >= file.sizeBytes { break }
            if control.isCancelled { return .cancelled }
            if control.mustAbortForPersistenceFailure { return .cancelled }
            if control.isPaused { setState(id, file.path, .partial(bytesOnDisk: onDisk)); return nil }

            attempt += 1
            if attempt > options.retry.maximumAttempts {
                let error =
                    lastError
                    ?? .interrupted(
                        path: file.path, atByte: onDisk,
                        underlying: "gave up after \(options.retry.maximumAttempts) attempts")
                setState(id, file.path, .failed(reason: error.description, attempts: attempt - 1))
                hub.emit(.fileFailed(path: file.path, error), for: id)
                return error
            }
            let backoff = options.retry.backoff(beforeAttempt: attempt)
            if backoff > 0 {
                // `Double(UInt64.max)` rounds to 2^64, so using it as the
                // conversion clamp can itself leave UInt64(...) out of range.
                // Int64.max nanoseconds is already effectively unbounded for
                // retries and is exactly inside UInt64's representable domain.
                let maximumSleep = Double(Int64.max) / 1_000_000_000
                let boundedBackoff = min(backoff, maximumSleep)
                try? await Task.sleep(
                    nanoseconds: UInt64(boundedBackoff * 1_000_000_000))
            }
            if control.isCancelled || control.mustAbortForPersistenceFailure {
                return .cancelled
            }

            let requestBytes = min(options.chunkBytes, file.sizeBytes - onDisk)
            // `requestBytes <= file.sizeBytes - onDisk`, so this addition is
            // representable even for a public one-file UInt64.max plan.
            let upper = onDisk + requestBytes - 1
            do {
                let current = try Self.validateArtifactPaths(
                    file.path, at: directory, expectedPhysicalRoot: job.artifactRoot,
                    expectedRootIdentity: job.artifactRootIdentity)
                finalURL = current.final
                partURL = current.part
                var fetchFailure: Error?
                do {
                    try await fetchRange(
                        file, from: onDisk, to: upper, repo: repo, into: prepared,
                        counters: counters, control: control)
                } catch {
                    fetchFailure = error
                }
                let afterFetch = try Self.validateArtifactPaths(
                    file.path, at: directory, expectedPhysicalRoot: job.artifactRoot,
                    expectedRootIdentity: job.artifactRootIdentity)
                finalURL = afterFetch.final
                partURL = afterFetch.part
                guard let landed = try Self.openArtifactLeaf(
                    file.path + Self.partSuffix, root: job.artifactRoot,
                    rootIdentity: job.artifactRootIdentity, requireSingleLink: true)
                else {
                    throw DownloadError.destinationNotWritable(
                        path: partURL.path, reason: "the transport removed its part file")
                }
                let landedIdentity = landed.identity
                _ = Darwin.close(landed.descriptor)
                guard landedIdentity == prepared.identity else {
                    throw DownloadError.destinationNotWritable(
                        path: partURL.path,
                        reason: "the transport part file was replaced while the request ran")
                }
                if let sinkFailure = fetchFailure as? HTTPTransportError,
                    case .discardFailed = sinkFailure
                {
                    // Never let a failed rollback turn the rejected range into
                    // the next attempt's resume offset. Remove only the exact
                    // inode prepared for this request, through its anchored
                    // parent; a raced replacement is not ours to delete.
                    try Self.unlinkArtifactLeafIfPresent(
                        file.path + Self.partSuffix, root: job.artifactRoot,
                        rootIdentity: job.artifactRootIdentity,
                        requireSingleLink: true,
                        expectedIdentity: prepared.identity)
                    throw DownloadError.destinationNotWritable(
                        path: partURL.path,
                        reason: sinkFailure.description)
                }
                // Cancellation does not outrank integrity cleanup. In the
                // narrow race where cancellation arrives after rollback has
                // failed, first ensure the rejected bytes cannot become a
                // future resume offset; only then report cancellation.
                if control.isCancelled || control.mustAbortForPersistenceFailure {
                    return .cancelled
                }
                // A retryable HTTP response is still a completed borrowed-sink
                // operation. Prove the sink kept naming the prepared inode
                // before deciding whether the response itself merits a retry.
                if let fetchFailure { throw fetchFailure }
            } catch let error as DownloadError {
                lastError = error
                guard Self.isRetryable(error) else {
                    setState(id, file.path, .failed(reason: error.description, attempts: attempt))
                    hub.emit(.fileFailed(path: file.path, error), for: id)
                    return error
                }
                continue
            } catch {
                lastError = .interrupted(
                    path: file.path, atByte: onDisk, underlying: String(describing: error))
                continue
            }

            let now: UInt64
            do {
                guard let current = try Self.openArtifactLeaf(
                    file.path + Self.partSuffix, root: job.artifactRoot,
                    rootIdentity: job.artifactRootIdentity, requireSingleLink: true)
                else {
                    throw DownloadError.destinationNotWritable(
                        path: partURL.path, reason: "the transport removed its part file")
                }
                now = current.sizeBytes
                _ = Darwin.close(current.descriptor)
            } catch {
                let failure = DownloadError.destinationNotWritable(
                    path: partURL.path, reason: String(describing: error))
                setState(id, file.path, .failed(reason: failure.description, attempts: attempt))
                hub.emit(.fileFailed(path: file.path, failure), for: id)
                return failure
            }
            if now > onDisk { attempt = 0 }  // progress renews the retry budget
            setState(id, file.path, .partial(bytesOnDisk: now))
            sampleRate(id)
            hub.emit(.progress(progress(of: id)), for: id)
        }

        do {
            let current = try Self.validateArtifactPaths(
                file.path, at: directory, expectedPhysicalRoot: job.artifactRoot,
                expectedRootIdentity: job.artifactRootIdentity)
            finalURL = current.final
            partURL = current.part
        } catch let error as DownloadError {
            setState(id, file.path, .failed(reason: error.description, attempts: attempt))
            hub.emit(.fileFailed(path: file.path, error), for: id)
            return error
        } catch {
            let failure = DownloadError.destinationNotWritable(
                path: directory.path, reason: String(describing: error))
            setState(id, file.path, .failed(reason: failure.description, attempts: attempt))
            hub.emit(.fileFailed(path: file.path, failure), for: id)
            return failure
        }
        let landedLeaf: OpenArtifactLeaf
        let landed: UInt64
        do {
            guard let current = try Self.openArtifactLeaf(
                file.path + Self.partSuffix, root: job.artifactRoot,
                rootIdentity: job.artifactRootIdentity, requireSingleLink: true)
            else {
                throw DownloadError.destinationNotWritable(
                    path: partURL.path, reason: "the completed part file disappeared")
            }
            landedLeaf = current
            landed = current.sizeBytes
        } catch {
            let failure = DownloadError.destinationNotWritable(
                path: partURL.path, reason: String(describing: error))
            setState(id, file.path, .failed(reason: failure.description, attempts: attempt))
            hub.emit(.fileFailed(path: file.path, failure), for: id)
            return failure
        }
        defer { _ = Darwin.close(landedLeaf.descriptor) }
        guard landed == file.sizeBytes else {
            let mismatch = SizeMismatch(path: file.path, expected: file.sizeBytes, found: landed)
            setState(id, file.path, .failed(reason: "wrong size", attempts: attempt))
            hub.emit(.fileFailed(path: file.path, .sizeMismatch(mismatch)), for: id)
            return .sizeMismatch(mismatch)
        }

        if options.verifyAfterEachFile {
            if control.isCancelled || control.mustAbortForPersistenceFailure {
                return .cancelled
            }
            let found: String
            do {
                found = try await digest(
                    landedLeaf, algorithm: file.digest.algorithm, control: control
                ).hex
            } catch is CancellationError {
                return .cancelled
            } catch {
                let error = DownloadError.interrupted(
                    path: file.path, atByte: landed, underlying: String(describing: error))
                setState(id, file.path, .failed(reason: error.description, attempts: attempt))
                hub.emit(.fileFailed(path: file.path, error), for: id)
                return error
            }
            guard file.digest.matches(found) else {
                // A byte that failed a digest is not a byte to resume from.
                do {
                    partURL = try Self.validateArtifactPaths(
                        file.path, at: directory,
                        expectedPhysicalRoot: job.artifactRoot,
                        expectedRootIdentity: job.artifactRootIdentity
                    ).part
                    try Self.unlinkArtifactLeafIfPresent(
                        file.path + Self.partSuffix, root: job.artifactRoot,
                        rootIdentity: job.artifactRootIdentity, requireSingleLink: true)
                } catch let error as DownloadError {
                    setState(id, file.path, .failed(reason: error.description, attempts: attempt))
                    hub.emit(.fileFailed(path: file.path, error), for: id)
                    return error
                } catch {
                    let failure = DownloadError.destinationNotWritable(
                        path: directory.path, reason: String(describing: error))
                    setState(
                        id, file.path, .failed(reason: failure.description, attempts: attempt))
                    hub.emit(.fileFailed(path: file.path, failure), for: id)
                    return failure
                }
                counters.addWasted(file.sizeBytes)
                // Wasted bytes are terminal accounting for this file, not a
                // future transfer sample. Emit them immediately so a single
                // corrupt file cannot finish without ever surfacing the loss.
                hub.emit(.progress(progress(of: id)), for: id)
                let mismatch = DigestMismatch(
                    path: file.path, algorithm: file.digest.algorithm,
                    expected: file.digest.hex, found: found)
                setState(id, file.path, .failed(reason: "digest mismatch", attempts: attempt))
                hub.emit(.fileFailed(path: file.path, .digestMismatch(mismatch)), for: id)
                return .digestMismatch(mismatch)
            }
        }

        if control.isCancelled || control.mustAbortForPersistenceFailure {
            return .cancelled
        }
        do {
            let current = try Self.validateArtifactPaths(
                file.path, at: directory, expectedPhysicalRoot: job.artifactRoot,
                expectedRootIdentity: job.artifactRootIdentity)
            finalURL = current.final
            partURL = current.part
            try Self.promotePartLeaf(
                for: file.path, expectedPart: landedLeaf.identity,
                root: job.artifactRoot, rootIdentity: job.artifactRootIdentity)
        } catch {
            let failure = DownloadError.destinationNotWritable(
                path: finalURL.path, reason: String(describing: error))
            setState(id, file.path, .failed(reason: failure.description, attempts: attempt))
            hub.emit(.fileFailed(path: file.path, failure), for: id)
            return failure
        }

        if options.verifyAfterEachFile {
            setState(id, file.path, .verified(bytes: file.sizeBytes, at: Date()))
            hub.emit(
                .fileVerified(path: file.path, digest: file.digest, bytes: file.sizeBytes), for: id)
        } else {
            setState(id, file.path, .fetched(bytesOnDisk: file.sizeBytes))
        }
        sampleRate(id)
        hub.emit(.progress(progress(of: id)), for: id)
        if control.mustAbortForPersistenceFailure { return .cancelled }
        do {
            try persist(id)
            return nil
        } catch {
            let failure = persistenceFailure(error, job: id)
            control.abortForPersistenceFailure()
            return failure
        }
    }

    private func fetchRange(
        _ file: RepoFile, from lower: UInt64, to upper: UInt64,
        repo: HuggingFaceRepoRef, into part: OpenArtifactLeaf, counters: JobCounters,
        control: JobControl
    ) async throws {
        var headers: [String: String] = [:]
        if let token = token?() { headers["authorization"] = "Bearer \(token)" }
        let request = HTTPRequest(
            url: repo.resolveURL(path: file.path, endpoint: endpoint),
            headers: headers, byteRange: lower...upper)
        let cancellation = HTTPDownloadCancellationGate()
        let sink: HTTPDownloadSink
        do {
            sink = try HTTPDownloadSink(
                borrowing: part.descriptor, request: request,
                expectedObjectLength: file.sizeBytes,
                onBytes: { counters.addNetwork($0) },
                cancellationRequested: {
                    control.isCancelled || control.mustAbortForPersistenceFailure
                        || cancellation.isCancelled
                })
        } catch let error as HTTPTransportError {
            throw DownloadError.interrupted(
                path: file.path, atByte: lower, underlying: error.description)
        }
        defer {
            cancellation.cancel()
            sink.abort()
            counters.addWasted(sink.discardedBodyBytes)
        }

        let response: HTTPResponse
        do {
            response = try await withTaskCancellationHandler {
                try await transport.download(request, into: sink)
            } onCancel: {
                // A custom transport may ignore cooperative task
                // cancellation and retain the sink. Revocation therefore
                // lives beside the capability, not only in URLSession.
                cancellation.cancel()
            }
        } catch let error as HTTPTransportError {
            if case .cancelled = error { throw DownloadError.cancelled }
            if case .discardFailed = error {
                // The caller owns the anchored name and will unlink/taint it
                // before deciding whether another attempt is possible.
                throw error
            }
            if case .responseRejected(let rejected, _) = error {
                // Production admission already rolled this response back.
                // Repeat the operation so a custom transport cannot write
                // under one response and then throw a different rejection.
                let cleanup = sink.discardCurrentResponse(because: error)
                if case .discardFailed = cleanup { throw cleanup }
                if let provenance = Self.provenanceFailure(
                    in: rejected, repo: repo, path: file.path, atByte: lower)
                {
                    throw provenance
                }
                if rejected.statusCode == 200 {
                    throw DownloadError.rangeNotSupported(path: file.path)
                }
                if rejected.statusCode != 206 {
                    throw DownloadError.transport(
                        status: rejected.statusCode, path: file.path, attempt: 0)
                }
            }
            throw DownloadError.interrupted(
                path: file.path, atByte: lower, underlying: error.description)
        }

        var admissionFailure: HTTPTransportError?
        do {
            // A conforming transport admitted this same response before its
            // first body byte. Repeating admission makes the manager fail
            // closed against a custom transport that admits one response,
            // writes bytes, and returns another. A mismatch rolls this range
            // back to `lower` inside the sink.
            try sink.admit(response)
        } catch let error as HTTPTransportError {
            if case .cancelled = error {
                admissionFailure = error
            } else {
                admissionFailure = sink.discardCurrentResponse(because: error)
            }
        }

        // Revision provenance retains its established priority over status and
        // protocol errors. Admission still ran first so an untrusted custom
        // transport cannot use that priority to keep bytes supplied under a
        // different returned response. All recorded Hugging Face resolve
        // responses carry this header on the origin redirect; URLSession
        // overlays it onto the final response.
        if let provenance = Self.provenanceFailure(
            in: response, repo: repo, path: file.path, atByte: lower)
        {
            let rejection = HTTPTransportError.responseRejected(
                response: response, reason: provenance.description)
            let result = sink.discardCurrentResponse(because: rejection)
            if case .discardFailed = result { throw result }
            throw provenance
        }
        if let error = admissionFailure {
            if case .cancelled = error { throw DownloadError.cancelled }
            if case .discardFailed = error { throw error }
            if response.statusCode == 200 {
                throw DownloadError.rangeNotSupported(path: file.path)
            }
            if response.statusCode != 206 {
                throw DownloadError.transport(
                    status: response.statusCode, path: file.path, attempt: 0)
            }
            throw DownloadError.interrupted(
                path: file.path, atByte: lower, underlying: error.description)
        }
        switch response.statusCode {
        case 206:
            break
        case 200:
            throw DownloadError.rangeNotSupported(path: file.path)
        case let status:
            throw DownloadError.transport(status: status, path: file.path, attempt: 0)
        }
        guard sink.isSealed else {
            let error = sink.discardCurrentResponse(
                because: .invalidSink(
                    reason: "the transport returned before sealing the response"))
            if case .discardFailed = error { throw error }
            throw DownloadError.interrupted(
                path: file.path, atByte: lower, underlying: error.description)
        }
        do {
            // Re-run the exact-size invariant after confirming that the
            // transport itself sealed before it returned.
            try sink.seal()
        } catch let error as HTTPTransportError {
            if case .cancelled = error { throw DownloadError.cancelled }
            throw DownloadError.interrupted(
                path: file.path, atByte: lower, underlying: error.description)
        }
        if let size = response.linkedSizeBytes, size != file.sizeBytes {
            let rejection = HTTPTransportError.responseRejected(
                response: response,
                reason: "x-linked-size \(size) does not match \(file.sizeBytes)")
            let result = sink.discardCurrentResponse(because: rejection)
            if case .discardFailed = result { throw result }
            throw DownloadError.sizeMismatch(
                SizeMismatch(path: file.path, expected: file.sizeBytes, found: size))
        }
    }

    /// Repository identity is more authoritative than transport status: a
    /// moved resolve response is a different artifact even when that response
    /// also ignored Range. Only a successful partial response is required to
    /// carry the header; retryable error pages need not do so.
    private static func provenanceFailure(
        in response: HTTPResponse, repo: HuggingFaceRepoRef,
        path: String, atByte: UInt64
    ) -> DownloadError? {
        if let served = response.repoCommit, served != repo.revision {
            return .revisionMoved(expected: repo.revision, served: served, path: path)
        }
        if response.statusCode == 206, response.repoCommit == nil {
            return .interrupted(
                path: path, atByte: atByte,
                underlying: "the response did not identify its x-repo-commit")
        }
        return nil
    }

    /// Hashing a multi-gigabyte weight is blocking POSIX work. Running it on
    /// the actor would queue Stop behind the very read Stop must interrupt.
    /// The descriptor is already bound to the anchored artifact tree; the
    /// caller keeps it open until this detached worker returns.
    private func digest(
        _ leaf: OpenArtifactLeaf, algorithm: DigestAlgorithm, control: JobControl
    ) async throws -> FileDigest {
        let operation = digestOperation
        let worker = Task.detached(priority: .utility) {
            try operation(leaf.descriptor, algorithm) {
                control.isCancelled || control.mustAbortForPersistenceFailure
                    || withUnsafeCurrentTask { $0?.isCancelled ?? false }
            }
        }
        return try await withTaskCancellationHandler {
            try await worker.value
        } onCancel: {
            worker.cancel()
        }
    }

    /// Which failures are worth another attempt. A digest mismatch, a moved
    /// revision and a server that will not serve ranges are not: retrying them
    /// spends bandwidth to reach the same conclusion.
    static func isRetryable(_ error: DownloadError) -> Bool {
        switch error {
        case .interrupted:
            return true
        case .transport(let status, _, _):
            return status == 408 || status == 429 || (500...599).contains(status)
        default:
            return false
        }
    }

    // MARK: - Verification and repair

    public func verify(_ id: DownloadJobID, depth: VerificationDepth) async throws
        -> VerificationReport
    {
        // Finished, paused and restored jobs have released their long-lived
        // grant. A manual verification is still an external-volume read, so
        // reacquire the job's own bookmark for exactly this operation.
        let acquiredScope = try activateSavedDestinationIfNeeded(id)
        defer { acquiredScope?.release() }

        guard let job = jobs[id] else { throw DownloadError.unknownJob(id) }
        let plan = job.state.plan
        let directory = job.destination.directory
        let validated = try Self.validate(
            plan, at: directory, expectedPhysicalRoot: job.artifactRoot,
            expectedRootIdentity: job.artifactRootIdentity)

        var ok: [String] = []
        var missing: [String] = []
        var wrongSize: [SizeMismatch] = []
        var wrongDigest: [DigestMismatch] = []
        var unreadable: [String: String] = [:]

        for file in plan.files {
            if job.control.isCancelled { throw DownloadError.cancelled }
            guard validated.pathsByFile[file.path] != nil else {
                throw Self.planRefusal(
                    "downloadPlan.path", file.path, "a validated artifact path")
            }
            let leaf: OpenArtifactLeaf?
            do {
                leaf = try Self.openArtifactLeaf(
                    file.path, root: job.artifactRoot,
                    rootIdentity: job.artifactRootIdentity, requireSingleLink: false)
            } catch {
                unreadable[file.path] = String(describing: error)
                continue
            }
            guard let leaf else {
                missing.append(file.path)
                continue
            }
            guard leaf.sizeBytes == file.sizeBytes else {
                wrongSize.append(
                    SizeMismatch(
                        path: file.path, expected: file.sizeBytes, found: leaf.sizeBytes))
                _ = Darwin.close(leaf.descriptor)
                continue
            }
            guard depth.digests(file) else {
                ok.append(file.path)
                _ = Darwin.close(leaf.descriptor)
                continue
            }
            do {
                defer { _ = Darwin.close(leaf.descriptor) }
                let found = try await digest(
                    leaf, algorithm: file.digest.algorithm, control: job.control)
                if file.digest == found {
                    ok.append(file.path)
                } else {
                    wrongDigest.append(
                        DigestMismatch(
                            path: file.path, algorithm: file.digest.algorithm,
                            expected: file.digest.hex, found: found.hex))
                }
            } catch is CancellationError {
                if job.control.isCancelled { throw DownloadError.cancelled }
                throw CancellationError()
            } catch let error as DownloadError where error == .cancelled {
                throw error
            } catch {
                unreadable[file.path] = String(describing: error)
            }
        }

        let planned = Set(plan.files.map(\.path))
        let extraneous = Self.filesOnDisk(under: directory)
            .filter { !planned.contains($0) && !$0.hasSuffix(Self.partSuffix) }
            .sorted()

        var report = VerificationReport(
            job: id, model: plan.model, depth: depth, checkedAt: Date(),
            ok: ok.sorted(), missing: missing.sorted(), wrongSize: wrongSize,
            wrongDigest: wrongDigest, unreadable: unreadable, extraneous: extraneous)
        let sizes = Dictionary(
            plan.files.map { ($0.path, $0.sizeBytes) }, uniquingKeysWith: { a, _ in a })
        let cost = try Self.repairByteCost(paths: report.pathsToRefetch, sizes: sizes)
        report = VerificationReport(
            job: id, model: plan.model, depth: depth, checkedAt: report.checkedAt,
            ok: report.ok, missing: report.missing, wrongSize: report.wrongSize,
            wrongDigest: report.wrongDigest, unreadable: report.unreadable,
            extraneous: report.extraneous, bytesToRefetch: cost)
        return report
    }

    @discardableResult
    public func repair(_ id: DownloadJobID, from report: VerificationReport) async throws
        -> DownloadJobID
    {
        guard let original = jobs[id] else { throw DownloadError.unknownJob(id) }
        guard original.runState != .running, original.runState != .verifying else {
            throw DownloadError.jobAlreadyRunning(id)
        }

        // A terminal or restored job no longer holds its original sandbox
        // grant. Re-open the per-job bookmark and bind it to the directory
        // identity recorded at registration before trusting a report or
        // touching a byte.
        let resolved = try resolveSavedDestination(for: original)
        var scoped = original
        scoped.destination = DownloadDestination(
            directory: resolved.scope.url, scope: resolved.scope)
        scoped.state = DownloadJobState(
            job: scoped.state.job, plan: scoped.state.plan,
            destinationPath: resolved.scope.url.standardizedFileURL.path,
            destinationBookmark: resolved.bookmarkData,
            artifactRootPath: scoped.state.artifactRootPath,
            artifactRootDevice: scoped.state.artifactRootDevice,
            artifactRootInode: scoped.state.artifactRootInode,
            options: scoped.state.options, fileStates: scoped.state.fileStates,
            runState: scoped.runState, createdAt: scoped.state.createdAt,
            updatedAt: Date())

        let targets: [RepairTarget]
        do {
            targets = try validatedRepairTargets(id: id, report: report, job: scoped)
        } catch {
            resolved.scope.release()
            throw error
        }
        let paths = Set(targets.map(\.path))
        guard !paths.isEmpty else {
            resolved.scope.release()
            return id
        }

        var states = scoped.state.fileStates
        for target in targets { states[target.path] = .pending }
        let now = Date()
        let state = DownloadJobState(
            job: scoped.state.job, plan: scoped.state.plan,
            destinationPath: scoped.state.destinationPath,
            destinationBookmark: scoped.state.destinationBookmark,
            artifactRootPath: scoped.state.artifactRootPath,
            artifactRootDevice: scoped.state.artifactRootDevice,
            artifactRootInode: scoped.state.artifactRootInode,
            options: scoped.state.options, fileStates: states,
            runState: .running, createdAt: scoped.state.createdAt, updatedAt: now)
        var repaired = Job(
            state: state, runState: .running, destination: scoped.destination,
            artifactRoot: scoped.artifactRoot,
            artifactRootIdentity: scoped.artifactRootIdentity,
            counters: scoped.counters, control: JobControl(), startedAt: now,
            lastSampleAt: now, lastSampleBytes: scoped.counters.networkBytes,
            smoothedBytesPerSecond: 0, task: nil)
        jobs[id] = repaired

        // Make the repair intent durable before the first deletion. If the
        // process dies after this save, restore re-derives the bytes from disk
        // and can only pause or fail verification; it cannot call a corrupt
        // final file verified.
        do {
            try persist(id)
        } catch {
            jobs[id] = original
            resolved.scope.release()
            throw persistenceFailure(error, job: id)
        }
        hub.restart(id)

        do {
            for target in targets {
                // Truncate to nothing, never resume. Bytes that failed a
                // digest are not a prefix worth keeping: corruption may start
                // at byte zero. Both names are unlinked relative to an opened,
                // identity-checked parent directory.
                try Self.unlinkArtifactLeafIfPresent(
                    target.path + Self.partSuffix, root: repaired.artifactRoot,
                    rootIdentity: repaired.artifactRootIdentity, requireSingleLink: true)
                try Self.unlinkArtifactLeafIfPresent(
                    target.path, root: repaired.artifactRoot,
                    rootIdentity: repaired.artifactRootIdentity, requireSingleLink: false)
            }
        } catch let failure as DownloadError {
            throw finishFailed(id, failure)
        } catch {
            let failure = DownloadError.destinationNotWritable(
                path: repaired.destination.directory.path,
                reason: String(describing: error))
            throw finishFailed(id, failure)
        }

        repaired = jobs[id] ?? repaired
        repaired.task = Task { [weak self] in
            await self?.transfer(id, restrictedTo: paths)
        }
        jobs[id] = repaired
        return id
    }

    private struct RepairTarget {
        let path: String
    }

    /// Validate the whole report before the first mutation. A report is useful
    /// repair authority only for the job, artifact root and immutable plan that
    /// produced it; its strings are never filesystem authority on their own.
    private func validatedRepairTargets(
        id: DownloadJobID, report: VerificationReport, job: Job
    ) throws -> [RepairTarget] {
        guard report.job == id else {
            throw Self.repairRefusal(
                "verificationReport.job", report.job.rawValue.uuidString,
                "the job being repaired (\(id.rawValue.uuidString))")
        }
        let plan = job.state.plan
        guard report.model == plan.model else {
            throw Self.repairRefusal(
                "verificationReport.model", report.model.rawValue,
                "the job plan's model (\(plan.model.rawValue))")
        }

        let validated = try Self.validate(
            plan, at: job.destination.directory,
            expectedPhysicalRoot: job.artifactRoot,
            expectedRootIdentity: job.artifactRootIdentity)
        let lexicalRoot = validated.lexicalRoot
        let recordedPath = URL(fileURLWithPath: job.state.destinationPath).standardizedFileURL
        guard lexicalRoot.path == recordedPath.path else {
            throw Self.repairRefusal(
                "verificationReport.artifactRoot", lexicalRoot.path,
                "the job's recorded artifact root (\(recordedPath.path))")
        }
        let planByPath = Dictionary(uniqueKeysWithValues: plan.files.map { ($0.path, $0) })

        let faultPaths = report.pathsToRefetch
        let allReported =
            report.ok + report.missing + report.wrongSize.map(\.path)
            + report.wrongDigest.map(\.path) + report.unreadable.keys.sorted()
        var reportedOnce = Set<String>()
        for path in allReported {
            guard HuggingFaceTreeClient.isSafeRepositoryPath(path) else {
                throw Self.repairRefusal(
                    "verificationReport.path", path, "a canonical repository-relative path")
            }
            guard planByPath[path] != nil else {
                throw Self.repairRefusal(
                    "verificationReport.path", path, "a file in this job's download plan")
            }
            guard reportedOnce.insert(path).inserted else {
                throw Self.repairRefusal(
                    "verificationReport.path", path, "one result category per planned file")
            }
        }
        guard reportedOnce == Set(planByPath.keys) else {
            throw Self.repairRefusal(
                "verificationReport.coverage", "\(reportedOnce.count) files",
                "all \(planByPath.count) files in this job's plan")
        }

        for mismatch in report.wrongSize {
            guard planByPath[mismatch.path]?.sizeBytes == mismatch.expected else {
                throw Self.repairRefusal(
                    "verificationReport.expectedSize", "\(mismatch.expected) for \(mismatch.path)",
                    "the size recorded in this job's plan")
            }
        }
        for mismatch in report.wrongDigest {
            guard let file = planByPath[mismatch.path],
                mismatch.algorithm == file.digest.algorithm,
                mismatch.expected.lowercased() == file.digest.hex
            else {
                throw Self.repairRefusal(
                    "verificationReport.expectedDigest", mismatch.path,
                    "the digest recorded in this job's plan")
            }
        }

        let expectedBytes = try Self.repairByteCost(
            paths: faultPaths, sizes: planByPath.mapValues(\.sizeBytes))
        guard report.bytesToRefetch == expectedBytes else {
            throw Self.repairRefusal(
                "verificationReport.bytesToRefetch", "\(report.bytesToRefetch)",
                "\(expectedBytes) bytes from this job's plan")
        }

        return try faultPaths.map { path in
            guard validated.pathsByFile[path] != nil else {
                throw Self.repairRefusal(
                    "verificationReport.path", path, "a validated file in this job's plan")
            }
            return RepairTarget(path: path)
        }
    }

    /// Validate a public plan as authority to touch one artifact tree. This is
    /// intentionally the single entry point used by preflight, registration,
    /// transfer, verification and repair: a plan produced anywhere other than
    /// the HF decoder must meet the same path and arithmetic contract.
    private static func validate(
        _ plan: DownloadPlan, at destination: URL, expectedPhysicalRoot: URL? = nil,
        expectedRootIdentity: ArtifactRootIdentity? = nil
    ) throws -> ValidatedPlan {
        guard HuggingFaceRepoRef.isCanonicalRepositoryID(plan.repo.repoID) else {
            throw planRefusal(
                "downloadPlan.repoID", plan.repo.repoID,
                "a canonical Hugging Face repository ID in the form 'owner/name'")
        }
        guard plan.repo.isPinnedCommit else {
            throw planRefusal(
                "downloadPlan.revision", plan.repo.revision,
                "a lowercase 40-hex commit")
        }
        guard !plan.files.isEmpty else {
            throw planRefusal(
                "downloadPlan.files", "0", "at least one file")
        }
        var filePaths = Set<String>()
        var storageEntries: [(path: String, role: String)] = []
        var totalBytes: UInt64 = 0

        for file in plan.files {
            guard HuggingFaceTreeClient.isSafeRepositoryPath(file.path) else {
                throw planRefusal(
                    "downloadPlan.path", file.path, "a canonical repository-relative path")
            }
            guard filePaths.insert(file.path).inserted else {
                throw planRefusal(
                    "downloadPlan.path", file.path, "a path occurring exactly once in the plan")
            }
            guard file.isPayload == RepoFileClassification.isPayload(path: file.path) else {
                throw planRefusal(
                    "downloadPlan.isPayload", "\(file.isPayload) for '\(file.path)'",
                    "the repository path classification")
            }
            let digestDigits = file.digest.algorithm == .sha256 ? 64 : 40
            guard isHexDigest(file.digest.hex, digits: digestDigits) else {
                throw planRefusal(
                    "downloadPlan.digest", file.path,
                    "a \(digestDigits)-digit hexadecimal \(file.digest.algorithm.rawValue) digest")
            }

            let (sum, overflow) = totalBytes.addingReportingOverflow(file.sizeBytes)
            guard !overflow else {
                throw planRefusal(
                    "downloadPlan.totalBytes", "overflow while adding '\(file.path)'",
                    "a byte total representable by UInt64")
            }
            totalBytes = sum

            for (storagePath, role) in [
                (file.path, "final path for '\(file.path)'"),
                (file.path + partSuffix, "part path for '\(file.path)'"),
            ] {
                storageEntries.append((storagePath, role))
            }
        }

        let expectedReconciliation = IndexReconciliation.between(
            files: plan.files, claim: plan.index)
        guard plan.reconciliation == expectedReconciliation else {
            throw planRefusal(
                "downloadPlan.reconciliation", "does not match the plan's files and index",
                "the reconciliation derived from downloadPlan.files and downloadPlan.index")
        }

        guard destination.isFileURL else {
            throw planRefusal(
                "downloadDestination", destination.absoluteString, "an absolute file URL")
        }
        let lexicalRoot = destination.standardizedFileURL
        let caseSensitive = volumeSupportsCaseSensitiveNames(at: lexicalRoot)
        var storageOwners: [String: (path: String, role: String)] = [:]
        for entry in storageEntries {
            let key = collisionKey(entry.path, caseSensitive: caseSensitive)
            if let previous = storageOwners.updateValue(entry, forKey: key) {
                throw planRefusal(
                    "downloadPlan.path", entry.path,
                    "a storage path distinct from \(previous.role) at '\(previous.path)'")
            }
        }

        // An exact-string set is not enough: `foo` and `foo/bar` require the
        // same object to be both a file and a directory. Include part paths in
        // the same namespace so a temporary file cannot become another file's
        // ancestor either.
        let storagePaths = Set(storageOwners.keys)
        for entry in storageEntries {
            let path = entry.path
            var components = path.split(separator: "/").map(String.init)
            while components.count > 1 {
                components.removeLast()
                let ancestor = components.joined(separator: "/")
                let ancestorKey = collisionKey(ancestor, caseSensitive: caseSensitive)
                if storagePaths.contains(ancestorKey) {
                    throw planRefusal(
                        "downloadPlan.path", path,
                        "a storage path with no planned file or part path at ancestor '\(ancestor)'")
                }
            }
        }

        let physicalRoot = lexicalRoot.resolvingSymlinksInPath().standardizedFileURL
        if let expectedPhysicalRoot,
            physicalRoot.path != expectedPhysicalRoot.standardizedFileURL.path
        {
            throw planRefusal(
                "downloadDestination.root", physicalRoot.path,
                "the registered artifact root (\(expectedPhysicalRoot.path))")
        }
        let rootIdentity = try inspectArtifactRoot(physicalRoot)
        if let expectedRootIdentity, rootIdentity != expectedRootIdentity {
            throw planRefusal(
                "downloadDestination.root", physicalRoot.path,
                "the directory object registered for this job")
        }

        var pathsByFile: [String: ArtifactPaths] = [:]
        for file in plan.files {
            pathsByFile[file.path] = ArtifactPaths(
                final: try validateArtifactURL(
                    file.path, lexicalRoot: lexicalRoot, physicalRoot: physicalRoot,
                    mutableLeaf: false),
                part: try validateArtifactURL(
                    file.path + partSuffix,
                    lexicalRoot: lexicalRoot, physicalRoot: physicalRoot,
                    mutableLeaf: true))
        }
        return ValidatedPlan(
            totalBytes: totalBytes, lexicalRoot: lexicalRoot,
            physicalRoot: physicalRoot, rootIdentity: rootIdentity,
            pathsByFile: pathsByFile)
    }

    /// Re-check one file immediately before a transfer or deletion. Structural
    /// plan validation has already happened; this catches a root replacement
    /// or a symlink inserted into the file's path after registration.
    private static func validateArtifactPaths(
        _ path: String, at destination: URL, expectedPhysicalRoot: URL,
        expectedRootIdentity: ArtifactRootIdentity
    ) throws -> ArtifactPaths {
        guard HuggingFaceTreeClient.isSafeRepositoryPath(path) else {
            throw planRefusal(
                "downloadPlan.path", path, "a canonical repository-relative path")
        }
        guard destination.isFileURL else {
            throw planRefusal(
                "downloadDestination", destination.absoluteString, "an absolute file URL")
        }
        let lexicalRoot = destination.standardizedFileURL
        let physicalRoot = lexicalRoot.resolvingSymlinksInPath().standardizedFileURL
        guard physicalRoot.path == expectedPhysicalRoot.standardizedFileURL.path else {
            throw planRefusal(
                "downloadDestination.root", physicalRoot.path,
                "the registered artifact root (\(expectedPhysicalRoot.path))")
        }
        guard try inspectArtifactRoot(physicalRoot) == expectedRootIdentity else {
            throw planRefusal(
                "downloadDestination.root", physicalRoot.path,
                "the directory object registered for this job")
        }
        return ArtifactPaths(
            final: try validateArtifactURL(
                path, lexicalRoot: lexicalRoot, physicalRoot: physicalRoot,
                mutableLeaf: false),
            part: try validateArtifactURL(
                path + partSuffix, lexicalRoot: lexicalRoot, physicalRoot: physicalRoot,
                mutableLeaf: true))
    }

    private static func validateArtifactURL(
        _ relativePath: String, lexicalRoot: URL, physicalRoot: URL,
        mutableLeaf: Bool
    ) throws -> URL {
        let candidate = lexicalRoot.appendingPathComponent(relativePath).standardizedFileURL
        guard isStrictDescendant(candidate, of: lexicalRoot) else {
            throw planRefusal(
                "downloadPlan.path", relativePath, "a path contained by the artifact root")
        }

        let expectedPhysical = physicalRoot.appendingPathComponent(relativePath)
            .standardizedFileURL
        let resolved = candidate.resolvingSymlinksInPath().standardizedFileURL
        guard isStrictDescendant(resolved, of: physicalRoot) else {
            throw planRefusal(
                "downloadPlan.path", relativePath,
                "a path whose resolved target stays inside the artifact root")
        }
        guard resolved.path == expectedPhysical.path else {
            throw planRefusal(
                "downloadPlan.path", relativePath,
                "a path with no symbolic-link redirection below the artifact root")
        }
        var metadata = stat()
        if lstat(candidate.path, &metadata) == 0 {
            guard metadata.st_mode & S_IFMT == S_IFREG else {
                throw planRefusal(
                    "downloadPlan.path", relativePath,
                    "an absent path or an existing regular file")
            }
            if mutableLeaf, metadata.st_nlink != 1 {
                throw planRefusal(
                    "downloadPlan.path", relativePath,
                    "an unlinked destination or a private single-link part file")
            }
        } else if errno != ENOENT {
            throw DownloadError.destinationNotWritable(
                path: candidate.path, reason: String(cString: strerror(errno)))
        }
        return candidate
    }

    /// Run one operation relative to an opened parent directory. Every path
    /// component is opened with `O_NOFOLLOW`, and the root descriptor is bound
    /// to the device/inode captured when the job registered. This prevents a
    /// symlink or replaced root from redirecting local mutations outside the
    /// registered artifact tree; leaf identity is checked separately wherever
    /// the platform API exposes enough information to do so.
    private static func withAnchoredParent<T>(
        of relativePath: String, root: URL, rootIdentity: ArtifactRootIdentity,
        createParents: Bool = false, _ operation: (Int32, String) throws -> T
    ) throws -> T {
        guard HuggingFaceTreeClient.isSafeRepositoryPath(relativePath) else {
            throw planRefusal(
                "downloadPlan.path", relativePath, "a canonical repository-relative path")
        }
        let components = relativePath.split(separator: "/").map(String.init)
        guard let leaf = components.last else {
            throw planRefusal("downloadPlan.path", relativePath, "a non-empty file path")
        }

        let rootDescriptor = root.path.withCString {
            Darwin.open($0, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
        }
        guard rootDescriptor >= 0 else {
            let code = errno
            throw DownloadError.destinationNotWritable(
                path: root.path, reason: String(cString: strerror(code)))
        }
        var current = rootDescriptor
        defer { _ = Darwin.close(current) }

        var rootMetadata = stat()
        guard Darwin.fstat(current, &rootMetadata) == 0 else {
            let code = errno
            throw DownloadError.destinationNotWritable(
                path: root.path, reason: String(cString: strerror(code)))
        }
        guard rootMetadata.st_mode & S_IFMT == S_IFDIR,
            ArtifactRootIdentity(
                device: UInt64(rootMetadata.st_dev), inode: UInt64(rootMetadata.st_ino))
                == rootIdentity
        else {
            throw planRefusal(
                "downloadDestination.root", root.path,
                "the directory object registered for this job")
        }

        for component in components.dropLast() {
            if createParents {
                let made = component.withCString { Darwin.mkdirat(current, $0, 0o755) }
                if made != 0 {
                    let code = errno
                    if code != EEXIST {
                        throw DownloadError.destinationNotWritable(
                            path: root.appendingPathComponent(relativePath).path,
                            reason: String(cString: strerror(code)))
                    }
                }
            }
            let next = component.withCString {
                Darwin.openat(current, $0, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
            }
            guard next >= 0 else {
                let code = errno
                if code == ENOENT, !createParents {
                    throw AnchoredPathError.missingParent
                }
                throw DownloadError.destinationNotWritable(
                    path: root.appendingPathComponent(relativePath).path,
                    reason: String(cString: strerror(code)))
            }
            var metadata = stat()
            guard Darwin.fstat(next, &metadata) == 0 else {
                let code = errno
                _ = Darwin.close(next)
                throw DownloadError.destinationNotWritable(
                    path: root.appendingPathComponent(relativePath).path,
                    reason: String(cString: strerror(code)))
            }
            guard metadata.st_mode & S_IFMT == S_IFDIR,
                UInt64(metadata.st_dev) == rootIdentity.device
            else {
                _ = Darwin.close(next)
                throw DownloadError.destinationNotWritable(
                    path: root.appendingPathComponent(relativePath).path,
                    reason: "the parent is not a directory on the registered volume")
            }
            _ = Darwin.close(current)
            current = next
        }
        return try operation(current, leaf)
    }

    private static func openArtifactLeaf(
        _ relativePath: String, root: URL, rootIdentity: ArtifactRootIdentity,
        requireSingleLink: Bool
    ) throws -> OpenArtifactLeaf? {
        do {
            return try withAnchoredParent(
                of: relativePath, root: root, rootIdentity: rootIdentity
            ) { parent, leaf in
                let descriptor = leaf.withCString {
                    Darwin.openat(parent, $0, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
                }
                guard descriptor >= 0 else {
                    let code = errno
                    if code == ENOENT { return nil }
                    throw DownloadError.destinationNotWritable(
                        path: root.appendingPathComponent(relativePath).path,
                        reason: String(cString: strerror(code)))
                }
                var metadata = stat()
                guard Darwin.fstat(descriptor, &metadata) == 0,
                    metadata.st_mode & S_IFMT == S_IFREG, metadata.st_size >= 0,
                    !requireSingleLink || metadata.st_nlink == 1
                else {
                    _ = Darwin.close(descriptor)
                    throw DownloadError.destinationNotWritable(
                        path: root.appendingPathComponent(relativePath).path,
                        reason: "the artifact leaf is not a permitted regular file")
                }
                return OpenArtifactLeaf(
                    descriptor: descriptor,
                    identity: ArtifactLeafIdentity(
                        device: UInt64(metadata.st_dev), inode: UInt64(metadata.st_ino)),
                    sizeBytes: UInt64(metadata.st_size))
            }
        } catch AnchoredPathError.missingParent {
            return nil
        }
    }

    /// Open a regular, private part leaf through its anchored parent and return
    /// the descriptor that owns all writes for the next request. The transport
    /// receives only a borrowed sink over this descriptor; reopening the leaf
    /// by name is never part of the body path. The identity is checked again
    /// before any response or error is trusted.
    private static func preparePartLeaf(
        for path: String, root: URL, rootIdentity: ArtifactRootIdentity
    ) throws -> OpenArtifactLeaf {
        let relativePath = path + partSuffix
        return try withAnchoredParent(
            of: relativePath, root: root, rootIdentity: rootIdentity,
            createParents: true
        ) { parent, leaf in
            let descriptor = leaf.withCString {
                Darwin.openat(
                    parent, $0, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0o600)
            }
            guard descriptor >= 0 else {
                let code = errno
                throw DownloadError.destinationNotWritable(
                    path: root.appendingPathComponent(relativePath).path,
                    reason: String(cString: strerror(code)))
            }
            var metadata = stat()
            guard Darwin.fstat(descriptor, &metadata) == 0 else {
                let code = errno
                _ = Darwin.close(descriptor)
                throw DownloadError.destinationNotWritable(
                    path: root.appendingPathComponent(relativePath).path,
                    reason: String(cString: strerror(code)))
            }
            guard metadata.st_mode & S_IFMT == S_IFREG,
                metadata.st_size >= 0, metadata.st_nlink == 1
            else {
                _ = Darwin.close(descriptor)
                throw DownloadError.destinationNotWritable(
                    path: root.appendingPathComponent(relativePath).path,
                    reason: "the part path is not a private regular file")
            }
            return OpenArtifactLeaf(
                descriptor: descriptor,
                identity: ArtifactLeafIdentity(
                    device: UInt64(metadata.st_dev), inode: UInt64(metadata.st_ino)),
                sizeBytes: UInt64(metadata.st_size))
        }
    }

    private static func unlinkArtifactLeafIfPresent(
        _ relativePath: String, root: URL, rootIdentity: ArtifactRootIdentity,
        requireSingleLink: Bool,
        expectedIdentity: ArtifactLeafIdentity? = nil
    ) throws {
        do {
            try withAnchoredParent(of: relativePath, root: root, rootIdentity: rootIdentity) {
                parent, leaf in
                var metadata = stat()
                let inspected = leaf.withCString {
                    Darwin.fstatat(parent, $0, &metadata, AT_SYMLINK_NOFOLLOW)
                }
                if inspected != 0 {
                    let code = errno
                    if code == ENOENT { return }
                    throw DownloadError.destinationNotWritable(
                        path: root.appendingPathComponent(relativePath).path,
                        reason: String(cString: strerror(code)))
                }
                guard metadata.st_mode & S_IFMT == S_IFREG,
                    !requireSingleLink || metadata.st_nlink == 1
                else {
                    throw DownloadError.destinationNotWritable(
                        path: root.appendingPathComponent(relativePath).path,
                        reason: "the mutation target is not a private regular file")
                }
                if let expectedIdentity {
                    let actual = ArtifactLeafIdentity(
                        device: UInt64(metadata.st_dev), inode: UInt64(metadata.st_ino))
                    // The prepared inode is already unlinked when another
                    // object now owns the name. Closing its manager-owned fd
                    // is sufficient cleanup; never unlink the replacement.
                    guard actual == expectedIdentity else { return }
                }
                let removed = leaf.withCString { Darwin.unlinkat(parent, $0, 0) }
                guard removed == 0 || errno == ENOENT else {
                    throw DownloadError.destinationNotWritable(
                        path: root.appendingPathComponent(relativePath).path,
                        reason: String(cString: strerror(errno)))
                }
            }
        } catch AnchoredPathError.missingParent {
            return
        }
    }

    private static func promotePartLeaf(
        for path: String, expectedPart: ArtifactLeafIdentity,
        root: URL, rootIdentity: ArtifactRootIdentity
    ) throws {
        try withAnchoredParent(
            of: path, root: root, rootIdentity: rootIdentity, createParents: true
        ) { parent, finalLeaf in
            let partLeaf = finalLeaf + partSuffix
            var partMetadata = stat()
            let inspectedPart = partLeaf.withCString {
                Darwin.fstatat(parent, $0, &partMetadata, AT_SYMLINK_NOFOLLOW)
            }
            guard inspectedPart == 0,
                partMetadata.st_mode & S_IFMT == S_IFREG,
                partMetadata.st_nlink == 1,
                ArtifactLeafIdentity(
                    device: UInt64(partMetadata.st_dev), inode: UInt64(partMetadata.st_ino))
                    == expectedPart
            else {
                throw DownloadError.destinationNotWritable(
                    path: root.appendingPathComponent(path + partSuffix).path,
                    reason: "the verified part file was replaced before promotion")
            }

            var finalMetadata = stat()
            let inspectedFinal = finalLeaf.withCString {
                Darwin.fstatat(parent, $0, &finalMetadata, AT_SYMLINK_NOFOLLOW)
            }
            if inspectedFinal == 0,
                finalMetadata.st_mode & S_IFMT != S_IFREG
            {
                throw DownloadError.destinationNotWritable(
                    path: root.appendingPathComponent(path).path,
                    reason: "the final path is not a regular file")
            } else if inspectedFinal != 0, errno != ENOENT {
                throw DownloadError.destinationNotWritable(
                    path: root.appendingPathComponent(path).path,
                    reason: String(cString: strerror(errno)))
            }

            let promoted = partLeaf.withCString { partName in
                finalLeaf.withCString { finalName in
                    Darwin.renameat(parent, partName, parent, finalName)
                }
            }
            guard promoted == 0 else {
                let code = errno
                throw DownloadError.destinationNotWritable(
                    path: root.appendingPathComponent(path).path,
                    reason: String(cString: strerror(code)))
            }

            // `renameat` is pathname-based: bracket it with identity checks so
            // a source substitution in the inspection-to-rename interval is
            // rejected before the file is recorded as verified. An arbitrary
            // writer can still mutate the same inode later; no pathname API can
            // make a user-writable artifact tree immutable.
            let installed = finalLeaf.withCString {
                Darwin.openat(parent, $0, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
            }
            guard installed >= 0 else {
                let code = errno
                throw DownloadError.destinationNotWritable(
                    path: root.appendingPathComponent(path).path,
                    reason: String(cString: strerror(code)))
            }
            defer { _ = Darwin.close(installed) }
            var installedMetadata = stat()
            guard Darwin.fstat(installed, &installedMetadata) == 0 else {
                let code = errno
                throw DownloadError.destinationNotWritable(
                    path: root.appendingPathComponent(path).path,
                    reason: String(cString: strerror(code)))
            }
            guard installedMetadata.st_mode & S_IFMT == S_IFREG,
                ArtifactLeafIdentity(
                    device: UInt64(installedMetadata.st_dev),
                    inode: UInt64(installedMetadata.st_ino)) == expectedPart
            else {
                throw DownloadError.destinationNotWritable(
                    path: root.appendingPathComponent(path).path,
                    reason: "the verified part file was replaced during promotion")
            }
        }
    }

    private static func isHexDigest(_ value: String, digits: Int) -> Bool {
        let bytes = value.utf8
        guard bytes.count == digits else { return false }
        return bytes.allSatisfy {
            (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0)
        }
    }

    private static func collisionKey(_ path: String, caseSensitive: Bool) -> String {
        guard !caseSensitive else { return path }
        return path.folding(options: [.caseInsensitive], locale: Locale(identifier: "en_US_POSIX"))
    }

    private static func volumeSupportsCaseSensitiveNames(at url: URL) -> Bool {
        var candidate = url
        while true {
            if let value = try? candidate.resourceValues(
                forKeys: [.volumeSupportsCaseSensitiveNamesKey]
            ).volumeSupportsCaseSensitiveNames {
                return value
            }
            let parent = candidate.deletingLastPathComponent()
            if parent.path == candidate.path { return false }
            candidate = parent
        }
    }

    private static func inspectArtifactRoot(_ root: URL) throws -> ArtifactRootIdentity? {
        var metadata = stat()
        guard lstat(root.path, &metadata) == 0 else {
            if errno == ENOENT { return nil }
            throw DownloadError.destinationNotWritable(
                path: root.path, reason: String(cString: strerror(errno)))
        }
        guard metadata.st_mode & S_IFMT == S_IFDIR else {
            throw planRefusal(
                "downloadDestination.root", root.path, "a directory")
        }
        return ArtifactRootIdentity(
            device: UInt64(metadata.st_dev), inode: UInt64(metadata.st_ino))
    }

    /// `StorageManaging.canHold` represents capacity as signed bytes. Refuse a
    /// value it cannot represent rather than allowing its internal conversion
    /// to wrap or trap after plan validation succeeded.
    private static func validateCapacityArithmetic(
        requiredBytes: UInt64, headroomBytes: UInt64
    ) throws {
        let (need, overflow) = requiredBytes.addingReportingOverflow(headroomBytes)
        guard !overflow, need <= UInt64(Int64.max) else {
            throw planRefusal(
                "downloadPlan.requiredBytesPlusHeadroom",
                "\(requiredBytes) + \(headroomBytes)",
                "a byte total representable by the storage capacity API")
        }
    }

    private static func isStrictDescendant(_ candidate: URL, of root: URL) -> Bool {
        let rootComponents = root.standardizedFileURL.pathComponents
        let candidateComponents = candidate.standardizedFileURL.pathComponents
        return candidateComponents.count > rootComponents.count
            && candidateComponents.prefix(rootComponents.count).elementsEqual(rootComponents)
    }

    /// UInt64 is the persisted report format, so an unrepresentable repair
    /// cost is an invalid plan/report pair, not a value to wrap or a process to
    /// trap. Internal visibility keeps the arithmetic independently testable.
    static func repairByteCost(paths: [String], sizes: [String: UInt64]) throws -> UInt64 {
        var total: UInt64 = 0
        for path in paths {
            let (sum, overflow) = total.addingReportingOverflow(sizes[path] ?? 0)
            guard !overflow else {
                throw repairRefusal(
                    "repair.bytesToRefetch", "overflow while adding '\(path)'",
                    "a byte total representable by UInt64")
            }
            total = sum
        }
        return total
    }

    private static func repairRefusal(
        _ name: String, _ value: String, _ allowed: String
    ) -> DownloadError {
        .invalidOption(name: name, value: value, allowed: allowed)
    }

    private static func planRefusal(
        _ name: String, _ value: String, _ allowed: String
    ) -> DownloadError {
        .invalidOption(name: name, value: value, allowed: allowed)
    }

    // MARK: - Bookkeeping

    private func setState(_ id: DownloadJobID, _ path: String, _ state: FileState) {
        guard var job = jobs[id] else { return }
        var states = job.state.fileStates
        states[path] = state
        job.state = updated(job.state, fileStates: states)
        jobs[id] = job
    }

    private func updated(
        _ state: DownloadJobState, fileStates: [String: FileState]
    ) -> DownloadJobState {
        DownloadJobState(
            job: state.job, plan: state.plan, destinationPath: state.destinationPath,
            destinationBookmark: state.destinationBookmark,
            artifactRootPath: state.artifactRootPath,
            artifactRootDevice: state.artifactRootDevice,
            artifactRootInode: state.artifactRootInode,
            options: state.options, fileStates: fileStates, runState: state.runState,
            createdAt: state.createdAt, updatedAt: Date())
    }

    private func persist(_ id: DownloadJobID) throws {
        guard var job = jobs[id] else { throw DownloadError.unknownJob(id) }
        job.state = DownloadJobState(
            job: job.state.job, plan: job.state.plan,
            destinationPath: job.state.destinationPath,
            destinationBookmark: job.state.destinationBookmark,
            artifactRootPath: job.state.artifactRootPath,
            artifactRootDevice: job.state.artifactRootDevice,
            artifactRootInode: job.state.artifactRootInode,
            options: job.state.options, fileStates: job.state.fileStates,
            runState: job.runState, createdAt: job.state.createdAt, updatedAt: Date())
        jobs[id] = job
        try saveState(job.state)
    }

    private func saveState(_ state: DownloadJobState) throws {
        do {
            try stateStore.save(state)
        } catch {
            throw DownloadError.statePersistenceFailed(
                job: state.job, reason: String(describing: error))
        }
    }

    private func persistenceFailure(_ error: Error, job id: DownloadJobID) -> DownloadError {
        if let failure = error as? DownloadError { return failure }
        return .statePersistenceFailed(job: id, reason: String(describing: error))
    }

    /// Resolve a saved per-job bookmark and bind it back to the exact path,
    /// physical root and directory object registered for the job. The caller
    /// owns the returned grant and must release it.
    private func resolveSavedDestination(for job: Job) throws -> ResolvedStorageBookmark {
        guard let bookmark = job.state.destinationBookmark else {
            throw StorageError.bookmarkDataUnresolvable(
                reason: "job \(job.state.job.rawValue) has no saved destination bookmark")
        }
        let resolved = try storage.resolveBookmarkData(bookmark)
        do {
            guard !resolved.scope.isReleased else {
                throw DownloadError.scopeLost(path: resolved.scope.url.path)
            }
            let recordedPath = URL(fileURLWithPath: job.state.destinationPath)
                .standardizedFileURL.path
            let resolvedPath = resolved.scope.url.standardizedFileURL.path
            guard recordedPath == resolvedPath else {
                throw Self.planRefusal(
                    "downloadDestination.bookmark", resolvedPath,
                    "the destination path registered for this job (\(recordedPath))")
            }
            _ = try Self.validate(
                job.state.plan, at: resolved.scope.url,
                expectedPhysicalRoot: job.artifactRoot,
                expectedRootIdentity: job.artifactRootIdentity)
            return resolved
        } catch {
            resolved.scope.release()
            throw error
        }
    }

    /// Acquire a bookmark only when the job no longer has a live grant. A
    /// stale rewrite is persisted before the caller reads or mutates the
    /// external volume; failure restores the original in-memory job.
    @discardableResult
    private func activateSavedDestinationIfNeeded(_ id: DownloadJobID) throws -> StorageScope? {
        guard let original = jobs[id] else { throw DownloadError.unknownJob(id) }
        if let scope = original.destination.scope, !scope.isReleased { return nil }

        let resolved = try resolveSavedDestination(for: original)
        var active = original
        active.destination = DownloadDestination(
            directory: resolved.scope.url, scope: resolved.scope)
        active.state = DownloadJobState(
            job: active.state.job, plan: active.state.plan,
            destinationPath: resolved.scope.url.standardizedFileURL.path,
            destinationBookmark: resolved.bookmarkData,
            artifactRootPath: active.state.artifactRootPath,
            artifactRootDevice: active.state.artifactRootDevice,
            artifactRootInode: active.state.artifactRootInode,
            options: active.state.options, fileStates: active.state.fileStates,
            runState: active.runState, createdAt: active.state.createdAt,
            updatedAt: Date())
        jobs[id] = active
        do {
            try persist(id)
        } catch {
            jobs[id] = original
            resolved.scope.release()
            throw persistenceFailure(error, job: id)
        }
        return resolved.scope
    }

    @discardableResult
    private func finishFailed(_ id: DownloadJobID, _ failure: DownloadError) -> DownloadError {
        jobs[id]?.runState = .failed
        let reported: DownloadError
        do {
            try persist(id)
            reported = failure
        } catch {
            reported = persistenceFailure(error, job: id)
        }
        hub.emit(.failed(reported, progress(of: id)), for: id)
        releaseScope(id)
        return reported
    }

    private func releaseScope(_ id: DownloadJobID) {
        jobs[id]?.destination.scope?.release()
    }

    @discardableResult
    private func finishCancelled(
        _ id: DownloadJobID, keepPartialFiles: Bool
    ) -> DownloadError? {
        guard jobs[id] != nil else { return .unknownJob(id) }

        if !keepPartialFiles {
            do {
                _ = try activateSavedDestinationIfNeeded(id)
            } catch let failure as DownloadError {
                return finishFailed(id, failure)
            } catch {
                let failure = DownloadError.interrupted(
                    path: jobs[id]?.state.destinationPath ?? "", atByte: 0,
                    underlying:
                        "could not reopen the destination to discard partial files: \(String(describing: error))")
                return finishFailed(id, failure)
            }
        }

        jobs[id]?.runState = .cancelled
        do {
            // Cancellation is durable before deleting a resumable prefix. A
            // state-store failure therefore leaves every part file intact.
            try persist(id)
        } catch {
            return finishFailed(id, persistenceFailure(error, job: id))
        }

        guard let job = jobs[id] else { return .unknownJob(id) }
        if !keepPartialFiles {
            for file in job.state.plan.files {
                do {
                    try Self.unlinkArtifactLeafIfPresent(
                        file.path + Self.partSuffix, root: job.artifactRoot,
                        rootIdentity: job.artifactRootIdentity, requireSingleLink: true)
                } catch let failure as DownloadError {
                    return finishFailed(id, failure)
                } catch {
                    let failure = DownloadError.destinationNotWritable(
                        path: job.destination.directory.path,
                        reason: String(describing: error))
                    return finishFailed(id, failure)
                }
            }

            do {
                let validated = try Self.validate(
                    job.state.plan, at: job.destination.directory,
                    expectedPhysicalRoot: job.artifactRoot,
                    expectedRootIdentity: job.artifactRootIdentity)
                let states = adoptFileStates(
                    plan: job.state.plan, pathsByFile: validated.pathsByFile)
                if var updatedJob = jobs[id] {
                    updatedJob.state = updated(updatedJob.state, fileStates: states)
                    jobs[id] = updatedJob
                }
                // The first cancelled record already made the destructive
                // intent durable. This second save records the post-unlink
                // disk truth so completed finals are retained and only the
                // discarded prefixes disappear from progress.
                try persist(id)
            } catch let failure as DownloadError {
                return finishFailed(id, failure)
            } catch {
                let failure = DownloadError.destinationNotWritable(
                    path: job.destination.directory.path,
                    reason: String(describing: error))
                return finishFailed(id, failure)
            }
        }
        hub.emit(.cancelled(progress(of: id)), for: id)
        releaseScope(id)
        return nil
    }

    private func sampleRate(_ id: DownloadJobID) {
        guard var job = jobs[id] else { return }
        let now = Date()
        let bytes = job.counters.networkBytes
        let dt = now.timeIntervalSince(job.lastSampleAt)
        guard dt > 0 else { return }
        let delta = bytes >= job.lastSampleBytes ? bytes - job.lastSampleBytes : 0
        let instantaneous = Double(delta) / dt
        // Exponentially weighted, time-constant based rather than
        // sample-count based, so a burst of tiny chunks does not move the
        // average faster than a burst of large ones.
        let alpha = 1 - exp(-dt / Self.rateTimeConstant)
        job.smoothedBytesPerSecond =
            job.smoothedBytesPerSecond == 0
            ? instantaneous
            : job.smoothedBytesPerSecond + alpha * (instantaneous - job.smoothedBytesPerSecond)
        job.lastSampleAt = now
        job.lastSampleBytes = bytes
        job.lastInstantaneous = instantaneous
        jobs[id] = job
    }

    private func progress(of id: DownloadJobID) -> DownloadProgress {
        guard let job = jobs[id] else {
            return DownloadProgress(
                job: id, planTotalBytes: 0, verifiedBytes: 0, fetchedUnverifiedBytes: 0,
                inFlightBytes: 0, remainingBytes: 0, filesTotal: 0, filesVerified: 0,
                filesFailed: 0, networkBytes: 0, wastedBytes: 0,
                instantaneousBytesPerSecond: 0, smoothedBytesPerSecond: 0,
                estimatedTimeRemaining: nil, startedAt: Date(), elapsed: 0)
        }
        var verified: UInt64 = 0
        var fetched: UInt64 = 0
        var inFlight: UInt64 = 0
        var remaining: UInt64 = 0
        var verifiedFiles = 0
        var failedFiles = 0

        // Each bucket is summed independently rather than one of them being
        // derived by subtraction. Derivation would make
        // `accountsForEveryByte` true by construction and therefore useless;
        // summed independently, the identity is a real check on the state
        // machine.
        for file in job.state.plan.files {
            switch job.state.fileStates[file.path] ?? .pending {
            case .pending:
                remaining += file.sizeBytes
            case .failed:
                failedFiles += 1
                remaining += file.sizeBytes
            case .partial(let bytes):
                let held = min(bytes, file.sizeBytes)
                inFlight += held
                remaining += file.sizeBytes - held
            case .fetched:
                fetched += file.sizeBytes
            case .verified:
                verifiedFiles += 1
                verified += file.sizeBytes
            }
        }

        let elapsed = Date().timeIntervalSince(job.startedAt)
        let smoothed = job.smoothedBytesPerSecond
        let outstanding = remaining + inFlight
        return DownloadProgress(
            job: id,
            planTotalBytes: job.state.plan.totalBytes,
            verifiedBytes: verified, fetchedUnverifiedBytes: fetched,
            inFlightBytes: inFlight, remainingBytes: remaining,
            filesTotal: job.state.plan.files.count, filesVerified: verifiedFiles,
            filesFailed: failedFiles,
            networkBytes: job.counters.networkBytes, wastedBytes: job.counters.wastedBytes,
            instantaneousBytesPerSecond: job.lastInstantaneous,
            smoothedBytesPerSecond: smoothed,
            estimatedTimeRemaining: smoothed > 0 ? Double(outstanding) / smoothed : nil,
            startedAt: job.startedAt, elapsed: elapsed)
    }

    private func emptyReport(_ id: DownloadJobID, depth: VerificationDepth) -> VerificationReport {
        VerificationReport(
            job: id, model: jobs[id]?.state.plan.model ?? ModelID(""), depth: depth,
            checkedAt: Date(), ok: [], missing: [], wrongSize: [], wrongDigest: [],
            unreadable: [:], extraneous: [])
    }

    /// Every regular file under `directory`, as destination-relative paths.
    static func filesOnDisk(under directory: URL) -> [String] {
        guard
            let walker = FileManager.default.enumerator(
                at: directory, includingPropertiesForKeys: [.isRegularFileKey])
        else { return [] }
        let prefix = directory.standardizedFileURL.path + "/"
        var out: [String] = []
        for case let url as URL in walker {
            guard
                (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true
            else { continue }
            let path = url.standardizedFileURL.path
            guard path.hasPrefix(prefix) else { continue }
            out.append(String(path.dropFirst(prefix.count)))
        }
        return out
    }
}
