#if DEBUG
import Foundation
import MinirunKit

/// A `DownloadManaging` that moves no bytes.
///
/// It exists so the download half of the product can be built and reviewed
/// before MinirunKit's real manager lands, and it is written to be *hard* on
/// the UI rather than easy: it pauses, it is interrupted, it resumes from a
/// byte offset, it fails a digest, and it keeps the byte-accounting identity
/// true at every event — `verified + fetchedUnverified + inFlight + remaining
/// == planTotal` — because a screen that only ever sees balanced accounting
/// never proves it can show unbalanced accounting.
///
/// **Clock:** the transfer runs on a simulated clock. The *rate* it reports is
/// a plausible one for the destination link, and the ETA it reports is the real
/// arithmetic on that rate — 1.56 TB at 0.92 GB/s is nineteen days, and the
/// card says nineteen days. Only the simulated clock advances faster than the
/// wall one, so a full transfer completes in about forty seconds of review time.
final class MockDownloadManager: DownloadManaging, @unchecked Sendable {

    /// What the world is going to do to this transfer, and when. Set before
    /// `start`; the app's fault menu drives it.
    struct Fault: Sendable, Equatable {
        var reason: InterruptionReason
        /// Fraction of the plan at which it strikes.
        var atFraction: Double
    }

    /// A file whose digest will not match, so the repair path is reachable.
    struct DigestFault: Sendable, Equatable {
        var pathIndex: Int
    }

    private struct Job {
        let id: DownloadJobID
        let plan: DownloadPlan
        let destination: String
        let options: DownloadOptions
        var state: DownloadJobSummary.JobState
        var progress: DownloadProgress
        var fileIndex: Int
        var bytesIntoCurrentFile: UInt64
        var task: Task<Void, Never>?
    }

    private let lock = NSLock()
    private var jobs: [DownloadJobID: Job] = [:]
    private var continuations: [DownloadJobID: AsyncStream<DownloadEvent>.Continuation] = [:]

    /// Bytes per second the simulated link is asked to deliver.
    var linkBytesPerSecond: Double = 0.92e9
    /// Wall seconds a whole transfer should take to review.
    var reviewSeconds: Double = 40
    var fault: Fault?
    var digestFault: DigestFault?
    /// Set on resume so the interrupted transfer can be made to succeed the
    /// second time, which is the ordinary case.
    var clearFaultOnResume = true

    private var entriesByModel: [ModelID: CatalogEntry]

    init(entries: [CatalogEntry] = CatalogFixtures.all) {
        entriesByModel = Dictionary(uniqueKeysWithValues: entries.map { ($0.id, $0) })
    }

    /// Catalog refresh may add descriptors while a multi-day simulated job is
    /// paused or in flight. Merge the new rows into the planning table; jobs
    /// keep their own immutable plan and are never discarded by this update.
    func updateEntries(_ entries: [CatalogEntry]) {
        lock.lock()
        for entry in entries { entriesByModel[entry.id] = entry }
        lock.unlock()
    }

    // MARK: Planning

    func plan(for model: ModelDescriptor) async throws -> DownloadPlan {
        let entry = withLock { entriesByModel[model.id] }
        guard let entry else {
            throw CatalogError.unknownModel(model.id)
        }
        try? await Task.sleep(for: .milliseconds(600))
        guard let plan = PlanBuilder.plan(for: entry) else {
            throw CatalogError.unrecognizedLayout(
                repoID: model.source.repo?.repoID ?? model.id.rawValue)
        }
        return plan
    }

    func preflight(
        _ plan: DownloadPlan, into destination: DownloadDestination, options: DownloadOptions
    ) async throws -> DownloadPreflight {
        let options = try options.validated()
        let storage = StorageManager()
        let info = storage.describe(destination.directory)
        let verdict = storage.canHold(
            bytes: plan.totalBytes, at: destination.directory,
            headroomBytes: options.freeSpaceHeadroomBytes)

        var refusals: [DownloadError] = []
        var fits = false
        switch verdict {
        case .fits:
            fits = true
        case .refused(let error):
            refusals = [error]
        case .unknown(let reason):
            refusals = [
                .destinationNotWritable(path: info.resolvedPath, reason: reason)
            ]
        }
        if !info.pathIsWritable && info.pathExists {
            refusals.append(
                .destinationNotWritable(
                    path: info.resolvedPath, reason: "the path exists and is not writable"))
            fits = false
        }

        return DownloadPreflight(
            destination: destination.directory.path,
            storage: StorageInfoSnapshot(info),
            requiredBytes: plan.totalBytes,
            alreadyPresentBytes: 0,
            headroomBytes: options.freeSpaceHeadroomBytes,
            dependableFreeBytes: info.volumeAvailableBytes,
            fits: fits,
            refusals: refusals,
            estimatedDuration: Double(plan.totalBytes) / linkBytesPerSecond)
    }

    // MARK: Running

    @discardableResult
    func start(
        _ plan: DownloadPlan, into destination: DownloadDestination, options: DownloadOptions
    ) async throws -> DownloadJobID {
        let options = try options.validated()
        let id = DownloadJobID()
        let job = Job(
            id: id, plan: plan, destination: destination.directory.path, options: options,
            state: .running,
            progress: emptyProgress(job: id, plan: plan),
            fileIndex: 0, bytesIntoCurrentFile: 0, task: nil)
        withLock { jobs[id] = job }

        emit(.planned(plan), to: id)
        run(id)
        return id
    }

    /// Re-subscribing replaces the sink; the app has exactly one.
    nonisolated func events(for job: DownloadJobID) -> AsyncStream<DownloadEvent> {
        let (stream, continuation) = AsyncStream<DownloadEvent>.makeStream(
            bufferingPolicy: .bufferingNewest(256))
        withLock {
            continuations[job]?.finish()
            continuations[job] = continuation
        }
        return stream
    }

    func pause(_ job: DownloadJobID) async throws {
        try mutate(job) { $0.state = .paused; $0.task?.cancel(); $0.task = nil }
        if let progress = snapshot(job)?.progress { emit(.paused(progress), to: job) }
    }

    func resume(_ job: DownloadJobID, scope: StorageScope?) async throws {
        guard let current = snapshot(job) else { throw DownloadError.unknownJob(job) }
        guard current.state == .paused else {
            throw DownloadError.jobNotPaused(job)
        }
        if clearFaultOnResume { fault = nil }
        try mutate(job) { $0.state = .running }
        if let progress = snapshot(job)?.progress { emit(.resumed(progress), to: job) }
        run(job)
    }

    func cancel(_ job: DownloadJobID, keepPartialFiles: Bool) async throws {
        try mutate(job) { $0.state = .cancelled; $0.task?.cancel(); $0.task = nil }
        if let progress = snapshot(job)?.progress { emit(.cancelled(progress), to: job) }
    }

    func verify(_ job: DownloadJobID, depth: VerificationDepth) async throws
        -> VerificationReport
    {
        guard let current = snapshot(job) else { throw DownloadError.unknownJob(job) }
        try mutate(job) { $0.state = .verifying }

        let files = current.plan.files
        let checked = depth == .digestEverything ? files : files.filter(\.isPayload)

        var wrongDigest: [DigestMismatch] = []
        if let digestFault, digestFault.pathIndex < checked.count, depth != .sizeOnly {
            let file = checked[digestFault.pathIndex]
            wrongDigest.append(
                DigestMismatch(
                    path: file.path, algorithm: file.digest.algorithm,
                    expected: file.digest.hex,
                    found: PlanBuilder.syntheticHex(file.path + "-corrupt", length: file.digest.hex.count)))
        }

        var bytesToRefetch: UInt64 = 0
        for mismatch in wrongDigest {
            guard let file = files.first(where: { $0.path == mismatch.path }) else { continue }
            let (next, overflow) = bytesToRefetch.addingReportingOverflow(file.sizeBytes)
            bytesToRefetch = overflow ? .max : next
        }
        let report = VerificationReport(
            job: job, model: current.plan.model, depth: depth, checkedAt: Date(),
            ok: files.map(\.path).filter { path in !wrongDigest.contains { $0.path == path } },
            missing: [], wrongSize: [], wrongDigest: wrongDigest, unreadable: [:],
            extraneous: [], bytesToRefetch: bytesToRefetch)
        emit(.verificationFinished(report), to: job)
        try mutate(job) { $0.state = report.isComplete ? .finished : .failed }
        return report
    }

    @discardableResult
    func repair(_ job: DownloadJobID, from report: VerificationReport) async throws
        -> DownloadJobID
    {
        digestFault = nil
        try mutate(job) { $0.state = .running }
        for path in report.pathsToRefetch {
            emit(.fileStarted(path: path, bytes: 0, resumedAtByte: 0), to: job)
            try? await Task.sleep(for: .milliseconds(250))
            emit(
                .fileVerified(path: path, digest: .sha256(hex: PlanBuilder.syntheticHex(path, length: 64)), bytes: 0),
                to: job)
        }
        _ = try await verify(job, depth: report.depth)
        return job
    }

    func jobs() async -> [DownloadJobSummary] {
        withLock {
            jobs.values.map {
                DownloadJobSummary(
                    job: $0.id, model: $0.plan.model, state: $0.state, progress: $0.progress)
            }
        }
    }

    // MARK: The simulation

    private func run(_ id: DownloadJobID) {
        guard let current = snapshot(id) else { return }
        let plan = current.plan
        let tickHz = 8.0
        let totalSimulatedSeconds = Double(plan.totalBytes) / linkBytesPerSecond
        // Simulated seconds advanced per wall tick, so the whole transfer takes
        // `reviewSeconds` of the reviewer's time regardless of its size.
        let simulatedSecondsPerTick = totalSimulatedSeconds / (reviewSeconds * tickHz)

        let task = Task { [weak self] in
            guard let self else { return }
            var elapsedSimulated = self.snapshot(id)?.progress.elapsed ?? 0

            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(Int(1000 / tickHz)))
                if Task.isCancelled { return }
                guard let job = self.snapshot(id), job.state == .running else { return }

                elapsedSimulated += simulatedSecondsPerTick
                // A little jitter, because a perfectly flat rate reads as a
                // progress bar rather than as a transfer.
                let jitter = 1.0 + 0.06 * sin(elapsedSimulated / 37)
                let bytesThisTick = UInt64(
                    max(0, self.linkBytesPerSecond * simulatedSecondsPerTick * jitter))

                let done = self.advance(id, by: bytesThisTick, elapsed: elapsedSimulated)
                guard let progress = self.snapshot(id)?.progress else { return }

                if let fault = self.fault,
                    Double(progress.verifiedBytes) / Double(max(1, plan.totalBytes))
                        >= fault.atFraction
                {
                    self.emitFault(fault, on: id, progress: progress)
                    return
                }

                self.emit(.progress(progress), to: id)
                if done {
                    self.finish(id)
                    return
                }
            }
        }
        try? mutate(id) { $0.task = task }
    }

    /// Advances the byte buckets, keeping the identity true. Returns whether
    /// the plan is complete.
    private func advance(_ id: DownloadJobID, by bytes: UInt64, elapsed: Double) -> Bool {
        withLock { advanceLocked(id, by: bytes, elapsed: elapsed) }
    }

    private func advanceLocked(_ id: DownloadJobID, by bytes: UInt64, elapsed: Double) -> Bool {
        guard var job = jobs[id] else { return true }

        var budget = bytes
        var newlyVerifiedFiles = 0
        var verified = job.progress.verifiedBytes
        var inFlight = job.progress.inFlightBytes

        while budget > 0 && job.fileIndex < job.plan.files.count {
            let file = job.plan.files[job.fileIndex]
            let wanted = file.sizeBytes - job.bytesIntoCurrentFile
            if budget >= wanted {
                budget -= wanted
                verified += file.sizeBytes
                inFlight = 0
                job.bytesIntoCurrentFile = 0
                job.fileIndex += 1
                newlyVerifiedFiles += 1
            } else {
                job.bytesIntoCurrentFile += budget
                inFlight = job.bytesIntoCurrentFile
                budget = 0
            }
        }

        let accounted = verified &+ inFlight
        let remaining = job.plan.totalBytes > accounted ? job.plan.totalBytes - accounted : 0
        let rate = elapsed > 0 ? Double(verified &+ inFlight) / elapsed : 0

        job.progress = DownloadProgress(
            job: id, planTotalBytes: job.plan.totalBytes,
            verifiedBytes: verified, fetchedUnverifiedBytes: 0, inFlightBytes: inFlight,
            remainingBytes: remaining,
            filesTotal: job.plan.files.count, filesVerified: job.fileIndex, filesFailed: 0,
            networkBytes: verified &+ inFlight, wastedBytes: 0,
            instantaneousBytesPerSecond: linkBytesPerSecond,
            smoothedBytesPerSecond: rate > 0 ? rate : linkBytesPerSecond,
            estimatedTimeRemaining: rate > 0 ? Double(remaining) / rate : nil,
            startedAt: job.progress.startedAt, elapsed: elapsed)

        let currentPath =
            job.fileIndex < job.plan.files.count ? job.plan.files[job.fileIndex].path : nil
        jobs[id] = job
        _ = currentPath
        return job.fileIndex >= job.plan.files.count
    }

    private func emitFault(_ fault: Fault, on id: DownloadJobID, progress: DownloadProgress) {
        try? mutate(id) { $0.state = .failed; $0.task?.cancel(); $0.task = nil }
        let error: DownloadError
        switch fault.reason {
        case .volumeDisappeared(let name):
            error = .scopeLost(path: "/Volumes/\(name)")
        case .network:
            error = .interrupted(
                path: currentPath(id) ?? "", atByte: progress.inFlightBytes,
                underlying: "The network connection was lost.")
        case .transfer(let reason):
            error = .interrupted(
                path: currentPath(id) ?? "", atByte: progress.inFlightBytes,
                underlying: reason)
        case .spaceExhausted(let deficit):
            error = .insufficientFreeSpace(
                needBytes: deficit, headroomBytes: 8 << 30,
                availableBytes: 0, volume: nil)
        case .thermal:
            error = .interrupted(
                path: currentPath(id) ?? "", atByte: progress.inFlightBytes,
                underlying: "Thermal state critical; the transfer was stood down.")
        }
        emit(.failed(error, progress), to: id)
    }

    private func finish(_ id: DownloadJobID) {
        guard let job = snapshot(id) else { return }
        let report = VerificationReport(
            job: id, model: job.plan.model, depth: .digestPayload, checkedAt: Date(),
            ok: job.plan.files.map(\.path),
            missing: [], wrongSize: [], wrongDigest: [], unreadable: [:], extraneous: [])
        let summary = DownloadSummary(
            job: id, model: job.plan.model, repo: job.plan.repo,
            destinationPath: job.destination, bytesWritten: job.plan.totalBytes,
            networkBytes: job.plan.totalBytes, wastedBytes: 0,
            wallSeconds: job.progress.elapsed,
            meanBytesPerSecond: job.progress.elapsed > 0
                ? Double(job.plan.totalBytes) / job.progress.elapsed : 0,
            verification: report)
        try? mutate(id) { $0.state = .finished }
        emit(.finished(summary), to: id)
    }

    // MARK: Plumbing

    private func emptyProgress(job: DownloadJobID, plan: DownloadPlan) -> DownloadProgress {
        DownloadProgress(
            job: job, planTotalBytes: plan.totalBytes, verifiedBytes: 0,
            fetchedUnverifiedBytes: 0, inFlightBytes: 0, remainingBytes: plan.totalBytes,
            filesTotal: plan.files.count, filesVerified: 0, filesFailed: 0,
            networkBytes: 0, wastedBytes: 0, instantaneousBytesPerSecond: 0,
            smoothedBytesPerSecond: 0, estimatedTimeRemaining: nil, startedAt: Date(),
            elapsed: 0)
    }

    /// One place that takes the lock, so no call site has to remember to.
    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }

    private func snapshot(_ id: DownloadJobID) -> Job? {
        withLock { jobs[id] }
    }

    private func currentPath(_ id: DownloadJobID) -> String? {
        guard let job = snapshot(id), job.fileIndex < job.plan.files.count else { return nil }
        return job.plan.files[job.fileIndex].path
    }

    private func mutate(_ id: DownloadJobID, _ body: (inout Job) -> Void) throws {
        let found: Bool = withLock {
            guard var job = jobs[id] else { return false }
            body(&job)
            jobs[id] = job
            return true
        }
        if !found { throw DownloadError.unknownJob(id) }
    }

    private func emit(_ event: DownloadEvent, to id: DownloadJobID) {
        withLock { continuations[id] }?.yield(event)
    }
}
#endif
