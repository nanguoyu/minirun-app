import Foundation
import MinirunKit
import Observation

/// Owns one model's transfer and turns `DownloadEvent`s into the state machine
/// the download card renders.
///
/// `paused` and `interrupted` are ordinary states here, not error paths: a
/// drive unplugged at 900 GB is a Tuesday.
@Observable
@MainActor
final class DownloadController {

    let model: ModelID
    private(set) var state: DownloadState = .notStarted
    private(set) var plan: DownloadPlan?
    private(set) var report: VerificationReport?
    private(set) var lastError: DownloadError?
    private(set) var operationError: String?
    /// The measured size, once verification has made the files authoritative
    /// over the manifests. Nil before that — and the UI renders the declared
    /// number in italic until it is not nil.
    private(set) var measuredBytes: UInt64?
    private(set) var destinationPath: String?
    private(set) var volumeName: String?
    private(set) var jobID: DownloadJobID?
    private(set) var createdAt = Date()
    private(set) var updatedAt = Date()
    /// True when the catalog moved or removed this descriptor after a plan was
    /// already resolved. The job is preserved; the UI labels that its immutable
    /// plan belongs to the earlier snapshot.
    private(set) var catalogEntryIsStale = false

    private let manager: any DownloadManaging
    private var entry: CatalogEntry
    private var eventTask: Task<Void, Never>?
    private var verificationTask: Task<Void, Never>?
    private var verificationToken: UUID?
    private var stateBeforeVerification: DownloadState?
    private(set) var isRepairing = false

    var onJobRegistered: ((DownloadJobID) -> Void)?
    var onArtifactVerified: ((DownloadJobID, URL) -> Void)?

    /// Whether removing this catalog row would discard user-visible work or a
    /// plan already resolved against an earlier descriptor.
    var hasCatalogBoundState: Bool {
        state != .notStarted || plan != nil || jobID != nil
    }

    var blocksNewTransfer: Bool {
        switch state {
        case .resolvingIndex, .awaitingDestination, .active, .paused, .verifying:
            return true
        case .notStarted, .interrupted, .cancelled, .ready, .incomplete, .failed:
            return false
        }
    }

    /// Whether this controller may still read or mutate files belonging to its
    /// artifact. A stopped terminal job does not; a paused/interrupted job does
    /// because it remains resumable, and repair gets its own explicit latch for
    /// the await before its replacement job begins emitting events.
    var protectsLocalCopyRemoval: Bool {
        if isRepairing { return true }
        switch state {
        case .resolvingIndex, .awaitingDestination, .active, .paused, .interrupted,
            .verifying:
            return true
        case .notStarted, .cancelled, .ready, .incomplete, .failed:
            return false
        }
    }

    init(entry: CatalogEntry, manager: any DownloadManaging) {
        self.entry = entry
        self.model = entry.id
        self.manager = manager
    }

    init(
        entry: CatalogEntry, manager: any DownloadManaging,
        restoredState: DownloadJobState, summary: DownloadJobSummary
    ) {
        self.entry = entry
        self.model = entry.id
        self.manager = manager
        self.plan = restoredState.plan
        self.jobID = restoredState.job
        self.destinationPath = restoredState.destinationPath
        self.volumeName = StorageManager().describe(
            URL(fileURLWithPath: restoredState.destinationPath, isDirectory: true)
        ).volumeName
        self.createdAt = restoredState.createdAt
        self.updatedAt = restoredState.updatedAt
        self.state = Self.stoppedState(from: summary, plan: restoredState.plan)
        // Restoration is deliberately zero-I/O and manager summaries are not
        // trusted to imply a running subscription. Attach to the event stream
        // only after the operator explicitly resumes; otherwise a stale or
        // malicious replay could overwrite this stopped state with `active`.
    }

    /// Adopt a refreshed descriptor only before a plan exists. Once a transfer
    /// has a plan, changing it underneath the job would be worse than retaining
    /// the old one, so the controller stays intact and marks the difference.
    func updateCatalogEntry(_ fresh: CatalogEntry?) {
        guard let fresh else {
            catalogEntryIsStale = true
            return
        }
        guard plan == nil, jobID == nil else {
            catalogEntryIsStale = fresh.descriptor != entry.descriptor
            return
        }
        entry = fresh
        catalogEntryIsStale = false
    }

    // MARK: Driving

    func resolvePlan() async {
        guard plan == nil else { return }
        operationError = nil
        state = .resolvingIndex
        do {
            plan = try await manager.plan(for: entry.descriptor)
            state = .awaitingDestination
        } catch {
            operationError = String(describing: error)
            state = .failed(.environmentUnavailable("\(error)"))
        }
    }

    @discardableResult
    func start(
        destination: URL, scope: StorageScope?, options: DownloadOptions
    ) async -> Bool {
        await resolvePlan()
        guard let plan else {
            // The manager takes ownership only when `start` is actually
            // called. A planning refusal happens before that boundary, so the
            // UI layer must close the picker grant itself.
            scope?.release()
            return false
        }
        destinationPath = destination.path
        volumeName = StorageManager().describe(destination).volumeName
        operationError = nil

        do {
            // Ownership of the live grant transfers to the manager with this
            // call. A paused or restored job re-resolves its persisted per-job
            // bookmark instead of retaining a stale scope in the UI layer.
            let id = try await manager.start(
                plan, into: DownloadDestination(directory: destination, scope: scope),
                options: options)
            jobID = id
            createdAt = Date()
            updatedAt = createdAt
            onJobRegistered?(id)
            subscribe(to: id, plan: plan)
            return true
        } catch let error as DownloadError {
            // `DownloadManaging` owns the grant after the call begins, but a
            // fail-closed or test conformer may reject before retaining it.
            // Release is idempotent, so this closes both implementations.
            scope?.release()
            lastError = error
            operationError = error.description
            state = .failed(.invalidArgument(error.description))
            return false
        } catch {
            scope?.release()
            operationError = String(describing: error)
            state = .failed(.environmentUnavailable("\(error)"))
            return false
        }
    }

    func pause() async {
        guard let jobID else { return }
        operationError = nil
        do {
            try await manager.pause(jobID)
        } catch {
            operationError = String(describing: error)
        }
    }

    func resume() async {
        guard let jobID else { return }
        operationError = nil
        do {
            // Restored jobs re-resolve their durable per-job bookmark. No
            // plain path or stale UI scope is offered as a fallback.
            try await manager.resume(jobID, scope: nil)
            if let plan { subscribe(to: jobID, plan: plan) }
        } catch let error as DownloadError {
            lastError = error
            operationError = error.description
        } catch {
            operationError = String(describing: error)
        }
    }

    func cancel() async {
        guard let jobID, let plan else { return }
        operationError = nil
        do {
            try await manager.cancel(jobID, keepPartialFiles: true)
            // The event stream normally supplies this state. Set it here too
            // because cancellation is already durable when the await returns.
            state = .cancelled(
                state.progress
                    ?? ProgressSnapshot.empty(
                        totalBytes: plan.totalBytes, filesTotal: plan.files.count))
            updatedAt = Date()
        } catch let error as DownloadError {
            lastError = error
            operationError = error.description
        } catch {
            operationError = String(describing: error)
        }
    }

    /// The manager currently reports a final file report but no intermediate
    /// count. Show an honest indeterminate state rather than animating invented
    /// geometry or tile totals.
    func verify(depth: VerificationDepth = .digestPayload) {
        guard let jobID, let plan else { return }
        guard verificationTask == nil else { return }
        let snapshot = state.progress ?? ProgressSnapshot.empty(
            totalBytes: plan.totalBytes, filesTotal: plan.files.count)
        let previousState = state
        let token = UUID()
        verificationToken = token
        stateBeforeVerification = previousState
        operationError = nil
        state = .verifying(.fileDigests, snapshot)

        verificationTask = Task { [weak self] in
            guard let self else { return }
            do {
                try Task.checkCancellation()
                let outcome = try await self.manager.verify(jobID, depth: depth)
                // Once the manager returns a report, that report is newer disk
                // truth than the state captured before verification. A cancel
                // racing with this return must not resurrect an old `.ready`
                // state after the manager has just proved a mismatch.
                guard self.verificationToken == token else { return }
                try self.apply(
                    verification: outcome, plan: plan, job: jobID, expectedDepth: depth)
                self.verificationTask = nil
                self.verificationToken = nil
                self.stateBeforeVerification = nil
                if outcome.isComplete, let destinationPath = self.destinationPath {
                    self.onArtifactVerified?(
                        jobID, URL(fileURLWithPath: destinationPath, isDirectory: true))
                }
            } catch is CancellationError {
                guard self.verificationToken == token else { return }
                let restore = self.stateBeforeVerification ?? previousState
                self.verificationTask = nil
                self.verificationToken = nil
                self.stateBeforeVerification = nil
                // Cancellation does not change what was on disk before the
                // pass began. Restore that exact ready/incomplete/paused truth
                // instead of leaving a permanent synthetic "verifying" state.
                if case .verifying = self.state { self.state = restore }
            } catch let error as DownloadError where error == .cancelled {
                guard self.verificationToken == token else { return }
                let restore = self.stateBeforeVerification ?? previousState
                self.verificationTask = nil
                self.verificationToken = nil
                self.stateBeforeVerification = nil
                if case .verifying = self.state { self.state = restore }
            } catch let error as DownloadError {
                guard self.verificationToken == token else { return }
                self.verificationTask = nil
                self.verificationToken = nil
                self.stateBeforeVerification = nil
                self.lastError = error
                self.operationError = error.description
                self.state = .failed(.environmentUnavailable(error.description))
            } catch {
                guard self.verificationToken == token else { return }
                self.verificationTask = nil
                self.verificationToken = nil
                self.stateBeforeVerification = nil
                self.operationError = String(describing: error)
                self.state = .failed(.environmentUnavailable("\(error)"))
            }
        }
    }

    func cancelVerification() {
        verificationTask?.cancel()
    }

    func repair() async {
        guard let jobID, let report, let plan else { return }
        guard !isRepairing else { return }
        isRepairing = true
        defer { isRepairing = false }
        operationError = nil
        do {
            let repaired = try await manager.repair(jobID, from: report)
            self.jobID = repaired
            subscribe(to: repaired, plan: plan)
        } catch let error as DownloadError {
            lastError = error
            operationError = error.description
        } catch {
            operationError = String(describing: error)
        }
    }

    #if DEBUG
        /// Marks the artifact as present and verified without moving bytes, so
        /// the Run screen is reachable only in a Debug review session. No
        /// equivalent fake-ready transition is compiled into Release.
        func markReadyForPreview() {
            guard let plan = self.plan ?? PlanBuilder.plan(for: entry) else {
                state = .failed(
                    .invalidArgument(
                        "No preview download plan exists for an unrecognized artifact layout."))
                return
            }
            self.plan = plan
            measuredBytes = plan.totalBytes
            destinationPath =
                destinationPath ?? "/Volumes/MINIRUN-NVME/\(entry.descriptor.id.rawValue)"
            volumeName = volumeName ?? "MINIRUN-NVME"
            state = .ready(measuredBytes: plan.totalBytes, fileCount: plan.files.count)
        }
    #endif

    func reset() {
        eventTask?.cancel()
        verificationTask?.cancel()
        verificationTask = nil
        verificationToken = nil
        stateBeforeVerification = nil
        jobID = nil
        plan = nil
        report = nil
        measuredBytes = nil
        operationError = nil
        state = .notStarted
    }

    // MARK: Events

    private func subscribe(to id: DownloadJobID, plan: DownloadPlan) {
        eventTask?.cancel()
        let stream = manager.events(for: id)
        eventTask = Task { [weak self] in
            for await event in stream {
                self?.apply(event, plan: plan)
            }
        }
    }

    private func apply(_ event: DownloadEvent, plan: DownloadPlan) {
        updatedAt = Date()
        switch event {
        case .planned:
            state = .active(
                ProgressSnapshot.empty(
                    totalBytes: plan.totalBytes, filesTotal: plan.files.count))

        case .progress(let progress):
            state = .active(Self.snapshot(from: progress, plan: plan))

        case .paused(let progress):
            state = .paused(Self.snapshot(from: progress, plan: plan))

        case .resumed(let progress):
            state = .active(Self.snapshot(from: progress, plan: plan))

        case .cancelled(let progress):
            state = .cancelled(Self.snapshot(from: progress, plan: plan))

        case .failed(let error, let progress):
            lastError = error
            operationError = error.description
            switch error {
            case .transport, .interrupted, .scopeLost, .insufficientFreeSpace:
                state = .interrupted(
                    Self.snapshot(from: progress, plan: plan), reason: reason(for: error))
            case .cancelled:
                state = .cancelled(Self.snapshot(from: progress, plan: plan))
            case .invalidOption, .destinationNotWritable, .destinationIsReadOnlyVolume,
                .revisionMoved, .rangeNotSupported, .sizeMismatch, .digestMismatch,
                .digestUnavailable, .unknownJob, .jobNotPaused, .jobAlreadyRunning,
                .statePersistenceFailed, .spaceUnknown:
                state = .failed(.environmentUnavailable(error.description))
            }

        case .finished(let summary):
            // `DownloadManager` has already completed its closing verification
            // before emitting this summary. Applying the included report is
            // idempotent; calling verify again would read the entire artifact a
            // second time and was the source of the old fake two-pass UI.
            guard summary.job == jobID, summary.model == plan.model,
                summary.repo == plan.repo,
                summary.verification.job == summary.job,
                summary.verification.model == summary.model,
                URL(fileURLWithPath: summary.destinationPath, isDirectory: true)
                    .standardizedFileURL.path
                    == URL(fileURLWithPath: destinationPath ?? "", isDirectory: true)
                    .standardizedFileURL.path,
                summary.wastedBytes <= summary.networkBytes,
                summary.wallSeconds.isFinite, summary.wallSeconds >= 0,
                summary.meanBytesPerSecond.isFinite, summary.meanBytesPerSecond >= 0
            else {
                let error = DownloadError.invalidOption(
                    name: "downloadSummary", value: "inconsistent terminal authority",
                    allowed: "the current immutable job, plan and destination")
                operationError = error.description
                state = .failed(.environmentUnavailable(error.description))
                return
            }
            do {
                try apply(
                    verification: summary.verification, plan: plan, job: summary.job,
                    expectedDepth: .digestPayload)
            } catch {
                operationError = String(describing: error)
                state = .failed(.environmentUnavailable(String(describing: error)))
                return
            }
            if summary.verification.isComplete {
                onArtifactVerified?(
                    summary.job,
                    URL(fileURLWithPath: summary.destinationPath, isDirectory: true))
            }

        case .verificationFinished(let outcome):
            guard let jobID else {
                operationError = "verification report arrived before a job was registered"
                state = .failed(
                    .environmentUnavailable(
                        "verification report arrived before a job was registered"))
                return
            }
            do {
                try apply(
                    verification: outcome, plan: plan, job: jobID,
                    expectedDepth: .digestPayload)
            } catch {
                operationError = String(describing: error)
                state = .failed(.environmentUnavailable(String(describing: error)))
            }

        case .fileFailed(_, let error):
            lastError = error
            operationError = error.description

        case .fileVerified, .fileStarted, .preflight:
            break
        }
    }

    private func apply(
        verification outcome: VerificationReport, plan: DownloadPlan,
        job: DownloadJobID, expectedDepth: VerificationDepth
    ) throws {
        try Self.validateVerificationReport(
            outcome, plan: plan, job: job, expectedDepth: expectedDepth)
        report = outcome
        if outcome.isComplete {
            // The flip from declared to measured. Files are now authoritative.
            measuredBytes = plan.totalBytes
            state = .ready(measuredBytes: plan.totalBytes, fileCount: plan.files.count)
        } else {
            let missingBytes = outcome.bytesToRefetch
            state = .incomplete(
                missingFiles: outcome.pathsToRefetch.count, missingBytes: missingBytes)
        }
    }

    private static func validateVerificationReport(
        _ report: VerificationReport, plan: DownloadPlan,
        job: DownloadJobID, expectedDepth: VerificationDepth
    ) throws {
        func refuse(_ field: String, _ reason: String) throws -> Never {
            throw DownloadError.invalidOption(
                name: "verificationReport.\(field)", value: reason,
                allowed: "the immutable download job and plan")
        }
        guard report.job == job else {
            try refuse("job", "another job identity")
        }
        guard report.model == plan.model else {
            try refuse("model", "another model identity")
        }
        guard report.depth == expectedDepth else {
            try refuse("depth", report.depth.rawValue)
        }

        let planByPath = Dictionary(
            plan.files.map { ($0.path, $0) }, uniquingKeysWith: { first, _ in first })
        guard planByPath.count == plan.files.count else {
            try refuse("plan", "duplicate planned paths")
        }
        let allReported = report.ok + report.missing + report.wrongSize.map(\.path)
            + report.wrongDigest.map(\.path) + report.unreadable.keys.sorted()
        var seen = Set<String>()
        for path in allReported {
            guard planByPath[path] != nil else {
                try refuse("path", path)
            }
            guard seen.insert(path).inserted else {
                try refuse("path", "duplicate result for \(path)")
            }
        }
        guard seen == Set(planByPath.keys) else {
            try refuse("coverage", "\(seen.count) of \(planByPath.count) planned files")
        }
        for mismatch in report.wrongSize {
            guard planByPath[mismatch.path]?.sizeBytes == mismatch.expected else {
                try refuse("expectedSize", mismatch.path)
            }
        }
        for mismatch in report.wrongDigest {
            guard let file = planByPath[mismatch.path],
                mismatch.algorithm == file.digest.algorithm,
                mismatch.expected.lowercased() == file.digest.hex
            else {
                try refuse("expectedDigest", mismatch.path)
            }
        }

        var expectedRefetchBytes: UInt64 = 0
        for path in report.pathsToRefetch {
            guard let size = planByPath[path]?.sizeBytes else {
                try refuse("path", path)
            }
            let (next, overflow) = expectedRefetchBytes.addingReportingOverflow(size)
            guard !overflow else { try refuse("bytesToRefetch", "overflow") }
            expectedRefetchBytes = next
        }
        guard report.bytesToRefetch == expectedRefetchBytes else {
            try refuse("bytesToRefetch", "\(report.bytesToRefetch)")
        }
    }

    /// The only product projection of a restored summary. Restoration never
    /// resumes work: active-looking persisted states are presented as paused,
    /// while terminal states remain terminal.
    static func stoppedState(
        from summary: DownloadJobSummary, plan: DownloadPlan
    ) -> DownloadState {
        let progress = snapshot(from: summary.progress, plan: plan)
        switch summary.state {
        case .planned, .running, .paused, .verifying, .finished:
            return .paused(progress)
        case .failed:
            return .failed(
                .environmentUnavailable(
                    "the saved transfer had already failed; continue with the files kept on disk"))
        case .cancelled:
            return .cancelled(progress)
        }
    }

    static func snapshot(from progress: DownloadProgress?, plan: DownloadPlan)
        -> ProgressSnapshot
    {
        guard let progress else {
            return ProgressSnapshot.empty(
                totalBytes: plan.totalBytes, filesTotal: plan.files.count)
        }
        let filesDone = min(max(0, progress.filesVerified), plan.files.count)
        let index = min(filesDone, max(0, plan.files.count - 1))
        let bytesPerSecond = progress.smoothedBytesPerSecond.isFinite
            && progress.smoothedBytesPerSecond >= 0
            ? progress.smoothedBytesPerSecond : 0
        let estimatedTimeRemaining = progress.estimatedTimeRemaining.flatMap {
            $0.isFinite && $0 >= 0 ? $0 : nil
        }
        return ProgressSnapshot(
            totalBytes: progress.planTotalBytes,
            verifiedBytes: progress.verifiedBytes,
            fetchedUnverifiedBytes: progress.fetchedUnverifiedBytes,
            inFlightBytes: progress.inFlightBytes,
            filesTotal: plan.files.count,
            filesDone: filesDone,
            currentFilePath: plan.files.isEmpty ? nil : plan.files[index].path,
            currentFileBytes: plan.files.isEmpty ? 0 : plan.files[index].sizeBytes,
            currentFileOffset: progress.inFlightBytes,
            bytesPerSecond: bytesPerSecond,
            estimatedTimeRemaining: estimatedTimeRemaining,
            networkBytes: progress.networkBytes,
            wastedBytes: progress.wastedBytes)
    }

    private func reason(for error: DownloadError) -> InterruptionReason {
        switch error {
        case .scopeLost(let path):
            return .volumeDisappeared(URL(fileURLWithPath: path).lastPathComponent)
        case .insufficientFreeSpace(let need, _, let available, _):
            let deficit = need > UInt64(max(0, available)) ? need - UInt64(max(0, available)) : 0
            return .spaceExhausted(deficit: deficit)
        case .transport:
            return .network
        case .interrupted(_, _, let underlying):
            return .transfer(underlying)
        default:
            return .transfer(error.description)
        }
    }
}
