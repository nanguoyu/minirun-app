import Foundation
import MinirunKit
import Observation

typealias InstalledArtifactVerificationOperation = @Sendable (
    DiscoveredArtifact, ArtifactVerificationRequest, ArtifactVerificationRoot,
    ArtifactVerificationResumption,
    @escaping @Sendable (ArtifactVerificationProgress) -> Void
) async throws -> VerificationReport

typealias InstalledModelScanOperation = @Sendable (
    any StorageManaging, ModelCatalogSnapshot, any ArtifactVerificationLedger
) async -> DiscoveryReport

typealias InstalledArtifactRemovalOperation = @Sendable (
    ArtifactVerificationRoot
) async throws -> ArtifactRemovalReceipt

enum InstalledArtifactAccessError: Error, Equatable, CustomStringConvertible {
    case notInCurrentScan(path: String)
    case locationUnavailable(path: String)
    case missingStorageAuthority(path: String)
    case outsideRegisteredLocation(artifact: String, location: String)
    case verificationEvidenceUnavailable(path: String)
    case verificationEvidenceNotRecorded(path: String)

    var description: String {
        switch self {
        case .notInCurrentScan(let path):
            return "verification refused because '\(path)' is no longer in the current storage scan"
        case .locationUnavailable(let path):
            return "verification refused because the registered storage location '\(path)' is unavailable"
        case .missingStorageAuthority(let path):
            return "verification refused because no registered storage bookmark owns '\(path)'"
        case .outsideRegisteredLocation(let artifact, let location):
            return
                "verification refused because '\(artifact)' is outside its registered storage location '\(location)'"
        case .verificationEvidenceUnavailable(let path):
            return
                "running '\(path)' was refused because its complete verification evidence no longer matches the opened files"
        case .verificationEvidenceNotRecorded(let path):
            return
                "verification read every requested file, but durable evidence for '\(path)' could not be confirmed"
        }
    }
}

enum InstalledArtifactRemovalError: Error, Equatable, CustomStringConvertible {
    case verificationInProgress(path: String)
    case removalInProgress(path: String)

    var description: String {
        switch self {
        case .verificationInProgress(let path):
            return "removal refused while '\(path)' is being verified"
        case .removalInProgress(let path):
            return "removal refused because '\(path)' is already being removed"
        }
    }
}

/// What is actually on the volumes the operator registered.
///
/// The app used to answer "is this model installed?" from a `DownloadState` —
/// which is a record of what *this app* fetched, and therefore says nothing
/// about an artifact copied on by hand, or one on a drive that is currently in a
/// drawer. Both are the normal case here. So the answer comes from a scan, and
/// the download state is left to describe transfers, which is what it is for.
///
/// Two rules this type exists to keep:
///
/// 1. **Scanning never verifies.** ``ArtifactLocator`` reads directory entries;
///    hashing 1.56 TB happens only through ``verify(_:_:)``, which is wired to a
///    button and to nothing else.
/// 2. **A scan never runs on the main actor.** Walking K3NVME is roughly half a
///    second of `stat` calls. Half a second of dropped frames every time a
///    volume mounts is a scan the operator would learn to dread.
@Observable
@MainActor
final class InstalledModels {

    private(set) var report: DiscoveryReport
    private(set) var isScanning = false
    /// Non-nil when remembering or forgetting a location failed. Shown, not
    /// logged.
    private(set) var locationError: String?
    private(set) var scanCount = 0

    /// How the last attempt to add a folder ended.
    ///
    /// This exists because of a specific failure: the owner's app had no
    /// storage location registered at all, so the scanner never ran and every
    /// model read as missing — and nothing on any screen said which of "you
    /// never added one", "you cancelled the picker" or "the bookmark could not
    /// be minted" had happened. Every one of those now has a sentence.
    enum AddOutcome: Equatable {
        case added(path: String)
        case alreadyRegistered(path: String)
        case cancelled
        case failed(String)

        var isFailure: Bool {
            switch self {
            case .added, .alreadyRegistered: return false
            case .cancelled, .failed: return true
            }
        }
    }

    private(set) var lastAddOutcome: AddOutcome?

    /// Whether anything at all has been registered. When this is false the
    /// scanner has never had a folder to walk, and a model list showing
    /// "not installed" is describing the app's ignorance, not the disk.
    var hasLocation: Bool { !locationKeys.isEmpty }

    /// A verification the operator started, and how far it has got.
    private(set) var verificationInFlight: String?
    private(set) var verificationProgress: ArtifactVerificationProgress?
    private(set) var verificationOutcome: [String: String] = [:]
    /// A verification the system stopped before it finished.
    ///
    /// This is not a failure and not a cancellation. Every file it had already
    /// digested is in the kit's checkpoint, so the pass resumes rather than
    /// restarts — automatically, when the app is frontmost again, because the
    /// operator asked for this work and never asked for it to stop.
    private(set) var pausedVerification: PausedVerification?
    private(set) var removalInFlight: String?
    private(set) var removalOutcome: [String: String] = [:]
    @ObservationIgnored private var verificationTask: Task<Void, Never>?
    /// Whether the pass that just ended ended well. Read once, after the task
    /// has left the stack, to close the background continuation honestly.
    @ObservationIgnored private var verificationSucceeded = false
    /// Built while the pass's own task is still on the stack, posted after it
    /// has left. Not observable: it is a message, not screen state.
    @ObservationIgnored private var pendingAnnouncement: VerificationAnnouncement?
    /// Product coordination hook, deliberately not observable state. A scan
    /// can prove that a recent cached catalog predates newly published local
    /// files; AppModel owns the network decision that follows.
    @ObservationIgnored var onReportChanged: (@MainActor (DiscoveryReport) -> Void)?
    private var verificationToken: UUID?

    /// An interrupted verification, and why it stopped.
    struct PausedVerification: Equatable {
        let artifact: DiscoveredArtifact
        let request: ArtifactVerificationRequest
        let reason: String

        var rootPath: String { artifact.rootPath }
    }

    private let storage: any StorageManaging
    private var catalog: ModelCatalogSnapshot
    private let ledger: any ArtifactVerificationLedger
    private let checkpoints: (any ArtifactVerificationCheckpointStore)?
    private let backgroundWork: any VerificationBackgroundWork
    private let announcer: (any VerificationAnnouncing)?
    /// Whether the operator is looking at this app right now. A completion
    /// notification exists for the case where they are not; when they are, the
    /// row changing in front of them has already said it.
    private let applicationIsFrontmost: @MainActor () -> Bool
    private var verificationExpired = false
    private let verificationOperation: InstalledArtifactVerificationOperation?
    private let removalOperation: InstalledArtifactRemovalOperation?
    private let scanOperation: InstalledModelScanOperation?
    private let monitor = VolumeChangeMonitor()
    /// `StorageManaging` owns the bookmark ledger, so Observation cannot see a
    /// bookmark appear or disappear by itself. Reading `locationKeys` also
    /// reads this token; replacing it makes an accepted picker grant visible in
    /// the same UI transaction instead of waiting for the asynchronous scan.
    private var locationObservationToken = UUID()
    /// Any scan request that arrived while a scan was in flight. Catalog,
    /// picker, mount and removal changes all use the same coalescing bit: the
    /// current result may already be stale, so exactly one follow-up pass is
    /// required.
    private var rescanPending = false

    init(
        storage: any StorageManaging,
        catalog: ModelCatalogSnapshot = ModelCatalogSnapshot(
            generatedAt: .distantPast, origin: .bundled, models: []),
        ledger: any ArtifactVerificationLedger = UserDefaultsVerificationLedger(),
        checkpoints: (any ArtifactVerificationCheckpointStore)? = nil,
        backgroundWork: (any VerificationBackgroundWork)? = nil,
        announcer: (any VerificationAnnouncing)? = nil,
        applicationIsFrontmost: @escaping @MainActor () -> Bool = { true },
        initialReport: DiscoveryReport = .empty,
        verificationOperation: InstalledArtifactVerificationOperation? = nil,
        removalOperation: InstalledArtifactRemovalOperation? = nil,
        scanOperation: InstalledModelScanOperation? = nil
    ) {
        self.storage = storage
        self.catalog = catalog
        self.ledger = ledger
        self.checkpoints = checkpoints
        self.backgroundWork = backgroundWork ?? UnsuspendedVerificationWork()
        self.announcer = announcer
        self.applicationIsFrontmost = applicationIsFrontmost
        self.report = initialReport
        self.verificationOperation = verificationOperation
        self.removalOperation = removalOperation
        self.scanOperation = scanOperation
    }

    // MARK: - Locations

    /// The prefix every location this screen manages is filed under.
    ///
    /// The kit's two well-known keys name one slot each — where downloads land,
    /// where the artifact root is. An operator has as many drives as they have
    /// drives, so each gets its own key under this prefix and the ledger is the
    /// list.
    static let locationKeyPrefix = "artifact-location."

    var locationKeys: [StorageKey] {
        _ = locationObservationToken
        return storage.knownLocations()
            .filter { $0.rawValue.hasPrefix(Self.locationKeyPrefix) }
            .sorted { $0.rawValue < $1.rawValue }
    }

    /// Remember a folder the operator picked and scan it straight away.
    ///
    /// The bookmark is minted before the scan, so a location that cannot be
    /// remembered fails here rather than after showing a list of artifacts that
    /// will be unreachable on the next launch.
    func addLocation(_ url: URL) {
        for key in locationKeys where storage.bookmark(key)?.recordedPath == url.path {
            do {
                let replay = try storage.resolve(key)
                replay.release()
                lastAddOutcome = .alreadyRegistered(path: url.path)
                return
            } catch {
                // A dead bookmark must not permanently block a fresh picker
                // grant for the same removable-drive path.
                storage.forget(key)
                locationObservationToken = UUID()
            }
        }
        let key = StorageKey(Self.locationKeyPrefix + UUID().uuidString)
        do {
            // Bookmarked inside an access bracket. On iOS, creating a bookmark
            // for a security-scoped URL outside one fails — the picker hands
            // back a URL whose grant has to be opened before it can be
            // recorded. On macOS the bracket is harmless, and one code path is
            // better than two that differ by a platform check.
            let scope = try storage.scope(for: url)
            defer { scope.release() }
            try storage.remember(url, as: key)
            // Prove the durable capability while the picker grant is still
            // live. A bookmark that serializes but cannot resolve is not a
            // successful add; discovering that only after relaunch produced
            // the old "sandbox refused" surprise.
            let replay = try storage.resolve(key)
            replay.release()
            locationObservationToken = UUID()
            locationError = nil
            lastAddOutcome = .added(path: url.path)
            // Straight away, and not on the next `onAppear`: the folder was
            // just picked, so this is the moment the operator is owed an answer
            // about what is in it.
            refresh()
        } catch {
            storage.forget(key)
            locationObservationToken = UUID()
            locationError = "\(error)"
            lastAddOutcome = .failed("\(error)")
        }
    }

    /// The picker was dismissed without a folder. Recorded rather than ignored:
    /// a silent no-op is indistinguishable from a broken button.
    func noteAddCancelled() { lastAddOutcome = .cancelled }

    func noteAddFailed(_ reason: String) {
        locationError = reason
        lastAddOutcome = .failed(reason)
    }

    func clearAddOutcome() { lastAddOutcome = nil }

    func removeLocation(_ key: StorageKey) {
        storage.forget(key)
        locationObservationToken = UUID()
        locationError = nil
        lastAddOutcome = nil
        refresh()
    }

    func recordedPath(of key: StorageKey) -> String? {
        storage.bookmark(key)?.recordedPath
    }

    // MARK: - Scanning

    /// Apply the same snapshot the Models page is rendering, then rescan so an
    /// index is identified against that snapshot and no other. Catalog refresh
    /// is the only caller that changes it after initialisation.
    func updateCatalog(_ snapshot: ModelCatalogSnapshot, refresh shouldRefresh: Bool = true) {
        catalog = snapshot
        guard shouldRefresh else { return }
        refresh()
    }

    /// Rescan every registered location.
    ///
    /// Re-entrant calls coalesce into one follow-up pass. They cannot simply be
    /// dropped: adding/removing a location or receiving a mount event while an
    /// older scan is blocked changes the answer that scan is about to publish.
    func refresh() {
        guard !isScanning else {
            rescanPending = true
            return
        }
        isScanning = true
        let storage = self.storage
        let catalog = self.catalog
        let ledger = self.ledger
        let scanOperation = self.scanOperation
        Task {
            let fresh: DiscoveryReport
            if let scanOperation {
                fresh = await scanOperation(storage, catalog, ledger)
            } else {
                let locator = ArtifactLocator(catalog: catalog, verificationLedger: ledger)
                fresh = await Task.detached(priority: .utility) {
                    locator.scanRegisteredLocations(of: storage)
                }.value
            }
            self.report = fresh
            self.scanCount += 1
            self.isScanning = false
            self.onReportChanged?(fresh)
            if self.rescanPending {
                self.rescanPending = false
                self.refresh()
            }
        }
    }

    /// Start listening for drives arriving and leaving.
    ///
    /// On iOS this observes nothing, by ``VolumeChangeMonitor``'s own admission,
    /// and the app refreshes on foreground instead.
    func startWatchingVolumes() {
        monitor.start { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        }
    }

    func stopWatchingVolumes() { monitor.stop() }

    // MARK: - The questions the screens ask

    func installations(of model: ModelID) -> [DiscoveredArtifact] {
        report.installations(of: model)
    }

    func isInstalled(_ model: ModelID) -> Bool { report.isInstalled(model) }

    /// Present **and** on a volume that is mounted right now. Deliberately not
    /// the same question as installed: a 1.56 TB artifact on an unplugged drive
    /// is installed and cannot be run.
    func isRunnableHere(_ model: ModelID) -> Bool { report.isRunnableHere(model) }

    var installedCount: Int {
        Set(report.locations.flatMap { $0.artifacts }.compactMap(\.model)).count
    }

    /// The mounted artifact a run would read, largest first — the most complete
    /// copy of the model this machine can currently reach.
    func preferredArtifact(for model: ModelID) -> DiscoveredArtifact? {
        report.locations
            .filter(\.isMounted)
            .flatMap { $0.artifacts(for: model) }
            .max { $0.bytesOnDisk < $1.bytesOnDisk }
    }

    /// The mounted artifact a run may actually open. A directory the scan
    /// already knows is short cannot become a request merely because a mock
    /// download controller once reached `ready`.
    func runnableArtifact(
        for model: ModelID,
        matching predicate: (DiscoveredArtifact) -> Bool = { _ in true }
    ) -> DiscoveredArtifact? {
        report.locations
            .filter(\.isMounted)
            .flatMap { $0.artifacts(for: model) }
            .filter { $0.isComplete != false && $0.verification == .fullyVerified }
            .filter(predicate)
            .max { lhs, rhs in
                let lhsVerification = verificationRank(lhs.verification)
                let rhsVerification = verificationRank(rhs.verification)
                if lhsVerification != rhsVerification {
                    return lhsVerification < rhsVerification
                }
                return lhs.bytesOnDisk < rhs.bytesOnDisk
            }
    }

    /// Build a run reference only when the consumer receives the same rooted
    /// filesystem objects the full verification pass authorized.
    func runnableArtifactReference(
        for model: ModelID,
        matching predicate: (DiscoveredArtifact) -> Bool = { _ in true }
    ) throws -> ArtifactReference? {
        guard let artifact = runnableArtifact(for: model, matching: predicate) else { return nil }
        let location = try owningLocation(of: artifact)
        guard location.storageKey != nil else {
            // DEBUG preview fixtures have no durable storage authority and the
            // preview runner reads no model bytes.
            return ArtifactReference(root: artifact.url)
        }
        guard let key = location.storageKey else {
            throw InstalledArtifactAccessError.missingStorageAuthority(path: location.rootPath)
        }
        let scope = try storage.resolve(key)
        do {
            let remapped = try Self.remappedArtifactAndAuthority(
                artifact, from: location, to: scope.url)
            guard let descriptor = catalog.descriptor(model),
                let repository = descriptor.source.repo,
                let record = ledger.record(
                    matching: ArtifactVerificationLookup(
                        rootPath: remapped.artifact.rootPath,
                        model: model,
                        repository: repository,
                        index: remapped.artifact.index)),
                record.state == .fullyVerified,
                let evidence = record.evidence
            else {
                throw InstalledArtifactAccessError.verificationEvidenceUnavailable(
                    path: artifact.rootPath)
            }
            let runtimeAuthority = try ArtifactRuntimeAuthority(
                root: remapped.authority, evidence: evidence)
            return ArtifactReference(
                root: remapped.artifact.url, scope: scope,
                runtimeAuthority: runtimeAuthority)
        } catch {
            scope.release()
            throw error
        }
    }

    /// Locate the exact scan row that vended this value. An artifact URL on its
    /// own is not storage authority: callers may only replay a path which the
    /// current report still attributes to one registered location.
    private func owningLocation(of artifact: DiscoveredArtifact) throws -> LocationScan {
        let artifactLocation = URL(fileURLWithPath: artifact.locationPath, isDirectory: true)
            .standardizedFileURL.path
        guard
            let location = report.locations.first(where: { scan in
                URL(fileURLWithPath: scan.rootPath, isDirectory: true)
                    .standardizedFileURL.path == artifactLocation
                    && scan.artifacts.contains(where: { candidate in
                        candidate == artifact
                    })
            })
        else {
            throw InstalledArtifactAccessError.notInCurrentScan(path: artifact.rootPath)
        }
        guard location.isMounted else {
            throw InstalledArtifactAccessError.locationUnavailable(path: location.rootPath)
        }
        return location
    }

    /// Replay only the path-component suffix below the scanned location onto a
    /// freshly resolved bookmark URL. String-prefix containment is not enough:
    /// `/Volumes/Model` is a prefix of `/Volumes/Models`, and a stale value may
    /// contain `..`. Standardising and comparing complete components makes
    /// both cases refusals rather than alternate verification targets.
    private static func relativeComponents(
        of artifact: DiscoveredArtifact, from location: LocationScan
    ) throws -> [String] {
        let scannedLocation = URL(
            fileURLWithPath: location.rootPath, isDirectory: true
        ).standardizedFileURL
        let scannedArtifact = URL(
            fileURLWithPath: artifact.rootPath, isDirectory: true
        ).standardizedFileURL
        let locationComponents = scannedLocation.pathComponents
        let artifactComponents = scannedArtifact.pathComponents
        guard artifactComponents.count >= locationComponents.count,
            Array(artifactComponents.prefix(locationComponents.count)) == locationComponents
        else {
            throw InstalledArtifactAccessError.outsideRegisteredLocation(
                artifact: artifact.rootPath, location: location.rootPath)
        }

        let relativeComponents = Array(artifactComponents.dropFirst(locationComponents.count))
        guard relativeComponents.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw InstalledArtifactAccessError.outsideRegisteredLocation(
                artifact: artifact.rootPath, location: location.rootPath)
        }
        return relativeComponents
    }

    private static func remappedArtifactAndAuthority(
        _ artifact: DiscoveredArtifact, from location: LocationScan, to resolvedLocation: URL
    ) throws -> (artifact: DiscoveredArtifact, authority: ArtifactVerificationRoot) {
        let relativeComponents = try relativeComponents(of: artifact, from: location)
        let locationURL = resolvedLocation.standardizedFileURL
        let authority: ArtifactVerificationRoot
        do {
            authority = try ArtifactVerificationRoot.open(
                relativeComponents: relativeComponents, beneath: locationURL)
        } catch {
            throw InstalledArtifactAccessError.outsideRegisteredLocation(
                artifact: artifact.rootPath, location: location.rootPath)
        }
        let remapped = DiscoveredArtifact(
            rootPath: authority.rootURL.path,
            locationPath: locationURL.path,
            index: artifact.index,
            model: artifact.model,
            displayName: artifact.displayName,
            bytesOnDisk: artifact.bytesOnDisk,
            fileCount: artifact.fileCount,
            expectedFileCount: artifact.expectedFileCount,
            expectedBytes: artifact.expectedBytes,
            verification: artifact.verification,
            verifiedAt: artifact.verifiedAt,
            scannedAt: artifact.scannedAt,
            metadataOverflowed: artifact.metadataOverflowed)
        return (remapped, authority)
    }

    private func verificationRank(_ verification: ArtifactVerification) -> Int {
        switch verification {
        case .unverified: return 0
        case .spotChecked: return 1
        case .fullyVerified: return 2
        }
    }

    // MARK: - Verification, only ever by request

    /// The current scan's non-I/O answer about whether this exact row carries
    /// enough storage authority to offer a destructive action. The operation
    /// still resolves and revalidates the bookmark after confirmation.
    func removalAccessRefusal(for artifact: DiscoveredArtifact) -> String? {
        do {
            let location = try owningLocation(of: artifact)
            guard location.storageKey != nil else {
                throw InstalledArtifactAccessError.missingStorageAuthority(
                    path: location.rootPath)
            }
            return nil
        } catch {
            return String(describing: error)
        }
    }

    /// Permanently remove the exact scanned artifact through its freshly
    /// resolved registered-location capability.
    ///
    /// The package operation performs the destructive traversal off the main
    /// actor and is itself descriptor-bound. This layer owns the security scope
    /// for the entire operation and clears positive verification evidence only
    /// after the filesystem reports success.
    func remove(_ artifact: DiscoveredArtifact) async throws -> ArtifactRemovalReceipt {
        guard verificationTask == nil else {
            throw InstalledArtifactRemovalError.verificationInProgress(
                path: verificationInFlight ?? artifact.rootPath)
        }
        guard removalInFlight == nil else {
            throw InstalledArtifactRemovalError.removalInProgress(
                path: removalInFlight ?? artifact.rootPath)
        }

        removalInFlight = artifact.rootPath
        removalOutcome[artifact.rootPath] = nil
        defer { removalInFlight = nil }

        do {
            let location = try owningLocation(of: artifact)
            guard let key = location.storageKey else {
                throw InstalledArtifactAccessError.missingStorageAuthority(
                    path: location.rootPath)
            }
            let scope = try storage.resolve(key)
            defer { scope.release() }
            let remapped = try Self.remappedArtifactAndAuthority(
                artifact, from: location, to: scope.url)
            let operation: InstalledArtifactRemovalOperation = removalOperation ?? { authority in
                try await Task.detached(priority: .utility) {
                    try authority.removeRecursively()
                }.value
            }
            let receipt = try await operation(remapped.authority)
            ledger.forget(rootPath: remapped.artifact.rootPath)
            checkpoints?.discard(rootPath: remapped.artifact.rootPath, kind: nil)
            if remapped.artifact.rootPath != artifact.rootPath {
                ledger.forget(rootPath: artifact.rootPath)
                checkpoints?.discard(rootPath: artifact.rootPath, kind: nil)
            }
            if pausedVerification?.rootPath == artifact.rootPath {
                pausedVerification = nil
            }
            removalOutcome[artifact.rootPath] =
                "Removed \(MRFormat.measuredBytes(receipt.removedRegularFileBytes)) from this device."
            refresh()
            return receipt
        } catch {
            removalOutcome[artifact.rootPath] = String(describing: error)
            throw error
        }
    }

    /// Digest an artifact against the digests its repository publishes.
    ///
    /// Called from a button and from nothing else. A full pass on K3 reads
    /// 1.56 TB, so the spot check is the default the UI offers and the full pass
    /// says its own size out loud before it starts.
    func verify(
        _ artifact: DiscoveredArtifact, _ request: ArtifactVerificationRequest,
        resumption: ArtifactVerificationResumption = .resumeIfPossible
    ) async {
        guard verificationTask == nil, removalInFlight == nil else { return }
        let token = UUID()
        verificationToken = token
        verificationInFlight = artifact.rootPath
        verificationProgress = nil
        verificationExpired = false
        if pausedVerification?.rootPath == artifact.rootPath { pausedVerification = nil }
        // Asked here and nowhere else. A permission prompt at launch is a
        // question about something the operator has not done yet; this is the
        // moment they start work long enough to walk away from.
        announcer?.prepare()
        // The system panel names the model and the size, because that panel may
        // be the only place this work is visible for the next half hour.
        backgroundWork.begin(
            title: "Verifying \(artifact.displayName)",
            subtitle: Self.backgroundSubtitle(for: request, artifact: artifact),
            totalBytes: artifact.bytesOnDisk
        ) { [weak self] in
            self?.expireVerification()
        }
        let operation: InstalledArtifactVerificationOperation
        if let verificationOperation {
            operation = verificationOperation
        } else {
            let verifier = ArtifactVerifier(
                tree: HuggingFaceTreeClient(transport: URLSessionTransport()),
                catalog: catalog, ledger: ledger, checkpoints: checkpoints)
            operation = { artifact, request, authority, resumption, progress in
                try await verifier.verify(
                    artifact, request, rootedAt: authority, resumption: resumption,
                    onProgress: progress)
            }
        }

        let task = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                // The scan may have refreshed since this pass was requested —
                // it always has, when a paused pass resumes on foreground.
                // Re-read the row the current report holds for this exact
                // location and root before asking which location owns it.
                let target = self.currentRow(matching: artifact) ?? artifact
                let location = try self.owningLocation(of: target)
                guard let key = location.storageKey else {
                    throw InstalledArtifactAccessError.missingStorageAuthority(
                        path: location.rootPath)
                }
                let scope = try self.storage.resolve(key)
                let remapped: (
                    artifact: DiscoveredArtifact, authority: ArtifactVerificationRoot
                )
                do {
                    remapped = try Self.remappedArtifactAndAuthority(
                        target, from: location, to: scope.url)
                } catch {
                    scope.release()
                    throw error
                }
                defer { scope.release() }

                let verificationReport = try await operation(
                    remapped.artifact, request, remapped.authority, resumption
                ) { [weak self] progress in
                    Task { @MainActor in
                        guard let self, self.verificationToken == token else { return }
                        self.verificationProgress = progress
                        // The system's own progress UI is driven from exactly
                        // the same samples as the row on screen.
                        self.backgroundWork.update(
                            subtitle: MRFormat.percent(progress.fraction, digits: 0)
                                + " of \(MRFormat.measuredBytes(progress.bytesToCheck))",
                            completedBytes: progress.bytesChecked,
                            totalBytes: progress.bytesToCheck)
                    }
                }
                guard self.verificationToken == token else { return }

                let resultingState: ArtifactVerification
                if verificationReport.isComplete {
                    switch request {
                    case .spotCheck: resultingState = .spotChecked
                    case .full: resultingState = .fullyVerified
                    }
                } else {
                    resultingState = .unverified
                }

                var checkedAt = verificationReport.checkedAt
                // The production verifier owns the durable ledger write. Do
                // not report success merely because it returned a complete
                // digest report: prove the exact record can already be read
                // back while the security scope and rooted fd are still live.
                if self.verificationOperation == nil {
                    guard let model = remapped.artifact.model,
                        let repository = self.catalog.descriptor(model)?.source.repo,
                        let record = self.ledger.record(
                            matching: ArtifactVerificationLookup(
                                rootPath: remapped.artifact.rootPath,
                                model: model,
                                repository: repository,
                                index: remapped.artifact.index)),
                        record.state == resultingState
                    else {
                        throw InstalledArtifactAccessError.verificationEvidenceNotRecorded(
                            path: remapped.artifact.rootPath)
                    }
                    checkedAt = record.checkedAt
                }

                self.report = Self.applyingVerification(
                    resultingState, checkedAt: checkedAt,
                    to: target, in: self.report)
                self.onReportChanged?(self.report)
                let sentence = ArtifactVerifier.sentence(
                    verificationReport, request: request,
                    selected: verificationReport.ok.count
                        + verificationReport.missing.count)
                self.verificationOutcome[artifact.rootPath] = sentence
                self.verificationSucceeded = verificationReport.isComplete
                self.pendingAnnouncement = VerificationAnnouncement(
                    rootPath: artifact.rootPath,
                    title: verificationReport.isComplete
                        ? "\(target.displayName) verified"
                        : "\(target.displayName) did not verify",
                    body: verificationReport.isComplete
                        ? Self.verifiedSummary(
                            artifact: target, progress: self.verificationProgress,
                            files: verificationReport.ok.count)
                        : sentence,
                    isFailure: !verificationReport.isComplete)
            } catch is CancellationError {
                guard self.verificationToken == token else { return }
                self.finishInterrupted(artifact, request)
            } catch ArtifactVerificationError.cancelled {
                guard self.verificationToken == token else { return }
                self.finishInterrupted(artifact, request)
            } catch {
                guard self.verificationToken == token else { return }
                // A refusal names its own reason — that is what these error
                // descriptions are for — so the notification carries it rather
                // than the word "failed".
                self.verificationOutcome[artifact.rootPath] = "\(error)"
                self.pendingAnnouncement = VerificationAnnouncement(
                    rootPath: artifact.rootPath,
                    title: "\(artifact.displayName) was not verified",
                    body: "\(error)", isFailure: true)
            }
        }
        verificationTask = task
        await task.value
        guard verificationToken == token else { return }
        verificationTask = nil
        verificationToken = nil
        verificationInFlight = nil
        verificationProgress = nil
        backgroundWork.end(success: verificationSucceeded)
        verificationSucceeded = false
        let announcement = pendingAnnouncement
        pendingAnnouncement = nil
        // Only when the operator is elsewhere. A banner over the screen that
        // already shows the answer is noise, and a paused pass says nothing at
        // all: it has not ended, and it will resume by itself.
        if let announcement, !applicationIsFrontmost() {
            announcer?.announce(announcement)
        }
        refresh()
    }

    /// The pass stopped before it finished. Whether the operator cancelled it
    /// or the system took its background time back decides whether this is an
    /// ending or a pause — and a pause keeps its checkpoint, so resuming reads
    /// only what is left.
    private func finishInterrupted(
        _ artifact: DiscoveredArtifact, _ request: ArtifactVerificationRequest
    ) {
        guard verificationExpired else {
            verificationOutcome[artifact.rootPath] = "Cancelled"
            return
        }
        verificationExpired = false
        // One sentence, shown on the row and held with the paused pass, because
        // two spellings of one state is one spelling too many.
        let reason = "Paused where it stopped; open Minirun to continue from there."
        pausedVerification = PausedVerification(
            artifact: artifact, request: request, reason: reason)
        verificationOutcome[artifact.rootPath] = reason
    }

    /// The system is taking its background time back. Stop on the next chunk
    /// boundary; every file already digested stays in the kit's checkpoint.
    private func expireVerification() {
        guard verificationTask != nil else { return }
        verificationExpired = true
        verificationTask?.cancel()
    }

    /// The app is no longer frontmost.
    ///
    /// A verification the system agreed to continue is left running — that is
    /// the whole point of the continued-processing task. Without that
    /// agreement the app is about to be suspended anyway, so the pass is
    /// stopped deliberately, on a file boundary, with its checkpoint intact,
    /// rather than frozen mid-read with a progress bar that never moves again.
    func applicationWillLeaveForeground() {
        guard verificationTask != nil else { return }
        guard !backgroundWork.isRunning else { return }
        expireVerification()
    }

    /// The app is frontmost again. Resume what the system paused.
    ///
    /// Automatic, and deliberately not a prompt: the operator explicitly
    /// started a pass that reads a terabyte, and nothing about walking away
    /// from the phone withdraws that. The resumed pass re-reads only what the
    /// checkpoint does not already cover.
    func resumePausedVerification() {
        guard let paused = pausedVerification, verificationTask == nil,
            removalInFlight == nil, currentRow(matching: paused.artifact) != nil
        else { return }
        pausedVerification = nil
        Task { await self.verify(paused.artifact, paused.request) }
    }

    /// The row the current scan holds for this artifact, which is the value
    /// `owningLocation` will accept.
    ///
    /// A held artifact value goes stale for a reason that has nothing to do
    /// with the disk: a rescan gives the same directory a new `scannedAt`, and
    /// therefore a new value identity, while every fact about it is unchanged.
    /// Returning to the foreground rescans and resumes in the same breath, so
    /// without this the resumed pass raced the scan it triggered and lost.
    /// Identity here is the registered location plus the root path; the
    /// bookmark walk and the verifier's own dev/inode evidence remain what
    /// decides whether those bytes may be trusted.
    private func currentRow(matching artifact: DiscoveredArtifact) -> DiscoveredArtifact? {
        report.locations
            .flatMap(\.artifacts)
            .first {
                $0.rootPath == artifact.rootPath
                    && $0.locationPath == artifact.locationPath
            }
    }

    private static func backgroundSubtitle(
        for request: ArtifactVerificationRequest, artifact: DiscoveredArtifact
    ) -> String {
        switch request {
        case .full:
            return "Digesting \(MRFormat.measuredBytes(artifact.bytesOnDisk))"
        case .spotCheck:
            return "Digesting a sample of the largest files"
        }
    }

    /// "1.56 TB, 470 files" — the two numbers that say which pass this was.
    private static func verifiedSummary(
        artifact: DiscoveredArtifact, progress: ArtifactVerificationProgress?,
        files: Int
    ) -> String {
        let bytes = progress?.bytesToCheck ?? artifact.bytesOnDisk
        return "\(MRFormat.measuredBytes(bytes)), \(MRFormat.grouped(files)) files"
    }

    /// Publish the verifier's confirmed result immediately. The asynchronous
    /// follow-up scan remains the filesystem truth and may supersede this
    /// value, but the UI no longer falls back to a stale pre-verification row
    /// during that scan. Production callers reach this only after a durable
    /// ledger round-trip has succeeded.
    static func applyingVerification(
        _ state: ArtifactVerification, checkedAt: Date,
        to artifact: DiscoveredArtifact, in report: DiscoveryReport
    ) -> DiscoveryReport {
        let locations = report.locations.map { location in
            let artifacts = location.artifacts.map { candidate in
                // A long verification may overlap a metadata-only rescan. Its
                // scannedAt value can therefore differ even though it still
                // describes the same artifact row. Identity here is the
                // registered location + root + parsed index, not the stale
                // value object's full Equatable payload.
                guard location.rootPath == artifact.locationPath,
                    candidate.rootPath == artifact.rootPath,
                    candidate.model == artifact.model,
                    candidate.index == artifact.index
                else { return candidate }
                return DiscoveredArtifact(
                    rootPath: candidate.rootPath,
                    locationPath: candidate.locationPath,
                    index: candidate.index,
                    model: candidate.model,
                    displayName: candidate.displayName,
                    bytesOnDisk: candidate.bytesOnDisk,
                    fileCount: candidate.fileCount,
                    expectedFileCount: candidate.expectedFileCount,
                    expectedBytes: candidate.expectedBytes,
                    verification: state,
                    verifiedAt: state == .unverified ? nil : checkedAt,
                    scannedAt: candidate.scannedAt,
                    metadataOverflowed: candidate.metadataOverflowed)
            }
            return LocationScan(
                rootPath: location.rootPath,
                displayName: location.displayName,
                isMounted: location.isMounted,
                storageKey: location.storageKey,
                artifacts: artifacts,
                unreadable: location.unreadable,
                scannedAt: location.scannedAt)
        }
        return DiscoveryReport(locations: locations, scannedAt: report.scannedAt)
    }

    func cancelVerification() {
        verificationTask?.cancel()
    }
}
