import Foundation
import MinirunKit
import Observation

/// The drive assessments this machine has taken, and the one that is running.
///
/// The assessment is the answer to a question the product could not previously
/// ask: *is this drive fast enough, on the right link, with room for the model?*
/// The record's own history is why it exists — the same enclosure delivered
/// 0.95 GB/s behind a hub and 2.84 GB/s on a USB4 tunnel, a 3.4× difference on
/// the token that nobody could see until it was measured.
///
/// Two rules this type keeps:
///
/// 1. **It never assesses by itself.** A screen appearing is not consent to read
///    three gigabytes off somebody's drive, exactly as it is not consent to
///    digest 1.56 TB. Every run is a button press.
/// 2. **The reading never happens on the main actor.** Two gigabytes of `pread`
///    on the main thread is a frozen window, and a progress bar that cannot
///    redraw is worse than no progress bar.
@Observable
@MainActor
final class VolumeAssessments {

    /// The last report for each registered location, keyed by its path.
    private(set) var reports: [String: VolumeAssessmentReport] = [:]
    /// The location being assessed right now, if any.
    private(set) var inFlight: String?
    private(set) var progress: ReadSpotProgress?
    /// Non-nil when an assessment could not even be started — a location whose
    /// grant would not resolve, say. Shown, not logged.
    private(set) var startError: String?

    private let storage: any StorageManaging
    private var catalog: ModelCatalogSnapshot
    private let ledger: any AssessmentLedger
    private let assessor: VolumeAssessor
    private var task: Task<VolumeAssessmentReport, Never>?

    init(
        storage: any StorageManaging,
        catalog: ModelCatalogSnapshot = ModelCatalogSnapshot(
            generatedAt: .distantPast, origin: .bundled, models: []),
        ledger: any AssessmentLedger = UserDefaultsAssessmentLedger(),
        assessor: VolumeAssessor = VolumeAssessor()
    ) {
        self.storage = storage
        self.catalog = catalog
        self.ledger = ledger
        self.assessor = assessor
    }

    /// What the button says it will read, so the size of the request is visible
    /// before it is authorised.
    var plannedBytes: UInt64 { assessor.spotTest.configuration.plannedBytes }

    var catalogModelIDs: [ModelID] { catalog.models.map(\.id) }

    func updateCatalog(_ snapshot: ModelCatalogSnapshot) {
        catalog = snapshot
    }

    func report(forLocationPath path: String) -> VolumeAssessmentReport? {
        if let cached = reports[path] { return cached }
        guard let stored = ledger.report(forLocationPath: path) else { return nil }
        reports[path] = stored
        return stored
    }

    /// The most recent report covering a path — the artifact's own location,
    /// or a location it sits under. Used by the model screen, which knows where
    /// an artifact is and not which folder was registered.
    func report(covering path: String) -> VolumeAssessmentReport? {
        let candidates = reports.values.filter {
            path == $0.locationPath || path.hasPrefix($0.locationPath + "/")
        }
        return candidates.max { $0.assessedAt < $1.assessedAt }
    }

    /// Load whatever was persisted for the locations that exist now.
    func load(locationPaths: [String]) {
        for path in locationPaths where reports[path] == nil {
            if let stored = ledger.report(forLocationPath: path) { reports[path] = stored }
        }
    }

    var isRunning: Bool { inFlight != nil }

    /// Assess one registered location. One at a time: two concurrent spot tests
    /// on the same bus would each measure the other.
    func assess(
        key: StorageKey, scan: LocationScan?, workloads: [ModelID: TokenWorkload]
    ) async {
        guard inFlight == nil else { return }
        let scope: StorageScope
        do {
            scope = try storage.resolve(key)
        } catch {
            startError = "\(error)"
            return
        }
        defer { scope.release() }

        let path = scope.url.path
        startError = nil
        inFlight = path
        progress = ReadSpotProgress(
            phase: .choosingTargets, bytesRead: 0, plannedBytes: plannedBytes)
        defer {
            inFlight = nil
            progress = nil
        }

        let assessor = self.assessor
        let catalog = self.catalog
        let url = scope.url
        let sink = ProgressSink()

        let task = Task.detached(priority: .userInitiated) { () -> VolumeAssessmentReport in
            assessor.assess(
                location: url, scan: scan, catalog: catalog, workloads: workloads,
                isCancelled: { Task.isCancelled },
                progress: { sink.offer($0) })
        }
        self.task = task
        defer { self.task = nil }

        // The reading runs off this actor; this loop is the only thing that
        // touches the published progress, and it samples rather than being
        // called, so a fast drive cannot flood the main actor with updates.
        let ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(100))
                guard let self, let latest = sink.latest else { continue }
                self.progress = latest
            }
        }
        defer { ticker.cancel() }

        let report = await task.value
        // A cancelled run leaves a report whose spot test refused with
        // "stopped before it finished". That is a true sentence and a useless
        // one to persist, so it is shown and not remembered.
        guard !task.isCancelled else { return }
        reports[path] = report
        ledger.record(report)
    }

    /// Stop the run in flight. The spot test checks between reads, so this lands
    /// within one chunk rather than at the end.
    func cancel() {
        task?.cancel()
    }

    func forget(locationPath: String) {
        reports.removeValue(forKey: locationPath)
        ledger.forget(locationPath: locationPath)
    }
}

/// A mailbox the reading thread drops progress into and the main actor picks it
/// out of.
///
/// Deliberately lossy: only the newest value survives, because a progress bar
/// that redraws every 4 MiB is a progress bar that costs more than the read.
private final class ProgressSink: @unchecked Sendable {
    private let lock = NSLock()
    private var value: ReadSpotProgress?

    func offer(_ progress: ReadSpotProgress) {
        lock.lock()
        value = progress
        lock.unlock()
    }

    var latest: ReadSpotProgress? {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}
