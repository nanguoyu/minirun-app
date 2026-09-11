import MinirunKit
import SwiftUI

/// Storage access, mounted volumes, and explicit drive assessment.
///
/// This page deliberately does not enumerate models or artifacts. A storage
/// location is a permission and a drive is a piece of hardware; model presence,
/// integrity, and provenance belong on the Models screen.
///
/// In the product-page language (DESIGN I.37) a folder is a row and not a card:
/// a drive glyph, the folder's name, its path in mono, and one dot with one
/// sentence — *Mounted · assessed 3 days ago*. What the drive measured is a
/// definition list with the digits aligned, and everything that acts on the
/// folder is a text link beside it. The one filled button on the page adds a
/// folder, because that is the one thing this page is for.
struct StorageSettingsView: View {
    @Environment(AppModel.self) private var model
    @State private var pendingLocationRemoval: StorageLocationRemovalRequest?

    var body: some View {
        page
        .frame(maxWidth: .infinity)
        .mrProductPage()
        .mrPhoneNavigationTitle("Storage")
        #if os(iOS)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    StorageRefreshButton(placement: .navigationBar)
                }
            }
        #endif
        .onAppear {
            model.refreshVolumes()
            model.installed.refresh()
            model.assessments.load(
                locationPaths: model.installed.locationKeys.compactMap {
                    model.installed.recordedPath(of: $0)
                })
        }
        .confirmationDialog(
            "Stop using this folder?",
            isPresented: locationRemovalConfirmationPresented,
            titleVisibility: .visible
        ) {
            if let request = pendingLocationRemoval {
                Button("Stop using folder", role: .destructive) {
                    pendingLocationRemoval = nil
                    model.installed.removeLocation(request.key)
                }
            }
            Button("Cancel", role: .cancel) { pendingLocationRemoval = nil }
        } message: {
            if let request = pendingLocationRemoval {
                Text(
                    "Models stored only in \(request.displayName) will no longer be available "
                        + "until you add this folder again. No files will be deleted."
                )
            }
        }
    }

    /// With nothing registered the first-run guidance IS the page, and on the
    /// phone it fills the screen instead of hanging off the top of a scroll
    /// view above an empty drive inventory.
    @ViewBuilder private var page: some View {
        #if os(iOS)
            if model.installed.hasLocation {
                scrollingPage
            } else {
                NoStorageLocationCard()
                    .padding(MRSpace.s4)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        #else
            scrollingPage
        #endif
    }

    private var scrollingPage: some View {
        ScrollView {
            StorageSettingsPage(pendingLocationRemoval: $pendingLocationRemoval)
        }
        .frame(maxWidth: .infinity)
    }

    private var locationRemovalConfirmationPresented: Binding<Bool> {
        Binding(
            get: { pendingLocationRemoval != nil },
            set: { if !$0 { pendingLocationRemoval = nil } })
    }
}

/// The Storage page, without the scroll view around it.
///
/// Split out for the same reason `ModelDetailPage` is: `ImageRenderer` draws
/// nothing at all for a macOS `ScrollView`, and this page has three states a
/// reviewer has to be able to look at — a folder on a plugged-in drive that has
/// been assessed, a folder whose drive is in a drawer, and no folders at all.
/// The screen owns the removal confirmation; the page only sets the binding,
/// which is what keeps it a pure function of the app model.
struct StorageSettingsPage: View {
    @Binding var pendingLocationRemoval: StorageLocationRemovalRequest?

    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // One or the other, never both: with nothing registered the card IS
            // the storage-locations section, and repeating its button in an
            // empty section underneath is the sort of doubling that made this
            // screen hard to read.
            if model.installed.hasLocation {
                storageLocations
            } else {
                NoStorageLocationCard()
            }
            // iOS never enumerates volumes — the code below says so — so the
            // phone showed "MOUNTED DRIVES 0" and a paragraph about drives it
            // can never list. The folder grants are the whole of what Storage
            // means there.
            #if os(macOS)
                volumes
            #endif
        }
        .padding(pagePadding)
        .frame(maxWidth: 860, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var pagePadding: CGFloat {
        #if os(macOS)
            MRSpace.s5
        #else
            MRSpace.s4
        #endif
    }

    // MARK: - Folders Minirun may access

    /// The persistent folder grants discovery may scan. What a scan finds is a
    /// model fact and is intentionally rendered on Models, not repeated here.
    private var storageLocations: some View {
        MRPageSection(title: "Folders Minirun can access", showsSeparator: false) {
            Text(StorageLocationsPresentation.explanation)
                .font(MRType.prose)
                .foregroundStyle(MRColor.tertiary)
                .fixedSize(horizontal: false, vertical: true)

            AddLocationOutcomeLine()

            if let locationError = model.installed.locationError {
                MRInlineNote(
                    message: locationError, title: "A folder could not be read")
            }

            ForEach(model.installed.locationKeys, id: \.rawValue) { key in
                MRHairline()
                locationRow(key)
            }

            HStack(spacing: MRSpace.s3) {
                AddLocationButton()
                if model.installed.isScanning {
                    HStack(spacing: MRSpace.s2) {
                        ProgressView().controlSize(.small)
                        Text("Scanning…")
                            .font(MRType.prose)
                            .foregroundStyle(MRColor.secondary)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(.top, MRSpace.s2)
        }
    }

    /// One folder: the drive it is on, what it is called, where it is, and the
    /// one sentence that says whether Minirun can read it right now.
    @ViewBuilder private func locationRow(_ key: StorageKey) -> some View {
        let path = model.installed.recordedPath(of: key)
        let scan = model.installed.report.locations.first { $0.storageKey == key }
        let report = path.flatMap { model.assessments.report(forLocationPath: $0) }
        let status = StorageFolderStatusPresentation.resolve(
            isMounted: scan?.isMounted ?? false,
            hasRecordedPath: path != nil,
            assessedAt: report?.assessedAt)
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            MRQuietRow {
                HStack(alignment: .top, spacing: MRSpace.s3) {
                    Image(
                        systemName: StorageFolderPresentation.systemImage(
                            isInternal: isInternal(path),
                            isMounted: scan?.isMounted ?? false)
                    )
                    .imageScale(.large)
                    .foregroundStyle(MRColor.tertiary)
                    .frame(width: 24)
                    .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: MRSpace.s1) {
                        Text(scan?.displayName ?? path ?? key.rawValue)
                            .font(MRType.prose.weight(.medium))
                            .foregroundStyle(MRColor.primary)
                            .fixedSize(horizontal: false, vertical: true)
                        MRPathLabel(path: path ?? "path not recorded")
                        MRStatusLine(tone: status.tone, sentence: status.sentence)
                    }
                }
            } trailing: {
                Button("Remove") {
                    pendingLocationRemoval = StorageLocationRemovalRequest(
                        key: key,
                        displayName: scan?.displayName ?? path ?? key.rawValue)
                }
                .mrTextLink(.quiet)
                .help("Stop scanning this folder. No files are deleted.")
            }

            // Only when there is something to draw. An empty measurement block
            // still costs this row a gap, and a folder whose drive is away then
            // reads as a row with a paragraph missing from it.
            if showsAssessment(path: path, scan: scan, report: report) {
                assessment(key, path: path, scan: scan, report: report)
            }
        }
        .padding(.vertical, MRSpace.s2)
    }

    /// Whether this folder has a measurement, a measurement in flight, or a
    /// control that could start one.
    private func showsAssessment(
        path: String?, scan: LocationScan?, report: VolumeAssessmentReport?
    ) -> Bool {
        if report != nil { return true }
        #if os(iOS)
            // The phone always says why it does not measure.
            return true
        #else
            return model.assessments.inFlight == path || scan?.isMounted == true
        #endif
    }

    /// Whether the volume holding `path` is the machine's own disk. Nil when no
    /// mounted volume claims that path — the glyph then says "external", which
    /// is the honest guess for a folder the app was handed.
    private func isInternal(_ path: String?) -> Bool? {
        guard let path else { return nil }
        return
            model.volumes
            .filter { path.hasPrefix($0.mountPath) }
            .max { $0.mountPath.count < $1.mountPath.count }?
            .isInternal
    }

    // MARK: - Drive performance

    /// The assessment for one location: the control that takes it, the progress
    /// while it runs, and the facts afterwards.
    ///
    /// The control is the only way an assessment ever happens, and it says how
    /// much it will read before it is pressed. Nothing on this screen writes to
    /// the volume.
    @ViewBuilder private func assessment(
        _ key: StorageKey, path: String?, scan: LocationScan?,
        report: VolumeAssessmentReport?
    ) -> some View {
        let assessments = model.assessments
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            #if os(iOS)
                // Read-only on the phone. Its port is the ceiling, and there is
                // no second port to move the drive to.
                if report == nil {
                    Text(
                        "Drive assessment is available on Mac. On iPhone, Minirun reports "
                            + "capacity and access without claiming an unmeasured read rate."
                    )
                    .font(MRType.prose)
                    .foregroundStyle(MRColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
                }
            #else
                if assessments.inFlight == path {
                    VStack(alignment: .leading, spacing: MRSpace.s2) {
                        MRProgressBar(fraction: assessments.progress?.fraction ?? 0)
                            .frame(maxWidth: 320)
                        HStack(spacing: MRSpace.s4) {
                            Text(assessments.progress.map(Self.progressLabel) ?? "starting…")
                                .font(MRType.prose)
                                .foregroundStyle(MRColor.secondary)
                            Button("Stop") { assessments.cancel() }
                                .mrTextLink(.quiet)
                        }
                    }
                } else if report == nil, scan?.isMounted == true {
                    // Only for a drive that is here. The status line above
                    // already says what a folder on an unplugged drive needs,
                    // and a greyed-out control under it would be a second,
                    // wordless answer to the same question.
                    Button("Measure this drive") {
                        Task { await model.assess(key) }
                    }
                    .mrTextLink()
                    .disabled(assessments.isRunning)
                    .help(Self.assessmentHelp(plannedBytes: assessments.plannedBytes))
                }
                if let startError = assessments.startError, assessments.inFlight == nil {
                    MRInlineNote(
                        message: startError, title: "That measurement did not start")
                }
            #endif

            if let report, let path {
                AssessmentFacts(
                    report: report,
                    // Both controls are absent on the read-only surface: a
                    // report the phone cannot take is a report it has no
                    // business discarding either.
                    onReassess: Self.canAssess
                        ? { Task { await model.assess(key) } } : nil,
                    onForget: Self.canAssess
                        ? { assessments.forget(locationPath: path) } : nil)
            }
        }
    }

    /// Whether this platform takes assessments at all. False on iOS, where the
    /// report is shown and never taken.
    private static var canAssess: Bool {
        #if os(iOS)
            return false
        #else
            return true
        #endif
    }

    static func assessmentHelp(plannedBytes: UInt64) -> String {
        "Read about " + MRFormat.bytesDecimal(plannedBytes)
            + " from files already on this volume and report its sequential and scattered "
            + "read rates. Nothing is written."
    }

    private static func progressLabel(_ progress: ReadSpotProgress) -> String {
        switch progress.phase {
        case .choosingTargets: return "choosing files to read"
        case .sequential:
            return "sequential · \(MRFormat.bytesDecimal(progress.bytesRead))"
        case .scattered:
            return "scattered · \(MRFormat.bytesDecimal(progress.bytesRead))"
        }
    }

    // MARK: - Mounted drives

    private var volumes: some View {
        MRPageSection(title: "Mounted drives") {
            Text(
                "These drives are visible to the system. Minirun can read only the folders "
                    + "you add above."
            )
            .font(MRType.prose)
            .foregroundStyle(MRColor.tertiary)
            .fixedSize(horizontal: false, vertical: true)
            if let volumeError = model.volumeError {
                MRInlineNote(message: volumeError, title: "The drive list is incomplete")
            }
            ForEach(model.volumes) { volume in
                MRHairline()
                mountedDriveRow(volume)
            }
        }
    }

    /// One mounted drive, as a row of this page.
    ///
    /// Deliberately not `VolumeRow`: that component carries a destination's
    /// capacity verdict and its named refusal, which is a question the download
    /// sheet asks and this page does not. Here a drive is a name, a path, what
    /// it is, and how much of it is free.
    private func mountedDriveRow(_ volume: VolumeDescriptor) -> some View {
        MRQuietRow {
            HStack(alignment: .top, spacing: MRSpace.s3) {
                Image(
                    systemName: StorageVolumePresentation.systemImage(
                        isInternal: volume.isInternal, isSelected: false)
                )
                .imageScale(.large)
                .foregroundStyle(MRColor.tertiary)
                .frame(width: 24)
                .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: MRSpace.s1) {
                    Text(volume.name ?? volume.mountPath)
                        .font(MRType.prose.weight(.medium))
                        .foregroundStyle(MRColor.primary)
                    MRPathLabel(path: volume.mountPath, systemImage: "externaldrive")
                    let traits = StorageFolderPresentation.traits(volume)
                    if !traits.isEmpty {
                        Text(traits.joined(separator: " · "))
                            .font(MRType.prose)
                            .foregroundStyle(MRColor.tertiary)
                    }
                    if let refusal = volume.writeRefusal {
                        Text(refusal)
                            .font(MRType.prose)
                            .foregroundStyle(MRColor.caution)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        } trailing: {
            VStack(alignment: .trailing, spacing: 1) {
                Text(StorageCapacityPresentation.freeSpace(volume.space.dependableBytes))
                    .font(MRType.figure)
                    .foregroundStyle(
                        volume.space.dependableBytes == nil
                            ? MRColor.caution : MRColor.primary)
                if let optimistic = volume.space.optimisticBytes,
                    let dependable = volume.space.dependableBytes, optimistic > dependable
                {
                    Text(StorageCapacityPresentation.reclaimableTotal(optimistic))
                        .font(MRType.prose)
                        .foregroundStyle(MRColor.tertiary)
                }
            }
        }
        .padding(.vertical, MRSpace.s1)
    }
}

/// The sentences and glyphs the Storage page composes, as values.
///
/// A folder's line is the whole of what this page claims about it — whether the
/// drive is plugged in, and whether anybody has measured how fast it reads —
/// and it is assembled from two independent sources, so it is resolved here and
/// checked in the suite rather than built inside a `body`.
struct StorageFolderStatus: Equatable {
    let tone: MRPageStatusTone
    let sentence: String
}

enum StorageFolderStatusPresentation {
    static func resolve(
        isMounted: Bool, hasRecordedPath: Bool, assessedAt: Date?, now: Date = Date()
    ) -> StorageFolderStatus {
        guard hasRecordedPath else {
            return StorageFolderStatus(
                tone: .attention,
                sentence: "This grant records no path. Remove it and add the folder again.")
        }
        guard isMounted else {
            return StorageFolderStatus(
                tone: .attention,
                sentence: "Not connected. Plug the drive in, or remove the folder.")
        }
        guard let assessedAt else {
            return StorageFolderStatus(
                tone: .ready, sentence: "Mounted · read rate not measured")
        }
        return StorageFolderStatus(
            tone: .ready,
            sentence: "Mounted · assessed \(MRFormat.relative(assessedAt, now: now))")
    }
}

enum StorageFolderPresentation {
    /// A drive glyph, and a crossed-out one when the drive is away — the row's
    /// status sentence says the same thing in words, so the glyph is never the
    /// only carrier.
    static func systemImage(isInternal: Bool?, isMounted: Bool) -> String {
        guard isMounted else { return "externaldrive.badge.xmark" }
        return isInternal == true ? "internaldrive" : "externaldrive"
    }

    static func traits(_ volume: VolumeDescriptor) -> [String] {
        var result: [String] = []
        if volume.isInternal == true { result.append("Internal") }
        if volume.space.isRemovable { result.append("Removable") }
        if volume.isEjectable == true { result.append("Ejectable") }
        if volume.space.isReadOnly { result.append("Read-only") }
        return result
    }
}

enum StorageLocationsPresentation {
    static let explanation =
        "Minirun scans only the folders you add here. A folder keeps its read access across "
        + "a relaunch and a drive replug; remove it whenever Minirun should stop using it."
}

struct StorageLocationRemovalRequest {
    let key: StorageKey
    let displayName: String
}

/// One visible action refreshes both storage facts shown on this page: the
/// mounted-volume inventory and the contents of every authorized folder.
/// macOS places it in the product-owned Storage header; iOS uses the native
/// navigation bar.
struct StorageRefreshButton: View {
    /// Storage's Rescan is in the phone's navigation bar and in the Mac's own
    /// column header, and those two surfaces draw a control differently. See
    /// `MRControlPlacement`.
    var placement: MRControlPlacement = .page

    @Environment(AppModel.self) private var model

    var body: some View {
        Button {
            model.refreshVolumes()
            model.installed.refresh()
        } label: {
            if model.installed.isScanning {
                HStack(spacing: MRSpace.s1) {
                    ProgressView().controlSize(.small)
                    Text("Scanning…")
                }
            } else {
                Text("Rescan")
            }
        }
        .mrOutlineAction(placement)
        .help("Recheck mounted drives and authorized folders")
        .accessibilityLabel("rescan storage")
        .disabled(model.installed.isScanning)
    }
}
