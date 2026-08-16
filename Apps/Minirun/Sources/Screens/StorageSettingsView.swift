import MinirunKit
import SwiftUI

/// Storage access, mounted volumes, and explicit drive assessment.
///
/// This page deliberately does not enumerate models or artifacts. A storage
/// location is a permission and a drive is a piece of hardware; model presence,
/// integrity, and provenance belong on the Models screen.
struct StorageSettingsView: View {
    @Environment(AppModel.self) private var model
    @State private var pendingLocationRemoval: StorageLocationRemovalRequest?

    var body: some View {
        page
        .frame(maxWidth: .infinity)
        .mrWorkspaceSurface()
        .mrPhoneNavigationTitle("Storage")
        #if os(iOS)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    StorageRefreshButton()
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
                    .padding(pagePadding)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        #else
            scrollingPage
        #endif
    }

    private var scrollingPage: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MRSpace.s5) {
                // One or the other, never both: with nothing registered the
                // card IS the storage-locations section, and repeating its
                // button in an empty panel underneath is the sort of doubling
                // that made this screen hard to read.
                if model.installed.hasLocation {
                    storageLocations
                } else {
                    NoStorageLocationCard()
                }
                // iOS never enumerates volumes — the code below says so — so
                // the phone showed "MOUNTED DRIVES 0" and a paragraph about
                // drives it can never list. The folder grants are the whole of
                // what Storage means there.
                #if os(macOS)
                    volumes
                #endif
            }
            .padding(pagePadding)
            .frame(maxWidth: 800, alignment: .leading)
        }
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
        Panel(
            title: "Folders Minirun can access",
            trailing: AnyView(
                HStack(spacing: MRSpace.s2) {
                    if model.installed.isScanning { ProgressView().controlSize(.small) }
                    AddLocationButton()
                    HelpTip(
                        "Adding a folder grants Minirun persistent read access. The grant "
                            + "survives a relaunch and a drive replug. Remove it here whenever "
                            + "Minirun should stop using that folder.")
                })
        ) {
            AddLocationOutcomeLine()

            if let locationError = model.installed.locationError {
                Text(locationError).font(MRType.smallProse).foregroundStyle(MRColor.refuse)
            }

            ForEach(model.installed.locationKeys, id: \.rawValue) { key in
                locationCard(key)
            }
        }
    }

    @ViewBuilder private func locationCard(_ key: StorageKey) -> some View {
        let path = model.installed.recordedPath(of: key)
        let scan = model.installed.report.locations.first { $0.storageKey == key }
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .firstTextBaseline, spacing: MRSpace.s3) {
                    locationIdentity(key: key, path: path, scan: scan)
                    Spacer(minLength: MRSpace.s3)
                    locationControls(key: key, path: path, scan: scan)
                }
                VStack(alignment: .leading, spacing: MRSpace.s2) {
                    locationIdentity(key: key, path: path, scan: scan)
                    locationControls(key: key, path: path, scan: scan)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            assessment(key, path: path, scan: scan)

        }
        .padding(MRSpace.s3)
        .mrCard(MRColor.raised)
    }

    private func locationIdentity(
        key: StorageKey, path: String?, scan: LocationScan?
    ) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(scan?.displayName ?? path ?? key.rawValue)
                .font(MRType.headline)
                .foregroundStyle(MRColor.primary)
            Text(path ?? "path not recorded")
                .font(MRType.micro)
                .foregroundStyle(MRColor.tertiary)
                .textSelection(.enabled)
                .lineLimit(2)
                .truncationMode(.middle)
        }
    }

    private func locationControls(
        key: StorageKey, path: String?, scan: LocationScan?
    ) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: MRSpace.s2) {
                locationStatus(scan)
                removeLocationButton(key: key, path: path, scan: scan)
            }
            VStack(alignment: .leading, spacing: MRSpace.s1) {
                locationStatus(scan)
                removeLocationButton(key: key, path: path, scan: scan)
            }
        }
    }

    private func locationStatus(_ scan: LocationScan?) -> some View {
        StatusChip(
            text: (scan?.isMounted ?? false) ? "mounted" : "not mounted",
            tone: (scan?.isMounted ?? false) ? .ok : .caution)
    }

    private func removeLocationButton(
        key: StorageKey, path: String?, scan: LocationScan?
    ) -> some View {
        Button("Remove") {
            pendingLocationRemoval = StorageLocationRemovalRequest(
                key: key,
                displayName: scan?.displayName ?? path ?? key.rawValue)
        }
        .controlSize(.small)
        .frame(minHeight: 44)
    }

    private var locationRemovalConfirmationPresented: Binding<Bool> {
        Binding(
            get: { pendingLocationRemoval != nil },
            set: { if !$0 { pendingLocationRemoval = nil } })
    }

    // MARK: - Drive performance

    /// The assessment for one location: the button that takes it, the progress
    /// while it runs, and the card afterwards.
    ///
    /// The button is the only way an assessment ever happens, and it says how
    /// much it will read before it is pressed. Nothing on this screen writes to
    /// the volume.
    @ViewBuilder private func assessment(
        _ key: StorageKey, path: String?, scan: LocationScan?
    ) -> some View {
        let assessments = model.assessments
        let report = path.flatMap { assessments.report(forLocationPath: $0) }
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            #if os(iOS)
                // Read-only on the phone. Its port is the ceiling, and there is
                // no second port to move the drive to.
                if report == nil {
                    Text(
                        "Drive assessment is available on Mac. On iPhone, Minirun reports "
                            + "capacity and access without claiming an unmeasured read rate."
                    )
                    // Prose, so the system's caption face rather than the
                    // telemetry mono this app reserves for numbers.
                    .font(.caption)
                    .foregroundStyle(MRColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
                }
            #else
                HStack(spacing: MRSpace.s2) {
                    if assessments.inFlight == path {
                        ProgressView(value: assessments.progress?.fraction ?? 0)
                            .frame(maxWidth: 160)
                        Text(assessments.progress.map(Self.progressLabel) ?? "starting…")
                            .font(MRType.micro)
                            .foregroundStyle(MRColor.tertiary)
                        Button("Stop") { assessments.cancel() }
                            .controlSize(.small)
                    } else {
                        Button(report == nil ? "Assess" : "Re-assess") {
                            Task { await model.assess(key) }
                        }
                        .controlSize(.small)
                        .disabled(assessments.isRunning || !(scan?.isMounted ?? false))
                        .help(
                            "Read about "
                                + MRFormat.bytesDecimal(assessments.plannedBytes)
                                + " from files already on this volume and report its sequential "
                                + "and scattered read rates. Nothing is written.")
                    }
                    Spacer()
                }
                if let startError = assessments.startError, assessments.inFlight == nil {
                    Text(startError).font(MRType.micro).foregroundStyle(MRColor.refuse)
                }
            #endif

            if let report, let path {
                AssessmentCard(
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

    private static func progressLabel(_ progress: ReadSpotProgress) -> String {
        switch progress.phase {
        case .choosingTargets: return "choosing files to read"
        case .sequential:
            return "sequential · \(MRFormat.bytesDecimal(progress.bytesRead))"
        case .scattered:
            return "scattered · \(MRFormat.bytesDecimal(progress.bytesRead))"
        }
    }

    private var volumes: some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            SectionHeader(title: "Mounted drives", trailing: "\(model.volumes.count)")
            Text(
                "These drives are visible to the system. Minirun can read only folders "
                    + "you add above."
            )
            .font(MRType.caption)
            .foregroundStyle(MRColor.secondary)
            .fixedSize(horizontal: false, vertical: true)
            if let volumeError = model.volumeError {
                Text(volumeError).font(MRType.micro).foregroundStyle(MRColor.refuse)
            }
            ForEach(model.volumes) { volume in
                VolumeRow(volume: volume, requiredBytes: 0, headroomBytes: 0)
            }
        }
    }
}

private struct StorageLocationRemovalRequest {
    let key: StorageKey
    let displayName: String
}

/// One visible action refreshes both storage facts shown on this page: the
/// mounted-volume inventory and the contents of every authorized folder.
/// macOS places it in the product-owned Storage header; iOS uses the native
/// navigation bar.
struct StorageRefreshButton: View {
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
        .help("Recheck mounted drives and authorized folders")
        .accessibilityLabel("rescan storage")
        .disabled(model.installed.isScanning)
    }
}
