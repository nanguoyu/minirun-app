import MinirunKit
import SwiftUI
import UniformTypeIdentifiers

#if canImport(AppKit)
    import AppKit
#endif

// =============================================================================
// MARK: - Why this file exists
// =============================================================================
//
// A hands-on review found the app reporting every model as not installed while
// a 1.56 TB artifact sat on an NVMe drive attached to the same machine. The
// cause was not the scanner: the app's container held **no storage-location
// bookmark at all**, so the scanner had never been given a folder to walk. The
// only way to register one was a small "Add folder…" button on a sub-screen of
// a sub-screen of Settings, and if the picker failed or was dismissed nothing
// on screen changed.
//
// So this file makes registering a location the loudest thing on any screen
// that has none and says out loud how the attempt ended — including
// "cancelled". Mounted drives are listed separately on Storage; they are not
// grants and therefore do not belong inside this control.

/// Opens the folder picker, pre-targeted where the caller asks.
///
/// `NSOpenPanel` rather than SwiftUI's `.fileImporter` on the Mac, because the
/// suggestion buttons need `directoryURL`: clicking "K3NVME" should open the
/// panel *at* K3NVME. The sandbox still requires the user to confirm, and that
/// is the point — the confirmation is what mints the grant.
enum FolderChooser {
    enum Outcome: Equatable {
        case chose(URL)
        case cancelled
    }

    #if os(macOS)
        @MainActor
        static func run(
            startingAt directory: URL?,
            prompt: String = "Use this folder",
            message: String =
                "Choose a folder Minirun may use for local model data. Nothing is written or "
                + "moved when the folder is added."
        ) -> Outcome {
            let panel = NSOpenPanel()
            panel.canChooseDirectories = true
            panel.canChooseFiles = false
            panel.allowsMultipleSelection = false
            panel.canCreateDirectories = true
            panel.prompt = prompt
            panel.message = message
            if let directory { panel.directoryURL = directory }
            guard panel.runModal() == .OK, let url = panel.url else { return .cancelled }
            return .chose(url)
        }
    #endif
}

enum StorageLocationActionPolicy {
    static let reviewDisabledMessage = "Sample review — storage access disabled"

    @discardableResult
    static func perform(
        isReviewPreview: Bool, action: () -> Void
    ) -> Bool {
        guard !isReviewPreview else { return false }
        action()
        return true
    }
}

/// The one control that registers a storage location, wherever it appears.
///
/// There is exactly one product action: open the system folder picker. A
/// mounted volume is only hardware Minirun can see; the folder the user confirms
/// is the durable read grant Minirun may scan.
struct AddLocationButton: View {
    @Environment(AppModel.self) private var model
    var prominent = false

    #if os(iOS)
        @State private var isPicking = false
    #endif

    @ViewBuilder var body: some View {
        if model.visualReviewNotice != nil {
            Label(
                StorageLocationActionPolicy.reviewDisabledMessage,
                systemImage: "externaldrive.badge.xmark")
                .font(MRType.caption)
                .foregroundStyle(MRColor.secondary)
        } else {
            #if os(macOS)
            Button("Add a folder…") { choose(startingAt: nil) }
                .buttonStyle(.borderedProminent)
                .tint(MRColor.tierPinned)
                .controlSize(prominent ? .large : .small)
                .frame(minHeight: 44)
            #else
            Button("Add a folder…") { isPicking = true }
                .buttonStyle(.borderedProminent)
                .tint(MRColor.tierPinned)
                .frame(minHeight: 44)
                .fileImporter(isPresented: $isPicking, allowedContentTypes: [.folder]) { result in
                    switch result {
                    case .success(let url): model.installed.addLocation(url)
                    case .failure(let error):
                        model.installed.noteAddFailed((error as NSError).localizedDescription)
                    }
                }
            #endif
        }
    }

    #if os(macOS)
        private func choose(startingAt directory: URL?) {
            StorageLocationActionPolicy.perform(
                isReviewPreview: model.visualReviewNotice != nil
            ) {
                switch FolderChooser.run(startingAt: directory) {
                case .chose(let url): model.installed.addLocation(url)
                case .cancelled: model.installed.noteAddCancelled()
                }
            }
        }
    #endif
}

/// What a screen shows when no folder has ever been registered.
///
/// Deliberately the largest thing on the screen. The state it describes is the
/// difference between "you own no models" and "nobody has told this app where
/// to look", and the app got that wrong in the only way that matters.
struct NoStorageLocationCard: View {
    @Environment(AppModel.self) private var model

    @ViewBuilder
    var body: some View {
        #if os(iOS)
            phoneEmptyState
        #else
            desktopCautionCard
        #endif
    }

    /// First-run guidance is not a warning. The amber-bordered caution card is
    /// the same treatment this app uses for a refused budget and a failed
    /// verification; wearing it before anything has happened told a new user
    /// that they had already done something wrong. The phone gets the neutral
    /// centred empty state the Chats tab uses.
    #if os(iOS)
        private var phoneEmptyState: some View {
            VStack(spacing: MRSpace.s4) {
                Image(systemName: "folder.badge.plus")
                    .font(.system(size: 38, weight: .regular))
                    .foregroundStyle(MRColor.secondary)
                    .accessibilityHidden(true)
                Text("No folders added")
                    .font(MRType.title)
                    .foregroundStyle(MRColor.primary)
                Text(
                    "Choose a folder for Minirun to scan. It keeps read access, so the same "
                        + "folder is there after a relaunch."
                )
                .font(MRType.body)
                .foregroundStyle(MRColor.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)
                .fixedSize(horizontal: false, vertical: true)
                AddLocationButton(prominent: true)
                AddLocationOutcomeLine()
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .accessibilityElement(children: .contain)
            .accessibilityLabel("no folders have been added")
        }
    #endif

    private var desktopCautionCard: some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            HStack(spacing: MRSpace.s2) {
                Image(systemName: "folder.badge.plus")
                    .font(.system(size: 22))
                    .foregroundStyle(MRColor.caution)
                Text("No folders added")
                    .font(MRType.title)
                    .foregroundStyle(MRColor.primary)
            }
            Text(
                "Choose a folder for Minirun to scan. The app keeps read access so the same "
                    + "folder is available after relaunch; you can remove that access here."
            )
            .font(MRType.body)
            .foregroundStyle(MRColor.secondary)
            .fixedSize(horizontal: false, vertical: true)
            AddLocationButton(prominent: true)
            AddLocationOutcomeLine()
        }
        .padding(MRSpace.s4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .mrCard(MRColor.caution.opacity(0.10), stroke: MRColor.caution.opacity(0.45))
        .accessibilityElement(children: .contain)
        .accessibilityLabel("no folders have been added")
    }
}

/// How the last attempt ended, in one line. Nothing at all when there has not
/// been one.
struct AddLocationOutcomeLine: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if let outcome = model.installed.lastAddOutcome {
            VStack(alignment: .leading, spacing: MRSpace.s2) {
                HStack(spacing: MRSpace.s2) {
                    Image(
                        systemName: outcome.isFailure
                            ? "exclamationmark.triangle" : "checkmark.circle"
                    )
                    .imageScale(.small)
                    Text(sentence(outcome))
                        .font(MRType.caption)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
                .foregroundStyle(outcome.isFailure ? MRColor.caution : MRColor.ok)
                if !outcome.isFailure {
                    HStack(spacing: MRSpace.s2) {
                        if model.installed.isScanning {
                            ProgressView().controlSize(.small)
                            Text("Scanning…")
                                .font(MRType.micro)
                                .foregroundStyle(MRColor.secondary)
                        }
                        NavigationLink("View models") { ModelCatalogView() }
                            .controlSize(.small)
                            .frame(minHeight: 44)
                    }
                }
            }
            .accessibilityElement(children: .contain)
        }
    }

    private func sentence(_ outcome: InstalledModels.AddOutcome) -> String {
        switch outcome {
        case .added(let path):
            return "Added \(path)."
        case .alreadyRegistered(let path):
            return "\(path) was already registered."
        case .cancelled:
            return "No folder was chosen, so nothing was registered."
        case .failed(let reason):
            return "That folder could not be registered: \(reason)"
        }
    }
}

/// The one action an "unavailable model" notice offers.
///
/// A notice used to end in two links, Storage and Models, and left the reader
/// to work out which one would fix their situation. It cannot be both: with no
/// registered folder the only fix is to add one, and once a folder is
/// registered the only fix is to put a model in it. So the notice asks the
/// installed-state exactly that question and shows one control.
struct UnavailableModelAction: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if model.installed.hasLocation {
            NavigationLink("Find models") { ModelCatalogView() }
                .buttonStyle(.bordered)
                .controlSize(.small)
        } else {
            AddLocationButton()
        }
    }
}
