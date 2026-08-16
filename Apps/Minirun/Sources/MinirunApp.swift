import SwiftUI

#if os(macOS)
    import AppKit

    /// SwiftUI collapses the entire title-bar host when its automatic toolbar
    /// items are removed, including the standard window controls. Minirun owns
    /// the header rows, but it remains a normal Mac window: this bridge restores
    /// the traffic lights over full-size content and never reads or mutates the
    /// outer frame.
    struct MinirunWindowChrome: NSViewRepresentable {
        final class HostView: NSView {
            private var updateObserver: NSObjectProtocol?

            override func viewDidMoveToWindow() {
                super.viewDidMoveToWindow()
                if let updateObserver {
                    NotificationCenter.default.removeObserver(updateObserver)
                }
                updateObserver = window.map { window in
                    NotificationCenter.default.addObserver(
                        forName: NSWindow.didUpdateNotification,
                        object: window,
                        queue: .main
                    ) { [weak self] _ in
                        self?.configureWindow()
                    }
                }
                configureWindow()
            }

            deinit {
                if let updateObserver {
                    NotificationCenter.default.removeObserver(updateObserver)
                }
            }

            func configureWindow() {
                guard let window else { return }
                window.titleVisibility = .hidden
                window.titlebarAppearsTransparent = true
                window.titlebarSeparatorStyle = .none
                if window.toolbar != nil { window.toolbar = nil }
                if !window.titlebarAccessoryViewControllers.isEmpty {
                    window.titlebarAccessoryViewControllers = []
                }
                window.styleMask.insert(.titled)
                window.styleMask.insert(.fullSizeContentView)
                for kind in [
                    NSWindow.ButtonType.closeButton,
                    .miniaturizeButton,
                    .zoomButton,
                ] {
                    window.standardWindowButton(kind)?.isHidden = false
                    window.standardWindowButton(kind)?.alphaValue = 1
                }
            }
        }

        func makeNSView(context: Context) -> HostView { HostView() }

        func updateNSView(_ view: HostView, context: Context) {
            DispatchQueue.main.async { view.configureWindow() }
        }
    }
#endif

/// Reads values only from this process's argument vector. Unlike an application
/// defaults key, a review switch cannot survive the process that received it.
enum MinirunLaunchArguments {
    static func value(for flag: String, in arguments: [String]) -> String? {
        guard let index = arguments.lastIndex(of: flag) else { return nil }
        let valueIndex = arguments.index(after: index)
        guard arguments.indices.contains(valueIndex) else { return nil }
        let value = arguments[valueIndex]
        guard !value.hasPrefix("-") else { return nil }
        return value
    }
}

/// The product-safe launch choice. A malformed review request is not equivalent
/// to an ordinary launch: treating it as `.live` would let a typo in a review
/// script open the operator's real conversations, defaults and bookmarks.
enum MinirunInitialModel: Equatable {
    case live
    case launchRefused
    #if DEBUG
        case visualReviewPreview
    #endif

    private static let reviewFlag = "-MinirunReviewPreview"

    static func resolve(
        arguments: [String],
        visualReviewAvailable: Bool? = nil
    ) -> MinirunInitialModel {
        let candidates = arguments.indices.dropFirst().filter {
            arguments[$0].hasPrefix(reviewFlag)
        }
        guard !candidates.isEmpty else { return .live }

        // One exact flag and one exact value is the entire accepted grammar.
        // Duplicates, a suffix typo, a missing value and case variants all fail
        // closed without echoing the potentially sensitive argument text.
        guard candidates.count == 1, let flagIndex = candidates.first,
            arguments[flagIndex] == reviewFlag
        else { return .launchRefused }
        let valueIndex = arguments.index(after: flagIndex)
        guard arguments.indices.contains(valueIndex), arguments[valueIndex] == "YES"
        else { return .launchRefused }

        #if DEBUG
            return (visualReviewAvailable ?? visualReviewAvailableInThisBuild)
                ? .visualReviewPreview : .launchRefused
        #else
            // The preview factory is not part of a Release binary. Even an
            // otherwise valid review request is therefore an unavailable
            // request, not permission to fall back to the live product.
            return .launchRefused
        #endif
    }

    @MainActor
    func makeModel(
        arguments: [String], userDefaults: UserDefaults = .standard
    ) -> AppModel {
        switch self {
        case .live:
            return AppModel.live(userDefaults: userDefaults)
        case .launchRefused:
            return AppModel.refusingUnsafeReviewLaunch(userDefaults: userDefaults)
        #if DEBUG
        case .visualReviewPreview:
            return AppModel.visualReviewPreview(
                arguments: arguments, userDefaults: userDefaults)
        #endif
        }
    }

    private static var visualReviewAvailableInThisBuild: Bool {
        #if DEBUG
            true
        #else
            false
        #endif
    }
}

enum MinirunProcessContext {
    /// A hosted XCTest launches the application executable before it runs the
    /// test bundle. That host must not open the operator's real bookmarks,
    /// catalog cache or verification ledger merely because unit tests need an
    /// NSApplication. The tests construct every AppModel they exercise.
    static func isUnitTestHost(environment: [String: String]) -> Bool {
        environment["XCTestConfigurationFilePath"] != nil
            || environment["XCTestBundlePath"] != nil
    }
}

#if os(macOS)
    /// Removes only the frame state written by the obsolete additive-panel
    /// implementation. SwiftUI restores a saved frame before consulting
    /// `defaultSize`, so retaining that state would preserve the 1,920-point
    /// windows produced by the old panel sizer after the implementation was
    /// fixed.
    ///
    /// This migration is versioned and runs once. Frames saved after it —
    /// including every deliberate user resize — are left alone on subsequent
    /// launches.
    enum MinirunWindowRestorationMigration {
        enum Result: Equatable {
            case notPermitted
            case alreadyCurrent
            case migrated(removedKeys: [String])
        }

        static let currentVersion = 1
        static let versionKey = "minirun.windowRestorationMigrationVersion"
        static let legacyPanelKeys = [
            "minirun.window.panelAttached",
            "minirun.window.widthWithoutPanel",
        ]

        static func applyIfNeeded(
            for launch: MinirunInitialModel,
            defaults: UserDefaults = .standard
        ) -> Result {
            guard launch == .live else { return .notPermitted }
            guard defaults.integer(forKey: versionKey) < currentVersion else {
                return .alreadyCurrent
            }

            let restorationKeys = defaults.dictionaryRepresentation().keys.filter {
                isObsoleteMainWindowRestorationKey($0)
            }
            let presentLegacyKeys = legacyPanelKeys.filter {
                defaults.object(forKey: $0) != nil
            }
            let keys = Array(Set(restorationKeys + presentLegacyKeys)).sorted()
            for key in keys { defaults.removeObject(forKey: key) }
            defaults.set(currentVersion, forKey: versionKey)
            return .migrated(removedKeys: keys)
        }

        static func isObsoleteMainWindowRestorationKey(_ key: String) -> Bool {
            let isFrame = key.hasPrefix("NSWindow Frame ")
                || key.hasPrefix("NSSplitView Subview Frames ")
            return isFrame && key.contains("MinirunApp.RootView")
                && key.contains("-AppWindow-")
        }
    }
#endif

/// Minirun: download a pre-converted artifact and run it on device with the
/// weights streaming from disk.
///
/// The product is one idea rendered as a screen — a **stated** memory budget,
/// with the tier split visible and live telemetry proving the run kept it. Four
/// invariants of the codebase are load-bearing here and not engineering trivia:
/// a budget is stated before a run and never emergent; invalid configuration is
/// refused with a named error and never clamped; where files and metadata
/// disagree, the files win; and every byte lands in exactly one bucket.
///
/// This target links the package's catalogue, storage and real K3 runtime. The
/// product registers K3 only when the current HF snapshot, complete local
/// verification and the artifact-published tokenizer identity agree; every
/// other model fails closed. Preview stand-ins are injected only by SwiftUI
/// previews and tests. See DESIGN.md.
@main
struct MinirunApp: App {

    @State private var model: AppModel
    #if os(macOS)
        /// The Mac ships outside the App Store, so the app is the only thing
        /// that can tell an operator their build is stale. Owned here, beside
        /// the model, because it is scoped to the process rather than to a
        /// window or a screen. Whether it actually runs is
        /// `SoftwareUpdatePolicy`'s decision, not this initialiser's.
        @State private var updater: SoftwareUpdater
    #endif

    init() {
        let arguments = ProcessInfo.processInfo.arguments
        let isUnitTestHost = MinirunProcessContext.isUnitTestHost(
            environment: ProcessInfo.processInfo.environment)
        let launch = isUnitTestHost
            ? MinirunInitialModel.launchRefused
            : MinirunInitialModel.resolve(arguments: arguments)
        #if os(macOS)
            // Before SwiftUI asks AppKit to restore the scene, and only for a
            // real product launch. Review/refusal launches never touch the
            // operator's defaults domain.
            _ = MinirunWindowRestorationMigration.applyIfNeeded(for: launch)
        #endif
        let initialModel = launch.makeModel(arguments: arguments)
        _model = State(initialValue: initialModel)
        #if os(macOS)
            _updater = State(initialValue: SoftwareUpdater.forLaunch(launch))
        #endif

        #if os(macOS) && DEBUG
            // Apply the review-only appearance override before AppKit creates
            // the window or its material-backed sidebars. Applying it later
            // repaints SwiftUI text without rebuilding those materials and can
            // produce a false dark-on-light (or light-on-dark) review capture.
            if let raw = MinirunLaunchArguments.value(
                for: "-MinirunAppearance", in: arguments)
            {
                NSApplication.shared.appearance = NSAppearance(
                    named: raw.lowercased() == "dark" ? .darkAqua : .aqua)
            }
        #endif
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(model)
                .tint(MRColor.tierPinned)
            #if os(macOS)
                .environment(updater)
                // A practical lower bound for sidebar + transcript. The
                // inspector divides this space when opened; no panel state is
                // allowed to mutate the outer window frame.
                .background(MinirunWindowChrome())
                .frame(
                    minWidth: ConversationWorkspaceLayout.minimumWindowWidth,
                    minHeight: ConversationWorkspaceLayout.minimumWindowHeight)
            #endif
        }
        #if os(macOS)
            .defaultSize(
                width: ConversationWorkspaceLayout.defaultWindowWidth,
                height: ConversationWorkspaceLayout.defaultWindowHeight)
            .windowStyle(.hiddenTitleBar)
            .commands { MinirunCommands(model: model, updater: updater) }
        #endif

        // There is deliberately no separate `Settings` scene: Settings is a
        // full surface in the detail pane — its own left nav over General,
        // Storage, Models and About — reached from the pinned sidebar row or
        // ⌘,. Two settings surfaces would be exactly the confusion this build
        // set out to remove.
    }
}
