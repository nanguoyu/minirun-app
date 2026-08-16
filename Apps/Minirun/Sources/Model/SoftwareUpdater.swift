import Foundation

/// Whether *this* launch of *this* build may run Sparkle.
///
/// Two independent facts have to hold, and they are separated here so both can
/// be tested without Sparkle, AppKit or a network:
///
/// 1. **The launch is the operator's own.** A review preview and a hosted-test
///    launch already refuse to touch the real conversation directory, bookmark
///    ledger and defaults domain. An updater is the same class of thing: it
///    writes scheduling state into the defaults domain and reaches a remote
///    feed. A capture script must not be able to do either.
/// 2. **The build is actually configured to verify an update.** Until the owner
///    generates the EdDSA key pair and pastes the public half into
///    `project.yml`, `SUPublicEDKey` holds a placeholder. Sparkle would refuse
///    that configuration anyway; starting the updater to watch it fail would
///    put a broken control in Settings and an error in the log on every launch
///    of every development build. Reporting "this build cannot verify an
///    update" is the honest state, and it is the one the panel shows.
///
/// This is the same rule the rest of the app follows: an unsupported case is
/// named and refused, never quietly half-performed.
enum SoftwareUpdatePolicy {
    /// The literal that ships in the repository, and the one value that is
    /// definitely not a key.
    static let publicKeyPlaceholder = "REPLACE_WITH_SPARKLE_PUBLIC_ED_KEY"

    /// Sparkle's `SUPublicEDKey` is an ed25519 public key: 32 raw bytes, which
    /// base64 is exactly 44 characters with one `=` of padding. Checking the
    /// shape rather than merely `!= placeholder` also catches a truncated
    /// paste, which is the realistic way this goes wrong by hand.
    static func isConfiguredPublicKey(_ raw: String?) -> Bool {
        guard let raw else { return false }
        let key = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard key != publicKeyPlaceholder, key.count == 44 else { return false }
        guard let decoded = Data(base64Encoded: key) else { return false }
        return decoded.count == 32
    }

    static func startsUpdater(
        for launch: MinirunInitialModel, publicKey: String?
    ) -> Bool {
        launch == .live && isConfiguredPublicKey(publicKey)
    }
}

#if os(macOS)
    import Sparkle

    /// The app's single Sparkle updater, and the only place `Sparkle` is
    /// imported.
    ///
    /// The Mac app is distributed outside the App Store — Developer ID signed,
    /// notarized, stapled, and served from the owner's own bucket — so nothing
    /// else would ever tell an operator their build is stale. Sparkle 2 is the
    /// standard answer for exactly that distribution, and it is the one that
    /// works from inside this app's sandbox: it installs through an XPC service
    /// bundled in its own framework rather than asking the app to write outside
    /// its container. See `project.yml` for the Info.plist keys and the two
    /// Mach-lookup entitlements that make that call possible.
    ///
    /// Consent is Sparkle's, not ours. `SUEnableAutomaticChecks` is deliberately
    /// unset, so the first automatic check happens only after Sparkle's standard
    /// permission prompt has been answered. Nothing here contacts the feed
    /// before that, and there is no analytics of any kind: Sparkle's optional
    /// system-profile parameters are left off, so a check is a request for one
    /// XML file and nothing else.
    ///
    /// A `SoftwareUpdater` always exists so the menu item and the Settings row
    /// are never conditionally compiled out of the screen; whether it is
    /// *running* is `SoftwareUpdatePolicy`'s decision, and `state` says which
    /// case a reader is looking at.
    @MainActor
    @Observable
    final class SoftwareUpdater {

        /// Why the controls look the way they do. A disabled toggle with no
        /// explanation is the failure this enum exists to prevent.
        enum State: Equatable {
            /// Sparkle is started and owns this app's update schedule.
            case running
            /// This build carries no usable `SUPublicEDKey`, so it could not
            /// verify an update even if one were published.
            case unsignedBuild
            /// A review-preview or hosted-test launch. The operator's real
            /// update state is deliberately untouched.
            case notThisLaunch
        }

        let state: State

        /// Mirrors Sparkle's own `canCheckForUpdates`, which is false while a
        /// check or install is already in flight. The menu item and the button
        /// follow it rather than guessing.
        private(set) var canCheckForUpdates: Bool = false

        /// Mirrors Sparkle's stored preference. Sparkle is the owner of this
        /// value — it lives in the app's defaults domain under Sparkle's own
        /// key — so this is a mirror kept current by KVO, never a second copy
        /// of the truth.
        private(set) var automaticallyChecksForUpdates: Bool = false

        private(set) var lastUpdateCheckDate: Date?

        @ObservationIgnored private let controller: SPUStandardUpdaterController?
        @ObservationIgnored private var observations: [NSKeyValueObservation] = []

        /// The launch-time factory. `bundle` is a parameter so a test can state
        /// the configuration it is asserting about instead of inheriting the
        /// host bundle's.
        static func forLaunch(
            _ launch: MinirunInitialModel, bundle: Bundle = .main
        ) -> SoftwareUpdater {
            let key = bundle.object(forInfoDictionaryKey: "SUPublicEDKey") as? String
            guard launch == .live else { return SoftwareUpdater(state: .notThisLaunch) }
            guard SoftwareUpdatePolicy.isConfiguredPublicKey(key) else {
                return SoftwareUpdater(state: .unsignedBuild)
            }
            return SoftwareUpdater(state: .running)
        }

        private init(state: State) {
            self.state = state
            guard state == .running else {
                controller = nil
                return
            }

            // `startingUpdater: true` is Sparkle's standard entry point: it
            // starts the scheduler, and on a configuration error it logs and
            // stays inert rather than throwing into the app's launch path.
            // Neither delegate is supplied — the standard user driver already
            // presents Sparkle's permission prompt, its update alert and its
            // installer, and a delegate here would only be a place to
            // accidentally add behaviour the operator did not ask for.
            let controller = SPUStandardUpdaterController(
                startingUpdater: true, updaterDelegate: nil, userDriverDelegate: nil)
            self.controller = controller

            let updater = controller.updater
            canCheckForUpdates = updater.canCheckForUpdates
            automaticallyChecksForUpdates = updater.automaticallyChecksForUpdates
            lastUpdateCheckDate = updater.lastUpdateCheckDate

            observations = [
                updater.observe(\.canCheckForUpdates, options: [.new]) {
                    [weak self] updater, _ in
                    let value = updater.canCheckForUpdates
                    Task { @MainActor in self?.canCheckForUpdates = value }
                },
                updater.observe(\.automaticallyChecksForUpdates, options: [.new]) {
                    [weak self] updater, _ in
                    let value = updater.automaticallyChecksForUpdates
                    Task { @MainActor in self?.automaticallyChecksForUpdates = value }
                },
                updater.observe(\.lastUpdateCheckDate, options: [.new]) {
                    [weak self] updater, _ in
                    let value = updater.lastUpdateCheckDate
                    Task { @MainActor in self?.lastUpdateCheckDate = value }
                },
            ]
        }

        /// The menu item and the button. Sparkle presents everything from here
        /// on: its own progress, its release notes, its install-and-relaunch
        /// prompt.
        func checkForUpdates() {
            guard let controller else { return }
            controller.checkForUpdates(nil)
        }

        /// Writes through to Sparkle, which persists it. The mirror above is
        /// updated by the KVO observation rather than here, so the value the
        /// toggle shows is always the value Sparkle actually holds.
        func setAutomaticallyChecksForUpdates(_ isOn: Bool) {
            controller?.updater.automaticallyChecksForUpdates = isOn
        }

        /// The one sentence the Settings row shows under its title.
        var explanation: String {
            switch state {
            case .running:
                return SoftwareUpdatePresentation.runningExplanation
            case .unsignedBuild:
                return SoftwareUpdatePresentation.unsignedBuildExplanation
            case .notThisLaunch:
                return SoftwareUpdatePresentation.reviewLaunchExplanation
            }
        }
    }
#endif

/// Sentences, kept out of the view so a test can name them.
enum SoftwareUpdatePresentation {
    static let panelTitle = "Software update"
    static let automaticTitle = "Automatically check for updates"
    static let checkNowTitle = "Check now"
    static let menuItemTitle = "Check for Updates…"

    static let runningExplanation =
        "Minirun asks downloads.minirun.dev once a day whether a newer signed build exists. "
        + "Nothing about this Mac or your conversations is sent."
    static let unsignedBuildExplanation =
        "This build carries no update-signing key, so it cannot verify a downloaded update. "
        + "Updates are disabled."
    static let reviewLaunchExplanation =
        "Updates are disabled in this launch mode so a review build cannot change the "
        + "installed app."

    /// "Never checked" is a fact; a blank trailing value is a bug the reader
    /// has to diagnose.
    static func lastCheck(_ date: Date?, now: Date = Date()) -> String {
        guard let date else { return "Never checked" }
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .full
        return "Checked \(formatter.localizedString(for: date, relativeTo: now))"
    }
}
