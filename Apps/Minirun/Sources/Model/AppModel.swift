import Foundation
import MinirunKit
import MinirunRunners
import Observation
import StorageCore

/// Read-only machine facts plus a closed bookmark surface for ephemeral app
/// states. It can
/// answer capacity questions without reading or writing the product's bookmark
/// defaults, and every operation that would mint a grant fails explicitly.
private struct EphemeralPreviewStorage: StorageManaging {
    private let readOnly = StorageManager(bookmarkLedger: InMemoryBookmarkLedger())

    func volumes() throws -> [VolumeDescriptor] { [] }
    func describe(_ url: URL) -> StorageInfo { readOnly.describe(url) }
    func freeSpace(at url: URL) -> FreeSpace { readOnly.freeSpace(at: url) }
    func canHold(bytes: UInt64, at url: URL, headroomBytes: UInt64) -> SpaceVerdict {
        readOnly.canHold(bytes: bytes, at: url, headroomBytes: headroomBytes)
    }
    func scope(for url: URL) throws -> StorageScope {
        throw StorageError.accessRefused(path: url.path)
    }
    func remember(_ url: URL, as key: StorageKey) throws -> StorageBookmark {
        throw StorageError.accessRefused(path: url.path)
    }
    func resolve(_ key: StorageKey) throws -> StorageScope {
        throw StorageError.noBookmark(key)
    }
    func bookmark(_ key: StorageKey) -> StorageBookmark? { nil }
    func forget(_ key: StorageKey) {}
    func knownLocations() -> [StorageKey] { [] }
    func bookmarkData(for url: URL) throws -> Data {
        throw StorageError.bookmarkCreationFailed(
            path: url.path, reason: "ephemeral storage is non-persistent")
    }
}

private struct DownloadServiceUnavailableError: Error, CustomStringConvertible, Sendable {
    let reason: String
    var description: String { "downloads are unavailable: \(reason)" }
}

/// A fail-closed backend used only when the product cannot establish its
/// durable job directory. It never substitutes an in-memory or temporary job:
/// every attempted operation returns the same visible setup failure.
private final class UnavailableDownloadManager: DownloadManaging, @unchecked Sendable {
    private let error: DownloadServiceUnavailableError

    init(reason: String) { error = DownloadServiceUnavailableError(reason: reason) }

    func plan(for model: ModelDescriptor) async throws -> DownloadPlan { throw error }
    func preflight(
        _ plan: DownloadPlan, into destination: DownloadDestination, options: DownloadOptions
    ) async throws -> DownloadPreflight { throw error }
    func start(
        _ plan: DownloadPlan, into destination: DownloadDestination, options: DownloadOptions
    ) async throws -> DownloadJobID { throw error }
    func events(for job: DownloadJobID) -> AsyncStream<DownloadEvent> {
        AsyncStream { $0.finish() }
    }
    func pause(_ job: DownloadJobID) async throws { throw error }
    func resume(_ job: DownloadJobID, scope: StorageScope?) async throws { throw error }
    func cancel(_ job: DownloadJobID, keepPartialFiles: Bool) async throws { throw error }
    func verify(_ job: DownloadJobID, depth: VerificationDepth) async throws
        -> VerificationReport
    { throw error }
    func repair(_ job: DownloadJobID, from report: VerificationReport) async throws
        -> DownloadJobID
    { throw error }
    func jobs() async -> [DownloadJobSummary] { [] }
    func restorePersistedJobs() async throws -> DownloadRestoreReport { throw error }
}

/// The App's download dependency is explicit at every construction site.
/// Release uses `persistent`; previews and tests ask for `preview` by name.
struct AppDownloadServices {
    let manager: any DownloadManaging
    let stateStore: (any DownloadStateStore)?
    let restoresPersistedJobs: Bool
    let setupError: String?

    #if DEBUG
        static func preview(entries: [CatalogEntry]) -> AppDownloadServices {
            AppDownloadServices(
                manager: MockDownloadManager(entries: entries), stateStore: nil,
                restoresPersistedJobs: false, setupError: nil)
        }
    #endif

    static func persistent(
        storage: any StorageManaging,
        applicationSupportRoot: URL? = nil,
        fileManager: FileManager = .default
    ) throws -> AppDownloadServices {
        let root: URL
        if let applicationSupportRoot {
            root = applicationSupportRoot
        } else {
            let base = try fileManager.url(
                for: .applicationSupportDirectory, in: .userDomainMask,
                appropriateFor: nil, create: true)
            root = base.appendingPathComponent(
                Bundle.main.bundleIdentifier ?? "wang.wangdongdong.minirun", isDirectory: true)
        }
        let directory = root.appendingPathComponent("DownloadJobs", isDirectory: true)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let store = FileDownloadStateStore(directory: directory)
        return AppDownloadServices(
            manager: DownloadManager(storage: storage, stateStore: store),
            stateStore: store, restoresPersistedJobs: true, setupError: nil)
    }

    static func unavailable(_ reason: String) -> AppDownloadServices {
        AppDownloadServices(
            manager: UnavailableDownloadManager(reason: reason), stateStore: nil,
            restoresPersistedJobs: false, setupError: reason)
    }
}

struct DownloadRecoveryIssue: Equatable, Identifiable {
    var id: DownloadJobID { job }
    let job: DownloadJobID
    let model: ModelID?
    let destinationPath: String
    let reason: String
}

/// One obsolete product-wide destination bookmark shipped before downloads
/// became job-local. It has no remaining reader and, unlike a per-job bookmark,
/// no UI through which the operator could revoke it. A formal launch removes it
/// idempotently before any download service is constructed.
enum AppStorageMigrations {
    static let legacyGlobalDownloadDestination = StorageKey("download-destination")

    static func removeLegacyGlobalDownloadDestination(from storage: any StorageManaging) {
        storage.forget(legacyGlobalDownloadDestination)
    }
}

/// A durable transfer is visible even when no storefront row currently exists
/// for its model. Catalog metadata may arrive later; it is not the authority
/// for whether a saved job exists on this machine.
struct SavedTransferPresentation: Equatable, Identifiable {
    var id: DownloadJobID { job }
    let job: DownloadJobID
    let model: ModelID
    let destinationPath: String
    let state: String
    let reason: String?
    let updatedAt: Date
}

/// One durable download job on this Mac, independent of whether its model is
/// still present in the current storefront response. The Downloads screen is
/// built from this projection so a catalog refresh can never hide local work.
struct TransferPresentation: Equatable, Identifiable {
    var id: DownloadJobID { job }
    let job: DownloadJobID
    let model: ModelID
    let displayName: String
    let destinationPath: String
    let declaredBytes: UInt64
    let state: DownloadState
    let updatedAt: Date
    let hasCatalogEntry: Bool
    let reason: String?
}

private struct PendingRestoredDownload {
    let state: DownloadJobState
    let summary: DownloadJobSummary
}

enum ConversationPersistence {
    static func open(
        _ factory: () throws -> ConversationStore = { try .applicationSupport() }
    ) -> (store: ConversationStore, error: String?) {
        do {
            return (try factory(), nil)
        } catch {
            // Ephemeral means exactly what happened: the conversation remains
            // usable for this launch but is not presented as durable storage.
            return (
                .ephemeral(),
                "Conversation storage could not be opened; chats from this launch will not "
                    + "be saved: \(error)")
        }
    }
}

/// Turns a kit catalog snapshot into the richer rows the App currently needs.
///
/// The snapshot decides the set of rows and every descriptor field. The old
/// preview entries are consulted only for App-side geometry and workload terms
/// that MinirunKit does not vend yet; iterating the snapshot rather than the
/// fixtures is what lets a cached or live catalog grow without an App release.
enum AppCatalogAdapter {
    static func entries(from snapshot: ModelCatalogSnapshot) -> [CatalogEntry] {
        snapshot.models.map(entry(from:))
    }

    static func entry(from descriptor: ModelDescriptor) -> CatalogEntry {
        // The owner may update one repository without releasing a new App.
        // Supplemental model geometry follows that repository identity, while
        // the live descriptor still carries the newly pinned commit and the
        // runner rechecks the opened layer census before accepting a plan.
        let supplemental: CatalogEntry?
        if let candidate = CatalogFixtures.entry(descriptor.id),
            case .huggingFaceRepo(let current) = descriptor.source,
            case .huggingFaceRepo(let known) = candidate.descriptor.source,
            current.repoID == known.repoID
        {
            supplemental = candidate
        } else {
            supplemental = nil
        }
        let sourceRepo: String?
        let sourceRevision: String?
        switch descriptor.source {
        case .huggingFaceRepo(let repo):
            sourceRepo = repo.repoID
            sourceRevision = repo.revision
        case .locallyBuilt:
            sourceRepo = nil
            sourceRevision = nil
        }

        let baseMemory = supplemental?.memory
        return CatalogEntry(
            descriptor: descriptor,
            index: ArtifactIndexSummary(
                layers: supplemental?.index.layers,
                globals: supplemental?.index.globals,
                files: descriptor.payloadFileCount,
                bytes: descriptor.payloadBytes,
                sourceRepo: sourceRepo,
                sourceRevision: sourceRevision),
            memory: MemoryProfile(
                census: baseMemory?.census ?? CatalogFixtures.unknownGeometryCensus,
                expertPoolBytes: baseMemory?.expertPoolBytes ?? 0,
                workingSetReserveBytes: baseMemory?.workingSetReserveBytes ?? 0,
                requiredMinimumBudgetBytes: baseMemory?.requiredMinimumBudgetBytes,
                onRecordMinimumBudgetBytes: baseMemory?.onRecordMinimumBudgetBytes,
                provenance: baseMemory == nil ? .unknown : .declaredByIndex,
                deepSeekV4: baseMemory?.deepSeekV4),
            terms: supplemental?.terms
                ?? WorkloadTerms(
                    deterministicBytesPerToken: 0,
                    expertBytesPerToken: 0,
                    computeSecondsPerToken: 0,
                    isMeasuredHere: false),
            architectureSubtitle: architectureSubtitle(descriptor.architecture))
    }

    private static func architectureSubtitle(_ architecture: ModelArchitecture) -> String {
        switch architecture {
        case .kimiK3MoE: return "Kimi K3 MoE"
        case .deepseekV4FlashMoE: return "DeepSeek V4 Flash MoE"
        case .minimaxH3Video: return "MiniMax H3 video DiT"
        // `qwen3MoE` survives in the kit only so a catalogue cached by an older
        // build still decodes. This build publishes no such model and must not
        // name one on a row it can never offer.
        case .qwen3MoE, .unrecognized: return "Architecture not recognized"
        }
    }
}

/// The product's storefront policy, kept separate from catalogue discovery.
///
/// MinirunKit deliberately discovers an open set of owned `*-minirun`
/// repositories. The App must not turn that discovery layer into a fixed list,
/// and the Models surface must keep that whole set visible. Runtime support is
/// a separate row-level fact: MiniMax H3 is a video DiT and this build cannot
/// chat with it, but hiding its published container would make the scan count,
/// storage state and catalog list disagree. A row becomes runnable only when
/// its own verified runtime provider is registered.
enum ProductCatalogPolicy {
    static func includes(_ descriptor: ModelDescriptor) -> Bool {
        if case .huggingFaceRepo = descriptor.source { return true }
        return false
    }

    static func applying(to snapshot: ModelCatalogSnapshot) -> ModelCatalogSnapshot {
        ModelCatalogSnapshot(
            schemaVersion: snapshot.schemaVersion,
            generatedAt: snapshot.generatedAt,
            origin: snapshot.origin,
            models: snapshot.models.filter(includes))
    }
}

/// Whether a model found by discovery can back a run in this build.
///
/// This deliberately has no download-state case. A transfer is history; a
/// discovery report is the current answer about which bytes can be reached.
enum ModelRunAvailability: Equatable, Sendable {
    case available
    case missingFromCatalog
    case notFoundByScan
    case storageUnavailable
    case incompleteArtifact
    case unverifiedArtifact
    case runtimeUnavailable
    case runtimeFilesUnavailable
    case unrecognizedArtifact

    var isAvailable: Bool { self == .available }

    var reason: String {
        switch self {
        case .available: return "available from scanned storage"
        case .missingFromCatalog: return "not present in the current catalog"
        case .notFoundByScan: return "not found in scanned storage"
        case .storageUnavailable: return "its storage location is unavailable"
        case .incompleteArtifact: return "the scanned artifact is incomplete"
        case .unverifiedArtifact: return "the scanned artifact is not fully verified"
        case .runtimeUnavailable:
            return "no verified runner and tokenizer ship in this build"
        case .runtimeFilesUnavailable:
            return "the verified artifact is missing files required by its runtime"
        case .unrecognizedArtifact: return "model format not recognized"
        }
    }

    /// Product copy for ordinary surfaces. `reason` remains the diagnostic
    /// value used by tests and named failures; this version says what the
    /// person can act on without exposing registry or tokenizer plumbing.
    var productReason: String {
        switch self {
        case .available: return "Ready"
        case .missingFromCatalog: return "Not in the current catalog"
        case .notFoundByScan: return "Not found in scanned storage"
        case .storageUnavailable: return "Its storage location is unavailable"
        case .incompleteArtifact: return "The scanned files are incomplete"
        case .unverifiedArtifact: return "The scanned files are not fully verified"
        case .runtimeUnavailable: return "This version has no verified runtime for this model"
        case .runtimeFilesUnavailable:
            return "Update this model copy, then verify it again"
        case .unrecognizedArtifact: return "This model format is not supported"
        }
    }

    /// Secondary copy for a compact status card. A successful state already
    /// says “Ready” in its headline, so repeating the same word underneath is
    /// noise; unavailable states still need the actionable reason.
    var productDetail: String? {
        isAvailable ? nil : productReason
    }
}

/// The next honest step in an empty chat workspace.
///
/// This is deliberately derived state rather than a persisted onboarding
/// cursor. A catalog refresh, a removed bookmark, a newly mounted drive, or a
/// completed verification can all change the correct next step between two
/// launches; replaying a saved wizard page would then be a lie.
enum ChatReadiness: Equatable, Sendable {
    case ready(model: ModelID)
    case loadingCatalog
    case catalogUnavailable
    case needsStorage
    case scanningStorage
    case needsVerification(model: ModelID)
    case needsModel
}

/// One row in either model picker. Identified scan results carry a ModelID;
/// unidentified artifacts stay visible but cannot become a selection.
struct ModelPickerChoice: Equatable, Identifiable, Sendable {
    let id: String
    let modelID: ModelID?
    let displayName: String
    let availability: ModelRunAvailability

    var isSelectable: Bool { modelID != nil && availability.isAvailable }
    var label: String {
        availability.isAvailable
            ? displayName
            : "\(displayName) — unavailable: \(availability.reason)"
    }
}

/// The operator's application-wide appearance. `system` deliberately stores
/// no guessed light/dark value: a running window follows the platform again as
/// soon as that option is selected.
enum AppAppearance: String, CaseIterable, Codable, Identifiable, Sendable {
    case system = "System"
    case light = "Light"
    case dark = "Dark"

    var id: String { rawValue }
}

/// Everything the app knows, in one observable.
///
/// It is the only place that holds a `StorageManaging`, a `DownloadManaging`, a
/// `RunnerFacade` or a `ConversationStore`; screens take what they need from
/// here and never construct one. That is what makes the swap to MinirunKit a
/// change to this file's initialiser rather than to every screen.
///
/// The spine is the **conversation**. A run belongs to a turn, a turn belongs to
/// a conversation, and everything that decides how a run behaves is stored on
/// the conversation — never here, and never in global settings. What global
/// settings holds is `NewChatDefaults`, which is copied at creation and never
/// consulted again.
enum AppSceneState: Equatable, Sendable {
    case active
    case inactive
    case background
}

@Observable
@MainActor
final class AppModel {

    // MARK: Navigation

    /// What the detail pane shows. `chats` is the honest empty workspace used
    /// before a runnable model exists; it does not create or persist a fake
    /// conversation merely to give Settings somewhere to return to.
    ///
    /// Models used to be a third case with its own sidebar row. It is now a
    /// section *inside* Settings: a sidebar that mixes conversations with
    /// anything else pushes the anything else off the bottom the moment the
    /// chat list grows, which is exactly what happened.
    enum Destination: Hashable {
        case chats
        case conversation(UUID)
        case settings
    }

    /// A launch refusal is intentionally a whole-app state. It exists when a
    /// visual-review switch was present but malformed or unavailable, before
    /// the product is allowed to open any operator state.
    struct LaunchRefusal: Equatable, Sendable {
        let title = "Minirun did not open"
        let message =
            "The visual review option was invalid or is unavailable in this build. "
            + "Quit and launch Minirun again without that option."
        let privacyNote = "No conversations, defaults, or storage locations were opened."
    }

    let launchRefusal: LaunchRefusal?

    /// Non-nil only for the explicitly requested DEBUG visual fixture. The
    /// banner prevents a sample catalog and deliberately disabled storage from
    /// being mistaken for the live product during hands-on review.
    private(set) var visualReviewNotice: String?

    /// One page of the Settings surface. The order here is the order of its
    /// left nav.
    enum SettingsSection: String, Hashable, CaseIterable, Identifiable, Sendable {
        case general = "General"
        case storage = "Storage"
        case models = "Models"
        case downloads = "Downloads"
        case about = "About"

        var id: String { rawValue }

        var systemImage: String {
            switch self {
            case .general: return "gearshape"
            case .storage: return "externaldrive"
            case .models: return "square.stack.3d.up"
            case .downloads: return "arrow.down.circle"
            case .about: return "info.circle"
            }
        }
    }

    /// A passive, serializable projection for SwiftUI scene restoration. It is
    /// never bound as a second navigation state: the view restores it into the
    /// model once, then mirrors model changes back to SceneStorage.
    struct SceneNavigationMirror: Equatable, Sendable {
        enum Place: String, Sendable {
            case chats
            case conversation
            case settings
        }

        let place: Place
        let conversationID: UUID?
        let settingsSection: SettingsSection?
    }

    var destination: Destination = .chats
    var settingsSection: SettingsSection = .general
    /// The Settings push path is part of application navigation, not view-local
    /// state. Keeping it here lets a programmatic route and a restored SwiftUI
    /// scene describe the same destination on iPhone and iPad.
    private(set) var settingsNavigationPath: [SettingsSection] = []
    /// A direct route can be written before iOS has constructed the lazily
    /// hosted Settings navigation stack. SwiftUI initializes that new stack by
    /// writing an empty path through its binding; without this short-lived
    /// intent, that framework write erases a Find Models / Choose Storage tap.
    /// The Settings root acknowledges the intent after its first render.
    private var pendingSettingsNavigationSection: SettingsSection?
    /// Gives a retained TabView child a new task identity every time a direct
    /// Settings route crosses the tab boundary. A plain onAppear is not enough:
    /// iOS may keep both tabs mounted after their first visit.
    private(set) var settingsNavigationActivationID = UUID()
    /// Settings is a detour, not a reset to the newest transcript. Remember
    /// the exact workspace that opened it so Back returns without surprising
    /// navigation or creating a conversation.
    private var settingsReturnDestination: Destination = .chats

    /// What the right-hand panel shows beside a conversation, or nil for no
    /// panel at all.
    ///
    /// The Mac has one inspector, toggled from the conversation toolbar, with a
    /// visible segmented control that switches its Settings and Instruments
    /// faces. The phone maps the same two cases to sheets in its ellipsis menu.
    enum ConversationPanel: String, Hashable, CaseIterable, Identifiable {
        case chatSettings = "Chat settings"
        case instruments = "Instruments"

        var id: String { rawValue }

        var systemImage: String {
            switch self {
            case .chatSettings: return "slider.horizontal.3"
            case .instruments: return "gauge.with.dots.needle.bottom.50percent"
            }
        }
    }

    /// Closed at launch. Opening the inspector divides the current window; it
    /// never changes the outer window's size.
    var conversationPanel: ConversationPanel?

    /// Opens the Settings surface at a named section. The one entry point, so a
    /// banner that wants the Models page cannot land on General instead.
    func openSettings(_ section: SettingsSection = .general) {
        if destination != .settings {
            settingsReturnDestination = destination
            pendingSettingsNavigationSection = section
            settingsNavigationActivationID = UUID()
        } else {
            pendingSettingsNavigationSection = nil
        }
        settingsSection = section
        settingsNavigationPath = [section]
        destination = .settings
    }

    /// Opens the Settings index itself. A direct tab selection is not the same
    /// route as a banner that names Models or Storage.
    func openSettingsRoot() {
        if destination != .settings { settingsReturnDestination = destination }
        pendingSettingsNavigationSection = nil
        settingsNavigationPath = []
        destination = .settings
    }

    /// Called by the iOS Settings root after its lazy NavigationStack or
    /// NavigationSplitView exists. Reasserting the named section here makes a
    /// direct route deterministic, then returns ordinary pop gestures to the
    /// normal path setter.
    func activatePendingSettingsNavigation() {
        guard destination == .settings,
            let section = pendingSettingsNavigationSection
        else { return }
        settingsSection = section
        settingsNavigationPath = [section]
        pendingSettingsNavigationSection = nil
    }

    /// Applies a user-driven Settings stack mutation without creating a second
    /// navigation truth in the view. Settings has one level at this boundary;
    /// feature-owned detail pushes below a section remain inside that feature.
    func setSettingsNavigationPath(_ path: [SettingsSection]) {
        guard destination == .settings else { return }
        if path.isEmpty, pendingSettingsNavigationSection != nil {
            // This is the lazy stack's initialization write, not a user pop.
            return
        }
        pendingSettingsNavigationSection = nil
        settingsNavigationPath = Array(path.suffix(1))
        if let section = settingsNavigationPath.last { settingsSection = section }
    }

    /// Applies a user-driven Chats stack mutation. SwiftUI may clear an
    /// inactive tab's stack while Settings is visible, so that write is ignored
    /// instead of bouncing the App back to Chats.
    func setConversationNavigationPath(_ path: [UUID]) {
        guard destination != .settings else { return }
        guard let id = path.last, conversation(id) != nil else {
            destination = .chats
            return
        }
        destination = .conversation(id)
    }

    func openConversation(_ id: UUID) {
        guard conversation(id) != nil else { return }
        destination = .conversation(id)
    }

    var sceneNavigationMirror: SceneNavigationMirror {
        switch destination {
        case .chats:
            return SceneNavigationMirror(
                place: .chats, conversationID: nil, settingsSection: nil)
        case .conversation(let id):
            return SceneNavigationMirror(
                place: .conversation, conversationID: id, settingsSection: nil)
        case .settings:
            return SceneNavigationMirror(
                place: .settings, conversationID: nil,
                settingsSection: settingsNavigationPath.last)
        }
    }

    func restoreNavigation(from mirror: SceneNavigationMirror) {
        switch mirror.place {
        case .chats:
            pendingSettingsNavigationSection = nil
            settingsNavigationPath = []
            destination = .chats
        case .conversation:
            pendingSettingsNavigationSection = nil
            settingsNavigationPath = []
            if let id = mirror.conversationID, conversation(id) != nil {
                destination = .conversation(id)
            } else {
                destination = .chats
            }
        case .settings:
            settingsReturnDestination = .chats
            if let section = mirror.settingsSection,
                section != .downloads || showsDownloadsSection
            {
                settingsSection = section
                settingsNavigationPath = [section]
                pendingSettingsNavigationSection = section
                settingsNavigationActivationID = UUID()
            } else {
                settingsNavigationPath = []
                pendingSettingsNavigationSection = nil
            }
            destination = .settings
        }
    }

    /// Leaves Settings for the exact workspace that opened it. This never
    /// creates a chat as a navigation side effect.
    func backToApp() {
        pendingSettingsNavigationSection = nil
        settingsNavigationPath = []
        switch settingsReturnDestination {
        case .conversation(let id) where conversation(id) != nil:
            destination = .conversation(id)
        case .conversation, .chats, .settings:
            destination = .chats
        }
    }

    /// The scene phase is mirrored here so lifecycle work and navigation are
    /// driven by the same application state on every iOS layout. Views report
    /// platform events; they do not own a second active/inactive state machine.
    private(set) var sceneState: AppSceneState = .inactive

    /// Handles the foreground-only boundaries already exposed by the product.
    /// Leaving active generation requests the runner's safe layer-boundary
    /// stop. Active downloads are paused through their durable manager. No
    /// transfer is converted into a terminal failure here.
    ///
    /// Verification is the exception, and deliberately so. It used to be
    /// cancelled here, which is what made pulling down Notification Center
    /// throw away half an hour of hashing. A pass the system agreed to continue
    /// now keeps running in the background; one it did not agree to continue
    /// stops on a file boundary with its checkpoint intact and resumes when the
    /// app is frontmost again.
    func transitionScene(to state: AppSceneState) {
        guard state != sceneState else { return }
        let wasActive = sceneState == .active
        sceneState = state

        if state == .active {
            refreshVolumes()
            installed.refresh()
            installed.resumePausedVerification()
            return
        }
        guard wasActive else { return }

        if run.isRunning { stopTurn() }
        installed.applicationWillLeaveForeground()

        var downloadsToPause: [DownloadController] = []
        for controller in downloadJobs.values {
            switch controller.state {
            case .active:
                downloadsToPause.append(controller)
            case .verifying:
                controller.cancelVerification()
            case .notStarted, .resolvingIndex, .awaitingDestination, .paused,
                .interrupted, .cancelled, .ready, .incomplete, .failed:
                break
            }
        }
        guard !downloadsToPause.isEmpty else { return }
        Task {
            for controller in downloadsToPause { await controller.pause() }
        }
    }

    // MARK: Conversations

    private(set) var conversations: [Conversation] = []
    /// Documents that would not decode. Reported in Settings rather than
    /// silently skipped.
    private(set) var conversationLoadFailures: [ConversationStore.LoadFailure] = []
    private(set) var conversationWriteError: String?
    private(set) var conversationStorageError: String?
    /// Composer text per conversation. Deliberately NOT persisted: an unsent
    /// draft is not a turn, and writing it to disk would make the transcript
    /// file disagree with what was actually run.
    var drafts: [UUID: String] = [:]

    let conversationStore: ConversationStore

    /// The conversation whose turn is in flight, if any.
    private(set) var runningConversationID: UUID?
    /// Which conversation owns the controller's current snapshot. Unlike
    /// `runningConversationID`, this remains set after a terminal event so the
    /// owner may inspect its final instruments. Every other conversation sees
    /// an idle snapshot, never this one.
    private(set) var runSnapshotConversationID: UUID?
    /// The message the in-flight turn is answering, so a stop or a fault can be
    /// recorded against the right turn.
    private var runningTurnStartedAt: Date?
    /// A failure while resolving a run's storage grant belongs to the chat that
    /// tried to run. It must not become a global banner in every conversation.
    private(set) var runPreparationErrors: [UUID: String] = [:]
    /// The named budget required by a prompt-specific preparation refusal.
    /// Kept separately from generic storage/runtime failures so the UI can
    /// offer Chat settings instead of sending the user to an unrelated page.
    private var runPreparationBudgetTargets: [UUID: UInt64] = [:]

    // MARK: Global defaults for NEW conversations

    /// The only run-shaped values global settings holds. Copied onto a
    /// conversation at creation; never read during a run.
    var defaults: NewChatDefaults {
        didSet { persistDefaults() }
    }

    var appearance: AppAppearance {
        didSet { persistAppearance() }
    }

    private let defaultsKey = "minirun.newChatDefaults"
    private static let appearanceKey = "minirun.appearance"
    private let userDefaults: UserDefaults
    /// False only for explicit preview/test instances. It is part of the truth
    /// boundary: changing review controls must never rewrite product defaults.
    let persistsDefaults: Bool

    // MARK: Catalog

    private(set) var entries: [CatalogEntry]
    private(set) var snapshot: ModelCatalogSnapshot
    /// Metadata derived from the exact rooted artifact most recently prepared
    /// for this model. It is deliberately launch-local: discovery and complete
    /// verification remain the authority for whether a copy may run, while
    /// this cache only keeps the visible dial identical to the request just
    /// constructed from that copy.
    private var preparedArtifactCensuses: [ModelID: ArtifactCensus] = [:]
    /// The same thing for DeepSeek V4, whose ladder needs the kit's second
    /// census and a floor priced from the product plan's own terms.
    ///
    /// It is read the same way and cached the same way, with one difference
    /// that matters for the dial: V4's census comes from layer manifests alone,
    /// so it can be taken as soon as a copy is verified rather than only from
    /// the run that consumes it. Without that, a new chat's default preset
    /// would have to be chosen against a ladder nobody had read yet.
    private var preparedDeepSeekV4Ladders:
        [ModelID: DeepSeekV4MemoryDialInputs.ArtifactProfile] = [:]
    /// Prompt-length-aware reserve most recently prepared for each conversation.
    /// Unlike the model census, this cannot be keyed only by model: MLA state
    /// grows with the exact transcript sent to the runner.
    private var preparedWorkingSetReserves: [UUID: UInt64] = [:]
    private(set) var divergences: [CatalogDivergence] = []
    private(set) var catalogError: CatalogError?
    /// A live list can succeed even when its offline snapshot cannot be
    /// replaced. Keep the current rows, and report that narrower durability
    /// failure independently from the network/catalog result.
    private(set) var catalogCacheWarning: String?
    /// The fast Hugging Face index used by Find Models. These references name
    /// repositories and immutable commits only; a row is not promoted into
    /// `entries` until its complete tree has been resolved.
    private(set) var publishedModels: [PublishedModelReference]
    private(set) var publishedModelsOrigin: ModelCatalogSnapshot.Origin
    private(set) var isRefreshingPublishedModels = false
    private var hasCompletedPublishedModelLoad: Bool
    private var didAttemptPublishedModelLoadThisLaunch = false
    private var publishedResolutionsInFlight: Set<PublishedModelReference> = []
    private var publishedResolutionFailures: [PublishedModelReference: CatalogError] = [:]
    private var attemptedLocalPublishedResolutions: Set<PublishedModelReference> = []
    private(set) var isRefreshing = false
    /// At most one list refresh may be inferred from a locally newer artifact
    /// in one launch. The follow-up resolves only the matching repository, not
    /// every published model tree.
    private var didRequestPublishedRefreshForLocalDivergence = false
    /// False for every ephemeral review/refusal model. Such a model must not
    /// touch the network or the product catalog cache merely because a sample
    /// Models screen appeared.
    let allowsLiveCatalogRefresh: Bool
    private let catalogService: ModelCatalog
    /// A previous successful network snapshot is the product's only offline
    /// catalogue authority. Bundled descriptors remain recognition metadata;
    /// they are never promoted into product rows because the network failed.
    private let catalogCache: (any CatalogCache)?

    var catalogSort: CatalogSort = .size

    enum CatalogSort: String, CaseIterable, Identifiable {
        case size = "Size"
        case name = "Name"
        var id: String { rawValue }
    }

    // MARK: Device and storage

    let device: DeviceProfile
    let storage: any StorageManaging
    /// What is on the drives, read from the drives. The catalog says what a
    /// model *is*; this says whether one is here.
    let installed: InstalledModels
    /// What each registered location measured when somebody asked it to. Never
    /// populated by a screen appearing.
    let assessments: VolumeAssessments
    private(set) var volumes: [VolumeDescriptor] = []
    private(set) var volumeError: String?
    private(set) var calibrations: [String: VolumeCalibration]

    // MARK: Per-model state

    /// Durable transfers are keyed by their job identity, not by model. One
    /// model may have copies on several drives and restoring one must never
    /// overwrite another merely because their descriptors share a ModelID.
    private(set) var downloadJobs: [DownloadJobID: DownloadController] = [:]
    private var downloadDrafts: [ModelID: DownloadController] = [:]
    private var selectedDownloadJob: [ModelID: DownloadJobID] = [:]
    private var pendingRestoredDownloads: [DownloadJobID: PendingRestoredDownload] = [:]
    private(set) var downloadRecoveryIssues: [DownloadRecoveryIssue] = []
    /// Fatal construction failure: no durable manager/store exists. This alone
    /// disables every new transfer.
    private(set) var downloadSetupError: String?
    /// A retryable problem restoring old jobs. The manager and store remain
    /// usable, so unrelated new transfers stay available.
    private(set) var downloadServiceError: String?
    private(set) var artifactRegistrationError: String?
    private(set) var localCopyRemovalInFlight: DiscoveredArtifact?
    private(set) var localCopyRemovalOutcome: [String: String] = [:]
    private(set) var isRecoveringDownloads: Bool
    private var didCompleteDownloadRestore = false
    private var downloadRestoreTask: Task<Bool, Never>?
    var history: [RunRecord]

    let run = RunController()
    private let downloadManager: any DownloadManaging
    private let downloadStateStore: (any DownloadStateStore)?
    private let restoresPersistedDownloads: Bool
    /// The only authority for executable model support. Catalogue metadata is
    /// intentionally not consulted as a substitute for a complete binding.
    private let runtimes: ModelRuntimeRegistry

    // MARK: Init

    init(
        /// Explicit preview/test entries. Nil in the product, whose rows come
        /// only from a successful live refresh or its successful cache.
        entries injectedEntries: [CatalogEntry]? = nil,
        catalogSnapshot injectedSnapshot: ModelCatalogSnapshot? = nil,
        catalogService injectedCatalogService: ModelCatalog? = nil,
        catalogCache injectedCatalogCache: (any CatalogCache)? = nil,
        allowsLiveCatalogRefresh: Bool = true,
        /// Complete runner + tokenizer bindings. Empty and verified-only by
        /// default; previews/tests inject their stand-ins explicitly.
        runtimes injectedRuntimes: ModelRuntimeRegistry = ModelRuntimeRegistry(),
        store: ConversationStore,
        conversationStorageError: String? = nil,
        userDefaults: UserDefaults = .standard,
        persistsDefaults: Bool = true,
        storage injectedStorage: (any StorageManaging)? = nil,
        /// Product persistence is explicit at `live()`. The initializer's
        /// default is process-local so a preview or unit-test model can never
        /// inspect or invalidate the operator's verification evidence.
        verificationLedger: any ArtifactVerificationLedger = InMemoryVerificationLedger(),
        downloadServices: AppDownloadServices,
        launchRefusal: LaunchRefusal? = nil,
        now: Date = Date(),
        /// Injectable so a test can point discovery at a fixture directory
        /// instead of at whatever happens to be plugged into the machine
        /// running the suite.
        installed injectedInstalled: InstalledModels? = nil,
        /// Injectable for the same reason: a test must not persist a report into
        /// the developer's own defaults.
        assessments injectedAssessments: VolumeAssessments? = nil,
        /// Whether to seed the archived device record — run history and a link
        /// calibration — into this instance.
        ///
        /// **False for the app the operator launches.** A seeded calibration is
        /// what let the Models list print "≈ 6:06/token here" for a model on no
        /// registered drive, beside a card saying no storage location had been
        /// added. Every claim about what is here has to come from a scan, and
        /// a fixture that survives into the shipping path is a claim that came
        /// from nowhere. Previews and tests still ask for it, by name.
        seedRecordedRuns: Bool = false,
        /// Tests and previews can inject an already-scanned report and keep it
        /// stable. The product always watches and scans its registered folders.
        startDiscovery: Bool = true
    ) {
        let initialSnapshot = injectedSnapshot
            ?? injectedEntries.map {
                ModelCatalogSnapshot(
                    generatedAt: now, origin: .bundled, models: $0.map(\.descriptor))
            }
            ?? ModelCatalogSnapshot(generatedAt: now, origin: .bundled, models: [])
        let entries = injectedEntries ?? AppCatalogAdapter.entries(from: initialSnapshot)
        self.entries = entries
        self.snapshot = initialSnapshot
        self.publishedModels = Self.publishedReferences(from: initialSnapshot)
        self.publishedModelsOrigin = initialSnapshot.origin
        self.hasCompletedPublishedModelLoad =
            injectedEntries != nil || initialSnapshot.origin != .bundled
        self.allowsLiveCatalogRefresh = allowsLiveCatalogRefresh
        if let injectedCatalogService {
            self.catalogService = injectedCatalogService
            self.catalogCache = injectedCatalogCache
        } else {
            let cache: any CatalogCache = injectedCatalogCache ?? FileCatalogCache.default
            self.catalogService = ModelCatalog(bundled: initialSnapshot, cache: cache)
            self.catalogCache = cache
        }
        self.runtimes = injectedRuntimes
        self.launchRefusal = launchRefusal
        self.device = DeviceProfile.current()
        let storage: any StorageManaging = injectedStorage ?? StorageManager()
        self.storage = storage
        self.installed = injectedInstalled
            ?? InstalledModels(
                storage: storage, catalog: initialSnapshot,
                ledger: verificationLedger,
                // Progress through a terabyte of hashing is durable for the
                // same reason the result is: the operator does not get to lose
                // half an hour because a notification banner arrived.
                checkpoints: UserDefaultsVerificationCheckpointStore(
                    defaults: userDefaults),
                backgroundWork: VerificationBackgroundWorkFactory.make())
        self.assessments = injectedAssessments
            ?? VolumeAssessments(storage: storage, catalog: initialSnapshot)
        #if DEBUG
            self.history = seedRecordedRuns ? CatalogFixtures.seededHistory(now: now) : []
        #else
            self.history = []
        #endif
        self.conversationStore = store
        self.conversationStorageError = conversationStorageError
        self.userDefaults = userDefaults
        self.persistsDefaults = persistsDefaults
        self.appearance = persistsDefaults
            ? AppAppearance(
                rawValue: userDefaults.string(forKey: Self.appearanceKey) ?? "") ?? .system
            : .system
        self.downloadManager = downloadServices.manager
        self.downloadStateStore = downloadServices.stateStore
        self.restoresPersistedDownloads = downloadServices.restoresPersistedJobs
        self.downloadSetupError = downloadServices.setupError
        self.downloadServiceError = nil
        self.isRecoveringDownloads = downloadServices.restoresPersistedJobs
        #if DEBUG
            self.calibrations =
                seedRecordedRuns
                ? Dictionary(
                    uniqueKeysWithValues: CatalogFixtures.seededCalibrations(now: now).map {
                        ($0.volumeMountPath, $0)
                    })
                : [:]
        #else
            self.calibrations = [:]
        #endif

        var loaded = NewChatDefaults()
        var migratedDefaults = false
        if persistsDefaults,
            let data = userDefaults.data(forKey: defaultsKey),
            let decoded = try? JSONDecoder().decode(NewChatDefaults.self, from: data)
        {
            loaded = decoded
        } else {
            // A default budget nobody has stated yet is the smallest budget the
            // model is on record under — the one number that is a measurement.
            //
            // Except where `defaultPreset` says a new chat starts somewhere
            // else. That default is a position on a ladder, and a ladder cannot
            // be read until a verified copy of the artifact is on disk, so
            // freezing a number here would freeze the wrong one. Those models
            // are left unset and derived on every read instead;
            // `setDefaultBudget` still stores an operator's own choice, and a
            // stored choice still wins.
            for entry in entries where Self.defaultPreset(for: entry.id) == .floor {
                loaded.setBudget(
                    entry.memory.requiredMinimumBudgetBytes
                        ?? entry.memory.onRecordMinimumBudgetBytes
                        ?? entry.memory.arithmeticFloorBytes,
                    for: entry.id)
            }
        }
        if (loaded.budgetSchemaVersion ?? 0) < NewChatDefaults.currentBudgetSchemaVersion {
            // Migrate only exact automatic boundaries. An operator-stated
            // value is never silently rewritten. macOS keeps the supported
            // 8 GB product policy; iOS moves the old 8 GB default to its
            // bounded 5.8 GB / two-token experimental tier.
            let policy = K3ProductMemoryBudget.currentPolicy
            if policy.isExperimental,
                loaded.budget(for: .kimiK3)
                    == K3ProductMemoryBudget.macOSProductPolicy.minimumBudgetBytes
            {
                loaded.setBudget(policy.minimumBudgetBytes, for: .kimiK3)
            } else if !policy.isExperimental,
                loaded.budget(for: .kimiK3)
                    == K3ProductMemoryBudget.iOSExperimentalPolicy.minimumBudgetBytes
            {
                loaded.setBudget(policy.minimumBudgetBytes, for: .kimiK3)
            }
            // The same rule for the seed this launch no longer writes. A stored
            // V4 budget that is exactly the old automatic 2 GB carries no
            // operator statement, so it is dropped and re-derived — which on a
            // Mac is what moves V4 off its floor and onto Balanced. Any other
            // stored number is a choice and stays.
            for entry in entries where Self.defaultPreset(for: entry.id) != .floor {
                let seeded = entry.memory.requiredMinimumBudgetBytes
                    ?? entry.memory.onRecordMinimumBudgetBytes
                    ?? entry.memory.arithmeticFloorBytes
                if loaded.budget(for: entry.id) == seeded {
                    loaded.clearBudget(for: entry.id)
                }
            }
            loaded.budgetSchemaVersion = NewChatDefaults.currentBudgetSchemaVersion
            migratedDefaults = true
        }
        if (loaded.instrumentPanelSchemaVersion ?? 0)
            < NewChatDefaults.currentInstrumentPanelSchemaVersion
        {
            // The prior product default was manual and the stored value could
            // not distinguish that inherited default from an operator choice.
            // Migrate it once; after the marker is written, an explicit Off in
            // General remains Off on every later launch.
            loaded.telemetryDensity = .cockpit
            loaded.instrumentPanelSchemaVersion =
                NewChatDefaults.currentInstrumentPanelSchemaVersion
            migratedDefaults = true
        }
        self.defaults = loaded
        if persistsDefaults, migratedDefaults,
            let data = try? JSONEncoder().encode(loaded)
        {
            userDefaults.set(data, forKey: defaultsKey)
        }

        for entry in entries { ensureDownloadDraft(for: entry) }
        refreshVolumes()
        installed.onReportChanged = { [weak self] report in self?.handleDiscoveryReport(report) }
        installed.updateCatalog(initialSnapshot, refresh: false)
        assessments.updateCatalog(initialSnapshot)
        // A snapshot with no models has a descriptor for nothing, so every
        // question such a scan puts to the verification ledger arrives with no
        // repository. That is not a cheap no-op: it is a lookup that cannot
        // match, made about artifacts that may hold a terabyte of checked bytes.
        // Discovery waits for a recognition table; `live()` always has at least
        // the bundled one.
        if startDiscovery, !initialSnapshot.models.isEmpty {
            installed.startWatchingVolumes()
            installed.refresh()
        }
        // Reports outlive a launch: a drive that measured 0.95 GB/s yesterday is
        // still worth showing today, labelled with the date it was taken.
        assessments.load(
            locationPaths: installed.locationKeys.compactMap { installed.recordedPath(of: $0) })
        loadConversations(recoveredAt: now)
        run.onTurnEnded = { [weak self] snapshot in
            self?.recordTurn(from: snapshot)
        }
    }

    /// The app's own instance, over the real Application Support directory.
    static func live(
        userDefaults: UserDefaults = .standard,
        catalogService: ModelCatalog? = nil,
        catalogCache: (any CatalogCache)? = nil,
        conversationFactory: () throws -> ConversationStore = {
            try .applicationSupport()
        },
        downloadServicesFactory: ((any StorageManaging) throws -> AppDownloadServices)? = nil,
        storageFactory: () -> any StorageManaging = { StorageManager() },
        startDiscovery: Bool = true
    ) -> AppModel {
        let effectiveCatalogCache: (any CatalogCache)? =
            catalogCache ?? (catalogService == nil ? FileCatalogCache.default : nil)
        let cachedSnapshot = effectiveCatalogCache.flatMap(Self.loadProductCatalogCache)
        // With no saved catalogue this used to launch under a snapshot holding
        // no models at all, which is not the neutral starting point it looks
        // like: discovery still ran, and an empty snapshot has a descriptor for
        // nothing, so every question it put to the verification ledger arrived
        // with no repository at all. On 2026-08-15 that erased complete
        // evidence for 1.7 TB of artifacts whose files had not moved a byte.
        //
        // The bundled catalogue is the recognition table this build was
        // compiled with. It is deliberately not publication evidence — while it
        // is the snapshot in hand the Models storefront renders no rows
        // (`catalogEntriesForPresentation`) and the remote browser lists none
        // (`publishedModelsForPresentation`) — but it is enough to name what is
        // already on the disk, which is what a scan needs and what the ledger
        // is entitled to be asked with.
        let initialSnapshot = cachedSnapshot
            ?? ProductCatalogPolicy.applying(to: ModelCatalog.bundled)
        let shouldLoadPublishedModels = cachedSnapshot == nil
        let storage = storageFactory()
        AppStorageMigrations.removeLegacyGlobalDownloadDestination(from: storage)
        let downloadServices: AppDownloadServices
        do {
            downloadServices = try downloadServicesFactory?(storage)
                ?? .persistent(storage: storage)
        } catch {
            // A resumable transfer without durable state is not resumable.
            // Keep the feature unavailable and name the setup error rather
            // than quietly substituting /tmp or an in-memory store.
            downloadServices = .unavailable(String(describing: error))
        }
        let conversations = ConversationPersistence.open(conversationFactory)
        // Built here rather than inside the initializer's default, because
        // this is the only launch that should ever touch the notification
        // centre: a hosted test bundle must not ask the operator for
        // permission to post about work it is not doing.
        let announcer = UserNotificationAnnouncer()
        let installed = InstalledModels(
            storage: storage, catalog: initialSnapshot,
            ledger: UserDefaultsVerificationLedger(defaults: userDefaults),
            checkpoints: UserDefaultsVerificationCheckpointStore(defaults: userDefaults),
            backgroundWork: VerificationBackgroundWorkFactory.make(),
            announcer: announcer,
            applicationIsFrontmost: VerificationAnnouncementPolicy.applicationIsFrontmost)
        // Nothing is pre-marked ready and nothing is seeded. The catalog and
        // discovery are MinirunKit's. Product runtimes are assembled from
        // independent verified providers. K3 and V4 are complete on arm64;
        // every provider still needs a scan-backed fully verified rooted
        // artifact before it can prepare.
        // A mock that hands the
        // launched app a verified 1.56 TB artifact it never looked for makes
        // the one screen that must not lie — is this model here — into a
        // fixture. What is on the disk comes from the scan or from nowhere.
        let model = AppModel(
            catalogSnapshot: initialSnapshot,
            catalogService: catalogService, catalogCache: effectiveCatalogCache,
            runtimes: .product, store: conversations.store,
            conversationStorageError: conversations.error, userDefaults: userDefaults,
            storage: storage,
            verificationLedger: UserDefaultsVerificationLedger(defaults: userDefaults),
            downloadServices: downloadServices,
            installed: installed,
            seedRecordedRuns: false, startDiscovery: startDiscovery)
        // A tap on the completion notification lands on the page that shows
        // the copy it is about. Wired at launch, because a tap may be what
        // launched this process.
        announcer.onOpenModels = { [weak model] in model?.openSettings(.models) }
        announcer.activate()
        Task { await model.restoreDownloads() }
        // A previously resolved, fully verified local model remains usable
        // offline indefinitely. With no cache, launch asks only for the fast
        // repository index; complete trees are resolved lazily for local rows
        // or a model the operator opens. Opening an ordinary chat never starts
        // Hugging Face work, and cache age alone never starts a tree walk.
        if shouldLoadPublishedModels {
            Task { await model.refreshPublishedModels() }
        }
        if model.conversations.isEmpty { _ = model.newConversation() }
        model.destination = model.conversations.first.map { .conversation($0.id) } ?? .chats
        model.applyStartOverride()
        return model
    }

    /// The product accepts only a previous successful Hugging Face response as
    /// launch authority. Older builds could write bundled fixtures into this
    /// slot, so loading the file is not by itself enough to make rows visible.
    private static func loadProductCatalogCache(
        _ cache: any CatalogCache
    ) -> ModelCatalogSnapshot? {
        guard let cached = try? cache.load(),
            cached.origin == .live || cached.origin == .cached
        else { return nil }
        return ProductCatalogPolicy.applying(to: cached).labelled(.cached)
    }

    /// A recent saved catalog normally prevents launch-time network work. One
    /// exception is observable without guessing: a recognized local artifact
    /// contains more product files or bytes than that saved immutable tree.
    /// This happens when the owner updates the repository and then updates a
    /// local copy. Refresh metadata once so the copy can be verified against
    /// the new publication; opening an ordinary chat still performs no HF I/O.
    static func cachedCatalogNeedsRefresh(
        snapshot: ModelCatalogSnapshot, report: DiscoveryReport
    ) -> Bool {
        guard snapshot.origin == .cached else { return false }
        for location in report.locations where location.isMounted {
            for artifact in location.artifacts {
                guard !artifact.metadataOverflowed, let id = artifact.model,
                    let descriptor = snapshot.descriptor(id)
                else { continue }
                if artifact.fileCount > descriptor.totalFileCount
                    || artifact.bytesOnDisk > descriptor.totalBytes
                {
                    return true
                }
            }
        }
        return false
    }

    private func handleDiscoveryReport(_ report: DiscoveryReport) {
        // A newly verified V4 copy is the first moment its ladder can be read,
        // and the dial needs it before any run: the default preset for a new
        // chat is a position on that ladder.
        refreshDeepSeekV4Ladder()
        schedulePublishedResolutionForLocalModels(in: report)
        guard !didRequestPublishedRefreshForLocalDivergence,
            Self.cachedCatalogNeedsRefresh(snapshot: snapshot, report: report)
        else { return }
        didRequestPublishedRefreshForLocalDivergence = true
        Task { await refreshPublishedModels() }
    }

    /// A fail-closed model for an unsafe visual-review request. It is capable
    /// only of backing the dedicated refusal screen: no persistent defaults,
    /// conversation directory, bookmark ledger, volume discovery or runtime is
    /// opened while this instance exists.
    static func refusingUnsafeReviewLaunch(
        userDefaults: UserDefaults = .standard
    ) -> AppModel {
        let storage = EphemeralPreviewStorage()
        let snapshot = ModelCatalogSnapshot(
            generatedAt: .distantPast, origin: .bundled, models: [])
        let report = DiscoveryReport(locations: [], scannedAt: Date())
        let installed = InstalledModels(
            storage: storage,
            catalog: snapshot,
            ledger: InMemoryVerificationLedger(),
            initialReport: report)
        let assessments = VolumeAssessments(
            storage: storage, ledger: InMemoryAssessmentLedger())
        return AppModel(
            catalogSnapshot: snapshot,
            catalogService: ModelCatalog(bundled: snapshot, cache: nil),
            allowsLiveCatalogRefresh: false,
            runtimes: ModelRuntimeRegistry(),
            store: .ephemeral(),
            userDefaults: userDefaults,
            persistsDefaults: false,
            storage: storage,
            downloadServices: .unavailable(
                "downloads are disabled in the launch-refusal state"),
            launchRefusal: LaunchRefusal(),
            installed: installed,
            assessments: assessments,
            seedRecordedRuns: false,
            startDiscovery: false)
    }

    /// Where a launch may be told to start, for review.
    ///
    /// `-MinirunStart Models` on the command line lands the app on the Models
    /// page; `-MinirunPanel none` opens a conversation with no side panel, and
    /// any other value names one. It exists because this project's rule is that
    /// a screen is verified
    /// by looking at it, and a reviewer capturing four sections should not have
    /// to click their way to each one. It reads directly from
    /// `ProcessInfo.arguments`, never from a defaults domain:
    /// nothing here can change where an ordinary double-click opens the app.
    private func applyStartOverride(
        arguments: [String] = ProcessInfo.processInfo.arguments,
        allowsStorageOverride: Bool = true
    ) {
        #if DEBUG
            // `-MinirunAddLocation /Volumes/K3NVME` registers a folder through
            // exactly the call the folder picker makes, so the add, the grant,
            // the immediate scan and the persistence of all three can be
            // exercised from a script. DEBUG only, and it cannot grant more
            // than the picker could: the sandbox decides whether the bookmark
            // mints, and a refusal shows up on screen like any other.
            if allowsStorageOverride,
                let path = MinirunLaunchArguments.value(
                    for: "-MinirunAddLocation", in: arguments)
            {
                installed.addLocation(URL(fileURLWithPath: path, isDirectory: true))
            }
        #endif
        if let raw = MinirunLaunchArguments.value(for: "-MinirunPanel", in: arguments) {
            conversationPanel = ConversationPanel.allCases.first {
                $0.rawValue.caseInsensitiveCompare(raw) == .orderedSame
            }
        }
        guard
            let raw = MinirunLaunchArguments.value(for: "-MinirunStart", in: arguments)
        else { return }
        guard
            let section = SettingsSection.allCases.first(where: {
                $0.rawValue.caseInsensitiveCompare(raw) == .orderedSame
            })
        else { return }
        openSettings(section)
    }

    #if DEBUG
        /// The state a reviewer should land in: K3 present and verified, so a
        /// conversation and its instruments are one keystroke away, and the
        /// other published containers not yet downloaded, so the empty and
        /// download paths are too. The factory is absent from Release builds.
        static func preview(userDefaults: UserDefaults = .standard) -> AppModel {
            let storage = EphemeralPreviewStorage()
            let installed = previewInstalledModels(
                model: .kimiK3, snapshot: CatalogFixtures.productPreviewSnapshot,
                storage: storage)
            let assessments = VolumeAssessments(
                storage: storage, ledger: InMemoryAssessmentLedger())
            let entries = CatalogFixtures.productPreview
            let snapshot = CatalogFixtures.productPreviewSnapshot
            let model = AppModel(
                entries: entries,
                catalogSnapshot: snapshot,
                catalogService: ModelCatalog(
                    live: MockCatalogSource(latency: .zero),
                    bundled: snapshot, cache: nil),
                allowsLiveCatalogRefresh: false,
                runtimes: .preview(entries: entries),
                store: .ephemeral(),
                userDefaults: userDefaults,
                persistsDefaults: false,
                storage: storage,
                downloadServices: .preview(entries: entries),
                installed: installed,
                assessments: assessments,
                seedRecordedRuns: false,
                startDiscovery: false)
            model.download(.kimiK3)?.markReadyForPreview()
            _ = model.newConversation()
            model.destination = model.conversations.first.map { .conversation($0.id) } ?? .chats
            return model
        }

        /// A launch-only review fixture. It uses the same explicit preview
        /// dependencies as SwiftUI previews, but its conversations and defaults
        /// are non-persistent and it never accepts the storage-location override.
        static func visualReviewPreview(
            arguments: [String], userDefaults: UserDefaults = .standard
        ) -> AppModel {
            let model = preview(userDefaults: userDefaults)
            model.visualReviewNotice =
                "Review preview — sample data; storage access and persistence are disabled."
            model.applyStartOverride(arguments: arguments, allowsStorageOverride: false)
            return model
        }

        /// An explicit fixture for previews and App tests. Production cannot
        /// call this because the helper is not compiled into Release.
        static func previewInstalledModels(
            model id: ModelID,
            snapshot: ModelCatalogSnapshot = ModelCatalog.bundled,
            verification: ArtifactVerification = .fullyVerified,
            storage: any StorageManaging = StorageManager()
        ) -> InstalledModels {
            previewInstalledModels(
                models: [id], snapshot: snapshot, verification: verification, storage: storage)
        }

        static func previewInstalledModels(
            models ids: [ModelID],
            snapshot: ModelCatalogSnapshot = ModelCatalog.bundled,
            verification: ArtifactVerification = .fullyVerified,
            storage: any StorageManaging = StorageManager()
        ) -> InstalledModels {
            let locationPath = "/Volumes/MINIRUN-PREVIEW"
            let now = Date()
            let artifacts = ids.compactMap { id -> DiscoveredArtifact? in
                guard
                    let descriptor = snapshot.descriptor(id)
                        ?? ModelCatalog.bundled.descriptor(id)
                else { return nil }
                return DiscoveredArtifact(
                    rootPath: "\(locationPath)/\(id.rawValue)",
                    locationPath: locationPath,
                    index: .unreadable,
                    model: id,
                    displayName: descriptor.displayName,
                    bytesOnDisk: descriptor.totalBytes,
                    fileCount: descriptor.totalFileCount,
                    expectedFileCount: descriptor.totalFileCount,
                    expectedBytes: descriptor.totalBytes,
                    verification: verification,
                    verifiedAt: verification == .unverified ? nil : now,
                    scannedAt: now)
            }
            let report = DiscoveryReport(
                locations: [
                    LocationScan(
                        rootPath: locationPath,
                        displayName: "MINIRUN-PREVIEW",
                        isMounted: true,
                        storageKey: nil,
                        artifacts: artifacts,
                        unreadable: [:],
                        scannedAt: now)
                ],
                scannedAt: now)
            return InstalledModels(
                storage: storage,
                catalog: snapshot,
                ledger: InMemoryVerificationLedger(),
                initialReport: report)
        }
    #endif

    // MARK: - Conversations

    private func loadConversations(recoveredAt: Date) {
        let result = conversationStore.load()
        var restored = result.conversations
        for index in restored.indices {
            var recovered = restored[index]
            let migrated = Self.migrateK3Conversation(
                &recovered, to: K3ProductMemoryBudget.currentPolicy)
            let interrupted = recovered.recordInterruptedReplyIfNeeded(at: recoveredAt)
            guard migrated || interrupted else { continue }

            if conversationStore.isPersistent {
                do {
                    // ConversationStore writes one complete document with
                    // `.atomic`; publish the recovered value in memory only
                    // after that durable replacement succeeds.
                    try conversationStore.save(recovered)
                    restored[index] = recovered
                } catch {
                    conversationWriteError =
                        "The recovered conversation could not be saved: \(error)"
                }
            } else {
                restored[index] = recovered
            }
        }
        conversations = restored.sorted { $0.updatedAt > $1.updatedAt }
        conversationLoadFailures = result.failures
    }

    /// One-time compatibility for conversations created before the iPhone K3
    /// tier existed. The exact former automatic budget is recognizable; every
    /// other stated budget remains the operator's. The token ceiling is a hard
    /// runtime capability and is therefore constrained for every K3 turn.
    @discardableResult
    static func migrateK3Conversation(
        _ conversation: inout Conversation, to policy: K3ProductMemoryBudget.Policy
    ) -> Bool {
        guard policy.isExperimental, conversation.settings.model == .kimiK3 else {
            return false
        }
        var changed = false
        if conversation.settings.memoryBudgetBytes
            == K3ProductMemoryBudget.macOSProductPolicy.minimumBudgetBytes
        {
            conversation.settings.memoryBudgetBytes = policy.minimumBudgetBytes
            changed = true
        }
        if conversation.settings.maximumNewTokens > policy.maximumNewTokens {
            conversation.settings.maximumNewTokens = policy.maximumNewTokens
            changed = true
        }
        return changed
    }

    func conversation(_ id: UUID) -> Conversation? {
        conversations.first { $0.id == id }
    }

    var selectedConversation: Conversation? {
        guard case .conversation(let id) = destination else { return nil }
        return conversation(id)
    }

    /// Creates a conversation from the current defaults. This copy is the whole
    /// of the inheritance — afterwards the conversation owns its settings.
    @discardableResult
    func newConversation() -> UUID? {
        newConversation(modelID: defaults.model)
    }

    /// Creates from the model projected by `chatReadiness`. This is the
    /// first-run/empty-state action: it prefers the stated default, but can use
    /// another currently runnable model without rewriting that default.
    /// `newConversation()` remains strict so existing Mac/default semantics do
    /// not change silently.
    @discardableResult
    func newReadyConversation() -> UUID? {
        guard case .ready(let model) = chatReadiness else { return nil }
        return newConversation(modelID: model)
    }

    /// Creates a conversation for an explicit model while inheriting every
    /// other new-chat default. A model-detail CTA is a choice for this chat,
    /// not permission to rewrite the operator's global default model.
    @discardableResult
    func newConversation(modelID id: ModelID) -> UUID? {
        guard modelAvailability(for: id).isAvailable, let entry = entry(id),
            let capabilities = capabilities(for: id)
        else {
            return nil
        }
        var settings = defaults.settings(
            for: entry.id,
            // Only reached when nothing is stored for this model, which is the
            // case `defaultPreset` owns: the new chat starts where the dial
            // says a new chat starts.
            fallbackBudget: defaultBudgetPlan(for: entry.id)?.budgetBytes
                ?? entry.memory.requiredMinimumBudgetBytes
                ?? entry.memory.onRecordMinimumBudgetBytes
                ?? entry.memory.arithmeticFloorBytes,
            // A chat should complete a response, not stop after the first BPE
            // token. K3 stops earlier at verified response framing, while this
            // runner-owned ceiling remains the final safety bound and the user
            // can stop at any layer boundary.
            maximumNewTokens: capabilities.maximumNewTokens)
        Self.constrain(&settings, to: capabilities)
        let conversation = Conversation(settings: settings)
        conversations.insert(conversation, at: 0)
        persist(conversation)
        destination = .conversation(conversation.id)
        return conversation.id
    }

    /// Every mutation of a conversation goes through here, so persistence is not
    /// something a call site can forget.
    func update(_ id: UUID, _ mutate: (inout Conversation) -> Void) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        mutate(&conversations[index])
        conversations[index].updatedAt = Date()
        persist(conversations[index])
        sortConversations()
    }

    func rename(_ id: UUID, to title: String) {
        update(id) { $0.rename(to: title) }
    }

    func delete(_ id: UUID) {
        if runningConversationID == id { stopTurn() }
        if runningConversationID != id, runSnapshotConversationID == id {
            run.clear()
            runSnapshotConversationID = nil
        }
        conversations.removeAll { $0.id == id }
        drafts.removeValue(forKey: id)
        runPreparationErrors.removeValue(forKey: id)
        runPreparationBudgetTargets.removeValue(forKey: id)
        preparedWorkingSetReserves.removeValue(forKey: id)
        do {
            try conversationStore.delete(id)
        } catch {
            conversationWriteError = "\(error)"
        }
        if case .conversation(let selected) = destination, selected == id {
            destination = conversations.first.map { .conversation($0.id) } ?? .chats
        }
    }

    private func persist(_ conversation: Conversation) {
        guard conversationStore.isPersistent else {
            conversationWriteError = conversationStorageError
            return
        }
        do {
            try conversationStore.save(conversation)
            conversationWriteError = nil
        } catch {
            // A transcript that could not be written is a fact the operator
            // needs, not a log line nobody reads.
            conversationWriteError = "\(error)"
        }
    }

    private func sortConversations() {
        conversations.sort { $0.updatedAt > $1.updatedAt }
    }

    // MARK: - Turns

    func isRunning(_ id: UUID) -> Bool {
        runningConversationID == id && run.isRunning
    }

    func isStopping(_ id: UUID) -> Bool {
        runningConversationID == id && run.isStopping
    }

    var hasActiveRun: Bool { run.isRunning }

    /// The controller is global because this build permits one decode at a
    /// time; what it publishes is not. A conversation may observe only the
    /// snapshot it owns.
    func runSnapshot(for id: UUID) -> RunSnapshot {
        runSnapshotConversationID == id ? run.snapshot : RunSnapshot()
    }

    func canStartTurn(in id: UUID) -> Bool {
        guard !run.isRunning, let conversation = conversation(id) else { return false }
        return canRun(conversation)
    }

    /// Why this idle conversation cannot start while another one owns the
    /// single runner. The owner's title is the only cross-chat fact exposed;
    /// no token, fault or telemetry crosses with it.
    func runBusyReason(for id: UUID) -> String? {
        guard run.isRunning, runningConversationID != id else { return nil }
        let title = runningConversation?.title ?? "another conversation"
        return "A turn is already running in “\(title)”."
    }

    func openRunningConversation() {
        guard let id = runningConversationID, conversation(id) != nil else { return }
        openConversation(id)
    }

    func runPreparationError(for id: UUID) -> String? {
        runPreparationErrors[id]
    }

    func runPreparationBudgetTarget(for id: UUID) -> UInt64? {
        runPreparationBudgetTargets[id]
    }

    func setDraft(_ text: String, in id: UUID) {
        drafts[id] = text
        // A prompt-specific reserve belongs to the exact draft that produced
        // it. Editing that draft invalidates both the projection and its old
        // refusal; the next Send will prepare the new text from scratch.
        preparedWorkingSetReserves.removeValue(forKey: id)
        runPreparationBudgetTargets.removeValue(forKey: id)
        runPreparationErrors.removeValue(forKey: id)
    }

    var runningConversation: Conversation? {
        runningConversationID.flatMap { conversation($0) }
    }

    /// Appends the user's turn and starts the run that answers it.
    func send(_ text: String, in id: UUID) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !run.isRunning else { return }
        guard var prospective = conversation(id), canRun(prospective) else { return }

        let message = Message(role: .user, text: trimmed)
        prospective.append(message)
        let prepared: PreparedTurn
        runPreparationBudgetTargets.removeValue(forKey: id)
        do {
            guard let value = try prepareTurn(for: prospective, previousDigest: nil) else { return }
            prepared = value
        } catch {
            if let target = runPreparationBudgetTargets[id] {
                runPreparationErrors[id] =
                    "This prompt needs \(MRFormat.bytesDecimal(target)); this chat states "
                    + "\(MRFormat.bytesDecimal(prospective.settings.memoryBudgetBytes))."
            } else {
                runPreparationErrors[id] = "\(error)"
            }
            return
        }

        // Preparation — including resolving the live storage grant — completed
        // before the transcript or draft changes. A failed guard is a no-op on
        // the conversation the user owns.
        runPreparationErrors.removeValue(forKey: id)
        runPreparationBudgetTargets.removeValue(forKey: id)
        update(id) { $0.append(message) }
        drafts[id] = ""
        startPreparedTurn(prepared, in: id)
    }

    /// Drops the last assistant turn and runs it again, unchanged. The point is
    /// that the digest should match, and the turn's own footer says whether it
    /// did.
    func regenerateLastTurn(in id: UUID) {
        guard !run.isRunning else { return }
        guard var prospective = conversation(id), prospective.lastUserMessage != nil,
            canRun(prospective), let removed = prospective.removeLastAssistantTurn()
        else {
            return
        }

        let prepared: PreparedTurn
        runPreparationBudgetTargets.removeValue(forKey: id)
        do {
            guard
                let value = try prepareTurn(
                    for: prospective, previousDigest: removed.telemetry?.logitsDigest)
            else { return }
            prepared = value
        } catch {
            if let target = runPreparationBudgetTargets[id] {
                runPreparationErrors[id] =
                    "This prompt needs \(MRFormat.bytesDecimal(target)); this chat states "
                    + "\(MRFormat.bytesDecimal(prospective.settings.memoryBudgetBytes))."
            } else {
                runPreparationErrors[id] = "\(error)"
            }
            return
        }

        runPreparationErrors.removeValue(forKey: id)
        runPreparationBudgetTargets.removeValue(forKey: id)
        update(id) { _ = $0.removeLastAssistantTurn() }
        startPreparedTurn(prepared, in: id)
    }

    func stopTurn() {
        run.stop(clearSnapshot: false)
    }

    /// Builds the request from what the CONVERSATION states. Refusals happen in
    /// the runner, not here — this method never adjusts a budget to make one
    /// fit.
    func makeRequest(for conversation: Conversation) -> RunRequest? {
        try? makeResolvedRequest(for: conversation)
    }

    private func makeResolvedRequest(for conversation: Conversation) throws -> RunRequest? {
        guard localCopyRemovalInFlight?.model != conversation.settings.model,
            modelAvailability(for: conversation.settings.model).isAvailable,
            let entry = entry(conversation.settings.model),
            let runtime = runtimes.runtime(for: entry.id),
            let artifact = try installed.runnableArtifactReference(
                for: entry.id, matching: runtime.accepts)
        else { return nil }
        let prepared = try runtime.prepare(artifact: artifact, for: conversation)
        capture(prepared, for: entry.id)
        return try makeResolvedRequest(
            for: conversation, entry: entry, artifact: artifact,
            tokenizer: prepared.tokenizer, artifactCensus: prepared.artifactCensus,
            workingSetReserveBytes: prepared.workingSetReserveBytes,
            supportsPinPlan: prepared.runner.capabilities.supportsPinPlan)
    }

    private func makeResolvedRequest(
        for conversation: Conversation, entry: CatalogEntry,
        artifact: ArtifactReference, tokenizer: any PromptTokenizing,
        artifactCensus: ArtifactCensus?,
        workingSetReserveBytes: (@Sendable (Int, Int) throws -> UInt64)? = nil,
        supportsPinPlan: Bool
    ) throws -> RunRequest {
        if let artifactCensus {
            preparedArtifactCensuses[entry.id] = artifactCensus
        }
        let assembled = try PromptAssembler.assemble(
            conversation: conversation, tokenizer: tokenizer)
        let preparedReserve = try workingSetReserveBytes?(
            assembled.tokenIDs.count, conversation.settings.maximumNewTokens)
        if let preparedReserve {
            preparedWorkingSetReserves[conversation.id] = preparedReserve
        } else {
            preparedWorkingSetReserves.removeValue(forKey: conversation.id)
        }
        return RunRequest(
            model: entry.id,
            artifact: artifact,
            prompt: assembled.prompt,
            memoryBudgetBytes: conversation.settings.memoryBudgetBytes,
            pinPlan: supportsPinPlan
                ? budgetPlan(
                    for: conversation, artifactCensus: artifactCensus,
                    workingSetReserveBytes: preparedReserve)?.pinPlan
                : nil,
            maximumNewTokens: conversation.settings.maximumNewTokens,
            knobs: conversation.settings.knobs,
            workingDirectory: FileManager.default.urls(
                for: .documentDirectory, in: .userDomainMask)[0])
    }

    /// Keep whatever the rooted artifact just told us about its own geometry.
    ///
    /// Launch-local, like `preparedArtifactCensuses`: discovery and complete
    /// verification stay the authority on whether a copy may run, and this only
    /// keeps the dial identical to the request built from that copy.
    private func capture(_ prepared: PreparedModelRuntime, for id: ModelID) {
        if let ladder = prepared.deepSeekV4Ladder {
            preparedDeepSeekV4Ladders[id] = ladder
        }
    }

    func assembledPrompt(for conversation: Conversation) -> AssembledPrompt? {
        guard localCopyRemovalInFlight?.model != conversation.settings.model,
            let runtime = runtimes.runtime(for: conversation.settings.model),
            let artifact = try? installed.runnableArtifactReference(
                for: conversation.settings.model, matching: runtime.accepts),
            let prepared = try? runtime.prepare(artifact: artifact, for: conversation)
        else { return nil }
        return try? PromptAssembler.assemble(
            conversation: conversation, tokenizer: prepared.tokenizer)
    }

    private struct PreparedTurn {
        let entry: CatalogEntry
        let request: RunRequest
        let runner: any RunnerFacade
        let artifactCensus: ArtifactCensus?
        let progressLayerCount: Int?
        let decodeTokens: ([Int]) -> String
        let previousDigest: String?
    }

    private func prepareTurn(
        for conversation: Conversation, previousDigest: String?
    ) throws -> PreparedTurn? {
        guard canRun(conversation), let entry = entry(conversation.settings.model),
            let runtime = runtimes.runtime(for: conversation.settings.model),
            let artifact = try installed.runnableArtifactReference(
                for: entry.id, matching: runtime.accepts)
        else { return nil }
        let preparedRuntime = try runtime.prepare(artifact: artifact, for: conversation)
        capture(preparedRuntime, for: entry.id)
        let request = try makeResolvedRequest(
            for: conversation, entry: entry, artifact: artifact,
            tokenizer: preparedRuntime.tokenizer,
            artifactCensus: preparedRuntime.artifactCensus,
            workingSetReserveBytes: preparedRuntime.workingSetReserveBytes,
            supportsPinPlan: preparedRuntime.runner.capabilities.supportsPinPlan)
        if let refusal = budgetPlan(for: conversation)?.refusal {
            // A prompt can move the MLA floor after the static conversation
            // controls were drawn. Refuse during preparation, before the user
            // turn or draft is mutated. Keep that exact draft's reserve visible
            // so Chat settings can name and raise to the request-specific
            // requirement before retrying.
            if let target = refusal.suggestedBudgetBytes {
                runPreparationBudgetTargets[conversation.id] = target
            }
            throw refusal.runError(model: entry.id)
        }
        let tokenizer = preparedRuntime.tokenizer
        return PreparedTurn(
            entry: entry, request: request, runner: preparedRuntime.runner,
            artifactCensus: preparedRuntime.artifactCensus,
            progressLayerCount: preparedRuntime.progressLayerCount,
            decodeTokens: { tokenizer.decode(tokenIDs: $0) },
            previousDigest: previousDigest)
    }

    private func startPreparedTurn(_ prepared: PreparedTurn, in id: UUID) {
        runningConversationID = id
        runSnapshotConversationID = id
        runningTurnStartedAt = Date()
        run.start(
            runner: prepared.runner, request: prepared.request,
            linkCeiling: calibration(for: prepared.entry.id)?.bytesPerSecond,
            layerCount: prepared.progressLayerCount
                ?? prepared.artifactCensus?.layerStoredBytes.count
                ?? prepared.entry.memory.layerCount,
            storedBytesPerLayer: prepared.artifactCensus?.layerStoredBytes.max()
                ?? prepared.entry.memory.widestDeterministicLayerStoredBytes,
            widenedBytesPerLayer: prepared.artifactCensus?.layerResidentBytes.max()
                ?? prepared.entry.memory.widenedResidentBytes,
            decodeTokens: prepared.decodeTokens,
            previousDigest: prepared.previousDigest)
        // A refusal is synchronous and terminal; the controller has already
        // reported it by the time `start` returns.
        if case .refused = run.snapshot.state { recordTurn(from: run.snapshot) }
    }

    /// Writes the finished turn into the transcript.
    ///
    /// Every terminal state lands here, including the ones that produced no
    /// text: a refused, stopped or halted turn is kept, with its named error, so
    /// the conversation records what happened rather than what one wishes had.
    private func recordTurn(from snapshot: RunSnapshot) {
        guard let id = runningConversationID else { return }
        defer {
            runningConversationID = nil
            runningTurnStartedAt = nil
        }
        let text = snapshot.generatedText
            ?? snapshot.tokens.map { $0.text ?? "‹\($0.tokenID)›" }.joined()
        var namedError: String?
        var failure: MessageFailure?
        switch snapshot.state {
        case .refused(let error):
            namedError = error.description
            failure = error.messageFailure
        case .halted(let fault):
            namedError = fault.namedError
            failure = fault.messageFailure
        case .cancelled:
            namedError = "cancelled — stopped at a layer boundary"
            failure = .stopped
        default: break
        }
        guard !text.isEmpty || namedError != nil else { return }

        var telemetry: MessageTelemetry?
        if let summary = snapshot.summary {
            let completedTokens = summary.tokenIDs.count
            let bytesPerToken: UInt64?
            if !summary.finalTelemetry.reportsByteAccounting {
                bytesPerToken = nil
            } else if completedTokens > 0,
                let completed = summary.finalTelemetry.completedTokenBytes
            {
                bytesPerToken = completed.totalBytesRead / UInt64(completedTokens)
            } else {
                bytesPerToken = summary.finalTelemetry.bytesPerToken
            }
            telemetry = MessageTelemetry(
                model: summary.model,
                wallSeconds: summary.wallSeconds,
                tokensPerSecond: summary.tokensPerSecond,
                bytesPerToken: bytesPerToken,
                peakFootprintBytes: summary.peakFootprintBytes,
                declaredBudgetBytes: summary.finalTelemetry.declaredBudgetBytes,
                budgetRespected: summary.budgetRespected,
                logitsDigest: snapshot.reproducibility?.logitsDigest,
                thermalState: summary.finalTelemetry.thermalState.rawValue,
                // What the run had attributed when it ended, which for a
                // stopped turn is the passes that completed before the Stop
                // and not a reconstruction of the one that did not.
                phases: snapshot.phaseSummaries.isEmpty ? nil : snapshot.phaseSummaries,
                prefillPhase: snapshot.prefillPhaseSummary,
                decodePhaseAggregate: snapshot.decodePhaseAggregate)
        }
        let message = Message(
            role: .assistant, text: text, telemetry: telemetry, namedError: namedError,
            failure: failure, tokenIDs: snapshot.tokens.map { $0.tokenID })
        update(id) { $0.append(message) }
        if snapshot.summary != nil { recordFinishedRun(snapshot) }
    }

    // MARK: - Catalog

    func entry(_ id: ModelID) -> CatalogEntry? { entries.first { $0.id == id } }

    func download(_ id: ModelID) -> DownloadController? {
        if let selected = selectedDownloadJob[id], let controller = downloadJobs[selected] {
            return controller
        }
        if let newest = downloadJobs(for: id).first { return newest }
        return downloadDrafts[id]
    }

    func downloadJobs(for id: ModelID) -> [DownloadController] {
        downloadJobs.values
            .filter { $0.model == id }
            .sorted {
                if $0.updatedAt != $1.updatedAt { return $0.updatedAt > $1.updatedAt }
                return ($0.jobID?.rawValue.uuidString ?? "")
                    > ($1.jobID?.rawValue.uuidString ?? "")
            }
    }

    /// Every durable job, not only the jobs whose catalog row disappeared.
    /// Pending restored jobs are already reconciled against their persisted
    /// plan and summary, so their stopped state can be shown without attaching
    /// a controller or starting any I/O.
    var transfersForPresentation: [TransferPresentation] {
        var transfers: [DownloadJobID: TransferPresentation] = [:]
        for pending in pendingRestoredDownloads.values {
            let state = pending.state
            transfers[state.job] = TransferPresentation(
                job: state.job,
                model: state.plan.model,
                displayName: entry(state.plan.model)?.descriptor.displayName
                    ?? state.plan.model.rawValue,
                destinationPath: state.destinationPath,
                declaredBytes: state.plan.totalBytes,
                state: DownloadController.stoppedState(
                    from: pending.summary, plan: state.plan),
                updatedAt: state.updatedAt,
                hasCatalogEntry: entry(state.plan.model) != nil,
                reason: "Waiting for current model information.")
        }
        for (job, controller) in downloadJobs {
            let currentEntry = entry(controller.model)
            transfers[job] = TransferPresentation(
                job: job,
                model: controller.model,
                displayName: currentEntry?.descriptor.displayName ?? controller.model.rawValue,
                destinationPath: controller.destinationPath ?? "Destination unavailable",
                declaredBytes: controller.plan?.totalBytes
                    ?? currentEntry?.descriptor.totalBytes ?? 0,
                state: controller.state,
                updatedAt: controller.updatedAt,
                hasCatalogEntry: currentEntry != nil,
                reason: currentEntry == nil
                    ? "This download is not in the current model list." : nil)
        }
        return transfers.values.sorted {
            if $0.updatedAt != $1.updatedAt { return $0.updatedAt > $1.updatedAt }
            return $0.job.rawValue.uuidString < $1.job.rawValue.uuidString
        }
    }

    /// Downloads is an operational surface, not permanent settings chrome.
    /// Keep it out of navigation until there is a durable transfer or a
    /// recovery condition that needs the operator's attention.
    var showsDownloadsSection: Bool {
        !transfersForPresentation.isEmpty
            || isRecoveringDownloads
            || downloadSetupError != nil
            || downloadServiceError != nil
            || !downloadRecoveryIssues.isEmpty
            || canRetryDownloadRecovery
    }

    func downloadController(for job: DownloadJobID) -> DownloadController? {
        downloadJobs[job]
    }

    /// Durable jobs whose model currently has no storefront row. This includes
    /// restored jobs awaiting catalog metadata and attached jobs whose row
    /// later disappeared; a catalog refresh cannot make local work invisible.
    var savedTransfersForPresentation: [SavedTransferPresentation] {
        var transfers: [DownloadJobID: SavedTransferPresentation] = [:]
        for pending in pendingRestoredDownloads.values {
            let state = pending.state
            transfers[state.job] = SavedTransferPresentation(
                job: state.job,
                model: state.plan.model,
                destinationPath: state.destinationPath,
                state: pending.summary.state.rawValue,
                reason: "Waiting for catalog metadata.",
                updatedAt: state.updatedAt)
        }
        for (job, controller) in downloadJobs where entry(controller.model) == nil {
            transfers[job] = SavedTransferPresentation(
                job: job,
                model: controller.model,
                destinationPath: controller.destinationPath ?? "Destination unavailable",
                state: controller.state.phrase,
                reason: "This saved transfer is not in the current catalog.",
                updatedAt: controller.updatedAt)
        }
        return transfers.values
        .sorted {
            if $0.updatedAt != $1.updatedAt { return $0.updatedAt > $1.updatedAt }
            return $0.job.rawValue.uuidString < $1.job.rawValue.uuidString
        }
    }

    func selectDownloadJob(_ job: DownloadJobID) {
        guard let controller = downloadJobs[job] else { return }
        selectedDownloadJob[controller.model] = job
    }

    var downloadStartRefusal: String? {
        if isRecoveringDownloads { return "Restoring saved transfers. Try again when this finishes." }
        if let downloadSetupError {
            return "Downloads are unavailable because durable job storage could not be opened: "
                + downloadSetupError
        }
        return nil
    }

    func downloadStartRefusal(for id: ModelID) -> String? {
        if let downloadStartRefusal { return downloadStartRefusal }
        if localCopyRemovalInFlight?.model == id {
            return "Wait for this model copy to finish being removed."
        }
        if downloadRecoveryIssues.contains(where: { $0.model == id }) {
            return "Resolve this model's saved transfer before starting another copy."
        }
        if downloadJobs.values.contains(where: { $0.model == id && $0.blocksNewTransfer }) {
            return "Finish, cancel, or resolve the existing transfer before starting another copy."
        }
        return nil
    }

    /// Why the exact scanned copy cannot be removed right now. This is checked
    /// once when drawing the destructive action and again immediately before
    /// the filesystem capability is opened.
    func localCopyRemovalRefusal(for artifact: DiscoveredArtifact) -> String? {
        guard let id = artifact.model else {
            return "This folder is not identified as a supported model copy."
        }
        if let active = localCopyRemovalInFlight {
            return active.rootPath == artifact.rootPath
                ? "This copy is being removed."
                : "Wait for the current model removal to finish."
        }
        if let verification = installed.verificationInFlight {
            return verification == artifact.rootPath
                ? "Cancel or finish verification before removing this copy."
                : "Wait for the current verification to finish."
        }
        if run.isRunning, runningConversation?.settings.model == id {
            return "Stop the response before removing the model files it may be reading."
        }
        if isRecoveringDownloads {
            return "Wait for saved downloads to finish restoring."
        }
        if downloadRecoveryIssues.contains(where: { $0.model == id }) {
            return "Resolve this model's saved download before removing a local copy."
        }
        if downloadDrafts[id]?.protectsLocalCopyRemoval == true
            || downloadJobs.values.contains(where: {
                $0.model == id && $0.protectsLocalCopyRemoval
            })
            || pendingRestoredDownloads.values.contains(where: {
                $0.state.plan.model == id && Self.savedTransferProtectsLocalFiles($0.summary.state)
            })
        {
            return "Pause is resumable; cancel or finish this model's transfer before removing its files."
        }
        if installed.removalAccessRefusal(for: artifact) != nil {
            return "Reconnect or re-add the storage location before removing this copy."
        }
        return nil
    }

    func localCopyRemovalMessage(for artifact: DiscoveredArtifact) -> String? {
        localCopyRemovalOutcome[artifact.rootPath]
    }

    /// Remove one exact scanned copy, then retire only durable job records
    /// bound to that same destination. A cleanup failure cannot undo the
    /// filesystem result, so it is returned as an explicit warning rather than
    /// pretending the bytes remain recoverable.
    @discardableResult
    func removeLocalCopy(_ artifact: DiscoveredArtifact) async -> Bool {
        if let refusal = localCopyRemovalRefusal(for: artifact) {
            localCopyRemovalOutcome[artifact.rootPath] = refusal
            return false
        }
        localCopyRemovalInFlight = artifact
        localCopyRemovalOutcome[artifact.rootPath] = nil
        defer { localCopyRemovalInFlight = nil }

        do {
            let receipt = try await installed.remove(artifact)
            let cleanupWarning = forgetDownloadRecords(boundTo: artifact)
            var sentence =
                "Removed \(MRFormat.measuredBytes(receipt.removedRegularFileBytes)) in "
                + "\(MRFormat.grouped(Int(clamping: receipt.removedEntryCount))) items."
            if let cleanupWarning {
                sentence += " The files are gone, but saved download cleanup needs attention: "
                    + cleanupWarning
            }
            localCopyRemovalOutcome[artifact.rootPath] = sentence
            return true
        } catch {
            localCopyRemovalOutcome[artifact.rootPath] = String(describing: error)
            return false
        }
    }

    /// Returns a draft that has no durable identity yet. As soon as manager
    /// `start` returns, its callback promotes the controller into
    /// `downloadJobs` under the real job id.
    func prepareDownload(for id: ModelID) -> DownloadController? {
        guard downloadStartRefusal(for: id) == nil, let entry = entry(id) else { return nil }
        if let draft = downloadDrafts[id] { return draft }
        let controller = makeDownloadController(entry: entry)
        downloadDrafts[id] = controller
        return controller
    }

    func state(of id: ModelID) -> DownloadState { download(id)?.state ?? .notStarted }

    func downloadPlanForPresentation(_ id: ModelID) -> DownloadPlan? {
        if let resolved = download(id)?.plan { return resolved }
        #if DEBUG
            guard visualReviewNotice != nil, let entry = entry(id) else { return nil }
            return PlanBuilder.plan(for: entry)
        #else
            return nil
        #endif
    }

    /// Reconstruct durable jobs without starting network work. The manager's
    /// contract returns every nonterminal record paused and this method never
    /// calls `resume`; restoration is a visibility operation, not consent to
    /// continue a terabyte transfer.
    func restoreDownloads() async {
        guard restoresPersistedDownloads else {
            isRecoveringDownloads = false
            return
        }
        guard !didCompleteDownloadRestore else { return }
        if let downloadRestoreTask {
            _ = await downloadRestoreTask.value
            return
        }
        isRecoveringDownloads = true
        let task = Task { @MainActor [weak self] in
            guard let self else { return false }
            return await self.performDownloadRestore()
        }
        downloadRestoreTask = task
        let completed = await task.value
        downloadRestoreTask = nil
        didCompleteDownloadRestore = completed
        isRecoveringDownloads = false
        // Whichever of restore and the lightweight Hugging Face listing
        // finishes second must get a chance to match saved jobs to published
        // metadata. Neither operation waits for or starts the other.
        schedulePublishedResolutionForSavedTransfers()
    }

    var canRetryDownloadRecovery: Bool {
        restoresPersistedDownloads && !isRecoveringDownloads && !didCompleteDownloadRestore
    }

    /// Retry is explicit and remains a zero-network reconstruction. It never
    /// calls resume; a successful report replaces the same DownloadJobID.
    func retryDownloadRecovery() async {
        guard restoresPersistedDownloads, !isRecoveringDownloads else { return }
        didCompleteDownloadRestore = false
        await restoreDownloads()
    }

    private func performDownloadRestore() async -> Bool {
        guard let downloadStateStore else {
            downloadServiceError = "the persistent download state store was not configured"
            return false
        }
        downloadServiceError = nil
        downloadRecoveryIssues = []

        let savedBefore: [DownloadJobState]
        do {
            savedBefore = try downloadStateStore.all()
        } catch {
            downloadServiceError = "saved transfers could not be read: \(error)"
            return false
        }

        var savedBeforeByID: [DownloadJobID: DownloadJobState] = [:]
        var duplicateStateIDs = Set<DownloadJobID>()
        for state in savedBefore {
            if savedBeforeByID.updateValue(state, forKey: state.job) != nil {
                duplicateStateIDs.insert(state.job)
            }
        }
        guard duplicateStateIDs.isEmpty else {
            downloadRecoveryIssues = duplicateStateIDs.sorted {
                $0.rawValue.uuidString < $1.rawValue.uuidString
            }.map { job in
                let state = savedBeforeByID[job]
                return DownloadRecoveryIssue(
                    job: job, model: state?.plan.model,
                    destinationPath: state?.destinationPath ?? "",
                    reason: "more than one durable record claims this job identity")
            }
            downloadServiceError =
                "saved transfer records contain duplicate job identities; no jobs were restored"
            return false
        }

        let report: DownloadRestoreReport
        do {
            report = try await downloadManager.restorePersistedJobs()
        } catch {
            downloadServiceError = "saved transfers could not be restored: \(error)"
            return false
        }

        let savedAfter: [DownloadJobState]
        do {
            savedAfter = try downloadStateStore.all()
        } catch {
            downloadServiceError = "restored transfer state could not be read back: \(error)"
            return false
        }
        var states: [DownloadJobID: DownloadJobState] = [:]
        for state in savedAfter {
            guard states.updateValue(state, forKey: state.job) == nil else {
                downloadRecoveryIssues.append(
                    DownloadRecoveryIssue(
                        job: state.job, model: state.plan.model,
                        destinationPath: state.destinationPath,
                        reason: "restoration produced duplicate durable records for this job"))
                downloadServiceError = "saved transfer state became ambiguous during restoration"
                return false
            }
        }
        var summaries: [DownloadJobID: DownloadJobSummary] = [:]
        for summary in await downloadManager.jobs() {
            guard summaries.updateValue(summary, forKey: summary.job) == nil else {
                downloadRecoveryIssues.append(
                    DownloadRecoveryIssue(
                        job: summary.job, model: summary.model, destinationPath: "",
                        reason: "the manager reported this job more than once"))
                downloadServiceError = "restored transfer state is ambiguous"
                return false
            }
        }

        // Treat the manager report as an exact reconciliation, not a hint. A
        // corrupt or hostile conformer must not be able to make a durable job
        // disappear, report it in two incompatible buckets, or advertise a
        // restored job without the post-restore evidence the UI will render.
        var structuralReasons: [DownloadJobID: [String]] = [:]
        func noteStructuralFailure(_ job: DownloadJobID, _ reason: String) {
            if !(structuralReasons[job]?.contains(reason) ?? false) {
                structuralReasons[job, default: []].append(reason)
            }
        }

        let savedIDs = Set(savedBeforeByID.keys)
        let postRestoreIDs = Set(states.keys)
        for job in postRestoreIDs.subtracting(savedIDs) {
            noteStructuralFailure(
                job, "restoration created durable state for a job that did not exist before")
        }
        for job in savedIDs.subtracting(postRestoreIDs) {
            noteStructuralFailure(
                job, "restoration removed this durable job instead of accounting for it")
        }
        for job in savedIDs.intersection(postRestoreIDs) {
            guard let saved = savedBeforeByID[job], let state = states[job] else { continue }
            if state.plan != saved.plan {
                noteStructuralFailure(job, "the manager changed this job's immutable download plan")
            }
            if URL(fileURLWithPath: state.destinationPath, isDirectory: true)
                .standardizedFileURL.path
                != URL(fileURLWithPath: saved.destinationPath, isDirectory: true)
                .standardizedFileURL.path
            {
                noteStructuralFailure(job, "the manager changed this job's destination path")
            }
            if state.artifactRootPath != saved.artifactRootPath
                || state.artifactRootDevice != saved.artifactRootDevice
                || state.artifactRootInode != saved.artifactRootInode
            {
                noteStructuralFailure(job, "the manager changed this job's physical artifact identity")
            }
            if state.options != saved.options {
                noteStructuralFailure(job, "the manager changed this job's transfer options")
            }
            if state.createdAt != saved.createdAt {
                noteStructuralFailure(job, "the manager changed this job's creation time")
            }
        }
        var restoredIDs = Set<DownloadJobID>()
        for job in report.restoredJobs where !restoredIDs.insert(job).inserted {
            noteStructuralFailure(job, "the manager reported this job restored more than once")
        }
        var failureIDs = Set<DownloadJobID>()
        var failureByID: [DownloadJobID: DownloadRestoreFailure] = [:]
        for failure in report.failures {
            if !failureIDs.insert(failure.job).inserted {
                noteStructuralFailure(
                    failure.job, "the manager reported more than one failure for this job")
            }
            if failureByID[failure.job] == nil { failureByID[failure.job] = failure }
            if let saved = savedBeforeByID[failure.job] {
                let reported = URL(fileURLWithPath: failure.destinationPath, isDirectory: true)
                    .standardizedFileURL.path
                let durable = URL(fileURLWithPath: saved.destinationPath, isDirectory: true)
                    .standardizedFileURL.path
                if reported != durable {
                    noteStructuralFailure(
                        failure.job, "the restore failure names a different destination path")
                }
            }
        }
        for job in failureIDs {
            if let before = savedBeforeByID[job], let after = states[job], before != after {
                noteStructuralFailure(
                    job, "the manager changed durable state while reporting restoration failure")
            }
        }
        for job in restoredIDs.intersection(failureIDs) {
            noteStructuralFailure(job, "the manager reported this job as both restored and failed")
        }

        let accountedIDs = restoredIDs.union(failureIDs)
        for job in savedIDs.subtracting(accountedIDs) {
            noteStructuralFailure(job, "the manager omitted this saved job from its restore report")
        }
        for job in accountedIDs.subtracting(savedIDs) {
            noteStructuralFailure(job, "the manager reported a job that was not in durable state")
        }
        for job in restoredIDs.subtracting(Set(summaries.keys)) {
            noteStructuralFailure(job, "the manager restored this job without a summary")
        }
        for job in Set(summaries.keys).subtracting(restoredIDs) {
            noteStructuralFailure(
                job, "the manager exposed a summary for a job not reported restored")
        }
        for job in restoredIDs {
            guard let state = states[job] else {
                noteStructuralFailure(
                    job, "the manager restored this job without post-restore durable state")
                continue
            }
            guard let summary = summaries[job] else { continue }
            if summary.model != state.plan.model {
                noteStructuralFailure(job, "the restored summary names a different model")
            }
            if summary.progress.job != job {
                noteStructuralFailure(job, "the restored summary carries another job identity")
            }
            if summary.progress.planTotalBytes != state.plan.totalBytes {
                noteStructuralFailure(job, "the restored summary disagrees with the download plan")
            }
            if let reason = Self.invalidRestoredProgressReason(
                summary.progress, plan: state.plan)
            {
                noteStructuralFailure(job, reason)
            }
            if summary.state != state.runState {
                noteStructuralFailure(
                    job,
                    "the restored summary says \(summary.state.rawValue), but durable state says "
                        + (state.runState?.rawValue ?? "<missing>"))
            }
            if summary.state != .paused, summary.state != .failed,
                summary.state != .cancelled
            {
                noteStructuralFailure(
                    job,
                    "the manager returned an active or unverified lifecycle state after a "
                        + "zero-network restore")
            }
        }

        guard structuralReasons.isEmpty else {
            downloadRecoveryIssues = structuralReasons.keys.sorted {
                $0.rawValue.uuidString < $1.rawValue.uuidString
            }.map { job in
                let state = states[job] ?? savedBeforeByID[job]
                return DownloadRecoveryIssue(
                    job: job,
                    model: state?.plan.model,
                    destinationPath: failureByID[job]?.destinationPath
                        ?? state?.destinationPath ?? "",
                    reason: structuralReasons[job, default: []].joined(separator: "; "))
            }
            downloadServiceError =
                "saved transfer restoration did not account for every durable job"
            return false
        }

        downloadRecoveryIssues = report.failures.map { failure in
            DownloadRecoveryIssue(
                job: failure.job,
                model: states[failure.job]?.plan.model
                    ?? savedBeforeByID[failure.job]?.plan.model,
                destinationPath: failure.destinationPath,
                reason: failure.reason)
        }
        let restorationIncomplete = !report.failures.isEmpty
        if restorationIncomplete {
            downloadServiceError =
                "\(report.failures.count) saved transfer"
                + (report.failures.count == 1 ? "" : "s")
                + " could not be restored"
        }

        // A retry is a fresh exact reconciliation against durable state. Keep
        // only still-restored pending identities; failures remain visible in
        // downloadRecoveryIssues rather than masquerading as successful rows.
        pendingRestoredDownloads = [:]

        for job in report.restoredJobs {
            // Exact reconciliation above proves both values exist. Keeping the
            // unwrap guarded makes a future refactor fail visibly rather than
            // trapping or falling back to stale pre-restore state.
            guard let state = states[job], let summary = summaries[job] else {
                downloadServiceError = "saved transfer restoration lost validated state"
                return false
            }
            if let entry = entry(state.plan.model) {
                attachRestoredDownload(state: state, summary: summary, entry: entry)
                pendingRestoredDownloads.removeValue(forKey: state.job)
            } else {
                pendingRestoredDownloads[state.job] = PendingRestoredDownload(
                    state: state, summary: summary)
            }
        }
        if restorationIncomplete, downloadServiceError == nil {
            downloadServiceError = "saved transfer restoration was incomplete"
        }
        return !restorationIncomplete
    }

    /// A restored manager summary crosses a public protocol boundary. Validate
    /// every value that the product will index, divide or present before a
    /// controller can consume it; durable state and the immutable plan remain
    /// the authority when a conformer returns malformed telemetry.
    private static func invalidRestoredProgressReason(
        _ progress: DownloadProgress, plan: DownloadPlan
    ) -> String? {
        guard progress.accountsForEveryByte else {
            return "the restored progress byte buckets do not account for the download plan"
        }
        guard progress.filesTotal == plan.files.count else {
            return "the restored progress file count disagrees with the download plan"
        }
        guard progress.filesVerified >= 0, progress.filesFailed >= 0,
            progress.filesVerified <= progress.filesTotal,
            progress.filesFailed <= progress.filesTotal - progress.filesVerified
        else {
            return "the restored progress carries impossible file counters"
        }
        guard progress.wastedBytes <= progress.networkBytes else {
            return "the restored progress reports more wasted bytes than network bytes"
        }
        guard progress.instantaneousBytesPerSecond.isFinite,
            progress.instantaneousBytesPerSecond >= 0,
            progress.smoothedBytesPerSecond.isFinite,
            progress.smoothedBytesPerSecond >= 0,
            progress.elapsed.isFinite, progress.elapsed >= 0
        else {
            return "the restored progress carries an invalid rate or elapsed time"
        }
        if let estimate = progress.estimatedTimeRemaining,
            !estimate.isFinite || estimate < 0
        {
            return "the restored progress carries an invalid time estimate"
        }
        return nil
    }

    private static func savedTransferProtectsLocalFiles(
        _ state: DownloadJobSummary.JobState
    ) -> Bool {
        switch state {
        case .planned, .running, .paused, .verifying:
            return true
        case .finished, .failed, .cancelled:
            return false
        }
    }

    private static func sameFilesystemPath(_ lhs: String?, _ rhs: String) -> Bool {
        guard let lhs else { return false }
        return URL(fileURLWithPath: lhs, isDirectory: true).standardizedFileURL.path
            == URL(fileURLWithPath: rhs, isDirectory: true).standardizedFileURL.path
    }

    private static func savedState(
        _ state: DownloadJobState, isBoundTo artifact: DiscoveredArtifact
    ) -> Bool {
        guard state.plan.model == artifact.model else { return false }
        return sameFilesystemPath(state.artifactRootPath, artifact.rootPath)
            || sameFilesystemPath(state.destinationPath, artifact.rootPath)
    }

    /// Returns a warning only when durable bookkeeping could not be fully
    /// retired. In-memory controllers are always removed after the bytes are
    /// gone so the current process never presents a stale Ready state.
    private func forgetDownloadRecords(boundTo artifact: DiscoveredArtifact) -> String? {
        var jobsToForget = Set<DownloadJobID>()
        var warnings: [String] = []

        if let downloadStateStore {
            do {
                for state in try downloadStateStore.all()
                where Self.savedState(state, isBoundTo: artifact) {
                    jobsToForget.insert(state.job)
                }
                for job in jobsToForget {
                    do {
                        try downloadStateStore.remove(job)
                    } catch {
                        warnings.append("job \(job.rawValue.uuidString): \(error)")
                    }
                }
            } catch {
                warnings.append("saved jobs could not be listed: \(error)")
            }
        }

        for (job, controller) in downloadJobs
        where controller.model == artifact.model
            && Self.sameFilesystemPath(controller.destinationPath, artifact.rootPath)
        {
            jobsToForget.insert(job)
        }
        for (job, pending) in pendingRestoredDownloads
        where Self.savedState(pending.state, isBoundTo: artifact) {
            jobsToForget.insert(job)
        }

        for job in jobsToForget {
            downloadJobs.removeValue(forKey: job)
            pendingRestoredDownloads.removeValue(forKey: job)
            downloadRecoveryIssues.removeAll { $0.job == job }
        }
        if let id = artifact.model, let selected = selectedDownloadJob[id],
            jobsToForget.contains(selected)
        {
            selectedDownloadJob[id] = downloadJobs(for: id).first?.jobID
        }
        if let id = artifact.model, let entry = entry(id) {
            ensureDownloadDraft(for: entry)
        }
        return warnings.isEmpty ? nil : warnings.joined(separator: "; ")
    }

    private func makeDownloadController(entry: CatalogEntry) -> DownloadController {
        let controller = DownloadController(entry: entry, manager: downloadManager)
        controller.onJobRegistered = { [weak self, weak controller] job in
            guard let self, let controller else { return }
            self.downloadDrafts[controller.model] = nil
            self.downloadJobs[job] = controller
            self.selectedDownloadJob[controller.model] = job
        }
        if downloadStateStore != nil {
            controller.onArtifactVerified = { [weak self] job, destination in
                self?.registerDownloadedArtifact(job: job, at: destination)
            }
        }
        return controller
    }

    private func ensureDownloadDraft(for entry: CatalogEntry) {
        guard downloadDrafts[entry.id] == nil,
            !downloadJobs.values.contains(where: { $0.model == entry.id })
        else { return }
        downloadDrafts[entry.id] = makeDownloadController(entry: entry)
    }

    private func attachRestoredDownload(
        state: DownloadJobState, summary: DownloadJobSummary, entry: CatalogEntry
    ) {
        guard downloadJobs[state.job] == nil else { return }
        let controller = DownloadController(
            entry: entry, manager: downloadManager, restoredState: state, summary: summary)
        controller.onArtifactVerified = { [weak self] job, destination in
            self?.registerDownloadedArtifact(job: job, at: destination)
        }
        downloadJobs[state.job] = controller
        downloadDrafts[entry.id] = nil
        if let selected = selectedDownloadJob[entry.id],
            let existing = downloadJobs[selected], existing.updatedAt >= controller.updatedAt
        {
            return
        }
        selectedDownloadJob[entry.id] = state.job
    }

    private func attachPendingRestoredDownloads() {
        var stillPending = pendingRestoredDownloads
        for pending in pendingRestoredDownloads.values {
            let state = pending.state
            guard let entry = entry(state.plan.model) else { continue }
            attachRestoredDownload(state: state, summary: pending.summary, entry: entry)
            stillPending.removeValue(forKey: state.job)
        }
        pendingRestoredDownloads = stillPending
    }

    private func registerDownloadedArtifact(job: DownloadJobID, at destination: URL) {
        // Concurrent jobs may point at different drives, so the durable
        // per-job bookmark is the only authority for this callback. The picker
        // does not create a second global destination grant.
        do {
            guard let state = try downloadStateStore?.load(job),
                let bookmark = state.destinationBookmark
            else {
                throw StorageError.bookmarkDataUnresolvable(
                    reason: "job \(job.rawValue) has no durable destination bookmark")
            }
            let resolved = try storage.resolveBookmarkData(bookmark)
            defer { resolved.scope.release() }
            let recorded = URL(fileURLWithPath: state.destinationPath, isDirectory: true)
                .standardizedFileURL
            guard resolved.scope.url.standardizedFileURL == destination.standardizedFileURL,
                recorded == destination.standardizedFileURL
            else {
                throw StorageError.volumeUnavailable(path: destination.path)
            }
            installed.addLocation(destination)
            switch installed.lastAddOutcome {
            case .added, .alreadyRegistered:
                artifactRegistrationError = nil
            case .failed(let reason):
                artifactRegistrationError =
                    "The verified download could not be added to storage: \(reason)"
            case .cancelled:
                artifactRegistrationError =
                    "The verified download could not be added to storage."
            case nil:
                artifactRegistrationError =
                    "The verified download did not produce a storage registration result."
            }
        } catch {
            artifactRegistrationError =
                "The verified download could not be added to storage: \(error)"
        }
    }

    /// The runner's own capabilities, so a settings screen can offer exactly the
    /// knobs and the token ceiling this model honours instead of offering
    /// something that will be refused later.
    func capabilities(for id: ModelID) -> RunnerCapabilities? {
        guard entry(id) != nil else { return nil }
        return runtimes.runtime(for: id)?.capabilities
    }

    /// Runtime fitness is a property of what this binary actually linked, not
    /// of catalogue metadata. A descriptor may advertise `decodeRunner` as a
    /// known capability while the current App has no verified runner/tokenizer
    /// pair; product UI must follow the latter.
    func hasVerifiedRuntime(for id: ModelID) -> Bool {
        runtimes.runtime(for: id)?.trust.isVerified == true
    }

    func effectiveProductFitness(for entry: CatalogEntry) -> PlatformFitness {
        let descriptor = entry.descriptor
        guard let runtime = runtimes.runtime(for: entry.id), runtime.trust.isVerified else {
            return PlatformFitness(
                verdict: .noRunner,
                reason:
                    "This version of Minirun can find, verify, and download "
                    + "\(descriptor.displayName), but cannot run it yet.",
                requiredFreeBytes: descriptor.totalBytes,
                headroomBytes: nil)
        }
        guard runtime.capabilities.layout == descriptor.layout else {
            return PlatformFitness(
                verdict: .noRunner,
                reason:
                    "The verified runtime does not support this catalog artifact layout.",
                requiredFreeBytes: descriptor.totalBytes,
                headroomBytes: nil)
        }

        let minimum = runtime.capabilities.minimumBudgetBytes
        let headroom = Self.saturatingSignedDifference(
            device.processMemoryBudgetBytes, minimum)
        guard device.processMemoryBudgetBytes >= minimum else {
            return PlatformFitness(
                verdict: .refused,
                reason:
                    "\(descriptor.displayName) requires a verified runtime budget of "
                    + "\(MRFormat.bytesDecimal(minimum)); this device offers "
                    + "\(MRFormat.bytesDecimal(device.processMemoryBudgetBytes)).",
                requiredFreeBytes: descriptor.totalBytes,
                headroomBytes: headroom)
        }
        if device.platform == .iOS && !device.hasIncreasedMemoryLimitEntitlement {
            return PlatformFitness(
                verdict: .runnableWithCaveats,
                reason:
                    "The verified runtime fits the reported budget, but iOS may lower the "
                    + "available memory without the increased-memory-limit entitlement.",
                requiredFreeBytes: descriptor.totalBytes,
                headroomBytes: headroom)
        }
        return PlatformFitness(
            verdict: .runnable,
            reason: "A verified runtime is available and fits this device's reported budget.",
            requiredFreeBytes: descriptor.totalBytes,
            headroomBytes: headroom)
    }

    private static func saturatingSignedDifference(_ lhs: UInt64, _ rhs: UInt64) -> Int64 {
        if lhs >= rhs {
            let difference = lhs - rhs
            return difference > UInt64(Int64.max) ? .max : Int64(difference)
        }
        let difference = rhs - lhs
        return difference > UInt64(Int64.max) ? .min : -Int64(difference)
    }

    /// Change the model for an existing chat as one state transition. Values
    /// the new runner cannot honour are removed in the same persisted edit, so
    /// no render or request can observe a V4-only knob attached to K3 (or the
    /// reverse).
    func selectModel(_ id: ModelID, in conversationID: UUID) {
        guard !isRunning(conversationID), modelAvailability(for: id).isAvailable,
            let capabilities = capabilities(for: id),
            let current = conversation(conversationID)
        else { return }
        let budget = Self.restatedBudget(
            current.settings.memoryBudgetBytes,
            wasDerived: budgetIsDerived(in: current),
            movingTo: defaultBudgetPlan(for: id))
        update(conversationID) { conversation in
            conversation.settings.model = id
            if let budget { conversation.settings.memoryBudgetBytes = budget }
            Self.constrain(&conversation.settings, to: capabilities)
        }
        preparedWorkingSetReserves.removeValue(forKey: conversationID)
    }

    /// **The rule for a budget when the chat's model changes.**
    ///
    /// A budget belongs to a model: 8.00 GB is K3's floor and more than three
    /// times V4's, so carrying one model's number onto another is carrying a
    /// statement about the wrong artifact. But the number may also be one the
    /// operator chose deliberately, and re-deriving that would be the settings
    /// panel overruling them.
    ///
    /// So the two cases are separated by where the current value came from:
    ///
    /// * **Derived** — it still equals the old model's new-chat default or one
    ///   of that model's own presets, so nobody has personally stated it. It is
    ///   re-derived to the new model's default, which is that model's own
    ///   preset.
    /// * **Customized** — anything else. It is kept, and moved only when the
    ///   new model's floor or this device's ceiling makes it impossible to
    ///   state; it is never silently jumped to a different preset.
    private static func restatedBudget(
        _ current: UInt64, wasDerived: Bool, movingTo target: BudgetPlan?
    ) -> UInt64? {
        guard let target else { return nil }
        if wasDerived { return target.budgetBytes }
        let ceiling = target.deviceCeilingBytes
        let floor = target.profile.refusalThresholdBytes
        guard floor <= ceiling else { return nil }
        return min(ceiling, max(floor, current))
    }

    /// Whether this chat's budget is still one the app derived — the model's
    /// new-chat default, or one of the presets the dial offers for it.
    private func budgetIsDerived(in conversation: Conversation) -> Bool {
        let stated = conversation.settings.memoryBudgetBytes
        if defaultBudgetPlan(for: conversation.settings.model)?.budgetBytes == stated {
            return true
        }
        guard let plan = budgetPlan(for: conversation) else { return false }
        return BudgetPlan.Preset.allCases.contains { plan.budget(for: $0) == stated }
    }

    /// The equivalent transition for the defaults copied into a new chat.
    func setDefaultModel(_ id: ModelID) {
        guard modelAvailability(for: id).isAvailable,
            let capabilities = capabilities(for: id)
        else { return }
        var updated = defaults
        updated.model = id
        if !capabilities.supportedKnobs.contains("deterministicReadAheadLayers") {
            updated.deterministicReadAheadLayers = nil
        }
        defaults = updated
    }

    private static func constrain(
        _ settings: inout ConversationSettings, to capabilities: RunnerCapabilities
    ) {
        settings.maximumNewTokens = min(
            max(1, settings.maximumNewTokens), max(1, capabilities.maximumNewTokens))
        if !capabilities.supportedKnobs.contains("deterministicReadAheadLayers") {
            settings.deterministicReadAheadLayers = nil
        }
        if !capabilities.supportedKnobs.contains("expertReadAhead") {
            settings.expertReadAhead = nil
        }
        if !capabilities.supportedKnobs.contains("expertPoolSlots") {
            settings.expertPoolSlots = nil
        }
    }

    /// Discovery plus runner registration, in that order: the catalog may say a
    /// model exists without a single byte of it being reachable on this device.
    func modelAvailability(for id: ModelID) -> ModelRunAvailability {
        guard let entry = entry(id) else { return .missingFromCatalog }
        let installations = installed.installations(of: id)
        guard !installations.isEmpty else { return .notFoundByScan }

        let mounted = installed.report.locations
            .filter(\.isMounted)
            .flatMap { $0.artifacts(for: id) }
        guard !mounted.isEmpty else { return .storageUnavailable }
        let complete = mounted.filter { $0.isComplete != false }
        guard !complete.isEmpty else {
            return .incompleteArtifact
        }
        let verified = complete.filter { $0.verification == .fullyVerified }
        guard !verified.isEmpty else {
            return .unverifiedArtifact
        }
        guard let runtime = runtimes.runtime(for: entry.id) else {
            return .runtimeUnavailable
        }
        guard verified.contains(where: runtime.accepts) else {
            return .runtimeFilesUnavailable
        }
        return .available
    }

    func canRun(_ conversation: Conversation) -> Bool {
        localCopyRemovalInFlight?.model != conversation.settings.model
            && modelAvailability(for: conversation.settings.model).isAvailable
            && budgetPlan(for: conversation)?.isRunnable == true
    }

    var canCreateConversation: Bool { modelAvailability(for: defaults.model).isAvailable }

    var canCreateReadyConversation: Bool {
        if case .ready = chatReadiness { return true }
        return false
    }

    var newConversationUnavailableReason: String? {
        let availability = modelAvailability(for: defaults.model)
        return availability.isAvailable ? nil : availability.reason
    }

    /// A restart-safe first-run path, projected only from current authorities.
    /// Prefer the stated default when it is runnable; otherwise offer any
    /// runnable model without silently changing that default.
    var chatReadiness: ChatReadiness {
        if modelAvailability(for: defaults.model).isAvailable {
            return .ready(model: defaults.model)
        }
        if let available = entries.first(where: {
            modelAvailability(for: $0.id).isAvailable
        }) {
            return .ready(model: available.id)
        }

        if entries.isEmpty {
            if isRefreshing || isRefreshingPublishedModels || isAwaitingInitialCatalog {
                return .loadingCatalog
            }
            if catalogError != nil { return .catalogUnavailable }
            return .needsModel
        }

        if installed.isScanning { return .scanningStorage }

        let mountedArtifacts = installed.report.locations
            .filter(\.isMounted)
            .flatMap(\.artifacts)
        let verificationCandidate = mountedArtifacts.first { artifact in
            guard artifact.isComplete != false, artifact.verification != .fullyVerified,
                let id = artifact.model, let runtime = runtimes.runtime(for: id),
                runtime.accepts(artifact)
            else { return false }
            return entry(id) != nil
        }
        if let id = verificationCandidate?.model {
            return .needsVerification(model: id)
        }

        if !installed.hasLocation { return .needsStorage }
        return .needsModel
    }

    /// Both model pickers contain only chat runtimes backed by a completely
    /// verified artifact in the latest storage scan. Models may still appear in
    /// the Models manager when this build cannot chat with them; storage and
    /// runtime support are deliberately separate truths. An unavailable
    /// persisted value remains in the conversation document, but never leaks
    /// into a menu whose promise is "models this chat can run now".
    func modelPickerChoices(current _: ModelID) -> [ModelPickerChoice] {
        let catalogModels = Set(entries.map(\.id))
        var seen: Set<ModelID> = []
        var choices: [ModelPickerChoice] = []

        for artifact in installed.report.locations.flatMap(\.artifacts) {
            guard let id = artifact.model else { continue }
            guard catalogModels.contains(id) else { continue }
            guard let catalogEntry = entry(id) else { continue }
            let availability = modelAvailability(for: id)
            guard availability.isAvailable else { continue }
            guard seen.insert(id).inserted else { continue }
            choices.append(
                ModelPickerChoice(
                    id: "model.\(id.rawValue)",
                    modelID: id,
                    displayName: catalogEntry.descriptor.displayName,
                    availability: availability))
        }

        return choices.sorted { lhs, rhs in
            if lhs.modelID == nil, rhs.modelID != nil { return false }
            if lhs.modelID != nil, rhs.modelID == nil { return true }
            return lhs.displayName.localizedStandardCompare(rhs.displayName) == .orderedAscending
        }
    }

    /// The published catalog ordered for the remote model browser.
    var visibleEntries: [CatalogEntry] {
        switch catalogSort {
        case .size:
            return entries.sorted { $0.descriptor.totalBytes > $1.descriptor.totalBytes }
        case .name:
            return entries.sorted { $0.descriptor.displayName < $1.descriptor.displayName }
        }
    }

    /// A bundled descriptor can recognize a scanned artifact, but cannot prove
    /// that a repository is currently published. Only the explicitly labelled
    /// DEBUG review fixture may render bundled sample rows.
    var catalogEntriesForPresentation: [CatalogEntry] {
        guard snapshot.origin != .bundled || visualReviewNotice != nil else { return [] }
        return visibleEntries
    }

    /// Lightweight remote rows. Unlike `catalogEntriesForPresentation`, these
    /// never imply that file counts or sizes have been resolved.
    ///
    /// The bundled catalogue is a recognition table, not proof that a
    /// repository is published right now, and it is the table a launch without
    /// a saved catalogue starts from. Gate these rows on origin exactly as the
    /// storefront rows are gated, so seeding recognition never turns into a
    /// claim about what Hugging Face is currently serving.
    var publishedModelsForPresentation: [PublishedModelReference] {
        guard publishedModelsOrigin != .bundled || visualReviewNotice != nil else { return [] }
        return publishedModels.sorted {
            $0.displayName.localizedStandardCompare($1.displayName) == .orderedAscending
        }
    }

    /// A fully resolved entry is reusable only at the exact listed commit.
    /// An older cached descriptor remains valid for its verified local bytes,
    /// but it must not supply sizes for a newer remote row.
    func resolvedEntry(for published: PublishedModelReference) -> CatalogEntry? {
        guard let entry = entry(published.id),
            entry.descriptor.source.repo == published.repository
        else { return nil }
        return entry
    }

    /// The Models settings page is an inventory of local artifacts, not a
    /// storefront. Hugging Face discovery has its own explicit browser sheet.
    /// Keep unverified and unsupported local containers visible: both are
    /// storage facts a person may need to inspect or repair.
    var localEntriesForPresentation: [CatalogEntry] {
        let localIDs = Set(
            installed.report.locations
                .flatMap(\.artifacts)
                .compactMap(\.model))
        // A remote listing requires live/cache publication evidence, but a
        // local row has stronger evidence of its own: discovery found bytes
        // whose index matches this descriptor. Keep that row available when
        // offline even if the only recognition table in memory is bundled.
        return visibleEntries.filter { localIDs.contains($0.id) }
    }

    /// Before the first live response there is no catalog evidence yet. That
    /// is a loading state, not proof that a persisted conversation's model has
    /// disappeared from Hugging Face.
    var isAwaitingInitialCatalog: Bool {
        visualReviewNotice == nil && !hasCompletedPublishedModelLoad && catalogError == nil
    }

    /// Refresh the fast Hugging Face repository index. This operation never
    /// walks an artifact tree and therefore cannot itself update verification,
    /// byte counts or download plans.
    func refreshPublishedModels() async {
        guard allowsLiveCatalogRefresh else { return }
        guard !isRefreshingPublishedModels else { return }
        didAttemptPublishedModelLoadThisLaunch = true
        isRefreshingPublishedModels = true
        catalogError = nil
        defer {
            isRefreshingPublishedModels = false
            hasCompletedPublishedModelLoad = true
        }
        do {
            publishedModels = try await catalogService.publishedModels()
            publishedModelsOrigin = .live
            attemptedLocalPublishedResolutions.removeAll()
            schedulePublishedResolutionForLocalModels(in: installed.report)
            schedulePublishedResolutionForSavedTransfers()
        } catch let error as CatalogError {
            catalogError = error
        } catch {
            catalogError = .malformedResponse("\(error)")
        }
    }

    /// Cache-first, stale-while-revalidate entry for Find Models. Opening the
    /// browser updates its lightweight index once per launch; reopening the
    /// sheet never repeats the request. The explicit Refresh button remains
    /// available after that attempt.
    func refreshPublishedModelsIfNeeded() async {
        guard !didAttemptPublishedModelLoadThisLaunch else { return }
        await refreshPublishedModels()
    }

    /// Resolve one selected remote row into complete tree-derived metadata and
    /// merge it with the offline runtime cache. Multiple UI tasks asking for
    /// the same row coalesce at the MainActor boundary.
    @discardableResult
    func resolvePublishedModel(_ published: PublishedModelReference) async throws
        -> CatalogEntry
    {
        if let resolved = resolvedEntry(for: published) { return resolved }
        if publishedResolutionsInFlight.contains(published) {
            while publishedResolutionsInFlight.contains(published) {
                try Task.checkCancellation()
                try await Task.sleep(for: .milliseconds(25))
            }
            if let resolved = resolvedEntry(for: published) { return resolved }
            throw publishedResolutionFailures[published]
                ?? CatalogError.unknownModel(published.id)
        }

        publishedResolutionFailures.removeValue(forKey: published)
        publishedResolutionsInFlight.insert(published)
        defer { publishedResolutionsInFlight.remove(published) }
        do {
            let result = try await catalogService.resolvePublishedModel(published)
            applyCatalog(result.snapshot, replacingPublishedModels: false)
            catalogCacheWarning = result.cacheWarning
            guard let resolved = resolvedEntry(for: published) else {
                throw CatalogError.malformedResponse(
                    "resolved model \(published.repository.repoID) was not retained")
            }
            return resolved
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as CatalogError {
            publishedResolutionFailures[published] = error
            throw error
        } catch {
            let failure = CatalogError.malformedResponse("\(error)")
            publishedResolutionFailures[published] = failure
            throw failure
        }
    }

    /// Ask MinirunKit's real catalog service. The live snapshot becomes the
    /// visible catalog and the scanner's identity table in one operation; any
    /// disagreement with the build's bundled baseline remains visible.
    func refreshCatalog() async {
        guard allowsLiveCatalogRefresh else { return }
        guard !isRefreshing else { return }
        isRefreshing = true
        catalogError = nil
        catalogCacheWarning = nil
        defer { isRefreshing = false }
        do {
            let result = try await catalogService.refreshResult()
            applyCatalog(result.snapshot)
            catalogCacheWarning = result.cacheWarning
        } catch let error as CatalogError {
            catalogError = error
            recoverCachedCatalog(after: error)
        } catch {
            let failure = CatalogError.malformedResponse("\(error)")
            catalogError = failure
            recoverCachedCatalog(after: failure)
        }
    }

    private func applyCatalog(
        _ fresh: ModelCatalogSnapshot, replacingPublishedModels: Bool = true
    ) {
        // Models is the live Hugging Face catalogue. A locally built model
        // carried by a bundled snapshot was not discovered by this request and
        // must not survive into a previously written cache as a visible preset.
        let fresh = ProductCatalogPolicy.applying(to: fresh)
        preparedArtifactCensuses = preparedArtifactCensuses.filter { id, _ in
            snapshot.descriptor(id) == fresh.descriptor(id)
        }
        preparedDeepSeekV4Ladders = preparedDeepSeekV4Ladders.filter { id, _ in
            snapshot.descriptor(id) == fresh.descriptor(id)
        }
        let freshEntries = AppCatalogAdapter.entries(from: fresh)
        snapshot = fresh
        entries = freshEntries
        if replacingPublishedModels {
            publishedModels = Self.publishedReferences(from: fresh)
            publishedModelsOrigin = fresh.origin
            hasCompletedPublishedModelLoad = true
        }
        divergences = fresh.divergence(
            from: ProductCatalogPolicy.applying(to: ModelCatalog.bundled))

        // A catalog refresh must not erase a transfer that may have been in
        // progress for days. Existing controllers keep their jobs and immutable
        // plans; new rows get a controller, and a changed/removed descriptor is
        // marked stale on the existing controller rather than reset.
        // The simulated review backend has a local planning table. The real
        // manager plans directly from the live descriptor's pinned HF tree and
        // has no catalog table to mutate.
        #if DEBUG
            (downloadManager as? MockDownloadManager)?.updateEntries(freshEntries)
        #endif
        let freshByID = Dictionary(uniqueKeysWithValues: freshEntries.map { ($0.id, $0) })
        for controller in downloadJobs.values {
            controller.updateCatalogEntry(freshByID[controller.model])
        }
        for id in Array(downloadDrafts.keys) {
            guard let controller = downloadDrafts[id] else { continue }
            if let freshEntry = freshByID[id] {
                controller.updateCatalogEntry(freshEntry)
            } else if controller.hasCatalogBoundState {
                controller.updateCatalogEntry(nil)
            } else {
                downloadDrafts.removeValue(forKey: id)
            }
        }
        attachPendingRestoredDownloads()
        for entry in freshEntries { ensureDownloadDraft(for: entry) }

        var updatedDefaults = defaults
        for entry in freshEntries where updatedDefaults.budget(for: entry.id) == nil {
            updatedDefaults.setBudget(
                entry.memory.requiredMinimumBudgetBytes
                    ?? entry.memory.onRecordMinimumBudgetBytes
                    ?? entry.memory.arithmeticFloorBytes,
                for: entry.id)
        }
        defaults = updatedDefaults
        installed.updateCatalog(fresh)
        assessments.updateCatalog(fresh)
    }

    private static func publishedReferences(
        from snapshot: ModelCatalogSnapshot
    ) -> [PublishedModelReference] {
        snapshot.models.compactMap { descriptor in
            guard let repository = descriptor.source.repo else { return nil }
            return PublishedModelReference(
                id: descriptor.id, displayName: descriptor.displayName,
                repository: repository)
        }
    }

    private func schedulePublishedResolutionForLocalModels(in report: DiscoveryReport) {
        let localIDs = Set(report.locations.flatMap(\.artifacts).compactMap(\.model))
        schedulePublishedResolutions(for: localIDs, report: report)
    }

    private func schedulePublishedResolutionForSavedTransfers() {
        let transferIDs = Set(downloadJobs.values.map(\.model))
            .union(pendingRestoredDownloads.values.map { $0.state.plan.model })
        schedulePublishedResolutions(for: transferIDs, report: installed.report)
    }

    private func schedulePublishedResolutions(
        for modelIDs: Set<ModelID>, report: DiscoveryReport
    ) {
        let references = publishedModels.filter { reference in
            guard modelIDs.contains(reference.id),
                !attemptedLocalPublishedResolutions.contains(reference)
            else { return false }
            return Self.publishedReferenceNeedsResolution(
                reference, snapshot: snapshot, report: report)
        }
        guard !references.isEmpty else { return }
        attemptedLocalPublishedResolutions.formUnion(references)
        Task {
            for reference in references {
                _ = try? await resolvePublishedModel(reference)
            }
        }
    }

    /// Whether local filesystem evidence requires resolving the current
    /// published tree. A different remote commit alone is not enough: the
    /// cached descriptor remains the authority for already verified offline
    /// bytes until the person explicitly opens the newer remote row.
    static func publishedReferenceNeedsResolution(
        _ reference: PublishedModelReference, snapshot: ModelCatalogSnapshot,
        report: DiscoveryReport
    ) -> Bool {
        guard let descriptor = snapshot.descriptor(reference.id) else { return true }
        return report.locations.flatMap(\.artifacts).contains { artifact in
            artifact.model == reference.id
                && !artifact.metadataOverflowed
                && (artifact.fileCount > descriptor.totalFileCount
                    || artifact.bytesOnDisk > descriptor.totalBytes)
        }
    }

    private func recoverCachedCatalog(after liveFailure: CatalogError) {
        guard let catalogCache else { return }
        do {
            guard let cached = try catalogCache.load() else { return }
            // Only a live fetch (or that same document already relabelled as a
            // cache) can populate the product catalog. Older builds could
            // write bundled snapshots here; accepting one would resurrect the
            // fixed four-row storefront after a network failure.
            guard cached.origin == .live || cached.origin == .cached else { return }
            applyCatalog(cached.labelled(.cached))
            // A cache is useful continuity, not evidence that the requested
            // network refresh succeeded.
            catalogError = liveFailure
        } catch {
            catalogError = .malformedResponse(
                "live refresh failed (\(liveFailure.description)); cached catalog could not "
                    + "be read (\(String(describing: error)))")
        }
    }

    // MARK: - Budgets

    /// The memory profile as it should be drawn right now: declared until a
    /// mounted artifact is fully verified by discovery, measured afterwards.
    /// Mock transfer history is not evidence about bytes on disk.
    private func profile(
        for id: ModelID, artifactCensus: ArtifactCensus? = nil
    ) -> MemoryProfile? {
        guard let entry = entry(id) else { return nil }
        let memory = entry.memory
        guard memory.provenance != .unknown,
            installed.runnableArtifact(for: id) != nil
        else { return memory }
        return MemoryProfile(
            census: artifactCensus ?? memory.census,
            expertPoolBytes: memory.expertPoolBytes,
            workingSetReserveBytes: memory.workingSetReserveBytes,
            requiredMinimumBudgetBytes: memory.requiredMinimumBudgetBytes,
            onRecordMinimumBudgetBytes: memory.onRecordMinimumBudgetBytes,
            provenance: .measured,
            deepSeekV4: preparedDeepSeekV4Ladders[id] ?? memory.deepSeekV4)
    }

    /// Read V4's ladder off the verified copy on disk, once.
    ///
    /// A no-op for every other model, for a copy that is not runnable, and for
    /// one already read. The read itself is `index.json`, the verified config
    /// and one `manifest.json` per layer — metadata the complete verification
    /// pass already covered, and no payload byte.
    private func refreshDeepSeekV4Ladder() {
        let id = ModelID.deepseekV4Flash
        guard preparedDeepSeekV4Ladders[id] == nil,
            let runtime = runtimes.runtime(for: id),
            let artifact = try? installed.runnableArtifactReference(
                for: id, matching: runtime.accepts),
            let ladder = try? DeepSeekV4MemoryDialInputs.inspect(artifact)
        else { return }
        preparedDeepSeekV4Ladders[id] = ladder
    }

    /// The dial for THIS conversation's stated budget.
    func budgetPlan(for conversation: Conversation) -> BudgetPlan? {
        budgetPlan(
            for: conversation,
            artifactCensus: preparedArtifactCensuses[conversation.settings.model],
            workingSetReserveBytes: preparedWorkingSetReserves[conversation.id])
    }

    /// Execution supplies the census derived from the rooted artifact it is
    /// about to open. A product view may draw catalog geometry before the first
    /// preparation, but a request never sends that historical layer table to a
    /// newly published artifact.
    private func budgetPlan(
        for conversation: Conversation, artifactCensus: ArtifactCensus?,
        workingSetReserveBytes: UInt64? = nil
    ) -> BudgetPlan? {
        guard let entry = entry(conversation.settings.model),
            let memory = profile(
                for: conversation.settings.model, artifactCensus: artifactCensus)
        else { return nil }
        let preparedMemory = MemoryProfile(
            census: memory.census,
            expertPoolBytes: memory.expertPoolBytes,
            workingSetReserveBytes: workingSetReserveBytes ?? memory.workingSetReserveBytes,
            requiredMinimumBudgetBytes: memory.requiredMinimumBudgetBytes,
            onRecordMinimumBudgetBytes: memory.onRecordMinimumBudgetBytes,
            provenance: memory.provenance,
            deepSeekV4: memory.deepSeekV4)
        return BudgetPlan(
            model: entry.id, modelName: entry.descriptor.displayName, profile: preparedMemory,
            budgetBytes: conversation.settings.memoryBudgetBytes,
            maximumNewTokens: conversation.settings.maximumNewTokens,
            deviceCeilingBytes: device.processMemoryBudgetBytes,
            // The dial draws what the run will do, and an unset knob means the
            // runner prefetches (spec v0.6.19). Drawing an empty staged tier
            // for that would show a serial run the operator is not going to get.
            readAheadDepth: ReadAheadSetting(conversation.settings.deterministicReadAheadLayers)
                .effectiveDepth)
    }

    /// **The preset a new chat starts from, per model and per platform.**
    ///
    /// Every model starts at its Floor except V4 on the Mac, and the asymmetry
    /// is deliberate rather than a tuning choice:
    ///
    /// * **macOS → Balanced.** The Mac has the memory. V4's Balanced is the
    ///   ladder point at which every deterministic layer is resident, which on
    ///   this owner's 34.4 GB machine is under a quarter of the device — and
    ///   every byte below it buys resident layers, so starting at the floor
    ///   would start every Mac chat re-reading 5.87 GB a token for no reason.
    /// * **iOS → Floor.** The phone's V4 has no device measurement at all: no
    ///   pinned V4 arm has been run there, and `os_proc_available_memory()` on
    ///   a phone is a much harder ceiling than physical RAM on a Mac. A default
    ///   that spent it on a projection would be the product stating a budget it
    ///   has no evidence for. The dial still offers Balanced; the operator can
    ///   state it.
    ///
    /// A default the operator has stated is stored in `NewChatDefaults` and
    /// wins over this; this is only the value they have not overruled.
    static func defaultPreset(for id: ModelID) -> BudgetPlan.Preset {
        #if os(macOS)
            if id == .deepseekV4Flash { return .balanced }
        #endif
        return .floor
    }

    /// The dial for the DEFAULT a new chat with this model would start from.
    /// Labelled as such everywhere it appears; never used to run anything.
    func defaultBudgetPlan(for id: ModelID) -> BudgetPlan? {
        guard let entry = entry(id),
            let memory = profile(for: id, artifactCensus: preparedArtifactCensuses[id])
        else { return nil }
        let base = BudgetPlan(
            model: id, modelName: entry.descriptor.displayName, profile: memory,
            budgetBytes: memory.refusalThresholdBytes,
            maximumNewTokens: max(1, capabilities(for: id)?.maximumNewTokens ?? 1),
            deviceCeilingBytes: device.processMemoryBudgetBytes,
            readAheadDepth: ReadAheadSetting(defaults.deterministicReadAheadLayers)
                .effectiveDepth)
        guard let stated = defaults.budget(for: id) else {
            let preset = base.budget(for: Self.defaultPreset(for: id))
            // A preset that does not fit this device is not a default. It is
            // shown disabled with its deficit in the dial, and a new chat falls
            // back to the boundary that does run.
            return base.with(
                budgetBytes: preset <= base.deviceCeilingBytes
                    ? preset : memory.refusalThresholdBytes)
        }
        return base.with(budgetBytes: stated)
    }

    func setDefaultBudget(_ bytes: UInt64, for id: ModelID) {
        defaults.setBudget(bytes, for: id)
    }

    func setBudget(_ bytes: UInt64, in conversationID: UUID) {
        update(conversationID) { $0.settings.memoryBudgetBytes = bytes }
        if let conversation = conversation(conversationID),
            budgetPlan(for: conversation)?.refusal == nil
        {
            runPreparationBudgetTargets.removeValue(forKey: conversationID)
            runPreparationErrors.removeValue(forKey: conversationID)
        }
    }

    private func persistDefaults() {
        guard persistsDefaults else { return }
        guard let data = try? JSONEncoder().encode(defaults) else { return }
        userDefaults.set(data, forKey: defaultsKey)
    }

    private func persistAppearance() {
        guard persistsDefaults else { return }
        userDefaults.set(appearance.rawValue, forKey: Self.appearanceKey)
    }

    // MARK: - Calibration and projection

    func calibration(for id: ModelID) -> VolumeCalibration? {
        // A completed transfer is history, not proof that the same path is
        // mounted now. Calibrations are projected only onto artifacts in the
        // current storage scan.
        guard let path = installed.preferredArtifact(for: id)?.rootPath else { return nil }
        return calibrations.values.first { path.hasPrefix($0.volumeMountPath) }
    }

    func projection(for entry: CatalogEntry) -> SpeedProjection? {
        guard hasVerifiedRuntime(for: entry.id) else { return nil }
        guard let plan = defaultBudgetPlan(for: entry.id) else { return nil }
        return SpeedProjector.project(
            plan: plan, terms: entry.terms, link: calibration(for: entry.id))
    }

    /// The newest persisted assistant turn with a real token rate for this
    /// model. This is deliberately labelled "last run" by the row: old chat
    /// telemetry proves what Minirun measured on this machine at that time, but
    /// it is not a projection for a newly published artifact revision.
    func lastRunSecondsPerToken(for id: ModelID) -> Double? {
        conversations
            .flatMap { conversation in
                conversation.messages.compactMap { message -> (Date, Double)? in
                    guard message.role == .assistant,
                        let telemetry = message.telemetry,
                        (telemetry.model ?? conversation.settings.model) == id,
                        telemetry.budgetRespected,
                        let rate = telemetry.tokensPerSecond,
                        rate.isFinite, rate > 0
                    else { return nil }
                    return (message.createdAt, 1 / rate)
                }
            }
            .max { $0.0 < $1.0 }?
            .1
    }

    // MARK: - Assessment

    /// The per-token terms each model would run under, at the budget a new chat
    /// with it would start from.
    ///
    /// The budget is part of the answer and not a detail: the staged pair being
    /// funded is what decides whether the deterministic read hides behind
    /// compute, and that is the step in the curve. An assessment that assumed
    /// the serial shape would understate a well-funded rig and an assessment
    /// that assumed the staged one would flatter a poor drive.
    var assessmentWorkloads: [ModelID: TokenWorkload] {
        var workloads: [ModelID: TokenWorkload] = [:]
        for entry in entries where runtimes.runtime(for: entry.id) != nil {
            let plan = defaultBudgetPlan(for: entry.id)
            workloads[entry.id] = TokenWorkload(
                deterministicBytesPerToken: entry.terms.deterministicBytesPerToken,
                expertBytesPerToken: entry.terms.expertBytesPerToken,
                computeSecondsPerToken: entry.terms.computeSecondsPerToken,
                // Zero, and no longer a function of the budget: the derived tier
                // ladder reaches an expert only past full deterministic
                // residency, so no budget any device here can state pins one.
                // The fraction this used to compute described a cache that does
                // not exist at any reachable budget.
                expertCacheFraction: 0,
                deterministicOverlapsCompute: plan?.stagedTierIsFunded ?? false)
        }
        return workloads
    }

    /// Assess one registered location. Called from a button and from nothing
    /// else — the reading is bounded, but it is still somebody's drive.
    func assess(_ key: StorageKey) async {
        let scan = installed.report.locations.first { $0.storageKey == key }
        await assessments.assess(key: key, scan: scan, workloads: assessmentWorkloads)
    }

    /// The assessment covering the artifact this model would actually be read
    /// from, when there is one.
    func assessment(for id: ModelID) -> VolumeAssessmentReport? {
        guard let artifact = installed.preferredArtifact(for: id) else { return nil }
        return assessments.report(covering: artifact.rootPath)
    }

    /// This model's projected token time on the volume it is installed on.
    /// Nil unless that volume has been measured — the app's standing rule.
    func assessedProjection(for id: ModelID) -> ModelVerdict? {
        assessment(for: id)?.verdicts.first { $0.fit.model == id && $0.projection != nil }
    }

    // MARK: - Volumes

    func refreshVolumes() {
        do {
            volumes = try storage.volumes()
            volumeError = nil
        } catch {
            volumes = []
            volumeError = "\(error)"
        }
    }

    // MARK: - History

    private func recordFinishedRun(_ snapshot: RunSnapshot) {
        guard let summary = snapshot.summary, let footer = snapshot.reproducibility else { return }
        history.insert(
            RunRecord(
                model: summary.model, at: Date(),
                declaredBudgetBytes: summary.finalTelemetry.declaredBudgetBytes,
                peakFootprintBytes: summary.peakFootprintBytes,
                secondsPerToken: summary.wallSeconds,
                thermalState: summary.finalTelemetry.thermalState,
                // The depth the run used, not the knob's spelling: a record that
                // said 0 for a run that prefetched would be a record of a
                // different run.
                readAheadDepth: ReadAheadSetting(
                    runningConversation?.settings.deterministicReadAheadLayers).effectiveDepth,
                tokenID: footer.tokenID,
                tokenText: summary.text, logitsDigest: footer.logitsDigest,
                budgetRespected: summary.budgetRespected),
            at: 0)
    }

    func history(for id: ModelID) -> [RunRecord] {
        history.filter { $0.model == id }
    }

    func measuredPoints(for id: ModelID) -> [MeasuredPoint] {
        history(for: id).map {
            MeasuredPoint(
                budgetBytes: $0.declaredBudgetBytes, secondsPerToken: $0.secondsPerToken,
                at: $0.at, logitsDigest: $0.logitsDigest)
        }
    }
}
