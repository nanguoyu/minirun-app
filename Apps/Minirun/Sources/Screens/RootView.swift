import MinirunKit
import SwiftUI

#if os(macOS)
    import AppKit
#endif

/// The platform fork, and the only one in the product.
///
/// macOS gets a split view whose sidebar is a list of conversations with
/// Settings pinned under them; iOS gets two tabs for the same two things.
/// Everything below this file is shared, because they are the same app.
struct RootView: View {
    @Environment(AppModel.self) private var model
    #if os(iOS)
        @Environment(\.scenePhase) private var scenePhase
    #endif

    @ViewBuilder
    var body: some View {
        Group {
            if let refusal = model.launchRefusal {
                LaunchRefusalView(refusal: refusal)
            } else {
                VStack(spacing: 0) {
                    if let notice = model.visualReviewNotice {
                        ReviewPreviewBanner(message: notice)
                    }
                    #if os(macOS)
                        MacRootView()
                    #else
                        PhoneRootView()
                    #endif
                }
            }
        }
        .preferredColorScheme(model.appearance.preferredColorScheme)
        #if os(iOS)
            .onChange(of: scenePhase, initial: true) { _, phase in
                model.transitionScene(to: phase.appSceneState)
            }
        #endif
    }
}

#if os(iOS)
    private extension ScenePhase {
        var appSceneState: AppSceneState {
            switch self {
            case .active: .active
            case .inactive: .inactive
            case .background: .background
            @unknown default: .inactive
            }
        }
    }
#endif

private extension AppAppearance {
    var preferredColorScheme: ColorScheme? {
        switch self {
        case .system: nil
        case .light: .light
        case .dark: .dark
        }
    }
}

/// The review fixture is intentionally incapable of touching real bookmarks.
/// That safety boundary must be visible when somebody operates the window,
/// rather than existing only in a launch argument they did not type.
private struct ReviewPreviewBanner: View {
    let message: String

    var body: some View {
        Label(message, systemImage: "eye.trianglebadge.exclamationmark")
            .font(MRType.caption)
            .foregroundStyle(MRColor.caution)
            .padding(.horizontal, MRSpace.s3)
            .padding(.vertical, MRSpace.s2)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(MRColor.caution.opacity(0.12))
            .accessibilityIdentifier("visual-review-preview-banner")
    }
}

/// A deliberately inert screen for an unsafe review request. It offers no
/// navigation into product state because reaching this view means that state
/// was never opened.
private struct LaunchRefusalView: View {
    let refusal: AppModel.LaunchRefusal

    var body: some View {
        VStack(spacing: MRSpace.s4) {
            Image(systemName: "exclamationmark.shield.fill")
                .font(.system(size: 34, weight: .semibold))
                .foregroundStyle(MRColor.caution)
                .accessibilityHidden(true)
            Text(refusal.title)
                .font(MRType.title)
                .foregroundStyle(MRColor.primary)
            Text(refusal.message)
                .font(MRType.body)
                .foregroundStyle(MRColor.secondary)
            Text(refusal.privacyNote)
                .font(MRType.caption)
                .foregroundStyle(MRColor.tertiary)
        }
        .multilineTextAlignment(.center)
        .padding(MRSpace.s6)
        .frame(maxWidth: 480)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .mrWorkspaceSurface()
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("launch-refusal")
    }
}

// MARK: - macOS

#if os(macOS)
    /// One product-owned top row for every macOS column. The window uses a
    /// hidden title bar, so this is where the title and the one relevant action
    /// live; AppKit never inserts toolbar group separators between them.
    struct MacColumnHeader<Trailing: View>: View {
        let title: String
        var protectsWindowControls = false
        var backAction: (() -> Void)?
        @ViewBuilder var trailing: Trailing

        init(
            title: String, protectsWindowControls: Bool = false,
            backAction: (() -> Void)? = nil,
            @ViewBuilder trailing: () -> Trailing
        ) {
            self.title = title
            self.protectsWindowControls = protectsWindowControls
            self.backAction = backAction
            self.trailing = trailing()
        }

        var body: some View {
            HStack(spacing: MRSpace.s3) {
                if let backAction {
                    Button(action: backAction) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 16, weight: .semibold))
                            .frame(width: 28, height: 28)
                    }
                    .buttonStyle(.plain)
                    .help("Back")
                    .accessibilityLabel("back")
                }
                Text(title)
                    .font(MRType.title)
                    .foregroundStyle(MRColor.primary)
                    .lineLimit(1)
                    .accessibilityAddTraits(.isHeader)
                Spacer(minLength: MRSpace.s3)
                trailing
            }
            .padding(.leading, protectsWindowControls ? 76 : MRSpace.s4)
            .padding(.trailing, MRSpace.s4)
            .frame(height: MacColumnHeaderLayout.height)
            .contentShape(Rectangle())
        }
    }

    enum MacColumnHeaderLayout {
        static let height: CGFloat = 52
    }

    extension MacColumnHeader where Trailing == EmptyView {
        init(
            title: String, protectsWindowControls: Bool = false,
            backAction: (() -> Void)? = nil
        ) {
            self.init(
                title: title, protectsWindowControls: protectsWindowControls,
                backAction: backAction
            ) { EmptyView() }
        }
    }

    /// The standard window controls own the first row of a full-size-content
    /// window. Sidebar titles live directly below that row, aligned with the
    /// sidebar's content instead of being pushed sideways around the controls.
    /// Chat and Settings both use this exact component so navigation cannot
    /// jump vertically when the operator moves between them.
    enum MacSidebarHeaderLayout {
        static let windowControlClearanceHeight: CGFloat = 28
        static let titleRowHeight: CGFloat = 44
        static let horizontalPadding: CGFloat = MRSpace.s4
        static var totalHeight: CGFloat {
            windowControlClearanceHeight + titleRowHeight
        }
    }

    struct MacSidebarHeader: View {
        let title: String

        var body: some View {
            VStack(spacing: 0) {
                Color.clear
                    .frame(height: MacSidebarHeaderLayout.windowControlClearanceHeight)
                    .accessibilityHidden(true)
                HStack {
                    Text(title)
                        .font(MRType.title)
                        .foregroundStyle(MRColor.primary)
                        .lineLimit(1)
                        .accessibilityAddTraits(.isHeader)
                    Spacer(minLength: MRSpace.s3)
                }
                .padding(.horizontal, MacSidebarHeaderLayout.horizontalPadding)
                .frame(height: MacSidebarHeaderLayout.titleRowHeight)
            }
            .frame(height: MacSidebarHeaderLayout.totalHeight)
            .contentShape(Rectangle())
        }
    }

    /// Sidebar · detail · panel.
    ///
    /// The sidebar lists **conversations** and nothing else, with Settings
    /// pinned at the bottom. Models used to sit in the sidebar too, and a
    /// sidebar that mixes a growing list with fixed rows loses the fixed rows
    /// the moment the list grows: Models is now a section of Settings.
    struct MacRootView: View {
        @Environment(AppModel.self) private var model

        var body: some View {
            Group {
                // Settings takes the WHOLE window. Its own nav column replaces
                // the chat sidebar rather than sitting beside it — a settings
                // surface with the conversation list still visible down the
                // left reads as a page you are visiting, and this one is a
                // place you go. "Back to app" is the way out.
                if model.destination == .settings {
                    SettingsView()
                } else {
                    chatSplit
                }
            }
            .ignoresSafeArea(.container, edges: .top)
        }

        private var chatSplit: some View {
            GeometryReader { geometry in
                HStack(spacing: 0) {
                    SidebarView()
                        .frame(width: ConversationWorkspaceLayout.sidebarWidth)
                        .mrSidebarSurface()

                    Divider()

                    HStack(spacing: 0) {
                        // The Mac chat workspace does not push destinations. A
                        // NavigationStack here inserted its own title-bar safe
                        // area above this product-owned 52-point header, making
                        // Chats taller than Settings. The detail is deliberately
                        // direct; iPhone keeps its real navigation stack.
                        content
                            .frame(maxWidth: .infinity, maxHeight: .infinity)

                        if isPanelPresented {
                            panel
                                .frame(width: effectiveInspectorWidth(for: geometry.size.width))
                                .overlay(alignment: .topLeading) {
                                    ZStack(alignment: .topLeading) {
                                        ConversationInspectorBodyBoundary()
                                        ConversationInspectorResizeHandle(
                                            width: inspectorWidthBinding(
                                                availableWindowWidth: geometry.size.width),
                                            maximumWidth: ConversationWorkspaceLayout
                                                .maximumInspectorWidth(
                                                    availableWindowWidth: geometry.size.width)
                                        )
                                        .padding(
                                            .top,
                                            ConversationInspectorBoundaryLayout.topInset)
                                    }
                                }
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .background(MinirunWindowChrome())
        }

        @ViewBuilder private var content: some View {
            switch model.destination {
            case .chats: chatHome
            case .conversation(let id): ConversationView(conversationID: id)
            case .settings: EmptyView()
            }
        }

        private var chatHome: some View {
            VStack(spacing: 0) {
                MacColumnHeader(title: "Chats")
                Divider()
                gettingReady
                    .padding(MRSpace.s7)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .mrWorkspaceSurface()
            .accessibilityElement(children: .contain)
            .accessibilityLabel("Chats")
        }

        @ViewBuilder private var gettingReady: some View {
            switch model.chatReadiness {
            case .ready(let modelID):
                GettingReadyContent(
                    systemImage: "bubble.left.and.bubble.right",
                    title: "Start a conversation",
                    detail: "\(modelName(modelID)) is verified and ready.",
                    actionTitle: "New chat",
                    action: { _ = model.newConversation(modelID: modelID) })

            case .loadingCatalog:
                GettingReadyContent(
                    systemImage: "arrow.triangle.2.circlepath",
                    title: "Loading models",
                    detail: "Checking the current model catalog.",
                    showsProgress: true)

            case .catalogUnavailable:
                GettingReadyContent(
                    systemImage: "wifi.exclamationmark",
                    title: "Models could not be loaded",
                    detail: "No saved catalog is available. Check the connection and try again.",
                    actionTitle: "Try again",
                    action: { Task { await model.refreshPublishedModels() } })

            case .needsStorage:
                GettingReadyContent(
                    systemImage: "folder.badge.plus",
                    title: "Choose a model folder",
                    detail: "Give Minirun access to the folder where local models are kept.",
                    actionTitle: "Open Storage",
                    action: { model.openSettings(.storage) })

            case .scanningStorage:
                GettingReadyContent(
                    systemImage: "externaldrive",
                    title: "Scanning your folders",
                    detail: "Looking for complete Minirun models in the folders you chose.",
                    showsProgress: true)

            case .needsVerification(let modelID):
                GettingReadyContent(
                    systemImage: "checkmark.shield",
                    title: "Verify \(modelName(modelID))",
                    detail: "Open Models and verify all published files before starting a chat.",
                    actionTitle: "Open Models",
                    action: { model.openSettings(.models) })

            case .needsModel:
                GettingReadyContent(
                    systemImage: "square.stack.3d.up",
                    title: "Choose a model",
                    detail: "Browse the current catalog, inspect local copies, or start a download.",
                    actionTitle: "Open Models",
                    action: { model.openSettings(.models) })
            }
        }

        private func modelName(_ id: ModelID) -> String {
            model.entry(id)?.descriptor.displayName ?? id.rawValue
        }

        /// The panel belongs to a conversation, and the conversation's own "…"
        /// menu is the only thing that opens or closes it. Away from a
        /// conversation there is nothing for it to be about, so there is no
        /// panel.
        @State private var inspectorWidth = ConversationWorkspaceLayout.inspectorIdealWidth

        private func effectiveInspectorWidth(for availableWindowWidth: CGFloat) -> CGFloat {
            ConversationWorkspaceLayout.clampedInspectorWidth(
                inspectorWidth, availableWindowWidth: availableWindowWidth)
        }

        private func inspectorWidthBinding(
            availableWindowWidth: CGFloat
        ) -> Binding<CGFloat> {
            Binding(
                get: { effectiveInspectorWidth(for: availableWindowWidth) },
                set: {
                    inspectorWidth = ConversationWorkspaceLayout.clampedInspectorWidth(
                        $0, availableWindowWidth: availableWindowWidth)
                })
        }

        private var isPanelPresented: Bool {
            guard case .conversation = model.destination else { return false }
            return model.conversationPanel != nil
        }

        @ViewBuilder private var panel: some View {
            if case .conversation(let id) = model.destination {
                ConversationInspector(conversationID: id)
            }
        }
    }

    private struct GettingReadyContent: View {
        let systemImage: String
        let title: String
        let detail: String
        var actionTitle: String?
        var showsProgress = false
        var action: (() -> Void)?

        var body: some View {
            VStack(spacing: MRSpace.s4) {
                if showsProgress {
                    ProgressView()
                        .controlSize(.large)
                        .accessibilityLabel(title)
                } else {
                    Image(systemName: systemImage)
                        .font(.system(size: 34, weight: .light))
                        .foregroundStyle(MRColor.secondary)
                        .accessibilityHidden(true)
                }
                Text(title)
                    .font(MRType.title)
                    .foregroundStyle(MRColor.primary)
                    .accessibilityAddTraits(.isHeader)
                Text(detail)
                    .font(MRType.body)
                    .foregroundStyle(MRColor.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 440)
                    .fixedSize(horizontal: false, vertical: true)
                if let actionTitle, let action {
                    Button(actionTitle, action: action)
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                }
            }
            .accessibilityElement(children: .contain)
        }
    }

    /// Window and inspector geometry is declared once and never mutated when a
    /// panel opens. The operator owns the outer window size; the inspector and
    /// transcript divide whatever width that window currently has.
    enum ConversationWorkspaceLayout {
        static let defaultWindowWidth: CGFloat = 1_120
        static let defaultWindowHeight: CGFloat = 720
        static let minimumWindowWidth: CGFloat = 960
        static let minimumWindowHeight: CGFloat = 560
        /// The right inspector divides the detail area only. Keeping the Chats
        /// column at one width prevents opening information from making the
        /// user's conversation list jump sideways.
        static let sidebarWidth: CGFloat = 260
        static let minimumConversationWidth: CGFloat = 380
        static let sidebarDividerWidth: CGFloat = 1
        static let inspectorMinimumWidth: CGFloat = 300
        static let inspectorIdealWidth: CGFloat = 360
        static let inspectorMaximumWidth: CGFloat = 520

        static func clampedInspectorWidth(_ width: CGFloat) -> CGFloat {
            min(inspectorMaximumWidth, max(inspectorMinimumWidth, width))
        }

        static func maximumInspectorWidth(availableWindowWidth: CGFloat) -> CGFloat {
            let roomAfterConversation =
                availableWindowWidth - sidebarWidth - sidebarDividerWidth
                - minimumConversationWidth
            return min(
                inspectorMaximumWidth,
                max(inspectorMinimumWidth, roomAfterConversation))
        }

        static func clampedInspectorWidth(
            _ width: CGFloat, availableWindowWidth: CGFloat
        ) -> CGFloat {
            min(
                maximumInspectorWidth(availableWindowWidth: availableWindowWidth),
                max(inspectorMinimumWidth, width))
        }
    }

    /// The detail/panel boundary belongs to the conversation body, not the
    /// product-owned title row. Starting below both 52-point column headers
    /// keeps that top row visually continuous while retaining the useful
    /// structure between transcript and inspector content.
    enum ConversationInspectorBoundaryLayout {
        static let lineWidth: CGFloat = 1
        static let resizeTargetWidth: CGFloat = 10
        static let topInset: CGFloat = MacColumnHeaderLayout.height + 1
    }

    /// The right column is an operational console: it contains controls and
    /// live instruments, not general information and not another sidebar.
    /// Keep one stable symbol in the same top-right position for show/hide.
    enum ConversationPanelToggleStyle {
        static let symbolName = "slider.horizontal.3"
        static let showHelp = "Show chat controls and instruments (⌥⌘I)"
        static let hideHelp = "Hide chat controls and instruments (⌥⌘I)"
    }

    private struct ConversationInspectorBodyBoundary: View {
        var body: some View {
            VStack(alignment: .leading, spacing: 0) {
                Color.clear
                    .frame(height: ConversationInspectorBoundaryLayout.topInset)
                    .accessibilityHidden(true)
                Rectangle()
                    .fill(MRColor.hairline)
                    .frame(width: ConversationInspectorBoundaryLayout.lineWidth)
                    .frame(maxHeight: .infinity)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .allowsHitTesting(false)
            .accessibilityHidden(true)
        }
    }

    /// A wider invisible target sits over the body-only hairline so the panel
    /// remains easy to resize across the space the current window can safely
    /// offer without making the visual separator heavy.
    private struct ConversationInspectorResizeHandle: View {
        @Binding var width: CGFloat
        let maximumWidth: CGFloat
        @State private var dragOriginWidth: CGFloat?

        var body: some View {
            Color.clear
                .frame(width: ConversationInspectorBoundaryLayout.resizeTargetWidth)
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            let origin = dragOriginWidth ?? width
                            if dragOriginWidth == nil { dragOriginWidth = origin }
                            width = min(
                                maximumWidth,
                                ConversationWorkspaceLayout.clampedInspectorWidth(
                                    origin - value.translation.width))
                        }
                        .onEnded { _ in dragOriginWidth = nil }
                )
                .onContinuousHover { phase in
                    switch phase {
                    case .active: NSCursor.resizeLeftRight.set()
                    case .ended: NSCursor.arrow.set()
                    }
                }
                .accessibilityElement()
                .accessibilityLabel("resize chat controls and instruments")
                .accessibilityValue("\(Int(width.rounded())) points wide")
                .accessibilityHint("Adjusts the width of the chat controls and instruments panel")
                .accessibilityAdjustableAction { direction in
                    let step: CGFloat = 20
                    switch direction {
                    case .increment:
                        width = min(
                            maximumWidth,
                            ConversationWorkspaceLayout.clampedInspectorWidth(width + step))
                    case .decrement:
                        width = ConversationWorkspaceLayout.clampedInspectorWidth(width - step)
                    @unknown default:
                        break
                    }
                }
        }
    }

    /// One resizable inspector with two explicit tabs. Selecting a tab changes
    /// the inspector's content; the close button hides it without changing the
    /// window frame or the selected conversation.
    private struct ConversationInspector: View {
        let conversationID: UUID

        @Environment(AppModel.self) private var model

        var body: some View {
            VStack(spacing: 0) {
                MacColumnHeader(title: "") {
                    Button { model.conversationPanel = nil } label: {
                        Image(systemName: ConversationPanelToggleStyle.symbolName)
                            .imageScale(.large)
                            .frame(width: 28, height: 28)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(MRColor.tierPinned)
                    .help(ConversationPanelToggleStyle.hideHelp)
                    .accessibilityLabel("hide chat controls and instruments")
                }

                Divider()

                HStack(spacing: MRSpace.s2) {
                    Picker("Inspector", selection: selection) {
                        ForEach(AppModel.ConversationPanel.allCases) { panel in
                            Text(panel.rawValue).tag(panel)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }
                .padding(.horizontal, MRSpace.s3)
                .padding(.vertical, MRSpace.s2)

                Divider()

                switch model.conversationPanel ?? .chatSettings {
                case .chatSettings:
                    ConversationSettingsView(conversationID: conversationID)
                case .instruments:
                    InstrumentPanelView(snapshot: model.runSnapshot(for: conversationID))
                }
            }
            .mrWorkspaceSurface()
            #if DEBUG
                .background(ConversationInspectorGeometryProbe())
            #endif
        }

        private var selection: Binding<AppModel.ConversationPanel> {
            Binding(
                get: { model.conversationPanel ?? .chatSettings },
                set: { model.conversationPanel = $0 })
        }
    }

    #if DEBUG
        /// Hosted layout tests observe the actual inspector SwiftUI places in
        /// its column. This seam has no storage and is compiled out of Release.
        @MainActor
        enum ConversationInspectorLayoutObservation {
            static var onLayout: ((CGSize) -> Void)?
        }

        private struct ConversationInspectorGeometryProbe: View {
            var body: some View {
                GeometryReader { geometry in
                    Color.clear
                        .onAppear { report(geometry.size) }
                        .onChange(of: geometry.size) { _, size in report(size) }
                }
                .accessibilityHidden(true)
            }

            private func report(_ size: CGSize) {
                ConversationInspectorLayoutObservation.onLayout?(size)
            }
        }
    #endif

    struct SidebarView: View {
        @Environment(AppModel.self) private var model
        @State private var renaming: UUID?
        @State private var pendingDeletion: ConversationDeletionRequest?

        var body: some View {
            @Bindable var model = model
            VStack(spacing: 0) {
                MacSidebarHeader(title: "Minirun")
                HStack(spacing: MRSpace.s2) {
                    Text("Chats")
                        .font(MRType.headline)
                        .foregroundStyle(MRColor.secondary)
                    Spacer()
                    Button {
                        _ = model.newConversation()
                    } label: {
                        Image(systemName: "square.and.pencil")
                            .frame(width: 28, height: 28)
                    }
                    .buttonStyle(.plain)
                    .help("New chat (⌘N)")
                    .disabled(!model.canCreateConversation)
                    .accessibilityLabel("new chat")
                }
                .padding(.leading, MRSpace.s4)
                .padding(.trailing, MRSpace.s4)
                .padding(.top, MRSpace.s3)
                .padding(.bottom, MRSpace.s2)

                List(selection: selection) {
                    ForEach(model.conversations) { conversation in
                        ConversationRow(
                            conversation: conversation,
                            modelName: modelName(conversation),
                            isRunning: model.isRunning(conversation.id)
                        )
                        .tag(conversation.id)
                        .contextMenu {
                            Button("Rename…") { renaming = conversation.id }
                            Button("Delete…", role: .destructive) {
                                pendingDeletion = ConversationDeletionRequest(
                                    conversation, isRunning: model.isRunning(conversation.id))
                            }
                        }
                    }
                    if model.conversations.isEmpty {
                        Text("No conversations yet.")
                            .font(MRType.caption)
                            .foregroundStyle(MRColor.secondary)
                    }
                }
                .listStyle(.sidebar)
                .scrollContentBackground(.hidden)
            }
            .sheet(item: $renaming) { id in
                RenameConversationSheet(conversationID: id)
            }
            .confirmationDialog(
                "Delete this conversation?",
                isPresented: deletionConfirmationPresented,
                titleVisibility: .visible
            ) {
                if let request = pendingDeletion {
                    Button("Delete conversation", role: .destructive) {
                        pendingDeletion = nil
                        model.delete(request.id)
                    }
                }
                Button("Cancel", role: .cancel) { pendingDeletion = nil }
            } message: {
                if let request = pendingDeletion { Text(request.message) }
            }
            // Pinned, so it never scrolls away no matter how long the chat list
            // grows. It is the ONE non-conversation row in this sidebar, and
            // everything that is not a chat lives behind it.
            .safeAreaInset(edge: .bottom, spacing: 0) {
                VStack(spacing: 0) {
                    Divider()
                    Button {
                        model.openSettings()
                    } label: {
                        HStack(spacing: MRSpace.s2) {
                            Image(systemName: "gearshape")
                            Text("Settings").font(MRType.body)
                            Spacer()
                        }
                        .padding(.horizontal, MRSpace.s3)
                        .padding(.vertical, MRSpace.s2)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help("Settings (⌘,)")
                    .accessibilityLabel("settings")
                }
                // No fill. `.bar` and every opaque colour this app owns read as
                // a white slab against the sidebar's material; the inset simply
                // lets the sidebar's own background through, which is the only
                // way the two can be the same tone in both appearances.
            }
        }

        /// The list selects conversations only, so selecting one always means
        /// the same thing. Settings is a button under the list rather than a
        /// row in it, and a `nil` selection is what "you are in Settings" looks
        /// like from here.
        private var selection: Binding<UUID?> {
            Binding(
                get: {
                    if case .conversation(let id) = model.destination { return id }
                    return nil
                },
                set: { id in
                    if let id { model.destination = .conversation(id) }
                })
        }

        private var deletionConfirmationPresented: Binding<Bool> {
            Binding(
                get: { pendingDeletion != nil },
                set: { if !$0 { pendingDeletion = nil } })
        }

        private func modelName(_ conversation: Conversation) -> String {
            model.entry(conversation.settings.model)?.descriptor.displayName
                ?? conversation.settings.model.rawValue
        }
    }

    /// Keyboard is a first-class input on the Mac.
    struct MinirunCommands: Commands {
        let model: AppModel
        let updater: SoftwareUpdater

        var body: some Commands {
            // Directly under "About Minirun", which is where every
            // non-App-Store Mac app puts it and therefore the only place an
            // operator will look for it. Disabled rather than hidden when the
            // updater is not running: a menu item that vanishes reads as a
            // build that lost a feature, and Settings → General says why.
            CommandGroup(after: .appInfo) {
                Button(SoftwareUpdatePresentation.menuItemTitle) {
                    updater.checkForUpdates()
                }
                .disabled(!updater.canCheckForUpdates)
            }
            CommandGroup(replacing: .newItem) {
                Button("New Chat") { _ = model.newConversation() }
                    .keyboardShortcut("n", modifiers: .command)
                    .disabled(!model.canCreateConversation)
            }
            CommandGroup(after: .toolbar) {
                Button(
                    model.conversationPanel == nil
                        ? "Show Conversation Information" : "Hide Conversation Information"
                ) {
                    model.conversationPanel = model.conversationPanel == nil ? .chatSettings : nil
                }
                    .keyboardShortcut("i", modifiers: [.option, .command])
                    .disabled(!hasSelectedConversation)
                Button("Instruments Panel") { model.conversationPanel = .instruments }
                    .keyboardShortcut("i", modifiers: [.option, .shift, .command])
                    .disabled(!hasSelectedConversation)
                Divider()
                Button("Models") { model.openSettings(.models) }
                    .keyboardShortcut("2", modifiers: .command)
            }
            CommandGroup(replacing: .appSettings) {
                Button("Settings…") { model.openSettings() }
                    .keyboardShortcut(",", modifiers: .command)
            }
            CommandMenu("Chat") {
                Button(model.run.isStopping ? "Stopping…" : "Stop") { model.stopTurn() }
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(!model.run.isRunning || model.run.isStopping)

                Button("Regenerate Last Turn") {
                    if case .conversation(let id) = model.destination {
                        model.regenerateLastTurn(in: id)
                    }
                }
                .keyboardShortcut("r", modifiers: [.shift, .command])
                .disabled(model.run.isRunning)

                Divider()

                Button("Refresh Models") {
                    Task { await model.refreshPublishedModels() }
                }
                .keyboardShortcut("r", modifiers: .command)
            }
        }

        private var hasSelectedConversation: Bool {
            if case .conversation = model.destination { return true }
            return false
        }
    }
#endif

// MARK: - iOS

#if os(iOS)
    /// Two tabs, matching the Mac's two sidebar destinations.
    ///
    /// Models was a third tab. It is a Settings section now for the same reason
    /// it left the Mac's sidebar — one information architecture, not two — and
    /// because on a phone the catalog is something you visit to install or
    /// inspect a model, not something you switch to mid-conversation. The chat
    /// banner that says a model is missing links straight into it.
    struct PhoneRootView: View {
        @Environment(AppModel.self) private var model
        @Environment(\.horizontalSizeClass) private var horizontalSizeClass
        @SceneStorage("minirun.iOS.navigation.place") private var mirroredPlace = ""
        @SceneStorage("minirun.iOS.navigation.conversation")
        private var mirroredConversation = ""
        @SceneStorage("minirun.iOS.navigation.settingsSection")
        private var mirroredSettingsSection = ""

        private enum Tab: Hashable { case chats, settings }

        var body: some View {
            TabView(selection: selectedTab) {
                Group {
                    if horizontalSizeClass == .regular {
                        RegularChatsRoot()
                    } else {
                        CompactChatsRoot()
                    }
                }
                .tabItem { Label("Chats", systemImage: "bubble.left.and.bubble.right") }
                .tag(Tab.chats)

                Group {
                    if horizontalSizeClass == .regular {
                        RegularSettingsRoot()
                    } else {
                        CompactSettingsRoot()
                    }
                }
                .tabItem { Label("Settings", systemImage: "gearshape") }
                .tag(Tab.settings)
            }
            .tint(MRColor.tierPinned)
            .onAppear { restoreOrInitializeSceneMirror() }
            .onChange(of: model.destination) { _, _ in mirrorNavigation() }
            .onChange(of: model.settingsNavigationPath) { _, _ in mirrorNavigation() }
        }

        private var selectedTab: Binding<Tab> {
            Binding(
                get: { model.destination == .settings ? .settings : .chats },
                set: { tab in
                    switch tab {
                    case .chats:
                        if model.destination == .settings { model.backToApp() }
                    case .settings:
                        if model.destination != .settings { model.openSettingsRoot() }
                    }
                })
        }

        private func restoreOrInitializeSceneMirror() {
            guard let place = AppModel.SceneNavigationMirror.Place(rawValue: mirroredPlace) else {
                mirrorNavigation()
                return
            }
            let conversationID = UUID(uuidString: mirroredConversation)
            let settingsSection = AppModel.SettingsSection(rawValue: mirroredSettingsSection)
            model.restoreNavigation(
                from: AppModel.SceneNavigationMirror(
                    place: place, conversationID: conversationID,
                    settingsSection: settingsSection))
            // A stale conversation id is deliberately repaired to Chats by
            // AppModel; mirror that repaired truth back into the scene record.
            mirrorNavigation()
        }

        private func mirrorNavigation() {
            let mirror = model.sceneNavigationMirror
            mirroredPlace = mirror.place.rawValue
            mirroredConversation = mirror.conversationID?.uuidString ?? ""
            mirroredSettingsSection = mirror.settingsSection?.rawValue ?? ""
        }
    }

    /// The compact stacks bind directly to AppModel. A banner opening Models,
    /// a running-turn shortcut and a restored destination therefore drive the
    /// same paths as taps in the UI.
    private struct CompactChatsRoot: View {
        @Environment(AppModel.self) private var model

        var body: some View {
            NavigationStack(path: path) {
                ConversationListView(openSettings: model.openSettings)
                    .navigationDestination(for: UUID.self) { id in
                        ConversationView(conversationID: id)
                            // A pushed conversation is the whole screen. The
                            // tab bar under the composer cost the transcript
                            // a band of height and put a second navigation
                            // surface below the thing being typed into; the
                            // platform hides it for a pushed detail and
                            // restores it on pop.
                            .toolbar(.hidden, for: .tabBar)
                    }
            }
        }

        private var path: Binding<[UUID]> {
            Binding(
                get: {
                    if case .conversation(let id) = model.destination { return [id] }
                    return []
                },
                set: model.setConversationNavigationPath)
        }
    }

    private struct CompactSettingsRoot: View {
        @Environment(AppModel.self) private var model

        var body: some View {
            NavigationStack(path: path) {
                IOSSettingsIndex()
                    .navigationDestination(for: AppModel.SettingsSection.self) { section in
                        SettingsSectionView(section: section)
                    }
            }
            .task(id: model.settingsNavigationActivationID) {
                // Let the lazily created stack finish its empty initialization
                // write before acknowledging a direct Models/Storage route.
                await Task.yield()
                model.activatePendingSettingsNavigation()
            }
        }

        private var path: Binding<[AppModel.SettingsSection]> {
            Binding(
                get: { model.settingsNavigationPath },
                set: model.setSettingsNavigationPath)
        }
    }

    /// iPad keeps the phone's two top-level destinations, but gives each one a
    /// native sidebar/detail relationship instead of stretching an iPhone
    /// push stack across a regular-width scene.
    private struct RegularChatsRoot: View {
        @Environment(AppModel.self) private var model

        var body: some View {
            NavigationSplitView {
                ConversationListView(openSettings: model.openSettings)
                    .navigationSplitViewColumnWidth(min: 280, ideal: 320, max: 380)
            } detail: {
                NavigationStack {
                    if case .conversation(let id) = model.destination,
                        model.conversation(id) != nil
                    {
                        ConversationView(conversationID: id)
                    } else {
                        IOSChatHome()
                    }
                }
            }
        }
    }

    private struct RegularSettingsRoot: View {
        @Environment(AppModel.self) private var model

        var body: some View {
            NavigationSplitView {
                IOSSettingsIndex(selection: selection)
                    .navigationSplitViewColumnWidth(min: 240, ideal: 280, max: 340)
            } detail: {
                NavigationStack {
                    if let section = model.settingsNavigationPath.last {
                        SettingsSectionView(section: section)
                    } else {
                        ContentUnavailableView(
                            "Settings", systemImage: "gearshape",
                            description: Text("Choose a section from the sidebar."))
                    }
                }
            }
            .task(id: model.settingsNavigationActivationID) {
                await Task.yield()
                model.activatePendingSettingsNavigation()
            }
        }

        private var selection: Binding<AppModel.SettingsSection?> {
            Binding(
                get: { model.settingsNavigationPath.last },
                set: { section in
                    if let section {
                        model.openSettings(section)
                    } else {
                        model.setSettingsNavigationPath([])
                    }
                })
        }
    }

    private struct IOSSettingsIndex: View {
        @Environment(AppModel.self) private var model
        var selection: Binding<AppModel.SettingsSection?>?

        init(selection: Binding<AppModel.SettingsSection?>? = nil) {
            self.selection = selection
        }

        var body: some View {
            if let selection {
                List(selection: selection) {
                    ForEach(visibleSections) { section in
                        Label(section.rawValue, systemImage: section.systemImage)
                            .tag(section)
                            .mrSettingsRow()
                    }
                }
                .mrSettingsList()
                .navigationTitle("Settings")
            } else {
                List(visibleSections) { section in
                    NavigationLink(value: section) {
                        Label(section.rawValue, systemImage: section.systemImage)
                    }
                    .mrSettingsRow()
                }
                .mrSettingsList()
                .navigationTitle("Settings")
            }
        }

        private var visibleSections: [AppModel.SettingsSection] {
            AppModel.SettingsSection.allCases.filter {
                $0 != .downloads || model.showsDownloadsSection
            }
        }
    }

    /// Honest empty/detail state for regular-width Chats. It projects the same
    /// readiness enum as the Mac workspace and uses `newReadyConversation`, so
    /// a runnable fallback model can create a chat without rewriting defaults.
    private struct IOSChatHome: View {
        @Environment(AppModel.self) private var model

        var body: some View {
            switch model.chatReadiness {
            case .ready(let modelID):
                ContentUnavailableView {
                    Label("Start a conversation", systemImage: "bubble.left.and.bubble.right")
                } description: {
                    Text("\(model.entry(modelID)?.descriptor.displayName ?? modelID.rawValue) is ready.")
                } actions: {
                    Button("New chat") { _ = model.newReadyConversation() }
                        .buttonStyle(.borderedProminent)
                }
            case .loadingCatalog, .scanningStorage:
                ContentUnavailableView {
                    Label("Getting ready", systemImage: "arrow.triangle.2.circlepath")
                } description: {
                    Text("Checking model and storage availability.")
                }
            case .needsStorage:
                ContentUnavailableView {
                    Label("Add model storage", systemImage: "externaldrive.badge.plus")
                } description: {
                    Text("Choose the folder that contains your Minirun models.")
                } actions: {
                    Button("Storage") { model.openSettings(.storage) }
                }
            case .needsVerification(let modelID):
                ContentUnavailableView {
                    Label("Verify model files", systemImage: "checkmark.shield")
                } description: {
                    Text("Verify \(model.entry(modelID)?.descriptor.displayName ?? modelID.rawValue) before starting a chat.")
                } actions: {
                    Button("Models") { model.openSettings(.models) }
                }
            case .catalogUnavailable, .needsModel:
                ContentUnavailableView {
                    Label("Find a model", systemImage: "square.stack.3d.up")
                } description: {
                    Text("Open Models to find or manage a Minirun model.")
                } actions: {
                    Button("Models") { model.openSettings(.models) }
                }
            }
        }
    }
#endif
