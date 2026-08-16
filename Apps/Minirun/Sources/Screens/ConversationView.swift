import MinirunKit
import SwiftUI

/// A conversation: the transcript, the live instruments, and the composer.
///
/// User turns align to the trailing edge and model turns to the leading edge,
/// matching the reading pattern of a familiar chat product. Runtime receipts
/// remain attached to the model reply because every answer here was produced by
/// a run whose budget was stated before it started.
///
/// The instrument strip stays on screen through the whole turn. It is the
/// product's soul and it does not collapse to nothing.
struct ConversationView: View {
    let conversationID: UUID

    @Environment(AppModel.self) private var model
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var showingCockpitSheet = false
    @State private var showingSettingsSheet = false
    #if os(iOS)
        @State private var settingsSheetDetent: PresentationDetent = .large
    #endif
    @State private var pendingFullVerification: DiscoveredArtifact?
    @State private var transcriptFollowsLatest = true
    @FocusState private var composerFocused: Bool

    private var conversation: Conversation? { model.conversation(conversationID) }
    private var snapshot: RunSnapshot { model.runSnapshot(for: conversationID) }
    private var isRunning: Bool { model.isRunning(conversationID) }
    private var isStopping: Bool { model.isStopping(conversationID) }

    private var entry: CatalogEntry? {
        conversation.flatMap { model.entry($0.settings.model) }
    }

    var body: some View {
        Group {
            #if os(macOS)
                VStack(spacing: 0) {
                    macHeader
                    Divider()
                    conversationContent
                }
            #else
                conversationContent
            #endif
        }
        .mrWorkspaceSurface()
        #if os(iOS)
            .navigationTitle(conversation?.title ?? "Chat")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { toolbar }
        #endif
        #if os(macOS)
            .onChange(of: isRunning) { _, running in
                if running, conversation?.settings.telemetryDensity == .cockpit {
                    model.conversationPanel = .instruments
                }
            }
        #else
            // A conversation that asks for automatic instruments gets them
            // when the turn starts. On iPhone they use a sheet rather than
            // covering the transcript permanently.
            .onChange(of: isRunning) { _, running in
                if running, conversation?.settings.telemetryDensity == .cockpit {
                    showingCockpitSheet = true
                }
            }
        #endif
        .sheet(isPresented: $showingCockpitSheet) { cockpitSheet }
        .sheet(isPresented: $showingSettingsSheet) {
            NavigationStack {
                ConversationSettingsView(conversationID: conversationID)
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Done") { showingSettingsSheet = false }
                        }
                    }
            }
            #if os(macOS)
                .frame(minWidth: 460, minHeight: 620)
            #else
                // The dial is the point of this sheet, and a medium detent cut
                // it in half on a 6.1-inch phone. It opens at full height; the
                // medium detent stays available for pulling it out of the way.
                .presentationDetents([.medium, .large], selection: $settingsSheetDetent)
                .presentationDragIndicator(.visible)
            #endif
        }
        #if os(iOS)
        .alert(
            "Verify every model file?",
            isPresented: fullVerificationConfirmationPresented
        ) {
            if let artifact = pendingFullVerification {
                Button(FullVerificationConfirmation.actionTitle) {
                    pendingFullVerification = nil
                    Task { await model.installed.verify(artifact, .full) }
                }
            }
            Button("Cancel", role: .cancel) { pendingFullVerification = nil }
        } message: {
            if let artifact = pendingFullVerification {
                Text(FullVerificationConfirmation.message(for: artifact))
            }
        }
        #else
        .confirmationDialog(
            "Verify every model file?",
            isPresented: fullVerificationConfirmationPresented,
            titleVisibility: .visible
        ) {
            if let artifact = pendingFullVerification {
                Button(FullVerificationConfirmation.actionTitle) {
                    pendingFullVerification = nil
                    Task { await model.installed.verify(artifact, .full) }
                }
            }
            Button("Cancel", role: .cancel) { pendingFullVerification = nil }
        } message: {
            if let artifact = pendingFullVerification {
                Text(FullVerificationConfirmation.message(for: artifact))
            }
        }
        #endif
    }

    @ViewBuilder private var conversationContent: some View {
        if let conversation {
            body(for: conversation)
        } else {
            EmptyStateView(
                headline: "This conversation is gone.",
                message: "It was deleted, or its document could not be read.")
        }
    }

    #if os(macOS)
        /// Product-owned header instead of a trailing toolbar group. AppKit
        /// draws a vertical separator before toolbar groups, and the old
        /// `sidebar.trailing` symbol looked like the control for the Chats
        /// column. This control-console symbol names what actually opens:
        /// per-chat controls and live instruments. When the panel is open, its
        /// copy occupies the same top-right window position instead of shifting
        /// left with the detail.
        private var macHeader: some View {
            MacColumnHeader(title: conversation?.title ?? "Chat") {
                if model.conversationPanel == nil {
                    Button {
                        model.conversationPanel = .chatSettings
                    } label: {
                        Image(systemName: ConversationPanelToggleStyle.symbolName)
                            .imageScale(.large)
                            .frame(width: 28, height: 28)
                    }
                    .buttonStyle(.plain)
                    .help(ConversationPanelToggleStyle.showHelp)
                    .accessibilityLabel("show chat controls and instruments")
                }
            }
        }
    #endif

    private func body(for conversation: Conversation) -> some View {
        VStack(spacing: 0) {
            banners(conversation)
            transcript(conversation)
            // The composer sits directly on the transcript. There used to be a
            // compact instrument strip here with a chevron that expanded a
            // second panel — a third place instruments could be, after the
            // side panel and the sheet. Instruments now live in exactly one
            // place: the right-hand panel, chosen from the "..." menu.
            Divider()
            composer(conversation)
        }
    }

    // MARK: - Transcript

    private func transcript(_ conversation: Conversation) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: MRSpace.s5) {
                    ForEach(conversation.messages) { message in
                        MessageBlock(
                            message: message,
                            modelName: model.entry(conversation.settings.model)?
                                .descriptor.displayName ?? conversation.settings.model.rawValue,
                            onRetry: message.failure?.allowsRetry == true
                                ? { model.regenerateLastTurn(in: conversationID) } : nil,
                            onVerifyModelFiles: message.failure == .verifyModelFiles
                                ? { beginFullVerification(for: conversation.settings.model) } : nil,
                            onReviewModel: message.failure == .reviewModel
                                ? { model.openSettings(.models) } : nil,
                            onAdjustChatSettings: message.failure == .adjustChatSettings
                                ? { showChatSettings() } : nil)
                            .id(message.id)
                    }
                    if isRunning || !snapshot.tokens.isEmpty && conversation.awaitsAnswer {
                        pendingTurn(conversation)
                            .id(Self.pendingID)
                    }
                    if case .refused(let error) = snapshot.state, conversation.awaitsAnswer {
                        refusedCard(error)
                    }
                    if let fault = snapshot.fault, conversation.awaitsAnswer {
                        faultCard(fault, conversation)
                    }
                }
                .padding(MRSpace.s5)
                .frame(maxWidth: 920, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .center)
            }
            #if os(iOS)
                .scrollDismissesKeyboard(.interactively)
            #endif
            .simultaneousGesture(
                DragGesture(minimumDistance: 4)
                    .onChanged { _ in transcriptFollowsLatest = false }
            )
            .onScrollGeometryChange(for: Bool.self) { geometry in
                geometry.visibleRect.maxY >= geometry.contentSize.height - 24
            } action: { _, isAtBottom in
                if isAtBottom { transcriptFollowsLatest = true }
            }
            .onChange(of: conversation.messages.count) { _, _ in
                guard transcriptFollowsLatest else { return }
                scrollToLatest(conversation.messages.last?.id, proxy: proxy)
            }
            .onChange(of: snapshot.tokens.count) { _, _ in
                guard transcriptFollowsLatest else { return }
                scrollToLatest(Self.pendingID, proxy: proxy)
            }
        }
    }

    private func scrollToLatest<ID: Hashable>(_ id: ID?, proxy: ScrollViewProxy) {
        guard let id else { return }
        withAnimation(reduceMotion ? nil : .easeOut(duration: 0.18)) {
            proxy.scrollTo(id, anchor: .bottom)
        }
    }

    private static let pendingID = "pending-turn"

    // An empty conversation shows the composer and nothing else. There used to
    // be an essay here — what a run is, what the budget means, and why a new K3
    // chat starts with a conservative token count. All of it is true; none of
    // it belongs in the way of the first thing somebody wants to type. The
    // budget and generation limit are in chat settings, one click away.

    /// The assistant turn being produced right now.
    private func pendingTurn(_ conversation: Conversation) -> some View {
        HStack(alignment: .top, spacing: MRSpace.s4) {
            VStack(alignment: .leading, spacing: MRSpace.s2) {
                HStack(alignment: .firstTextBaseline, spacing: 0) {
                    Text(
                        snapshot.generatedText
                            ?? snapshot.tokens.map { $0.text ?? "‹\($0.tokenID)›" }.joined())
                        .font(MRType.body)
                        .foregroundStyle(MRColor.primary)
                    Caret(active: isRunning)
                }
                .lineSpacing(5)
                if snapshot.tokens.isEmpty, let phase = snapshot.phase {
                    Text("\(phase.name) — \(phase.detail)")
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.tertiary)
                        .animation(nil, value: phase.detail)
                }
                if let footer = snapshot.reproducibility {
                    HStack(spacing: MRSpace.s2) {
                        Text(footer.line)
                            .font(MRType.micro)
                            .foregroundStyle(MRColor.secondary)
                            .textSelection(.enabled)
                        DigestLabel(digest: footer.logitsDigest, label: nil)
                    }
                }
            }
            .frame(maxWidth: 640, alignment: .leading)
            Spacer(minLength: MRSpace.s7)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("assistant turn in progress")
        .accessibilityValue(
            snapshot.tokens.isEmpty
                ? "no tokens yet"
                : snapshot.generatedText
                    ?? snapshot.tokens.map { $0.text ?? "token \($0.tokenID)" }.joined())
    }

    // MARK: - Cards and banners

    private func refusedCard(_ error: RunError) -> some View {
        NamedErrorCard(
            headline: "Minirun refused this turn.",
            message:
                "Nothing was read. The configuration was refused before the run started, "
                + "rather than adjusted until it fitted.",
            namedError: error.description,
            tone: .refuse,
            actions: AnyView(
                Button("Open this chat's settings") { openSettings() }
                    .controlSize(.small)))
    }

    private func faultCard(_ fault: RunFault, _ conversation: Conversation) -> some View {
        let recovery = fault.messageFailure
        return NamedErrorCard(
            headline: fault.headline,
            message: fault.body,
            namedError: fault.namedError,
            tone: fault.isFatal ? .refuse : .caution,
            numbers: [
                (
                    "bytes read",
                    fault.bytesReadSoFar.map { MRFormat.bytesDecimal($0) } ?? "Not reported"
                ),
                ("tokens completed", "\(fault.tokensCompleted)"),
            ],
            actions: AnyView(
                HStack(spacing: MRSpace.s2) {
                    if recovery.allowsRetry {
                        Button("Try this turn again") {
                            model.regenerateLastTurn(in: conversationID)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!model.canRun(conversation))
                    }
                    if recovery == .verifyModelFiles {
                        Button("Verify all model files") {
                            beginFullVerification(for: conversation.settings.model)
                        }
                    }
                    if recovery == .reviewModel {
                        Button("Review model") { model.openSettings(.models) }
                    }
                    if recovery == .adjustChatSettings {
                        Button("Chat settings") { showChatSettings() }
                    }
                    Button("Copy error") { copy(fault.namedError) }
                }
                .controlSize(.small)))
    }

    @ViewBuilder private func banners(_ conversation: Conversation) -> some View {
        let availability = model.modelAvailability(for: conversation.settings.model)
        if model.isAwaitingInitialCatalog {
            banner(
                tone: .neutral,
                text: "Preparing your models…",
                actionTitle: "View models") { model.openSettings(.models) }
        } else if !availability.isAvailable {
            let name = entry?.descriptor.displayName ?? conversation.settings.model.rawValue
            banner(
                tone: availability == .incompleteArtifact ? .refuse : .caution,
                text: "\(name) is unavailable: \(availability.productReason).",
                actionTitle: "Manage models") { model.openSettings(.models) }
        }
        if let reason = model.runBusyReason(for: conversationID) {
            banner(
                tone: .caution,
                text: reason,
                actionTitle: "Open running chat") { model.openRunningConversation() }
        }
        if let error = model.runPreparationError(for: conversationID) {
            if model.runPreparationBudgetTarget(for: conversationID) != nil {
                banner(
                    tone: .refuse,
                    text: error,
                    actionTitle: "Chat settings") { showChatSettings() }
            } else {
                banner(
                    tone: .refuse,
                    text: "This turn could not start: \(error)",
                    actionTitle: "Storage") { model.openSettings(.storage) }
            }
        }
        if let telemetry = snapshot.telemetry, isRunning,
            telemetry.thermalState == .serious || telemetry.thermalState == .critical
        {
            banner(
                tone: .refuse,
                text:
                    "Thermal state \(telemetry.thermalState.rawValue). Minirun is not "
                    + "throttling the run — the device is.",
                actionTitle: "Stop") { model.stopTurn() }
        }
        if let error = model.conversationStorageError {
            banner(
                tone: .refuse,
                text: error,
                actionTitle: "Storage") { model.openSettings(.storage) }
        } else if let error = model.conversationWriteError {
            banner(
                tone: .refuse,
                text: "This conversation could not be written to disk: \(error)",
                actionTitle: "Storage") { model.openSettings(.storage) }
        }
    }

    private func banner(
        tone: StatusChip.Tone, text: String, actionTitle: String, action: @escaping () -> Void
    ) -> some View {
        HStack(spacing: MRSpace.s3) {
            // Same rule as the composer footer: bounded lines, not `fixedSize`.
            // A banner sits above the transcript, so its minimum height is the
            // detail column's minimum height too.
            Text(text)
                .font(MRType.caption)
                .foregroundStyle(MRColor.primary)
                .lineLimit(3)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button(actionTitle, action: action)
                .controlSize(.small)
                .layoutPriority(1)
        }
        .padding(.horizontal, MRSpace.s4)
        .padding(.vertical, MRSpace.s2)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(tone.color.opacity(0.14))
        .overlay(alignment: .bottom) {
            Rectangle().fill(tone.color.opacity(0.4)).frame(height: 1)
        }
        .accessibilityElement(children: .contain)
    }

    // MARK: - Composer

    private func composer(_ conversation: Conversation) -> some View {
        @Bindable var model = model
        let plan = model.budgetPlan(for: conversation)
        return VStack(alignment: .leading, spacing: MRSpace.s2) {
            if let plan {
                if plan.refusal != nil {
                    RefusalCard(plan: plan) { target in
                        model.setBudget(target, in: conversationID)
                    }
                }
                if let refusal = plan.readAheadRefusal {
                    readAheadRecoveryCard(refusal)
                }
            }
            ConversationComposerBar(
                text: Binding(
                    get: { model.drafts[conversationID] ?? "" },
                    set: { model.setDraft($0, in: conversationID) }),
                isRunning: isRunning,
                isStopping: isStopping,
                canSend: canSend(conversation),
                isInputDisabled: model.hasActiveRun,
                focus: $composerFocused,
                onSend: send,
                onStop: { model.stopTurn() })
            if isStopping {
                Text("Finishing the current layer before releasing model memory.")
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityLabel(
                        "Stopping safely after the current model layer, then releasing model memory")
            }
        }
        .padding(composerPadding)
        .background(Color.clear)
        .onAppear { composerFocused = true }
    }

    private var composerPadding: CGFloat {
        #if os(iOS)
            MRSpace.s3
        #else
            MRSpace.s4
        #endif
    }

    /// Old conversation documents may contain a depth that the runtime never
    /// implemented. Keep the saved statement intact, explain why Send is
    /// disabled, and let the person choose the replacement explicitly. The
    /// composer used to render an empty budget-refusal card for this case,
    /// leaving a disabled Send button with no visible way forward.
    private func readAheadRecoveryCard(_ refusal: ReadAheadRefusal) -> some View {
        NamedErrorCard(
            headline: "Choose Auto or Off to continue.",
            message: refusal.message,
            namedError: refusal.namedError,
            tone: .caution,
            actions: AnyView(
                HStack(spacing: MRSpace.s2) {
                    Button("Use Auto") {
                        model.update(conversationID) {
                            $0.settings.deterministicReadAheadLayers = nil
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .mrMinimumTouchTarget()

                    Button("Turn Off") {
                        model.update(conversationID) {
                            $0.settings.deterministicReadAheadLayers = 0
                        }
                    }
                    .buttonStyle(.bordered)
                    .mrMinimumTouchTarget()
                }
                .controlSize(.small)
            )
        )
    }

    private func canSend(_ conversation: Conversation) -> Bool {
        guard !(model.drafts[conversationID] ?? "").trimmingCharacters(
            in: .whitespacesAndNewlines
        ).isEmpty else { return false }
        return model.canStartTurn(in: conversationID) && model.canRun(conversation)
    }

    private func send() {
        model.send(model.drafts[conversationID] ?? "", in: conversationID)
    }

    private var fullVerificationConfirmationPresented: Binding<Bool> {
        Binding(
            get: { pendingFullVerification != nil },
            set: { presented in
                if !presented { pendingFullVerification = nil }
            })
    }

    /// A runtime short read can originate from an existing copy or a completed
    /// download. Discovery is the current authority in both cases. Full
    /// verification is always confirmed because a flagship pass may read more
    /// than a terabyte; if no current copy exists, Models is the honest next
    /// screen instead of a no-op button.
    private func beginFullVerification(for modelID: ModelID) {
        guard let artifact = model.installed.preferredArtifact(for: modelID) else {
            model.openSettings(.models)
            return
        }
        pendingFullVerification = artifact
    }

    // MARK: - Toolbar

    private func showChatSettings() {
        #if os(macOS)
            model.conversationPanel = .chatSettings
        #else
            settingsSheetDetent = .large
            showingSettingsSheet = true
        #endif
    }

    #if os(iOS)
        @ToolbarContentBuilder private var toolbar: some ToolbarContent {
            if conversation != nil {
                ToolbarItem(placement: .primaryAction) { panelMenu }
            }
        }
    #endif

    /// The phone has no persistent inspector, so its one button remains a menu
    /// for the two sheets.
    private var panelMenu: some View {
        Menu {
            Button {
                showChatSettings()
            } label: {
                Label("Chat settings", systemImage: "slider.horizontal.3")
            }
            Button {
                showingCockpitSheet = true
            } label: {
                Label(
                    "Instruments",
                    systemImage: "gauge.with.dots.needle.bottom.50percent")
            }
        } label: {
            Image(systemName: "ellipsis")
                .frame(minWidth: 44, minHeight: 44)
        }
        .menuIndicator(.hidden)
        .help("Choose what the panel shows")
        .accessibilityLabel("panel menu")
    }

    /// On the Mac the settings live in the panel, which is where the operator
    /// can watch the dial and the transcript at once; on the phone they are a
    /// sheet.
    private func openSettings() {
        #if os(macOS)
            model.conversationPanel = .chatSettings
        #else
            settingsSheetDetent = .large
            showingSettingsSheet = true
        #endif
    }

    private var cockpitSheet: some View {
        NavigationStack {
            InstrumentPanelView(snapshot: snapshot)
                .navigationTitle("Instruments")
                #if os(iOS)
                    .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    #if os(iOS)
                        if isRunning {
                            ToolbarItem(placement: .cancellationAction) {
                                Button(isStopping ? "Stopping…" : "Stop", role: .destructive) {
                                    model.stopTurn()
                                }
                                .disabled(isStopping)
                            }
                        }
                    #endif
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { showingCockpitSheet = false }
                    }
                }
        }
        #if os(macOS)
            .frame(minWidth: 420, minHeight: 620)
        #else
            .presentationDetents([.large])
            .presentationDragIndicator(.visible)
        #endif
    }

    private func copy(_ text: String) {
        #if canImport(AppKit)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
        #elseif canImport(UIKit)
            UIPasteboard.general.string = text
        #endif
    }
}

// MARK: - The composer

/// The message field and the one control beside it, in each platform's idiom.
///
/// The Mac keeps the band the previous polish round settled on: a bordered
/// field and a prominent labelled button, both exactly `MRComposer.controlHeight`
/// tall so a single line of text is one horizontal row.
///
/// The phone does not have that idiom. A grey rounded rectangle reading `Send`,
/// taller than the field it sits beside, is a desktop control transplanted onto
/// a phone; every iOS product a person already uses puts a capsule field under
/// the thumb and a small circular glyph at its trailing edge. That is what this
/// draws, with the same one-slot rule: while a turn runs the circle becomes the
/// red stop control, and the stopping state keeps its spinner.
struct ConversationComposerBar: View {
    @Binding var text: String
    let isRunning: Bool
    let isStopping: Bool
    let canSend: Bool
    let isInputDisabled: Bool
    var focus: FocusState<Bool>.Binding?
    let onSend: () -> Void
    let onStop: () -> Void

    /// The glyph's drawn size on iOS. The tappable frame around it is the
    /// platform's 44-point minimum; the circle itself stays compact so it does
    /// not out-weigh the field.
    static let iOSControlGlyphSize: CGFloat = 32

    var body: some View {
        #if os(iOS)
            phoneBar
        #else
            desktopBar
        #endif
    }

    // MARK: iOS

    #if os(iOS)
        private var phoneBar: some View {
            HStack(alignment: .bottom, spacing: MRSpace.s2) {
                messageField
                    .padding(.horizontal, MRSpace.s3)
                    .padding(.vertical, 9)
                    .frame(minHeight: 38)
                    .background(
                        Capsule(style: .continuous).fill(MRColor.raised)
                    )
                    .overlay(
                        Capsule(style: .continuous)
                            .strokeBorder(MRColor.hairline, lineWidth: 1)
                    )
                phoneControl
            }
        }

        @ViewBuilder private var phoneControl: some View {
            if isRunning {
                Button(action: onStop) {
                    ZStack {
                        Image(systemName: "stop.circle.fill")
                            .font(.system(size: Self.iOSControlGlyphSize))
                            .foregroundStyle(MRColor.refuse)
                            .opacity(isStopping ? 0.35 : 1)
                        if isStopping {
                            ProgressView()
                                .controlSize(.small)
                                .tint(MRColor.refuse)
                        }
                    }
                    .frame(
                        width: MRAccessibility.minimumIOSTouchTarget,
                        height: MRAccessibility.minimumIOSTouchTarget)
                    .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .keyboardShortcut(".", modifiers: .command)
                .disabled(isStopping)
                .accessibilityLabel(isStopping ? "stopping this turn" : "stop this turn")
            } else {
                Button(action: onSend) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: Self.iOSControlGlyphSize))
                        .foregroundStyle(canSend ? MRColor.tierPinned : MRColor.tertiary)
                        .frame(
                            width: MRAccessibility.minimumIOSTouchTarget,
                            height: MRAccessibility.minimumIOSTouchTarget)
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(!canSend)
                .accessibilityLabel("send")
            }
        }
    #endif

    // MARK: macOS

    #if os(macOS)
        private var desktopBar: some View {
            HStack(alignment: .bottom, spacing: MRSpace.s3) {
                messageField
                    .padding(.horizontal, MRSpace.s3)
                    .padding(.vertical, MRSpace.s2)
                    // A single line of the field is exactly the button's height,
                    // so the two controls form one band instead of a tall box
                    // with a small control hanging off its bottom edge.
                    .frame(minHeight: MRComposer.controlHeight)
                    .mrCard(MRColor.raised, radius: MRRadius.control)

                // One slot, two states. The platform's prominent button, at the
                // field's own height and corner radius, rather than a default
                // control left to size itself.
                if isRunning {
                    Button(action: onStop) {
                        desktopButtonLabel {
                            if isStopping {
                                ProgressView()
                                    .controlSize(.small)
                                    .scaleEffect(0.7)
                            }
                            Text(isStopping ? "Stopping…" : "Stop")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .buttonBorderShape(.roundedRectangle(radius: MRRadius.control))
                    .tint(MRColor.refuse)
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(isStopping)
                    .accessibilityLabel(isStopping ? "stopping this turn" : "stop this turn")
                } else {
                    Button(action: onSend) {
                        desktopButtonLabel { Text("Send") }
                    }
                    .buttonStyle(.borderedProminent)
                    .buttonBorderShape(.roundedRectangle(radius: MRRadius.control))
                    .tint(MRColor.tierPinned)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(!canSend)
                    .accessibilityLabel("send")
                }
            }
        }

        /// The composer button's label geometry, shared by Send and Stop so the
        /// two states cannot differ in height, weight or minimum width as they
        /// swap.
        private func desktopButtonLabel<Content: View>(
            @ViewBuilder _ content: () -> Content
        ) -> some View {
            HStack(spacing: MRSpace.s2, content: content)
                .font(MRType.body.weight(.semibold))
                .frame(minWidth: 64)
                .frame(height: MRComposer.controlHeight)
                .contentShape(Rectangle())
        }
    #endif

    // MARK: Shared

    @ViewBuilder private var messageField: some View {
        let field = TextField("Message", text: $text, axis: .vertical)
            .textFieldStyle(.plain)
            .font(MRType.body)
            .lineLimit(fieldLineLimit)
            .accessibilityLabel("message")
            .disabled(isInputDisabled)
        if let focus {
            field.focused(focus)
        } else {
            field
        }
    }

    private var fieldLineLimit: ClosedRange<Int> {
        #if os(iOS)
            1...5
        #else
            1...6
        #endif
    }
}

// MARK: - One turn

enum ConversationErrorCopy {
    static func summary(for namedError: String, failure: MessageFailure? = nil) -> String {
        if let failure { return failure.summary }
        // Migration only: conversations written before MessageFailure existed
        // have an exact error string but no structured recovery. New turns
        // never use this text inspection to choose an action.
        let lowercased = namedError.lowercased()
        if lowercased.contains("layer census") {
            return "The model files changed and this reply could not start."
        }
        if lowercased.contains("cancelled") || lowercased.contains("stopped") {
            return "Generation stopped."
        }
        return "Minirun could not complete this reply."
    }
}

/// A familiar two-sided chat turn. User messages sit on the trailing edge;
/// model messages sit on the leading edge. Receipts remain attached to the
/// model turn, while raw engineering failures live behind a collapsed detail
/// instead of masquerading as an answer.
struct MessageBlock: View {
    let message: Message
    let modelName: String
    var onRetry: (() -> Void)?
    var onVerifyModelFiles: (() -> Void)?
    var onReviewModel: (() -> Void)?
    var onAdjustChatSettings: (() -> Void)?
    @State private var showsErrorDetails = false
    @State private var showsPhaseSplit = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var isUser: Bool { message.role == .user }

    var body: some View {
        HStack(alignment: .top, spacing: MRSpace.s4) {
            if isUser { Spacer(minLength: MRSpace.s7) }
            bubble
            if !isUser { Spacer(minLength: MRSpace.s7) }
        }
        .frame(maxWidth: .infinity)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(isUser ? "your message" : "\(modelName) reply")
        .accessibilityValue(spokenValue)
    }

    @ViewBuilder private var bubble: some View {
        if isUser {
            messageContent
                .padding(.horizontal, MRSpace.s4)
                .padding(.vertical, MRSpace.s3)
                .background(
                    MRColor.tierPinned.opacity(0.13),
                    in: RoundedRectangle(cornerRadius: MRRadius.card, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: MRRadius.card, style: .continuous)
                        .stroke(MRColor.tierPinned.opacity(0.25), lineWidth: 1)
                }
                .frame(maxWidth: 640, alignment: .trailing)
        } else {
            messageContent
                .frame(maxWidth: 640, alignment: .leading)
        }
    }

    private var messageContent: some View {
        VStack(alignment: isUser ? .trailing : .leading, spacing: MRSpace.s2) {
            if !message.text.isEmpty { messageText }
            if let telemetry = message.telemetry {
                telemetryLine(telemetry)
                phaseSplitDisclosure(telemetry)
            }
            if let namedError = message.namedError { conversationError(namedError) }
        }
    }

    private var messageText: some View {
        Text(message.text)
            .font(MRType.body)
            .foregroundStyle(MRColor.primary)
            .textSelection(.enabled)
            .lineSpacing(5)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func telemetryLine(_ telemetry: MessageTelemetry) -> some View {
        HStack(spacing: MRSpace.s2) {
            Text(telemetry.line)
                .font(MRType.micro)
                .foregroundStyle(MRColor.secondary)
                .textSelection(.enabled)
            if !telemetry.budgetRespected {
                StatusChip(text: "budget exceeded", tone: .refuse)
            }
            if let digest = telemetry.logitsDigest {
                DigestLabel(digest: digest, label: nil, showsCopyButton: false)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    /// The turn's own decomposition, folded away.
    ///
    /// A transcript is a conversation, not a cockpit: the bars are here because
    /// this record kept them, and they stay shut until somebody asks. A turn
    /// recorded before the engines published a split, or one whose runner could
    /// not attribute a pass, shows no control at all rather than an empty one.
    @ViewBuilder private func phaseSplitDisclosure(_ telemetry: MessageTelemetry) -> some View {
        let bars = [telemetry.prefillPhase, telemetry.decodePhaseAggregate].compactMap { $0 }
        if !bars.isEmpty {
            VStack(alignment: .leading, spacing: MRSpace.s2) {
                Button {
                    withAnimation(reduceMotion ? nil : .easeInOut(duration: MRMotion.quick)) {
                        showsPhaseSplit.toggle()
                    }
                } label: {
                    HStack(spacing: MRSpace.s1) {
                        Image(systemName: showsPhaseSplit ? "chevron.down" : "chevron.right")
                            .imageScale(.small)
                        Text("Where the time went")
                    }
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.secondary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("where the time went")
                .accessibilityHint(showsPhaseSplit ? "hides the split" : "shows the split")

                if showsPhaseSplit {
                    VStack(alignment: .leading, spacing: MRSpace.s3) {
                        ForEach(bars, id: \.passKind) { bar in
                            PhaseSplitBar(summary: bar, namedTermLimit: 4)
                        }
                    }
                    .padding(MRSpace.s3)
                    .frame(maxWidth: 420, alignment: .leading)
                    .mrCard()
                }
            }
        }
    }

    private var spokenValue: String {
        var value = message.text
        if let namedError = message.namedError {
            value += ". " + ConversationErrorCopy.summary(
                for: namedError, failure: message.failure)
        }
        if let telemetry = message.telemetry { value += ". " + telemetry.spokenLine }
        return value
    }

    private func conversationError(_ namedError: String) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
                Label(
                    ConversationErrorCopy.summary(for: namedError, failure: message.failure),
                    systemImage: namedError.lowercased().contains("cancelled")
                        ? "stop.circle" : "exclamationmark.triangle")
                    .font(MRType.caption)
                    .foregroundStyle(MRColor.caution)
                Button {
                    withAnimation(reduceMotion ? nil : .easeInOut(duration: 0.15)) {
                        showsErrorDetails.toggle()
                    }
                } label: {
                    Image(systemName: "chevron.right")
                        .rotationEffect(.degrees(showsErrorDetails ? 90 : 0))
                        .frame(minWidth: 44, minHeight: 44)
                }
                .buttonStyle(.plain)
                .foregroundStyle(MRColor.secondary)
                .help(showsErrorDetails ? "Hide error information" : "Show error information")
                .accessibilityLabel(
                    showsErrorDetails ? "hide error information" : "show error information")
            }
            if showsErrorDetails {
                Text(namedError)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            recoveryActions
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder private var recoveryActions: some View {
        if onRetry != nil || onVerifyModelFiles != nil || onReviewModel != nil
            || onAdjustChatSettings != nil
        {
            HStack(spacing: MRSpace.s2) {
                if let onRetry {
                    Button("Try this turn again", action: onRetry)
                        .buttonStyle(.borderedProminent)
                }
                if let onVerifyModelFiles {
                    Button("Verify all model files", action: onVerifyModelFiles)
                }
                if let onReviewModel {
                    Button("Review model", action: onReviewModel)
                }
                if let onAdjustChatSettings {
                    Button("Chat settings", action: onAdjustChatSettings)
                }
            }
            .controlSize(.small)
        }
    }
}

/// A thin caret at the insertion point, blinking at 1 Hz only while a token is
/// in flight. Still under Reduce Motion.
struct Caret: View {
    let active: Bool
    @State private var visible = true
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Rectangle()
            .fill(MRColor.streamDet)
            .frame(width: 2, height: 18)
            .opacity(active ? (visible ? 1 : 0) : 0)
            .task(id: active) {
                guard active, !reduceMotion else {
                    visible = true
                    return
                }
                while !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(500))
                    visible.toggle()
                }
            }
            .accessibilityHidden(true)
    }
}
