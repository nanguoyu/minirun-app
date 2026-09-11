import SwiftUI

/// Settings: a surface, not a page.
///
/// It has its own left nav and four persistent sections. Downloads joins them
/// only while there is transfer or recovery state worth showing. Everything
/// that is not a conversation lives here — including **Models**, which used to be a
/// sidebar row of its own. The audit that produced the split is still the rule:
///
/// | was here | went |
/// | --- | --- |
/// | memory dial (per selected model) | the conversation — every turn runs at the budget its chat states |
/// | read-ahead depth | the conversation |
/// | selected model | the conversation |
/// | prompt / prompt token ids | the conversation's composer |
/// | storage & drive assessment | stayed — a volume is a property of the machine |
/// | About | moved IN, so Settings is one place instead of two |
/// | the model catalog | moved IN, because the sidebar is for conversations |
///
/// What is left that touches a run is a *default*, and it says so: it is copied
/// onto a conversation at creation and never read again.
///
/// Phase 2 of the product-page language (DESIGN I.37) reaches these screens:
/// the panels and their ALL-CAPS labels are gone, a section is a hairline and a
/// 13 pt sentence-case heading, a preference is a label with its control on the
/// same line, and the only warnings drawn are the ones with something to do
/// about them.
struct SettingsView: View {
    var body: some View {
        #if os(macOS)
            MacSettingsView()
        #else
            PhoneSettingsView()
        #endif
    }
}

// MARK: - The rail

/// One row of the Settings navigation: a symbol, its name in sentence case,
/// and — when it is the section on screen — a soft wash of the product accent
/// behind it.
///
/// The rail around it is still a real sidebar `List`, so arrow keys, hover and
/// the sidebar material stay the platform's. What this replaces is the system's
/// emphasized blue capsule, which was the one selection colour in the product
/// that was not the product's own accent.
struct SettingsSidebarRow: View {
    let section: AppModel.SettingsSection
    var isSelected = false

    var body: some View {
        // The sidebar `List` draws the selection itself — the system's rounded
        // accent highlight with inverted text. Painting a second fill and an
        // accent-coloured label on top of it produced blue-on-blue; the row
        // therefore sets no colour of its own while selected and lets the
        // list's highlight carry the state.
        HStack(spacing: MRSpace.s2) {
            Image(systemName: section.systemImage)
                .imageScale(.medium)
                .frame(width: 20, alignment: .center)
                .foregroundStyle(isSelected ? AnyShapeStyle(.primary) : AnyShapeStyle(MRColor.secondary))
                .accessibilityHidden(true)
            Text(section.rawValue)
                .font(MRType.prose.weight(isSelected ? .semibold : .regular))
            Spacer(minLength: 0)
        }
        .padding(.vertical, 2)
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }
}

// MARK: - macOS: left nav plus a page

#if os(macOS)
    /// Settings owns a fixed navigation rail and a push-capable detail stack.
    /// The rail still uses a real sidebar `List`, so keyboard navigation and
    /// material remain native. It deliberately is not a
    /// second `NavigationSplitView`: nesting another split in the same window
    /// makes SwiftUI recreate a window-toolbar sidebar button above Minirun's
    /// own header, shifting the title down and drawing the stray separator the
    /// product review caught.
    struct MacSettingsView: View {
        @Environment(AppModel.self) private var model

        var body: some View {
            @Bindable var model = model
            HStack(spacing: 0) {
                VStack(spacing: 0) {
                    MacSidebarHeader(title: "Settings")
                    List(selection: $model.settingsSection) {
                        ForEach(visibleSections) { section in
                            SettingsSidebarRow(
                                section: section,
                                isSelected: model.settingsSection == section)
                                .tag(section)
                        }
                    }
                    .listStyle(.sidebar)
                    .scrollContentBackground(.hidden)
                }
                .safeAreaInset(edge: .bottom, spacing: 0) {
                    VStack(spacing: 0) {
                        MRHairline()
                        Button {
                            model.backToApp()
                        } label: {
                            HStack(spacing: MRSpace.s2) {
                                Image(systemName: "chevron.left")
                                    .imageScale(.small)
                                Text("Back to app")
                            }
                            .padding(.horizontal, MRSpace.s3)
                            .padding(.vertical, MRSpace.s2)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .contentShape(Rectangle())
                        }
                        .mrTextLink()
                        .accessibilityLabel("back to app")
                    }
                }
                .frame(width: ConversationWorkspaceLayout.sidebarWidth)
                .mrSidebarSurface()

                // A vertical rule, so `Divider` and not `MRHairline`: the
                // page's hairline is a 1-point-high rectangle and would draw
                // as a line across the top of the detail column instead of
                // down its leading edge.
                Divider()

                // Its own stack: a model pushed from the catalog must not cover
                // the nav that got you there.
                NavigationStack {
                    SettingsSectionView(section: model.settingsSection)
                }
                .id(model.settingsSection)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .background(MinirunWindowChrome())
            .onChange(of: model.showsDownloadsSection) { _, isVisible in
                if !isVisible, model.settingsSection == .downloads {
                    model.settingsSection = .models
                }
            }
        }

        private var visibleSections: [AppModel.SettingsSection] {
            AppModel.SettingsSection.allCases.filter {
                $0 != .downloads || model.showsDownloadsSection
            }
        }
    }
#endif

// MARK: - iOS: a list that pushes

#if os(iOS)
    struct PhoneSettingsView: View {
        @Environment(AppModel.self) private var model

        var body: some View {
            List {
                ForEach(visibleSections) { section in
                    NavigationLink {
                        SettingsSectionView(section: section)
                    } label: {
                        SettingsSidebarRow(section: section)
                    }
                    .mrSettingsRow()
                }
            }
            .mrSettingsList()
            .navigationTitle("Settings")
        }

        private var visibleSections: [AppModel.SettingsSection] {
            AppModel.SettingsSection.allCases.filter {
                $0 != .downloads || model.showsDownloadsSection
            }
        }
    }
#endif

// MARK: - The sections

/// One page of Settings, shared by both platforms.
struct SettingsSectionView: View {
    let section: AppModel.SettingsSection

    @ViewBuilder
    var body: some View {
        #if os(macOS)
            VStack(spacing: 0) {
                MacColumnHeader(title: section.rawValue) {
                    if section == .storage {
                        StorageRefreshButton()
                    } else if section == .models {
                        FindModelsButton()
                    }
                }
                MRHairline()
                content
            }
            .modifier(SettingsSectionSurface(section: section))
        #else
            content
        #endif
    }

    @ViewBuilder private var content: some View {
        switch section {
        case .general: GeneralSettingsView()
        case .storage: StorageSettingsView()
        case .models: ModelCatalogView()
        case .downloads: DownloadsView()
        case .about: AboutView()
        }
    }
}

/// Which of the two surfaces a Settings section is drawn on.
///
/// General, Storage and About are product pages now — one sheet of `panel`
/// divided by hairlines — so their column header has to sit on the same sheet
/// or the seam shows above the first section. Models and Downloads still speak
/// the panel language and stack cards on `abyss`; they take the workspace
/// surface until phase 2 reaches them.
private struct SettingsSectionSurface: ViewModifier {
    let section: AppModel.SettingsSection

    @ViewBuilder
    func body(content: Content) -> some View {
        switch section {
        case .general, .storage, .about: content.mrProductPage()
        case .models, .downloads: content.mrWorkspaceSurface()
        }
    }
}

// MARK: - A preference row

/// One line of a settings group: the name of a preference, the sentence that
/// says what it does, and the control that changes it.
///
/// The same skeleton as `MRFact` — a label column of one width, so every
/// control on the page starts at the same x — with room for the sentence a
/// preference needs and a fact does not. It replaces `PreferenceRow`, whose
/// bordered card, 24-point glyph and two-line label made every preference look
/// like an instrument.
struct SettingRow<Control: View>: View {
    let title: String
    var caption: String?
    @ViewBuilder var control: Control

    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    /// Wide enough for "Instruments panel" and its sentence to breathe, and
    /// narrow enough that a menu beside it is still a control rather than a
    /// second column of prose.
    static var labelWidth: CGFloat { 260 }

    var body: some View {
        MRSettingRowLayout(
            labelWidth: Self.labelWidth,
            forcesStack: PreferenceRowLayoutPolicy.usesVerticalLayout(
                horizontalSizeClass: horizontalSizeClass,
                dynamicTypeSize: dynamicTypeSize)
        ) {
            label
            control
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .contain)
    }

    private var label: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title)
                .font(MRType.prose.weight(.medium))
                .foregroundStyle(MRColor.primary)
                .fixedSize(horizontal: false, vertical: true)
            if let caption {
                Text(caption)
                    .font(MRType.prose)
                    .foregroundStyle(MRColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Label and control on one line while the page is wide enough to read them as
/// two columns, stacked when it is not.
///
/// Neither `ViewThatFits` nor `horizontalSizeClass` can make this call.
/// `ViewThatFits` asks each candidate for its ideal width, and a wrapping
/// caption answers with the width of its longest sentence on one line, so the
/// wide candidate never "fits"; `horizontalSizeClass` does not exist on macOS
/// at all, so a 390-point Mac window kept the wide layout and pushed its
/// controls off the page — which is exactly the bug the first render of this
/// revision showed. Asking the container how wide it is is the actual
/// question, and it is the same conclusion `MRColumns` reached.
struct MRSettingRowLayout: Layout {
    var labelWidth: CGFloat = 260
    var spacing: CGFloat = MRSpace.s3
    var stackSpacing: CGFloat = MRSpace.s2
    /// Below this the row stacks: a 260-point label beside a menu needs about
    /// this much before the control has anywhere to be.
    var minimumWideWidth: CGFloat = 560
    /// An accessibility text size stacks at any width, because the label alone
    /// then wants the whole row.
    var forcesStack = false

    func sizeThatFits(
        proposal: ProposedViewSize, subviews: Subviews, cache: inout ()
    ) -> CGSize {
        guard subviews.count == 2 else {
            return CGSize(width: proposal.width ?? 0, height: 0)
        }
        let width = proposal.width ?? minimumWideWidth
        if isWide(width) {
            let label = subviews[0].sizeThatFits(
                ProposedViewSize(width: labelWidth, height: nil))
            let control = subviews[1].sizeThatFits(
                ProposedViewSize(width: controlWidth(in: width), height: nil))
            return CGSize(width: width, height: max(label.height, control.height))
        }
        let label = subviews[0].sizeThatFits(ProposedViewSize(width: width, height: nil))
        let control = subviews[1].sizeThatFits(ProposedViewSize(width: width, height: nil))
        return CGSize(width: width, height: label.height + stackSpacing + control.height)
    }

    func placeSubviews(
        in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()
    ) {
        guard subviews.count == 2 else { return }
        if isWide(bounds.width) {
            subviews[0].place(
                at: CGPoint(x: bounds.minX, y: bounds.minY), anchor: .topLeading,
                proposal: ProposedViewSize(width: labelWidth, height: nil))
            subviews[1].place(
                at: CGPoint(x: bounds.minX + labelWidth + spacing, y: bounds.minY),
                anchor: .topLeading,
                proposal: ProposedViewSize(width: controlWidth(in: bounds.width), height: nil))
            return
        }
        let labelHeight = subviews[0]
            .sizeThatFits(ProposedViewSize(width: bounds.width, height: nil)).height
        subviews[0].place(
            at: CGPoint(x: bounds.minX, y: bounds.minY), anchor: .topLeading,
            proposal: ProposedViewSize(width: bounds.width, height: nil))
        subviews[1].place(
            at: CGPoint(x: bounds.minX, y: bounds.minY + labelHeight + stackSpacing),
            anchor: .topLeading,
            proposal: ProposedViewSize(width: bounds.width, height: nil))
    }

    /// Whether a row of `width` points reads as two columns. A value, so the
    /// suite can check the threshold without laying anything out.
    func isWide(_ width: CGFloat) -> Bool { !forcesStack && width >= minimumWideWidth }

    private func controlWidth(in width: CGFloat) -> CGFloat {
        max(0, width - labelWidth - spacing)
    }
}

/// A value a settings row states rather than offers: a version, a last-checked
/// date, a size. Tabular, in the page's second ink, never mono.
struct SettingFigure: View {
    let text: String
    var color: Color = MRColor.secondary

    var body: some View {
        Text(text)
            .font(MRType.figure)
            .foregroundStyle(color)
            .fixedSize(horizontal: false, vertical: true)
    }
}

// MARK: - General

/// Defaults a NEW chat starts from, plus where conversations are kept.
///
/// Appearance, defaults for new chats, and private conversation storage.
struct GeneralSettingsView: View {
    @Environment(AppModel.self) private var model

    @ViewBuilder
    var body: some View {
        #if os(iOS)
            phoneForm
        #else
            ScrollView {
                GeneralSettingsPage()
            }
            .frame(maxWidth: .infinity)
        #endif
    }

    // MARK: - iOS: a native inset-grouped form
    //
    // The desktop page stacks preference rows under sentence-case headings. On
    // the phone that reads as a Mac layout ported across: a "System ⌄" menu on
    // its own line beneath the word Theme, a switch under the words that
    // describe it, and a lone "—" where a value belongs. The platform's own
    // answer is one row per setting, label leading, control trailing, with the
    // explanation as a section footer — and that is what I.29 established here.

    #if os(iOS)
        private var phoneForm: some View {
            Form {
                Section {
                    Picker("Theme", selection: appearanceBinding) {
                        ForEach(AppAppearance.allCases) { appearance in
                            Text(appearance.rawValue).tag(appearance)
                        }
                    }
                    .pickerStyle(.menu)
                    .mrSettingsRow()
                } header: {
                    Text("Appearance")
                } footer: {
                    Text("Use the system setting, or choose this app's appearance.")
                }

                newChatsSection
                conversationIssuesSection
            }
            .mrSettingsList()
            .navigationTitle("General")
        }

        private var newChatsSection: some View {
            let choices = model.modelPickerChoices(current: model.defaults.model)
            let currentIsAvailable = choices.contains { $0.modelID == model.defaults.model }
            let availability = model.modelAvailability(for: model.defaults.model)
            return Section {
                phoneModelRow(choices: choices, currentIsAvailable: currentIsAvailable)
                phoneBudgetRow(availability: availability)
                Toggle(
                    "Instruments panel",
                    isOn: Binding(
                        get: { model.defaults.telemetryDensity == .cockpit },
                        set: { model.defaults.telemetryDensity = $0 ? .cockpit : .strip })
                )
                .mrSettingsRow()
            } header: {
                Text("New chats")
            } footer: {
                VStack(alignment: .leading, spacing: MRSpace.s2) {
                    Text(
                        "These are copied onto a chat when it is created. Instruments opens "
                            + "live performance metrics when generation begins."
                    )
                    if !availability.isAvailable {
                        phoneUnavailableFooter(availability)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }

        /// A picker when there is something to pick, and an honest trailing
        /// value when there is not. The previous screen hung a "No verified
        /// models" menu button under the label, which looked like a control
        /// somebody had failed to fill in.
        @ViewBuilder private func phoneModelRow(
            choices: [ModelPickerChoice], currentIsAvailable: Bool
        ) -> some View {
            if currentIsAvailable, !choices.isEmpty {
                Picker(
                    "Model",
                    selection: Binding(
                        get: { model.defaults.model },
                        set: { newValue in
                            guard model.modelAvailability(for: newValue).isAvailable else {
                                return
                            }
                            model.setDefaultModel(newValue)
                        })
                ) {
                    ForEach(choices) { choice in
                        if let id = choice.modelID {
                            Text(choice.displayName).tag(id)
                        }
                    }
                }
                .pickerStyle(.menu)
                .mrSettingsRow()
            } else if choices.isEmpty {
                LabeledContent("Model") {
                    Text("None available").foregroundStyle(MRColor.secondary)
                }
                .mrSettingsRow()
            } else {
                Menu("Choose model") {
                    ForEach(choices) { choice in
                        if let id = choice.modelID {
                            Button(choice.displayName) { model.setDefaultModel(id) }
                        }
                    }
                }
                .mrSettingsRow()
            }
        }

        @ViewBuilder private func phoneBudgetRow(
            availability: ModelRunAvailability
        ) -> some View {
            if availability.isAvailable,
                let plan = model.defaultBudgetPlan(for: model.defaults.model),
                let entry = model.entry(model.defaults.model)
            {
                NavigationLink {
                    DefaultBudgetPage(plan: plan, entry: entry)
                } label: {
                    LabeledContent("Memory budget") {
                        SettingFigure(
                            text: MRFormat.bytesDecimal(plan.budgetBytes),
                            color: plan.isRunnable ? MRColor.primary : MRColor.refuse)
                    }
                }
                .mrSettingsRow()
            } else {
                // Not "—". A dash is a value; this is the absence of one, and
                // the row says which fact would make it appear.
                LabeledContent("Memory budget") {
                    Text("Set when a model is ready").foregroundStyle(MRColor.secondary)
                }
                .mrSettingsRow()
            }
        }

        private func phoneUnavailableFooter(
            _ availability: ModelRunAvailability
        ) -> some View {
            VStack(alignment: .leading, spacing: MRSpace.s2) {
                MRInlineNote(
                    message: availability.productReason,
                    title: model.installed.isScanning
                        ? "Scanning storage" : "Model unavailable",
                    tone: model.installed.isScanning ? .moving : .attention)
                UnavailableModelAction()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }

        /// A transcript that will not decode is still reported by name. The
        /// "Private storage" row is gone: About already states that Minirun
        /// keeps everything on the device, and a row whose only control was a
        /// green tick was information, not a setting.
        @ViewBuilder private var conversationIssuesSection: some View {
            if !model.conversationLoadFailures.isEmpty {
                Section("Conversations") {
                    ForEach(model.conversationLoadFailures, id: \.filename) { failure in
                        VStack(alignment: .leading, spacing: 2) {
                            Label(
                                "\(failure.filename) could not be read",
                                systemImage: "doc.badge.xmark")
                                .foregroundStyle(MRColor.refuse)
                            Text(failure.reason)
                                .font(.caption)
                                .foregroundStyle(MRColor.secondary)
                                .textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .mrSettingsRow()
                    }
                }
            }
        }

        private var appearanceBinding: Binding<AppAppearance> {
            Binding(get: { model.appearance }, set: { model.appearance = $0 })
        }
    #endif
}

/// The General page, without the scroll view around it.
///
/// The same split `ModelDetailPage` made, and for the same reason:
/// `ImageRenderer` draws nothing at all for a macOS `ScrollView`, so a page
/// that lives inside one cannot be reviewed offscreen — and a redesign nobody
/// can look at is one that gets reviewed by launching the app.
struct GeneralSettingsPage: View {
    @Environment(AppModel.self) private var model
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    #if os(macOS)
        // Optional on purpose. SwiftUI previews and the hosted test bundle
        // render this page without an updater in the environment, and a
        // non-optional `@Environment(SoftwareUpdater.self)` traps when the
        // value is absent. The section simply does not appear in those contexts.
        @Environment(SoftwareUpdater.self) private var updater: SoftwareUpdater?
    #endif

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(GeneralSettingsPresentation.subtitle)
                .font(MRType.pageSubtitle)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.bottom, MRSpace.s2)

            appearance
            softwareUpdate
            newChatDefaults
            conversationStorage
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

    // MARK: Appearance

    private var appearance: some View {
        MRPageSection(title: "Appearance") {
            SettingRow(
                title: "Theme",
                caption: "Use the system setting, or choose this app's appearance."
            ) {
                themePicker
            }
        }
    }

    @ViewBuilder private var themePicker: some View {
        if horizontalSizeClass == .compact || dynamicTypeSize.isAccessibilitySize {
            Picker("Theme", selection: appearanceBinding) {
                ForEach(AppAppearance.allCases) { appearance in
                    Text(appearance.rawValue).tag(appearance)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(minHeight: MRPageControl.height)
        } else {
            Picker("Theme", selection: appearanceBinding) {
                ForEach(AppAppearance.allCases) { appearance in
                    Text(appearance.rawValue).tag(appearance)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(maxWidth: 230)
        }
    }

    private var appearanceBinding: Binding<AppAppearance> {
        Binding(get: { model.appearance }, set: { model.appearance = $0 })
    }

    // MARK: Software update

    /// macOS only, and only when this launch owns an updater.
    ///
    /// The Mac app is distributed outside the App Store; the phone is not, and
    /// iOS has no equivalent control to offer. So this section is a real
    /// platform difference rather than a layout one, and the phone form
    /// deliberately has no counterpart to it.
    @ViewBuilder private var softwareUpdate: some View {
        #if os(macOS)
            if let updater {
                MRPageSection(title: SoftwareUpdatePresentation.panelTitle) {
                    SettingRow(
                        title: SoftwareUpdatePresentation.automaticTitle,
                        caption: updater.explanation
                    ) {
                        Toggle(
                            SoftwareUpdatePresentation.automaticTitle,
                            isOn: Binding(
                                get: { updater.automaticallyChecksForUpdates },
                                set: { updater.setAutomaticallyChecksForUpdates($0) })
                        )
                        .labelsHidden()
                        .disabled(updater.state != .running)
                    }

                    SettingRow(title: "Check for updates") {
                        HStack(spacing: MRSpace.s3) {
                            Button(SoftwareUpdatePresentation.checkNowTitle) {
                                updater.checkForUpdates()
                            }
                            .mrOutlineAction()
                            .disabled(!updater.canCheckForUpdates)
                            SettingFigure(text: updateStatusFigure(updater))
                        }
                    }
                }
            }
        #endif
    }

    #if os(macOS)
        /// The quiet figure beside the button: when this build last asked, or
        /// the one sentence that says it cannot ask at all.
        private func updateStatusFigure(_ updater: SoftwareUpdater) -> String {
            updater.state == .running
                ? SoftwareUpdatePresentation.lastCheck(updater.lastUpdateCheckDate)
                : "Unavailable in this build"
        }
    #endif

    // MARK: New chats

    private var newChatDefaults: some View {
        let choices = model.modelPickerChoices(current: model.defaults.model)
        let currentIsAvailable = choices.contains { $0.modelID == model.defaults.model }
        let availability = model.modelAvailability(for: model.defaults.model)
        return MRPageSection(title: "New chats") {
            Text("These are copied onto a chat when it is created.")
                .font(MRType.prose)
                .foregroundStyle(MRColor.tertiary)
                .fixedSize(horizontal: false, vertical: true)

            SettingRow(title: "Model", caption: "Default for new conversations.") {
                modelControl(choices: choices, currentIsAvailable: currentIsAvailable)
            }

            budgetRow(availability: availability)

            SettingRow(
                title: "Instruments panel",
                caption: "Open live performance metrics when generation begins."
            ) {
                Toggle(
                    "Open automatically",
                    isOn: Binding(
                        get: { model.defaults.telemetryDensity == .cockpit },
                        set: { model.defaults.telemetryDensity = $0 ? .cockpit : .strip })
                )
                .labelsHidden()
                .help("Open Instruments automatically for each new chat while it generates")
            }

            if !availability.isAvailable {
                unavailableNote(availability)
            }
        }
    }

    @ViewBuilder private func modelControl(
        choices: [ModelPickerChoice], currentIsAvailable: Bool
    ) -> some View {
        if currentIsAvailable {
            Picker(
                "Model",
                selection: Binding(
                    get: { model.defaults.model },
                    set: { newValue in
                        guard model.modelAvailability(for: newValue).isAvailable else {
                            return
                        }
                        model.setDefaultModel(newValue)
                    })
            ) {
                ForEach(choices) { choice in
                    if let id = choice.modelID {
                        Text(choice.displayName).tag(id)
                    }
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(maxWidth: 240)
        } else {
            Menu(choices.isEmpty ? "No verified models" : "Choose model") {
                ForEach(choices) { choice in
                    if let id = choice.modelID {
                        Button(choice.displayName) { model.setDefaultModel(id) }
                    }
                }
            }
            .disabled(choices.isEmpty)
            .frame(maxWidth: 240)
        }
    }

    @ViewBuilder private func budgetRow(availability: ModelRunAvailability) -> some View {
        if availability.isAvailable,
            let plan = model.defaultBudgetPlan(for: model.defaults.model),
            let entry = model.entry(model.defaults.model)
        {
            SettingRow(
                title: "Memory budget",
                caption: "Default for each new conversation."
            ) {
                NavigationLink {
                    DefaultBudgetPage(plan: plan, entry: entry)
                } label: {
                    HStack(spacing: MRSpace.s2) {
                        Text(MRFormat.bytesDecimal(plan.budgetBytes))
                            .font(MRType.figure)
                            .foregroundStyle(
                                plan.isRunnable ? MRColor.accent : MRColor.refuse)
                        Image(systemName: "chevron.right")
                            .imageScale(.small)
                    }
                }
                .mrTextLink()
                .accessibilityLabel("memory budget for new chats")
            }
        } else {
            SettingRow(
                title: "Memory budget",
                caption: "Available when the default model is ready."
            ) {
                SettingFigure(text: "Set when a model is ready", color: MRColor.tertiary)
            }
        }
    }

    /// A note, and only when there is something to do about it: the model this
    /// app would start a chat with cannot start one yet, and the control under
    /// the sentence is the one thing that changes that.
    private func unavailableNote(_ availability: ModelRunAvailability) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            MRInlineNote(
                message: availability.productReason,
                title: model.installed.isScanning ? "Scanning storage" : "Model unavailable",
                tone: model.installed.isScanning ? .moving : .attention)
            UnavailableModelAction()
        }
    }

    // MARK: Conversations

    private var conversationStorage: some View {
        MRPageSection(title: "Conversations") {
            SettingRow(
                title: "Private storage",
                caption: ConversationStoragePresentation.folderDescription
            ) {
                #if os(macOS)
                    Button("Reveal in Finder") {
                        NSWorkspace.shared.selectFile(
                            nil,
                            inFileViewerRootedAtPath: model.conversationStore.directory.path)
                    }
                    .mrTextLink()
                #else
                    MRStatusLine(tone: .ready, sentence: "On this device only")
                #endif
            }

            ForEach(model.conversationLoadFailures, id: \.filename) { failure in
                MRInlineNote(
                    message: failure.reason,
                    title: "\(failure.filename) could not be read",
                    tone: .attention,
                    systemImage: "doc.badge.xmark")
            }
        }
    }
}

/// The memory dial as a pushed page, for both the Mac page and the phone form.
struct DefaultBudgetPage: View {
    let plan: BudgetPlan
    let entry: CatalogEntry

    @Environment(AppModel.self) private var model

    var body: some View {
        ScrollView {
            MemoryDialView(
                plan: plan,
                entry: entry,
                role: .defaultForNewChats,
                // Asked of the plan and not of the model id: a bounded streaming
                // run has no deterministic read-ahead to state, and which models
                // those are is the plan's answer rather than this screen's.
                readAheadDepth: plan.strategy == .boundedLayerStreaming
                    ? nil
                    : Binding(
                        get: { model.defaults.deterministicReadAheadLayers },
                        set: { model.defaults.deterministicReadAheadLayers = $0 }),
                onChange: { model.setDefaultBudget($0, for: entry.id) }
            )
            .padding(MRSpace.s4)
            .frame(maxWidth: 720, alignment: .leading)
        }
        .frame(maxWidth: .infinity)
        .mrWorkspaceSurface()
        .mrPhoneNavigationTitle("Default budget")
    }
}

enum ConversationStoragePresentation {
    static let folderDescription = "Saved privately in Minirun's app data"
}

enum GeneralSettingsPresentation {
    static let subtitle = "Choose how new chats start and where conversations are kept."
}

enum PreferenceRowLayoutPolicy {
    static func usesVerticalLayout(
        horizontalSizeClass: UserInterfaceSizeClass?,
        dynamicTypeSize: DynamicTypeSize
    ) -> Bool {
        horizontalSizeClass == .compact || dynamicTypeSize.isAccessibilitySize
    }
}
