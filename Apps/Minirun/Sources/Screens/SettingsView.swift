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
struct SettingsView: View {
    var body: some View {
        #if os(macOS)
            MacSettingsView()
        #else
            PhoneSettingsView()
        #endif
    }
}

// MARK: - macOS: left nav plus a page

#if os(macOS)
    /// Settings owns a fixed navigation rail and a push-capable detail stack.
    /// The rail still uses a real sidebar `List`, so keyboard navigation,
    /// selection tint and material remain native. It deliberately is not a
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
                            Label(section.rawValue, systemImage: section.systemImage)
                                .tag(section)
                        }
                    }
                    .listStyle(.sidebar)
                    .scrollContentBackground(.hidden)
                }
                .safeAreaInset(edge: .bottom, spacing: 0) {
                    VStack(spacing: 0) {
                        Divider()
                        Button {
                            model.backToApp()
                        } label: {
                            Label("Back to app", systemImage: "chevron.left")
                                .padding(.horizontal, MRSpace.s3)
                                .padding(.vertical, MRSpace.s2)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("back to app")
                    }
                }
                .frame(width: ConversationWorkspaceLayout.sidebarWidth)
                .mrSidebarSurface()

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
                        Label(section.rawValue, systemImage: section.systemImage)
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
                Divider()
                content
            }
            .mrWorkspaceSurface()
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

/// Defaults a NEW chat starts from, plus where conversations are kept.
///
/// Appearance, defaults for new chats, and private conversation storage.
struct GeneralSettingsView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    #if os(macOS)
        // Optional on purpose. SwiftUI previews and the hosted test bundle
        // render this page without an updater in the environment, and a
        // non-optional `@Environment(SoftwareUpdater.self)` traps when the
        // value is absent. The panel simply does not appear in those contexts.
        @Environment(SoftwareUpdater.self) private var updater: SoftwareUpdater?
    #endif

    @ViewBuilder
    var body: some View {
        #if os(iOS)
            phoneForm
        #else
            desktopPage
        #endif
    }

    #if os(macOS)
        private var desktopPage: some View {
            ScrollView {
                VStack(alignment: .leading, spacing: MRSpace.s5) {
                    Text(GeneralSettingsPresentation.subtitle)
                        .font(MRType.body)
                        .foregroundStyle(MRColor.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                    appearance
                    softwareUpdate
                    newChatDefaults
                    conversationStorage
                }
                .padding(pagePadding)
                .frame(maxWidth: 760, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .top)
            }
            .mrWorkspaceSurface()
        }

        /// macOS only, and only when this launch owns an updater.
        ///
        /// The Mac app is distributed outside the App Store; the phone is not,
        /// and iOS has no equivalent control to offer. So this panel is a real
        /// platform difference rather than a layout one, and the phone form
        /// below deliberately has no counterpart to it.
        @ViewBuilder private var softwareUpdate: some View {
            if let updater {
                Panel(title: SoftwareUpdatePresentation.panelTitle) {
                    PreferenceRow(
                        systemImage: "arrow.triangle.2.circlepath",
                        title: SoftwareUpdatePresentation.automaticTitle,
                        subtitle: updater.explanation
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

                    Divider().overlay(MRColor.hairline)

                    PreferenceRow(
                        systemImage: "clock.arrow.circlepath",
                        title: "Check for updates",
                        subtitle: updater.state == .running
                            ? SoftwareUpdatePresentation.lastCheck(updater.lastUpdateCheckDate)
                            : "Unavailable in this build"
                    ) {
                        Button(SoftwareUpdatePresentation.checkNowTitle) {
                            updater.checkForUpdates()
                        }
                        .controlSize(.small)
                        .disabled(!updater.canCheckForUpdates)
                    }
                }
            }
        }
    #endif

    // MARK: - iOS: a native inset-grouped form
    //
    // The desktop page above stacks product cards whose control sits under its
    // own label. On the phone that reads as a Mac layout ported across: a
    // "System ⌄" menu on its own line beneath the word Theme, a switch under
    // the words that describe it, and a lone "—" where a value belongs. The
    // platform's own answer is one row per setting, label leading, control
    // trailing, with the explanation as a section footer.

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
                Text("New Chats")
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
                    defaultBudget(plan: plan, entry: entry)
                } label: {
                    LabeledContent("Memory budget") {
                        Text(MRFormat.bytesDecimal(plan.budgetBytes))
                            .font(MRType.metric)
                            .foregroundStyle(plan.isRunnable ? MRColor.primary : MRColor.refuse)
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
                Label {
                    Text(
                        (model.installed.isScanning
                            ? "Scanning storage. " : "Model unavailable. ")
                            + availability.productReason)
                } icon: {
                    if model.installed.isScanning {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: "exclamationmark.circle.fill")
                    }
                }
                .foregroundStyle(MRColor.caution)
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
    #endif

    private var appearance: some View {
        Panel(title: "Appearance") {
            PreferenceRow(
                systemImage: "circle.lefthalf.filled",
                title: "Theme",
                subtitle: "Use the system setting or choose this app's appearance"
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
            .frame(minHeight: 44)
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
        Binding(
            get: { model.appearance },
            set: { model.appearance = $0 })
    }

    private var pagePadding: CGFloat {
        #if os(macOS)
            MRSpace.s6
        #else
            MRSpace.s4
        #endif
    }

    private var newChatDefaults: some View {
        let choices = model.modelPickerChoices(current: model.defaults.model)
        let currentIsAvailable = choices.contains { $0.modelID == model.defaults.model }
        let availability = model.modelAvailability(for: model.defaults.model)
        return Panel(
            title: "New chats",
            trailing: AnyView(
                Text("Applied on creation")
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary))
        ) {
            PreferenceRow(
                systemImage: "cube",
                title: "Model",
                subtitle: "Default for new conversations"
            ) {
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

            Divider().overlay(MRColor.hairline)

            if availability.isAvailable,
                let plan = model.defaultBudgetPlan(for: model.defaults.model),
                let entry = model.entry(model.defaults.model)
            {
                NavigationLink {
                    defaultBudget(plan: plan, entry: entry)
                } label: {
                    PreferenceRow(
                        systemImage: "memorychip",
                        title: "Memory budget",
                        subtitle: "Default for each new conversation"
                    ) {
                        HStack(spacing: MRSpace.s2) {
                            Text(MRFormat.bytesDecimal(plan.budgetBytes))
                                .font(MRType.metric)
                                .foregroundStyle(
                                    plan.isRunnable ? MRColor.primary : MRColor.refuse)
                            Image(systemName: "chevron.right")
                                .imageScale(.small)
                                .foregroundStyle(MRColor.tertiary)
                        }
                    }
                }
                .buttonStyle(.plain)
            } else {
                PreferenceRow(
                    systemImage: "memorychip",
                    title: "Memory budget",
                    subtitle: "Available when the default model is ready"
                ) {
                    Text("—")
                        .font(MRType.metric)
                        .foregroundStyle(MRColor.tertiary)
                }
            }

            Divider().overlay(MRColor.hairline)

            PreferenceRow(
                systemImage: "gauge.with.dots.needle.bottom.50percent",
                title: "Instruments panel",
                subtitle: "Open live performance metrics when generation begins"
            ) {
                Toggle(
                    "Open automatically",
                    isOn: Binding(
                        get: { model.defaults.telemetryDensity == .cockpit },
                        set: {
                            model.defaults.telemetryDensity = $0 ? .cockpit : .strip
                        })
                )
                .labelsHidden()
                .help("Open Instruments automatically for each new chat while it generates")
            }

            if !availability.isAvailable {
                unavailableCallout(availability)
            }
        }
    }

    private func defaultBudget(plan: BudgetPlan, entry: CatalogEntry) -> some View {
        ScrollView {
            MemoryDialView(
                plan: plan,
                entry: entry,
                role: .defaultForNewChats,
                readAheadDepth: entry.id == .deepseekV4Flash
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

    private func unavailableCallout(_ availability: ModelRunAvailability) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            HStack(alignment: .top, spacing: MRSpace.s2) {
                if model.installed.isScanning {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: "exclamationmark.circle.fill")
                        .foregroundStyle(MRColor.caution)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.installed.isScanning ? "Scanning storage" : "Model unavailable")
                        .font(MRType.caption)
                        .foregroundStyle(MRColor.caution)
                    Text(availability.productReason)
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            UnavailableModelAction()
        }
        .padding(MRSpace.s3)
        .frame(maxWidth: .infinity, alignment: .leading)
        .mrCard(MRColor.caution.opacity(0.08), stroke: MRColor.caution.opacity(0.35))
    }

    private var conversationStorage: some View {
        Panel(title: "Conversations") {
            PreferenceRow(
                systemImage: "lock.shield",
                title: "Private storage",
                subtitle: ConversationStoragePresentation.folderDescription
            ) {
                #if os(macOS)
                    Button("Reveal in Finder") {
                        NSWorkspace.shared.selectFile(
                            nil, inFileViewerRootedAtPath: model.conversationStore.directory.path)
                    }
                    .controlSize(.small)
                #else
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(MRColor.ok)
                #endif
            }

            if !model.conversationLoadFailures.isEmpty {
                Divider().overlay(MRColor.hairline)
                ForEach(model.conversationLoadFailures, id: \.filename) { failure in
                    VStack(alignment: .leading, spacing: 2) {
                        Label("\(failure.filename) could not be read", systemImage: "doc.badge.xmark")
                            .font(MRType.caption)
                            .foregroundStyle(MRColor.refuse)
                        Text(failure.reason)
                            .font(MRType.micro)
                            .foregroundStyle(MRColor.secondary)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
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

private struct PreferenceRow<Trailing: View>: View {
    let systemImage: String
    let title: String
    let subtitle: String
    @ViewBuilder let trailing: Trailing

    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    @ViewBuilder
    var body: some View {
        if PreferenceRowLayoutPolicy.usesVerticalLayout(
            horizontalSizeClass: horizontalSizeClass,
            dynamicTypeSize: dynamicTypeSize)
        {
            VStack(alignment: .leading, spacing: MRSpace.s3) {
                label
                trailing
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .contain)
        } else {
            HStack(alignment: .center, spacing: MRSpace.s3) {
                label
                    .frame(minWidth: 240, alignment: .leading)
                    .layoutPriority(1)
                Spacer(minLength: MRSpace.s3)
                trailing
                    // Controls must retain their intrinsic width on iPad and
                    // macOS. Letting the flexible label win can collapse a
                    // segmented picker to zero width while still making the
                    // horizontal candidate appear to fit.
                    .fixedSize(horizontal: true, vertical: false)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .contain)
        }
    }

    private var label: some View {
        HStack(alignment: .top, spacing: MRSpace.s3) {
            Image(systemName: systemImage)
                .foregroundStyle(MRColor.secondary)
                .frame(width: 24)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(MRType.body)
                    .foregroundStyle(MRColor.primary)
                Text(subtitle)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
