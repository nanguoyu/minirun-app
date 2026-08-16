import MinirunKit
import SwiftUI

/// The local model inventory. Published models live behind the explicit
/// Find Models action instead of sharing one list with bytes on this device.
struct ModelCatalogView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0) {
                if !model.installed.hasLocation {
                    noStorageCallout
                        .padding(MRSpace.s3)
                }
                if let error = model.artifactRegistrationError {
                    NamedErrorCard(
                        headline: "A verified download is not registered in Storage.",
                        message: "Its files were left in place.",
                        namedError: error,
                        tone: .caution)
                    .padding(MRSpace.s3)
                }

                if model.installed.isScanning && model.localEntriesForPresentation.isEmpty {
                    HStack(spacing: MRSpace.s2) {
                        ProgressView().controlSize(.small)
                        Text("Scanning added folders…")
                            .font(MRType.caption)
                            .foregroundStyle(MRColor.secondary)
                    }
                    .padding(MRSpace.s4)
                } else if model.installed.hasLocation
                    && model.localEntriesForPresentation.isEmpty
                {
                    emptyLocalInventory
                        .padding(MRSpace.s4)
                } else {
                    ForEach(model.localEntriesForPresentation) { entry in
                        localRow(entry)
                        Divider()
                    }
                }
            }
        }
        .safeAreaInset(edge: .top, spacing: 0) {
            VStack(spacing: 0) {
                catalogBar
                Text("Models found in folders you added to Storage.")
                    .font(MRType.smallProse)
                    .foregroundStyle(MRColor.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, MRSpace.s3)
                    .padding(.bottom, MRSpace.s2)
                    .background(Self.barBackground)
                Divider()
            }
        }
        .mrWorkspaceSurface()
        .mrPhoneNavigationTitle("Models")
        #if os(iOS)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    FindModelsButton()
                }
            }
        #endif
        .task { model.installed.refresh() }
    }

    private func localRow(_ entry: CatalogEntry) -> some View {
        NavigationLink {
            ModelDetailView(modelID: entry.id)
        } label: {
            CatalogModelRowContent(
                entry: entry,
                trailingText: ModelRowPresentation.trailingArgument(
                    fitness: model.effectiveProductFitness(for: entry),
                    lastRunSecondsPerToken: model.lastRunSecondsPerToken(for: entry.id)))
            .padding(.horizontal, MRSpace.s4)
        }
        .buttonStyle(.plain)
    }

    private var catalogBar: some View {
        @Bindable var model = model
        return ViewThatFits(in: .horizontal) {
            HStack(spacing: MRSpace.s3) {
                scanSummary
                Spacer()
                sortPicker(selection: $model.catalogSort)
            }
            VStack(alignment: .leading, spacing: MRSpace.s2) {
                scanSummary
                sortPicker(selection: $model.catalogSort)
            }
        }
        .padding(MRSpace.s3)
        .background(Self.barBackground)
    }

    /// The status bar is part of the page, not a separate slab. `panel` painted
    /// a band in a different tone from the list under it on iPhone, so the
    /// first thing the screen showed was a seam.
    private static var barBackground: Color {
        #if os(iOS)
            MRColor.abyss
        #else
            MRColor.panel
        #endif
    }

    private var scanSummary: some View {
        HStack(spacing: MRSpace.s2) {
            if model.installed.isScanning { ProgressView().controlSize(.small) }
            Text(installedSummary)
                .font(MRType.smallProse)
                .foregroundStyle(MRColor.tertiary)
        }
        .help("Counted by scanning the folders listed in Storage.")
    }

    private func sortPicker(selection: Binding<AppModel.CatalogSort>) -> some View {
        HStack(spacing: MRSpace.s1) {
            Text("Sort")
                .font(MRType.smallProse)
                .foregroundStyle(MRColor.secondary)
            Picker("Sort", selection: selection) {
                ForEach(AppModel.CatalogSort.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            #if os(macOS)
                .frame(maxWidth: 180)
            #endif
        }
    }

    /// Nothing has gone wrong here — nobody has told the app where to look yet.
    /// The amber caution card said the opposite, and it was the first thing a
    /// new install saw on both of the screens that ask for a folder.
    @ViewBuilder private var noStorageCallout: some View {
        #if os(iOS)
            neutralNoStorageState
        #else
            cautionNoStorageCallout
        #endif
    }

    #if os(iOS)
        private var neutralNoStorageState: some View {
            EmptyStateView(
                systemImage: "folder.badge.plus",
                headline: "No model folders added",
                message: "Add a folder in Storage so Minirun can find local model containers.",
                actionTitle: "Open Storage",
                action: { model.openSettings(.storage) })
                .frame(minHeight: 360)
        }
    #endif

    private var cautionNoStorageCallout: some View {
        HStack(alignment: .top, spacing: MRSpace.s3) {
            Image(systemName: "externaldrive")
                .font(.system(size: 20))
                .foregroundStyle(MRColor.caution)
            VStack(alignment: .leading, spacing: MRSpace.s2) {
                Text("No model folders added")
                    .font(MRType.headline)
                    .foregroundStyle(MRColor.primary)
                Text("Add a folder in Storage so Minirun can find local model containers.")
                    .font(MRType.body)
                    .foregroundStyle(MRColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                NavigationLink("Open Storage") { StorageSettingsView() }
                    .buttonStyle(.bordered)
                    .frame(minHeight: 44)
            }
        }
        .padding(MRSpace.s4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .mrCard(MRColor.caution.opacity(0.08), stroke: MRColor.caution.opacity(0.35))
    }

    private var emptyLocalInventory: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            Image(systemName: "cube.transparent")
                .font(.system(size: 24))
                .foregroundStyle(MRColor.tertiary)
            Text("No local models found")
                .font(MRType.headline)
                .foregroundStyle(MRColor.primary)
            Text(
                "The added folders do not currently contain a recognized Minirun model. "
                    + "Use Find Models to browse published containers."
            )
            .font(MRType.body)
            .foregroundStyle(MRColor.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: 520, alignment: .leading)
    }

    private var installedSummary: String {
        if model.installed.locationKeys.isEmpty { return "No storage location added" }
        return ModelCatalogPresentation.scanSummary(
            installedCount: model.localEntriesForPresentation.count,
            mountedLocationCount: model.installed.report.locations.filter(\.isMounted).count,
            unavailableLocationCount: model.installed.report.locations.filter { !$0.isMounted }.count)
    }
}

/// A textual product action, deliberately distinct from Refresh. It opens the
/// remote discovery surface; it does not turn the local inventory into a
/// storefront or move the window when the sheet appears.
struct FindModelsButton: View {
    @State private var isPresented = false

    var body: some View {
        Button("Find Models") { isPresented = true }
            .help("Browse Minirun models published on Hugging Face")
            .accessibilityLabel("find models")
            .sheet(isPresented: $isPresented) {
                RemoteModelBrowserView()
            }
    }
}

/// Hugging Face discovery is a modal browser: a complete published list with
/// local search, explicit refresh and a route to the same model
/// detail/download flow. Opening it is cache-first and refreshes the small
/// indexed repository list once per launch; it never walks model trees merely
/// to reveal the browser.
private struct RemoteModelBrowserView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""

    var body: some View {
        NavigationStack {
            #if os(macOS)
                VStack(spacing: 0) {
                    browserHeader
                    Divider()
                    browserContent
                }
            #else
                browserContent
                    .navigationTitle("Find Models")
                    .navigationBarTitleDisplayMode(.inline)
                    .searchable(
                        text: $query,
                        placement: .navigationBarDrawer(displayMode: .always),
                        prompt: "Search models")
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) {
                            Button("Done") { dismiss() }
                        }
                        ToolbarItem(placement: .primaryAction) {
                            RemoteCatalogRefreshButton()
                        }
                    }
            #endif
        }
        #if os(macOS)
            .frame(minWidth: 720, idealWidth: 820, minHeight: 560, idealHeight: 680)
        #else
            .presentationDetents([.large])
            .presentationDragIndicator(.visible)
        #endif
        .task { await model.refreshPublishedModelsIfNeeded() }
    }

    #if os(macOS)
        private var browserHeader: some View {
            HStack(spacing: MRSpace.s3) {
                Text("Find Models")
                    .font(MRType.title)
                    .foregroundStyle(MRColor.primary)
                Spacer()
                RemoteCatalogRefreshButton()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }
            .padding(.horizontal, MRSpace.s4)
            .frame(height: RemoteModelBrowserPresentation.headerHeight)
            .background(MRColor.panel)
        }
    #endif

    private var browserContent: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0) {
                statusBar

                if let error = model.catalogError {
                    NamedErrorCard(
                        headline: "Hugging Face could not be reached.",
                        message: model.snapshot.origin == .cached
                            ? "Showing the last saved model list."
                            : "No saved model list is available yet.",
                        namedError: error.description,
                        tone: .caution)
                    .padding(MRSpace.s3)
                }
                if let warning = model.catalogCacheWarning {
                    NamedErrorCard(
                        headline: "The latest models are shown.",
                        message: "This list could not be saved for offline use.",
                        namedError: warning,
                        tone: .caution)
                    .padding(MRSpace.s3)
                }
                if let error = model.downloadSetupError {
                    NamedErrorCard(
                        headline: "Downloads are unavailable.",
                        message: "You can still inspect published and local model details.",
                        namedError: error)
                    .padding(MRSpace.s3)
                }

                if model.isAwaitingInitialCatalog
                    || model.isRefreshingPublishedModels && publishedModels.isEmpty
                {
                    HStack(spacing: MRSpace.s2) {
                        ProgressView().controlSize(.small)
                        Text("Loading published models…")
                            .font(MRType.caption)
                            .foregroundStyle(MRColor.secondary)
                    }
                    .padding(MRSpace.s4)
                } else if filteredModels.isEmpty {
                    remoteEmptyState
                        .padding(MRSpace.s4)
                } else {
                    ForEach(filteredModels) { published in
                        remoteRow(published)
                        Divider()
                    }
                }
            }
        }
        .mrWorkspaceSurface()
    }

    private var statusBar: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: MRSpace.s3) {
                publishedStatus
                Spacer()
                #if os(macOS)
                    TextField("Search models", text: $query)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 260)
                #endif
            }
            publishedStatus
        }
        .padding(MRSpace.s3)
        .background(MRColor.panel)
        .overlay(alignment: .bottom) { Divider() }
    }

    private var publishedStatus: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(RemoteModelBrowserPresentation.summary(count: publishedModels.count))
                .font(MRType.caption)
                .foregroundStyle(MRColor.primary)
            Text(
                ModelCatalogPresentation.sourceStatus(
                    origin: model.publishedModelsOrigin,
                    isRefreshing: model.isRefreshingPublishedModels,
                    hasRefreshError: model.catalogError != nil,
                    isReviewPreview: model.visualReviewNotice != nil)
            )
            .font(MRType.smallProse)
            .foregroundStyle(MRColor.tertiary)
        }
    }

    private func remoteRow(_ published: PublishedModelReference) -> some View {
        let hasLocalCopy = !model.installed.installations(of: published.id).isEmpty
        let resolved = model.resolvedEntry(for: published)
        let localEntry = hasLocalCopy ? model.entry(published.id) : nil
        return NavigationLink {
            if localEntry != nil {
                // The browser may list a newer commit than the verified copy
                // on disk. Opening that row must keep the local descriptor and
                // verification authority intact; updates are a separate,
                // explicit download operation.
                ModelDetailView(modelID: published.id)
            } else {
                RemoteModelDetailLoader(published: published)
            }
        } label: {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: MRSpace.s3) {
                    remoteIdentity(published)
                    Spacer(minLength: MRSpace.s3)
                    remoteFacts(hasLocalCopy: hasLocalCopy, details: resolved ?? localEntry)
                    disclosureIndicator
                }
                VStack(alignment: .leading, spacing: MRSpace.s2) {
                    HStack(alignment: .top, spacing: MRSpace.s2) {
                        remoteIdentity(published)
                        Spacer(minLength: MRSpace.s2)
                        disclosureIndicator
                    }
                    remoteFacts(hasLocalCopy: hasLocalCopy, details: resolved ?? localEntry)
                        .frame(maxWidth: .infinity, alignment: .trailing)
                }
            }
            .padding(.horizontal, MRSpace.s4)
            .padding(.vertical, MRSpace.s3)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityHint("Opens model details")
    }

    private func remoteIdentity(_ published: PublishedModelReference) -> some View {
        HStack(alignment: .top, spacing: MRSpace.s3) {
            ModelPublisherMark(
                identity: .resolve(
                    modelID: published.id,
                    displayName: published.displayName,
                    repositoryID: published.repository.repoID),
                fallbackSystemName: "shippingbox",
                fallbackColor: MRColor.streamDet)
            VStack(alignment: .leading, spacing: 3) {
                Text(published.displayName)
                    .font(MRType.headline)
                    .foregroundStyle(MRColor.primary)
                Text(published.repository.repoID)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .lineLimit(2)
            }
        }
    }

    private func remoteFacts(hasLocalCopy: Bool, details: CatalogEntry?) -> some View {
        VStack(alignment: .trailing, spacing: 3) {
            if hasLocalCopy {
                Text("On this device")
                    .font(MRType.caption)
                    .foregroundStyle(MRColor.ok)
            }
            if let details {
                Text(MRFormat.publishedBytes(details.descriptor.totalBytes))
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
            } else {
                Text("Details load on open")
                    .font(MRType.smallProse)
                    .foregroundStyle(MRColor.tertiary)
            }
        }
    }

    private var disclosureIndicator: some View {
        Image(systemName: "chevron.right")
            .font(.caption.weight(.semibold))
            .foregroundStyle(MRColor.tertiary)
            .accessibilityHidden(true)
    }

    private var remoteEmptyState: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            Text(query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "No published models found"
                : "No matching models")
                .font(MRType.headline)
            Text(query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "Refresh when a connection is available."
                : "Try a model name or repository name.")
                .font(MRType.body)
                .foregroundStyle(MRColor.secondary)
        }
    }

    private var publishedModels: [PublishedModelReference] {
        model.publishedModelsForPresentation
    }

    private var filteredModels: [PublishedModelReference] {
        publishedModels.filter { RemoteModelBrowserPresentation.matches($0, query: query) }
    }
}

/// Tree resolution starts only after a person opens one published row. Until
/// it completes, no size, layout or download promise is rendered.
private struct RemoteModelDetailLoader: View {
    @Environment(AppModel.self) private var model
    let published: PublishedModelReference
    @State private var failure: String?
    @State private var retryGeneration = 0

    var body: some View {
        Group {
            if model.resolvedEntry(for: published) != nil {
                ModelDetailView(modelID: published.id)
            } else if let failure {
                VStack(alignment: .leading, spacing: MRSpace.s3) {
                    NamedErrorCard(
                        headline: "Model details could not be loaded.",
                        message: "No download or verification state was changed.",
                        namedError: failure,
                        tone: .caution)
                    Button("Try again") {
                        self.failure = nil
                        retryGeneration += 1
                    }
                }
                .padding(MRSpace.s4)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                .mrWorkspaceSurface()
            } else {
                VStack(spacing: MRSpace.s3) {
                    ProgressView()
                    Text("Loading \(published.displayName) details…")
                        .font(MRType.body)
                        .foregroundStyle(MRColor.secondary)
                    Text("Checking this repository's complete file list.")
                        .font(MRType.smallProse)
                        .foregroundStyle(MRColor.tertiary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .mrWorkspaceSurface()
            }
        }
        .task(id: retryGeneration) {
            guard model.resolvedEntry(for: published) == nil else { return }
            do {
                _ = try await model.resolvePublishedModel(published)
            } catch is CancellationError {
                return
            } catch {
                failure = String(describing: error)
            }
        }
    }
}

private struct CatalogModelRowContent: View {
    @Environment(AppModel.self) private var model
    let entry: CatalogEntry
    let trailingText: String?

    var body: some View {
        ModelRow(
            entry: entry,
            state: model.state(of: entry.id),
            fitness: model.effectiveProductFitness(for: entry),
            trailingText: trailingText,
            installations: model.installed.installations(of: entry.id),
            locationNames: locationNames,
            hasStorageLocation: model.installed.hasLocation)
    }

    private var locationNames: [String: String] {
        Dictionary(
            model.installed.report.locations.map { ($0.rootPath, $0.displayName) },
            uniquingKeysWith: { first, _ in first })
    }
}

private struct RemoteCatalogRefreshButton: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Button {
            Task { await model.refreshPublishedModels() }
        } label: {
            if model.isRefreshingPublishedModels {
                HStack(spacing: MRSpace.s1) {
                    ProgressView().controlSize(.small)
                    Text("Updating…")
                }
            } else {
                Text("Refresh")
            }
        }
        .help("Refresh the published model list")
        .disabled(model.isRefreshingPublishedModels || !model.allowsLiveCatalogRefresh)
    }
}

enum RemoteModelBrowserPresentation {
    static let headerHeight: CGFloat = 52

    static func summary(count: Int) -> String {
        count == 1 ? "1 published model" : "\(count) published models"
    }

    static func matches(_ published: PublishedModelReference, query: String) -> Bool {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !needle.isEmpty else { return true }
        let haystacks = [
            published.displayName, published.id.rawValue, published.repository.repoID,
        ]
        return haystacks.contains {
            $0.localizedCaseInsensitiveContains(needle)
        }
    }
}

enum ModelCatalogPresentation {
    static func sourceStatus(
        origin: ModelCatalogSnapshot.Origin,
        isRefreshing: Bool,
        hasRefreshError: Bool,
        isReviewPreview: Bool
    ) -> String {
        if isReviewPreview { return "Sample model list" }
        if isRefreshing { return "Current list · updating" }
        switch origin {
        case .live: return "Up to date"
        case .cached: return hasRefreshError ? "Saved list · offline" : "Saved model list"
        case .bundled: return hasRefreshError ? "Update unavailable" : "Loading…"
        }
    }

    static func scanSummary(
        installedCount: Int, mountedLocationCount: Int, unavailableLocationCount: Int
    ) -> String {
        let models = installedCount == 1 ? "1 local model" : "\(installedCount) local models"
        let mounted = mountedLocationCount == 1
            ? "1 location mounted" : "\(mountedLocationCount) locations mounted"
        guard unavailableLocationCount > 0 else { return "\(models) · \(mounted)" }
        let unavailable = unavailableLocationCount == 1
            ? "1 unavailable" : "\(unavailableLocationCount) unavailable"
        return "\(models) · \(mounted) · \(unavailable)"
    }
}
