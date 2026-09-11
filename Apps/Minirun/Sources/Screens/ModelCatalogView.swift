import MinirunKit
import SwiftUI

/// The local model inventory. Published models live behind the explicit
/// Find Models action instead of sharing one list with bytes on this device.
///
/// One surface and hairlines, in the product-page language: the bar with the
/// scan summary and the sort, then a row per model — mark, name, what it is,
/// and the sentence that says where it stands.
struct ModelCatalogView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        ScrollView {
            ModelCatalogList()
        }
        .safeAreaInset(edge: .top, spacing: 0) {
            VStack(spacing: 0) {
                catalogBar
                MRHairline()
            }
        }
        .mrProductPage()
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

    private var catalogBar: some View {
        @Bindable var model = model
        return VStack(alignment: .leading, spacing: MRSpace.s2) {
            ViewThatFits(in: .horizontal) {
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
            Text("Models found in folders you added to Storage.")
                .font(MRType.prose)
                .foregroundStyle(MRColor.tertiary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, pagePadding)
        .padding(.vertical, MRSpace.s3)
        .background(MRColor.panel)
    }

    private var scanSummary: some View {
        HStack(spacing: MRSpace.s2) {
            if model.installed.isScanning { ProgressView().controlSize(.small) }
            Text(installedSummary)
                .font(MRType.prose)
                .foregroundStyle(MRColor.secondary)
        }
        .help("Counted by scanning the folders listed in Storage.")
    }

    private func sortPicker(selection: Binding<AppModel.CatalogSort>) -> some View {
        HStack(spacing: MRSpace.s1) {
            Text("Sort")
                .font(MRType.prose)
                .foregroundStyle(MRColor.tertiary)
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

    private var pagePadding: CGFloat {
        #if os(macOS)
            MRSpace.s5
        #else
            MRSpace.s4
        #endif
    }

    private var installedSummary: String {
        if model.installed.locationKeys.isEmpty { return "No storage location added" }
        return ModelCatalogPresentation.scanSummary(
            installedCount: model.localEntriesForPresentation.count,
            mountedLocationCount: model.installed.report.locations.filter(\.isMounted).count,
            unavailableLocationCount: model.installed.report.locations.filter { !$0.isMounted }.count)
    }
}

/// The rows themselves, without the scroll view around them.
///
/// Separate for the reason `ModelDetailPage` is separate: `ImageRenderer` draws
/// nothing at all for a macOS `ScrollView`, and a list nobody can look at
/// offscreen is a list that gets reviewed by launching the app.
struct ModelCatalogList: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        LazyVStack(alignment: .leading, spacing: 0) {
            if !model.installed.hasLocation {
                noStorageCallout
                    .padding(.horizontal, pagePadding)
                    .padding(.top, MRSpace.s4)
            }
            if let error = model.artifactRegistrationError {
                NamedErrorCard(
                    headline: "A verified download is not registered in Storage.",
                    message: "Its files were left in place.",
                    namedError: error,
                    tone: .caution)
                .padding(.horizontal, pagePadding)
                .padding(.top, MRSpace.s4)
            }

            if model.installed.isScanning && model.localEntriesForPresentation.isEmpty {
                HStack(spacing: MRSpace.s2) {
                    ProgressView().controlSize(.small)
                    Text("Scanning added folders…")
                        .font(MRType.prose)
                        .foregroundStyle(MRColor.secondary)
                }
                .padding(pagePadding)
            } else if model.installed.hasLocation
                && model.localEntriesForPresentation.isEmpty
            {
                emptyLocalInventory
                    .padding(pagePadding)
            } else {
                ForEach(model.localEntriesForPresentation) { entry in
                    localRow(entry)
                        .padding(.horizontal, pagePadding)
                        .overlay(alignment: .top) { MRHairline() }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func localRow(_ entry: CatalogEntry) -> some View {
        NavigationLink {
            ModelDetailView(modelID: entry.id)
        } label: {
            HStack(spacing: MRSpace.s3) {
                CatalogModelRowContent(entry: entry)
                MRDisclosureChevron()
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityHint("Opens model details")
    }

    private var pagePadding: CGFloat {
        #if os(macOS)
            MRSpace.s5
        #else
            MRSpace.s4
        #endif
    }

    /// Nothing has gone wrong here — nobody has told the app where to look yet.
    /// The amber caution card said the opposite, and it was the first thing a
    /// new install saw on both of the screens that ask for a folder.
    @ViewBuilder private var noStorageCallout: some View {
        #if os(iOS)
            EmptyStateView(
                systemImage: "folder.badge.plus",
                headline: "No model folders added",
                message: "Add a folder in Storage so Minirun can find local model containers.",
                actionTitle: "Open Storage",
                action: { model.openSettings(.storage) })
                .frame(minHeight: 360)
        #else
            VStack(alignment: .leading, spacing: MRSpace.s3) {
                MRInlineNote(
                    message: "Add a folder in Storage so Minirun can find local model "
                        + "containers.",
                    title: "No model folders added",
                    tone: .idle,
                    systemImage: "folder.badge.plus")
                NavigationLink("Open Storage") { StorageSettingsView() }
                    .buttonStyle(.plain)
                    .foregroundStyle(MRColor.accent)
                    .font(MRType.control)
                    .frame(minHeight: MRPageControl.linkHeight)
            }
        #endif
    }

    private var emptyLocalInventory: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            Image(systemName: "cube.transparent")
                .font(.system(size: 24))
                .foregroundStyle(MRColor.tertiary)
                .accessibilityHidden(true)
            Text("No local models found")
                .font(MRType.headline)
                .foregroundStyle(MRColor.primary)
            Text(
                "The added folders do not currently contain a recognized Minirun model. "
                    + "Use Find Models to browse published containers."
            )
            .font(MRType.prose)
            .foregroundStyle(MRColor.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: 520, alignment: .leading)
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
                    MRHairline()
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
        .mrDismissesWhenAChatOpens()
        .task { await model.refreshPublishedModelsIfNeeded() }
    }

    #if os(macOS)
        private var browserHeader: some View {
            HStack(spacing: MRSpace.s3) {
                Text("Find Models")
                    .font(MRType.pageTitle)
                    .tracking(-0.5)
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
                    .padding(MRSpace.s4)
                }
                if let warning = model.catalogCacheWarning {
                    NamedErrorCard(
                        headline: "The latest models are shown.",
                        message: "This list could not be saved for offline use.",
                        namedError: warning,
                        tone: .caution)
                    .padding(MRSpace.s4)
                }
                if let error = model.downloadSetupError {
                    NamedErrorCard(
                        headline: "Downloads are unavailable.",
                        message: "You can still inspect published and local model details.",
                        namedError: error)
                    .padding(MRSpace.s4)
                }

                if model.isAwaitingInitialCatalog
                    || model.isRefreshingPublishedModels && publishedModels.isEmpty
                {
                    HStack(spacing: MRSpace.s2) {
                        ProgressView().controlSize(.small)
                        Text("Loading published models…")
                            .font(MRType.prose)
                            .foregroundStyle(MRColor.secondary)
                    }
                    .padding(MRSpace.s4)
                } else if filteredModels.isEmpty {
                    remoteEmptyState
                        .padding(MRSpace.s4)
                } else {
                    ForEach(filteredModels) { published in
                        remoteRow(published)
                            .padding(.horizontal, MRSpace.s4)
                            .overlay(alignment: .top) { MRHairline() }
                    }
                }
            }
        }
        .mrProductPage()
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
        .padding(MRSpace.s4)
        .background(MRColor.panel)
    }

    private var publishedStatus: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(RemoteModelBrowserPresentation.summary(count: publishedModels.count))
                .font(MRType.prose)
                .foregroundStyle(MRColor.primary)
            Text(
                ModelCatalogPresentation.sourceStatus(
                    origin: model.publishedModelsOrigin,
                    isRefreshing: model.isRefreshingPublishedModels,
                    hasRefreshError: model.catalogError != nil,
                    isReviewPreview: model.visualReviewNotice != nil)
            )
            .font(MRType.prose)
            .foregroundStyle(MRColor.tertiary)
        }
    }

    private func remoteRow(_ published: PublishedModelReference) -> some View {
        let hasLocalCopy = !model.installed.installations(of: published.id).isEmpty
        let resolved = model.resolvedEntry(for: published)
        let localEntry = hasLocalCopy ? model.entry(published.id) : nil
        let details = resolved ?? localEntry
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
            HStack(spacing: MRSpace.s3) {
                MRListRow {
                    MRRowIdentity(
                        title: published.displayName,
                        subtitle: details.flatMap {
                            ModelPurposePresentation.line(for: $0.descriptor)
                        }
                    ) {
                        ModelPublisherMark(
                            identity: .resolve(
                                modelID: published.id,
                                displayName: published.displayName,
                                repositoryID: published.repository.repoID),
                            fallbackSystemName: "shippingbox",
                            fallbackColor: MRColor.tertiary,
                            size: 32)
                    }
                } trailing: {
                    MRRowStatus(
                        tone: hasLocalCopy ? .ready : .idle,
                        sentence: hasLocalCopy ? "On this device" : "Not on this device")
                    if let details {
                        MRRowQuantity(
                            text: MRFormat.publishedBytes(details.descriptor.totalBytes),
                            provenance: .declared)
                    } else {
                        Text("Details load on open")
                            .font(MRType.prose)
                            .foregroundStyle(MRColor.tertiary)
                    }
                }
                MRDisclosureChevron()
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityHint("Opens model details")
    }

    private var remoteEmptyState: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            Text(query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "No published models found"
                : "No matching models")
                .font(MRType.headline)
                .foregroundStyle(MRColor.primary)
            Text(query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "Refresh when a connection is available."
                : "Try a model name or repository name.")
                .font(MRType.prose)
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
                    .mrOutlineAction()
                }
                .padding(MRSpace.s4)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                .mrProductPage()
            } else {
                VStack(spacing: MRSpace.s3) {
                    ProgressView()
                    Text("Loading \(published.displayName) details…")
                        .font(MRType.pageSubtitle)
                        .foregroundStyle(MRColor.secondary)
                    Text("Checking this repository's complete file list.")
                        .font(MRType.prose)
                        .foregroundStyle(MRColor.tertiary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .mrProductPage()
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

    var body: some View {
        ModelRow(
            entry: entry,
            state: model.state(of: entry.id),
            fitness: model.effectiveProductFitness(for: entry),
            note: ModelRowPresentation.note(
                fitness: model.effectiveProductFitness(for: entry),
                lastRunSecondsPerToken: model.lastRunSecondsPerToken(for: entry.id)),
            installations: model.installed.installations(of: entry.id),
            locations: locations,
            hasStorageLocation: model.installed.hasLocation)
    }

    private var locations: [String: ModelRowLocation] {
        Dictionary(
            model.installed.report.locations.map {
                ($0.rootPath, ModelRowLocation(name: $0.displayName, isMounted: $0.isMounted))
            },
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
