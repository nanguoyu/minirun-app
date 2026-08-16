import MinirunKit
import MinirunRunners
import SwiftUI

/// Readiness · Artifact · Storage · Budget · Recent runs.
///
/// Recent runs are receipts when they exist. The whole section disappears
/// when it has no evidence instead of turning an empty database count into a
/// product feature.
struct ModelDetailView: View {
    let modelID: ModelID

    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var showingDestination = false
    @State private var pendingFullVerification: DiscoveredArtifact?
    @State private var pendingDownloadVerification = false
    @State private var pendingRemoval: DiscoveredArtifact?
    @State private var sourceDetailsExpanded = false

    private var entry: CatalogEntry? { model.entry(modelID) }
    private var download: DownloadController? { model.download(modelID) }

    var body: some View {
        Group {
            #if os(macOS)
                VStack(spacing: 0) {
                    MacColumnHeader(
                        title: entry?.descriptor.displayName ?? modelID.rawValue,
                        backAction: { dismiss() })
                    Divider()
                    detailContent
                }
            #else
                detailContent
            #endif
        }
        .mrWorkspaceSurface()
        .mrPhoneNavigationTitle(entry?.descriptor.displayName ?? modelID.rawValue)
        .sheet(isPresented: $showingDestination) {
            DestinationPicker(modelID: modelID)
        }
        #if os(macOS)
            .navigationBarBackButtonHidden(true)
        #endif
        #if os(iOS)
        .alert(
            "Verify every file?",
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
        .alert(
            "Re-verify downloaded files?",
            isPresented: $pendingDownloadVerification
        ) {
            Button(DownloadVerificationConfirmation.actionTitle) {
                pendingDownloadVerification = false
                download?.verify()
            }
            Button("Cancel", role: .cancel) { pendingDownloadVerification = false }
        } message: {
            Text(
                DownloadVerificationConfirmation.message(
                    readBytes: download?.plan?.totalBytes
                        ?? entry?.descriptor.totalBytes ?? 0))
        }
        .alert(
            "Permanently remove this copy?",
            isPresented: removalConfirmationPresented
        ) {
            if let artifact = pendingRemoval {
                Button(LocalCopyRemovalConfirmation.actionTitle, role: .destructive) {
                    pendingRemoval = nil
                    Task { await model.removeLocalCopy(artifact) }
                }
            }
            Button("Cancel", role: .cancel) { pendingRemoval = nil }
        } message: {
            if let artifact = pendingRemoval {
                Text(
                    LocalCopyRemovalConfirmation.message(
                        modelName: entry?.descriptor.displayName ?? artifact.displayName,
                        artifact: artifact))
            }
        }
        #else
        .confirmationDialog(
            "Verify every file?",
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
        .confirmationDialog(
            "Re-verify downloaded files?",
            isPresented: $pendingDownloadVerification,
            titleVisibility: .visible
        ) {
            Button(DownloadVerificationConfirmation.actionTitle) {
                pendingDownloadVerification = false
                download?.verify()
            }
            Button("Cancel", role: .cancel) { pendingDownloadVerification = false }
        } message: {
            Text(
                DownloadVerificationConfirmation.message(
                    readBytes: download?.plan?.totalBytes
                        ?? entry?.descriptor.totalBytes ?? 0))
        }
        .confirmationDialog(
            "Permanently remove this copy?",
            isPresented: removalConfirmationPresented,
            titleVisibility: .visible
        ) {
            if let artifact = pendingRemoval {
                Button(LocalCopyRemovalConfirmation.actionTitle, role: .destructive) {
                    pendingRemoval = nil
                    Task { await model.removeLocalCopy(artifact) }
                }
            }
            Button("Cancel", role: .cancel) { pendingRemoval = nil }
        } message: {
            if let artifact = pendingRemoval {
                Text(
                    LocalCopyRemovalConfirmation.message(
                        modelName: entry?.descriptor.displayName ?? artifact.displayName,
                        artifact: artifact))
            }
        }
        #endif
    }

    private var detailContent: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MRSpace.s5) {
                if let entry {
                    header(entry)
                    #if os(iOS)
                        if entry.id == .kimiK3,
                            K3ProductMemoryBudget.currentPolicy.isExperimental
                        {
                            k3ExperimentalTier
                        }
                    #endif
                    artifact(entry)
                    if entry.descriptor.layout == .unrecognized {
                        unsupportedArtifact(entry)
                        discoveredCopies
                    } else {
                        storage(entry)
                    }
                    if model.hasVerifiedRuntime(for: entry.id) {
                        budget(entry)
                        let records = model.history(for: entry.id)
                        if !records.isEmpty { history(records) }
                    }
                } else {
                    EmptyStateView(
                        headline: "Unknown model.",
                        message: "This build has no descriptor for \(modelID.rawValue).")
                }
            }
            .padding(pagePadding)
            .frame(maxWidth: 860, alignment: .leading)
        }
        .frame(maxWidth: .infinity)
    }

    private var pagePadding: CGFloat {
        #if os(macOS)
            MRSpace.s5
        #else
            MRSpace.s4
        #endif
    }

    // MARK: Sections

    #if os(iOS)
        private var k3ExperimentalTier: some View {
            Panel(title: "iPhone runtime") {
                HStack(alignment: .top, spacing: MRSpace.s3) {
                    Image(systemName: "testtube.2")
                        .foregroundStyle(MRColor.caution)
                    VStack(alignment: .leading, spacing: MRSpace.s1) {
                        Text("Bounded experimental tier")
                            .font(MRType.headline)
                        Text(
                            "Uses the 5.80 GB device record and limits each response to two "
                                + "tokens while this memory profile is validated. File "
                                + "verification and the memory plan remain required."
                        )
                        .font(MRType.caption)
                        .foregroundStyle(MRColor.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .accessibilityElement(children: .combine)
            }
        }
    #endif

    private func header(_ entry: CatalogEntry) -> some View {
        let fitness = model.effectiveProductFitness(for: entry)
        let availability = model.modelAvailability(for: entry.id)
        let compatibility = ModelCompatibilityPresentation.resolve(
            modelName: entry.descriptor.displayName,
            fitness: fitness, availability: availability)
        let installations = model.installed.installations(of: entry.id)
        let mounted = model.installed.report.locations
            .filter(\.isMounted)
            .flatMap { $0.artifacts(for: entry.id) }
        let integrity = ModelIntegrityPresentation.resolve(
            installations: installations, mounted: mounted)
        return Panel(title: "Readiness") {
            readinessRow(title: "App compatibility", value: compatibility)
            Divider().overlay(MRColor.hairline)
            readinessRow(title: "Local files", value: integrity)
        }
    }

    private func readinessRow(
        title: String, value: ModelReadinessPresentationValue
    ) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: MRSpace.s3) {
                readinessExplanation(title: title, value: value)
                Spacer(minLength: MRSpace.s3)
                StatusChip(text: value.status, tone: value.tone)
            }
            VStack(alignment: .leading, spacing: MRSpace.s2) {
                StatusChip(text: value.status, tone: value.tone)
                readinessExplanation(title: title, value: value)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(title), \(value.status)")
        .accessibilityValue(value.reason)
    }

    private func readinessExplanation(
        title: String, value: ModelReadinessPresentationValue
    ) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s1) {
            Text(title).mrLabel()
            Text(value.reason)
                .font(MRType.caption)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func unsupportedArtifact(_ entry: CatalogEntry) -> some View {
        Panel(title: "Not supported in this version") {
            Text(
                "Minirun found this repository in the live catalog, but this version does not "
                    + "recognize its artifact layout or architecture. Download and run actions "
                    + "stay unavailable until support is explicit."
            )
            .font(MRType.caption)
            .foregroundStyle(MRColor.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder private var discoveredCopies: some View {
        let installations = model.installed.installations(of: modelID)
        if !installations.isEmpty {
            VStack(alignment: .leading, spacing: MRSpace.s3) {
                SectionHeader(title: "On this device")
                ForEach(installations) { artifact in
                    installedArtifactCard(artifact)
                }
            }
        }
    }

    private func artifact(_ entry: CatalogEntry) -> some View {
        let measured = model.installed.preferredArtifact(for: modelID)?.bytesOnDisk
            ?? download?.measuredBytes
        return Panel(
            title: "Artifact",
            trailing: AnyView(
                Button {
                    withAnimation(.easeInOut(duration: MRMotion.quick)) {
                        sourceDetailsExpanded.toggle()
                    }
                } label: {
                    Image(systemName: "chevron.down")
                        .rotationEffect(.degrees(sourceDetailsExpanded ? 180 : 0))
                        .frame(width: 44, height: 44)
                }
                .buttonStyle(.plain)
                .foregroundStyle(MRColor.secondary)
                .help(sourceDetailsExpanded ? "Hide source details" : "Show source details")
                .accessibilityLabel(
                    sourceDetailsExpanded ? "hide source details" : "show source details"))
        ) {
            MetricRow(
                label: "total size",
                value: measured.map { MRFormat.measuredBytes($0) }
                    ?? MRFormat.publishedBytes(entry.descriptor.totalBytes),
                provenance: measured == nil ? .declared : .measured,
                detail: measured == nil ? "published by the catalog" : "measured on disk")
            if sourceDetailsExpanded {
                Divider().overlay(MRColor.hairline)
                VStack(alignment: .leading, spacing: MRSpace.s2) {
                    MetricRow(label: "license", value: entry.descriptor.licenseName)
                    MetricRow(
                        label: "payload", value: MRFormat.publishedBytes(entry.descriptor.payloadBytes),
                        provenance: .declared,
                        detail: publishedFileCount(entry.descriptor.payloadFileCount))
                    MetricRow(
                        label: "metadata",
                        value: MRFormat.publishedBytes(entry.descriptor.metadataBytes),
                        provenance: .declared,
                        detail: publishedFileCount(entry.descriptor.metadataFileCount))
                    MetricRow(
                        label: "largest file",
                        value: MRFormat.publishedBytes(entry.descriptor.largestFileBytes),
                        provenance: .declared)
                    if let layers = entry.index.layers {
                        MetricRow(label: "layers", value: "\(layers)", provenance: .declared)
                    }
                    Divider().overlay(MRColor.hairline)
                    switch entry.descriptor.source {
                    case .huggingFaceRepo(let repo):
                        MetricRow(label: "repository", value: repo.repoID)
                        MetricRow(
                            label: "repository version",
                            value: MRFormat.shortDigest(repo.revision, head: 12, tail: 6),
                            detail: repo.isPinnedCommit
                                ? "exact version used to verify these files"
                                : "moving version — cannot be verified")
                        DigestLabel(digest: repo.revision, label: "full repository version")
                    case .locallyBuilt(let tool):
                        MetricRow(label: "created with", value: tool)
                    }
                }
            }
        }
    }

    private func publishedFileCount(_ count: Int) -> String {
        count > 0 ? "\(MRFormat.grouped(count)) files" : "file count not published"
    }

    private func storage(_ entry: CatalogEntry) -> some View {
        let installations = model.installed.installations(of: modelID)
        return VStack(alignment: .leading, spacing: MRSpace.s3) {
            SectionHeader(title: "On this device")

            if installations.isEmpty || hasTransferActivity {
                DownloadCard(
                    modelName: entry.descriptor.displayName,
                    state: model.state(of: modelID),
                    declaredBytes: entry.descriptor.totalBytes,
                    destinationPath: download?.destinationPath,
                    updatedAt: download?.jobID == nil ? nil : download?.updatedAt,
                    operationError: download?.operationError,
                    onPause: { Task { await download?.pause() } },
                    onResume: { Task { await download?.resume() } },
                    onCancel: { Task { await download?.cancel() } },
                    onVerify: { pendingDownloadVerification = true },
                    onCancelVerification: { download?.cancelVerification() },
                    onChooseVolume: { showingDestination = true })
            }
            if !installations.isEmpty {
                ForEach(installations) { artifact in
                    installedArtifactCard(artifact)
                }
            }

            assessedSpeed

            if installations.isEmpty || hasTransferActivity {
                HStack(spacing: MRSpace.s2) {
                    NavigationLink {
                        DownloadDetailView(modelID: modelID)
                    } label: {
                        Text("Download details")
                            .frame(minHeight: 44)
                    }
                    Spacer()
                }
                .controlSize(.small)

                if let refusal = model.downloadStartRefusal(for: modelID) {
                    Text(refusal)
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.caution)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private var hasTransferActivity: Bool {
        switch model.state(of: modelID) {
        case .notStarted, .cancelled, .ready: return false
        case .resolvingIndex, .awaitingDestination, .active, .paused, .interrupted,
            .verifying, .incomplete, .failed:
            return true
        }
    }

    private func installedArtifactCard(_ artifact: DiscoveredArtifact) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .firstTextBaseline, spacing: MRSpace.s3) {
                    installedArtifactIdentity(artifact)
                    Spacer(minLength: MRSpace.s3)
                    ValueText(
                        text: MRFormat.measuredBytes(artifact.bytesOnDisk),
                        provenance: .measured)
                }
                VStack(alignment: .leading, spacing: MRSpace.s1) {
                    installedArtifactIdentity(artifact)
                    ValueText(
                        text: MRFormat.measuredBytes(artifact.bytesOnDisk),
                        provenance: .measured)
                }
            }

            ViewThatFits(in: .horizontal) {
                HStack(spacing: MRSpace.s2) {
                    verificationStatus(artifact)
                    Spacer()
                    verificationTimestamp(artifact)
                }
                VStack(alignment: .leading, spacing: MRSpace.s1) {
                    verificationStatus(artifact)
                    verificationTimestamp(artifact)
                }
            }

            verificationControls(artifact)
            removalControls(artifact)
        }
        .padding(MRSpace.s3)
        .mrCard(MRColor.raised)
    }

    private func verificationStatus(_ artifact: DiscoveredArtifact) -> some View {
        HStack(spacing: MRSpace.s2) {
            Image(systemName: verificationIcon(artifact))
                .foregroundStyle(verificationColor(artifact))
            Text(ModelArtifactPresentation.status(artifact))
                .font(MRType.caption)
                .foregroundStyle(verificationColor(artifact))
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder private func verificationTimestamp(_ artifact: DiscoveredArtifact) -> some View {
        if let verifiedAt = artifact.verifiedAt {
            Text(MRFormat.timestamp(verifiedAt))
                .font(MRType.micro)
                .foregroundStyle(MRColor.tertiary)
        }
    }

    private func installedArtifactIdentity(_ artifact: DiscoveredArtifact) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(locationName(for: artifact))
                .font(MRType.headline)
                .foregroundStyle(MRColor.primary)
            Text(artifact.rootPath)
                .font(MRType.micro)
                .foregroundStyle(MRColor.tertiary)
                .textSelection(.enabled)
                .lineLimit(2)
                .truncationMode(.middle)
        }
    }

    @ViewBuilder private func verificationControls(_ artifact: DiscoveredArtifact) -> some View {
        if model.installed.verificationInFlight == artifact.rootPath {
            VStack(alignment: .leading, spacing: MRSpace.s1) {
                if let progress = model.installed.verificationProgress {
                    ProgressView(value: progress.fraction)
                        .frame(maxWidth: 260)
                    Text(progress.currentPath)
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.tertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                } else {
                    ProgressView("Starting verification…")
                        .controlSize(.small)
                }
                Button("Cancel verification", role: .cancel) {
                    model.installed.cancelVerification()
                }
                .controlSize(.small)
            }
        } else {
            VStack(alignment: .leading, spacing: MRSpace.s1) {
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: MRSpace.s2) {
                        verificationActionButtons(artifact)
                        Spacer()
                    }
                    VStack(alignment: .leading, spacing: MRSpace.s1) {
                        verificationActionButtons(artifact)
                    }
                }
                Text("Spot-check samples a few files. It does not enable chat.")
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .controlSize(.small)
        }
        if let outcome = model.installed.verificationOutcome[artifact.rootPath] {
            Text(outcome)
                .font(MRType.micro)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder private func verificationActionButtons(
        _ artifact: DiscoveredArtifact
    ) -> some View {
        Button("Spot-check") {
            Task { await model.installed.verify(artifact, .spotCheck()) }
        }
        .frame(minHeight: 44)
        .help("Check a bounded sample of the largest files.")
        .disabled(model.installed.verificationInFlight != nil)
        Button("Verify all files") {
            pendingFullVerification = artifact
        }
        .frame(minHeight: 44)
        .help(
            "Check every published digest. This reads about "
                + MRFormat.measuredBytes(artifact.expectedBytes ?? artifact.bytesOnDisk)
                + " and writes nothing.")
        .disabled(model.installed.verificationInFlight != nil)
        #if os(macOS)
            Button("Reveal in Finder") {
                NSWorkspace.shared.selectFile(
                    nil, inFileViewerRootedAtPath: artifact.rootPath)
            }
            .frame(minHeight: 44)
        #endif
    }

    @ViewBuilder private func removalControls(_ artifact: DiscoveredArtifact) -> some View {
        Divider().overlay(MRColor.hairline)
        let refusal = model.localCopyRemovalRefusal(for: artifact)
        if model.localCopyRemovalInFlight?.rootPath == artifact.rootPath {
            ProgressView("Removing this copy…")
                .controlSize(.small)
        } else {
            HStack(spacing: MRSpace.s2) {
                Button("Remove local copy…", role: .destructive) {
                    pendingRemoval = artifact
                }
                .frame(minHeight: 44)
                .disabled(refusal != nil)
                .help(refusal ?? "Permanently remove this exact local model directory.")
                Spacer()
            }
            .controlSize(.small)
            if let refusal {
                Text(refusal)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        if let outcome = model.localCopyRemovalMessage(for: artifact) {
            Text(outcome)
                .font(MRType.micro)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var fullVerificationConfirmationPresented: Binding<Bool> {
        Binding(
            get: { pendingFullVerification != nil },
            set: { presented in
                if !presented { pendingFullVerification = nil }
            })
    }

    private var removalConfirmationPresented: Binding<Bool> {
        Binding(
            get: { pendingRemoval != nil },
            set: { presented in
                if !presented { pendingRemoval = nil }
            })
    }

    private func locationName(for artifact: DiscoveredArtifact) -> String {
        model.installed.report.locations
            .first { $0.rootPath == artifact.locationPath }?.displayName
            ?? URL(fileURLWithPath: artifact.locationPath).lastPathComponent
    }

    private func verificationIcon(_ artifact: DiscoveredArtifact) -> String {
        if artifact.isComplete == false { return "exclamationmark.triangle.fill" }
        switch artifact.verification {
        case .unverified: return "shield.lefthalf.filled"
        case .spotChecked: return "checkmark.shield"
        case .fullyVerified: return "checkmark.seal.fill"
        }
    }

    private func verificationColor(_ artifact: DiscoveredArtifact) -> Color {
        if artifact.isComplete == false { return MRColor.caution }
        switch artifact.verification {
        case .unverified: return MRColor.secondary
        case .spotChecked: return MRColor.verify
        case .fullyVerified: return MRColor.ok
        }
    }

    /// What a token would cost on the volume this model is actually installed
    /// on, at the rate that volume measured.
    ///
    /// Shown only when both halves are true — the model is installed somewhere,
    /// and that somewhere has been assessed. Without a measurement there is no
    /// line at all, which is the same rule the dial's projection keeps.
    @ViewBuilder private var assessedSpeed: some View {
        if let report = model.assessment(for: modelID),
            let verdict = model.assessedProjection(for: modelID),
            let projection = verdict.projection
        {
            VStack(alignment: .leading, spacing: 1) {
                Text(
                    AssessmentFormat.onVolumeLine(
                        volumeName: report.volumeName, projection: projection,
                        measuredAt: report.assessedAt)
                )
                .font(MRType.metric)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
                if report.isStale() {
                    Text("that measurement is more than a week old — re-assess the volume.")
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.caution)
                }
                if let degraded = StorageAssessmentPresentation.fasterLinkSentence(report) {
                    Text(degraded)
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.caution)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .accessibilityElement(children: .combine)
        }
    }

    /// The dial here states a **default**, not a run: this screen is about the
    /// model, and a run belongs to a conversation. The role label on the dial
    /// says which of the two it is, every time.
    @ViewBuilder private func budget(_ entry: CatalogEntry) -> some View {
        if model.capabilities(for: modelID) != nil {
            VStack(alignment: .leading, spacing: MRSpace.s3) {
                SectionHeader(title: "Budget")
                if let plan = model.defaultBudgetPlan(for: modelID) {
                    MemoryDialView(
                        plan: plan,
                        entry: entry,
                        role: .defaultForNewChats,
                        onChange: { model.setDefaultBudget($0, for: modelID) })
                }
                Button("Start a chat with \(entry.descriptor.displayName)") {
                    _ = model.newConversation(modelID: modelID)
                }
                .frame(minHeight: 44)
                .buttonStyle(.borderedProminent)
                .tint(MRColor.tierPinned)
                .disabled(!model.modelAvailability(for: modelID).isAvailable)
            }
        } else {
            Panel(title: "Budget") {
                Text(
                    "No verified runner and tokenizer ship for \(entry.descriptor.displayName) "
                        + "in this build, so it is unavailable for chats and there is no run "
                        + "budget to state."
                )
                .font(MRType.caption)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func history(_ records: [RunRecord]) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            SectionHeader(title: "Recent runs")
            Group {
                #if os(macOS)
                    desktopHistoryTable(records)
                #else
                    LazyVStack(spacing: MRSpace.s2) {
                        ForEach(records) { record in
                            compactHistoryCard(record)
                        }
                    }
                #endif
            }
        }
    }

    #if os(macOS)
        private func desktopHistoryTable(_ records: [RunRecord]) -> some View {
            VStack(spacing: 0) {
                historyHeader
                ForEach(records) { record in
                    historyRow(record)
                }
            }
            .mrCard()
        }
    #endif

    private func compactHistoryCard(_ record: RunRecord) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
                Text(MRFormat.timestamp(record.at))
                    .font(MRType.caption)
                    .foregroundStyle(MRColor.primary)
                Spacer(minLength: MRSpace.s2)
                Text(record.thermalState.rawValue)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.secondary)
            }
            Divider().overlay(MRColor.hairline)
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: MRSpace.s4) {
                    historyMetric(
                        label: "budget",
                        value: MRFormat.bytesDecimal(record.declaredBudgetBytes))
                    historyMetric(
                        label: "time / token",
                        value: String(format: "%.1f s", record.secondsPerToken))
                    historyMetric(
                        label: "peak",
                        value: MRFormat.bytesDecimal(record.peakFootprintBytes),
                        color: record.budgetRespected ? MRColor.primary : MRColor.refuse)
                }
                VStack(alignment: .leading, spacing: MRSpace.s2) {
                    historyMetric(
                        label: "budget",
                        value: MRFormat.bytesDecimal(record.declaredBudgetBytes))
                    historyMetric(
                        label: "time / token",
                        value: String(format: "%.1f s", record.secondsPerToken))
                    historyMetric(
                        label: "peak",
                        value: MRFormat.bytesDecimal(record.peakFootprintBytes),
                        color: record.budgetRespected ? MRColor.primary : MRColor.refuse)
                }
            }
            DigestLabel(digest: record.logitsDigest, label: "logits", showsCopyButton: false)
        }
        .padding(MRSpace.s3)
        .mrCard(MRColor.raised)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(
            "run at \(MRFormat.timestamp(record.at)), budget "
                + "\(MRFormat.bytesDecimal(record.declaredBudgetBytes)), "
                + "\(Int(record.secondsPerToken)) seconds per token, peak "
                + "\(MRFormat.bytesDecimal(record.peakFootprintBytes)), thermal "
                + "\(record.thermalState.rawValue)")
    }

    private func historyMetric(
        label: String, value: String, color: Color = MRColor.primary
    ) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).mrLabel()
            Text(value)
                .font(MRType.metric)
                .foregroundStyle(color)
        }
    }

    #if os(macOS)
        private var historyHeader: some View {
            HStack {
                Text("date").mrLabel().frame(width: 100, alignment: .leading)
                Text("budget").mrLabel().frame(width: 80, alignment: .trailing)
                Text("s/token").mrLabel().frame(width: 70, alignment: .trailing)
                Text("peak").mrLabel().frame(width: 80, alignment: .trailing)
                Text("thermal").mrLabel().frame(width: 70, alignment: .leading)
                Text("logits").mrLabel()
                Spacer()
            }
            .padding(.horizontal, MRSpace.s3)
            .padding(.vertical, MRSpace.s2)
        }

        private func historyRow(_ record: RunRecord) -> some View {
            HStack {
                Text(MRFormat.timestamp(record.at))
                    .font(MRType.micro).frame(width: 100, alignment: .leading)
                Text(MRFormat.bytesDecimal(record.declaredBudgetBytes))
                    .font(MRType.micro).frame(width: 80, alignment: .trailing)
                Text(String(format: "%.1f", record.secondsPerToken))
                    .font(MRType.micro).frame(width: 70, alignment: .trailing)
                Text(MRFormat.bytesDecimal(record.peakFootprintBytes))
                    .font(MRType.micro)
                    .foregroundStyle(record.budgetRespected ? MRColor.primary : MRColor.refuse)
                    .frame(width: 80, alignment: .trailing)
                Text(record.thermalState.rawValue)
                    .font(MRType.micro).frame(width: 70, alignment: .leading)
                DigestLabel(digest: record.logitsDigest, showsCopyButton: false)
                Spacer()
            }
            .foregroundStyle(MRColor.secondary)
            .padding(.horizontal, MRSpace.s3)
            .padding(.vertical, MRSpace.s2)
            .overlay(alignment: .top) { Rectangle().fill(MRColor.hairline).frame(height: 1) }
            .accessibilityElement(children: .combine)
            .accessibilityLabel(
                "run at \(MRFormat.timestamp(record.at)), budget "
                    + "\(MRFormat.bytesDecimal(record.declaredBudgetBytes)), "
                    + "\(Int(record.secondsPerToken)) seconds per token, peak "
                    + "\(MRFormat.bytesDecimal(record.peakFootprintBytes)), thermal "
                    + "\(record.thermalState.rawValue)")
        }
    #endif

}

enum ModelFitnessPresentation {
    static func status(_ verdict: PlatformFitness.Verdict) -> String {
        switch verdict {
        case .runnable: return "Ready"
        case .runnableWithCaveats: return "Runs with limits"
        case .refused: return "Can't run on this device"
        case .noRunner: return "Can't run in this version"
        }
    }
}

/// The two model-detail rows share a small value type but never share their
/// authority: compatibility comes from the linked product runtime and device,
/// while integrity comes only from discovered local bytes.
struct ModelReadinessPresentationValue: Equatable {
    let status: String
    let reason: String
    let tone: StatusChip.Tone

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.status == rhs.status && lhs.reason == rhs.reason && lhs.tone == rhs.tone
    }
}

/// Whether this App and device can use the model family. File integrity is a
/// separate row: a perfectly verified copy can still target an unsupported
/// layout, and an unverified copy does not make a supported runtime disappear.
enum ModelCompatibilityPresentation {
    static func resolve(
        modelName: String, fitness: PlatformFitness, availability: ModelRunAvailability
    ) -> ModelReadinessPresentationValue {
        switch fitness.verdict {
        case .noRunner:
            return ModelReadinessPresentationValue(
                status: "Not supported by this version",
                reason: "This version can manage \(modelName), but cannot use it in a chat.",
                tone: .neutral)
        case .refused:
            return ModelReadinessPresentationValue(
                status: "Not supported on this device",
                reason: fitness.reason, tone: .refuse)
        case .runnable, .runnableWithCaveats:
            break
        }

        switch availability {
        case .runtimeFilesUnavailable:
            return ModelReadinessPresentationValue(
                status: "Local version not supported",
                reason: "This version supports \(modelName), but this local copy needs an update.",
                tone: .caution)
        case .runtimeUnavailable:
            return ModelReadinessPresentationValue(
                status: "Not supported by this version",
                reason: "This version can manage \(modelName), but cannot use it in a chat.",
                tone: .neutral)
        case .missingFromCatalog:
            return ModelReadinessPresentationValue(
                status: "Catalog entry unavailable",
                reason: "Compatibility cannot be confirmed without current model information.",
                tone: .neutral)
        case .unrecognizedArtifact:
            return ModelReadinessPresentationValue(
                status: "Format not supported",
                reason: "This version does not recognize the local artifact format.",
                tone: .neutral)
        case .available, .notFoundByScan, .storageUnavailable, .incompleteArtifact,
            .unverifiedArtifact:
            return ModelReadinessPresentationValue(
                status: fitness.verdict == .runnable ? "Supported" : "Supported with limits",
                reason: fitness.verdict == .runnable
                    ? "This version supports \(modelName) on this device."
                    : "This version supports \(modelName), with the device limit shown below.",
                tone: fitness.verdict == .runnable ? .ok : .caution)
        }
    }
}

/// What has been established about local bytes, independent of whether any
/// runtime can consume them. A spot-check is visibly a sample and never reads
/// as the full-verification gate used by chat.
enum ModelIntegrityPresentation {
    static func resolve(
        installations: [DiscoveredArtifact], mounted: [DiscoveredArtifact]
    ) -> ModelReadinessPresentationValue {
        guard !installations.isEmpty else {
            return ModelReadinessPresentationValue(
                status: "Not on this device",
                reason: "Download a copy or add the folder that already contains it.",
                tone: .neutral)
        }
        guard !mounted.isEmpty else {
            return ModelReadinessPresentationValue(
                status: "Storage unavailable",
                reason: "Reconnect the storage that contains this model.",
                tone: .caution)
        }
        let complete = mounted.filter { $0.isComplete != false }
        guard !complete.isEmpty else {
            return ModelReadinessPresentationValue(
                status: "Files incomplete",
                reason: "This local copy is missing files published by the model repository.",
                tone: .caution)
        }
        if complete.contains(where: { $0.verification == .fullyVerified }) {
            return ModelReadinessPresentationValue(
                status: "Fully verified",
                reason: "Every published model file matched its recorded digest.",
                tone: .ok)
        }
        if complete.contains(where: { $0.verification == .spotChecked }) {
            return ModelReadinessPresentationValue(
                status: "Spot-check only",
                reason: "A sample matched. Verify all files before using this copy in a chat.",
                tone: .verify)
        }
        return ModelReadinessPresentationValue(
            status: "Full verification required",
            reason: "Verify all files before using this local copy in a chat.",
            tone: .verify)
    }
}

/// One honest, compact status line for a discovered copy. Completeness comes
/// from directory metadata; verification comes from an explicit digest pass,
/// and the two are never collapsed into a generic "installed" badge.
enum ModelArtifactPresentation {
    static func status(_ artifact: DiscoveredArtifact) -> String {
        if let missing = artifact.missingFileCount, missing > 0 {
            return "Incomplete — \(MRFormat.grouped(missing)) file\(missing == 1 ? "" : "s") missing"
        }
        let files = "\(MRFormat.grouped(artifact.fileCount)) file\(artifact.fileCount == 1 ? "" : "s")"
        switch artifact.verification {
        case .unverified: return "\(files) · Not verified"
        case .spotChecked: return "\(files) · Spot-check passed"
        case .fullyVerified: return "\(files) · Fully verified"
        }
    }
}

enum FullVerificationConfirmation {
    static let actionTitle = "Start full verification"

    static func bytes(for artifact: DiscoveredArtifact) -> UInt64 {
        artifact.expectedBytes ?? artifact.bytesOnDisk
    }

    static func message(for artifact: DiscoveredArtifact) -> String {
        "This will read approximately \(MRFormat.publishedBytes(bytes(for: artifact))) and "
            + "may take a long time. It checks published digests, does not modify model files, "
            + "and can be cancelled."
    }
}

enum DownloadVerificationConfirmation {
    static let actionTitle = "Start re-verification"

    static func message(readBytes: UInt64) -> String {
        "This will read approximately \(MRFormat.publishedBytes(readBytes)) and may take a "
            + "long time. It checks every downloaded file against the exact repository version, "
            + "does not modify model files, and can be cancelled."
    }
}

enum LocalCopyRemovalConfirmation {
    static let actionTitle = "Remove local copy"

    static func message(modelName: String, artifact: DiscoveredArtifact) -> String {
        "Model: \(modelName)\n"
            + "Location: \(artifact.rootPath)\n"
            + "Measured size: \(MRFormat.measuredBytes(artifact.bytesOnDisk))\n\n"
            + "This permanently deletes this exact model directory and cannot be undone. "
            + "Other copies and the registered storage location are not removed."
    }
}
