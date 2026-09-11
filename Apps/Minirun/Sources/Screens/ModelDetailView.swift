import MinirunKit
import MinirunRunners
import SwiftUI

#if canImport(AppKit)
    import AppKit
#endif

/// The model's product page.
///
/// One surface, hairline separators, read top to bottom: what this is and what
/// state it is in, then the transfer as the single focal block, then the facts
/// as a quiet definition list, then the copies, the memory default, the run
/// receipts and the earlier attempts.
///
/// It used to be six bordered cards of equal weight under six ALL-CAPS labels —
/// Readiness, Artifact, On this device, Budget, Recent runs — and the eye had
/// nowhere to land. Nothing was removed to fix that; the same facts are here,
/// arranged by how much they matter. Receipts still disappear entirely when
/// there are none, rather than turning an empty database count into a feature.
struct ModelDetailView: View {
    let modelID: ModelID

    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var showingDestination = false
    @State private var pendingFullVerification: DiscoveredArtifact?
    @State private var pendingDownloadVerification = false
    @State private var pendingRemoval: DiscoveredArtifact?
    @State private var pendingForget: DownloadJobID?
    @State private var forgetWarning: String?

    private var entry: CatalogEntry? { model.entry(modelID) }
    private var download: DownloadController? { model.download(modelID) }

    var body: some View {
        Group {
            #if os(macOS)
                VStack(spacing: 0) {
                    MacColumnHeader(
                        title: entry?.descriptor.displayName ?? modelID.rawValue,
                        backAction: { dismiss() })
                    MRHairline()
                    detailContent
                }
            #else
                detailContent
            #endif
        }
        .mrProductPage()
        .mrPhoneNavigationTitle(entry?.descriptor.displayName ?? modelID.rawValue)
        .sheet(isPresented: $showingDestination) {
            DestinationPicker(modelID: modelID, onStarted: { showingDestination = false })
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
        .alert(
            ForgetTransferConfirmation.title,
            isPresented: forgetConfirmationPresented
        ) {
            forgetConfirmationActions
        } message: {
            Text(
                ForgetTransferConfirmation.message(
                    destinationPath: forgetDestinationPath))
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
        .confirmationDialog(
            ForgetTransferConfirmation.title,
            isPresented: forgetConfirmationPresented,
            titleVisibility: .visible
        ) {
            forgetConfirmationActions
        } message: {
            Text(
                ForgetTransferConfirmation.message(
                    destinationPath: forgetDestinationPath))
        }
        #endif
    }

    // MARK: - Confirmations

    private var forgetConfirmationPresented: Binding<Bool> {
        Binding(get: { pendingForget != nil }, set: { if !$0 { pendingForget = nil } })
    }

    @ViewBuilder private var forgetConfirmationActions: some View {
        if let job = pendingForget {
            Button(ForgetTransferConfirmation.actionTitle, role: .destructive) {
                pendingForget = nil
                forgetWarning = model.forgetDownloadJob(job)
            }
        }
        Button("Cancel", role: .cancel) { pendingForget = nil }
    }

    /// The destination the confirmation names. It is the pending record's own
    /// path, not the screen's current transfer: *Earlier transfers* can offer
    /// to forget a record the block above is not showing.
    private var forgetDestinationPath: String? {
        guard let job = pendingForget else { return nil }
        return model.downloadController(for: job)?.destinationPath
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

    // MARK: - The page

    /// The scrolling half of the screen. The page itself is a separate view so
    /// it can be laid out — and rendered to a PNG in the suite — without a
    /// scroll view around it: `ImageRenderer` draws nothing at all for a macOS
    /// `ScrollView`, and a redesign nobody can look at offscreen is a redesign
    /// that gets reviewed by launching the app.
    private var detailContent: some View {
        ScrollView {
            ModelDetailPage(
                modelID: modelID,
                showingDestination: $showingDestination,
                pendingFullVerification: $pendingFullVerification,
                pendingDownloadVerification: $pendingDownloadVerification,
                pendingRemoval: $pendingRemoval,
                pendingForget: $pendingForget,
                forgetWarning: $forgetWarning)
        }
        .frame(maxWidth: .infinity)
    }
}

/// Everything below the title bar: header, transfer, facts, copies, memory,
/// receipts, earlier transfers.
///
/// It owns no confirmation of its own — every destructive or expensive action
/// sets one of the bindings above and the screen presents the dialog — so this
/// view is a pure function of the app model plus those pending values, which is
/// what makes it renderable in isolation.
struct ModelDetailPage: View {
    let modelID: ModelID
    @Binding var showingDestination: Bool
    @Binding var pendingFullVerification: DiscoveredArtifact?
    @Binding var pendingDownloadVerification: Bool
    @Binding var pendingRemoval: DiscoveredArtifact?
    @Binding var pendingForget: DownloadJobID?
    @Binding var forgetWarning: String?

    @Environment(AppModel.self) private var model
    @Environment(\.openURL) private var openURL
    @State private var sourceDetailsExpanded = false

    private var entry: CatalogEntry? { model.entry(modelID) }
    private var download: DownloadController? { model.download(modelID) }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let entry {
                productHeader(entry)
                #if os(iOS)
                    if entry.id == .kimiK3,
                        K3ProductMemoryBudget.currentPolicy.isExperimental
                    {
                        k3ExperimentalTier
                    }
                #endif
                if entry.descriptor.layout == .unrecognized {
                    unsupportedArtifact
                } else {
                    transferSection(entry)
                }
                // The copies come before About, and on both platforms. They are
                // the actionable state of this page — where the bytes are,
                // whether they have been checked, and the controls that check or
                // remove them — and they used to sit below a column of facts and
                // a column of readiness sentences, which on the Mac put them
                // under the fold on the day a 517 GB transfer finished.
                copiesSection(entry)
                aboutSection(entry)
                if model.hasVerifiedRuntime(for: entry.id) {
                    budget(entry)
                    let records = model.history(for: entry.id)
                    if !records.isEmpty { history(records) }
                }
                earlierTransfers
            } else {
                EmptyStateView(
                    headline: "Unknown model.",
                    message: "This build has no descriptor for \(modelID.rawValue).")
            }
        }
        .padding(pagePadding)
        .frame(maxWidth: 860, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .leading)
        .task(id: keptFilesReconciliationKey) {
            await download?.reconcileKeptFiles()
        }
    }

    private var pagePadding: CGFloat {
        #if os(macOS)
            MRSpace.s5
        #else
            MRSpace.s4
        #endif
    }

    /// The job the transfer block is showing, and only while it has stopped. A
    /// running, paused or verifying record is bookkeeping the app is still
    /// using, and the page does not offer to drop it.
    private var forgettableJob: DownloadJobID? {
        guard let job = download?.jobID, model.canForgetDownloadJob(job) else { return nil }
        return job
    }

    // MARK: - Header

    private func productHeader(_ entry: CatalogEntry) -> some View {
        let status = status(entry)
        return MRProductHeader(
            title: entry.descriptor.displayName,
            subtitle: ModelPurposePresentation.line(for: entry.descriptor),
            status: (tone: status.tone, sentence: status.sentence)
        ) {
            MRProductTile {
                ModelPublisherMark(
                    identity: .resolve(
                        modelID: entry.id,
                        displayName: entry.descriptor.displayName,
                        repositoryID: entry.descriptor.source.repo?.repoID),
                    fallbackSystemName: "cube",
                    fallbackColor: MRColor.tertiary,
                    size: 40)
            }
        } actions: {
            headerActions(entry)
        }
        .padding(.bottom, MRSpace.s2)
    }

    /// The one status sentence at the top of the page, resolved from the disk
    /// first and the transfer record second.
    private func status(_ entry: CatalogEntry) -> ModelPageStatus {
        ModelPageStatusPresentation.resolve(
            state: model.state(of: modelID),
            remains: remains,
            installations: model.installed.installations(of: entry.id),
            mounted: mountedInstallations(entry),
            transferDestinationName: transferDestinationName,
            copyLocationName: describedArtifact.map { locationName(for: $0) })
    }

    /// Download / Pause · Cancel / New chat · Verify again, and nothing else.
    /// Every other control on this page is a text link.
    @ViewBuilder private func headerActions(_ entry: CatalogEntry) -> some View {
        switch model.state(of: modelID) {
        case .active:
            Button("Pause") { Task { await download?.pause() } }.mrOutlineAction()
            Button("Cancel") { Task { await download?.cancel() } }.mrOutlineAction()
        case .paused:
            Button("Resume") { Task { await download?.resume() } }.mrFilledAction()
            Button("Cancel") { Task { await download?.cancel() } }.mrOutlineAction()
        case .verifying:
            Button("Cancel verification") { download?.cancelVerification() }
                .mrOutlineAction()
        case .resolvingIndex:
            EmptyView()
        case .incomplete:
            Button("Verify again") { pendingDownloadVerification = true }.mrFilledAction()
        case .interrupted, .cancelled, .failed:
            startAction
        case .notStarted, .awaitingDestination, .ready:
            if preferredArtifact != nil {
                chatAction(entry)
                Button("Verify again") { verifyPreferredCopy() }
                    .mrOutlineAction()
                    .disabled(model.installed.verificationInFlight != nil)
            } else if describedArtifact != nil {
                // The copy is on a drive that is not plugged in. There is
                // nothing to press: a Download button here would offer to
                // fetch 167 GB the operator already owns, and the status
                // sentence above already says what to do — plug it in.
                EmptyView()
            } else if model.state(of: modelID).isReady {
                // The transfer says the bytes are down and checked; the scan
                // has not seen them yet. Offering another download here would
                // ask for 517 GB the drive may already hold.
                Button("Verify again") { pendingDownloadVerification = true }
                    .mrOutlineAction()
            } else {
                startAction
            }
        }
    }

    /// The one button that begins, or continues, a transfer. A stopped one may
    /// not offer to continue with files nobody found, so its title and its
    /// enabled state both come from the reconciliation.
    private var startAction: some View {
        let state = model.state(of: modelID)
        let title = state.hasStopped
            ? TransferRemainsPresentation.actionTitle(remains)
            : "Download…"
        return Button(title) { showingDestination = true }
            .mrFilledAction()
            .disabled(
                model.downloadStartRefusal(for: modelID) != nil
                    || (state.hasStopped
                        && !TransferRemainsPresentation.actionIsEnabled(remains)))
    }

    @ViewBuilder private func chatAction(_ entry: CatalogEntry) -> some View {
        if model.hasVerifiedRuntime(for: entry.id) {
            Button("New chat") { _ = model.newConversation(modelID: entry.id) }
                .mrFilledAction()
                .disabled(!model.modelAvailability(for: entry.id).isAvailable)
        }
    }

    private func verifyPreferredCopy() {
        guard let artifact = preferredArtifact else { return }
        pendingFullVerification = artifact
    }

    #if os(iOS)
        private var k3ExperimentalTier: some View {
            MRInlineNote(
                message:
                    "Uses the 5.80 GB device record and limits each response to two "
                    + "tokens while this memory profile is validated. File "
                    + "verification and the memory plan remain required.",
                title: "Bounded experimental tier",
                tone: .attention,
                systemImage: "testtube.2"
            )
            .padding(.top, MRSpace.s4)
        }
    #endif

    private var unsupportedArtifact: some View {
        MRInlineNote(
            message:
                "Minirun found this repository in the live catalog, but this version does not "
                + "recognize its artifact layout or architecture. Download and run actions "
                + "stay unavailable until support is explicit.",
            title: "Not supported in this version",
            tone: .attention
        )
        .padding(.top, MRSpace.s4)
    }

    // MARK: - The transfer

    /// What a re-check of the destination found. `.checking` until one has been
    /// done — the block never derives this from the progress snapshot it holds.
    private var remains: TransferRemains {
        TransferRemains.resolve(
            download?.keptFiles, destinationPath: download?.destinationPath)
    }

    /// The transfer block appears while there is a transfer to talk about: one
    /// in motion, one that stopped, or none at all with nothing on the drive
    /// yet. A model that is simply installed and quiet has no transfer, and
    /// this section is then absent rather than empty.
    private var showsTransfer: Bool {
        let state = model.state(of: modelID)
        if state.isBusy || state.hasStopped { return true }
        switch state {
        case .paused, .incomplete: return true
        case .ready: return model.installed.installations(of: modelID).isEmpty
        case .notStarted, .awaitingDestination:
            return model.installed.installations(of: modelID).isEmpty
        default: return false
        }
    }

    @ViewBuilder private func transferSection(_ entry: CatalogEntry) -> some View {
        if showsTransfer {
            MRPageSection {
                transferBlock(entry)
                // The named condition a stopped transfer is in, and what to do
                // about it — a drive to reconnect, a folder to re-grant. Only
                // the states that have one produce this.
                if model.state(of: modelID).hasStopped,
                    let detail = TransferRemainsPresentation.detail(remains)
                {
                    MRInlineNote(message: detail)
                }
                if let error = download?.operationError {
                    NamedErrorCard(
                        headline: "The last action did not finish.",
                        message: "The transfer state was kept unchanged.",
                        namedError: error,
                        tone: .caution)
                }
                if let forgetWarning {
                    MRInlineNote(message: forgetWarning)
                }
                // Only while the page offers a start. A running transfer's own
                // existence is not a warning to print under it, and a refusal
                // is only a note when there is a disabled button it explains.
                if model.state(of: modelID).offersStart,
                    let refusal = model.downloadStartRefusal(for: modelID)
                {
                    MRInlineNote(message: refusal)
                }
            }
        }
    }

    @ViewBuilder private func transferBlock(_ entry: CatalogEntry) -> some View {
        let state = model.state(of: modelID)
        switch state {
        case .active(let progress), .paused(let progress):
            MRTransferBlock(
                headline: MRFormat.bytesDecimal(progress.bytesOnDisk),
                headlineDetail: "of \(MRFormat.bytesDecimal(progress.totalBytes))",
                meta: TransferPacePresentation.line(progress, isMoving: state.isBusy),
                fraction: progress.fraction,
                path: download?.destinationPath
            ) {
                transferLinks
            }
        case .verifying(let phase, let progress):
            MRTransferBlock(
                headline: MRFormat.bytesDecimal(phase.bytesChecked ?? progress.totalBytes),
                headlineDetail: phase.bytesChecked == nil
                    ? "to check" : "of \(MRFormat.bytesDecimal(progress.totalBytes)) checked",
                meta: phase.filesLine,
                fraction: phase.fraction,
                sentence: phase.sentence,
                path: download?.destinationPath
            ) {
                transferLinks
            }
        case .resolvingIndex:
            MRTransferBlock(
                sentence: "Resolving the index and walking the tree…",
                path: download?.destinationPath
            ) {
                transferLinks
            }
        case .cancelled(let progress):
            stoppedBlock(totalBytes: progress.totalBytes, snapshot: progress)
        case .interrupted(let progress, let reason):
            VStack(alignment: .leading, spacing: MRSpace.s3) {
                MRInlineNote(
                    message: reason.recoverySentence,
                    title: "Interrupted — \(reason.sentence)")
                stoppedBlock(totalBytes: progress.totalBytes, snapshot: progress)
            }
        case .failed(let error):
            VStack(alignment: .leading, spacing: MRSpace.s3) {
                NamedErrorCard(
                    headline: "The transfer failed.",
                    message: "What it left on the drive is stated below.",
                    namedError: error.description)
                stoppedBlock(totalBytes: entry.descriptor.totalBytes, snapshot: nil)
            }
        case .incomplete(let files, let bytes):
            NamedErrorCard(
                headline: "\(files) file\(files == 1 ? "" : "s") did not verify.",
                message:
                    "Those bytes are not trusted and will be re-fetched from zero. A byte that "
                    + "failed a digest is not a byte to resume from.",
                namedError: "verificationIncomplete(pathsToRefetch: \(files))",
                tone: .caution,
                numbers: [("data to re-fetch", MRFormat.measuredBytes(bytes))])
        case .ready(let measured, let fileCount):
            MRTransferBlock(
                headline: MRFormat.bytesDecimal(measured),
                headlineDetail: "on the drive",
                meta:
                    "\(MRFormat.grouped(fileCount)) planned files checked",
                fraction: 1,
                sentence: "Every planned file was checked against its published digest.",
                path: download?.destinationPath
            ) {
                transferLinks
            }
        case .notStarted, .awaitingDestination:
            MRTransferBlock(
                headline: MRFormat.publishedBytes(entry.descriptor.totalBytes),
                headlineDetail: "to download",
                meta: publishedFileCount(entry.descriptor.totalFileCount),
                fraction: nil,
                sentence: "Nothing has been transferred yet."
            ) {
                transferLinks
            }
        }
    }

    /// A stopped transfer, drawn entirely from a re-check of the destination.
    /// No rate, no estimate and no resume offset: a cancelled job has no speed,
    /// and the offset it stopped at is not a fact about the drive.
    ///
    /// The kept-files sentence is the page's status line, above — it is the
    /// whole state of this model right now — so the block holds the numbers and
    /// the destination and does not print it twice. When nothing was kept there
    /// is no number and no bar either: `0 GB` under an empty rail is a claim
    /// about a drive that has nothing on it.
    private func stoppedBlock(
        totalBytes: UInt64, snapshot: ProgressSnapshot?
    ) -> some View {
        MRTransferBlock(
            headline: keptHeadline,
            headlineDetail: keptHeadline == nil
                ? nil : "of \(MRFormat.bytesDecimal(totalBytes))",
            meta: TransferRemainsPresentation.progressLine(remains, snapshot: snapshot),
            fraction: keptHeadline == nil
                ? nil
                : TransferRemainsPresentation.keptFraction(remains, totalBytes: totalBytes),
            sentence: nil,
            path: download?.destinationPath,
            accessibilitySummary: TransferRemainsPresentation.sentence(remains)
        ) {
            transferLinks
        }
    }

    /// The kept bytes, and only when a look at the drive found some. Nothing
    /// kept gets no big number at all: `0 GB` is a number, and this block's
    /// number is reserved for something that is there.
    private var keptHeadline: String? {
        guard case .kept(_, _, let bytes) = remains else { return nil }
        return MRFormat.bytesDecimal(bytes)
    }

    @ViewBuilder private var transferLinks: some View {
        NavigationLink {
            DownloadDetailView(modelID: modelID)
        } label: {
            Text("Transfer details")
        }
        .mrTextLink()

        #if os(macOS)
            if let path = download?.destinationPath {
                Button("Show in Finder") {
                    NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: path)
                }
                .mrTextLink()
            }
        #endif

        if let job = forgettableJob {
            Button(ForgetTransferConfirmation.cardActionTitle) { pendingForget = job }
                .mrTextLink(.quiet)
        }
    }

    private var transferDestinationName: String? {
        if let volume = download?.volumeName, !volume.isEmpty { return volume }
        guard let path = download?.destinationPath else { return nil }
        return URL(fileURLWithPath: path).deletingLastPathComponent().lastPathComponent
    }

    /// Re-check the drive when this screen appears, when the transfer's state
    /// changes, and when a storage scan has looked at the volumes again. Each
    /// of those is a moment when what is on disk may differ from what the last
    /// download event said, and the check is size-only and cheap.
    private var keptFilesReconciliationKey: KeptFilesReconciliationKey {
        KeptFilesReconciliationKey(
            state: model.state(of: modelID).phrase,
            destinationPath: download?.destinationPath,
            scannedAt: model.installed.report.scannedAt)
    }

    // MARK: - About

    private func aboutSection(_ entry: CatalogEntry) -> some View {
        MRPageSection(title: "About") {
            facts(entry)
            assessmentWarnings
        }
    }

    private func facts(_ entry: CatalogEntry) -> some View {
        MRFactList {
            readinessFacts(entry)
            sizeFact(entry)
            if let precision = ArtifactPrecisionPresentation.sentence(for: entry.descriptor.layout)
            {
                MRFact("Precision", precision, provenance: .declared)
            }
            MRFact("License", entry.descriptor.licenseName, provenance: .declared)
            sourceFact(entry)
            revisionFact(entry)
            if let artifact = describedArtifact {
                MRFact(label: "Location") {
                    MRPathLabel(path: artifact.rootPath)
                }
                if let verifiedAt = artifact.verifiedAt {
                    MRFact("Last verified", MRFormat.timestamp(verifiedAt))
                }
            }
            lastRunFact
            assessedSpeedFact
            if sourceDetailsExpanded { moreFacts(entry) }
            Button(sourceDetailsExpanded ? "Fewer details" : "More details") {
                withAnimation(.easeInOut(duration: MRMotion.quick)) {
                    sourceDetailsExpanded.toggle()
                }
            }
            .mrTextLink()
            .accessibilityLabel(
                sourceDetailsExpanded ? "hide source details" : "show source details")
        }
    }

    /// Measured where a copy or a finished transfer has been counted, published
    /// otherwise — and the type says which, upright versus italic.
    @ViewBuilder private func sizeFact(_ entry: CatalogEntry) -> some View {
        let measured = describedArtifact?.bytesOnDisk ?? download?.measuredBytes
        if let measured, let artifact = describedArtifact {
            MRFact(
                "Size on disk",
                "\(MRFormat.measuredBytes(measured)) · "
                    + "\(MRFormat.grouped(artifact.fileCount)) files")
        } else if let measured {
            MRFact("Size on disk", MRFormat.measuredBytes(measured))
        } else {
            MRFact(
                "Published size",
                "\(MRFormat.publishedBytes(entry.descriptor.totalBytes)) · "
                    + publishedFileCount(entry.descriptor.totalFileCount),
                provenance: .declared)
        }
    }

    @ViewBuilder private func sourceFact(_ entry: CatalogEntry) -> some View {
        switch entry.descriptor.source {
        case .huggingFaceRepo(let repo):
            MRFact(label: "Source") {
                // A `Button` and not a `Link`: `Link` is AppKit-backed on macOS
                // and draws as an unrenderable placeholder in `ImageRenderer`,
                // which is where this page is reviewed. `openURL` is what
                // `Link` calls anyway.
                Button(repo.repoID) { openURL(ModelSourceLink.url(for: repo)) }
                    .mrTextLink()
                    .accessibilityLabel("open \(repo.repoID) on Hugging Face")
            }
        case .locallyBuilt(let tool):
            MRFact("Created with", tool, provenance: .declared)
        }
    }

    @ViewBuilder private func revisionFact(_ entry: CatalogEntry) -> some View {
        if case .huggingFaceRepo(let repo) = entry.descriptor.source {
            MRFact(label: "Revision") {
                VStack(alignment: .leading, spacing: 1) {
                    DigestLabel(digest: repo.revision)
                    if !repo.isPinnedCommit {
                        Text("moving version — cannot be verified")
                            .font(MRType.prose)
                            .foregroundStyle(MRColor.caution)
                    }
                }
            }
        }
    }

    /// The last receipt this model produced on this machine, and nothing when
    /// it has produced none.
    @ViewBuilder private var lastRunFact: some View {
        if let record = model.history(for: modelID).first {
            MRFact(
                "Last run",
                String(format: "%.1f s per token", record.secondsPerToken)
                    + " · peak \(MRFormat.bytesDecimal(record.peakFootprintBytes))")
        }
    }

    /// What a token would cost on the volume this model is actually installed
    /// on, at the rate that volume measured. Shown only when both halves are
    /// true — installed somewhere, and that somewhere assessed. Without a
    /// measurement there is no line at all, which is the rule the dial's own
    /// projection keeps.
    @ViewBuilder private var assessedSpeedFact: some View {
        if let report = model.assessment(for: modelID),
            let verdict = model.assessedProjection(for: modelID),
            let projection = verdict.projection
        {
            MRFact(
                "Speed here",
                AssessmentFormat.range(projection)
                    + " on \(report.volumeName ?? "this volume")",
                provenance: .projected)
        }
    }

    /// The two things that make a measured rate worth doubting, and only when
    /// there is a rate on the page to doubt.
    @ViewBuilder private var assessmentWarnings: some View {
        if let report = model.assessment(for: modelID),
            let verdict = model.assessedProjection(for: modelID),
            verdict.projection != nil
        {
            if report.isStale() {
                MRInlineNote(
                    message: "That measurement is more than a week old — re-assess the volume.")
            }
            if let degraded = StorageAssessmentPresentation.fasterLinkSentence(report) {
                MRInlineNote(message: degraded)
            }
        }
    }

    /// The source details the artifact panel used to hide behind a chevron.
    /// Same facts, same disclosure, now in the page's own list.
    @ViewBuilder private func moreFacts(_ entry: CatalogEntry) -> some View {
        MRFact(
            "Payload",
            "\(MRFormat.publishedBytes(entry.descriptor.payloadBytes)) · "
                + publishedFileCount(entry.descriptor.payloadFileCount),
            provenance: .declared)
        MRFact(
            "Metadata",
            "\(MRFormat.publishedBytes(entry.descriptor.metadataBytes)) · "
                + publishedFileCount(entry.descriptor.metadataFileCount),
            provenance: .declared)
        MRFact(
            "Largest file", MRFormat.publishedBytes(entry.descriptor.largestFileBytes),
            provenance: .declared)
        if let layers = entry.index.layers {
            MRFact("Layers", "\(layers)", provenance: .declared)
        }
    }

    private func publishedFileCount(_ count: Int) -> String {
        count > 0 ? "\(MRFormat.grouped(count)) files" : "file count not published"
    }

    /// Two lines, each a dot and a phrase: whether this build can chat with the
    /// model, and what has been established about the bytes. They are separate
    /// facts and never collapse into one badge — a perfectly verified copy can
    /// still target an unsupported layout.
    ///
    /// They are the first two rows of the About list rather than a column of
    /// their own. Two short phrases do not need a third of the page's width,
    /// and spending it on them is what pushed the copies on the drive below the
    /// fold.
    @ViewBuilder private func readinessFacts(_ entry: CatalogEntry) -> some View {
        let fitness = model.effectiveProductFitness(for: entry)
        let availability = model.modelAvailability(for: entry.id)
        let compatibility = ModelCompatibilityPresentation.resolve(
            modelName: entry.descriptor.displayName,
            fitness: fitness, availability: availability)
        let integrity = ModelIntegrityPresentation.resolve(
            installations: model.installed.installations(of: entry.id),
            mounted: mountedInstallations(entry))
        let files = filesReadiness(integrity)
        MRFact(label: "Chat") {
            MRReadinessLine(tone: .from(compatibility.tone), phrase: compatibility.line)
        }
        MRFact(label: "Files") {
            MRReadinessLine(tone: files.tone, phrase: files.sentence)
        }
    }

    /// While a transfer is running there is nothing on the drive yet, and
    /// `ModelIntegrityPresentation` says so — *not on this device*, under a
    /// progress bar that is doing something about exactly that. The transfer is
    /// the answer to "what about the files", so while it runs it is what this
    /// line says.
    private func filesReadiness(
        _ integrity: ModelReadinessPresentationValue
    ) -> ModelPageStatus {
        let state = model.state(of: modelID)
        guard model.installed.installations(of: modelID).isEmpty, state.isBusy else {
            return ModelPageStatus(tone: .from(integrity.tone), sentence: integrity.line)
        }
        return ModelPageStatus(
            tone: .moving, sentence: "checked as each file lands")
    }

    // MARK: - Copies

    /// The mounted copy a run would read.
    private var preferredArtifact: DiscoveredArtifact? {
        model.installed.preferredArtifact(for: modelID)
    }

    /// The copy this page describes, mounted or not.
    ///
    /// `preferredArtifact` is deliberately mounted-only — it answers "what
    /// would a run open" — and a page that used it for its facts said
    /// *published size* about a model whose bytes it had measured, because the
    /// drive was in a drawer. What the last scan measured is still what is on
    /// that drive.
    private var describedArtifact: DiscoveredArtifact? {
        preferredArtifact ?? model.installed.installations(of: modelID)
            .max { $0.bytesOnDisk < $1.bytesOnDisk }
    }

    /// Whether the location holding this copy is plugged in right now.
    private func isMounted(_ artifact: DiscoveredArtifact) -> Bool {
        model.installed.report.locations
            .first { $0.rootPath == artifact.locationPath }?.isMounted ?? false
    }

    private func mountedInstallations(_ entry: CatalogEntry) -> [DiscoveredArtifact] {
        model.installed.report.mountedInstallations(of: entry.id)
    }

    @ViewBuilder private func copiesSection(_ entry: CatalogEntry) -> some View {
        let installations = model.installed.installations(of: entry.id)
        if !installations.isEmpty {
            MRPageSection(
                title: installations.count == 1 ? "This copy" : "Copies on this device"
            ) {
                ForEach(installations) { artifact in
                    copyRow(artifact, showsPath: installations.count > 1)
                    if artifact.rootPath != installations.last?.rootPath { MRHairline() }
                }
            }
        }
    }

    private func copyRow(_ artifact: DiscoveredArtifact, showsPath: Bool) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            MRQuietRow {
                VStack(alignment: .leading, spacing: MRSpace.s1) {
                    MRStatusLine(
                        tone: isMounted(artifact)
                            ? ModelPageStatusPresentation.tone(for: artifact) : .attention,
                        sentence: "\(locationName(for: artifact)) · "
                            + ModelArtifactPresentation.status(artifact)
                            + (isMounted(artifact) ? "" : " · not connected"))
                    // Only where there is more than one copy. With a single
                    // copy the facts above already carry its path and its last
                    // verification, and a row that repeats them is furniture.
                    if showsPath {
                        MRPathLabel(path: artifact.rootPath)
                        if let verifiedAt = artifact.verifiedAt {
                            Text("Last verified \(MRFormat.timestamp(verifiedAt))")
                                .font(MRType.prose)
                                .foregroundStyle(MRColor.tertiary)
                        }
                    }
                }
            } trailing: {
                ValueText(
                    text: MRFormat.measuredBytes(artifact.bytesOnDisk),
                    provenance: .measured, font: MRType.figure)
            }
            copyControls(artifact)
        }
    }

    @ViewBuilder private func copyControls(_ artifact: DiscoveredArtifact) -> some View {
        if model.installed.verificationInFlight == artifact.rootPath {
            VStack(alignment: .leading, spacing: MRSpace.s2) {
                if let progress = model.installed.verificationProgress {
                    MRProgressBar(fraction: progress.fraction)
                        .frame(maxWidth: 320)
                    Text(progress.currentPath)
                        .font(MRType.path)
                        .foregroundStyle(MRColor.tertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                } else {
                    Text("Starting verification…")
                        .font(MRType.prose)
                        .foregroundStyle(MRColor.secondary)
                }
                // A running pass needs a control that reads as one; a grey
                // caption-coloured link did not, and the owner could not find
                // how to stop a verification he had started by mistake.
                Button("Cancel verification") { model.installed.cancelVerification() }
                    .mrOutlineAction()
            }
        } else {
            HStack(spacing: MRSpace.s4) {
                Button("Verify all files") { pendingFullVerification = artifact }
                    .mrTextLink()
                    .help(
                        "Check every published digest. This reads about "
                            + MRFormat.measuredBytes(
                                artifact.expectedBytes ?? artifact.bytesOnDisk)
                            + " and writes nothing.")
                    .disabled(model.installed.verificationInFlight != nil)
                Button("Spot-check") {
                    Task { await model.installed.verify(artifact, .spotCheck()) }
                }
                .mrTextLink(.quiet)
                .help("Check a bounded sample of the largest files. It does not enable chat.")
                .disabled(model.installed.verificationInFlight != nil)
                #if os(macOS)
                    Button("Show in Finder") {
                        NSWorkspace.shared.selectFile(
                            nil, inFileViewerRootedAtPath: artifact.rootPath)
                    }
                    .mrTextLink(.quiet)
                #endif
                removalControl(artifact)
                Spacer(minLength: 0)
            }
        }
        if let outcome = model.installed.verificationOutcome[artifact.rootPath] {
            Text(outcome)
                .font(MRType.prose)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        if let outcome = model.localCopyRemovalMessage(for: artifact) {
            MRInlineNote(message: outcome, tone: .idle)
        }
    }

    @ViewBuilder private func removalControl(_ artifact: DiscoveredArtifact) -> some View {
        let refusal = model.localCopyRemovalRefusal(for: artifact)
        if model.localCopyRemovalInFlight?.rootPath == artifact.rootPath {
            Text("Removing this copy…")
                .font(MRType.prose)
                .foregroundStyle(MRColor.secondary)
        } else {
            Button("Remove local copy…") { pendingRemoval = artifact }
                .mrTextLink(.destructive)
                .disabled(refusal != nil)
                .help(refusal ?? "Permanently remove this exact local model directory.")
        }
    }

    private func locationName(for artifact: DiscoveredArtifact) -> String {
        model.installed.report.locations
            .first { $0.rootPath == artifact.locationPath }?.displayName
            ?? URL(fileURLWithPath: artifact.locationPath).lastPathComponent
    }

    // MARK: - Memory default

    /// The dial here states a **default**, not a run: this screen is about the
    /// model, and a run belongs to a conversation. The role label on the dial
    /// says which of the two it is, every time.
    @ViewBuilder private func budget(_ entry: CatalogEntry) -> some View {
        if model.capabilities(for: modelID) != nil {
            MRPageSection(title: "Memory for new chats") {
                if let plan = model.defaultBudgetPlan(for: modelID) {
                    MemoryDialView(
                        plan: plan,
                        entry: entry,
                        role: .defaultForNewChats,
                        onChange: { model.setDefaultBudget($0, for: modelID) })
                }
            }
        } else {
            MRPageSection(title: "Memory for new chats") {
                Text(
                    "No verified runner and tokenizer ship for \(entry.descriptor.displayName) "
                        + "in this build, so it is unavailable for chats and there is no run "
                        + "budget to state."
                )
                .font(MRType.prose)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - Receipts

    private func history(_ records: [RunRecord]) -> some View {
        MRPageSection(title: "Recent runs") {
            #if os(macOS)
                desktopHistoryTable(records)
            #else
                LazyVStack(spacing: 0) {
                    ForEach(records) { record in
                        compactHistoryCard(record)
                    }
                }
            #endif
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
        }
    #endif

    private func compactHistoryCard(_ record: RunRecord) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
                Text(MRFormat.timestamp(record.at))
                    .font(MRType.figure)
                    .foregroundStyle(MRColor.primary)
                Spacer(minLength: MRSpace.s2)
                Text(record.thermalState.rawValue)
                    .font(MRType.prose)
                    .foregroundStyle(MRColor.secondary)
            }
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: MRSpace.s5) {
                    historyMetric(
                        label: "Budget",
                        value: MRFormat.bytesDecimal(record.declaredBudgetBytes))
                    historyMetric(
                        label: "Time / token",
                        value: String(format: "%.1f s", record.secondsPerToken))
                    historyMetric(
                        label: "Peak",
                        value: MRFormat.bytesDecimal(record.peakFootprintBytes),
                        color: record.budgetRespected ? MRColor.primary : MRColor.refuse)
                }
                VStack(alignment: .leading, spacing: MRSpace.s2) {
                    historyMetric(
                        label: "Budget",
                        value: MRFormat.bytesDecimal(record.declaredBudgetBytes))
                    historyMetric(
                        label: "Time / token",
                        value: String(format: "%.1f s", record.secondsPerToken))
                    historyMetric(
                        label: "Peak",
                        value: MRFormat.bytesDecimal(record.peakFootprintBytes),
                        color: record.budgetRespected ? MRColor.primary : MRColor.refuse)
                }
            }
            DigestLabel(digest: record.logitsDigest, label: "logits", showsCopyButton: false)
        }
        .padding(.vertical, MRSpace.s3)
        .overlay(alignment: .top) { MRHairline() }
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
            Text(label)
                .font(MRType.prose)
                .foregroundStyle(MRColor.tertiary)
            Text(value)
                .font(MRType.figure)
                .foregroundStyle(color)
        }
    }

    #if os(macOS)
        private var historyHeader: some View {
            HStack {
                Text("Date").frame(width: 110, alignment: .leading)
                Text("Budget").frame(width: 80, alignment: .trailing)
                Text("s/token").frame(width: 70, alignment: .trailing)
                Text("Peak").frame(width: 80, alignment: .trailing)
                Text("Thermal").frame(width: 70, alignment: .leading)
                Text("Logits")
                Spacer()
            }
            .font(MRType.prose)
            .foregroundStyle(MRColor.tertiary)
            .padding(.vertical, MRSpace.s2)
        }

        private func historyRow(_ record: RunRecord) -> some View {
            HStack {
                Text(MRFormat.timestamp(record.at))
                    .frame(width: 110, alignment: .leading)
                Text(MRFormat.bytesDecimal(record.declaredBudgetBytes))
                    .frame(width: 80, alignment: .trailing)
                Text(String(format: "%.1f", record.secondsPerToken))
                    .frame(width: 70, alignment: .trailing)
                Text(MRFormat.bytesDecimal(record.peakFootprintBytes))
                    .foregroundStyle(record.budgetRespected ? MRColor.primary : MRColor.refuse)
                    .frame(width: 80, alignment: .trailing)
                Text(record.thermalState.rawValue)
                    .frame(width: 70, alignment: .leading)
                DigestLabel(digest: record.logitsDigest, showsCopyButton: false)
                Spacer()
            }
            .font(MRType.figure)
            .foregroundStyle(MRColor.secondary)
            .padding(.vertical, MRSpace.s2)
            .overlay(alignment: .top) { MRHairline() }
            .accessibilityElement(children: .combine)
            .accessibilityLabel(
                "run at \(MRFormat.timestamp(record.at)), budget "
                    + "\(MRFormat.bytesDecimal(record.declaredBudgetBytes)), "
                    + "\(Int(record.secondsPerToken)) seconds per token, peak "
                    + "\(MRFormat.bytesDecimal(record.peakFootprintBytes)), thermal "
                    + "\(record.thermalState.rawValue)")
        }
    #endif

    // MARK: - Earlier transfers

    /// Every other record this model has, quietly, with the timestamp that
    /// says which attempt it was and a Forget for the ones that have stopped.
    /// Never the record the transfer block above is already showing.
    @ViewBuilder private var earlierTransfers: some View {
        let others = model.downloadJobs(for: modelID)
            .filter { $0.jobID != nil && $0.jobID != download?.jobID }
        if !others.isEmpty {
            MRPageSection(title: "Earlier transfers") {
                ForEach(others, id: \.jobID) { controller in
                    MRQuietRow {
                        VStack(alignment: .leading, spacing: 1) {
                            Text(controller.state.phrase.capitalizedSentence)
                                .font(MRType.prose)
                                .foregroundStyle(MRColor.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                            if let path = controller.destinationPath {
                                MRPathLabel(path: path)
                            }
                        }
                    } trailing: {
                        Text(MRFormat.timestamp(controller.updatedAt))
                            .font(MRType.figure)
                            .foregroundStyle(MRColor.tertiary)
                        if let job = controller.jobID, model.canForgetDownloadJob(job) {
                            Button(ForgetTransferConfirmation.rowActionTitle) {
                                pendingForget = job
                            }
                            .mrTextLink(.quiet)
                        }
                    }
                    if controller.jobID != others.last?.jobID { MRHairline() }
                }
            }
        }
    }
}

extension String {
    /// A state phrase written as a sentence: the phrases are lowercase because
    /// they were built for a chip, and this page has no chips.
    fileprivate var capitalizedSentence: String {
        guard let first = first else { return self }
        return String(first).uppercased() + dropFirst()
    }
}

// MARK: - Presentation

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

/// The one line under the model's name: what it is for, how it is built, and
/// whose licence it carries.
///
/// Every term comes from the descriptor. A parameter count would belong here
/// and is not published in any field this app has, so the line says what is
/// known instead of rounding a number nobody measured.
enum ModelPurposePresentation {
    static func line(for descriptor: ModelDescriptor) -> String? {
        let terms = [
            purpose(descriptor.architecture),
            family(descriptor.architecture),
            ModelLicensePresentation.shortName(descriptor.licenseName),
        ]
        .compactMap { $0 }
        return terms.isEmpty ? nil : terms.joined(separator: " · ")
    }

    static func purpose(_ architecture: ModelArchitecture) -> String? {
        switch architecture {
        case .kimiK3MoE, .deepseekV4FlashMoE, .deepseekV41FlashMoE, .qwen3MoE:
            return "Text chat"
        case .minimaxH3Video:
            return "Video generation"
        case .unrecognized:
            return nil
        }
    }

    static func family(_ architecture: ModelArchitecture) -> String? {
        switch architecture {
        case .kimiK3MoE, .deepseekV4FlashMoE, .deepseekV41FlashMoE, .qwen3MoE:
            return "Mixture of experts"
        case .minimaxH3Video:
            return "Diffusion transformer"
        case .unrecognized:
            return nil
        }
    }
}

/// A licence in a header line is an identifier, not the licence text. The
/// catalog's own tag for K3 is a whole sentence — *other (repository tag
/// license:other; see the repository LICENSE)* — and the full string still
/// appears in the facts below, where there is room for it.
enum ModelLicensePresentation {
    static func shortName(_ license: String) -> String? {
        let head = license.prefix { $0 != "(" && $0 != ";" && $0 != "," }
            .trimmingCharacters(in: .whitespaces)
        return head.isEmpty ? nil : head
    }
}

/// What the published layout says about how these weights are stored.
///
/// This is read off the declared `ArtifactLayout` and nothing else — it is the
/// repository's own file shape, not an inspection of the bytes — so it is
/// rendered in the declared face. Layouts whose tile formats do not name a
/// numeric width state nothing rather than guessing one.
enum ArtifactPrecisionPresentation {
    static func sentence(for layout: ArtifactLayout) -> String? {
        switch layout {
        case .k3FlagshipLayerStreams:
            return "FP4 experts · BF16 dense layers"
        case .v4FlashUnitBundle:
            return "FP4 experts · FP8 matrices"
        case .v41FlashUnitBundle:
            return "FP4 experts · FP8 matrices · FP8 memory tables"
        case .h3UnitBundle, .qwenContainerSet, .unrecognized:
            return nil
        }
    }
}

/// The repository's page, for a person rather than for the API client.
enum ModelSourceLink {
    static func url(for repo: HuggingFaceRepoRef) -> URL {
        HuggingFaceEndpoint.default
            .appendingPathComponent(repo.repoID)
            .appendingPathComponent("tree")
            .appendingPathComponent(repo.revision)
    }
}

/// One quiet line beside the big number: how fast, and which file.
///
/// A stopped transfer never reaches here. It has no rate and no estimate, and
/// the line it would print would be a memory of a drive that has since changed.
enum TransferPacePresentation {
    static func line(_ progress: ProgressSnapshot, isMoving: Bool) -> String? {
        var terms: [String] = []
        if isMoving, progress.bytesPerSecond > 0 {
            terms.append(MRFormat.throughput(progress.bytesPerSecond))
        }
        if progress.filesTotal > 0 {
            terms.append(
                "file \(min(progress.filesDone + 1, progress.filesTotal)) "
                    + "of \(progress.filesTotal)")
        }
        // The estimate is the status sentence's, under the name; printing it
        // twice on one screen would make the page argue with itself.
        return terms.isEmpty ? nil : terms.joined(separator: " · ")
    }
}

/// The page's one status sentence, and the colour of the dot beside it.
///
/// The order of authority is the product's existing one and the reason this is
/// a value rather than a view: **the disk speaks last**. A transfer record that
/// says `ready` cannot claim a copy the scan has not seen, and a scan that
/// found nothing cannot silence a transfer that is running right now.
struct ModelPageStatus: Equatable {
    let tone: MRPageStatusTone
    let sentence: String
}

enum ModelPageStatusPresentation {
    static func resolve(
        state: DownloadState,
        remains: TransferRemains,
        installations: [DiscoveredArtifact],
        mounted: [DiscoveredArtifact],
        transferDestinationName: String?,
        copyLocationName: String?
    ) -> ModelPageStatus {
        // A transfer in motion, or one that stopped mid-flight, is what the
        // page is about while it lasts. A finished or cancelled one speaks only
        // while the scan has found nothing: after that the drive outranks it.
        switch state {
        case .notStarted, .awaitingDestination:
            break
        case .cancelled, .ready:
            if installations.isEmpty,
                let transfer = transferSentence(
                    state: state, remains: remains,
                    destinationName: transferDestinationName)
            {
                return transfer
            }
        default:
            if let transfer = transferSentence(
                state: state, remains: remains, destinationName: transferDestinationName)
            {
                return transfer
            }
        }

        // Then the drive.
        guard !installations.isEmpty else {
            return ModelPageStatus(
                tone: .idle,
                sentence: "Not on this device. Download a copy, or add the folder that "
                    + "already contains it.")
        }
        guard !mounted.isEmpty else {
            let drive = copyLocationName.map { "\($0) is not connected." }
                ?? "The drive holding this copy is not connected."
            return ModelPageStatus(
                tone: .attention,
                sentence: drive + " Nothing needs to be downloaded again.")
        }
        let place = copyLocationName ?? "this device"
        let complete = mounted.filter { $0.isComplete != false }
        guard !complete.isEmpty else {
            return ModelPageStatus(
                tone: .attention,
                sentence: "On \(place) · files published by the repository are missing.")
        }
        if complete.contains(where: { $0.verification == .fullyVerified }) {
            return ModelPageStatus(
                tone: .ready,
                sentence: "Ready on \(place) · every file matches its published digest.")
        }
        if complete.contains(where: { $0.verification == .spotChecked }) {
            return ModelPageStatus(
                tone: .attention,
                sentence: "On \(place) · a sample matched. Verify all files before using "
                    + "this copy in a chat.")
        }
        return ModelPageStatus(
            tone: .attention,
            sentence: "On \(place) · verify all files before using this copy in a chat.")
    }

    /// What a transfer record alone has to say, with no drive consulted.
    ///
    /// Nil for the two states that are not a transfer at all — nothing has been
    /// started, or a destination has not been chosen — because those are
    /// questions only the drive can answer. Every screen that shows a transfer
    /// says it in these words: the model page above, and each row of the
    /// Downloads list. One sentence per state, written once.
    static func transferSentence(
        state: DownloadState, remains: TransferRemains, destinationName: String?
    ) -> ModelPageStatus? {
        switch state {
        case .resolvingIndex:
            return ModelPageStatus(
                tone: .moving,
                sentence: "Preparing the transfer — resolving the index and walking the tree.")
        case .active(let progress):
            return ModelPageStatus(
                tone: .moving, sentence: downloading(progress, to: destinationName))
        case .paused:
            return ModelPageStatus(
                tone: .attention,
                sentence: to("Paused", destinationName) + ". Resume to continue.")
        case .verifying(let phase, _):
            if case .checking(let checked, let total, _, _) = phase {
                return ModelPageStatus(
                    tone: .moving,
                    sentence: "Checking the files on \(destinationName ?? "the drive") against the published digests · \(MRFormat.grouped(checked)) of \(MRFormat.grouped(total))")
            }
            return ModelPageStatus(
                tone: .moving,
                sentence: "Checking every downloaded file against the published digests.")
        case .interrupted(_, let reason):
            return ModelPageStatus(tone: .attention, sentence: "Interrupted — \(reason.sentence)")
        case .incomplete(let files, _):
            return ModelPageStatus(
                tone: .attention,
                sentence: "\(files) file\(files == 1 ? "" : "s") did not verify. "
                    + "Those bytes will be fetched again.")
        case .failed(let error):
            return ModelPageStatus(
                tone: .attention, sentence: "The transfer failed — \(error.description)")
        case .cancelled:
            return ModelPageStatus(
                tone: .idle,
                sentence: "Cancelled. " + TransferRemainsPresentation.sentence(remains))
        case .ready:
            return ModelPageStatus(
                tone: .ready,
                sentence: "Downloaded and checked. Every planned file matched its "
                    + "published digest.")
        case .notStarted, .awaitingDestination:
            return nil
        }
    }

    /// The dot beside one discovered copy, from the same rules the page-wide
    /// sentence uses.
    static func tone(for artifact: DiscoveredArtifact) -> MRPageStatusTone {
        if artifact.isComplete == false { return .attention }
        switch artifact.verification {
        case .unverified, .spotChecked: return .attention
        case .fullyVerified: return .ready
        }
    }

    private static func downloading(
        _ progress: ProgressSnapshot, to destination: String?
    ) -> String {
        var sentence = to("Downloading", destination)
        if let remaining = progress.estimatedTimeRemaining, remaining > 0 {
            sentence += " · about \(MRFormat.duration(remaining)) left"
        }
        return sentence
    }

    private static func to(_ verb: String, _ destination: String?) -> String {
        guard let destination, !destination.isEmpty else { return verb }
        return "\(verb) to \(destination)"
    }
}

/// The two model-detail readiness items share a small value type but never
/// share their authority: compatibility comes from the linked product runtime
/// and device, while integrity comes only from discovered local bytes.
struct ModelReadinessPresentationValue: Equatable {
    let status: String
    let reason: String
    let tone: StatusChip.Tone
    /// The same fact in a phrase that fits on one line beside its dot, for the
    /// readiness rows in the About list. It states the state in words rather
    /// than repeating the model's name or explaining what to do — the sentence
    /// that does that is ``reason``, and it is what the notes and the
    /// conversation banner still use.
    let line: String

    init(status: String, reason: String, tone: StatusChip.Tone, line: String) {
        self.status = status
        self.reason = reason
        self.tone = tone
        self.line = line
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.status == rhs.status && lhs.reason == rhs.reason && lhs.tone == rhs.tone
            && lhs.line == rhs.line
    }
}

/// Whether this App and device can use the model family. File integrity is a
/// separate item: a perfectly verified copy can still target an unsupported
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
                tone: .neutral, line: "not in this version")
        case .refused:
            return ModelReadinessPresentationValue(
                status: "Not supported on this device",
                reason: fitness.reason, tone: .refuse, line: "not on this device")
        case .runnable, .runnableWithCaveats:
            break
        }

        switch availability {
        case .runtimeFilesUnavailable:
            return ModelReadinessPresentationValue(
                status: "Local version not supported",
                reason: "This version supports \(modelName), but this local copy needs an update.",
                tone: .caution, line: "this copy needs an update")
        case .runtimeUnavailable:
            return ModelReadinessPresentationValue(
                status: "Not supported by this version",
                reason: "This version can manage \(modelName), but cannot use it in a chat.",
                tone: .neutral, line: "not in this version")
        case .missingFromCatalog:
            return ModelReadinessPresentationValue(
                status: "Catalog entry unavailable",
                reason: "Compatibility cannot be confirmed without current model information.",
                tone: .neutral, line: "unknown without model information")
        case .unrecognizedArtifact:
            return ModelReadinessPresentationValue(
                status: "Format not supported",
                reason: "This version does not recognize the local artifact format.",
                tone: .neutral, line: "format not supported")
        case .available, .notFoundByScan, .storageUnavailable, .incompleteArtifact,
            .unverifiedArtifact:
            return ModelReadinessPresentationValue(
                status: fitness.verdict == .runnable ? "Supported" : "Supported with limits",
                reason: fitness.verdict == .runnable
                    ? "This version supports \(modelName) on this device."
                    : "This version supports \(modelName), with the device limit shown below.",
                tone: fitness.verdict == .runnable ? .ok : .caution,
                line: fitness.verdict == .runnable ? "supported here" : "supported, with limits")
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
                tone: .neutral, line: "not on this device")
        }
        guard !mounted.isEmpty else {
            return ModelReadinessPresentationValue(
                status: "Storage unavailable",
                reason: "Reconnect the storage that contains this model.",
                tone: .caution, line: "drive not connected")
        }
        let complete = mounted.filter { $0.isComplete != false }
        guard !complete.isEmpty else {
            return ModelReadinessPresentationValue(
                status: "Files incomplete",
                reason: "This local copy is missing files published by the model repository.",
                tone: .caution, line: "files missing")
        }
        if complete.contains(where: { $0.verification == .fullyVerified }) {
            return ModelReadinessPresentationValue(
                status: "Fully verified",
                reason: "Every published model file matched its recorded digest.",
                tone: .ok, line: "fully verified")
        }
        if complete.contains(where: { $0.verification == .spotChecked }) {
            return ModelReadinessPresentationValue(
                status: "Spot-check only",
                reason: "A sample matched. Verify all files before using this copy in a chat.",
                tone: .verify, line: "spot-check only")
        }
        return ModelReadinessPresentationValue(
            status: "Full verification required",
            reason: "Verify all files before using this local copy in a chat.",
            tone: .verify, line: "not verified")
    }
}

/// The three facts that make a previous look at the drive stale: the transfer
/// stopped or resumed, it points somewhere else, or the scanner has been over
/// the volumes again.
private struct KeptFilesReconciliationKey: Equatable {
    let state: String
    let destinationPath: String?
    let scannedAt: Date
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
