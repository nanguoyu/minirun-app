import MinirunKit
import SwiftUI

/// Every durable download job on this device, regardless of whether its model
/// currently appears in the storefront. Long-running work belongs to one
/// stable control surface; a catalog refresh may rename it, but may not hide it.
///
/// One transfer per block, in the product-page language: the publisher's mark
/// and the model's name, the sentence that says where the transfer stands, and
/// then the transfer itself — the number, the bar, the pace, the destination.
struct DownloadsView: View {
    @Environment(AppModel.self) private var model
    @State private var destinationRequest: TransferDestinationRequest?
    @State private var pendingForget: DownloadJobID?
    @State private var forgetWarning: String?

    var body: some View {
        Group {
            if showsQuietEmptyState {
                downloadsEmptyState
                    .padding(pagePadding)
            } else {
                ScrollView {
                    DownloadsPage(
                        destinationRequest: $destinationRequest,
                        pendingForget: $pendingForget,
                        forgetWarning: $forgetWarning)
                }
            }
        }
        .mrProductPage()
        .mrPhoneNavigationTitle("Downloads")
        .sheet(item: $destinationRequest) { request in
            DestinationPicker(
                modelID: request.model,
                preferredStartingDirectory: request.destination,
                onStarted: { destinationRequest = nil })
        }
        #if os(iOS)
            .alert(
                ForgetTransferConfirmation.title,
                isPresented: forgetConfirmationPresented,
                presenting: pendingForget
            ) { job in
                forgetConfirmationActions(job)
            } message: { job in
                Text(ForgetTransferConfirmation.message(destinationPath: destinationPath(job)))
            }
        #else
            .confirmationDialog(
                ForgetTransferConfirmation.title,
                isPresented: forgetConfirmationPresented,
                titleVisibility: .visible,
                presenting: pendingForget
            ) { job in
                forgetConfirmationActions(job)
            } message: { job in
                Text(ForgetTransferConfirmation.message(destinationPath: destinationPath(job)))
            }
        #endif
    }

    private var forgetConfirmationPresented: Binding<Bool> {
        Binding(get: { pendingForget != nil }, set: { if !$0 { pendingForget = nil } })
    }

    @ViewBuilder private func forgetConfirmationActions(_ job: DownloadJobID) -> some View {
        Button(ForgetTransferConfirmation.actionTitle, role: .destructive) {
            pendingForget = nil
            forgetWarning = model.forgetDownloadJob(job)
        }
        Button("Cancel", role: .cancel) { pendingForget = nil }
    }

    private func destinationPath(_ job: DownloadJobID) -> String? {
        model.downloadController(for: job)?.destinationPath
    }

    private var showsQuietEmptyState: Bool {
        model.transfersForPresentation.isEmpty
            && !model.isRecoveringDownloads
            && model.downloadSetupError == nil
            && model.downloadServiceError == nil
            && model.downloadRecoveryIssues.isEmpty
            && !model.canRetryDownloadRecovery
    }

    private var pagePadding: CGFloat {
        #if os(macOS)
            MRSpace.s5
        #else
            MRSpace.s4
        #endif
    }

    private var downloadsEmptyState: some View {
        VStack(spacing: MRSpace.s3) {
            Image(systemName: "arrow.down.circle")
                .font(.system(size: 30, weight: .regular))
                .foregroundStyle(MRColor.secondary)
                .accessibilityHidden(true)
            Text("No downloads")
                .font(MRType.pageTitle)
                .tracking(-0.5)
                .foregroundStyle(MRColor.primary)
            Text(
                "Choose a model when you want another local copy. Active and saved transfers "
                    + "stay here across launches."
            )
            .font(MRType.pageSubtitle)
            .foregroundStyle(MRColor.secondary)
            .multilineTextAlignment(.center)
            .fixedSize(horizontal: false, vertical: true)
            NavigationLink("Browse models") {
                ModelCatalogView()
            }
            .buttonStyle(.plain)
            .foregroundStyle(MRColor.accent)
            .font(MRType.control)
            .frame(minHeight: MRPageControl.height)
        }
        .frame(maxWidth: 480, maxHeight: .infinity)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// The transfers themselves, without the scroll view around them.
///
/// Separate for the reason `ModelDetailPage` is separate: `ImageRenderer` draws
/// nothing at all for a macOS `ScrollView`, and it holds no confirmation of its
/// own, so it is a pure function of the app model plus the pending values the
/// screen presents dialogs for.
struct DownloadsPage: View {
    @Binding var destinationRequest: TransferDestinationRequest?
    @Binding var pendingForget: DownloadJobID?
    @Binding var forgetWarning: String?

    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if model.isRecoveringDownloads {
                HStack(spacing: MRSpace.s2) {
                    ProgressView().controlSize(.small)
                    Text("Restoring saved downloads…")
                        .font(MRType.prose)
                        .foregroundStyle(MRColor.secondary)
                }
                .padding(.bottom, MRSpace.s4)
            }

            serviceMessages

            ForEach(model.transfersForPresentation) { transfer in
                transferBlock(transfer)
            }
        }
        .padding(pagePadding)
        .frame(maxWidth: 820, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .top)
    }

    private var pagePadding: CGFloat {
        #if os(macOS)
            MRSpace.s5
        #else
            MRSpace.s4
        #endif
    }

    @ViewBuilder private var serviceMessages: some View {
        if let error = model.downloadSetupError {
            NamedErrorCard(
                headline: "Downloads are unavailable.",
                message: "The durable download store could not be opened.",
                namedError: error)
            .padding(.bottom, MRSpace.s4)
        }
        if let error = model.downloadServiceError {
            NamedErrorCard(
                headline: "Saved downloads need attention.",
                message: "No transfer was resumed automatically.",
                namedError: error,
                tone: .caution)
            .padding(.bottom, MRSpace.s4)
        }
        ForEach(model.downloadRecoveryIssues) { issue in
            NamedErrorCard(
                headline: "A saved download could not be restored.",
                message: issue.destinationPath.isEmpty
                    ? "The job remains stopped."
                    : "The job at \(issue.destinationPath) remains stopped.",
                namedError: issue.reason,
                tone: .caution)
            .padding(.bottom, MRSpace.s4)
        }
        if model.canRetryDownloadRecovery {
            Button("Retry restoring downloads") {
                Task { await model.retryDownloadRecovery() }
            }
            .mrOutlineAction()
            .disabled(model.isRecoveringDownloads)
            .padding(.bottom, MRSpace.s4)
        }
        if let forgetWarning {
            MRInlineNote(message: forgetWarning)
                .padding(.bottom, MRSpace.s4)
        }
    }

    // MARK: - One transfer

    private func transferBlock(_ transfer: TransferPresentation) -> some View {
        let controller = model.downloadController(for: transfer.job)
        let remains = TransferRemains.resolve(
            controller?.keptFiles, destinationPath: controller?.destinationPath)
        let status = DownloadsRowPresentation.status(
            state: transfer.state, remains: remains,
            destinationName: controller?.volumeName)
        return MRPageSection {
            header(transfer, status: status)
            block(transfer, remains: remains)
            if let reason = controller?.operationError {
                NamedErrorCard(
                    headline: "The last action did not finish.",
                    message: "The transfer state was kept unchanged.",
                    namedError: reason,
                    tone: .caution)
            }
            if let reason = transfer.reason {
                MRInlineNote(message: reason)
                refreshModelsButton
            }
            actions(transfer, controller: controller, remains: remains)
        }
        // A stopped transfer states what is on the drive, so the drive is
        // re-read whenever this list appears or a transfer's state changes.
        .task(id: transfer.state.phrase) {
            await controller?.reconcileKeptFiles()
        }
    }

    private func header(_ transfer: TransferPresentation, status: ModelPageStatus) -> some View {
        HStack(alignment: .top, spacing: MRSpace.s3) {
            ModelPublisherMark(
                identity: .resolve(
                    modelID: transfer.model,
                    displayName: transfer.displayName,
                    repositoryID: model.entry(transfer.model)?
                        .descriptor.source.repo?.repoID),
                fallbackSystemName: "shippingbox",
                fallbackColor: MRColor.tertiary,
                size: 32)
            VStack(alignment: .leading, spacing: MRSpace.s1) {
                Text(transfer.displayName)
                    .font(MRType.headline)
                    .foregroundStyle(MRColor.primary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityAddTraits(.isHeader)
                MRStatusLine(tone: status.tone, sentence: status.sentence)
            }
            Spacer(minLength: 0)
            Text(MRFormat.timestamp(transfer.updatedAt))
                .font(MRType.figure)
                .foregroundStyle(MRColor.tertiary)
        }
    }

    @ViewBuilder private func block(
        _ transfer: TransferPresentation, remains: TransferRemains
    ) -> some View {
        let controller = model.downloadController(for: transfer.job)
        let path = controller?.destinationPath ?? transfer.destinationPath
        switch transfer.state {
        case .active(let progress), .paused(let progress):
            MRTransferBlock(
                headline: MRFormat.bytesDecimal(progress.verifiedBytes),
                headlineDetail: "of \(MRFormat.bytesDecimal(progress.totalBytes))",
                meta: TransferPacePresentation.line(
                    progress, isMoving: transfer.state.isBusy),
                fraction: progress.fraction,
                path: path
            ) {
                links(transfer)
            }
        case .verifying(let phase, let progress):
            MRTransferBlock(
                headline: MRFormat.bytesDecimal(phase.bytesChecked ?? progress.totalBytes),
                headlineDetail: phase.bytesChecked == nil
                    ? "to check" : "of \(MRFormat.bytesDecimal(progress.totalBytes)) checked",
                meta: phase.filesLine,
                fraction: phase.fraction,
                sentence: phase.sentence,
                path: path
            ) {
                links(transfer)
            }
        case .resolvingIndex:
            MRTransferBlock(
                sentence: "Resolving the index and walking the tree…", path: path
            ) {
                links(transfer)
            }
        case .ready(let measured, let fileCount):
            MRTransferBlock(
                headline: MRFormat.bytesDecimal(measured),
                headlineDetail: "on the drive",
                meta: "\(MRFormat.grouped(fileCount)) planned files checked",
                fraction: 1,
                path: path
            ) {
                links(transfer)
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
        case .failed(let error):
            NamedErrorCard(
                headline: "The transfer failed.",
                message: "What it left on the drive is stated below.",
                namedError: error.description)
            stoppedBlock(transfer, remains: remains, path: path)
        case .interrupted(_, let reason):
            MRInlineNote(
                message: reason.recoverySentence,
                title: "Interrupted — \(reason.sentence)")
            stoppedBlock(transfer, remains: remains, path: path)
        case .cancelled:
            stoppedBlock(transfer, remains: remains, path: path)
        case .notStarted, .awaitingDestination:
            MRTransferBlock(
                headline: MRFormat.publishedBytes(transfer.declaredBytes),
                headlineDetail: "to download",
                sentence: "Nothing has been transferred yet.",
                path: path
            ) {
                links(transfer)
            }
        }
    }

    /// A stopped transfer, drawn entirely from a re-check of the destination.
    /// No rate, no estimate and no resume offset: a cancelled job has no speed,
    /// and the offset it stopped at is not a fact about the drive.
    private func stoppedBlock(
        _ transfer: TransferPresentation, remains: TransferRemains, path: String?
    ) -> some View {
        let snapshot = transfer.state.progress
        let totalBytes = snapshot?.totalBytes ?? transfer.declaredBytes
        let kept = DownloadsRowPresentation.keptHeadline(remains)
        return MRTransferBlock(
            headline: kept,
            headlineDetail: kept == nil ? nil : "of \(MRFormat.bytesDecimal(totalBytes))",
            meta: TransferRemainsPresentation.progressLine(remains, snapshot: snapshot),
            fraction: kept == nil
                ? nil
                : TransferRemainsPresentation.keptFraction(remains, totalBytes: totalBytes),
            sentence: DownloadsRowPresentation.keptSentence(
                state: transfer.state, remains: remains),
            path: path,
            accessibilitySummary: TransferRemainsPresentation.sentence(remains)
        ) {
            links(transfer)
        }
    }

    @ViewBuilder private func links(_ transfer: TransferPresentation) -> some View {
        if transfer.hasCatalogEntry {
            NavigationLink {
                DownloadDetailView(modelID: transfer.model)
            } label: {
                Text("Transfer details")
            }
            .mrTextLink()

            NavigationLink {
                ModelDetailView(modelID: transfer.model)
            } label: {
                Text("Open model")
            }
            .mrTextLink()
        }
    }

    // MARK: - Actions

    @ViewBuilder private func actions(
        _ transfer: TransferPresentation, controller: DownloadController?,
        remains: TransferRemains
    ) -> some View {
        HStack(spacing: MRSpace.s2) {
            switch transfer.state {
            case .active:
                if let controller {
                    Button("Pause") { Task { await controller.pause() } }.mrOutlineAction()
                    Button("Cancel") { Task { await controller.cancel() } }.mrOutlineAction()
                }
            case .paused:
                if let controller {
                    Button("Resume") { Task { await controller.resume() } }.mrFilledAction()
                    Button("Cancel") { Task { await controller.cancel() } }.mrOutlineAction()
                }
            case .verifying:
                if let controller {
                    Button("Cancel verification") { controller.cancelVerification() }
                        .mrOutlineAction()
                }
            case .interrupted, .cancelled, .failed:
                startAction(transfer, controller: controller, remains: remains)
                if model.canForgetDownloadJob(transfer.job) {
                    Button(ForgetTransferConfirmation.cardActionTitle) {
                        pendingForget = transfer.job
                    }
                    .mrTextLink(.quiet)
                }
            case .notStarted, .awaitingDestination, .resolvingIndex, .ready, .incomplete:
                EmptyView()
            }
            Spacer(minLength: 0)
        }
    }

    @ViewBuilder private func startAction(
        _ transfer: TransferPresentation, controller: DownloadController?,
        remains: TransferRemains
    ) -> some View {
        if transfer.hasCatalogEntry, controller != nil {
            Button(TransferRemainsPresentation.actionTitle(remains)) {
                destinationRequest = TransferDestinationRequest(
                    job: transfer.job,
                    model: transfer.model,
                    destination: URL(
                        fileURLWithPath: transfer.destinationPath, isDirectory: true))
            }
            .mrFilledAction()
            .disabled(!TransferRemainsPresentation.actionIsEnabled(remains))
        }
    }

    @ViewBuilder private var refreshModelsButton: some View {
        if model.allowsLiveCatalogRefresh {
            Button("Refresh models") {
                Task { await model.refreshPublishedModels() }
            }
            .mrOutlineAction()
            .disabled(model.isRefreshingPublishedModels)
        }
    }
}

/// What one row of the Downloads list is allowed to say. Separated from the
/// view because the sentence under a transfer's name is the same claim the
/// model page makes about the same job, and two screens that compose it
/// separately are two screens that can disagree.
enum DownloadsRowPresentation {
    /// The sentence beside the dot. A transfer, and only a transfer: this
    /// screen is a list of jobs, so the drive does not get to speak over one
    /// here the way it does on the model page — that page is one tap away and
    /// says what the scan found.
    static func status(
        state: DownloadState, remains: TransferRemains, destinationName: String?
    ) -> ModelPageStatus {
        ModelPageStatusPresentation.transferSentence(
            state: state, remains: remains, destinationName: destinationName)
            ?? ModelPageStatus(tone: .idle, sentence: "Nothing has been transferred yet.")
    }

    /// The kept bytes, and only when a look at the drive found some. Nothing
    /// kept gets no big number at all: `0 GB` is a number, and this block's
    /// number is reserved for something that is there.
    static func keptHeadline(_ remains: TransferRemains) -> String? {
        guard case .kept(_, _, let bytes) = remains else { return nil }
        return MRFormat.bytesDecimal(bytes)
    }

    /// What the reconciliation found, under the numbers — except after a
    /// cancellation, where the sentence beside the dot is already that exact
    /// sentence and printing it twice would make the row argue with itself.
    static func keptSentence(state: DownloadState, remains: TransferRemains) -> String? {
        guard state.hasStopped else { return nil }
        if case .cancelled = state { return nil }
        return TransferRemainsPresentation.sentence(remains)
    }
}

struct TransferDestinationRequest: Identifiable {
    var id: DownloadJobID { job }
    let job: DownloadJobID
    let model: ModelID
    let destination: URL
}
