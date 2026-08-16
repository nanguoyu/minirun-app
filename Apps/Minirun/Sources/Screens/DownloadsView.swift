import MinirunKit
import SwiftUI

/// Every durable download job on this device, regardless of whether its model
/// currently appears in the storefront. Long-running work belongs to one
/// stable control surface; a catalog refresh may rename it, but may not hide it.
struct DownloadsView: View {
    @Environment(AppModel.self) private var model
    @State private var destinationRequest: TransferDestinationRequest?

    var body: some View {
        Group {
            if showsQuietEmptyState {
                downloadsEmptyState
                    .padding(pagePadding)
            } else {
                transferList
            }
        }
        .mrWorkspaceSurface()
        .mrPhoneNavigationTitle("Downloads")
        .sheet(item: $destinationRequest) { request in
            DestinationPicker(
                modelID: request.model,
                preferredStartingDirectory: request.destination)
        }
    }

    private var showsQuietEmptyState: Bool {
        model.transfersForPresentation.isEmpty
            && !model.isRecoveringDownloads
            && model.downloadSetupError == nil
            && model.downloadServiceError == nil
            && model.downloadRecoveryIssues.isEmpty
            && !model.canRetryDownloadRecovery
    }

    private var transferList: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MRSpace.s4) {
                if model.isRecoveringDownloads {
                    HStack(spacing: MRSpace.s2) {
                        ProgressView().controlSize(.small)
                        Text("Restoring saved downloads…")
                            .font(MRType.caption)
                            .foregroundStyle(MRColor.secondary)
                    }
                }

                serviceMessages

                ForEach(model.transfersForPresentation) { transfer in
                    transferCard(transfer)
                }
            }
            .padding(pagePadding)
            .frame(maxWidth: 820, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .top)
        }
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
                .font(MRType.title)
                .foregroundStyle(MRColor.primary)
            Text(
                "Choose a model when you want another local copy. Active and saved transfers "
                    + "stay here across launches."
            )
            .font(MRType.body)
            .foregroundStyle(MRColor.secondary)
            .multilineTextAlignment(.center)
            .fixedSize(horizontal: false, vertical: true)
            NavigationLink("Browse models") {
                ModelCatalogView()
            }
            .buttonStyle(.borderedProminent)
            .frame(minHeight: 44)
        }
        .frame(maxWidth: 480, maxHeight: .infinity)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder private var serviceMessages: some View {
        if let error = model.downloadSetupError {
            NamedErrorCard(
                headline: "Downloads are unavailable.",
                message: "The durable download store could not be opened.",
                namedError: error)
        }
        if let error = model.downloadServiceError {
            NamedErrorCard(
                headline: "Saved downloads need attention.",
                message: "No transfer was resumed automatically.",
                namedError: error,
                tone: .caution)
        }
        ForEach(model.downloadRecoveryIssues) { issue in
            NamedErrorCard(
                headline: "A saved download could not be restored.",
                message: issue.destinationPath.isEmpty
                    ? "The job remains stopped."
                    : "The job at \(issue.destinationPath) remains stopped.",
                namedError: issue.reason,
                tone: .caution)
        }
        if model.canRetryDownloadRecovery {
            Button("Retry restoring downloads") {
                Task { await model.retryDownloadRecovery() }
            }
            .frame(minHeight: 44)
            .disabled(model.isRecoveringDownloads)
        }
    }

    private func transferCard(_ transfer: TransferPresentation) -> some View {
        let controller = model.downloadController(for: transfer.job)
        return VStack(alignment: .leading, spacing: MRSpace.s2) {
            DownloadCard(
                modelName: transfer.displayName,
                state: transfer.state,
                declaredBytes: transfer.declaredBytes,
                destinationPath: transfer.destinationPath,
                updatedAt: transfer.updatedAt,
                operationError: controller?.operationError,
                onPause: controller.map { controller in
                    { Task { await controller.pause() } }
                },
                onResume: controller.map { controller in
                    { Task { await controller.resume() } }
                },
                onCancel: controller.map { controller in
                    { Task { await controller.cancel() } }
                },
                onCancelVerification: controller.map { controller in
                    { controller.cancelVerification() }
                },
                onChooseVolume: transfer.hasCatalogEntry && controller != nil
                    ? {
                        destinationRequest = TransferDestinationRequest(
                            job: transfer.job,
                            model: transfer.model,
                            destination: URL(
                                fileURLWithPath: transfer.destinationPath,
                                isDirectory: true))
                    }
                    : nil)

            if let reason = transfer.reason {
                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
                        transferReason(reason)
                        Spacer(minLength: MRSpace.s2)
                        refreshModelsButton
                    }
                    VStack(alignment: .leading, spacing: MRSpace.s2) {
                        transferReason(reason)
                        refreshModelsButton
                    }
                }
                .font(MRType.caption)
                .foregroundStyle(MRColor.caution)
            }

            if transfer.hasCatalogEntry {
                NavigationLink {
                    ModelDetailView(modelID: transfer.model)
                } label: {
                    Text("Open model")
                        .frame(minHeight: 44)
                }
                .controlSize(.small)
            }
        }
    }

    private func transferReason(_ reason: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
            Image(systemName: "exclamationmark.circle")
            Text(reason)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder private var refreshModelsButton: some View {
        if model.allowsLiveCatalogRefresh {
            Button("Refresh models") {
                Task { await model.refreshPublishedModels() }
            }
            .frame(minHeight: 44)
            .disabled(model.isRefreshingPublishedModels)
        }
    }
}

private struct TransferDestinationRequest: Identifiable {
    var id: DownloadJobID { job }
    let job: DownloadJobID
    let model: ModelID
    let destination: URL
}
