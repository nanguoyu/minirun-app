import MinirunKit
import Observation
import SwiftUI
import UniformTypeIdentifiers

/// A user-confirmed destination with one live grant.
///
/// The picker supplies the URL and opens its security scope. The returned
/// scope is transferred to `DownloadManaging.start`; the manager mints and
/// persists the job's own bookmark before starting any transfer. The picker
/// keeps no second, global destination bookmark.
struct AuthorizedDownloadDestination {
    let directory: URL
    let scope: StorageScope
}

/// Owns the picker grant until the operator explicitly starts or leaves.
/// Replacing a selection closes the old grant; `take()` transfers ownership to
/// `DownloadController.start` without releasing it.
@Observable
@MainActor
final class DownloadDestinationSelection {
    private(set) var destination: AuthorizedDownloadDestination?

    func replace(with destination: AuthorizedDownloadDestination) {
        self.destination?.scope.release()
        self.destination = destination
    }

    func take() -> AuthorizedDownloadDestination? {
        defer { destination = nil }
        return destination
    }

    func cancel() {
        destination?.scope.release()
        destination = nil
    }
}

enum DownloadDestinationAuthorization {
    static func authorize(
        _ directory: URL, storage: any StorageManaging
    ) throws -> AuthorizedDownloadDestination {
        AuthorizedDownloadDestination(
            directory: directory, scope: try storage.scope(for: directory))
    }
}

/// The one capacity decision shared by row copy and the Continue button.
///
/// Filesystem capacity is signed while catalog and headroom sizes are unsigned;
/// keeping the arithmetic inside `StorageManaging` avoids both bit-pattern
/// conversion and unchecked addition at the view layer.
enum DownloadDestinationCapacity {
    static func verdict(
        for volume: VolumeDescriptor, requiredBytes: UInt64, headroomBytes: UInt64,
        storage: any StorageManaging
    ) -> SpaceVerdict {
        if let refusal = volume.writeRefusal {
            return .refused(
                .destinationNotWritable(path: volume.mountPath, reason: refusal))
        }
        return storage.canHold(
            bytes: requiredBytes,
            at: URL(fileURLWithPath: volume.mountPath, isDirectory: true),
            headroomBytes: headroomBytes)
    }
}

/// Where the bytes go.
///
/// Every refusal here is named and carries its numbers: a read-only volume says
/// which volume and why, and a volume that is too small states the deficit to
/// the byte. There are no disabled rows without a reason.
struct DestinationPicker: View {
    let modelID: ModelID
    var preferredStartingDirectory: URL? = nil

    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    @State private var selected: VolumeDescriptor?
    @State private var destinationSelection = DownloadDestinationSelection()
    @State private var picking = false
    @State private var pickError: String?
    @State private var options = DownloadOptions()
    @State private var advancedOptionsExpanded = false

    private var entry: CatalogEntry? { model.entry(modelID) }
    private var requiredBytes: UInt64 { entry?.descriptor.totalBytes ?? 0 }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: MRSpace.s4) {
                    summary
                    #if os(iOS)
                        poweredDriveGuidance
                    #endif
                    volumeList
                    folderPicker
                    if let pendingDestination = destinationSelection.destination {
                        confirmation(for: pendingDestination)
                    }
                    if let pickError { pickErrorCard(pickError) }
                    optionsPanel
                }
                .padding(MRSpace.s4)
            }
            .mrWorkspaceSurface()
            .mrPhoneNavigationTitle("Destination")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Start download") {
                        startDownload()
                    }
                    .disabled(!canStartDownload)
                }
            }
        }
        #if os(macOS)
            .frame(minWidth: 560, minHeight: 620)
        #else
            .presentationDetents([.large])
            .presentationDragIndicator(.visible)
        #endif
        .onAppear {
            model.refreshVolumes()
            if let preferredStartingDirectory {
                selected = model.volumes.first {
                    path(preferredStartingDirectory, isInside: volumeURL($0))
                }
            }
        }
        .onDisappear {
            destinationSelection.cancel()
        }
    }

    private var summary: some View {
        Panel(title: "Download summary") {
            MetricRow(
                label: "artifact", value: entry?.descriptor.displayName ?? modelID.rawValue)
            MetricRow(
                label: "size", value: MRFormat.publishedBytes(requiredBytes),
                provenance: .declared, detail: "declared by index, not yet checked")
            MetricRow(
                label: "files",
                value: MRFormat.grouped(
                    (entry?.descriptor.payloadFileCount ?? 0)
                        + (entry?.descriptor.metadataFileCount ?? 0)))
            MetricRow(
                label: "free-space headroom",
                value: MRFormat.bytesDecimal(options.freeSpaceHeadroomBytes),
                detail: "kept clear on top of the artifact")
            if let entry, !model.hasVerifiedRuntime(for: entry.id) {
                NamedErrorCard(
                    headline: "This version cannot use this model in chat.",
                    message: "You can still download it for storage or another device.",
                    namedError: "app compatibility: not supported",
                    tone: .caution)
            }
        }
    }

    private var volumeList: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            SectionHeader(title: "Volumes", trailing: "\(model.volumes.count)")
            if let volumeError = model.volumeError {
                Text(volumeError)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.refuse)
            }
            ForEach(model.volumes) { volume in
                Button {
                    guard volume.writeRefusal == nil else { return }
                    selected = volume
                } label: {
                    VolumeRow(
                        volume: volume,
                        requiredBytes: requiredBytes,
                        headroomBytes: options.freeSpaceHeadroomBytes,
                        capacityVerdict: capacityVerdict(for: volume),
                        isSelected: selected?.mountPath == volume.mountPath)
                }
                .buttonStyle(.plain)
                .disabled(volume.writeRefusal != nil)
            }
        }
    }

    /// The system-confirmed security-scoped grant. Volume rows above are only
    /// shortcuts for pre-positioning this picker; enumerating `/Volumes` is a
    /// hardware fact, not sandbox write authority.
    private var folderPicker: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            SectionHeader(title: "Destination folder")
            Button {
                beginFolderSelection(
                    startingAt: destinationSelection.destination?.directory
                        ?? preferredStartingDirectory ?? selected.map(volumeURL))
            } label: {
                Label(
                    destinationSelection.destination == nil
                        ? "Choose destination folder…" : "Choose a different folder…",
                    systemImage: "folder.badge.plus")
            }
            #if os(iOS)
                .fileImporter(
                    isPresented: $picking, allowedContentTypes: [.folder],
                    allowsMultipleSelection: false
                ) { result in
                    handlePick(result)
                }
            #endif
            .frame(minHeight: 44)
            Text(
                "Choosing a folder grants access but does not start the download. Review the "
                    + "destination and press Start download when ready.")
                .font(MRType.caption)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func pickErrorCard(_ text: String) -> some View {
        NamedErrorCard(
            headline: "That folder could not be used.",
            message: "Nothing was written, and no grant was taken.",
            namedError: text,
            tone: .refuse)
    }

    private var optionsPanel: some View {
        Panel(title: "Transfer options") {
            DisclosureGroup("Advanced transfer options", isExpanded: $advancedOptionsExpanded) {
                VStack(alignment: .leading, spacing: MRSpace.s2) {
                    Stepper(
                        "Chunk size \(MRFormat.bytesDecimal(options.chunkBytes))",
                        onIncrement: {
                            options.chunkBytes = min(1 << 30, options.chunkBytes * 2)
                        },
                        onDecrement: {
                            options.chunkBytes = max(1 << 20, options.chunkBytes / 2)
                        })
                    Stepper(
                        "Concurrent files \(options.maximumConcurrentFiles)",
                        value: $options.maximumConcurrentFiles, in: 1...16)
                    Toggle("Verify after each file", isOn: $options.verifyAfterEachFile)
                }
                .padding(.top, MRSpace.s2)
            }
            if let refusal = validationRefusal {
                Text(refusal)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.refuse)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    #if os(iOS)
        private var poweredDriveGuidance: some View {
            Panel(title: "External drives") {
                Label {
                    Text(
                        "Some external drives may need their own power. If a drive disconnects "
                            + "or does not appear, connect it through a powered hub or dock."
                    )
                    .font(MRType.caption)
                    .foregroundStyle(MRColor.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                } icon: {
                    Image(systemName: "bolt.horizontal.circle")
                        .foregroundStyle(MRColor.caution)
                }
            }
        }
    #endif

    /// Options are refused, never clamped — and the refusal names the knob, the
    /// value and the allowed range.
    private var validationRefusal: String? {
        do {
            _ = try options.validated()
            return nil
        } catch let error as DownloadError {
            return error.description
        } catch {
            return "\(error)"
        }
    }

    private var canStartDownload: Bool {
        guard destinationSelection.destination != nil, validationRefusal == nil else {
            return false
        }
        return pendingDestinationVerdict.isFit
    }

    private var pendingDestinationVerdict: SpaceVerdict {
        guard let pendingDestination = destinationSelection.destination else {
            return .unknown(reason: "Choose a destination folder first.")
        }
        return model.storage.canHold(
            bytes: requiredBytes,
            at: pendingDestination.directory,
            headroomBytes: options.freeSpaceHeadroomBytes)
    }

    private func confirmation(
        for destination: AuthorizedDownloadDestination
    ) -> some View {
        Panel(title: "Ready to start") {
            MetricRow(
                label: "destination",
                value: destination.directory.lastPathComponent.isEmpty
                    ? destination.directory.path : destination.directory.lastPathComponent,
                detail: destination.directory.path)
            MetricRow(
                label: "download size",
                value: MRFormat.publishedBytes(requiredBytes),
                provenance: .declared)
            switch pendingDestinationVerdict {
            case .fits(let spare):
                MetricRow(
                    label: "space after download",
                    value: MRFormat.bytesDecimal(spare),
                    valueColor: MRColor.ok)
            case .refused(let error):
                NamedErrorCard(
                    headline: "This folder does not have enough writable space.",
                    message: "Choose another destination before starting.",
                    namedError: error.description,
                    tone: .refuse)
            case .unknown(let reason):
                NamedErrorCard(
                    headline: "Available space could not be confirmed.",
                    message: "Minirun will not start without a safe capacity decision.",
                    namedError: reason,
                    tone: .caution)
            }
        }
    }

    private func capacityVerdict(for volume: VolumeDescriptor) -> SpaceVerdict {
        DownloadDestinationCapacity.verdict(
            for: volume, requiredBytes: requiredBytes,
            headroomBytes: options.freeSpaceHeadroomBytes, storage: model.storage)
    }

    private func handlePick(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            acceptPickedFolder(url)
        case .failure(let error):
            pickError = (error as NSError).localizedDescription
        }
    }

    private func beginFolderSelection(startingAt directory: URL?) {
        #if os(macOS)
            switch FolderChooser.run(
                startingAt: directory,
                prompt: "Use this destination",
                message:
                    "Choose or create the artifact folder for this model. Minirun will write "
                    + "the downloaded repository files directly into the folder you confirm."
            ) {
            case .chose(let url): acceptPickedFolder(url)
            case .cancelled: break
            }
        #else
            _ = directory
            picking = true
        #endif
    }

    private func acceptPickedFolder(_ url: URL) {
        do {
            let destination = try DownloadDestinationAuthorization.authorize(
                url, storage: model.storage)
            destinationSelection.replace(with: destination)
            selected = model.volumes.first {
                path(destination.directory, isInside: volumeURL($0))
            }
            pickError = nil
        } catch let error as StorageError {
            pickError = error.description
        } catch {
            pickError = "\(error)"
        }
    }

    private func startDownload() {
        guard canStartDownload, let destination = destinationSelection.take() else { return }
        guard let download = model.prepareDownload(for: modelID) else {
            destination.scope.release()
            pickError = model.downloadStartRefusal(for: modelID)
                ?? "The selected model is no longer in the current catalog."
            return
        }

        // Clear the view's reference before crossing the ownership boundary.
        // From the `start` call onward the manager/controller closes the grant
        // on every success and failure path; sheet dismissal must not race it.
        pickError = nil
        Task {
            let started = await download.start(
                destination: destination.directory, scope: destination.scope,
                options: options)
            if started {
                dismiss()
            } else {
                pickError = download.operationError
                    ?? "The download could not be started with this destination."
            }
        }
    }

    private func path(_ candidate: URL, isInside root: URL) -> Bool {
        let candidateParts = candidate.standardizedFileURL.pathComponents
        let rootParts = root.standardizedFileURL.pathComponents
        guard candidateParts.count >= rootParts.count else { return false }
        return Array(candidateParts.prefix(rootParts.count)) == rootParts
    }

    private func volumeURL(_ volume: VolumeDescriptor) -> URL {
        URL(fileURLWithPath: volume.mountPath, isDirectory: true)
    }
}
