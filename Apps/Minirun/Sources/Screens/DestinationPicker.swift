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
    /// Called on the main actor once the manager has accepted the job. The
    /// presenter closes the sheet through its own state: `dismiss` from a
    /// task, at the root of a `NavigationStack` inside a sheet, was observed
    /// to leave the sheet standing until the operator pressed Escape.
    var onStarted: (() -> Void)? = nil

    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    @State private var selected: VolumeDescriptor?
    @State private var destinationSelection = DownloadDestinationSelection()
    @State private var picking = false
    /// Between Start and the manager's answer: the plan is validated against
    /// the folder, the folder is created, the job registered. Seconds on a
    /// 624-file plan, during which the button must say so rather than go grey.
    @State private var isStarting = false
    @State private var pickError: String?
    @State private var options = DownloadOptions()
    @State private var advancedOptionsExpanded = false

    private var entry: CatalogEntry? { model.entry(modelID) }
    /// The whole published size — what a fresh copy would need.
    private var publishedBytes: UInt64 { entry?.descriptor.totalBytes ?? 0 }

    /// What still has to land at `directory`: the published size minus what a
    /// same-sized file already present there covers, and minus the bytes a
    /// part file already holds. A resume must be judged against the bytes it
    /// will write, not against the whole model, or a drive that plainly has
    /// room for the rest refuses the transfer that would finish it.
    private func remainingBytes(at directory: URL?) -> UInt64 {
        guard let directory, let plan = model.downloadPlanForPresentation(modelID) else {
            return publishedBytes
        }
        let kept = KeptFiles.measuring(plan, in: directory)
        guard case .present = kept.destination else { return publishedBytes }
        let covered = kept.completeBytes.addingReportingOverflow(kept.partialBytes)
        let landed = covered.overflow ? UInt64.max : covered.partialValue
        return landed >= publishedBytes ? 0 : publishedBytes - landed
    }

    /// Bytes still to write into the artifact folder under the pending
    /// destination, or the whole model until a destination is chosen.
    private var requiredBytes: UInt64 { remainingBytes(at: layout?.directory) }

    /// Bytes still to write for a volume row: the artifact folder this
    /// repository would take inside that volume, if it is already there.
    private func requiredBytes(on volume: VolumeDescriptor) -> UInt64 {
        guard let repository,
            let folder = ArtifactFolderPolicy.resolve(
                picked: volumeURL(volume), repository: repository, model: modelID)
        else { return publishedBytes }
        return remainingBytes(at: folder.directory)
    }

    /// The repository whose name the artifact folder takes.
    private var repository: HuggingFaceRepoRef? { entry?.descriptor.source.repo }

    /// The folder name Minirun will create inside whatever the operator picks.
    /// Known before a folder is chosen, because it comes from the catalog row.
    private var subfolderName: String? {
        repository.flatMap(ArtifactFolderPolicy.subfolderName(for:))
    }

    /// Where the bytes will actually go, once a parent has been confirmed.
    ///
    /// Classified against the resolved plan when one exists — that is the same
    /// authority `DownloadController.start` applies at Continue time — and
    /// against the repository alone before then.
    private var layout: DownloadDestinationLayout? {
        guard let picked = destinationSelection.destination?.directory,
            let repository
        else { return nil }
        if let plan = model.downloadPlanForPresentation(modelID) {
            return ArtifactFolderPolicy.resolve(picked: picked, plan: plan)
        }
        return ArtifactFolderPolicy.resolve(
            picked: picked, repository: repository, model: modelID)
    }

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
                        .disabled(isStarting)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button {
                        startDownload()
                    } label: {
                        if isStarting {
                            HStack(spacing: MRSpace.s1) {
                                ProgressView().controlSize(.small)
                                Text("Starting…")
                            }
                        } else {
                            Text("Start download")
                        }
                    }
                    .disabled(!canStartDownload || isStarting)
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
                selected = volume(containing: preferredStartingDirectory)
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
                label: "size", value: MRFormat.publishedBytes(publishedBytes),
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
                        requiredBytes: requiredBytes(on: volume),
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
            SectionHeader(title: "Where to keep this model")
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
            Text(chooserMessage)
                .font(MRType.caption)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text(
                "Choosing a folder grants access but does not start the download. Review the "
                    + "destination and press Start download when ready.")
                .font(MRType.caption)
                .foregroundStyle(MRColor.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// One sentence, shown in the sheet and put into the system folder chooser,
    /// so the operator reads the same promise in both places: the folder they
    /// confirm is the parent, not the artifact.
    private var chooserMessage: String {
        guard let subfolderName else {
            return "Choose where to keep this model. Minirun creates a folder for it inside "
                + "the one you choose."
        }
        return "Choose where to keep this model. Minirun creates a folder named "
            + "\(subfolderName) inside it."
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
        guard destinationSelection.destination != nil, validationRefusal == nil,
            let layout, layout.state.allowsWriting
        else { return false }
        return pendingDestinationVerdict.isFit
    }

    /// Capacity is a fact about the volume, and the artifact folder does not
    /// exist yet, so the verdict is always taken against the parent.
    private var pendingDestinationVerdict: SpaceVerdict {
        guard let pendingDestination = destinationSelection.destination else {
            return .unknown(reason: "Choose a destination folder first.")
        }
        return model.storage.canHold(
            bytes: requiredBytes,
            at: layout?.parent ?? pendingDestination.directory,
            headroomBytes: options.freeSpaceHeadroomBytes)
    }

    private func confirmation(
        for destination: AuthorizedDownloadDestination
    ) -> some View {
        Panel(title: "Ready to start") {
            MetricRow(
                label: "chosen folder",
                value: destination.directory.lastPathComponent.isEmpty
                    ? destination.directory.path : destination.directory.lastPathComponent,
                detail: destination.directory.path)
            if let layout {
                MetricRow(
                    label: "model folder", value: layout.subfolderName,
                    detail: layout.state == .absent
                        ? "created inside the folder you chose"
                        : "already there; this transfer reuses it")
                Text(layout.previewSentence)
                    .font(MRType.caption)
                    .foregroundStyle(MRColor.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            } else if subfolderName == nil {
                NamedErrorCard(
                    headline: "This model publishes no folder name.",
                    message: "Minirun will not write a repository into a folder it cannot name.",
                    namedError: "artifact folder: no canonical repository name",
                    tone: .refuse)
            }
            if let refusal = layout?.state.refusal {
                NamedErrorCard(
                    headline: "That folder already holds something else.",
                    message: "Nothing was written. Choose another folder, or move what is "
                        + "there out of the way.",
                    namedError: refusal,
                    tone: .refuse)
            }
            if requiredBytes < publishedBytes {
                MetricRow(
                    label: "still to download",
                    value: MRFormat.bytesDecimal(requiredBytes),
                    detail: "\(MRFormat.bytesDecimal(publishedBytes - requiredBytes)) of "
                        + "\(MRFormat.publishedBytes(publishedBytes)) is already in the folder")
            } else {
                MetricRow(
                    label: "download size",
                    value: MRFormat.publishedBytes(requiredBytes),
                    provenance: .declared)
            }
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
            for: volume, requiredBytes: requiredBytes(on: volume),
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
                message: chooserMessage
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
            selected = volume(containing: destination.directory)
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
        isStarting = true
        Task { @MainActor in
            let started = await download.start(
                destination: destination.directory, scope: destination.scope,
                options: options)
            isStarting = false
            if started {
                // The transfer is running and the model page is about to show
                // it; the sheet has nothing more to say.
                if let onStarted { onStarted() } else { dismiss() }
            } else {
                pickError = download.operationError
                    ?? "The download could not be started with this destination."
            }
        }
    }

    /// The volume a path lives on: the deepest mount that contains it. Every
    /// path is inside `/`, so "the first volume that contains it" was always
    /// the internal disk, and picking a folder on an external drive flipped
    /// the selection back to Macintosh HD.
    private func volume(containing directory: URL) -> VolumeDescriptor? {
        model.volumes
            .filter { path(directory, isInside: volumeURL($0)) }
            .max { lhs, rhs in
                volumeURL(lhs).standardizedFileURL.pathComponents.count
                    < volumeURL(rhs).standardizedFileURL.pathComponents.count
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
