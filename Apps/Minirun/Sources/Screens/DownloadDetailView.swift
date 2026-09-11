import MinirunKit
import SwiftUI

/// The plan, and what it disagrees with.
///
/// `index.json` is a claim; the repo tree is the truth. This screen shows both
/// numbers next to each other and states the delta, because the alternative —
/// picking one and printing it — is the thing the codebase refuses to do.
struct DownloadDetailView: View {
    let modelID: ModelID

    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var filter = ""
    @State private var showsPayloadOnly = true
    @State private var pendingForget: DownloadJobID?
    @State private var forgetWarning: String?

    var body: some View {
        Group {
            #if os(macOS)
                VStack(spacing: 0) {
                    MacColumnHeader(title: "Download details", backAction: { dismiss() })
                    MRHairline()
                    content
                }
            #else
                content
            #endif
        }
        .mrProductPage()
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

    private var content: some View {
        ScrollView {
            DownloadDetailPage(
                modelID: modelID,
                filter: $filter,
                showsPayloadOnly: $showsPayloadOnly,
                pendingForget: $pendingForget,
                forgetWarning: $forgetWarning)
        }
        .frame(maxWidth: .infinity)
        .mrPhoneNavigationTitle("Download details")
    }
}

/// Everything below the title bar: the transfers this model has, what the index
/// claims against what the tree holds, the two digest algorithms, and the file
/// list.
///
/// It owns no confirmation of its own — Forget sets a binding and the screen
/// presents the dialog — so this view is a pure function of the app model plus
/// those pending values, which is what makes it renderable in isolation. See
/// `ModelDetailPage`: `ImageRenderer` draws nothing at all for a macOS
/// `ScrollView`.
struct DownloadDetailPage: View {
    let modelID: ModelID
    @Binding var filter: String
    @Binding var showsPayloadOnly: Bool
    @Binding var pendingForget: DownloadJobID?
    @Binding var forgetWarning: String?

    @Environment(AppModel.self) private var model

    private var download: DownloadController? { model.download(modelID) }
    private var plan: DownloadPlan? { model.downloadPlanForPresentation(modelID) }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if model.downloadJobs(for: modelID).count > 1 {
                transfers
            }
            if download?.catalogEntryIsStale == true {
                NamedErrorCard(
                    headline: "This plan belongs to an earlier catalog.",
                    message:
                        "The catalog changed after this transfer was planned. Its job and "
                        + "progress were kept; the plan was not rewritten underneath it.",
                    namedError: "catalog entry changed or was removed",
                    tone: .caution)
                .padding(.top, MRSpace.s4)
            }
            if let plan {
                reconciliation(plan)
                digests(plan)
                files(plan)
            } else {
                EmptyStateView(
                    headline: "No plan yet.",
                    message: "Resolve the index first; the plan is built from the tree.")
                    .padding(.top, MRSpace.s4)
            }
        }
        .padding(pagePadding)
        .frame(maxWidth: 900, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var pagePadding: CGFloat {
        #if os(macOS)
            MRSpace.s5
        #else
            MRSpace.s4
        #endif
    }

    // MARK: - Transfers

    /// One row per job, and — for a job that has stopped and is not the one
    /// this screen is showing — a way to drop its record. A first attempt that
    /// was cancelled and cleaned up by hand has nothing left to describe, and
    /// without this it would stay in this list for the life of the install.
    private var transfers: some View {
        let jobs = model.downloadJobs(for: modelID)
        return MRPageSection(title: "Transfers", showsSeparator: false) {
            ForEach(jobs, id: \.jobID) { controller in
                if let job = controller.jobID {
                    transferRow(controller, job: job, isSelected: download?.jobID == job)
                    if job != jobs.last?.jobID { MRHairline() }
                }
            }
            if let forgetWarning {
                MRInlineNote(message: forgetWarning)
            }
        }
    }

    private func transferRow(
        _ controller: DownloadController, job: DownloadJobID, isSelected: Bool
    ) -> some View {
        MRQuietRow {
            Button {
                model.selectDownloadJob(job)
            } label: {
                HStack(alignment: .top, spacing: MRSpace.s2) {
                    Image(systemName: isSelected ? "largecircle.fill.circle" : "circle")
                        .foregroundStyle(isSelected ? MRColor.accent : MRColor.tertiary)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(controller.state.phrase.capitalizedTransferSentence)
                            .font(MRType.prose)
                            .foregroundStyle(MRColor.primary)
                            .fixedSize(horizontal: false, vertical: true)
                        MRPathLabel(path: controller.destinationPath
                            ?? "Destination is not available")
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityAddTraits(isSelected ? [.isSelected] : [])
        } trailing: {
            Text(MRFormat.timestamp(controller.updatedAt))
                .font(MRType.figure)
                .foregroundStyle(MRColor.tertiary)
            if !isSelected, model.canForgetDownloadJob(job) {
                Button(ForgetTransferConfirmation.rowActionTitle) { pendingForget = job }
                    .mrTextLink(.quiet)
            }
        }
    }

    // MARK: - Index versus tree

    private func reconciliation(_ plan: DownloadPlan) -> some View {
        let record = plan.reconciliation
        return MRPageSection(title: "Index versus tree") {
            MRStatusLine(
                tone: record.agrees ? .ready : .attention,
                sentence: record.agrees
                    ? "They agree."
                    : "They disagree — the declared payload count and bytes are not the "
                        + "tree's.")
            MRFactList(labelWidth: 150) {
                MRFact(label: "Declared by index") {
                    factValue(
                        "\(MRFormat.grouped(plan.index?.declaredFileCount ?? 0)) files",
                        detail: plan.index.map { MRFormat.publishedBytes($0.declaredBytes) }
                            ?? "not published",
                        provenance: .declared)
                }
                MRFact(label: "Payload in the tree") {
                    factValue(
                        "\(MRFormat.grouped(record.payloadFileCount)) files",
                        detail: MRFormat.publishedBytes(record.payloadBytes))
                }
                MRFact(label: "Whole tree") {
                    factValue(
                        "\(MRFormat.grouped(record.treeFileCount)) files",
                        detail: MRFormat.publishedBytes(record.treeBytes))
                }
                MRFact(label: "Delta") {
                    Text(
                        "\(record.fileCountDelta) files · "
                            + MRFormat.bytesDecimal(record.byteDelta)
                    )
                    .font(MRType.figure)
                    .foregroundStyle(record.agrees ? MRColor.ok : MRColor.caution)
                }
                MRFact(label: "Revision") {
                    factValue(
                        plan.repo.revision,
                        detail: plan.repo.isPinnedCommit
                            ? "an exact repository version; every download URL uses it"
                            : "a moving version cannot be verified and is refused",
                        valueFont: MRType.path,
                        valueColor: plan.repo.isPinnedCommit
                            ? MRColor.primary : MRColor.refuse)
                }
            }
            if let note = record.note {
                Text(note)
                    .font(MRType.prose)
                    .foregroundStyle(MRColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// A value over the sentence that explains it. `MRFact` takes one view, and
    /// half the facts on this screen are a quantity plus the phrase that says
    /// where the quantity came from.
    private func factValue(
        _ value: String,
        detail: String? = nil,
        provenance: ValueText.Provenance = .measured,
        valueFont: Font? = nil,
        valueColor: Color? = nil
    ) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            if let valueFont {
                Text(value)
                    .font(valueFont)
                    .foregroundStyle(valueColor ?? MRColor.primary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                MRFactValue(text: value, provenance: provenance)
                    .foregroundStyle(valueColor ?? MRColor.primary)
            }
            if let detail {
                Text(detail)
                    .font(MRType.prose)
                    .foregroundStyle(MRColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - Digests

    /// Two digest algorithms, on purpose. A verifier that assumed one would
    /// silently skip the files that use the other.
    private func digests(_ plan: DownloadPlan) -> some View {
        let sha = plan.files.filter { $0.digest.algorithm == .sha256 }
        let git = plan.files.filter { $0.digest.algorithm == .gitBlobSHA1 }
        return MRPageSection(title: "Digests") {
            MRFactList(labelWidth: 150) {
                MRFact(label: "sha256 (LFS objects)") {
                    factValue(
                        MRFormat.grouped(sha.count),
                        detail: "tree lfs.oid, and x-linked-etag on the resolve redirect")
                }
                MRFact(label: "git blob sha1") {
                    factValue(
                        MRFormat.grouped(git.count),
                        detail: "plain git blobs; they carry no sha256 at all")
                }
            }
            // Prose, not a monospaced paragraph. This is the one sentence on
            // the screen that has to be read rather than scanned, and set in
            // SF Mono it read as a log line nobody reads.
            Text(
                "The CDN's own ETag on the 200 or 206 is the Xet hash, a different number for "
                    + "the same bytes. Minirun never reads it as a digest: doing so is the one "
                    + "mistake here that produces a green check on wrong bytes."
            )
            .font(MRType.prose)
            .foregroundStyle(MRColor.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: - Files

    private func files(_ plan: DownloadPlan) -> some View {
        let shown = plan.files
            .filter { showsPayloadOnly ? $0.isPayload : true }
            .filter { filter.isEmpty || $0.path.localizedCaseInsensitiveContains(filter) }
            .prefix(300)
        return MRPageSection(title: "Files") {
            Text("\(MRFormat.grouped(plan.files.count)) in the tree")
                .font(MRType.prose)
                .foregroundStyle(MRColor.tertiary)
            ViewThatFits(in: .horizontal) {
                HStack(spacing: MRSpace.s3) {
                    fileFilter
                    payloadToggle
                }
                VStack(alignment: .leading, spacing: MRSpace.s2) {
                    fileFilter
                    payloadToggle
                }
            }
            .font(MRType.prose)

            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(shown), id: \.path) { file in
                    fileRow(file)
                        .overlay(alignment: .top) { MRHairline() }
                }
            }
            if plan.files.count > shown.count {
                Text("\(MRFormat.grouped(plan.files.count - shown.count)) more not shown.")
                    .font(MRType.prose)
                    .foregroundStyle(MRColor.tertiary)
            }
        }
    }

    private var fileFilter: some View {
        TextField("Filter by path", text: $filter)
            .textFieldStyle(.roundedBorder)
            #if os(macOS)
                .frame(maxWidth: 280)
            #endif
    }

    private var payloadToggle: some View {
        Toggle("Payload only", isOn: $showsPayloadOnly)
            .toggleStyle(.switch)
            .font(MRType.prose)
            .frame(minHeight: MRPageControl.height)
    }

    /// The path stays monospaced — it is a path — and the size stays tabular so
    /// a column of them lines up. Everything else on the row is prose.
    private func fileRow(_ file: RepoFile) -> some View {
        MRListRow(minimumWideWidth: 460, trailingWidth: 190) {
            VStack(alignment: .leading, spacing: 1) {
                Text(file.path)
                    .font(MRType.path)
                    .foregroundStyle(MRColor.primary)
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                Text(file.isPayload ? "Payload" : "Metadata")
                    .font(MRType.prose)
                    .foregroundStyle(MRColor.tertiary)
            }
        } trailing: {
            // One line, not two. Stacked under a path on a phone, a size over
            // an algorithm name doubles the height of every row in a list
            // three hundred rows long.
            HStack(alignment: .firstTextBaseline, spacing: MRSpace.s3) {
                Text(MRFormat.bytesDecimal(file.sizeBytes))
                    .font(MRType.figure)
                    .foregroundStyle(MRColor.primary)
                Text(file.digest.algorithm.rawValue)
                    .font(MRType.path)
                    .foregroundStyle(MRColor.tertiary)
            }
        }
        .accessibilityElement(children: .combine)
    }
}

extension String {
    /// A state phrase written as a sentence: the phrases are lowercase because
    /// they were built for a chip, and these screens have no chips.
    fileprivate var capitalizedTransferSentence: String {
        guard let first = first else { return self }
        return String(first).uppercased() + dropFirst()
    }
}

/// The words on the confirmation that drops a stopped transfer's record.
///
/// The destination is named, because a record is only worth dropping when the
/// operator recognises which attempt it was — and the sentence about the drive
/// is the whole promise: nothing under that path is read, moved or deleted.
enum ForgetTransferConfirmation {
    static let title = "Forget this transfer?"
    static let actionTitle = "Forget"
    /// The trailing action inside a transfer row, where the row already says
    /// which transfer it is.
    static let rowActionTitle = "Forget"
    /// The action beside a transfer block's primary button, where the block
    /// carries the noun itself.
    static let cardActionTitle = "Forget this transfer"
    static let filesAreSafeSentence = "Files on the drive are not touched."

    static func message(destinationPath: String?) -> String {
        "Destination: \(destinationPath ?? "not recorded")\n\n"
            + "This removes Minirun's record of the transfer. "
            + filesAreSafeSentence
    }
}
