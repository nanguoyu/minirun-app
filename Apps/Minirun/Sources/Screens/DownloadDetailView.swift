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
    @State private var filter = ""
    @State private var showsPayloadOnly = true

    private var download: DownloadController? { model.download(modelID) }
    private var plan: DownloadPlan? { model.downloadPlanForPresentation(modelID) }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MRSpace.s5) {
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
                }
                if let plan {
                    reconciliation(plan)
                    digests(plan)
                    files(plan)
                } else {
                    EmptyStateView(
                        headline: "No plan yet.",
                        message: "Resolve the index first; the plan is built from the tree.")
                }
            }
            .padding(pagePadding)
            .frame(maxWidth: 900, alignment: .leading)
        }
        .frame(maxWidth: .infinity)
        .mrWorkspaceSurface()
        .mrPhoneNavigationTitle("Download details")
    }

    private var pagePadding: CGFloat {
        #if os(macOS)
            MRSpace.s5
        #else
            MRSpace.s4
        #endif
    }

    private var transfers: some View {
        Panel(title: "Transfers") {
            ForEach(model.downloadJobs(for: modelID), id: \.jobID) { controller in
                if let job = controller.jobID {
                    Button {
                        model.selectDownloadJob(job)
                    } label: {
                        HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
                            Image(
                                systemName: download?.jobID == job
                                    ? "largecircle.fill.circle" : "circle")
                            VStack(alignment: .leading, spacing: 2) {
                                Text(controller.state.phrase)
                                    .font(MRType.caption)
                                    .foregroundStyle(MRColor.primary)
                                Text(
                                    controller.destinationPath
                                        ?? "Destination is not available")
                                    .font(MRType.micro)
                                    .foregroundStyle(MRColor.tertiary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                            Spacer()
                            Text(MRFormat.timestamp(controller.updatedAt))
                                .font(MRType.micro)
                                .foregroundStyle(MRColor.tertiary)
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private func reconciliation(_ plan: DownloadPlan) -> some View {
        let record = plan.reconciliation
        return Panel(title: "Index versus tree") {
            ViewThatFits(in: .horizontal) {
                HStack {
                    reconciliationStatus(record)
                    Spacer()
                    reconciliationBalance(record)
                }
                VStack(alignment: .leading, spacing: MRSpace.s2) {
                    reconciliationStatus(record)
                    reconciliationBalance(record)
                }
            }
            MetricRow(
                label: "declared by index",
                value: "\(MRFormat.grouped(plan.index?.declaredFileCount ?? 0)) files",
                provenance: .declared,
                detail: plan.index.map { MRFormat.publishedBytes($0.declaredBytes) }
                    ?? "not published")
            MetricRow(
                label: "payload in the tree",
                value: "\(MRFormat.grouped(record.payloadFileCount)) files",
                detail: MRFormat.publishedBytes(record.payloadBytes))
            MetricRow(
                label: "whole tree",
                value: "\(MRFormat.grouped(record.treeFileCount)) files",
                detail: MRFormat.publishedBytes(record.treeBytes))
            MetricRow(
                label: "delta",
                value: "\(record.fileCountDelta) files · \(MRFormat.bytesDecimal(record.byteDelta))",
                valueColor: record.agrees ? MRColor.ok : MRColor.caution)
            if let note = record.note {
                Text(note)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Divider().overlay(MRColor.hairline)
            MetricRow(
                label: "revision", value: plan.repo.revision,
                valueColor: plan.repo.isPinnedCommit ? MRColor.primary : MRColor.refuse,
                detail: plan.repo.isPinnedCommit
                    ? "an exact repository version; every download URL uses it"
                    : "a moving version cannot be verified and is refused")
        }
    }

    private func reconciliationStatus(_ record: IndexReconciliation) -> some View {
        Text(record.agrees ? "They agree." : "They disagree.")
            .font(MRType.headline)
            .foregroundStyle(record.agrees ? MRColor.ok : MRColor.caution)
    }

    private func reconciliationBalance(_ record: IndexReconciliation) -> some View {
        BalanceDot(
            balanced: record.agrees,
            failingIdentity: "declared payload count and bytes ≠ the tree's")
    }

    /// Two digest algorithms, on purpose. A verifier that assumed one would
    /// silently skip the files that use the other.
    private func digests(_ plan: DownloadPlan) -> some View {
        let sha = plan.files.filter { $0.digest.algorithm == .sha256 }
        let git = plan.files.filter { $0.digest.algorithm == .gitBlobSHA1 }
        return Panel(title: "Digests") {
            MetricRow(
                label: "sha256 (LFS objects)", value: MRFormat.grouped(sha.count),
                detail: "tree lfs.oid, and x-linked-etag on the resolve redirect")
            MetricRow(
                label: "git blob sha1", value: MRFormat.grouped(git.count),
                detail: "plain git blobs; they carry no sha256 at all")
            Text(
                "The CDN's own ETag on the 200 or 206 is the Xet hash, a different number for "
                    + "the same bytes. Minirun never reads it as a digest: doing so is the one "
                    + "mistake here that produces a green check on wrong bytes."
            )
            .font(MRType.micro)
            .foregroundStyle(MRColor.tertiary)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func files(_ plan: DownloadPlan) -> some View {
        let shown = plan.files
            .filter { showsPayloadOnly ? $0.isPayload : true }
            .filter { filter.isEmpty || $0.path.localizedCaseInsensitiveContains(filter) }
            .prefix(300)
        return VStack(alignment: .leading, spacing: MRSpace.s2) {
            SectionHeader(
                title: "Files", trailing: "\(MRFormat.grouped(plan.files.count)) in the tree")
            ViewThatFits(in: .horizontal) {
                HStack {
                    fileFilter
                    payloadToggle
                }
                VStack(alignment: .leading, spacing: MRSpace.s2) {
                    fileFilter
                    payloadToggle
                }
            }
            .font(MRType.caption)

            ForEach(Array(shown), id: \.path) { file in
                #if os(macOS)
                    desktopFileRow(file)
                #else
                    compactFileCard(file)
                #endif
            }
            if plan.files.count > shown.count {
                Text("\(MRFormat.grouped(plan.files.count - shown.count)) more not shown.")
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
            }
        }
    }

    private var fileFilter: some View {
        TextField("Filter by path", text: $filter)
            .textFieldStyle(.roundedBorder)
    }

    private var payloadToggle: some View {
        Toggle("Payload only", isOn: $showsPayloadOnly)
            .toggleStyle(.switch)
            .frame(minHeight: 44)
    }

    #if os(macOS)
        private func desktopFileRow(_ file: RepoFile) -> some View {
            HStack(spacing: MRSpace.s2) {
                Text(file.path)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Text(file.isPayload ? "payload" : "metadata")
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                Text(MRFormat.bytesDecimal(file.sizeBytes))
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.primary)
                    .frame(width: 78, alignment: .trailing)
                Text(file.digest.algorithm.rawValue)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .frame(width: 84, alignment: .leading)
            }
            .accessibilityElement(children: .combine)
        }
    #endif

    private func compactFileCard(_ file: RepoFile) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            Text(file.path)
                .font(MRType.micro)
                .foregroundStyle(MRColor.primary)
                .lineLimit(3)
                .truncationMode(.middle)
            ViewThatFits(in: .horizontal) {
                HStack(spacing: MRSpace.s3) {
                    fileFact(file.isPayload ? "payload" : "metadata")
                    fileFact(MRFormat.bytesDecimal(file.sizeBytes))
                    fileFact(file.digest.algorithm.rawValue)
                }
                VStack(alignment: .leading, spacing: MRSpace.s1) {
                    fileFact(file.isPayload ? "payload" : "metadata")
                    fileFact(MRFormat.bytesDecimal(file.sizeBytes))
                    fileFact(file.digest.algorithm.rawValue)
                }
            }
        }
        .padding(MRSpace.s3)
        .mrCard(MRColor.raised)
        .accessibilityElement(children: .combine)
    }

    private func fileFact(_ text: String) -> some View {
        Text(text)
            .font(MRType.micro)
            .foregroundStyle(MRColor.tertiary)
    }
}
