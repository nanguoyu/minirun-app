#if DEBUG
import Foundation
import MinirunKit

/// Serves the preview catalog. Replaced at integration by
/// `HuggingFaceCatalogSource`, which walks the live tree and paginates on the
/// `Link` header.
struct MockCatalogSource: ModelCatalogSource {

    /// When set, `fetch()` returns a snapshot that disagrees with the bundled
    /// one, so the divergence report has something to report. A repo whose byte
    /// count moved is news, and the screen that says so needs to be reachable.
    var simulateDivergence: Bool = false
    var failure: CatalogError?
    var latency: Duration = .milliseconds(400)

    func fetch() async throws -> ModelCatalogSnapshot {
        try? await Task.sleep(for: latency)
        if let failure { throw failure }
        let bundled = CatalogFixtures.snapshot
        guard simulateDivergence else {
            return ModelCatalogSnapshot(
                generatedAt: Date(), origin: .live, models: bundled.models)
        }
        let moved = bundled.models.map { descriptor -> ModelDescriptor in
            guard descriptor.id == .minimaxH3 else { return descriptor }
            return ModelDescriptor(
                id: descriptor.id, displayName: descriptor.displayName,
                architecture: descriptor.architecture, layout: descriptor.layout,
                source: .huggingFaceRepo(
                    HuggingFaceRepoRef(
                        repoID: "nanguoyu/MiniMax-H3-minirun",
                        revision: "9c1d5e0b7a3f42688d15b0c6ee9a37d4f2016b58")),
                payloadBytes: descriptor.payloadBytes + 2_147_483_648,
                metadataBytes: descriptor.metadataBytes,
                payloadFileCount: descriptor.payloadFileCount + 4,
                metadataFileCount: descriptor.metadataFileCount,
                largestFileBytes: descriptor.largestFileBytes,
                minimumBudgetBytes: descriptor.minimumBudgetBytes,
                runner: descriptor.runner, licenseName: descriptor.licenseName,
                licenseAcknowledgementRequired: descriptor.licenseAcknowledgementRequired,
                notes: descriptor.notes)
        }
        return ModelCatalogSnapshot(generatedAt: Date(), origin: .live, models: moved)
    }
}

/// Builds a `DownloadPlan` from a descriptor without a network.
///
/// The file list is synthesized, but the two things the product argues from are
/// not: the payload/metadata classification rule is the real one, and the
/// synthesized sizes are made to sum EXACTLY to the descriptor's byte counts,
/// so the reconciliation the download screen shows is a real arithmetic check
/// and not a hard-coded "agrees".
enum PlanBuilder {

    /// The preview planner only knows the three published layouts it can name
    /// file by file. A newly discovered repository remains visible in the
    /// catalog but gets no synthetic plan: inventing paths for an unknown
    /// layout would turn discovery into a claim about bytes this build cannot
    /// interpret.
    static func plan(for entry: CatalogEntry) -> DownloadPlan? {
        let descriptor = entry.descriptor
        guard descriptor.layout != .unrecognized,
            descriptor.architecture != .unrecognized
        else { return nil }
        let repo: HuggingFaceRepoRef
        switch descriptor.source {
        case .huggingFaceRepo(let reference):
            repo = reference
        case .locallyBuilt(let tool):
            repo = HuggingFaceRepoRef(repoID: tool, revision: String(repeating: "0", count: 40))
        }

        guard let payload = payloadFiles(for: entry),
            let metadata = metadataFiles(for: entry)
        else { return nil }
        let files = metadata + payload

        let payloadBytes = payload.reduce(UInt64(0)) { $0 + $1.sizeBytes }
        let treeBytes = files.reduce(UInt64(0)) { $0 + $1.sizeBytes }
        let claim = entry.index.claim

        let reconciliation = IndexReconciliation(
            treeFileCount: files.count,
            treeBytes: treeBytes,
            payloadFileCount: payload.count,
            payloadBytes: payloadBytes,
            fileCountDelta: claim.declaredFileCount - payload.count,
            byteDelta: Int64(bitPattern: claim.declaredBytes) - Int64(bitPattern: payloadBytes),
            note:
                "The index declares \(MRFormat.grouped(claim.declaredFileCount)) payload files "
                + "and \(MRFormat.publishedBytes(claim.declaredBytes)). The tree carries "
                + "\(MRFormat.grouped(files.count)) files in total, of which "
                + "\(MRFormat.grouped(payload.count)) are payload by the classification rule. "
                + "Where they disagree, the tree is the truth.")

        return DownloadPlan(
            model: descriptor.id, repo: repo, files: files, index: claim,
            reconciliation: reconciliation)
    }

    private static func payloadFiles(for entry: CatalogEntry) -> [RepoFile]? {
        let descriptor = entry.descriptor
        let count = descriptor.payloadFileCount
        guard count > 0 else { return [] }
        let sizes = distribute(
            total: descriptor.payloadBytes, across: count, largest: descriptor.largestFileBytes)
        var files: [RepoFile] = []
        files.reserveCapacity(count)
        for index in 0..<count {
            guard let path = payloadPath(
                layout: descriptor.layout, index: index, entry: entry)
            else { return nil }
            // H3 keeps a handful of payload files as plain git blobs rather
            // than LFS objects. They carry a git-blob SHA-1; a verifier that
            // assumes one algorithm silently skips them.
            let isGitBlob = descriptor.id == .minimaxH3 && index >= count - 16
            files.append(
                RepoFile(
                    path: path, sizeBytes: sizes[index],
                    digest: isGitBlob
                        ? .gitBlobSHA1(hex: syntheticHex(path, length: 40))
                        : .sha256(hex: syntheticHex(path, length: 64)),
                    isPayload: RepoFileClassification.isPayload(path: path)))
        }
        return files
    }

    private static func metadataFiles(for entry: CatalogEntry) -> [RepoFile]? {
        let descriptor = entry.descriptor
        let count = descriptor.metadataFileCount
        guard count > 0 else { return [] }
        let sizes = distribute(total: descriptor.metadataBytes, across: count, largest: nil)
        let rootNames = ["index.json", "README.md", "LICENSE", ".gitattributes", "config.json"]
        var files: [RepoFile] = []
        files.reserveCapacity(count)
        for index in 0..<count {
            let path: String
            if index < min(rootNames.count, count - 1) {
                path = rootNames[index]
            } else {
                guard let unit = metadataUnitName(
                    layout: descriptor.layout,
                    index: index - min(rootNames.count, count - 1))
                else {
                    return nil
                }
                path = "\(unit)/manifest.json"
            }
            files.append(
                RepoFile(
                    path: path, sizeBytes: sizes[index],
                    digest: .gitBlobSHA1(hex: syntheticHex(path, length: 40)),
                    isPayload: RepoFileClassification.isPayload(path: path)))
        }
        return files
    }

    private static func payloadPath(
        layout: ArtifactLayout, index: Int, entry: CatalogEntry
    ) -> String? {
        guard let unit = unitName(layout: layout, index: index) else { return nil }
        switch layout {
        case .k3FlagshipLayerStreams:
            return index % 4 == 3
                ? "\(unit)/\(unit)-deterministic.bin"
                : "\(unit)/\(unit)-w\(index % 4 + 1).mxfp4tile"
        case .v4FlashUnitBundle:
            return index % 3 == 0
                ? "\(unit)/\(unit)-w\(index).fp8tile"
                : "\(unit)/\(unit)-w\(index).mxfp4tile"
        case .h3UnitBundle:
            return index % 5 == 0
                ? "\(unit)/\(unit)-\(index).qstream"
                : "\(unit)/\(unit)-\(index).h3tile"
        case .qwenContainerSet, .unrecognized:
            return nil
        }
    }

    private static func unitName(layout: ArtifactLayout, index: Int) -> String? {
        switch layout {
        case .k3FlagshipLayerStreams:
            return String(format: "layer%02d", index / 4)
        case .v4FlashUnitBundle:
            return String(format: "layers%02d", index / 12)
        case .h3UnitBundle:
            return String(format: "dit-block-%02d", index / 16)
        case .qwenContainerSet, .unrecognized:
            return nil
        }
    }

    /// Metadata needs one manifest path per synthetic record. Payload unit
    /// grouping deliberately maps several files to one layer, so reusing that
    /// helper here produced duplicate `layer01/manifest.json` keys and made a
    /// preview plan impossible to persist safely.
    private static func metadataUnitName(layout: ArtifactLayout, index: Int) -> String? {
        switch layout {
        case .k3FlagshipLayerStreams:
            return String(format: "layer%02d", index)
        case .v4FlashUnitBundle:
            return String(format: "layers%02d", index)
        case .h3UnitBundle:
            return String(format: "dit-block-%02d", index)
        case .qwenContainerSet, .unrecognized:
            return nil
        }
    }

    /// Sizes that sum to `total` exactly. The last file absorbs the remainder,
    /// so the plan's byte identity holds by construction rather than by luck.
    private static func distribute(
        total: UInt64, across count: Int, largest: UInt64?
    ) -> [UInt64] {
        guard count > 0 else { return [] }
        guard count > 1 else { return [total] }
        var sizes = [UInt64](repeating: 0, count: count)
        var remaining = total
        if let largest, largest < total {
            sizes[0] = largest
            remaining -= largest
        }
        let others = count - (sizes[0] > 0 ? 1 : 0)
        let each = remaining / UInt64(others)
        var assigned: UInt64 = 0
        for index in (sizes[0] > 0 ? 1 : 0)..<count {
            sizes[index] = each
            assigned += each
        }
        sizes[count - 1] += remaining - assigned
        return sizes
    }

    /// Deterministic, so the same path always shows the same digest and the
    /// screens do not flicker between launches. Not a real hash and never
    /// presented as one.
    static func syntheticHex(_ seed: String, length: Int) -> String {
        var state = UInt64(5_381)
        for byte in seed.utf8 {
            state = state &* 1_099_511_628_211 &+ UInt64(byte)
        }
        var out = ""
        while out.count < length {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            out += String(format: "%016lx", state)
        }
        return String(out.prefix(length))
    }
}
#endif
