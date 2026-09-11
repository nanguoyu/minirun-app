import MinirunKit
import SwiftUI

/// The leading catalog symbol is a projection of what the storage scan found.
/// Download state is only a fallback when no artifact is visible in any
/// registered location; it must never overwrite a directory fact.
enum ModelRowLeadingState: Equatable {
    case locationUnknown
    case scannedIncomplete
    case scannedUnverified
    case scannedSpotChecked
    case scannedFullyVerified
    case download(DownloadState)

    static func resolve(
        installations: [DiscoveredArtifact], hasStorageLocation: Bool,
        downloadState: DownloadState
    ) -> ModelRowLeadingState {
        guard !installations.isEmpty else {
            if !hasStorageLocation && !downloadState.isReady { return .locationUnknown }
            return .download(downloadState)
        }

        let complete = installations.filter { $0.isComplete == true }
        if complete.contains(where: { $0.verification == .fullyVerified }) {
            return .scannedFullyVerified
        }
        if complete.contains(where: { $0.verification == .spotChecked }) {
            return .scannedSpotChecked
        }
        if !complete.isEmpty { return .scannedUnverified }
        if installations.contains(where: { $0.isComplete == false }) {
            return .scannedIncomplete
        }
        return .scannedUnverified
    }

    var glyphName: String {
        switch self {
        case .locationUnknown: return "cube.transparent"
        case .scannedIncomplete, .scannedUnverified: return "cube"
        case .scannedSpotChecked, .scannedFullyVerified: return "cube.fill"
        case .download: return "cube"
        }
    }

    var accessibilityLabel: String {
        switch self {
        case .locationUnknown: return "storage location not configured"
        case .scannedIncomplete: return "artifact found but incomplete"
        case .scannedUnverified: return "artifact found, not verified"
        case .scannedSpotChecked: return "artifact found and spot-checked"
        case .scannedFullyVerified: return "artifact found and fully verified"
        case .download(let state): return state.phrase
        }
    }
}

enum ModelRowPresentation {
    /// The third line under a model's name: what this build can do with it, and
    /// what it last cost here.
    ///
    /// It replaced three `StatusChip` pills. A pill is a shape that says
    /// "badge", and three of them on a row made the catalog look like a
    /// storefront with sale stickers; the same three facts are one quiet
    /// sentence, and most rows still have nothing to say here at all.
    ///
    /// A model with no runner in this build gets no run time, because it has
    /// never run here — *container only* is the whole of what it is.
    static func note(
        fitness: PlatformFitness, lastRunSecondsPerToken: Double? = nil
    ) -> String? {
        var terms: [String] = []
        switch fitness.verdict {
        case .noRunner: terms.append("Container only")
        case .refused: terms.append("Won't run on this device")
        case .runnableWithCaveats: terms.append("Runs with limits")
        case .runnable: break
        }
        if fitness.verdict != .noRunner,
            let seconds = lastRunSecondsPerToken, seconds.isFinite, seconds > 0
        {
            terms.append("Last run \(MRFormat.clock(seconds))/token")
        }
        return terms.isEmpty ? nil : terms.joined(separator: " · ")
    }
}

/// What a list row knows about the folder a copy was found in: what to call it,
/// and whether it is plugged in right now.
///
/// A row that knows only the name says "Ready on K3NVME" about a drive in a
/// drawer, which is the exact claim the model page was rebuilt to stop making.
struct ModelRowLocation: Equatable {
    let name: String
    let isMounted: Bool
}

/// One short sentence for one row of the models list, and the colour of the dot
/// beside it.
///
/// Same order of authority as the model page — a transfer in motion speaks
/// first, and then **the disk speaks last** — in the words a 230-point column
/// has room for. The page says *Ready on K3NVME · every file matches its
/// published digest*; the row says *Ready on K3NVME*, and the page is one tap
/// away for the rest.
enum ModelListStatusPresentation {
    static func resolve(
        state: DownloadState,
        installations: [DiscoveredArtifact],
        locations: [String: ModelRowLocation],
        hasStorageLocation: Bool
    ) -> ModelPageStatus {
        switch state {
        case .notStarted, .awaitingDestination:
            break
        case .cancelled, .ready:
            if installations.isEmpty, let transfer = transfer(state) { return transfer }
        default:
            if let transfer = transfer(state) { return transfer }
        }

        guard let described = described(installations, locations: locations) else {
            // Nothing found, and with no folder registered nothing has been
            // looked at either. "Not on this device" would be a guess wearing
            // a state's clothes.
            return ModelPageStatus(
                tone: .idle,
                sentence: hasStorageLocation
                    ? "Not on this device" : "No folder added to look in")
        }
        let place = locations[described.locationPath]?.name
            ?? URL(fileURLWithPath: described.locationPath).lastPathComponent
        let copies = installations.count > 1 ? " · \(installations.count) copies" : ""
        guard locations[described.locationPath]?.isMounted == true else {
            return ModelPageStatus(
                tone: .attention, sentence: "\(place) is not connected" + copies)
        }
        guard described.isComplete != false else {
            return ModelPageStatus(
                tone: .attention, sentence: "On \(place) · files are missing" + copies)
        }
        switch described.verification {
        case .fullyVerified:
            return ModelPageStatus(tone: .ready, sentence: "Ready on \(place)" + copies)
        case .spotChecked:
            return ModelPageStatus(
                tone: .attention, sentence: "On \(place) · a sample matched" + copies)
        case .unverified:
            return ModelPageStatus(
                tone: .attention, sentence: "On \(place) · not verified" + copies)
        }
    }

    /// The copy a row speaks about: the best one that is plugged in, and
    /// otherwise the best one anywhere. Better means complete before
    /// incomplete, then verified before sampled before unread, then larger.
    static func described(
        _ installations: [DiscoveredArtifact], locations: [String: ModelRowLocation]
    ) -> DiscoveredArtifact? {
        let mounted = installations.filter { locations[$0.locationPath]?.isMounted == true }
        return (mounted.isEmpty ? installations : mounted).max { first, second in
            rank(first) < rank(second)
        }
    }

    private static func rank(_ artifact: DiscoveredArtifact) -> (Int, Int, UInt64) {
        let complete = artifact.isComplete == false ? 0 : 1
        let checked: Int
        switch artifact.verification {
        case .unverified: checked = 0
        case .spotChecked: checked = 1
        case .fullyVerified: checked = 2
        }
        return (complete, checked, artifact.bytesOnDisk)
    }

    /// The transfer half, in row-sized words. The model page's own sentences
    /// name the drive and the time left; a row states the position and stops.
    private static func transfer(_ state: DownloadState) -> ModelPageStatus? {
        func of(_ progress: ProgressSnapshot) -> String {
            "\(MRFormat.bytesDecimal(progress.verifiedBytes)) of "
                + MRFormat.bytesDecimal(progress.totalBytes)
        }
        switch state {
        case .resolvingIndex:
            return ModelPageStatus(tone: .moving, sentence: "Preparing the transfer")
        case .active(let progress):
            return ModelPageStatus(tone: .moving, sentence: "Downloading · \(of(progress))")
        case .paused(let progress):
            return ModelPageStatus(tone: .attention, sentence: "Paused · \(of(progress))")
        case .verifying:
            return ModelPageStatus(tone: .moving, sentence: "Checking downloaded files")
        case .interrupted(_, let reason):
            return ModelPageStatus(
                tone: .attention, sentence: "Interrupted — \(reason.sentence)")
        case .incomplete(let files, _):
            return ModelPageStatus(
                tone: .attention,
                sentence: "\(files) file\(files == 1 ? "" : "s") did not verify")
        case .failed:
            return ModelPageStatus(tone: .attention, sentence: "The transfer failed")
        case .cancelled:
            return ModelPageStatus(tone: .idle, sentence: "Cancelled")
        case .ready:
            return ModelPageStatus(tone: .ready, sentence: "Downloaded and checked")
        case .notStarted, .awaitingDestination:
            return nil
        }
    }
}

/// The visual identity a model's publisher uses in catalog surfaces.
///
/// Known product IDs are authoritative. Repository and display-name tokens
/// extend the same treatment to newly published repositories without turning
/// an unrelated future name into one of these brands. Unknown publishers keep
/// the neutral artifact glyph until an explicit identity is added.
enum ModelPublisherIdentity: Equatable {
    case kimi
    case deepSeek
    case miniMax
    case qwen
    case unknown

    static func resolve(
        modelID: ModelID, displayName: String, repositoryID: String?
    ) -> ModelPublisherIdentity {
        if modelID == .kimiK3 { return .kimi }
        if modelID == .deepseekV4Flash { return .deepSeek }
        if modelID == .deepseekV41Flash { return .deepSeek }
        if modelID == .minimaxH3 { return .miniMax }

        let tokens = [modelID.rawValue, displayName, repositoryID ?? ""]
            .flatMap {
                $0.lowercased().split { !$0.isLetter && !$0.isNumber }
            }
            .map(String.init)
        if tokens.contains("kimi") || tokens.contains("moonshotai") { return .kimi }
        if tokens.contains("deepseek") { return .deepSeek }
        if tokens.contains("minimax") { return .miniMax }
        if tokens.contains(where: { brandToken($0, prefix: "qwen") }) { return .qwen }
        return .unknown
    }

    var assetName: String? {
        switch self {
        case .kimi: return "PublisherKimi"
        case .deepSeek: return "PublisherDeepSeek"
        case .miniMax: return "PublisherMiniMax"
        case .qwen: return "PublisherQwen"
        case .unknown: return nil
        }
    }

    var publisherName: String? {
        switch self {
        case .kimi: return "Moonshot AI"
        case .deepSeek: return "DeepSeek"
        case .miniMax: return "MiniMax"
        case .qwen: return "Qwen"
        case .unknown: return nil
        }
    }

    private static func brandToken(_ token: String, prefix: String) -> Bool {
        guard token.hasPrefix(prefix) else { return false }
        let suffix = token.dropFirst(prefix.count)
        return suffix.isEmpty || suffix.first?.isNumber == true
    }
}

/// One consistently sized publisher mark for both local and remote lists.
/// Installation and verification remain separate facts in the row's pills;
/// a logo must never be recoloured to imply storage state.
struct ModelPublisherMark: View {
    let identity: ModelPublisherIdentity
    let fallbackSystemName: String
    let fallbackColor: Color
    /// A row's mark is 28 points. A product page's header wants the same mark
    /// at 40 inside its 64-point tile, and one component draws both rather
    /// than two components disagreeing about which logo a publisher has.
    var size: CGFloat = 28

    var body: some View {
        Group {
            if let assetName = identity.assetName {
                Image(assetName)
                    .resizable()
                    .renderingMode(.original)
                    .scaledToFit()
                    .padding(size * 0.07)
            } else {
                Image(systemName: fallbackSystemName)
                    .font(.system(size: size * 0.6, weight: .medium))
                    .foregroundStyle(fallbackColor)
            }
        }
        .frame(width: size, height: size)
        .accessibilityHidden(true)
    }
}

/// One row of the models list.
///
/// Rows and hairlines: a publisher's mark, the model's name, one line saying
/// what it is, and — in the trailing column — a dot with the sentence that
/// says where this one stands, over its size.
///
/// It used to carry up to five `StatusChip` pills: one per discovered copy,
/// plus one for compatibility, plus one when no folder had been added. Every
/// pill was the same shape, so a verified 517 GB copy on a plugged-in drive
/// and a device that cannot run the model read as the same kind of badge. The
/// facts did not change; they became one sentence and one dot, and where a
/// model has two copies the sentence says so and the page holds the details.
struct ModelRow: View {
    let entry: CatalogEntry
    let state: DownloadState
    let fitness: PlatformFitness
    /// The quiet third line: what this build can do with the model, and what it
    /// last cost. Nil leaves the row deliberately quiet.
    let note: String?
    /// Every place on this machine the model was actually found. The row states
    /// one of them and says how many there are; per-copy bytes, paths and
    /// verification live on the model page, which is one tap away.
    var installations: [DiscoveredArtifact] = []
    /// What each location is called and whether it is plugged in, by path.
    var locations: [String: ModelRowLocation] = [:]
    /// Whether the app has been granted any folder to scan at all.
    ///
    /// False changes what this row is entitled to claim. With no registered
    /// folder the app has not looked at a single byte of anybody's disk, and
    /// saying "not on this device" there is a guess wearing a state's clothes.
    var hasStorageLocation = true

    private var isDimmed: Bool { fitness.verdict == .refused }

    private var leadingState: ModelRowLeadingState {
        .resolve(
            installations: installations, hasStorageLocation: hasStorageLocation,
            downloadState: state)
    }

    private var status: ModelPageStatus {
        ModelListStatusPresentation.resolve(
            state: state, installations: installations,
            locations: locations, hasStorageLocation: hasStorageLocation)
    }

    /// The copy whose measured bytes the row prints, when there is one.
    private var describedArtifact: DiscoveredArtifact? {
        ModelListStatusPresentation.described(installations, locations: locations)
    }

    var body: some View {
        MRListRow {
            MRRowIdentity(
                title: entry.descriptor.displayName,
                subtitle: ModelPurposePresentation.line(for: entry.descriptor),
                note: note,
                noteColor: isDimmed ? MRColor.caution : MRColor.tertiary
            ) {
                mark
            }
        } trailing: {
            MRRowStatus(tone: status.tone, sentence: status.sentence)
            size
        }
        .opacity(isDimmed ? 0.62 : 1)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(entry.descriptor.displayName)
        .accessibilityValue(spoken)
    }

    private var mark: some View {
        ModelPublisherMark(
            identity: .resolve(
                modelID: entry.id,
                displayName: entry.descriptor.displayName,
                repositoryID: entry.descriptor.source.repo?.repoID),
            fallbackSystemName: leadingState.glyphName,
            fallbackColor: MRColor.tertiary,
            size: 32)
    }

    /// Measured where a copy has been counted, published otherwise — and the
    /// type says which, upright versus italic. This is the one number on the
    /// screen a reader compares between rows, so both faces are tabular.
    @ViewBuilder private var size: some View {
        if let artifact = describedArtifact {
            MRRowQuantity(
                text: MRFormat.measuredBytes(artifact.bytesOnDisk), provenance: .measured)
        } else {
            MRRowQuantity(
                text: MRFormat.publishedBytes(entry.descriptor.totalBytes),
                provenance: .declared)
        }
    }

    private var spoken: String {
        [
            status.sentence,
            describedArtifact.map { MRFormat.measuredBytes($0.bytesOnDisk) }
                ?? MRFormat.publishedBytes(entry.descriptor.totalBytes),
            note,
            fitness.reason,
        ]
        .compactMap { $0 }
        .joined(separator: ". ")
    }
}

/// Normal storage surfaces use a human-scale decimal value and name absence as
/// unknown. Converting nil to zero would turn "the API did not answer" into a
/// measured claim that the drive is full.
enum StorageCapacityPresentation {
    static func value(_ bytes: Int64?) -> String {
        guard let bytes, bytes > 0 else { return "capacity unknown" }
        return MRFormat.bytesDecimal(bytes)
    }

    static func freeSpace(_ bytes: Int64?) -> String {
        guard let bytes else { return "free space unknown" }
        if bytes <= 0 { return "no free space" }
        return "\(MRFormat.bytesDecimal(bytes)) free"
    }

    static func reclaimableTotal(_ bytes: Int64) -> String {
        "Up to \(MRFormat.bytesDecimal(bytes)) available after cleanup"
    }
}

enum StorageVolumePresentation {
    static func systemImage(isInternal: Bool?, isSelected: Bool) -> String {
        let base = isInternal == true ? "internaldrive" : "externaldrive"
        return isSelected ? "\(base).fill" : base
    }
}

/// One row per volume, with the badges the storage layer actually reports and a
/// refusal that is always named.
struct VolumeRow: View {
    let volume: VolumeDescriptor
    let requiredBytes: UInt64
    let headroomBytes: UInt64
    /// Supplied by the destination screen so displayed copy and the enabled
    /// state come from the same checked `StorageManaging.canHold` verdict.
    var capacityVerdict: SpaceVerdict? = nil
    var isSelected = false

    private var verdict: SpaceVerdict {
        if let capacityVerdict { return capacityVerdict }
        if volume.writeRefusal != nil {
            return .refused(.destinationIsReadOnlyVolume(path: volume.mountPath))
        }
        guard let dependable = volume.space.dependableBytes else {
            return .unknown(
                reason: "this volume reported no available capacity; Minirun will not treat an "
                    + "unprobed volume as an empty one")
        }
        guard dependable >= 0 else {
            return .unknown(reason: "this volume reported an unusable negative capacity")
        }
        let (need, overflowed) = requiredBytes.addingReportingOverflow(headroomBytes)
        if overflowed || need > UInt64(dependable) {
            return .refused(
                .insufficientFreeSpace(
                    needBytes: requiredBytes, headroomBytes: headroomBytes,
                    availableBytes: dependable, volume: volume.name))
        }
        return .fits(spareBytes: dependable - Int64(need))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
                Image(
                    systemName: StorageVolumePresentation.systemImage(
                        isInternal: volume.isInternal, isSelected: isSelected)
                )
                    .foregroundStyle(isSelected ? MRColor.tierPinned : MRColor.tertiary)
                    .accessibilityHidden(true)
                Text(volume.name ?? volume.mountPath)
                    .font(MRType.headline)
                    .foregroundStyle(MRColor.primary)
                if let type = volume.filesystemType {
                    Text(type).font(MRType.micro).foregroundStyle(MRColor.tertiary)
                }
                Spacer()
                if let dependable = volume.space.dependableBytes {
                    ValueText(
                        text: StorageCapacityPresentation.freeSpace(dependable),
                        provenance: .measured)
                } else {
                    Text(StorageCapacityPresentation.freeSpace(nil))
                        .font(MRType.metric)
                        .foregroundStyle(MRColor.caution)
                }
            }

            if !traits.isEmpty {
                Text(traits.joined(separator: " · "))
                    .font(MRType.micro)
                    .foregroundStyle(volume.space.isReadOnly ? MRColor.caution : MRColor.secondary)
            }

            Text(volume.mountPath)
                .font(MRType.micro)
                .foregroundStyle(MRColor.tertiary)
                .lineLimit(1)
                .truncationMode(.middle)

            if let optimistic = volume.space.optimisticBytes,
                let dependable = volume.space.dependableBytes, optimistic > dependable
            {
                Text(StorageCapacityPresentation.reclaimableTotal(optimistic))
                .font(MRType.micro)
                .foregroundStyle(MRColor.tertiary)
            }

            if requiredBytes > 0 || headroomBytes > 0 {
                refusalLine
            } else if let refusal = volume.writeRefusal {
                Text(refusal)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.caution)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(MRSpace.s3)
        .frame(maxWidth: .infinity, alignment: .leading)
        .mrCard(
            isSelected ? MRColor.raised : MRColor.panel,
            stroke: isSelected ? MRColor.tierPinned : MRColor.hairline)
        .accessibilityElement(children: .combine)
        .accessibilityValue(spoken)
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }

    @ViewBuilder private var refusalLine: some View {
        switch verdict {
        case .fits(let spare):
            Text("fits — \(MRFormat.bytesDecimal(spare)) to spare after the headroom")
                .font(MRType.micro)
                .foregroundStyle(MRColor.ok)
        case .refused(let error):
            VStack(alignment: .leading, spacing: 2) {
                Text(refusalSentence(error))
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.refuse)
                    .fixedSize(horizontal: false, vertical: true)
                Text(error.description)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .textSelection(.enabled)
                    .lineLimit(2)
            }
        case .unknown(let reason):
            Text(reason)
                .font(MRType.micro)
                .foregroundStyle(MRColor.caution)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func refusalSentence(_ error: DownloadError) -> String {
        switch error {
        case .destinationIsReadOnlyVolume:
            return volume.writeRefusal ?? "this volume is read-only"
        case .insufficientFreeSpace(let need, let headroom, let available, _):
            let (total, overflowed) = need.addingReportingOverflow(headroom)
            guard !overflowed, available >= 0, total > UInt64(available) else {
                return "needs \(MRFormat.bytesDecimal(need)) plus "
                    + "\(MRFormat.bytesDecimal(headroom)) headroom; "
                    + "\(MRFormat.bytesDecimal(available)) available"
            }
            return "needs \(MRFormat.bytesDecimal(need)); \(MRFormat.bytesDecimal(available)) "
                + "available — short by \(MRFormat.bytesDecimal(total - UInt64(available)))"
        default:
            return error.description
        }
    }

    private var spoken: String {
        switch verdict {
        case .fits(let spare): return "fits, \(MRFormat.bytesDecimal(spare)) spare"
        case .refused(let error): return "refused: \(refusalSentence(error))"
        case .unknown(let reason): return "unknown: \(reason)"
        }
    }

    private var traits: [String] {
        var result: [String] = []
        if volume.isInternal == true { result.append("Internal") }
        if volume.space.isRemovable { result.append("Removable") }
        if volume.isEjectable == true { result.append("Ejectable") }
        if volume.space.isReadOnly { result.append("Read-only") }
        return result
    }
}

/// A navigation affordance that reads as a line of this page rather than a
/// lone bordered rectangle floating under a card.
///
/// `NavigationLink { … } label: { Text(…) }` renders on macOS as a button with
/// a border, which is what "Download details" used to be: a control shaped like
/// nothing else on the screen, detached from the card whose details it opens.
/// The rest of the product spells the same idea as a title and the chevron
/// every list row uses, so this does too.
struct DisclosureLink<Destination: View>: View {
    let title: String
    @ViewBuilder let destination: () -> Destination

    var body: some View {
        NavigationLink {
            destination()
        } label: {
            HStack(spacing: MRSpace.s1) {
                Text(title)
                    .font(MRType.caption)
                    .foregroundStyle(MRColor.secondary)
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(MRColor.tertiary)
                    .accessibilityHidden(true)
            }
            // Still a 44-pt target; it simply stops looking like a button.
            .frame(minHeight: 44)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

/// What a stopped transfer left on the drive, as a card is entitled to say it.
///
/// Every case here comes from a reconciliation that happened, or says that one
/// has not happened yet. The card never derives this from the progress snapshot
/// it is holding: that snapshot is a memory of the last event, and after the
/// operator deletes the directory it is a memory of a drive that has changed.
enum TransferRemains: Equatable, Sendable {
    /// The drive has not been looked at yet.
    case checking
    /// Nothing was ever written anywhere: there is no destination to examine.
    case noDestination
    case driveNotConnected(volume: String)
    case unreadable(reason: String)
    case nothingKept
    case kept(completeFiles: Int, partialFiles: Int, bytes: UInt64)

    static func resolve(_ kept: KeptFiles?, destinationPath: String?) -> TransferRemains {
        guard let kept else {
            return destinationPath == nil ? .noDestination : .checking
        }
        switch kept.destination {
        case .volumeNotMounted(let volume):
            return .driveNotConnected(volume: volume)
        case .unreadable(let reason):
            return .unreadable(reason: reason)
        case .missing:
            return .nothingKept
        case .present:
            guard !kept.isEmpty else { return .nothingKept }
            return .kept(
                completeFiles: kept.completeFileCount,
                partialFiles: kept.partialFileCount, bytes: kept.bytes)
        }
    }
}

/// The sentences a stopped transfer is allowed to print, and the one action it
/// offers. Separated from the view so the claim can be tested without pixels —
/// this is exactly the surface that shipped a lie.
enum TransferRemainsPresentation {
    static func sentence(_ remains: TransferRemains) -> String {
        switch remains {
        case .checking:
            return "Checking what this transfer left on the drive…"
        case .noDestination:
            return "This transfer never reached a drive."
        case .driveNotConnected:
            return "The drive this transfer used is not connected."
        case .unreadable:
            return "The folder this transfer used could not be read."
        case .nothingKept:
            return "Nothing from this transfer is on the drive any more."
        case .kept(let complete, let partial, let bytes):
            let size = MRFormat.bytesDecimal(bytes)
            guard complete > 0 else {
                return "\(MRFormat.grouped(partial)) partial file"
                    + "\(partial == 1 ? "" : "s") kept on disk (\(size))."
            }
            let kept = "\(MRFormat.grouped(complete)) file\(complete == 1 ? "" : "s") "
                + "kept on disk (\(size))"
            guard partial > 0 else { return "\(kept)." }
            return "\(kept), \(MRFormat.grouped(partial)) partial."
        }
    }

    /// The named condition, where there is one. Never a second sentence for a
    /// state that already said everything it knows.
    static func detail(_ remains: TransferRemains) -> String? {
        switch remains {
        case .driveNotConnected(let volume):
            return volume.isEmpty ? nil : "Reconnect \(volume) to continue this transfer."
        case .unreadable(let reason):
            return reason
        case .checking, .noDestination, .nothingKept, .kept:
            return nil
        }
    }

    /// A cancelled transfer has no speed and no ETA, so the button is the only
    /// thing left to be exact about: it may not offer to continue with files
    /// nobody found.
    static func actionTitle(_ remains: TransferRemains) -> String {
        switch remains {
        case .nothingKept: return "Download again…"
        case .noDestination: return "Download…"
        case .checking, .driveNotConnected, .unreadable, .kept:
            return "Continue with kept files…"
        }
    }

    static func actionIsEnabled(_ remains: TransferRemains) -> Bool {
        switch remains {
        // Unreadable stays enabled: the action opens the destination picker,
        // and a fresh grant from it is exactly what an unreadable folder
        // needs. Only a drive that is not there, or a look not yet taken,
        // leaves nothing for the button to do.
        case .checking, .driveNotConnected: return false
        case .noDestination, .nothingKept, .kept, .unreadable: return true
        }
    }

    /// The per-file position, kept only while the reconciliation agrees with
    /// it. A snapshot that says "file 94 of 624" after 93 files were deleted is
    /// the same lie in smaller type.
    static func progressLine(
        _ remains: TransferRemains, snapshot: ProgressSnapshot?
    ) -> String? {
        guard case .kept(let complete, _, _) = remains, let snapshot,
            snapshot.filesTotal > 0, complete == snapshot.filesDone
        else { return nil }
        return "file \(min(complete + 1, snapshot.filesTotal)) of \(snapshot.filesTotal)"
    }

    /// The bar, drawn from the bytes that are there now. Nil where nothing has
    /// been measured: an empty bar is a claim too.
    static func keptFraction(_ remains: TransferRemains, totalBytes: UInt64) -> Double? {
        switch remains {
        case .checking, .noDestination, .driveNotConnected, .unreadable:
            return nil
        case .nothingKept:
            return 0
        case .kept(_, _, let bytes):
            guard totalBytes > 0 else { return 0 }
            return min(1, Double(bytes) / Double(totalBytes))
        }
    }
}
