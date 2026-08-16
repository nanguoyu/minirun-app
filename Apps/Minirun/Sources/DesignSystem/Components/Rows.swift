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
    static func trailingArgument(
        fitness: PlatformFitness, lastRunSecondsPerToken: Double? = nil
    ) -> String? {
        if fitness.verdict == .noRunner {
            return "Storage and verification only"
        }
        if let seconds = lastRunSecondsPerToken, seconds.isFinite, seconds > 0 {
            return "Last run \(MRFormat.clock(seconds))/token"
        }
        return nil
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

    var body: some View {
        Group {
            if let assetName = identity.assetName {
                Image(assetName)
                    .resizable()
                    .renderingMode(.original)
                    .scaledToFit()
                    .padding(2)
            } else {
                Image(systemName: fallbackSystemName)
                    .font(.system(size: 17, weight: .medium))
                    .foregroundStyle(fallbackColor)
            }
        }
        .frame(width: 28, height: 28)
        .accessibilityHidden(true)
    }
}

/// One catalog row. Rows, not cards — the catalog is a list of facts.
///
/// The trailing bottom line is the catalog's whole argument and is never cut:
/// what the model needs, and what it would cost here.
struct ModelRow: View {
    let entry: CatalogEntry
    let state: DownloadState
    let fitness: PlatformFitness
    /// Contextual action or evidence supplied by the owning surface. Local
    /// inventory uses an actual last-run time; the remote browser uses a clear
    /// Open/Download action. Nil leaves the row deliberately quiet.
    let trailingText: String?
    /// Every place on this machine the model was actually found. One pill each,
    /// because two copies on two drives is a fact worth seeing and a single
    /// "installed" badge hides it.
    var installations: [DiscoveredArtifact] = []
    /// Volume display name by location path, so a pill can say "On K3NVME"
    /// rather than repeating a mount path.
    var locationNames: [String: String] = [:]
    /// Whether the app has been granted any folder to scan at all.
    ///
    /// False changes what this row is entitled to claim. A cloud glyph means
    /// "published, not fetched"; with no registered folder the app has not
    /// looked at a single byte of anybody's disk, and saying "not downloaded"
    /// there is a guess wearing a state's clothes.
    var hasStorageLocation = true

    #if os(iOS)
        @Environment(\.horizontalSizeClass) private var sizeClass
        private var isNarrow: Bool { sizeClass == .compact }
    #else
        private var isNarrow: Bool { false }
    #endif

    private var isDimmed: Bool { fitness.verdict == .refused }
    private var leadingState: ModelRowLeadingState {
        .resolve(
            installations: installations, hasStorageLocation: hasStorageLocation,
            downloadState: state)
    }

    var body: some View {
        Group {
            #if os(macOS)
                ViewThatFits(in: .horizontal) {
                    wide
                    narrow
                }
            #else
                if isNarrow { narrow } else { wide }
            #endif
        }
        .padding(.vertical, MRSpace.s2)
        .opacity(isDimmed ? 0.62 : 1)
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityLabel(entry.descriptor.displayName)
        .accessibilityValue(spoken)
    }

    /// Two columns: identity on the left, the argument on the right.
    private var wide: some View {
        HStack(alignment: .top, spacing: MRSpace.s3) {
            glyph
            identity
            Spacer(minLength: MRSpace.s3)
            if let trailingText {
                VStack(alignment: .trailing, spacing: MRSpace.s1) {
                    Text(trailingText)
                        .font(MRType.micro)
                        .foregroundStyle(MRColor.tertiary)
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                }
            }
        }
    }

    /// One column on a phone. The argument wraps onto its own line rather than
    /// squeezing the model's name into a vertical alphabet — it is the row's
    /// whole point and must not be cut.
    private var narrow: some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            HStack(alignment: .top, spacing: MRSpace.s3) {
                glyph
                identity
                Spacer(minLength: MRSpace.s2)
            }
            if let trailingText {
                Text(trailingText)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var glyph: some View {
        ModelPublisherMark(
            identity: .resolve(
                modelID: entry.id,
                displayName: entry.descriptor.displayName,
                repositoryID: entry.descriptor.source.repo?.repoID),
            fallbackSystemName: leadingState.glyphName,
            fallbackColor: glyphColor)
    }

    /// Nothing is known about this model's presence, and the reason is that
    /// nowhere has been registered to look.
    private var isUnknownForWantOfALocation: Bool {
        leadingState == .locationUnknown
    }

    private var identity: some View {
        VStack(alignment: .leading, spacing: MRSpace.s1) {
            Text(entry.descriptor.displayName)
                .font(MRType.headline)
                .foregroundStyle(MRColor.primary)
                .fixedSize(horizontal: false, vertical: true)
            subtitleLine
            if let transferStatus = state.compactTransferStatus {
                Text(transferStatus)
                    .font(MRType.micro)
                    .foregroundStyle(transferStatusColor)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if fitness.verdict == .noRunner {
                StatusChip(text: "Not supported by this version", tone: .neutral)
            } else if fitness.verdict == .runnableWithCaveats {
                StatusChip(text: "Runs with limits", tone: .caution)
            } else if isDimmed {
                StatusChip(text: "Won't run on this device", tone: .refuse)
            }
            installationPills
        }
    }

    /// One pill per place the artifact was found: where it is, whether it is
    /// complete, and how hard anybody has looked at it.
    ///
    /// A row states the model size once. Per-copy byte details live in model
    /// detail; when the count falls short of what the catalog expects, the pill
    /// says how far short instead of calling a partial download installed.
    @ViewBuilder private var installationPills: some View {
        if isUnknownForWantOfALocation {
            StatusChip(
                text: "no storage location", tone: .caution,
                systemImage: "externaldrive.badge.questionmark")
        }
        if !installations.isEmpty {
            FlowRow(spacing: MRSpace.s2) {
                ForEach(installations) { artifact in
                    StatusChip(
                        text: pillText(artifact),
                        tone: pillTone(artifact),
                        systemImage: "externaldrive")
                }
            }
        }
    }

    private func pillText(_ artifact: DiscoveredArtifact) -> String {
        let place = locationNames[artifact.locationPath] ?? artifact.locationPath
        var parts = ["On \(place)"]
        if let missing = artifact.missingFileCount, missing > 0 {
            parts.append("\(MRFormat.grouped(missing)) files short")
        } else {
            parts.append(artifact.verification.label)
        }
        return parts.joined(separator: " · ")
    }

    private func pillTone(_ artifact: DiscoveredArtifact) -> StatusChip.Tone {
        if let missing = artifact.missingFileCount, missing > 0 { return .caution }
        switch artifact.verification {
        case .unverified: return .neutral
        case .spotChecked: return .verify
        case .fullyVerified: return .ok
        }
    }

    private var glyphColor: Color {
        switch leadingState {
        case .locationUnknown, .scannedIncomplete: return MRColor.caution
        case .scannedUnverified: return MRColor.tertiary
        case .scannedSpotChecked: return MRColor.verify
        case .scannedFullyVerified: return MRColor.ok
        case .download(let state):
            switch state {
            case .ready: return MRColor.ok
            case .incomplete, .failed: return MRColor.refuse
            case .verifying: return MRColor.verify
            case .active, .resolvingIndex: return MRColor.tierPinned
            default: return MRColor.tertiary
            }
        }
    }

    private var transferStatusColor: Color {
        switch state {
        case .failed, .incomplete: return MRColor.refuse
        case .interrupted: return MRColor.caution
        case .ready: return MRColor.ok
        case .verifying: return MRColor.verify
        case .active, .resolvingIndex: return MRColor.tierPinned
        default: return MRColor.secondary
        }
    }

    /// Published size, and nothing else.
    ///
    /// This line used to carry architecture, layer count and file count — three
    /// facts a person browsing a list of four models cannot act on, and which
    /// the detail screen already states properly. Repository and tool
    /// identifiers are source details, not the model's identity.
    private var subtitleLine: some View {
        ValueText(
            text: MRFormat.publishedBytes(entry.descriptor.totalBytes),
            provenance: .declared, font: MRType.declared)
    }

    private var subtitle: String {
        MRFormat.publishedBytes(entry.descriptor.totalBytes)
    }

    private var spoken: String {
        [leadingState.accessibilityLabel, subtitle, trailingText, fitness.reason]
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

/// The download card. `paused` and `interrupted` carry the same visual weight
/// as `active` — a drive unplugged at 900 GB is a Tuesday, not a disaster.
/// Only `failed` is red.
struct DownloadCard: View {
    let modelName: String
    let state: DownloadState
    let declaredBytes: UInt64
    var destinationPath: String? = nil
    var updatedAt: Date? = nil
    var operationError: String? = nil
    var onPause: (() -> Void)? = nil
    var onResume: (() -> Void)? = nil
    var onCancel: (() -> Void)? = nil
    var onVerify: (() -> Void)? = nil
    var onCancelVerification: (() -> Void)? = nil
    var onChooseVolume: (() -> Void)? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            header
            destination
            switch state {
            case .notStarted, .awaitingDestination:
                Text("Nothing transferred yet.")
                    .font(MRType.caption)
                    .foregroundStyle(MRColor.secondary)
            case .resolvingIndex:
                ProgressView().controlSize(.small)
                Text("Resolving the index and walking the tree…")
                    .font(MRType.caption)
                    .foregroundStyle(MRColor.secondary)
            case .active(let progress), .paused(let progress):
                progressBody(progress)
            case .interrupted(let progress, let reason):
                progressBody(progress)
                interruption(reason, progress)
            case .verifying(let phase, _):
                VerificationProgress(phase: phase)
            case .cancelled(let progress):
                progressBody(progress)
                Text("Cancelled. Verified and partial files remain on disk for a new job to reuse.")
                    .font(MRType.caption)
                    .foregroundStyle(MRColor.secondary)
            case .ready(let measured, let fileCount):
                readyBody(measured: measured, fileCount: fileCount)
            case .incomplete(let files, let bytes):
                incompleteBody(files: files, bytes: bytes)
            case .failed(let error):
                NamedErrorCard(
                    headline: "The transfer failed.",
                    message: "Nothing was lost that had already been verified.",
                    namedError: error.description)
            }
            if let operationError {
                NamedErrorCard(
                    headline: "The last action did not finish.",
                    message: "The transfer state was kept unchanged.",
                    namedError: operationError,
                    tone: .caution)
            }
            actions
        }
        .padding(MRSpace.s4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .mrCard(MRColor.panel, stroke: strokeColor)
        .accessibilityElement(children: .contain)
    }

    private var strokeColor: Color {
        if case .failed = state { return MRColor.refuse.opacity(0.5) }
        return MRColor.hairline
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(modelName).font(MRType.headline).foregroundStyle(MRColor.primary)
            Spacer()
            stateChip
        }
    }

    @ViewBuilder private var destination: some View {
        if destinationPath != nil || updatedAt != nil {
            HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
                Image(systemName: "folder")
                    .foregroundStyle(MRColor.tertiary)
                    .accessibilityHidden(true)
                if let destinationPath {
                    Text(destinationPath)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                }
                Spacer(minLength: MRSpace.s2)
                if let updatedAt {
                    Text(MRFormat.timestamp(updatedAt))
                }
            }
            .font(MRType.micro)
            .foregroundStyle(MRColor.tertiary)
        }
    }

    private var stateChip: StatusChip {
        switch state {
        case .notStarted, .awaitingDestination: return StatusChip(text: "not downloaded")
        case .resolvingIndex: return StatusChip(text: "resolving index")
        case .active: return StatusChip(text: "transferring", tone: .neutral)
        case .paused: return StatusChip(text: "paused", tone: .neutral)
        case .interrupted: return StatusChip(text: "interrupted", tone: .neutral)
        case .verifying: return StatusChip(text: "verifying", tone: .verify)
        case .cancelled: return StatusChip(text: "cancelled", tone: .neutral)
        case .ready: return StatusChip(text: "payload checked", tone: .ok)
        case .incomplete: return StatusChip(text: "incomplete", tone: .caution)
        case .failed: return StatusChip(text: "failed", tone: .refuse)
        }
    }

    private func progressBody(_ progress: ProgressSnapshot) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            ProgressView(value: progress.fraction)
                .tint(MRColor.tierPinned)
            HStack {
                Text(
                    "\(MRFormat.bytesDecimal(progress.verifiedBytes)) of "
                        + "\(MRFormat.bytesDecimal(progress.totalBytes))"
                )
                .font(MRType.metric)
                .foregroundStyle(MRColor.primary)
                .animation(nil, value: progress.verifiedBytes)
                Spacer()
                BalanceDot(
                    balanced: progress.accountsForEveryByte,
                    failingIdentity:
                        "verified + fetched + in flight + remaining ≠ plan total")
            }
            if let sentence = progress.resumeSentence {
                Text(sentence)
                    .font(MRType.micro)
                    .foregroundStyle(MRColor.tertiary)
                    .lineLimit(1)
            }
            HStack(spacing: MRSpace.s3) {
                Text(MRFormat.throughput(progress.bytesPerSecond))
                Text("ETA \(MRFormat.duration(progress.estimatedTimeRemaining))")
            }
            .font(MRType.micro)
            .foregroundStyle(MRColor.secondary)
        }
    }

    private func interruption(
        _ reason: InterruptionReason, _ progress: ProgressSnapshot
    ) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s1) {
            Text("Interrupted — \(reason.sentence)")
                .font(MRType.body)
                .foregroundStyle(MRColor.primary)
                .fixedSize(horizontal: false, vertical: true)
            Text(
                "\(MRFormat.bytesDecimal(progress.verifiedBytes)) of "
                    + "\(MRFormat.bytesDecimal(progress.totalBytes)) kept. "
                    + reason.recoverySentence
            )
            .font(MRType.caption)
            .foregroundStyle(MRColor.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func readyBody(measured: UInt64, fileCount: Int) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s2) {
            HStack(alignment: .firstTextBaseline, spacing: MRSpace.s2) {
                ValueText(
                    text: MRFormat.bytesDecimal(measured), provenance: .measured,
                    font: MRType.readout)
                Text("measured").mrLabel(MRColor.ok)
            }
            Text(
                "\(MRFormat.grouped(fileCount)) planned files checked; payload digests verified."
            )
                .font(MRType.caption)
                .foregroundStyle(MRColor.secondary)
            if measured != declaredBytes {
                Text(
                    "Published size: \(MRFormat.publishedBytes(declaredBytes)). Measured on "
                        + "disk: \(MRFormat.measuredBytes(measured))."
                )
                .font(MRType.micro)
                .foregroundStyle(MRColor.caution)
                .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func incompleteBody(files: Int, bytes: UInt64) -> some View {
        NamedErrorCard(
            headline: "\(files) file\(files == 1 ? "" : "s") did not verify.",
            message:
                "Those bytes are not trusted and will be re-fetched from zero. A byte that "
                + "failed a digest is not a byte to resume from.",
            namedError: "verificationIncomplete(pathsToRefetch: \(files))",
            tone: .caution,
            numbers: [("data to re-fetch", MRFormat.measuredBytes(bytes))])
    }

    @ViewBuilder private var actions: some View {
        HStack(spacing: MRSpace.s2) {
            switch state {
            case .notStarted, .awaitingDestination:
                if let onChooseVolume {
                    Button("Download…", action: onChooseVolume)
                        .buttonStyle(.borderedProminent)
                }
            case .active:
                if let onPause { Button("Pause", action: onPause) }
                if let onCancel { Button("Cancel", role: .destructive, action: onCancel) }
            case .paused:
                if let onResume {
                    Button("Resume", action: onResume).buttonStyle(.borderedProminent)
                }
                if let onCancel { Button("Cancel", role: .destructive, action: onCancel) }
            case .interrupted, .cancelled:
                if let onChooseVolume {
                    Button("Continue with kept files…", action: onChooseVolume)
                        .buttonStyle(.borderedProminent)
                }
            case .incomplete:
                if let onVerify {
                    Button("Re-verify", action: onVerify).buttonStyle(.borderedProminent)
                }
            case .ready:
                if let onVerify { Button("Re-verify", action: onVerify) }
            case .verifying:
                if let onCancelVerification {
                    Button("Cancel verification", role: .cancel, action: onCancelVerification)
                }
            case .failed:
                if let onChooseVolume {
                    Button("Continue with kept files…", action: onChooseVolume)
                        .buttonStyle(.borderedProminent)
                }
            default:
                EmptyView()
            }
        }
        .controlSize(.small)
        .font(MRType.caption)
    }
}
