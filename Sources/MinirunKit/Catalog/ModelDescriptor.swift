import Foundation
import StorageCore

/// Stable identity for a model, independent of which repository publishes it.
public struct ModelID: RawRepresentable, Hashable, Codable, Sendable, CustomStringConvertible {
    public let rawValue: String

    public init(rawValue: String) { self.rawValue = rawValue }
    public init(_ rawValue: String) { self.rawValue = rawValue }
    public var description: String { rawValue }

    public static let kimiK3 = ModelID("kimi-k3")
    public static let deepseekV4Flash = ModelID("deepseek-v4-flash")
    public static let minimaxH3 = ModelID("minimax-h3")

    /// Stable identity for a repository discovered after this build shipped.
    ///
    /// The namespace keeps an observed repository id from colliding with the
    /// product ids above. The spelling is deliberately preserved: changing
    /// case or normalising a publisher's name would invent an identity the
    /// catalogue did not observe.
    public static func discoveredHuggingFaceRepository(_ repoID: String) -> ModelID {
        ModelID("hf:\(repoID)")
    }
}

public enum ModelArchitecture: String, Codable, Sendable, CaseIterable {
    case kimiK3MoE
    case deepseekV4FlashMoE
    /// Retained only so a catalogue cached by an older build still decodes. No
    /// model of this architecture is published or offered by this build.
    case qwen3MoE
    case minimaxH3Video
    /// The repository is visible, but this build has no architecture adapter
    /// for it. This is a statement of missing knowledge, not an architecture.
    case unrecognized
}

/// How the published bytes are arranged on disk.
///
/// The layout is what a downloader has to know and a runner has to agree with;
/// it is not the architecture. Two models can share an architecture family and
/// still be laid out differently, and it is the layout that decides whether a
/// directory a user picked is the thing the runner expects.
public enum ArtifactLayout: String, Codable, Sendable, CaseIterable {
    /// `layerNN/` plus `global/`; `.mxfp4tile` and `-deterministic.bin`.
    case k3FlagshipLayerStreams
    /// `layersNN/`, `mtpNN/`, `global00/`; `.fp8tile`, `.mxfp4tile`, `.bin`.
    case v4FlashUnitBundle
    /// `dit-block-NN/`, `vae-*`, `adaln-cache/`; `.h3tile`, `.qstream`.
    case h3UnitBundle
    /// A locally built container set. Retained only so a catalogue cached by an
    /// older build still decodes; no published artifact uses this layout.
    case qwenContainerSet
    /// The repository is visible, but this build has no declared layout for
    /// it. Callers must not infer one from filenames.
    case unrecognized
}

/// A repository at one commit.
///
/// The revision is a commit and never a branch, and that is a correctness
/// property rather than a preference: every download URL embeds it, and a
/// branch name would let the bytes under a half-finished 1.5 TB transfer change
/// without a single response looking wrong.
public struct HuggingFaceRepoRef: Codable, Sendable, Hashable {
    public let repoID: String
    public let revision: String

    public init(repoID: String, revision: String) {
        self.repoID = repoID
        self.revision = revision
    }

    public var isPinnedCommit: Bool {
        revision.count == 40 && revision.allSatisfy { $0.isHexDigit && !$0.isUppercase }
    }

    /// Whether `repoID` is exactly the two-component Hugging Face spelling
    /// this package can safely place into URL paths and persisted plans.
    ///
    /// Percent escapes, backslashes, Unicode lookalikes, whitespace and
    /// controls are rejected rather than left to different URL/filesystem
    /// layers to interpret differently.
    public static func isCanonicalRepositoryID(_ repoID: String) -> Bool {
        let components = repoID.split(separator: "/", omittingEmptySubsequences: false)
        guard components.count == 2 else { return false }
        return components.allSatisfy { component in
            guard !component.isEmpty, component != ".", component != ".." else { return false }
            return component.utf8.allSatisfy { byte in
                (48...57).contains(byte)
                    || (65...90).contains(byte)
                    || (97...122).contains(byte)
                    || byte == 45 || byte == 46 || byte == 95
            }
        }
    }

    public func resolveURL(path: String, endpoint: URL = HuggingFaceEndpoint.default) -> URL {
        HuggingFaceRepoRef.url(endpoint, segments: [repoID, "resolve", revision, path])
    }

    public func treeURL(endpoint: URL = HuggingFaceEndpoint.default) -> URL {
        var components = URLComponents(
            url: HuggingFaceRepoRef.url(
                endpoint, segments: ["api", "models", repoID, "tree", revision]),
            resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "recursive", value: "1"),
            URLQueryItem(name: "expand", value: "1"),
        ]
        return components.url!
    }

    /// Join path segments onto an endpoint without a trailing slash.
    ///
    /// `URL.appendingPathComponent(_:isDirectory:)` is not used because the
    /// pieces being joined already contain separators — `repoID` is
    /// `owner/name`, `path` is `layer01/layer01-w1.mxfp4tile` — and the
    /// directory form leaves a trailing `/` that the tree API answers with a
    /// 404. Splitting on `/` and percent-encoding each piece keeps separators
    /// as separators and everything else escaped.
    static func url(_ endpoint: URL, segments: [String]) -> URL {
        var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)!
        let existing = components.percentEncodedPath.split(separator: "/").map(String.init)
        let pieces = (existing + segments.flatMap { $0.split(separator: "/").map(String.init) })
            .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? $0 }
        components.percentEncodedPath = "/" + pieces.joined(separator: "/")
        return components.url!
    }

    private enum CodingKeys: String, CodingKey { case repoID = "repoId", revision }
}

public enum ModelSource: Codable, Sendable, Equatable {
    case huggingFaceRepo(HuggingFaceRepoRef)
    /// No repository: the artifact is produced on the machine that runs it.
    case locallyBuilt(tool: String)

    public var repo: HuggingFaceRepoRef? {
        if case .huggingFaceRepo(let ref) = self { return ref }
        return nil
    }

    private enum CodingKeys: String, CodingKey { case kind, repo, tool }
    private enum Kind: String, Codable { case huggingFaceRepo, locallyBuilt }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(Kind.self, forKey: .kind) {
        case .huggingFaceRepo:
            self = .huggingFaceRepo(try container.decode(HuggingFaceRepoRef.self, forKey: .repo))
        case .locallyBuilt:
            self = .locallyBuilt(tool: try container.decode(String.self, forKey: .tool))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .huggingFaceRepo(let ref):
            try container.encode(Kind.huggingFaceRepo, forKey: .kind)
            try container.encode(ref, forKey: .repo)
        case .locallyBuilt(let tool):
            try container.encode(Kind.locallyBuilt, forKey: .kind)
            try container.encode(tool, forKey: .tool)
        }
    }
}

/// Whether this build can run the model, as opposed to merely fetch it.
public enum RunnerAvailability: String, Codable, Sendable {
    /// A `RunnerFacade` for this model is registered in this build.
    case decodeRunner
    /// The artifact is published and downloadable; nothing here runs it.
    case none
}

public enum Platform: String, Codable, Sendable { case macOS, iOS }

/// What the machine asking the question can offer.
public struct DeviceProfile: Codable, Sendable, Equatable {
    public let platform: Platform
    /// `os_proc_available_memory()` on iOS — the number jetsam judges against,
    /// not physical RAM. Physical memory on macOS, where nothing enforces a
    /// per-process ceiling.
    public let processMemoryBudgetBytes: UInt64
    public let hasMLXGPU: Bool
    public let hasIncreasedMemoryLimitEntitlement: Bool

    public init(
        platform: Platform, processMemoryBudgetBytes: UInt64,
        hasMLXGPU: Bool, hasIncreasedMemoryLimitEntitlement: Bool
    ) {
        self.platform = platform
        self.processMemoryBudgetBytes = processMemoryBudgetBytes
        self.hasMLXGPU = hasMLXGPU
        self.hasIncreasedMemoryLimitEntitlement = hasIncreasedMemoryLimitEntitlement
    }

    public static func current() -> DeviceProfile {
        #if os(iOS)
            let budget =
                ProcessFootprint.availableMemory()
                ?? ProcessInfo.processInfo.physicalMemory
            return DeviceProfile(
                platform: .iOS,
                processMemoryBudgetBytes: budget,
                hasMLXGPU: true,
                // Read from the built app's entitlements, which this package
                // cannot see; the app target overrides this when it knows.
                hasIncreasedMemoryLimitEntitlement: false)
        #else
            return DeviceProfile(
                platform: .macOS,
                processMemoryBudgetBytes: ProcessInfo.processInfo.physicalMemory,
                hasMLXGPU: true,
                hasIncreasedMemoryLimitEntitlement: false)
        #endif
    }
}

/// Whether a given device can run a given model, and why.
public struct PlatformFitness: Codable, Sendable, Equatable {
    public enum Verdict: String, Codable, Sendable {
        case runnable
        case runnableWithCaveats
        case refused
        case noRunner
    }

    public let verdict: Verdict
    /// One sentence, shown verbatim. Not a code the UI switches on.
    public let reason: String
    /// Bytes of volume the artifact needs. The storage picker filters on this;
    /// iPhone internal storage cannot hold K3 under any setting.
    public let requiredFreeBytes: UInt64
    /// `processMemoryBudgetBytes - minimumBudgetBytes`. Nil when the model has
    /// no measured minimum, because a headroom computed from a guess is a guess
    /// wearing a number's clothes.
    public let headroomBytes: Int64?

    public init(
        verdict: Verdict, reason: String, requiredFreeBytes: UInt64, headroomBytes: Int64?
    ) {
        self.verdict = verdict
        self.reason = reason
        self.requiredFreeBytes = requiredFreeBytes
        self.headroomBytes = headroomBytes
    }
}

/// Everything known about one publishable model before a byte is fetched.
public struct ModelDescriptor: Codable, Sendable, Equatable, Identifiable {
    public let id: ModelID
    public let displayName: String
    public let architecture: ModelArchitecture
    public let layout: ArtifactLayout
    public let source: ModelSource
    /// Files a run reads.
    public let payloadBytes: UInt64
    /// Manifests and root documents.
    public let metadataBytes: UInt64
    public var totalBytes: UInt64 {
        let (total, overflow) = payloadBytes.addingReportingOverflow(metadataBytes)
        return overflow ? .max : total
    }
    public let payloadFileCount: Int
    public let metadataFileCount: Int
    /// The largest single file, which is what sizes the download chunker's
    /// buffers and what decides whether a filesystem can hold the artifact at
    /// all.
    public let largestFileBytes: UInt64
    /// The smallest budget under which this model is *on record* as having run.
    ///
    /// A smaller request is refused with ``RunError/budgetBelowMinimum(declared:minimum:model:)``
    /// — never clamped, never waved through. Spec §5.3 asks for a stated
    /// budget, and a budget nobody has ever run under is not a statement about
    /// anything.
    public let minimumBudgetBytes: UInt64?
    public let runner: RunnerAvailability
    public let licenseName: String
    public let licenseAcknowledgementRequired: Bool
    /// Verbatim operator-facing sentences. Shown, never parsed.
    public let notes: [String]

    public init(
        id: ModelID, displayName: String, architecture: ModelArchitecture,
        layout: ArtifactLayout, source: ModelSource,
        payloadBytes: UInt64, metadataBytes: UInt64,
        payloadFileCount: Int, metadataFileCount: Int, largestFileBytes: UInt64,
        minimumBudgetBytes: UInt64?, runner: RunnerAvailability,
        licenseName: String, licenseAcknowledgementRequired: Bool, notes: [String]
    ) {
        self.id = id
        self.displayName = displayName
        self.architecture = architecture
        self.layout = layout
        self.source = source
        self.payloadBytes = payloadBytes
        self.metadataBytes = metadataBytes
        self.payloadFileCount = payloadFileCount
        self.metadataFileCount = metadataFileCount
        self.largestFileBytes = largestFileBytes
        self.minimumBudgetBytes = minimumBudgetBytes
        self.runner = runner
        self.licenseName = licenseName
        self.licenseAcknowledgementRequired = licenseAcknowledgementRequired
        self.notes = notes
    }

    public var totalFileCount: Int {
        let (total, overflow) = payloadFileCount.addingReportingOverflow(metadataFileCount)
        return overflow ? .max : total
    }

    public func fitness(on device: DeviceProfile) -> PlatformFitness {
        let required = totalBytes
        guard runner == .decodeRunner else {
            return PlatformFitness(
                verdict: .noRunner,
                reason:
                    "\(displayName) is published and downloadable, but no runner for it ships in this build.",
                requiredFreeBytes: required, headroomBytes: nil)
        }
        guard let minimum = minimumBudgetBytes else {
            return PlatformFitness(
                verdict: .runnableWithCaveats,
                reason:
                    "\(displayName) has a runner but no measured minimum budget, so no fit can be stated.",
                requiredFreeBytes: required, headroomBytes: nil)
        }
        let headroom = Self.saturatingSignedDifference(
            device.processMemoryBudgetBytes, minimum)
        guard device.processMemoryBudgetBytes >= minimum else {
            return PlatformFitness(
                verdict: .refused,
                reason:
                    "\(displayName) has only ever run at \(minimum) bytes, and this device offers \(device.processMemoryBudgetBytes).",
                requiredFreeBytes: required, headroomBytes: headroom)
        }
        if device.platform == .iOS && !device.hasIncreasedMemoryLimitEntitlement {
            return PlatformFitness(
                verdict: .runnableWithCaveats,
                reason:
                    "\(displayName) fits the reported budget, but without the increased-memory-limit entitlement iOS may lower it mid-run.",
                requiredFreeBytes: required, headroomBytes: headroom)
        }
        return PlatformFitness(
            verdict: .runnable,
            reason: "\(displayName) has run at \(minimum) bytes and this device reports more.",
            requiredFreeBytes: required, headroomBytes: headroom)
    }

    private static func saturatingSignedDifference(_ lhs: UInt64, _ rhs: UInt64) -> Int64 {
        if lhs >= rhs {
            let difference = lhs - rhs
            return difference > UInt64(Int64.max) ? .max : Int64(difference)
        }
        let difference = rhs - lhs
        return difference > UInt64(Int64.max) ? .min : -Int64(difference)
    }
}
