import Foundation
import MinirunKit
import MinirunRunners

/// The evidence attached to a tokenizer before it may enter the product
/// runtime registry.
///
/// A type conforming to `PromptTokenizing` proves only that it can return
/// integers. It does not prove that those integers belong to this model, that
/// the vocabulary is the pinned one, or that the model's chat template was
/// applied. Product preparation therefore records the current catalog's
/// immutable artifact revision plus the tokenizer source revision and SHA-256
/// which were verified together. None of these are compiled version locks.
struct VerifiedTokenizerIdentity: Equatable, Sendable {
    let model: ModelID
    let artifactRepository: String
    let artifactRevision: String
    let sourceRepository: String
    let sourceRevision: String
    let sha256: String

    init(
        model: ModelID, artifactRepository: String, artifactRevision: String,
        sourceRepository: String, sourceRevision: String, sha256: String
    ) {
        precondition(!artifactRepository.isEmpty, "a verified tokenizer needs an artifact repository")
        precondition(
            artifactRevision.count == 40,
            "a verified tokenizer needs a pinned artifact revision")
        precondition(!sourceRepository.isEmpty, "a verified tokenizer needs a source repository")
        precondition(sourceRevision.count == 40, "a verified tokenizer needs a pinned source revision")
        precondition(
            sha256.count == 64 && sha256.allSatisfy(\.isHexDigit),
            "a verified tokenizer needs a complete SHA-256")
        self.model = model
        self.artifactRepository = artifactRepository
        self.artifactRevision = artifactRevision
        self.sourceRepository = sourceRepository
        self.sourceRevision = sourceRevision
        self.sha256 = sha256.lowercased()
    }
}

/// A tokenizer that is eligible for the launched product. PreviewTokenizer does
/// not conform: attaching evidence to the implementation that consumes the
/// vocabulary makes it impossible to pass the stand-in to `verified(...)` by
/// accident and provide unrelated evidence beside it.
protocol VerifiedPromptTokenizing: PromptTokenizing {
    var verifiedIdentity: VerifiedTokenizerIdentity { get }
}

/// Marker for a runner integration intended to execute real model code.
/// `MockDecodeRunner` deliberately does not conform, so the verified runtime
/// constructor cannot accept it without an explicit, review-visible violation.
protocol ProductRunnerFacade: RunnerFacade {}

/// One complete execution binding for one model.
///
/// Runner and tokenizer are deliberately inseparable here. Registering only a
/// runner would invite a caller to fall back to `PreviewTokenizer`; registering
/// only a tokenizer would make catalogue metadata look executable. Neither is
/// a runtime, so neither can make a model available.
struct ModelRuntime {
    enum Trust: Equatable, Sendable {
        case verified
        #if DEBUG
            case preview
        #endif

        var isVerified: Bool {
            if self == .verified { return true }
            return false
        }
    }

    let capabilities: RunnerCapabilities
    let trust: Trust
    private let artifactCompatibility: @Sendable (ArtifactIndexIdentity) -> Bool
    private let preparation: @MainActor (
        ArtifactReference, Conversation
    ) throws -> PreparedModelRuntime

    static func verified(
        capabilities: RunnerCapabilities,
        acceptsArtifact: @escaping @Sendable (ArtifactIndexIdentity) -> Bool = { _ in true },
        inspectArtifact: @escaping @MainActor (
            ArtifactReference
        ) throws -> PreparedArtifactRuntimeInspection = { _ in
            PreparedArtifactRuntimeInspection(census: nil)
        },
        makeTokenizer: @escaping @MainActor (
            ArtifactReference
        ) throws -> any VerifiedPromptTokenizing,
        makeRunner: @escaping @MainActor (Conversation) -> any ProductRunnerFacade
    ) -> ModelRuntime {
        return ModelRuntime(
            capabilities: capabilities,
            trust: .verified,
            artifactCompatibility: acceptsArtifact,
            preparation: { artifact, conversation in
                let tokenizer = try makeTokenizer(artifact)
                precondition(
                    tokenizer.verifiedIdentity.model == capabilities.model,
                    "tokenizer and runner must name the same model")
                let inspection = try inspectArtifact(artifact)
                return PreparedModelRuntime(
                    tokenizer: tokenizer, runner: makeRunner(conversation),
                    artifactCensus: inspection.census,
                    deepSeekV4Ladder: inspection.deepSeekV4Ladder,
                    progressLayerCount: inspection.progressLayerCount,
                    workingSetReserveBytes: inspection.workingSetReserveBytes)
            })
    }

    #if DEBUG
        /// Preview/test construction is separate and named. A preview runtime
        /// can never enter a verified registry, and this constructor does not
        /// exist in a Release binary.
        static func preview(
            capabilities: RunnerCapabilities,
            tokenizer: any PromptTokenizing,
            inspectArtifact: @escaping @MainActor (
                ArtifactReference
            ) throws -> PreparedArtifactRuntimeInspection = { _ in
                PreparedArtifactRuntimeInspection(census: nil)
            },
            makeRunner: @escaping @MainActor (Conversation) -> any RunnerFacade
        ) -> ModelRuntime {
            ModelRuntime(
                capabilities: capabilities,
                trust: .preview,
                artifactCompatibility: { _ in true },
                preparation: { artifact, conversation in
                    let inspection = try inspectArtifact(artifact)
                    return PreparedModelRuntime(
                        tokenizer: tokenizer, runner: makeRunner(conversation),
                        artifactCensus: inspection.census,
                        deepSeekV4Ladder: inspection.deepSeekV4Ladder,
                        progressLayerCount: inspection.progressLayerCount,
                        workingSetReserveBytes: inspection.workingSetReserveBytes)
                })
        }
    #endif

    @MainActor
    func prepare(
        artifact: ArtifactReference, for conversation: Conversation
    ) throws -> PreparedModelRuntime {
        try preparation(artifact, conversation)
    }

    /// A cheap structural gate used before the product offers a model as
    /// runnable. Byte-level authority is still established by full artifact
    /// verification and repeated by `prepare`; this prevents an older copy
    /// which lacks newly required runtime files from looking ready first.
    func accepts(_ artifact: DiscoveredArtifact) -> Bool {
        artifactCompatibility(artifact.index)
    }
}

struct PreparedArtifactRuntimeInspection {
    let census: ArtifactCensus?
    /// DeepSeek V4's ladder input, for the runtime whose planner needs the
    /// kit's second census rather than `ArtifactCensus`.
    ///
    /// A separate field and not a second `ArtifactCensus`, for the reason
    /// `docs/design/v4-memory-dial.md` §3.1 gives: that type derives resident
    /// and saved from one array, and a V4 plan built through it is wrong by
    /// 3.1% in whichever direction the caller picks.
    let deepSeekV4Ladder: DeepSeekV4MemoryDialInputs.ArtifactProfile?
    /// Exact execution layers from the verified model configuration.
    ///
    /// This is separate from `census`: a runtime can report honest progress
    /// without claiming the per-layer residency geometry required by the
    /// memory planner. V4 is such a runtime; K3 derives both from one census.
    let progressLayerCount: Int?
    let workingSetReserveBytes: (@Sendable (_ promptTokens: Int, _ maximumNewTokens: Int) throws
        -> UInt64)?

    init(
        census: ArtifactCensus?,
        deepSeekV4Ladder: DeepSeekV4MemoryDialInputs.ArtifactProfile? = nil,
        progressLayerCount: Int? = nil,
        workingSetReserveBytes: (@Sendable (Int, Int) throws -> UInt64)? = nil
    ) {
        self.census = census
        self.deepSeekV4Ladder = deepSeekV4Ladder
        self.progressLayerCount = progressLayerCount
        self.workingSetReserveBytes = workingSetReserveBytes
    }
}

struct PreparedModelRuntime {
    let tokenizer: any PromptTokenizing
    let runner: any RunnerFacade
    /// Exact metadata from the rooted artifact this prepared runtime will
    /// consume. Nil for models whose planner has no artifact-specific census.
    let artifactCensus: ArtifactCensus?
    /// The same, for V4's own planner. Nil for every other model.
    let deepSeekV4Ladder: DeepSeekV4MemoryDialInputs.ArtifactProfile?
    /// Exact layer count used by live progress, when it is independently
    /// available without a planner census.
    let progressLayerCount: Int?
    /// Request-specific reserve arithmetic derived from the same opened model
    /// configuration. Nil for runtimes whose state does not grow with prompt
    /// or generation length.
    let workingSetReserveBytes: (@Sendable (Int, Int) throws -> UInt64)?
}

/// One independently reviewable product runtime contribution.
///
/// The registry is intentionally assembled from providers rather than from a
/// switch over known ModelIDs. Adding V4 or a future model therefore means
/// supplying one complete verified binding; it cannot require weakening K3's
/// evidence gate or teaching the catalogue about executable code.
struct ProductModelRuntimeProvider {
    let model: ModelID
    private let makeRuntime: () -> ModelRuntime

    init(model: ModelID, makeRuntime: @escaping () -> ModelRuntime) {
        self.model = model
        self.makeRuntime = makeRuntime
    }

    fileprivate func resolve() -> ModelRuntime {
        let runtime = makeRuntime()
        precondition(runtime.trust.isVerified, "a product provider must supply a verified runtime")
        precondition(
            runtime.capabilities.model == model,
            "a product provider's declared model must match its runner")
        return runtime
    }
}

/// The sole authority for whether this app build can execute a model.
///
/// Catalogue `descriptor.runner` is descriptive metadata and may outlive or
/// precede an actual integration. It must never turn into executable capability
/// on its own. The product-safe default is therefore an empty verified registry;
/// previews and tests opt into a separately constructed preview registry.
struct ModelRuntimeRegistry {
    private let runtimes: [ModelID: ModelRuntime]

    /// Product registry. Every supplied binding must carry tokenizer evidence;
    /// an empty registry remains the fail-closed default for tests and for
    /// product builds which deliberately omit all runtime integrations.
    init(verified runtimes: [ModelRuntime] = []) {
        precondition(
            runtimes.allSatisfy { $0.trust.isVerified },
            "preview runtimes cannot enter the product registry")
        self.runtimes = Self.index(runtimes)
    }

    init(providers: [ProductModelRuntimeProvider]) {
        self.init(verified: providers.map { $0.resolve() })
    }

    #if DEBUG
        /// Preview/test registry. It is absent from Release, so a product
        /// binary has no constructor that can admit a stand-in runtime.
        init(preview runtimes: [ModelRuntime]) {
            precondition(
                runtimes.allSatisfy { !$0.trust.isVerified },
                "verified and preview runtimes belong in different registries")
            self.runtimes = Self.index(runtimes)
        }
    #endif

    func runtime(for id: ModelID) -> ModelRuntime? { runtimes[id] }

    var registeredModelIDs: Set<ModelID> { Set(runtimes.keys) }

    private static func index(_ runtimes: [ModelRuntime]) -> [ModelID: ModelRuntime] {
        var result: [ModelID: ModelRuntime] = [:]
        for runtime in runtimes {
            let model = runtime.capabilities.model
            precondition(result[model] == nil, "duplicate runtime for \(model.rawValue)")
            result[model] = runtime
        }
        return result
    }
}
