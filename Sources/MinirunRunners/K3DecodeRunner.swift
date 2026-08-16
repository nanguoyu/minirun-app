import BenchScenarios
import Foundation
import MinirunKit
import ModelAdapters

/// Kimi K3, decoded from a streamed artifact.
///
/// The decode machinery is not reimplemented here and not touched: this binds
/// ``K3Model`` — the same assembly `ArtifactFirstTokenTests.testFirstFullModelToken`
/// and `K3DecodeScenario` build, in the same order — to the event stream the app
/// consumes. What the facade adds is the four things a UI needs and a
/// `HarnessScenario` cannot give it: a token as each pass produces it, telemetry
/// while the pass is still running, a Stop that lands inside a forward pass, and
/// a teardown whose completion is part of the record.
///
/// ## Why this wraps `K3Model` and not `K3DecodeScenario`
///
/// `K3DecodeScenario.run` is one straight line from a published artifact to a
/// `ScenarioResult`: it translates the published layout unconditionally, and it
/// reports through `ScenarioContext` closures that carry a phase and a byte
/// count but never a token id. Wrapping it would have given a stream with no
/// tokens in it, runnable only against the 1.56 TB drive — nothing this Mac can
/// test. So the wrap is one level lower, at the model, and the scenario's
/// *numbers* are borrowed rather than copied: the budget floor, the working-set
/// reserve, the pool and evaluation defaults and the flagship config all come
/// from `K3DecodeScenario` itself, so the arm this runner runs is the arm the
/// M9C and M10 records were measured on.
public struct K3DecodeRunner: RunnerFacade {

    /// Which K3 artifact this runner is pointed at, and therefore what it may
    /// be asked for.
    ///
    /// The floor and the ceiling are not preferences. 5.8 GB remains the
    /// historical one-token scenario budget. Product chat uses a separate
    /// 8.0 GB floor because the second pass overlaps the complete KDA/MLA state
    /// with the next widened layer; longer prompts add their exact MLA cache
    /// term at request preparation. The 64-token API ceiling is a safety bound,
    /// not a speed claim.
    public enum Scale: Sendable, Equatable {
        /// The published 1.56 TB artifact.
        case flagship
        /// Any other K3 artifact — a fixture, a scaled repack — whose floor and
        /// ceiling the caller states because no run record exists to state them.
        case stated(minimumBudgetBytes: UInt64, maximumNewTokens: Int)
    }

    /// The knobs the K3 path can honour, by ``RunKnobs`` property name.
    /// Everything else is refused by name rather than ignored.
    public static let knobs: Set<String> = [
        "expertReadAhead", "expertsPerDispatch", "queueDepth", "granularity",
        "mlxCacheLimitBytes", "logitChunkRows", "deterministicReadAheadLayers",
        "readAheadReserveBytes",
    ]

    public let scale: Scale

    public init(scale: Scale = .flagship) { self.scale = scale }

    /// Derive the memory planner's exact K3 census from verified artifact
    /// metadata, without reading a weight payload.
    ///
    /// The owned repository may be republished independently of the App. Its
    /// deterministic manifests therefore outrank the historical flagship
    /// layer table: F32 vectors stay F32 in residency while BF16 tensors widen,
    /// and a publication that changes that mix changes the plan identity. The
    /// remaining global-read terms are the recorded K3 execution contract; the
    /// runner continues to validate the opened global shapes before any token.
    public static func inspectArtifactCensus(
        _ artifact: ArtifactReference, workingDirectory: URL
    ) throws -> ArtifactCensus {
        try inspectArtifactRuntimeProfile(
            artifact, workingDirectory: workingDirectory).census
    }

    /// The artifact-specific layer census and the exact KDA/MLA state geometry
    /// are inspected together from the same opened configuration. The owned HF
    /// repository may move to another immutable commit without an App release;
    /// no compiled revision is used here.
    public static func inspectArtifactRuntimeProfile(
        _ artifact: ArtifactReference, workingDirectory: URL
    ) throws -> K3ArtifactRuntimeProfile {
        try FileManager.default.createDirectory(
            at: workingDirectory, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
        let mirror = try artifact.runtimeAuthority.map {
            _ in try K3RootedArtifactMirror(
                reference: artifact, workingDirectory: workingDirectory)
        }
        let reference = mirror?.reference ?? artifact
        let fileAccess = mirror?.fileAccess
        let resolved = try K3ArtifactResolution.resolve(
            reference, fileAccess: fileAccess)

        let scratch = workingDirectory.appendingPathComponent(
            "k3-census-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        let translated: URL
        switch resolved.shape {
        case .translated(let directory):
            translated = directory
        case .published(let published):
            try FileManager.default.createDirectory(
                at: scratch, withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
            _ = try PublishedArtifactLayout.translate(
                artifact: published.path, into: scratch.path, config: resolved.config,
                blobReference: fileAccess == nil ? .absolutePath : .artifactReference,
                fileAccess: fileAccess)
            translated = scratch
        }

        let metadata = try ArtifactWeightSource.inspectMetadataCensus(
            translated: translated.path, config: resolved.config,
            fileAccess: fileAccess)
        let recorded = try ArtifactCensus.measuredFlagship()
        var deterministicBytes = recorded.globalsBytesReadPerToken
        for bytes in metadata.layerStoredBytes {
            let sum = deterministicBytes.addingReportingOverflow(bytes)
            guard !sum.overflow else {
                throw RunError.artifactNotReady(
                    "the artifact's deterministic byte census exceeds UInt64.max")
            }
            deterministicBytes = sum.partialValue
        }
        let census = try ArtifactCensus(
            layerStoredBytes: metadata.layerStoredBytes,
            layerResidentBytes: metadata.layerResidentBytes,
            globalsStoredBytes: recorded.globalsStoredBytes,
            globalsBytesReadPerToken: recorded.globalsBytesReadPerToken,
            expertStrideBytes: metadata.expertStrideBytes,
            expertsPerLayer: metadata.expertsPerLayer,
            routedExpertsPerToken: metadata.routedExpertsPerToken,
            moeLayerCount: metadata.moeLayerCount,
            measuredDeterministicBytesPerToken: deterministicBytes)
        return try K3ArtifactRuntimeProfile(
            census: census,
            generationState: K3GenerationStateMemory(config: resolved.config))
    }

    public var capabilities: RunnerCapabilities {
        let minimum: UInt64
        let tokens: Int
        switch scale {
        case .flagship:
            minimum = K3ProductMemoryBudget.minimumBudgetBytes
            tokens = K3ProductMemoryBudget.maximumNewTokens
        case .stated(let minimumBudgetBytes, let maximumNewTokens):
            minimum = minimumBudgetBytes
            tokens = maximumNewTokens
        }
        return RunnerCapabilities(
            model: .kimiK3,
            layout: .k3FlagshipLayerStreams,
            supportedKnobs: Self.knobs,
            minimumBudgetBytes: minimum,
            maximumNewTokens: tokens,
            // No vocabulary travels with this runner. The prompt ids are the
            // experiment's input and a guessed tokenization would change it.
            acceptsTextPrompts: false,
            requiresMLX: true,
            supportsPinPlan: true,
            // One pass, one token: `K3Model.forward` returns a capture, never a
            // callback per step. A per-token cursor would need a callback on
            // the model, which is out of bounds for this change.
            tokenEventGranularity: .perPass)
    }

    public func validate(_ request: RunRequest) throws {
        try validateCommon(request)
        let effective = K3EffectiveKnobs.resolve(request.knobs)
        guard (0...1).contains(effective.deterministicReadAheadLayers) else {
            throw RunError.knobOutOfRange(
                name: "deterministicReadAheadLayers",
                value: "\(effective.deterministicReadAheadLayers)",
                allowed: "0 (off) or 1 (one layer)")
        }
        if let plan = request.pinPlan {
            guard plan.readAheadDepth == effective.deterministicReadAheadLayers else {
                throw RunError.pinPlanInvalid(
                    "the plan projects read-ahead depth \(plan.readAheadDepth ?? -1), but the "
                        + "effective run requests \(effective.deterministicReadAheadLayers)")
            }
            // This constructor performs the complete artifact-independent plan
            // validation. Do it synchronously, before a session can emit an
            // acceptance; the engine repeats the construction when it binds
            // the plan to the opened layer table.
            _ = try ArtifactWeightSource.Configuration(
                plan: plan,
                deterministicReadAhead: effective.deterministicReadAheadLayers)
            if plan.pinnedGlobals != nil {
                throw RunError.pinPlanUnitUnsupported("globals")
            }
            if plan.expertHotSet != nil {
                throw RunError.pinPlanUnitUnsupported("expert hot set")
            }
            if let requestedReserve = request.knobs.readAheadReserveBytes,
                requestedReserve != plan.readAheadReserveBytes
            {
                throw RunError.knobOutOfRange(
                    name: "readAheadReserveBytes", value: "\(requestedReserve)",
                    allowed: "\(plan.readAheadReserveBytes) from the accepted memory plan")
            }
        }
        if let authority = request.artifact.runtimeAuthority {
            guard authority.rootURL.standardizedFileURL.path
                == request.artifact.root.standardizedFileURL.path
            else {
                throw RunError.artifactNotReady(
                    "the rooted runtime authority does not name the requested artifact")
            }
            try authority.validateCurrentBinding()
        } else {
            // Preview/tests and package callers can still use a local fixture.
            // The formal App always supplies rooted authority.
            _ = try K3ArtifactResolution.resolve(request.artifact)
        }
    }

    public func start(_ request: RunRequest) throws -> RunSession {
        try validate(request)
        var knobs = K3EffectiveKnobs.resolve(request.knobs)
        // An unset reserve lets the runner choose. Once a PinPlan is present,
        // its reserve is the only honest choice: it includes every byte the
        // plan made permanently resident, so reporting the scenario default
        // while scheduling against this larger number would describe a
        // different run.
        if let plan = request.pinPlan {
            knobs.readAheadReserveBytes = plan.readAheadReserveBytes
        }
        let handle = RunHandleID()
        let latch = RunCancellationLatch()
        let payloadFlow = RunPayloadFlowMeter()
        let (events, continuation) = AsyncThrowingStream<RunEvent, Error>.makeStream()
        let engine = K3RunEngine(
            request: request, capabilities: capabilities, knobs: knobs, handle: handle,
            latch: latch, continuation: continuation,
            payloadFlow: payloadFlow,
            usesProductMemoryBudget: scale == .flagship)

        // The consuming Task's cancellation is the second door onto the same
        // latch. `onTermination` also fires on a normal finish, where closing a
        // latch nothing reads again is a no-op.
        continuation.onTermination = { _ in latch.cancel() }

        // A dedicated thread, never the cooperative pool. `K3Model.forward`
        // blocks by design — the storage layer is synchronous — and a
        // 357-second blocking call on a cooperative thread starves every other
        // async task in the process.
        let thread = Thread { engine.run() }
        thread.name = "minirun.k3-decode"
        thread.qualityOfService = .userInitiated
        thread.stackSize = 8 << 20
        thread.start()

        return RunSession(
            handle: handle, events: events, payloadFlow: payloadFlow.stream,
            cancel: { latch.cancel() })
    }
}

/// The knobs after defaults are filled in — what the run will actually use.
///
/// The defaults are read off a `K3DecodeScenario` instance rather than restated
/// here. They are the values every number in the M9C and M10 records was
/// measured with, and a second copy of them is a second thing to forget to
/// update.
struct K3EffectiveKnobs {

    private static let scenarioDefaults = K3DecodeScenario(
        resolveArtifact: { throw ScenarioError.notReady("defaults only") })

    var expertReadAhead: Int
    var expertsPerDispatch: Int
    var queueDepth: Int
    var granularity: ExpertFetchGranularity
    var mlxCacheLimitBytes: Int
    var logitChunkRows: Int
    var deterministicReadAheadLayers: Int
    var readAheadReserveBytes: UInt64

    static func resolve(_ knobs: RunKnobs) -> K3EffectiveKnobs {
        let defaults = scenarioDefaults
        return K3EffectiveKnobs(
            expertReadAhead: knobs.expertReadAhead ?? defaults.readAhead,
            expertsPerDispatch: knobs.expertsPerDispatch ?? defaults.expertsPerDispatch,
            queueDepth: knobs.queueDepth ?? defaults.queueDepth,
            granularity: knobs.granularity
                ?? (defaults.granularity == .perTile ? .perTile : .perExpert),
            mlxCacheLimitBytes: knobs.mlxCacheLimitBytes ?? defaults.mlxCacheLimitBytes,
            logitChunkRows: knobs.logitChunkRows ?? defaults.logitChunkRows,
            deterministicReadAheadLayers: knobs.deterministicReadAheadLayers
                ?? defaults.deterministicReadAhead,
            readAheadReserveBytes: knobs.readAheadReserveBytes ?? defaults.readAheadReserveBytes)
    }

    var fetchGranularity: FetchGranularity {
        granularity == .perTile ? .perTile : .perExpert
    }

    /// Stated back to the caller in `RunEvent.accepted`, fully resolved: the
    /// acceptance says what the run uses, not what the caller happened to set.
    var asRunKnobs: RunKnobs {
        var knobs = RunKnobs()
        knobs.expertReadAhead = expertReadAhead
        knobs.expertsPerDispatch = expertsPerDispatch
        knobs.queueDepth = queueDepth
        knobs.granularity = granularity
        knobs.mlxCacheLimitBytes = mlxCacheLimitBytes
        knobs.logitChunkRows = logitChunkRows
        knobs.deterministicReadAheadLayers = deterministicReadAheadLayers
        knobs.readAheadReserveBytes = readAheadReserveBytes
        return knobs
    }

    func asStrings(
        blobReference: PublishedArtifactLayout.BlobReference
    ) -> [String: String] {
        let blobReferenceName: String
        switch blobReference {
        case .symlink: blobReferenceName = "symlink"
        case .absolutePath: blobReferenceName = "absolutePath"
        case .artifactReference: blobReferenceName = "artifactReference"
        }
        return [
            "expertReadAhead": "\(expertReadAhead)",
            "expertsPerDispatch": "\(expertsPerDispatch)",
            "queueDepth": "\(queueDepth)",
            "granularity": "\(fetchGranularity)",
            "mlxCacheLimitBytes": "\(mlxCacheLimitBytes)",
            "logitChunkRows": "\(logitChunkRows)",
            "deterministicReadAheadLayers": "\(deterministicReadAheadLayers)",
            "readAheadReserveBytes": "\(readAheadReserveBytes)",
            "blobReference": blobReferenceName,
        ]
    }
}
