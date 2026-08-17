import Foundation

/// What a run is asked to continue from.
public enum Prompt: Sendable, Equatable {
    case tokenIDs([Int])
    /// Refused by any runner whose ``RunnerCapabilities/acceptsTextPrompts`` is
    /// false. Never tokenized by a guess: a wrong tokenization produces logits
    /// that are wrong for a reason no metric in the run's record would name.
    case text(String)
}

/// Where the artifact is, and the grant that reaches it.
public struct ArtifactReference: Sendable {
    public let root: URL
    /// Held for the whole run. The runner's own bracket releases it, on the
    /// failure paths too.
    public let scope: StorageScope?
    /// Full-verification evidence plus the rooted descriptors a product runner
    /// must consume. Preview fixtures may omit this because their runners read
    /// no artifact bytes; a live runtime must not substitute `root` for it.
    public let runtimeAuthority: ArtifactRuntimeAuthority?
    /// Layouts that need more than one root — the Qwen container set wants a
    /// checkpoint directory as well as a container directory.
    public let auxiliaryRoots: [String: URL]

    public init(
        root: URL, scope: StorageScope? = nil,
        runtimeAuthority: ArtifactRuntimeAuthority? = nil,
        auxiliaryRoots: [String: URL] = [:]
    ) {
        self.root = root
        self.scope = scope
        self.runtimeAuthority = runtimeAuthority
        self.auxiliaryRoots = auxiliaryRoots
    }
}

public enum ExpertFetchGranularity: String, Sendable, Codable, CaseIterable {
    case perExpert, perTile
}

/// The tuning surface of a run.
///
/// Every knob is optional and every knob a runner cannot honour is **refused by
/// name**. Dropping an unsupported knob on the floor would leave a run recorded
/// under settings it did not use, which is the same class of error as clamping
/// a budget: the record stops describing the thing that happened.
public struct RunKnobs: Sendable, Equatable, Codable {
    public var expertReadAhead: Int?
    public var expertsPerDispatch: Int?
    public var queueDepth: Int?
    /// How many routed experts the pool may hold. V4 honours it; K3 does not.
    public var expertPoolSlots: Int?
    public var deterministicReadAheadLayers: Int?
    public var readAheadReserveBytes: UInt64?
    public var mlxCacheLimitBytes: Int?
    public var logitChunkRows: Int?
    public var granularity: ExpertFetchGranularity?
    /// Capturing routing margins changes the thing being timed. A run with this
    /// on must not be quoted for tokens per second.
    public var captureRoutingMargins: Bool?
    /// Start every projection's expert reads the moment the router names the
    /// ids, instead of at each projection's own gather.
    ///
    /// A routed layer reads `w1`, `w3` and `w2` for the *same* selected
    /// experts, so all three schedules are known at `routingSelect` and the
    /// last two are being waited for later than they had to be. Turning this on
    /// widens the pool the layer needs — the reads are in flight together — so
    /// V4 refuses it rather than clamping when `expertPoolSlots` cannot cover
    /// the widened window. K3 does not honour it: its router consumes the
    /// layer's own hidden state, so there is nothing earlier to move it to.
    public var expertProjectionPrefetch: Bool?
    /// Keep one routed-expert backend per layer alive for the whole run,
    /// sharing one pool, instead of building and destroying one per layer per
    /// pass.
    ///
    /// It is a lifetime, not a width: the containers, the window and the gather
    /// are unchanged. What changes is that readers and the pool are created
    /// once rather than `layers x passes` times, and that a schedule issued for
    /// a layer that has not started yet has somewhere to live. V4 honours it;
    /// K3's expert backend is already run-scoped.
    public var expertRunScopedBackends: Bool?
    /// Tiles a layer that has not started yet may hold in the shared pool.
    ///
    /// Only reachable with ``expertRunScopedBackends``, and only useful where
    /// ids are knowable early — in V4 that is the hash-routed layers, whose
    /// experts are a table lookup on the token id. It widens the pool the run
    /// needs by exactly this many slots, and V4 refuses rather than clamps when
    /// ``expertPoolSlots`` cannot cover it.
    public var expertCrossLayerPrefetch: Int?
    /// Hand MLX the pager's own bytes instead of copying each tile into
    /// MLX-owned arrays (ADR-0002).
    ///
    /// Adoption removes the per-tile `memmove`, and in exchange every tile in a
    /// dispatch stays leased until MLX evaluates the graph — so the pool must
    /// cover the gather width on top of the read-ahead windows. Refused, not
    /// clamped, when it cannot.
    public var expertTileAdoption: Bool?

    public init() {}

    /// Names of the knobs actually set, in a stable order.
    public var setKnobNames: [String] {
        var names: [String] = []
        if expertReadAhead != nil { names.append("expertReadAhead") }
        if expertsPerDispatch != nil { names.append("expertsPerDispatch") }
        if queueDepth != nil { names.append("queueDepth") }
        if expertPoolSlots != nil { names.append("expertPoolSlots") }
        if deterministicReadAheadLayers != nil { names.append("deterministicReadAheadLayers") }
        if readAheadReserveBytes != nil { names.append("readAheadReserveBytes") }
        if mlxCacheLimitBytes != nil { names.append("mlxCacheLimitBytes") }
        if logitChunkRows != nil { names.append("logitChunkRows") }
        if granularity != nil { names.append("granularity") }
        if captureRoutingMargins != nil { names.append("captureRoutingMargins") }
        if expertProjectionPrefetch != nil { names.append("expertProjectionPrefetch") }
        if expertRunScopedBackends != nil { names.append("expertRunScopedBackends") }
        if expertCrossLayerPrefetch != nil { names.append("expertCrossLayerPrefetch") }
        if expertTileAdoption != nil { names.append("expertTileAdoption") }
        return names
    }

    public func validated(for capabilities: RunnerCapabilities) throws -> RunKnobs {
        for name in setKnobNames where !capabilities.supportedKnobs.contains(name) {
            throw RunError.knobNotSupported(name: name, byModel: capabilities.model)
        }
        if let value = expertReadAhead, value < 0 {
            throw RunError.knobOutOfRange(
                name: "expertReadAhead", value: "\(value)", allowed: ">= 0")
        }
        if let value = expertsPerDispatch, value < 1 {
            throw RunError.knobOutOfRange(
                name: "expertsPerDispatch", value: "\(value)", allowed: ">= 1")
        }
        if let value = queueDepth, value < 1 {
            throw RunError.knobOutOfRange(name: "queueDepth", value: "\(value)", allowed: ">= 1")
        }
        if let value = expertPoolSlots, value < 1 {
            throw RunError.knobOutOfRange(
                name: "expertPoolSlots", value: "\(value)", allowed: ">= 1")
        }
        if let value = deterministicReadAheadLayers, value < 0 {
            throw RunError.knobOutOfRange(
                name: "deterministicReadAheadLayers", value: "\(value)", allowed: ">= 0")
        }
        if let value = logitChunkRows, value < 1 {
            throw RunError.knobOutOfRange(
                name: "logitChunkRows", value: "\(value)", allowed: ">= 1")
        }
        if let value = mlxCacheLimitBytes, value < 0 {
            throw RunError.knobOutOfRange(
                name: "mlxCacheLimitBytes", value: "\(value)", allowed: ">= 0")
        }
        if let value = expertCrossLayerPrefetch, value < 0 {
            throw RunError.knobOutOfRange(
                name: "expertCrossLayerPrefetch", value: "\(value)", allowed: ">= 0")
        }
        return self
    }
}

public struct RunRequest: Sendable {
    public let model: ModelID
    public let artifact: ArtifactReference
    public let prompt: Prompt
    /// Stated before the run. Never inferred from what the run turned out to
    /// need, and never clamped up to what it needed (spec §5.3).
    public let memoryBudgetBytes: UInt64
    /// The residency decision the memory dial made for this exact budget.
    ///
    /// `nil` preserves the streamed-everything arm used by records made before
    /// the dial existed. When present, a runner must either honour the plan or
    /// refuse it by name; silently using only the byte budget would make the
    /// run disagree with the screen that launched it.
    public let pinPlan: PinPlan?
    public let maximumNewTokens: Int
    public let knobs: RunKnobs
    public let workingDirectory: URL

    public init(
        model: ModelID, artifact: ArtifactReference, prompt: Prompt,
        memoryBudgetBytes: UInt64, pinPlan: PinPlan? = nil, maximumNewTokens: Int,
        knobs: RunKnobs = RunKnobs(), workingDirectory: URL
    ) {
        self.model = model
        self.artifact = artifact
        self.prompt = prompt
        self.memoryBudgetBytes = memoryBudgetBytes
        self.pinPlan = pinPlan
        self.maximumNewTokens = maximumNewTokens
        self.knobs = knobs
        self.workingDirectory = workingDirectory
    }
}

/// Whether tokens arrive one at a time or in batches at pass boundaries.
public enum TokenEventGranularity: String, Sendable, Codable { case perToken, perPass }

public struct RunnerCapabilities: Sendable, Equatable {
    public let model: ModelID
    public let layout: ArtifactLayout
    public let supportedKnobs: Set<String>
    public let minimumBudgetBytes: UInt64
    public let maximumNewTokens: Int
    public let acceptsTextPrompts: Bool
    public let requiresMLX: Bool
    /// Whether this runner can honour the K3-shaped residency plan carried by
    /// ``RunRequest/pinPlan``. A byte budget is model-neutral; a pin plan is
    /// not. Callers must not attach one merely because the memory dial was able
    /// to draw one for another model family.
    public let supportsPinPlan: Bool
    /// States, rather than implies, when tokens appear. A caller that draws a
    /// per-token cursor against a `.perPass` runner draws a cursor that jumps.
    public let tokenEventGranularity: TokenEventGranularity

    public init(
        model: ModelID, layout: ArtifactLayout, supportedKnobs: Set<String>,
        minimumBudgetBytes: UInt64, maximumNewTokens: Int, acceptsTextPrompts: Bool,
        requiresMLX: Bool, supportsPinPlan: Bool = false,
        tokenEventGranularity: TokenEventGranularity
    ) {
        self.model = model
        self.layout = layout
        self.supportedKnobs = supportedKnobs
        self.minimumBudgetBytes = minimumBudgetBytes
        self.maximumNewTokens = maximumNewTokens
        self.acceptsTextPrompts = acceptsTextPrompts
        self.requiresMLX = requiresMLX
        self.supportsPinPlan = supportsPinPlan
        self.tokenEventGranularity = tokenEventGranularity
    }
}

public struct RunHandleID: Hashable, Sendable, Codable {
    public let rawValue: UUID
    public init(rawValue: UUID = UUID()) { self.rawValue = rawValue }
}

public enum ThermalStateName: String, Sendable, Codable, CaseIterable {
    case nominal, fair, serious, critical, unknown

    public init(name: String) {
        self = ThermalStateName(rawValue: name) ?? .unknown
    }
}

/// Every byte a run read, in exactly one bucket.
///
/// The split is optional because the existing scenarios report only the total
/// while running and break it down in their result JSON; the buckets are
/// populated on `finished`. ``isBalanced`` says whether the identity currently
/// holds, so a consumer never has to guess whether a nil means "not split yet"
/// or "split and lost some".
public struct ByteAccounting: Sendable, Codable, Equatable {
    public let totalBytesRead: UInt64
    public let deterministicBytesRead: UInt64?
    public let expertBytesRead: UInt64?

    public init(
        totalBytesRead: UInt64, deterministicBytesRead: UInt64? = nil,
        expertBytesRead: UInt64? = nil
    ) {
        self.totalBytesRead = totalBytesRead
        self.deterministicBytesRead = deterministicBytesRead
        self.expertBytesRead = expertBytesRead
    }

    /// Bytes the split does not name. Nil while the split is not available, and
    /// also when the buckets overrun the total — an over-attributed run has no
    /// remainder to report, it has a broken identity, and answering `0` there
    /// would read exactly like a run that balanced.
    public var unattributedBytes: UInt64? {
        guard let deterministic = deterministicBytesRead, let expert = expertBytesRead else {
            return nil
        }
        let named = deterministic.addingReportingOverflow(expert)
        guard !named.overflow else { return nil }
        return totalBytesRead >= named.partialValue ? totalBytesRead - named.partialValue : nil
    }

    /// The split accounts for the total exactly. False when the parts overrun
    /// the whole, which is what a double-counted read looks like.
    public var isBalanced: Bool {
        guard let deterministic = deterministicBytesRead, let expert = expertBytesRead else {
            return true
        }
        let named = deterministic.addingReportingOverflow(expert)
        return !named.overflow && named.partialValue == totalBytesRead
    }

    /// Returns the bytes completed after an earlier cumulative boundary.
    ///
    /// Run byte counters are monotonic. A component moving backwards is not a
    /// negative amount of work; it is evidence that the two samples cannot be
    /// compared, so subtraction fails closed. A split is retained only when it
    /// is available at both boundaries.
    public func subtracting(_ earlier: ByteAccounting) -> ByteAccounting? {
        guard totalBytesRead >= earlier.totalBytesRead else { return nil }

        let deterministic: UInt64?
        let expert: UInt64?
        switch (
            deterministicBytesRead, expertBytesRead,
            earlier.deterministicBytesRead, earlier.expertBytesRead
        ) {
        case let (currentDeterministic?, currentExpert?, earlierDeterministic?, earlierExpert?):
            guard currentDeterministic >= earlierDeterministic,
                currentExpert >= earlierExpert
            else { return nil }
            deterministic = currentDeterministic - earlierDeterministic
            expert = currentExpert - earlierExpert
        default:
            deterministic = nil
            expert = nil
        }

        let result = ByteAccounting(
            totalBytesRead: totalBytesRead - earlier.totalBytesRead,
            deterministicBytesRead: deterministic,
            expertBytesRead: expert)
        return result.isBalanced ? result : nil
    }
}

/// Routed-expert observations that can be sampled without exposing a
/// model-specific backend to the product. Every field is cumulative from the
/// start of this run; a consumer may safely derive rates from successive
/// samples without inventing another counter.
public struct RunExpertTelemetry: Sendable, Codable, Equatable {
    public let phaseSeconds: Double
    public let ioWaitSeconds: Double
    public let bytesRead: UInt64
    public let cachePolicy: String
    public let cacheHits: Int
    public let cacheMisses: Int
    public let cacheEvictions: Int
    public let cacheRejectedAdmissions: Int
    public let pinnedExpertCount: Int

    public init(
        phaseSeconds: Double, ioWaitSeconds: Double, bytesRead: UInt64,
        cachePolicy: String, cacheHits: Int, cacheMisses: Int,
        cacheEvictions: Int, cacheRejectedAdmissions: Int,
        pinnedExpertCount: Int
    ) {
        self.phaseSeconds = phaseSeconds
        self.ioWaitSeconds = ioWaitSeconds
        self.bytesRead = bytesRead
        self.cachePolicy = cachePolicy
        self.cacheHits = cacheHits
        self.cacheMisses = cacheMisses
        self.cacheEvictions = cacheEvictions
        self.cacheRejectedAdmissions = cacheRejectedAdmissions
        self.pinnedExpertCount = pinnedExpertCount
    }
}

/// Deterministic read-ahead observations at the same instant as a
/// ``RunTelemetry`` sample. The three byte exits remain separate so a product
/// can prove the lifecycle identity instead of treating a non-zero total as a
/// success signal.
public struct RunReadAheadTelemetry: Sendable, Codable, Equatable {
    public let depth: Int
    public let stagedBytes: UInt64
    public let peakStagedBytes: UInt64
    public let materialisedStagedBytes: UInt64
    public let discardedStagedBytes: UInt64
    public let outstandingStagedBytes: UInt64
    public let prefetchWaitSeconds: Double
    public let stageReadSeconds: Double
    /// Present only for a terminal census. A live sample cannot infer a slot
    /// count from outstanding bytes, and must not manufacture one.
    public let nonFreeSlots: Int?

    public init(
        depth: Int, stagedBytes: UInt64, peakStagedBytes: UInt64,
        materialisedStagedBytes: UInt64, discardedStagedBytes: UInt64,
        outstandingStagedBytes: UInt64, prefetchWaitSeconds: Double,
        stageReadSeconds: Double, nonFreeSlots: Int?
    ) {
        self.depth = depth
        self.stagedBytes = stagedBytes
        self.peakStagedBytes = peakStagedBytes
        self.materialisedStagedBytes = materialisedStagedBytes
        self.discardedStagedBytes = discardedStagedBytes
        self.outstandingStagedBytes = outstandingStagedBytes
        self.prefetchWaitSeconds = prefetchWaitSeconds
        self.stageReadSeconds = stageReadSeconds
        self.nonFreeSlots = nonFreeSlots
    }
}

/// Optional evidence for instruments that are not universal runner concepts.
/// A missing field means “not reported”; it never means a measured zero.
public struct RunInstrumentationTelemetry: Sendable, Codable, Equatable {
    public let expert: RunExpertTelemetry?
    public let readAhead: RunReadAheadTelemetry?

    public init(
        expert: RunExpertTelemetry? = nil,
        readAhead: RunReadAheadTelemetry? = nil
    ) {
        self.expert = expert
        self.readAhead = readAhead
    }
}

public enum RunGenerationStage: String, Sendable, Codable, Equatable {
    case preparing
    case prefill
    case decode
    case terminal
}

public struct RunTelemetry: Sendable, Codable, Equatable {
    public let at: Date
    public let elapsed: TimeInterval
    public let phase: String
    /// Structured generation stage. Nil only for records written before this
    /// field existed or for a runner that cannot name its stage.
    public let generationStage: RunGenerationStage?
    public let tokensPerSecond: Double?
    public let bytes: ByteAccounting
    public let bytesPerSecond: Double?
    /// The byte counters sampled at the most recent completed-token boundary.
    ///
    /// Progressive `bytes` includes work already performed for the token that
    /// is currently in flight. Dividing that value by the number of completed
    /// tokens makes the apparent per-token cost rise throughout the next pass.
    /// A runner that can name token boundaries publishes this snapshot so a UI
    /// can keep the completed-token average stable and report the in-flight
    /// bytes separately. Older runners may leave it nil.
    public let completedTokenBytes: ByteAccounting?
    /// Cumulative byte boundary at the completion of prefill. Decode-only
    /// bytes/token subtracts this once rather than charging prompt ingestion to
    /// every generated token.
    public let prefillBytes: ByteAccounting?
    public let bytesPerToken: UInt64?
    /// Prefill and decode are different workloads. These fields keep their
    /// elapsed time and denominator explicit instead of averaging them into a
    /// number that cannot be compared across prompt lengths.
    public let prefillSeconds: TimeInterval?
    public let decodeSeconds: TimeInterval?
    public let decodeTokensCompleted: Int?
    /// Whether `bytes` and its derived rates are measurements from this run.
    ///
    /// Some runners do not yet have a trustworthy observer for every read.
    /// They still publish the rest of their telemetry, but set this to false
    /// so a zero placeholder cannot be presented as measured traffic. Nil is
    /// reserved for records written before this field existed and preserves
    /// their original, reported-byte interpretation.
    public let byteAccountingReported: Bool?
    /// The promise the run made about itself.
    public let declaredBudgetBytes: UInt64
    /// `phys_footprint` — what iOS enforces its limit against.
    public let footprintBytes: UInt64
    public let peakFootprintBytes: UInt64
    public let residentBytes: UInt64
    /// `os_proc_available_memory()`, where the platform has it.
    public let availableBytes: UInt64?
    public let mlxActiveBytes: UInt64
    public let mlxCacheBytes: UInt64
    public let mlxPeakBytes: UInt64
    public let thermalState: ThermalStateName
    public let lowPowerMode: Bool
    public let batteryLevel: Float?
    /// Optional, runner-supplied evidence for the extended instrument panel.
    /// Older records decode this as nil.
    public let instrumentation: RunInstrumentationTelemetry?

    public init(
        at: Date, elapsed: TimeInterval, phase: String, tokensPerSecond: Double?,
        generationStage: RunGenerationStage? = nil,
        bytes: ByteAccounting, bytesPerSecond: Double?,
        completedTokenBytes: ByteAccounting? = nil,
        prefillBytes: ByteAccounting? = nil,
        bytesPerToken: UInt64?,
        prefillSeconds: TimeInterval? = nil,
        decodeSeconds: TimeInterval? = nil,
        decodeTokensCompleted: Int? = nil,
        byteAccountingReported: Bool? = true,
        declaredBudgetBytes: UInt64, footprintBytes: UInt64, peakFootprintBytes: UInt64,
        residentBytes: UInt64, availableBytes: UInt64?, mlxActiveBytes: UInt64,
        mlxCacheBytes: UInt64, mlxPeakBytes: UInt64, thermalState: ThermalStateName,
        lowPowerMode: Bool, batteryLevel: Float?,
        instrumentation: RunInstrumentationTelemetry? = nil
    ) {
        self.at = at
        self.elapsed = elapsed
        self.phase = phase
        self.generationStage = generationStage
        self.tokensPerSecond = tokensPerSecond
        self.bytes = bytes
        self.bytesPerSecond = bytesPerSecond
        self.completedTokenBytes = completedTokenBytes
        self.prefillBytes = prefillBytes
        self.bytesPerToken = bytesPerToken
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
        self.decodeTokensCompleted = decodeTokensCompleted
        self.byteAccountingReported = byteAccountingReported
        self.declaredBudgetBytes = declaredBudgetBytes
        self.footprintBytes = footprintBytes
        self.peakFootprintBytes = peakFootprintBytes
        self.residentBytes = residentBytes
        self.availableBytes = availableBytes
        self.mlxActiveBytes = mlxActiveBytes
        self.mlxCacheBytes = mlxCacheBytes
        self.mlxPeakBytes = mlxPeakBytes
        self.thermalState = thermalState
        self.lowPowerMode = lowPowerMode
        self.batteryLevel = batteryLevel
        self.instrumentation = instrumentation
    }

    public var budgetFraction: Double {
        declaredBudgetBytes == 0 ? 0 : Double(peakFootprintBytes) / Double(declaredBudgetBytes)
    }

    /// The promise was kept.
    public var budgetRespected: Bool { peakFootprintBytes <= declaredBudgetBytes }

    /// Legacy telemetry predates the explicit availability bit and therefore
    /// retains the byte-accounting semantics it had when it was recorded.
    public var reportsByteAccounting: Bool { byteAccountingReported ?? true }
}

/// One fixed-window observation of model payload bytes completed into this
/// process.
///
/// This is deliberately narrower than a disk benchmark and stricter than a
/// request counter:
///
/// - it includes deterministic layer/global weights and routed-expert payloads;
/// - it includes successful read-ahead that is later discarded;
/// - it excludes catalogue, tokenizer, manifest, header, and download traffic;
/// - a byte is recorded only after the complete read succeeds;
/// - the interval is measured with a monotonic clock by the runner.
///
/// Page-cache hits are still reads completed into the process, so the derived
/// rate may exceed the physical link rate. Consumers should label this “model
/// data flow”, not “drive speed”.
public struct RunPayloadFlowSample: Sendable, Codable, Equatable {
    public let sequence: UInt64
    public let at: Date
    public let intervalSeconds: TimeInterval
    public let deterministicBytes: UInt64
    public let expertBytes: UInt64
    /// Sticky. Once a run loses exact byte evidence, later samples remain
    /// unusable even if their individual window would fit in UInt64.
    public let didOverflow: Bool

    public init(
        sequence: UInt64, at: Date, intervalSeconds: TimeInterval,
        deterministicBytes: UInt64, expertBytes: UInt64,
        didOverflow: Bool = false
    ) {
        self.sequence = sequence
        self.at = at
        self.intervalSeconds = intervalSeconds
        self.deterministicBytes = deterministicBytes
        self.expertBytes = expertBytes
        self.didOverflow = didOverflow
    }

    public var totalBytes: UInt64? {
        guard !didOverflow else { return nil }
        let total = deterministicBytes.addingReportingOverflow(expertBytes)
        return total.overflow ? nil : total.partialValue
    }

    public var isValid: Bool {
        intervalSeconds.isFinite && intervalSeconds > 0 && totalBytes != nil
    }

    public var deterministicBytesPerSecond: Double? {
        guard isValid else { return nil }
        return Double(deterministicBytes) / intervalSeconds
    }

    public var expertBytesPerSecond: Double? {
        guard isValid else { return nil }
        return Double(expertBytes) / intervalSeconds
    }
}

public struct RunPhase: Sendable, Equatable {
    public let name: String
    public let generationStage: RunGenerationStage?
    public let fraction: Double?
    public let detail: String

    public init(
        name: String, generationStage: RunGenerationStage? = nil,
        fraction: Double? = nil, detail: String = ""
    ) {
        self.name = name
        self.generationStage = generationStage
        self.fraction = fraction
        self.detail = detail
    }
}

public struct TokenEvent: Sendable, Equatable, Identifiable {
    /// 0 is the token the prefill produced. Also the identity: within one run a
    /// token's position is what distinguishes it, and two runs' events are
    /// never mixed in one list.
    public var id: Int { index }
    public let index: Int
    public let tokenID: Int
    /// Nil when the runner carries no vocabulary.
    public let text: String?
    public let secondsSincePreviousToken: Double
    public let isPrefillToken: Bool
    /// Nil unless `captureRoutingMargins` was on.
    public let top1Top2Gap: Double?

    public init(
        index: Int, tokenID: Int, text: String?, secondsSincePreviousToken: Double,
        isPrefillToken: Bool, top1Top2Gap: Double? = nil
    ) {
        self.index = index
        self.tokenID = tokenID
        self.text = text
        self.secondsSincePreviousToken = secondsSincePreviousToken
        self.isPrefillToken = isPrefillToken
        self.top1Top2Gap = top1Top2Gap
    }
}

/// What the runner agreed to after metadata and runtime preflight, but before
/// model payload materialisation begins.
public struct RunAcceptance: Sendable, Equatable {
    public let handle: RunHandleID
    public let model: ModelID
    public let declaredBudgetBytes: UInt64
    /// The plan the runner accepted, not a plan reconstructed after the run.
    public let pinPlan: PinPlan?
    public let effectiveKnobs: RunKnobs
    public let artifactRoot: String
    public let startedAt: Date

    public init(
        handle: RunHandleID, model: ModelID, declaredBudgetBytes: UInt64,
        pinPlan: PinPlan? = nil,
        effectiveKnobs: RunKnobs, artifactRoot: String, startedAt: Date
    ) {
        self.handle = handle
        self.model = model
        self.declaredBudgetBytes = declaredBudgetBytes
        self.pinPlan = pinPlan
        self.effectiveKnobs = effectiveKnobs
        self.artifactRoot = artifactRoot
        self.startedAt = startedAt
    }
}

public struct ThermalTransition: Sendable, Codable, Equatable {
    public let state: ThermalStateName
    public let atSeconds: Double
    public let phase: String

    public init(state: ThermalStateName, atSeconds: Double, phase: String) {
        self.state = state
        self.atSeconds = atSeconds
        self.phase = phase
    }
}

/// A digest is evidence about one specific logits vector, not about a run in
/// the abstract. The token index prevents a multi-token run from silently
/// comparing its final logits with an earlier pass's keeper value.
public struct RunLogitsDigest: Sendable, Codable, Equatable {
    public enum Algorithm: String, Sendable, Codable { case sha256 }

    public let algorithm: Algorithm
    public let tokenIndex: Int
    public let hex: String

    public init(algorithm: Algorithm = .sha256, tokenIndex: Int, hex: String) {
        self.algorithm = algorithm
        self.tokenIndex = tokenIndex
        self.hex = hex
    }

    public var isValid: Bool {
        tokenIndex >= 0 && hex.count == 64 && hex.utf8.allSatisfy {
            (48...57).contains($0) || (97...102).contains($0)
        }
    }
}

public struct RunSummary: Sendable, Codable, Equatable {
    public let handle: RunHandleID
    public let model: ModelID
    public let tokenIDs: [Int]
    public let text: String?
    public let wallSeconds: Double
    public let tokensPerSecond: Double?
    public let finalTelemetry: RunTelemetry
    public let peakFootprintBytes: UInt64
    public let budgetRespected: Bool
    public let thermalTrail: [ThermalTransition]
    /// SHA-256 of the logits bytes the runner actually produced, when the
    /// runner captured them. `nil` is deliberately different from a fixture
    /// digest: callers may omit the reproducibility footer, but must never
    /// invent a measurement the runner did not make.
    public let logitsDigest: RunLogitsDigest?
    /// The scenario's own result JSON, byte for byte. MinirunKit does not
    /// re-summarize it: spec §17.4's record belongs to the scenario that
    /// produced it, and a second summary is a second thing to keep in sync.
    public let resultJSON: Data?
    public let resultJSONFilename: String?
    public let headline: String
    public let lines: [String]
    public let gitRevision: String

    public init(
        handle: RunHandleID, model: ModelID, tokenIDs: [Int], text: String?,
        wallSeconds: Double, tokensPerSecond: Double?, finalTelemetry: RunTelemetry,
        peakFootprintBytes: UInt64, budgetRespected: Bool,
        thermalTrail: [ThermalTransition], logitsDigest: RunLogitsDigest? = nil,
        resultJSON: Data?, resultJSONFilename: String?,
        headline: String, lines: [String], gitRevision: String
    ) {
        self.handle = handle
        self.model = model
        self.tokenIDs = tokenIDs
        self.text = text
        self.wallSeconds = wallSeconds
        self.tokensPerSecond = tokensPerSecond
        self.finalTelemetry = finalTelemetry
        self.peakFootprintBytes = peakFootprintBytes
        self.budgetRespected = budgetRespected
        self.thermalTrail = thermalTrail
        self.logitsDigest = logitsDigest
        self.resultJSON = resultJSON
        self.resultJSONFilename = resultJSONFilename
        self.headline = headline
        self.lines = lines
        self.gitRevision = gitRevision
    }
}

public enum RunEvent: Sendable {
    case accepted(RunAcceptance)
    case phase(RunPhase)
    case token(TokenEvent)
    case telemetry(RunTelemetry)
    /// Where one completed pass's wall time went. Emitted at the same boundary
    /// as the pass's token, and only when the engine's own subtraction produced
    /// a decomposition it can stand behind — a split that cannot be trusted is
    /// worse than none, so a runner that cannot attribute a pass says nothing
    /// rather than publishing zeros.
    case phaseSummary(RunPhaseSummary)
    case log(String)
    case finished(RunSummary)
    case cancelled(RunSummary)
}

public enum RunError: Error, Equatable, CustomStringConvertible {
    case budgetBelowMinimum(declared: UInt64, minimum: UInt64, model: ModelID)
    case knobNotSupported(name: String, byModel: ModelID)
    case knobOutOfRange(name: String, value: String, allowed: String)
    case promptNotSupported(reason: String)
    case tooManyTokens(requested: Int, maximum: Int, model: ModelID)
    case artifactNotReady(String)
    case artifactLayoutMismatch(expected: ArtifactLayout, atPath: String)
    case scopeRefused(path: String)
    case budgetExceeded(peakBytes: UInt64, declaredBytes: UInt64)
    case pinPlanBudgetMismatch(planBytes: UInt64, declaredBytes: UInt64)
    case pinPlanAccountingMismatch
    case pinPlanInvalid(String)
    case pinPlanUnitUnsupported(String)
    case runnerUnavailable(ModelID, reason: String)
    case cancelled
    /// The scenario's own error text, verbatim. Never reworded — a reworded
    /// error is a second story about the same failure.
    case underlying(String)

    public var description: String {
        switch self {
        case .budgetBelowMinimum(let declared, let minimum, let model):
            return
                "\(model) asked for \(declared) bytes; this runtime requires at least \(minimum)"
        case .knobNotSupported(let name, let model):
            return "\(model)'s runner cannot honour '\(name)'"
        case .knobOutOfRange(let name, let value, let allowed):
            return "\(name) is \(value); allowed: \(allowed)"
        case .promptNotSupported(let reason):
            return "this prompt cannot be used: \(reason)"
        case .tooManyTokens(let requested, let maximum, let model):
            return "\(model)'s runner produces at most \(maximum) tokens; \(requested) were asked for"
        case .artifactNotReady(let what):
            return "the artifact is not ready: \(what)"
        case .artifactLayoutMismatch(let expected, let path):
            return "'\(path)' is not a \(expected.rawValue) artifact"
        case .scopeRefused(let path):
            return "access to '\(path)' was refused"
        case .budgetExceeded(let peak, let declared):
            return "peak footprint \(peak) exceeded the declared budget \(declared)"
        case .pinPlanBudgetMismatch(let plan, let declared):
            return "the memory plan is for \(plan) bytes, but this run declares \(declared)"
        case .pinPlanAccountingMismatch:
            return "the memory plan does not balance its residency and per-token byte accounts"
        case .pinPlanInvalid(let reason):
            return "the memory plan is invalid: \(reason)"
        case .pinPlanUnitUnsupported(let unit):
            return "this runner cannot honour the memory plan's \(unit) tier"
        case .runnerUnavailable(let model, let reason):
            return "no runner for \(model): \(reason)"
        case .cancelled:
            return "cancelled"
        case .underlying(let text):
            return text
        }
    }
}

/// A run in progress.
public struct RunSession: Sendable {
    public let handle: RunHandleID
    /// The stream **is** the run: consuming it is what observes progress, and
    /// abandoning it is one of the two ways to stop.
    public let events: AsyncThrowingStream<RunEvent, Error>
    /// Fixed-window model payload flow, independent of layer/token events.
    /// Nil means this runner does not provide comparable flow evidence.
    public let payloadFlow: AsyncStream<RunPayloadFlowSample>?
    public let cancel: @Sendable () -> Void

    public init(
        handle: RunHandleID, events: AsyncThrowingStream<RunEvent, Error>,
        payloadFlow: AsyncStream<RunPayloadFlowSample>? = nil,
        cancel: @escaping @Sendable () -> Void
    ) {
        self.handle = handle
        self.events = events
        self.payloadFlow = payloadFlow
        self.cancel = cancel
    }
}

/// What the app binds against.
///
/// Cancellation has two doors and both close the same latch: ``RunSession/cancel``
/// or cancelling the task consuming ``RunSession/events``. A cancelled run ends
/// as `RunEvent.cancelled(RunSummary)` — a named failure class, never a
/// silently short result.
public protocol RunnerFacade: Sendable {
    var capabilities: RunnerCapabilities { get }

    /// Refuses **before a byte is read**: a budget below the measured minimum,
    /// a knob this runner cannot honour, a text prompt with no tokenizer, a
    /// root that is not this layout.
    func validate(_ request: RunRequest) throws

    /// Calls ``validate(_:)`` first.
    func start(_ request: RunRequest) throws -> RunSession
}

extension RunnerFacade {
    /// The checks every runner owes, in one place so they cannot drift apart.
    /// A runner's own `validate` calls this and then adds what only it knows.
    public func validateCommon(_ request: RunRequest) throws {
        let capabilities = self.capabilities
        guard request.model == capabilities.model else {
            throw RunError.runnerUnavailable(
                request.model, reason: "this runner runs \(capabilities.model)")
        }
        guard request.memoryBudgetBytes >= capabilities.minimumBudgetBytes else {
            throw RunError.budgetBelowMinimum(
                declared: request.memoryBudgetBytes,
                minimum: capabilities.minimumBudgetBytes, model: capabilities.model)
        }
        guard request.maximumNewTokens >= 1 else {
            throw RunError.knobOutOfRange(
                name: "maximumNewTokens", value: "\(request.maximumNewTokens)", allowed: ">= 1")
        }
        guard request.maximumNewTokens <= capabilities.maximumNewTokens else {
            throw RunError.tooManyTokens(
                requested: request.maximumNewTokens, maximum: capabilities.maximumNewTokens,
                model: capabilities.model)
        }
        if case .text = request.prompt, !capabilities.acceptsTextPrompts {
            throw RunError.promptNotSupported(
                reason: "\(capabilities.model)'s runner carries no tokenizer vocabulary, so it takes token ids")
        }
        if case .tokenIDs(let ids) = request.prompt, ids.isEmpty {
            throw RunError.promptNotSupported(reason: "the prompt has no tokens")
        }
        if let plan = request.pinPlan {
            guard capabilities.supportsPinPlan else {
                throw RunError.pinPlanUnitUnsupported(
                    "\(capabilities.model.rawValue) runtime")
            }
            guard plan.budgetBytes == request.memoryBudgetBytes else {
                throw RunError.pinPlanBudgetMismatch(
                    planBytes: plan.budgetBytes, declaredBytes: request.memoryBudgetBytes)
            }
            if let reason = plan.runtimeValidationError {
                if !plan.isResidencyBalanced || !plan.isAccountingBalanced {
                    throw RunError.pinPlanAccountingMismatch
                }
                throw RunError.pinPlanInvalid(reason)
            }
            guard plan.requestedMaximumNewTokens == request.maximumNewTokens else {
                throw RunError.pinPlanInvalid(
                    "the plan is for \(plan.requestedMaximumNewTokens ?? 0) new tokens, but "
                        + "this run requests \(request.maximumNewTokens)")
            }
        }
        _ = try request.knobs.validated(for: capabilities)
    }
}

/// The one latch both cancellation doors close.
///
/// A small class rather than a `Task`-based mechanism because the thing being
/// cancelled is a *blocking* forward pass on a dedicated thread: it cannot
/// await anything, so it polls, and what it polls has to be safe to read from
/// any thread.
public final class RunCancellationLatch: @unchecked Sendable {

    private let lock = NSLock()
    private var cancelled = false

    public init() {}

    public var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    public func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }

    /// Throws ``RunError/cancelled`` if the latch is closed. Called at every
    /// point a run can be interrupted; a loop that does not call it is a loop
    /// the Stop button cannot stop.
    public func check() throws {
        if isCancelled { throw RunError.cancelled }
    }
}
