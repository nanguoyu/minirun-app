import Darwin
import Foundation
import MLX

/// Return V4's evaluated, no-longer-owned buffers to both allocators.
///
/// `MLX.Memory.clearCache()` releases recyclable MLX allocations, but V4 also
/// streams large weight and output-head windows through malloc-backed raw
/// pointers. Darwin's allocator may otherwise keep those empty pages in the
/// process footprint and grow by another high-water allocation on a later
/// layer or token even though no model state owns them. Pressure relief is
/// deliberately placed only at V4's existing evaluated layer/chunk/terminal
/// boundaries; it never runs while a lazy graph may still consume a buffer.
///
/// The optional recorder is how a pass finds out what this costs. Both calls
/// return pages to an allocator and neither is free; V4 makes them 43 times
/// per pass plus once per logit window, and until the audit of 2026-08-14
/// nobody had measured the total.
enum DeepSeekV4TransientMemory {
    static func reclaim(recordingTo phaseAccounting: DeepSeekV4PhaseAccounting? = nil) {
        measuringPhase(phaseAccounting?.recordReclaim(nanoseconds:)) {
            MLX.Memory.clearCache()
            _ = malloc_zone_pressure_relief(nil, 0)
        }
    }
}

/// Whether a V4 run reclaims at *every* evaluated layer/window boundary.
///
/// Reclaiming at every boundary is the conservative policy that bought the
/// accepted 2 GB product floor: the MLX cache limit is 0 there, so nothing is
/// being recycled and returning the pages costs only the call. A run under a
/// larger operator-declared budget sets a non-zero MLX cache limit instead,
/// and clearing that cache at every boundary would defeat it exactly.
///
/// Such a run therefore reclaims only near its ceiling. The margin is
/// a single stated number, not a knob: **reclaim once the last telemetry
/// sample's process footprint has reached 90% of the declared budget.** That
/// is high enough to leave the recycled buffers in place for the ordinary
/// pass, and low enough to start giving pages back well before the engine's
/// own budget enforcement — which fails the run at 100% — can trip.
///
/// The footprint is published by the run engine from the same telemetry
/// sample the run reports; this type never samples the process itself, so the
/// decision is always made from an observation the record also contains.
public final class DeepSeekV4BoundaryReclaimPolicy: @unchecked Sendable {
    /// The stated margin, as a share of the declared budget. It is not a
    /// knob: the initializer below implements exactly this number, written as
    /// "the budget less a tenth of it".
    public static let budgetMarginPercent: UInt64 = 90

    private let lock = NSLock()
    private let thresholdBytes: UInt64?
    private var observedFootprintBytes: UInt64 = 0

    /// Reclaim at every boundary. This is the product policy and the default.
    public init() {
        thresholdBytes = nil
    }

    /// Reclaim only once the observed footprint reaches the stated margin of
    /// `declaredBudgetBytes`.
    public init(nearBudgetBytes declaredBudgetBytes: UInt64) {
        // 90% by subtraction rather than multiplication: exact for every
        // budget, and it cannot overflow.
        thresholdBytes = declaredBudgetBytes - declaredBudgetBytes / 10
    }

    /// True when this policy reclaims at every boundary regardless of the
    /// observed footprint.
    public var reclaimsAtEveryBoundary: Bool { thresholdBytes == nil }

    /// The footprint at or above which a bounded policy reclaims. `nil` for
    /// the unconditional policy.
    public var thresholdFootprintBytes: UInt64? { thresholdBytes }

    /// Publish the footprint of the most recent telemetry sample.
    public func observeFootprint(_ bytes: UInt64) {
        lock.lock()
        observedFootprintBytes = bytes
        lock.unlock()
    }

    /// The decision the next boundary will take, without taking it.
    public var reclaimsAtNextBoundary: Bool {
        guard let thresholdBytes else { return true }
        lock.lock()
        let observed = observedFootprintBytes
        lock.unlock()
        return observed >= thresholdBytes
    }

    func reclaimAtBoundary(recordingTo phaseAccounting: DeepSeekV4PhaseAccounting? = nil) {
        guard reclaimsAtNextBoundary else { return }
        DeepSeekV4TransientMemory.reclaim(recordingTo: phaseAccounting)
    }
}

/// A thread-safe cancellation authority for one V4 execution.
///
/// The between-stage checks stop ordinary MLX and file work. The active expert
/// backend is also cancelled so a user pressing Stop can wake a blocked pager
/// acquire instead of waiting for the current layer to finish. A control is
/// terminal after cancellation and must not be reused for another turn.
public final class DeepSeekV4ExecutionControl: @unchecked Sendable {
    private let lock = NSLock()
    private var cancellationReason: String?
    private var activeBackend: PagedRoutedExpertBackend?
    private var residency: DeepSeekV4RunExpertBackends?

    /// Which boundaries this execution reclaims at. It travels with the
    /// control because the control is already threaded to every V4 entry
    /// point that owns a layer or output-head window.
    public let boundaryReclaim: DeepSeekV4BoundaryReclaimPolicy

    /// Serve every layer's expert backend from one run-scoped residency
    /// instead of building and destroying one per layer per pass.
    ///
    /// It travels on the control for the same reason `boundaryReclaim` does:
    /// the control is already threaded to every V4 entry point that owns a
    /// layer. Installing one changes no arithmetic — the same containers, the
    /// same configuration, the same gather — only the lifetime of the readers
    /// and the number of times the shared pool is allocated. Nil is the
    /// per-layer lifetime every record before 2026-08-17 measured.
    public func installExpertResidency(_ residency: DeepSeekV4RunExpertBackends?) {
        lock.lock()
        self.residency = residency
        lock.unlock()
    }

    public var expertResidency: DeepSeekV4RunExpertBackends? {
        lock.lock()
        defer { lock.unlock() }
        return residency
    }

    public init(
        boundaryReclaim: DeepSeekV4BoundaryReclaimPolicy = DeepSeekV4BoundaryReclaimPolicy()
    ) {
        self.boundaryReclaim = boundaryReclaim
    }

    public func cancel(reason: String = "generation stopped") {
        lock.lock()
        if cancellationReason == nil { cancellationReason = reason }
        let recorded = cancellationReason ?? reason
        let backend = activeBackend
        let live = residency
        lock.unlock()
        backend?.cancel(reason: recorded)
        // A run-scoped residency owns readers for layers that are not the
        // active one, and they share the pool a blocked acquire is waiting on.
        // Stop has to reach all of them or a cancelled run can still be asleep
        // in a slot wait for a layer that will never come.
        live?.cancel(reason: recorded)
    }

    public func throwIfCancelled() throws {
        lock.lock()
        let cancelled = cancellationReason != nil
        lock.unlock()
        if cancelled { throw CancellationError() }
    }

    fileprivate func activate(_ backend: PagedRoutedExpertBackend) throws {
        lock.lock()
        if let reason = cancellationReason {
            lock.unlock()
            backend.cancel(reason: reason)
            throw CancellationError()
        }
        activeBackend = backend
        lock.unlock()
    }

    fileprivate func deactivate(_ backend: PagedRoutedExpertBackend) {
        lock.lock()
        if activeBackend === backend { activeBackend = nil }
        lock.unlock()
    }
}

public struct DeepSeekV4PrefillResult: Sendable, Equatable {
    public let logits: [Float]
    public let greedyTokenID: Int
    public let completedLayers: Int

    public init(logits: [Float], greedyTokenID: Int, completedLayers: Int) {
        self.logits = logits
        self.greedyTokenID = greedyTokenID
        self.completedLayers = completedLayers
    }
}

/// Opaque, run-scoped causal state for one admitted V4 model artifact.
///
/// The state owns only the materialized attention caches produced by the
/// complete, config-derived main-layer schedule plus strong references to the
/// exact opened metadata/rooted-file capabilities that produced them. It
/// never retains decoded layer matrices or an expert pager. There is
/// deliberately no public initializer: callers cannot splice together caches
/// from different layers, revisions, or opened filesystem objects.
public struct DeepSeekV4GenerationState {
    fileprivate enum LayerState {
        case hashWindow(DeepSeekV4HashWindowBlock.State)
        case compressedIndexed(DeepSeekV4CompressedIndexedBlock.State)
        case compressedLearned(DeepSeekV4CompressedLearnedBlock.State)

        var position: Int {
            switch self {
            case .hashWindow(let state): state.position
            case .compressedIndexed(let state): state.position
            case .compressedLearned(let state): state.position
            }
        }

        var decodePositionLimit: Int {
            switch self {
            case .hashWindow(let state): state.decodePositionLimit
            case .compressedIndexed(let state): state.decodePositionLimit
            case .compressedLearned(let state): state.decodePositionLimit
            }
        }

        func matches(_ plan: DeepSeekV4LayerPlan) -> Bool {
            switch (self, plan) {
            case (.hashWindow, .hashWindow),
                (.compressedIndexed, .compressedIndexed),
                (.compressedLearned, .compressedLearned):
                return true
            default:
                return false
            }
        }
    }

    fileprivate let artifact: DeepSeekV4ModelArtifact
    fileprivate let layerStates: [LayerState]
    private let nextPosition: Int
    private let positionLimit: Int

    fileprivate init(
        artifact: DeepSeekV4ModelArtifact,
        layerStates: [LayerState],
        nextPosition: Int,
        positionLimit: Int
    ) {
        self.artifact = artifact
        self.layerStates = layerStates
        self.nextPosition = nextPosition
        self.positionLimit = positionLimit
    }

    /// Absolute position consumed by the next decode step.
    public var position: Int { nextPosition }
    /// Absolute, run-scoped position ceiling chosen before prefill.
    public var decodePositionLimit: Int { positionLimit }
    /// Number of materialized main-layer cache states.
    public var layerCount: Int { layerStates.count }
}

/// Product generation state whose layer caches are replaced in place.
///
/// The immutable ``DeepSeekV4GenerationState`` API remains available to
/// callers that need retryable, atomic state publication. A product chat has a
/// different requirement: its declared memory ceiling must not temporarily
/// fund both a complete old cache generation and a complete new one. This
/// session therefore gives up retryability after a decode begins. Each layer's
/// new cache replaces that layer's old cache immediately, and any failure or
/// Stop invalidates and releases the entire session instead of publishing a
/// partially advanced state.
public final class DeepSeekV4GenerationSession {
    fileprivate enum Lifecycle { case ready, advancing, invalid }

    fileprivate let artifact: DeepSeekV4ModelArtifact
    fileprivate var layerStates: [DeepSeekV4GenerationState.LayerState]
    fileprivate var nextPosition: Int
    fileprivate let positionLimit: Int
    fileprivate var lifecycle: Lifecycle = .ready

    fileprivate init(
        artifact: DeepSeekV4ModelArtifact,
        layerStates: [DeepSeekV4GenerationState.LayerState],
        nextPosition: Int,
        positionLimit: Int
    ) {
        self.artifact = artifact
        self.layerStates = layerStates
        self.nextPosition = nextPosition
        self.positionLimit = positionLimit
    }

    public var position: Int { nextPosition }
    public var decodePositionLimit: Int { positionLimit }
    public var layerCount: Int { layerStates.count }
    public var isValid: Bool { lifecycle == .ready }

    /// Explicitly drop every retained cache. A released session is terminal.
    public func release() {
        lifecycle = .invalid
        layerStates.removeAll(keepingCapacity: false)
        DeepSeekV4TransientMemory.reclaim()
    }

    fileprivate func beginAdvance() throws {
        guard lifecycle == .ready else {
            throw DeepSeekV4Error.attention(
                "generation session is already advancing or was released")
        }
        lifecycle = .advancing
    }

    fileprivate func finishAdvance(at position: Int) {
        nextPosition = position
        lifecycle = .ready
    }
}

/// Logits and the atomically advanced state from one V4 generation step.
public struct DeepSeekV4GenerationResult {
    public let logits: [Float]
    public let greedyTokenID: Int
    public let completedLayers: Int
    public let state: DeepSeekV4GenerationState

    fileprivate init(
        logits: [Float],
        greedyTokenID: Int,
        completedLayers: Int,
        state: DeepSeekV4GenerationState
    ) {
        self.logits = logits
        self.greedyTokenID = greedyTokenID
        self.completedLayers = completedLayers
        self.state = state
    }
}

/// First product token together with the sole owner of its bounded cache.
public struct DeepSeekV4GenerationSessionStart {
    public let logits: [Float]
    public let greedyTokenID: Int
    public let completedLayers: Int
    public let session: DeepSeekV4GenerationSession

    fileprivate init(
        logits: [Float], greedyTokenID: Int, completedLayers: Int,
        session: DeepSeekV4GenerationSession
    ) {
        self.logits = logits
        self.greedyTokenID = greedyTokenID
        self.completedLayers = completedLayers
        self.session = session
    }
}

extension DeepSeekV4ModelArtifact {
    /// Execute one causal prefill against the fully admitted publication.
    ///
    /// Embeddings are expanded to the four residual streams exactly as the
    /// official V4 model does. Main layers then run in verified config order.
    /// Each layer owns a fresh bounded expert pager; its result is evaluated,
    /// the pager is shut down, and MLX's recyclable cache is cleared before the
    /// next layer begins. No layer weight or expert tile is intentionally kept
    /// alive across that boundary.
    ///
    /// This compatibility entry point deliberately discards causal state; use
    /// ``prefillGeneration(tokenIDs:decodePositionLimit:expertConfiguration:logitChunkRows:stream:control:onLayerCompleted:)``
    /// for bounded incremental decode. Neither entry point is by itself a
    /// product chat runner: product admission still requires runner/tokenizer
    /// binding, manual-stop event semantics, and real-artifact output/memory
    /// gates.
    public func prefill(
        tokenIDs: [Int],
        expertConfiguration: PagedRoutedExpertBackend.Configuration = .init(),
        logitChunkRows: Int? = nil,
        stream: StreamOrDevice = .default,
        control: DeepSeekV4ExecutionControl = .init(),
        onLayerCompleted: (_ completed: Int, _ total: Int) -> Void = { _, _ in }
    ) throws -> DeepSeekV4PrefillResult {
        guard !tokenIDs.isEmpty else {
            throw DeepSeekV4Error.configuration("prefill requires at least one token")
        }
        try control.throwIfCancelled()

        let embedding = try global.embeddingRows(
            tokenIDs, cancellationCheck: control.throwIfCancelled)
        var residual = try Self.initialResidual(
            embedding: embedding,
            multiplicity: plan.config.hyperConnectionMultiplicity)

        residual = try Self.executeSequence(
            initialResidual: residual,
            layers: layers,
            phaseAccounting: phaseAccounting,
            cancellationCheck: control.throwIfCancelled,
            boundaryReclaim: control.boundaryReclaim,
            execute: { layer, input in
                try execute(
                    layer: layer,
                    residual: input,
                    tokenIDs: tokenIDs,
                    expertConfiguration: expertConfiguration,
                    stream: stream,
                    control: control)
            },
            onLayerCompleted: onLayerCompleted)

        try control.throwIfCancelled()
        let parameters = try global.loadHeadParameters(
            cancellationCheck: control.throwIfCancelled)
        let hidden = try global.finalHidden(from: residual, parameters: parameters)
        guard hidden.ndim == 2, hidden.shape[0] == tokenIDs.count,
            hidden.shape[1] == plan.config.hiddenSize
        else {
            throw DeepSeekV4Error.configuration(
                "final hidden does not match the prompt and model geometry")
        }
        let last = hidden[(hidden.shape[0] - 1)..., 0...]
        // The prefill twin of the decode slice above, and deferred for the same
        // reason: `logits` opens with a blocking pull one statement later.
        MLX.asyncEval([last])
        let logits = try global.logits(
            for: last,
            chunkRows: logitChunkRows,
            cancellationCheck: control.throwIfCancelled,
            boundaryReclaim: control.boundaryReclaim)
        let greedy = try Self.greedyToken(in: logits)
        return DeepSeekV4PrefillResult(
            logits: logits,
            greedyTokenID: greedy,
            completedLayers: layers.count)
    }

    /// Prefill all admitted V4 layers and create one bounded generation state.
    ///
    /// `decodePositionLimit` is an absolute position, not an allocation hint:
    /// every layer cache is bound to it before the first payload read. A limit
    /// smaller than the prompt or above the verified config ceiling is refused
    /// before embeddings, layer tensors, or expert tiles are opened.
    public func prefillGeneration(
        tokenIDs: [Int],
        decodePositionLimit: Int,
        expertConfiguration: PagedRoutedExpertBackend.Configuration = .init(),
        logitChunkRows: Int? = nil,
        stream: StreamOrDevice = .default,
        control: DeepSeekV4ExecutionControl = .init(),
        onLayerCompleted: (_ completed: Int, _ total: Int) -> Void = { _, _ in }
    ) throws -> DeepSeekV4GenerationResult {
        let components = try generationPrefill(
            tokenIDs: tokenIDs,
            decodePositionLimit: decodePositionLimit,
            expertConfiguration: expertConfiguration,
            logitChunkRows: logitChunkRows,
            stream: stream,
            control: control,
            onLayerCompleted: onLayerCompleted)
        let state = DeepSeekV4GenerationState(
            artifact: self,
            layerStates: components.states,
            nextPosition: tokenIDs.count,
            positionLimit: decodePositionLimit)
        return DeepSeekV4GenerationResult(
            logits: components.logits,
            greedyTokenID: components.greedyTokenID,
            completedLayers: layers.count,
            state: state)
    }

    /// Prefill the product-owned, bounded cache session.
    ///
    /// Unlike the immutable compatibility API, this session is the only owner
    /// of the returned layer-state array once the call returns. Subsequent
    /// decode passes can therefore replace entries without triggering an
    /// array copy or retaining a second complete cache generation.
    public func prefillGenerationSession(
        tokenIDs: [Int],
        decodePositionLimit: Int,
        expertConfiguration: PagedRoutedExpertBackend.Configuration = .init(),
        logitChunkRows: Int? = nil,
        stream: StreamOrDevice = .default,
        control: DeepSeekV4ExecutionControl = .init(),
        onLayerCompleted: (_ completed: Int, _ total: Int) -> Void = { _, _ in }
    ) throws -> DeepSeekV4GenerationSessionStart {
        let components = try generationPrefill(
            tokenIDs: tokenIDs,
            decodePositionLimit: decodePositionLimit,
            expertConfiguration: expertConfiguration,
            logitChunkRows: logitChunkRows,
            stream: stream,
            control: control,
            onLayerCompleted: onLayerCompleted)
        let session = DeepSeekV4GenerationSession(
            artifact: self,
            layerStates: components.states,
            nextPosition: tokenIDs.count,
            positionLimit: decodePositionLimit)
        return DeepSeekV4GenerationSessionStart(
            logits: components.logits,
            greedyTokenID: components.greedyTokenID,
            completedLayers: layers.count,
            session: session)
    }

    /// Advance one token through every admitted layer cache and return the
    /// next logits/state pair.
    ///
    /// The caller's `state` remains immutable. New per-layer states are kept
    /// local until every layer and the output head complete. A failure or Stop
    /// therefore publishes no partially advanced model state, and the prior
    /// state remains valid with a fresh execution control.
    public func decode(
        tokenID: Int,
        state: DeepSeekV4GenerationState,
        expertConfiguration: PagedRoutedExpertBackend.Configuration = .init(),
        logitChunkRows: Int? = nil,
        stream: StreamOrDevice = .default,
        control: DeepSeekV4ExecutionControl = .init(),
        onLayerCompleted: (_ completed: Int, _ total: Int) -> Void = { _, _ in }
    ) throws -> DeepSeekV4GenerationResult {
        try validateGenerationState(state)
        guard tokenID >= 0, tokenID < plan.config.vocabularySize else {
            throw DeepSeekV4Error.configuration(
                "decode token id \(tokenID) is outside the verified vocabulary")
        }
        guard state.position < state.decodePositionLimit else {
            throw DeepSeekV4Error.attention(
                "decode position \(state.position) reaches this run's limit "
                    + "\(state.decodePositionLimit)")
        }
        try control.throwIfCancelled()

        let embedding = try global.embeddingRows(
            [tokenID], cancellationCheck: control.throwIfCancelled)
        let initial = try Self.initialResidual(
            embedding: embedding,
            multiplicity: plan.config.hyperConnectionMultiplicity)
        let execution = try Self.executeStateSequence(
            initialResidual: initial,
            layers: layers,
            phaseAccounting: phaseAccounting,
            cancellationCheck: control.throwIfCancelled,
            boundaryReclaim: control.boundaryReclaim,
            execute: { offset, layer, residual in
                try executeGenerationDecode(
                    layer: layer,
                    priorState: state.layerStates[offset],
                    residual: residual,
                    tokenID: tokenID,
                    startPosition: state.position,
                    expertConfiguration: expertConfiguration,
                    stream: stream,
                    control: control)
            },
            onLayerCompleted: onLayerCompleted)
        let advancedPosition = state.position + 1
        try Self.validateStateSchedule(
            layers: layers,
            states: execution.states,
            expectedPosition: advancedPosition,
            expectedPositionLimit: state.decodePositionLimit,
            statePosition: \.position,
            statePositionLimit: \.decodePositionLimit,
            matches: { $0.matches($1.plan) })

        let head = try outputHead(
            residual: execution.residual,
            expectedRows: 1,
            logitChunkRows: logitChunkRows,
            control: control)
        let advanced = DeepSeekV4GenerationState(
            artifact: self,
            layerStates: execution.states,
            nextPosition: advancedPosition,
            positionLimit: state.decodePositionLimit)
        return DeepSeekV4GenerationResult(
            logits: head.logits,
            greedyTokenID: head.greedyTokenID,
            completedLayers: layers.count,
            state: advanced)
    }

    /// Advance the product cache with at most one old/new layer-state pair
    /// alive beyond the one complete retained state set.
    ///
    /// This operation is intentionally consuming. If any layer, the output
    /// head, or cancellation fails, `session` is invalidated and releases all
    /// retained caches. Product code must terminate the run rather than retry
    /// from a partly replaced generation.
    public func decode(
        tokenID: Int,
        session: DeepSeekV4GenerationSession,
        expertConfiguration: PagedRoutedExpertBackend.Configuration = .init(),
        logitChunkRows: Int? = nil,
        stream: StreamOrDevice = .default,
        control: DeepSeekV4ExecutionControl = .init(),
        onLayerCompleted: (_ completed: Int, _ total: Int) -> Void = { _, _ in }
    ) throws -> DeepSeekV4PrefillResult {
        try session.beginAdvance()
        do {
            try validateGenerationSession(session)
            guard tokenID >= 0, tokenID < plan.config.vocabularySize else {
                throw DeepSeekV4Error.configuration(
                    "decode token id \(tokenID) is outside the verified vocabulary")
            }
            guard session.position < session.decodePositionLimit else {
                throw DeepSeekV4Error.attention(
                    "decode position \(session.position) reaches this run's limit "
                        + "\(session.decodePositionLimit)")
            }
            try control.throwIfCancelled()

            let embedding = try global.embeddingRows(
                [tokenID], cancellationCheck: control.throwIfCancelled)
            let initial = try Self.initialResidual(
                embedding: embedding,
                multiplicity: plan.config.hyperConnectionMultiplicity)
            let startPosition = session.position
            let residual = try Self.executeReplacingStateSequence(
                initialResidual: initial,
                layers: layers,
                states: &session.layerStates,
                phaseAccounting: phaseAccounting,
                cancellationCheck: control.throwIfCancelled,
                boundaryReclaim: control.boundaryReclaim,
                execute: { _, layer, residual, priorState in
                    try executeGenerationDecode(
                        layer: layer,
                        priorState: priorState,
                        residual: residual,
                        tokenID: tokenID,
                        startPosition: startPosition,
                        expertConfiguration: expertConfiguration,
                        stream: stream,
                        control: control)
                },
                onLayerCompleted: onLayerCompleted)
            let advancedPosition = startPosition + 1
            try Self.validateStateSchedule(
                layers: layers,
                states: session.layerStates,
                expectedPosition: advancedPosition,
                expectedPositionLimit: session.decodePositionLimit,
                statePosition: \.position,
                statePositionLimit: \.decodePositionLimit,
                matches: { $0.matches($1.plan) })
            let head = try outputHead(
                residual: residual,
                expectedRows: 1,
                logitChunkRows: logitChunkRows,
                control: control)
            session.finishAdvance(at: advancedPosition)
            return DeepSeekV4PrefillResult(
                logits: head.logits,
                greedyTokenID: head.greedyTokenID,
                completedLayers: layers.count)
        } catch {
            session.release()
            throw error
        }
    }

    private func generationPrefill(
        tokenIDs: [Int],
        decodePositionLimit: Int,
        expertConfiguration: PagedRoutedExpertBackend.Configuration,
        logitChunkRows: Int?,
        stream: StreamOrDevice,
        control: DeepSeekV4ExecutionControl,
        onLayerCompleted: (_ completed: Int, _ total: Int) -> Void
    ) throws -> (
        logits: [Float], greedyTokenID: Int,
        states: [DeepSeekV4GenerationState.LayerState]
    ) {
        try Self.validateGenerationRequest(
            tokenIDs: tokenIDs,
            decodePositionLimit: decodePositionLimit,
            vocabularySize: plan.config.vocabularySize,
            maximumPositionCount: plan.config.maximumPositionCount)
        try control.throwIfCancelled()

        let embedding = try global.embeddingRows(
            tokenIDs, cancellationCheck: control.throwIfCancelled)
        let initial = try Self.initialResidual(
            embedding: embedding,
            multiplicity: plan.config.hyperConnectionMultiplicity)
        let execution = try Self.executeStateSequence(
            initialResidual: initial,
            layers: layers,
            phaseAccounting: phaseAccounting,
            cancellationCheck: control.throwIfCancelled,
            boundaryReclaim: control.boundaryReclaim,
            execute: { _, layer, residual in
                try executeGenerationPrefill(
                    layer: layer,
                    residual: residual,
                    tokenIDs: tokenIDs,
                    decodePositionLimit: decodePositionLimit,
                    expertConfiguration: expertConfiguration,
                    stream: stream,
                    control: control)
            },
            onLayerCompleted: onLayerCompleted)
        try Self.validateStateSchedule(
            layers: layers,
            states: execution.states,
            expectedPosition: tokenIDs.count,
            expectedPositionLimit: decodePositionLimit,
            statePosition: \.position,
            statePositionLimit: \.decodePositionLimit,
            matches: { $0.matches($1.plan) })

        let head = try outputHead(
            residual: execution.residual,
            expectedRows: tokenIDs.count,
            logitChunkRows: logitChunkRows,
            control: control)
        return (head.logits, head.greedyTokenID, execution.states)
    }

    static func initialResidual(
        embedding: MLXArray,
        multiplicity: Int
    ) throws -> MLXArray {
        guard embedding.ndim == 2, embedding.shape[0] > 0,
            embedding.shape[1] > 0, multiplicity > 0,
            embedding.dtype.isFloatingPoint, !embedding.dtype.isComplex
        else {
            throw DeepSeekV4Error.configuration(
                "embedding must be a nonempty real matrix and multiplicity must be positive")
        }
        let expanded = expandedDimensions(embedding, axis: 1)
        let residual = repeated(expanded, count: multiplicity, axis: 1)
        // Deferred: an expand-and-repeat of an embedding row the caller has
        // already materialised. The sequence driver counts the same array as
        // its `.passBoundary` a statement later, so a blocking eval here would
        // be an uncounted round trip in front of a deferred one.
        MLX.asyncEval([residual])
        return residual
    }

    static func validateGenerationRequest(
        tokenIDs: [Int],
        decodePositionLimit: Int,
        vocabularySize: Int,
        maximumPositionCount: Int
    ) throws {
        guard !tokenIDs.isEmpty else {
            throw DeepSeekV4Error.configuration(
                "generation prefill requires at least one token")
        }
        guard vocabularySize > 0,
            tokenIDs.allSatisfy({ $0 >= 0 && $0 < vocabularySize })
        else {
            throw DeepSeekV4Error.configuration(
                "generation prompt contains a token outside the verified vocabulary")
        }
        guard decodePositionLimit >= tokenIDs.count,
            decodePositionLimit <= maximumPositionCount
        else {
            throw DeepSeekV4Error.attention(
                "decode position limit \(decodePositionLimit) must contain the prompt "
                    + "and not exceed \(maximumPositionCount)")
        }
    }

    static func executeStateSequence<Layer, State>(
        initialResidual: MLXArray,
        layers: [Layer],
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        cancellationCheck: () throws -> Void,
        boundaryReclaim: DeepSeekV4BoundaryReclaimPolicy = DeepSeekV4BoundaryReclaimPolicy(),
        execute: (
            _ offset: Int,
            _ layer: Layer,
            _ residual: MLXArray
        ) throws -> (residual: MLXArray, state: State),
        onLayerCompleted: (_ completed: Int, _ total: Int) -> Void
    ) throws -> (residual: MLXArray, states: [State]) {
        guard !layers.isEmpty else {
            throw DeepSeekV4Error.configuration(
                "generation state requires at least one model layer")
        }
        var residual = initialResidual
        phaseAccounting?.recordAsyncEval(.passBoundary)
        MLX.asyncEval([residual])
        var states = [State]()
        states.reserveCapacity(layers.count)
        for (offset, layer) in layers.enumerated() {
            try cancellationCheck()
            let next = try autoreleasepool {
                let result = try execute(offset, layer, residual)
                phaseAccounting?.recordAsyncEval(.layerBoundary)
                MLX.asyncEval([result.residual])
                return result
            }
            residual = next.residual
            states.append(next.state)
            // The evaluated residual and run-scoped cache survive. Recyclable
            // tensor/command storage from the completed layer does not — under
            // the unconditional policy, or once the bounded policy's stated
            // margin is reached.
            boundaryReclaim.reclaimAtBoundary(recordingTo: phaseAccounting)
            onLayerCompleted(offset + 1, layers.count)
        }
        try cancellationCheck()
        guard states.count == layers.count else {
            throw DeepSeekV4Error.configuration(
                "generation state did not cover every model layer")
        }
        return (residual, states)
    }

    /// Replace a complete state schedule one layer at a time.
    ///
    /// `states` must be uniquely owned by the caller. On any failure it is
    /// cleared before the error escapes; retaining a partially advanced set
    /// would make both correctness and the memory bound ambiguous.
    static func executeReplacingStateSequence<Layer, State>(
        initialResidual: MLXArray,
        layers: [Layer],
        states: inout [State],
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        cancellationCheck: () throws -> Void,
        boundaryReclaim: DeepSeekV4BoundaryReclaimPolicy = DeepSeekV4BoundaryReclaimPolicy(),
        execute: (
            _ offset: Int,
            _ layer: Layer,
            _ residual: MLXArray,
            _ priorState: State
        ) throws -> (residual: MLXArray, state: State),
        onLayerCompleted: (_ completed: Int, _ total: Int) -> Void
    ) throws -> MLXArray {
        guard !layers.isEmpty, states.count == layers.count else {
            states.removeAll(keepingCapacity: false)
            throw DeepSeekV4Error.configuration(
                "bounded generation replacement requires one state per model layer")
        }
        var residual = initialResidual
        phaseAccounting?.recordAsyncEval(.passBoundary)
        MLX.asyncEval([residual])
        do {
            for (offset, layer) in layers.enumerated() {
                try cancellationCheck()
                var prior: State? = states[offset]
                let next = try autoreleasepool {
                    let result = try execute(offset, layer, residual, prior!)
                    // The layer boundary, counted here for every sequence the
                    // driver runs. On the generation path `execute` has
                    // already submitted this same array a statement earlier,
                    // so one of the two submissions finds nothing unscheduled
                    // and returns immediately; the census counts the *forcing
                    // point*, which is one, and the inner site is commented
                    // rather than counted.
                    phaseAccounting?.recordAsyncEval(.layerBoundary)
                    MLX.asyncEval([result.residual])
                    return result
                }
                states[offset] = next.state
                prior = nil
                residual = next.residual
                boundaryReclaim.reclaimAtBoundary(recordingTo: phaseAccounting)
                onLayerCompleted(offset + 1, layers.count)
            }
            try cancellationCheck()
            return residual
        } catch {
            states.removeAll(keepingCapacity: false)
            DeepSeekV4TransientMemory.reclaim(recordingTo: phaseAccounting)
            throw error
        }
    }

    static func validateStateSchedule<Layer, State>(
        layers: [Layer],
        states: [State],
        expectedPosition: Int,
        expectedPositionLimit: Int,
        statePosition: KeyPath<State, Int>,
        statePositionLimit: KeyPath<State, Int>,
        matches: (State, Layer) -> Bool
    ) throws {
        guard expectedPosition > 0,
            expectedPosition <= expectedPositionLimit,
            states.count == layers.count,
            !layers.isEmpty
        else {
            throw DeepSeekV4Error.attention(
                "generation state has an incomplete layer schedule or invalid position")
        }
        for (state, layer) in zip(states, layers) {
            guard state[keyPath: statePosition] == expectedPosition,
                state[keyPath: statePositionLimit] == expectedPositionLimit,
                matches(state, layer)
            else {
                throw DeepSeekV4Error.attention(
                    "generation layer states do not share one position, limit, and family")
            }
        }
    }

    private func validateGenerationState(
        _ state: DeepSeekV4GenerationState
    ) throws {
        guard state.artifact === self else {
            throw DeepSeekV4Error.artifact(
                "generation state belongs to a different opened model artifact")
        }
        guard state.decodePositionLimit <= plan.config.maximumPositionCount else {
            throw DeepSeekV4Error.attention(
                "generation state exceeds the verified model position limit")
        }
        try Self.validateStateSchedule(
            layers: layers,
            states: state.layerStates,
            expectedPosition: state.position,
            expectedPositionLimit: state.decodePositionLimit,
            statePosition: \.position,
            statePositionLimit: \.decodePositionLimit,
            matches: { $0.matches($1.plan) })
    }

    private func validateGenerationSession(
        _ session: DeepSeekV4GenerationSession
    ) throws {
        guard session.artifact === self else {
            throw DeepSeekV4Error.artifact(
                "generation session belongs to a different opened model artifact")
        }
        guard session.decodePositionLimit <= plan.config.maximumPositionCount else {
            throw DeepSeekV4Error.attention(
                "generation session exceeds the verified model position limit")
        }
        try Self.validateStateSchedule(
            layers: layers,
            states: session.layerStates,
            expectedPosition: session.position,
            expectedPositionLimit: session.decodePositionLimit,
            statePosition: \.position,
            statePositionLimit: \.decodePositionLimit,
            matches: { $0.matches($1.plan) })
    }

    private func outputHead(
        residual: MLXArray,
        expectedRows: Int,
        logitChunkRows: Int?,
        control: DeepSeekV4ExecutionControl
    ) throws -> (logits: [Float], greedyTokenID: Int) {
        try control.throwIfCancelled()
        let parameters = try global.loadHeadParameters(
            cancellationCheck: control.throwIfCancelled)
        let hidden = try global.finalHidden(from: residual, parameters: parameters)
        guard expectedRows > 0,
            hidden.ndim == 2,
            hidden.shape == [expectedRows, plan.config.hiddenSize]
        else {
            throw DeepSeekV4Error.configuration(
                "final hidden does not match the generation step geometry")
        }
        let last = hidden[(expectedRows - 1)..., 0...]
        // Not counted: `finalHidden` submitted `hidden` a statement ago and
        // records that as `.passBoundary`; slicing it forces nothing new.
        // Deferred with it — making this one blocking would reinstate exactly
        // the round trip the pass boundary just gave up, for a slice.
        MLX.asyncEval([last])
        let logits = try global.logits(
            for: last,
            chunkRows: logitChunkRows,
            cancellationCheck: control.throwIfCancelled,
            boundaryReclaim: control.boundaryReclaim)
        let greedy = try Self.greedyToken(in: logits)
        try control.throwIfCancelled()
        return (logits, greedy)
    }

    static func executeSequence<Layer>(
        initialResidual: MLXArray,
        layers: [Layer],
        phaseAccounting: DeepSeekV4PhaseAccounting? = nil,
        cancellationCheck: () throws -> Void,
        boundaryReclaim: DeepSeekV4BoundaryReclaimPolicy = DeepSeekV4BoundaryReclaimPolicy(),
        execute: (Layer, MLXArray) throws -> MLXArray,
        onLayerCompleted: (_ completed: Int, _ total: Int) -> Void
    ) throws -> MLXArray {
        var residual = initialResidual
        phaseAccounting?.recordAsyncEval(.passBoundary)
        MLX.asyncEval([residual])
        for (offset, layer) in layers.enumerated() {
            try cancellationCheck()
            residual = try autoreleasepool {
                let next = try execute(layer, residual)
                phaseAccounting?.recordAsyncEval(.layerBoundary)
                MLX.asyncEval([next])
                return next
            }
            // Evaluated activations remain live. Only recyclable command and
            // tensor storage from the completed layer is discarded here.
            boundaryReclaim.reclaimAtBoundary(recordingTo: phaseAccounting)
            onLayerCompleted(offset + 1, layers.count)
        }
        try cancellationCheck()
        return residual
    }

    static func greedyToken(in logits: [Float]) throws -> Int {
        guard let first = logits.first, first.isFinite,
            logits.allSatisfy(\.isFinite)
        else {
            throw DeepSeekV4Error.configuration(
                "output head produced empty or non-finite logits")
        }
        var bestIndex = 0
        var bestValue = first
        for index in logits.indices.dropFirst() where logits[index] > bestValue {
            bestIndex = index
            bestValue = logits[index]
        }
        return bestIndex
    }

    private func execute(
        layer: Layer,
        residual: MLXArray,
        tokenIDs: [Int],
        expertConfiguration: PagedRoutedExpertBackend.Configuration,
        stream: StreamOrDevice,
        control: DeepSeekV4ExecutionControl
    ) throws -> MLXArray {
        try withLayerBackend(
            layer: layer,
            expertConfiguration: expertConfiguration,
            control: control
        ) { backend in
            let output: MLXArray
            switch layer.plan {
            case .hashWindow(let geometry):
                output = try DeepSeekV4HashWindowBlock.prefill(
                    residual: residual,
                    tokenIDs: tokenIDs,
                    vocabularySize: plan.config.vocabularySize,
                    artifact: layer.artifact,
                    expertUnit: layer.expertUnit,
                    geometry: geometry,
                    routedExperts: backend,
                    stream: stream,
                    cancellationCheck: control.throwIfCancelled).residual

            case .compressedIndexed(let geometry):
                switch geometry.routingKind {
                case .tokenHash:
                    output = try DeepSeekV4CompressedIndexedBlock.prefill(
                        residual: residual,
                        tokenIDs: tokenIDs,
                        vocabularySize: plan.config.vocabularySize,
                        artifact: layer.artifact,
                        expertUnit: layer.expertUnit,
                        geometry: geometry,
                        routedExperts: backend,
                        stream: stream,
                        cancellationCheck: control.throwIfCancelled).residual
                case .learned:
                    output = try DeepSeekV4CompressedIndexedBlock.prefill(
                        residual: residual,
                        artifact: layer.artifact,
                        expertUnit: layer.expertUnit,
                        geometry: geometry,
                        routedExperts: backend,
                        stream: stream,
                        cancellationCheck: control.throwIfCancelled).residual
                }

            case .compressedLearned(let geometry):
                output = try DeepSeekV4CompressedLearnedBlock.prefill(
                    residual: residual,
                    artifact: layer.artifact,
                    expertUnit: layer.expertUnit,
                    geometry: geometry,
                    routedExperts: backend,
                    stream: stream,
                    cancellationCheck: control.throwIfCancelled).residual
            }
            // Deferred, and counted by `executeSequence`'s layer boundary one
            // statement later, exactly as the two generation twins are.
            MLX.asyncEval([output])
            try control.throwIfCancelled()
            return output
        }
    }

    private func executeGenerationPrefill(
        layer: Layer,
        residual: MLXArray,
        tokenIDs: [Int],
        decodePositionLimit: Int,
        expertConfiguration: PagedRoutedExpertBackend.Configuration,
        stream: StreamOrDevice,
        control: DeepSeekV4ExecutionControl
    ) throws -> (
        residual: MLXArray,
        state: DeepSeekV4GenerationState.LayerState
    ) {
        try withLayerBackend(
            layer: layer,
            expertConfiguration: expertConfiguration,
            control: control
        ) { backend in
            let step: (
                residual: MLXArray,
                state: DeepSeekV4GenerationState.LayerState
            )
            switch layer.plan {
            case .hashWindow(let geometry):
                let result = try DeepSeekV4HashWindowBlock.prefill(
                    residual: residual,
                    tokenIDs: tokenIDs,
                    vocabularySize: plan.config.vocabularySize,
                    artifact: layer.artifact,
                    expertUnit: layer.expertUnit,
                    geometry: geometry,
                    routedExperts: backend,
                    decodePositionLimit: decodePositionLimit,
                    stream: stream,
                    cancellationCheck: control.throwIfCancelled)
                guard let state = result.state else {
                    throw DeepSeekV4Error.attention(
                        "hash-window prefill did not create its declared cache")
                }
                step = (result.residual, .hashWindow(state))

            case .compressedIndexed(let geometry):
                let result: DeepSeekV4CompressedIndexedBlock.PrefillResult
                switch geometry.routingKind {
                case .tokenHash:
                    result = try DeepSeekV4CompressedIndexedBlock.prefill(
                        residual: residual,
                        tokenIDs: tokenIDs,
                        vocabularySize: plan.config.vocabularySize,
                        artifact: layer.artifact,
                        expertUnit: layer.expertUnit,
                        geometry: geometry,
                        routedExperts: backend,
                        decodePositionLimit: decodePositionLimit,
                        stream: stream,
                        cancellationCheck: control.throwIfCancelled)
                case .learned:
                    result = try DeepSeekV4CompressedIndexedBlock.prefill(
                        residual: residual,
                        artifact: layer.artifact,
                        expertUnit: layer.expertUnit,
                        geometry: geometry,
                        routedExperts: backend,
                        decodePositionLimit: decodePositionLimit,
                        stream: stream,
                        cancellationCheck: control.throwIfCancelled)
                }
                guard let state = result.state else {
                    throw DeepSeekV4Error.attention(
                        "ratio-four prefill did not create its declared cache")
                }
                step = (result.residual, .compressedIndexed(state))

            case .compressedLearned(let geometry):
                let result = try DeepSeekV4CompressedLearnedBlock.prefill(
                    residual: residual,
                    artifact: layer.artifact,
                    expertUnit: layer.expertUnit,
                    geometry: geometry,
                    routedExperts: backend,
                    decodePositionLimit: decodePositionLimit,
                    stream: stream,
                    cancellationCheck: control.throwIfCancelled)
                guard let state = result.state else {
                    throw DeepSeekV4Error.attention(
                        "ratio-128 prefill did not create its declared cache")
                }
                step = (result.residual, .compressedLearned(state))
            }
            // Not counted, for the reason `executeGenerationDecode`'s twin is
            // not: `executeStateSequence` forces this array immediately
            // after this closure returns and counts the boundary there.
            // Deferred with it — an uncounted *blocking* eval one statement
            // before a deferred boundary would be the round trip the boundary
            // just gave up, hidden from the census that exists to find it.
            MLX.asyncEval([step.residual])
            try control.throwIfCancelled()
            return step
        }
    }

    private func executeGenerationDecode(
        layer: Layer,
        priorState: DeepSeekV4GenerationState.LayerState,
        residual: MLXArray,
        tokenID: Int,
        startPosition: Int,
        expertConfiguration: PagedRoutedExpertBackend.Configuration,
        stream: StreamOrDevice,
        control: DeepSeekV4ExecutionControl
    ) throws -> (
        residual: MLXArray,
        state: DeepSeekV4GenerationState.LayerState
    ) {
        guard priorState.matches(layer.plan) else {
            throw DeepSeekV4Error.attention(
                "generation cache family does not match layer \(layer.plan.layer)")
        }
        return try withLayerBackend(
            layer: layer,
            expertConfiguration: expertConfiguration,
            control: control
        ) { backend in
            let step: (
                residual: MLXArray,
                state: DeepSeekV4GenerationState.LayerState
            )
            switch (layer.plan, priorState) {
            case (.hashWindow(let geometry), .hashWindow(let state)):
                let result = try DeepSeekV4HashWindowBlock.decode(
                    residual: residual,
                    tokenID: tokenID,
                    vocabularySize: plan.config.vocabularySize,
                    artifact: layer.artifact,
                    expertUnit: layer.expertUnit,
                    geometry: geometry,
                    routedExperts: backend,
                    startPosition: startPosition,
                    state: state,
                    stream: stream,
                    cancellationCheck: control.throwIfCancelled)
                step = (result.residual, .hashWindow(result.state))

            case (.compressedIndexed(let geometry), .compressedIndexed(let state)):
                let result: DeepSeekV4CompressedIndexedBlock.DecodeResult
                switch geometry.routingKind {
                case .tokenHash:
                    result = try DeepSeekV4CompressedIndexedBlock.decode(
                        residual: residual,
                        tokenID: tokenID,
                        vocabularySize: plan.config.vocabularySize,
                        artifact: layer.artifact,
                        expertUnit: layer.expertUnit,
                        geometry: geometry,
                        routedExperts: backend,
                        startPosition: startPosition,
                        state: state,
                        stream: stream,
                        cancellationCheck: control.throwIfCancelled)
                case .learned:
                    result = try DeepSeekV4CompressedIndexedBlock.decode(
                        residual: residual,
                        artifact: layer.artifact,
                        expertUnit: layer.expertUnit,
                        geometry: geometry,
                        routedExperts: backend,
                        startPosition: startPosition,
                        state: state,
                        stream: stream,
                        cancellationCheck: control.throwIfCancelled)
                }
                step = (result.residual, .compressedIndexed(result.state))

            case (.compressedLearned(let geometry), .compressedLearned(let state)):
                let result = try DeepSeekV4CompressedLearnedBlock.decode(
                    residual: residual,
                    artifact: layer.artifact,
                    expertUnit: layer.expertUnit,
                    geometry: geometry,
                    routedExperts: backend,
                    startPosition: startPosition,
                    state: state,
                    stream: stream,
                    cancellationCheck: control.throwIfCancelled)
                step = (result.residual, .compressedLearned(result.state))

            default:
                throw DeepSeekV4Error.attention(
                    "generation cache family changed during layer execution")
            }
            // Not recorded in the eval census: `executeReplacingStateSequence`
            // forces this same array immediately after this closure returns,
            // and counts the boundary there. One forcing point, one count —
            // and deferred at both ends, so the count does not understate what
            // the pass actually waits for. **This is the decode path**: a
            // blocking `eval` here would be 43 uncounted round trips a pass.
            MLX.asyncEval([step.residual])
            try control.throwIfCancelled()
            return step
        }
    }

    /// Build one layer's expert backend, bracketed.
    ///
    /// V4 builds and destroys a backend per layer per pass: a fresh
    /// `BufferPool` with page-aligned slots, and per container a `TileReader`
    /// that spawns `queueDepth + 1` threads. The audit of 2026-08-14 (P0-3)
    /// counted ~387 thread create/join cycles per token from this shape and
    /// could not say what they cost. This bracket and
    /// ``shutDownLayerBackend(_:recordingTo:)`` are that cost, and they sit
    /// outside ``DeepSeekV4PhaseMetrics/expertPhaseSeconds``, which brackets
    /// the gather alone.
    static func buildLayerBackend(
        containers: RoutedExpertContainers,
        configuration: PagedRoutedExpertBackend.Configuration,
        successfulReadObserver: (@Sendable (UInt64) -> Void)?,
        phaseAccounting: DeepSeekV4PhaseAccounting?
    ) throws -> PagedRoutedExpertBackend {
        try measuringPhase(phaseAccounting?.recordExpertBackendBuild(nanoseconds:)) {
            try PagedRoutedExpertBackend(
                containers: containers,
                configuration: configuration,
                successfulReadObserver: successfulReadObserver,
                phaseObserver: phaseAccounting)
        }
    }

    /// Tear one layer's expert backend down, bracketed: the reader threads
    /// this joins are the other half of the lifecycle above.
    static func shutDownLayerBackend(
        _ backend: PagedRoutedExpertBackend,
        recordingTo phaseAccounting: DeepSeekV4PhaseAccounting?
    ) {
        measuringPhase(phaseAccounting?.recordExpertBackendShutdown(nanoseconds:)) {
            backend.shutdown()
        }
    }

    private func withLayerBackend<Result>(
        layer: Layer,
        expertConfiguration: PagedRoutedExpertBackend.Configuration,
        control: DeepSeekV4ExecutionControl,
        body: (PagedRoutedExpertBackend) throws -> Result
    ) throws -> Result {
        try control.throwIfCancelled()
        // A run-scoped residency owns the backend's lifetime, so this bracket
        // neither builds nor shuts one down: it borrows the layer's backend,
        // marks it active for cancellation, and gives it back. The build cost
        // it would have paid is charged once per layer per *run*, by the
        // residency, through the same `expertBackendLifecycle` bracket.
        let residency = control.expertResidency
        let backend: PagedRoutedExpertBackend
        if let residency {
            backend = try residency.backend(for: layer.plan.layer)
        } else {
            backend = try Self.buildLayerBackend(
                containers: layer.expertUnit.containers,
                configuration: expertConfiguration,
                successfulReadObserver: { [readAccounting] bytes in
                    readAccounting.recordExpert(bytes, didOverflow: false)
                },
                phaseAccounting: phaseAccounting)
        }
        do {
            try control.activate(backend)
        } catch {
            if residency == nil {
                Self.shutDownLayerBackend(backend, recordingTo: phaseAccounting)
            }
            throw error
        }
        defer {
            control.deactivate(backend)
            if residency == nil {
                Self.shutDownLayerBackend(backend, recordingTo: phaseAccounting)
                let accounting = backend.expertReadAccounting
                if accounting.didOverflow {
                    readAccounting.recordExpert(0, didOverflow: true)
                }
            }
        }
        do {
            let result = try body(backend)
            try control.throwIfCancelled()
            return result
        } catch {
            DeepSeekV4TransientMemory.reclaim(recordingTo: phaseAccounting)
            throw error
        }
    }
}
