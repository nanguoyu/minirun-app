import BenchScenarios
import Darwin
import Foundation
import MLX
import MinirunKit
import ModelAdapters
import StorageCore

#if canImport(UIKit)
    import UIKit
#endif

/// What keeps a declared budget from being a number nobody enforces in time.
///
/// ## Why it exists
///
/// ``DeepSeekV41RunEngine`` already compared `peak - floor` against the
/// declared budget. It did it from two places — after a token, and in
/// `onBlockCompleted` — and on the owner's iPhone 16 Pro, at the 3.4 GB floor,
/// that was not enough: the app was killed by Jetsam twice during **prefill**,
/// `vm-pageshortage`, 5,156 MB and 5,044 MB resident, without the runner's own
/// refusal firing once.
///
/// Three things were wrong with sampling only at those two points, and they
/// compound:
///
/// 1. **`onBlockCompleted` is the narrowest moment of a block, not the widest.**
///    `DeepSeekV41Model.forward` calls it after `source.withBlock`, whose
///    `autoreleasepool` and `DeepSeekV4TransientMemory.reclaim()` have already
///    returned the block's dense weights; the routed gather's operands went
///    with them. Every sample was taken at the trough.
/// 2. **The widest term is inside a block and scales with the prompt.** A
///    prefill holds `3 x tokens x expertsPerToken` half-tiles of one block at
///    once — 112,803,840 B a token at the published geometry. Nothing sampled
///    between the first and the last of them.
/// 3. **The lifetime peak could not rescue it.** `FootprintPeak` consulted
///    `ledger_phys_footprint_peak`, but only above the baseline the process
///    already had, and a phone that is killed never reaches a sample at all.
///
/// So this samples at the seam where the bytes are actually allocated: before
/// every gather operand and every block tensor, through the same closure the
/// cancellation check travels down. The most a run can cross its ceiling by is
/// then one allocation between two checks — ``Policy/budgetOvershootAllowanceBytes``
/// states it.
///
/// The high-water mark of what **this run** is holding.
///
/// ## Why it is measured against a floor
///
/// A V4.1 run does not begin in an empty process. The gate performs a
/// complete verification of 517 GB before it starts, and the App verifies
/// in-process too; measured on this machine, the process footprint after
/// that pass and after the run's own reclaim was **10.0 GiB**, and
/// `malloc_zone_pressure_relief` did not return it.
///
/// That figure is now known to have been almost entirely one allocation,
/// and not the verification pass at all: ``DeepSeekV41RunPositions`` has
/// the arithmetic and the iPhone kills it caused. With the rotary tables
/// sized to the run rather than to `max_position_embeddings`, the floor a
/// V4.1 run starts from is 10.0 GiB smaller — which changes nothing about
/// what this type enforces, because the budget has always bounded what the
/// run **adds** above whatever it found. Enforcing a declared
/// budget against the absolute footprint therefore charges the run for
/// bytes it never allocated and makes every in-process gate impossible at
/// this artifact's scale — the 24 GB probe was cancelled at 24.07 GB while
/// holding 8.1 GiB of pinned weights.
///
/// So the budget bounds what the run **adds**: `peak - floor`, where the
/// floor is the footprint at ``prepareForExecution()``, after the run's own
/// cache clear and allocator reclaim. Both numbers are recorded — the run
/// record carries `entryFootprintBytes` beside `peakFootprintBytes` — so
/// the absolute high-water mark is one addition away and nothing is hidden
/// behind the floor.
///
/// This is a deliberate divergence from ``DeepSeekV4RunEngine``, whose
/// absolute rule has never been the binding constraint at V4's 166 GB, and
/// it is recorded as such in ADR 0021.
final class DeepSeekV41BudgetSentry: @unchecked Sendable {
    private let lock = NSLock()
    private let lifetimeBaseline: UInt64?
    private var floorBytes: UInt64 = 0
    private var absoluteBytes: UInt64 = 0
    private var armed = false
    private var sampleCount: UInt64 = 0
    private var worstOvershootBytes: UInt64 = 0

    let declaredBudgetBytes: UInt64
    private let cancellationCheck: () throws -> Void
    private let exceeded: (UInt64) -> Void

    init(
        declaredBudgetBytes: UInt64,
        cancellationCheck: @escaping () throws -> Void,
        exceeded: @escaping (UInt64) -> Void
    ) {
        self.declaredBudgetBytes = declaredBudgetBytes
        self.cancellationCheck = cancellationCheck
        self.exceeded = exceeded
        self.lifetimeBaseline = ProcessFootprint.current()?.peakFootprintBytes
    }

    /// The floor this run starts from. Set once, before the first allocation,
    /// and never lowered by a later sample. Arms the enforcement.
    func setFloor(_ bytes: UInt64) {
        lock.lock()
        floorBytes = bytes
        armed = true
        lock.unlock()
    }

    var floor: UInt64 { lock.withLock { floorBytes } }
    /// What the process is holding, absolutely.
    var absolute: UInt64 { lock.withLock { absoluteBytes } }
    /// What this run added, which is the number the budget bounds.
    var addedBytes: UInt64 {
        lock.withLock { absoluteBytes > floorBytes ? absoluteBytes - floorBytes : 0 }
    }
    /// How many times the budget was actually looked at. Published because
    /// "the runner enforces its budget" is a claim about a cadence.
    var samples: UInt64 { lock.withLock { sampleCount } }
    /// The largest amount by which an observed peak stood above the declared
    /// budget. Zero on a run that stayed inside it.
    var worstOvershoot: UInt64 { lock.withLock { worstOvershootBytes } }

    /// Fold one footprint reading in and return what this run has added.
    @discardableResult
    func observe(current: UInt64, lifetime: UInt64) -> UInt64 {
        lock.lock()
        sampleCount &+= 1
        absoluteBytes = max(absoluteBytes, current)
        if let lifetimeBaseline, lifetime > lifetimeBaseline {
            absoluteBytes = max(absoluteBytes, lifetime)
        }
        let added = absoluteBytes > floorBytes ? absoluteBytes - floorBytes : 0
        if added > declaredBudgetBytes {
            worstOvershootBytes = max(worstOvershootBytes, added - declaredBudgetBytes)
        }
        lock.unlock()
        return added
    }

    /// Sample the platform and fold the reading in.
    @discardableResult
    func sample() -> UInt64 {
        guard let footprint = ProcessFootprint.current() else { return addedBytes }
        return observe(
            current: footprint.footprintBytes, lifetime: footprint.peakFootprintBytes)
    }

    /// The closure that travels down into the artifact in place of the bare
    /// cancellation check: sample, enforce, then answer the ordinary question.
    ///
    /// Enforcement is armed only once the floor is known, because before that
    /// `peak - floor` is `peak` and every run would refuse itself.
    func check() throws {
        try cancellationCheck()
        let isArmed = lock.withLock { armed }
        guard isArmed else { return }
        let added = sample()
        if added > declaredBudgetBytes { exceeded(added) }
        try cancellationCheck()
    }
}

/// The positions a run can actually reach, which is what its model is built
/// for.
///
/// ## Why this is not `max_position_embeddings`
///
/// It was, until 2026-09-11, and that is what killed the owner's iPhone three
/// times. ``DeepSeekV41Model`` builds one rotary table per backbone block
/// eagerly in its initializer, `positionCount x rope_head_dim/2` float32
/// cosines and the same in sines; DeepSeek V4.1 Flash declares
/// `max_position_embeddings = 1,048,576`, so forty blocks at `rope_head_dim
/// = 64` is **exactly 10.0 GiB of tables** — allocated inside
/// `factory.prepare`, before the budget floor is taken, before the first
/// payload byte is read, and before the flight recorder has anything to
/// record. A Mac absorbed it into the floor the budget is measured from (the
/// run record's 10.0 GiB "entry footprint" was these tables, not the
/// verification pass). An 8 GB phone was killed about nineteen blocks in.
///
/// A chat reaches `promptTokens + maximumNewTokens - 1`, which is the same
/// number ``DeepSeekV41RunEngine`` passes to `prefill` as its decode position
/// limit — 74 for the eleven-token chat that was killed, against 1,048,576.
enum DeepSeekV41RunPositions {
    /// - Returns: a count in `1...config.maximumPositionCount`. A request that
    ///   asks for more positions than the checkpoint has is *not* refused here:
    ///   the engine refuses it by name once it has validated the prompt, and
    ///   clamping only decides how many table rows get built in the meantime.
    static func count(
        promptTokenCount: Int, maximumNewTokens: Int, configuredMaximum: Int
    ) -> Int {
        let reached = max(0, promptTokenCount)
            .addingReportingOverflow(max(1, maximumNewTokens) - 1)
        guard !reached.overflow else { return max(1, configuredMaximum) }
        return min(max(1, reached.partialValue), max(1, configuredMaximum))
    }

    /// The same number, from a request. The prompt is ids or the run is already
    /// refused: ``DeepSeekV41DecodeRunner`` declares `acceptsTextPrompts: false`
    /// and `validate` has run before this.
    static func count(for request: RunRequest, configuredMaximum: Int) -> Int {
        let promptTokenCount: Int
        if case .tokenIDs(let ids) = request.prompt {
            promptTokenCount = ids.count
        } else {
            promptTokenCount = 0
        }
        return count(
            promptTokenCount: promptTokenCount,
            maximumNewTokens: request.maximumNewTokens,
            configuredMaximum: configuredMaximum)
    }
}

/// Opens the exact V4.1 files named by full-verification evidence and performs
/// complete metadata/header admission before a run may be accepted.
///
/// The one structural difference from V4's factory is that **two** configuration
/// documents are opened, not one. ADR 0020 made `inference/config.json` — the
/// file DeepSeek's own runner reads — the authority and the Transformers
/// `config.json` a cross-check, and the container republishes the first at its
/// root as `inference-config.json` with `index.json` naming it under
/// `arguments`. A runner that read only the file `index.json` calls
/// `configuration` would be running a different model than the reference.
struct DeepSeekV41ArtifactWorkloadFactory: DeepSeekV41RunWorkloadFactory {
    let requiresRuntimeAuthority = true

    func prepare(
        request: RunRequest, knobs: DeepSeekV41EffectiveKnobs,
        cancellation: DeepSeekV4RunCancellation,
        sentry: DeepSeekV41BudgetSentry,
        preparation: @escaping DeepSeekV41PreparationReporter
    ) throws -> any DeepSeekV41RunWorkload {
        guard let authority = request.artifact.runtimeAuthority else {
            throw RunError.artifactNotReady(
                "DeepSeek V4.1 requires a rooted runtime authority")
        }
        try cancellation.check()
        try DeepSeekV41DecodeRunner.validateExecutionEvidence(authority)
        preparation("execution evidence validated")

        let identity = authority.evidence.index
        guard let configurationIdentity = identity.configuration,
            let tokenizerIdentity = identity.tokenizer
        else {
            throw RunError.artifactNotReady(
                "the V4.1 execution identities disappeared after validation")
        }
        let huggingFaceData = try Self.verifiedData(
            file: configurationIdentity.file, expectedBytes: configurationIdentity.bytes,
            expectedSHA256: configurationIdentity.sha256, beneath: authority,
            maximumBytes: 1 << 20, cancellation: cancellation)
        let tokenizerData = try Self.verifiedData(
            file: tokenizerIdentity.file, expectedBytes: tokenizerIdentity.bytes,
            expectedSHA256: tokenizerIdentity.sha256, beneath: authority,
            maximumBytes: 32 << 20, cancellation: cancellation)
        let indexData = try authority.openFile("index.json").readAll(maximumBytes: 8 << 20)
        let index = try DeepSeekV41ArtifactIndex(json: indexData)
        // The authority: read through the same rooted opener, and its digest
        // checked against the *index's* own record of it rather than against a
        // constant here.
        let inferenceData = try authority.openFile(index.argumentsFile).readAll(
            maximumBytes: 1 << 20)
        try cancellation.check()

        let config = try DeepSeekV41Config(
            inferenceJSON: inferenceData, huggingFaceJSON: huggingFaceData)
        preparation("configuration loaded")

        let vocabulary = try autoreleasepool {
            try DeepSeekV4Vocabulary(data: tokenizerData)
        }
        preparation("tokenizer loaded")
        for control in DeepSeekV41ChatControl.allCases {
            _ = try vocabulary.id(for: control.rawValue)
        }
        let beginningOfSentence = try vocabulary.id(
            for: DeepSeekV41ChatControl.beginningOfSentence.rawValue)
        let endOfSentence = try vocabulary.id(
            for: DeepSeekV41ChatControl.endOfSentence.rawValue)
        guard vocabulary.vocabularySize == config.vocabularySize else {
            throw RunError.artifactNotReady(
                "the V4.1 tokenizer and model config declare different vocabulary sizes")
        }
        guard beginningOfSentence == config.bosTokenID,
            endOfSentence == config.eosTokenID
        else {
            throw RunError.artifactNotReady(
                "the V4.1 tokenizer controls do not match config bos/eos token ids")
        }

        let access = ModelFileAccess(identityAssurance: .rootedDescriptorIdentity) {
            repositoryPath in
            try authority.openFileDescriptor(repositoryPath)
        }
        let readAccounting = DeepSeekV4ReadAccounting(
            successfulDeterministicReadObserver: {
                [payloadFlow = cancellation.payloadFlow] bytes in
                payloadFlow.recordDeterministic(bytes)
            },
            successfulExpertReadObserver: {
                [payloadFlow = cancellation.payloadFlow] bytes in
                payloadFlow.recordExpert(bytes)
            })

        let artifact = try DeepSeekV41ModelArtifact(
            config: config,
            indexData: indexData,
            fileAccess: access,
            expertPoolSlots: knobs.expertPoolSlots,
            expertQueueDepth: knobs.queueDepth,
            // ADR 0013's held-authority path, and its premise is exactly met
            // here: this run has already hashed every one of these bytes in the
            // complete verification pass over this exact revision, the
            // authority is held open for the run's lifetime, and every open
            // goes back through it with the recorded identity rechecked. ADR
            // 0013 measured the same term at 45.1 s of a 58.2 s V4 decode pass;
            // V4.1 re-reads 72 GB of expert tiles for one prefill, and pure-
            // Swift SHA-256 over that is the pass rather than a check on it.
            verifiesTileDigests: false,
            adoptsExpertOperands: knobs.expertTileAdoption,
            headWindowRows: knobs.logitChunkRows,
            boundsLiveOperands: knobs.boundsLiveOperands,
            readAccounting: readAccounting,
            diagnostics: .off,
            // The sentry's check and not the bare cancellation check: every
            // seam that asks "should I stop?" now also asks "am I over?", and
            // the seams are where the bytes are — before a block tensor is
            // read and before a gather operand is stacked.
            cancellationCheck: sentry.check,
            loadManifest: { unitID in
                try cancellation.check()
                return try authority.openFile("\(unitID)/manifest.json").readAll(
                    maximumBytes: 8 << 20)
            })
        guard artifact.sourceRepository == configurationIdentity.sourceRepo,
            artifact.sourceRevision == configurationIdentity.sourceRevision,
            artifact.sourceRepository == tokenizerIdentity.sourceRepo,
            artifact.sourceRevision == tokenizerIdentity.sourceRevision
        else {
            throw RunError.artifactNotReady(
                "the opened V4.1 units, config, and tokenizer do not share one source identity")
        }

        preparation("unit manifests reconciled")

        let licence = try authority.openFileDescriptor(tokenizerIdentity.license)
        _ = close(licence)
        try authority.validateCurrentBinding()
        try cancellation.check()

        // The Engram compressed-token map is built from the *verified*
        // tokenizer, and the hasher refuses a checkpoint whose compressed
        // vocabulary is not the one the shipped multipliers were drawn from.
        let tokenMap = try DeepSeekV41EngramTokenMap.build(vocabulary: vocabulary)
        preparation("engram token map built")
        let workload = try DeepSeekV41ArtifactWorkload(
            artifact: artifact, authority: authority, tokenMap: tokenMap,
            knobs: knobs, cancellation: cancellation, sentry: sentry,
            endOfSentenceIDs: [endOfSentence],
            positionCount: DeepSeekV41RunPositions.count(
                for: request, configuredMaximum: config.maximumPositionCount))
        // The step that killed three chats, and the reason it is named: this is
        // where forty blocks of rotary tables are built. See
        // ``DeepSeekV41RunPositions``.
        preparation("model built")
        return workload
    }

    private static func verifiedData(
        file: String, expectedBytes: UInt64, expectedSHA256: String,
        beneath authority: ArtifactRuntimeAuthority,
        maximumBytes: UInt64,
        cancellation: DeepSeekV4RunCancellation
    ) throws -> Data {
        try cancellation.check()
        let data = try authority.openFile(file).readAll(maximumBytes: maximumBytes)
        guard UInt64(data.count) == expectedBytes,
            SHA256.hexString(SHA256.hash(data)) == expectedSHA256
        else {
            throw RunError.artifactNotReady(
                "\(file) does not match the size and SHA-256 recorded by index.json")
        }
        try cancellation.check()
        return data
    }
}

/// The production workload: the admitted artifact, the model over it, and one
/// bounded generation state. All three are dropped by ``shutdown()`` before the
/// engine emits a terminal event.
final class DeepSeekV41ArtifactWorkload: DeepSeekV41RunWorkload {
    private var artifact: DeepSeekV41ModelArtifact?
    private var authority: ArtifactRuntimeAuthority?
    private var model: DeepSeekV41Model?
    private var state: DeepSeekV41GenerationState?
    private let knobs: DeepSeekV41EffectiveKnobs
    private let cancellation: DeepSeekV4RunCancellation
    private let sentry: DeepSeekV41BudgetSentry
    private var beganExecution = false

    let blockCount: Int
    let vocabularySize: Int
    let eosTokenIDs: Set<Int>
    let maximumPositionCount: Int
    let sourceRepository: String
    let sourceRevision: String
    let expertPoolBudgetBytes: UInt64
    /// One expert's half of a pair tile, in bytes: the unit the pool reserves
    /// in and the unit the routed gather's operands are counted in.
    let expertTileStrideBytes: UInt64
    private let config: DeepSeekV41Config
    private let readAccounting: DeepSeekV4ReadAccounting
    private let phaseAccounting: DeepSeekV4PhaseAccounting
    private var releasedPinnedBytes: UInt64 = 0
    private var releasedPinnedBlocks = 0
    private var releasedPinsHead = false
    private var releasedTileReads = 0
    private var releasedTileHits = 0

    init(
        artifact: DeepSeekV41ModelArtifact,
        authority: ArtifactRuntimeAuthority,
        tokenMap: DeepSeekV41EngramTokenMap.Map,
        knobs: DeepSeekV41EffectiveKnobs,
        cancellation: DeepSeekV4RunCancellation,
        sentry: DeepSeekV41BudgetSentry,
        endOfSentenceIDs: Set<Int>,
        positionCount: Int
    ) throws {
        let config = artifact.config
        self.artifact = artifact
        self.authority = authority
        self.knobs = knobs
        self.cancellation = cancellation
        self.sentry = sentry
        self.blockCount = config.numberOfLayers
        self.vocabularySize = config.vocabularySize
        self.eosTokenIDs = endOfSentenceIDs
        self.maximumPositionCount = config.maximumPositionCount
        self.sourceRepository = artifact.sourceRepository
        self.sourceRevision = artifact.sourceRevision
        self.expertPoolBudgetBytes = artifact.expertPoolBudgetBytes
        // `try?`, so that an artifact with no representable stride is refused
        // by the pool-budget guard below with the sentence it already had,
        // rather than by a different error from one line earlier.
        self.expertTileStrideBytes = (try? artifact.expertTileStrideBytes())
            .map(UInt64.init) ?? 0
        self.config = config
        self.readAccounting = artifact.readAccounting
        self.phaseAccounting = artifact.phaseAccounting
        guard expertPoolBudgetBytes > 0 else {
            throw RunError.artifactNotReady(
                "the V4.1 artifact has no representable routed-expert tile stride")
        }
        self.model = try DeepSeekV41Model(
            config: config,
            source: artifact,
            engramIndices: {
                try DeepSeekV41PublishedEngramIndexSource(map: tokenMap, config: config)
            },
            // The positions this run can reach, and not the checkpoint's
            // declared ceiling: see ``DeepSeekV41RunPositions``.
            positionCount: positionCount,
            // The reference quantizes the activation to E4M3 before every
            // expert GEMM and it costs percent, not ulps
            // (`docs/experiments/2026-09-11-v41-phase2-moe.md` §5). A run that
            // skipped it would not be slightly more accurate; it would compute
            // a different function.
            expertActivation: .referenceFP8,
            boundsLiveOperands: knobs.boundsLiveOperands,
            phaseAccounting: artifact.phaseAccounting,
            // The guard-only finiteness sweeps are diagnostics and this run
            // still fails closed: the head refuses a non-finite window and the
            // greedy pick refuses a non-finite vector.
            diagnostics: .off)
    }

    var readAccountingSnapshot: DeepSeekV4ReadAccountingSnapshot? {
        readAccounting.snapshot
    }

    var phaseMetricsSnapshot: DeepSeekV4PhaseMetrics? { phaseAccounting.snapshot }

    var deterministicCensus: DeepSeekV4MemoryDial.Census? { artifact?.census }

    func productMemoryPlan(
        declaredBudgetBytes: UInt64,
        promptTokenCount: Int,
        maximumNewTokens: Int,
        mlxCacheBytes: UInt64,
        pinnedDeterministicBytes: UInt64,
        enforcesProductLimits: Bool
    ) throws -> DeepSeekV41ProductMemoryPlan {
        try DeepSeekV41ProductMemoryBudget.plan(
            declaredBudgetBytes: declaredBudgetBytes,
            config: config,
            promptTokenCount: promptTokenCount,
            maximumNewTokens: maximumNewTokens,
            expertPoolBytes: expertPoolBudgetBytes,
            mlxCacheBytes: mlxCacheBytes,
            pinnedDeterministicBytes: pinnedDeterministicBytes,
            expertTileStrideBytes: expertTileStrideBytes,
            enforcesProductLimits: enforcesProductLimits)
    }

    func installPinnedTier(blocks: Set<Int>, outputHead: Bool) throws {
        try artifact?.installPinnedTier(blocks: blocks, outputHead: outputHead)
    }

    var pinnedResidentBytes: UInt64 { artifact?.pinnedResidentBytes ?? releasedPinnedBytes }
    var pinnedBlockCount: Int { artifact?.pinnedBlockCount ?? releasedPinnedBlocks }
    var pinsOutputHead: Bool { artifact?.pinsOutputHead ?? releasedPinsHead }
    var expertTileReads: Int { artifact?.expertTileReads ?? releasedTileReads }
    var expertTileHits: Int { artifact?.expertTileHits ?? releasedTileHits }

    func prefill(
        tokenIDs: [Int], decodePositionLimit: Int,
        cancellationCheck: () throws -> Void,
        onBlockCompleted: (Int, Int) -> Void
    ) throws -> DeepSeekV41WorkloadStep {
        try beginIfNeeded(cancellationCheck: cancellationCheck)
        guard let model else {
            throw RunError.artifactNotReady("the V4.1 workload was already released")
        }
        let result = try model.prefill(
            tokenIDs: tokenIDs,
            decodePositionLimit: decodePositionLimit,
            cancellationCheck: cancellationCheck,
            onBlockCompleted: onBlockCompleted)
        state = result.state
        return DeepSeekV41WorkloadStep(
            tokenID: result.step.greedyTokenID, logits: result.step.logits,
            completedBlocks: result.step.completedBlocks)
    }

    func decode(
        tokenID: Int, cancellationCheck: () throws -> Void,
        onBlockCompleted: (Int, Int) -> Void
    ) throws -> DeepSeekV41WorkloadStep {
        try beginIfNeeded(cancellationCheck: cancellationCheck)
        guard let model, state != nil else {
            throw RunError.artifactNotReady(
                "V4.1 decode has no admitted artifact and prefill state")
        }
        // Consuming, like V4's session decode and for the same reason: the
        // caches are replaced in place, so a run's declared ceiling never funds
        // a complete old generation and a complete new one at once.
        var advancing = state!
        state = nil
        do {
            let step = try model.decode(
                tokenID: tokenID, state: &advancing,
                cancellationCheck: cancellationCheck,
                onBlockCompleted: onBlockCompleted)
            state = advancing
            return DeepSeekV41WorkloadStep(
                tokenID: step.greedyTokenID, logits: step.logits,
                completedBlocks: step.completedBlocks)
        } catch {
            advancing.release()
            throw error
        }
    }

    func validateTerminalBinding() throws {
        guard let authority else {
            throw RunError.artifactNotReady("the V4.1 runtime authority was released early")
        }
        try authority.validateAllFilesCurrent()
    }

    func shutdown() {
        releasedPinnedBytes = artifact?.pinnedResidentBytes ?? releasedPinnedBytes
        releasedPinnedBlocks = artifact?.pinnedBlockCount ?? releasedPinnedBlocks
        releasedPinsHead = artifact?.pinsOutputHead ?? releasedPinsHead
        releasedTileReads = artifact?.expertTileReads ?? releasedTileReads
        releasedTileHits = artifact?.expertTileHits ?? releasedTileHits
        state?.release()
        state = nil
        model = nil
        artifact?.release()
        artifact = nil
        authority = nil
        Self.reclaimTransientMemory()
    }

    func prepareForExecution() throws {
        MLX.Memory.cacheLimit = knobs.mlxCacheLimitBytes
        Self.reclaimTransientMemory()
        beganExecution = true
    }

    private func beginIfNeeded(cancellationCheck: () throws -> Void) throws {
        try cancellationCheck()
        if !beganExecution { try prepareForExecution() }
    }

    private static func reclaimTransientMemory() {
        MLX.Memory.clearCache()
        _ = malloc_zone_pressure_relief(nil, 0)
    }
}

/// The product engine for V4.1: acceptance, the pass loop, the events a chat
/// draws, and the run record a gate compares.
///
/// Byte for byte the same *shape* as ``DeepSeekV4RunEngine``, because a record
/// a reader can put beside a V4 record is worth more than a tidier one, and
/// because the logits digest has to be comparable: ``logitsDigest(_:)`` uses
/// `DeepSeekV4LogitsDump.rawBytes`, so a V4 and a V4.1 digest of the same vector
/// are the same string.
final class DeepSeekV41RunEngine {
    struct TeardownRecord: Codable, Sendable, Equatable {
        var workloadWasPrepared = false
        var workloadReleased = false
        var scopeWasHeld = false
        var scopeReleased = false
        var mlxCacheBytesAfterClear: UInt64 = 0
    }

    private enum Outcome: String { case finished, cancelled }

    private let request: RunRequest
    private let capabilities: RunnerCapabilities
    private let knobs: DeepSeekV41EffectiveKnobs
    private let enforcesProductMemoryPolicy: Bool
    private let handle: RunHandleID
    private let cancellation: DeepSeekV4RunCancellation
    private let factory: any DeepSeekV41RunWorkloadFactory
    private let continuation: AsyncThrowingStream<RunEvent, Error>.Continuation

    private let started = Date()
    private let startedTick = MonotonicClock.now()
    private var phaseName = "starting"
    private var generationStage: RunGenerationStage = .preparing
    private var thermalTrail: [ThermalTransition] = []
    private let sentry: DeepSeekV41BudgetSentry
    private var tokenIDs: [Int] = []
    private var passSeconds: [Double] = []
    private var lastLogits: [Float] = []
    private let logitsDump = DeepSeekV41LogitsDump.shared
    private var workload: (any DeepSeekV41RunWorkload)?
    private var environment: EnvironmentReport?
    private var sourceRepository = ""
    private var sourceRevision = ""
    private var productMemoryPlan: DeepSeekV41ProductMemoryPlan?
    private var pinPlan: PinPlan?
    private var completedTokenBytes: ByteAccounting?
    private var prefillBytes: ByteAccounting?
    private var phaseBoundary: DeepSeekV4PhaseMetrics?
    private var prefillPhaseMetrics: DeepSeekV4PhaseMetrics?
    private var decodePhaseMetrics: [DeepSeekV4PhaseMetrics] = []
    private var terminalReadAccounting: DeepSeekV4ReadAccountingSnapshot?
    private var teardownRecord: TeardownRecord?
    private var terminalPinnedBytes: UInt64 = 0
    /// What the process was already holding when this run's first allocation
    /// was made. See where it is set.
    private var entryFootprintBytes: UInt64 = 0
    private var terminalPinnedBlocks = 0
    private var terminalPinsHead = false

    init(
        request: RunRequest, capabilities: RunnerCapabilities,
        knobs: DeepSeekV41EffectiveKnobs,
        enforcesProductMemoryPolicy: Bool,
        handle: RunHandleID,
        cancellation: DeepSeekV4RunCancellation,
        factory: any DeepSeekV41RunWorkloadFactory,
        continuation: AsyncThrowingStream<RunEvent, Error>.Continuation
    ) {
        self.request = request
        self.capabilities = capabilities
        self.knobs = knobs
        self.enforcesProductMemoryPolicy = enforcesProductMemoryPolicy
        self.handle = handle
        self.cancellation = cancellation
        self.factory = factory
        self.continuation = continuation
        sentry = DeepSeekV41BudgetSentry(
            declaredBudgetBytes: request.memoryBudgetBytes,
            cancellationCheck: cancellation.check,
            exceeded: { [cancellation] added in cancellation.exceedBudget(peakBytes: added) })
    }

    func run() {
        recordThermal(.unknown, at: 0, phase: phaseName)
        do {
            try execute()
            let teardown = tearDown()
            continuation.yield(.finished(summary(outcome: .finished, teardown: teardown)))
            continuation.finish()
        } catch {
            let overBudget = cancellation.budgetExceededPeak
            let artifactFailure = cancellation.artifactFailure
            let wasCancelled = Self.isCancellation(error) || cancellation.isCancelled
            let teardown = tearDown()
            if let peak = overBudget {
                continuation.finish(
                    throwing: RunError.budgetExceeded(
                        peakBytes: peak, declaredBytes: request.memoryBudgetBytes))
            } else if let artifactFailure {
                continuation.finish(throwing: RunError.artifactNotReady(artifactFailure))
            } else if wasCancelled {
                continuation.yield(.cancelled(summary(outcome: .cancelled, teardown: teardown)))
                continuation.finish()
            } else {
                continuation.finish(throwing: error)
            }
        }
    }

    private func execute() throws {
        try cancellation.check()
        environment = EnvironmentReport.current(path: request.artifact.root.path)
        let prepared = try factory.prepare(
            request: request, knobs: knobs, cancellation: cancellation, sentry: sentry,
            preparation: notePreparation)
        workload = prepared
        sourceRepository = prepared.sourceRepository
        sourceRevision = prepared.sourceRevision

        guard prepared.blockCount > 0, prepared.vocabularySize > 0 else {
            throw RunError.artifactNotReady(
                "the admitted V4.1 workload has no blocks or vocabulary")
        }
        guard let cacheBytes = UInt64(exactly: knobs.mlxCacheLimitBytes) else {
            throw RunError.knobOutOfRange(
                name: "mlxCacheLimitBytes", value: "\(knobs.mlxCacheLimitBytes)",
                allowed: "a non-negative UInt64 byte count")
        }
        guard case .tokenIDs(let promptTokenIDs) = request.prompt else {
            throw RunError.promptNotSupported(
                reason: "the V4.1 runner accepts only ids from its verified tokenizer")
        }
        guard promptTokenIDs.allSatisfy({ $0 >= 0 && $0 < prepared.vocabularySize }) else {
            throw RunError.promptNotSupported(
                reason: "the prompt contains an id outside the verified V4.1 vocabulary")
        }
        let positionLimit = promptTokenIDs.count.addingReportingOverflow(
            request.maximumNewTokens - 1)
        guard !positionLimit.overflow,
            positionLimit.partialValue <= prepared.maximumPositionCount
        else {
            throw RunError.promptNotSupported(
                reason: "the prompt and requested response exceed the verified V4.1 "
                    + "position limit")
        }

        let explicitlyReserved = prepared.expertPoolBudgetBytes.addingReportingOverflow(
            cacheBytes)
        guard !explicitlyReserved.overflow else {
            throw RunError.artifactNotReady(
                "the V4.1 expert pool and MLX cache reservation exceed UInt64.max")
        }
        let requiredBudget: UInt64
        if enforcesProductMemoryPolicy {
            let plan = try prepared.productMemoryPlan(
                declaredBudgetBytes: request.memoryBudgetBytes,
                promptTokenCount: promptTokenIDs.count,
                maximumNewTokens: request.maximumNewTokens,
                mlxCacheBytes: cacheBytes,
                pinnedDeterministicBytes: 0,
                enforcesProductLimits: true)
            productMemoryPlan = plan
            requiredBudget = plan.requiredBudgetBytes
        } else {
            let planned = try planPinnedTier(
                workload: prepared, promptTokenCount: promptTokenIDs.count,
                cacheBytes: cacheBytes)
            pinPlan = planned
            let withPins = explicitlyReserved.partialValue.addingReportingOverflow(
                planned?.pinnedBytes ?? 0)
            guard !withPins.overflow else {
                throw RunError.artifactNotReady(
                    "the V4.1 reservation and pinned tier exceed UInt64.max")
            }
            requiredBudget = withPins.partialValue
        }
        guard requiredBudget <= request.memoryBudgetBytes else {
            throw RunError.budgetBelowMinimum(
                declared: request.memoryBudgetBytes,
                minimum: requiredBudget,
                model: capabilities.model)
        }
        try refuseABudgetTheDeviceCannotGive(requiredBudget: requiredBudget)
        try cancellation.check()

        try prepared.prepareForExecution()
        // The floor this run starts from, after its own reclaim and before it
        // allocates anything. Recorded because a run does not begin in an empty
        // process: this harness verifies 517 GB in-process first, and the App
        // verifies in-process too, so "peak footprint" is only interpretable
        // beside the number it started from.
        entryFootprintBytes = ProcessFootprint.current()?.footprintBytes ?? 0
        sentry.setFloor(entryFootprintBytes)
        continuation.yield(
            .log(
                "V4.1 entry footprint " + ByteSize.format(entryFootprintBytes)
                    + " before pinning"))
        if let pinPlan, pinPlan.pinnedBytes > 0 {
            try prepared.installPinnedTier(
                blocks: Set(
                    pinPlan.pinnedLayers.compactMap { decision in
                        guard case .deterministicLayer(let block) = decision.unit else {
                            return nil
                        }
                        return block
                    }),
                outputHead: pinPlan.pinnedGlobals?.unit == .outputHead)
        }
        // The floor is set, so this one folds: it is the first reading the
        // budget is actually measured from, and on a pinning run it is taken
        // after the tier is resident.
        continuation.yield(.telemetry(sample(phase: "execution floor", stage: .preparing)))

        continuation.yield(
            .accepted(
                RunAcceptance(
                    handle: handle, model: capabilities.model,
                    declaredBudgetBytes: request.memoryBudgetBytes,
                    pinPlan: pinPlan, effectiveKnobs: knobs.asRunKnobs,
                    artifactRoot: request.artifact.root.path,
                    startedAt: started)))
        if let pinPlan {
            let head = pinPlan.pinnedGlobals?.unit == .outputHead ? " + output head" : ""
            continuation.yield(
                .log(
                    "V4.1 memory dial: \(pinPlan.pinnedLayers.count) of "
                        + "\(pinPlan.layerCount ?? 0) blocks\(head) pinned, "
                        + ByteSize.format(pinPlan.pinnedBytes) + " resident, "
                        + ByteSize.format(pinPlan.projectedBytesPerTokenSaved)
                        + " saved per later pass"))
        }
        continuation.yield(
            .log(
                "V4.1 source \(sourceRepository)@\(sourceRevision); "
                    + "\(prepared.blockCount) blocks; expert pool "
                    + ByteSize.format(prepared.expertPoolBudgetBytes)))
        if let plan = productMemoryPlan {
            continuation.yield(
                .log(
                    "V4.1 bounded memory plan: retained state "
                        + ByteSize.format(plan.retainedStateBytes)
                        + ", one-block replacement "
                        + ByteSize.format(plan.replacementBlockStateBytes)
                        + ", headroom " + ByteSize.format(plan.headroomBytes)))
        }

        phaseName = "prefill"
        generationStage = .prefill
        continuation.yield(
            .phase(
                RunPhase(
                    name: phaseName, generationStage: generationStage,
                    fraction: 0, detail: "reading prompt")))
        phaseBoundary = prepared.phaseMetricsSnapshot
        let prefillStarted = MonotonicClock.now()
        let first = try autoreleasepool {
            try prepared.prefill(
                tokenIDs: promptTokenIDs,
                decodePositionLimit: positionLimit.partialValue,
                // The sentry's check, so the model's own per-block seam is a
                // budget check too and not only a cancellation one.
                cancellationCheck: sentry.check,
                onBlockCompleted: observeBlock)
        }
        try record(
            first, seconds: MonotonicClock.seconds(since: prefillStarted), prefill: true)

        if !prepared.eosTokenIDs.contains(first.tokenID), request.maximumNewTokens > 1 {
            for index in 1..<request.maximumNewTokens {
                try cancellation.check()
                phaseName = "decode token \(index + 1)"
                generationStage = .decode
                continuation.yield(
                    .phase(
                        RunPhase(
                            name: phaseName, generationStage: generationStage,
                            fraction: 0, detail: "reading blocks")))
                phaseBoundary = prepared.phaseMetricsSnapshot
                let passStarted = MonotonicClock.now()
                let next = try autoreleasepool {
                    try prepared.decode(
                        tokenID: tokenIDs.last!,
                        cancellationCheck: sentry.check,
                        onBlockCompleted: observeBlock)
                }
                try record(
                    next, seconds: MonotonicClock.seconds(since: passStarted),
                    prefill: false)
                if prepared.eosTokenIDs.contains(next.tokenID) { break }
            }
        }

        try cancellation.check()
        try prepared.validateTerminalBinding()
        try cancellation.check()
    }

    /// Ask the device whether it can give this budget, and refuse in a sentence
    /// if it cannot — **before** the first allocation.
    ///
    /// Spec §4.4 asks the runtime to treat the budget as advisory and moving
    /// rather than fixed, and names `os_proc_available_memory()` as the source.
    /// ``ProcessFootprint/availableMemory()`` is that call; it exists only on
    /// the platforms that impose a per-process limit, and it returns what this
    /// process may still allocate before it is at risk — which is exactly the
    /// question a declared budget asks.
    ///
    /// The comparison is against `required + overshoot`, not `required`: the
    /// budget promise this runner makes is "stops rather than crossing, by at
    /// most ``DeepSeekV41ProductMemoryBudget/Policy/budgetOvershootAllowanceBytes``",
    /// and a device that cannot fund the overshoot cannot fund the promise.
    ///
    /// On macOS the call does not exist and this is a no-op: a Mac's answer to
    /// an over-large budget is paging, and phase 3's arms are what say what a
    /// Mac can hold. The iPhone 16 Pro's answer was `SIGKILL`.
    private func refuseABudgetTheDeviceCannotGive(requiredBudget: UInt64) throws {
        guard let available = ProcessFootprint.availableMemory() else { return }
        let allowance = DeepSeekV41ProductMemoryBudget.currentPolicy
            .budgetOvershootAllowanceBytes
        let needed = requiredBudget.addingReportingOverflow(allowance)
        guard needed.overflow || needed.partialValue > available else {
            continuation.yield(
                .log(
                    "V4.1 device memory: " + ByteSize.format(available)
                        + " still available to this process, "
                        + ByteSize.format(needed.partialValue) + " needed"))
            return
        }
        throw RunError.runnerUnavailable(
            capabilities.model,
            reason: "this device says \(ByteSize.format(available)) is still available to "
                + "Minirun and this chat needs \(ByteSize.format(requiredBudget)) plus "
                + "\(ByteSize.format(allowance)) of stated overshoot; close something or "
                + "state a smaller budget, because a run that starts here is killed "
                + "rather than refused")
    }

    /// Turn a stated budget into a residency plan, or refuse it by name.
    ///
    /// The dial's floor is priced from the product plan's own terms, built with
    /// an unbounded declared budget so it prices the run rather than admitting
    /// it. The 2.9 GB product floor is deliberately not part of it: that is the
    /// product scale's policy, and a stated harness declares its own.
    private func planPinnedTier(
        workload: any DeepSeekV41RunWorkload, promptTokenCount: Int, cacheBytes: UInt64
    ) throws -> PinPlan? {
        guard let census = workload.deterministicCensus else { return nil }
        let priced = try workload.productMemoryPlan(
            declaredBudgetBytes: .max,
            promptTokenCount: promptTokenCount,
            maximumNewTokens: request.maximumNewTokens,
            mlxCacheBytes: cacheBytes,
            pinnedDeterministicBytes: 0,
            enforcesProductLimits: false)
        guard let floor = DeepSeekV41MemoryDialInputs.floor(pricing: priced) else {
            return nil
        }
        guard request.memoryBudgetBytes >= floor.totalBytes else { return nil }
        return try DeepSeekV4MemoryDial.plan(
            budgetBytes: request.memoryBudgetBytes,
            census: census,
            floor: floor,
            maximumNewTokens: request.maximumNewTokens)
    }

    private func record(
        _ step: DeepSeekV41WorkloadStep, seconds: Double, prefill: Bool
    ) throws {
        guard step.completedBlocks == workload?.blockCount,
            step.tokenID >= 0,
            step.tokenID < (workload?.vocabularySize ?? 0),
            !step.logits.isEmpty,
            step.logits.count == workload?.vocabularySize,
            step.logits.allSatisfy(\.isFinite)
        else {
            throw RunError.artifactNotReady(
                "the V4.1 generation step returned incomplete blocks, token, or logits")
        }
        passSeconds.append(seconds)
        completedTokenBytes = byteAccounting()
        if prefill { prefillBytes = completedTokenBytes }
        recordPhaseSplit(seconds: seconds, prefill: prefill)
        tokenIDs.append(step.tokenID)
        lastLogits = step.logits
        logitsDump?.record(
            position: tokenIDs.count - 1, logits: step.logits, tokenID: step.tokenID,
            isPrefillToken: prefill)
        continuation.yield(
            .token(
                TokenEvent(
                    index: tokenIDs.count - 1, tokenID: step.tokenID,
                    text: nil, secondsSincePreviousToken: seconds,
                    isPrefillToken: prefill)))
        let telemetry = sample(phase: phaseName)
        recordThermal(
            telemetry.thermalState, at: telemetry.elapsed, phase: telemetry.phase)
        continuation.yield(.telemetry(telemetry))
        enforceBudget(telemetry)
        try cancellation.check()
    }

    private func recordPhaseSplit(seconds: Double, prefill: Bool) {
        guard let earlier = phaseBoundary,
            let current = workload?.phaseMetricsSnapshot,
            let split = current.subtracting(earlier)?.withPassSeconds(seconds),
            split.isAccountingBalanced
        else {
            phaseBoundary = nil
            return
        }
        phaseBoundary = nil
        if prefill {
            prefillPhaseMetrics = split
        } else {
            decodePhaseMetrics.append(split)
        }
        if let summary = split.runPhaseSummary(
            passKind: prefill ? .prefill : .decode,
            passIndex: prefill ? 0 : decodePhaseMetrics.count)
        {
            continuation.yield(.phaseSummary(summary))
        }
        continuation.yield(.log(split.summaryLine))
        // Its own line beside the phase's: a V4 pass measures none of it, and a
        // V4 record has to read exactly as it did before the split existed.
        if let gather = split.expertGatherLine { continuation.yield(.log(gather)) }
    }

    private func observeBlock(completed: Int, total: Int) {
        guard completed >= 1, total >= 1, completed <= total else {
            cancellation.failArtifact(
                "the V4.1 workload reported an invalid block progress boundary")
            return
        }
        let telemetry = sample(phase: phaseName)
        recordThermal(
            telemetry.thermalState, at: telemetry.elapsed, phase: telemetry.phase)
        continuation.yield(
            .phase(
                RunPhase(
                    name: phaseName, generationStage: generationStage,
                    fraction: Double(completed) / Double(total),
                    detail: "block \(completed)/\(total)")))
        continuation.yield(.telemetry(telemetry))
        enforceBudget(telemetry)
    }

    /// One named preparation step, as a telemetry sample.
    ///
    /// Telemetry and a log line, deliberately **not** a `.phase`: the chat's
    /// phase label belongs to the generation, and what this is for is the
    /// flight recorder. Before it existed the runner's first event of any kind
    /// was `.accepted`, so a process killed while preparing left a trace with a
    /// start line and nothing after it — three times, on the owner's iPhone.
    ///
    /// `folds: false` because the sentry's floor is not set yet. Folding these
    /// readings in would make `absolute` the whole process's footprint before
    /// the floor was taken, and `peak - floor` would then refuse every run at
    /// its first real sample.
    private func notePreparation(_ step: String) {
        let telemetry = sample(phase: step, stage: .preparing, folds: false)
        continuation.yield(
            .log(
                "V4.1 preparing: \(step) · footprint "
                    + ByteSize.format(telemetry.footprintBytes)))
        continuation.yield(.telemetry(telemetry))
    }

    private func enforceBudget(_ telemetry: RunTelemetry) {
        if telemetry.peakFootprintBytes > request.memoryBudgetBytes {
            cancellation.exceedBudget(peakBytes: telemetry.peakFootprintBytes)
        }
    }

    /// - Parameter folds: whether this reading joins the run's high-water mark.
    ///   False only for the preparation samples taken before
    ///   ``DeepSeekV41BudgetSentry/setFloor(_:)``; see ``notePreparation(_:)``.
    private func sample(
        phase: String, stage: RunGenerationStage? = nil, folds: Bool = true
    ) -> RunTelemetry {
        let harness = HarnessTelemetry.sample(
            declaredBudgetBytes: request.memoryBudgetBytes)
        let peak: UInt64
        if folds {
            peak = sentry.observe(
                current: harness.footprintBytes, lifetime: harness.peakFootprintBytes)
            cancellation.executionControl.boundaryReclaim.observeFootprint(
                harness.footprintBytes)
        } else {
            peak = sentry.addedBytes
        }
        let elapsed = MonotonicClock.seconds(since: startedTick)
        let prefillSeconds = passSeconds.first
        let decodeSeconds = passSeconds.dropFirst().reduce(0, +)
        let decodeTokens = max(0, tokenIDs.count - 1)
        let observed = byteAccounting()
        let accounting = observed ?? ByteAccounting(totalBytesRead: 0)
        let completed = completedTokenBytes
        let decodeAccounting = completed.flatMap { completed in
            prefillBytes.flatMap { completed.subtracting($0) }
        }
        return RunTelemetry(
            at: Date(), elapsed: elapsed, phase: phase,
            tokensPerSecond: decodeTokens > 0 && decodeSeconds > 0
                ? Double(decodeTokens) / decodeSeconds : nil,
            generationStage: stage ?? generationStage,
            bytes: accounting,
            bytesPerSecond: observed != nil && elapsed > 0
                ? Double(accounting.totalBytesRead) / elapsed : nil,
            completedTokenBytes: completed,
            prefillBytes: prefillBytes,
            bytesPerToken: decodeTokens > 0
                ? decodeAccounting.map { $0.totalBytesRead / UInt64(decodeTokens) } : nil,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            decodeTokensCompleted: decodeTokens,
            byteAccountingReported: observed != nil,
            declaredBudgetBytes: request.memoryBudgetBytes,
            footprintBytes: harness.footprintBytes,
            peakFootprintBytes: peak,
            // The basis declaration. `peak` is already `absolute - floor`,
            // which is what `enforceBudget` compares, so a reader that does not
            // also know the floor cannot put `footprintBytes` beside it without
            // comparing two different measurements. Publishing the floor is
            // what makes the pair readable; it changes nothing about what is
            // enforced. Zero means the platform gave no entry sample, and
            // `RunTelemetry` stores that as "no declaration".
            entryFootprintBytes: sentry.floor,
            residentBytes: harness.residentBytes,
            availableBytes: harness.availableBytes,
            mlxActiveBytes: UInt64(max(0, MLX.Memory.activeMemory)),
            mlxCacheBytes: UInt64(max(0, MLX.Memory.cacheMemory)),
            mlxPeakBytes: UInt64(max(0, MLX.Memory.peakMemory)),
            thermalState: ThermalStateName(name: harness.thermalState),
            lowPowerMode: harness.lowPowerModeEnabled,
            batteryLevel: nil,
            // The cadence, per sample rather than only in the terminal
            // summary's prose. A phone that is killed never reaches a terminal
            // summary, and the flight recorder's last written line is then the
            // only evidence of whether the budget was being watched at all.
            instrumentation: RunInstrumentationTelemetry(
                budgetSentry: RunBudgetSentryTelemetry(
                    checks: sentry.samples,
                    worstOvershootBytes: sentry.worstOvershoot,
                    allowanceBytes: DeepSeekV41ProductMemoryBudget.currentPolicy
                        .budgetOvershootAllowanceBytes)))
    }

    @discardableResult
    private func tearDown() -> TeardownRecord {
        if let teardownRecord { return teardownRecord }
        var record = TeardownRecord()
        record.workloadWasPrepared = workload != nil
        terminalReadAccounting = workload?.readAccountingSnapshot
        terminalPinnedBytes = workload?.pinnedResidentBytes ?? 0
        terminalPinnedBlocks = workload?.pinnedBlockCount ?? 0
        terminalPinsHead = workload?.pinsOutputHead ?? false
        workload?.shutdown()
        record.workloadReleased = true
        cancellation.payloadFlow.finish()

        if let scope = request.artifact.scope {
            record.scopeWasHeld = true
            scope.release()
            record.scopeReleased = scope.isReleased
        }
        Self.reclaimTransientMemory()
        record.mlxCacheBytesAfterClear = UInt64(max(0, MLX.Memory.cacheMemory))
        teardownRecord = record
        return record
    }

    private func summary(outcome: Outcome, teardown: TeardownRecord) -> RunSummary {
        let telemetry = sample(phase: outcome.rawValue, stage: .terminal)
        let wall = MonotonicClock.seconds(since: startedTick)
        let prefillSeconds = passSeconds.first
        let decodeSeconds = passSeconds.dropFirst().reduce(0, +)
        let decodeTokens = max(0, tokenIDs.count - 1)
        let rate = decodeTokens > 0 && decodeSeconds > 0
            ? Double(decodeTokens) / decodeSeconds : nil
        let digest = lastLogits.isEmpty ? nil : Self.logitsDigest(lastLogits)
        let publication = request.artifact.runtimeAuthority?.evidence.repository
        let tileReads = workload?.expertTileReads ?? 0
        let tileHits = workload?.expertTileHits ?? 0
        let payload = Payload(
            runner: "deepseek-v41-decode",
            handle: handle.rawValue.uuidString,
            revision: BuildInfo.gitRevision,
            outcome: outcome.rawValue,
            environment: environment
                ?? EnvironmentReport.current(path: request.artifact.root.path),
            artifactRoot: request.artifact.root.path,
            publicationRepository: publication?.repoID ?? "",
            publicationRevision: publication?.revision ?? "",
            sourceRepository: sourceRepository,
            sourceRevision: sourceRevision,
            promptTokenIDs: {
                if case .tokenIDs(let ids) = request.prompt { return ids }
                return []
            }(),
            requestedNewTokens: request.maximumNewTokens,
            generatedTokenIDs: tokenIDs,
            logitsSHA256: digest,
            wallSeconds: wall,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            decodeTokensCompleted: decodeTokens,
            passSeconds: passSeconds,
            prefillPhaseMetrics: prefillPhaseMetrics,
            decodePhaseMetrics: decodePhaseMetrics,
            tokensPerSecond: rate,
            byteAccountingReported: telemetry.reportsByteAccounting,
            declaredBudgetBytes: request.memoryBudgetBytes,
            peakFootprintBytes: telemetry.peakFootprintBytes,
            budgetRespected: telemetry.budgetRespected,
            telemetry: telemetry,
            thermalTrail: thermalTrail,
            configuration: knobs.asStrings,
            productMemoryPlan: productMemoryPlan,
            pinPlan: pinPlan,
            entryFootprintBytes: entryFootprintBytes,
            absolutePeakFootprintBytes: sentry.absolute,
            pinnedResidentBytes: terminalPinnedBytes,
            pinnedBlockCount: terminalPinnedBlocks,
            pinnedOutputHead: terminalPinsHead,
            expertTileReads: tileReads,
            expertTileHits: tileHits,
            teardown: teardown)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        let headline: String
        switch outcome {
        case .finished:
            headline = String(
                format: "%d V4.1 token%@ in %.1f s, peak %@",
                tokenIDs.count, tokenIDs.count == 1 ? "" : "s", wall,
                ByteSize.format(telemetry.peakFootprintBytes))
        case .cancelled:
            headline = String(
                format: "V4.1 stopped after %d of %d token%@, %.1f s, peak %@",
                tokenIDs.count, request.maximumNewTokens,
                request.maximumNewTokens == 1 ? "" : "s", wall,
                ByteSize.format(telemetry.peakFootprintBytes))
        }
        var lines = [
            "source \(sourceRepository)@\(sourceRevision)",
            "thermal " + thermalTrail.map(\.state.rawValue).joined(separator: " -> "),
            "teardown: workload released before terminal event",
        ]
        if telemetry.reportsByteAccounting {
            lines.insert(
                "payload read \(ByteSize.format(telemetry.bytes.totalBytesRead)); "
                    + "deterministic \(ByteSize.format(telemetry.bytes.deterministicBytesRead ?? 0)); "
                    + "expert \(ByteSize.format(telemetry.bytes.expertBytesRead ?? 0))",
                at: 1)
        }
        lines.insert(
            "expert tiles: \(tileReads) read, \(tileHits) served resident", at: 1)
        lines.insert(
            "entry footprint " + ByteSize.format(entryFootprintBytes)
                + "; this run added " + ByteSize.format(telemetry.peakFootprintBytes)
                + "; process peak " + ByteSize.format(sentry.absolute),
            at: 1)
        // The cadence, beside the peak. "The runner stops rather than crossing
        // this budget" is a claim about how often it looked, and a record that
        // states a peak without stating the sample count is a record that
        // cannot distinguish a run that stayed inside its ceiling from a run
        // nobody measured.
        lines.insert(
            "budget checked \(sentry.samples) times; worst overshoot "
                + ByteSize.format(sentry.worstOvershoot) + " of an allowed "
                + ByteSize.format(
                    DeepSeekV41ProductMemoryBudget.currentPolicy
                        .budgetOvershootAllowanceBytes),
            at: 2)
        if terminalPinnedBlocks > 0 {
            lines.insert(
                "pinned \(terminalPinnedBlocks) blocks"
                    + (terminalPinsHead ? " + output head" : "") + ", "
                    + ByteSize.format(terminalPinnedBytes) + " resident",
                at: 1)
        }
        if let last = decodePhaseMetrics.last ?? prefillPhaseMetrics {
            lines.insert(last.summaryLine, at: 1)
            if let gather = last.expertGatherLine { lines.insert(gather, at: 1) }
        }
        if let digest { lines.insert("logits sha256 \(digest)", at: 1) }

        return RunSummary(
            handle: handle, model: capabilities.model,
            tokenIDs: tokenIDs, text: nil,
            wallSeconds: wall, tokensPerSecond: rate,
            finalTelemetry: telemetry,
            peakFootprintBytes: telemetry.peakFootprintBytes,
            budgetRespected: telemetry.budgetRespected,
            thermalTrail: thermalTrail,
            logitsDigest: tokenIDs.indices.last.flatMap { index in
                digest.map { RunLogitsDigest(tokenIndex: index, hex: $0) }
            },
            resultJSON: try? encoder.encode(payload),
            resultJSONFilename: "deepseek-v41-decode-run.json",
            headline: headline, lines: lines,
            gitRevision: BuildInfo.gitRevision)
    }

    private func byteAccounting() -> ByteAccounting? {
        guard let snapshot = terminalReadAccounting ?? workload?.readAccountingSnapshot,
            snapshot.isBalanced
        else { return nil }
        return ByteAccounting(
            totalBytesRead: snapshot.totalBytesRead,
            deterministicBytesRead: snapshot.deterministicBytesRead,
            expertBytesRead: snapshot.expertBytesRead)
    }

    private func recordThermal(
        _ state: ThermalStateName, at seconds: Double, phase: String
    ) {
        let resolved = state == .unknown
            ? ThermalStateName(name: ThermalState.currentName) : state
        guard thermalTrail.last?.state != resolved else { return }
        thermalTrail.append(
            ThermalTransition(state: resolved, atSeconds: seconds, phase: phase))
    }

    /// SHA-256 over the float32 bit patterns, little-endian, in vocabulary
    /// order — the same packing V4's digest uses, deliberately, so the two
    /// gates' digests are the same kind of string.
    static func logitsDigest(_ logits: [Float]) -> String {
        SHA256.hexString(SHA256.hash(DeepSeekV4LogitsDump.rawBytes(logits)))
    }

    private static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        return (error as? RunError) == .cancelled
    }

    private static func reclaimTransientMemory() {
        MLX.Memory.clearCache()
        _ = malloc_zone_pressure_relief(nil, 0)
    }

    private struct Payload: Codable {
        let runner: String
        let handle: String
        let revision: String
        let outcome: String
        let environment: EnvironmentReport
        let artifactRoot: String
        let publicationRepository: String
        let publicationRevision: String
        let sourceRepository: String
        let sourceRevision: String
        let promptTokenIDs: [Int]
        let requestedNewTokens: Int
        let generatedTokenIDs: [Int]
        let logitsSHA256: String?
        let wallSeconds: Double
        let prefillSeconds: Double?
        let decodeSeconds: Double
        let decodeTokensCompleted: Int
        let passSeconds: [Double]
        let prefillPhaseMetrics: DeepSeekV4PhaseMetrics?
        let decodePhaseMetrics: [DeepSeekV4PhaseMetrics]
        let tokensPerSecond: Double?
        let byteAccountingReported: Bool
        let declaredBudgetBytes: UInt64
        let peakFootprintBytes: UInt64
        let budgetRespected: Bool
        let telemetry: RunTelemetry
        let thermalTrail: [ThermalTransition]
        let configuration: [String: String]
        let productMemoryPlan: DeepSeekV41ProductMemoryPlan?
        let pinPlan: PinPlan?
        /// The process footprint when this run began, after its own reclaim.
        /// A peak is only interpretable against it.
        let entryFootprintBytes: UInt64
        /// The whole process's high-water mark, floor included. `peakFootprint`
        /// above is this minus the floor, which is what the budget bounds.
        let absolutePeakFootprintBytes: UInt64
        let pinnedResidentBytes: UInt64
        let pinnedBlockCount: Int
        let pinnedOutputHead: Bool
        /// What the pool actually did, beside what the dial planned: a tile
        /// read is a tile that crossed the link and a hit is one that did not.
        let expertTileReads: Int
        let expertTileHits: Int
        let teardown: TeardownRecord
    }
}
