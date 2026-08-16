import Darwin
import Foundation
import MinirunKit
import StorageCore

/// Whether a product metric is backed by evidence from this run.
///
/// `notReported` is deliberately not success. A missing sample must stay
/// neutral instead of borrowing the visual language of a passing check.
enum MetricEvidenceStatus: Equatable, Sendable {
    case notReported
    case valid
    case failed(String)
}

/// The single immutable value the instrument panel renders.
///
/// The run publishes one of these; views observe it and nothing else. No view
/// reaches into the runner, and no view samples anything itself — which is what
/// makes the three-clock rule (runner telemetry, fixed-window payload flow,
/// and event-driven layer/token progress) enforceable rather than aspirational.
struct RunSnapshot: Equatable, Sendable {

    enum State: Equatable, Sendable {
        case idle
        case validating
        case running
        case finished
        case cancelled
        /// The run stopped at a layer boundary because the world intervened.
        /// The telemetry stays on screen; it is not cleared.
        case halted(RunFault)
        case refused(RunError)

        var isRunning: Bool { self == .running || self == .validating }
    }

    var state: State = .idle
    var acceptance: RunAcceptance?
    var phase: RunPhase?
    var telemetry: RunTelemetry?
    var tokens: [TokenEvent] = []
    /// Tokens decoded together so UTF-8 sequences split across BPE boundaries
    /// never render replacement characters. Nil means this run supplied no
    /// verified decoder.
    var generatedText: String?
    var log: [String] = []
    var startedAt: Date?
    var elapsed: TimeInterval = 0

    /// One cell per layer.
    var ladder: [LayerCell] = []
    /// 60-second window at 0.5-second runner-owned intervals, oldest first.
    var flow: [FlowSample] = []
    /// Nil until the instrument stream reports each metric. Zero is a valid
    /// measurement for several counters, so it cannot double as "missing".
    var stalls: StallAccounting?
    var cache: CacheStrip?
    var readAhead: ReadAheadAccounting?
    var summary: RunSummary?
    /// Where each completed pass's wall time went, in the order the runner
    /// published them. Empty for a runner that publishes no decomposition, and
    /// for a pass whose own attribution failed closed — neither is a pass that
    /// spent no time anywhere.
    var phaseSummaries: [RunPhaseSummary] = []
    var fault: RunFault?
    /// The measured ceiling the ribbon draws its reference line at.
    var linkCeilingBytesPerSecond: Double?
    /// Set once and never unset: `budgetRespected == false` is a permanent fact
    /// about a run.
    var budgetBreached = false
    var thermalTrail: [ThermalTransition] = []
    var reproducibility: ReproducibilityFooter?

    /// Accounting is tri-state, and the three words are the chip's own promise:
    /// the counts are pending until this run has reported a completed pass,
    /// balanced when the partition it reported is exact, and a mismatch names
    /// the identity that failed.
    ///
    /// A failing identity — the byte split, the staged read-ahead ledger, or an
    /// accepted read-ahead depth this product cannot honour — outranks
    /// everything else, because a broken identity is a fact about the run and
    /// not a stage of it. Nothing else may hold the chip at pending: a run that
    /// publishes no read-ahead evidence at all still reports its bytes, and
    /// making the byte chip wait on a staged-slot census kept V4 — which has no
    /// deterministic read-ahead and never publishes one — reading pending for
    /// an entire run whose split was on screen beside it.
    var accountingStatus: MetricEvidenceStatus {
        let byteStatus = byteAccountingStatus
        if case .failed(let identity) = byteStatus { return .failed(identity) }

        if case .failed(let identity) = readAheadAccountingStatus {
            return .failed(identity)
        }

        guard acceptance != nil else { return .notReported }
        let readAheadSetting = ReadAheadSetting(
            acceptance?.effectiveKnobs.deterministicReadAheadLayers
        )
        guard readAheadSetting.isProductSupported else {
            return .failed("accepted read-ahead depth is outside 0...1")
        }
        return byteStatus
    }

    /// The identity the chip checks: model-layer bytes plus expert bytes are
    /// exactly the total the run reports.
    ///
    /// It is answered from the boundary the run actually reported — the
    /// completed pass — rather than from the cumulative counter alone, which is
    /// mid-flight for the pass in progress and is why `pending` means "no pass
    /// has reported yet". A broken partition is still a mismatch wherever it
    /// appears, so the cumulative counter is checked too; it just cannot make
    /// the counts look settled before a pass has settled them.
    var byteAccountingStatus: MetricEvidenceStatus {
        guard let telemetry, telemetry.reportsByteAccounting else { return .notReported }
        let boundary = telemetry.completedTokenBytes
        guard telemetry.bytes.isBalanced, boundary?.isBalanced != false else {
            return .failed("model layers + experts ≠ total bytes read")
        }
        guard let boundary, boundary.deterministicBytesRead != nil,
            boundary.expertBytesRead != nil
        else { return .notReported }
        return .valid
    }

    var readAheadAccountingStatus: MetricEvidenceStatus {
        guard let readAhead else { return .notReported }
        guard readAhead.isBalanced else {
            return .failed("materialised + discarded + outstanding ≠ staged")
        }
        guard let censusIsClean = readAhead.censusIsClean else { return .notReported }
        guard censusIsClean else {
            return .failed(
                "final slot census has \(readAhead.nonFreeSlots ?? 0) slots that are not free")
        }
        return .valid
    }

    /// The terminal summary is authoritative if a transport omitted an
    /// otherwise useful progressive token event.
    var tokensCompleted: Int { max(tokens.count, summary?.tokenIDs.count ?? 0) }

    /// Generated tokens completed after prompt ingestion. Prefill produces the
    /// first sampled token, but it is not decode throughput and must never be
    /// used as the denominator for decode rate or decode bytes/token.
    var decodeTokensCompleted: Int {
        if let reported = telemetry?.decodeTokensCompleted, reported >= 0 {
            return reported
        }
        return tokens.filter { !$0.isPrefillToken }.count
    }

    var tokensPerSecond: Double? {
        if let reported = telemetry?.tokensPerSecond,
            reported.isFinite, reported >= 0
        {
            return reported
        }
        let decode = tokens.filter { !$0.isPrefillToken }
        guard !decode.isEmpty else { return nil }
        let seconds = decode.reduce(0.0) { partial, token in
            let duration = token.secondsSincePreviousToken
            return duration.isFinite && duration >= 0 ? partial + duration : partial
        }
        guard seconds > 0 else { return nil }
        return Double(decode.count) / seconds
    }

    var bytesPerToken: UInt64? {
        guard telemetry?.reportsByteAccounting != false else { return nil }
        let count = decodeTokensCompleted
        guard count > 0 else { return nil }
        if let reported = telemetry?.bytesPerToken {
            return reported
        }
        guard let completed = telemetry?.completedTokenBytes,
            let prefill = telemetry?.prefillBytes,
            let decode = completed.subtracting(prefill)
        else { return nil }
        return decode.totalBytesRead / UInt64(count)
    }

    var byteSplitPerToken: (deterministic: UInt64, expert: UInt64)? {
        let count = decodeTokensCompleted
        guard count > 0, let telemetry,
            telemetry.reportsByteAccounting,
            let completed = telemetry.completedTokenBytes,
            let prefill = telemetry.prefillBytes,
            let bytes = completed.subtracting(prefill),
            let deterministic = bytes.deterministicBytesRead,
            let expert = bytes.expertBytesRead
        else { return nil }
        let denominator = UInt64(count)
        return (deterministic / denominator, expert / denominator)
    }

    // MARK: Where the time went

    /// The prompt-ingestion pass's decomposition. There is at most one.
    var prefillPhaseSummary: RunPhaseSummary? {
        phaseSummaries.first { $0.passKind == .prefill }
    }

    /// Every decode pass of this turn summed into one, whose `secondsPerPass`
    /// is the mean decode pass. A turn is many passes and one bar per pass
    /// would be a chart of the same shape repeated; the sum keeps the counts
    /// and the seconds honest and states how many passes it covers.
    var decodePhaseAggregate: RunPhaseSummary? {
        RunPhaseSummary.aggregate(phaseSummaries.filter { $0.passKind == .decode })
    }

    /// Both bars the panel draws, in reading order, and nothing when the run
    /// published no decomposition at all.
    var phaseBars: [RunPhaseSummary] {
        [prefillPhaseSummary, decodePhaseAggregate].compactMap { $0 }
    }

    /// The stage this snapshot is in, preferring the live phase while a run is
    /// in flight and the terminal telemetry once it is not.
    var generationStage: RunGenerationStage? {
        if state.isRunning {
            return phase?.generationStage ?? telemetry?.generationStage
        }
        return telemetry?.generationStage ?? phase?.generationStage
    }

    // MARK: The prefill pass
    //
    // Prompt ingestion is minutes long on a flagship artifact, and for all of
    // it the decode readouts have nothing to report: no decode token has
    // completed, so a decode rate and decode bytes/token are both honestly
    // "—". The three values below are what the run HAS published during that
    // time. None of them is a second clock — the throughput is the runner's own
    // published rate, and the progress is the layer ladder the panel already
    // draws.

    /// Model-payload throughput for the run so far, as the runner published it.
    /// Nil when this run does not report byte accounting, so a runner without a
    /// complete observer cannot present a zero as measured traffic.
    var readBytesPerSecond: Double? {
        guard let telemetry, telemetry.reportsByteAccounting,
            let rate = telemetry.bytesPerSecond, rate.isFinite, rate > 0
        else { return nil }
        return rate
    }

    /// Layers completed, and how many the ladder has, for the pass in flight.
    var passLayerProgress: (completed: Int, total: Int)? {
        guard !ladder.isEmpty else { return nil }
        return (LayerLadderAccessibility.completedCount(in: ladder), ladder.count)
    }

    /// An **estimate** of the seconds the prefill pass still needs.
    ///
    /// Derivation, in full: the elapsed seconds the runner reports, divided by
    /// the layers it has finished, times the layers it has not. Two caveats,
    /// which is why the panel labels the number an estimate and never a
    /// measurement: the elapsed clock is the whole run's, so it also contains
    /// the short metadata-only preparation before the first layer; and it
    /// assumes the remaining layers cost what the finished ones did. Nil until
    /// enough layers have finished for that ratio to mean anything, and nil
    /// once the pass is complete.
    var prefillRemainingSecondsEstimate: Double? {
        guard generationStage == .prefill,
            let progress = passLayerProgress,
            progress.completed >= 2, progress.completed < progress.total,
            let elapsed = telemetry?.elapsed, elapsed.isFinite, elapsed > 0
        else { return nil }
        let perLayer = elapsed / Double(progress.completed)
        let remaining = perLayer * Double(progress.total - progress.completed)
        return remaining.isFinite && remaining > 0 ? remaining : nil
    }

    /// Bytes attributable to the pass currently in flight, never folded into
    /// ``bytesPerToken`` until that token reaches its boundary.
    var currentTokenBytes: ByteAccounting? {
        guard state.isRunning, let telemetry, telemetry.reportsByteAccounting else { return nil }
        guard let completed = telemetry.completedTokenBytes else {
            return tokensCompleted == 0 ? telemetry.bytes : nil
        }
        return telemetry.bytes.subtracting(completed)
    }

    mutating func merge(_ sample: InstrumentSample) {
        if let ladder = sample.ladder { self.ladder = ladder }
        if let stalls = sample.stalls { self.stalls = stalls }
        if let cache = sample.cache { self.cache = cache }
        if let readAhead = sample.readAhead { self.readAhead = readAhead }
        if let flow = sample.flow {
            appendFlow(flow)
        }
    }

    /// Applies the runner's monotonic fixed-window stream. Invalid, overflowed,
    /// duplicate, or out-of-order samples do not become a chart point.
    mutating func apply(_ sample: RunPayloadFlowSample) {
        guard sample.isValid,
            let deterministic = sample.deterministicBytesPerSecond,
            let expert = sample.expertBytesPerSecond
        else { return }
        if let previous = flow.last?.sequence, sample.sequence <= previous { return }
        appendFlow(
            FlowSample(
                sequence: sample.sequence, at: sample.at,
                deterministicBytesPerSecond: deterministic,
                expertBytesPerSecond: expert))
    }

    /// Apply the canonical runner sample and derive only quantities whose
    /// numerators and clocks are both present. This is the product path for
    /// K3; the optional App-local instrument stream remains a preview/test
    /// seam for runners that predate these fields.
    mutating func apply(_ sample: RunTelemetry) {
        telemetry = sample

        if let expert = sample.instrumentation?.expert {
            let active = max(0, expert.phaseSeconds - expert.ioWaitSeconds)
            let achieved = expert.ioWaitSeconds > 0
                ? Double(expert.bytesRead) / expert.ioWaitSeconds : 0
            stalls = StallAccounting(
                computeSeconds: active,
                computeWaitedOnIOSeconds: expert.ioWaitSeconds,
                wallSeconds: expert.phaseSeconds,
                achievedBytesPerSecond: achieved)
            cache = CacheStrip(
                policy: expert.cachePolicy,
                hits: expert.cacheHits,
                misses: expert.cacheMisses,
                evictions: expert.cacheEvictions,
                rejectedAdmissions: expert.cacheRejectedAdmissions,
                pinnedExpertCount: expert.pinnedExpertCount)
        }

        if let reported = sample.instrumentation?.readAhead {
            readAhead = ReadAheadAccounting(
                depth: reported.depth,
                stagedBytes: reported.stagedBytes,
                peakStagedBytes: reported.peakStagedBytes,
                materialisedStagedBytes: reported.materialisedStagedBytes,
                discardedStagedBytes: reported.discardedStagedBytes,
                outstandingStagedBytes: reported.outstandingStagedBytes,
                prefetchWaitSeconds: reported.prefetchWaitSeconds,
                stageReadSeconds: reported.stageReadSeconds,
                nonFreeSlots: reported.nonFreeSlots)
        }
    }

    private mutating func appendFlow(_ sample: FlowSample) {
        flow.append(sample)
        if flow.count > 120 { flow.removeFirst(flow.count - 120) }
    }

    /// Canonical phase events are enough to drive honest layer progress even
    /// when a runner has not adopted the App's optional instrumentation
    /// stream. A phase with fraction zero means layer 1 is in flight; each
    /// later fraction is emitted only after the corresponding layer boundary.
    mutating func apply(_ phase: RunPhase) {
        self.phase = phase
        let carriesLayerProgress = phase.generationStage == .prefill
            || phase.generationStage == .decode
            || phase.name == "prefill" || phase.name.hasPrefix("decode ")
        guard !ladder.isEmpty, carriesLayerProgress,
            let fraction = phase.fraction, fraction.isFinite
        else { return }

        let bounded = min(1, max(0, fraction))
        let completed = min(
            ladder.count,
            max(0, Int((bounded * Double(ladder.count)).rounded())))
        for index in ladder.indices {
            if index < completed {
                ladder[index].status = .done
            } else if index == completed, completed < ladder.count {
                ladder[index].status = .computing
            } else {
                ladder[index].status = .pending
            }
        }
    }
}

/// One rung of the layer ladder.
struct LayerCell: Equatable, Sendable, Identifiable {
    enum Status: String, Equatable, Sendable {
        case pending, staged, computing, done
    }
    let index: Int
    var status: Status
    let storedBytes: UInt64
    let widenedBytes: UInt64
    var wallSeconds: Double?
    var wasStaged: Bool

    var id: Int { index }

    var tooltip: String {
        var parts = [
            "layer \(index)",
            "stored \(MRFormat.bytesDecimal(storedBytes))",
            "widened \(MRFormat.bytesDecimal(widenedBytes))",
        ]
        if let seconds = wallSeconds { parts.append(String(format: "%.3f s", seconds)) }
        parts.append(wasStaged ? "staged" : "serial")
        return parts.joined(separator: " · ")
    }
}

/// One runner-owned fixed-window sample of model payload flow.
struct FlowSample: Equatable, Sendable, Identifiable {
    let sequence: UInt64?
    let at: Date
    let deterministicBytesPerSecond: Double
    let expertBytesPerSecond: Double
    var id: String { sequence.map(String.init) ?? "legacy-\(at.timeIntervalSinceReferenceDate)" }
    var totalBytesPerSecond: Double { deterministicBytesPerSecond + expertBytesPerSecond }

    init(
        sequence: UInt64? = nil, at: Date,
        deterministicBytesPerSecond: Double, expertBytesPerSecond: Double
    ) {
        self.sequence = sequence
        self.at = at
        self.deterministicBytesPerSecond = deterministicBytesPerSecond
        self.expertBytesPerSecond = expertBytesPerSecond
    }
}

/// The `Stalls` pair. Only both together attribute a stall.
struct StallAccounting: Equatable, Sendable {
    var computeSeconds: Double
    var computeWaitedOnIOSeconds: Double
    var wallSeconds: Double
    var achievedBytesPerSecond: Double

    var ioWaitFraction: Double? {
        guard wallSeconds.isFinite, wallSeconds > 0,
            computeSeconds.isFinite, computeSeconds >= 0,
            computeWaitedOnIOSeconds.isFinite, computeWaitedOnIOSeconds >= 0
        else { return nil }
        let accounted = computeSeconds + computeWaitedOnIOSeconds
        let tolerance = max(1e-6, wallSeconds * 1e-6)
        guard accounted.isFinite, abs(accounted - wallSeconds) <= tolerance else { return nil }
        return min(1, computeWaitedOnIOSeconds / wallSeconds)
    }
    var overlapFraction: Double? { ioWaitFraction.map { max(0, 1 - $0) } }

    /// The verdict in plain words, chosen by comparing what was achieved to the
    /// measured ceiling — not by a threshold on the wait fraction alone.
    func verdict(linkCeilingBytesPerSecond: Double?) -> String {
        guard overlapFraction != nil else { return "Not reported" }
        guard let ceiling = linkCeilingBytesPerSecond,
            ceiling.isFinite, ceiling > 0
        else {
            return "Link ceiling not measured"
        }
        let achieved = achievedBytesPerSecond
        guard achieved.isFinite, achieved >= 0 else { return "Throughput not reported" }
        if achieved >= ceiling * 0.85 {
            return "storage-bound: \(MRFormat.throughput(achieved)) against a "
                + "\(MRFormat.throughput(ceiling)) port"
        }
        return "pipeline-bound: \(MRFormat.throughput(achieved)) of requests against a "
            + "\(MRFormat.throughput(ceiling)) port"
    }
}

struct CacheStrip: Equatable, Sendable {
    var policy: String? = nil
    var hits: Int?
    var misses: Int?
    var evictions: Int?
    var rejectedAdmissions: Int?
    var pinnedExpertCount: Int?

    /// The production K3 pool currently reports `no-cache`: it intentionally
    /// retains no routed-expert region between requests. Calling that a 0% hit
    /// rate suggests a malfunctioning cache when no cache was attempted.
    var isDisabled: Bool {
        guard let policy else { return false }
        switch policy.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "none", "no-cache": return true
        default: return false
        }
    }

    /// A zero-request sample has counters, but it does not have a hit rate.
    /// Keeping that distinction avoids presenting 0/0 as either 0% or 100%.
    var hitRate: Double? {
        guard !isDisabled else { return nil }
        guard let hits, let misses, hits >= 0, misses >= 0 else { return nil }
        let total = hits.addingReportingOverflow(misses)
        guard !total.overflow, total.partialValue > 0 else { return nil }
        return Double(hits) / Double(total.partialValue)
    }

    var line: String {
        if isDisabled {
            guard let misses, misses >= 0 else { return "Off" }
            if misses == 0 { return "Off · no storage loads yet" }
            return "Off · \(count(misses)) loaded from storage"
        }
        if let hits, let misses, hits == 0, misses == 0 {
            return "No expert requests yet"
        }
        let rate = hitRate.map { MRFormat.percent($0) } ?? "—"
        let lookups = "\(count(hits)) reused · \(count(misses)) loaded"
        return "\(rate) hit rate · \(lookups) · \(count(evictions)) evicted · "
            + "\(count(rejectedAdmissions)) not cached · \(count(pinnedExpertCount)) held"
    }

    var accessibilityLine: String {
        if isDisabled {
            guard let misses, misses >= 0 else { return "off" }
            return misses == 0
                ? "off, no storage loads yet"
                : "off, \(misses) loaded from storage"
        }
        if let hits, let misses, hits == 0, misses == 0 {
            return "no expert requests yet"
        }
        let rate = hitRate.map { MRFormat.percent($0) } ?? "not reported"
        return "reuse rate \(rate), \(spoken(hits)) reused, "
            + "\(spoken(misses)) loaded, \(spoken(evictions)) evicted, "
            + "\(spoken(rejectedAdmissions)) not cached, "
            + "\(spoken(pinnedExpertCount)) held in memory"
    }

    private func count(_ value: Int?) -> String {
        guard let value, value >= 0 else { return "—" }
        return MRFormat.grouped(value)
    }

    private func spoken(_ value: Int?) -> String {
        guard let value, value >= 0 else { return "not reported" }
        return String(value)
    }
}

/// The read-ahead half of the overlap meter, and the accounting identity the
/// balance dot watches.
struct ReadAheadAccounting: Equatable, Sendable {
    var depth: Int
    var stagedBytes: UInt64
    var peakStagedBytes: UInt64
    var materialisedStagedBytes: UInt64
    var discardedStagedBytes: UInt64
    var outstandingStagedBytes: UInt64
    var prefetchWaitSeconds: Double
    var stageReadSeconds: Double
    /// Nil for a live sample; only a terminal slot census can prove this
    /// identity. Zero is therefore never used as a stand-in for “unknown”.
    var nonFreeSlots: Int?

    /// `1 - wait / stageRead`. 1.0 means the reads were entirely hidden.
    var overlapEfficiency: Double? {
        guard stageReadSeconds.isFinite, stageReadSeconds > 0,
            prefetchWaitSeconds.isFinite, prefetchWaitSeconds >= 0
        else { return nil }
        return max(0, 1 - prefetchWaitSeconds / stageReadSeconds)
    }

    var isBalanced: Bool {
        let completed = materialisedStagedBytes.addingReportingOverflow(discardedStagedBytes)
        guard !completed.overflow else { return false }
        let accounted = completed.partialValue.addingReportingOverflow(outstandingStagedBytes)
        return !accounted.overflow && accounted.partialValue == stagedBytes
    }

    var censusIsClean: Bool? { nonFreeSlots.map { $0 == 0 } }
}

/// The reproducibility line under the token stream. A first-class user-visible
/// output, not a debug affordance.
struct ReproducibilityFooter: Equatable, Sendable {
    var samplingMode: String
    var tokenID: Int
    var top1Top2Gap: Double?
    var logitsDigest: String
    /// Set when this run's digest equals the previous run's, which is the whole
    /// point of `Regenerate`.
    var matchesPreviousRun: Bool?

    var line: String {
        var parts = [samplingMode, "token \(tokenID)"]
        if let gap = top1Top2Gap { parts.append(String(format: "top1−top2 gap %.4f", gap)) }
        parts.append("logits sha256 \(MRFormat.shortDigest(logitsDigest))")
        return parts.joined(separator: " · ")
    }
}

/// A named failure with the numbers its card needs. Every error card in this
/// product is plain sentence · exact numbers · the named error string.
struct RunFault: Equatable, Sendable {
    enum Kind: Equatable, Sendable {
        /// The exact fields StorageCore reported. A removable-volume failure is
        /// a presentation of this case, not a second, reconstructed error.
        case storagePOSIX(
            operation: String, path: String, code: Int32, atLayer: Int)
        case shortRead(operation: String, path: String, offset: UInt64, expected: Int, actual: Int)
        case storageInvalidArgument(String)
        case storageEnvironmentUnavailable(String)
        case budgetExceeded(peakBytes: UInt64, declaredBytes: UInt64)
        case thermal(ThermalStateName)
        case runnerRefused(RunError)
    }

    let kind: Kind
    let at: Date
    /// Bytes read before the fault, when the runner has a complete observer.
    /// Nil must remain unknown; zero is a valid measured value.
    let bytesReadSoFar: UInt64?
    let tokensCompleted: Int

    var headline: String {
        switch kind {
        case .storagePOSIX(let operation, _, _, let layer):
            if let volume = unavailableExternalVolume {
                return "\(volume) became unavailable at layer \(layer)."
            }
            return "\(operation) failed at layer \(layer)."
        case .shortRead(_, _, _, let expected, let actual):
            return "Short read — \(actual) of \(expected) bytes."
        case .storageInvalidArgument:
            return "The storage request was invalid."
        case .storageEnvironmentUnavailable:
            return "Storage information became unavailable."
        case .budgetExceeded(let peak, let declared):
            let over = peak > declared ? peak - declared : 0
            return "Peak footprint exceeded the stated budget by \(MRFormat.bytesDecimal(over))."
        case .thermal(let state):
            return "Thermal state \(state.rawValue)."
        case .runnerRefused(let error):
            return "The runner refused this run: \(error.description)"
        }
    }

    var body: String {
        switch kind {
        case .storagePOSIX:
            if unavailableExternalVolume != nil {
                let byteCopy = bytesReadSoFar.map {
                    "\(MRFormat.bytesDecimal($0)) read"
                } ?? "Byte count not reported"
                return "\(byteCopy), \(tokensCompleted) token\(tokensCompleted == 1 ? "" : "s") "
                    + "completed. Reconnect the drive and try the turn again — the budget "
                    + "you stated is kept."
            }
            return "The storage operation failed. Partial telemetry is kept; the exact POSIX "
                + "operation, path and errno are shown below."
        case .shortRead:
            return "The requested bytes were not all available. Verify the full local model "
                + "copy before trying again; this run cannot safely resume where it stopped."
        case .storageInvalidArgument:
            return "Nothing is inferred or retried with different parameters. The storage "
                + "layer's exact refusal is shown below."
        case .storageEnvironmentUnavailable:
            return "The storage layer could not establish a required environment fact. "
                + "Partial telemetry is kept and the exact error is shown below."
        case .budgetExceeded:
            return "The run completed; the budget did not hold. The number is written to "
                + "history — reporting a breach honestly beats hiding it."
        case .thermal:
            return "Minirun is not throttling the run — the device is. Reducing work here "
                + "would make the stated budget a lie about the run."
        case .runnerRefused:
            return "Nothing was read. The configuration was refused before the run started."
        }
    }

    /// The named error string, shown verbatim under a disclosure.
    var namedError: String {
        switch kind {
        case .storagePOSIX(let operation, let path, let code, _):
            return StorageCoreError.posix(
                operation: operation, path: path, code: code
            ).description
        case .shortRead(let operation, let path, let offset, let expected, let actual):
            return StorageCoreError.shortTransfer(
                operation: operation, path: path, offset: offset, expected: expected,
                actual: actual
            ).description
        case .storageInvalidArgument(let detail):
            return StorageCoreError.invalidArgument(detail).description
        case .storageEnvironmentUnavailable(let detail):
            return StorageCoreError.environmentUnavailable(detail).description
        case .budgetExceeded(let peak, let declared):
            return RunError.budgetExceeded(peakBytes: peak, declaredBytes: declared).description
        case .thermal(let state):
            return "thermalState(\(state.rawValue))"
        case .runnerRefused(let error):
            return error.description
        }
    }

    /// Only a disconnect is resumable. A short read is not.
    var isResumable: Bool {
        unavailableExternalVolume != nil
    }

    var isFatal: Bool {
        switch kind {
        case .budgetExceeded, .thermal: return false
        default: return true
        }
    }

    /// Persisted recovery semantics derived from the structured error, before
    /// the exact error is reduced to its user-selectable diagnostic string.
    var messageFailure: MessageFailure {
        switch kind {
        case .storagePOSIX:
            return unavailableExternalVolume == nil ? .unknown : .retryAfterStorageReturns
        case .shortRead:
            return .verifyModelFiles
        case .storageInvalidArgument, .storageEnvironmentUnavailable:
            return .reviewModel
        case .budgetExceeded, .thermal:
            return .informational
        case .runnerRefused(let error):
            return error.messageFailure
        }
    }

    /// A POSIX error on a removable-volume path is only called a disconnect for
    /// errno values that actually describe vanished or unusable media. The
    /// path and errno remain the storage layer's fields; this helper decides
    /// presentation and never parses `description`.
    private var unavailableExternalVolume: String? {
        guard case .storagePOSIX(_, let path, let code, _) = kind,
            [EIO, ENXIO, ENODEV, ESTALE].contains(code)
        else { return nil }
        let components = URL(fileURLWithPath: path).standardizedFileURL.pathComponents
        guard let volumes = components.firstIndex(of: "Volumes"),
            components.indices.contains(volumes + 1)
        else { return nil }
        return components[volumes + 1]
    }
}

extension RunError {
    /// Product recovery for a named runner refusal. This switch intentionally
    /// follows enum cases, never `description` text.
    var messageFailure: MessageFailure {
        switch self {
        case .budgetBelowMinimum, .knobNotSupported, .knobOutOfRange,
            .promptNotSupported, .tooManyTokens, .pinPlanBudgetMismatch,
            .pinPlanAccountingMismatch, .pinPlanInvalid, .pinPlanUnitUnsupported:
            return .adjustChatSettings
        case .artifactNotReady, .artifactLayoutMismatch, .runnerUnavailable,
            .underlying:
            return .reviewModel
        case .scopeRefused:
            return .retryAfterStorageReturns
        case .budgetExceeded:
            return .informational
        case .cancelled:
            return .stopped
        }
    }
}
