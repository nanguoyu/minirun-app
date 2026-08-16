import Foundation
import MinirunKit
import Observation
import StorageCore

/// Owns one run and publishes exactly one immutable `RunSnapshot`.
///
/// Three clocks, and none of them is the display's. Telemetry reports runtime
/// state, payload flow arrives in runner-owned fixed windows, and layer/token
/// progress advances on events. Nothing here polls, and no view reaches past
/// this object into the runner.
@Observable
@MainActor
final class RunController {

    private(set) var snapshot = RunSnapshot()
    /// The budget that was stated at start. Kept separately from the dial's
    /// live value because the dial is editable and this is not: a run screen
    /// shows the budget the run was accepted under, forever.
    private(set) var statedBudgetBytes: UInt64?
    private(set) var statedKnobs = RunKnobs()

    /// The previous run's digest, so `Regenerate` can say whether the new one
    /// matched rather than implying it.
    private(set) var previousDigest: String?

    private var session: RunSession?
    private var eventTask: Task<Void, Never>?
    private var instrumentTask: Task<Void, Never>?
    private var payloadFlowTask: Task<Void, Never>?
    private var decodeTokens: (([Int]) -> String)?

    /// A Stop request has crossed the product boundary, but the blocking model
    /// thread has not yet reached its next safe layer boundary. Keeping this
    /// separate from `isRunning` prevents the UI from claiming the turn has
    /// stopped before the runner has released its readers and working set.
    private(set) var isStopping = false

    /// Called once, with the final snapshot, when a turn reaches a terminal
    /// state — finished, cancelled, halted or refused. A conversation records
    /// every one of those, including the ones that produced no text.
    var onTurnEnded: ((RunSnapshot) -> Void)?

    var isRunning: Bool { snapshot.state.isRunning }

    // MARK: Starting

    /// Refuses before a byte is read, and shows the refusal instead of running.
    func start(
        runner: any RunnerFacade, request: RunRequest, linkCeiling: Double?,
        layerCount: Int, storedBytesPerLayer: UInt64, widenedBytesPerLayer: UInt64,
        decodeTokens: (([Int]) -> String)? = nil,
        previousDigest: String? = nil
    ) {
        teardown()
        isStopping = false
        statedBudgetBytes = nil
        statedKnobs = RunKnobs()
        self.previousDigest = previousDigest
        self.decodeTokens = decodeTokens

        var fresh = RunSnapshot()
        fresh.state = .validating
        fresh.linkCeilingBytesPerSecond = linkCeiling
        fresh.ladder = (0..<max(0, layerCount)).map {
            LayerCell(
                index: $0, status: .pending, storedBytes: storedBytesPerLayer,
                widenedBytes: widenedBytesPerLayer, wallSeconds: nil, wasStaged: false)
        }
        snapshot = fresh

        do {
            try runner.validate(request)
        } catch let error as RunError {
            request.artifact.scope?.release()
            snapshot.state = .refused(error)
            snapshot.log.append(error.description)
            releaseCompletedSessionResources()
            return
        } catch let error as StorageCoreError {
            request.artifact.scope?.release()
            snapshot.state = .refused(.underlying(error.description))
            snapshot.log.append(error.description)
            releaseCompletedSessionResources()
            return
        } catch {
            request.artifact.scope?.release()
            snapshot.state = .refused(.underlying("\(error)"))
            releaseCompletedSessionResources()
            return
        }

        statedBudgetBytes = request.memoryBudgetBytes
        statedKnobs = request.knobs

        let session: RunSession
        do {
            session = try runner.start(request)
        } catch let error as RunError {
            request.artifact.scope?.release()
            snapshot.state = .refused(error)
            releaseCompletedSessionResources()
            return
        } catch let error as StorageCoreError {
            request.artifact.scope?.release()
            snapshot.state = .refused(.underlying(error.description))
            snapshot.log.append(error.description)
            releaseCompletedSessionResources()
            return
        } catch {
            request.artifact.scope?.release()
            snapshot.state = .refused(.underlying("\(error)"))
            releaseCompletedSessionResources()
            return
        }

        self.session = session
        snapshot.state = .running
        snapshot.startedAt = Date()

        if let instrumented = runner as? any InstrumentedRunner {
            let stream = instrumented.instrumentation(for: session.handle)
            instrumentTask = Task { [weak self] in
                for await sample in stream {
                    self?.apply(sample)
                }
            }
        }

        if let payloadFlow = session.payloadFlow {
            let handle = session.handle
            payloadFlowTask = Task { [weak self] in
                for await sample in payloadFlow {
                    guard !Task.isCancelled else { break }
                    self?.apply(sample, for: handle)
                }
            }
        }

        eventTask = Task { [weak self] in
            do {
                for try await event in session.events {
                    self?.apply(event)
                }
            } catch let error as StorageCoreError {
                self?.halt(with: error)
            } catch let error as RunError {
                self?.halt(with: error)
            } catch {
                self?.halt(with: .underlying("\(error)"))
            }
        }
    }

    // MARK: Stopping

    /// Always available during a run. Cancels at the next layer boundary and
    /// reports partial telemetry — the run is not discarded.
    ///
    /// The event task is deliberately NOT cancelled here: the runner answers a
    /// cancel with a `.cancelled(RunSummary)` at its next boundary, and tearing
    /// the subscription down first would throw away exactly the partial
    /// telemetry this method promises to report.
    func stop(clearSnapshot: Bool = true) {
        guard isRunning, !isStopping else { return }
        isStopping = true
        session?.cancel()
        if clearSnapshot {
            teardown()
            snapshot = RunSnapshot()
        }
    }

    /// Drops the subscription without waiting for a last word. Used when a new
    /// run replaces this one, where a stale event applied to the fresh snapshot
    /// would be a bug rather than a record.
    private func teardown() {
        session?.cancel()
        eventTask?.cancel()
        instrumentTask?.cancel()
        payloadFlowTask?.cancel()
        eventTask = nil
        instrumentTask = nil
        payloadFlowTask = nil
        session = nil
        decodeTokens = nil
        isStopping = false
    }

    /// The runner has already emitted its terminal event (or closed with a
    /// named error), so no more events can belong to this session. Release the
    /// event stream, instrumentation subscription and tokenizer closure immediately;
    /// retaining the decoder until a future chat would keep its vocabulary in
    /// memory even though the turn is visibly over. The fixed-window payload
    /// stream is allowed to drain its final partial window: both production
    /// runners close it after readers quiesce and before emitting terminal.
    private func releaseCompletedSessionResources() {
        instrumentTask?.cancel()
        instrumentTask = nil
        eventTask = nil
        session = nil
        decodeTokens = nil
        isStopping = false
    }

    func clear() {
        teardown()
        snapshot = RunSnapshot()
        statedBudgetBytes = nil
        statedKnobs = RunKnobs()
        previousDigest = nil
    }

    // MARK: Event application

    private func apply(_ event: RunEvent) {
        switch event {
        case .accepted(let acceptance):
            snapshot.acceptance = acceptance
            snapshot.log.append(
                "accepted · budget \(MRFormat.bytesDecimal(acceptance.declaredBudgetBytes))")

        case .phase(let phase):
            snapshot.apply(phase)

        case .token(let token):
            snapshot.tokens.append(token)
            if let decodeTokens {
                snapshot.generatedText = decodeTokens(snapshot.tokens.map(\.tokenID))
            }

        case .telemetry(let telemetry):
            snapshot.apply(telemetry)
            snapshot.elapsed = telemetry.elapsed
            // A breach is a permanent fact about a run: once latched, it stays.
            if !telemetry.budgetRespected { snapshot.budgetBreached = true }

        case .phaseSummary(let summary):
            // Appended, never merged: two passes of the same kind are two
            // measurements, and the panel sums them itself so the aggregate
            // can always say how many passes it covers.
            snapshot.phaseSummaries.append(summary)

        case .log(let line):
            snapshot.log.append(line)

        case .finished(let summary):
            snapshot.summary = summary
            snapshot.apply(summary.finalTelemetry)
            snapshot.thermalTrail = summary.thermalTrail
            snapshot.elapsed = summary.wallSeconds
            snapshot.generatedText = summary.text
                ?? decodeTokens?(summary.tokenIDs)
            if !summary.budgetRespected {
                snapshot.budgetBreached = true
                snapshot.fault = RunFault(
                    kind: .budgetExceeded(
                        peakBytes: summary.peakFootprintBytes,
                        declaredBytes: summary.finalTelemetry.declaredBudgetBytes),
                    at: Date(),
                    bytesReadSoFar: summary.finalTelemetry.reportsByteAccounting
                        ? summary.finalTelemetry.bytes.totalBytesRead : nil,
                    tokensCompleted: summary.tokenIDs.count)
            }
            applyReproducibility(from: summary, becomesPrevious: true)
            snapshot.state = .finished
            releaseCompletedSessionResources()
            onTurnEnded?(snapshot)

        case .cancelled(let summary):
            snapshot.summary = summary
            snapshot.apply(summary.finalTelemetry)
            snapshot.elapsed = summary.wallSeconds
            snapshot.generatedText = summary.text
                ?? decodeTokens?(summary.tokenIDs)
            applyReproducibility(from: summary, becomesPrevious: false)
            snapshot.state = .cancelled
            releaseCompletedSessionResources()
            onTurnEnded?(snapshot)
        }
    }

    /// A digest belongs to the logits bytes a runner actually captured. Token
    /// events do not carry those bytes, so the footer appears only when the
    /// terminal summary supplies the digest; a preview/mock run therefore
    /// cannot accidentally inherit a keeper digest from a fixture.
    private func applyReproducibility(from summary: RunSummary, becomesPrevious: Bool) {
        guard let digest = summary.logitsDigest, digest.isValid,
            summary.tokenIDs.indices.contains(digest.tokenIndex)
        else {
            snapshot.reproducibility = nil
            return
        }
        let observed = snapshot.tokens.first { $0.index == digest.tokenIndex }
        snapshot.reproducibility = ReproducibilityFooter(
            samplingMode: "greedy", tokenID: summary.tokenIDs[digest.tokenIndex],
            top1Top2Gap: observed?.top1Top2Gap,
            logitsDigest: digest.hex,
            matchesPreviousRun: previousDigest.map { $0 == digest.hex })
        if becomesPrevious { previousDigest = digest.hex }
    }

    private func apply(_ sample: InstrumentSample) {
        snapshot.merge(sample)
    }

    private func apply(_ sample: RunPayloadFlowSample, for handle: RunHandleID) {
        guard session?.handle == handle || snapshot.acceptance?.handle == handle else {
            return
        }
        snapshot.apply(sample)
    }

    /// A run that stopped because the world intervened. The telemetry stays on
    /// screen; it is not cleared.
    private func halt(with error: StorageCoreError) {
        let layer = snapshot.ladder.lastIndex { $0.status == .done } ?? 0
        let kind: RunFault.Kind
        switch error {
        case .shortTransfer(let operation, let path, let offset, let expected, let actual):
            kind = .shortRead(
                operation: operation, path: path, offset: offset,
                expected: expected, actual: actual)
        case .posix(let operation, let path, let code):
            kind = .storagePOSIX(
                operation: operation, path: path, code: code, atLayer: layer)
        case .invalidArgument(let detail):
            kind = .storageInvalidArgument(detail)
        case .environmentUnavailable(let detail):
            kind = .storageEnvironmentUnavailable(detail)
        }
        halt(kind: kind, namedError: error.description)
    }

    private func halt(with error: RunError) {
        halt(kind: .runnerRefused(error), namedError: error.description)
    }

    private func halt(kind: RunFault.Kind, namedError: String) {
        let fault = RunFault(
            kind: kind, at: Date(),
            bytesReadSoFar: snapshot.telemetry.flatMap {
                $0.reportsByteAccounting ? $0.bytes.totalBytesRead : nil
            },
            tokensCompleted: snapshot.tokens.count)
        snapshot.fault = fault
        snapshot.log.append(namedError)
        snapshot.state = .halted(fault)
        releaseCompletedSessionResources()
        onTurnEnded?(snapshot)
    }
}
