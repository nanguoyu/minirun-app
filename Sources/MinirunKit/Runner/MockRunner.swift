import Foundation
import StorageCore

/// A `RunnerFacade` that reads nothing and reports plausibly.
///
/// It exists so the app can be built, laid out and driven before the
/// scenario-wrapping runners land, and so the protocol's contract is exercised
/// by a test rather than only by a 357-second device run. It is deliberately
/// honest about being a mock: `MockRunner` refuses exactly what a real runner
/// refuses — a budget below the minimum, an unsupported knob, a text prompt
/// with no tokenizer, too many tokens — and its telemetry says so in the phase
/// names and the headline. Nothing it emits should ever be quoted as a
/// measurement.
public struct MockRunner: RunnerFacade {
    public let capabilities: RunnerCapabilities
    /// Token ids handed back, in order. The first is the prefill's.
    public let scriptedTokenIDs: [Int]
    /// Simulated wall time per token. Zero by default, so a test runs at test
    /// speed rather than at pretend-inference speed.
    public let secondsPerToken: Double
    /// Set to have `start` produce a run that ends in this error.
    public let failure: RunError?

    public init(
        capabilities: RunnerCapabilities = MockRunner.defaultCapabilities,
        scriptedTokenIDs: [Int] = [1128, 4802, 27],
        secondsPerToken: Double = 0,
        failure: RunError? = nil
    ) {
        self.capabilities = capabilities
        self.scriptedTokenIDs = scriptedTokenIDs
        self.secondsPerToken = secondsPerToken
        self.failure = failure
    }

    /// A K3-shaped runner: one token, no text prompts, the flagship layout.
    /// Shaped like the real thing so an app bound against it does not have to
    /// change when the real one arrives.
    public static let defaultCapabilities = RunnerCapabilities(
        model: .kimiK3,
        layout: .k3FlagshipLayerStreams,
        supportedKnobs: [
            "expertReadAhead", "expertsPerDispatch", "queueDepth", "granularity",
            "mlxCacheLimitBytes", "logitChunkRows", "deterministicReadAheadLayers",
            "readAheadReserveBytes",
        ],
        minimumBudgetBytes: 5_800_000_000,
        maximumNewTokens: 1,
        acceptsTextPrompts: false,
        requiresMLX: false,
        supportsPinPlan: true,
        tokenEventGranularity: .perPass)

    public func validate(_ request: RunRequest) throws {
        try validateCommon(request)
    }

    public func start(_ request: RunRequest) throws -> RunSession {
        try validate(request)

        let handle = RunHandleID()
        let latch = CancelLatch()
        let capabilities = self.capabilities
        let scripted = Array(scriptedTokenIDs.prefix(max(1, request.maximumNewTokens)))
        let secondsPerToken = self.secondsPerToken
        let failure = self.failure

        let stream = AsyncThrowingStream<RunEvent, Error> { continuation in
            let task = Task {
                let startedAt = Date()
                continuation.yield(
                    .accepted(
                        RunAcceptance(
                            handle: handle, model: capabilities.model,
                            declaredBudgetBytes: request.memoryBudgetBytes,
                            pinPlan: request.pinPlan,
                            effectiveKnobs: request.knobs,
                            artifactRoot: request.artifact.root.path, startedAt: startedAt)))
                continuation.yield(
                    .log("MockRunner: no artifact is read and no number here is a measurement."))

                var tokens: [Int] = []
                var trail: [ThermalTransition] = []

                for (index, id) in scripted.enumerated() {
                    if latch.isCancelled || Task.isCancelled { break }
                    let phase = index == 0 ? "prefill" : "decode \(index)"
                    continuation.yield(
                        .phase(
                            RunPhase(
                                name: phase,
                                fraction: Double(index + 1) / Double(scripted.count),
                                detail: "simulated")))
                    if secondsPerToken > 0 {
                        try? await Task.sleep(nanoseconds: UInt64(secondsPerToken * 1e9))
                    }
                    tokens.append(id)
                    continuation.yield(
                        .token(
                            TokenEvent(
                                index: index, tokenID: id, text: nil,
                                secondsSincePreviousToken: secondsPerToken,
                                isPrefillToken: index == 0,
                                top1Top2Gap: request.knobs.captureRoutingMargins == true
                                    ? 0.5 : nil)))
                    let telemetry = Self.telemetry(
                        phase: phase, elapsed: Date().timeIntervalSince(startedAt),
                        budget: request.memoryBudgetBytes, tokens: tokens.count)
                    trail.append(
                        ThermalTransition(
                            state: telemetry.thermalState, atSeconds: telemetry.elapsed,
                            phase: phase))
                    continuation.yield(.telemetry(telemetry))
                }

                let elapsed = Date().timeIntervalSince(startedAt)
                let telemetry = Self.telemetry(
                    phase: "finished", elapsed: elapsed,
                    budget: request.memoryBudgetBytes, tokens: tokens.count)
                let cancelled = latch.isCancelled || Task.isCancelled
                let summary = RunSummary(
                    handle: handle, model: capabilities.model, tokenIDs: tokens, text: nil,
                    wallSeconds: elapsed,
                    tokensPerSecond: elapsed > 0 ? Double(tokens.count) / elapsed : nil,
                    finalTelemetry: telemetry,
                    peakFootprintBytes: telemetry.peakFootprintBytes,
                    budgetRespected: telemetry.budgetRespected,
                    thermalTrail: trail, resultJSON: nil, resultJSONFilename: nil,
                    headline: cancelled
                        ? "MockRunner cancelled after \(tokens.count) simulated token(s)"
                        : "MockRunner produced \(tokens.count) simulated token(s)",
                    lines: [
                        "This is MockRunner. No artifact was opened and no byte was read.",
                        "Declared budget: \(request.memoryBudgetBytes) bytes.",
                    ],
                    gitRevision: BuildInfo.gitRevision)

                if let failure {
                    continuation.finish(throwing: failure)
                    return
                }
                continuation.yield(cancelled ? .cancelled(summary) : .finished(summary))
                continuation.finish()
            }
            continuation.onTermination = { reason in
                if case .cancelled = reason { latch.cancel() }
                task.cancel()
            }
        }

        return RunSession(handle: handle, events: stream, cancel: { latch.cancel() })
    }

    /// Memory numbers read from the live process, so the shape of the telemetry
    /// is real even though the run is not. Everything MLX-specific is zero
    /// rather than invented: this runner links no MLX, and a plausible-looking
    /// fabricated cache size is exactly the sort of number that gets quoted.
    static func telemetry(
        phase: String, elapsed: TimeInterval, budget: UInt64, tokens: Int
    ) -> RunTelemetry {
        let footprint = ProcessFootprint.current()
        let usage = MemoryUsage.current()
        return RunTelemetry(
            at: Date(), elapsed: elapsed, phase: phase,
            tokensPerSecond: elapsed > 0 ? Double(tokens) / elapsed : nil,
            bytes: ByteAccounting(
                totalBytesRead: 0, deterministicBytesRead: 0, expertBytesRead: 0),
            bytesPerSecond: 0, bytesPerToken: 0,
            declaredBudgetBytes: budget,
            footprintBytes: footprint?.footprintBytes ?? 0,
            peakFootprintBytes: footprint?.peakFootprintBytes ?? 0,
            residentBytes: usage?.residentBytes ?? 0,
            availableBytes: footprint?.availableBytes,
            mlxActiveBytes: 0, mlxCacheBytes: 0, mlxPeakBytes: 0,
            thermalState: ThermalStateName(name: ThermalState.currentName),
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            batteryLevel: nil)
    }
}

/// The one latch both cancellation doors close.
public final class CancelLatch: @unchecked Sendable {
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
}
