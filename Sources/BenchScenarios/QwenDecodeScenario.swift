import Foundation
import MLX
import ModelAdapters
import StorageCore

/// Qwen3-30B-A3B decoding on the phone, experts streamed from wherever the
/// containers are.
///
/// This is M8.5's exit shape moved onto the device. The Mac phase measured
/// 2.09 tok/s at a 2.31 GB peak with 1,019,215,872 bytes per token; the question
/// here is what the same code does when the bytes come off internal NAND, inside
/// a foreground memory budget that can kill the process rather than merely
/// swapping.
///
/// ## What it deliberately shares with the Mac bench
///
/// The prompt token *ids* are the ones `QwenDecodeBenchTests` carries, copied
/// rather than re-tokenised, for the reason that suite gives: this measurement
/// should not also depend on a Swift tokenizer being right. Where a prompt is
/// one of phase A's three margin-screened ones, its expected first token is
/// carried too — so the device run is a correctness check as well as a timing
/// one, and a phone that produces a different token says so on screen instead of
/// producing a plausible number nobody checks.
public struct QwenDecodeScenario: HarnessScenario {

    /// One prompt, as the Mac bench carries it.
    public struct Prompt: Sendable {
        public let name: String
        public let kind: String
        public let text: String
        public let ids: [Int]
        /// Greedy first token, where phase A screened the prompt and the Mac
        /// agreed with the mlx-lm reference on it. Nil for the free prompts,
        /// which were never gated.
        public let expectedFirstToken: Int?

        public init(
            name: String, kind: String, text: String, ids: [Int], expectedFirstToken: Int? = nil
        ) {
            self.name = name
            self.kind = kind
            self.text = text
            self.ids = ids
            self.expectedFirstToken = expectedFirstToken
        }
    }

    /// The three margin-screened prompts, with the first tokens phase A gated
    /// on, plus the shortest chat prompt for a readable sample. Screened first:
    /// if the device disagrees with the Mac, it should disagree on a prompt
    /// where disagreement means something.
    public static let defaultPrompts: [Prompt] = [
        .init(
            name: "en-count", kind: "screened", text: "one two three four",
            ids: [603, 1378, 2326, 3040]),
        .init(
            name: "zh-everest", kind: "screened", text: "世界上最高的山峰是",
            ids: [102_506, 106_667, 57811, 100_035, 20412], expectedFirstToken: 100_088),
        .init(
            name: "mix-return", kind: "screened", text: "def add(a, b): return a",
            ids: [750, 912, 2877, 11, 293, 1648, 470, 264], expectedFirstToken: 488),
        .init(
            name: "free-why-sky", kind: "free", text: "Why is the sky blue?",
            ids: [
                151_644, 872, 198, 10234, 374, 279, 12884, 6303, 30, 151_645, 198, 151_644,
                77091, 198,
            ]),
    ]

    public let id = "qwen-decode"
    public let title = "Qwen3-30B streamed decode"
    public var summary: String {
        "\(tokens) greedy tokens per prompt, \(repetitions)× — experts streamed through a "
            + "\(cacheBudgetExperts)-slot pool."
    }

    private let resolveModel: @Sendable () throws -> URL
    private let resolveContainers: @Sendable () throws -> URL
    private let prompts: [Prompt]
    private let tokens: Int
    private let repetitions: Int
    private let cacheBudgetExperts: Int
    private let expertsPerDispatch: Int
    private let prefetchScope: QwenStreamedExpertBackend.PrefetchScope
    /// The number this run promises to stay under, chosen before it starts.
    ///
    /// Spec §5.3 wants a stated budget rather than an emergent footprint, and on
    /// iOS the statement has teeth: exceed the real limit and the process is
    /// killed rather than slowed. 4.0 GB against the ~5-5.5 GB measured
    /// foreground budget leaves room for the fact that the Mac's 2.31 GB peak
    /// was RSS and iOS counts a different quantity.
    private let declaredBudgetBytes: UInt64
    /// Record each step's top-1/top-2 logit separation. Off by default: it
    /// costs a full 151,936-element readback per step, which is a change to the
    /// thing being timed, so a run with this on is a correctness run and its
    /// tok/s must not be quoted against a run with it off.
    private let captureMargins: Bool
    private let accessBracket: (@Sendable () -> Void)?

    public init(
        resolveModel: @escaping @Sendable () throws -> URL,
        resolveContainers: @escaping @Sendable () throws -> URL,
        prompts: [Prompt] = QwenDecodeScenario.defaultPrompts,
        tokens: Int = 64,
        repetitions: Int = 2,
        cacheBudgetExperts: Int = 24,
        expertsPerDispatch: Int = 8,
        prefetchScope: QwenStreamedExpertBackend.PrefetchScope = .layer,
        declaredBudgetBytes: UInt64 = 4_000_000_000,
        captureMargins: Bool = false,
        accessBracket: (@Sendable () -> Void)? = nil
    ) {
        self.resolveModel = resolveModel
        self.resolveContainers = resolveContainers
        self.prompts = prompts
        self.tokens = tokens
        self.repetitions = repetitions
        self.cacheBudgetExperts = cacheBudgetExperts
        self.expertsPerDispatch = expertsPerDispatch
        self.prefetchScope = prefetchScope
        self.declaredBudgetBytes = declaredBudgetBytes
        self.captureMargins = captureMargins
        self.accessBracket = accessBracket
    }

    public func run(_ context: ScenarioContext) throws -> ScenarioResult {
        let modelDirectory = try resolveModel()
        let containerDirectory = try resolveContainers()
        defer { accessBracket?() }

        context.log("model:      \(modelDirectory.path)")
        context.log("containers: \(containerDirectory.path)")
        context.log(
            "budget: \(ByteSize.format(declaredBudgetBytes)) declared, "
                + "\(cacheBudgetExperts) pool slots, \(expertsPerDispatch)/dispatch, "
                + "prefetch \(prefetchScope.rawValue)")

        // MLX's own cache is a second memory pool, and one this project does not
        // account for unless it is bounded. 256 MiB is what the Mac phase used,
        // so the two runs are comparable.
        //
        // `GPU.set(cacheLimit:)` is deprecated in favour of `Memory.cacheLimit`
        // and is kept anyway: this is the call `QwenDecodeBenchTests` makes, and
        // the device arm exists to be compared against that Mac arm. Changing
        // the API here would be changing one thing between the two runs for a
        // reason unrelated to the measurement.
        MLX.GPU.set(cacheLimit: 256 << 20)
        MLX.GPU.clearCache()

        context.report(ScenarioProgress(phase: "opening", detail: "checkpoint"))
        let checkpoint = try QwenCheckpoint(directory: modelDirectory.path)
        let source = try QwenResidentWeights(checkpoint: checkpoint)
        let config = checkpoint.config
        context.log(
            "config: \(config.numHiddenLayers) layers, \(config.numExperts) experts "
                + "top-\(config.numExpertsPerToken), hidden \(config.hiddenSize)")

        let containers = try QwenExpertContainers(directory: containerDirectory.path)
        context.log("containers: \(containers.entries.count) entries opened")

        // The resident weights are the only thing this run holds unconditionally,
        // so the number is stated before a token is decoded rather than inferred
        // afterwards from the peak.
        let resident = try source.residentBytes()
        context.log("resident non-expert weights: \(ByteSize.format(UInt64(resident)))")

        var runs: [PromptRun] = []
        var warmed = false

        for repetition in 0..<max(1, repetitions) {
            for prompt in prompts {
                try context.checkCancellation()

                let configuration = QwenStreamedExpertBackend.Configuration(
                    cacheBudgetExperts: cacheBudgetExperts,
                    prefetchScope: prefetchScope,
                    expertsPerDispatch: expertsPerDispatch)
                let backend = try QwenStreamedExpertBackend(
                    containers: containers, config: config, configuration: configuration)
                defer { backend.shutdown() }
                let model = QwenMoEModel(source: source, backend: backend)

                if !warmed {
                    // The first pass through the model pays for faulting in the
                    // non-expert weights, compiling MLX kernels and starting 144
                    // reader threads. Left in, that lands on whichever prompt is
                    // measured first and makes it look slow — the same
                    // correction the Mac bench applies, so the numbers compare.
                    context.log("warm-up: 2 tokens, not measured")
                    context.report(ScenarioProgress(phase: "warm-up", detail: "2 tokens"))
                    _ = try model.generate(prompt: prompt.ids, maxTokens: 2)
                    warmed = true
                    MLX.GPU.clearCache()
                }

                context.log(
                    "[rep \(repetition + 1)] \(prompt.name) (\(prompt.kind)): \(prompt.text)")

                var steps: [StepRecord] = []
                var peak = HarnessTelemetry.sample(declaredBudgetBytes: declaredBudgetBytes)
                var firstToken: Int?
                let runStarted = MonotonicClock.now()
                var lastBytes: UInt64 = 0

                var lastSeparation: Separation?
                let generated = try model.generateTimed(
                    prompt: prompt.ids, maxTokens: tokens,
                    onLogits: captureMargins
                        ? { logits in lastSeparation = Self.separation(of: logits.asArray(Float.self)) }
                        : nil
                ) { step in
                    try context.checkCancellation()
                    if step.index == 0 { firstToken = step.tokenId }
                    let telemetry = HarnessTelemetry.sample(
                        declaredBudgetBytes: declaredBudgetBytes)
                    if telemetry.peakFootprintBytes > peak.peakFootprintBytes {
                        peak = telemetry
                    }
                    let bytes = backend.bytesRequested
                    let stepBytes = bytes - lastBytes
                    lastBytes = bytes

                    let memory = MLX.Memory.snapshot()
                    var margin: Double?
                    var top1: Int?
                    var top2: Int?
                    if captureMargins, let separation = lastSeparation {
                        margin = separation.ulps
                        top1 = separation.top1Id
                        top2 = separation.top2Id
                    }

                    steps.append(
                        StepRecord(
                            index: step.index, tokenId: step.tokenId,
                            inputTokens: step.inputTokens, seconds: step.seconds,
                            bytesRequested: stepBytes,
                            footprintBytes: telemetry.footprintBytes,
                            availableBytes: telemetry.availableBytes,
                            thermalState: telemetry.thermalState,
                            kvCacheBytes: model.kvCacheBytes,
                            mlxActiveBytes: memory.activeMemory,
                            mlxCacheBytes: memory.cacheMemory,
                            mlxPeakBytes: memory.peakMemory,
                            poolInUseBytes: backend.statistics.census.inUse
                                * backend.statistics.slotBytes,
                            logitMarginULPs: margin, top1Id: top1, top2Id: top2))

                    // Decode-only rate: step 0 is the prefill and averaging it
                    // in would describe neither phase.
                    let decodeSteps = steps.dropFirst()
                    let decodeSeconds = decodeSteps.reduce(0) { $0 + $1.seconds }
                    let rate = decodeSeconds > 0
                        ? Double(decodeSteps.count) / decodeSeconds : nil

                    context.report(
                        ScenarioProgress(
                            phase: step.index == 0 ? "prefill" : "decode",
                            fraction: Double(step.index + 1) / Double(tokens),
                            detail: "\(prompt.name) \(step.index + 1)/\(tokens)",
                            tokensPerSecond: rate,
                            bytesRead: bytes,
                            bytesPerSecond: step.seconds > 0
                                ? Double(stepBytes) / step.seconds : nil,
                            telemetry: telemetry))

                    if step.index % 8 == 0 || step.index == tokens - 1 {
                        context.log(
                            String(
                                format:
                                    "  step %3d  %6.3f s  %@  footprint %@  thermal %@",
                                step.index, step.seconds,
                                rate.map { String(format: "%5.2f tok/s", $0) } ?? "  prefill",
                                ByteSize.format(telemetry.footprintBytes),
                                telemetry.thermalState))
                    }
                }

                let wall = MonotonicClock.seconds(since: runStarted)
                let pool = backend.statistics
                let readers = backend.readerStatistics
                let shortReads = readers.reduce(0) { $0 + $1.shortReadCount }
                let errors = readers.reduce(0) { $0 + $1.errorCount }

                let decode = steps.dropFirst()
                let decodeSeconds = decode.reduce(0) { $0 + $1.seconds }
                let median = Self.median(decode.map(\.seconds))
                let rate = decodeSeconds > 0 ? Double(decode.count) / decodeSeconds : 0
                let perTokenBytes = decode.isEmpty
                    ? 0 : decode.reduce(UInt64(0)) { $0 + $1.bytesRequested } / UInt64(decode.count)

                var verdict = "not gated"
                if let expected = prompt.expectedFirstToken {
                    verdict = firstToken == expected
                        ? "first token \(expected) as on the Mac"
                        : "FIRST TOKEN \(firstToken ?? -1), Mac had \(expected)"
                }

                context.log(
                    String(
                        format: "  %@: %.3f tok/s median-step %.3f s, %@/token, peak %@ (%.0f%% "
                            + "of budget), %@",
                        prompt.name, rate, median, ByteSize.format(perTokenBytes),
                        ByteSize.format(peak.peakFootprintBytes),
                        peak.budgetFraction * 100, verdict))
                context.log(
                    "  pool: \(pool.peakInUseSlots)/\(pool.slotCount) slots peak "
                        + "(\(ByteSize.format(UInt64(pool.peakInUseBytes))) of "
                        + "\(ByteSize.format(UInt64(pool.budgetBytes)))), "
                        + "\(pool.blockedAcquireCount) blocked, "
                        + "\(shortReads) short reads, \(errors) errors")

                // The growth decomposition, per token, over the decode steps
                // only. Each term is a *measured* delta, not an inference, so
                // the residual at the end is the honest size of what this run
                // cannot name.
                let growth = Growth(steps: steps)
                context.log(
                    String(
                        format: "  growth/token: footprint %+.1f KiB = KV %+.1f + MLX active "
                            + "%+.1f + MLX cache %+.1f + pool %+.1f, unexplained %+.1f KiB",
                        growth.footprintPerToken / 1024, growth.kvPerToken / 1024,
                        growth.mlxActivePerToken / 1024, growth.mlxCachePerToken / 1024,
                        growth.poolPerToken / 1024, growth.unexplainedPerToken / 1024))
                if captureMargins {
                    let margins = steps.compactMap(\.logitMarginULPs).sorted()
                    if !margins.isEmpty {
                        let below = margins.filter { $0 < 4 }.count
                        context.log(
                            String(
                                format: "  logit margins (bf16 ULPs): min %.2f  p50 %.2f  max %.2f"
                                    + "  —  %d of %d below the 4-ULP screen",
                                margins[0], margins[margins.count / 2],
                                margins[margins.count - 1], below, margins.count))
                    }
                }

                if peak.peakFootprintBytes > declaredBudgetBytes {
                    // Not thrown: the run completed, and a budget overshoot that
                    // did not kill the process is a finding rather than a crash.
                    // It is logged loudly because a quiet one would let the
                    // milestone's memory claim rot.
                    context.log(
                        "  BUDGET EXCEEDED: peak \(ByteSize.format(peak.peakFootprintBytes)) "
                            + "over the declared \(ByteSize.format(declaredBudgetBytes))")
                }

                runs.append(
                    PromptRun(
                        prompt: prompt.name, kind: prompt.kind, text: prompt.text,
                        repetition: repetition, tokens: generated.count,
                        firstTokenId: firstToken, expectedFirstTokenId: prompt.expectedFirstToken,
                        tokenIds: generated.map(\.tokenId),
                        tokensPerSecond: rate, medianStepSeconds: median,
                        wallSeconds: wall, bytesPerToken: perTokenBytes,
                        totalBytesRequested: backend.bytesRequested,
                        expertReads: backend.expertReads,
                        acquireWaitSeconds: backend.acquireWaitSeconds,
                        pool: pool, shortReads: shortReads, errors: errors,
                        peak: peak, growth: growth, steps: steps))
            }
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let payload = Payload(
            scenario: id,
            revision: BuildInfo.gitRevision,
            environment: EnvironmentReport.current(path: containerDirectory.path),
            modelDirectory: modelDirectory.path,
            containerDirectory: containerDirectory.path,
            declaredBudgetBytes: declaredBudgetBytes,
            residentNonExpertBytes: resident,
            tokensPerRun: tokens,
            configuration: [
                "cacheBudgetExperts": "\(cacheBudgetExperts)",
                "expertsPerDispatch": "\(expertsPerDispatch)",
                "prefetchScope": prefetchScope.rawValue,
            ],
            runs: runs)

        let rates = runs.map(\.tokensPerSecond).sorted()
        let mismatches = runs.filter {
            $0.expectedFirstTokenId != nil && $0.firstTokenId != $0.expectedFirstTokenId
        }
        let peakOfPeaks = runs.map(\.peak.peakFootprintBytes).max() ?? 0

        var headline = String(
            format: "%.2f-%.2f tok/s, peak %@", rates.first ?? 0, rates.last ?? 0,
            ByteSize.format(peakOfPeaks))
        if !mismatches.isEmpty { headline += ", \(mismatches.count) TOKEN MISMATCH" }

        return ScenarioResult(
            headline: headline,
            lines: runs.map {
                String(
                    format: "%@ rep%d: %.3f tok/s, %@/token, peak %@", $0.prompt,
                    $0.repetition + 1, $0.tokensPerSecond, ByteSize.format($0.bytesPerToken),
                    ByteSize.format($0.peak.peakFootprintBytes))
            },
            json: try? encoder.encode(payload),
            jsonFilename: "\(id).json")
    }

    /// Top-1 versus top-2 separation, in bfloat16 ULPs at top-1.
    ///
    /// A transcription of `bf16_ulp` / `logit_margin` in
    /// `Tools/qwen_reference/trace_reference.py`, deliberately: phase A's
    /// screening threshold of 4 ULPs is stated in that function's units, and a
    /// second definition here that differed by a factor of two would silently
    /// move the threshold. bfloat16 carries 7 explicit mantissa bits plus the
    /// implicit one, so the step at a value with exponent `e` is `2^(e-7)` —
    /// 0.125 at magnitude 28, which is the number phase B reported.
    static func bf16ULP(_ value: Double) -> Double {
        if value == 0 { return Double(Float.leastNormalMagnitude) }
        return pow(2.0, (log2(abs(value))).rounded(.down) - 7)
    }

    struct Separation {
        let top1Id: Int
        let top2Id: Int
        let top1: Double
        let top2: Double
        let ulps: Double
    }

    /// The two largest logits and their separation. One pass, no sort: the
    /// vocabulary is 151,936 wide and this runs inside a decode loop.
    static func separation(of values: [Float]) -> Separation? {
        guard values.count >= 2 else { return nil }
        var best = 0
        var second = 1
        if values[1] > values[0] { best = 1; second = 0 }
        for index in 2..<values.count {
            let value = values[index]
            if value > values[best] {
                second = best
                best = index
            } else if value > values[second] {
                second = index
            }
        }
        let top1 = Double(values[best])
        let top2 = Double(values[second])
        let unit = bf16ULP(top1)
        return Separation(
            top1Id: best, top2Id: second, top1: top1, top2: top2,
            ulps: unit > 0 ? (top1 - top2) / unit : .infinity)
    }

    static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count % 2 == 0
            ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    }

    struct StepRecord: Encodable {
        let index: Int
        let tokenId: Int
        let inputTokens: Int
        let seconds: Double
        let bytesRequested: UInt64
        let footprintBytes: UInt64
        let availableBytes: UInt64?
        let thermalState: String
        // --- phase 2: where the footprint growth actually lives ---
        /// Bytes in the KV caches, read from the arrays. Required growth.
        let kvCacheBytes: Int
        /// MLX's live allocations.
        let mlxActiveBytes: Int
        /// MLX's buffer cache — freed blocks it is holding for reuse, capped by
        /// the cache limit this scenario sets.
        let mlxCacheBytes: Int
        let mlxPeakBytes: Int
        /// The pager's own slots. A stated constant, sampled anyway so the
        /// table can show it did not move.
        let poolInUseBytes: Int
        /// Top-1 minus top-2 of this step's logits, in bf16 ULPs at top-1 —
        /// the phase A instrument, on device. Nil unless margin capture is on.
        let logitMarginULPs: Double?
        let top1Id: Int?
        let top2Id: Int?
    }

    /// Per-token deltas over a run's decode steps, each one measured.
    ///
    /// Slopes are taken from the first decode step to the last rather than
    /// fitted, because the question is "how much did it grow", not "what curve
    /// did it follow" — and a fit would smuggle in an assumption of linearity
    /// that §7 of the record explicitly declines to make.
    struct Growth: Encodable {
        var tokens: Int = 0
        var footprintPerToken: Double = 0
        var kvPerToken: Double = 0
        var mlxActivePerToken: Double = 0
        var mlxCachePerToken: Double = 0
        var poolPerToken: Double = 0
        var unexplainedPerToken: Double = 0
        var kvFirst: Int = 0
        var kvLast: Int = 0
        var mlxCacheFirst: Int = 0
        var mlxCacheLast: Int = 0
        var mlxActiveFirst: Int = 0
        var mlxActiveLast: Int = 0

        init(steps: [StepRecord]) {
            // Skip the prefill: it allocates a whole prompt's KV at once and
            // would be counted as one token's growth.
            let decode = Array(steps.dropFirst())
            guard let first = decode.first, let last = decode.last, decode.count > 1 else {
                return
            }
            let n = Double(decode.count - 1)
            tokens = decode.count
            footprintPerToken = (Double(last.footprintBytes) - Double(first.footprintBytes)) / n
            kvPerToken = Double(last.kvCacheBytes - first.kvCacheBytes) / n
            mlxActivePerToken = Double(last.mlxActiveBytes - first.mlxActiveBytes) / n
            mlxCachePerToken = Double(last.mlxCacheBytes - first.mlxCacheBytes) / n
            poolPerToken = Double(last.poolInUseBytes - first.poolInUseBytes) / n
            unexplainedPerToken =
                footprintPerToken - kvPerToken - mlxActivePerToken - mlxCachePerToken
                - poolPerToken
            kvFirst = first.kvCacheBytes
            kvLast = last.kvCacheBytes
            mlxCacheFirst = first.mlxCacheBytes
            mlxCacheLast = last.mlxCacheBytes
            mlxActiveFirst = first.mlxActiveBytes
            mlxActiveLast = last.mlxActiveBytes
        }
    }

    struct PromptRun: Encodable {
        let prompt: String
        let kind: String
        let text: String
        let repetition: Int
        let tokens: Int
        let firstTokenId: Int?
        let expectedFirstTokenId: Int?
        let tokenIds: [Int]
        let tokensPerSecond: Double
        let medianStepSeconds: Double
        let wallSeconds: Double
        let bytesPerToken: UInt64
        let totalBytesRequested: UInt64
        let expertReads: Int
        let acquireWaitSeconds: Double
        let pool: BufferPoolStatistics
        let shortReads: Int
        let errors: Int
        let peak: HarnessTelemetry
        let growth: Growth
        let steps: [StepRecord]
    }

    struct Payload: Encodable {
        let scenario: String
        let revision: String
        let environment: EnvironmentReport
        let modelDirectory: String
        let containerDirectory: String
        let declaredBudgetBytes: UInt64
        let residentNonExpertBytes: Int
        let tokensPerRun: Int
        let configuration: [String: String]
        let runs: [PromptRun]
    }
}
