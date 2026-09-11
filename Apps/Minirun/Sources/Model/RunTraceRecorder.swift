import Darwin
import Foundation
import MinirunKit
import StorageCore

/// One run's flight recorder: append-only NDJSON, flushed to the disk after
/// every line, kept where the app container can be pulled off a phone.
///
/// ## Why it exists
///
/// A Jetsam kill produces no terminal event, no summary, and no log the app can
/// write afterwards — the process is simply gone. Twice now the only evidence
/// of a V4.1 kill on the owner's iPhone has been the operating system's own
/// report (`vm-pageshortage`, a resident figure) and a conversation document
/// whose settings said what the chat *intended*. Neither says what the run was
/// holding at the moment it died, which block it was in, or whether the budget
/// was being enforced at all.
///
/// So every run writes its own trace as it goes:
///
/// ```text
/// Library/Application Support/<bundle>/RunTraces/<conversationID>-<startISO>.ndjson
/// ```
///
/// one JSON object per line — `start`, then a `sample` per telemetry event,
/// then `end`. Each line is `write(2)`n and `fsync(2)`d before the call returns,
/// so what the file holds is what had happened by the last line in it. A run
/// that is killed leaves a truncated trace, and a truncated trace is the point.
///
/// ## What a sample carries, and why it carries two footprints
///
/// `footprintBytes` is `phys_footprint` — the number iOS enforces its per-process
/// limit against — and `residentBytes` is `resident_size`, which is roughly the
/// `rpages` figure a Jetsam report prints. They disagree in both directions:
/// footprint counts compressed, IOKit and purgeable pages that resident does
/// not, and resident counts clean file-backed pages that footprint does not. A
/// trace that carried one of them could not answer the first question a kill
/// raises, which is *which one grew*. `budgetedFootprintBytes` is the third
/// number and the only one the budget is compared against: on V4.1 that is
/// `footprint - entry`, because the run does not begin in an empty process.
///
/// ## What it is not
///
/// It is not telemetry for a screen, and nothing reads it back into the app.
/// `Tools/v41_flash/read_run_trace.py` prints it as a table. The newest
/// ``Retention/keep`` traces are kept and older ones are deleted when a run
/// starts, so an app container stays a thing a person can copy off a phone.
final class RunTraceRecorder: @unchecked Sendable {

    enum Retention {
        /// Traces kept on disk. Twenty is about two evenings of chats and a few
        /// megabytes; the number is here so the reason is one edit away from
        /// the value.
        static let keep = 20
    }

    // MARK: The three line shapes

    /// The platform memory policy a run was admitted under. V4.1 states one;
    /// every other model leaves it absent rather than inventing terms.
    struct PolicyLine: Codable, Equatable {
        let name: String
        let floorBytes: UInt64
        let transientExecutionBytes: UInt64
        let pinnedTransientExecutionBytes: UInt64
        let budgetOvershootAllowanceBytes: UInt64
        let maximumPromptTokens: Int
        let maximumNewTokens: Int
        let expertPoolSlots: Int
        let headWindowRows: Int
        let boundsLiveOperands: Bool
        let isExperimental: Bool
    }

    struct PinPlanLine: Codable, Equatable {
        let pinnedBytes: UInt64
        let pinnedLayerCount: Int
        let pinsOutputHead: Bool
        let workingFloorBytes: UInt64
    }

    struct Start: Codable, Equatable {
        var kind = "start"
        let at: Date
        let conversation: String
        let model: String
        let scale: String
        let declaredBudgetBytes: UInt64
        let maximumNewTokens: Int
        let promptTokenCount: Int?
        let policy: PolicyLine?
        let pinPlan: PinPlanLine?
        /// What the process was already holding when the run was handed over.
        let footprintBytes: UInt64
        let residentBytes: UInt64
        /// `os_proc_available_memory()`, where the platform has it.
        let availableBytes: UInt64?
        let gitRevision: String?
    }

    struct Sample: Codable, Equatable {
        var kind = "sample"
        /// `"telemetry"` for a line the runner's own event produced, `"timer"`
        /// for one the ``Heartbeat`` took on its own.
        ///
        /// They are not interchangeable and a reader must not average them: a
        /// telemetry line's run-basis numbers were measured by the runner at
        /// that instant, where a timer line carries the last basis the runner
        /// published beside footprints the recorder measured itself. Only the
        /// two absolute footprints mean the same thing on both.
        var source = "telemetry"
        let elapsed: TimeInterval
        let stage: String
        let phase: String
        /// The block the run had reached, from the phase detail the runner
        /// publishes. Nil between blocks and during preparation.
        let block: Int?
        let blockCount: Int?
        let footprintBytes: UInt64
        let residentBytes: UInt64
        /// The current value **on the budget's own basis**. Compare this one
        /// against `declaredBudgetBytes`, never `footprintBytes`.
        let budgetedFootprintBytes: UInt64
        let peakFootprintBytes: UInt64
        let entryFootprintBytes: UInt64?
        let declaredBudgetBytes: UInt64
        let availableBytes: UInt64?
        let mlxActiveBytes: UInt64
        let mlxCacheBytes: UInt64
        let sentryChecks: UInt64?
        let sentryWorstOvershootBytes: UInt64?
        let tokens: Int
    }

    struct End: Codable, Equatable {
        var kind = "end"
        let at: Date
        let elapsed: TimeInterval
        let outcome: String
        let tokens: Int
        let peakFootprintBytes: UInt64?
        let budgetRespected: Bool?
        let namedError: String?
    }

    // MARK: Location

    /// The real location, beside `Conversations/` and named the same way.
    ///
    /// `bundleIdentifier` is nil in some unit-test host configurations, so the
    /// fallback is spelled out rather than force-unwrapped into a crash nobody
    /// can reproduce — the identifier a pull off the device names:
    /// `devicectl device copy from --domain-type appDataContainer
    /// --domain-identifier wang.wangdongdong.minirun`.
    static func applicationSupportDirectory(
        fileManager: FileManager = .default
    ) throws -> URL {
        let base = try fileManager.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil,
            create: true)
        return base
            .appendingPathComponent(
                Bundle.main.bundleIdentifier ?? "wang.wangdongdong.minirun",
                isDirectory: true)
            .appendingPathComponent("RunTraces", isDirectory: true)
    }

    /// `<conversationID>-<startISO>.ndjson`, with the timestamp in ISO 8601's
    /// basic form so the name carries no colon and sorts chronologically inside
    /// one conversation.
    static func filename(conversationID: UUID, startedAt: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.formatOptions = [
            .withYear, .withMonth, .withDay, .withTime, .withTimeZone,
        ]
        return "\(conversationID.uuidString)-\(formatter.string(from: startedAt)).ndjson"
    }

    // MARK: Lifetime

    private let lock = NSLock()
    private let handle: FileHandle
    private let encoder: JSONEncoder
    private var isClosed = false
    let url: URL

    /// Opens a trace, after pruning the directory to ``Retention/keep``.
    ///
    /// Failing to write a trace must never fail a run: every error here becomes
    /// `nil` and the chat proceeds with no recorder. That is the one place this
    /// type is allowed to be quiet, because the alternative is a diagnostic
    /// that can take down the thing it was added to diagnose.
    init?(
        directory: URL, conversationID: UUID, startedAt: Date,
        fileManager: FileManager = .default
    ) {
        let url = directory.appendingPathComponent(
            Self.filename(conversationID: conversationID, startedAt: startedAt),
            isDirectory: false)
        do {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
            Self.prune(
                directory: directory, keeping: Retention.keep - 1,
                fileManager: fileManager)
            guard fileManager.createFile(atPath: url.path, contents: nil) else { return nil }
            handle = try FileHandle(forWritingTo: url)
        } catch {
            return nil
        }
        self.url = url
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        self.encoder = encoder
    }

    deinit {
        heartbeat?.cancel()
        try? handle.close()
    }

    /// The newest `keep` traces stay; the rest are deleted.
    ///
    /// Ordered by modification date and not by name: a name begins with a
    /// conversation id, so sorting by name would keep the most recent chats of
    /// whichever conversation sorts last rather than the most recent chats.
    static func prune(
        directory: URL, keeping keep: Int, fileManager: FileManager = .default
    ) {
        guard let names = try? fileManager.contentsOfDirectory(atPath: directory.path) else {
            return
        }
        var traces: [(url: URL, modified: Date)] = []
        for name in names where name.hasSuffix(".ndjson") {
            let url = directory.appendingPathComponent(name, isDirectory: false)
            let values = try? url.resourceValues(forKeys: [.contentModificationDateKey])
            traces.append((url, values?.contentModificationDate ?? .distantPast))
        }
        traces.sort { left, right in
            if left.modified == right.modified {
                return left.url.lastPathComponent > right.url.lastPathComponent
            }
            return left.modified > right.modified
        }
        guard traces.count > max(0, keep) else { return }
        for trace in traces.dropFirst(max(0, keep)) {
            try? fileManager.removeItem(at: trace.url)
        }
    }

    // MARK: Writing

    func record(_ line: Start) { append(line) }
    func record(_ line: Sample) { append(line) }

    /// The last line, after which the handle is closed. A second call writes
    /// nothing: a run ends once, and a trace with two ends would be a trace a
    /// reader has to guess about.
    func record(_ line: End) {
        endHeartbeat()
        append(line)
        lock.lock()
        defer { lock.unlock() }
        guard !isClosed else { return }
        isClosed = true
        try? handle.close()
    }

    private func append<Line: Encodable>(_ line: Line) {
        guard let data = try? encoder.encode(line) else { return }
        lock.lock()
        defer { lock.unlock() }
        guard !isClosed else { return }
        do {
            try handle.write(contentsOf: data + Data("\n".utf8))
            // Not `synchronize()`'s convenience: the promise this file makes is
            // that a line which returned is a line the kernel has committed, and
            // a Jetsam kill is not a graceful shutdown that would flush it.
            try handle.synchronize()
        } catch {
            isClosed = true
            try? handle.close()
        }
    }

    // MARK: The heartbeat

    /// The timer that samples memory whether or not the runner has said
    /// anything.
    ///
    /// ## Why a timer and not events
    ///
    /// The first version of this recorder wrote a `sample` line per telemetry
    /// event, and a V4.1 runner published its first telemetry after the first
    /// block of the first pass. Everything before that — opening the artifact,
    /// reconciling forty-one manifests, building the model — was silent. On
    /// 2026-09-11 the owner's iPhone 16 Pro was killed three times inside that
    /// silence, and the traces it left were a start line and nothing else: no
    /// sample, no end. The evidence of a ~5 GB death was a start line saying
    /// 0.111 GB.
    ///
    /// So the recorder no longer waits to be told. From the moment a run is
    /// handed over — before `validate`, before `prepare` — a timer writes
    /// `phys_footprint` and `resident_size` every ``Heartbeat/interval``,
    /// tagged with whatever stage the controller last learned. A kill then
    /// falls at most one interval after a line that says how big the process
    /// had grown and what it was doing.
    ///
    /// It runs on its own serial queue rather than a run-loop timer on purpose:
    /// the main actor can be blocked, and a heartbeat that stops when the app
    /// is busy stops exactly when it is needed.
    enum Heartbeat {
        /// 200 ms. Fast enough that the last line before a kill is close to the
        /// kill — a V4.1 preparation grew about 268 MB per block, so the gap
        /// between two lines is under one block — and slow enough that a
        /// sixty-second chat writes three hundred lines rather than a log file.
        static let interval: TimeInterval = 0.2
    }

    /// What the runner last published about the run's own basis, carried onto
    /// timer lines so a reader does not have to join them to a telemetry line.
    private struct Basis {
        var stage = RunGenerationStage.preparing.rawValue
        var phase = "submitted"
        var block: (block: Int, count: Int)?
        var budgetedFootprintBytes: UInt64 = 0
        var peakFootprintBytes: UInt64 = 0
        var entryFootprintBytes: UInt64?
        var declaredBudgetBytes: UInt64 = 0
        var sentryChecks: UInt64?
        var sentryWorstOvershootBytes: UInt64?
        var tokens = 0
    }

    /// The current memory reading, as the heartbeat takes it.
    typealias MemoryReading = (footprint: UInt64, resident: UInt64, available: UInt64?)

    /// `phys_footprint` and `resident_size` for this process. The default
    /// source; tests supply their own.
    static func platformMemory() -> MemoryReading? {
        guard let footprint = ProcessFootprint.current() else { return nil }
        return (
            footprint: footprint.footprintBytes,
            resident: MemoryUsage.current()?.residentBytes ?? 0,
            available: footprint.availableBytes
        )
    }

    private let basisLock = NSLock()
    private var basis = Basis()
    private var heartbeat: DispatchSourceTimer?
    private var heartbeatStartedAt = Date()
    private var heartbeatMemory: () -> MemoryReading? = RunTraceRecorder.platformMemory
    private var heartbeatNow: () -> Date = Date.init

    /// Begin the heartbeat. Called when the run is submitted, not when it
    /// starts producing: the point is to cover the part that produces nothing.
    ///
    /// A second call is ignored — a run has one heartbeat.
    func beginHeartbeat(
        declaredBudgetBytes: UInt64,
        startedAt: Date = Date(),
        interval: TimeInterval = Heartbeat.interval,
        memory: @escaping () -> MemoryReading? = RunTraceRecorder.platformMemory,
        now: @escaping () -> Date = Date.init
    ) {
        basisLock.lock()
        guard heartbeat == nil else {
            basisLock.unlock()
            return
        }
        basis.declaredBudgetBytes = declaredBudgetBytes
        heartbeatStartedAt = startedAt
        heartbeatMemory = memory
        heartbeatNow = now
        let timer = DispatchSource.makeTimerSource(queue: Self.heartbeatQueue)
        heartbeat = timer
        basisLock.unlock()

        timer.schedule(
            deadline: .now() + interval, repeating: interval, leeway: .milliseconds(20))
        timer.setEventHandler { [weak self] in self?.recordHeartbeatSample() }
        timer.resume()
    }

    /// Carry forward what the runner last said, so the next timer line can
    /// repeat it. Cheap enough to call from every event.
    func noteRunnerState(_ telemetry: RunTelemetry, block: (block: Int, count: Int)?, tokens: Int) {
        basisLock.lock()
        basis.stage = telemetry.generationStage?.rawValue ?? basis.stage
        basis.phase = telemetry.phase
        basis.block = block
        basis.budgetedFootprintBytes = telemetry.budgetedFootprintBytes
        basis.peakFootprintBytes = telemetry.peakFootprintBytes
        basis.entryFootprintBytes = telemetry.entryFootprintBytes
        basis.declaredBudgetBytes = telemetry.declaredBudgetBytes
        basis.sentryChecks = telemetry.instrumentation?.budgetSentry?.checks
        basis.sentryWorstOvershootBytes =
            telemetry.instrumentation?.budgetSentry?.worstOvershootBytes
        basis.tokens = tokens
        basisLock.unlock()
    }

    /// The stage the controller knows, from a phase event, which arrives before
    /// the telemetry that confirms it.
    func noteStage(_ stage: RunGenerationStage?, phase: String, block: (block: Int, count: Int)?) {
        basisLock.lock()
        if let stage { basis.stage = stage.rawValue }
        basis.phase = phase
        basis.block = block
        basisLock.unlock()
    }

    /// Write one timer line now, on the calling thread. The heartbeat's own
    /// body, exposed so a test does not have to wait on a clock to assert what
    /// a line says.
    @discardableResult
    func recordHeartbeatSample() -> Sample? {
        // Everything mutable is copied out under the lock and the two closures
        // are called outside it: a `task_info` call is not something to hold a
        // lock across, and a test's clock is not something to call one under.
        basisLock.lock()
        let basis = self.basis
        let memory = heartbeatMemory
        let now = heartbeatNow
        let startedAt = heartbeatStartedAt
        basisLock.unlock()
        guard let reading = memory() else { return nil }
        let elapsed = now().timeIntervalSince(startedAt)
        let line = Sample(
            source: "timer",
            elapsed: max(0, elapsed),
            stage: basis.stage,
            phase: basis.phase,
            block: basis.block?.block,
            blockCount: basis.block?.count,
            footprintBytes: reading.footprint,
            residentBytes: reading.resident,
            budgetedFootprintBytes: basis.budgetedFootprintBytes,
            peakFootprintBytes: basis.peakFootprintBytes,
            entryFootprintBytes: basis.entryFootprintBytes,
            declaredBudgetBytes: basis.declaredBudgetBytes,
            availableBytes: reading.available,
            // The App cannot see MLX's allocator from here, and a zero that
            // looks like a measurement would be worse than a zero a reader
            // knows to skip: `source` says which lines carry these.
            mlxActiveBytes: 0, mlxCacheBytes: 0,
            sentryChecks: basis.sentryChecks,
            sentryWorstOvershootBytes: basis.sentryWorstOvershootBytes,
            tokens: basis.tokens)
        append(line)
        return line
    }

    /// Stop the heartbeat. Idempotent, and called by ``record(_:)-(End)`` so a
    /// closed trace cannot grow another line.
    func endHeartbeat() {
        basisLock.lock()
        let timer = heartbeat
        heartbeat = nil
        basisLock.unlock()
        timer?.cancel()
    }

    /// One serial queue for every recorder in the process. A run has one
    /// recorder and a chat has one run, so this is a queue with one client;
    /// it is `static` so the queue does not outlive its source by accident.
    private static let heartbeatQueue = DispatchQueue(
        label: "wang.wangdongdong.minirun.run-trace-heartbeat", qos: .utility)

    // MARK: Reading a phase detail

    /// `"block 12/40"` — the detail string the streamed runners publish — as the
    /// pair a trace line carries.
    ///
    /// Parsed rather than recomputed from `RunPhase.fraction`, because a
    /// fraction is a rounded quotient and the block index is the thing a reader
    /// of the trace lines up jumps against.
    static func blockProgress(in detail: String) -> (block: Int, count: Int)? {
        let parts = detail.split(separator: " ")
        guard parts.count == 2, parts[0] == "block" else { return nil }
        let pair = parts[1].split(separator: "/")
        guard pair.count == 2, let block = Int(pair[0]), let count = Int(pair[1]) else {
            return nil
        }
        return (block, count)
    }
}
