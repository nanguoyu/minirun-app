import Combine
import Foundation
import StorageCore

/// Runs one scenario at a time and keeps what the screen shows.
///
/// The app's view is meant to be thin enough that nothing in it needs testing;
/// that only works if the state machine lives here, where the macOS suite can
/// drive it. Everything published is a value type, and the scenario itself runs
/// off the main thread — the storage layer blocks by design, and a harness that
/// ran it on the main queue would report its own beachball as latency.
@MainActor
public final class HarnessController: ObservableObject {

    public enum Status: Equatable {
        case idle
        case running(String)
        case finished(String)
        case failed(String)

        public var isRunning: Bool {
            if case .running = self { return true }
            return false
        }
    }

    @Published public private(set) var status: Status = .idle
    @Published public private(set) var progress: ScenarioProgress?
    @Published public private(set) var telemetry: HarnessTelemetry
    /// Newest last. Bounded, because a 64-token run at eight lines a token is
    /// more text than a `List` should be asked to diff.
    @Published public private(set) var log: [LogLine] = []
    @Published public private(set) var results: [String: ScenarioResult] = [:]
    /// Where result JSON went, so the operator can find it in the Files app
    /// without a debugger attached.
    @Published public private(set) var writtenFiles: [String] = []

    /// Called once when a run ends, either way. The auto-run path uses it to
    /// terminate the process so that `devicectl ... --console` returns instead
    /// of hanging on an app that has nothing left to do.
    public var onRunFinished: ((Status) -> Void)?

    public struct LogLine: Identifiable, Equatable {
        public let id = UUID()
        public let timestamp: Date
        public let text: String
    }

    public let workingDirectory: URL
    private let maximumLogLines: Int
    private var cancelFlag = ManagedAtomicFlag()
    private var telemetryTimer: Timer?
    /// The budget the currently running scenario declared, so the gauge is
    /// against the promise rather than against physical memory.
    private var declaredBudgetBytes: UInt64 = 0

    public init(workingDirectory: URL, maximumLogLines: Int = 2000) {
        self.workingDirectory = workingDirectory
        self.maximumLogLines = maximumLogLines
        self.telemetry = HarnessTelemetry.sample(declaredBudgetBytes: 0)
    }

    // MARK: - Telemetry

    /// Samples memory and thermal state on a timer, so the numbers move even
    /// while a scenario is between progress reports — which, at one token every
    /// few seconds, is most of the time.
    public func startTelemetry(interval: TimeInterval = 1.0) {
        telemetryTimer?.invalidate()
        let timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) {
            [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.telemetry = HarnessTelemetry.sample(
                    declaredBudgetBytes: self.declaredBudgetBytes)
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        telemetryTimer = timer
    }

    public func stopTelemetry() {
        telemetryTimer?.invalidate()
        telemetryTimer = nil
    }

    // MARK: - Running

    public func run(_ scenario: any HarnessScenario, declaredBudgetBytes: UInt64 = 0) {
        guard !status.isRunning else { return }
        cancelFlag.clear()
        self.declaredBudgetBytes = declaredBudgetBytes
        status = .running(scenario.title)
        progress = nil
        append("=== \(scenario.title) — \(Self.stamp()) ===")
        append("revision \(BuildInfo.gitRevision)")

        let directory = workingDirectory
        let flag = cancelFlag
        let context = ScenarioContext(
            workingDirectory: directory,
            log: { [weak self] line in
                Task { @MainActor [weak self] in self?.append(line) }
            },
            report: { [weak self] update in
                Task { @MainActor [weak self] in self?.progress = update }
            },
            isCancelled: { flag.isSet })

        Task.detached(priority: .userInitiated) { [weak self] in
            let outcome: Result<ScenarioResult, Error>
            do {
                outcome = .success(try scenario.run(context))
            } catch {
                outcome = .failure(error)
            }
            await MainActor.run { [weak self] in
                guard let self else { return }
                switch outcome {
                case .success(let result):
                    self.finish(scenario: scenario, result: result)
                case .failure(let error):
                    // Every failure class in spec §19.1 arrives here, and the
                    // harness's job is to name it on screen rather than to
                    // reduce it to "failed". A surfaced disconnect is a passing
                    // result for the failure-handling design; it is only the
                    // silence that would be a defect.
                    let described = (error as? ScenarioError).map(String.init(describing:))
                        ?? String(describing: error)
                    self.append("FAILED: \(described)")
                    self.status = .failed(described)
                }
                self.onRunFinished?(self.status)
            }
        }
    }

    public func cancel() {
        guard status.isRunning else { return }
        cancelFlag.set()
        append("stop requested — the scenario will end at its next checkpoint")
    }

    private func finish(scenario: any HarnessScenario, result: ScenarioResult) {
        results[scenario.id] = result
        append(result.headline)
        for line in result.lines { append("  \(line)") }

        if let json = result.json, let name = result.jsonFilename {
            // Written into Documents, which `UIFileSharingEnabled` makes visible
            // in the Files app, because the alternative is reading numbers off a
            // screenshot. Prefixed with the timestamp so a second run does not
            // silently overwrite the first one's evidence.
            let stamped = "\(Self.fileStamp())-\(name)"
            let url = workingDirectory.appendingPathComponent(stamped)
            do {
                try json.write(to: url, options: .atomic)
                writtenFiles.append(stamped)
                append("wrote \(stamped) (\(ByteSize.format(UInt64(json.count))))")
            } catch {
                append("could not write \(stamped): \(error)")
            }
        }
        status = .finished(result.headline)
    }

    // MARK: - Log

    public func append(_ text: String) {
        log.append(LogLine(timestamp: Date(), text: text))
        if log.count > maximumLogLines {
            log.removeFirst(log.count - maximumLogLines)
        }
        // Also to stdout, which is not redundant: `devicectl device process
        // launch --console` streams it back to the Mac, and that is the only
        // way a run driven without a human at the screen reports anything.
        print(text)
        fflush(stdout)
    }

    public func clearLog() { log.removeAll() }

    /// The whole log as one string, for the share sheet. A run whose log can
    /// only be photographed is a run that will be transcribed wrongly.
    public var logText: String {
        let formatter = ISO8601DateFormatter()
        return log.map { "\(formatter.string(from: $0.timestamp)) \($0.text)" }
            .joined(separator: "\n")
    }

    private static func stamp() -> String {
        ISO8601DateFormatter().string(from: Date())
    }

    private static func fileStamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        formatter.timeZone = TimeZone(identifier: "UTC")
        return formatter.string(from: Date())
    }
}

/// A cancellation flag both the main actor and a detached task may touch.
///
/// `NSLock` around a `Bool` rather than an `Atomic`: the package has no
/// atomics dependency, this is polled a few times per second, and the cost is
/// irrelevant next to a whole-expert read.
public final class ManagedAtomicFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    public init() {}

    public var isSet: Bool {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    public func set() {
        lock.lock()
        value = true
        lock.unlock()
    }

    public func clear() {
        lock.lock()
        value = false
        lock.unlock()
    }
}
