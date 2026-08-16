import Foundation
import StorageCore

/// A live reading of what the process is costing and what the device is doing
/// about it.
///
/// Every field here is on the harness screen because a device run can fail in a
/// way a Mac run cannot: the process is killed, and the number that killed it is
/// not the number a Mac benchmark reports. So the footprint is quoted against a
/// *stated* budget rather than against physical memory, and the thermal state
/// travels with it because a run that slowed down is a different result from a
/// run that was throttled (spec §17.4 requires the second to be recorded).
public struct HarnessTelemetry: Codable, Equatable, Sendable {

    /// `phys_footprint` -- what iOS enforces its limit against.
    public var footprintBytes: UInt64
    public var peakFootprintBytes: UInt64
    /// `os_proc_available_memory()`, where the platform has it.
    public var availableBytes: UInt64?
    /// Resident set size, kept beside the footprint rather than instead of it,
    /// because the Mac phase's numbers are RSS and a comparison needs both.
    public var residentBytes: UInt64
    public var peakResidentBytes: UInt64
    /// The budget this run declared for itself before it started. Not a
    /// measurement -- a promise, which the peak either keeps or breaks.
    public var declaredBudgetBytes: UInt64
    public var thermalState: String
    public var lowPowerModeEnabled: Bool

    public var budgetFraction: Double {
        declaredBudgetBytes == 0 ? 0 : Double(peakFootprintBytes) / Double(declaredBudgetBytes)
    }

    public static func sample(declaredBudgetBytes: UInt64) -> HarnessTelemetry {
        let footprint = ProcessFootprint.current()
        let usage = MemoryUsage.current()
        return HarnessTelemetry(
            footprintBytes: footprint?.footprintBytes ?? 0,
            peakFootprintBytes: footprint?.peakFootprintBytes ?? 0,
            availableBytes: footprint?.availableBytes,
            residentBytes: usage?.residentBytes ?? 0,
            peakResidentBytes: usage?.peakResidentBytes ?? 0,
            declaredBudgetBytes: declaredBudgetBytes,
            thermalState: ThermalState.currentName,
            lowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled)
    }
}

/// What a running scenario tells the screen.
///
/// Deliberately a value type with no reference to the view: a scenario runs on a
/// background thread and must not know that a `View` exists, and the harness has
/// to stay a thing the macOS test suite can drive.
public struct ScenarioProgress: Sendable {
    /// Which part of the run this is -- "prefill", "decode", "reading". Shown
    /// verbatim, so scenarios should keep it short.
    public var phase: String
    /// Nil when the scenario cannot know its own length (a duration-bounded
    /// read loop, for instance).
    public var fraction: Double?
    public var detail: String
    /// Decode rate, once there is one. Nil during prefill.
    public var tokensPerSecond: Double?
    public var bytesRead: UInt64
    /// Bytes per second over the run so far.
    public var bytesPerSecond: Double?
    public var telemetry: HarnessTelemetry?

    public init(
        phase: String, fraction: Double? = nil, detail: String = "",
        tokensPerSecond: Double? = nil, bytesRead: UInt64 = 0, bytesPerSecond: Double? = nil,
        telemetry: HarnessTelemetry? = nil
    ) {
        self.phase = phase
        self.fraction = fraction
        self.detail = detail
        self.tokensPerSecond = tokensPerSecond
        self.bytesRead = bytesRead
        self.bytesPerSecond = bytesPerSecond
        self.telemetry = telemetry
    }
}

/// What a scenario is handed when it runs.
///
/// Closures rather than a delegate protocol, so a test can drive a scenario with
/// three lines and no stub class, and so nothing in this file needs to be
/// `@MainActor`.
public struct ScenarioContext: Sendable {
    /// Where the harness keeps its own files: the app's Documents directory on
    /// the phone, a temporary directory in a test.
    public let workingDirectory: URL
    /// Appended to the on-screen log and to the run's log file.
    public let log: @Sendable (String) -> Void
    public let report: @Sendable (ScenarioProgress) -> Void
    /// Polled by long loops. A scenario that ignores this is a scenario the
    /// Stop button cannot stop.
    public let isCancelled: @Sendable () -> Bool

    public init(
        workingDirectory: URL,
        log: @escaping @Sendable (String) -> Void,
        report: @escaping @Sendable (ScenarioProgress) -> Void = { _ in },
        isCancelled: @escaping @Sendable () -> Bool = { false }
    ) {
        self.workingDirectory = workingDirectory
        self.log = log
        self.report = report
        self.isCancelled = isCancelled
    }

    /// Throws if the operator pressed Stop. Cancellation is a named failure
    /// class in spec §19.1, so it travels as an error rather than as a silently
    /// short result.
    public func checkCancellation() throws {
        if isCancelled() { throw ScenarioError.cancelled }
    }
}

public enum ScenarioError: Error, CustomStringConvertible {
    case cancelled
    /// The scenario needs something the operator has not supplied yet -- a
    /// picked folder, a model directory. Named separately from a failure
    /// because it is not one.
    case notReady(String)
    case failed(String)

    public var description: String {
        switch self {
        case .cancelled: return "cancelled"
        case .notReady(let what): return "not ready: \(what)"
        case .failed(let why): return why
        }
    }
}

/// What a finished scenario leaves behind.
public struct ScenarioResult: Sendable {
    /// One line, for the screen. The number the operator came for.
    public var headline: String
    /// Additional lines kept on screen after the run.
    public var lines: [String]
    /// The result JSON, written into the working directory under
    /// ``jsonFilename``. Spec §17.4: a documented result carries its own
    /// metadata, so scenarios emit the file rather than the record's author
    /// re-typing the numbers.
    public var json: Data?
    public var jsonFilename: String?

    public init(
        headline: String, lines: [String] = [], json: Data? = nil, jsonFilename: String? = nil
    ) {
        self.headline = headline
        self.lines = lines
        self.json = json
        self.jsonFilename = jsonFilename
    }
}

/// One Run button.
public protocol HarnessScenario: Sendable {
    /// Stable across runs; used in filenames.
    var id: String { get }
    var title: String { get }
    /// One sentence under the title, saying what pressing Run will cost.
    var summary: String { get }
    /// Runs on a background thread. Blocking is expected and correct -- the
    /// storage layer is synchronous by design.
    func run(_ context: ScenarioContext) throws -> ScenarioResult
}
