import Foundation
import StorageCore

/// What this device is, measured on the device rather than assumed from the Mac.
///
/// The cheapest scenario, and the first one to run: it needs no artifact, no
/// picked folder and no model, so it is what tells the operator the app
/// installed correctly and the entitlements took effect. It is also where the
/// port's platform deltas get recorded — page size above all, because the v0.4
/// alignment decision was made on the Mac's 16 KiB pages and is only free if the
/// phone agrees.
public struct EnvironmentScenario: HarnessScenario {

    public let id = "environment"
    public let title = "Environment"
    public let summary = "Device, OS, page size, memory budget, thermal state. Reads nothing."

    /// Whether the run should assert that the entitlements are live. The
    /// harness sets this; a macOS test does not, because the question is
    /// meaningless there.
    public let expectsIncreasedMemoryLimit: Bool

    public init(expectsIncreasedMemoryLimit: Bool = false) {
        self.expectsIncreasedMemoryLimit = expectsIncreasedMemoryLimit
    }

    public func run(_ context: ScenarioContext) throws -> ScenarioResult {
        let report = EnvironmentReport.current(path: context.workingDirectory.path)
        let footprint = ProcessFootprint.current()
        var lines: [String] = []

        lines.append(contentsOf: report.humanSummary().split(
            separator: "\n", omittingEmptySubsequences: false).map(String.init))

        let pageSize = report.device.pageSizeBytes
        lines.append("page size: \(pageSize) B")
        // The v0.4 decision in one assertion. If this ever prints 4096 the
        // container alignment is no longer free and the record must say so.
        if pageSize == 16384 {
            context.log("page size is 16 KiB here, as on the Mac — container alignment is free")
        } else {
            context.log(
                "page size is \(pageSize) B, NOT the 16 KiB the container alignment assumes")
        }

        if let footprint {
            lines.append("footprint: \(ByteSize.format(footprint.footprintBytes))")
            lines.append("peak footprint: \(ByteSize.format(footprint.peakFootprintBytes))")
            if let available = footprint.availableBytes {
                lines.append("os_proc_available_memory: \(ByteSize.format(available))")
                // The number the milestone's memory claim is made against. On a
                // phone without the entitlement this lands near 3 GB; with it,
                // materially higher. Reporting it is how the record can state
                // the entitlement's effect instead of asserting it.
                context.log(
                    "available to this process: \(ByteSize.format(available)) "
                        + "(+ \(ByteSize.format(footprint.footprintBytes)) already used "
                        + "= \(ByteSize.format(available + footprint.footprintBytes)) budget)")
            } else {
                lines.append("os_proc_available_memory: unavailable on this platform")
                if expectsIncreasedMemoryLimit {
                    context.log(
                        "no os_proc_available_memory here — the budget cannot be read, only "
                            + "inferred from where a run dies")
                }
            }
        }

        let storage = report.storage
        if let free = storage.volumeAvailableBytes {
            lines.append("documents free: \(ByteSize.format(UInt64(max(0, free))))")
        }
        if let total = storage.volumeTotalBytes {
            lines.append("documents volume: \(ByteSize.format(UInt64(max(0, total))))")
        }
        lines.append("documents: \(context.workingDirectory.path)")

        for line in lines { context.log(line) }

        let payload = EnvironmentPayload(
            scenario: id,
            revision: BuildInfo.gitRevision,
            environment: report,
            footprint: footprint,
            pageSizeBytes: pageSize,
            workingDirectory: context.workingDirectory.path,
            volumeFreeBytes: storage.volumeAvailableBytes.map { UInt64(max(0, $0)) },
            volumeTotalBytes: storage.volumeTotalBytes.map { UInt64(max(0, $0)) })

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        let budget = footprint?.availableBytes.map { $0 + (footprint?.footprintBytes ?? 0) }
        return ScenarioResult(
            headline: budget.map { "budget \(ByteSize.format($0)), page \(pageSize) B" }
                ?? "page \(pageSize) B",
            lines: lines,
            json: try? encoder.encode(payload),
            jsonFilename: "environment.json")
    }

    struct EnvironmentPayload: Encodable {
        let scenario: String
        let revision: String
        let environment: EnvironmentReport
        let footprint: ProcessFootprint?
        let pageSizeBytes: Int
        let workingDirectory: String
        let volumeFreeBytes: UInt64?
        let volumeTotalBytes: UInt64?
    }
}
