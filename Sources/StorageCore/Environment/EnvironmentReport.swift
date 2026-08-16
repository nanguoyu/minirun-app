import Foundation

/// A snapshot of the machine, OS, and storage volume a measurement would run
/// on (WP-002).
///
/// Collected entirely from `sysctl`, `ProcessInfo`, and `URLResourceValues`.
/// Nothing is transmitted anywhere: the report is printed, and that is all it
/// does. Spec §20 and WP-002 both require that no data is collected or sent
/// automatically.
public struct EnvironmentReport: Codable, Equatable {
    public let schemaVersion: Int
    public let reportType: String
    public let timestamp: String
    public let gitRevision: String
    public let tool: ToolInfo
    public let device: DeviceInfo
    public let os: OSInfo
    public let storage: StorageInfo
    public let process: ProcessInfoSnapshot

    public static let currentSchemaVersion = 1

    public static func current(path: String) -> EnvironmentReport {
        EnvironmentReport(
            schemaVersion: currentSchemaVersion,
            reportType: "environment",
            timestamp: JSONOutput.timestamp(),
            gitRevision: BuildInfo.gitRevision,
            tool: ToolInfo.current,
            device: DeviceInfo.current(),
            os: OSInfo.current(),
            storage: StorageInfo.describing(path: path),
            process: ProcessInfoSnapshot.current()
        )
    }
}

/// Identity of the binary that produced a report.
public struct ToolInfo: Codable, Equatable {
    public let name: String
    public let version: String

    public static var current: ToolInfo {
        ToolInfo(name: "minirun", version: BuildInfo.version)
    }
}

/// Process-level state worth recording next to a measurement.
public struct ProcessInfoSnapshot: Codable, Equatable {
    public let residentBytes: UInt64?
    public let peakResidentBytes: UInt64?
    /// Coarse thermal annotation; see `ThermalState`.
    public let thermalState: String
    /// True when the OS has put the process into a reduced-activity mode, which
    /// would invalidate a timing run.
    public let lowPowerModeEnabled: Bool

    public static func current() -> ProcessInfoSnapshot {
        let memory = MemoryUsage.current()
        return ProcessInfoSnapshot(
            residentBytes: memory?.residentBytes,
            peakResidentBytes: memory?.peakResidentBytes,
            thermalState: ThermalState.currentName,
            lowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled
        )
    }

    private enum CodingKeys: String, CodingKey {
        case residentBytes, peakResidentBytes, thermalState, lowPowerModeEnabled
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeAlways(residentBytes, forKey: .residentBytes)
        try container.encodeAlways(peakResidentBytes, forKey: .peakResidentBytes)
        try container.encode(thermalState, forKey: .thermalState)
        try container.encode(lowPowerModeEnabled, forKey: .lowPowerModeEnabled)
    }
}

extension EnvironmentReport {
    /// Human-readable rendering for a terminal.
    public func humanSummary() -> String {
        var lines: [String] = []

        func row(_ label: String, _ value: String?) {
            guard let value, !value.isEmpty else { return }
            lines.append("  \(label.padding(toLength: 18, withPad: " ", startingAt: 0))\(value)")
        }

        lines.append("minirun \(tool.version) (\(gitRevision))")
        lines.append("")
        lines.append("device")
        row("model", device.modelIdentifier)
        row("cpu", device.cpuBrand)
        var cores = device.cpuCoreCount.map { "\($0) logical" } ?? "unknown"
        if let performance = device.performanceCoreCount,
            let efficiency = device.efficiencyCoreCount
        {
            cores += " (\(performance) performance + \(efficiency) efficiency)"
        }
        cores += ", \(device.activeProcessorCount) active"
        row("cores", cores)
        row("memory", ByteSize.format(device.physicalMemoryBytes))
        row("page size", ByteSize.format(UInt64(device.pageSizeBytes)))

        lines.append("")
        lines.append("os")
        row("version", "\(os.name) \(os.version)" + (os.build.map { " (\($0))" } ?? ""))
        row("kernel", [os.kernelType, os.kernelVersion].compactMap { $0 }.joined(separator: " "))

        lines.append("")
        lines.append("process")
        row("resident", process.residentBytes.map(ByteSize.format))
        row("peak resident", process.peakResidentBytes.map(ByteSize.format))
        row("thermal", process.thermalState)
        row("low power", process.lowPowerModeEnabled ? "enabled" : "disabled")

        lines.append("")
        lines.append("storage")
        row("path", storage.resolvedPath)
        if !storage.pathExists {
            row("exists", "no (volume below describes the nearest existing parent)")
        } else if let size = storage.fileSizeBytes {
            row("file size", ByteSize.format(size))
        }
        row("access", "\(storage.pathIsReadable ? "r" : "-")\(storage.pathIsWritable ? "w" : "-")")
        row("volume", storage.volumeName)
        row("mounted at", storage.volumeMountPath)
        row("filesystem", storage.filesystemType)
        row("capacity", storage.volumeTotalBytes.map { ByteSize.format(UInt64($0)) })
        row("free", storage.volumeAvailableBytes.map { ByteSize.format(UInt64($0)) })
        row(
            "free (purgeable)",
            storage.volumeAvailableBytesForImportantUsage.map { ByteSize.format(UInt64($0)) })
        var flags: [String] = []
        if storage.volumeIsInternal == true { flags.append("internal") }
        if storage.volumeIsRemovable == true { flags.append("removable") }
        if storage.volumeIsEjectable == true { flags.append("ejectable") }
        if storage.volumeIsReadOnly == true { flags.append("read-only") }
        row("flags", flags.isEmpty ? nil : flags.joined(separator: ", "))

        return lines.joined(separator: "\n")
    }
}
