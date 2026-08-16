import Darwin
import Foundation

/// Hardware identity of the machine a measurement ran on.
///
/// Populates the `device` object of the benchmark schema (spec §18.3). Fields
/// are optional because the underlying sysctl keys are not present on every
/// machine; a null in the output means "this machine did not report it", which
/// is information, unlike a fabricated zero.
///
/// GPU identification is deliberately absent: naming the GPU needs Metal, and
/// spec §7.1 keeps the storage layer free of the compute stack. It belongs
/// with the Metal backend, once there is one.
public struct DeviceInfo: Codable, Equatable {
    /// `hw.model`, e.g. `Mac16,10`. The stable way to identify the machine.
    public let modelIdentifier: String?
    /// `machdep.cpu.brand_string`, e.g. `Apple M4 Pro`.
    public let cpuBrand: String?
    /// Total logical CPUs.
    public let cpuCoreCount: Int?
    /// Physical CPUs.
    public let physicalCoreCount: Int?
    /// Performance cores (`hw.perflevel0.logicalcpu`). Null on symmetric CPUs.
    public let performanceCoreCount: Int?
    /// Efficiency cores (`hw.perflevel1.logicalcpu`). Null on symmetric CPUs.
    public let efficiencyCoreCount: Int?
    /// CPUs currently available to this process, which can be lower than the
    /// physical count under thermal or scheduling limits.
    public let activeProcessorCount: Int
    /// `hw.memsize`.
    public let physicalMemoryBytes: UInt64
    /// VM page size, the natural alignment for I/O buffers.
    public let pageSizeBytes: Int

    public static func current() -> DeviceInfo {
        DeviceInfo(
            modelIdentifier: Sysctl.string("hw.model"),
            cpuBrand: Sysctl.string("machdep.cpu.brand_string"),
            cpuCoreCount: Sysctl.int("hw.logicalcpu_max") ?? Sysctl.int("hw.ncpu"),
            physicalCoreCount: Sysctl.int("hw.physicalcpu_max") ?? Sysctl.int("hw.physicalcpu"),
            performanceCoreCount: Sysctl.int("hw.perflevel0.logicalcpu"),
            efficiencyCoreCount: Sysctl.int("hw.perflevel1.logicalcpu"),
            activeProcessorCount: ProcessInfo.processInfo.activeProcessorCount,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            pageSizeBytes: Int(getpagesize())
        )
    }

    private enum CodingKeys: String, CodingKey {
        case modelIdentifier, cpuBrand, cpuCoreCount, physicalCoreCount
        case performanceCoreCount, efficiencyCoreCount, activeProcessorCount
        case physicalMemoryBytes, pageSizeBytes
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeAlways(modelIdentifier, forKey: .modelIdentifier)
        try container.encodeAlways(cpuBrand, forKey: .cpuBrand)
        try container.encodeAlways(cpuCoreCount, forKey: .cpuCoreCount)
        try container.encodeAlways(physicalCoreCount, forKey: .physicalCoreCount)
        try container.encodeAlways(performanceCoreCount, forKey: .performanceCoreCount)
        try container.encodeAlways(efficiencyCoreCount, forKey: .efficiencyCoreCount)
        try container.encode(activeProcessorCount, forKey: .activeProcessorCount)
        try container.encode(physicalMemoryBytes, forKey: .physicalMemoryBytes)
        try container.encode(pageSizeBytes, forKey: .pageSizeBytes)
    }
}

/// Operating system identity. Populates the `os` object of spec §18.3.
public struct OSInfo: Codable, Equatable {
    /// Product platform name (`macOS` or `iOS`).
    public let name: String
    /// Marketing version, e.g. `15.7.7`.
    public let version: String
    /// Build identifier from `kern.osversion`, e.g. `24G720`. Point releases
    /// with identical marketing versions differ here, and storage behaviour has
    /// changed across builds before.
    public let build: String?
    /// `kern.osrelease`, the Darwin kernel version.
    public let kernelVersion: String?
    /// `kern.ostype`, `Darwin`.
    public let kernelType: String?

    public static func current() -> OSInfo {
        let version = ProcessInfo.processInfo.operatingSystemVersion
        #if os(iOS)
            let platformName = "iOS"
        #elseif os(macOS)
            let platformName = "macOS"
        #else
            // StorageCore currently ships on macOS and iOS. Keep an honest
            // Darwin fallback for tools that compile it on another Darwin OS
            // instead of mislabelling their reproducibility record as macOS.
            let platformName = "Darwin"
        #endif
        return OSInfo(
            name: platformName,
            version: "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)",
            build: Sysctl.string("kern.osversion"),
            kernelVersion: Sysctl.string("kern.osrelease"),
            kernelType: Sysctl.string("kern.ostype")
        )
    }

    private enum CodingKeys: String, CodingKey {
        case name, version, build, kernelVersion, kernelType
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(version, forKey: .version)
        try container.encodeAlways(build, forKey: .build)
        try container.encodeAlways(kernelVersion, forKey: .kernelVersion)
        try container.encodeAlways(kernelType, forKey: .kernelType)
    }
}
