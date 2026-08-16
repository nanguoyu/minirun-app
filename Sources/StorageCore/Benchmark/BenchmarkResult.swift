import Foundation

/// Machine-readable benchmark result, schema version 1 (spec §18.3).
///
/// The top-level keys are exactly those of the provisional schema in the
/// specification, including `model` and `compute`, which are explicitly null at
/// M0: there is no model and no GPU work yet. They are emitted rather than
/// omitted so that a consumer can tell "this run had no model" apart from "this
/// tool predates the field".
public struct BenchmarkResult: Encodable {
    public static let schemaVersion = 1

    public let timestamp: String
    public let gitRevision: String
    public let device: DeviceInfo
    public let os: OSInfo
    public let storage: StorageInfo
    public let runtime: RuntimeInfo
    public let workload: WorkloadInfo
    public let memory: MemoryReport
    public let io: IOReport
    public let latency: LatencyStatistics
    public let correctness: CorrectnessReport

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case timestamp
        case gitRevision = "git_revision"
        case device
        case os
        case storage
        case model
        case runtime
        case workload
        case memory
        case io
        case compute
        case latency
        case correctness
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(Self.schemaVersion, forKey: .schemaVersion)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encode(gitRevision, forKey: .gitRevision)
        try container.encode(device, forKey: .device)
        try container.encode(os, forKey: .os)
        try container.encode(storage, forKey: .storage)
        try container.encodeNil(forKey: .model)
        try container.encode(runtime, forKey: .runtime)
        try container.encode(workload, forKey: .workload)
        try container.encode(memory, forKey: .memory)
        try container.encode(io, forKey: .io)
        try container.encodeNil(forKey: .compute)
        try container.encode(latency, forKey: .latency)
        try container.encode(correctness, forKey: .correctness)
    }
}

/// What ran, and under what conditions.
public struct RuntimeInfo: Encodable {
    public let name: String
    public let version: String
    public let ioMode: String
    /// Prose description of the concurrency mechanism, so a reader does not
    /// have to know what `pread` implies about threading.
    public let workerModel: String
    public let thermalStateStart: String
    public let thermalStateEnd: String
    public let lowPowerModeEnabled: Bool
    /// Spec §17.4 requires the command line to travel with the result.
    public let commandLine: [String]
}

/// Every parameter of the run. Reproducing a result should need nothing else.
public struct WorkloadInfo: Encodable {
    public let path: String
    public let fileSizeBytes: UInt64
    public let readSizeBytes: Int
    public let queueDepth: Int
    public let durationSeconds: Double
    public let warmupSeconds: Double
    public let pattern: String
    public let mode: String
    /// Channel topology used by the dispatchio engine. Echoed for every run,
    /// including pread runs, so a sweep's parameter columns line up.
    public let dispatchioChannels: String
    public let cacheMode: String
    /// What the cache mode does and does not guarantee.
    public let cacheNote: String
    /// True when the file is small enough relative to physical memory that the
    /// result may describe the page cache rather than storage. See
    /// `CacheConfounding`. A true here does not invalidate the run; it says the
    /// number needs reading in that light.
    public let cacheConfoundedRisk: Bool
    public let seed: UInt64
    public let verify: Bool
    public let verifySampleInterval: Int
    /// Whole read-size blocks in the file; the addressable space of the run.
    public let blockCount: UInt64
    public let addressableBytes: UInt64
}

public struct MemoryReport: Encodable {
    public let peakResidentBytes: UInt64?
    public let residentBytesAtEnd: UInt64?

    private enum CodingKeys: String, CodingKey {
        case peakResidentBytes, residentBytesAtEnd
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeAlways(peakResidentBytes, forKey: .peakResidentBytes)
        try container.encodeAlways(residentBytesAtEnd, forKey: .residentBytesAtEnd)
    }
}

public struct IOReport: Encodable {
    public let readCount: Int
    public let bytesRequested: UInt64
    public let bytesRead: UInt64
    public let shortReadCount: Int
    public let errorCount: Int
    public let firstError: String?
    /// Measured window only; warm-up is excluded.
    public let wallSeconds: Double
    /// Decimal megabytes, 10^6 bytes — the unit drive vendors quote.
    public let throughputMbPerSecond: Double
    /// Binary mebibytes, 2^20 bytes. Both are emitted because "MB/s" is
    /// ambiguous by a factor of 1.049 and storage results get compared.
    public let throughputMibPerSecond: Double
    public let iops: Double

    private enum CodingKeys: String, CodingKey {
        case readCount, bytesRequested, bytesRead, shortReadCount, errorCount, firstError
        case wallSeconds, throughputMbPerSecond, throughputMibPerSecond, iops
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(readCount, forKey: .readCount)
        try container.encode(bytesRequested, forKey: .bytesRequested)
        try container.encode(bytesRead, forKey: .bytesRead)
        try container.encode(shortReadCount, forKey: .shortReadCount)
        try container.encode(errorCount, forKey: .errorCount)
        try container.encodeAlways(firstError, forKey: .firstError)
        try container.encode(wallSeconds, forKey: .wallSeconds)
        try container.encode(throughputMbPerSecond, forKey: .throughputMbPerSecond)
        try container.encode(throughputMibPerSecond, forKey: .throughputMibPerSecond)
        try container.encode(iops, forKey: .iops)
    }
}

public struct CorrectnessReport: Encodable {
    /// `passed`, `failed`, or `not_checked` when `--verify` was off.
    public let status: String
    public let verifyEnabled: Bool
    public let verifiedReadCount: Int
    public let verifiedBytes: Int
    public let mismatchedBytes: Int
    public let firstMismatchOffset: UInt64?
    public let shortReadCount: Int
    public let errorCount: Int

    public var passed: Bool { status != "failed" }

    private enum CodingKeys: String, CodingKey {
        case status, verifyEnabled, verifiedReadCount, verifiedBytes
        case mismatchedBytes, firstMismatchOffset, shortReadCount, errorCount
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(status, forKey: .status)
        try container.encode(verifyEnabled, forKey: .verifyEnabled)
        try container.encode(verifiedReadCount, forKey: .verifiedReadCount)
        try container.encode(verifiedBytes, forKey: .verifiedBytes)
        try container.encode(mismatchedBytes, forKey: .mismatchedBytes)
        try container.encodeAlways(firstMismatchOffset, forKey: .firstMismatchOffset)
        try container.encode(shortReadCount, forKey: .shortReadCount)
        try container.encode(errorCount, forKey: .errorCount)
    }
}

extension BenchmarkResult {
    /// Concise summary for a terminal. The JSON is the record; this is for the
    /// person deciding whether the run was worth recording.
    public func humanSummary() -> String {
        var lines: [String] = []

        func row(_ label: String, _ value: String?) {
            guard let value, !value.isEmpty else { return }
            lines.append("  \(label.padding(toLength: 12, withPad: " ", startingAt: 0))\(value)")
        }

        lines.append(
            "minirun bench  \(workload.mode)  \(workload.pattern)  "
                + "\(ByteSize.format(UInt64(workload.readSizeBytes))) x qd\(workload.queueDepth)")
        lines.append("")

        var location = storage.resolvedPath
        if let volume = storage.volumeName, let filesystem = storage.filesystemType {
            location += " (\(ByteSize.format(workload.fileSizeBytes)), \(filesystem) on \"\(volume)\")"
        }
        row("path", location)
        if workload.mode == "dispatchio" {
            row("channels", workload.dispatchioChannels)
        }
        row("cache", "\(workload.cacheMode)")
        row(
            "window",
            String(format: "%.2f s measured", io.wallSeconds)
                + (workload.warmupSeconds > 0
                    ? String(format: ", %.2f s warmup excluded", workload.warmupSeconds) : ""))
        row(
            "reads",
            "\(io.readCount) (\(String(format: "%.1f", io.iops)) IOPS)")
        row("bytes", ByteSize.format(io.bytesRead))
        row(
            "throughput",
            String(format: "%.1f MB/s", io.throughputMbPerSecond)
                + String(format: " (%.1f MiB/s)", io.throughputMibPerSecond))
        row(
            "latency",
            String(
                format: "p50 %.1f  p95 %.1f  p99 %.1f  max %.1f  (us)",
                latency.p50Us, latency.p95Us, latency.p99Us, latency.maxUs))
        row(
            "thermal",
            "\(runtime.thermalStateStart) -> \(runtime.thermalStateEnd)")
        row("peak rss", memory.peakResidentBytes.map(ByteSize.format))

        switch correctness.status {
        case "not_checked":
            row("verify", "off (pass --verify to check content)")
        case "passed":
            row(
                "verify",
                "\(correctness.verifiedReadCount) reads sampled, "
                    + "\(ByteSize.format(UInt64(correctness.verifiedBytes))) checked, 0 mismatches")
        default:
            row(
                "verify",
                "FAILED: \(correctness.mismatchedBytes) mismatched bytes"
                    + (correctness.firstMismatchOffset.map { ", first at offset \($0)" } ?? ""))
        }

        if io.shortReadCount > 0 {
            row("short reads", "\(io.shortReadCount)")
        }
        if io.errorCount > 0 {
            row("errors", "\(io.errorCount): \(io.firstError ?? "unknown")")
        }

        lines.append("")
        lines.append("  \(workload.cacheNote)")
        if workload.cacheConfoundedRisk {
            lines.append(
                "  This file is small relative to physical memory, so the figures above may "
                    + "describe the page cache rather than storage.")
        }

        return lines.joined(separator: "\n")
    }
}
