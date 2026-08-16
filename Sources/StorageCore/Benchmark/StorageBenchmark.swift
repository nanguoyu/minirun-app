import Darwin
import Foundation

/// Runs a storage read benchmark and assembles the result (WP-003).
///
/// Opens the file, applies the cache policy, hands the read loop to one of the
/// two engines, and packages the tallies together with the environment the run
/// happened in. Warm-up is excluded by the engines themselves: a read that
/// starts before the warm-up deadline is issued but not counted, so the
/// transition costs nothing and the offset stream stays continuous across it.
public enum StorageBenchmark {
    /// - Parameter warning: Receives advisory messages once the file has been
    ///   opened and measured but before reads begin, so a caller can surface
    ///   them while there is still time to abandon the run. Presentation is the
    ///   caller's business; this type does not write to any stream.
    public static func run(
        configuration: BenchmarkConfiguration,
        warning: ((String) -> Void)? = nil
    ) throws -> BenchmarkResult {
        try configuration.validate()

        let descriptor = open(configuration.path, O_RDONLY)
        guard descriptor >= 0 else {
            throw StorageCoreError.posix(
                operation: "open", path: configuration.path, code: errno)
        }
        defer { close(descriptor) }

        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            throw StorageCoreError.posix(
                operation: "fstat", path: configuration.path, code: errno)
        }
        guard status.st_mode & S_IFMT == S_IFREG else {
            throw StorageCoreError.invalidArgument(
                "'\(configuration.path)' is not a regular file")
        }
        let fileSizeBytes = UInt64(max(status.st_size, 0))
        try configuration.validate(fileSizeBytes: fileSizeBytes)

        if configuration.disableCache {
            guard fcntl(descriptor, F_NOCACHE, 1) != -1 else {
                throw StorageCoreError.posix(
                    operation: "fcntl(F_NOCACHE)", path: configuration.path, code: errno)
            }
        }

        if let warning,
            let message = CacheConfounding.warning(
                fileSizeBytes: fileSizeBytes,
                physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                cacheMode: configuration.cacheMode)
        {
            warning(message)
        }

        let thermalStart = ThermalState.currentName
        let begin = MonotonicClock.now()
        let warmupDeadline = begin &+ MonotonicClock.nanoseconds(configuration.warmupSeconds)
        let window = RunWindow(
            warmupDeadline: warmupDeadline,
            stopDeadline: warmupDeadline
                &+ MonotonicClock.nanoseconds(configuration.durationSeconds))

        var accounting: ReadAccounting
        switch configuration.mode {
        case .pread:
            accounting = try PreadEngine.run(
                configuration: configuration,
                fileDescriptor: descriptor,
                fileSizeBytes: fileSizeBytes,
                window: window)
        case .dispatchio:
            accounting = try DispatchIOEngine.run(
                configuration: configuration,
                fileDescriptor: descriptor,
                fileSizeBytes: fileSizeBytes,
                window: window)
        }

        let finished = MonotonicClock.now()
        let thermalEnd = ThermalState.currentName

        guard accounting.readCount > 0 else {
            throw StorageCoreError.invalidArgument(
                "no reads completed in the measured window"
                    + (accounting.firstError.map { "; first error: \($0)" } ?? "")
                    + ". Increase --duration or reduce --warmup.")
        }

        // The measured window runs from the end of warm-up to the moment the
        // last in-flight read returned. Reads that started before the stop
        // deadline are counted, so the time they took is counted too.
        let wallSeconds = MonotonicClock.seconds(from: warmupDeadline, to: finished)
        guard let latency = LatencyStatistics.compute(nanosecondSamples: &accounting.latencySamplesNs)
        else {
            throw StorageCoreError.invalidArgument("no latency samples were recorded")
        }

        let blockCount = OffsetScheduler.blockCount(
            fileSizeBytes: fileSizeBytes, readSizeBytes: configuration.readSizeBytes)
        let memory = MemoryUsage.current()

        let correctnessStatus: String
        if !configuration.verify {
            correctnessStatus = "not_checked"
        } else if accounting.verification.mismatchedBytes > 0 || accounting.errorCount > 0
            || accounting.shortReadCount > 0
        {
            correctnessStatus = "failed"
        } else {
            correctnessStatus = "passed"
        }

        return BenchmarkResult(
            timestamp: JSONOutput.timestamp(),
            gitRevision: BuildInfo.gitRevision,
            device: DeviceInfo.current(),
            os: OSInfo.current(),
            storage: StorageInfo.describing(path: configuration.path),
            runtime: RuntimeInfo(
                name: "minirun",
                version: BuildInfo.version,
                ioMode: configuration.mode.rawValue,
                workerModel: workerModel(for: configuration),
                thermalStateStart: thermalStart,
                thermalStateEnd: thermalEnd,
                lowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled,
                commandLine: ProcessInfo.processInfo.arguments),
            workload: WorkloadInfo(
                path: configuration.path,
                fileSizeBytes: fileSizeBytes,
                readSizeBytes: configuration.readSizeBytes,
                queueDepth: configuration.queueDepth,
                durationSeconds: configuration.durationSeconds,
                warmupSeconds: configuration.warmupSeconds,
                pattern: configuration.pattern.rawValue,
                mode: configuration.mode.rawValue,
                dispatchioChannels: configuration.dispatchIOChannels.rawValue,
                cacheMode: configuration.cacheMode.rawValue,
                cacheNote: configuration.cacheMode.note,
                cacheConfoundedRisk: CacheConfounding.isAtRisk(
                    fileSizeBytes: fileSizeBytes,
                    physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory),
                seed: configuration.seed,
                verify: configuration.verify,
                verifySampleInterval: configuration.verifySampleInterval,
                blockCount: blockCount,
                addressableBytes: blockCount * UInt64(configuration.readSizeBytes)),
            memory: MemoryReport(
                peakResidentBytes: memory?.peakResidentBytes,
                residentBytesAtEnd: memory?.residentBytes),
            io: IOReport(
                readCount: accounting.readCount,
                bytesRequested: accounting.bytesRequested,
                bytesRead: accounting.bytesRead,
                shortReadCount: accounting.shortReadCount,
                errorCount: accounting.errorCount,
                firstError: accounting.firstError,
                wallSeconds: wallSeconds,
                throughputMbPerSecond: rate(accounting.bytesRead, wallSeconds, 1_000_000),
                throughputMibPerSecond: rate(accounting.bytesRead, wallSeconds, 1_048_576),
                iops: wallSeconds > 0 ? Double(accounting.readCount) / wallSeconds : 0),
            latency: latency,
            correctness: CorrectnessReport(
                status: correctnessStatus,
                verifyEnabled: configuration.verify,
                verifiedReadCount: accounting.verifiedReadCount,
                verifiedBytes: accounting.verification.comparedBytes,
                mismatchedBytes: accounting.verification.mismatchedBytes,
                firstMismatchOffset: accounting.verification.firstMismatchOffset,
                shortReadCount: accounting.shortReadCount,
                errorCount: accounting.errorCount)
        )
    }

    private static func rate(_ bytes: UInt64, _ seconds: Double, _ unit: Double) -> Double {
        seconds > 0 ? Double(bytes) / seconds / unit : 0
    }

    private static func workerModel(for configuration: BenchmarkConfiguration) -> String {
        switch configuration.mode {
        case .pread:
            return "\(configuration.queueDepth) threads issuing blocking pread on a shared descriptor"
        case .dispatchio:
            switch configuration.dispatchIOChannels {
            case .one:
                return
                    "one DispatchIO random-access channel, "
                    + "\(configuration.queueDepth) reads in flight"
            case .perSlot:
                return
                    "\(configuration.queueDepth) DispatchIO random-access channels on duplicated "
                    + "descriptors, one read in flight each"
            }
        }
    }
}
