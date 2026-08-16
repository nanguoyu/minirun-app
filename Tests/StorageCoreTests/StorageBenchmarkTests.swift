import Foundation
import XCTest

@testable import StorageCore

/// End-to-end runs of both engines.
///
/// These assert correctness only — read counts, byte accounting, verification.
/// Nothing here asserts a throughput or a latency: those depend on the machine,
/// its cache state, and what else is running, and a test that fails on a busy
/// laptop teaches nobody anything.
final class StorageBenchmarkTests: XCTestCase {
    private var directory: URL!
    private var filePath: String!
    private let seed: UInt64 = 4242
    private let fileSize: UInt64 = 8 << 20  // 8 MiB
    private let readSize = 64 << 10  // 64 KiB

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-bench-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        filePath = directory.appendingPathComponent("data.bin").path
        _ = try FileGenerator.generate(
            FileGenerationRequest(path: filePath, sizeBytes: fileSize, seed: seed))
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func configuration(
        mode: IOMode,
        pattern: AccessPattern = .random,
        seed: UInt64? = nil,
        verify: Bool = true,
        sampleInterval: Int = 1,
        queueDepth: Int = 2,
        duration: Double = 0.2
    ) -> BenchmarkConfiguration {
        BenchmarkConfiguration(
            path: filePath,
            readSizeBytes: readSize,
            queueDepth: queueDepth,
            durationSeconds: duration,
            warmupSeconds: 0.05,
            pattern: pattern,
            mode: mode,
            disableCache: false,
            seed: seed ?? self.seed,
            verify: verify,
            verifySampleInterval: sampleInterval)
    }

    private func assertHealthy(_ result: BenchmarkResult, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertGreaterThan(result.io.readCount, 0, file: file, line: line)
        XCTAssertEqual(result.io.errorCount, 0, result.io.firstError ?? "", file: file, line: line)
        XCTAssertEqual(result.io.shortReadCount, 0, file: file, line: line)
        XCTAssertEqual(
            result.io.bytesRead, UInt64(result.io.readCount) * UInt64(readSize),
            "every read returned a full block", file: file, line: line)
        XCTAssertEqual(
            result.io.bytesRequested, result.io.bytesRead, file: file, line: line)
        XCTAssertEqual(result.latency.sampleCount, result.io.readCount, file: file, line: line)
    }

    func testPreadRandomRunVerifiesClean() throws {
        let result = try StorageBenchmark.run(configuration: configuration(mode: .pread))
        assertHealthy(result)
        XCTAssertEqual(result.correctness.status, "passed")
        XCTAssertEqual(result.correctness.mismatchedBytes, 0)
        XCTAssertGreaterThan(result.correctness.verifiedReadCount, 0)
        XCTAssertNil(result.correctness.firstMismatchOffset)
    }

    func testDispatchIORandomRunVerifiesClean() throws {
        let result = try StorageBenchmark.run(configuration: configuration(mode: .dispatchio))
        assertHealthy(result)
        XCTAssertEqual(result.correctness.status, "passed")
        XCTAssertEqual(result.correctness.mismatchedBytes, 0)
        XCTAssertGreaterThan(result.correctness.verifiedReadCount, 0)
    }

    func testSequentialRunsVerifyClean() throws {
        for mode in IOMode.allCases {
            let result = try StorageBenchmark.run(
                configuration: configuration(mode: mode, pattern: .sequential, queueDepth: 4))
            assertHealthy(result)
            XCTAssertEqual(result.correctness.status, "passed", "\(mode)")
        }
    }

    func testNoCacheRunsVerifyClean() throws {
        for mode in IOMode.allCases {
            var config = configuration(mode: mode)
            config.disableCache = true
            let result = try StorageBenchmark.run(configuration: config)
            assertHealthy(result)
            XCTAssertEqual(result.correctness.status, "passed", "\(mode)")
            XCTAssertEqual(result.workload.cacheMode, "f_nocache")
        }
    }

    /// The verification path has to be able to fail, or a clean result proves
    /// nothing. Reading with the wrong seed is the cheapest way to corrupt the
    /// expectation without corrupting the file.
    func testVerificationDetectsWrongSeed() throws {
        for mode in IOMode.allCases {
            let result = try StorageBenchmark.run(
                configuration: configuration(mode: mode, seed: seed &+ 1))
            XCTAssertEqual(result.correctness.status, "failed", "\(mode)")
            XCTAssertGreaterThan(result.correctness.mismatchedBytes, 0, "\(mode)")
            XCTAssertNotNil(result.correctness.firstMismatchOffset, "\(mode)")
        }
    }

    func testVerificationOffReportsNotChecked() throws {
        let result = try StorageBenchmark.run(
            configuration: configuration(mode: .pread, verify: false))
        assertHealthy(result)
        XCTAssertEqual(result.correctness.status, "not_checked")
        XCTAssertEqual(result.correctness.verifiedReadCount, 0)
        XCTAssertFalse(result.correctness.verifyEnabled)
    }

    func testSamplingIntervalLimitsVerifiedReads() throws {
        let result = try StorageBenchmark.run(
            configuration: configuration(mode: .pread, sampleInterval: 8))
        assertHealthy(result)
        XCTAssertLessThan(
            result.correctness.verifiedReadCount, result.io.readCount,
            "sampling must verify fewer reads than it performs")
        XCTAssertEqual(result.correctness.status, "passed")
    }

    /// Per-slot mode duplicates the descriptor once per slot and hands each
    /// duplicate back in a cleanup handler. A leak or a double close here would
    /// surface as an exhausted descriptor table, so run it repeatedly and check
    /// the results stay clean.
    func testPerSlotChannelsVerifyCleanAcrossRepeatedRuns() throws {
        for _ in 0..<3 {
            var config = configuration(mode: .dispatchio, queueDepth: 4, duration: 0.15)
            config.dispatchIOChannels = .perSlot
            let result = try StorageBenchmark.run(configuration: config)
            assertHealthy(result)
            XCTAssertEqual(result.correctness.status, "passed")
            XCTAssertEqual(result.workload.dispatchioChannels, "per-slot")
        }
    }

    func testPerSlotChannelsWorkSequentially() throws {
        var config = configuration(
            mode: .dispatchio, pattern: .sequential, queueDepth: 3, duration: 0.15)
        config.dispatchIOChannels = .perSlot
        let result = try StorageBenchmark.run(configuration: config)
        assertHealthy(result)
        XCTAssertEqual(result.correctness.status, "passed")
    }

    func testPerSlotChannelsHonourNoCache() throws {
        var config = configuration(mode: .dispatchio, queueDepth: 4, duration: 0.15)
        config.dispatchIOChannels = .perSlot
        config.disableCache = true
        let result = try StorageBenchmark.run(configuration: config)
        assertHealthy(result)
        XCTAssertEqual(result.correctness.status, "passed")
        XCTAssertEqual(result.workload.cacheMode, "f_nocache")
    }

    func testPerSlotVerificationStillDetectsWrongSeed() throws {
        var config = configuration(mode: .dispatchio, seed: seed &+ 1, duration: 0.15)
        config.dispatchIOChannels = .perSlot
        let result = try StorageBenchmark.run(configuration: config)
        XCTAssertEqual(result.correctness.status, "failed")
        XCTAssertGreaterThan(result.correctness.mismatchedBytes, 0)
    }

    func testChannelModeDefaultsToOneAndIsEchoedByBothEngines() throws {
        for mode in IOMode.allCases {
            let result = try StorageBenchmark.run(
                configuration: configuration(mode: mode, duration: 0.1))
            XCTAssertEqual(
                result.workload.dispatchioChannels, "one",
                "\(mode) must echo the topology so sweep columns line up")
        }
    }

    func testQueueDepthOneWorks() throws {
        for mode in IOMode.allCases {
            let result = try StorageBenchmark.run(
                configuration: configuration(mode: mode, queueDepth: 1))
            assertHealthy(result)
        }
    }

    func testWorkloadEchoesEveryParameter() throws {
        var config = configuration(mode: .dispatchio, pattern: .sequential, queueDepth: 3)
        config.disableCache = true
        let result = try StorageBenchmark.run(configuration: config)

        XCTAssertEqual(result.workload.path, filePath)
        XCTAssertEqual(result.workload.fileSizeBytes, fileSize)
        XCTAssertEqual(result.workload.readSizeBytes, readSize)
        XCTAssertEqual(result.workload.queueDepth, 3)
        XCTAssertEqual(result.workload.durationSeconds, config.durationSeconds)
        XCTAssertEqual(result.workload.warmupSeconds, config.warmupSeconds)
        XCTAssertEqual(result.workload.pattern, "sequential")
        XCTAssertEqual(result.workload.mode, "dispatchio")
        XCTAssertEqual(result.workload.cacheMode, "f_nocache")
        XCTAssertEqual(result.workload.seed, seed)
        XCTAssertTrue(result.workload.verify)
        XCTAssertEqual(result.workload.blockCount, fileSize / UInt64(readSize))
        XCTAssertEqual(result.workload.addressableBytes, fileSize)
        XCTAssertFalse(result.workload.cacheNote.isEmpty)
    }

    /// The test fixture is 8 MiB against many gigabytes of memory, so a run
    /// against it must both flag the risk and say so before the reads start.
    func testSmallFixtureFlagsCacheConfoundingAndWarnsEarly() throws {
        var warnings: [String] = []
        let result = try StorageBenchmark.run(
            configuration: configuration(mode: .pread, duration: 0.1)
        ) { warnings.append($0) }

        XCTAssertTrue(result.workload.cacheConfoundedRisk)
        XCTAssertEqual(warnings.count, 1)
        XCTAssertTrue(warnings[0].lowercased().contains("warning"))
        XCTAssertTrue(result.humanSummary().contains("page cache rather than storage"))
    }

    func testWarningCallbackIsOptional() throws {
        let result = try StorageBenchmark.run(
            configuration: configuration(mode: .pread, duration: 0.1))
        XCTAssertTrue(result.workload.cacheConfoundedRisk)
    }

    /// Every optional in the result document must appear as an explicit null
    /// rather than vanish, so analysis code can index instead of probe. A clean
    /// run is the case that would otherwise drop `first_error` and
    /// `first_mismatch_offset`.
    func testOptionalFieldsAppearAsExplicitNulls() throws {
        let result = try StorageBenchmark.run(
            configuration: configuration(mode: .pread, duration: 0.1))
        let data = try JSONOutput.encoder().encode(result)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        func assertExplicitNull(_ path: [String], file: StaticString = #filePath, line: UInt = #line) {
            var node: Any? = object
            for key in path.dropLast() {
                node = (node as? [String: Any])?[key]
            }
            let container = try? XCTUnwrap(node as? [String: Any], file: file, line: line)
            let leaf = path[path.count - 1]
            XCTAssertTrue(
                container?.keys.contains(leaf) ?? false,
                "\(path.joined(separator: ".")) must be present", file: file, line: line)
            XCTAssertTrue(
                container?[leaf] is NSNull,
                "\(path.joined(separator: ".")) must be null, got "
                    + String(describing: container?[leaf]),
                file: file, line: line)
        }

        XCTAssertTrue(object["model"] is NSNull)
        XCTAssertTrue(object["compute"] is NSNull)
        assertExplicitNull(["io", "first_error"])
        assertExplicitNull(["correctness", "first_mismatch_offset"])
    }

    func testMissingFileFailsExplicitly() {
        var config = configuration(mode: .pread)
        config.path = directory.appendingPathComponent("absent.bin").path
        XCTAssertThrowsError(try StorageBenchmark.run(configuration: config)) { error in
            guard case StorageCoreError.posix(let operation, _, _) = error else {
                return XCTFail("expected a posix error, got \(error)")
            }
            XCTAssertEqual(operation, "open")
        }
    }

    func testDirectoryPathIsRejected() {
        var config = configuration(mode: .pread)
        config.path = directory.path
        XCTAssertThrowsError(try StorageBenchmark.run(configuration: config))
    }

    func testReadSizeLargerThanFileIsRejected() {
        var config = configuration(mode: .pread)
        config.readSizeBytes = Int(fileSize) * 2
        XCTAssertThrowsError(try StorageBenchmark.run(configuration: config)) { error in
            XCTAssertTrue("\(error)".contains("no complete block fits"), "\(error)")
        }
    }
}

final class CacheConfoundingTests: XCTestCase {
    private let memory: UInt64 = 32 << 30  // 32 GiB

    func testSmallFileIsAtRisk() {
        XCTAssertTrue(
            CacheConfounding.isAtRisk(fileSizeBytes: 256 << 20, physicalMemoryBytes: memory))
    }

    func testFileAtTwiceMemoryIsNotAtRisk() {
        XCTAssertFalse(
            CacheConfounding.isAtRisk(fileSizeBytes: memory * 2, physicalMemoryBytes: memory))
        XCTAssertFalse(
            CacheConfounding.isAtRisk(fileSizeBytes: memory * 4, physicalMemoryBytes: memory))
    }

    /// A file the size of memory is still at risk: the threshold is twice
    /// memory, because a file only slightly larger than RAM is partly cached
    /// and harder to interpret than either extreme.
    func testFileEqualToMemoryIsStillAtRisk() {
        XCTAssertTrue(
            CacheConfounding.isAtRisk(fileSizeBytes: memory, physicalMemoryBytes: memory))
    }

    func testBoundaryIsExclusive() {
        let threshold = memory * CacheConfounding.memoryMultiplier
        XCTAssertTrue(
            CacheConfounding.isAtRisk(fileSizeBytes: threshold - 1, physicalMemoryBytes: memory))
        XCTAssertFalse(
            CacheConfounding.isAtRisk(fileSizeBytes: threshold, physicalMemoryBytes: memory))
    }

    func testUnknownMemoryIsNotReportedAsRisk() {
        XCTAssertFalse(CacheConfounding.isAtRisk(fileSizeBytes: 1024, physicalMemoryBytes: 0))
    }

    func testHugeMemoryDoesNotOverflow() {
        XCTAssertTrue(
            CacheConfounding.isAtRisk(fileSizeBytes: 1 << 40, physicalMemoryBytes: UInt64.max))
    }

    /// F_NOCACHE does not clear the risk, because it does not evict pages an
    /// earlier run left resident. Both modes warn; only the advice differs.
    func testBothCacheModesWarnButDifferInAdvice() throws {
        let cached = try XCTUnwrap(
            CacheConfounding.warning(
                fileSizeBytes: 256 << 20, physicalMemoryBytes: memory, cacheMode: .pageCache))
        let noCache = try XCTUnwrap(
            CacheConfounding.warning(
                fileSizeBytes: 256 << 20, physicalMemoryBytes: memory, cacheMode: .noCache))
        XCTAssertTrue(cached.contains("page cache"))
        XCTAssertTrue(noCache.contains("F_NOCACHE"))
        XCTAssertNotEqual(cached, noCache)
    }

    func testNoWarningForLargeFile() {
        XCTAssertNil(
            CacheConfounding.warning(
                fileSizeBytes: memory * 3, physicalMemoryBytes: memory, cacheMode: .pageCache))
    }
}

final class ReadAccountingTests: XCTestCase {
    /// The signal the Dispatch I/O issuer uses to stop submitting after a
    /// failure. Inducing a real mid-run read error needs fault injection that
    /// M0 does not have, so the mechanism itself is covered directly.
    func testHasErrorFlipsOnFirstRecordedError() {
        let shared = SharedAccounting()
        XCTAssertFalse(shared.hasError)

        var clean = ReadAccounting()
        clean.readCount = 10
        shared.merge(clean)
        XCTAssertFalse(shared.hasError, "successful reads must not look like failures")

        var failed = ReadAccounting()
        failed.recordError("EIO")
        shared.merge(failed)
        XCTAssertTrue(shared.hasError)
        XCTAssertEqual(shared.snapshot.firstError, "EIO")
    }

    func testFirstErrorSurvivesLaterErrors() {
        var accounting = ReadAccounting()
        accounting.recordError("first")
        accounting.recordError("second")
        XCTAssertEqual(accounting.errorCount, 2)
        XCTAssertEqual(accounting.firstError, "first", "the first failure is the diagnostic one")
    }

    func testMergeKeepsEarliestFirstError() {
        var left = ReadAccounting()
        left.recordError("left")
        var right = ReadAccounting()
        right.recordError("right")
        left.merge(right)
        XCTAssertEqual(left.errorCount, 2)
        XCTAssertEqual(left.firstError, "left")
    }
}

final class BenchmarkConfigurationTests: XCTestCase {
    private func valid() -> BenchmarkConfiguration {
        BenchmarkConfiguration(
            path: "/tmp/x.bin", readSizeBytes: 4096, queueDepth: 4, durationSeconds: 1)
    }

    func testValidConfigurationPasses() {
        XCTAssertNoThrow(try valid().validate())
    }

    func testRejectsEmptyPath() {
        var config = valid()
        config.path = ""
        XCTAssertThrowsError(try config.validate())
    }

    func testRejectsNonPositiveReadSize() {
        var config = valid()
        config.readSizeBytes = 0
        XCTAssertThrowsError(try config.validate())
        config.readSizeBytes = -1
        XCTAssertThrowsError(try config.validate())
    }

    func testRejectsOversizedReadSize() {
        var config = valid()
        config.readSizeBytes = BenchmarkConfiguration.maximumReadSizeBytes + 1
        XCTAssertThrowsError(try config.validate())
    }

    func testRejectsBadQueueDepth() {
        var config = valid()
        config.queueDepth = 0
        XCTAssertThrowsError(try config.validate())
        config.queueDepth = BenchmarkConfiguration.maximumQueueDepth + 1
        XCTAssertThrowsError(try config.validate())
    }

    func testRejectsBadDuration() {
        var config = valid()
        config.durationSeconds = 0
        XCTAssertThrowsError(try config.validate())
        config.durationSeconds = -1
        XCTAssertThrowsError(try config.validate())
        config.durationSeconds = .infinity
        XCTAssertThrowsError(try config.validate())
        config.durationSeconds = BenchmarkConfiguration.maximumDurationSeconds + 1
        XCTAssertThrowsError(try config.validate())
    }

    func testRejectsNegativeWarmup() {
        var config = valid()
        config.warmupSeconds = -0.5
        XCTAssertThrowsError(try config.validate())
        config.warmupSeconds = 0
        XCTAssertNoThrow(try config.validate())
    }

    func testRejectsBadVerifyInterval() {
        var config = valid()
        config.verifySampleInterval = 0
        XCTAssertThrowsError(try config.validate())
    }

    func testRejectsReadSizeLargerThanFile() {
        let config = valid()
        XCTAssertThrowsError(try config.validate(fileSizeBytes: 100))
        XCTAssertNoThrow(try config.validate(fileSizeBytes: 4096))
    }

    func testCacheModeFollowsFlag() {
        var config = valid()
        XCTAssertEqual(config.cacheMode, .pageCache)
        config.disableCache = true
        XCTAssertEqual(config.cacheMode, .noCache)
        XCTAssertFalse(config.cacheMode.note.isEmpty)
        XCTAssertTrue(config.cacheMode.note.contains("not the same as a cold cache"))
    }
}
