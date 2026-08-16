import Darwin
import Foundation
@testable import MinirunKit
import XCTest

final class FileDigestTests: XCTestCase {
    private let memoryCeilingBytes: UInt64 = 96 << 20

    private final class CancellationProbe: @unchecked Sendable {
        private let lock = NSLock()
        private var checks = 0

        func shouldCancel() -> Bool {
            lock.lock()
            checks += 1
            let cancel = checks >= 3
            lock.unlock()
            return cancel
        }

        var checkCount: Int {
            lock.lock()
            defer { lock.unlock() }
            return checks
        }
    }

    func testFileDigestCrossesChunkBoundariesWithoutChangingEitherDigest() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("cross-boundary.bin")

        var payload = Data(count: FileDigestComputer.chunkBytes * 2 + 37)
        payload.withUnsafeMutableBytes { bytes in
            guard let base = bytes.bindMemory(to: UInt8.self).baseAddress else { return }
            for index in 0..<bytes.count {
                base[index] = UInt8(truncatingIfNeeded: index &* 31 &+ 17)
            }
        }
        try payload.write(to: file)

        XCTAssertEqual(
            try FileDigestComputer.sha256(ofFileAt: file),
            FileDigestComputer.sha256(of: payload))
        XCTAssertEqual(
            try FileDigestComputer.gitBlobSHA1(ofFileAt: file),
            FileDigestComputer.gitBlobSHA1(of: payload))
    }

    func testFileDigestHonoursCancellationAtAChunkBoundary() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("cancel.bin")
        try makeSparseFile(file, size: UInt64(FileDigestComputer.chunkBytes * 8))
        let probe = CancellationProbe()

        XCTAssertThrowsError(
            try FileDigestComputer.digest(
                ofFileAt: file, algorithm: .sha256,
                cancellationRequested: { probe.shouldCancel() })
        ) { error in
            XCTAssertTrue(error is CancellationError, "got \(error)")
        }
        XCTAssertGreaterThanOrEqual(probe.checkCount, 3)
    }

    /// A digest is allowed one fixed read buffer; it is not allowed to retain
    /// one Foundation allocation per chunk until a 1.5 TB verification ends.
    /// The sparse file keeps this regression fast and disk-light while still
    /// forcing 128 iterations through the production read loop.
    func testLargeSparseFileDigestHasAConstantResidentMemoryBound() throws {
        #if os(macOS)
            let directory = try makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            let file = directory.appendingPathComponent("512-mib-sparse.bin")
            try makeSparseFile(file, size: 512 << 20)

            let measurement = try measureDigest(file)
            print(
                "MINIRUN_DIGEST_MEMORY bytes=\(512 << 20) "
                    + "chunk_bytes=\(FileDigestComputer.chunkBytes) "
                    + "peak_resident_growth_bytes=\(measurement.peakResidentGrowthBytes)")
            XCTAssertEqual(measurement.digest.count, 64)
            XCTAssertLessThanOrEqual(
                measurement.peakResidentGrowthBytes, memoryCeilingBytes,
                "streaming a sparse 512 MiB file grew the process footprint by "
                    + "\(measurement.peakResidentGrowthBytes) bytes")
        #else
            throw XCTSkip("resident-memory sampling is a macOS regression gate")
        #endif
    }

    /// Opt-in benchmark for a real artifact file. Example:
    ///
    /// ```text
    /// MINIRUN_DIGEST_BENCH_FILE=/path/to/weight.bin \
    ///   xcrun xctest -XCTest MinirunKitTests.FileDigestTests/testRealFileDigestBenchmark <bundle>
    /// ```
    ///
    /// It reports throughput and the peak resident delta together: a faster
    /// verifier that grows with the artifact is not an improvement.
    func testRealFileDigestBenchmark() throws {
        #if os(macOS)
            guard let path = ProcessInfo.processInfo.environment["MINIRUN_DIGEST_BENCH_FILE"],
                !path.isEmpty
            else {
                throw XCTSkip("set MINIRUN_DIGEST_BENCH_FILE to a local artifact file")
            }
            let file = URL(fileURLWithPath: path)
            let size = try FileDigestComputer.byteCount(ofFileAt: file)
            let measurement = try measureDigest(file)
            let rate = measurement.elapsed > 0
                ? Double(size) / measurement.elapsed : 0
            let elapsedText = String(format: "%.6f", measurement.elapsed)
            let rateText = String(format: "%.0f", rate)
            print(
                "MINIRUN_DIGEST_BENCH "
                    + "file=\(file.lastPathComponent) bytes=\(size) "
                    + "chunk_bytes=\(FileDigestComputer.chunkBytes) "
                    + "seconds=\(elapsedText) "
                    + "bytes_per_second=\(rateText) "
                    + "peak_resident_growth_bytes=\(measurement.peakResidentGrowthBytes) "
                    + "os=\(ProcessInfo.processInfo.operatingSystemVersionString)")
            XCTAssertLessThanOrEqual(
                measurement.peakResidentGrowthBytes, memoryCeilingBytes)
        #else
            throw XCTSkip("the real-file digest benchmark currently samples macOS memory")
        #endif
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("minirun-file-digest-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func makeSparseFile(_ url: URL, size: UInt64) throws {
        let descriptor: Int32 = try url.withUnsafeFileSystemRepresentation { path in
            guard let path else { throw CocoaError(.fileWriteInvalidFileName) }
            let value = open(path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0o600)
            guard value >= 0 else { throw currentPOSIXError() }
            return value
        }
        defer { _ = close(descriptor) }
        guard size <= UInt64(off_t.max), ftruncate(descriptor, off_t(size)) == 0 else {
            throw currentPOSIXError()
        }
    }

    #if os(macOS)
        private struct DigestMeasurement {
            let digest: String
            let elapsed: TimeInterval
            let peakResidentGrowthBytes: UInt64
        }

        private final class DigestResultBox: @unchecked Sendable {
            private let lock = NSLock()
            private var value: Result<String, Error>?

            func store(_ result: Result<String, Error>) {
                lock.lock()
                value = result
                lock.unlock()
            }

            func load() -> Result<String, Error>? {
                lock.lock()
                defer { lock.unlock() }
                return value
            }
        }

        private func measureDigest(_ file: URL) throws -> DigestMeasurement {
            let baseline = try physicalFootprintBytes()
            let started = ContinuousClock.now
            let finished = DispatchSemaphore(value: 0)
            let result = DigestResultBox()
            let thread = Thread {
                result.store(Result { try FileDigestComputer.sha256(ofFileAt: file) })
                finished.signal()
            }
            thread.name = "minirun.file-digest-memory-test"
            thread.start()

            var peak = baseline
            while finished.wait(timeout: .now() + .milliseconds(2)) == .timedOut {
                peak = max(peak, try physicalFootprintBytes())
            }
            peak = max(peak, try physicalFootprintBytes())
            let elapsed = ContinuousClock.now - started
            guard let outcome = result.load() else {
                XCTFail("digest thread finished without publishing a result")
                throw CocoaError(.fileReadUnknown)
            }
            return DigestMeasurement(
                digest: try outcome.get(),
                elapsed: Double(elapsed.components.seconds)
                    + Double(elapsed.components.attoseconds) / 1e18,
                peakResidentGrowthBytes: peak > baseline ? peak - baseline : 0)
        }

        private func physicalFootprintBytes() throws -> UInt64 {
            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(
                MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
            let status = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(
                    to: integer_t.self, capacity: Int(count)
                ) { rebound in
                    task_info(
                        mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
                }
            }
            guard status == KERN_SUCCESS else {
                throw NSError(
                    domain: NSMachErrorDomain, code: Int(status),
                    userInfo: [NSLocalizedDescriptionKey: "task_info(TASK_VM_INFO) failed"])
            }
            return UInt64(info.phys_footprint)
        }
    #endif

    private func currentPOSIXError() -> POSIXError {
        POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
}
