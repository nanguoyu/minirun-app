import Foundation
import XCTest

@testable import StorageCore

final class FileGeneratorTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-generator-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func path(_ name: String) -> String {
        directory.appendingPathComponent(name).path
    }

    func testWritesRequestedSize() throws {
        let target = path("size.bin")
        let result = try FileGenerator.generate(
            FileGenerationRequest(path: target, sizeBytes: 12_345, seed: 1))
        XCTAssertEqual(result.sizeBytes, 12_345)

        let data = try Data(contentsOf: URL(fileURLWithPath: target))
        XCTAssertEqual(data.count, 12_345)
    }

    func testSameSeedAndSizeProduceIdenticalFiles() throws {
        let first = path("a.bin")
        let second = path("b.bin")
        _ = try FileGenerator.generate(
            FileGenerationRequest(path: first, sizeBytes: 200_000, seed: 42))
        _ = try FileGenerator.generate(
            FileGenerationRequest(path: second, sizeBytes: 200_000, seed: 42))

        XCTAssertEqual(
            try Data(contentsOf: URL(fileURLWithPath: first)),
            try Data(contentsOf: URL(fileURLWithPath: second)))
    }

    func testDifferentSeedsProduceDifferentFiles() throws {
        let first = path("s42.bin")
        let second = path("s43.bin")
        _ = try FileGenerator.generate(
            FileGenerationRequest(path: first, sizeBytes: 100_000, seed: 42))
        _ = try FileGenerator.generate(
            FileGenerationRequest(path: second, sizeBytes: 100_000, seed: 43))

        XCTAssertNotEqual(
            try Data(contentsOf: URL(fileURLWithPath: first)),
            try Data(contentsOf: URL(fileURLWithPath: second)))
    }

    /// Chunking is a throughput knob. If it changed the bytes, a file
    /// generated on one machine could not be verified on another.
    func testChunkSizeDoesNotAffectContent() throws {
        var previous: Data?
        for chunk in [1, 7, 4096, 65_536, 1 << 20] {
            let target = path("chunk-\(chunk).bin")
            _ = try FileGenerator.generate(
                FileGenerationRequest(
                    path: target, sizeBytes: 70_001, seed: 7, chunkBytes: chunk))
            let data = try Data(contentsOf: URL(fileURLWithPath: target))
            XCTAssertEqual(data.count, 70_001)
            if let previous { XCTAssertEqual(data, previous, "chunk \(chunk) differs") }
            previous = data
        }
    }

    /// The generated file must match what `verify` expects at every offset —
    /// this is the link between the generator and `bench --verify`.
    func testFileContentMatchesRecomputedExpectation() throws {
        let target = path("verify.bin")
        let size: UInt64 = 300_000
        _ = try FileGenerator.generate(
            FileGenerationRequest(path: target, sizeBytes: size, seed: 2024, chunkBytes: 8192))
        let data = try Data(contentsOf: URL(fileURLWithPath: target))

        for offset: UInt64 in [0, 1, 8, 4095, 8192, 100_003, 299_000] {
            let count = Int(min(1024, size - offset))
            let observed = data.subdata(in: Int(offset)..<(Int(offset) + count))
            let result = observed.withUnsafeBytes {
                DeterministicContent.verify($0, seed: 2024, fileOffset: offset)
            }
            XCTAssertTrue(result.isClean, "offset \(offset): \(result)")
        }
    }

    func testChunkLargerThanFileIsClamped() throws {
        let target = path("small.bin")
        let result = try FileGenerator.generate(
            FileGenerationRequest(
                path: target, sizeBytes: 100, seed: 5, chunkBytes: 1 << 20))
        XCTAssertEqual(result.sizeBytes, 100)
        XCTAssertEqual(result.chunkBytes, 100)
    }

    func testOverwritesExistingFile() throws {
        let target = path("overwrite.bin")
        try Data(repeating: 0xFF, count: 500_000).write(to: URL(fileURLWithPath: target))
        _ = try FileGenerator.generate(
            FileGenerationRequest(path: target, sizeBytes: 1_000, seed: 1))
        let data = try Data(contentsOf: URL(fileURLWithPath: target))
        XCTAssertEqual(data.count, 1_000, "truncation must drop the old tail")
    }

    func testValidationRejectsBadRequests() {
        XCTAssertThrowsError(
            try FileGenerationRequest(path: "", sizeBytes: 100, seed: 1).validate())
        XCTAssertThrowsError(
            try FileGenerationRequest(path: "/tmp/x", sizeBytes: 0, seed: 1).validate())
        XCTAssertThrowsError(
            try FileGenerationRequest(
                path: "/tmp/x", sizeBytes: 100, seed: 1, chunkBytes: 0
            ).validate())
        XCTAssertThrowsError(
            try FileGenerationRequest(
                path: "/tmp/x", sizeBytes: 100, seed: 1, chunkBytes: (1 << 30) + 1
            ).validate())
        XCTAssertNoThrow(
            try FileGenerationRequest(path: "/tmp/x", sizeBytes: 100, seed: 1).validate())
    }

    func testUnwritablePathFailsExplicitly() {
        XCTAssertThrowsError(
            try FileGenerator.generate(
                FileGenerationRequest(
                    path: "/minirun-should-not-be-writable/x.bin", sizeBytes: 100, seed: 1))
        ) { error in
            guard case StorageCoreError.posix(let operation, _, _) = error else {
                return XCTFail("expected a posix error, got \(error)")
            }
            XCTAssertEqual(operation, "open")
        }
    }

    func testProgressReachesTotal() throws {
        var last: (UInt64, UInt64)?
        var callbacks = 0
        _ = try FileGenerator.generate(
            FileGenerationRequest(
                path: path("progress.bin"), sizeBytes: 40_000, seed: 1, chunkBytes: 4_096)
        ) { written, total in
            callbacks += 1
            last = (written, total)
        }
        XCTAssertGreaterThan(callbacks, 1)
        XCTAssertEqual(last?.0, 40_000)
        XCTAssertEqual(last?.1, 40_000)
    }
}

final class AlignedBufferTests: XCTestCase {
    func testBufferIsPageAligned() throws {
        let buffer = try AlignedBuffer(count: 4096)
        XCTAssertEqual(buffer.count, 4096)
        XCTAssertEqual(UInt(bitPattern: buffer.pointer) % UInt(AlignedBuffer.pageSize), 0)
    }

    func testNonPowerOfTwoAlignmentIsRoundedUp() throws {
        let buffer = try AlignedBuffer(count: 1024, alignment: 100)
        XCTAssertEqual(buffer.alignment, 128)
        XCTAssertEqual(UInt(bitPattern: buffer.pointer) % 128, 0)
    }

    func testBufferIsWritable() throws {
        let buffer = try AlignedBuffer(count: 64)
        DeterministicContent.fill(buffer.mutableBytes, seed: 1, fileOffset: 0)
        let result = DeterministicContent.verify(buffer.bytes, seed: 1, fileOffset: 0)
        XCTAssertTrue(result.isClean)
    }
}
