import Foundation
import XCTest

@testable import StorageCore

final class SplitMix64Tests: XCTestCase {
    /// The whole verification scheme rests on this: the closed form must equal
    /// the iterated stream, or `bench --verify` would be checking against
    /// content the generator never wrote.
    func testClosedFormMatchesIteratedStream() {
        for seed: UInt64 in [0, 1, 42, 0xDEAD_BEEF, UInt64.max] {
            var stream = SplitMix64(seed: seed)
            for index in 0..<256 {
                XCTAssertEqual(
                    stream.next(),
                    SplitMix64.value(seed: seed, index: UInt64(index)),
                    "seed \(seed), index \(index)")
            }
        }
    }

    func testDistinctSeedsProduceDistinctStreams() {
        let a = (0..<64).map { SplitMix64.value(seed: 42, index: UInt64($0)) }
        let b = (0..<64).map { SplitMix64.value(seed: 43, index: UInt64($0)) }
        XCTAssertNotEqual(a, b)
    }

    func testDerivedStreamsDifferFromBaseSeed() {
        let base = SplitMix64.value(seed: 42, index: 0)
        let derived = SplitMix64.derive(seed: 42, label: 0)
        XCTAssertNotEqual(base, derived)
        XCTAssertNotEqual(
            SplitMix64.derive(seed: 42, label: 0), SplitMix64.derive(seed: 42, label: 1))
    }

    /// A weak but useful smoke test: a broken mixer usually shows up as a
    /// badly skewed bit balance.
    func testOutputBitsAreRoughlyBalanced() {
        var ones = 0
        let samples = 4096
        for index in 0..<samples {
            ones += SplitMix64.value(seed: 7, index: UInt64(index)).nonzeroBitCount
        }
        let fraction = Double(ones) / Double(samples * 64)
        XCTAssertEqual(fraction, 0.5, accuracy: 0.02)
    }
}

final class DeterministicContentTests: XCTestCase {
    func testSameSeedProducesSameBytes() {
        let a = DeterministicContent.bytes(seed: 42, fileOffset: 0, count: 1024)
        let b = DeterministicContent.bytes(seed: 42, fileOffset: 0, count: 1024)
        XCTAssertEqual(a, b)
    }

    func testDifferentSeedProducesDifferentBytes() {
        let a = DeterministicContent.bytes(seed: 42, fileOffset: 0, count: 1024)
        let b = DeterministicContent.bytes(seed: 43, fileOffset: 0, count: 1024)
        XCTAssertNotEqual(a, b)
        // Not merely different: overwhelmingly different.
        let equalBytes = zip(a, b).filter(==).count
        XCTAssertLessThan(equalBytes, 40, "expected roughly 1/256 of bytes to coincide")
    }

    /// Content at an arbitrary offset must be recomputable without generating
    /// what precedes it. This is what lets a random-read benchmark verify block
    /// 90,000 without materializing the first 89,999.
    func testArbitraryOffsetMatchesFullGeneration() {
        let whole = DeterministicContent.bytes(seed: 99, fileOffset: 0, count: 8192)
        for offset in [0, 1, 7, 8, 9, 63, 64, 1000, 4095, 4096, 8191] {
            for count in [1, 3, 8, 16, 100] where offset + count <= whole.count {
                let slice = DeterministicContent.bytes(
                    seed: 99, fileOffset: UInt64(offset), count: count)
                XCTAssertEqual(
                    slice, Array(whole[offset..<(offset + count)]),
                    "offset \(offset), count \(count)")
            }
        }
    }

    func testUnalignedRangesSpanLaneBoundaries() {
        let whole = DeterministicContent.bytes(seed: 5, fileOffset: 0, count: 64)
        // A range starting mid-lane and ending mid-lane, crossing several lanes.
        let slice = DeterministicContent.bytes(seed: 5, fileOffset: 3, count: 19)
        XCTAssertEqual(slice, Array(whole[3..<22]))
    }

    func testGenerationAtLargeOffsetsIsStable() {
        // Well past 2^32 bytes, to catch 32-bit truncation in lane indexing.
        let offset: UInt64 = 8_589_934_592 + 17
        let a = DeterministicContent.bytes(seed: 3, fileOffset: offset, count: 64)
        let b = DeterministicContent.bytes(seed: 3, fileOffset: offset, count: 64)
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, DeterministicContent.bytes(seed: 3, fileOffset: 0, count: 64))
    }

    func testEmptyRangeIsHandled() {
        XCTAssertEqual(DeterministicContent.bytes(seed: 1, fileOffset: 0, count: 0), [])
    }

    func testVerifyAcceptsCorrectContent() {
        for offset in [UInt64(0), 1, 7, 4096, 1_048_583] {
            let bytes = DeterministicContent.bytes(seed: 11, fileOffset: offset, count: 517)
            let result = bytes.withUnsafeBytes {
                DeterministicContent.verify($0, seed: 11, fileOffset: offset)
            }
            XCTAssertTrue(result.isClean, "offset \(offset)")
            XCTAssertEqual(result.comparedBytes, 517)
            XCTAssertNil(result.firstMismatchOffset)
        }
    }

    func testVerifyDetectsCorruptedBytes() {
        var bytes = DeterministicContent.bytes(seed: 11, fileOffset: 4096, count: 256)
        bytes[10] ^= 0xFF
        bytes[200] ^= 0x01
        let result = bytes.withUnsafeBytes {
            DeterministicContent.verify($0, seed: 11, fileOffset: 4096)
        }
        XCTAssertFalse(result.isClean)
        XCTAssertEqual(result.mismatchedBytes, 2)
        XCTAssertEqual(result.firstMismatchOffset, 4096 + 10)
    }

    func testVerifyDetectsWrongSeed() {
        let bytes = DeterministicContent.bytes(seed: 11, fileOffset: 0, count: 4096)
        let result = bytes.withUnsafeBytes {
            DeterministicContent.verify($0, seed: 12, fileOffset: 0)
        }
        XCTAssertGreaterThan(result.mismatchedBytes, 4000)
    }

    /// Reading the right bytes from the wrong place must not verify clean —
    /// otherwise a benchmark with a broken offset scheduler would look correct.
    func testVerifyDetectsWrongOffset() {
        let bytes = DeterministicContent.bytes(seed: 11, fileOffset: 8192, count: 4096)
        let result = bytes.withUnsafeBytes {
            DeterministicContent.verify($0, seed: 11, fileOffset: 12288)
        }
        XCTAssertGreaterThan(result.mismatchedBytes, 4000)
    }

    func testVerifyHandlesUnalignedEdges() {
        var bytes = DeterministicContent.bytes(seed: 11, fileOffset: 3, count: 13)
        let clean = bytes.withUnsafeBytes {
            DeterministicContent.verify($0, seed: 11, fileOffset: 3)
        }
        XCTAssertTrue(clean.isClean)

        bytes[0] ^= 0xFF
        bytes[12] ^= 0xFF
        let dirty = bytes.withUnsafeBytes {
            DeterministicContent.verify($0, seed: 11, fileOffset: 3)
        }
        XCTAssertEqual(dirty.mismatchedBytes, 2)
        XCTAssertEqual(dirty.firstMismatchOffset, 3)
    }

    func testVerificationMergeAccumulates() {
        var total = DeterministicContent.Verification.clean
        total.merge(
            .init(comparedBytes: 100, mismatchedBytes: 2, firstMismatchOffset: 500))
        total.merge(
            .init(comparedBytes: 100, mismatchedBytes: 1, firstMismatchOffset: 20))
        XCTAssertEqual(total.comparedBytes, 200)
        XCTAssertEqual(total.mismatchedBytes, 3)
        XCTAssertEqual(total.firstMismatchOffset, 20, "earliest mismatch wins")
    }
}
