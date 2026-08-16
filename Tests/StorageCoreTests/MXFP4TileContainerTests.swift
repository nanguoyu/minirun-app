import Darwin
import Foundation
import XCTest

@testable import StorageCore

/// The synthetic container: layout rules, deterministic content, and the
/// truncation rejection the pager depends on.
final class MXFP4TileContainerTests: XCTestCase {

    private var directory: URL!

    /// Smallest geometry the format allows: the binding constraint is
    /// `rows * cols` divisible by `regionAlignment * 32` = 524288, because the
    /// scale sub-region is the smaller of the two and still has to be a whole
    /// number of alignment units. 512*1024 = 524288 exactly, giving packed =
    /// 262144 bytes and scales = 16384.
    private let geometry = TileGeometry(rows: 512, cols: 1024)

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-container-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let directory, FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.removeItem(at: directory)
        }
    }

    private func path(_ name: String = "tiles.bin") -> String {
        directory.appendingPathComponent(name).path
    }

    // MARK: - Layout

    func testLayoutMatchesTheGeometry() throws {
        let layout = try MXFP4TileContainer.layout(geometry: geometry, tileCount: 6, seed: 1)
        XCTAssertEqual(layout.geometry.packedBytes, 512 * 1024 / 2)
        XCTAssertEqual(layout.geometry.scaleBytes, 512 * 1024 / 32)
        XCTAssertEqual(layout.scaleOffsetInTile, layout.geometry.packedBytes)
        XCTAssertEqual(
            layout.tileStride, layout.geometry.packedBytes + layout.geometry.scaleBytes)
        XCTAssertEqual(layout.firstTileOffset, UInt64(MXFP4TileContainer.headerBytes))
        XCTAssertEqual(layout.totalRows, 6 * 512)
        XCTAssertEqual(
            layout.totalBytes,
            UInt64(MXFP4TileContainer.headerBytes) + UInt64(6 * layout.tileStride))
    }

    /// Every offset the pager will hand to `pread`, and every offset it will
    /// adopt as an `MLXArray`, has to be page-aligned or the zero-copy path is
    /// silently lost.
    func testEveryTileAndSubRegionIsPageAligned() throws {
        let layout = try MXFP4TileContainer.layout(geometry: geometry, tileCount: 5, seed: 3)
        // Checked against the format's own constant and against the host page
        // size, because both have to hold: the file must be portable and the
        // adopted pointers must satisfy this machine's Metal requirements.
        let pageSize = UInt64(max(AlignedBuffer.pageSize, MXFP4TileContainer.regionAlignment))
        XCTAssertEqual(layout.firstTileOffset % pageSize, 0)
        XCTAssertEqual(UInt64(layout.tileStride) % pageSize, 0)
        XCTAssertEqual(UInt64(layout.scaleOffsetInTile) % pageSize, 0)
        for tile in 0..<5 {
            XCTAssertEqual(layout.tileOffset(tile) % pageSize, 0, "tile \(tile) is not aligned")
        }
    }

    func testUnsupportedGeometriesAreRejected() {
        // cols not a multiple of the MXFP4 group size.
        XCTAssertThrowsError(
            try MXFP4TileContainer.layout(
                geometry: TileGeometry(rows: 512, cols: 1000), tileCount: 1, seed: 0))
        // Sub-regions not a whole number of alignment units.
        XCTAssertThrowsError(
            try MXFP4TileContainer.layout(
                geometry: TileGeometry(rows: 8, cols: 1024), tileCount: 1, seed: 0))
        XCTAssertThrowsError(
            try MXFP4TileContainer.layout(
                geometry: TileGeometry(rows: 256, cols: 1024), tileCount: 1, seed: 0))
        XCTAssertThrowsError(
            try MXFP4TileContainer.layout(
                geometry: TileGeometry(rows: 0, cols: 1024), tileCount: 1, seed: 0))
        XCTAssertThrowsError(
            try MXFP4TileContainer.layout(geometry: geometry, tileCount: 0, seed: 0))
    }

    // MARK: - Content

    func testContentIsDeterministicAndSeedDependent() throws {
        let layout = try MXFP4TileContainer.layout(geometry: geometry, tileCount: 2, seed: 42)
        var first = [UInt8](repeating: 0, count: layout.geometry.packedBytes)
        var again = [UInt8](repeating: 0, count: layout.geometry.packedBytes)
        first.withUnsafeMutableBytes { MXFP4TileContainer.fillPacked($0, layout: layout, tile: 1) }
        again.withUnsafeMutableBytes { MXFP4TileContainer.fillPacked($0, layout: layout, tile: 1) }
        XCTAssertEqual(first, again)

        var otherTile = [UInt8](repeating: 0, count: layout.geometry.packedBytes)
        otherTile.withUnsafeMutableBytes {
            MXFP4TileContainer.fillPacked($0, layout: layout, tile: 0)
        }
        XCTAssertNotEqual(first, otherTile, "tiles must not share content")

        let otherSeed = try MXFP4TileContainer.layout(geometry: geometry, tileCount: 2, seed: 43)
        var otherSeedBytes = [UInt8](repeating: 0, count: layout.geometry.packedBytes)
        otherSeedBytes.withUnsafeMutableBytes {
            MXFP4TileContainer.fillPacked($0, layout: otherSeed, tile: 1)
        }
        XCTAssertNotEqual(first, otherSeedBytes, "a different seed must give different bytes")
    }

    /// Scale bytes stay inside the window that keeps every product finite and
    /// normal. Byte 255 is the e8m0 NaN encoding; the extremes are subnormal or
    /// overflowing, and either would break a bit-identity comparison for
    /// reasons unrelated to the pager.
    func testScaleBytesStayInTheSafeWindow() throws {
        let layout = try MXFP4TileContainer.layout(geometry: geometry, tileCount: 4, seed: 7)
        var histogram = Set<UInt8>()
        for tile in 0..<4 {
            var scales = [UInt8](repeating: 0, count: layout.geometry.scaleBytes)
            scales.withUnsafeMutableBytes {
                MXFP4TileContainer.fillScales($0, layout: layout, tile: tile)
            }
            for byte in scales {
                XCTAssertGreaterThanOrEqual(byte, 100)
                XCTAssertLessThanOrEqual(byte, 140)
                histogram.insert(byte)
            }
        }
        XCTAssertEqual(histogram.count, 41, "the whole [100, 140] window should be exercised")
    }

    // MARK: - Round trip

    func testRoundTripThroughTheReader() throws {
        let written = try MXFP4TileContainer.write(
            path: path(), geometry: geometry, tileCount: 7, seed: 99, chunkTiles: 3)
        let read = try MXFP4TileContainer.open(path: path())
        XCTAssertEqual(read, written)

        let size = try FileManager.default.attributesOfItem(atPath: path())[.size] as? Int
        XCTAssertEqual(UInt64(size ?? 0), written.totalBytes)

        // Every tile's bytes must match what the generator says they are.
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path()))
        defer { try? handle.close() }
        for tile in 0..<7 {
            try handle.seek(toOffset: written.tileOffset(tile))
            let data = try XCTUnwrap(handle.read(upToCount: written.tileStride))
            XCTAssertEqual(data.count, written.tileStride)
            let mismatch = try data.withUnsafeBytes {
                try MXFP4TileContainer.verifyTile($0, layout: written, tile: tile)
            }
            XCTAssertNil(mismatch)
        }
    }

    func testDescriptorOpenReadsTheAuthorizedObjectAfterPathReplacement() throws {
        let written = try MXFP4TileContainer.write(
            path: path(), geometry: geometry, tileCount: 2, seed: 99)
        let descriptor = Darwin.open(path(), O_RDONLY | O_CLOEXEC)
        XCTAssertGreaterThanOrEqual(descriptor, 0)
        defer { if descriptor >= 0 { _ = Darwin.close(descriptor) } }

        try Data(repeating: 0xEE, count: 128).write(
            to: URL(fileURLWithPath: path()), options: .atomic)

        XCTAssertEqual(
            try MXFP4TileContainer.open(fileDescriptor: descriptor, path: "authorized-container"),
            written)
        XCTAssertThrowsError(try MXFP4TileContainer.open(path: path()))
    }

    func testChunkSizeDoesNotChangeTheBytes() throws {
        try MXFP4TileContainer.write(
            path: path("a.bin"), geometry: geometry, tileCount: 5, seed: 5, chunkTiles: 1)
        try MXFP4TileContainer.write(
            path: path("b.bin"), geometry: geometry, tileCount: 5, seed: 5, chunkTiles: 4)
        let a = try Data(contentsOf: URL(fileURLWithPath: path("a.bin")))
        let b = try Data(contentsOf: URL(fileURLWithPath: path("b.bin")))
        XCTAssertEqual(a, b)
    }

    func testRegionDescriptorsCoverTheWholeFile() throws {
        let layout = try MXFP4TileContainer.write(
            path: path(), geometry: geometry, tileCount: 4, seed: 1)
        var covered: UInt64 = layout.firstTileOffset
        for tile in 0..<4 {
            let region = layout.region(tile)
            XCTAssertEqual(region.offset, covered)
            XCTAssertEqual(region.length, layout.tileStride)
            XCTAssertEqual(region.identity.index, UInt64(tile))
            covered += UInt64(region.length)
        }
        XCTAssertEqual(covered, layout.totalBytes)
    }

    // MARK: - Explicit failure

    func testTruncationIsRejected() throws {
        let layout = try MXFP4TileContainer.write(
            path: path(), geometry: geometry, tileCount: 6, seed: 11)
        let handle = try FileHandle(forWritingAtPath: path())
        try XCTUnwrap(handle).truncate(atOffset: layout.totalBytes - 1)
        try XCTUnwrap(handle).close()

        XCTAssertThrowsError(try MXFP4TileContainer.open(path: path())) { error in
            guard case TileContainerError.truncated(_, let expected, let actual) = error else {
                return XCTFail("expected truncated, got \(error)")
            }
            XCTAssertEqual(expected, layout.totalBytes)
            XCTAssertEqual(actual, layout.totalBytes - 1)
        }
    }

    func testHeaderOnlyFileIsRejected() throws {
        let layout = try MXFP4TileContainer.write(
            path: path(), geometry: geometry, tileCount: 6, seed: 11)
        let handle = try XCTUnwrap(FileHandle(forWritingAtPath: path()))
        try handle.truncate(atOffset: UInt64(MXFP4TileContainer.headerBytes))
        try handle.close()
        XCTAssertThrowsError(try MXFP4TileContainer.open(path: path())) { error in
            guard case TileContainerError.truncated(_, let expected, _) = error else {
                return XCTFail("expected truncated, got \(error)")
            }
            XCTAssertEqual(expected, layout.totalBytes)
        }
    }

    func testFileShorterThanTheHeaderIsRejected() throws {
        try Data([0, 1, 2, 3]).write(to: URL(fileURLWithPath: path()))
        XCTAssertThrowsError(try MXFP4TileContainer.open(path: path())) { error in
            guard case TileContainerError.truncated = error else {
                return XCTFail("expected truncated, got \(error)")
            }
        }
    }

    func testBadMagicIsRejected() throws {
        try MXFP4TileContainer.write(path: path(), geometry: geometry, tileCount: 2, seed: 1)
        let handle = try XCTUnwrap(FileHandle(forUpdatingAtPath: path()))
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: Data("NOTATILE".utf8))
        try handle.close()
        XCTAssertThrowsError(try MXFP4TileContainer.open(path: path())) { error in
            guard case TileContainerError.badMagic(let found) = error else {
                return XCTFail("expected badMagic, got \(error)")
            }
            XCTAssertEqual(found, "NOTATILE")
        }
    }

    func testUnsupportedVersionIsRejected() throws {
        try MXFP4TileContainer.write(path: path(), geometry: geometry, tileCount: 2, seed: 1)
        let handle = try XCTUnwrap(FileHandle(forUpdatingAtPath: path()))
        try handle.seek(toOffset: 8)
        try handle.write(contentsOf: Data([9, 0, 0, 0]))
        try handle.close()
        XCTAssertThrowsError(try MXFP4TileContainer.open(path: path())) { error in
            guard case TileContainerError.unsupportedVersion(let version) = error else {
                return XCTFail("expected unsupportedVersion, got \(error)")
            }
            XCTAssertEqual(version, 9)
        }
    }

    /// A header whose derived fields contradict its geometry must not be
    /// trusted to steer reads.
    func testContradictoryHeaderIsRejected() throws {
        try MXFP4TileContainer.write(path: path(), geometry: geometry, tileCount: 2, seed: 1)
        let handle = try XCTUnwrap(FileHandle(forUpdatingAtPath: path()))
        // tileStride lives at offset 8 + 4*6 + 8*3 = 56.
        try handle.seek(toOffset: 56)
        try handle.write(contentsOf: Data([0xFF, 0xFF, 0, 0, 0, 0, 0, 0]))
        try handle.close()
        XCTAssertThrowsError(try MXFP4TileContainer.open(path: path())) { error in
            guard case TileContainerError.malformedHeader = error else {
                return XCTFail("expected malformedHeader, got \(error)")
            }
        }
    }

    func testMissingFileIsRejected() {
        XCTAssertThrowsError(try MXFP4TileContainer.open(path: path("absent.bin"))) { error in
            guard case StorageCoreError.posix(let operation, _, _) = error else {
                return XCTFail("expected posix error, got \(error)")
            }
            XCTAssertEqual(operation, "open")
        }
    }
}
