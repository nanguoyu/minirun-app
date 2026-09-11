import Foundation
import XCTest

@testable import StorageCore

/// The `.engrampage` reader, read against a fixture the *converter* wrote.
///
/// `Tools/v41_flash/make_engrampage_fixture.py` produces the two part files and
/// the JSON beside them through `engram_container.encode_header`,
/// `pack_pages` and `read_row` — the same three functions the published 517 GB
/// artifact went through. So the claim these tests make is not "Swift agrees
/// with a description of the format", it is "Swift agrees with the bytes Python
/// wrote", which is the only version of that claim that can fail usefully.
final class EngramRowPageContainerTests: XCTestCase {
    private struct Fixture: Decodable {
        struct Part: Decodable {
            let file: String
            let partIndex: Int
            let rowBase: UInt64
            let rows: UInt64
            let pages: UInt64
            let bytes: UInt64
            let pageBytes: Int
            let scaleOffsetInPage: Int
            let padBytesPerPage: Int
            let firstPageOffset: UInt64
            let scaleGroupSize: Int

            enum CodingKeys: String, CodingKey {
                case file, rows, pages, bytes
                case partIndex = "part_index"
                case rowBase = "row_base"
                case pageBytes = "page_bytes"
                case scaleOffsetInPage = "scale_offset_in_page"
                case padBytesPerPage = "pad_bytes_per_page"
                case firstPageOffset = "first_page_offset"
                case scaleGroupSize = "scale_group_size"
            }
        }

        struct Row: Decodable {
            let row: UInt64
            let part: Int
            let pageOffset: UInt64
            let valueOffsetInPage: Int
            let scaleOffsetInPage: Int
            let values: String
            let scales: String

            enum CodingKeys: String, CodingKey {
                case row, part, values, scales
                case pageOffset = "page_offset"
                case valueOffsetInPage = "value_offset_in_page"
                case scaleOffsetInPage = "scale_offset_in_page"
            }
        }

        let tableRows: UInt64
        let rowValueBytes: Int
        let rowScaleBytes: Int
        let rowsPerPage: Int
        let parts: [Part]
        let rows: [Row]

        enum CodingKeys: String, CodingKey {
            case parts, rows
            case tableRows = "table_rows"
            case rowValueBytes = "row_value_bytes"
            case rowScaleBytes = "row_scale_bytes"
            case rowsPerPage = "rows_per_page"
        }
    }

    private var fixtureDirectory: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("../Fixtures/deepseek-v41/engram")
            .standardizedFileURL
    }

    private func loadFixture() throws -> Fixture {
        try JSONDecoder().decode(
            Fixture.self,
            from: Data(contentsOf: fixtureDirectory
                .appendingPathComponent("engram-fixture.json")))
    }

    private func path(_ part: Fixture.Part) -> String {
        fixtureDirectory.appendingPathComponent(part.file).path
    }

    func testEveryHeaderFieldMatchesWhatTheConverterWrote() throws {
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.parts.count, 2)
        for part in fixture.parts {
            let layout = try EngramRowPageContainer.open(path: path(part))
            XCTAssertEqual(layout.tableRows, fixture.tableRows)
            XCTAssertEqual(layout.rowBase, part.rowBase)
            XCTAssertEqual(layout.partRows, part.rows)
            XCTAssertEqual(layout.pageCount, part.pages)
            XCTAssertEqual(layout.totalBytes, part.bytes)
            XCTAssertEqual(layout.partIndex, part.partIndex)
            XCTAssertEqual(layout.pageBytes, part.pageBytes)
            XCTAssertEqual(layout.rowsPerPage, fixture.rowsPerPage)
            XCTAssertEqual(layout.rowValueBytes, fixture.rowValueBytes)
            XCTAssertEqual(layout.rowScaleBytes, fixture.rowScaleBytes)
            XCTAssertEqual(layout.scaleOffsetInPage, part.scaleOffsetInPage)
            XCTAssertEqual(layout.padBytesPerPage, part.padBytesPerPage)
            XCTAssertEqual(layout.firstPageOffset, part.firstPageOffset)
            XCTAssertEqual(layout.scaleGroupSize, part.scaleGroupSize)
            XCTAssertEqual(layout.valueDType, .fp8E4M3)
            XCTAssertEqual(layout.scaleDType, .uint8E8M0)
        }
    }

    func testTheTwoPartsPartitionTheTableExactlyOnce() throws {
        let fixture = try loadFixture()
        var expected: UInt64 = 0
        for part in fixture.parts.sorted(by: { $0.partIndex < $1.partIndex }) {
            let layout = try EngramRowPageContainer.open(path: path(part))
            XCTAssertEqual(layout.rowBase, expected)
            expected = layout.rowEnd
        }
        XCTAssertEqual(expected, fixture.tableRows)
    }

    func testEveryRowIsTheBytesPythonWroteAtThePythonOffsets() throws {
        let fixture = try loadFixture()
        var layouts = [Int: EngramRowPageContainer.Layout]()
        var descriptors = [Int: Int32]()
        for part in fixture.parts {
            let reference = path(part)
            let descriptor = open(reference, O_RDONLY)
            XCTAssertGreaterThanOrEqual(descriptor, 0)
            descriptors[part.partIndex] = descriptor
            layouts[part.partIndex] = try EngramRowPageContainer.open(
                fileDescriptor: descriptor, path: reference)
        }
        defer { for descriptor in descriptors.values { close(descriptor) } }

        XCTAssertEqual(UInt64(fixture.rows.count), fixture.tableRows)
        for expected in fixture.rows {
            let layout = try XCTUnwrap(layouts[expected.part])
            let descriptor = try XCTUnwrap(descriptors[expected.part])
            let location = try layout.locate(row: expected.row)
            XCTAssertEqual(location.pageOffset, expected.pageOffset)
            XCTAssertEqual(location.valueOffsetInPage, expected.valueOffsetInPage)
            XCTAssertEqual(location.scaleOffsetInPage, expected.scaleOffsetInPage)
            // The whole point of the layout: the page is one aligned read.
            XCTAssertEqual(location.pageOffset % 4096, 0)

            let row = try EngramRowPageContainer.readRow(
                fileDescriptor: descriptor, path: path(fixture.parts[expected.part]),
                layout: layout, row: expected.row)
            XCTAssertEqual(row.row, expected.row)
            XCTAssertEqual(row.values, Self.bytes(expected.values))
            XCTAssertEqual(row.scales, Self.bytes(expected.scales))
            XCTAssertEqual(row.values.count, fixture.rowValueBytes)
            XCTAssertEqual(row.scales.count, fixture.rowScaleBytes)
        }
    }

    func testABatchOfTwentyFourRowsAgreesWithTwentyFourSingleReads() throws {
        let fixture = try loadFixture()
        let part = fixture.parts[0]
        let reference = path(part)
        let descriptor = open(reference, O_RDONLY)
        XCTAssertGreaterThanOrEqual(descriptor, 0)
        defer { close(descriptor) }
        let layout = try EngramRowPageContainer.open(
            fileDescriptor: descriptor, path: reference)

        // Twenty-four rows is what one Engram module reads per token. These are
        // deliberately unsorted, deliberately repeat one row, and deliberately
        // straddle several pages.
        let rows: [UInt64] = [
            41, 0, 14, 15, 29, 30, 7, 44, 3, 59, 58, 16,
            1, 22, 37, 8, 45, 14, 52, 11, 26, 33, 47, 2,
        ]
        let batch = try EngramRowPageContainer.readRows(
            fileDescriptor: descriptor, path: reference, layout: layout, rows: rows)
        XCTAssertEqual(batch.count, rows.count)
        XCTAssertEqual(batch.map(\.row), rows)
        for (index, row) in rows.enumerated() {
            let single = try EngramRowPageContainer.readRow(
                fileDescriptor: descriptor, path: reference, layout: layout, row: row)
            XCTAssertEqual(batch[index], single)
        }

        // The batch touches fewer pages than it has rows, and every one of them
        // is aligned — which is the property a prefetcher would schedule on.
        let offsets = try EngramRowPageContainer.pageOffsets(layout: layout, rows: rows)
        XCTAssertEqual(offsets, offsets.sorted())
        XCTAssertEqual(Set(offsets).count, offsets.count)
        XCTAssertLessThan(offsets.count, rows.count)
        XCTAssertTrue(offsets.allSatisfy { $0 % 4096 == 0 })
    }

    func testTheLastPageOfTheLastPartIsPartialAndStillReadable() throws {
        let fixture = try loadFixture()
        let last = try XCTUnwrap(fixture.parts.last)
        let layout = try EngramRowPageContainer.open(path: path(last))
        // 97 rows at 15 a page is 6 full pages and a 7-row remainder, so the
        // final page is short and zero-filled past its last row.
        XCTAssertNotEqual(layout.partRows % UInt64(layout.rowsPerPage), 0)
        let finalRow = layout.rowEnd - 1
        let location = try layout.locate(row: finalRow)
        XCTAssertEqual(location.page, layout.pageCount - 1)
        let descriptor = open(path(last), O_RDONLY)
        XCTAssertGreaterThanOrEqual(descriptor, 0)
        defer { close(descriptor) }
        let row = try EngramRowPageContainer.readRow(
            fileDescriptor: descriptor, path: path(last), layout: layout, row: finalRow)
        let expected = try XCTUnwrap(fixture.rows.first { $0.row == finalRow })
        XCTAssertEqual(row.values, Self.bytes(expected.values))
        XCTAssertEqual(row.scales, Self.bytes(expected.scales))
    }

    func testARowOutsideThisPartIsRefusedRatherThanWrapped() throws {
        let fixture = try loadFixture()
        let first = fixture.parts[0]
        let layout = try EngramRowPageContainer.open(path: path(first))
        XCTAssertThrowsError(try layout.locate(row: layout.rowEnd)) { error in
            guard case EngramContainerError.rowOutOfRange = error else {
                return XCTFail("expected rowOutOfRange, got \(error)")
            }
        }
        XCTAssertThrowsError(try layout.locate(row: fixture.tableRows + 1))
        XCTAssertFalse(layout.contains(row: layout.rowEnd))
        XCTAssertTrue(layout.contains(row: layout.rowBase))
    }

    // MARK: - Headers that contradict themselves

    private func header(of part: Fixture.Part) throws -> [UInt8] {
        let data = try Data(contentsOf: fixtureDirectory.appendingPathComponent(part.file))
        return Array(data.prefix(EngramRowPageContainer.headerBytes))
    }

    private func store(_ value: UInt64, at offset: Int, in header: inout [UInt8]) {
        withUnsafeBytes(of: value.littleEndian) { bytes in
            for (index, byte) in bytes.enumerated() { header[offset + index] = byte }
        }
    }

    private func store(_ value: UInt32, at offset: Int, in header: inout [UInt8]) {
        withUnsafeBytes(of: value.littleEndian) { bytes in
            for (index, byte) in bytes.enumerated() { header[offset + index] = byte }
        }
    }

    func testTheUnmutatedHeaderDecodes() throws {
        let fixture = try loadFixture()
        let layout = try EngramRowPageContainer.decodeHeader(header(of: fixture.parts[0]))
        XCTAssertEqual(layout.partIndex, 0)
    }

    func testABadMagicIsRefused() throws {
        let fixture = try loadFixture()
        var bytes = try header(of: fixture.parts[0])
        bytes[0] = UInt8(ascii: "X")
        XCTAssertThrowsError(try EngramRowPageContainer.decodeHeader(bytes)) { error in
            guard case EngramContainerError.badMagic = error else {
                return XCTFail("expected badMagic, got \(error)")
            }
        }
    }

    func testAnUnreadableVersionIsRefused() throws {
        let fixture = try loadFixture()
        var bytes = try header(of: fixture.parts[0])
        store(UInt32(2), at: 8, in: &bytes)
        XCTAssertThrowsError(try EngramRowPageContainer.decodeHeader(bytes)) { error in
            guard case EngramContainerError.unsupportedVersion(2) = error else {
                return XCTFail("expected unsupportedVersion, got \(error)")
            }
        }
    }

    func testAPageCountThatContradictsTheRowCountIsRefused() throws {
        let fixture = try loadFixture()
        var bytes = try header(of: fixture.parts[0])
        store(UInt64(3), at: 56, in: &bytes)
        XCTAssertThrowsError(try EngramRowPageContainer.decodeHeader(bytes)) { error in
            guard case EngramContainerError.malformedHeader = error else {
                return XCTFail("expected malformedHeader, got \(error)")
            }
        }
    }

    func testARowsPerPageThatIsNotTheDensestPackingIsRefused() throws {
        let fixture = try loadFixture()
        var bytes = try header(of: fixture.parts[0])
        store(UInt32(14), at: 52, in: &bytes)
        XCTAssertThrowsError(try EngramRowPageContainer.decodeHeader(bytes)) { error in
            guard case EngramContainerError.geometryUnsupported = error else {
                return XCTFail("expected geometryUnsupported, got \(error)")
            }
        }
    }

    func testAScaleGroupThatDoesNotCoverTheRowIsRefused() throws {
        let fixture = try loadFixture()
        var bytes = try header(of: fixture.parts[0])
        store(UInt32(16), at: 80, in: &bytes)
        XCTAssertThrowsError(try EngramRowPageContainer.decodeHeader(bytes)) { error in
            guard case EngramContainerError.geometryUnsupported = error else {
                return XCTFail("expected geometryUnsupported, got \(error)")
            }
        }
    }

    func testAPartThatRunsPastItsTableIsRefused() throws {
        let fixture = try loadFixture()
        var bytes = try header(of: fixture.parts[0])
        store(UInt64(10), at: 16, in: &bytes)
        XCTAssertThrowsError(try EngramRowPageContainer.decodeHeader(bytes)) { error in
            guard case EngramContainerError.geometryUnsupported = error else {
                return XCTFail("expected geometryUnsupported, got \(error)")
            }
        }
    }

    func testANonZeroReservedByteIsRefused() throws {
        let fixture = try loadFixture()
        var bytes = try header(of: fixture.parts[0])
        bytes[EngramRowPageContainer.headerBytes - 1] = 1
        XCTAssertThrowsError(try EngramRowPageContainer.decodeHeader(bytes)) { error in
            guard case EngramContainerError.malformedHeader = error else {
                return XCTFail("expected malformedHeader, got \(error)")
            }
        }
    }

    func testAFileShorterThanItsHeaderDescribesIsRefused() throws {
        let fixture = try loadFixture()
        let source = try Data(
            contentsOf: fixtureDirectory.appendingPathComponent(fixture.parts[0].file))
        let truncated = FileManager.default.temporaryDirectory
            .appendingPathComponent("engram-truncated-\(UUID().uuidString).engrampage")
        try source.dropLast(4096).write(to: truncated)
        defer { try? FileManager.default.removeItem(at: truncated) }
        XCTAssertThrowsError(
            try EngramRowPageContainer.open(path: truncated.path)
        ) { error in
            guard case EngramContainerError.truncated = error else {
                return XCTFail("expected truncated, got \(error)")
            }
        }
    }

    private static func bytes(_ hex: String) -> [UInt8] {
        var out = [UInt8]()
        out.reserveCapacity(hex.count / 2)
        var index = hex.startIndex
        while index < hex.endIndex {
            let next = hex.index(index, offsetBy: 2)
            out.append(UInt8(hex[index..<next], radix: 16) ?? 0)
            index = next
        }
        return out
    }
}
