import XCTest

@testable import StorageCore

final class ByteSizeTests: XCTestCase {
    func testParsesPlainByteCounts() throws {
        XCTAssertEqual(try ByteSize.parse("0"), 0)
        XCTAssertEqual(try ByteSize.parse("1"), 1)
        XCTAssertEqual(try ByteSize.parse("268435456"), 268_435_456)
        XCTAssertEqual(try ByteSize.parse("1_048_576"), 1_048_576)
    }

    func testParsesBinarySuffixes() throws {
        XCTAssertEqual(try ByteSize.parse("1KiB"), 1024)
        XCTAssertEqual(try ByteSize.parse("256MiB"), 256 * 1024 * 1024)
        XCTAssertEqual(try ByteSize.parse("256M"), 256 * 1024 * 1024)
        XCTAssertEqual(try ByteSize.parse("2GiB"), 2 * 1024 * 1024 * 1024)
        XCTAssertEqual(try ByteSize.parse("2 gib"), 2 * 1024 * 1024 * 1024)
    }

    func testParsesDecimalSuffixes() throws {
        XCTAssertEqual(try ByteSize.parse("1KB"), 1_000)
        XCTAssertEqual(try ByteSize.parse("1MB"), 1_000_000)
        XCTAssertEqual(try ByteSize.parse("1GB"), 1_000_000_000)
    }

    func testBinaryAndDecimalSuffixesDiffer() throws {
        XCTAssertNotEqual(try ByteSize.parse("1MiB"), try ByteSize.parse("1MB"))
    }

    func testRejectsMalformedInput() {
        XCTAssertThrowsError(try ByteSize.parse(""))
        XCTAssertThrowsError(try ByteSize.parse("   "))
        XCTAssertThrowsError(try ByteSize.parse("MiB"))
        XCTAssertThrowsError(try ByteSize.parse("-1"))
        XCTAssertThrowsError(try ByteSize.parse("12quux"))
        XCTAssertThrowsError(try ByteSize.parse("1.5MiB"))
    }

    func testRejectsOverflow() {
        XCTAssertThrowsError(try ByteSize.parse("18446744073709551615TiB"))
    }

    func testFormatsBinarySizes() {
        XCTAssertEqual(ByteSize.format(0), "0 B")
        XCTAssertEqual(ByteSize.format(512), "512 B")
        XCTAssertEqual(ByteSize.format(1024), "1.0 KiB")
        XCTAssertEqual(ByteSize.format(268_435_456), "256.0 MiB")
    }
}
