import XCTest

@testable import StorageCore

final class UInt64AccountingTests: XCTestCase {
    func testCheckedSumRefusesOverflowInsteadOfWrapping() {
        XCTAssertNil(UInt64Accounting.checkedSum([UInt64.max, 1]))
        XCTAssertEqual(UInt64Accounting.checkedSum([1, 2, 3]), 6)
    }

    func testSaturatingSumKeepsAStickyOverflowBit() {
        var sum = UInt64Accounting.SaturatingSum(value: UInt64.max - 1)
        sum.add(1)
        XCTAssertEqual(sum.value, UInt64.max)
        XCTAssertFalse(sum.didOverflow)

        sum.add(1)
        XCTAssertEqual(sum.value, UInt64.max)
        XCTAssertTrue(sum.didOverflow)

        sum.add(42)
        XCTAssertEqual(sum.value, UInt64.max)
        XCTAssertTrue(sum.didOverflow)
    }
}
