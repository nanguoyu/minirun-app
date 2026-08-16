import XCTest

@testable import StorageCore

/// The configuration-time deadlock check.
final class PagedReadPlanTests: XCTestCase {

    func testAcceptsPlansThatSatisfyTheInvariant() throws {
        let plan = try PagedReadPlan(slotCount: 8, readAhead: 3, evalEveryK: 4)
        XCTAssertEqual(plan.slotCount, 8)
        XCTAssertEqual(plan.maximumReadAhead, 4)
        // Exactly at the boundary is legal.
        _ = try PagedReadPlan(slotCount: 8, readAhead: 4, evalEveryK: 4)
        _ = try PagedReadPlan(slotCount: 2, readAhead: 1, evalEveryK: 1)
    }

    /// The check throws instead of clamping. A clamp would change either the
    /// evaluation cadence or the memory budget without the caller knowing, and
    /// the budget is the one property this milestone exists to bound.
    func testRejectsPlansThatCouldDeadlock() {
        XCTAssertThrowsError(try PagedReadPlan(slotCount: 4, readAhead: 2, evalEveryK: 3)) {
            error in
            guard case PagerError.configuration(let detail) = error else {
                return XCTFail("expected configuration error, got \(error)")
            }
            XCTAssertTrue(detail.contains("evalEveryK"), detail)
            XCTAssertTrue(detail.contains("Raise slotCount to at least 5"), detail)
        }
        XCTAssertThrowsError(try PagedReadPlan(slotCount: 1, readAhead: 1, evalEveryK: 1))
    }

    func testRejectsNonsenseSizes() {
        XCTAssertThrowsError(try PagedReadPlan(slotCount: 0, readAhead: 1, evalEveryK: 1))
        XCTAssertThrowsError(try PagedReadPlan(slotCount: 4, readAhead: 0, evalEveryK: 1))
        XCTAssertThrowsError(try PagedReadPlan(slotCount: 4, readAhead: 1, evalEveryK: 0))
    }
}
