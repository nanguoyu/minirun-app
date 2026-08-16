import XCTest

@testable import MinirunKit

final class DownloadAccountingTests: XCTestCase {
    func testProgressOverflowCannotMasqueradeAsBalanced() {
        let progress = DownloadProgress(
            job: DownloadJobID(), planTotalBytes: UInt64.max,
            verifiedBytes: UInt64.max, fetchedUnverifiedBytes: 1,
            inFlightBytes: 0, remainingBytes: 0,
            filesTotal: 1, filesVerified: 0, filesFailed: 0,
            networkBytes: 0, wastedBytes: 0,
            instantaneousBytesPerSecond: 0, smoothedBytesPerSecond: 0,
            estimatedTimeRemaining: nil, startedAt: Date(), elapsed: 0)

        XCTAssertFalse(progress.accountsForEveryByte)
        XCTAssertEqual(progress.fractionComplete, 0)

        let impossibleEmptyPlan = DownloadProgress(
            job: DownloadJobID(), planTotalBytes: 0,
            verifiedBytes: 1, fetchedUnverifiedBytes: 0,
            inFlightBytes: 0, remainingBytes: 0,
            filesTotal: 1, filesVerified: 0, filesFailed: 0,
            networkBytes: 0, wastedBytes: 0,
            instantaneousBytesPerSecond: 0, smoothedBytesPerSecond: 0,
            estimatedTimeRemaining: nil, startedAt: Date(), elapsed: 0)
        XCTAssertEqual(impossibleEmptyPlan.fractionComplete, 0)
    }

    func testJobCountersSaturateAndRememberOverflow() {
        let counters = JobCounters()
        counters.addNetwork(UInt64.max)
        counters.addNetwork(1)
        counters.addWasted(UInt64.max)
        counters.addWasted(1)

        XCTAssertEqual(counters.networkBytes, UInt64.max)
        XCTAssertEqual(counters.wastedBytes, UInt64.max)
        XCTAssertTrue(counters.didOverflow)
    }
}
