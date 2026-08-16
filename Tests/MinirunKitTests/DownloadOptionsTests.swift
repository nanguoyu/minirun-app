import Foundation
import XCTest

@testable import MinirunKit

/// Invalid configuration is refused with a named error, never clamped. Each of
/// these asserts the *name* as well as the throw, because "invalid option" with
/// no name is a message an operator cannot act on.
final class DownloadOptionsTests: XCTestCase {

    private func refusal(_ mutate: (inout DownloadOptions) -> Void) -> DownloadError? {
        var options = DownloadOptions()
        mutate(&options)
        do {
            _ = try options.validated()
            return nil
        } catch {
            return error as? DownloadError
        }
    }

    func testTheDefaultsValidate() throws {
        XCTAssertEqual(try DownloadOptions().validated(), DownloadOptions())
    }

    func testAChunkBelowTheFloorIsRefusedAndNotRaised() throws {
        guard case .invalidOption(let name, let value, let allowed)? = refusal({
            $0.chunkBytes = 1024
        }) else { return XCTFail("expected a refusal") }
        XCTAssertEqual(name, "chunkBytes")
        XCTAssertEqual(value, "1024")
        XCTAssertTrue(allowed.contains("\(1 << 20)"))
        // And nothing was quietly raised to the floor.
        var options = DownloadOptions()
        options.chunkBytes = 1024
        XCTAssertEqual(options.chunkBytes, 1024)
    }

    func testAChunkAboveTheCeilingIsRefused() {
        guard case .invalidOption(let name, _, _)? = refusal({ $0.chunkBytes = 2 << 30 }) else {
            return XCTFail("expected a refusal")
        }
        XCTAssertEqual(name, "chunkBytes")
    }

    func testAChunkThatIsNotAWholeNumberOfMebibytesIsRefused() {
        guard case .invalidOption(_, _, let allowed)? = refusal({
            $0.chunkBytes = (1 << 20) + 1
        }) else { return XCTFail("expected a refusal") }
        XCTAssertTrue(allowed.contains("MiB"))
    }

    func testConcurrentFileCountIsBounded() {
        XCTAssertNotNil(refusal { $0.maximumConcurrentFiles = 0 })
        XCTAssertNotNil(refusal { $0.maximumConcurrentFiles = 17 })
        XCTAssertNil(refusal { $0.maximumConcurrentFiles = 16 })
    }

    /// Parallel chunks within a file are refused rather than accepted and
    /// ignored. Accepting the knob and running sequentially anyway would leave
    /// a run recorded under a setting it did not use.
    func testParallelChunksWithinAFileAreRefusedWithTheReason() {
        guard case .invalidOption(let name, let value, let allowed)? = refusal({
            $0.maximumConcurrentChunksPerFile = 4
        }) else { return XCTFail("expected a refusal") }
        XCTAssertEqual(name, "maximumConcurrentChunksPerFile")
        XCTAssertEqual(value, "4")
        XCTAssertTrue(allowed.contains("part file"))
    }

    func testRetryPolicyIsBounded() {
        XCTAssertNotNil(refusal { $0.retry.maximumAttempts = 0 })
        XCTAssertNotNil(refusal { $0.retry.maximumAttempts = 11 })
        XCTAssertNotNil(refusal { $0.retry.backoffMultiplier = 0.5 })
        XCTAssertNotNil(refusal { $0.retry.initialBackoff = .infinity })
        XCTAssertNotNil(refusal { $0.retry.backoffMultiplier = .infinity })
        XCTAssertNotNil(refusal { $0.retry.maximumBackoff = .infinity })
        XCTAssertNotNil(refusal { $0.retry.initialBackoff = -1 })
        XCTAssertNotNil(refusal {
            $0.retry.initialBackoff = 30
            $0.retry.maximumBackoff = 5
        })
    }

    func testBackoffGrowsGeometricallyAndIsCapped() {
        var policy = RetryPolicy()
        policy.initialBackoff = 1
        policy.backoffMultiplier = 2
        policy.maximumBackoff = 10
        XCTAssertEqual(policy.backoff(beforeAttempt: 1), 0)
        XCTAssertEqual(policy.backoff(beforeAttempt: 2), 1)
        XCTAssertEqual(policy.backoff(beforeAttempt: 3), 2)
        XCTAssertEqual(policy.backoff(beforeAttempt: 4), 4)
        XCTAssertEqual(policy.backoff(beforeAttempt: 10), 10, "capped, not unbounded")
    }

    func testOptionsRoundTripThroughJSON() throws {
        var options = DownloadOptions()
        options.chunkBytes = 8 << 20
        options.allowMetered = true
        let data = try MinirunKitJSON.encoder().encode(options)
        XCTAssertEqual(
            try MinirunKitJSON.decoder().decode(DownloadOptions.self, from: data), options)
    }

    /// A refusal has to survive being written down, because a preflight is a
    /// document somebody is shown when they ask why a terabyte would not start.
    func testEveryRefusalRoundTripsThroughJSON() throws {
        let refusals: [DownloadError] = [
            .invalidOption(name: "chunkBytes", value: "1", allowed: "a lot"),
            .destinationNotWritable(path: "/x", reason: "read-only"),
            .destinationIsReadOnlyVolume(path: "/Volumes/K3NVME"),
            .insufficientFreeSpace(
                needBytes: 10, headroomBytes: 2, availableBytes: 3, volume: "NVMe"),
            .insufficientFreeSpace(
                needBytes: 10, headroomBytes: 2, availableBytes: 3, volume: nil),
            .revisionMoved(expected: "a", served: "b", path: "p"),
            .rangeNotSupported(path: "p"),
            .sizeMismatch(SizeMismatch(path: "p", expected: 1, found: 2)),
            .digestMismatch(
                DigestMismatch(path: "p", algorithm: .gitBlobSHA1, expected: "a", found: "b")),
            .digestUnavailable(path: "p"),
            .transport(status: 503, path: "p", attempt: 2),
            .interrupted(path: "p", atByte: 9, underlying: "reset"),
            .scopeLost(path: "p"),
            .cancelled,
            .unknownJob(DownloadJobID()),
            .jobNotPaused(DownloadJobID()),
            .jobAlreadyRunning(DownloadJobID()),
            .statePersistenceFailed(job: DownloadJobID(), reason: "disk full"),
            .spaceUnknown(path: "p", reason: "unprobeable"),
        ]
        for refusal in refusals {
            let data = try MinirunKitJSON.encoder().encode(refusal)
            let decoded = try MinirunKitJSON.decoder().decode(DownloadError.self, from: data)
            XCTAssertEqual(decoded, refusal, "\(refusal) did not survive a round trip")
            XCTAssertEqual(decoded.description, refusal.description)
        }
    }
}
