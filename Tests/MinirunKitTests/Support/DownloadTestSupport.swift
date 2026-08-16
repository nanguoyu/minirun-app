import Foundation
import XCTest

@testable import MinirunKit

enum FixtureBytes {
    /// Deterministic pseudo-random bytes. Deterministic so a failing digest
    /// names a bug rather than a coin flip, and pseudo-random so a truncated
    /// or byte-swapped file cannot accidentally still hash correctly.
    static func make(_ count: Int, seed: UInt64) -> Data {
        var state = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        var out = [UInt8]()
        out.reserveCapacity(count)
        for _ in 0..<count {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            out.append(UInt8((state >> 33) & 0xFF))
        }
        return Data(out)
    }
}

/// A small artifact shaped like the published ones: files under per-unit
/// directories, a `manifest.json` beside them, and root documents — so the
/// payload/metadata rule is exercised by the download tests too, not only by
/// the recorded trees.
enum FixtureArtifact {
    static let largeFilePath = "layer00/layer00-w1.mxfp4tile"

    static func files(largeBytes: Int = 3 << 20) -> [String: Data] {
        var files: [String: Data] = [:]
        files["README.md"] = Data("# fixture\n".utf8)
        files["index.json"] = Data(
            #"{"format":"quantized_tile_container","files":3,"bytes":0}"#.utf8)
        files[largeFilePath] = FixtureBytes.make(largeBytes, seed: 1)
        files["layer00/layer00-deterministic.bin"] = FixtureBytes.make(4096, seed: 2)
        files["layer00/manifest.json"] = Data(#"{"unit":"layer00","files":2}"#.utf8)
        files["global/global-embed.bin"] = FixtureBytes.make(8192, seed: 3)
        files["global/manifest.json"] = Data(#"{"unit":"global","files":1}"#.utf8)
        return files
    }

    /// Files the payload rule counts as payload, given the map above.
    static func payloadPaths(_ files: [String: Data]) -> [String] {
        files.keys.filter { RepoFileClassification.isPayload(path: $0) }.sorted()
    }
}

/// Options sized for a test: chunks at the smallest the validator accepts, and
/// a headroom that does not depend on how full the developer's disk is.
func testOptions(
    chunkBytes: UInt64 = 1 << 20, concurrentFiles: Int = 2, verifyEachFile: Bool = true,
    attempts: Int = 3
) -> DownloadOptions {
    var options = DownloadOptions()
    options.chunkBytes = chunkBytes
    options.maximumConcurrentFiles = concurrentFiles
    options.verifyAfterEachFile = verifyEachFile
    options.freeSpaceHeadroomBytes = 1 << 20
    options.retry.maximumAttempts = attempts
    options.retry.initialBackoff = 0.01
    options.retry.maximumBackoff = 0.05
    return options
}

extension XCTestCase {
    /// Consume a job's events to its terminal one.
    ///
    /// Also asserts the progress identity on every progress event that goes
    /// past: `verified + fetchedUnverified + inFlight + remaining` must equal
    /// the plan total, always, including mid-transfer and after a failure.
    func drainEvents(
        of manager: DownloadManager, job: DownloadJobID, timeout: TimeInterval = 60,
        file: StaticString = #filePath, line: UInt = #line
    ) async -> [DownloadEvent] {
        let collector = Task { () -> [DownloadEvent] in
            var events: [DownloadEvent] = []
            for await event in manager.events(for: job) {
                events.append(event)
                if case .progress(let progress) = event {
                    XCTAssertTrue(
                        progress.accountsForEveryByte,
                        """
                        progress buckets do not partition the plan: \
                        verified \(progress.verifiedBytes) + fetched \(progress.fetchedUnverifiedBytes) \
                        + inFlight \(progress.inFlightBytes) + remaining \(progress.remainingBytes) \
                        != \(progress.planTotalBytes)
                        """,
                        file: file, line: line)
                }
                if event.isTerminal { break }
            }
            return events
        }
        let watchdog = Task {
            try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
            collector.cancel()
        }
        let events = await collector.value
        watchdog.cancel()
        return events
    }
}

extension Array where Element == DownloadEvent {
    var summary: DownloadSummary? {
        for event in self { if case .finished(let summary) = event { return summary } }
        return nil
    }
    var failure: DownloadError? {
        for event in self { if case .failed(let error, _) = event { return error } }
        return nil
    }
    var cancellation: DownloadProgress? {
        for event in self { if case .cancelled(let progress) = event { return progress } }
        return nil
    }
    var verifiedPaths: [String] {
        compactMap { if case .fileVerified(let path, _, _) = $0 { return path } else { return nil } }
    }
    var fileFailures: [(String, DownloadError)] {
        compactMap {
            if case .fileFailed(let path, let error) = $0 { return (path, error) } else { return nil }
        }
    }
    var lastProgress: DownloadProgress? {
        var found: DownloadProgress?
        for event in self { if case .progress(let progress) = event { found = progress } }
        return found
    }
}
