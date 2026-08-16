import Foundation
import XCTest

@testable import MinirunKit

/// The one arm that reads a real drive.
///
/// Off by default, because a suite that reads half a gigabyte off somebody's
/// enclosure every time it runs is a suite people stop running. It is also
/// strictly **read-only**: `/Volumes/K3NVME` holds the published artifact and is
/// read-only by standing instruction, and this arm opens files there with
/// `O_RDONLY` and creates nothing at all.
///
///     TEST_RUNNER_MINIRUN_ASSESS_K3NVME=1 xcodebuild test \
///         -scheme minirun-Package -destination 'platform=macOS' \
///         -only-testing:MinirunKitTests/ReadSpotIntegrationTests
///
/// `xcodebuild test` does not forward the caller's environment to the test
/// process, and an env-gated arm invoked that way skips silently — the worst
/// failure mode a measurement has. The `TEST_RUNNER_` prefix is the one form
/// `xcodebuild` does pass through, so it is what the gate reads first; the bare
/// name is accepted too, for the `build-for-testing` + `xcrun xctest` route
/// AGENTS.md describes.
final class ReadSpotIntegrationTests: XCTestCase {

    static let gate = "MINIRUN_ASSESS_K3NVME"
    static let defaultVolume = "/Volumes/K3NVME"

    private var volumePath: String {
        ProcessInfo.processInfo.environment["TEST_RUNNER_MINIRUN_ASSESS_VOLUME"]
            ?? ProcessInfo.processInfo.environment["MINIRUN_ASSESS_VOLUME"]
            ?? Self.defaultVolume
    }

    override func setUpWithError() throws {
        let environment = ProcessInfo.processInfo.environment
        let armed =
            environment["TEST_RUNNER_" + Self.gate] == "1" || environment[Self.gate] == "1"
        try XCTSkipUnless(
            armed,
            "set TEST_RUNNER_\(Self.gate)=1 to run the reduced read spot test against "
                + "\(Self.defaultVolume) — it reads about 512 MB and writes nothing")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: volumePath),
            "\(volumePath) is not mounted")
    }

    /// A reduced spot test — ~512 MB, a quarter of what the app's button runs.
    ///
    /// It asserts a *sane* rate and deliberately not a link class. Which port
    /// this enclosure is plugged into today is the very thing the instrument
    /// exists to discover, and a test that failed when the answer was USB3 would
    /// be a test that fails when the instrument is working.
    func testAReducedSpotTestMeasuresARealVolume() throws {
        let root = URL(fileURLWithPath: volumePath)
        let test = ReadSpotTest(configuration: .reduced)

        let targets = try test.targets(under: root)
        XCTAssertFalse(targets.isEmpty)

        let before = try FileManager.default.contentsOfDirectory(atPath: volumePath).sorted()
        let result = try test.run(on: root)
        let after = try FileManager.default.contentsOfDirectory(atPath: volumePath).sorted()
        // The standing rule, asserted rather than trusted.
        XCTAssertEqual(before, after, "the spot test must not create anything on \(volumePath)")

        XCTAssertEqual(result.shortReadCount, 0)
        XCTAssertGreaterThan(result.sequentialBytesRead, 0)
        XCTAssertGreaterThan(
            result.sequentialBytesPerSecond, 0.2e9,
            "0.2 GB/s is below every link this project has measured; a rate under it means "
                + "something other than the drive is being timed")
        XCTAssertGreaterThan(result.scatteredBytesPerSecond, 0)

        // Printed rather than asserted: this is the number the whole module
        // exists to produce, and a record of it in the test log is free.
        print(
            "K3NVME reduced spot test: "
                + String(format: "%.2f GB/s sequential", result.sequentialBytesPerSecond / 1e9)
                + String(format: ", %.2f GB/s scattered", result.scatteredBytesPerSecond / 1e9)
                + ", \(result.totalBytesRead) B read in "
                + String(format: "%.2f s", result.durationSeconds)
                + ", class \(LinkClassifier.linkClass(forBytesPerSecond: result.sequentialBytesPerSecond).rawValue)")
    }
}
