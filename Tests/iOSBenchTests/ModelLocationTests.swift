import Foundation
import StorageCore
import XCTest

@testable import iOSBench

/// The layout check, against manifests shaped like the two the repository's
/// own tools write.
final class ModelLocationResolverTests: XCTestCase {

    /// Shaped after `Tools/k3_reference/convert_tiny_experts.py`, whose
    /// manifest `RoutedExpertContainers` reads.
    private func tinyManifest(files: [String]) -> [String: Any] {
        [
            "containers": files.enumerated().map { index, file in
                [
                    "layer": 1, "projection": "w1", "file": file,
                    "experts_per_tile": 4, "out_features_per_expert": 256,
                    "in_features": 512, "tiles": [["tile": index, "experts": [0, 1, 2, 3]]],
                ] as [String: Any]
            }
        ]
    }

    /// Shaped after `Tools/k3_flagship/build_full_artifact.py`.
    private func flagshipManifest(complete: Bool) -> [String: Any] {
        [
            "schema_version": 1,
            "bundle_kind": "k3_full_text_artifact",
            "complete": complete,
            "units": [
                [
                    "unit_id": "layer01-experts", "kind": "experts", "layer": 1,
                    "directory": "layer01",
                    "files": ["w1": "layer01-w1.bin", "w2": "layer01-w2.bin"],
                ] as [String: Any],
                [
                    "unit_id": "global-embed", "kind": "global", "layer": NSNull(),
                    "directory": "global", "files": ["embed": "embed.bin"],
                ] as [String: Any],
            ],
        ]
    }

    func testDirectoryWithoutManifestIsNotAnArtifact() {
        let directory = TemporaryDirectory(self)
        let report = ModelLocationResolver.describe(directory: directory.url)

        XCTAssertNil(report.layout)
        XCTAssertFalse(report.isComplete)
        XCTAssertTrue(report.files.isEmpty)
        XCTAssertNil(report.manifestProblem)
    }

    func testCompleteTinyContainerSetIsRecognised() {
        let directory = TemporaryDirectory(self)
        directory.write(4096, to: "l1-w1.bin")
        directory.write(2048, to: "l1-w2.bin")
        directory.writeManifest(tinyManifest(files: ["l1-w1.bin", "l1-w2.bin"]))

        let report = ModelLocationResolver.describe(directory: directory.url)

        XCTAssertEqual(report.layout, .tinyContainerSet)
        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(report.files.count, 2)
        XCTAssertEqual(report.presentBytes, 6144)
        XCTAssertTrue(report.missingFiles.isEmpty)
        XCTAssertNil(report.builderReportsComplete, "this layout carries no completeness flag")
    }

    func testMissingFilesAreNamedRatherThanCollapsedToAFlag() {
        let directory = TemporaryDirectory(self)
        directory.write(4096, to: "l1-w1.bin")
        directory.writeManifest(tinyManifest(files: ["l1-w1.bin", "l1-w2.bin", "l1-w3.bin"]))

        let report = ModelLocationResolver.describe(directory: directory.url)

        XCTAssertEqual(report.layout, .tinyContainerSet)
        XCTAssertFalse(report.isComplete)
        XCTAssertEqual(report.missingFiles, ["l1-w2.bin", "l1-w3.bin"])
        XCTAssertEqual(report.presentBytes, 4096, "only files that exist count")
    }

    func testFlagshipBundleResolvesPerUnitSubdirectories() {
        let directory = TemporaryDirectory(self)
        directory.write(1024, to: "layer01/layer01-w1.bin")
        directory.write(1024, to: "layer01/layer01-w2.bin")
        directory.write(512, to: "global/embed.bin")
        directory.writeManifest(flagshipManifest(complete: true))

        let report = ModelLocationResolver.describe(directory: directory.url)

        XCTAssertEqual(report.layout, .fullArtifactBundle)
        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(report.presentBytes, 2560)
        XCTAssertEqual(report.builderReportsComplete, true)
        XCTAssertEqual(
            Set(report.files.map(\.relativePath)),
            ["layer01/layer01-w1.bin", "layer01/layer01-w2.bin", "global/embed.bin"])
    }

    func testBuilderIncompleteFlagIsReportedSeparatelyFromFilePresence() {
        let directory = TemporaryDirectory(self)
        directory.write(1024, to: "layer01/layer01-w1.bin")
        directory.write(1024, to: "layer01/layer01-w2.bin")
        directory.write(512, to: "global/embed.bin")
        directory.writeManifest(flagshipManifest(complete: false))

        let report = ModelLocationResolver.describe(directory: directory.url)

        // Every named file is there, yet the builder says it is not done —
        // an artifact still being written names only the units it has
        // finished. Both facts have to reach the screen.
        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(report.builderReportsComplete, false)
    }

    func testCorruptManifestIsReportedNotIgnored() {
        let directory = TemporaryDirectory(self)
        try! Data("{not json".utf8).write(to: directory.url.appendingPathComponent("manifest.json"))

        let report = ModelLocationResolver.describe(directory: directory.url)

        XCTAssertEqual(report.layout, .unrecognized)
        XCTAssertNotNil(report.manifestProblem)
        XCTAssertFalse(report.isComplete)
    }

    func testUnknownManifestShapeIsUnrecognisedRatherThanEmpty() {
        let directory = TemporaryDirectory(self)
        directory.writeManifest(["schema_version": 1, "tensors": ["a", "b"]])

        let report = ModelLocationResolver.describe(directory: directory.url)

        XCTAssertEqual(report.layout, .unrecognized)
        XCTAssertEqual(
            report.manifestProblem,
            "manifest.json has neither a 'containers' nor a 'units' list")
    }

    func testManifestNamingNoFilesIsNotComplete() {
        let directory = TemporaryDirectory(self)
        directory.writeManifest(tinyManifest(files: []))

        let report = ModelLocationResolver.describe(directory: directory.url)

        XCTAssertEqual(report.layout, .tinyContainerSet)
        XCTAssertFalse(report.isComplete, "zero files must not satisfy allSatisfy")
        XCTAssertEqual(report.manifestProblem, "manifest.json names no files")
    }

    // MARK: - Volume facts

    func testFreeSpaceComesFromStorageCore() {
        let directory = TemporaryDirectory(self)
        let report = ModelLocationResolver.describe(directory: directory.url)

        // Not a duplicated implementation: the same StorageInfo the benchmark
        // schema uses (spec §18.3).
        XCTAssertEqual(report.storage.resolvedPath, StorageInfo.describing(path: directory.url.path).resolvedPath)
        XCTAssertTrue(report.storage.pathExists)
        XCTAssertTrue(report.storage.pathIsDirectory)
        let free = try? XCTUnwrap(report.freeBytes)
        XCTAssertGreaterThan(free ?? 0, 0)
    }

    func testReportIsCodable() throws {
        let directory = TemporaryDirectory(self)
        directory.write(16, to: "l1-w1.bin")
        directory.writeManifest(tinyManifest(files: ["l1-w1.bin"]))
        let report = ModelLocationResolver.describe(directory: directory.url)

        let data = try JSONEncoder().encode(report)
        let decoded = try JSONDecoder().decode(ModelLocationReport.self, from: data)
        XCTAssertEqual(decoded.layout, report.layout)
        XCTAssertEqual(decoded.files, report.files)
        XCTAssertEqual(decoded.presentBytes, report.presentBytes)
    }

    func testSummaryNamesThePathFreeSpaceAndArtifactState() {
        let directory = TemporaryDirectory(self)
        directory.write(4096, to: "l1-w1.bin")
        directory.writeManifest(tinyManifest(files: ["l1-w1.bin", "l1-w2.bin"]))

        let lines = ModelLocationResolver.describe(directory: directory.url).summaryLines

        XCTAssertTrue(lines.contains { $0.hasPrefix("path: ") })
        XCTAssertTrue(lines.contains { $0.hasPrefix("free: ") && !$0.contains("unavailable") })
        XCTAssertTrue(lines.contains { $0.contains("tinyContainerSet") })
        XCTAssertTrue(lines.contains { $0.contains("1 file(s) missing") && $0.contains("l1-w2.bin") })
    }
}

/// The picker's non-UIKit half, which is where all of its behaviour lives.
final class PickedLocationHandlerTests: XCTestCase {

    func testAcceptBookmarksInsideTheBracketAndDescribes() throws {
        let directory = TemporaryDirectory(self)
        directory.write(32, to: "l1-w1.bin")
        directory.writeManifest([
            "containers": [
                [
                    "layer": 1, "projection": "w1", "file": "l1-w1.bin",
                    "experts_per_tile": 4, "out_features_per_expert": 256,
                    "in_features": 512, "tiles": [["tile": 0, "experts": [0, 1, 2, 3]]],
                ] as [String: Any]
            ]
        ])

        let coding = FakeSecurityScopedURLCoding()
        let persistence = InMemoryBookmarkPersistence()
        let handler = PickedLocationHandler(
            store: SecurityScopedLocationStore(coding: coding, persistence: persistence))

        let report = try handler.accept(directory.url)

        XCTAssertEqual(report.layout, .tinyContainerSet)
        XCTAssertTrue(report.isComplete)
        XCTAssertNotNil(persistence.data(forKey: PickedLocationHandler.modelDirectoryKey))
        // One bracket, closed, with the bookmark created inside it.
        XCTAssertEqual(coding.startCount, 1)
        XCTAssertEqual(coding.stopCount, 1)
        XCTAssertEqual(coding.bookmarkCount, 1)
    }

    func testAcceptRejectsAFileAndStillClosesTheBracket() {
        let directory = TemporaryDirectory(self)
        let file = directory.write(8, to: "not-a-directory.bin")

        let coding = FakeSecurityScopedURLCoding()
        let persistence = InMemoryBookmarkPersistence()
        let handler = PickedLocationHandler(
            store: SecurityScopedLocationStore(coding: coding, persistence: persistence))

        XCTAssertThrowsError(try handler.accept(file)) { error in
            XCTAssertEqual(error as? LocationPickError, .notADirectory(path: file.path))
        }
        XCTAssertEqual(coding.startCount, 1)
        XCTAssertEqual(coding.stopCount, 1, "the rejection path must not leak the grant")
        XCTAssertEqual(coding.bookmarkCount, 0, "nothing should be remembered")
        XCTAssertTrue(persistence.keys.isEmpty)
    }

    func testDescribeRememberedReturnsNilBeforeAnythingIsPicked() throws {
        let handler = PickedLocationHandler(
            store: SecurityScopedLocationStore(
                coding: FakeSecurityScopedURLCoding(),
                persistence: InMemoryBookmarkPersistence()))
        XCTAssertNil(try handler.describeRemembered())
    }

    func testDescribeRememberedThrowsWhenTheVolumeIsGone() throws {
        let directory = TemporaryDirectory(self)
        let coding = FakeSecurityScopedURLCoding()
        let handler = PickedLocationHandler(
            store: SecurityScopedLocationStore(
                coding: coding, persistence: InMemoryBookmarkPersistence()))
        _ = try handler.accept(directory.url)

        // "picked but detached" must not read as "never picked".
        coding.failResolution = true
        XCTAssertThrowsError(try handler.describeRemembered()) { error in
            guard case .resolutionFailed? = error as? SecurityScopedBookmarkError else {
                return XCTFail("expected resolutionFailed, got \(error)")
            }
        }
    }

    func testDescribeRememberedSurvivesRelaunch() throws {
        let directory = TemporaryDirectory(self)
        directory.writeManifest(["containers": [] as [Any]])

        let coding = FakeSecurityScopedURLCoding()
        let persistence = InMemoryBookmarkPersistence()
        _ = try PickedLocationHandler(
            store: SecurityScopedLocationStore(coding: coding, persistence: persistence)
        ).accept(directory.url)

        // A second handler over the same persistence is what the next launch
        // looks like.
        let relaunched = PickedLocationHandler(
            store: SecurityScopedLocationStore(coding: coding, persistence: persistence))
        let report = try XCTUnwrap(try relaunched.describeRemembered())

        XCTAssertEqual(report.directoryPath, directory.url.path)
        XCTAssertTrue(coding.isBalanced)
    }

    func testForgetClearsTheRememberedLocation() throws {
        let directory = TemporaryDirectory(self)
        let persistence = InMemoryBookmarkPersistence()
        let handler = PickedLocationHandler(
            store: SecurityScopedLocationStore(
                coding: FakeSecurityScopedURLCoding(), persistence: persistence))
        _ = try handler.accept(directory.url)
        XCTAssertFalse(persistence.keys.isEmpty)

        handler.forget()
        XCTAssertTrue(persistence.keys.isEmpty)
        XCTAssertNil(try handler.describeRemembered())
    }
}
