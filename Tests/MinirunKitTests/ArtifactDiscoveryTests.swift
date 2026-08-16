import Foundation
import XCTest

@testable import MinirunKit

/// Discovery is the app's answer to "is this model installed?", and it answers
/// it from a directory rather than from a registry. These fixtures are the three
/// `index.json` shapes that actually exist on the drive, transcribed field for
/// field, because a reader that only ever saw one of them is a reader that
/// reports two real artifacts as unidentified.
final class ArtifactDiscoveryTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-discovery-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: - Fixtures

    /// K3's shape: totals at the top level, one upstream repository.
    static let k3Index = """
        {
          "format": "quantized_tile_container",
          "format_version": 1,
          "source_repo": "moonshotai/Kimi-K3",
          "source_revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
          "relationship": "byte_preserving_repack",
          "layers": 94,
          "files": 372,
          "bytes": 1559976181760
        }
        """

    /// V4-Flash's shape: one upstream, a `units` array, `total_bytes`.
    static let v4Index = """
        {
          "source_repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
          "source_revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
          "relationship": "byte-preserving repack",
          "units": [
            { "unit": "layers00", "files": 12, "bytes": 3566452184 },
            { "unit": "layers01", "files": 10, "bytes": 3566452184 }
          ],
          "total_bytes": 166893192184
        }
        """

    /// H3's shape: no `source_repo` at all — a map of five upstreams instead.
    static let h3Index = """
        {
          "sources": {
            "dit": {
              "repo": "pipenetwork/MiniMax-H3-MLX-8bit",
              "revision": "3ac52081470b0488921c3ec3ba84a39097bf2361",
              "role": "diffusion transformer, affine 8-bit, group 64",
              "license": "MiniMax H3 Community License Agreement"
            },
            "text_encoder": {
              "repo": "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit",
              "revision": "64314cde0ac6d90f132bc94ae58e0c82f77396c6",
              "role": "text encoder"
            },
            "official": {
              "repo": "MiniMaxAI/MiniMax-H3",
              "revision": "6818f6c32d12b210915e44ad56a4228c2608f160",
              "role": "video VAE"
            }
          },
          "relationship": "byte-preserving repack with AdaLN pruning",
          "units": [
            { "unit": "adaln-cache", "files": 6, "bytes": 1959064288 },
            { "unit": "dit-block-00", "files": 4, "bytes": 1000 }
          ],
          "total_bytes": 63969279450
        }
        """

    @discardableResult
    private func makeArtifact(
        named name: String, index: String, payload: [String: Int] = [:], in parent: URL? = nil
    ) throws -> URL {
        let directory = (parent ?? root!).appendingPathComponent(name)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(index.utf8).write(to: directory.appendingPathComponent("index.json"))
        for (path, bytes) in payload {
            let url = directory.appendingPathComponent(path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(repeating: 0x41, count: bytes).write(to: url)
        }
        return directory
    }

    private func locator() -> ArtifactLocator {
        ArtifactLocator(catalog: ModelCatalog.bundled)
    }

    // MARK: - Identity, one test per real shape

    func testTheFlatShapeNamesOneUpstreamAndIsIdentifiedAsK3() throws {
        let identity = ArtifactIndexIdentity.parse(Data(Self.k3Index.utf8))
        XCTAssertEqual(identity.shape, .singleSourceTotals)
        XCTAssertEqual(identity.repositories.map(\.repoID), ["moonshotai/Kimi-K3"])
        XCTAssertEqual(
            identity.repositories.first?.revision, "9f62e4e9fffbd0a83ddd60e1c209d828994b3569")
        XCTAssertEqual(identity.format, "quantized_tile_container")
        XCTAssertEqual(identity.declaredFileCount, 372)
        XCTAssertEqual(identity.declaredBytes, 1_559_976_181_760)

        let matcher = ArtifactIdentityMatcher(catalog: ModelCatalog.bundled)
        XCTAssertEqual(matcher.match(identity), .kimiK3)
    }

    func testK3TokenizerIdentityIsAcceptedFromTheVerifiedIndexWithoutAnArtifactCommitConstant()
        throws
    {
        let data = try JSONSerialization.data(withJSONObject: [
            "source_repo": "moonshotai/Kimi-K3",
            "source_revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
            "files": 372,
            "bytes": NSNumber(value: UInt64(1_559_976_181_760)),
            "tokenizer": [
                "file": "tiktoken.model",
                "bytes": 2_795_286,
                "sha256": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
                "source_repo": "moonshotai/Kimi-K3",
                "source_revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
                "license": "LICENSE",
            ],
        ])

        let tokenizer = try XCTUnwrap(ArtifactIndexIdentity.parse(data).tokenizer)
        XCTAssertEqual(tokenizer.file, "tiktoken.model")
        XCTAssertEqual(tokenizer.bytes, 2_795_286)
        XCTAssertEqual(tokenizer.sourceRepo, "moonshotai/Kimi-K3")
        XCTAssertEqual(tokenizer.sourceRevision.count, 40)
        XCTAssertEqual(tokenizer.sha256.count, 64)
    }

    func testMalformedTokenizerEvidenceIsAbsentRatherThanPartlyTrusted() throws {
        let malformed: [[String: Any]] = [
            ["file": "../tokenizer", "bytes": 1, "sha256": String(repeating: "a", count: 64),
             "source_repo": "moonshotai/Kimi-K3", "source_revision": String(repeating: "b", count: 40),
             "license": "LICENSE"],
            ["file": "tiktoken.model", "bytes": -1, "sha256": String(repeating: "a", count: 64),
             "source_repo": "moonshotai/Kimi-K3", "source_revision": String(repeating: "b", count: 40),
             "license": "LICENSE"],
            ["file": "tiktoken.model", "bytes": 1, "sha256": "not-a-digest",
             "source_repo": "moonshotai/Kimi-K3", "source_revision": String(repeating: "b", count: 40),
             "license": "LICENSE"],
            ["file": "tiktoken.model", "bytes": 1, "sha256": String(repeating: "a", count: 64),
             "source_repo": "moonshotai/Kimi-K3", "source_revision": "main",
             "license": "LICENSE"],
        ]
        for tokenizer in malformed {
            let data = try JSONSerialization.data(withJSONObject: [
                "source_repo": "moonshotai/Kimi-K3", "files": 1, "bytes": 1,
                "tokenizer": tokenizer,
            ])
            XCTAssertNil(ArtifactIndexIdentity.parse(data).tokenizer)
        }
    }

    func testV4ConfigurationAndTokenizerIdentitiesRequireCompletePinnedEvidence() throws {
        let revision = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
        let data = try JSONSerialization.data(withJSONObject: [
            "source_repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "source_revision": revision,
            "units": [["unit": "global00", "files": 3, "bytes": 17]],
            "total_bytes": 17,
            "configuration": [
                "file": "config.json",
                "bytes": 1_888,
                "sha256": "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
                "source_repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
                "source_revision": revision,
            ],
            "tokenizer": [
                "file": "tokenizer.json",
                "bytes": 6_367_146,
                "sha256": "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
                "source_repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
                "source_revision": revision,
                "license": "LICENSE",
            ],
        ])

        let identity = ArtifactIndexIdentity.parse(data)
        XCTAssertEqual(identity.configuration?.file, "config.json")
        XCTAssertEqual(identity.configuration?.bytes, 1_888)
        XCTAssertEqual(identity.configuration?.sourceRevision, revision)
        XCTAssertEqual(identity.tokenizer?.file, "tokenizer.json")
        XCTAssertEqual(identity.tokenizer?.bytes, 6_367_146)
        XCTAssertEqual(identity.tokenizer?.sourceRevision, revision)
    }

    func testMalformedV4ConfigurationEvidenceIsAbsentRatherThanPartlyTrusted() throws {
        let revision = String(repeating: "b", count: 40)
        let base: [String: Any] = [
            "file": "config.json", "bytes": 1,
            "sha256": String(repeating: "a", count: 64),
            "source_repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "source_revision": revision,
        ]
        var cases: [[String: Any]] = []
        let malformedFields: [(String, Any)] = [
            ("file", "nested/config.json"), ("bytes", -1),
            ("sha256", "not-a-digest"), ("source_revision", "main"),
        ]
        for (field, value) in malformedFields {
            var candidate = base
            candidate[field] = value
            cases.append(candidate)
        }
        for configuration in cases {
            let data = try JSONSerialization.data(withJSONObject: [
                "source_repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
                "units": [["unit": "global00", "files": 1, "bytes": 1]],
                "total_bytes": 1,
                "configuration": configuration,
            ])
            XCTAssertNil(ArtifactIndexIdentity.parse(data).configuration)
        }
    }

    func testTheUnitShapeSumsItsUnitsForAFileCount() throws {
        let identity = ArtifactIndexIdentity.parse(Data(Self.v4Index.utf8))
        XCTAssertEqual(identity.shape, .singleSourceUnits)
        XCTAssertEqual(identity.declaredFileCount, 22)
        XCTAssertEqual(identity.declaredBytes, 166_893_192_184)
        XCTAssertEqual(
            ArtifactIdentityMatcher(catalog: ModelCatalog.bundled).match(identity),
            .deepseekV4Flash)
    }

    func testTheMultiSourceShapeKeepsEveryUpstreamAndStillIdentifies() throws {
        let identity = ArtifactIndexIdentity.parse(Data(Self.h3Index.utf8))
        XCTAssertEqual(identity.shape, .multiSourceUnits)
        // Sorted by the map's key — dit, official, text_encoder — so that two
        // scans of one directory never disagree about the order.
        XCTAssertEqual(
            identity.repositories.map(\.repoID),
            [
                "pipenetwork/MiniMax-H3-MLX-8bit", "MiniMaxAI/MiniMax-H3",
                "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit",
            ])
        XCTAssertEqual(identity.repositories.first?.role, "diffusion transformer, affine 8-bit, group 64")
        XCTAssertEqual(
            ArtifactIdentityMatcher(catalog: ModelCatalog.bundled).match(identity), .minimaxH3)
    }

    /// The rule the owner asked for out loud: an index shape nobody wrote a
    /// reader for is an unidentified artifact, never a crash and never a wrong
    /// model.
    func testAnUnknownShapeIsReportedRatherThanGuessedAt() throws {
        let identity = ArtifactIndexIdentity.parse(
            Data(#"{"schema": "something-else", "weights": 12}"#.utf8))
        XCTAssertEqual(identity.shape, .unrecognized)
        XCTAssertTrue(identity.repositories.isEmpty)
        XCTAssertEqual(identity.topLevelKeys, ["schema", "weights"])
        XCTAssertNil(ArtifactIdentityMatcher(catalog: ModelCatalog.bundled).match(identity))
    }

    func testAnIndexThatIsNotJSONAtAllIsStillAnAnswer() throws {
        let identity = ArtifactIndexIdentity.parse(Data("not json {{{".utf8))
        XCTAssertEqual(identity, .unreadable)
    }

    // MARK: - Scanning a location

    func testTheThreeRealShapesAreAllFoundUnderOneLocation() throws {
        try makeArtifact(named: "k3-artifact", index: Self.k3Index)
        try makeArtifact(named: "v4-artifact", index: Self.v4Index)
        try makeArtifact(named: "h3-artifact", index: Self.h3Index)

        let scan = locator().scan(root)
        XCTAssertTrue(scan.isMounted)
        XCTAssertEqual(scan.artifacts.count, 3)
        XCTAssertEqual(
            Set(scan.artifacts.compactMap(\.model)),
            [.kimiK3, .deepseekV4Flash, .minimaxH3])
        XCTAssertTrue(scan.unreadable.isEmpty)
    }

    func testAnUnidentifiedArtifactIsListedWithItsBytesRatherThanHidden() throws {
        try makeArtifact(
            named: "mystery", index: #"{"weights": 3}"#, payload: ["a.bin": 1024])
        let scan = locator().scan(root)
        let found = try XCTUnwrap(scan.artifacts.first)
        XCTAssertNil(found.model)
        XCTAssertFalse(found.isIdentified)
        XCTAssertEqual(found.displayName, "mystery")
        XCTAssertEqual(found.bytesOnDisk, 1024 + UInt64(#"{"weights": 3}"#.utf8.count))
        XCTAssertNil(found.expectedFileCount)
        XCTAssertNil(found.isComplete)
    }

    /// The downloader keeps its tree cache and part files inside the artifact
    /// root. Counting them would push a half-finished artifact's file count
    /// *above* the catalog's expectation, which is the one direction a
    /// completeness check must never be wrong in.
    func testDownloaderBookkeepingIsNotCountedAsArtifact() throws {
        let directory = try makeArtifact(
            named: "k3-artifact", index: Self.k3Index,
            payload: [
                "global/embed_tokens.bin": 100,
                "layer00/layer00-deterministic.bin": 200,
                ".cache/huggingface/trees/abc.json": 4096,
                ".DS_Store": 8192,
            ])
        _ = directory

        let scan = locator().scan(root)
        let found = try XCTUnwrap(scan.artifacts.first)
        XCTAssertEqual(found.fileCount, 3)  // index.json + the two payload files
        XCTAssertEqual(found.bytesOnDisk, 300 + UInt64(Self.k3Index.utf8.count))
    }

    func testArtifactByteTallySaturatesAndRecordsOverflow() {
        var tally = ArtifactLocator.Tally()
        XCTAssertFalse(tally.add(fileSize: UInt64.max))
        XCTAssertTrue(tally.add(fileSize: 1))
        XCTAssertEqual(tally.bytes, UInt64.max)
        XCTAssertEqual(tally.files, 2)
        XCTAssertTrue(tally.overflowed)

        // Once overflow has occurred, later metadata cannot wrap the total
        // back to a small value that could be mistaken for a complete artifact.
        XCTAssertTrue(tally.add(fileSize: 42))
        XCTAssertEqual(tally.bytes, UInt64.max)
        XCTAssertEqual(tally.files, 3)

        let discovered = DiscoveredArtifact(
            rootPath: "/overflow", locationPath: "/", index: .unreadable,
            model: .kimiK3, displayName: "Overflow", bytesOnDisk: tally.bytes,
            fileCount: tally.files, expectedFileCount: 1, expectedBytes: 1,
            verification: .unverified, verifiedAt: nil, scannedAt: Date(),
            metadataOverflowed: tally.overflowed)
        XCTAssertNil(discovered.isComplete)
        XCTAssertNil(discovered.fractionOnDisk)
    }

    func testCompletenessComparesAgainstTheCatalogAndSaysHowFarShort() throws {
        try makeArtifact(named: "k3-artifact", index: Self.k3Index, payload: ["global/a.bin": 10])
        let found = try XCTUnwrap(locator().scan(root).artifacts.first)
        let expected = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        XCTAssertEqual(found.expectedFileCount, expected.totalFileCount)
        XCTAssertEqual(found.expectedBytes, expected.totalBytes)
        XCTAssertEqual(found.missingFileCount, expected.totalFileCount - 2)
        XCTAssertEqual(found.isComplete, false)
    }

    func testAnArtifactRootIsALeafSoANestedIndexIsNotASecondArtifact() throws {
        let outer = try makeArtifact(named: "k3-artifact", index: Self.k3Index)
        let inner = outer.appendingPathComponent("layer00")
        try FileManager.default.createDirectory(at: inner, withIntermediateDirectories: true)
        try Data(Self.v4Index.utf8).write(to: inner.appendingPathComponent("index.json"))

        let scan = locator().scan(root)
        XCTAssertEqual(scan.artifacts.count, 1)
        XCTAssertEqual(scan.artifacts.first?.model, .kimiK3)
    }

    func testTheWalkStopsAtTheDeclaredDepth() throws {
        let nested = root.appendingPathComponent("a/b/c")
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try Data(Self.k3Index.utf8).write(to: nested.appendingPathComponent("index.json"))

        XCTAssertTrue(locator().scan(root).artifacts.isEmpty)

        var deeper = locator()
        deeper.maximumDepth = 4
        XCTAssertEqual(deeper.scan(root).artifacts.count, 1)
    }

    func testALocationThatIsNotThereIsAnUnmountedRowRatherThanAThrow() {
        let scan = locator().scan(URL(fileURLWithPath: "/Volumes/DefinitelyNotMounted-\(UUID())"))
        XCTAssertFalse(scan.isMounted)
        XCTAssertTrue(scan.artifacts.isEmpty)
    }

    // MARK: - Verification is never a side effect of looking

    func testAScanReportsUnverifiedAndDoesNotDigestAnything() throws {
        try makeArtifact(named: "k3-artifact", index: Self.k3Index, payload: ["global/a.bin": 64])
        let ledger = InMemoryVerificationLedger()
        let locator = ArtifactLocator(catalog: ModelCatalog.bundled, verificationLedger: ledger)
        let found = try XCTUnwrap(locator.scan(root).artifacts.first)
        XCTAssertEqual(found.verification, .unverified)
        XCTAssertNil(found.verifiedAt)
        // Nothing was written to the ledger, which is the observable half of
        // "the scan did not verify": a scan that hashed would have had a result
        // to file.
        XCTAssertNil(
            ledger.record(
                matching: ArtifactVerificationLookup(
                    rootPath: found.rootPath, model: found.model,
                    repository: ModelCatalog.bundled.descriptor(.kimiK3)?.source.repo,
                    index: found.index)))
    }

    func testALegacyRecordedVerificationIsDowngradedByTheNextScan() throws {
        try makeArtifact(named: "k3-artifact", index: Self.k3Index)
        let ledger = InMemoryVerificationLedger()
        let locator = ArtifactLocator(catalog: ModelCatalog.bundled, verificationLedger: ledger)
        let before = try XCTUnwrap(locator.scan(root).artifacts.first)
        ledger.record(
            ArtifactVerificationRecord(
                rootPath: before.rootPath, state: .spotChecked, checkedAt: Date(),
                detail: "4 of the largest payload files matched."))

        let after = try XCTUnwrap(locator.scan(root).artifacts.first)
        XCTAssertEqual(after.verification, .unverified)
        XCTAssertNil(after.verifiedAt)
    }

    /// A spot check considers the biggest files first but its byte argument is
    /// a hard ceiling. Oversized entries are skipped so a smaller positive-size
    /// payload can still provide evidence within the stated budget.
    func testASpotCheckSpendsItsBudgetOnTheLargestFilesFirst() throws {
        let files = (1...10).map { index in
            RepoFile(
                path: "layer0\(index)/tile.mxfp4tile", sizeBytes: UInt64(index) * 100,
                digest: .sha256(hex: String(repeating: "a", count: 64)), isPayload: true)
        }
        let chosen = try ArtifactVerifier.select(files, for: .spotCheck(maximumBytes: 2500))
        XCTAssertEqual(chosen.map(\.sizeBytes).reduce(0, +), 2500)
        XCTAssertEqual(Set(chosen.map(\.sizeBytes)), [1000, 900, 600])
        XCTAssertThrowsError(
            try ArtifactVerifier.select(files, for: .spotCheck(maximumBytes: 1))) { error in
                XCTAssertEqual(error as? ArtifactVerificationError, .budgetTooSmall(maximumBytes: 1))
            }
        XCTAssertEqual(try ArtifactVerifier.select(files, for: .full).count, 10)
    }

    func testAFullCheckIncludesPublishedMetadata() throws {
        let files = [
            RepoFile(path: "README.md", sizeBytes: 10, digest: .gitBlobSHA1(hex: "ab"), isPayload: false),
            RepoFile(path: "global/a.bin", sizeBytes: 10, digest: .sha256(hex: "cd"), isPayload: true),
        ]
        XCTAssertEqual(
            try ArtifactVerifier.select(files, for: .full).map(\.path),
            ["README.md", "global/a.bin"])
    }

    // MARK: - The questions the UI asks

    func testInstalledAndRunnableHereAreDifferentQuestions() throws {
        try makeArtifact(named: "k3-artifact", index: Self.k3Index)
        let mounted = locator().scan(root)
        let away = LocationScan(
            rootPath: "/Volumes/Elsewhere", displayName: "Elsewhere", isMounted: false,
            storageKey: nil,
            artifacts: [
                DiscoveredArtifact(
                    rootPath: "/Volumes/Elsewhere/v4-artifact",
                    locationPath: "/Volumes/Elsewhere", index: .unreadable,
                    model: .deepseekV4Flash, displayName: "DeepSeek V4 Flash 0731",
                    bytesOnDisk: 1, fileCount: 1, expectedFileCount: nil, expectedBytes: nil,
                    verification: .unverified, verifiedAt: nil, scannedAt: Date())
            ],
            unreadable: [:], scannedAt: Date())

        let report = DiscoveryReport(locations: [mounted, away], scannedAt: Date())
        XCTAssertTrue(report.isInstalled(.kimiK3))
        XCTAssertTrue(report.isRunnableHere(.kimiK3))
        // On the drawer drive: installed, and not runnable from here.
        XCTAssertTrue(report.isInstalled(.deepseekV4Flash))
        XCTAssertFalse(report.isRunnableHere(.deepseekV4Flash))
        XCTAssertFalse(report.isInstalled(.minimaxH3))
    }

    // MARK: - The real drive

    /// Scans the actual artifacts, READ-ONLY, when the drive is present and the
    /// operator asked. Off by default: it walks 1.5 TB worth of directory
    /// entries, which is metadata work but not work a unit-test run should do
    /// unasked, and it fails on any machine that is not this one.
    ///
    /// The `TEST_RUNNER_` prefix is not decoration: `xcodebuild` starts the
    /// test host with its own environment, and a plain `MINIRUN_SCAN_K3NVME=1`
    /// reaches the build and not the process running this method — the test
    /// skips and the run still says it succeeded.
    ///
    ///     TEST_RUNNER_MINIRUN_SCAN_K3NVME=1 xcodebuild test \
    ///         -scheme minirun-Package -destination 'platform=macOS' \
    ///         -only-testing:MinirunKitTests/ArtifactDiscoveryTests
    ///
    /// It reads directory entries and nothing else. The drive is read-only by
    /// instruction and nothing in this file opens a payload file, let alone
    /// writes one.
    func testTheRealDriveScansToThreeIdentifiedArtifacts() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MINIRUN_SCAN_K3NVME"] == "1",
            "set MINIRUN_SCAN_K3NVME=1 with /Volumes/K3NVME mounted")
        let volume = URL(fileURLWithPath: "/Volumes/K3NVME")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: volume.path))

        let scan = locator().scan(volume)
        XCTAssertTrue(scan.isMounted)
        XCTAssertEqual(scan.displayName, "K3NVME")
        XCTAssertEqual(
            Set(scan.artifacts.compactMap(\.model)), [.kimiK3, .deepseekV4Flash, .minimaxH3],
            "the drive holds one artifact per published model and each names its upstream")

        for artifact in scan.artifacts {
            XCTAssertGreaterThan(artifact.fileCount, 0)
            XCTAssertGreaterThan(artifact.bytesOnDisk, 0)
            XCTAssertEqual(artifact.verification, .unverified, "a scan never digests")
            // The file count the payload rule predicts is the count on disk.
            // These three directories are complete downloads; a mismatch here
            // means the classification rule and the drive have parted company.
            XCTAssertEqual(
                artifact.missingFileCount, 0,
                "\(artifact.displayName) holds \(artifact.fileCount) files; "
                    + "the catalog expects \(artifact.expectedFileCount ?? -1)")
        }
    }
}
