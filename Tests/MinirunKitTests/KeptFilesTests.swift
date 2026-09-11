import Foundation
import XCTest

@testable import MinirunKit

/// The re-check behind "kept on disk".
///
/// The product used to say "verified and partial files remain on disk for a new
/// job to reuse" from the last progress event it happened to see, so it went on
/// saying it after the operator deleted the directory by hand. These tests are
/// the rule that replaces that memory: a file counts when it is there at its
/// planned size, a `.minirun-part` file counts as partial, and a wrong-size or
/// missing file counts as nothing at all — with no digest read anywhere.
final class KeptFilesTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-kept-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDown() {
        if let root { try? FileManager.default.removeItem(at: root) }
        super.tearDown()
    }

    // MARK: - Fixtures

    private func makePlan() -> DownloadPlan {
        let files = [
            RepoFile(
                path: "layer00/layer00-w1.mxfp4tile", sizeBytes: 4096,
                digest: .sha256(hex: String(repeating: "a", count: 64)), isPayload: true),
            RepoFile(
                path: "layer00/layer00-w2.mxfp4tile", sizeBytes: 2048,
                digest: .sha256(hex: String(repeating: "b", count: 64)), isPayload: true),
            RepoFile(
                path: "layer00/manifest.json", sizeBytes: 64,
                digest: .gitBlobSHA1(hex: String(repeating: "c", count: 40)), isPayload: false),
            RepoFile(
                path: "README.md", sizeBytes: 16,
                digest: .gitBlobSHA1(hex: String(repeating: "d", count: 40)), isPayload: false),
        ]
        return DownloadPlan(
            model: .kimiK3,
            repo: HuggingFaceRepoRef(
                repoID: "nanguoyu/Kimi-K3-minirun",
                revision: "159987d3ac437e0aceaff0763d43ddeb549b1842"),
            files: files, index: nil,
            reconciliation: IndexReconciliation.between(files: files, claim: nil))
    }

    private func write(_ relativePath: String, bytes: Int) throws {
        let url = root.appendingPathComponent(relativePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(repeating: 0x5A, count: bytes).write(to: url)
    }

    // MARK: - The rule

    func testAFileAtItsPlannedSizeCountsAndAWrongSizedOneDoesNot() throws {
        let plan = makePlan()
        try write("layer00/layer00-w1.mxfp4tile", bytes: 4096)
        try write("layer00/layer00-w2.mxfp4tile", bytes: 1000)  // short: not kept
        try write("README.md", bytes: 16)

        let kept = KeptFiles.measuring(plan, in: root)

        XCTAssertEqual(kept.destination, .present)
        XCTAssertEqual(kept.completeFileCount, 2)
        XCTAssertEqual(kept.completeBytes, 4112)
        XCTAssertEqual(kept.partialFileCount, 0)
        XCTAssertEqual(kept.partialBytes, 0)
        XCTAssertEqual(kept.plannedFileCount, 4)
        XCTAssertEqual(kept.plannedBytes, 6224)
        XCTAssertFalse(kept.isEmpty)
        XCTAssertTrue(kept.wasExamined)
    }

    func testAPartFileCountsAsPartialAndNeverAsComplete() throws {
        let plan = makePlan()
        try write("layer00/layer00-w1.mxfp4tile" + DownloadManager.partSuffix, bytes: 1024)
        try write("layer00/manifest.json", bytes: 64)

        let kept = KeptFiles.measuring(plan, in: root)

        XCTAssertEqual(kept.completeFileCount, 1)
        XCTAssertEqual(kept.completeBytes, 64)
        XCTAssertEqual(kept.partialFileCount, 1)
        XCTAssertEqual(kept.partialBytes, 1024)
        XCTAssertEqual(kept.fileCount, 2)
        XCTAssertEqual(kept.bytes, 1088)
    }

    /// A part file for a file that is also present whole is not counted twice.
    func testACompleteFileWinsOverItsLeftoverPartFile() throws {
        let plan = makePlan()
        try write("layer00/layer00-w1.mxfp4tile", bytes: 4096)
        try write("layer00/layer00-w1.mxfp4tile" + DownloadManager.partSuffix, bytes: 4096)

        let kept = KeptFiles.measuring(plan, in: root)

        XCTAssertEqual(kept.completeFileCount, 1)
        XCTAssertEqual(kept.partialFileCount, 0)
        XCTAssertEqual(kept.bytes, 4096)
    }

    func testAnEmptyPartFileIsNotAKeptFile() throws {
        let plan = makePlan()
        try write("layer00/layer00-w1.mxfp4tile" + DownloadManager.partSuffix, bytes: 0)

        let kept = KeptFiles.measuring(plan, in: root)

        XCTAssertTrue(kept.isEmpty)
        XCTAssertEqual(kept.bytes, 0)
    }

    func testADirectoryNamedLikeAPlannedFileIsNotADownloadedFile() throws {
        let plan = makePlan()
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("README.md"), withIntermediateDirectories: true)

        let kept = KeptFiles.measuring(plan, in: root)

        XCTAssertTrue(kept.isEmpty)
    }

    /// The state the owner actually hit: a cancelled job whose files were then
    /// deleted by hand. Nothing is kept, and the directory's absence is a fact
    /// about the drive rather than an unexamined guess.
    func testADeletedDestinationKeepsNothingAndSaysItWasLookedAt() throws {
        let plan = makePlan()
        try write("layer00/layer00-w1.mxfp4tile", bytes: 4096)
        try FileManager.default.removeItem(at: root)

        let kept = KeptFiles.measuring(plan, in: root)

        XCTAssertEqual(kept.destination, .missing)
        XCTAssertTrue(kept.isEmpty)
        XCTAssertEqual(kept.bytes, 0)
        XCTAssertTrue(kept.wasExamined)
        XCTAssertEqual(kept.plannedFileCount, 4)
    }

    func testAnEmptyDestinationDirectoryKeepsNothing() {
        let kept = KeptFiles.measuring(makePlan(), in: root)

        XCTAssertEqual(kept.destination, .present)
        XCTAssertTrue(kept.isEmpty)
    }

    /// A path on a drive that is not in the mount table is unexamined, not
    /// empty: an unplugged drive has lost nothing.
    func testAPathOnAnUnmountedVolumeIsReportedAsSuchRatherThanAsEmpty() {
        let missingVolume = URL(
            fileURLWithPath: "/Volumes/minirun-not-mounted-\(UUID().uuidString)/artifact",
            isDirectory: true)

        let kept = KeptFiles.measuring(makePlan(), in: missingVolume)

        guard case .volumeNotMounted(let volume) = kept.destination else {
            return XCTFail("expected an unmounted volume, got \(kept.destination)")
        }
        XCTAssertTrue(volume.hasPrefix("minirun-not-mounted-"))
        XCTAssertFalse(kept.wasExamined)
        XCTAssertTrue(kept.isEmpty)
    }

    func testAMissingDirectoryOnAMountedVolumeIsMissingRatherThanUnmounted() {
        let kept = KeptFiles.measuring(
            makePlan(), in: root.appendingPathComponent("nowhere", isDirectory: true))

        XCTAssertEqual(kept.destination, .missing)
    }

    // MARK: - Through the protocol

    func testTheManagerReconcilesOverItsOwnValidatedPaths() async throws {
        let plan = makePlan()
        try write("layer00/layer00-w1.mxfp4tile", bytes: 4096)
        try write("layer00/manifest.json" + DownloadManager.partSuffix, bytes: 12)
        let manager = DownloadManager(
            storage: StorageManager(bookmarkLedger: InMemoryBookmarkLedger()),
            stateStore: InMemoryDownloadStateStore())

        let kept = await manager.keptFiles(of: plan, in: root)

        XCTAssertEqual(kept.destination, .present)
        XCTAssertEqual(kept.completeFileCount, 1)
        XCTAssertEqual(kept.completeBytes, 4096)
        XCTAssertEqual(kept.partialFileCount, 1)
        XCTAssertEqual(kept.partialBytes, 12)
    }

    func testTheManagerReportsADeletedDestinationAsHoldingNothing() async throws {
        let plan = makePlan()
        try FileManager.default.removeItem(at: root)
        let manager = DownloadManager(
            storage: StorageManager(bookmarkLedger: InMemoryBookmarkLedger()),
            stateStore: InMemoryDownloadStateStore())

        let kept = await manager.keptFiles(of: plan, in: root)

        XCTAssertEqual(kept.destination, .missing)
        XCTAssertTrue(kept.isEmpty)
    }

    /// The default conformance exists so every stand-in answers the same
    /// question without reimplementing the rule.
    func testTheProtocolDefaultMeasuresTheSameDirectory() async throws {
        struct Stub: DownloadManaging {
            func plan(for model: ModelDescriptor) async throws -> DownloadPlan {
                throw DownloadError.unknownJob(DownloadJobID())
            }
            func preflight(
                _ plan: DownloadPlan, into destination: DownloadDestination,
                options: DownloadOptions
            ) async throws -> DownloadPreflight {
                throw DownloadError.unknownJob(DownloadJobID())
            }
            func start(
                _ plan: DownloadPlan, into destination: DownloadDestination,
                options: DownloadOptions
            ) async throws -> DownloadJobID {
                throw DownloadError.unknownJob(DownloadJobID())
            }
            func events(for job: DownloadJobID) -> AsyncStream<DownloadEvent> {
                AsyncStream { $0.finish() }
            }
            func pause(_ job: DownloadJobID) async throws {}
            func resume(_ job: DownloadJobID, scope: StorageScope?) async throws {}
            func cancel(_ job: DownloadJobID, keepPartialFiles: Bool) async throws {}
            func verify(_ job: DownloadJobID, depth: VerificationDepth) async throws
                -> VerificationReport
            {
                throw DownloadError.unknownJob(job)
            }
            func repair(_ job: DownloadJobID, from report: VerificationReport) async throws
                -> DownloadJobID
            {
                throw DownloadError.unknownJob(job)
            }
            func jobs() async -> [DownloadJobSummary] { [] }
        }

        let plan = makePlan()
        try write("README.md", bytes: 16)

        let kept = await Stub().keptFiles(of: plan, in: root)

        XCTAssertEqual(kept.completeFileCount, 1)
        XCTAssertEqual(kept.completeBytes, 16)
    }
}
