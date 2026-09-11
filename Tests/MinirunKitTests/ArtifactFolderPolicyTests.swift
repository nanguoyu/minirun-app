import Foundation
import XCTest

@testable import MinirunKit

/// The rule that keeps a 517 GB repository out of a drive's root.
///
/// The shipped behaviour wrote the repository files straight into the folder
/// the operator confirmed, so picking the volume `/Volumes/K3NVME` put
/// `index.json`, `LICENSE` and `layer00/` beside his own directories. These
/// tests pin the replacement: the picked folder is the parent, the artifact
/// gets a folder named exactly as the repository is published, and each of the
/// four states that folder can be in decides something different.
final class ArtifactFolderPolicyTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-folder-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDown() {
        if let root { try? FileManager.default.removeItem(at: root) }
        super.tearDown()
    }

    // MARK: - Fixtures

    private func makePlan(
        model: ModelID = .kimiK3, repoID: String = "nanguoyu/Kimi-K3-minirun"
    ) -> DownloadPlan {
        let files = [
            RepoFile(
                path: "layer00/layer00-w1.mxfp4tile", sizeBytes: 4096,
                digest: .sha256(hex: String(repeating: "a", count: 64)), isPayload: true),
            RepoFile(
                path: "layer00/manifest.json", sizeBytes: 64,
                digest: .gitBlobSHA1(hex: String(repeating: "c", count: 40)), isPayload: false),
            RepoFile(
                path: "index.json", sizeBytes: 128,
                digest: .gitBlobSHA1(hex: String(repeating: "d", count: 40)), isPayload: false),
            RepoFile(
                path: "LICENSE", sizeBytes: 16,
                digest: .gitBlobSHA1(hex: String(repeating: "e", count: 40)), isPayload: false),
        ]
        return DownloadPlan(
            model: model,
            repo: HuggingFaceRepoRef(
                repoID: repoID, revision: "159987d3ac437e0aceaff0763d43ddeb549b1842"),
            files: files, index: nil,
            reconciliation: IndexReconciliation.between(files: files, claim: nil))
    }

    /// Shaped like the documents `Tools/*/build_*_artifact.py` write: the
    /// upstream repository under `source_repo`, its commit under
    /// `source_revision`, and the totals beside them.
    private func writeIndex(
        in directory: URL, sourceRepo: String,
        sourceRevision: String = String(repeating: "1", count: 40)
    ) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let document: [String: Any] = [
            "format": "quantized_tile_container",
            "source_repo": sourceRepo,
            "source_revision": sourceRevision,
            "files": 372,
            "bytes": 17_213_128_704,
        ]
        let data = try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys])
        try data.write(to: directory.appendingPathComponent("index.json"))
    }

    private func makeFile(_ url: URL, bytes: Int = 8) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(repeating: 0x5A, count: bytes).write(to: url)
    }

    // MARK: - The name

    func testTheSubfolderIsTheRepositoryNameExactlyAsPublished() {
        let published = [
            "nanguoyu/DeepSeek-V4.1-Flash-minirun": "DeepSeek-V4.1-Flash-minirun",
            "nanguoyu/Kimi-K3-minirun": "Kimi-K3-minirun",
            "nanguoyu/DeepSeek-V4-Flash-0731-minirun": "DeepSeek-V4-Flash-0731-minirun",
            "nanguoyu/MiniMax-H3-minirun": "MiniMax-H3-minirun",
        ]
        for (repoID, name) in published {
            XCTAssertEqual(ArtifactFolderPolicy.subfolderName(forRepositoryID: repoID), name)
        }
    }

    func testANameIsNotDerivedFromARepositoryIDThisPackageWouldRefuseInAURL() {
        for repoID in [
            "nanguoyu", "nanguoyu/a/b", "nanguoyu/..", "nanguoyu/", "/Kimi-K3-minirun",
            "nanguoyu/Kimi K3", "nanguoyu/Kimi%2FK3",
        ] {
            XCTAssertNil(
                ArtifactFolderPolicy.subfolderName(forRepositoryID: repoID),
                "\(repoID) should publish no folder name")
        }
    }

    // MARK: - Where the bytes go

    func testAPickedVolumeBecomesTheParentAndTheArtifactGetsItsOwnFolder() throws {
        let plan = makePlan()
        let layout = try XCTUnwrap(ArtifactFolderPolicy.resolve(picked: root, plan: plan))

        XCTAssertEqual(layout.parent.path, root.standardizedFileURL.path)
        XCTAssertEqual(
            layout.directory.path,
            root.standardizedFileURL.appendingPathComponent("Kimi-K3-minirun").path)
        XCTAssertEqual(layout.subfolderName, "Kimi-K3-minirun")
        XCTAssertEqual(layout.state, .absent)
        XCTAssertEqual(
            layout.previewSentence, "Will be saved to " + layout.directory.path)
        XCTAssertTrue(layout.chooserMessage.contains("Kimi-K3-minirun"))
    }

    /// *Continue with kept files* re-enters the picker with the job's recorded
    /// destination, which already is the artifact folder.
    func testAnArtifactFolderPickedDirectlyDoesNotNestASecondOneInsideItself() throws {
        let plan = makePlan()
        let artifact = root.appendingPathComponent("Kimi-K3-minirun", isDirectory: true)
        try FileManager.default.createDirectory(at: artifact, withIntermediateDirectories: true)

        let layout = try XCTUnwrap(ArtifactFolderPolicy.resolve(picked: artifact, plan: plan))

        XCTAssertEqual(layout.directory.path, artifact.standardizedFileURL.path)
        XCTAssertEqual(layout.parent.path, root.standardizedFileURL.path)
        XCTAssertEqual(layout.state, .empty)
    }

    func testAModelWithNoPublishableRepositoryNameResolvesNoLayout() {
        let plan = makePlan(repoID: "nanguoyu")
        XCTAssertNil(ArtifactFolderPolicy.resolve(picked: root, plan: plan))
    }

    // MARK: - The four states

    func testAFolderThatDoesNotExistIsAbsent() {
        let plan = makePlan()
        XCTAssertEqual(
            ArtifactFolderPolicy.classify(
                root.appendingPathComponent("Kimi-K3-minirun"), plan: plan),
            .absent)
    }

    func testAnEmptyFolderIsReused() throws {
        let plan = makePlan()
        let artifact = root.appendingPathComponent("Kimi-K3-minirun", isDirectory: true)
        try FileManager.default.createDirectory(at: artifact, withIntermediateDirectories: true)

        XCTAssertEqual(ArtifactFolderPolicy.classify(artifact, plan: plan), .empty)
    }

    func testAFolderHoldingOnlyFinderDebrisIsStillEmpty() throws {
        let plan = makePlan()
        let artifact = root.appendingPathComponent("Kimi-K3-minirun", isDirectory: true)
        try makeFile(artifact.appendingPathComponent(".DS_Store"))

        XCTAssertEqual(ArtifactFolderPolicy.classify(artifact, plan: plan), .empty)
    }

    func testAFolderWhoseIndexNamesThisArtifactsUpstreamIsReusable() throws {
        let plan = makePlan()
        let artifact = root.appendingPathComponent("Kimi-K3-minirun", isDirectory: true)
        try writeIndex(in: artifact, sourceRepo: "moonshotai/Kimi-K3")
        try makeFile(artifact.appendingPathComponent("layer00/layer00-w1.mxfp4tile"), bytes: 4096)

        XCTAssertEqual(ArtifactFolderPolicy.classify(artifact, plan: plan), .reusable)
    }

    func testAFolderWhoseIndexNamesThePublishingRepositoryIsReusable() throws {
        let plan = makePlan()
        let artifact = root.appendingPathComponent("Kimi-K3-minirun", isDirectory: true)
        try writeIndex(in: artifact, sourceRepo: "nanguoyu/Kimi-K3-minirun")
        try makeFile(artifact.appendingPathComponent("holiday-photo.heic"))

        XCTAssertEqual(ArtifactFolderPolicy.classify(artifact, plan: plan), .reusable)
    }

    func testAHalfFinishedTransferWithNoIndexYetIsStillItsOwnFolder() throws {
        let plan = makePlan()
        let artifact = root.appendingPathComponent("Kimi-K3-minirun", isDirectory: true)
        try makeFile(artifact.appendingPathComponent("layer00/layer00-w1.mxfp4tile"), bytes: 1024)
        try makeFile(
            artifact.appendingPathComponent("index.json" + DownloadManager.partSuffix))

        XCTAssertEqual(ArtifactFolderPolicy.classify(artifact, plan: plan), .reusable)
    }

    func testAFolderOfNothingButPartFilesIsReusableEvenBeforeAPlanExists() throws {
        let artifact = root.appendingPathComponent("Kimi-K3-minirun", isDirectory: true)
        try makeFile(
            artifact.appendingPathComponent("index.json" + DownloadManager.partSuffix))

        XCTAssertEqual(
            ArtifactFolderPolicy.classify(
                artifact,
                repository: HuggingFaceRepoRef(
                    repoID: "nanguoyu/Kimi-K3-minirun",
                    revision: "159987d3ac437e0aceaff0763d43ddeb549b1842"),
                model: .kimiK3),
            .reusable)
    }

    func testAFolderHoldingSomethingElseIsRefusedByName() throws {
        let plan = makePlan()
        let artifact = root.appendingPathComponent("Kimi-K3-minirun", isDirectory: true)
        try makeFile(artifact.appendingPathComponent("tax-return-2025.pdf"))

        let state = ArtifactFolderPolicy.classify(artifact, plan: plan)

        XCTAssertEqual(
            state,
            .occupied(
                reason: artifact.standardizedFileURL.path
                    + " already exists and holds something else"))
        XCTAssertFalse(state.allowsWriting)
    }

    func testAFolderHoldingAnotherModelsArtifactSaysWhichOne() throws {
        let plan = makePlan()
        let artifact = root.appendingPathComponent("Kimi-K3-minirun", isDirectory: true)
        try writeIndex(in: artifact, sourceRepo: "deepseek-ai/DeepSeek-V4.1-Flash")

        let refusal = try XCTUnwrap(
            ArtifactFolderPolicy.classify(artifact, plan: plan).refusal)

        XCTAssertTrue(refusal.contains(artifact.standardizedFileURL.path))
        XCTAssertTrue(refusal.contains("deepseek-ai/DeepSeek-V4.1-Flash"))
    }

    func testASymlinkWhereTheFolderShouldBeIsRefusedRatherThanFollowed() throws {
        let plan = makePlan()
        let elsewhere = root.appendingPathComponent("elsewhere", isDirectory: true)
        try FileManager.default.createDirectory(at: elsewhere, withIntermediateDirectories: true)
        let artifact = root.appendingPathComponent("Kimi-K3-minirun", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: artifact, withDestinationURL: elsewhere)

        let refusal = try XCTUnwrap(
            ArtifactFolderPolicy.classify(artifact, plan: plan).refusal)

        XCTAssertTrue(refusal.contains("symbolic link"))
    }

    func testAFileWhereTheFolderShouldBeIsRefused() throws {
        let plan = makePlan()
        let artifact = root.appendingPathComponent("Kimi-K3-minirun")
        try makeFile(artifact)

        let refusal = try XCTUnwrap(
            ArtifactFolderPolicy.classify(artifact, plan: plan).refusal)

        XCTAssertTrue(refusal.contains("is not a folder"))
    }

    func testTheOwnersDriveRootStopsBeingAWritableDestination() throws {
        // What actually happened: `k3-artifact` was already on the volume and
        // the transfer wrote beside it. The volume is now only ever a parent.
        let plan = makePlan(model: .deepseekV41Flash, repoID: "nanguoyu/DeepSeek-V4.1-Flash-minirun")
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("k3-artifact", isDirectory: true),
            withIntermediateDirectories: true)

        let layout = try XCTUnwrap(ArtifactFolderPolicy.resolve(picked: root, plan: plan))

        XCTAssertEqual(
            layout.directory.lastPathComponent, "DeepSeek-V4.1-Flash-minirun")
        XCTAssertEqual(layout.state, .absent)
        XCTAssertTrue(layout.state.allowsWriting)
    }
}
