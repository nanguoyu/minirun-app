import Foundation
import XCTest

@testable import MinirunKit

/// Resumability across launches is only real if the job record survives being
/// written down and read back — plan, digests, options and per-file state
/// included.
final class DownloadStateStoreTests: XCTestCase {

    private var directory: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-state-\(UUID().uuidString)", isDirectory: true)
    }

    override func tearDown() {
        if let directory { try? FileManager.default.removeItem(at: directory) }
        super.tearDown()
    }

    private func makeState(_ job: DownloadJobID = DownloadJobID()) -> DownloadJobState {
        let repo = HuggingFaceRepoRef(
            repoID: "nanguoyu/Kimi-K3-minirun",
            revision: "159987d3ac437e0aceaff0763d43ddeb549b1842")
        let files = [
            RepoFile(
                path: "layer00/layer00-w1.mxfp4tile", sizeBytes: 5_240_799_232,
                digest: .sha256(hex: String(repeating: "a", count: 64)), isPayload: true),
            RepoFile(
                path: "layer00/manifest.json", sizeBytes: 812,
                digest: .gitBlobSHA1(hex: String(repeating: "b", count: 40)), isPayload: false),
            RepoFile(
                path: "README.md", sizeBytes: 40,
                digest: .gitBlobSHA1(hex: String(repeating: "c", count: 40)), isPayload: false),
        ]
        // Deliberately wrong by one file and one byte, so the round trip has a
        // disagreement to carry rather than an agreement to lose quietly.
        let claim = IndexClaim(
            declaredFileCount: 2, declaredBytes: 5_240_799_233,
            sourceRepo: "moonshotai/Kimi-K3", sourceRevision: String(repeating: "d", count: 40))
        let plan = DownloadPlan(
            model: .kimiK3, repo: repo, files: files, index: claim,
            reconciliation: IndexReconciliation.between(files: files, claim: claim))
        return DownloadJobState(
            job: job, plan: plan, destinationPath: "/Volumes/NVMe/k3",
            destinationBookmark: Data([0xDE, 0xAD, 0xBE, 0xEF]),
            artifactRootPath: "/Volumes/NVMe/k3", artifactRootDevice: 12,
            artifactRootInode: 34,
            options: try! testOptions().validated(),
            fileStates: [
                "layer00/layer00-w1.mxfp4tile": .partial(bytesOnDisk: 1_073_741_824),
                "layer00/manifest.json": .verified(bytes: 812, at: Date(timeIntervalSince1970: 100)),
                "README.md": .failed(reason: "reset by peer", attempts: 2),
            ], runState: .paused,
            createdAt: Date(timeIntervalSince1970: 1), updatedAt: Date(timeIntervalSince1970: 2))
    }

    func testAJobStateSurvivesAFileRoundTrip() throws {
        let store = FileDownloadStateStore(directory: directory)
        let state = makeState()
        try store.save(state)

        let reread = try XCTUnwrap(FileDownloadStateStore(directory: directory).load(state.job))
        XCTAssertEqual(reread, state)
        XCTAssertEqual(reread.plan.files.count, 3)
        XCTAssertEqual(reread.plan.file(at: "README.md")?.digest.algorithm, .gitBlobSHA1)
        XCTAssertEqual(
            reread.plan.file(at: "layer00/layer00-w1.mxfp4tile")?.digest.algorithm, .sha256)
        XCTAssertEqual(reread.destinationBookmark, Data([0xDE, 0xAD, 0xBE, 0xEF]))
        XCTAssertEqual(reread.artifactRootPath, "/Volumes/NVMe/k3")
        XCTAssertEqual(reread.artifactRootDevice, 12)
        XCTAssertEqual(reread.artifactRootInode, 34)
        XCTAssertEqual(reread.runState, .paused)
        XCTAssertEqual(
            reread.fileStates["layer00/layer00-w1.mxfp4tile"],
            .partial(bytesOnDisk: 1_073_741_824))
        XCTAssertEqual(
            reread.fileStates["README.md"], .failed(reason: "reset by peer", attempts: 2))
    }

    func testEveryFileStateRoundTrips() throws {
        let states: [FileState] = [
            .pending,
            .partial(bytesOnDisk: 17),
            .fetched(bytesOnDisk: 4096),
            .verified(bytes: 4096, at: Date(timeIntervalSince1970: 99)),
            .failed(reason: "digest mismatch", attempts: 3),
        ]
        for state in states {
            let data = try MinirunKitJSON.encoder().encode(state)
            XCTAssertEqual(try MinirunKitJSON.decoder().decode(FileState.self, from: data), state)
        }
        XCTAssertEqual(FileState.partial(bytesOnDisk: 17).bytesOnDisk, 17)
        XCTAssertEqual(FileState.pending.bytesOnDisk, 0)
        XCTAssertEqual(FileState.failed(reason: "x", attempts: 1).bytesOnDisk, 0)
    }

    func testTheStoreListsAndRemovesJobs() throws {
        let store = FileDownloadStateStore(directory: directory)
        let first = makeState()
        let second = makeState()
        try store.save(first)
        try store.save(second)
        XCTAssertEqual(Set(try store.all().map(\.job)), [first.job, second.job])

        try store.remove(first.job)
        XCTAssertNil(try store.load(first.job))
        XCTAssertEqual(try store.all().map(\.job), [second.job])
        // Removing something that is not there is not an error: a job may have
        // been cleaned up by an earlier launch.
        XCTAssertNoThrow(try store.remove(first.job))
    }

    func testListingFailsClosedWhenOneJobRecordIsCorrupt() throws {
        let store = FileDownloadStateStore(directory: directory)
        try store.save(makeState())
        try Data("not json".utf8).write(
            to: directory.appendingPathComponent("corrupt.json"))

        XCTAssertThrowsError(try store.all())
    }

    func testLegacyRecordWithoutLifecycleOrRootIdentityStillDecodes() throws {
        let encoded = try MinirunKitJSON.encoder().encode(makeState())
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object.removeValue(forKey: "runState")
        object.removeValue(forKey: "artifactRootPath")
        object.removeValue(forKey: "artifactRootDevice")
        object.removeValue(forKey: "artifactRootInode")
        let legacy = try JSONSerialization.data(withJSONObject: object)

        let decoded = try MinirunKitJSON.decoder().decode(
            DownloadJobState.self, from: legacy)
        XCTAssertNil(decoded.runState)
        XCTAssertNil(decoded.artifactRootPath)
        XCTAssertNil(decoded.artifactRootDevice)
        XCTAssertNil(decoded.artifactRootInode)
    }

    func testTheInMemoryStoreBehavesTheSameWay() throws {
        let store = InMemoryDownloadStateStore()
        let state = makeState()
        XCTAssertNil(try store.load(state.job))
        try store.save(state)
        XCTAssertEqual(try store.load(state.job), state)
        XCTAssertEqual(try store.all().count, 1)
        try store.remove(state.job)
        XCTAssertEqual(try store.all().count, 0)
    }

    /// The reconciliation travels with the plan, so an operator reading a
    /// resumed job still sees that the index and the tree disagreed.
    func testAReconciliationDisagreementIsCarriedInTheSavedPlan() throws {
        let store = FileDownloadStateStore(directory: directory)
        let state = makeState()
        try store.save(state)
        let reread = try XCTUnwrap(store.load(state.job))
        XCTAssertFalse(reread.plan.reconciliation.agrees)
        XCTAssertEqual(reread.plan.reconciliation.payloadFileCount, 1)
        XCTAssertEqual(reread.plan.reconciliation.fileCountDelta, 1)
        XCTAssertEqual(reread.plan.reconciliation.byteDelta, 1)
        XCTAssertNotNil(reread.plan.reconciliation.note)
    }
}
