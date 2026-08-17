import Darwin
import Foundation
import XCTest

@testable import MinirunKit

/// Serves one revision's tree, and its pinned repository-info sibling set, from
/// an in-memory description. Same shape as the transports in the evidence and
/// resume suites; each is private to its file.
private struct CarryForwardTreeTransport: HTTPTransport {
    let body: Data

    func data(for request: HTTPRequest) async throws -> HTTPResponse {
        if request.url.path.contains("/revision/") {
            let entries = (try? JSONSerialization.jsonObject(with: body) as? [[String: Any]]) ?? []
            let siblings = entries.compactMap { entry -> [String: String]? in
                guard let path = entry["path"] as? String else { return nil }
                return ["rfilename": path]
            }
            let revision = request.url.path.split(separator: "/").last.map(String.init) ?? ""
            return HTTPResponse(
                statusCode: 200, headers: [:],
                body: try JSONSerialization.data(withJSONObject: [
                    "sha": revision, "siblings": siblings,
                ]))
        }
        return HTTPResponse(statusCode: 200, headers: [:], body: body)
    }

    func download(_ request: HTTPRequest, into sink: HTTPDownloadSink) async throws
        -> HTTPResponse
    {
        throw HTTPTransportError.notHTTP(url: request.url.absoluteString)
    }
}

/// Records which files a pass actually digested, and can stop it at a chosen
/// file boundary.
private final class DigestLog: @unchecked Sendable {
    private let lock = NSLock()
    private var started: [String] = []
    private let pauseAfterStarts: Int?
    private let reached = DispatchSemaphore(value: 0)
    private let resumeHash = DispatchSemaphore(value: 0)

    init(pauseAfterStarts: Int? = nil) { self.pauseAfterStarts = pauseAfterStarts }

    var digested: [String] {
        lock.lock()
        defer { lock.unlock() }
        return started.sorted()
    }

    func noteStart(_ path: String) {
        lock.lock()
        started.append(path)
        let shouldPause = started.count == pauseAfterStarts
        lock.unlock()
        if shouldPause {
            reached.signal()
            resumeHash.wait()
        }
    }

    func waitUntilPaused() async {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                self.reached.wait()
                continuation.resume()
            }
        }
    }

    func resume() { resumeHash.signal() }
}

/// A model card edited upstream moves the pinned commit, and therefore the
/// whole digest plan identity, without moving one payload byte. Before ADR 0014
/// that cost a K3 owner 1.56 TB of reading and — if their own copy of the card
/// was stale — their evidence as well.
///
/// These tests pin what carry-forward may and may not do: it spends an earlier
/// revision's per-file evidence exactly where the published digest, the
/// published size and the local object are all unchanged, it reads everything
/// else, and it never turns a mismatch into a pass.
final class ArtifactVerificationCarryForwardTests: XCTestCase {
    private static let firstRevision = String(repeating: "a", count: 40)
    private static let secondRevision = String(repeating: "b", count: 40)
    private static let repoID = "fixture/K3-minirun"
    private static let payloadPaths = [
        "layer00/weights.bin", "layer01/weights.bin", "layer02/weights.bin",
        "layer03/weights.bin",
    ]
    private static let oldCard = Data("# Kimi K3\n\nA model card.\n".utf8)
    private static let newCard = Data(
        "# Kimi K3 for Minirun\n\nA model card, rewritten, naming the app.\n".utf8)

    private var location: URL!
    private var artifactURL: URL!
    private var defaults: UserDefaults!
    private var defaultsSuite: String!
    private var ledgerPrefix: String!

    override func setUpWithError() throws {
        let base = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-carry-forward-\(UUID().uuidString)")
        location = base.appendingPathComponent("location")
        artifactURL = location.appendingPathComponent("artifact")
        try FileManager.default.createDirectory(at: location, withIntermediateDirectories: true)
        defaultsSuite = "minirun.verification.carry.\(UUID().uuidString)"
        defaults = try XCTUnwrap(UserDefaults(suiteName: defaultsSuite))
        ledgerPrefix = "ledger.\(UUID().uuidString)."
        try install()
    }

    override func tearDownWithError() throws {
        if let location { try? FileManager.default.removeItem(at: location) }
        if let defaultsSuite { defaults?.removePersistentDomain(forName: defaultsSuite) }
    }

    // MARK: - The incident

    func testARewrittenModelCardReadsOneFileAndCarriesTheRestForward() async throws {
        let ledger = makeLedger()
        try await verifyAtFirstRevision(ledger: ledger)

        // Upstream rewrote README.md and the operator's copy followed. Nothing
        // else in either tree moved.
        try write(Self.newCard, to: "README.md")
        let second = revisionTwo()
        let artifact = try discover(catalog: second.catalog, ledger: ledger)
        XCTAssertEqual(
            artifact.verification, .unverified,
            "a record about the previous revision is withheld, exactly as before")

        let log = DigestLog()
        let report = try await makeVerifier(
            revision: second, ledger: ledger, checkpoints: nil, digestLog: log
        ).verify(artifact, .full)

        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(log.digested, ["README.md"], "only the file whose published entry moved")
        let carried = try XCTUnwrap(report.carryForward)
        XCTAssertEqual(carried.sourceRevision, Self.firstRevision)
        XCTAssertEqual(carried.sourceState, .fullyVerified)
        XCTAssertEqual(carried.carriedForward, 5)
        XCTAssertEqual(carried.reread, 1)
        XCTAssertEqual(carried.newFiles, 0)
        XCTAssertEqual(carried.removedFiles, 0)
        XCTAssertEqual(
            ArtifactVerifier.sentence(report, request: .full, selected: 6),
            "every published file matched the digests kimi-k3 publishes. "
                + "1 file re-read, 5 carried forward from the previous revision.")

        XCTAssertEqual(
            try discover(catalog: second.catalog, ledger: ledger).verification, .fullyVerified)
        // And the record it wrote is the record a pass that read every byte
        // would have written. Carry-forward changes what was read, not what is
        // remembered.
        let carriedEvidence = try XCTUnwrap(storedEvidence())
        let scratch = InMemoryVerificationLedger()
        _ = try await makeVerifier(
            revision: second, ledger: scratch, checkpoints: nil, digestLog: DigestLog()
        ).verify(artifact, .full, resumption: .fromScratch)
        let reference = try XCTUnwrap(
            scratch.carryForwardCandidate(rootPath: artifact.rootPath)?.evidence)
        XCTAssertEqual(
            try MinirunKitJSON.encoder(pretty: false).encode(carriedEvidence),
            try MinirunKitJSON.encoder(pretty: false).encode(reference))
    }

    func testAChangedPayloadIsReadAgainAndAMismatchLeavesTheEarlierRecordStanding()
        async throws
    {
        let ledger = makeLedger()
        try await verifyAtFirstRevision(ledger: ledger)
        let before = try XCTUnwrap(storedRecordData())

        // The repository republishes one payload. The local copy is still the
        // old bytes at the old length, which is the failure this must catch.
        let changed = Self.payloadPaths[1]
        var files = repoFiles(card: Self.oldCard)
        files = files.map { file in
            guard file.path == changed else { return file }
            return RepoFile(
                path: file.path, sizeBytes: file.sizeBytes,
                digest: .sha256(hex: FileDigestComputer.sha256(of: Data(repeating: 0x7A, count: Int(file.sizeBytes)))),
                isPayload: true)
        }
        let second = revision(Self.secondRevision, files: files)
        let artifact = try discover(catalog: second.catalog, ledger: ledger)

        let log = DigestLog()
        let report = try await makeVerifier(
            revision: second, ledger: ledger, checkpoints: nil, digestLog: log
        ).verify(artifact, .full)

        XCTAssertFalse(report.isComplete)
        XCTAssertEqual(report.wrongDigest.map(\.path), [changed])
        XCTAssertEqual(log.digested, [changed], "a refusal is still reached by reading")
        XCTAssertEqual(
            storedRecordData(), before,
            "a failure about the new revision does not destroy the record about the old one")
        XCTAssertEqual(
            try discover(catalog: second.catalog, ledger: ledger).verification, .unverified,
            "the retained record is still withheld from the revision it does not answer")
        XCTAssertEqual(
            try discover(catalog: revisionOne().catalog, ledger: ledger).verification,
            .fullyVerified,
            "and still answers the revision it does describe")
    }

    func testAFileAddedAtTheNewRevisionIsRead() async throws {
        let ledger = makeLedger()
        try await verifyAtFirstRevision(ledger: ledger)

        let notice = Data("Third-party notices.\n".utf8)
        try write(notice, to: "NOTICE.md")
        var files = repoFiles(card: Self.oldCard)
        files.append(
            RepoFile(
                path: "NOTICE.md", sizeBytes: UInt64(notice.count),
                digest: .gitBlobSHA1(hex: FileDigestComputer.gitBlobSHA1(of: notice)),
                isPayload: false))
        let second = revision(Self.secondRevision, files: files)
        let artifact = try discover(catalog: second.catalog, ledger: ledger)

        let log = DigestLog()
        let report = try await makeVerifier(
            revision: second, ledger: ledger, checkpoints: nil, digestLog: log
        ).verify(artifact, .full)

        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(log.digested, ["NOTICE.md"])
        let carried = try XCTUnwrap(report.carryForward)
        XCTAssertEqual(carried.newFiles, 1)
        XCTAssertEqual(carried.carriedForward, 6)
        XCTAssertEqual(carried.removedFiles, 0)
    }

    func testAFileRemovedAtTheNewRevisionIsDroppedFromThePlanAndNothingIsRead() async throws {
        let ledger = makeLedger()
        try await verifyAtFirstRevision(ledger: ledger)

        let dropped = Self.payloadPaths[3]
        let files = repoFiles(card: Self.oldCard).filter { $0.path != dropped }
        let second = revision(Self.secondRevision, files: files)
        let artifact = try discover(catalog: second.catalog, ledger: ledger)

        let log = DigestLog()
        let report = try await makeVerifier(
            revision: second, ledger: ledger, checkpoints: nil, digestLog: log
        ).verify(artifact, .full)

        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(log.digested, [], "a pass may legitimately read nothing at all")
        XCTAssertFalse(report.ok.contains(dropped), "an unpublished file is not in the plan")
        let carried = try XCTUnwrap(report.carryForward)
        XCTAssertEqual(carried.carriedForward, 5)
        XCTAssertEqual(carried.reread, 0)
        XCTAssertEqual(carried.removedFiles, 1)
        XCTAssertEqual(
            try discover(catalog: second.catalog, ledger: ledger).verification, .fullyVerified)
    }

    func testAPhysicallyChangedFileIsNeverCarriedEvenWhenItsBytesStillMatch() async throws {
        let ledger = makeLedger()
        try await verifyAtFirstRevision(ledger: ledger)

        // Same content, same length, same inode — only the timestamps move.
        // The digest would match, and that is not the point: the recorded
        // identity no longer describes this object, so it is read again.
        let touched = Self.payloadPaths[2]
        let handle = try FileHandle(forWritingTo: artifactURL.appendingPathComponent(touched))
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: Self.payloadBytes(2).prefix(1))
        try handle.synchronize()
        try handle.close()

        try write(Self.newCard, to: "README.md")
        let second = revisionTwo()
        let artifact = try discover(catalog: second.catalog, ledger: ledger)

        let log = DigestLog()
        let report = try await makeVerifier(
            revision: second, ledger: ledger, checkpoints: nil, digestLog: log
        ).verify(artifact, .full)

        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(log.digested, ["README.md", touched])
        XCTAssertEqual(try XCTUnwrap(report.carryForward).carriedForward, 4)
    }

    func testEvidencePredatingTheCurrentSchemaCarriesNothing() async throws {
        let ledger = makeLedger()
        try await verifyAtFirstRevision(ledger: ledger)
        let current = try XCTUnwrap(storedEvidence())

        // A record written before the current evidence authority is not trusted
        // for a state, and it is not trusted for a byte either.
        let downgraded = ArtifactVerificationEvidence(
            schemaVersion: ArtifactVerificationEvidence.currentSchemaVersion - 1,
            model: current.model, repository: current.repository,
            treeIdentity: current.treeIdentity,
            completenessAuthorityIdentity: current.completenessAuthorityIdentity,
            selectedPlanIdentity: current.selectedPlanIdentity,
            treeFileCount: current.treeFileCount,
            treePayloadFileCount: current.treePayloadFileCount,
            treeMetadataFileCount: current.treeMetadataFileCount,
            treeBytes: current.treeBytes, treePayloadBytes: current.treePayloadBytes,
            treeMetadataBytes: current.treeMetadataBytes,
            selectedBytes: current.selectedBytes, index: current.index,
            persistentVolume: current.persistentVolume, root: current.root,
            files: current.files)
        ledger.record(
            ArtifactVerificationRecord(
                rootPath: artifactURL.path, state: .fullyVerified, checkedAt: Date(),
                detail: "legacy", evidence: downgraded))
        XCTAssertNil(ledger.carryForwardCandidate(rootPath: artifactURL.path))

        try write(Self.newCard, to: "README.md")
        let second = revisionTwo()
        let artifact = try discover(catalog: second.catalog, ledger: ledger)
        let log = DigestLog()
        let report = try await makeVerifier(
            revision: second, ledger: ledger, checkpoints: nil, digestLog: log
        ).verify(artifact, .full)

        XCTAssertTrue(report.isComplete)
        XCTAssertNil(report.carryForward)
        XCTAssertEqual(log.digested.count, 6, "a full pass, exactly as before ADR 0014")
    }

    func testAnInterruptedCarriedPassResumesAndWritesTheSameRecordAsAFreshOne() async throws {
        let ledger = makeLedger()
        try await verifyAtFirstRevision(ledger: ledger)

        // Three files move: the card and two payloads. Nothing else does.
        try write(Self.newCard, to: "README.md")
        let rewritten = [Self.payloadPaths[0], Self.payloadPaths[1]]
        for path in rewritten {
            try write(Data(repeating: 0x11, count: 2048), to: path)
        }
        let second = revision(Self.secondRevision, files: repoFiles(card: Self.newCard))
        let artifact = try discover(catalog: second.catalog, ledger: ledger)
        let checkpoints = InMemoryVerificationCheckpointStore()

        // Path order is README.md, index.json, then the four payloads. Stopping
        // at the third digest start means the card and the first rewritten
        // payload are done and index.json was carried on the way past.
        let interrupted = DigestLog(pauseAfterStarts: 3)
        let firstPass = Task {
            try await makeVerifier(
                revision: second, ledger: ledger, checkpoints: checkpoints,
                digestLog: interrupted
            ).verify(artifact, .full)
        }
        await interrupted.waitUntilPaused()
        firstPass.cancel()
        interrupted.resume()
        do {
            _ = try await firstPass.value
            XCTFail("a cancelled pass reported a result")
        } catch let error as ArtifactVerificationError {
            XCTAssertEqual(error, .cancelled)
        }

        let saved = try XCTUnwrap(checkpoints.checkpoint(matching: checkpointKey(second)))
        XCTAssertEqual(
            saved.files.map(\.path), ["README.md", "index.json", rewritten[0]],
            "a carried file enters the checkpoint beside a digested one")

        let resumed = DigestLog()
        let report = try await makeVerifier(
            revision: second, ledger: ledger, checkpoints: checkpoints, digestLog: resumed
        ).verify(artifact, .full)
        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(
            resumed.digested, [rewritten[1]],
            "the resumed pass reads only what neither the checkpoint nor the record covers")
        XCTAssertNil(
            checkpoints.checkpoint(matching: checkpointKey(second)),
            "a completed pass leaves no progress behind")

        let written = try XCTUnwrap(storedEvidence())
        let scratch = InMemoryVerificationLedger()
        _ = try await makeVerifier(
            revision: second, ledger: scratch, checkpoints: nil, digestLog: DigestLog()
        ).verify(artifact, .full, resumption: .fromScratch)
        XCTAssertEqual(
            try MinirunKitJSON.encoder(pretty: false).encode(written),
            try MinirunKitJSON.encoder(pretty: false).encode(
                try XCTUnwrap(
                    scratch.carryForwardCandidate(rootPath: artifact.rootPath)?.evidence)))
    }

    func testSpotCheckedEvidenceCarriesItsOwnPayloadsAndClaimsNothingAboutTheTree()
        async throws
    {
        let ledger = makeLedger()
        let first = revisionOne()
        let sampled = try await makeVerifier(
            revision: first, ledger: ledger, checkpoints: nil, digestLog: DigestLog()
        ).verify(
            try discover(catalog: first.catalog, ledger: ledger),
            .spotCheck(maximumBytes: 1 << 20))
        XCTAssertTrue(sampled.isComplete)

        try write(Self.newCard, to: "README.md")
        let second = revisionTwo()
        let artifact = try discover(catalog: second.catalog, ledger: ledger)
        let log = DigestLog()
        let report = try await makeVerifier(
            revision: second, ledger: ledger, checkpoints: nil, digestLog: log
        ).verify(artifact, .full)

        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(
            log.digested, ["README.md", "index.json"],
            "a spot check covers no metadata, so the metadata is read")
        let carried = try XCTUnwrap(report.carryForward)
        XCTAssertEqual(carried.sourceState, .spotChecked)
        XCTAssertEqual(carried.carriedForward, 4)
        XCTAssertEqual(carried.newFiles, 0, "a sample describes no complete tree")
        XCTAssertEqual(carried.removedFiles, 0)
        XCTAssertEqual(
            try discover(catalog: second.catalog, ledger: ledger).verification, .fullyVerified)
    }

    func testVerifyingFromScratchCarriesNothing() async throws {
        let ledger = makeLedger()
        try await verifyAtFirstRevision(ledger: ledger)
        try write(Self.newCard, to: "README.md")
        let second = revisionTwo()
        let artifact = try discover(catalog: second.catalog, ledger: ledger)

        let log = DigestLog()
        let report = try await makeVerifier(
            revision: second, ledger: ledger, checkpoints: nil, digestLog: log
        ).verify(artifact, .full, resumption: .fromScratch)
        XCTAssertTrue(report.isComplete)
        XCTAssertNil(report.carryForward)
        XCTAssertEqual(log.digested.count, 6, "read it all again means read it all again")
    }

    func testARecordThatAlreadyAnswersThisRevisionIsNotCarried() async throws {
        let ledger = makeLedger()
        try await verifyAtFirstRevision(ledger: ledger)

        let first = revisionOne()
        let artifact = try discover(catalog: first.catalog, ledger: ledger)
        XCTAssertEqual(artifact.verification, .fullyVerified)
        let log = DigestLog()
        let report = try await makeVerifier(
            revision: first, ledger: ledger, checkpoints: nil, digestLog: log
        ).verify(artifact, .full)
        XCTAssertTrue(report.isComplete)
        XCTAssertNil(report.carryForward)
        XCTAssertEqual(
            log.digested.count, 6,
            "verifying an already-verified row is the operator asking for the bytes")
    }

    // MARK: - The real drive

    /// The same contract against a real artifact on a real external volume,
    /// two real published commits, and the real tree API.
    ///
    /// Off by default, because it reads from the drive and talks to Hugging
    /// Face. It is deliberately a **spot check** at both revisions: that is a
    /// bounded, first-class request kind, and it keeps the read to the stated
    /// budget instead of the 167 GB (V4) or 1.56 TB (K3) a full pass costs. The
    /// drive is opened read-only — `O_RDONLY`, `O_NOFOLLOW`, no write path is
    /// reachable from here.
    ///
    ///     TEST_RUNNER_MINIRUN_CARRY_FORWARD_ARTIFACT=/Volumes/K3NVME/v4-artifact \
    ///     TEST_RUNNER_MINIRUN_CARRY_FORWARD_REPO=nanguoyu/DeepSeek-V4-Flash-0731-minirun \
    ///     TEST_RUNNER_MINIRUN_CARRY_FORWARD_FROM=<previous 40-hex commit> \
    ///     TEST_RUNNER_MINIRUN_CARRY_FORWARD_TO=<current 40-hex commit> \
    ///     xcodebuild test -scheme minirun-Package -destination 'platform=macOS' \
    ///       -only-testing:MinirunKitTests/ArtifactVerificationCarryForwardTests/testRealDriveEvidenceSurvivesARepublishedRevision
    func testRealDriveEvidenceSurvivesARepublishedRevision() async throws {
        func setting(_ name: String) -> String? {
            let environment = ProcessInfo.processInfo.environment
            return environment["TEST_RUNNER_" + name] ?? environment[name]
        }
        guard let artifactPath = setting("MINIRUN_CARRY_FORWARD_ARTIFACT"),
            let repoID = setting("MINIRUN_CARRY_FORWARD_REPO"),
            let from = setting("MINIRUN_CARRY_FORWARD_FROM"),
            let to = setting("MINIRUN_CARRY_FORWARD_TO")
        else {
            throw XCTSkip(
                "set MINIRUN_CARRY_FORWARD_{ARTIFACT,REPO,FROM,TO} to check a real artifact")
        }
        let root = URL(fileURLWithPath: artifactPath)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: root.path))

        let client = HuggingFaceTreeClient(transport: URLSessionTransport())
        let before = HuggingFaceRepoRef(repoID: repoID, revision: from)
        let after = HuggingFaceRepoRef(repoID: repoID, revision: to)
        let publishedBefore = try await client.completeTree(in: before).files
        let publishedAfter = try await client.completeTree(in: after).files

        // What actually moved between the two commits, as an observation
        // rather than an assumption about which file it was.
        let entriesBefore = Dictionary(
            publishedBefore.map { ($0.path, $0) }, uniquingKeysWith: { first, _ in first })
        let moved = publishedAfter.filter { entriesBefore[$0.path] != $0 }.map(\.path)
        print(
            "carry-forward drive check: \(publishedBefore.count) → \(publishedAfter.count) "
                + "published files, entries that moved: \(moved)")

        let suite = "minirun.verification.drive.\(UUID().uuidString)"
        let driveDefaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { driveDefaults.removePersistentDomain(forName: suite) }
        let ledger = UserDefaultsVerificationLedger(
            defaults: driveDefaults, prefix: "drive.\(UUID().uuidString).")

        func artifact(under catalog: ModelCatalogSnapshot) throws -> DiscoveredArtifact {
            let scan = ArtifactLocator(catalog: catalog, verificationLedger: ledger)
                .scan(root.deletingLastPathComponent())
            return try XCTUnwrap(
                scan.artifacts.first { $0.rootPath == root.standardizedFileURL.path })
        }
        // The descriptor is derived from each fetched tree. On this path it is
        // only the second consistency check the verifier applies to a tree it
        // already reconciled against the pinned repository-info sibling set,
        // and a shipped catalogue pins neither of these two commits.
        let catalogBefore = Self.catalog(for: publishedBefore, repository: before)
        let catalogAfter = Self.catalog(for: publishedAfter, repository: after)

        let first = try await ArtifactVerifier(
            tree: client, catalog: catalogBefore, ledger: ledger
        ).verify(try artifact(under: catalogBefore), .spotCheck())
        XCTAssertTrue(first.isComplete, "the sampled payloads match the previous commit")
        XCTAssertNil(first.carryForward, "the first pass has nothing to carry")

        let second = try await ArtifactVerifier(
            tree: client, catalog: catalogAfter, ledger: ledger
        ).verify(try artifact(under: catalogAfter), .spotCheck())
        XCTAssertTrue(second.isComplete)
        let carried = try XCTUnwrap(
            second.carryForward, "the record about the previous commit was spent")
        XCTAssertEqual(carried.sourceRevision, from)
        XCTAssertEqual(carried.reread, 0, "a republished model card costs no payload read")
        XCTAssertEqual(carried.carriedForward, first.ok.count)
        XCTAssertEqual(
            try artifact(under: catalogAfter).verification, .spotChecked,
            "and the row reads checked at the revision the catalogue now pins")
    }

    private static func catalog(
        for files: [RepoFile], repository: HuggingFaceRepoRef
    ) -> ModelCatalogSnapshot {
        let payload = files.filter(\.isPayload)
        let metadata = files.filter { !$0.isPayload }
        let base = ModelCatalog.bundled.descriptor(.deepseekV4Flash)!
        let descriptor = ModelDescriptor(
            id: base.id, displayName: base.displayName, architecture: base.architecture,
            layout: base.layout, source: .huggingFaceRepo(repository),
            payloadBytes: payload.reduce(0) { $0 + $1.sizeBytes },
            metadataBytes: metadata.reduce(0) { $0 + $1.sizeBytes },
            payloadFileCount: payload.count, metadataFileCount: metadata.count,
            largestFileBytes: files.map(\.sizeBytes).max() ?? 0,
            minimumBudgetBytes: base.minimumBudgetBytes, runner: base.runner,
            licenseName: base.licenseName,
            licenseAcknowledgementRequired: base.licenseAcknowledgementRequired,
            notes: base.notes)
        return ModelCatalogSnapshot(generatedAt: Date(), origin: .live, models: [descriptor])
    }

    // MARK: - Fixture

    private struct Revision {
        let repository: HuggingFaceRepoRef
        let files: [RepoFile]
        let catalog: ModelCatalogSnapshot
    }

    private func revision(_ commit: String, files: [RepoFile]) -> Revision {
        Revision(
            repository: HuggingFaceRepoRef(repoID: Self.repoID, revision: commit),
            files: files,
            catalog: Self.makeCatalog(files: files, revision: commit))
    }

    private func revisionOne() -> Revision {
        revision(Self.firstRevision, files: repoFiles(card: Self.oldCard))
    }

    private func revisionTwo() -> Revision {
        revision(Self.secondRevision, files: repoFiles(card: Self.newCard))
    }

    private func verifyAtFirstRevision(ledger: any ArtifactVerificationLedger) async throws {
        let first = revisionOne()
        let artifact = try discover(catalog: first.catalog, ledger: ledger)
        let report = try await makeVerifier(
            revision: first, ledger: ledger, checkpoints: nil, digestLog: DigestLog()
        ).verify(artifact, .full)
        XCTAssertTrue(report.isComplete)
        XCTAssertNil(report.carryForward, "the first pass has nothing to carry")
    }

    private func install() throws {
        try FileManager.default.createDirectory(
            at: artifactURL, withIntermediateDirectories: true)
        try Data(ArtifactDiscoveryTests.k3Index.utf8).write(
            to: artifactURL.appendingPathComponent("index.json"))
        try Self.oldCard.write(to: artifactURL.appendingPathComponent("README.md"))
        for (offset, path) in Self.payloadPaths.enumerated() {
            let url = artifactURL.appendingPathComponent(path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Self.payloadBytes(offset).write(to: url)
        }
    }

    private func write(_ data: Data, to path: String) throws {
        let url = artifactURL.appendingPathComponent(path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    }

    private static func payloadBytes(_ offset: Int) -> Data {
        Data(repeating: UInt8(0x40 + offset), count: 4096 + offset)
    }

    /// The published tree, read from the bytes currently on disk for every file
    /// except the card, which the caller names so a revision can differ from
    /// the working copy.
    private func repoFiles(card: Data) -> [RepoFile] {
        let index = Data(ArtifactDiscoveryTests.k3Index.utf8)
        var files = [
            RepoFile(
                path: "index.json", sizeBytes: UInt64(index.count),
                digest: .gitBlobSHA1(hex: FileDigestComputer.gitBlobSHA1(of: index)),
                isPayload: false),
            RepoFile(
                path: "README.md", sizeBytes: UInt64(card.count),
                digest: .gitBlobSHA1(hex: FileDigestComputer.gitBlobSHA1(of: card)),
                isPayload: false),
        ]
        for path in Self.payloadPaths {
            let bytes =
                (try? Data(contentsOf: artifactURL.appendingPathComponent(path))) ?? Data()
            files.append(
                RepoFile(
                    path: path, sizeBytes: UInt64(bytes.count),
                    digest: .sha256(hex: FileDigestComputer.sha256(of: bytes)),
                    isPayload: true))
        }
        return files
    }

    private static func makeCatalog(
        files: [RepoFile], revision: String
    ) -> ModelCatalogSnapshot {
        let base = ModelCatalog.bundled.descriptor(.kimiK3)!
        let payload = files.filter(\.isPayload)
        let metadata = files.filter { !$0.isPayload }
        let descriptor = ModelDescriptor(
            id: base.id, displayName: base.displayName, architecture: base.architecture,
            layout: base.layout,
            source: .huggingFaceRepo(
                HuggingFaceRepoRef(repoID: repoID, revision: revision)),
            payloadBytes: payload.reduce(0) { $0 + $1.sizeBytes },
            metadataBytes: metadata.reduce(0) { $0 + $1.sizeBytes },
            payloadFileCount: payload.count, metadataFileCount: metadata.count,
            largestFileBytes: files.map(\.sizeBytes).max() ?? 0,
            minimumBudgetBytes: base.minimumBudgetBytes, runner: base.runner,
            licenseName: base.licenseName,
            licenseAcknowledgementRequired: base.licenseAcknowledgementRequired,
            notes: base.notes)
        return ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [descriptor])
    }

    private func makeLedger() -> UserDefaultsVerificationLedger {
        UserDefaultsVerificationLedger(defaults: defaults, prefix: ledgerPrefix)
    }

    private func makeVerifier(
        revision: Revision, ledger: any ArtifactVerificationLedger,
        checkpoints: (any ArtifactVerificationCheckpointStore)?, digestLog: DigestLog
    ) -> ArtifactVerifier {
        let tree = HuggingFaceTreeClient(
            transport: CarryForwardTreeTransport(body: treeBody(revision.files)),
            endpoint: URL(string: "https://carry.invalid")!)
        return ArtifactVerifier(
            tree: tree, catalog: revision.catalog, ledger: ledger, checkpoints: checkpoints,
            hooks: ArtifactVerifierHooks(
                digestStarted: { digestLog.noteStart($0) },
                volumeSupportsPersistentIDs: { _ in true },
                descriptorSupportsPersistentIDs: { _, _ in true }))
    }

    private func treeBody(_ files: [RepoFile]) -> Data {
        let entries: [[String: Any]] = files.map { file in
            var entry: [String: Any] = [
                "type": "file", "path": file.path,
                "size": NSNumber(value: file.sizeBytes),
            ]
            switch file.digest {
            case .sha256(let hex):
                entry["lfs"] = ["oid": hex, "size": NSNumber(value: file.sizeBytes)]
            case .gitBlobSHA1(let hex):
                entry["oid"] = hex
            }
            return entry
        }
        return try! JSONSerialization.data(withJSONObject: entries, options: [.sortedKeys])
    }

    private func discover(
        catalog: ModelCatalogSnapshot, ledger: any ArtifactVerificationLedger
    ) throws -> DiscoveredArtifact {
        let artifacts = ArtifactLocator(catalog: catalog, verificationLedger: ledger)
            .scan(location).artifacts
        XCTAssertEqual(artifacts.count, 1)
        return try XCTUnwrap(artifacts.first)
    }

    private func checkpointKey(
        _ revision: Revision
    ) -> ArtifactVerificationCheckpointKey {
        let selected = try! ArtifactVerifier.select(revision.files, for: .full)
        return ArtifactVerificationCheckpointKey(
            rootPath: artifactURL.path, model: .kimiK3, repository: revision.repository,
            treeIdentity: ArtifactDigestPlanIdentity.compute(revision.files),
            selectedPlanIdentity: ArtifactDigestPlanIdentity.compute(selected),
            requestKind: .full)
    }

    private func storedRecordData() -> Data? {
        defaults.data(
            forKey: ledgerPrefix + ArtifactVerificationPathIdentity.canonical(artifactURL.path))
    }

    private func storedEvidence() throws -> ArtifactVerificationEvidence? {
        guard let data = storedRecordData() else { return nil }
        return try MinirunKitJSON.decoder()
            .decode(ArtifactVerificationRecord.self, from: data).evidence
    }
}
