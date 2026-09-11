import Darwin
import Foundation
import XCTest

@testable import MinirunKit

/// Serves one revision's tree and its pinned repository-info sibling set from an
/// in-memory description. Same shape as the transport in the carry-forward
/// suite; each is private to its file.
private struct HandOffTreeTransport: HTTPTransport {
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

private final class DigestLog: @unchecked Sendable {
    private let lock = NSLock()
    private var started: [String] = []

    var digested: [String] {
        lock.lock()
        defer { lock.unlock() }
        return started.sorted()
    }

    func noteStart(_ path: String) {
        lock.lock()
        started.append(path)
        lock.unlock()
    }
}

/// A finished transfer has already read every published file and matched it
/// against the digest the repository publishes. Before ADR 0019 that work was
/// thrown away: the copy appeared as *Not verified* and the page asked for a
/// third full read of the drive — on the owner's 517 GB V4.1 Flash download,
/// half an hour of reading to learn what the transfer had just proved.
///
/// These tests pin what the hand-off may and may not do: it produces the exact
/// record a full ``ArtifactVerifier`` pass produces, it reads no file body to do
/// it, and it refuses — by name, writing nothing — wherever the transfer cannot
/// account for a published file or the volume has moved underneath it.
final class ArtifactDownloadEvidenceTests: XCTestCase {
    private static let revision = String(repeating: "c", count: 40)
    private static let repoID = "fixture/K3-minirun"
    private static let payloadPaths = [
        "layer00/weights.bin", "layer01/weights.bin", "layer02/weights.bin",
    ]
    private static let card = Data("# Kimi K3\n\nA model card.\n".utf8)

    private var location: URL!
    private var artifactURL: URL!

    override func setUpWithError() throws {
        let base = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-download-evidence-\(UUID().uuidString)")
        location = base.appendingPathComponent("location")
        artifactURL = location.appendingPathComponent("artifact")
        try FileManager.default.createDirectory(at: artifactURL, withIntermediateDirectories: true)
        try Data(ArtifactDiscoveryTests.k3Index.utf8).write(
            to: artifactURL.appendingPathComponent("index.json"))
        try Self.card.write(to: artifactURL.appendingPathComponent("README.md"))
        for (offset, path) in Self.payloadPaths.enumerated() {
            let url = artifactURL.appendingPathComponent(path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Self.payloadBytes(offset).write(to: url)
        }
    }

    override func tearDownWithError() throws {
        if let location { try? FileManager.default.removeItem(at: location) }
    }

    // MARK: - The record

    /// The whole claim, in one test: the record the hand-off writes is
    /// byte-for-byte the record a full pass writes, and the hand-off reads no
    /// file body to produce it.
    func testTheRecordIsTheOneAFullPassWouldHaveWrittenAndNoBytesAreRead() async throws {
        let handedOver = InMemoryVerificationLedger()
        let log = DigestLog()
        let record = try await makeVerifier(ledger: handedOver, digestLog: log)
            .recordEvidence(fromDownload: try proof(), index: index())

        XCTAssertEqual(record.state, .fullyVerified)
        XCTAssertEqual(record.rootPath, artifactURL.path)
        XCTAssertEqual(log.digested, [], "the hand-off reads no file body at all")

        let readByAFullPass = InMemoryVerificationLedger()
        let report = try await makeVerifier(ledger: readByAFullPass, digestLog: DigestLog())
            .verify(try discover(ledger: readByAFullPass), .full)
        XCTAssertTrue(report.isComplete)

        let handed = try XCTUnwrap(
            handedOver.carryForwardCandidate(rootPath: artifactURL.path)?.evidence)
        let read = try XCTUnwrap(
            readByAFullPass.carryForwardCandidate(rootPath: artifactURL.path)?.evidence)
        XCTAssertEqual(
            try MinirunKitJSON.encoder(pretty: false).encode(handed),
            try MinirunKitJSON.encoder(pretty: false).encode(read))
    }

    /// The verifier's own reader accepts it, so the model page reads verified
    /// and a chat may open the files — without anybody pressing Verify.
    func testTheLedgerReadsItBackAsFullyVerified() async throws {
        let ledger = InMemoryVerificationLedger()
        XCTAssertEqual(try discover(ledger: ledger).verification, .unverified)

        try await makeVerifier(ledger: ledger, digestLog: DigestLog())
            .recordEvidence(fromDownload: try proof(), index: index())

        let row = try discover(ledger: ledger)
        XCTAssertEqual(row.verification, .fullyVerified)
        XCTAssertNotNil(row.verifiedAt)
        let record = try XCTUnwrap(
            ledger.record(
                matching: ArtifactVerificationLookup(
                    rootPath: artifactURL.path, model: .kimiK3, repository: repository(),
                    index: index(), currentTreeIdentity: treeIdentity())))
        XCTAssertEqual(record.state, .fullyVerified)
        XCTAssertEqual(
            record.detail,
            "every published file matched the digests kimi-k3 publishes, "
                + "read by the transfer that fetched them.")
    }

    /// And it carries forward, which is the point of writing it in the ledger's
    /// own shape rather than in one of its own: a model card rewritten upstream
    /// moves the whole plan identity, and must still cost one file rather than
    /// the artifact (ADR 0014).
    func testARecordedTransferCarriesForwardThroughARepublishedRevision() async throws {
        let ledger = InMemoryVerificationLedger()
        try await makeVerifier(ledger: ledger, digestLog: DigestLog())
            .recordEvidence(fromDownload: try proof(), index: index())

        let newCard = Data("# Kimi K3 for Minirun\n\nRewritten upstream.\n".utf8)
        try newCard.write(to: artifactURL.appendingPathComponent("README.md"))
        let second = String(repeating: "d", count: 40)
        let log = DigestLog()
        let report = try await makeVerifier(
            revision: second, ledger: ledger, digestLog: log
        ).verify(try discover(ledger: ledger, revision: second), .full)

        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(log.digested, ["README.md"], "only the file whose published entry moved")
        let carried = try XCTUnwrap(report.carryForward)
        XCTAssertEqual(carried.sourceState, .fullyVerified)
        XCTAssertEqual(carried.carriedForward, 4)
        XCTAssertEqual(carried.reread, 1)
    }

    // MARK: - Refusals

    /// A file the transfer only size-checked has no digest behind it, and the
    /// ledger is told nothing rather than told a size.
    func testAFileTheTransferNeverDigestedIsRefusedByName() throws {
        let summary = try summary(
            omittingEvidenceFor: ["layer01/weights.bin", "README.md"])
        XCTAssertThrowsError(try DownloadVerificationProof(plan: plan(), summary: summary)) {
            guard case ArtifactDownloadEvidenceError.bytesNeverDigested(let paths) =
                $0
            else { return XCTFail("expected a named refusal, got \($0)") }
            XCTAssertEqual(paths, ["README.md", "layer01/weights.bin"])
        }
    }

    /// A transfer that did not pass hands over nothing at all.
    func testAFailedTransferHandsOverNothing() throws {
        let files = repoFiles()
        let report = VerificationReport(
            job: DownloadJobID(), model: .kimiK3, depth: .digestPayload, checkedAt: Date(),
            ok: files.dropLast().map(\.path), missing: [files.last!.path], wrongSize: [],
            wrongDigest: [], unreadable: [:], extraneous: [])
        let summary = DownloadSummary(
            job: report.job, model: .kimiK3, repo: repository(),
            destinationPath: artifactURL.path, bytesWritten: 0, networkBytes: 0,
            wastedBytes: 0, wallSeconds: 1, meanBytesPerSecond: 0, verification: report,
            filesVerified: [])
        XCTAssertThrowsError(try DownloadVerificationProof(plan: plan(), summary: summary)) {
            guard case ArtifactDownloadEvidenceError.transferDidNotPass = $0 else {
                return XCTFail("expected a named refusal, got \($0)")
            }
        }
    }

    /// A plan that is not the repository's complete published tree cannot earn
    /// a claim about every published file, however well its own files check out.
    func testAPartialPlanIsRefusedRatherThanRecordedAsTheWholeTree() async throws {
        let ledger = InMemoryVerificationLedger()
        let whole = try proof()
        let partial = DownloadVerificationProof(
            model: whole.model, repository: whole.repository, rootPath: whole.rootPath,
            files: whole.files.filter { $0.path != "layer02/weights.bin" },
            checkedAt: whole.checkedAt)
        do {
            try await makeVerifier(ledger: ledger, digestLog: DigestLog())
                .recordEvidence(fromDownload: partial, index: index())
            XCTFail("a partial plan was recorded as a complete verification")
        } catch let error as ArtifactDownloadEvidenceError {
            guard case .planIsNotTheCompleteTree(let reason) = error else {
                return XCTFail("expected a completeness refusal, got \(error)")
            }
            XCTAssertTrue(reason.contains("layer02/weights.bin"), reason)
        }
        XCTAssertEqual(try discover(ledger: ledger).verification, .unverified)
    }

    /// The identity in the proof is the object the closing pass checked. If the
    /// file on disk is no longer that object, the digest the transfer matched is
    /// not a fact about what is there now, and nothing is written.
    func testAFileThatMovedSinceTheClosingPassIsRefused() async throws {
        let ledger = InMemoryVerificationLedger()
        let original = try proof()
        let stale = DownloadVerificationProof(
            model: original.model, repository: original.repository,
            rootPath: original.rootPath,
            files: original.files.map { file in
                guard file.path == "layer01/weights.bin" else { return file }
                let identity = file.filesystem
                return ArtifactVerifiedFile(
                    path: file.path, expectedSizeBytes: file.expectedSizeBytes,
                    digest: file.digest, isPayload: file.isPayload,
                    filesystem: ArtifactFilesystemIdentity(
                        device: identity.device, inode: identity.inode,
                        sizeBytes: identity.sizeBytes,
                        modificationSeconds: identity.modificationSeconds - 60,
                        modificationNanoseconds: identity.modificationNanoseconds,
                        statusChangeSeconds: identity.statusChangeSeconds,
                        statusChangeNanoseconds: identity.statusChangeNanoseconds))
            },
            checkedAt: original.checkedAt)
        do {
            try await makeVerifier(ledger: ledger, digestLog: DigestLog())
                .recordEvidence(fromDownload: stale, index: index())
            XCTFail("a moved object was recorded as verified")
        } catch let error as ArtifactDownloadEvidenceError {
            XCTAssertEqual(error, .identityMoved(path: "layer01/weights.bin"))
        }
        XCTAssertEqual(try discover(ledger: ledger).verification, .unverified)
    }

    /// Durable dev/inode evidence is meaningless on a volume that recycles
    /// object ids, and the verifier refuses to record it. So does the hand-off.
    func testAVolumeWithoutPersistentIdentifiersIsRefused() async throws {
        let ledger = InMemoryVerificationLedger()
        let verifier = ArtifactVerifier(
            tree: HuggingFaceTreeClient(
                transport: HandOffTreeTransport(body: treeBody(repoFiles())),
                endpoint: URL(string: "https://hand-off.invalid")!),
            catalog: catalog(), ledger: ledger, checkpoints: nil,
            hooks: ArtifactVerifierHooks(
                volumeSupportsPersistentIDs: { _ in false },
                descriptorSupportsPersistentIDs: { _, _ in false }))
        do {
            try await verifier.recordEvidence(fromDownload: try proof(), index: index())
            XCTFail("evidence was recorded on a volume with no persistent identifiers")
        } catch let error as ArtifactVerificationError {
            guard case .persistentFileIDsUnavailable = error else {
                return XCTFail("expected a persistent-id refusal, got \(error)")
            }
        }
        XCTAssertEqual(try discover(ledger: ledger).verification, .unverified)
    }

    /// The record is filed against the revision whose digests were matched. A
    /// catalog that has moved on since the transfer started does not get to
    /// reinterpret those bytes here.
    func testACatalogThatMovedOnIsRefusedRatherThanReinterpreted() async throws {
        let ledger = InMemoryVerificationLedger()
        let moved = String(repeating: "e", count: 40)
        let verifier = ArtifactVerifier(
            tree: HuggingFaceTreeClient(
                transport: HandOffTreeTransport(body: treeBody(repoFiles())),
                endpoint: URL(string: "https://hand-off.invalid")!),
            catalog: catalog(revision: moved), ledger: ledger, checkpoints: nil,
            hooks: ArtifactVerifierHooks(
                volumeSupportsPersistentIDs: { _ in true },
                descriptorSupportsPersistentIDs: { _, _ in true }))
        do {
            try await verifier.recordEvidence(fromDownload: try proof(), index: index())
            XCTFail("evidence was recorded against a revision nobody read")
        } catch let error as ArtifactDownloadEvidenceError {
            guard case .repositoryMoved = error else {
                return XCTFail("expected a revision refusal, got \(error)")
            }
        }
        XCTAssertEqual(try discover(ledger: ledger).verification, .unverified)
    }

    // MARK: - Fixture

    private static func payloadBytes(_ offset: Int) -> Data {
        Data(repeating: UInt8(0x40 + offset), count: 4096 + offset)
    }

    private func repository(_ revision: String = ArtifactDownloadEvidenceTests.revision)
        -> HuggingFaceRepoRef
    {
        HuggingFaceRepoRef(repoID: Self.repoID, revision: revision)
    }

    private func repoFiles() -> [RepoFile] {
        let indexData = Data(ArtifactDiscoveryTests.k3Index.utf8)
        let card = (try? Data(contentsOf: artifactURL.appendingPathComponent("README.md")))
            ?? Self.card
        var files = [
            RepoFile(
                path: "index.json", sizeBytes: UInt64(indexData.count),
                digest: .gitBlobSHA1(hex: FileDigestComputer.gitBlobSHA1(of: indexData)),
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
        return files.sorted { $0.path < $1.path }
    }

    private func treeIdentity() -> String {
        ArtifactDigestPlanIdentity.compute(repoFiles())
    }

    private func plan() -> DownloadPlan {
        let files = repoFiles()
        return DownloadPlan(
            model: .kimiK3, repo: repository(), files: files, index: nil,
            reconciliation: IndexReconciliation.between(files: files, claim: nil))
    }

    private func index() -> ArtifactIndexIdentity {
        ArtifactIndexIdentity.parse(Data(ArtifactDiscoveryTests.k3Index.utf8))
    }

    /// The per-file evidence a finished job's closing pass would carry: the
    /// published entry, and the object `fstat` reports for it right now.
    private func verifiedFiles(omitting omitted: Set<String> = []) throws
        -> [ArtifactVerifiedFile]
    {
        try repoFiles().compactMap { file in
            guard !omitted.contains(file.path) else { return nil }
            let descriptor = open(
                artifactURL.appendingPathComponent(file.path).path,
                O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
            XCTAssertGreaterThanOrEqual(descriptor, 0, file.path)
            defer { _ = close(descriptor) }
            return ArtifactVerifiedFile(
                path: file.path, expectedSizeBytes: file.sizeBytes, digest: file.digest,
                isPayload: file.isPayload,
                filesystem: try ArtifactFilesystemEvidence.identity(
                    of: descriptor, path: file.path, expectedDirectory: false))
        }
    }

    private func summary(omittingEvidenceFor omitted: Set<String> = []) throws
        -> DownloadSummary
    {
        let files = repoFiles()
        let job = DownloadJobID()
        let report = VerificationReport(
            job: job, model: .kimiK3, depth: .digestPayload,
            checkedAt: Date(timeIntervalSince1970: 1_786_000_000),
            ok: files.map(\.path), missing: [], wrongSize: [], wrongDigest: [],
            unreadable: [:], extraneous: [])
        return DownloadSummary(
            job: job, model: .kimiK3, repo: repository(),
            destinationPath: artifactURL.path,
            bytesWritten: files.reduce(0) { $0 + $1.sizeBytes },
            networkBytes: 0, wastedBytes: 0, wallSeconds: 1, meanBytesPerSecond: 0,
            verification: report,
            filesVerified: try verifiedFiles(omitting: omitted))
    }

    private func proof() throws -> DownloadVerificationProof {
        try DownloadVerificationProof(plan: plan(), summary: try summary())
    }

    private func catalog(revision: String = ArtifactDownloadEvidenceTests.revision)
        -> ModelCatalogSnapshot
    {
        let files = repoFiles()
        let payload = files.filter(\.isPayload)
        let metadata = files.filter { !$0.isPayload }
        let base = ModelCatalog.bundled.descriptor(.kimiK3)!
        let descriptor = ModelDescriptor(
            id: base.id, displayName: base.displayName, architecture: base.architecture,
            layout: base.layout, source: .huggingFaceRepo(repository(revision)),
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

    private func makeVerifier(
        revision: String = ArtifactDownloadEvidenceTests.revision,
        ledger: any ArtifactVerificationLedger, digestLog: DigestLog
    ) -> ArtifactVerifier {
        ArtifactVerifier(
            tree: HuggingFaceTreeClient(
                transport: HandOffTreeTransport(body: treeBody(repoFiles())),
                endpoint: URL(string: "https://hand-off.invalid")!),
            catalog: catalog(revision: revision), ledger: ledger, checkpoints: nil,
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
        ledger: any ArtifactVerificationLedger,
        revision: String = ArtifactDownloadEvidenceTests.revision
    ) throws -> DiscoveredArtifact {
        let artifacts = ArtifactLocator(
            catalog: catalog(revision: revision), verificationLedger: ledger
        ).scan(location).artifacts
        XCTAssertEqual(artifacts.count, 1)
        return try XCTUnwrap(artifacts.first)
    }
}
