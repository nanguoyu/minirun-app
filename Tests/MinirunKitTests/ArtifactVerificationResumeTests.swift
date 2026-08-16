import Darwin
import Foundation
import XCTest

@testable import MinirunKit

/// Serves the fixture tree, and its pinned repository-info sibling set, from
/// one in-memory description. Identical in shape to the transport in
/// `ArtifactVerificationEvidenceTests`; both are private to their file.
private struct ResumeTreeTransport: HTTPTransport {
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

/// Counts which files were actually digested, and stops the pass at a chosen
/// file boundary so an interruption lands where a suspended app's would.
private final class DigestLedger: @unchecked Sendable {
    private let lock = NSLock()
    private var started: [String] = []
    private var pauseAfterStarts: Int?
    private let reached = DispatchSemaphore(value: 0)
    private let resumeHash = DispatchSemaphore(value: 0)

    init(pauseAfterStarts: Int? = nil) { self.pauseAfterStarts = pauseAfterStarts }

    var digestedPaths: [String] {
        lock.lock()
        defer { lock.unlock() }
        return started
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

/// A 1.56 TB pass is half an hour of reading, and iOS suspends the app that
/// started it within seconds of it leaving the screen. These tests pin the
/// contract that makes that survivable: an interrupted pass resumes where it
/// stopped, it resumes only against bytes whose filesystem identity has not
/// moved, and the record it finally writes is the record an uninterrupted pass
/// would have written.
final class ArtifactVerificationResumeTests: XCTestCase {
    private static let revision = String(repeating: "3", count: 40)
    private static let payloadPaths = [
        "layer00/weights.bin", "layer01/weights.bin", "layer02/weights.bin",
        "layer03/weights.bin",
    ]

    private var location: URL!
    private var artifactURL: URL!
    private var defaults: UserDefaults!
    private var defaultsSuite: String!
    private var ledgerPrefix: String!

    override func setUpWithError() throws {
        let base = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-verification-resume-\(UUID().uuidString)")
        location = base.appendingPathComponent("location")
        artifactURL = location.appendingPathComponent("artifact")
        try FileManager.default.createDirectory(at: location, withIntermediateDirectories: true)
        defaultsSuite = "minirun.verification.resume.\(UUID().uuidString)"
        defaults = try XCTUnwrap(UserDefaults(suiteName: defaultsSuite))
        ledgerPrefix = "ledger.\(UUID().uuidString)."
        try install()
    }

    override func tearDownWithError() throws {
        if let location { try? FileManager.default.removeItem(at: location) }
        if let defaultsSuite { defaults?.removePersistentDomain(forName: defaultsSuite) }
    }

    // MARK: - The contract

    func testResumedPassDigestsOnlyTheFilesTheInterruptedPassDidNotFinish() async throws {
        let files = repoFiles()
        let catalog = makeCatalog(files: files)

        let reference = try await uninterruptedEvidence(files: files, catalog: catalog)

        let checkpoints = InMemoryVerificationCheckpointStore()
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)

        // Stop the pass while the third file is being hashed: two files are
        // recorded, the third is not, and the remaining ones are untouched.
        let interrupted = DigestLedger(pauseAfterStarts: 3)
        let firstPass = Task {
            try await makeVerifier(
                catalog: catalog, ledger: ledger, checkpoints: checkpoints, files: files,
                digestLedger: interrupted
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
        XCTAssertEqual(interrupted.digestedPaths.count, 3)
        XCTAssertEqual(
            try discover(catalog: catalog, ledger: ledger).verification, .unverified,
            "partial progress is never a verified row")

        let saved = try XCTUnwrap(
            checkpoints.checkpoint(matching: checkpointKey(artifact: artifact, files: files)))
        XCTAssertEqual(saved.files.count, 2, "only completed digests are checkpointed")

        let resumed = DigestLedger()
        let report = try await makeVerifier(
            catalog: catalog, ledger: ledger, checkpoints: checkpoints, files: files,
            digestLedger: resumed
        ).verify(artifact, .full)
        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(
            resumed.digestedPaths.count, files.count - 2,
            "the two files the interrupted pass finished were not read again")
        XCTAssertEqual(
            Set(resumed.digestedPaths).intersection(saved.files.map(\.path)), [],
            "no checkpointed file was digested a second time")

        let evidence = try XCTUnwrap(storedEvidence(artifact))
        XCTAssertEqual(
            try MinirunKitJSON.encoder(pretty: false).encode(evidence),
            try MinirunKitJSON.encoder(pretty: false).encode(reference),
            "a resumed pass writes the record an uninterrupted pass would have written")
        XCTAssertEqual(
            try discover(catalog: catalog, ledger: ledger).verification, .fullyVerified)
        XCTAssertNil(
            checkpoints.checkpoint(matching: checkpointKey(artifact: artifact, files: files)),
            "a completed pass leaves no progress behind")
    }

    func testProgressCountsResumedFilesSoAResumedPassDoesNotRestartItsBar() async throws {
        let files = repoFiles()
        let catalog = makeCatalog(files: files)
        let checkpoints = InMemoryVerificationCheckpointStore()
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)

        let interrupted = DigestLedger(pauseAfterStarts: 3)
        let firstPass = Task {
            try await makeVerifier(
                catalog: catalog, ledger: ledger, checkpoints: checkpoints, files: files,
                digestLedger: interrupted
            ).verify(artifact, .full)
        }
        await interrupted.waitUntilPaused()
        firstPass.cancel()
        interrupted.resume()
        _ = try? await firstPass.value

        let saved = try XCTUnwrap(
            checkpoints.checkpoint(matching: checkpointKey(artifact: artifact, files: files)))
        let observed = LockedProgress()
        _ = try await makeVerifier(
            catalog: catalog, ledger: ledger, checkpoints: checkpoints, files: files,
            digestLedger: DigestLedger()
        ).verify(artifact, .full) { observed.record($0) }

        let last = try XCTUnwrap(observed.samples.last)
        XCTAssertEqual(last.filesToCheck, files.count)
        XCTAssertEqual(
            last.bytesToCheck, files.reduce(0) { $0 + $1.sizeBytes },
            "the denominator stays the whole pass, not the remainder")
        // The plan is path-sorted and the checkpoint holds its first entries,
        // so the first file this pass actually digests is already reported at
        // the byte count the interrupted pass proved.
        let firstFresh = try XCTUnwrap(
            observed.samples.first { sample in
                !saved.files.contains { $0.path == sample.currentPath }
            })
        XCTAssertEqual(
            firstFresh.filesChecked, saved.files.count,
            "resumed files count towards the file total")
        XCTAssertEqual(
            firstFresh.bytesChecked, saved.files.reduce(0) { $0 + $1.expectedSizeBytes },
            "a resumed pass starts its progress at the bytes it already proved")
    }

    func testAFileChangedBetweenPassesIsDigestedAgain() async throws {
        let files = repoFiles()
        let catalog = makeCatalog(files: files)
        let checkpoints = InMemoryVerificationCheckpointStore()
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)

        let interrupted = DigestLedger(pauseAfterStarts: 4)
        let firstPass = Task {
            try await makeVerifier(
                catalog: catalog, ledger: ledger, checkpoints: checkpoints, files: files,
                digestLedger: interrupted
            ).verify(artifact, .full)
        }
        await interrupted.waitUntilPaused()
        firstPass.cancel()
        interrupted.resume()
        _ = try? await firstPass.value
        let saved = try XCTUnwrap(
            checkpoints.checkpoint(matching: checkpointKey(artifact: artifact, files: files)))
        XCTAssertEqual(saved.files.count, 3)
        let rewritten = try XCTUnwrap(saved.files.first { $0.isPayload }).path

        // Same length, same inode, same name — only the bytes and the two
        // timestamps move. That is exactly the failure the recorded identity
        // exists to catch.
        let handle = try FileHandle(
            forWritingTo: artifactURL.appendingPathComponent(rewritten))
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: Data([0xFF]))
        try handle.synchronize()
        try handle.close()

        let resumed = DigestLedger()
        let report = try await makeVerifier(
            catalog: catalog, ledger: ledger, checkpoints: checkpoints, files: files,
            digestLedger: resumed
        ).verify(artifact, .full)
        XCTAssertTrue(
            resumed.digestedPaths.contains(rewritten),
            "a file whose identity moved must be read again, not trusted")
        XCTAssertFalse(report.isComplete)
        XCTAssertTrue(report.wrongDigest.contains { $0.path == rewritten })
        XCTAssertEqual(
            try discover(catalog: catalog, ledger: ledger).verification, .unverified)
    }

    func testAnotherRequestKindOrTreeCannotSpendRecordedProgress() async throws {
        let files = repoFiles()
        let catalog = makeCatalog(files: files)
        let checkpoints = InMemoryVerificationCheckpointStore()
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)

        let interrupted = DigestLedger(pauseAfterStarts: 4)
        let firstPass = Task {
            try await makeVerifier(
                catalog: catalog, ledger: ledger, checkpoints: checkpoints, files: files,
                digestLedger: interrupted
            ).verify(artifact, .full)
        }
        await interrupted.waitUntilPaused()
        firstPass.cancel()
        interrupted.resume()
        _ = try? await firstPass.value
        let fullProgress = try XCTUnwrap(
            checkpoints.checkpoint(matching: checkpointKey(artifact: artifact, files: files)))
        XCTAssertEqual(fullProgress.files.count, 3)

        // A spot check selects payloads only, so its plan identity differs and
        // it files under its own slot. The full pass's progress survives it.
        let spot = DigestLedger()
        let spotReport = try await makeVerifier(
            catalog: catalog, ledger: ledger, checkpoints: checkpoints, files: files,
            digestLedger: spot
        ).verify(artifact, .spotCheck(maximumBytes: 1 << 20))
        XCTAssertTrue(spotReport.isComplete)
        XCTAssertEqual(
            Set(spot.digestedPaths), Set(files.filter(\.isPayload).map(\.path)),
            "a spot check re-reads its own payloads rather than spending full-pass progress")
        XCTAssertEqual(
            checkpoints.checkpoint(matching: checkpointKey(artifact: artifact, files: files))?
                .files.count,
            3,
            "a spot check must not evict a full pass's recorded progress")

        // A republished tree is a different question, so nothing carries over.
        let extended = files + [
            RepoFile(
                path: "README.md", sizeBytes: UInt64(Self.readme.count),
                digest: .gitBlobSHA1(hex: FileDigestComputer.gitBlobSHA1(of: Self.readme)),
                isPayload: false)
        ]
        try Self.readme.write(to: artifactURL.appendingPathComponent("README.md"))
        let newCatalog = makeCatalog(files: extended)
        let newArtifact = try discover(catalog: newCatalog, ledger: ledger)
        let afterRepublish = DigestLedger()
        let republished = try await makeVerifier(
            catalog: newCatalog, ledger: ledger, checkpoints: checkpoints, files: extended,
            digestLedger: afterRepublish
        ).verify(newArtifact, .full)
        XCTAssertTrue(republished.isComplete)
        XCTAssertEqual(
            afterRepublish.digestedPaths.count, extended.count,
            "progress recorded against another tree identity is never spent")
    }

    func testVerifyingFromScratchDiscardsRecordedProgress() async throws {
        let files = repoFiles()
        let catalog = makeCatalog(files: files)
        let checkpoints = InMemoryVerificationCheckpointStore()
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)

        let interrupted = DigestLedger(pauseAfterStarts: 4)
        let firstPass = Task {
            try await makeVerifier(
                catalog: catalog, ledger: ledger, checkpoints: checkpoints, files: files,
                digestLedger: interrupted
            ).verify(artifact, .full)
        }
        await interrupted.waitUntilPaused()
        firstPass.cancel()
        interrupted.resume()
        _ = try? await firstPass.value
        XCTAssertNotNil(
            checkpoints.checkpoint(matching: checkpointKey(artifact: artifact, files: files)))

        let scratch = DigestLedger()
        let report = try await makeVerifier(
            catalog: catalog, ledger: ledger, checkpoints: checkpoints, files: files,
            digestLedger: scratch
        ).verify(artifact, .full, resumption: .fromScratch)
        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(
            scratch.digestedPaths.count, files.count,
            "verify-from-scratch reads every selected file again")
    }

    func testUserDefaultsCheckpointStoreSurvivesAProcessBoundary() async throws {
        let files = repoFiles()
        let catalog = makeCatalog(files: files)
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)
        let prefix = "checkpoint.\(UUID().uuidString)."

        let interrupted = DigestLedger(pauseAfterStarts: 3)
        let firstPass = Task {
            try await makeVerifier(
                catalog: catalog, ledger: ledger,
                checkpoints: UserDefaultsVerificationCheckpointStore(
                    defaults: defaults, prefix: prefix),
                files: files, digestLedger: interrupted
            ).verify(artifact, .full)
        }
        await interrupted.waitUntilPaused()
        firstPass.cancel()
        interrupted.resume()
        _ = try? await firstPass.value

        // A second store instance addressing the same defaults domain is what
        // the next launch holds.
        let resumed = DigestLedger()
        let report = try await makeVerifier(
            catalog: catalog, ledger: ledger,
            checkpoints: UserDefaultsVerificationCheckpointStore(
                defaults: defaults, prefix: prefix),
            files: files, digestLedger: resumed
        ).verify(artifact, .full)
        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(resumed.digestedPaths.count, files.count - 2)
    }

    // MARK: - Fixture

    private static let readme = Data("resume fixture\n".utf8)

    private func uninterruptedEvidence(
        files: [RepoFile], catalog: ModelCatalogSnapshot
    ) async throws -> ArtifactVerificationEvidence {
        let ledger = InMemoryVerificationLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)
        let report = try await makeVerifier(
            catalog: catalog, ledger: ledger, checkpoints: nil, files: files,
            digestLedger: DigestLedger()
        ).verify(artifact, .full)
        XCTAssertTrue(report.isComplete)
        let record = try XCTUnwrap(
            ledger.record(
                matching: ArtifactVerificationLookup(
                    rootPath: artifact.rootPath, model: .kimiK3,
                    repository: HuggingFaceRepoRef(
                        repoID: "fixture/K3-minirun", revision: Self.revision),
                    index: artifact.index)))
        return try XCTUnwrap(record.evidence)
    }

    private func install() throws {
        try FileManager.default.createDirectory(
            at: artifactURL, withIntermediateDirectories: true)
        try Data(ArtifactDiscoveryTests.k3Index.utf8).write(
            to: artifactURL.appendingPathComponent("index.json"))
        for (offset, path) in Self.payloadPaths.enumerated() {
            let url = artifactURL.appendingPathComponent(path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Self.payloadBytes(offset).write(to: url)
        }
    }

    private static func payloadBytes(_ offset: Int) -> Data {
        Data(repeating: UInt8(0x40 + offset), count: 4096 + offset)
    }

    private func repoFiles() -> [RepoFile] {
        let index = Data(ArtifactDiscoveryTests.k3Index.utf8)
        var files = [
            RepoFile(
                path: "index.json", sizeBytes: UInt64(index.count),
                digest: .gitBlobSHA1(hex: FileDigestComputer.gitBlobSHA1(of: index)),
                isPayload: false)
        ]
        for (offset, path) in Self.payloadPaths.enumerated() {
            let bytes = Self.payloadBytes(offset)
            files.append(
                RepoFile(
                    path: path, sizeBytes: UInt64(bytes.count),
                    digest: .sha256(hex: FileDigestComputer.sha256(of: bytes)),
                    isPayload: true))
        }
        return files
    }

    private func makeCatalog(files: [RepoFile]) -> ModelCatalogSnapshot {
        let base = ModelCatalog.bundled.descriptor(.kimiK3)!
        let payload = files.filter(\.isPayload)
        let metadata = files.filter { !$0.isPayload }
        let descriptor = ModelDescriptor(
            id: base.id, displayName: base.displayName, architecture: base.architecture,
            layout: base.layout,
            source: .huggingFaceRepo(
                HuggingFaceRepoRef(repoID: "fixture/K3-minirun", revision: Self.revision)),
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
        catalog: ModelCatalogSnapshot, ledger: any ArtifactVerificationLedger,
        checkpoints: (any ArtifactVerificationCheckpointStore)?, files: [RepoFile],
        digestLedger: DigestLedger
    ) -> ArtifactVerifier {
        let tree = HuggingFaceTreeClient(
            transport: ResumeTreeTransport(body: treeBody(files)),
            endpoint: URL(string: "https://resume.invalid")!)
        return ArtifactVerifier(
            tree: tree, catalog: catalog, ledger: ledger, checkpoints: checkpoints,
            hooks: ArtifactVerifierHooks(
                digestStarted: { digestLedger.noteStart($0) },
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
        artifact: DiscoveredArtifact, files: [RepoFile],
        kind: ArtifactVerificationRequestKind = .full
    ) -> ArtifactVerificationCheckpointKey {
        let selected = try! ArtifactVerifier.select(
            files, for: kind == .full ? .full : .spotCheck(maximumBytes: 1 << 20))
        return ArtifactVerificationCheckpointKey(
            rootPath: artifact.rootPath, model: .kimiK3,
            repository: HuggingFaceRepoRef(
                repoID: "fixture/K3-minirun", revision: Self.revision),
            treeIdentity: ArtifactDigestPlanIdentity.compute(files),
            selectedPlanIdentity: ArtifactDigestPlanIdentity.compute(selected),
            requestKind: kind)
    }

    private func storedEvidence(
        _ artifact: DiscoveredArtifact
    ) throws -> ArtifactVerificationEvidence? {
        let key = ledgerPrefix
            + ArtifactVerificationPathIdentity.canonical(artifact.rootPath)
        guard let data = defaults.data(forKey: key) else { return nil }
        return try MinirunKitJSON.decoder()
            .decode(ArtifactVerificationRecord.self, from: data).evidence
    }
}

private final class LockedProgress: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [ArtifactVerificationProgress] = []

    var samples: [ArtifactVerificationProgress] {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }

    func record(_ progress: ArtifactVerificationProgress) {
        lock.lock()
        stored.append(progress)
        lock.unlock()
    }
}
