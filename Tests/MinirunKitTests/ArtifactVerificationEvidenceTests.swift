import Darwin
import Foundation
import XCTest

@testable import MinirunKit

private struct VerificationTreeTransport: HTTPTransport {
    let body: Data
    var authorityPaths: [String]? = nil

    func data(for request: HTTPRequest) async throws -> HTTPResponse {
        if request.url.path.contains("/revision/") {
            let entries = (try? JSONSerialization.jsonObject(with: body) as? [[String: Any]]) ?? []
            let decodedPaths = entries.compactMap { entry -> String? in
                guard (entry["type"] as? String ?? "file") == "file",
                    let path = entry["path"] as? String
                else { return nil }
                return path
            }
            let siblings = (authorityPaths ?? decodedPaths).map { ["rfilename": $0] }
            let revision = request.url.path.split(separator: "/").last.map(String.init) ?? ""
            let authority = try JSONSerialization.data(withJSONObject: [
                "sha": revision, "siblings": siblings,
            ])
            return HTTPResponse(statusCode: 200, headers: [:], body: authority)
        }
        return HTTPResponse(statusCode: 200, headers: [:], body: body)
    }

    func download(_ request: HTTPRequest, into sink: HTTPDownloadSink) async throws
        -> HTTPResponse
    {
        throw HTTPTransportError.notHTTP(url: request.url.absoluteString)
    }
}

private struct LegacyArtifactVerificationRecord: Encodable {
    let rootPath: String
    let state: ArtifactVerification
    let checkedAt: Date
    let detail: String
}

private final class DigestCheckpointGate: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    private let target: Int
    private let reached = DispatchSemaphore(value: 0)
    private let resumeHash = DispatchSemaphore(value: 0)

    init(target: Int) { self.target = target }

    func checkpoint() {
        lock.lock()
        count += 1
        let shouldPause = count == target
        lock.unlock()
        if shouldPause {
            reached.signal()
            resumeHash.wait()
        }
    }

    func waitUntilReached() async {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                self.reached.wait()
                continuation.resume()
            }
        }
    }

    func resume() { resumeHash.signal() }
}

private final class DigestConcurrencyProbe: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var peak = 0
    private var active = 0
    private(set) var starts = 0

    func started() {
        lock.lock()
        active += 1
        starts += 1
        peak = max(peak, active)
        lock.unlock()
    }

    func finished() {
        lock.lock()
        active -= 1
        lock.unlock()
    }
}

private final class LockedFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = false

    var value: Bool {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }

    func set() {
        lock.lock()
        stored = true
        lock.unlock()
    }
}

private enum InjectedFilesystemFailure: Error { case stat }

/// A remembered digest is a cache of evidence, not authority over a pathname.
/// These tests exercise every identity component that lets a metadata-only scan
/// decide whether the cache still describes the physical bytes on disk.
final class ArtifactVerificationEvidenceTests: XCTestCase {
    private static let revisionA = String(repeating: "1", count: 40)
    private static let revisionB = String(repeating: "2", count: 40)
    private static let payloadPath = "layer00/weights.bin"
    private static let manifestPath = "layer00/manifest.json"
    private static let manifestData = Data("{\"offset\":0}".utf8)

    private var location: URL!
    private var artifactURL: URL!
    private var payloadURL: URL!
    private var defaults: UserDefaults!
    private var defaultsSuite: String!
    private var ledgerPrefix: String!

    override func setUpWithError() throws {
        let base = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-verification-evidence-\(UUID().uuidString)")
        location = base.appendingPathComponent("location")
        artifactURL = location.appendingPathComponent("artifact")
        payloadURL = artifactURL.appendingPathComponent(Self.payloadPath)
        try FileManager.default.createDirectory(at: location, withIntermediateDirectories: true)

        defaultsSuite = "minirun.verification.tests.\(UUID().uuidString)"
        defaults = try XCTUnwrap(UserDefaults(suiteName: defaultsSuite))
        ledgerPrefix = "ledger.\(UUID().uuidString)."
    }

    override func tearDownWithError() throws {
        if let location { try? FileManager.default.removeItem(at: location) }
        if let defaultsSuite { defaults?.removePersistentDomain(forName: defaultsSuite) }
    }

    // MARK: - Positive persistence

    func testFullVerificationRejectsEmptyAndTruncatedTreesBeforeDigesting() async throws {
        let bytes = Data("complete fixture".utf8)
        try install(bytes)
        let complete = repoFiles(for: bytes)
        let catalog = makeCatalog(files: complete)
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)

        for returned in [[], Array(complete.dropLast())] {
            let probe = DigestConcurrencyProbe()
            let verifier = makeVerifier(
                catalog: catalog, ledger: ledger, files: returned,
                hooks: ArtifactVerifierHooks(
                    digestStarted: { _ in probe.started() },
                    digestFinished: { _ in probe.finished() },
                    volumeSupportsPersistentIDs: { _ in true }))
            do {
                _ = try await verifier.verify(artifact, .full)
                XCTFail("an incomplete tree was accepted")
            } catch let error as ArtifactVerificationError {
                guard case .publishedTreeMismatch = error else {
                    return XCTFail("unexpected error: \(error)")
                }
            }
            XCTAssertEqual(probe.starts, 0)
        }
    }

    func testSameTruncatedTreeCannotBootstrapBothDescriptorAndVerificationAuthority() async throws {
        let bytes = Data("same source truncation".utf8)
        try install(bytes)
        let complete = repoFiles(for: bytes)
        let truncated = Array(complete.dropLast())
        let catalogDerivedFromTruncation = makeCatalog(files: truncated)
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalogDerivedFromTruncation, ledger: ledger)
        let probe = DigestConcurrencyProbe()
        let verifier = makeVerifier(
            catalog: catalogDerivedFromTruncation, ledger: ledger, files: truncated,
            authorityFiles: complete,
            hooks: ArtifactVerifierHooks(
                digestStarted: { _ in probe.started() },
                digestFinished: { _ in probe.finished() },
                volumeSupportsPersistentIDs: { _ in true }))

        do {
            _ = try await verifier.verify(artifact, .full)
            XCTFail("one truncated response bootstrapped its own completeness authority")
        } catch let error as CatalogError {
            guard case .malformedResponse(let detail) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains("not complete"), detail)
        }
        XCTAssertEqual(probe.starts, 0)
        XCTAssertEqual(try discover(catalog: catalogDerivedFromTruncation, ledger: ledger).verification, .unverified)
    }

    func testSuccessfulRecordBindsPinnedTreeAndPhysicalPayloadAcrossLedgerInstances() async throws {
        let bytes = Data("verified payload".utf8)
        try install(bytes)
        let files = repoFiles(for: bytes)
        let catalog = makeCatalog(files: files)
        let firstLedger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: firstLedger)

        let report = try await makeVerifier(
            catalog: catalog, ledger: firstLedger, files: files
        ).verify(artifact, .full)
        XCTAssertTrue(report.isComplete)

        // A new value wrapper models a relaunch reading the same UserDefaults
        // document. The scan performs only open/fstat work and inherits the
        // positive state because every bound identity still matches.
        let relaunchedLedger = makeLedger()
        let after = try discover(catalog: catalog, ledger: relaunchedLedger)
        XCTAssertEqual(after.verification, .fullyVerified)
        XCTAssertNotNil(after.verifiedAt)

        let record = try XCTUnwrap(
            relaunchedLedger.record(
                matching: lookup(
                    artifact: after, catalog: catalog,
                    treeIdentity: ArtifactDigestPlanIdentity.compute(files))))
        let evidence = try XCTUnwrap(record.evidence)
        XCTAssertNotNil(evidence.persistentVolume)
        XCTAssertEqual(evidence.model, .kimiK3)
        XCTAssertEqual(evidence.repository, makeCatalogRepo())
        XCTAssertEqual(evidence.treeIdentity, ArtifactDigestPlanIdentity.compute(files))
        XCTAssertEqual(
            evidence.completenessAuthorityIdentity,
            RepositoryFileSetIdentity.compute(files.map(\.path)))
        XCTAssertEqual(
            evidence.selectedPlanIdentity,
            ArtifactDigestPlanIdentity.compute(files))
        XCTAssertEqual(evidence.files.map(\.path), files.map(\.path).sorted())
        XCTAssertEqual(
            evidence.files.first(where: { $0.path == Self.payloadPath })?.filesystem.sizeBytes,
            UInt64(bytes.count))
        XCTAssertNotEqual(evidence.root.inode, 0)
        XCTAssertNotEqual(
            evidence.files.first(where: { $0.path == Self.payloadPath })?.filesystem.inode, 0)
        XCTAssertEqual(report.depth, .digestEverything)
    }

    func testEvidenceSurvivesRemountDeviceRenumberingOnTheSamePersistentVolume() async throws {
        let bytes = Data("same removable volume".utf8)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        let record = try XCTUnwrap(
            ledger.record(
                matching: lookup(
                    artifact: artifact, catalog: catalog,
                    treeIdentity: ArtifactDigestPlanIdentity.compute(files))))
        let evidence = try XCTUnwrap(record.evidence)
        let volume = try XCTUnwrap(evidence.persistentVolume)
        let remountedDevice = evidence.root.device == UInt64.max
            ? evidence.root.device - 1 : evidence.root.device + 1

        XCTAssertNoThrow(
            try ArtifactFilesystemEvidence.validate(
                evidence: evidence, at: artifactURL,
                hooks: ArtifactFilesystemValidationHooks(
                    betweenIdentityPasses: {},
                    persistentIDCapability: { _, _ in true },
                    persistentVolumeIdentity: { _, _ in volume },
                    filesystemIdentity: { descriptor, path, isDirectory in
                        let current = try ArtifactFilesystemEvidence.identity(
                            of: descriptor, path: path, expectedDirectory: isDirectory)
                        return ArtifactFilesystemIdentity(
                            device: remountedDevice, inode: current.inode,
                            sizeBytes: current.sizeBytes,
                            modificationSeconds: current.modificationSeconds,
                            modificationNanoseconds: current.modificationNanoseconds,
                            statusChangeSeconds: current.statusChangeSeconds,
                            statusChangeNanoseconds: current.statusChangeNanoseconds)
                    })))
    }

    func testEvidenceRejectsADifferentVolumeEvenWhenObjectMetadataMatches() async throws {
        let bytes = Data("different removable volume".utf8)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        let record = try XCTUnwrap(
            ledger.record(
                matching: lookup(
                    artifact: artifact, catalog: catalog,
                    treeIdentity: ArtifactDigestPlanIdentity.compute(files))))
        let evidence = try XCTUnwrap(record.evidence)
        XCTAssertThrowsError(
            try ArtifactFilesystemEvidence.validate(
                evidence: evidence, at: artifactURL,
                hooks: ArtifactFilesystemValidationHooks(
                    betweenIdentityPasses: {},
                    persistentIDCapability: { _, _ in true },
                    persistentVolumeIdentity: { _, _ in
                        ArtifactPersistentVolumeIdentity(
                            uuid: "00000000-0000-4000-8000-000000000001")
                    }))) { error in
                        guard case ArtifactFilesystemEvidenceError.changed(let path) = error else {
                            return XCTFail("unexpected error: \(error)")
                        }
                        XCTAssertEqual(path, self.artifactURL.path)
                    }
    }

    func testLegacyEvidenceStillRequiresExactMountSessionIdentity() async throws {
        let bytes = Data("legacy exact device".utf8)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        let record = try XCTUnwrap(
            ledger.record(
                matching: lookup(
                    artifact: artifact, catalog: catalog,
                    treeIdentity: ArtifactDigestPlanIdentity.compute(files))))
        let evidence = try XCTUnwrap(record.evidence)
        let legacy = copy(evidence, persistentVolume: nil)
        let remountedDevice = evidence.root.device == UInt64.max
            ? evidence.root.device - 1 : evidence.root.device + 1

        XCTAssertThrowsError(
            try ArtifactFilesystemEvidence.validate(
                evidence: legacy, at: artifactURL,
                hooks: ArtifactFilesystemValidationHooks(
                    betweenIdentityPasses: {},
                    persistentIDCapability: { _, _ in true },
                    persistentVolumeIdentity: { _, _ in
                        evidence.persistentVolume
                    },
                    filesystemIdentity: { descriptor, path, isDirectory in
                        let current = try ArtifactFilesystemEvidence.identity(
                            of: descriptor, path: path, expectedDirectory: isDirectory)
                        return ArtifactFilesystemIdentity(
                            device: remountedDevice, inode: current.inode,
                            sizeBytes: current.sizeBytes,
                            modificationSeconds: current.modificationSeconds,
                            modificationNanoseconds: current.modificationNanoseconds,
                            statusChangeSeconds: current.statusChangeSeconds,
                            statusChangeNanoseconds: current.statusChangeNanoseconds)
                    })))
    }

    func testVerifiedEvidenceSurvivesPrivateVarBookmarkAlias() async throws {
        let bytes = Data("darwin path alias".utf8)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        let canonicalPath = ArtifactVerificationPathIdentity.canonical(artifact.rootPath)
        guard canonicalPath == "/var" || canonicalPath.hasPrefix("/var/") else {
            throw XCTSkip("the test temporary directory is not beneath Darwin /var")
        }
        let bookmarkPath = "/private" + canonicalPath
        XCTAssertNotEqual(bookmarkPath, canonicalPath)
        XCTAssertTrue(FileManager.default.fileExists(atPath: bookmarkPath))
        XCTAssertNotNil(defaults.data(forKey: ledgerPrefix + canonicalPath))
        XCTAssertNil(defaults.data(forKey: ledgerPrefix + bookmarkPath))

        let lookupFromBookmark = ArtifactVerificationLookup(
            rootPath: bookmarkPath, model: artifact.model,
            repository: artifact.model.flatMap { catalog.descriptor($0)?.source.repo },
            index: artifact.index,
            currentTreeIdentity: ArtifactDigestPlanIdentity.compute(files))
        let remembered = try XCTUnwrap(ledger.record(matching: lookupFromBookmark))
        XCTAssertEqual(remembered.state, .fullyVerified)
        XCTAssertEqual(
            ArtifactVerificationPathIdentity.canonical(remembered.rootPath), canonicalPath)

        ledger.forget(rootPath: bookmarkPath)
        XCTAssertNil(
            ledger.record(
                matching: lookup(
                    artifact: artifact, catalog: catalog,
                    treeIdentity: ArtifactDigestPlanIdentity.compute(files))))
    }

    func testVerificationPathIdentityNormalizesOnlyTheDarwinVarAlias() {
        XCTAssertEqual(
            ArtifactVerificationPathIdentity.canonical(
                "/private/var/mobile/Library/LiveFiles/example/artifact"),
            "/var/mobile/Library/LiveFiles/example/artifact")
        XCTAssertEqual(
            ArtifactVerificationPathIdentity.storageAliases(
                for: "/var/mobile/Library/LiveFiles/example/artifact"),
            [
                "/var/mobile/Library/LiveFiles/example/artifact",
                "/private/var/mobile/Library/LiveFiles/example/artifact",
            ])
        XCTAssertEqual(
            ArtifactVerificationPathIdentity.canonical("/private/variant/artifact"),
            "/private/variant/artifact")
        XCTAssertNotEqual(
            ArtifactVerificationPathIdentity.canonical("/private/var/model-a"),
            ArtifactVerificationPathIdentity.canonical("/private/var/model-b"))
    }

    func testRelaunchQueriesPersistentIDsFromTheOpenedArtifactDescriptor() async throws {
        let bytes = Data("descriptor capability".utf8)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        let record = try XCTUnwrap(
            ledger.record(
                matching: lookup(
                    artifact: artifact, catalog: catalog,
                    treeIdentity: ArtifactDigestPlanIdentity.compute(files))))
        let evidence = try XCTUnwrap(record.evidence)
        let didQuery = LockedFlag()

        try ArtifactFilesystemEvidence.validate(
            evidence: evidence, at: artifactURL,
            hooks: ArtifactFilesystemValidationHooks(
                betweenIdentityPasses: {},
                persistentIDCapability: { descriptor, path in
                    XCTAssertGreaterThanOrEqual(descriptor, 0)
                    XCTAssertEqual(path, self.artifactURL.path)
                    didQuery.set()
                    return true
                }))

        XCTAssertTrue(didQuery.value)
    }

    func testSuccessfulSpotCheckReusesOnlyItsSelectedPayloadEvidence() async throws {
        let smaller = Data("small".utf8)
        let larger = Data("the larger payload".utf8)
        try install(smaller)
        let largerPath = "layer01/weights.bin"
        let largerURL = artifactURL.appendingPathComponent(largerPath)
        try FileManager.default.createDirectory(
            at: largerURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try larger.write(to: largerURL)

        let files = repoFiles(for: smaller) + [
            RepoFile(
                path: largerPath, sizeBytes: UInt64(larger.count),
                digest: .sha256(hex: FileDigestComputer.sha256(of: larger)), isPayload: true),
        ]
        let catalog = makeCatalog(files: files)
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)

        let report = try await makeVerifier(
            catalog: catalog, ledger: ledger, files: files
        ).verify(artifact, .spotCheck(maximumBytes: UInt64(smaller.count)))
        XCTAssertTrue(report.isComplete)

        let after = try discover(catalog: catalog, ledger: makeLedger())
        XCTAssertEqual(after.verification, .spotChecked)
        let record = try XCTUnwrap(
            makeLedger().record(
                matching: lookup(
                    artifact: after, catalog: catalog,
                    treeIdentity: ArtifactDigestPlanIdentity.compute(files))))
        let evidence = try XCTUnwrap(record.evidence)
        XCTAssertEqual(evidence.treePayloadFileCount, 2)
        XCTAssertEqual(evidence.files.map(\.path), [Self.payloadPath])
        XCTAssertEqual(evidence.selectedBytes, UInt64(smaller.count))
        XCTAssertEqual(report.depth, .digestPayload)
    }

    func testSynthesizedStructurallyInvalidEvidenceNeverBecomesAuthority() async throws {
        let bytes = Data("valid bytes, invalid persisted evidence".utf8)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        let good = try XCTUnwrap(
            ledger.record(
                matching: lookup(
                    artifact: artifact, catalog: catalog,
                    treeIdentity: ArtifactDigestPlanIdentity.compute(files))))
        let evidence = try XCTUnwrap(good.evidence)
        let invalid = ArtifactVerificationEvidence(
            model: evidence.model, repository: evidence.repository,
            treeIdentity: evidence.treeIdentity,
            completenessAuthorityIdentity: evidence.completenessAuthorityIdentity,
            selectedPlanIdentity: evidence.selectedPlanIdentity,
            treeFileCount: evidence.treeFileCount,
            treePayloadFileCount: evidence.treePayloadFileCount,
            treeMetadataFileCount: evidence.treeMetadataFileCount,
            treeBytes: evidence.treeBytes, treePayloadBytes: evidence.treePayloadBytes,
            treeMetadataBytes: evidence.treeMetadataBytes,
            selectedBytes: evidence.selectedBytes + 1,
            index: evidence.index, root: evidence.root, files: evidence.files)
        ledger.record(
            ArtifactVerificationRecord(
                rootPath: artifact.rootPath, state: .fullyVerified,
                checkedAt: good.checkedAt, detail: "synthesized invalid evidence",
                evidence: invalid))

        let after = try discover(catalog: catalog, ledger: ledger)
        XCTAssertEqual(after.verification, .unverified)
        XCTAssertNil(after.verifiedAt)
    }

    // MARK: - Filesystem replacement and mutation

    func testSamePathDirectoryReplacementDoesNotInheritVerification() async throws {
        let bytes = Data("same bytes in a different directory object".utf8)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        let oldRecord = try XCTUnwrap(
            ledger.record(
                matching: lookup(
                    artifact: artifact, catalog: catalog,
                    treeIdentity: ArtifactDigestPlanIdentity.compute(files))))

        let held = location.appendingPathComponent(".held-artifact")
        try FileManager.default.moveItem(at: artifactURL, to: held)
        try install(bytes)

        let after = try discover(catalog: catalog, ledger: ledger)
        XCTAssertEqual(after.verification, .unverified)
        let currentRoot = try rootIdentity(at: artifactURL)
        XCTAssertNotEqual(oldRecord.evidence?.root.inode, currentRoot.inode)
    }

    func testPayloadInodeReplacementAtTheSamePathDoesNotInheritVerification() async throws {
        let bytes = Data("same length and digest, new inode".utf8)
        let (catalog, ledger, artifact, _) = try await verifiedFixture(bytes: bytes)
        let held = location.appendingPathComponent(".held-payload")
        try FileManager.default.moveItem(at: payloadURL, to: held)
        try bytes.write(to: payloadURL)

        let after = try discover(catalog: catalog, ledger: ledger)
        XCTAssertEqual(after.verification, .unverified)
        XCTAssertNil(after.verifiedAt)
        XCTAssertNil(
            defaults.data(forKey: ledgerKey(artifact)),
            "a contradiction in the physical object still erases the record")
    }

    func testSameSizeInPlaceRewriteChangesStatusEvidenceAndInvalidates() async throws {
        let bytes = Data("same-size payload".utf8)
        let (catalog, ledger, artifact, _) = try await verifiedFixture(bytes: bytes)
        try await Task.sleep(nanoseconds: 10_000_000)
        try rewriteFirstByte(of: payloadURL, to: 0x58)
        XCTAssertEqual(try FileDigestComputer.byteCount(ofFileAt: payloadURL), UInt64(bytes.count))

        let after = try discover(catalog: catalog, ledger: ledger)
        XCTAssertEqual(after.verification, .unverified)
        XCTAssertNil(after.verifiedAt)
        XCTAssertNil(defaults.data(forKey: ledgerKey(artifact)))
    }

    func testPublishedManifestMutationInvalidatesAFullRecord() async throws {
        let bytes = Data("manifest-bound".utf8)
        let (catalog, ledger, _, _) = try await verifiedFixture(bytes: bytes)
        let changed = Data("{\"offset\":1}".utf8)
        XCTAssertEqual(changed.count, Self.manifestData.count)
        try changed.write(to: artifactURL.appendingPathComponent(Self.manifestPath))

        let after = try discover(catalog: catalog, ledger: ledger)
        XCTAssertEqual(after.verification, .unverified)
        XCTAssertNil(after.verifiedAt)
    }

    func testHardLinkedPublishedFileIsRejectedWithoutHashing() async throws {
        let bytes = Data("hard-link".utf8)
        try install(bytes)
        let aliasPath = "layer01/weights.bin"
        let aliasURL = artifactURL.appendingPathComponent(aliasPath)
        try FileManager.default.createDirectory(
            at: aliasURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        XCTAssertEqual(link(payloadURL.path, aliasURL.path), 0)
        var files = repoFiles(for: bytes)
        files.append(
            RepoFile(
                path: aliasPath, sizeBytes: UInt64(bytes.count),
                digest: .sha256(hex: FileDigestComputer.sha256(of: bytes)),
                isPayload: true))
        let catalog = makeCatalog(files: files)
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)
        let probe = DigestConcurrencyProbe()
        let report = try await makeVerifier(
            catalog: catalog, ledger: ledger, files: files,
            hooks: ArtifactVerifierHooks(
                digestStarted: { _ in probe.started() },
                digestFinished: { _ in probe.finished() },
                volumeSupportsPersistentIDs: { _ in true })
        ).verify(artifact, .full)
        XCTAssertFalse(report.isComplete)
        XCTAssertTrue(report.unreadable.keys.contains(Self.payloadPath))
        XCTAssertEqual(probe.starts, 2, "metadata may hash, but neither hard link may hash")
        XCTAssertEqual(try discover(catalog: catalog, ledger: ledger).verification, .unverified)
    }

    func testCaseAliasToTheSameInodeIsRejected() async throws {
        let bytes = Data("case-alias".utf8)
        try install(bytes)
        let aliasPath = "INDEX.JSON"
        let aliasURL = artifactURL.appendingPathComponent(aliasPath)
        guard FileManager.default.fileExists(atPath: aliasURL.path) else {
            throw XCTSkip("the test volume is case-sensitive")
        }
        let index = Data(ArtifactDiscoveryTests.k3Index.utf8)
        var files = repoFiles(for: bytes)
        files.append(
            RepoFile(
                path: aliasPath, sizeBytes: UInt64(index.count),
                digest: .gitBlobSHA1(hex: FileDigestComputer.gitBlobSHA1(of: index)),
                isPayload: false))
        let catalog = makeCatalog(files: files)
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)
        do {
            _ = try await makeVerifier(
                catalog: catalog, ledger: ledger, files: files,
                hooks: ArtifactVerifierHooks(volumeSupportsPersistentIDs: { _ in true })
            ).verify(artifact, .full)
            XCTFail("two repository spellings for one inode were accepted")
        } catch let error as ArtifactVerificationError {
            guard case .filesystemUnavailable(let path, let reason) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(path, "index.json")
            XCTAssertTrue(reason.contains("aliases the same filesystem object"))
        }
        XCTAssertEqual(try discover(catalog: catalog, ledger: ledger).verification, .unverified)
    }

    // MARK: - Catalog-side disagreement

    // A lookup carries two very different kinds of fact. `rootPath` and the
    // physical objects behind it are the disk. `model` and `repository` are
    // whatever catalogue the calling process has managed to load, which on a
    // first launch is nothing and after a publication is last week's. On
    // 2026-08-15 the ledger treated both as grounds for deletion and destroyed
    // complete evidence for 1.7 TB of artifacts that had not moved a byte.
    // Every test in this section asserts the stored bytes as well as the
    // reported state: withholding and forgetting look identical from the
    // outside and are not remotely the same thing.

    func testLookupWithoutAModelWithholdsEvidenceWithoutErasingIt() async throws {
        let bytes = Data("empty-snapshot launch".utf8)
        let (catalog, ledger, artifact, _) = try await verifiedFixture(bytes: bytes)
        let key = ledgerKey(artifact)
        let remembered = try XCTUnwrap(defaults.data(forKey: key))

        XCTAssertNil(
            ledger.record(
                matching: ArtifactVerificationLookup(
                    rootPath: artifact.rootPath, model: nil, repository: nil,
                    index: artifact.index)))
        XCTAssertEqual(defaults.data(forKey: key), remembered)

        // The launch as it actually ran: no cached catalog, so the scan matched
        // against a snapshot holding no models at all. `ArtifactIdentityMatcher`
        // still names the model from its compiled-in upstream table, but an
        // empty snapshot has no descriptor and therefore no repository to put
        // in the lookup — which is the disagreement that used to delete.
        let empty = ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [])
        let unrecognized = try discover(catalog: empty, ledger: ledger)
        XCTAssertNil(empty.descriptor(.kimiK3)?.source.repo)
        XCTAssertEqual(unrecognized.verification, .unverified)
        XCTAssertEqual(defaults.data(forKey: key), remembered)

        // …and the launch that follows it, once a catalogue is available.
        let recovered = try discover(catalog: catalog, ledger: ledger)
        XCTAssertEqual(recovered.verification, .fullyVerified)
        XCTAssertNotNil(recovered.verifiedAt)
    }

    func testCatalogPinnedToAnotherRevisionWithholdsEvidenceWithoutErasingIt() async throws {
        let bytes = Data("revision-bound".utf8)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        let key = ledgerKey(artifact)
        let remembered = try XCTUnwrap(defaults.data(forKey: key))
        let elsewhere = makeCatalog(files: files, revision: Self.revisionB)

        let underOtherRevision = try discover(catalog: elsewhere, ledger: ledger)
        XCTAssertEqual(underOtherRevision.verification, .unverified)
        XCTAssertEqual(
            defaults.data(forKey: key), remembered,
            "a catalogue pinned elsewhere than the disk says nothing about the disk")

        let underRecordedRevision = try discover(catalog: catalog, ledger: ledger)
        XCTAssertEqual(underRecordedRevision.verification, .fullyVerified)
        XCTAssertNotNil(underRecordedRevision.verifiedAt)
    }

    func testPublishingRepositoryChangeWithholdsEvidenceWithoutErasingIt() async throws {
        let bytes = Data("repo-bound".utf8)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        let key = ledgerKey(artifact)
        let remembered = try XCTUnwrap(defaults.data(forKey: key))
        let changedCatalog = makeCatalog(
            files: files, repoID: "fixture/Other-K3-minirun")

        let after = try discover(catalog: changedCatalog, ledger: ledger)
        XCTAssertEqual(after.model, .kimiK3)
        XCTAssertEqual(after.verification, .unverified)
        XCTAssertEqual(defaults.data(forKey: key), remembered)

        XCTAssertEqual(
            try discover(catalog: catalog, ledger: ledger).verification, .fullyVerified)
    }

    func testIndexChangingToAnotherModelWithholdsEvidenceWithoutErasingIt() async throws {
        let bytes = Data("model-bound".utf8)
        let (catalog, ledger, artifact, _) = try await verifiedFixture(bytes: bytes)
        let key = ledgerKey(artifact)
        let remembered = try XCTUnwrap(defaults.data(forKey: key))
        try Data(ArtifactDiscoveryTests.v4Index.utf8).write(
            to: artifactURL.appendingPathComponent("index.json"))

        let after = try discover(catalog: catalog, ledger: ledger)
        XCTAssertEqual(after.model, .deepseekV4Flash)
        XCTAssertEqual(after.verification, .unverified)
        XCTAssertEqual(defaults.data(forKey: key), remembered)
    }

    // MARK: - The index.json decision

    // `index` is the one lookup field a caller reads from the disk rather than
    // from its catalogue, so it was the candidate for staying a delete. These
    // two tests are why it does not. On a full record the delete is redundant:
    // `index.json` is evidenced and a replacement is caught physically. On a
    // spot-checked record it is unprovable: nothing physical covers
    // `index.json` at all, so the inequality would be deleting on a guess.

    func testReplacedIndexOnAFullRecordIsErasedByThePhysicalPass() async throws {
        let bytes = Data("index-bound".utf8)
        let (catalog, ledger, artifact, _) = try await verifiedFixture(bytes: bytes)
        let key = ledgerKey(artifact)
        let remembered = try XCTUnwrap(defaults.data(forKey: key))
        let evidence = try XCTUnwrap(storedEvidence(remembered))
        XCTAssertTrue(
            evidence.files.contains { $0.path == "index.json" },
            "a full pass covers every tree file, index.json included")

        let indexURL = artifactURL.appendingPathComponent("index.json")
        try await Task.sleep(nanoseconds: 10_000_000)
        try Data(ArtifactDiscoveryTests.v4Index.utf8).write(to: indexURL)
        // While the document names another model the record answers a different
        // question, so it is withheld rather than destroyed.
        XCTAssertEqual(try discover(catalog: catalog, ledger: ledger).verification, .unverified)
        XCTAssertEqual(defaults.data(forKey: key), remembered)

        // Restoring the original bytes makes the catalogue agree again — and
        // the filesystem pass then rejects the record anyway, because the
        // evidenced `index.json` object is not the object now on disk. The
        // index-inequality delete would have bought nothing here.
        try Data(ArtifactDiscoveryTests.k3Index.utf8).write(to: indexURL)
        let after = try discover(catalog: catalog, ledger: ledger)
        XCTAssertEqual(after.model, .kimiK3)
        XCTAssertEqual(after.index, evidence.index)
        XCTAssertEqual(after.verification, .unverified)
        XCTAssertNil(
            defaults.data(forKey: key),
            "a replaced index.json is a physical contradiction and is caught as one")
    }

    func testSpotCheckEvidenceCannotCoverIndexJsonSoIndexInequalityIsWithheld() async throws {
        let bytes = Data("spot-checked".utf8)
        try install(bytes)
        let files = repoFiles(for: bytes)
        let catalog = makeCatalog(files: files)
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)
        let report = try await makeVerifier(catalog: catalog, ledger: ledger, files: files)
            .verify(artifact, .spotCheck(maximumBytes: UInt64(bytes.count)))
        XCTAssertTrue(report.isComplete)

        let key = ledgerKey(artifact)
        let remembered = try XCTUnwrap(defaults.data(forKey: key))
        let evidence = try XCTUnwrap(storedEvidence(remembered))
        XCTAssertTrue(
            evidence.files.allSatisfy(\.isPayload),
            "a spot check evidences payloads only, by structural rule")
        XCTAssertFalse(evidence.files.contains { $0.path == "index.json" })

        // Give index.json a new modification time. The physical pass still
        // succeeds, because there is no recorded identity for it to contradict:
        // an index-inequality delete could not have been the physical check
        // firing first.
        try await Task.sleep(nanoseconds: 10_000_000)
        try rewriteFirstByte(of: artifactURL.appendingPathComponent("index.json"), to: 0x7B)
        XCTAssertNoThrow(
            try ArtifactFilesystemEvidence.validate(evidence: evidence, at: artifactURL))

        // So a caller that parsed a different index is told nothing and costs
        // the record nothing.
        let otherIndex = ArtifactIndexIdentity.parse(
            Data(ArtifactDiscoveryTests.k3Index.replacingOccurrences(
                of: "\"files\": 372", with: "\"files\": 373").utf8))
        XCTAssertNotEqual(otherIndex, evidence.index)
        XCTAssertNil(
            ledger.record(
                matching: ArtifactVerificationLookup(
                    rootPath: artifact.rootPath, model: .kimiK3,
                    repository: makeCatalogRepo(), index: otherIndex)))
        XCTAssertEqual(defaults.data(forKey: key), remembered)
        XCTAssertEqual(try discover(catalog: catalog, ledger: ledger).verification, .spotChecked)
    }

    // MARK: - Logical provenance

    func testDifferentTreeIdentityInvalidatesEvenWhenTheSelectedPayloadIsUnchanged() async throws {
        let bytes = Data("tree-bound".utf8)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        let metadata = RepoFile(
            path: "README.md", sizeBytes: 4,
            digest: .gitBlobSHA1(hex: FileDigestComputer.gitBlobSHA1(of: Data("read".utf8))),
            isPayload: false)
        let changedTree = files + [metadata]
        XCTAssertEqual(
            ArtifactDigestPlanIdentity.compute(files.filter(\.isPayload)),
            ArtifactDigestPlanIdentity.compute(changedTree.filter(\.isPayload)))
        XCTAssertNotEqual(
            ArtifactDigestPlanIdentity.compute(files),
            ArtifactDigestPlanIdentity.compute(changedTree))

        // A current tree identity is not a catalogue opinion: only the verifier
        // supplies one, only from its own fetch, and only for the pinned commit
        // this record already names. A pinned commit is immutable, so one
        // commit presenting two trees contradicts the record itself. Unlike a
        // model or a repository disagreement, this one still deletes.
        XCTAssertNil(
            ledger.record(
                matching: lookup(
                    artifact: artifact, catalog: catalog,
                    treeIdentity: ArtifactDigestPlanIdentity.compute(changedTree))))
        XCTAssertNil(defaults.data(forKey: ledgerKey(artifact)))
        XCTAssertEqual(
            try discover(catalog: catalog, ledger: ledger).verification, .unverified)
    }

    func testLegacyCodableRecordMigratesButNeverRetainsVerifiedState() throws {
        let bytes = Data("legacy".utf8)
        try install(bytes)
        let catalog = makeCatalog(files: repoFiles(for: bytes))
        let artifact = try discover(catalog: catalog, ledger: makeLedger())
        let legacy = LegacyArtifactVerificationRecord(
            rootPath: artifact.rootPath, state: .fullyVerified,
            checkedAt: Date(timeIntervalSince1970: 1_700_000_000), detail: "legacy pass")
        let encoded = try MinirunKitJSON.encoder(pretty: false).encode(legacy)
        defaults.set(encoded, forKey: ledgerPrefix + artifact.rootPath)

        let after = try discover(catalog: catalog, ledger: makeLedger())
        XCTAssertEqual(after.verification, .unverified)
        XCTAssertNil(after.verifiedAt)

        let migratedData = try XCTUnwrap(defaults.data(forKey: ledgerPrefix + artifact.rootPath))
        let migrated = try MinirunKitJSON.decoder().decode(
            ArtifactVerificationRecord.self, from: migratedData)
        XCTAssertEqual(migrated.schemaVersion, ArtifactVerificationRecord.currentSchemaVersion)
        XCTAssertEqual(migrated.state, .unverified)
        XCTAssertNil(migrated.evidence)
    }

    func testSchemaOneEvidenceDecodesThenDowngradesWithoutTrust() async throws {
        let bytes = Data("old-evidence".utf8)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        let good = try XCTUnwrap(
            ledger.record(
                matching: lookup(
                    artifact: artifact, catalog: catalog,
                    treeIdentity: ArtifactDigestPlanIdentity.compute(files))))
        let encoded = try MinirunKitJSON.encoder(pretty: false).encode(good)
        var root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        var evidence = try XCTUnwrap(root["evidence"] as? [String: Any])
        let allFiles = try XCTUnwrap(evidence["files"] as? [[String: Any]])
        let payloads = allFiles.filter { ($0["isPayload"] as? Bool) == true }
        evidence["schemaVersion"] = 1
        evidence["payloads"] = payloads
        evidence["selectedPayloadBytes"] = NSNumber(value: bytes.count)
        for key in [
            "files", "treeFileCount", "treeMetadataFileCount", "treeBytes",
            "treePayloadBytes", "treeMetadataBytes", "selectedBytes",
        ] {
            evidence.removeValue(forKey: key)
        }
        root["evidence"] = evidence
        defaults.set(
            try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys]),
            forKey: ledgerPrefix + ArtifactVerificationPathIdentity.canonical(artifact.rootPath))

        let after = try discover(catalog: catalog, ledger: makeLedger())
        XCTAssertEqual(after.verification, .unverified)
        let migrated = try XCTUnwrap(
            makeLedger().record(
                matching: lookup(artifact: after, catalog: catalog)))
        XCTAssertEqual(migrated.state, .unverified)
        XCTAssertNil(migrated.evidence)
    }

    func testSchemaTwoEvidenceWithoutIndependentCompletenessDowngradesWithoutTrust() async throws {
        let bytes = Data("pre-authority-evidence".utf8)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        let good = try XCTUnwrap(
            ledger.record(
                matching: lookup(
                    artifact: artifact, catalog: catalog,
                    treeIdentity: ArtifactDigestPlanIdentity.compute(files))))
        let encoded = try MinirunKitJSON.encoder(pretty: false).encode(good)
        var root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        var evidence = try XCTUnwrap(root["evidence"] as? [String: Any])
        evidence["schemaVersion"] = 2
        evidence.removeValue(forKey: "completenessAuthorityIdentity")
        root["evidence"] = evidence
        defaults.set(
            try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys]),
            forKey: ledgerPrefix + ArtifactVerificationPathIdentity.canonical(artifact.rootPath))

        let after = try discover(catalog: catalog, ledger: makeLedger())
        XCTAssertEqual(after.verification, .unverified)
        XCTAssertEqual(
            makeLedger().record(matching: lookup(artifact: after, catalog: catalog))?.state,
            .unverified)
    }

    func testLegacyPathOnlyConformerDefaultsMatchingLookupToFailClosed() {
        struct LegacyLedger: ArtifactVerificationLedger {
            func record(_ record: ArtifactVerificationRecord) {}
            func record(forRootPath rootPath: String) -> ArtifactVerificationRecord? {
                ArtifactVerificationRecord(
                    rootPath: rootPath, state: .fullyVerified, checkedAt: Date(),
                    detail: "legacy path-only authority")
            }
            func forget(rootPath: String) {}
        }
        let lookup = ArtifactVerificationLookup(
            rootPath: "/tmp/artifact", model: .kimiK3,
            repository: makeCatalogRepo(), index: .unreadable)
        let legacy = LegacyLedger()
        XCTAssertEqual(legacy.record(forRootPath: lookup.rootPath)?.state, .fullyVerified)

        // The old concrete method remains source-compatible, but protocol
        // dispatch must not let it act as positive verification authority.
        let ledger: any ArtifactVerificationLedger = legacy
        XCTAssertNil(ledger.record(forRootPath: lookup.rootPath))
        XCTAssertNil(ledger.record(matching: lookup))
    }

    func testDeprecatedPathOnlyLookupsNeverReturnPositiveAuthority() async throws {
        let bytes = Data("path-only-is-not-authority".utf8)
        let (_, ledger, artifact, _) = try await verifiedFixture(bytes: bytes)
        XCTAssertNil(ledger.record(forRootPath: artifact.rootPath))

        let unverified = ArtifactVerificationRecord(
            rootPath: artifact.rootPath, state: .unverified, checkedAt: Date(),
            detail: "negative state remains source-compatible")
        ledger.record(unverified)
        let pathOnly = try XCTUnwrap(ledger.record(forRootPath: artifact.rootPath))
        XCTAssertEqual(pathOnly.rootPath, unverified.rootPath)
        XCTAssertEqual(pathOnly.state, unverified.state)
        XCTAssertEqual(pathOnly.detail, unverified.detail)

        let memory = InMemoryVerificationLedger()
        memory.record(
            ArtifactVerificationRecord(
                rootPath: artifact.rootPath, state: .fullyVerified, checkedAt: Date(),
                detail: "positive path-only claim"))
        XCTAssertNil(memory.record(forRootPath: artifact.rootPath))
    }

    func testCompareAndRemoveCannotDeleteANewerRecord() async throws {
        let bytes = Data("defaults-cas-remove".utf8)
        let (catalog, _, artifact, _) = try await verifiedFixture(bytes: bytes)
        let gate = DigestCheckpointGate(target: 1)
        let staleLedger = UserDefaultsVerificationLedger(
            defaults: defaults, prefix: ledgerPrefix,
            hooks: UserDefaultsVerificationLedgerHooks(
                beforeCompareAndSwap: { _ in gate.checkpoint() }))
        // The lookup has to be one that actually reaches the removal path. A
        // model or repository disagreement no longer does — it is the calling
        // catalogue's opinion and the record is kept — so this uses the pinned
        // commit presenting a different tree, which contradicts the record.
        let contradicting = ArtifactVerificationLookup(
            rootPath: artifact.rootPath, model: artifact.model,
            repository: catalog.descriptor(.kimiK3)?.source.repo,
            index: artifact.index,
            currentTreeIdentity: String(repeating: "b", count: 64))
        let staleLookup = Task.detached { staleLedger.record(matching: contradicting) }
        await gate.waitUntilReached()
        let newer = ArtifactVerificationRecord(
            rootPath: artifact.rootPath, state: .unverified, checkedAt: Date(),
            detail: "newer record wins")
        makeLedger().record(newer)
        gate.resume()
        let staleResult = await staleLookup.value
        XCTAssertNil(staleResult)
        let surviving = try XCTUnwrap(makeLedger().record(matching: contradicting))
        XCTAssertEqual(surviving.state, newer.state)
        XCTAssertEqual(surviving.detail, newer.detail)
    }

    func testCompareAndMigrateCannotReplaceANewerRecord() async throws {
        let bytes = Data("defaults-cas-replace".utf8)
        try install(bytes)
        let catalog = makeCatalog(files: repoFiles(for: bytes))
        let artifact = try discover(catalog: catalog, ledger: makeLedger())
        let legacy = LegacyArtifactVerificationRecord(
            rootPath: artifact.rootPath, state: .fullyVerified,
            checkedAt: Date(timeIntervalSince1970: 1_700_000_000), detail: "legacy")
        defaults.set(
            try MinirunKitJSON.encoder(pretty: false).encode(legacy),
            forKey: ledgerPrefix + artifact.rootPath)

        let gate = DigestCheckpointGate(target: 1)
        let staleLedger = UserDefaultsVerificationLedger(
            defaults: defaults, prefix: ledgerPrefix,
            hooks: UserDefaultsVerificationLedgerHooks(
                beforeCompareAndSwap: { _ in gate.checkpoint() }))
        let currentLookup = lookup(artifact: artifact, catalog: catalog)
        let staleLookup = Task.detached { staleLedger.record(matching: currentLookup) }
        await gate.waitUntilReached()
        let newer = ArtifactVerificationRecord(
            rootPath: artifact.rootPath, state: .unverified, checkedAt: Date(),
            detail: "newer record wins")
        makeLedger().record(newer)
        gate.resume()
        let staleResult = await staleLookup.value
        XCTAssertNil(staleResult)
        let surviving = try XCTUnwrap(makeLedger().record(matching: currentLookup))
        XCTAssertEqual(surviving.state, newer.state)
        XCTAssertEqual(surviving.detail, newer.detail)
    }

    func testUserDefaultsNotificationMayReenterLedgerWithoutDeadlock() throws {
        let ledger = makeLedger()
        let lookup = ArtifactVerificationLookup(
            rootPath: "/tmp/reentry", model: nil, repository: nil, index: .unreadable)
        let expectation = expectation(description: "defaults observer reentered ledger")
        let token = NotificationCenter.default.addObserver(
            forName: UserDefaults.didChangeNotification, object: defaults, queue: nil
        ) { _ in
            _ = ledger.record(matching: lookup)
            expectation.fulfill()
        }
        defer { NotificationCenter.default.removeObserver(token) }
        ledger.record(
            ArtifactVerificationRecord(
                rootPath: "/tmp/reentry", state: .unverified, checkedAt: Date(),
                detail: "reentry"))
        wait(for: [expectation], timeout: 2)
    }

    // MARK: - Verification-time races and cancellation

    func testRootedVerificationHashesHeldTreeButRejectsIntermediateSymlinkReplacement()
        async throws
    {
        let bytes = Data("rooted descriptor bytes".utf8)
        let intermediate = location.appendingPathComponent("models", isDirectory: true)
        artifactURL = intermediate.appendingPathComponent("k3", isDirectory: true)
        payloadURL = artifactURL.appendingPathComponent(Self.payloadPath)
        try install(bytes)
        let files = repoFiles(for: bytes)
        let catalog = makeCatalog(files: files)
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)
        let authority = try ArtifactVerificationRoot.open(
            relativeComponents: ["models", "k3"], beneath: location)
        let gate = DigestCheckpointGate(target: 1)
        let probe = DigestConcurrencyProbe()
        let verifier = makeVerifier(
            catalog: catalog, ledger: ledger, files: files,
            hooks: ArtifactVerifierHooks(
                digestCheckpoint: { _ in gate.checkpoint() },
                digestStarted: { _ in probe.started() },
                digestFinished: { _ in probe.finished() },
                descriptorSupportsPersistentIDs: { _, _ in true }))

        let verification = Task {
            try await verifier.verify(artifact, .full, rootedAt: authority)
        }
        await gate.waitUntilReached()
        let heldIntermediate = location.appendingPathComponent(
            "held-models", isDirectory: true)
        let replacement = location.appendingPathComponent(
            "replacement", isDirectory: true)
        do {
            try FileManager.default.createDirectory(
                at: replacement.appendingPathComponent("k3", isDirectory: true),
                withIntermediateDirectories: true)
            try FileManager.default.moveItem(at: intermediate, to: heldIntermediate)
            try FileManager.default.createSymbolicLink(
                at: intermediate, withDestinationURL: replacement)
        } catch {
            gate.resume()
            _ = try? await verification.value
            throw error
        }
        gate.resume()

        do {
            _ = try await verification.value
            XCTFail("an intermediate symlink replacement earned verification evidence")
        } catch let error as ArtifactVerificationError {
            guard case .filesystemChanged = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
        // Only the first file was open when the rename happened. Reaching all
        // three digest starts proves subsequent payload opens stayed beneath
        // the held artifact fd instead of following the replacement pathname.
        XCTAssertEqual(probe.starts, files.count)
        XCTAssertEqual(probe.peak, 1)
        XCTAssertNil(defaults.data(forKey: ledgerPrefix + artifact.rootPath))
    }

    func testRuntimeAuthorityOpensOnlyVerifiedObjectsAndDetectsReplacement() async throws {
        let original = Data("runtime-authorized-bytes".utf8)
        let fixture = try await verifiedFixture(bytes: original)
        let discovered = try discover(catalog: fixture.catalog, ledger: fixture.ledger)
        let record = try XCTUnwrap(
            fixture.ledger.record(
                matching: lookup(
                    artifact: discovered, catalog: fixture.catalog,
                    treeIdentity: ArtifactDigestPlanIdentity.compute(fixture.files))))
        let evidence = try XCTUnwrap(record.evidence)
        let root = try ArtifactVerificationRoot.open(
            relativeComponents: ["artifact"], beneath: location)
        let authority = try ArtifactRuntimeAuthority(root: root, evidence: evidence)

        XCTAssertThrowsError(try authority.openFile("not-in-the-published-tree.bin"))
        let held = try authority.openFile(Self.payloadPath)
        XCTAssertEqual(try held.readAll(maximumBytes: 1_024), original)

        try Data("replacement-with-other-identity".utf8).write(
            to: payloadURL, options: .atomic)
        XCTAssertThrowsError(
            try held.readAll(maximumBytes: 1_024),
            "an unlinked verified object must no longer be accepted as current")
        XCTAssertThrowsError(try authority.openFile(Self.payloadPath))
        XCTAssertThrowsError(try authority.validateAllFilesCurrent())
    }

    // MARK: - Descriptor-bound removal

    func testArtifactRemovalDeletesOnlyTheOpenedArtifactAndNeverFollowsSymlinks() throws {
        let artifact = location.appendingPathComponent("artifact-to-remove", isDirectory: true)
        let nested = artifact.appendingPathComponent("layers/00", isDirectory: true)
        let outside = location.appendingPathComponent("outside.txt", isDirectory: false)
        let sibling = location.appendingPathComponent("keep-me", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: sibling, withIntermediateDirectories: true)
        let first = Data("first".utf8)
        let second = Data("second payload".utf8)
        try first.write(to: artifact.appendingPathComponent("index.json"))
        try second.write(to: nested.appendingPathComponent("weights.bin"))
        try Data("outside".utf8).write(to: outside)
        try FileManager.default.createSymbolicLink(
            at: artifact.appendingPathComponent("outside-link"), withDestinationURL: outside)
        let root = try ArtifactVerificationRoot.open(
            relativeComponents: ["artifact-to-remove"], beneath: location)

        let receipt = try root.removeRecursively()

        XCTAssertEqual(receipt.rootPath, artifact.path)
        XCTAssertEqual(receipt.removedEntryCount, 5)
        XCTAssertEqual(receipt.removedRegularFileBytes, UInt64(first.count + second.count))
        XCTAssertFalse(FileManager.default.fileExists(atPath: artifact.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: outside.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: sibling.path))
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: location.path)
        XCTAssertFalse(leftovers.contains { $0.hasPrefix(".minirun-removing-") })
    }

    func testArtifactRemovalRefusesTheRegisteredLocationItself() throws {
        let marker = location.appendingPathComponent("keep.txt")
        try Data("keep".utf8).write(to: marker)
        let root = try ArtifactVerificationRoot.open(relativeComponents: [], beneath: location)

        XCTAssertThrowsError(try root.removeRecursively()) { error in
            XCTAssertEqual(
                error as? ArtifactRemovalError,
                .refusesRegisteredLocation(path: self.location.path))
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: marker.path))
    }

    func testArtifactRemovalReversesAtomicSwapWhenVisiblePathWasReplaced() throws {
        let artifact = location.appendingPathComponent("artifact", isDirectory: true)
        let original = artifact.appendingPathComponent("original.bin")
        try FileManager.default.createDirectory(at: artifact, withIntermediateDirectories: true)
        try Data("original".utf8).write(to: original)
        let held = location.appendingPathComponent("held-original", isDirectory: true)
        let root = try ArtifactVerificationRoot.open(
            relativeComponents: ["artifact"], beneath: location)

        XCTAssertThrowsError(
            try root.removeRecursively(
                hooks: ArtifactRemovalHooks(beforeAtomicIsolation: {
                    do {
                        try FileManager.default.moveItem(at: artifact, to: held)
                        try FileManager.default.createDirectory(
                            at: artifact, withIntermediateDirectories: true)
                        try Data("replacement".utf8).write(
                            to: artifact.appendingPathComponent("replacement.bin"))
                    } catch {
                        XCTFail("race fixture failed: \(error)")
                    }
                }))) { error in
                    XCTAssertEqual(
                        error as? ArtifactRemovalError,
                        .changed(path: artifact.path))
                }

        XCTAssertEqual(
            try Data(contentsOf: held.appendingPathComponent("original.bin")),
            Data("original".utf8))
        XCTAssertEqual(
            try Data(contentsOf: artifact.appendingPathComponent("replacement.bin")),
            Data("replacement".utf8))
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: location.path)
        XCTAssertFalse(leftovers.contains { $0.hasPrefix(".minirun-removing-") })
    }

    func testOneVerifierNeverRunsTwoDigestPassesConcurrently() async throws {
        let bytes = Data(repeating: 0x31, count: FileDigestComputer.chunkBytes * 2)
        try install(bytes)
        let files = repoFiles(for: bytes)
        let catalog = makeCatalog(files: files)
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)
        let gate = DigestCheckpointGate(target: 1)
        let probe = DigestConcurrencyProbe()
        let verifier = makeVerifier(
            catalog: catalog, ledger: ledger, files: files,
            hooks: ArtifactVerifierHooks(
                digestCheckpoint: { _ in gate.checkpoint() },
                digestStarted: { _ in probe.started() },
                digestFinished: { _ in probe.finished() },
                volumeSupportsPersistentIDs: { _ in true }))
        let first = Task { try await verifier.verify(artifact, .full) }
        await gate.waitUntilReached()
        do {
            _ = try await verifier.verify(artifact, .full)
            XCTFail("the second pass was not rejected")
        } catch let error as ArtifactVerificationError {
            XCTAssertEqual(error, .verificationAlreadyInProgress)
        }
        gate.resume()
        let firstReport = try await first.value
        XCTAssertTrue(firstReport.isComplete)
        XCTAssertEqual(probe.peak, 1)
    }

    func testSeparateVerifiersSerializeTheSameArtifact() async throws {
        let bytes = Data(repeating: 0x32, count: FileDigestComputer.chunkBytes * 2)
        try install(bytes)
        let files = repoFiles(for: bytes)
        let catalog = makeCatalog(files: files)
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)
        let gate = DigestCheckpointGate(target: 1)
        let firstVerifier = makeVerifier(
            catalog: catalog, ledger: ledger, files: files,
            hooks: ArtifactVerifierHooks(digestCheckpoint: { _ in gate.checkpoint() }))
        let secondVerifier = makeVerifier(catalog: catalog, ledger: ledger, files: files)

        let first = Task { try await firstVerifier.verify(artifact, .full) }
        await gate.waitUntilReached()
        do {
            _ = try await secondVerifier.verify(artifact, .full)
            XCTFail("separate verifier instances hashed one artifact concurrently")
        } catch let error as ArtifactVerificationError {
            XCTAssertEqual(error, .verificationAlreadyInProgress)
        }
        gate.resume()
        let firstReport = try await first.value
        XCTAssertTrue(firstReport.isComplete)
    }

    func testTransientMissingRootSuppressesButDoesNotEraseEvidence() async throws {
        let bytes = Data("removable-volume-transient".utf8)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        let current = lookup(
            artifact: artifact, catalog: catalog,
            treeIdentity: ArtifactDigestPlanIdentity.compute(files))
        let key = ledgerPrefix + ArtifactVerificationPathIdentity.canonical(artifact.rootPath)
        let remembered = try XCTUnwrap(defaults.data(forKey: key))
        let held = location.appendingPathComponent(".temporarily-unmounted-artifact")
        try FileManager.default.moveItem(at: artifactURL, to: held)
        XCTAssertNil(ledger.record(matching: current))
        XCTAssertEqual(defaults.data(forKey: key), remembered)

        // A rename is only a portable stand-in for temporary unavailability;
        // it may also change the directory's status-change timestamp. Restoring
        // the pathname does not itself resurrect authority; the next matching
        // lookup must perform the ordinary physical-identity validation.
        try FileManager.default.moveItem(at: held, to: artifactURL)
        XCTAssertEqual(defaults.data(forKey: key), remembered)
    }

    func testUnsupportedPersistentIDVolumeIsNamedAndDoesNotDigest() async throws {
        let bytes = Data("unsupported-volume".utf8)
        try install(bytes)
        let files = repoFiles(for: bytes)
        let catalog = makeCatalog(files: files)
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)
        for capability: Bool? in [false, nil] {
            let probe = DigestConcurrencyProbe()
            let verifier = makeVerifier(
                catalog: catalog, ledger: ledger, files: files,
                hooks: ArtifactVerifierHooks(
                    digestStarted: { _ in probe.started() },
                    digestFinished: { _ in probe.finished() },
                    volumeSupportsPersistentIDs: { _ in capability }))
            do {
                _ = try await verifier.verify(artifact, .full)
                XCTFail("unsupported or unknown filesystem capability was accepted")
            } catch let error as ArtifactVerificationError {
                XCTAssertEqual(
                    error, .persistentFileIDsUnavailable(path: artifact.rootPath))
            }
            XCTAssertEqual(probe.starts, 0)
        }
    }

    func testRootDescriptorClosesWhenInitialIdentityReadFails() throws {
        try FileManager.default.createDirectory(
            at: artifactURL, withIntermediateDirectories: true)
        var captured: Int32 = -1
        XCTAssertThrowsError(
            try ArtifactFilesystemEvidence.openRootAndIdentity(at: artifactURL) { descriptor, _ in
                captured = descriptor
                throw InjectedFilesystemFailure.stat
            })
        XCTAssertGreaterThanOrEqual(captured, 0)
        errno = 0
        XCTAssertEqual(fcntl(captured, F_GETFD), -1)
        XCTAssertEqual(errno, EBADF)
    }

    func testSecondFilesystemIdentityPassDetectsEarlierFileMutation() async throws {
        let bytes = Data("two-pass".utf8)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        let record = try XCTUnwrap(
            ledger.record(
                matching: lookup(
                    artifact: artifact, catalog: catalog,
                    treeIdentity: ArtifactDigestPlanIdentity.compute(files))))
        let evidence = try XCTUnwrap(record.evidence)
        let changed = Data("{\"offset\":1}".utf8)
        let manifestURL = artifactURL.appendingPathComponent(Self.manifestPath)
        XCTAssertThrowsError(
            try ArtifactFilesystemEvidence.validate(
                evidence: evidence, at: artifactURL,
                hooks: ArtifactFilesystemValidationHooks(
                    betweenIdentityPasses: {
                        try! changed.write(to: manifestURL)
                    }))) { error in
                        guard case ArtifactFilesystemEvidenceError.changed(let path) = error else {
                            return XCTFail("unexpected error: \(error)")
                        }
                        XCTAssertEqual(path, Self.manifestPath)
                    }
    }

    func testMutationBetweenDigestChunksFailsClosedAndClearsOldAuthority() async throws {
        let byteCount = FileDigestComputer.chunkBytes * 3
        let bytes = Data(repeating: 0x41, count: byteCount)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        // Full verification hashes index + manifest before the large payload;
        // the fourth checkpoint is the payload's second chunk.
        let gate = DigestCheckpointGate(target: 4)
        let verifier = makeVerifier(
            catalog: catalog, ledger: ledger, files: files,
            hooks: ArtifactVerifierHooks(digestCheckpoint: { _ in gate.checkpoint() }))

        let task = Task { try await verifier.verify(artifact, .full) }
        await gate.waitUntilReached()
        let handle = try FileHandle(forWritingTo: payloadURL)
        try handle.seek(toOffset: UInt64(FileDigestComputer.chunkBytes + 17))
        try handle.write(contentsOf: Data([0x42]))
        try handle.synchronize()
        try handle.close()
        gate.resume()

        do {
            let report = try await task.value
            XCTAssertFalse(report.isComplete)
            XCTAssertTrue(
                report.wrongDigest.contains { $0.path == Self.payloadPath },
                "a mutation not visible in timestamps must still fail its digest")
        } catch let error as ArtifactVerificationError {
            guard case .filesystemChanged(let path, _) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(path, Self.payloadPath)
        }
        XCTAssertEqual(
            try discover(catalog: catalog, ledger: ledger).verification, .unverified)
    }

    func testCancellationDuringDetachedDigestPreservesApplicablePriorRecord() async throws {
        let byteCount = FileDigestComputer.chunkBytes * 3
        let bytes = Data(repeating: 0x5A, count: byteCount)
        let (catalog, ledger, artifact, files) = try await verifiedFixture(bytes: bytes)
        let before = try discover(catalog: catalog, ledger: ledger)
        XCTAssertEqual(before.verification, .fullyVerified)
        let priorDate = try XCTUnwrap(before.verifiedAt)

        let gate = DigestCheckpointGate(target: 4)
        let verifier = makeVerifier(
            catalog: catalog, ledger: ledger, files: files,
            hooks: ArtifactVerifierHooks(digestCheckpoint: { _ in gate.checkpoint() }))
        let task = Task { try await verifier.verify(artifact, .full) }
        await gate.waitUntilReached()
        task.cancel()
        gate.resume()

        do {
            _ = try await task.value
            XCTFail("cancelled verification completed")
        } catch let error as ArtifactVerificationError {
            XCTAssertEqual(error, .cancelled)
        }
        let after = try discover(catalog: catalog, ledger: ledger)
        XCTAssertEqual(after.verification, .fullyVerified)
        XCTAssertEqual(after.verifiedAt, priorDate)
    }

    // MARK: - Untrusted arithmetic

    func testVerificationArithmeticRefusesUInt64OverflowByName() throws {
        let digest = FileDigest.sha256(hex: String(repeating: "a", count: 64))
        let files = [
            RepoFile(path: "unit/a.bin", sizeBytes: .max, digest: digest, isPayload: true),
            RepoFile(path: "unit/b.bin", sizeBytes: 1, digest: digest, isPayload: true),
        ]

        assertArithmeticOverflow {
            _ = try ArtifactVerifier.select(files, for: .full)
        }
        assertArithmeticOverflow {
            _ = try ArtifactVerifier.checkedByteTotal(files, context: "test total")
        }
        assertArithmeticOverflow {
            _ = try ArtifactVerifier.bytesToRefetch(files, paths: files.map(\.path))
        }

        // The spot selector does not form `max + 1`: after taking the first
        // file it proves the next one cannot fit using subtraction.
        XCTAssertEqual(
            try ArtifactVerifier.select(files, for: .spotCheck(maximumBytes: .max)).map(\.path),
            ["unit/a.bin"])
    }

    // MARK: - Helpers

    private func install(_ payload: Data) throws {
        try FileManager.default.createDirectory(
            at: payloadURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(ArtifactDiscoveryTests.k3Index.utf8).write(
            to: artifactURL.appendingPathComponent("index.json"))
        try Self.manifestData.write(
            to: artifactURL.appendingPathComponent(Self.manifestPath))
        try payload.write(to: payloadURL)
    }

    private func makeLedger() -> UserDefaultsVerificationLedger {
        UserDefaultsVerificationLedger(defaults: defaults, prefix: ledgerPrefix)
    }

    /// The exact defaults key the ledger files this artifact under. Reading the
    /// stored bytes is the only way to tell a withheld record from a deleted
    /// one; both return nil to the caller.
    private func ledgerKey(_ artifact: DiscoveredArtifact) -> String {
        ledgerPrefix + ArtifactVerificationPathIdentity.canonical(artifact.rootPath)
    }

    private func storedEvidence(_ data: Data) throws -> ArtifactVerificationEvidence? {
        try MinirunKitJSON.decoder()
            .decode(ArtifactVerificationRecord.self, from: data).evidence
    }

    /// Change one file's modification and status-change times without touching
    /// its parent directory, its length, or its inode.
    private func rewriteFirstByte(of url: URL, to byte: UInt8) throws {
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: Data([byte]))
        try handle.synchronize()
    }

    private func makeCatalogRepo(
        repoID: String = "fixture/K3-minirun",
        revision: String = ArtifactVerificationEvidenceTests.revisionA
    ) -> HuggingFaceRepoRef {
        HuggingFaceRepoRef(repoID: repoID, revision: revision)
    }

    private func makeCatalog(
        files: [RepoFile],
        repoID: String = "fixture/K3-minirun",
        revision: String = ArtifactVerificationEvidenceTests.revisionA
    ) -> ModelCatalogSnapshot {
        let base = ModelCatalog.bundled.descriptor(.kimiK3)!
        let payload = files.filter(\.isPayload)
        let metadata = files.filter { !$0.isPayload }
        let descriptor = ModelDescriptor(
            id: base.id, displayName: base.displayName, architecture: base.architecture,
            layout: base.layout,
            source: .huggingFaceRepo(makeCatalogRepo(repoID: repoID, revision: revision)),
            payloadBytes: payload.reduce(0) { $0 + $1.sizeBytes },
            metadataBytes: metadata.reduce(0) { $0 + $1.sizeBytes },
            payloadFileCount: payload.count,
            metadataFileCount: metadata.count,
            largestFileBytes: files.map(\.sizeBytes).max() ?? 0,
            minimumBudgetBytes: base.minimumBudgetBytes, runner: base.runner,
            licenseName: base.licenseName,
            licenseAcknowledgementRequired: base.licenseAcknowledgementRequired,
            notes: base.notes)
        return ModelCatalogSnapshot(generatedAt: Date(), origin: .bundled, models: [descriptor])
    }

    private func repoFiles(for payload: Data) -> [RepoFile] {
        let index = Data(ArtifactDiscoveryTests.k3Index.utf8)
        return [
            RepoFile(
                path: Self.payloadPath, sizeBytes: UInt64(payload.count),
                digest: .sha256(hex: FileDigestComputer.sha256(of: payload)), isPayload: true),
            RepoFile(
                path: "index.json", sizeBytes: UInt64(index.count),
                digest: .gitBlobSHA1(hex: FileDigestComputer.gitBlobSHA1(of: index)),
                isPayload: false),
            RepoFile(
                path: Self.manifestPath, sizeBytes: UInt64(Self.manifestData.count),
                digest: .gitBlobSHA1(
                    hex: FileDigestComputer.gitBlobSHA1(of: Self.manifestData)),
                isPayload: false),
        ]
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

    private func makeVerifier(
        catalog: ModelCatalogSnapshot, ledger: any ArtifactVerificationLedger,
        files: [RepoFile], authorityFiles: [RepoFile]? = nil,
        hooks: ArtifactVerifierHooks = .none
    ) -> ArtifactVerifier {
        let tree = HuggingFaceTreeClient(
            transport: VerificationTreeTransport(
                body: treeBody(files), authorityPaths: authorityFiles?.map(\.path)),
            endpoint: URL(string: "https://verification.invalid")!)
        return ArtifactVerifier(tree: tree, catalog: catalog, ledger: ledger, hooks: hooks)
    }

    private func discover(
        catalog: ModelCatalogSnapshot, ledger: any ArtifactVerificationLedger
    ) throws -> DiscoveredArtifact {
        let artifacts = ArtifactLocator(catalog: catalog, verificationLedger: ledger)
            .scan(location).artifacts
        XCTAssertEqual(artifacts.count, 1)
        return try XCTUnwrap(artifacts.first)
    }

    private func lookup(
        artifact: DiscoveredArtifact, catalog: ModelCatalogSnapshot,
        treeIdentity: String? = nil
    ) -> ArtifactVerificationLookup {
        ArtifactVerificationLookup(
            rootPath: artifact.rootPath, model: artifact.model,
            repository: artifact.model.flatMap { catalog.descriptor($0)?.source.repo },
            index: artifact.index, currentTreeIdentity: treeIdentity)
    }

    private func verifiedFixture(
        bytes: Data
    ) async throws -> (
        catalog: ModelCatalogSnapshot, ledger: UserDefaultsVerificationLedger,
        artifact: DiscoveredArtifact, files: [RepoFile]
    ) {
        try install(bytes)
        let files = repoFiles(for: bytes)
        let catalog = makeCatalog(files: files)
        let ledger = makeLedger()
        let artifact = try discover(catalog: catalog, ledger: ledger)
        let report = try await makeVerifier(
            catalog: catalog, ledger: ledger, files: files
        ).verify(artifact, .full)
        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(
            try discover(catalog: catalog, ledger: ledger).verification, .fullyVerified)
        return (catalog, ledger, artifact, files)
    }

    private func rootIdentity(at url: URL) throws -> ArtifactFilesystemIdentity {
        let descriptor = try ArtifactFilesystemEvidence.openRoot(at: url)
        defer { _ = close(descriptor) }
        return try ArtifactFilesystemEvidence.identity(
            of: descriptor, path: url.path, expectedDirectory: true)
    }

    private func copy(
        _ source: ArtifactVerificationEvidence,
        persistentVolume: ArtifactPersistentVolumeIdentity?
    ) -> ArtifactVerificationEvidence {
        ArtifactVerificationEvidence(
            schemaVersion: source.schemaVersion, model: source.model,
            repository: source.repository, treeIdentity: source.treeIdentity,
            completenessAuthorityIdentity: source.completenessAuthorityIdentity,
            selectedPlanIdentity: source.selectedPlanIdentity,
            treeFileCount: source.treeFileCount,
            treePayloadFileCount: source.treePayloadFileCount,
            treeMetadataFileCount: source.treeMetadataFileCount,
            treeBytes: source.treeBytes,
            treePayloadBytes: source.treePayloadBytes,
            treeMetadataBytes: source.treeMetadataBytes,
            selectedBytes: source.selectedBytes, index: source.index,
            persistentVolume: persistentVolume, root: source.root,
            files: source.files)
    }

    private func assertArithmeticOverflow(
        _ operation: () throws -> Void, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertThrowsError(try operation(), file: file, line: line) { error in
            guard case ArtifactVerificationError.arithmeticOverflow = error else {
                return XCTFail("unexpected error: \(error)", file: file, line: line)
            }
        }
    }
}
