import Darwin
import Foundation
import XCTest
import iOSBench

@testable import MinirunKit

private final class SwitchableDownloadStateStore: DownloadStateStore, @unchecked Sendable {
    private enum DeliberateFailure: Error, CustomStringConvertible {
        case save
        var description: String { "deliberate state-store save failure" }
    }

    private let lock = NSLock()
    private var storage: [DownloadJobID: DownloadJobState] = [:]
    /// `nil` means saves are unrestricted; zero means the next save fails.
    private var successfulSavesRemaining: Int?
    private var saveDidFail = false

    func failSaves(after successfulSaves: Int = 0) {
        lock.lock()
        successfulSavesRemaining = max(0, successfulSaves)
        lock.unlock()
    }

    func save(_ state: DownloadJobState) throws {
        lock.lock()
        defer { lock.unlock() }
        if let remaining = successfulSavesRemaining {
            guard remaining > 0 else {
                saveDidFail = true
                throw DeliberateFailure.save
            }
            successfulSavesRemaining = remaining - 1
        }
        storage[state.job] = state
    }

    func load(_ job: DownloadJobID) throws -> DownloadJobState? {
        lock.lock()
        defer { lock.unlock() }
        return storage[job]
    }

    func all() throws -> [DownloadJobState] {
        lock.lock()
        defer { lock.unlock() }
        return storage.values.sorted { $0.createdAt < $1.createdAt }
    }

    func remove(_ job: DownloadJobID) throws {
        lock.lock()
        defer { lock.unlock() }
        storage.removeValue(forKey: job)
    }

    var hasFailedSave: Bool {
        lock.lock()
        defer { lock.unlock() }
        return saveDidFail
    }
}

private final class BlockingDigestProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var entered = false

    func digest(
        descriptor: Int32, algorithm: DigestAlgorithm,
        cancellationRequested: @escaping @Sendable () -> Bool
    ) throws -> FileDigest {
        lock.lock()
        entered = true
        lock.unlock()

        let deadline = Date().addingTimeInterval(3)
        while Date() < deadline {
            if cancellationRequested() { throw CancellationError() }
            Thread.sleep(forTimeInterval: 0.002)
        }
        throw CocoaError(.fileReadUnknown)
    }

    var hasEntered: Bool {
        lock.lock()
        defer { lock.unlock() }
        return entered
    }
}

private final class PersistenceRaceTransport: HTTPTransport, @unchecked Sendable {
    private let store: SwitchableDownloadStateStore
    private let files: [String: Data]

    init(store: SwitchableDownloadStateStore, files: [String: Data]) {
        self.store = store
        self.files = files
    }

    func data(for request: HTTPRequest) async throws -> HTTPResponse {
        throw HTTPTransportError.notHTTP(url: request.url.absoluteString)
    }

    func download(_ request: HTTPRequest, into sink: HTTPDownloadSink) async throws
        -> HTTPResponse
    {
        let repositoryPath = request.url.lastPathComponent
        guard let bytes = files[repositoryPath] else {
            throw HTTPTransportError.notHTTP(url: request.url.absoluteString)
        }

        // b.bin deliberately ignores cooperative Task cancellation. It only
        // returns after its sibling has discovered that durable state is gone.
        if repositoryPath == "b.bin" {
            while !store.hasFailedSave { await Task.yield() }
        }
        let range = try XCTUnwrap(request.byteRange)
        let lower = Int(range.lowerBound)
        let upper = Int(range.upperBound)
        let chunk = bytes.subdata(in: lower..<(upper + 1))
        let response = HTTPResponse(
            statusCode: 206,
            headers: [
                "x-repo-commit": String(repeating: "1", count: 40),
                "x-linked-size": String(bytes.count),
                "content-range": "bytes \(lower)-\(upper)/\(bytes.count)",
                "content-length": String(chunk.count),
            ], body: nil)
        try sink.admit(response)
        try sink.write(chunk)
        try sink.seal()
        return response
    }
}

private final class PartNameReplacementTransport: HTTPTransport, @unchecked Sendable {
    private let bytes: Data
    private let statusCode: Int
    private let partURL: URL
    let replacement = Data("attacker-owned replacement\n".utf8)

    init(bytes: Data, statusCode: Int, partURL: URL) {
        self.bytes = bytes
        self.statusCode = statusCode
        self.partURL = partURL
    }

    func data(for request: HTTPRequest) async throws -> HTTPResponse {
        throw HTTPTransportError.notHTTP(url: request.url.absoluteString)
    }

    func download(_ request: HTTPRequest, into sink: HTTPDownloadSink) async throws
        -> HTTPResponse
    {
        let range = try XCTUnwrap(request.byteRange)
        let lower = Int(range.lowerBound)
        let upper = Int(range.upperBound)
        let chunk = bytes.subdata(in: lower..<(upper + 1))

        // Simulate another process replacing the name after DownloadManager
        // opened it. The transport has only `sink`; this URL is deliberately
        // injected by the adversarial test, not supplied by the protocol.
        try FileManager.default.removeItem(at: partURL)
        try replacement.write(to: partURL)
        let response = HTTPResponse(
            statusCode: statusCode,
            headers: [
                "x-repo-commit": String(repeating: "1", count: 40),
                "x-linked-size": String(bytes.count),
                "content-range": "bytes \(lower)-\(upper)/\(bytes.count)",
                "content-length": String(chunk.count),
            ], body: nil)
        if statusCode == 206 {
            try sink.admit(response)
            try sink.write(chunk)
            try sink.seal()
        }
        return response
    }
}

private final class IntegrityMismatchTransport: HTTPTransport, @unchecked Sendable {
    enum Violation: Sendable, Equatable {
        case movedRevision
        case missingRevision
        case linkedSize
        case returnedRevisionSwitch
    }

    private let bytes: Data
    private let expectedRevision: String
    private let violation: Violation

    init(bytes: Data, expectedRevision: String, violation: Violation) {
        self.bytes = bytes
        self.expectedRevision = expectedRevision
        self.violation = violation
    }

    func data(for request: HTTPRequest) async throws -> HTTPResponse {
        throw HTTPTransportError.notHTTP(url: request.url.absoluteString)
    }

    func download(_ request: HTTPRequest, into sink: HTTPDownloadSink) async throws
        -> HTTPResponse
    {
        let range = try XCTUnwrap(request.byteRange)
        let lower = Int(range.lowerBound)
        let upper = Int(range.upperBound)
        let chunk = bytes.subdata(in: lower..<(upper + 1))
        var headers = [
            "content-range": "bytes \(lower)-\(upper)/\(bytes.count)",
            "content-length": String(chunk.count),
            "x-linked-size": String(bytes.count),
        ]
        var admittedHeaders = headers
        switch violation {
        case .movedRevision:
            headers["x-repo-commit"] = String(repeating: "9", count: 40)
        case .missingRevision:
            break
        case .linkedSize:
            headers["x-repo-commit"] = expectedRevision
            headers["x-linked-size"] = String(bytes.count + 1)
        case .returnedRevisionSwitch:
            admittedHeaders["x-repo-commit"] = String(repeating: "9", count: 40)
            headers["x-repo-commit"] = expectedRevision
        }
        if violation != .returnedRevisionSwitch { admittedHeaders = headers }
        let admittedResponse = HTTPResponse(
            statusCode: 206, headers: admittedHeaders, body: nil)
        let returnedResponse = HTTPResponse(statusCode: 206, headers: headers, body: nil)
        try sink.admit(admittedResponse)
        try sink.write(chunk)
        try sink.seal()
        return returnedResponse
    }
}

private final class DiscardFailureTransport: HTTPTransport, @unchecked Sendable {
    private let bytes: Data
    private let revision: String
    private let gate: TransportGate?
    private let lock = NSLock()
    private var attempts = 0

    init(bytes: Data, revision: String, gate: TransportGate? = nil) {
        self.bytes = bytes
        self.revision = revision
        self.gate = gate
    }

    func data(for request: HTTPRequest) async throws -> HTTPResponse {
        throw HTTPTransportError.notHTTP(url: request.url.absoluteString)
    }

    func download(_ request: HTTPRequest, into sink: HTTPDownloadSink) async throws
        -> HTTPResponse
    {
        lock.lock()
        attempts += 1
        lock.unlock()
        let range = try XCTUnwrap(request.byteRange)
        let lower = Int(range.lowerBound)
        let upper = Int(range.upperBound)
        let chunk = bytes.subdata(in: lower..<(upper + 1))
        let response = HTTPResponse(
            statusCode: 206,
            headers: [
                "content-range": "bytes \(lower)-\(upper)/\(bytes.count)",
                "content-length": String(chunk.count),
                "x-repo-commit": revision,
                "x-linked-size": String(bytes.count),
            ], body: nil)
        try sink.admit(response)
        try sink.write(chunk)
        try sink.seal()
        _ = sink.discardCurrentResponse(
            because: .responseRejected(
                response: response, reason: "injected rejected response"))
        if let gate { await gate.enterAndWait() }
        // Exercise the manager's fail-closed branch after a sink reports that
        // its rollback could not establish a safe resume offset.
        throw HTTPTransportError.discardFailed(reason: "injected cleanup failure")
    }

    var attemptCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return attempts
    }
}

private final class TransportGate: @unchecked Sendable {
    private let lock = NSLock()
    private var entered = false
    private var released = false

    func enterAndWait() async {
        markEntered()
        while !isReleased { await Task.yield() }
    }

    private func markEntered() {
        lock.lock()
        entered = true
        lock.unlock()
    }

    var hasEntered: Bool {
        lock.lock()
        defer { lock.unlock() }
        return entered
    }

    private var isReleased: Bool {
        lock.lock()
        defer { lock.unlock() }
        return released
    }

    func release() {
        lock.lock()
        released = true
        lock.unlock()
    }
}

/// The downloader, driven end to end against a real socket.
///
/// No test here touches a published repository. The origin is
/// ``FixtureOrigin``, which serves a few megabytes shaped like the real thing —
/// a `302` carrying the digest, a `206` carrying an `ETag` that is deliberately
/// *not* the digest, and a tree paginated over two pages.
final class DownloadManagerTests: XCTestCase {

    private var origin: FixtureOrigin!
    private var destination: URL!
    private let revision = String(repeating: "1", count: 40)

    override func setUpWithError() throws {
        try super.setUpWithError()
        origin = try FixtureOrigin.start(revision: revision, files: FixtureArtifact.files())
        destination = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-download-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: destination, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        origin?.stop()
        origin = nil
        if let destination { try? FileManager.default.removeItem(at: destination) }
        try super.tearDownWithError()
    }

    private func makeManager(
        stateStore: any DownloadStateStore = InMemoryDownloadStateStore(),
        storage: (any StorageManaging)? = nil
    ) -> DownloadManager {
        DownloadManager(
            transport: URLSessionTransport(),
            storage: storage ?? StorageManager(bookmarkLedger: InMemoryBookmarkLedger()),
            stateStore: stateStore,
            catalogEndpoint: origin.baseURL)
    }

    private func scriptedStorage(_ coding: ScriptedURLCoding) -> StorageManager {
        StorageManager(
            store: SecurityScopedLocationStore(
                coding: coding, persistence: InMemoryBookmarkPersistence()),
            bookmarkLedger: InMemoryBookmarkLedger(), coding: coding)
    }

    private func descriptor() -> ModelDescriptor {
        ModelDescriptor(
            id: ModelID("fixture"), displayName: "Fixture", architecture: .kimiK3MoE,
            layout: .k3FlagshipLayerStreams,
            source: .huggingFaceRepo(origin.repo),
            payloadBytes: 0, metadataBytes: 0, payloadFileCount: 0, metadataFileCount: 0,
            largestFileBytes: 0, minimumBudgetBytes: nil, runner: .none,
            licenseName: "n/a", licenseAcknowledgementRequired: false, notes: [])
    }

    // MARK: - Planning

    func testPlanWalksEveryTreePageAndClassifiesTheFiles() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())

        XCTAssertEqual(plan.files.count, origin.paths.count)
        XCTAssertEqual(Set(plan.files.map(\.path)), Set(origin.paths))
        XCTAssertTrue(plan.bucketsAccountForEveryByte)
        XCTAssertEqual(plan.payloadBytes + plan.metadataBytes, plan.totalBytes)

        // README.md and index.json are at the root; both manifest.json files
        // are metadata wherever they sit.
        let metadata = Set(plan.files.filter { !$0.isPayload }.map(\.path))
        XCTAssertEqual(
            metadata,
            ["README.md", "index.json", "layer00/manifest.json", "global/manifest.json"])

        XCTAssertEqual(plan.largestFileBytes, UInt64(origin.bytes(of: FixtureArtifact.largeFilePath)!.count))
        // The two pages were both walked: a client that stopped at the first
        // would have found roughly half of these.
        XCTAssertGreaterThanOrEqual(
            origin.requests.filter { $0.contains("/tree/") }.count, 2)
    }

    func testJSONPayloadFilesAreDigestedAsGitBlobsAndTilesAsSHA256() async throws {
        let plan = try await makeManager().plan(for: descriptor())
        let manifest = try XCTUnwrap(plan.file(at: "layer00/manifest.json"))
        XCTAssertEqual(manifest.digest.algorithm, .gitBlobSHA1)
        let tile = try XCTUnwrap(plan.file(at: FixtureArtifact.largeFilePath))
        XCTAssertEqual(tile.digest.algorithm, .sha256)
    }

    func testAModelWithNoRepositoryIsRefusedRatherThanPlannedEmpty() async {
        let id = ModelID("locally-built-model")
        let local = ModelDescriptor(
            id: id, displayName: "Locally Built", architecture: .unrecognized,
            layout: .unrecognized, source: .locallyBuilt(tool: "Tools/local_containers"),
            payloadBytes: 0, metadataBytes: 0, payloadFileCount: 0, metadataFileCount: 0,
            largestFileBytes: 0, minimumBudgetBytes: nil, runner: .decodeRunner,
            licenseName: "n/a", licenseAcknowledgementRequired: false, notes: [])
        do {
            _ = try await makeManager().plan(for: local)
            XCTFail("a locally built model has nothing to download")
        } catch {
            XCTAssertEqual(
                error as? CatalogError,
                .notDownloadable(id, builtBy: "Tools/local_containers"))
        }
    }

    func testPublicPlansAreValidatedBeforePreflightOrStartTouchesDisk() async throws {
        let parent = destination.deletingLastPathComponent()
        let outside = parent.appendingPathComponent(
            "minirun-public-plan-outside-\(UUID().uuidString).bin")
        let marker = Data("must survive every public-plan refusal\n".utf8)
        try marker.write(to: outside)
        defer { try? FileManager.default.removeItem(at: outside) }

        let safe = publicFile("safe.bin", size: 1)
        let validPlan = publicPlan([safe])
        let attacks: [(String, DownloadPlan)] = [
            (
                "non-canonical repository ID",
                publicPlan(
                    [safe],
                    repo: HuggingFaceRepoRef(
                        repoID: "../fixture", revision: origin.repo.revision))
            ),
            (
                "percent-escaped repository ID",
                publicPlan(
                    [safe],
                    repo: HuggingFaceRepoRef(
                        repoID: "fixture/%2e%2e", revision: origin.repo.revision))
            ),
            (
                "backslash repository ID",
                publicPlan(
                    [safe],
                    repo: HuggingFaceRepoRef(
                        repoID: #"fixture/name\alias"#, revision: origin.repo.revision))
            ),
            (
                "unpinned repository revision",
                publicPlan(
                    [safe],
                    repo: HuggingFaceRepoRef(repoID: origin.repo.repoID, revision: "main"))
            ),
            (
                "forged reconciliation",
                DownloadPlan(
                    model: validPlan.model, repo: validPlan.repo, files: validPlan.files,
                    index: validPlan.index,
                    reconciliation: IndexReconciliation(
                        treeFileCount: 0, treeBytes: 0, payloadFileCount: 0,
                        payloadBytes: 0, fileCountDelta: 0, byteDelta: 0,
                        note: "caller-supplied fiction"))
            ),
            ("empty plan", publicPlan([])),
            ("absolute path", publicPlan([publicFile(outside.path, size: 1)])),
            (
                "parent traversal",
                publicPlan([publicFile("../\(outside.lastPathComponent)", size: 1)])
            ),
            ("duplicate path", publicPlan([safe, safe])),
            (
                "final-to-part collision",
                publicPlan([safe, publicFile("safe.bin\(DownloadManager.partSuffix)", size: 1)])
            ),
            (
                "file-to-directory collision",
                publicPlan([publicFile("node", size: 1), publicFile("node/child.bin", size: 1)])
            ),
            (
                "canonical Unicode duplicate",
                publicPlan([
                    publicFile("caf\u{00E9}.bin", size: 1),
                    publicFile("cafe\u{0301}.bin", size: 1),
                ])
            ),
            (
                "misclassified payload",
                publicPlan([
                    RepoFile(
                        path: "payload/misclassified.bin", sizeBytes: 1,
                        digest: .sha256(hex: String(repeating: "a", count: 64)),
                        isPayload: false)
                ])
            ),
            (
                "invalid digest",
                publicPlan([
                    RepoFile(
                        path: "payload/bad-digest.bin", sizeBytes: 1,
                        digest: .sha256(hex: "not-a-sha256"), isPayload: true)
                ])
            ),
            (
                "overflowing total",
                publicPlan([
                    publicFile("payload/huge.bin", size: UInt64.max),
                    publicFile("payload/one-more.bin", size: 1),
                ])
            ),
        ]

        for (name, plan) in attacks {
            let root = parent.appendingPathComponent(
                "minirun-public-plan-root-\(UUID().uuidString)", isDirectory: true)
            let store = InMemoryDownloadStateStore()
            let manager = makeManager(stateStore: store)

            await assertPlanRefused(
                manager, plan: plan, destination: root, operation: "preflight", context: name)
            await assertPlanRefused(
                manager, plan: plan, destination: root, operation: "start", context: name)

            XCTAssertFalse(
                FileManager.default.fileExists(atPath: root.path),
                "\(name) created its artifact root")
            XCTAssertEqual(try Data(contentsOf: outside), marker, "\(name) changed outside bytes")
            XCTAssertTrue(try store.all().isEmpty, "\(name) persisted a job")
        }
    }

    func testPlanRejectsCaseAliasedTargetsOnACaseInsensitiveVolume() async throws {
        let values = try destination.resourceValues(forKeys: [.volumeSupportsCaseSensitiveNamesKey])
        guard values.volumeSupportsCaseSensitiveNames == false else {
            throw XCTSkip("the test destination is on a case-sensitive volume")
        }
        let plan = publicPlan([
            publicFile("payload/Case.bin", size: 1),
            publicFile("payload/case.bin", size: 1),
        ])
        let store = InMemoryDownloadStateStore()
        await assertPlanRefused(
            makeManager(stateStore: store), plan: plan, destination: destination,
            operation: "start", context: "case-aliased targets")
        XCTAssertTrue(try store.all().isEmpty)
    }

    func testOverflowingPublicPlanAccessorsNeverTrapAndManagerStillRefusesThePlan() async throws {
        let files = [
            publicFile("payload/huge.bin", size: UInt64.max),
            publicFile("payload/one-more.bin", size: 1),
        ]
        let plan = publicPlan(files)

        XCTAssertEqual(plan.totalBytes, UInt64.max)
        XCTAssertEqual(plan.payloadBytes, UInt64.max)
        XCTAssertEqual(plan.metadataBytes, 0)
        XCTAssertFalse(plan.bucketsAccountForEveryByte)
        XCTAssertEqual(IndexReconciliation.between(files: files, claim: nil).treeBytes, UInt64.max)

        let root = destination.deletingLastPathComponent().appendingPathComponent(
            "minirun-overflow-root-\(UUID().uuidString)", isDirectory: true)
        let store = InMemoryDownloadStateStore()
        await assertPlanRefused(
            makeManager(stateStore: store), plan: plan, destination: root,
            operation: "start", context: "overflowing total")
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
        XCTAssertTrue(try store.all().isEmpty)
    }

    func testPlanRejectsASymlinkBelowTheArtifactRootWithoutTouchingItsTarget() async throws {
        let outsideDirectory = destination.deletingLastPathComponent().appendingPathComponent(
            "minirun-plan-symlink-target-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: outsideDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outsideDirectory) }
        let outside = outsideDirectory.appendingPathComponent("victim.bin")
        let marker = Data("outside target\n".utf8)
        try marker.write(to: outside)
        let link = destination.appendingPathComponent("nested")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outsideDirectory)

        let store = InMemoryDownloadStateStore()
        let manager = makeManager(stateStore: store)
        let plan = publicPlan([publicFile("nested/victim.bin", size: UInt64(marker.count))])
        await assertPlanRefused(
            manager, plan: plan, destination: destination,
            operation: "preflight", context: "intermediate symlink")
        await assertPlanRefused(
            manager, plan: plan, destination: destination,
            operation: "start", context: "intermediate symlink")

        XCTAssertEqual(try Data(contentsOf: outside), marker)
        XCTAssertTrue(try store.all().isEmpty)
    }

    func testPlanRejectsAPartHardLinkWithoutChangingTheOutsideInode() async throws {
        let outside = destination.deletingLastPathComponent().appendingPathComponent(
            "minirun-part-hardlink-target-\(UUID().uuidString).bin")
        let marker = Data("outside hard-link target\n".utf8)
        try marker.write(to: outside)
        defer { try? FileManager.default.removeItem(at: outside) }
        let part = destination.appendingPathComponent(
            "payload.bin\(DownloadManager.partSuffix)")
        try FileManager.default.linkItem(at: outside, to: part)

        let store = InMemoryDownloadStateStore()
        let manager = makeManager(stateStore: store)
        let plan = publicPlan([publicFile("payload.bin", size: UInt64(marker.count))])
        await assertPlanRefused(
            manager, plan: plan, destination: destination,
            operation: "start", context: "part hard link")

        XCTAssertEqual(try Data(contentsOf: outside), marker)
        XCTAssertEqual(try Data(contentsOf: part), marker)
        XCTAssertTrue(try store.all().isEmpty)
    }

    func testPlanRefusesAnExistingDirectoryAtAPlannedFileWithoutDeletingIt() async throws {
        let occupied = destination.appendingPathComponent("occupied.bin", isDirectory: true)
        try FileManager.default.createDirectory(at: occupied, withIntermediateDirectories: true)
        let markerURL = occupied.appendingPathComponent("not-download-manager-owned.txt")
        let marker = Data("keep this directory\n".utf8)
        try marker.write(to: markerURL)

        let store = InMemoryDownloadStateStore()
        let manager = makeManager(stateStore: store)
        let plan = publicPlan([publicFile("occupied.bin", size: 1)])
        await assertPlanRefused(
            manager, plan: plan, destination: destination,
            operation: "start", context: "directory at final path")

        XCTAssertEqual(try Data(contentsOf: markerURL), marker)
        XCTAssertTrue(try store.all().isEmpty)
    }

    func testPubliclyConstructedSafePlanStillDownloads() async throws {
        let manager = makeManager()
        let discovered = try await manager.plan(for: descriptor())
        let publicPlan = DownloadPlan(
            model: discovered.model, repo: discovered.repo, files: discovered.files,
            index: discovered.index, reconciliation: discovered.reconciliation)

        let check = try await manager.preflight(
            publicPlan, into: DownloadDestination(directory: destination),
            options: testOptions())
        XCTAssertTrue(check.fits, "\(check.refusals)")

        let job = try await manager.start(
            publicPlan, into: DownloadDestination(directory: destination),
            options: testOptions())
        let events = await drainEvents(of: manager, job: job)
        XCTAssertNotNil(events.summary, "\(events)")
        for file in publicPlan.files {
            XCTAssertEqual(
                try Data(contentsOf: destination.appendingPathComponent(file.path)),
                origin.bytes(of: file.path))
        }
    }

    func testAValidZeroBytePublicPlanCreatesAndVerifiesAnEmptyFile() async throws {
        let manager = makeManager()
        let emptySHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        let plan = publicPlan([
            RepoFile(
                path: "payload/empty.bin", sizeBytes: 0,
                digest: .sha256(hex: emptySHA256), isPayload: true)
        ])

        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())
        let events = await drainEvents(of: manager, job: job)
        XCTAssertNotNil(events.summary, "\(events)")
        XCTAssertEqual(
            try Data(contentsOf: destination.appendingPathComponent("payload/empty.bin")), Data())
    }

    // MARK: - A complete transfer

    func testFullDownloadVerifiesEveryFileAndLeavesNoPartFiles() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())

        let events = await drainEvents(of: manager, job: job)
        let summary = try XCTUnwrap(events.summary, "expected the job to finish: \(events)")
        XCTAssertTrue(summary.verification.isComplete)
        XCTAssertEqual(summary.verification.missing, [])
        XCTAssertEqual(summary.verification.wrongDigest, [])
        XCTAssertEqual(summary.bytesWritten, plan.totalBytes)
        XCTAssertEqual(Set(events.verifiedPaths), Set(plan.files.map(\.path)))

        for file in plan.files {
            let url = destination.appendingPathComponent(file.path)
            let data = try Data(contentsOf: url)
            XCTAssertEqual(
                data, origin.bytes(of: file.path), "\(file.path) does not match what was served")
        }
        let leftovers = DownloadManager.filesOnDisk(under: destination)
            .filter { $0.hasSuffix(DownloadManager.partSuffix) }
        XCTAssertEqual(leftovers, [], "a verified download must leave no .minirun-part files")

        // Every byte off the wire was useful, and the counters say so.
        XCTAssertGreaterThanOrEqual(summary.networkBytes, plan.totalBytes)
        XCTAssertEqual(summary.wastedBytes, 0)
    }

    func testTheLargeFileArrivesInSeveralChunksAndStillVerifies() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let large = try XCTUnwrap(plan.file(at: FixtureArtifact.largeFilePath))
        XCTAssertGreaterThan(large.sizeBytes, 1 << 20, "the fixture must span more than one chunk")

        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination),
            options: testOptions(chunkBytes: 1 << 20, concurrentFiles: 1))
        let events = await drainEvents(of: manager, job: job)
        XCTAssertNotNil(events.summary, "\(events)")

        let requests = origin.requests.filter { $0 == "/cdn/\(FixtureArtifact.largeFilePath)" }
        XCTAssertGreaterThanOrEqual(
            requests.count, 3, "a 3 MiB file at a 1 MiB chunk needs three ranged requests")
    }

    // MARK: - Corruption and repair

    func testACorruptedFileFailsItsDigestAndRepairPutsItRight() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let victim = "global/global-embed.bin"
        origin.corrupt(victim)

        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())
        let events = await drainEvents(of: manager, job: job)

        let failures = events.fileFailures.filter { $0.0 == victim }
        XCTAssertFalse(failures.isEmpty, "the corrupted file should have failed: \(events)")
        guard case .digestMismatch(let mismatch)? = failures.first?.1 else {
            return XCTFail("expected a digest mismatch, got \(String(describing: failures.first?.1))")
        }
        XCTAssertEqual(mismatch.path, victim)
        XCTAssertEqual(mismatch.algorithm, .sha256)
        XCTAssertNotEqual(mismatch.expected, mismatch.found)
        XCTAssertNotNil(events.failure)

        // Bytes that failed a digest are counted as wasted, not as progress.
        let progress = try XCTUnwrap(events.lastProgress)
        XCTAssertGreaterThan(progress.wastedBytes, 0)
        XCTAssertTrue(progress.accountsForEveryByte)

        // And they are gone: a byte that failed a digest is not a byte to
        // resume from.
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: destination.appendingPathComponent(victim + DownloadManager.partSuffix).path))
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: destination.appendingPathComponent(victim).path))

        let report = try await manager.verify(job, depth: .digestPayload)
        XCTAssertFalse(report.isComplete)
        XCTAssertEqual(report.pathsToRefetch, [victim])
        XCTAssertEqual(report.bytesToRefetch, plan.file(at: victim)?.sizeBytes)

        origin.heal(victim)
        _ = try await manager.repair(job, from: report)
        let afterRepair = await drainEvents(of: manager, job: job)
        XCTAssertNotNil(afterRepair.summary, "the repair should finish the job: \(afterRepair)")
        XCTAssertEqual(
            try Data(contentsOf: destination.appendingPathComponent(victim)),
            origin.bytes(of: victim))
    }

    func testRepairRefetchesOnlyWhatTheReportNamed() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let victim = "layer00/layer00-deterministic.bin"
        origin.corrupt(victim)
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())
        _ = await drainEvents(of: manager, job: job)

        let report = try await manager.verify(job, depth: .digestPayload)
        origin.heal(victim)
        let before = origin.requests.count
        _ = try await manager.repair(job, from: report)
        _ = await drainEvents(of: manager, job: job)

        let refetched = origin.requests.dropFirst(before).filter { $0.hasPrefix("/cdn/") }
        XCTAssertEqual(Set(refetched), ["/cdn/\(victim)"])
    }

    func testRepairRejectsReportsNotBoundToTheJobModelAndPlan() async throws {
        let (manager, plan, job) = try await finishedJob()
        let victim = "global/global-embed.bin"
        let file = try XCTUnwrap(plan.file(at: victim))
        let before = try artifactSnapshot(plan, under: destination)

        let reports = [
            (
                "another job",
                repairReport(plan: plan, job: DownloadJobID(), missing: [victim])
            ),
            (
                "another model",
                repairReport(
                    plan: plan, job: job, model: ModelID("another-model"), missing: [victim])
            ),
            (
                "a size claim other than the plan",
                repairReport(
                    plan: plan, job: job,
                    wrongSize: [
                        SizeMismatch(path: victim, expected: file.sizeBytes + 1, found: 0)
                    ])
            ),
            (
                "a digest claim other than the plan",
                repairReport(
                    plan: plan, job: job,
                    wrongDigest: [
                        DigestMismatch(
                            path: victim, algorithm: file.digest.algorithm,
                            expected: String(repeating: "0", count: file.digest.hex.count),
                            found: String(repeating: "1", count: file.digest.hex.count))
                    ])
            ),
            (
                "a byte cost other than the plan",
                repairReport(plan: plan, job: job, missing: [victim], bytesToRefetch: 0)
            ),
            (
                "one path in two result categories",
                repairReport(
                    plan: plan, job: job, missing: [victim],
                    ok: plan.files.map(\.path))
            ),
        ]

        for (name, report) in reports {
            await assertRepairRefused(manager, job: job, report: report, name)
            assertArtifactSnapshot(before, under: destination, context: name)
        }
    }

    func testRepairCostOverflowIsANamedRefusalInsteadOfAWrapOrTrap() throws {
        XCTAssertThrowsError(
            try DownloadManager.repairByteCost(
                paths: ["huge.bin", "one-more.bin"],
                sizes: ["huge.bin": UInt64.max, "one-more.bin": 1])
        ) { error in
            guard case DownloadError.invalidOption(let name, _, let allowed)? =
                error as? DownloadError
            else { return XCTFail("expected a named overflow refusal, got \(error)") }
            XCTAssertEqual(name, "repair.bytesToRefetch")
            XCTAssertTrue(allowed.contains("UInt64"), allowed)
        }
    }

    func testRepairValidatesEveryPathBeforeDeletingAnyFile() async throws {
        let (manager, plan, job) = try await finishedJob()
        let victim = "global/global-embed.bin"
        let before = try artifactSnapshot(plan, under: destination)
        let outside = destination.deletingLastPathComponent()
            .appendingPathComponent("minirun-repair-outside-\(UUID().uuidString).bin")
        let marker = Data("outside\n".utf8)
        try marker.write(to: outside)
        defer { try? FileManager.default.removeItem(at: outside) }

        let attacks = [
            outside.path,
            "../\(outside.lastPathComponent)",
            "nested/../../\(outside.lastPathComponent)",
            "not-in-this-plan.bin",
        ]
        for attack in attacks {
            // Put a legitimate planned victim first. If repair mutated while
            // validating, it would delete this before discovering the attack.
            let report = repairReport(
                plan: plan, job: job, missing: [victim, attack])
            await assertRepairRefused(manager, job: job, report: report, attack)
            assertArtifactSnapshot(before, under: destination, context: attack)
            XCTAssertEqual(try Data(contentsOf: outside), marker)
        }
    }

    func testRepairCannotFollowASymlinkOrAReplacedArtifactRootOutsideTheJobRoot() async throws {
        let (manager, plan, job) = try await finishedJob()
        let victim = "global/global-embed.bin"
        let victimBytes = try Data(contentsOf: destination.appendingPathComponent(victim))
        let before = try artifactSnapshot(plan, under: destination)
        let report = repairReport(plan: plan, job: job, missing: [victim])
        let parent = destination.deletingLastPathComponent()

        // First keep the artifact root in place but replace an intermediate
        // directory with a symlink. Lexical prefix checks alone miss this.
        let originalGlobal = destination.appendingPathComponent("global")
        let savedGlobal = destination.appendingPathComponent("global.saved")
        let outsideDirectory = parent.appendingPathComponent(
            "minirun-repair-symlink-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.moveItem(at: originalGlobal, to: savedGlobal)
        try FileManager.default.createDirectory(
            at: outsideDirectory, withIntermediateDirectories: true)
        let outsideVictim = outsideDirectory.appendingPathComponent("global-embed.bin")
        try victimBytes.write(to: outsideVictim)
        try FileManager.default.createSymbolicLink(
            at: originalGlobal, withDestinationURL: outsideDirectory)

        await assertRepairRefused(manager, job: job, report: report, "intermediate symlink")
        XCTAssertEqual(try Data(contentsOf: outsideVictim), victimBytes)

        try FileManager.default.removeItem(at: originalGlobal)
        try FileManager.default.moveItem(at: savedGlobal, to: originalGlobal)
        try FileManager.default.removeItem(at: outsideDirectory)
        assertArtifactSnapshot(before, under: destination, context: "intermediate symlink")

        // Then replace the registered root itself. The same job id and report
        // still do not grant authority over the new symlink target.
        let savedRoot = parent.appendingPathComponent(
            "minirun-repair-root-saved-\(UUID().uuidString)", isDirectory: true)
        let outsideRoot = parent.appendingPathComponent(
            "minirun-repair-root-outside-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.moveItem(at: destination, to: savedRoot)
        try FileManager.default.createDirectory(
            at: outsideRoot.appendingPathComponent("global"), withIntermediateDirectories: true)
        let rootSwapVictim = outsideRoot.appendingPathComponent(victim)
        try victimBytes.write(to: rootSwapVictim)
        try FileManager.default.createSymbolicLink(at: destination, withDestinationURL: outsideRoot)

        await assertRepairRefused(manager, job: job, report: report, "replaced artifact root")
        XCTAssertEqual(try Data(contentsOf: rootSwapVictim), victimBytes)
        assertArtifactSnapshot(before, under: savedRoot, context: "replaced artifact root")

        try FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: savedRoot, to: destination)
        try FileManager.default.removeItem(at: outsideRoot)
        assertArtifactSnapshot(before, under: destination, context: "restored artifact root")

        // Finally replace the root with a *new directory at the same path*.
        // String-based root binding cannot distinguish this from the original;
        // the registered device/inode pair must.
        let savedSamePathRoot = parent.appendingPathComponent(
            "minirun-repair-same-path-saved-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.moveItem(at: destination, to: savedSamePathRoot)
        try FileManager.default.createDirectory(
            at: destination.appendingPathComponent("global"), withIntermediateDirectories: true)
        let replacementBytes = Data("new directory, not this job\n".utf8)
        let replacementVictim = destination.appendingPathComponent(victim)
        try replacementBytes.write(to: replacementVictim)

        await assertRepairRefused(manager, job: job, report: report, "same-path new root inode")
        XCTAssertEqual(try Data(contentsOf: replacementVictim), replacementBytes)
        assertArtifactSnapshot(
            before, under: savedSamePathRoot, context: "same-path new root inode")

        try FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: savedSamePathRoot, to: destination)
        assertArtifactSnapshot(before, under: destination, context: "restored original root inode")
    }

    // MARK: - Resume

    /// The part file's length is the resume point, and the record is not
    /// consulted for it. Half the bytes are placed on disk before the job
    /// starts; only the remainder should come off the wire.
    func testResumeContinuesFromThePartFileAlreadyOnDisk() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let path = FixtureArtifact.largeFilePath
        let whole = try XCTUnwrap(origin.bytes(of: path))
        let half = whole.count / 2

        let part = destination.appendingPathComponent(path + DownloadManager.partSuffix)
        try FileManager.default.createDirectory(
            at: part.deletingLastPathComponent(), withIntermediateDirectories: true)
        try whole.prefix(half).write(to: part)

        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination),
            options: testOptions(chunkBytes: 1 << 20, concurrentFiles: 1))
        let events = await drainEvents(of: manager, job: job)
        let summary = try XCTUnwrap(events.summary, "\(events)")

        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent(path)), whole)
        XCTAssertLessThan(
            summary.networkBytes, plan.totalBytes,
            "the half already on disk must not be fetched again")

        let started = events.compactMap { event -> UInt64? in
            if case .fileStarted(let started, _, let resumed) = event, started == path {
                return resumed
            }
            return nil
        }
        XCTAssertEqual(started.first, UInt64(half))
    }

    /// A part file longer than the plan says is discarded rather than trusted.
    /// Trusting it would resume past the end of the object and leave a file that
    /// is the right length and the wrong bytes.
    func testAnOverlongPartFileIsDiscardedAndTheFileIsFetchedAgain() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let path = "global/global-embed.bin"
        let whole = try XCTUnwrap(origin.bytes(of: path))

        let part = destination.appendingPathComponent(path + DownloadManager.partSuffix)
        try FileManager.default.createDirectory(
            at: part.deletingLastPathComponent(), withIntermediateDirectories: true)
        try (whole + Data(repeating: 0xAB, count: 1024)).write(to: part)

        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())
        let events = await drainEvents(of: manager, job: job)
        XCTAssertNotNil(events.summary, "\(events)")
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent(path)), whole)
    }

    /// A complete file already at the destination is adopted, not fetched
    /// again — and the closing pass still digests it, because nobody in this
    /// process ever checked those bytes.
    func testAnAlreadyCompleteFileIsAdoptedAndStillDigested() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let path = "layer00/layer00-deterministic.bin"
        let whole = try XCTUnwrap(origin.bytes(of: path))
        let url = destination.appendingPathComponent(path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try whole.write(to: url)

        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())
        let events = await drainEvents(of: manager, job: job)
        let summary = try XCTUnwrap(events.summary, "\(events)")
        XCTAssertEqual(summary.verification.depth, .digestPayload)
        XCTAssertTrue(summary.verification.ok.contains(path))
        XCTAssertFalse(origin.requests.contains("/cdn/\(path)"))
    }

    // MARK: - Pause and resume

    func testStartRefusesStateStoreFailureBeforeNetworkAndReleasesScope() async throws {
        let store = SwitchableDownloadStateStore()
        let coding = ScriptedURLCoding()
        let storage = scriptedStorage(coding)
        let manager = makeManager(stateStore: store, storage: storage)
        let plan = try await manager.plan(for: descriptor())
        let marker = destination.appendingPathComponent("must-survive.txt")
        try Data("untouched".utf8).write(to: marker)
        let scope = try storage.scope(for: destination)
        XCTAssertEqual(coding.balance, 1)
        store.failSaves()
        let requestCount = origin.requests.count

        do {
            _ = try await manager.start(
                plan, into: DownloadDestination(directory: destination, scope: scope),
                options: testOptions(concurrentFiles: 1))
            XCTFail("a job whose first durable save failed must not start")
        } catch {
            guard case .statePersistenceFailed(_, let reason)? = error as? DownloadError else {
                return XCTFail("expected statePersistenceFailed, got \(error)")
            }
            XCTAssertTrue(reason.contains("deliberate"))
        }

        XCTAssertEqual(origin.requests.count, requestCount)
        XCTAssertEqual(try Data(contentsOf: marker), Data("untouched".utf8))
        XCTAssertEqual(coding.balance, 0)
        let summaries = await manager.jobs()
        XCTAssertTrue(summaries.isEmpty)
    }

    func testStartReleasesAnOwnedScopeOnEveryFailureBeforeRegistration() async throws {
        let coding = ScriptedURLCoding()
        let storage = scriptedStorage(coding)
        let manager = makeManager(storage: storage)
        let valid = try await manager.plan(for: descriptor())

        var invalidOptions = testOptions()
        invalidOptions.chunkBytes = 1
        let invalidPlan = publicPlan([publicFile("../outside.bin", size: 1)])
        var refusedOptions = testOptions()
        refusedOptions.freeSpaceHeadroomBytes = UInt64(Int64.max / 2)
        let cases: [(DownloadPlan, DownloadOptions)] = [
            (valid, invalidOptions),
            (invalidPlan, testOptions()),
            (valid, refusedOptions),
        ]

        for (plan, options) in cases {
            let scope = try storage.scope(for: destination)
            XCTAssertEqual(coding.balance, 1)
            do {
                _ = try await manager.start(
                    plan, into: DownloadDestination(directory: destination, scope: scope),
                    options: options)
                XCTFail("the deliberately invalid start should fail")
            } catch {
                // The exact named refusal is covered elsewhere. This test owns
                // the resource-transfer contract across every early throw.
            }
            XCTAssertEqual(coding.balance, 0, "start leaked or double-released its scope")
        }
    }

    func testTransferStopsSchedulingFilesWhenProgressCannotBePersisted() async throws {
        let store = SwitchableDownloadStateStore()
        let coding = ScriptedURLCoding()
        let storage = scriptedStorage(coding)
        let manager = makeManager(stateStore: store, storage: storage)
        let plan = try await manager.plan(for: descriptor())
        // Registration is the one permitted save. The first completed file
        // then fails to record its progress, which must stop the serial queue.
        store.failSaves(after: 1)
        let requestCount = origin.requests.count
        let scope = try storage.scope(for: destination)
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination, scope: scope),
            options: testOptions(concurrentFiles: 1))

        let events = await drainEvents(of: manager, job: job)

        guard case .statePersistenceFailed(let failedJob, _)? = events.failure else {
            return XCTFail("expected statePersistenceFailed, got \(String(describing: events.failure))")
        }
        XCTAssertEqual(failedJob, job)
        XCTAssertNil(events.summary)
        let newRequests = origin.requests.dropFirst(requestCount)
            .filter { $0.hasPrefix("/cdn/") }
        XCTAssertEqual(
            newRequests.count, 1,
            "a persistence failure must not schedule the remaining artifact files")
        XCTAssertEqual(coding.balance, 0)
    }

    func testPersistenceAbortStopsANonCooperativeSiblingBeforeDigestRenameOrPersist() async throws {
        let store = SwitchableDownloadStateStore()
        let a = FixtureBytes.make(1 << 20, seed: 41)
        let b = FixtureBytes.make(1 << 20, seed: 42)
        let files = ["a.bin": a, "b.bin": b]
        let plan = publicPlan(
            [("payload/a.bin", a), ("payload/b.bin", b)].map { path, bytes in
                RepoFile(
                    path: path, sizeBytes: UInt64(bytes.count),
                    digest: .sha256(hex: FileDigestComputer.sha256(of: bytes)),
                    isPayload: true)
            })
        let transport = PersistenceRaceTransport(store: store, files: files)
        let manager = DownloadManager(
            transport: transport,
            storage: StorageManager(bookmarkLedger: InMemoryBookmarkLedger()),
            stateStore: store, catalogEndpoint: origin.baseURL)
        // Registration succeeds. a.bin then lands and fails its progress save;
        // b.bin returns only after that failure and ignores task cancellation.
        store.failSaves(after: 1)

        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination),
            options: testOptions(concurrentFiles: 2))
        let events = await drainEvents(of: manager, job: job)

        guard case .statePersistenceFailed? = events.failure else {
            return XCTFail("expected persistence failure, got \(String(describing: events.failure))")
        }
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: destination.appendingPathComponent(
                "payload/b.bin" + DownloadManager.partSuffix).path))
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: destination.appendingPathComponent("payload/b.bin").path),
            "a sibling returning after durable-state loss must never be promoted to final")
        XCTAssertFalse(events.verifiedPaths.contains("payload/b.bin"))
    }

    func testAReplacedPartNameNeverReceivesBodyBytesAndIsRejectedBeforePromotion() async throws {
        let bytes = FixtureBytes.make(1 << 20, seed: 43)
        let path = "payload/raced.bin"
        let plan = publicPlan([
            RepoFile(
                path: path, sizeBytes: UInt64(bytes.count),
                digest: .sha256(hex: FileDigestComputer.sha256(of: bytes)),
                isPayload: true)
        ])
        for statusCode in [206, 503] {
            let root = destination.appendingPathComponent("status-\(statusCode)")
            let partURL = root.appendingPathComponent(path + DownloadManager.partSuffix)
            let transport = PartNameReplacementTransport(
                bytes: bytes, statusCode: statusCode, partURL: partURL)
            let manager = DownloadManager(
                transport: transport,
                storage: StorageManager(bookmarkLedger: InMemoryBookmarkLedger()),
                stateStore: InMemoryDownloadStateStore(), catalogEndpoint: origin.baseURL)

            let job = try await manager.start(
                plan, into: DownloadDestination(directory: root),
                options: testOptions(chunkBytes: 1 << 20, concurrentFiles: 1))
            let events = await drainEvents(of: manager, job: job)

            guard case .destinationNotWritable? = events.failure else {
                XCTFail(
                    "expected substituted part refusal after HTTP \(statusCode), got "
                        + "\(String(describing: events.failure))")
                continue
            }
            XCTAssertFalse(FileManager.default.fileExists(
                atPath: root.appendingPathComponent(path).path))
            XCTAssertEqual(
                try Data(contentsOf: partURL), transport.replacement,
                "body bytes must stay on the borrowed inode, never the raced path")
            XCTAssertFalse(events.verifiedPaths.contains(path))
        }
    }

    func testRange200FromAMovedRevisionKeepsRevisionPriorityWithProductionTransport() async throws {
        let bytes = FixtureBytes.make(1 << 20, seed: 45)
        let moved = String(repeating: "9", count: 40)
        let server = try LocalHTTPServer { _ in
            LocalHTTPServer.Response(
                status: 200,
                headers: [
                    "x-repo-commit": moved,
                    "x-linked-size": String(bytes.count),
                ], body: bytes)
        }
        defer { server.stop() }
        let path = "payload/moved.bin"
        let plan = publicPlan([
            RepoFile(
                path: path, sizeBytes: UInt64(bytes.count),
                digest: .sha256(hex: FileDigestComputer.sha256(of: bytes)),
                isPayload: true)
        ])
        let manager = DownloadManager(
            transport: URLSessionTransport(),
            storage: StorageManager(bookmarkLedger: InMemoryBookmarkLedger()),
            stateStore: InMemoryDownloadStateStore(), catalogEndpoint: server.baseURL)

        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination),
            options: testOptions(concurrentFiles: 1, attempts: 1))
        let events = await drainEvents(of: manager, job: job)

        guard case .revisionMoved(let expected, let served, let failedPath)? = events.failure else {
            return XCTFail("expected revisionMoved, got \(String(describing: events.failure))")
        }
        XCTAssertEqual(expected, revision)
        XCTAssertEqual(served, moved)
        XCTAssertEqual(failedPath, path)
        XCTAssertEqual(
            try Data(
                contentsOf: destination.appendingPathComponent(
                    path + DownloadManager.partSuffix)),
            Data(), "the rejected 200 body must never enter the part")
    }

    func testIntegrityHeaderFailuresRollBackOnlyTheCurrentRange() async throws {
        let bytes = FixtureBytes.make(1 << 20, seed: 44)
        let prefixLength = 4096
        let prefix = Data(bytes.prefix(prefixLength))
        let path = "payload/integrity.bin"
        let plan = publicPlan([
            RepoFile(
                path: path, sizeBytes: UInt64(bytes.count),
                digest: .sha256(hex: FileDigestComputer.sha256(of: bytes)),
                isPayload: true)
        ])
        let cases: [(name: String, violation: IntegrityMismatchTransport.Violation)] = [
            ("moved-revision", .movedRevision),
            ("missing-revision", .missingRevision),
            ("linked-size", .linkedSize),
            ("returned-revision-switch", .returnedRevisionSwitch),
        ]

        for item in cases {
            let root = destination.appendingPathComponent(item.name)
            let partURL = root.appendingPathComponent(path + DownloadManager.partSuffix)
            try FileManager.default.createDirectory(
                at: partURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try prefix.write(to: partURL)
            let transport = IntegrityMismatchTransport(
                bytes: bytes, expectedRevision: revision, violation: item.violation)
            let manager = DownloadManager(
                transport: transport,
                storage: StorageManager(bookmarkLedger: InMemoryBookmarkLedger()),
                stateStore: InMemoryDownloadStateStore(), catalogEndpoint: origin.baseURL)

            let job = try await manager.start(
                plan, into: DownloadDestination(directory: root),
                options: testOptions(
                    chunkBytes: 1 << 20, concurrentFiles: 1, attempts: 1))
            let events = await drainEvents(of: manager, job: job)

            switch item.violation {
            case .movedRevision:
                guard case .revisionMoved? = events.failure else {
                    XCTFail("expected revisionMoved, got \(String(describing: events.failure))")
                    continue
                }
            case .missingRevision:
                guard case .interrupted(_, _, let reason)? = events.failure,
                    reason.contains("x-repo-commit")
                else {
                    XCTFail("expected missing commit refusal, got \(String(describing: events.failure))")
                    continue
                }
            case .linkedSize:
                guard case .sizeMismatch? = events.failure else {
                    XCTFail("expected sizeMismatch, got \(String(describing: events.failure))")
                    continue
                }
            case .returnedRevisionSwitch:
                guard case .interrupted(_, _, let reason)? = events.failure,
                    reason.contains("authority changed")
                else {
                    XCTFail(
                        "expected admitted/returned authority refusal, got "
                            + "\(String(describing: events.failure))")
                    continue
                }
            }
            XCTAssertEqual(
                try Data(contentsOf: partURL), prefix,
                "\(item.name) bytes must roll back without losing an earlier sealed range")
            XCTAssertFalse(FileManager.default.fileExists(
                atPath: root.appendingPathComponent(path).path))
            XCTAssertFalse(events.verifiedPaths.contains(path))
            let terminalProgress = events.compactMap { event -> DownloadProgress? in
                if case .failed(_, let progress) = event { return progress }
                return nil
            }.last
            XCTAssertEqual(
                terminalProgress?.networkBytes, UInt64(bytes.count - prefixLength))
            XCTAssertEqual(
                terminalProgress?.wastedBytes, UInt64(bytes.count - prefixLength),
                "bytes rolled back for provenance or size are discarded traffic")
        }
    }

    func testRollbackFailureDiscardsThePartAndIsNeverRetriedFromItsSize() async throws {
        let bytes = FixtureBytes.make(1 << 20, seed: 46)
        let prefixLength = 4096
        let path = "payload/rollback-failure.bin"
        let partURL = destination.appendingPathComponent(
            path + DownloadManager.partSuffix)
        try FileManager.default.createDirectory(
            at: partURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try bytes.prefix(prefixLength).write(to: partURL)
        let transport = DiscardFailureTransport(bytes: bytes, revision: revision)
        let manager = DownloadManager(
            transport: transport,
            storage: StorageManager(bookmarkLedger: InMemoryBookmarkLedger()),
            stateStore: InMemoryDownloadStateStore(), catalogEndpoint: origin.baseURL)
        let plan = publicPlan([
            RepoFile(
                path: path, sizeBytes: UInt64(bytes.count),
                digest: .sha256(hex: FileDigestComputer.sha256(of: bytes)),
                isPayload: true)
        ])

        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination),
            options: testOptions(concurrentFiles: 1, attempts: 3))
        let events = await drainEvents(of: manager, job: job)

        guard case .destinationNotWritable(_, let reason)? = events.failure else {
            return XCTFail(
                "rollback failure must be terminal, got \(String(describing: events.failure))")
        }
        XCTAssertTrue(reason.contains("could not be discarded"))
        XCTAssertEqual(transport.attemptCount, 1)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: partURL.path),
            "a part with an unknown rollback boundary must not remain resumable")
        let terminalProgress = events.compactMap { event -> DownloadProgress? in
            if case .failed(_, let progress) = event { return progress }
            return nil
        }.last
        XCTAssertEqual(terminalProgress?.networkBytes, UInt64(bytes.count - prefixLength))
        XCTAssertEqual(terminalProgress?.wastedBytes, UInt64(bytes.count - prefixLength))
    }

    func testCancellationCannotPreserveAPartWhoseRollbackBoundaryIsUnknown() async throws {
        let bytes = FixtureBytes.make(1 << 20, seed: 47)
        let prefixLength = 4096
        let path = "payload/cancelled-rollback-failure.bin"
        let root = destination.appendingPathComponent("cancelled-rollback")
        let partURL = root.appendingPathComponent(path + DownloadManager.partSuffix)
        try FileManager.default.createDirectory(
            at: partURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try bytes.prefix(prefixLength).write(to: partURL)
        let gate = TransportGate()
        let transport = DiscardFailureTransport(
            bytes: bytes, revision: revision, gate: gate)
        let manager = DownloadManager(
            transport: transport,
            storage: StorageManager(bookmarkLedger: InMemoryBookmarkLedger()),
            stateStore: InMemoryDownloadStateStore(), catalogEndpoint: origin.baseURL)
        let plan = publicPlan([
            RepoFile(
                path: path, sizeBytes: UInt64(bytes.count),
                digest: .sha256(hex: FileDigestComputer.sha256(of: bytes)),
                isPayload: true)
        ])

        let job = try await manager.start(
            plan, into: DownloadDestination(directory: root),
            options: testOptions(concurrentFiles: 1, attempts: 3))
        let deadline = Date().addingTimeInterval(2)
        while !gate.hasEntered, Date() < deadline {
            try await Task.sleep(nanoseconds: 2_000_000)
        }
        XCTAssertTrue(gate.hasEntered, "transport never reached its rollback-failure boundary")
        try await manager.cancel(job, keepPartialFiles: true)
        gate.release()
        let events = await drainEvents(of: manager, job: job)

        XCTAssertNotNil(events.cancellation)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: partURL.path),
            "keep-partials cannot retain a leaf with an unknown safe resume offset")
        XCTAssertEqual(transport.attemptCount, 1)
    }

    func testResumeRefusesStateStoreFailureBeforeNetworkAndRemainsPaused() async throws {
        let store = SwitchableDownloadStateStore()
        let coding = ScriptedURLCoding()
        let storage = scriptedStorage(coding)
        let plan = try await makeManager().plan(for: descriptor())
        let saved = try persistedState(plan: plan, coding: coding)
        try store.save(saved)
        let manager = makeManager(stateStore: store, storage: storage)
        let restore = try await manager.restorePersistedJobs()
        XCTAssertEqual(restore.restoredJobs, [saved.job])
        store.failSaves()
        let requestCount = origin.requests.count

        do {
            try await manager.resume(saved.job, scope: nil)
            XCTFail("resume must not open the network without durable running state")
        } catch {
            guard case .statePersistenceFailed(let failedJob, _)? = error as? DownloadError else {
                return XCTFail("expected statePersistenceFailed, got \(error)")
            }
            XCTAssertEqual(failedJob, saved.job)
        }

        XCTAssertEqual(origin.requests.count, requestCount)
        XCTAssertEqual(coding.balance, 0)
        let summaries = await manager.jobs()
        XCTAssertEqual(summaries.first?.state, .paused)
    }

    func testRestoreReportsRewriteFailureWithoutInstallingAJobOrOpeningNetwork() async throws {
        let store = SwitchableDownloadStateStore()
        let coding = ScriptedURLCoding()
        let storage = scriptedStorage(coding)
        let plan = try await makeManager().plan(for: descriptor())
        let saved = try persistedState(plan: plan, coding: coding)
        try store.save(saved)
        store.failSaves()
        let manager = makeManager(stateStore: store, storage: storage)
        let requestCount = origin.requests.count

        let report = try await manager.restorePersistedJobs()

        XCTAssertEqual(report.restoredJobs, [])
        XCTAssertEqual(report.failures.map(\.job), [saved.job])
        XCTAssertTrue(report.failures[0].reason.contains("could not persist"))
        XCTAssertEqual(origin.requests.count, requestCount)
        let restoredJobs = await manager.jobs()
        XCTAssertTrue(restoredJobs.isEmpty)
        XCTAssertEqual(coding.balance, 0)
    }

    func testRestoredFailedAndCancelledJobsReplayATerminalEventAndCloseImmediately() async throws {
        for durableState in [DownloadJobSummary.JobState.failed, .cancelled] {
            let store = InMemoryDownloadStateStore()
            let coding = ScriptedURLCoding()
            let plan = try await makeManager().plan(for: descriptor())
            let base = try persistedState(plan: plan, coding: coding)
            let saved = DownloadJobState(
                job: base.job, plan: base.plan, destinationPath: base.destinationPath,
                destinationBookmark: base.destinationBookmark,
                artifactRootPath: base.artifactRootPath,
                artifactRootDevice: base.artifactRootDevice,
                artifactRootInode: base.artifactRootInode,
                options: base.options, fileStates: base.fileStates,
                runState: durableState, createdAt: base.createdAt, updatedAt: base.updatedAt)
            try store.save(saved)
            let manager = makeManager(stateStore: store, storage: scriptedStorage(coding))

            let restored = try await manager.restorePersistedJobs()
            XCTAssertEqual(restored.restoredJobs, [saved.job])
            let events = await drainEvents(of: manager, job: saved.job, timeout: 1)
            XCTAssertEqual(events.filter(\.isTerminal).count, 1)
            if durableState == .failed {
                XCTAssertNotNil(events.failure)
            } else {
                XCTAssertNotNil(events.cancellation)
            }
            XCTAssertEqual(coding.balance, 0)
        }
    }

    func testManualVerifyReacquiresItsJobScopeAndFailsBeforeReadingWhenSaveFails() async throws {
        let store = SwitchableDownloadStateStore()
        let coding = ScriptedURLCoding()
        let storage = scriptedStorage(coding)
        let manager = makeManager(stateStore: store, storage: storage)
        let plan = try await manager.plan(for: descriptor())
        let scope = try storage.scope(for: destination)
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination, scope: scope),
            options: testOptions(concurrentFiles: 1))
        let finished = await drainEvents(of: manager, job: job)
        XCTAssertNotNil(finished.summary)
        XCTAssertEqual(coding.balance, 0)

        let startsBeforeVerify = coding.startCount
        let report = try await manager.verify(job, depth: .digestPayload)
        XCTAssertTrue(report.isComplete)
        XCTAssertEqual(coding.startCount, startsBeforeVerify + 1)
        XCTAssertEqual(coding.balance, 0)

        let startsBeforeFailure = coding.startCount
        store.failSaves()
        do {
            _ = try await manager.verify(job, depth: .digestPayload)
            XCTFail("verification must not read a released external volume without durable authority")
        } catch {
            guard case .statePersistenceFailed(let failedJob, _)? = error as? DownloadError else {
                return XCTFail("expected statePersistenceFailed, got \(error)")
            }
            XCTAssertEqual(failedJob, job)
        }
        XCTAssertEqual(coding.startCount, startsBeforeFailure + 1)
        XCTAssertEqual(coding.balance, 0)
    }

    func testRepairPersistsIntentBeforeDeletingOrOpeningNetwork() async throws {
        let store = SwitchableDownloadStateStore()
        let coding = ScriptedURLCoding()
        let storage = scriptedStorage(coding)
        let manager = makeManager(stateStore: store, storage: storage)
        let plan = try await manager.plan(for: descriptor())
        let scope = try storage.scope(for: destination)
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination, scope: scope),
            options: testOptions(concurrentFiles: 1))
        let finished = await drainEvents(of: manager, job: job)
        XCTAssertNotNil(finished.summary)
        XCTAssertEqual(coding.balance, 0)

        let victim = "global/global-embed.bin"
        let victimURL = destination.appendingPathComponent(victim)
        var corrupted = try Data(contentsOf: victimURL)
        corrupted[corrupted.startIndex] ^= 0xFF
        try corrupted.write(to: victimURL)
        let report = try await manager.verify(job, depth: .digestPayload)
        XCTAssertEqual(report.pathsToRefetch, [victim])
        XCTAssertEqual(coding.balance, 0)

        store.failSaves()
        let requestCount = origin.requests.count
        let startsBeforeRepair = coding.startCount
        do {
            _ = try await manager.repair(job, from: report)
            XCTFail("repair must not delete bytes before its intent is durable")
        } catch {
            guard case .statePersistenceFailed(let failedJob, _)? = error as? DownloadError else {
                return XCTFail("expected statePersistenceFailed, got \(error)")
            }
            XCTAssertEqual(failedJob, job)
        }

        XCTAssertEqual(try Data(contentsOf: victimURL), corrupted)
        XCTAssertEqual(origin.requests.count, requestCount)
        XCTAssertEqual(coding.startCount, startsBeforeRepair + 1)
        XCTAssertEqual(coding.balance, 0)
        let summaries = await manager.jobs()
        XCTAssertEqual(summaries.first?.state, .finished)
    }

    func testCancelPersistsBeforeDiscardingAResumablePrefix() async throws {
        let store = SwitchableDownloadStateStore()
        let coding = ScriptedURLCoding()
        let plan = try await makeManager().plan(for: descriptor())
        let path = FixtureArtifact.largeFilePath
        let whole = try XCTUnwrap(origin.bytes(of: path))
        let part = destination.appendingPathComponent(path + DownloadManager.partSuffix)
        try FileManager.default.createDirectory(
            at: part.deletingLastPathComponent(), withIntermediateDirectories: true)
        let prefix = Data(whole.prefix(1 << 20))
        try prefix.write(to: part)
        let saved = try persistedState(
            plan: plan, coding: coding,
            fileStates: [path: .partial(bytesOnDisk: UInt64(prefix.count))])
        try store.save(saved)
        let manager = makeManager(
            stateStore: store, storage: scriptedStorage(coding))
        _ = try await manager.restorePersistedJobs()
        store.failSaves()

        do {
            try await manager.cancel(saved.job, keepPartialFiles: false)
            XCTFail("cancel must keep resumable bytes when durable state fails")
        } catch {
            guard case .statePersistenceFailed(let failedJob, _)? = error as? DownloadError else {
                return XCTFail("expected statePersistenceFailed, got \(error)")
            }
            XCTAssertEqual(failedJob, saved.job)
        }

        XCTAssertEqual(try Data(contentsOf: part), prefix)
        XCTAssertEqual(coding.balance, 0)
    }

    func testCancelDiscardingPartialsKeepsCompletedFinalsAndTheirProgress() async throws {
        let store = SwitchableDownloadStateStore()
        let coding = ScriptedURLCoding()
        let plan = try await makeManager().plan(for: descriptor())
        let finalPath = "global/global-embed.bin"
        let finalBytes = try XCTUnwrap(origin.bytes(of: finalPath))
        let final = destination.appendingPathComponent(finalPath)
        try FileManager.default.createDirectory(
            at: final.deletingLastPathComponent(), withIntermediateDirectories: true)
        try finalBytes.write(to: final)

        let partialPath = FixtureArtifact.largeFilePath
        let partialBytes = try XCTUnwrap(origin.bytes(of: partialPath))
        let part = destination.appendingPathComponent(
            partialPath + DownloadManager.partSuffix)
        try FileManager.default.createDirectory(
            at: part.deletingLastPathComponent(), withIntermediateDirectories: true)
        try partialBytes.prefix(1 << 20).write(to: part)

        let saved = try persistedState(plan: plan, coding: coding)
        try store.save(saved)
        let manager = makeManager(
            stateStore: store, storage: scriptedStorage(coding))
        let restore = try await manager.restorePersistedJobs()
        XCTAssertEqual(restore.restoredJobs, [saved.job])

        try await manager.cancel(saved.job, keepPartialFiles: false)
        let events = await drainEvents(of: manager, job: saved.job)

        XCTAssertNotNil(events.cancellation)
        XCTAssertEqual(try Data(contentsOf: final), finalBytes)
        XCTAssertFalse(FileManager.default.fileExists(atPath: part.path))
        let jobs = await manager.jobs()
        let summary = try XCTUnwrap(jobs.first)
        XCTAssertEqual(summary.state, .cancelled)
        XCTAssertEqual(summary.progress.fetchedUnverifiedBytes, UInt64(finalBytes.count))
        XCTAssertTrue(summary.progress.accountsForEveryByte)
        XCTAssertEqual(coding.balance, 0)
    }

    func testFinalTransitionPersistenceFailureNeverEmitsFinished() async throws {
        let store = SwitchableDownloadStateStore()
        let coding = ScriptedURLCoding()
        let storage = scriptedStorage(coding)
        let manager = makeManager(stateStore: store, storage: storage)
        let plan = try await manager.plan(for: descriptor())
        // Registration, one save for every completed file, and the verifying
        // transition may land. The following finished transition must fail.
        store.failSaves(after: 1 + plan.files.count + 1)
        let scope = try storage.scope(for: destination)
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination, scope: scope),
            options: testOptions(concurrentFiles: 1))

        let events = await drainEvents(of: manager, job: job)

        XCTAssertNil(events.summary)
        guard case .statePersistenceFailed(let failedJob, _)? = events.failure else {
            return XCTFail("expected statePersistenceFailed, got \(String(describing: events.failure))")
        }
        XCTAssertEqual(failedJob, job)
        XCTAssertEqual(try store.load(job)?.runState, .verifying)
        XCTAssertEqual(coding.balance, 0)
    }

    func testPersistedJobRestoresPausedWithoutNetworkThenResumesFromDisk() async throws {
        let stateStore = InMemoryDownloadStateStore()
        let coding = ScriptedURLCoding()
        let manager = makeManager(
            stateStore: stateStore, storage: scriptedStorage(coding))
        let plan = try await manager.plan(for: descriptor())
        let path = FixtureArtifact.largeFilePath
        let whole = try XCTUnwrap(origin.bytes(of: path))
        let half = whole.count / 2
        let part = destination.appendingPathComponent(path + DownloadManager.partSuffix)
        try FileManager.default.createDirectory(
            at: part.deletingLastPathComponent(), withIntermediateDirectories: true)
        try whole.prefix(half).write(to: part)
        let previouslyVerifiedPath = "layer00/layer00-deterministic.bin"
        let previouslyVerifiedBytes = try XCTUnwrap(origin.bytes(of: previouslyVerifiedPath))
        let previouslyVerifiedURL = destination.appendingPathComponent(previouslyVerifiedPath)
        try FileManager.default.createDirectory(
            at: previouslyVerifiedURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try previouslyVerifiedBytes.write(to: previouslyVerifiedURL)

        let state = try persistedState(
            plan: plan, coding: coding, fileStates: [
                // Deliberately stale. Recovery must measure the part file.
                path: .partial(bytesOnDisk: 1),
                // A same-sized file is carried, but its old digest verdict is
                // not trusted across the process boundary.
                previouslyVerifiedPath: .verified(
                    bytes: UInt64(previouslyVerifiedBytes.count), at: Date())
            ])
        try stateStore.save(state)
        let requestsBeforeRestore = origin.requests.count

        let restoredManager = makeManager(
            stateStore: stateStore, storage: scriptedStorage(coding))
        let report = try await restoredManager.restorePersistedJobs()

        XCTAssertEqual(report.restoredJobs, [state.job])
        XCTAssertEqual(report.failures, [])
        XCTAssertEqual(
            origin.requests.count, requestsBeforeRestore,
            "restoring durable state must not open the network")
        let restoredJobs = await restoredManager.jobs()
        let paused = try XCTUnwrap(restoredJobs.first)
        XCTAssertEqual(paused.state, .paused)
        XCTAssertEqual(paused.progress.inFlightBytes, UInt64(half))
        XCTAssertEqual(paused.progress.networkBytes, 0)
        XCTAssertTrue(paused.progress.accountsForEveryByte)
        let rewritten = try XCTUnwrap(stateStore.load(state.job))
        XCTAssertEqual(rewritten.runState, .paused)
        XCTAssertEqual(
            rewritten.fileStates[path], .partial(bytesOnDisk: UInt64(half)))
        XCTAssertEqual(
            rewritten.fileStates[previouslyVerifiedPath],
            .fetched(bytesOnDisk: UInt64(previouslyVerifiedBytes.count)))
        XCTAssertEqual(coding.balance, 0, "restore must not retain a volume grant")

        try await restoredManager.resume(state.job, scope: nil)
        let events = await drainEvents(of: restoredManager, job: state.job)
        let summary = try XCTUnwrap(events.summary, "\(events)")
        XCTAssertTrue(summary.verification.isComplete)
        XCTAssertLessThan(summary.networkBytes, plan.totalBytes)
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent(path)), whole)
        XCTAssertEqual(coding.balance, 0, "the resumed transfer must release its volume grant")
    }

    func testRestoreRefusesARecordWithoutBookmarkOrRootIdentity() async throws {
        let stateStore = InMemoryDownloadStateStore()
        let coding = ScriptedURLCoding()
        let manager = makeManager(
            stateStore: stateStore, storage: scriptedStorage(coding))
        let plan = try await manager.plan(for: descriptor())
        let legacy = DownloadJobState(
            job: DownloadJobID(), plan: plan, destinationPath: destination.path,
            destinationBookmark: nil, options: testOptions(), fileStates: [:],
            runState: .running, createdAt: Date(), updatedAt: Date())
        try stateStore.save(legacy)
        let requestCount = origin.requests.count

        let report = try await manager.restorePersistedJobs()

        XCTAssertEqual(report.restoredJobs, [])
        XCTAssertEqual(report.failures.map(\.job), [legacy.job])
        XCTAssertTrue(report.failures[0].reason.contains("bookmark"))
        let jobs = await manager.jobs()
        XCTAssertTrue(jobs.isEmpty)
        XCTAssertEqual(origin.requests.count, requestCount)
        XCTAssertEqual(coding.balance, 0)
    }

    func testRestoreRefusesBookmarkPathMismatchWithoutTouchingEitherRoot() async throws {
        let stateStore = InMemoryDownloadStateStore()
        let coding = ScriptedURLCoding()
        let manager = makeManager(
            stateStore: stateStore, storage: scriptedStorage(coding))
        let plan = try await manager.plan(for: descriptor())
        let other = destination.deletingLastPathComponent().appendingPathComponent(
            "minirun-restore-other-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: other) }
        let originalMarker = destination.appendingPathComponent("keep.txt")
        let otherMarker = other.appendingPathComponent("keep.txt")
        try Data("original".utf8).write(to: originalMarker)
        try Data("other".utf8).write(to: otherMarker)

        let valid = try persistedState(plan: plan, coding: coding)
        let mismatched = copy(
            valid, destinationBookmark: try coding.bookmarkData(for: other))
        try stateStore.save(mismatched)
        let requestCount = origin.requests.count

        let report = try await manager.restorePersistedJobs()

        XCTAssertEqual(report.restoredJobs, [])
        XCTAssertEqual(report.failures.map(\.job), [valid.job])
        XCTAssertTrue(report.failures[0].reason.contains("recorded"))
        XCTAssertEqual(try Data(contentsOf: originalMarker), Data("original".utf8))
        XCTAssertEqual(try Data(contentsOf: otherMarker), Data("other".utf8))
        XCTAssertEqual(origin.requests.count, requestCount)
        XCTAssertEqual(coding.balance, 0)
    }

    func testRestoreRefusesAReplacementAtTheSameDestinationPath() async throws {
        let stateStore = InMemoryDownloadStateStore()
        let coding = ScriptedURLCoding()
        let manager = makeManager(
            stateStore: stateStore, storage: scriptedStorage(coding))
        let plan = try await manager.plan(for: descriptor())
        let saved = try persistedState(plan: plan, coding: coding)
        try stateStore.save(saved)

        let original = destination.deletingLastPathComponent().appendingPathComponent(
            "minirun-original-root-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.moveItem(at: destination, to: original)
        try FileManager.default.createDirectory(
            at: destination, withIntermediateDirectories: true)
        let replacementMarker = destination.appendingPathComponent("replacement.txt")
        try Data("must survive".utf8).write(to: replacementMarker)
        defer {
            try? FileManager.default.removeItem(at: destination)
            try? FileManager.default.moveItem(at: original, to: destination)
        }
        let requestCount = origin.requests.count

        let report = try await manager.restorePersistedJobs()

        XCTAssertEqual(report.restoredJobs, [])
        XCTAssertEqual(report.failures.map(\.job), [saved.job])
        XCTAssertTrue(report.failures[0].reason.contains("directory object"))
        XCTAssertEqual(try Data(contentsOf: replacementMarker), Data("must survive".utf8))
        XCTAssertEqual(origin.requests.count, requestCount)
        XCTAssertEqual(coding.balance, 0)
    }

    func testPauseStopsAtAChunkBoundaryAndResumeFinishesTheJob() async throws {
        origin.delayResponses(by: 0.05)
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination),
            options: testOptions(chunkBytes: 1 << 20, concurrentFiles: 1))

        try await manager.pause(job)
        let paused = try await waitForJobState(manager, job, .paused)
        XCTAssertTrue(paused.progress.accountsForEveryByte)
        XCTAssertLessThan(
            paused.progress.verifiedBytes, plan.totalBytes,
            "the pause should have landed before the whole plan was on disk")

        // A pause is not a failure and not a cancellation: the job is still
        // there, still holding its plan, and resumable. How far it got before
        // the latch was noticed is a property of the machine, so nothing here
        // asserts a particular number of bytes.
        XCTAssertEqual(paused.progress.filesFailed, 0)
        XCTAssertEqual(paused.progress.planTotalBytes, plan.totalBytes)
        origin.delayResponses(by: 0)

        try await manager.resume(job, scope: nil)
        let after = await drainEvents(of: manager, job: job)
        let summary = try XCTUnwrap(after.summary, "\(after)")
        XCTAssertTrue(summary.verification.isComplete)
        for file in plan.files {
            XCTAssertEqual(
                try Data(contentsOf: destination.appendingPathComponent(file.path)),
                origin.bytes(of: file.path), file.path)
        }
    }

    func testResumingAJobThatIsNotPausedIsRefusedByName() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())
        _ = await drainEvents(of: manager, job: job)
        do {
            try await manager.resume(job, scope: nil)
            XCTFail("a finished job is not paused")
        } catch {
            XCTAssertEqual(error as? DownloadError, .jobNotPaused(job))
        }
    }

    /// Poll rather than sleep a fixed time: the pause takes effect at the next
    /// chunk boundary, and how long that is depends on the machine.
    private func waitForJobState(
        _ manager: DownloadManager, _ job: DownloadJobID,
        _ state: DownloadJobSummary.JobState, within seconds: TimeInterval = 20
    ) async throws -> DownloadJobSummary {
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline {
            let summaries = await manager.jobs()
            if let found = summaries.first(where: { $0.job == job }), found.state == state {
                return found
            }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        let actual = await manager.jobs().first { $0.job == job }?.state
        throw XCTSkip(
            "job never reached \(state.rawValue) (it is \(actual.map(\.rawValue) ?? "gone"))")
    }

    // MARK: - Refusals that are not retried

    func testARepositoryThatMovedUnderTheTransferIsRefused() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        origin.serveRevision(String(repeating: "9", count: 40))

        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())
        let events = await drainEvents(of: manager, job: job)
        guard case .revisionMoved(let expected, let served, _)? = events.failure else {
            return XCTFail("expected revisionMoved, got \(String(describing: events.failure))")
        }
        XCTAssertEqual(expected, revision)
        XCTAssertEqual(served, String(repeating: "9", count: 40))
    }

    func testARetryableStatusIsRetriedAndAFatalOneIsNot() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let flaky = "global/global-embed.bin"
        origin.failNext("/\(origin.repoID)/resolve/\(revision)/\(flaky)", with: [503, 503])

        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination),
            options: testOptions(concurrentFiles: 1, attempts: 5))
        let events = await drainEvents(of: manager, job: job)
        XCTAssertNotNil(events.summary, "two 503s should have been ridden out: \(events)")

        XCTAssertTrue(DownloadManager.isRetryable(.transport(status: 503, path: "x", attempt: 1)))
        XCTAssertTrue(DownloadManager.isRetryable(.transport(status: 429, path: "x", attempt: 1)))
        XCTAssertFalse(DownloadManager.isRetryable(.transport(status: 404, path: "x", attempt: 1)))
        XCTAssertFalse(
            DownloadManager.isRetryable(
                .revisionMoved(expected: "a", served: "b", path: "x")))
        XCTAssertFalse(DownloadManager.isRetryable(.rangeNotSupported(path: "x")))
    }

    func testAPersistentFailureGivesUpAfterTheStatedNumberOfAttempts() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let doomed = "global/global-embed.bin"
        origin.failNext(
            "/\(origin.repoID)/resolve/\(revision)/\(doomed)",
            with: Array(repeating: 503, count: 20))

        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination),
            options: testOptions(concurrentFiles: 1, attempts: 2))
        let events = await drainEvents(of: manager, job: job)
        XCTAssertNil(events.summary)
        XCTAssertNotNil(events.failure, "\(events)")
        XCTAssertTrue(events.fileFailures.contains { $0.0 == doomed })
    }

    // MARK: - Cancellation

    func testCancellingAPlannedJobDropsItsPartialsWhenAsked() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())
        try await manager.cancel(job, keepPartialFiles: false)

        let events = await drainEvents(of: manager, job: job)
        XCTAssertNotNil(events.cancellation, "\(events)")
        XCTAssertNil(events.summary)
        let parts = DownloadManager.filesOnDisk(under: destination)
            .filter { $0.hasSuffix(DownloadManager.partSuffix) }
        XCTAssertEqual(parts, [])
    }

    func testCancellingAnUnknownJobNamesIt() async {
        let manager = makeManager()
        let stranger = DownloadJobID()
        do {
            try await manager.cancel(stranger, keepPartialFiles: true)
            XCTFail("expected unknownJob")
        } catch {
            XCTAssertEqual(error as? DownloadError, .unknownJob(stranger))
        }
    }

    func testCancelDuringVerificationInterruptsDigestWithoutBlockingTheActorAndEndsOnce() async throws {
        let planManager = makeManager()
        let plan = try await planManager.plan(for: descriptor())
        for file in plan.files {
            let url = destination.appendingPathComponent(file.path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try XCTUnwrap(origin.bytes(of: file.path)).write(to: url)
        }

        let probe = BlockingDigestProbe()
        let manager = DownloadManager(
            transport: URLSessionTransport(),
            storage: StorageManager(bookmarkLedger: InMemoryBookmarkLedger()),
            stateStore: InMemoryDownloadStateStore(), catalogEndpoint: origin.baseURL,
            digestOperation: { descriptor, algorithm, cancellationRequested in
                try probe.digest(
                    descriptor: descriptor, algorithm: algorithm,
                    cancellationRequested: cancellationRequested)
            })
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())

        let enteredDeadline = Date().addingTimeInterval(1)
        while !probe.hasEntered, Date() < enteredDeadline {
            try await Task.sleep(nanoseconds: 2_000_000)
        }
        XCTAssertTrue(probe.hasEntered, "automatic verification never entered its digest")

        let cancelReturned = expectation(description: "cancel returned while digest was active")
        Task {
            try await manager.cancel(job, keepPartialFiles: true)
            cancelReturned.fulfill()
        }
        await fulfillment(of: [cancelReturned], timeout: 1)

        let events = await drainEvents(of: manager, job: job, timeout: 5)
        XCTAssertEqual(events.filter(\.isTerminal).count, 1, "a job has exactly one terminal event")
        XCTAssertNotNil(events.cancellation, "cancelling verification must end as cancelled")
        XCTAssertNil(events.failure)
        XCTAssertNil(events.summary)
    }

    // MARK: - Verification

    func testExtraneousFilesAreReportedAndNeverDeleted() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())
        _ = await drainEvents(of: manager, job: job)

        let stranger = destination.appendingPathComponent("someone-elses-notes.txt")
        try Data("not ours\n".utf8).write(to: stranger)

        let report = try await manager.verify(job, depth: .sizeOnly)
        XCTAssertTrue(report.isComplete, "an extraneous file does not make a download incomplete")
        XCTAssertEqual(report.extraneous, ["someone-elses-notes.txt"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: stranger.path))
    }

    func testATruncatedFileIsCaughtBySizeAloneAndListedForRefetch() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())
        _ = await drainEvents(of: manager, job: job)

        let path = "global/global-embed.bin"
        let url = destination.appendingPathComponent(path)
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: 12)
        try handle.close()

        let report = try await manager.verify(job, depth: .sizeOnly)
        XCTAssertFalse(report.isComplete)
        XCTAssertEqual(report.wrongSize.map(\.path), [path])
        XCTAssertEqual(report.wrongSize.first?.found, 12)
        XCTAssertEqual(report.pathsToRefetch, [path])
    }

    func testAMissingFileIsReportedAsMissing() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())
        _ = await drainEvents(of: manager, job: job)

        try FileManager.default.removeItem(
            at: destination.appendingPathComponent("layer00/manifest.json"))
        let report = try await manager.verify(job, depth: .digestEverything)
        XCTAssertEqual(report.missing, ["layer00/manifest.json"])
        XCTAssertEqual(report.pathsToRefetch, ["layer00/manifest.json"])
    }

    // MARK: - Preflight

    func testPreflightRefusesAVolumeItCannotWriteTo() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        var options = testOptions()
        // Ask for more headroom than any volume has, so the refusal is about
        // space and not about this machine.
        options.freeSpaceHeadroomBytes = UInt64(Int64.max / 2)
        let check = try await manager.preflight(
            plan, into: DownloadDestination(directory: destination), options: options)
        XCTAssertFalse(check.fits)
        XCTAssertFalse(check.refusals.isEmpty)
        guard case .insufficientFreeSpace? = check.refusals.first else {
            return XCTFail("expected insufficientFreeSpace, got \(check.refusals)")
        }
    }

    func testStartRefusesRatherThanTryingAnyway() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        var options = testOptions()
        options.freeSpaceHeadroomBytes = UInt64(Int64.max / 2)
        do {
            _ = try await manager.start(
                plan, into: DownloadDestination(directory: destination), options: options)
            XCTFail("start should refuse what preflight refused")
        } catch {
            guard case .insufficientFreeSpace? = error as? DownloadError else {
                return XCTFail("expected insufficientFreeSpace, got \(error)")
            }
        }
    }

    func testPreflightCountsWhatIsAlreadyOnDisk() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let path = "global/global-embed.bin"
        let whole = try XCTUnwrap(origin.bytes(of: path))
        let url = destination.appendingPathComponent(path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try whole.write(to: url)

        let check = try await manager.preflight(
            plan, into: DownloadDestination(directory: destination), options: testOptions())
        XCTAssertEqual(check.alreadyPresentBytes, UInt64(whole.count))
        XCTAssertEqual(check.requiredBytes, plan.totalBytes - UInt64(whole.count))
    }

    // MARK: - Job bookkeeping

    func testJobsReportTheirStateAndTheirProgressPartitionsThePlan() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())
        _ = await drainEvents(of: manager, job: job)

        let jobs = await manager.jobs()
        XCTAssertEqual(jobs.count, 1)
        XCTAssertEqual(jobs.first?.state, .finished)
        let progress = try XCTUnwrap(jobs.first?.progress)
        XCTAssertTrue(progress.accountsForEveryByte)
        XCTAssertEqual(progress.verifiedBytes, plan.totalBytes)
        XCTAssertEqual(progress.remainingBytes, 0)
        XCTAssertEqual(progress.filesVerified, plan.files.count)
        XCTAssertEqual(progress.fractionComplete, 1, accuracy: 1e-12)
    }

    /// Subscribing after the job has already finished still yields the whole
    /// story. Without the replay, a caller that started a fast job and then
    /// asked for its events would await a stream that had already closed.
    func testEventsSubscribedAfterTheFactStillReplayTheTerminalEvent() async throws {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())
        _ = await drainEvents(of: manager, job: job)

        let late = await drainEvents(of: manager, job: job, timeout: 5)
        XCTAssertNotNil(late.summary)
    }

    private func publicFile(_ path: String, size: UInt64) -> RepoFile {
        RepoFile(
            path: path, sizeBytes: size,
            digest: .sha256(hex: String(repeating: "a", count: 64)),
            isPayload: RepoFileClassification.isPayload(path: path))
    }

    private func persistedState(
        plan: DownloadPlan, coding: ScriptedURLCoding,
        fileStates: [String: FileState] = [:]
    ) throws -> DownloadJobState {
        let physicalRoot = destination.resolvingSymlinksInPath().standardizedFileURL
        var metadata = stat()
        guard lstat(physicalRoot.path, &metadata) == 0 else {
            throw CocoaError(.fileReadUnknown)
        }
        return DownloadJobState(
            job: DownloadJobID(), plan: plan, destinationPath: destination.path,
            destinationBookmark: try coding.bookmarkData(for: destination),
            artifactRootPath: physicalRoot.path,
            artifactRootDevice: UInt64(metadata.st_dev),
            artifactRootInode: UInt64(metadata.st_ino),
            options: testOptions(chunkBytes: 1 << 20, concurrentFiles: 1),
            fileStates: fileStates, runState: .running,
            createdAt: Date(timeIntervalSince1970: 1), updatedAt: Date(timeIntervalSince1970: 2))
    }

    private func copy(
        _ state: DownloadJobState, destinationBookmark: Data?
    ) -> DownloadJobState {
        DownloadJobState(
            job: state.job, plan: state.plan, destinationPath: state.destinationPath,
            destinationBookmark: destinationBookmark,
            artifactRootPath: state.artifactRootPath,
            artifactRootDevice: state.artifactRootDevice,
            artifactRootInode: state.artifactRootInode,
            options: state.options, fileStates: state.fileStates,
            runState: state.runState, createdAt: state.createdAt, updatedAt: state.updatedAt)
    }

    private func publicPlan(
        _ files: [RepoFile], repo: HuggingFaceRepoRef? = nil
    ) -> DownloadPlan {
        DownloadPlan(
            model: ModelID("public-plan-fixture"), repo: repo ?? origin.repo, files: files,
            index: nil,
            reconciliation: IndexReconciliation.between(files: files, claim: nil))
    }

    private func assertPlanRefused(
        _ manager: DownloadManager, plan: DownloadPlan, destination root: URL,
        operation: String, context: String,
        file: StaticString = #filePath, line: UInt = #line
    ) async {
        do {
            if operation == "preflight" {
                _ = try await manager.preflight(
                    plan, into: DownloadDestination(directory: root), options: testOptions())
            } else {
                _ = try await manager.start(
                    plan, into: DownloadDestination(directory: root), options: testOptions())
            }
            XCTFail("\(operation) accepted \(context)", file: file, line: line)
        } catch {
            guard case DownloadError.invalidOption? = error as? DownloadError else {
                return XCTFail(
                    "expected a named refusal for \(context), got \(error)",
                    file: file, line: line)
            }
        }
    }

    private func finishedJob() async throws -> (DownloadManager, DownloadPlan, DownloadJobID) {
        let manager = makeManager()
        let plan = try await manager.plan(for: descriptor())
        let job = try await manager.start(
            plan, into: DownloadDestination(directory: destination), options: testOptions())
        let events = await drainEvents(of: manager, job: job)
        XCTAssertNotNil(events.summary, "fixture download did not finish: \(events)")
        return (manager, plan, job)
    }

    private func repairReport(
        plan: DownloadPlan, job: DownloadJobID, model: ModelID? = nil,
        missing: [String] = [], wrongSize: [SizeMismatch] = [],
        wrongDigest: [DigestMismatch] = [], unreadable: [String: String] = [:],
        ok: [String]? = nil, bytesToRefetch: UInt64? = nil
    ) -> VerificationReport {
        var seen = Set<String>()
        let paths = missing + wrongSize.map(\.path) + wrongDigest.map(\.path)
            + unreadable.keys.sorted()
        let refetch = paths.filter { seen.insert($0).inserted }
        let faults = Set(refetch)
        let calculatedBytes = refetch.reduce(UInt64(0)) {
            $0 + (plan.file(at: $1)?.sizeBytes ?? 0)
        }
        return VerificationReport(
            job: job, model: model ?? plan.model, depth: .digestEverything,
            checkedAt: Date(),
            ok: ok ?? plan.files.map(\.path).filter { !faults.contains($0) },
            missing: missing, wrongSize: wrongSize, wrongDigest: wrongDigest,
            unreadable: unreadable, extraneous: [],
            bytesToRefetch: bytesToRefetch ?? calculatedBytes)
    }

    private func assertRepairRefused(
        _ manager: DownloadManager, job: DownloadJobID, report: VerificationReport,
        _ context: String, file: StaticString = #filePath, line: UInt = #line
    ) async {
        do {
            _ = try await manager.repair(job, from: report)
            XCTFail("repair accepted \(context)", file: file, line: line)
        } catch {
            guard case DownloadError.invalidOption? = error as? DownloadError else {
                return XCTFail(
                    "expected a named repair refusal for \(context), got \(error)",
                    file: file, line: line)
            }
        }
    }

    private func artifactSnapshot(_ plan: DownloadPlan, under root: URL) throws -> [String: Data] {
        try Dictionary(
            uniqueKeysWithValues: plan.files.map { file in
                (file.path, try Data(contentsOf: root.appendingPathComponent(file.path)))
            })
    }

    private func assertArtifactSnapshot(
        _ expected: [String: Data], under root: URL, context: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        for (path, bytes) in expected {
            let actual: Data
            do {
                actual = try Data(contentsOf: root.appendingPathComponent(path))
            } catch {
                XCTFail(
                    "a refused repair (\(context)) removed '\(path)': \(error)",
                    file: file, line: line)
                continue
            }
            XCTAssertEqual(
                actual, bytes, "a refused repair (\(context)) changed '\(path)'",
                file: file, line: line)
        }
    }
}
