import Foundation
import XCTest

@testable import MinirunKit

/// The four checks that make the download layer trustworthy, run against
/// responses recorded from the live API rather than against anything written
/// to make the code pass.
final class RepoTreeTests: XCTestCase {

    private func files(_ repo: RecordedFixtures.Repo) throws -> [RepoFile] {
        try RecordedFixtures.treePages(repo).flatMap { try HuggingFaceTreeClient.decodePage($0) }
    }

    // MARK: - Check 1: the payload rule reproduces each index.json's own numbers

    func testPayloadRuleReproducesEachRepositorysOwnCounts() throws {
        for repo in RecordedFixtures.all {
            let all = try files(repo)
            let payload = all.filter(\.isPayload)
            let metadata = all.filter { !$0.isPayload }

            XCTAssertEqual(
                payload.count, repo.expectedPayloadFiles,
                "\(repo.repoID): payload file count")
            XCTAssertEqual(
                metadata.count, repo.expectedMetadataFiles,
                "\(repo.repoID): metadata file count")
            XCTAssertEqual(
                payload.reduce(UInt64(0)) { $0 + $1.sizeBytes }, repo.expectedPayloadBytes,
                "\(repo.repoID): payload bytes")
            XCTAssertEqual(
                metadata.reduce(UInt64(0)) { $0 + $1.sizeBytes }, repo.expectedMetadataBytes,
                "\(repo.repoID): metadata bytes")
            XCTAssertEqual(
                all.map(\.sizeBytes).max(), repo.expectedLargestFileBytes,
                "\(repo.repoID): largest file")

            // The rule is only worth stating because it agrees with what the
            // publisher counted. This is that agreement, per repository.
            let claim = try XCTUnwrap(IndexClaim.parse(RecordedFixtures.indexJSON(repo)))
            XCTAssertEqual(claim.declaredFileCount, repo.declaredFiles)
            XCTAssertEqual(claim.declaredBytes, repo.declaredBytes)

            let reconciliation = IndexReconciliation.between(files: all, claim: claim)
            XCTAssertTrue(
                reconciliation.agrees,
                "\(repo.repoID) disagrees with its own index.json: \(reconciliation.note ?? "")")
            XCTAssertNil(reconciliation.note)
        }
    }

    func testEveryMetadataFileIsARootFileOrAManifest() throws {
        for repo in RecordedFixtures.all {
            for file in try files(repo) where !file.isPayload {
                let isRoot = !file.path.contains("/")
                XCTAssertTrue(
                    isRoot || file.lastComponent == "manifest.json",
                    "\(repo.repoID): '\(file.path)' is classified as metadata but is neither")
            }
        }
    }

    // MARK: - Check 2: x-linked-etag is the tree's lfs.oid, and ETag is not

    func testResolveRedirectCarriesTheTreesDigestAndTheCDNsETagDoesNot() throws {
        for repo in RecordedFixtures.all {
            let pair = try RecordedFixtures.resolvePair(repo)
            let lfs = try XCTUnwrap(pair.treeEntry["lfs"] as? [String: Any])
            let treeOID = try XCTUnwrap(lfs["oid"] as? String)

            let redirect = HTTPResponse(
                statusCode: pair.redirect.status, headers: pair.redirect.headers, body: nil)
            XCTAssertEqual(redirect.statusCode, 302)
            XCTAssertEqual(
                redirect.linkedSHA256, treeOID,
                "\(repo.repoID): x-linked-etag must equal the tree's lfs.oid")
            XCTAssertEqual(redirect.linkedSizeBytes, (lfs["size"] as? NSNumber)?.uint64Value)
            // A resolve capture remains useful after the repository advances:
            // the response must bind its bytes to the immutable commit that
            // served them, but it need not equal today's catalog revision.
            // This is the same rule the product uses for owner-published
            // repository updates.
            let resolveRevision = try XCTUnwrap(redirect.repoCommit)
            XCTAssertTrue(
                HuggingFaceRepoRef(repoID: repo.repoID, revision: resolveRevision)
                    .isPinnedCommit)

            // The recorded CDN response. Its ETag is a Xet content hash of
            // different bytes; treating it as the digest is the one mistake
            // here that produces a green check over wrong bytes, so the
            // fixture is kept precisely to keep that difference visible.
            let content = HTTPResponse(
                statusCode: pair.content.status, headers: pair.content.headers, body: nil)
            XCTAssertEqual(content.statusCode, 206)
            let cdnETag = try XCTUnwrap(content.header("etag")).replacingOccurrences(
                of: "\"", with: "")
            XCTAssertNotEqual(
                cdnETag, treeOID,
                "\(repo.repoID): the recorded CDN ETag happens to equal the SHA-256, which would make this test vacuous")
            XCTAssertNil(
                content.linkedSHA256,
                "\(repo.repoID): the CDN response must offer no digest at all")
            XCTAssertEqual(content.contentRange?.start, 0)
            XCTAssertEqual(content.contentRange?.end, 15)
            XCTAssertEqual(
                content.contentRange?.total, (lfs["size"] as? NSNumber)?.uint64Value)
        }
    }

    // MARK: - Check 3: a non-LFS file's git-blob SHA-1, recomputed

    func testNonLFSPayloadFilesAreDigestedAsGitBlobs() throws {
        for repo in RecordedFixtures.all {
            let nonLFS = try files(repo).filter {
                $0.isPayload && $0.digest.algorithm == .gitBlobSHA1
            }
            XCTAssertEqual(
                nonLFS.count, repo.expectedNonLFSPayloadFiles,
                "\(repo.repoID): payload files stored as plain git blobs")
        }
    }

    /// The git-blob id is SHA-1 over `blob <n>\0` and then the content — not the
    /// SHA-1 of the file. Recomputed here against git's own answer for a known
    /// blob so the header can never quietly go missing.
    func testGitBlobSHA1MatchesGitsOwnAnswer() throws {
        // `printf 'hello world' | git hash-object --stdin` is a published,
        // stable value.
        XCTAssertEqual(
            FileDigestComputer.gitBlobSHA1(of: Data("hello world".utf8)),
            "95d09f2b10159347eece71399a7e2e907ea3df4f")
        // The empty blob, likewise.
        XCTAssertEqual(
            FileDigestComputer.gitBlobSHA1(of: Data()),
            "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")

        let file = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-blob-\(UUID().uuidString)")
        try Data("hello world".utf8).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        XCTAssertEqual(
            try FileDigestComputer.gitBlobSHA1(ofFileAt: file),
            "95d09f2b10159347eece71399a7e2e907ea3df4f")
        XCTAssertNotEqual(
            try FileDigestComputer.sha256(ofFileAt: file),
            "95d09f2b10159347eece71399a7e2e907ea3df4f")
    }

    // MARK: - Decoding

    func testDirectoriesAreDroppedAndEveryFileCarriesADigest() throws {
        for repo in RecordedFixtures.all {
            let entries = try RecordedFixtures.treePages(repo).flatMap { page -> [[String: Any]] in
                (try? JSONSerialization.jsonObject(with: page) as? [[String: Any]]) ?? []
            }
            let directories = entries.filter { ($0["type"] as? String) == "directory" }
            XCTAssertFalse(
                directories.isEmpty, "\(repo.repoID): the recorded tree should contain directories")

            let decoded = try files(repo)
            XCTAssertEqual(decoded.count, entries.count - directories.count)
            for file in decoded {
                XCTAssertFalse(file.digest.hex.isEmpty)
                XCTAssertEqual(file.digest.hex, file.digest.hex.lowercased())
            }
        }
    }

    func testTreeDecodingRejectsAnEntryWithNoDigest() {
        let body = Data(#"[{"type":"file","path":"a/b.bin","size":10}]"#.utf8)
        XCTAssertThrowsError(try HuggingFaceTreeClient.decodePage(body)) { error in
            guard case CatalogError.malformedResponse(let detail)? = error as? CatalogError else {
                return XCTFail("expected a malformedResponse, got \(error)")
            }
            XCTAssertTrue(detail.contains("a/b.bin"))
        }
    }

    func testTreeDecodingRejectsAbsoluteTraversalAndNonCanonicalPaths() throws {
        let paths = [
            "", "/private/tmp/escape.bin", "../escape.bin", "a/../../escape.bin",
            "a/../escape.bin", "a/./escape.bin", "a//escape.bin", "C:/escape.bin",
            #"C:\escape.bin"#, #"\\server\share\escape.bin"#,
        ]
        for path in paths {
            let fragment = path.isEmpty ? "canonical repository-relative path" : path
            assertMalformed(
                try treeData([gitEntry(path: path, size: 1)]), contains: fragment,
                "expected '\(path)' to be refused")
        }
    }

    func testTreeDecodingRejectsDuplicatePathsWithinAPage() throws {
        let entry = gitEntry(path: "layer00/manifest.json", size: 12)
        assertMalformed(
            try treeData([entry, entry]), contains: "more than once",
            "a duplicate path must not produce an ambiguous plan")
    }

    func testTreeDecodingRejectsInvalidDigestsAndSizes() throws {
        let invalidLFSOIDs = [
            String(repeating: "a", count: 63), String(repeating: "a", count: 65),
            String(repeating: "g", count: 64),
        ]
        for oid in invalidLFSOIDs {
            assertMalformed(
                try treeData([lfsEntry(path: "weights.bin", size: 1, oid: oid)]),
                contains: "sha256")
        }

        assertMalformed(
            try treeData([
                [
                    "type": "file", "path": "plain.json",
                    "oid": String(repeating: "a", count: 40),
                ]
            ]), contains: "size")
        assertMalformed(
            try treeData([
                [
                    "type": "file", "path": "weights.bin",
                    "oid": String(repeating: "b", count: 40),
                    "lfs": ["oid": String(repeating: "a", count: 64)],
                ]
            ]), contains: "size")

        let invalidSizes: [Any] = [-1, 1.5, true, "12", NSNull()]
        for size in invalidSizes {
            assertMalformed(
                try treeData([gitEntry(path: "plain.json", size: size)]), contains: "size")
        }
    }

    func testTreeDecodingAcceptsZeroSizeAndTheDocumentedLFSTopLevelFallback() throws {
        let entries: [[String: Any]] = [
            gitEntry(path: "empty.json", size: 0),
            [
                "type": "file", "path": "weights.bin", "size": 17,
                "oid": String(repeating: "b", count: 40),
                "lfs": ["oid": String(repeating: "A", count: 64)],
            ],
        ]
        let files = try HuggingFaceTreeClient.decodePage(treeData(entries))
        XCTAssertEqual(files.map(\.path), ["empty.json", "weights.bin"])
        XCTAssertEqual(files.map(\.sizeBytes), [0, 17])
        XCTAssertEqual(files.last?.digest.hex, String(repeating: "a", count: 64))
    }

    func testPaginationAcceptsSameOriginAndKeepsAuthorizationOnThatOrigin() async throws {
        let endpoint = URL(string: "https://tree.example:443")!
        let second = URL(string: "https://tree.example/page-2")!
        let transport = ScriptedTreeTransport([
            HTTPResponse(
                statusCode: 200,
                headers: ["link": "<\(second.absoluteString)>; rel=\"next\""],
                body: try treeData([gitEntry(path: "a.json", size: 1)])),
            HTTPResponse(
                statusCode: 200, headers: [:],
                body: try treeData([gitEntry(path: "b.json", size: 2)])),
            HTTPResponse(
                statusCode: 200, headers: [:],
                body: try authorityData(
                    revision: testRepo.revision, paths: ["a.json", "b.json"])),
        ])
        let client = HuggingFaceTreeClient(
            transport: transport, endpoint: endpoint, token: { "tree-secret" })

        let files = try await client.files(in: testRepo)
        XCTAssertEqual(files.map(\.path), ["a.json", "b.json"])
        let requests = await transport.capturedRequests()
        XCTAssertEqual(requests.count, 3)
        XCTAssertTrue(requests.allSatisfy { $0.headers["authorization"] == "Bearer tree-secret" })
        XCTAssertTrue(requests[2].url.path.contains("/revision/\(testRepo.revision)"))
        XCTAssertEqual(
            requests[2].url.query?.components(separatedBy: "expand=").count, 3,
            "sha and siblings must both be requested explicitly")
    }

    func testPaginationRejectsEveryCrossOriginAxisBeforeSendingTheToken() async throws {
        let endpoint = URL(string: "https://tree.example")!
        let hostileTargets = [
            "http://tree.example/page-2",       // scheme
            "https://attacker.example/page-2",  // host
            "https://tree.example:444/page-2",  // port
        ]

        for hostile in hostileTargets {
            let transport = ScriptedTreeTransport([
                HTTPResponse(
                    statusCode: 200, headers: ["link": "<\(hostile)>; rel=\"next\""],
                    body: try treeData([gitEntry(path: "a.json", size: 1)]))
            ])
            let client = HuggingFaceTreeClient(
                transport: transport, endpoint: endpoint, token: { "must-not-leak" })

            do {
                _ = try await client.files(in: testRepo)
                XCTFail("expected cross-origin pagination to be refused: \(hostile)")
            } catch {
                guard case CatalogError.malformedResponse(let detail)? = error as? CatalogError
                else { return XCTFail("expected malformedResponse, got \(error)") }
                XCTAssertTrue(detail.contains("cross-origin"), detail)
            }

            let requests = await transport.capturedRequests()
            XCTAssertEqual(requests.count, 1, "the hostile origin must never be requested")
            XCTAssertEqual(requests.first?.url.host, "tree.example")
            XCTAssertEqual(
                requests.first?.headers["authorization"], "Bearer must-not-leak",
                "the token belongs only on the original origin")
        }
    }

    func testPaginationRejectsDuplicatePathsAcrossPages() async throws {
        let endpoint = URL(string: "https://tree.example")!
        let second = URL(string: "https://tree.example/page-2")!
        let duplicate = gitEntry(path: "same.json", size: 1)
        let transport = ScriptedTreeTransport([
            HTTPResponse(
                statusCode: 200,
                headers: ["link": "<\(second.absoluteString)>; rel=\"next\""],
                body: try treeData([duplicate])),
            HTTPResponse(statusCode: 200, headers: [:], body: try treeData([duplicate])),
        ])

        do {
            _ = try await HuggingFaceTreeClient(transport: transport, endpoint: endpoint)
                .files(in: testRepo)
            XCTFail("a duplicate path split across pages must be refused")
        } catch {
            guard case CatalogError.malformedResponse(let detail)? = error as? CatalogError else {
                return XCTFail("expected malformedResponse, got \(error)")
            }
            XCTAssertTrue(detail.contains("more than once"), detail)
        }
    }

    func testMissingFinalPageLinkCannotMasqueradeAsACompleteTree() async throws {
        let transport = ScriptedTreeTransport([
            HTTPResponse(
                statusCode: 200, headers: [:],
                body: try treeData([gitEntry(path: "a.json", size: 1)])),
            HTTPResponse(
                statusCode: 200, headers: [:],
                body: try authorityData(
                    revision: testRepo.revision, paths: ["a.json", "b.json"])),
        ])

        do {
            _ = try await HuggingFaceTreeClient(
                transport: transport, endpoint: URL(string: "https://tree.example")!
            ).files(in: testRepo)
            XCTFail("a stripped pagination link produced a complete tree")
        } catch let error as CatalogError {
            guard case .malformedResponse(let detail) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains("not complete"), detail)
            XCTAssertTrue(detail.contains("b.json"), detail)
        }
    }

    func testRepositoryInfoMustBindTheExactPinnedRevision() async throws {
        let transport = ScriptedTreeTransport([
            HTTPResponse(
                statusCode: 200, headers: [:],
                body: try treeData([gitEntry(path: "a.json", size: 1)])),
            HTTPResponse(
                statusCode: 200, headers: [:],
                body: try authorityData(
                    revision: String(repeating: "2", count: 40), paths: ["a.json"])),
        ])

        do {
            _ = try await HuggingFaceTreeClient(
                transport: transport, endpoint: URL(string: "https://tree.example")!
            ).files(in: testRepo)
            XCTFail("repository-info authority from another commit was accepted")
        } catch let error as CatalogError {
            guard case .malformedResponse(let detail) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains("expected \(testRepo.revision)"), detail)
        }
    }

    func testIndexClaimRejectsIntegersThatAreInvalidOrCannotFitItsPublicTypes() {
        let tooManyFiles = UInt64(Int.max) + 1
        let invalid = [
            #"{"files":-1,"bytes":1}"#,
            #"{"files":1.5,"bytes":1}"#,
            #"{"files":true,"bytes":1}"#,
            "{\"files\":\(tooManyFiles),\"bytes\":1}",
            #"{"files":1,"bytes":18446744073709551616}"#,
            #"{"total_bytes":1,"units":[{"files":18446744073709551615},{"files":1}]}"#,
        ]
        for json in invalid {
            XCTAssertNil(
                IndexClaim.parse(Data(json.utf8)),
                "invalid index arithmetic was accepted: \(json)")
        }
    }

    func testMaliciousTreeAndIndexAggregatesSaturateInsteadOfTrapping() throws {
        let oid = String(repeating: "a", count: 64)
        let body = Data(
            """
            [
              {"type":"file","path":"payload/a.bin","lfs":{"oid":"\(oid)","size":18446744073709551615}},
              {"type":"file","path":"payload/b.bin","lfs":{"oid":"\(oid)","size":1}}
            ]
            """.utf8)
        let files = try HuggingFaceTreeClient.decodePage(body)
        XCTAssertEqual(files.map(\.sizeBytes), [UInt64.max, 1])

        let lowClaim = IndexClaim(
            declaredFileCount: Int.min, declaredBytes: 0,
            sourceRepo: nil, sourceRevision: nil)
        let low = IndexReconciliation.between(files: files, claim: lowClaim)
        XCTAssertEqual(low.treeBytes, UInt64.max)
        XCTAssertEqual(low.payloadBytes, UInt64.max)
        XCTAssertEqual(low.fileCountDelta, Int.min)
        XCTAssertEqual(low.byteDelta, Int64.min)

        let highClaim = IndexClaim(
            declaredFileCount: Int.max, declaredBytes: UInt64.max,
            sourceRepo: nil, sourceRevision: nil)
        let high = IndexReconciliation.between(files: [], claim: highClaim)
        XCTAssertEqual(high.fileCountDelta, Int.max)
        XCTAssertEqual(high.byteDelta, Int64.max)
    }

    func testTreeAndSmallFileRejectAmbiguousRepositoryInputsBeforeNetworkAccess() async throws {
        let transport = ScriptedTreeTransport([])
        let client = HuggingFaceTreeClient(
            transport: transport, endpoint: URL(string: "https://catalog.example")!)
        let revision = String(repeating: "a", count: 40)
        let invalidRepositories = [
            "nanguoyu/%2e%2e-minirun",
            #"nanguoyu/Future\alias-minirun"#,
            "nanguoyu/Future model-minirun",
            "nanguoyu/Futuré-minirun",
        ]

        for repoID in invalidRepositories {
            let repo = HuggingFaceRepoRef(repoID: repoID, revision: revision)
            do {
                _ = try await client.files(in: repo)
                XCTFail("files(in:) accepted \(repoID)")
            } catch {
                guard case CatalogError.malformedResponse? = error as? CatalogError else {
                    return XCTFail("expected malformedResponse for \(repoID), got \(error)")
                }
            }
            do {
                _ = try await client.fetchSmallFile("index.json", in: repo)
                XCTFail("fetchSmallFile accepted \(repoID)")
            } catch {
                guard case CatalogError.malformedResponse? = error as? CatalogError else {
                    return XCTFail("expected malformedResponse for \(repoID), got \(error)")
                }
            }
        }

        do {
            _ = try await client.fetchSmallFile("../index.json", in: testRepo)
            XCTFail("fetchSmallFile accepted a traversal path")
        } catch {
            guard case CatalogError.malformedResponse? = error as? CatalogError else {
                return XCTFail("expected malformedResponse for traversal, got \(error)")
            }
        }
        let requests = await transport.capturedRequests()
        XCTAssertTrue(requests.isEmpty)
    }

    private var testRepo: HuggingFaceRepoRef {
        HuggingFaceRepoRef(
            repoID: "nanguoyu/Fixture-minirun", revision: String(repeating: "1", count: 40))
    }

    private func gitEntry(path: String, size: Any) -> [String: Any] {
        [
            "type": "file", "path": path, "size": size,
            "oid": String(repeating: "a", count: 40),
        ]
    }

    private func lfsEntry(path: String, size: Any, oid: String) -> [String: Any] {
        [
            "type": "file", "path": path, "size": size,
            "oid": String(repeating: "b", count: 40),
            "lfs": ["oid": oid, "size": size],
        ]
    }

    private func treeData(_ entries: [[String: Any]]) throws -> Data {
        try JSONSerialization.data(withJSONObject: entries)
    }

    private func authorityData(revision: String, paths: [String]) throws -> Data {
        try JSONSerialization.data(withJSONObject: [
            "sha": revision,
            "siblings": paths.map { ["rfilename": $0] },
        ])
    }

    private func assertMalformed(
        _ body: Data, contains fragment: String, _ message: String = "",
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try HuggingFaceTreeClient.decodePage(body), message, file: file, line: line
        ) { error in
            guard case CatalogError.malformedResponse(let detail)? = error as? CatalogError else {
                return XCTFail("expected malformedResponse, got \(error)", file: file, line: line)
            }
            XCTAssertTrue(detail.contains(fragment), detail, file: file, line: line)
        }
    }
}

private actor ScriptedTreeTransport: HTTPTransport {
    private var responses: [HTTPResponse]
    private var requests: [HTTPRequest] = []

    init(_ responses: [HTTPResponse]) { self.responses = responses }

    func data(for request: HTTPRequest) async throws -> HTTPResponse {
        requests.append(request)
        guard !responses.isEmpty else { throw ScriptedTreeTransportError.unexpectedRequest }
        return responses.removeFirst()
    }

    func download(_ request: HTTPRequest, into sink: HTTPDownloadSink) async throws
        -> HTTPResponse
    {
        throw ScriptedTreeTransportError.unexpectedRequest
    }

    func capturedRequests() -> [HTTPRequest] { requests }
}

private enum ScriptedTreeTransportError: Error {
    case unexpectedRequest
}
