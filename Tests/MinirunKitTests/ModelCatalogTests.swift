import Foundation
import XCTest

@testable import MinirunKit

final class ModelCatalogTests: XCTestCase {

    // MARK: - The copy compiled into this build

    func testBundledSnapshotDecodesAndNamesItselfBundled() {
        let snapshot = ModelCatalog.bundled
        XCTAssertEqual(snapshot.schemaVersion, ModelCatalogSnapshot.currentSchemaVersion)
        XCTAssertEqual(snapshot.origin, .bundled)
        XCTAssertEqual(snapshot.models.count, 3)
        XCTAssertNotNil(snapshot.descriptor(.kimiK3))
        XCTAssertNotNil(snapshot.descriptor(.deepseekV4Flash))
        XCTAssertNotNil(snapshot.descriptor(.minimaxH3))
    }

    /// Every bundled model is a published repository whose tree can be walked.
    /// A locally built entry could never be verified against anything, so the
    /// catalogue does not carry one.
    func testEveryBundledModelIsAPublishedRepository() {
        for model in ModelCatalog.bundled.models {
            XCTAssertNotNil(
                model.source.repo,
                "\(model.id) has no repository behind it")
        }
    }

    /// Every revision in the bundled catalogue is a commit. A branch name would
    /// let the bytes under a pinned plan change without a single response
    /// looking wrong.
    func testEveryBundledRevisionIsAPinnedCommit() {
        for model in ModelCatalog.bundled.models {
            guard let repo = model.source.repo else { continue }
            XCTAssertTrue(
                repo.isPinnedCommit,
                "\(model.id) points at '\(repo.revision)', which is not a 40-hex commit")
        }
    }

    /// The bundled numbers are the ones the recorded trees produce. If someone
    /// edits one without the other, this fails rather than mis-sizing a
    /// terabyte at run time.
    func testBundledNumbersAgreeWithTheRecordedTrees() throws {
        let pairs: [(RecordedFixtures.Repo, ModelID)] = [
            (RecordedFixtures.kimiK3, .kimiK3),
            (RecordedFixtures.deepseekV4Flash, .deepseekV4Flash),
            (RecordedFixtures.minimaxH3, .minimaxH3),
        ]
        for (fixture, id) in pairs {
            let descriptor = try XCTUnwrap(ModelCatalog.bundled.descriptor(id))
            XCTAssertEqual(descriptor.source.repo?.repoID, fixture.repoID)
            XCTAssertEqual(descriptor.source.repo?.revision, fixture.revision)
            XCTAssertEqual(descriptor.payloadBytes, fixture.expectedPayloadBytes)
            XCTAssertEqual(descriptor.metadataBytes, fixture.expectedMetadataBytes)
            XCTAssertEqual(descriptor.payloadFileCount, fixture.expectedPayloadFiles)
            XCTAssertEqual(descriptor.metadataFileCount, fixture.expectedMetadataFiles)
            XCTAssertEqual(descriptor.largestFileBytes, fixture.expectedLargestFileBytes)
            XCTAssertEqual(
                descriptor.totalBytes, fixture.expectedPayloadBytes + fixture.expectedMetadataBytes)
            XCTAssertEqual(
                descriptor.totalFileCount,
                fixture.expectedPayloadFiles + fixture.expectedMetadataFiles)
        }
    }

    func testSnapshotRoundTripsThroughItsOwnCoder() throws {
        let data = try MinirunKitJSON.encoder().encode(ModelCatalog.bundled)
        let decoded = try MinirunKitJSON.decoder().decode(ModelCatalogSnapshot.self, from: data)
        XCTAssertEqual(decoded, ModelCatalog.bundled)
    }

    // MARK: - Open live catalogue

    func testPublishedModelListingIsOneIndexedRequestAndWalksNoRepositoryTree() async throws {
        let knownRevision = String(repeating: "a", count: 40)
        let unknownRevision = String(repeating: "b", count: 40)
        let list = try JSONSerialization.data(withJSONObject: [
            ["id": "nanguoyu/Kimi-K3-minirun", "sha": knownRevision],
            ["id": "nanguoyu/Zeta-minirun", "sha": unknownRevision],
        ])
        let transport = CatalogScriptedTransport([
            .init(pathFragment: "/api/models", body: list)
        ])
        let source = HuggingFaceCatalogSource(
            endpoint: URL(string: "https://catalog.example")!, transport: transport)

        let published = try await source.listPublishedModels()

        XCTAssertEqual(published.map(\.displayName), ["Kimi K3", "Zeta"])
        XCTAssertEqual(published.map(\.id), [
            .kimiK3,
            .discoveredHuggingFaceRepository("nanguoyu/Zeta-minirun"),
        ])
        XCTAssertEqual(published.map(\.repository.revision), [knownRevision, unknownRevision])
        let requests = await transport.capturedRequests()
        XCTAssertEqual(requests.count, 1)
        XCTAssertTrue(requests[0].url.path.hasSuffix("/api/models"))
        let queryItems = URLComponents(
            url: requests[0].url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        XCTAssertEqual(queryItems.filter { $0.name == "expand" }.map(\.value), ["sha"])
        XCTAssertFalse(queryItems.contains { $0.name == "full" })
    }

    func testResolvingOnePublishedModelWalksOnlyThatPinnedTree() async throws {
        let revision = String(repeating: "c", count: 40)
        let entries = [
            catalogGitEntry(path: "README.md", size: 3),
            catalogGitEntry(path: "unit/weights.bin", size: 17),
        ]
        let transport = CatalogScriptedTransport([
            .init(
                pathFragment: "/nanguoyu/Zeta-minirun/tree/",
                body: try catalogTreeData(entries)),
            .init(
                pathFragment: "/nanguoyu/Zeta-minirun/revision/",
                body: try catalogAuthorityData(revision: revision, entries: entries)),
        ])
        let source = HuggingFaceCatalogSource(
            endpoint: URL(string: "https://catalog.example")!, transport: transport)
        let reference = PublishedModelReference(
            id: .discoveredHuggingFaceRepository("nanguoyu/Zeta-minirun"),
            displayName: "Zeta",
            repository: HuggingFaceRepoRef(
                repoID: "nanguoyu/Zeta-minirun", revision: revision))

        let descriptor = try await source.resolvePublishedModel(reference)

        XCTAssertEqual(descriptor.id, reference.id)
        XCTAssertEqual(descriptor.source.repo, reference.repository)
        XCTAssertEqual(descriptor.payloadBytes, 17)
        XCTAssertEqual(descriptor.metadataBytes, 3)
        let requests = await transport.capturedRequests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertFalse(requests.contains { $0.url.path.hasSuffix("/api/models") })
    }

    func testResolvingOnePublishedModelMergesWithTheExistingOfflineCatalog() async throws {
        let existing = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let seeded = ModelCatalogSnapshot(
            generatedAt: .distantPast, origin: .cached, models: [existing])
        let cache = MemoryCatalogCache(seeded: seeded)
        let revision = String(repeating: "d", count: 40)
        let entries = [catalogGitEntry(path: "weights.bin", size: 23)]
        let source = HuggingFaceCatalogSource(
            endpoint: URL(string: "https://catalog.example")!,
            transport: CatalogScriptedTransport([
                .init(
                    pathFragment: "/nanguoyu/Zeta-minirun/tree/",
                    body: try catalogTreeData(entries)),
                .init(
                    pathFragment: "/nanguoyu/Zeta-minirun/revision/",
                    body: try catalogAuthorityData(revision: revision, entries: entries)),
            ]))
        let catalog = ModelCatalog(
            live: source,
            bundled: ModelCatalogSnapshot(
                generatedAt: .distantPast, origin: .bundled, models: []),
            cache: cache)
        let reference = PublishedModelReference(
            id: .discoveredHuggingFaceRepository("nanguoyu/Zeta-minirun"),
            displayName: "Zeta",
            repository: HuggingFaceRepoRef(
                repoID: "nanguoyu/Zeta-minirun", revision: revision))

        let result = try await catalog.resolvePublishedModel(reference)

        XCTAssertEqual(Set(result.snapshot.models.map(\.id)), Set([.kimiK3, reference.id]))
        XCTAssertEqual(try cache.load(), result.snapshot)
    }

    func testLiveCatalogueKeepsKnownIdentityAndSurfacesANewMinirunRepository() async throws {
        let bundledKnown = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let knownRevision = try XCTUnwrap(bundledKnown.source.repo?.revision)
        let unknownRevision = String(repeating: "b", count: 40)
        let list = try JSONSerialization.data(withJSONObject: [
            ["id": "nanguoyu/Kimi-K3-minirun", "sha": knownRevision],
            ["id": "nanguoyu/Zeta-minirun", "sha": unknownRevision],
            ["id": "nanguoyu/not-a-product-repository", "sha": knownRevision],
        ])
        let knownEntries = [
            catalogGitEntry(path: "README.md", size: 2),
            catalogGitEntry(path: "layer01/weights.bin", size: 11),
        ]
        let unknownEntries = [
            catalogGitEntry(path: "README.md", size: 3),
            catalogGitEntry(path: "unit/manifest.json", size: 5),
            catalogGitEntry(path: "unit/weights.bin", size: 7),
        ]
        let knownTree = try catalogTreeData(knownEntries)
        let unknownTree = try catalogTreeData(unknownEntries)
        let transport = CatalogScriptedTransport([
            .init(pathFragment: "/api/models", body: list),
            .init(pathFragment: "/nanguoyu/Kimi-K3-minirun/tree/", body: knownTree),
            .init(
                pathFragment: "/nanguoyu/Kimi-K3-minirun/revision/",
                body: try catalogAuthorityData(revision: knownRevision, entries: knownEntries)),
            .init(pathFragment: "/nanguoyu/Zeta-minirun/tree/", body: unknownTree),
            .init(
                pathFragment: "/nanguoyu/Zeta-minirun/revision/",
                body: try catalogAuthorityData(revision: unknownRevision, entries: unknownEntries)),
        ])

        let snapshot = try await HuggingFaceCatalogSource(
            endpoint: URL(string: "https://catalog.example")!, transport: transport
        ).fetch()

        let known = try XCTUnwrap(snapshot.descriptor(.kimiK3))
        XCTAssertEqual(known.id, .kimiK3)
        XCTAssertEqual(known.architecture, bundledKnown.architecture)
        XCTAssertEqual(known.layout, bundledKnown.layout)
        XCTAssertEqual(known.runner, bundledKnown.runner)
        XCTAssertEqual(known.source.repo?.revision, knownRevision)

        let discoveredID = ModelID.discoveredHuggingFaceRepository(
            "nanguoyu/Zeta-minirun")
        let discovered = try XCTUnwrap(snapshot.descriptor(discoveredID))
        XCTAssertEqual(discovered.displayName, "Zeta")
        XCTAssertEqual(discovered.architecture, .unrecognized)
        XCTAssertEqual(discovered.layout, .unrecognized)
        XCTAssertEqual(discovered.runner, .none)
        XCTAssertNil(discovered.minimumBudgetBytes)
        XCTAssertEqual(discovered.source.repo?.revision, unknownRevision)
        XCTAssertEqual(discovered.payloadBytes, 7)
        XCTAssertEqual(discovered.metadataBytes, 8)
        XCTAssertEqual(discovered.payloadFileCount, 1)
        XCTAssertEqual(discovered.metadataFileCount, 2)
        XCTAssertEqual(discovered.largestFileBytes, 7)
        XCTAssertEqual(discovered.totalBytes, 15)
        XCTAssertEqual(discovered.notes.count, 1)
        XCTAssertTrue(discovered.notes[0].contains("does not recognize"))

        let cached = try MinirunKitJSON.encoder().encode(snapshot)
        let decoded = try MinirunKitJSON.decoder().decode(ModelCatalogSnapshot.self, from: cached)
        XCTAssertEqual(
            decoded.models, snapshot.models,
            "an open-catalog entry must survive the cache boundary with its unknown capability intact")
        XCTAssertEqual(decoded.schemaVersion, snapshot.schemaVersion)
        XCTAssertEqual(decoded.origin, snapshot.origin)

        let requests = await transport.capturedRequests()
        XCTAssertEqual(requests.count, 5, "unrelated author repositories must not be walked")
        XCTAssertTrue(requests[0].url.absoluteString.contains("author=nanguoyu"))
        XCTAssertTrue(requests[0].url.absoluteString.contains("expand=sha"))
    }

    func testRecognizedRepositoriesUseStableRepositoryIdentity() throws {
        let bundled = ModelCatalog.bundled.models.compactMap { descriptor -> (HuggingFaceRepoRef, ModelID)? in
            guard let repo = descriptor.source.repo,
                descriptor.architecture != .unrecognized,
                descriptor.layout != .unrecognized
            else { return nil }
            return (repo, descriptor.id)
        }
        XCTAssertEqual(HuggingFaceCatalogSource.recognizedRepos.count, bundled.count)
        for (repo, id) in bundled {
            XCTAssertEqual(HuggingFaceCatalogSource.recognizedRepos[repo.repoID], id)
        }
        XCTAssertEqual(bundled.count, 3)
    }

    func testKnownRepositoryAtANewRevisionKeepsItsFormatIdentityButPinsTheNewTree() async throws {
        let bundled = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let revision = String(repeating: "f", count: 40)
        XCTAssertNotEqual(revision, bundled.source.repo?.revision)
        let list = try JSONSerialization.data(withJSONObject: [
            ["id": "nanguoyu/Kimi-K3-minirun", "sha": revision]
        ])
        let entries = [
            catalogGitEntry(path: "README.md", size: 2),
            catalogGitEntry(path: "layer01/weights.bin", size: 11),
        ]
        let transport = CatalogScriptedTransport([
            .init(pathFragment: "/api/models", body: list),
            .init(
                pathFragment: "/nanguoyu/Kimi-K3-minirun/tree/",
                body: try catalogTreeData(entries)),
            .init(
                pathFragment: "/nanguoyu/Kimi-K3-minirun/revision/",
                body: try catalogAuthorityData(revision: revision, entries: entries)),
        ])

        let snapshot = try await HuggingFaceCatalogSource(
            endpoint: URL(string: "https://catalog.example")!, transport: transport
        ).fetch()

        let current = try XCTUnwrap(snapshot.descriptor(.kimiK3))
        XCTAssertEqual(current.source.repo?.revision, revision)
        XCTAssertEqual(current.architecture, bundled.architecture)
        XCTAssertEqual(current.layout, bundled.layout)
        XCTAssertEqual(current.runner, bundled.runner)
        XCTAssertEqual(current.minimumBudgetBytes, bundled.minimumBudgetBytes)
        XCTAssertEqual(current.payloadBytes, 11)
        XCTAssertEqual(current.metadataBytes, 2)

        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-revision-bound-catalog-\(UUID().uuidString)")
        let cache = FileCatalogCache(url: directory.appendingPathComponent("catalog.json"))
        defer { try? FileManager.default.removeItem(at: directory) }
        try cache.save(snapshot)
        let cached = try XCTUnwrap(cache.load())
        XCTAssertEqual(cached.descriptor(.kimiK3), current)
    }

    func testLiveCatalogueRejectsAmbiguousDuplicateRepositoryEntries() async throws {
        let list = try JSONSerialization.data(withJSONObject: [
            [
                "id": "nanguoyu/Future-minirun",
                "sha": String(repeating: "a", count: 40),
            ],
            [
                "id": "nanguoyu/Future-minirun",
                "sha": String(repeating: "b", count: 40),
            ],
        ])
        let transport = CatalogScriptedTransport([
            .init(pathFragment: "/api/models", body: list)
        ])

        do {
            _ = try await HuggingFaceCatalogSource(
                endpoint: URL(string: "https://catalog.example")!, transport: transport
            ).fetch()
            XCTFail("an ambiguous repository listing must be refused")
        } catch {
            guard case CatalogError.malformedResponse(let detail)? = error as? CatalogError else {
                return XCTFail("expected malformedResponse, got \(error)")
            }
            XCTAssertTrue(detail.contains("more than once"), detail)
        }
        let requests = await transport.capturedRequests()
        XCTAssertEqual(requests.count, 1)
    }

    func testLiveCatalogueFollowsAuthorizedListPaginationAndMergesExactDuplicates() async throws {
        let revision = String(repeating: "a", count: 40)
        let pageOne = try JSONSerialization.data(withJSONObject: [
            ["id": "nanguoyu/Future-minirun", "sha": revision]
        ])
        let pageTwo = try JSONSerialization.data(withJSONObject: [
            ["id": "nanguoyu/Future-minirun", "sha": revision]
        ])
        let entries = [catalogGitEntry(path: "weights.bin", size: 7)]
        let transport = CatalogScriptedTransport([
            .init(
                pathFragment: "/hub/api/models?author=nanguoyu&expand=sha", body: pageOne,
                headers: [
                    "Link": "<https://catalog.example/hub/api/models?author=nanguoyu&expand=sha&cursor=second>; rel=\"next\""
                ]),
            .init(pathFragment: "cursor=second", body: pageTwo),
            .init(
                pathFragment: "/nanguoyu/Future-minirun/tree/",
                body: try catalogTreeData(entries)),
            .init(
                pathFragment: "/nanguoyu/Future-minirun/revision/",
                body: try catalogAuthorityData(revision: revision, entries: entries)),
        ])

        let snapshot = try await HuggingFaceCatalogSource(
            endpoint: URL(string: "https://catalog.example/hub")!, transport: transport,
            token: { "catalog-token" }
        ).fetch()

        XCTAssertEqual(snapshot.models.count, 1)
        XCTAssertEqual(snapshot.models[0].source.repo?.revision, revision)
        let requests = await transport.capturedRequests()
        XCTAssertEqual(requests.count, 4)
        XCTAssertEqual(requests[0].headers["authorization"], "Bearer catalog-token")
        XCTAssertEqual(requests[1].headers["authorization"], "Bearer catalog-token")
    }

    func testLiveCatalogueRejectsConflictingDuplicateAcrossPages() async throws {
        let firstRevision = String(repeating: "a", count: 40)
        let secondRevision = String(repeating: "b", count: 40)
        let pageOne = try JSONSerialization.data(withJSONObject: [
            ["id": "nanguoyu/Future-minirun", "sha": firstRevision]
        ])
        let pageTwo = try JSONSerialization.data(withJSONObject: [
            ["id": "nanguoyu/Future-minirun", "sha": secondRevision]
        ])
        let transport = CatalogScriptedTransport([
            .init(
                pathFragment: "/api/models", body: pageOne,
                headers: [
                    "link": "<https://catalog.example/api/models?author=nanguoyu&expand=sha&cursor=2>; rel=next"
                ]),
            .init(pathFragment: "cursor=2", body: pageTwo),
        ])

        do {
            _ = try await HuggingFaceCatalogSource(
                endpoint: URL(string: "https://catalog.example")!, transport: transport
            ).fetch()
            XCTFail("an ambiguous cross-page repository listing must be refused")
        } catch let error as CatalogError {
            guard case .malformedResponse(let detail) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains("more than once"), detail)
        }
        let requests = await transport.capturedRequests()
        XCTAssertEqual(requests.count, 2, "no repository tree may be read after list ambiguity")
    }

    func testListPageIdentityDoesNotAliasEncodedCursorDelimitersWithSeparateItems() async throws {
        let empty = try JSONSerialization.data(withJSONObject: [])
        let transport = CatalogScriptedTransport([
            .init(
                pathFragment: "/api/models", body: empty,
                headers: [
                    "link": "<https://catalog.example/api/models?author=nanguoyu&expand=sha&cursor=x%26limit%3D1>; rel=next"
                ]),
            .init(
                pathFragment: "cursor=x%26limit%3D1", body: empty,
                headers: [
                    "link": "<https://catalog.example/api/models?author=nanguoyu&expand=sha&cursor=x&limit=1>; rel=next"
                ]),
            .init(pathFragment: "cursor=x&limit=1", body: empty),
        ])

        let snapshot = try await HuggingFaceCatalogSource(
            endpoint: URL(string: "https://catalog.example")!, transport: transport
        ).fetch()

        XCTAssertTrue(snapshot.models.isEmpty)
        let requests = await transport.capturedRequests()
        XCTAssertEqual(requests.count, 3)
    }

    func testLiveCatalogueRejectsUnauthorizedAndRepeatedNextPages() async throws {
        let empty = try JSONSerialization.data(withJSONObject: [])
        let unauthorizedTargets = [
            "https://evil.example/api/models?author=nanguoyu&expand=sha&cursor=2",
            "http://catalog.example/api/models?author=nanguoyu&expand=sha&cursor=2",
            "https://catalog.example:444/api/models?author=nanguoyu&expand=sha&cursor=2",
            "https://catalog.example/api/models/other?author=nanguoyu&expand=sha&cursor=2",
            "https://catalog.example/api/models?author=attacker&expand=sha&cursor=2",
            "https://catalog.example/api/models?author=nanguoyu&expand=downloads&cursor=2",
            "https://catalog.example/api/models?author=nanguoyu&author=nanguoyu&expand=sha&cursor=2",
            "https://catalog.example/api/models?author=nanguoyu&expand=sha&cursor=2&cursor=3",
            "https://catalog.example/api/models?author=nanguoyu&expand=sha&access_token=secret",
            "https://catalog.example/api/models?author=nanguoyu&expand=sha&expand=downloads",
            "https://catalog.example/api/models?author=nanguoyu&expand=sha&cursor=",
            "https://catalog.example/api/models?author=nanguoyu&expand=sha&p=-1",
            "https://catalog.example/api/models?author=nanguoyu&expand=sha&limit=all",
        ]
        for target in unauthorizedTargets {
            let transport = CatalogScriptedTransport([
                .init(
                    pathFragment: "/api/models", body: empty,
                    headers: ["link": "<\(target)>; rel=\"next\""])
            ])
            do {
                _ = try await HuggingFaceCatalogSource(
                    endpoint: URL(string: "https://catalog.example")!, transport: transport
                ).fetch()
                XCTFail("unauthorized next page was accepted: \(target)")
            } catch let error as CatalogError {
                guard case .malformedResponse = error else {
                    return XCTFail("unexpected error for \(target): \(error)")
                }
            }
            let requests = await transport.capturedRequests()
            XCTAssertEqual(requests.count, 1, "an unauthorized target must not receive a request")
        }

        let loopTransport = CatalogScriptedTransport([
            .init(
                pathFragment: "/api/models", body: empty,
                headers: [
                    "link": "<https://catalog.example/api/models?expand=sha&author=nanguoyu>; rel=next"
                ])
        ])
        do {
            _ = try await HuggingFaceCatalogSource(
                endpoint: URL(string: "https://catalog.example")!, transport: loopTransport
            ).fetch()
            XCTFail("a reordered duplicate page was accepted")
        } catch let error as CatalogError {
            guard case .malformedResponse(let detail) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains("paginated back"), detail)
        }
        let loopRequests = await loopTransport.capturedRequests()
        XCTAssertEqual(loopRequests.count, 1)
    }

    func testNewMinirunRepositoryWithoutAPinnedCommitIsRefusedBeforeTreeAccess() async throws {
        let list = try JSONSerialization.data(withJSONObject: [
            ["id": "nanguoyu/Future-minirun", "sha": "main"]
        ])
        let transport = CatalogScriptedTransport([
            .init(pathFragment: "/api/models", body: list)
        ])

        do {
            _ = try await HuggingFaceCatalogSource(
                endpoint: URL(string: "https://catalog.example")!, transport: transport
            ).fetch()
            XCTFail("a moving repository revision must be refused")
        } catch {
            XCTAssertEqual(
                error as? CatalogError,
                .unpinnedRevision(repoID: "nanguoyu/Future-minirun", given: "main"))
        }
        let requests = await transport.capturedRequests()
        XCTAssertEqual(requests.count, 1, "the unpinned repository tree must not be requested")
    }

    func testLiveCatalogueIgnoresCrossAuthorAndNonCanonicalRepositories() async throws {
        let revision = String(repeating: "a", count: 40)
        let list = try JSONSerialization.data(withJSONObject: [
            ["id": "attacker/Injected-minirun", "sha": revision],
            ["id": "nanguoyu/../Injected-minirun", "sha": revision],
            ["id": "nanguoyu/%2e%2e-minirun", "sha": revision],
            ["id": "nanguoyu/owner\\x-minirun", "sha": revision],
            ["id": "nanguoyu/space name-minirun", "sha": revision],
            ["id": "nanguoyu/control\u{0001}-minirun", "sha": revision],
            ["id": "nanguoyu/-minirun", "sha": revision],
            ["id": "nanguoyu/Nested-minirun/extra", "sha": revision],
            ["id": "nanguoyu/Case-MINIRUN", "sha": revision],
        ])
        let transport = CatalogScriptedTransport([
            .init(pathFragment: "/api/models", body: list)
        ])

        let snapshot = try await HuggingFaceCatalogSource(
            endpoint: URL(string: "https://catalog.example")!, transport: transport
        ).fetch()

        XCTAssertTrue(snapshot.models.isEmpty)
        XCTAssertNil(
            snapshot.descriptor(.kimiK3),
            "a live fetch that found nothing must not be backfilled from the bundled catalogue")
        let requests = await transport.capturedRequests()
        XCTAssertEqual(requests.count, 1)
    }

    func testLiveCatalogueSaturatesMaliciousTreeByteTotals() async throws {
        let revision = String(repeating: "c", count: 40)
        let list = try JSONSerialization.data(withJSONObject: [
            ["id": "nanguoyu/Overflow-minirun", "sha": revision]
        ])
        let oid = String(repeating: "d", count: 64)
        let gitOID = String(repeating: "e", count: 40)
        let tree = Data(
            """
            [
              {"type":"file","path":"payload/a.bin","lfs":{"oid":"\(oid)","size":18446744073709551615}},
              {"type":"file","path":"payload/b.bin","lfs":{"oid":"\(oid)","size":1}},
              {"type":"file","path":"README.md","size":5,"oid":"\(gitOID)"}
            ]
            """.utf8)
        let transport = CatalogScriptedTransport([
            .init(pathFragment: "/api/models", body: list),
            .init(pathFragment: "/nanguoyu/Overflow-minirun/tree/", body: tree),
            .init(
                pathFragment: "/nanguoyu/Overflow-minirun/revision/",
                body: try catalogAuthorityData(
                    revision: revision,
                    paths: ["payload/a.bin", "payload/b.bin", "README.md"])),
        ])

        let snapshot = try await HuggingFaceCatalogSource(
            endpoint: URL(string: "https://catalog.example")!, transport: transport
        ).fetch()
        let model = try XCTUnwrap(
            snapshot.descriptor(
                .discoveredHuggingFaceRepository("nanguoyu/Overflow-minirun")))

        XCTAssertEqual(model.payloadBytes, UInt64.max)
        XCTAssertEqual(model.metadataBytes, 5)
        XCTAssertEqual(model.totalBytes, UInt64.max)
        XCTAssertEqual(model.largestFileBytes, UInt64.max)
    }

    func testOpenCatalogueRejectsStableTreeTruncationAgainstPinnedSiblingAuthority() async throws {
        let revision = String(repeating: "d", count: 40)
        let list = try JSONSerialization.data(withJSONObject: [
            ["id": "nanguoyu/Future-minirun", "sha": revision]
        ])
        let truncated = [catalogGitEntry(path: "README.md", size: 2)]
        let transport = CatalogScriptedTransport([
            .init(pathFragment: "/api/models", body: list),
            .init(
                pathFragment: "/nanguoyu/Future-minirun/tree/",
                body: try catalogTreeData(truncated)),
            .init(
                pathFragment: "/nanguoyu/Future-minirun/revision/",
                body: try catalogAuthorityData(
                    revision: revision,
                    paths: ["README.md", "unit/weights.bin"])),
        ])

        do {
            _ = try await HuggingFaceCatalogSource(
                endpoint: URL(string: "https://catalog.example")!, transport: transport
            ).fetch()
            XCTFail("an open repository derived a descriptor from a truncated tree")
        } catch let error as CatalogError {
            guard case .malformedResponse(let detail) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains("not complete"), detail)
            XCTAssertTrue(detail.contains("unit/weights.bin"), detail)
        }
    }

    // MARK: - Fitness

    func testK3IsRefusedOnADeviceThatCannotOfferItsMeasuredMinimum() throws {
        let k3 = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let small = DeviceProfile(
            platform: .iOS, processMemoryBudgetBytes: 3_000_000_000,
            hasMLXGPU: true, hasIncreasedMemoryLimitEntitlement: true)
        let fitness = k3.fitness(on: small)
        XCTAssertEqual(fitness.verdict, .refused)
        XCTAssertEqual(fitness.requiredFreeBytes, k3.totalBytes)
        XCTAssertEqual(fitness.headroomBytes, 3_000_000_000 - 5_800_000_000)
    }

    func testAModelWithNoRunnerSaysSoRatherThanGuessing() throws {
        let h3 = try XCTUnwrap(ModelCatalog.bundled.descriptor(.minimaxH3))
        let big = DeviceProfile(
            platform: .macOS, processMemoryBudgetBytes: 128 << 30,
            hasMLXGPU: true, hasIncreasedMemoryLimitEntitlement: false)
        XCTAssertEqual(h3.fitness(on: big).verdict, .noRunner)
        XCTAssertNil(h3.fitness(on: big).headroomBytes)
    }

    func testAnIPhoneWithoutTheEntitlementIsRunnableWithCaveats() throws {
        let k3 = try XCTUnwrap(ModelCatalog.bundled.descriptor(.kimiK3))
        let phone = DeviceProfile(
            platform: .iOS, processMemoryBudgetBytes: 10_000_000_000,
            hasMLXGPU: true, hasIncreasedMemoryLimitEntitlement: false)
        XCTAssertEqual(k3.fitness(on: phone).verdict, .runnableWithCaveats)

        let entitled = DeviceProfile(
            platform: .iOS, processMemoryBudgetBytes: 10_000_000_000,
            hasMLXGPU: true, hasIncreasedMemoryLimitEntitlement: true)
        XCTAssertEqual(k3.fitness(on: entitled).verdict, .runnable)
    }

    func testDescriptorSummariesAndHeadroomSaturateAtPublicIntegerBounds() {
        let model = ModelDescriptor(
            id: ModelID("overflow-fixture"), displayName: "Overflow fixture",
            architecture: .kimiK3MoE, layout: .k3FlagshipLayerStreams,
            source: .locallyBuilt(tool: "test"),
            payloadBytes: .max, metadataBytes: 1,
            payloadFileCount: .max, metadataFileCount: 1, largestFileBytes: .max,
            minimumBudgetBytes: .max, runner: .decodeRunner,
            licenseName: "test", licenseAcknowledgementRequired: false, notes: [])
        XCTAssertEqual(model.totalBytes, UInt64.max)
        XCTAssertEqual(model.totalFileCount, Int.max)

        let noMemory = DeviceProfile(
            platform: .macOS, processMemoryBudgetBytes: 0,
            hasMLXGPU: true, hasIncreasedMemoryLimitEntitlement: false)
        XCTAssertEqual(model.fitness(on: noMemory).headroomBytes, Int64.min)
    }

    // MARK: - Divergence

    func testDivergenceNamesEveryFieldThatMoved() throws {
        let bundled = ModelCatalog.bundled
        let k3 = try XCTUnwrap(bundled.descriptor(.kimiK3))
        let moved = ModelDescriptor(
            id: k3.id, displayName: k3.displayName, architecture: k3.architecture,
            layout: k3.layout,
            source: .huggingFaceRepo(
                HuggingFaceRepoRef(
                    repoID: k3.source.repo!.repoID, revision: String(repeating: "a", count: 40))),
            payloadBytes: k3.payloadBytes + 1, metadataBytes: k3.metadataBytes,
            payloadFileCount: k3.payloadFileCount, metadataFileCount: k3.metadataFileCount,
            largestFileBytes: k3.largestFileBytes, minimumBudgetBytes: k3.minimumBudgetBytes,
            runner: k3.runner, licenseName: k3.licenseName,
            licenseAcknowledgementRequired: k3.licenseAcknowledgementRequired, notes: k3.notes)
        let live = ModelCatalogSnapshot(generatedAt: Date(), origin: .live, models: [moved])

        let divergence = live.divergence(from: bundled)
        let fields = Set(divergence.filter { $0.model == .kimiK3 }.map(\.field))
        XCTAssertTrue(fields.contains("payloadBytes"))
        XCTAssertTrue(fields.contains("revision"))
        // The models the live snapshot did not mention are reported missing
        // rather than assumed unchanged.
        XCTAssertEqual(
            divergence.filter { $0.field == "presence" }.count,
            ModelCatalog.bundled.models.count - 1)
    }

    func testIdenticalSnapshotsDivergeInNothing() {
        XCTAssertTrue(ModelCatalog.bundled.divergence(from: ModelCatalog.bundled).isEmpty)
    }

    // MARK: - Policy and fallback

    func testBundledOnlyNeverConsultsTheSourceOrTheCache() async {
        let source = ExplodingCatalogSource()
        let catalog = ModelCatalog(live: source, cache: ExplodingCache())
        let snapshot = await catalog.snapshot(.bundledOnly)
        XCTAssertEqual(snapshot.origin, .bundled)
        let calls = await source.callCount
        XCTAssertEqual(calls, 0)
    }

    func testAFailedLiveFetchFallsBackToTheCacheAndSaysSo() async throws {
        let cached = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .live, models: ModelCatalog.bundled.models)
        let cache = MemoryCatalogCache(seeded: cached)
        let catalog = ModelCatalog(live: ExplodingCatalogSource(), cache: cache)
        let snapshot = await catalog.snapshot(.forceLive)
        XCTAssertEqual(snapshot.origin, .cached)
        XCTAssertEqual(snapshot.models.count, ModelCatalog.bundled.models.count)
    }

    func testAFailedLiveFetchWithNoCacheFallsBackToBundled() async {
        let catalog = ModelCatalog(live: ExplodingCatalogSource(), cache: MemoryCatalogCache())
        let origin = await catalog.snapshot(.forceLive).origin
        XCTAssertEqual(origin, .bundled)
    }

    /// `snapshot` never throws; `refresh` does. An operator who pressed Refresh
    /// is owed the reason it did not happen.
    func testRefreshThrowsWhereSnapshotWouldHaveFallenBack() async {
        let catalog = ModelCatalog(live: ExplodingCatalogSource(), cache: MemoryCatalogCache())
        do {
            _ = try await catalog.refresh()
            XCTFail("refresh should have thrown")
        } catch {
            XCTAssertEqual(error as? CatalogError, .malformedResponse("exploding source"))
        }
    }

    func testRefreshResultKeepsFreshSnapshotAndSurfacesCacheSaveFailure() async throws {
        let fresh = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .live, models: ModelCatalog.bundled.models)
        let catalog = ModelCatalog(
            live: FixedCatalogSource(snapshot: fresh), cache: SaveFailingCatalogCache())

        let result = try await catalog.refreshResult()

        XCTAssertEqual(result.snapshot, fresh)
        XCTAssertNotNil(result.cacheWarning)
        XCTAssertTrue(result.cacheWarning?.contains("offline cache refused") == true)
        let memoized = await catalog.snapshot(.preferCachedWithin(3600))
        XCTAssertEqual(
            memoized, fresh,
            "a durability failure must not discard the usable live snapshot")
    }

    func testRefreshResultHasNoWarningWhenTheFreshSnapshotWasSaved() async throws {
        let fresh = ModelCatalogSnapshot(
            generatedAt: Date(), origin: .live, models: ModelCatalog.bundled.models)
        let result = try await ModelCatalog(
            live: FixedCatalogSource(snapshot: fresh), cache: MemoryCatalogCache()
        ).refreshResult()

        XCTAssertEqual(result.snapshot, fresh)
        XCTAssertNil(result.cacheWarning)
    }

    func testDescriptorThrowsForAModelTheCatalogueDoesNotHave() async {
        let catalog = ModelCatalog(live: ExplodingCatalogSource(), cache: MemoryCatalogCache())
        do {
            _ = try await catalog.descriptor(ModelID("not-a-model"))
            XCTFail("expected unknownModel")
        } catch {
            XCTAssertEqual(error as? CatalogError, .unknownModel(ModelID("not-a-model")))
        }
    }

    func testFileCacheRoundTrip() throws {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-catalog-\(UUID().uuidString)")
            .appendingPathComponent("catalog.json")
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let cache = FileCatalogCache(url: url)
        XCTAssertNil(try cache.load())
        try cache.save(ModelCatalog.bundled)
        XCTAssertEqual(try cache.load(), ModelCatalog.bundled)
    }

    // MARK: - URL construction

    func testResolveAndTreeURLsCarryTheRevision() {
        let repo = HuggingFaceRepoRef(
            repoID: "nanguoyu/Kimi-K3-minirun",
            revision: "159987d3ac437e0aceaff0763d43ddeb549b1842")
        XCTAssertEqual(
            repo.resolveURL(path: "layer01/layer01-w1.mxfp4tile").absoluteString,
            "https://huggingface.co/nanguoyu/Kimi-K3-minirun/resolve/159987d3ac437e0aceaff0763d43ddeb549b1842/layer01/layer01-w1.mxfp4tile"
        )
        let tree = repo.treeURL().absoluteString
        XCTAssertTrue(tree.hasPrefix(
            "https://huggingface.co/api/models/nanguoyu/Kimi-K3-minirun/tree/159987d3ac437e0aceaff0763d43ddeb549b1842"
        ))
        XCTAssertTrue(tree.contains("recursive=1"))
        XCTAssertTrue(tree.contains("expand=1"))
    }

    func testAnUnpinnedRevisionIsRefused() {
        XCTAssertFalse(HuggingFaceRepoRef(repoID: "x/y", revision: "main").isPinnedCommit)
        XCTAssertFalse(
            HuggingFaceRepoRef(repoID: "x/y", revision: String(repeating: "A", count: 40))
                .isPinnedCommit,
            "an upper-case hex string is not the form the API returns and must not be accepted")
    }

    func testRepositoryIDCanonicalFormRejectsAmbiguousURLSpellings() {
        XCTAssertTrue(
            HuggingFaceRepoRef.isCanonicalRepositoryID("nanguoyu/Future.Model_2-minirun"))
        for invalid in [
            "nanguoyu/%2e%2e-minirun",
            "nanguoyu/owner\\x-minirun",
            "nanguoyu/space name-minirun",
            "nanguoyu/control\u{0001}-minirun",
            "nanguoyu/ümlaut-minirun",
            "/absolute-minirun",
            "nanguoyu/nested/repo-minirun",
            "nanguoyu/..",
        ] {
            XCTAssertFalse(
                HuggingFaceRepoRef.isCanonicalRepositoryID(invalid),
                "ambiguous repository id was accepted: \(invalid.debugDescription)")
        }
    }
}

// MARK: - Doubles

actor ExplodingCatalogSource: ModelCatalogSource {
    private(set) var callCount = 0
    func fetch() async throws -> ModelCatalogSnapshot {
        callCount += 1
        throw CatalogError.malformedResponse("exploding source")
    }
}

struct FixedCatalogSource: ModelCatalogSource {
    let snapshot: ModelCatalogSnapshot
    func fetch() async throws -> ModelCatalogSnapshot { snapshot }
}

final class MemoryCatalogCache: CatalogCache, @unchecked Sendable {
    private let lock = NSLock()
    private var stored: ModelCatalogSnapshot?

    init(seeded: ModelCatalogSnapshot? = nil) { stored = seeded }

    func load() throws -> ModelCatalogSnapshot? {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }

    func save(_ snapshot: ModelCatalogSnapshot) throws {
        lock.lock()
        stored = snapshot
        lock.unlock()
    }
}

struct ExplodingCache: CatalogCache {
    func load() throws -> ModelCatalogSnapshot? {
        XCTFail("the cache must not be consulted under .bundledOnly")
        return nil
    }
    func save(_ snapshot: ModelCatalogSnapshot) throws {
        XCTFail("the cache must not be written under .bundledOnly")
    }
}

struct SaveFailingCatalogCache: CatalogCache {
    func load() throws -> ModelCatalogSnapshot? { nil }
    func save(_ snapshot: ModelCatalogSnapshot) throws {
        throw CatalogError.malformedResponse("offline cache refused")
    }
}

private actor CatalogScriptedTransport: HTTPTransport {
    struct Step: Sendable {
        let pathFragment: String
        let body: Data
        let headers: [String: String]

        init(pathFragment: String, body: Data, headers: [String: String] = [:]) {
            self.pathFragment = pathFragment
            self.body = body
            self.headers = headers
        }
    }

    private var steps: [Step]
    private var requests: [HTTPRequest] = []

    init(_ steps: [Step]) { self.steps = steps }

    func data(for request: HTTPRequest) async throws -> HTTPResponse {
        requests.append(request)
        guard !steps.isEmpty else { throw CatalogScriptedTransportError.unexpectedRequest }
        let step = steps.removeFirst()
        guard request.url.absoluteString.contains(step.pathFragment) else {
            throw CatalogScriptedTransportError.wrongRequest(
                expected: step.pathFragment, actual: request.url.absoluteString)
        }
        return HTTPResponse(statusCode: 200, headers: step.headers, body: step.body)
    }

    func download(_ request: HTTPRequest, into sink: HTTPDownloadSink) async throws
        -> HTTPResponse
    {
        throw CatalogScriptedTransportError.unexpectedRequest
    }

    func capturedRequests() -> [HTTPRequest] { requests }
}

private enum CatalogScriptedTransportError: Error {
    case unexpectedRequest
    case wrongRequest(expected: String, actual: String)
}

private func catalogGitEntry(path: String, size: UInt64) -> [String: Any] {
    [
        "type": "file", "path": path, "size": NSNumber(value: size),
        "oid": String(repeating: "a", count: 40),
    ]
}

private func catalogTreeData(_ entries: [[String: Any]]) throws -> Data {
    try JSONSerialization.data(withJSONObject: entries)
}

private func catalogAuthorityData(
    revision: String, entries: [[String: Any]]
) throws -> Data {
    try catalogAuthorityData(
        revision: revision,
        paths: entries.compactMap { $0["path"] as? String })
}

private func catalogAuthorityData(revision: String, paths: [String]) throws -> Data {
    try JSONSerialization.data(withJSONObject: [
        "sha": revision,
        "siblings": paths.map { ["rfilename": $0] },
    ])
}
