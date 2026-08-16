import Foundation
import XCTest

@testable import MinirunKit

/// The one suite that talks to huggingface.co, and only when asked.
///
/// Off by default and gated on `MINIRUN_LIVE_HF=1`, because the ordinary suite
/// must run on a plane and must never depend on somebody else's uptime. It is
/// strictly read-only: the model list, one tree, and one `index.json` — no
/// payload byte is fetched, and the largest thing it downloads is a 27 KB
/// document.
///
/// What it is for is drift. The bundled catalogue's numbers were walked from
/// the live API on 2026-08-11 and everything this package sizes, budgets and
/// refuses against comes from them; if a repository moves, that is news, and
/// this is where it is heard.
///
///     MINIRUN_LIVE_HF=1 xcodebuild test -scheme minirun-Package \
///         -destination 'platform=macOS' -only-testing:MinirunKitTests/LiveHuggingFaceTests
final class LiveHuggingFaceTests: XCTestCase {

    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MINIRUN_LIVE_HF"] == "1",
            "set MINIRUN_LIVE_HF=1 to run the live, read-only Hugging Face checks")
    }

    func testTheLiveCatalogueStillMatchesTheBundledOne() async throws {
        let live = try await HuggingFaceCatalogSource().fetch()
        XCTAssertEqual(live.origin, .live)

        let divergence = live.divergence(from: ModelCatalog.bundled)
        // Reported, not resolved. A failure here is a statement that the
        // published repositories moved, and the bundled snapshot needs
        // re-recording — not that this code is broken.
        XCTAssertEqual(
            divergence, [],
            "the live catalogue diverges from the bundled one: "
                + divergence.map { "\($0.model).\($0.field): \($0.bundled) → \($0.live)" }
                    .joined(separator: "; "))
    }

    func testEveryLiveRevisionIsStillAPinnedCommit() async throws {
        for model in try await HuggingFaceCatalogSource().fetch().models {
            guard let repo = model.source.repo else { continue }
            XCTAssertTrue(repo.isPinnedCommit, "\(model.id): '\(repo.revision)'")
        }
    }

    /// The invariant the whole download layer rests on, checked against the
    /// live tree rather than the recording of it.
    func testTheLiveTreeStillAgreesWithItsOwnIndexJSON() async throws {
        let transport = URLSessionTransport()
        let client = HuggingFaceTreeClient(transport: transport)
        for fixture in RecordedFixtures.all {
            let repo = HuggingFaceRepoRef(repoID: fixture.repoID, revision: fixture.revision)
            let files = try await client.files(in: repo)
            let body = try await client.fetchSmallFile("index.json", in: repo)
            let claim = try XCTUnwrap(body.flatMap(IndexClaim.parse))
            let reconciliation = IndexReconciliation.between(files: files, claim: claim)
            XCTAssertTrue(
                reconciliation.agrees, "\(fixture.repoID): \(reconciliation.note ?? "")")
        }
    }
}
