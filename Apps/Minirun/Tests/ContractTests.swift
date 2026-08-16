import MinirunKit
import StorageCore
import XCTest

@testable import MinirunApp

/// The invariants the contract copy exists to preserve. When MinirunKit lands
/// these tests should pass unchanged against the real types; if they do not,
/// the seam moved and this is where it shows.
final class ContractTests: XCTestCase {

    // MARK: The payload / metadata classification rule

    /// A file is metadata iff it sits at the repo root, or its last path
    /// component is `manifest.json`. Everything else is payload. This rule
    /// reproduces every published index's own counts, so it is an invariant and
    /// not a heuristic.
    func testClassificationRule() {
        XCTAssertFalse(RepoFileClassification.isPayload(path: "index.json"))
        XCTAssertFalse(RepoFileClassification.isPayload(path: "README.md"))
        XCTAssertFalse(RepoFileClassification.isPayload(path: ".gitattributes"))
        XCTAssertFalse(RepoFileClassification.isPayload(path: "layer01/manifest.json"))
        XCTAssertFalse(RepoFileClassification.isPayload(path: "dit-block-00/manifest.json"))
        XCTAssertTrue(RepoFileClassification.isPayload(path: "layer01/layer01-w1.mxfp4tile"))
        XCTAssertTrue(RepoFileClassification.isPayload(path: "global/global-deterministic.bin"))
        XCTAssertTrue(RepoFileClassification.isPayload(path: "layer01/notes.json"))
    }

    /// The plans the app builds must add up: payload + metadata == total, and
    /// the tree's payload figures must match what the index declares.
    func testEveryPlanBalancesAndReconciles() throws {
        for entry in CatalogFixtures.all {
            let plan = try XCTUnwrap(PlanBuilder.plan(for: entry))
            XCTAssertEqual(
                plan.payloadBytes + plan.metadataBytes, plan.totalBytes,
                "\(entry.id) payload + metadata ≠ total")
            XCTAssertEqual(
                plan.payloadBytes, entry.descriptor.payloadBytes,
                "\(entry.id) synthesized payload bytes drifted from the descriptor")
            XCTAssertEqual(
                plan.metadataBytes, entry.descriptor.metadataBytes,
                "\(entry.id) synthesized metadata bytes drifted from the descriptor")
            XCTAssertTrue(
                plan.reconciliation.agrees,
                "\(entry.id) index and tree disagree: "
                    + "\(plan.reconciliation.fileCountDelta) files, "
                    + "\(plan.reconciliation.byteDelta) bytes")
            XCTAssertEqual(
                Set(plan.files.map(\.path)).count, plan.files.count,
                "\(entry.id) preview plan contains duplicate repository paths")
        }
    }

    /// H3 keeps a handful of payload files as plain git blobs. A verifier that
    /// assumed one algorithm would silently skip them.
    func testH3CarriesBothDigestAlgorithms() throws {
        let plan = try XCTUnwrap(PlanBuilder.plan(for: CatalogFixtures.minimaxH3))
        let payload = plan.files.filter(\.isPayload)
        let gitBlobs = payload.filter { $0.digest.algorithm == .gitBlobSHA1 }
        XCTAssertEqual(gitBlobs.count, 16)
        XCTAssertTrue(gitBlobs.allSatisfy { $0.digest.hex.count == 40 })
        XCTAssertTrue(
            payload.filter { $0.digest.algorithm == .sha256 }
                .allSatisfy { $0.digest.hex.count == 64 })
    }

    func testAnUnrecognizedCatalogEntryNeverGetsASyntheticDownloadPlan() async throws {
        let descriptor = ModelDescriptor(
            id: .discoveredHuggingFaceRepository("nanguoyu/Future-minirun"),
            displayName: "Future",
            architecture: .unrecognized,
            layout: .unrecognized,
            source: .huggingFaceRepo(
                HuggingFaceRepoRef(
                    repoID: "nanguoyu/Future-minirun",
                    revision: String(repeating: "a", count: 40))),
            payloadBytes: 100,
            metadataBytes: 10,
            payloadFileCount: 2,
            metadataFileCount: 1,
            largestFileBytes: 50,
            minimumBudgetBytes: nil,
            runner: .none,
            licenseName: "unstated",
            licenseAcknowledgementRequired: true,
            notes: [])
        let entry = AppCatalogAdapter.entry(from: descriptor)

        XCTAssertNil(PlanBuilder.plan(for: entry))
        do {
            _ = try await MockDownloadManager(entries: [entry]).plan(for: descriptor)
            XCTFail("an unknown layout must not be turned into guessed file paths")
        } catch {
            XCTAssertEqual(
                error as? CatalogError,
                .unrecognizedLayout(repoID: "nanguoyu/Future-minirun"))
        }
    }

    // MARK: Pinned revisions

    func testEveryPublishedRepoIsPinnedToACommit() {
        for entry in CatalogFixtures.all {
            guard case .huggingFaceRepo(let repo) = entry.descriptor.source else { continue }
            XCTAssertTrue(
                repo.isPinnedCommit,
                "\(entry.id) is not pinned to a 40-hex commit: \(repo.revision)")
            XCTAssertTrue(
                repo.resolveURL(path: "index.json", endpoint: HuggingFaceEndpoint.default)
                    .absoluteString.contains(repo.revision),
                "every URL must carry the revision")
        }
    }

    // MARK: Options are refused, never clamped

    func testInvalidOptionsAreRefusedByName() {
        var options = DownloadOptions()
        options.chunkBytes = 1234
        XCTAssertThrowsError(try options.validated()) { error in
            guard case DownloadError.invalidOption(let name, let value, let allowed)? =
                error as? DownloadError
            else { return XCTFail("expected invalidOption, got \(error)") }
            XCTAssertEqual(name, "chunkBytes")
            XCTAssertEqual(value, "1234")
            XCTAssertFalse(allowed.isEmpty)
        }

        var concurrency = DownloadOptions()
        concurrency.maximumConcurrentFiles = 99
        XCTAssertThrowsError(try concurrency.validated())

        XCTAssertNoThrow(try DownloadOptions().validated())
    }

    // MARK: Byte accounting

    func testProgressAccountsForEveryByte() {
        let job = DownloadJobID()
        let progress = DownloadProgress(
            job: job, planTotalBytes: 1_000, verifiedBytes: 400,
            fetchedUnverifiedBytes: 100, inFlightBytes: 60, remainingBytes: 440,
            filesTotal: 3, filesVerified: 1, filesFailed: 0, networkBytes: 560,
            wastedBytes: 0, instantaneousBytesPerSecond: 0, smoothedBytesPerSecond: 0,
            estimatedTimeRemaining: nil, startedAt: Date(), elapsed: 1)
        XCTAssertTrue(progress.accountsForEveryByte)

        let broken = DownloadProgress(
            job: job, planTotalBytes: 1_000, verifiedBytes: 400,
            fetchedUnverifiedBytes: 100, inFlightBytes: 60, remainingBytes: 1,
            filesTotal: 3, filesVerified: 1, filesFailed: 0, networkBytes: 560,
            wastedBytes: 0, instantaneousBytesPerSecond: 0, smoothedBytesPerSecond: 0,
            estimatedTimeRemaining: nil, startedAt: Date(), elapsed: 1)
        XCTAssertFalse(
            broken.accountsForEveryByte,
            "the identity has to be able to FAIL, or asserting it proves nothing")
    }

    func testByteAccountingDetectsUnattributedBytes() {
        let balanced = ByteAccounting(
            totalBytesRead: 300, deterministicBytesRead: 200, expertBytesRead: 100)
        XCTAssertTrue(balanced.isBalanced)
        XCTAssertEqual(balanced.unattributedBytes, 0)

        let leaking = ByteAccounting(
            totalBytesRead: 320, deterministicBytesRead: 200, expertBytesRead: 100)
        XCTAssertFalse(leaking.isBalanced)
        XCTAssertEqual(leaking.unattributedBytes, 20)

        // Splits are optional while a run is in flight; absent is not unbalanced.
        let partial = ByteAccounting(
            totalBytesRead: 320, deterministicBytesRead: nil, expertBytesRead: nil)
        XCTAssertTrue(partial.isBalanced)
        XCTAssertNil(partial.unattributedBytes)
    }

    /// The reconciliation this app's local copy got wrong, now settled by the
    /// kit: when the buckets overrun the total, there is no remainder to report.
    ///
    /// The old copy answered `0` here, and `0` is exactly what a run that
    /// balanced perfectly answers — so a double-counted read displayed as a
    /// clean split. Nil means unstated, and a screen that has to say something
    /// says "not stated" rather than "balanced".
    func testOverAttributedBytesHaveNoRemainderRatherThanAZeroOne() {
        let doubleCounted = ByteAccounting(
            totalBytesRead: 300, deterministicBytesRead: 200, expertBytesRead: 150)
        XCTAssertFalse(doubleCounted.isBalanced)
        XCTAssertNil(
            doubleCounted.unattributedBytes,
            "an over-attributed run has a broken identity, not a remainder of zero")

        let exact = ByteAccounting(
            totalBytesRead: 300, deterministicBytesRead: 200, expertBytesRead: 100)
        XCTAssertEqual(exact.unattributedBytes, 0)
    }

    // MARK: The CDN ETag trap

    /// `x-linked-etag` is the SHA-256. The CDN's own `ETag` is the Xet hash — a
    /// different number for the same bytes — and there is deliberately no
    /// accessor for it.
    func testLinkedETagIsUnquotedAndTheCDNETagIsNotExposed() {
        let response = HTTPResponse(
            statusCode: 302,
            headers: [
                "X-Linked-Etag": "\"ad7cd23e5f1b09c4\"",
                "X-Linked-Size": "5240799232",
                "X-Repo-Commit": "159987d3ac437e0aceaff0763d43ddeb549b1842",
                "ETag": "4aed07421f0b3d95",
            ],
            body: nil)
        XCTAssertEqual(response.linkedSHA256, "ad7cd23e5f1b09c4")
        XCTAssertEqual(response.linkedSizeBytes, 5_240_799_232)
        XCTAssertEqual(response.repoCommit, "159987d3ac437e0aceaff0763d43ddeb549b1842")
        XCTAssertNotEqual(response.linkedSHA256, response.headers["etag"])
    }

    func testContentRangeParses() {
        let response = HTTPResponse(
            statusCode: 206,
            headers: ["content-range": "bytes 1048576-2097151/5240799232"], body: nil)
        let range = response.contentRange
        XCTAssertEqual(range?.start, 1_048_576)
        XCTAssertEqual(range?.end, 2_097_151)
        XCTAssertEqual(range?.total, 5_240_799_232)
    }

    // MARK: Divergence

    func testDivergenceIsReportedFieldByField() async throws {
        let live = try await MockCatalogSource(simulateDivergence: true, latency: .zero).fetch()
        let divergences = live.divergence(from: CatalogFixtures.snapshot)
        XCTAssertFalse(divergences.isEmpty)
        XCTAssertTrue(divergences.contains { $0.field == "revision" })
        XCTAssertTrue(divergences.contains { $0.field == "payloadBytes" })
        XCTAssertTrue(divergences.allSatisfy { $0.model == .minimaxH3 })
    }

    func testNoDivergenceWhenTheLiveCatalogAgrees() async throws {
        let live = try await MockCatalogSource(latency: .zero).fetch()
        XCTAssertTrue(live.divergence(from: CatalogFixtures.snapshot).isEmpty)
    }
}

// MARK: - Design tokens

#if canImport(AppKit)
    import AppKit
    import SwiftUI

    /// Every token that is ever set as a *foreground* must be readable on every
    /// surface it can land on, in both appearances.
    ///
    /// This test exists because a whole screen shipped unreadable and the
    /// arithmetic tests all passed. Contrast is arithmetic too — there is no
    /// reason to leave it to the eye of whoever last opened the file.
    ///
    /// 4.5:1 is the WCAG AA floor for body text. It is applied here to the small
    /// print as well, because in this product the small print is the argument.
    final class MRTokenContrastTests: XCTestCase {

        private static let floor = 4.5

        /// The surfaces a foreground can be drawn on.
        private var surfaces: [(String, Color)] {
            [("abyss", MRColor.abyss), ("panel", MRColor.panel), ("raised", MRColor.raised)]
        }

        private var foregrounds: [(String, Color)] {
            [
                ("primary", MRColor.primary),
                ("secondary", MRColor.secondary),
                ("tertiary", MRColor.tertiary),
                ("ok", MRColor.ok),
                ("caution", MRColor.caution),
                ("refuse", MRColor.refuse),
                ("verify", MRColor.verify),
                ("streamDet", MRColor.streamDet),
                ("tierPinned", MRColor.tierPinned),
                // Added when the dial grew a fifth tier. Staged and pinned were
                // one blue while the dial had one tier for both, and a legend
                // swatch is a foreground like any other.
                ("tierStaged", MRColor.tierStaged),
            ]
        }

        func testEveryForegroundTokenIsReadableInBothAppearances() {
            for isDark in [false, true] {
                for (fgName, fg) in foregrounds {
                    for (bgName, bg) in surfaces {
                        let ratio = Self.contrast(fg, bg, dark: isDark)
                        XCTAssertGreaterThanOrEqual(
                            ratio, Self.floor,
                            "\(fgName) on \(bgName) in \(isDark ? "dark" : "light") is "
                                + String(format: "%.2f", ratio) + ":1")
                    }
                }
            }
        }

        /// A token that resolves to the same colour in both appearances is a
        /// token that was never made adaptive — the failure mode the blank
        /// sidebar was first blamed for. Surfaces and foregrounds must both
        /// move when the appearance does.
        func testEverySurfaceAndForegroundChangesWithTheAppearance() {
            for (name, color) in surfaces + foregrounds {
                let light = Self.components(color, dark: false)
                let dark = Self.components(color, dark: true)
                XCTAssertNotEqual(
                    light.0 + light.1 + light.2, dark.0 + dark.1 + dark.2, accuracy: 0.0001,
                    "\(name) resolves identically in light and dark")
            }
        }

        /// Text is never drawn in the colour of the surface it sits on.
        func testTextIsNeverTheSameColourAsItsSurface() {
            for isDark in [false, true] {
                for (fgName, fg) in foregrounds {
                    for (bgName, bg) in surfaces {
                        XCTAssertGreaterThan(
                            Self.contrast(fg, bg, dark: isDark), 1.0001,
                            "\(fgName) is invisible on \(bgName) in \(isDark ? "dark" : "light")")
                    }
                }
            }
        }

        // MARK: Resolution

        private static func components(_ color: Color, dark: Bool) -> (Double, Double, Double) {
            var out = (0.0, 0.0, 0.0)
            let appearance = NSAppearance(named: dark ? .darkAqua : .aqua)!
            appearance.performAsCurrentDrawingAppearance {
                guard let resolved = NSColor(color).usingColorSpace(.sRGB) else { return }
                out = (
                    Double(resolved.redComponent), Double(resolved.greenComponent),
                    Double(resolved.blueComponent)
                )
            }
            return out
        }

        private static func luminance(_ rgb: (Double, Double, Double)) -> Double {
            func channel(_ value: Double) -> Double {
                value <= 0.03928 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4)
            }
            return 0.2126 * channel(rgb.0) + 0.7152 * channel(rgb.1) + 0.0722 * channel(rgb.2)
        }

        private static func contrast(_ a: Color, _ b: Color, dark: Bool) -> Double {
            let la = luminance(components(a, dark: dark))
            let lb = luminance(components(b, dark: dark))
            return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
        }
    }
#endif
