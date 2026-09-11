import Foundation
import MinirunKit
import MinirunRunners
import StorageCore
import SwiftUI
import XCTest

@testable import MinirunApp

/// The model page's status line, subtitle and facts, as values.
///
/// The sentence beside the dot is the whole top of the screen, and it is the
/// one claim on the page that is composed from three sources at once — a
/// transfer record, a storage scan, and whether the drive is plugged in. That
/// makes it exactly the surface that can ship a lie, so it is resolved in a
/// function and checked here rather than assembled inside a `body`.
final class ModelPageStatusTests: XCTestCase {

    // MARK: - While a transfer is moving

    func testARunningTransferIsBlueAndNamesTheDriveAndTheTimeLeft() {
        let status = ModelPageStatusPresentation.resolve(
            state: .active(Self.snapshot(verified: 47_600_000_000)),
            remains: .checking,
            installations: [], mounted: [],
            transferDestinationName: "K3NVME",
            copyLocationName: nil)

        XCTAssertEqual(status.tone, .moving)
        XCTAssertEqual(status.sentence, "Downloading to K3NVME · about 3.7 h left")
    }

    func testAPausedTransferIsAmberBecauseItIsWaitingForTheOperator() {
        let status = ModelPageStatusPresentation.resolve(
            state: .paused(Self.snapshot(verified: 47_600_000_000)),
            remains: .checking,
            installations: [], mounted: [],
            transferDestinationName: "K3NVME",
            copyLocationName: nil)

        XCTAssertEqual(status.tone, .attention)
        XCTAssertEqual(status.sentence, "Paused to K3NVME. Resume to continue.")
    }

    func testVerificationIsBlueAndSaysWhatIsBeingChecked() {
        let status = ModelPageStatusPresentation.resolve(
            state: .verifying(.fileDigests, Self.snapshot(verified: 517_269_118_806)),
            remains: .checking,
            installations: [], mounted: [],
            transferDestinationName: "K3NVME",
            copyLocationName: nil)

        XCTAssertEqual(status.tone, .moving)
        XCTAssertTrue(status.sentence.contains("published digests"))
    }

    // MARK: - When it has stopped

    func testACancelledTransferThatKeptNothingIsGreyAndSaysSo() {
        let status = ModelPageStatusPresentation.resolve(
            state: .cancelled(Self.snapshot(verified: 0)),
            remains: .nothingKept,
            installations: [], mounted: [],
            transferDestinationName: "K3NVME",
            copyLocationName: nil)

        XCTAssertEqual(status.tone, .idle)
        XCTAssertEqual(
            status.sentence,
            "Cancelled. Nothing from this transfer is on the drive any more.")
    }

    func testAnInterruptedTransferIsAmberAndCarriesTheNamedReason() {
        let status = ModelPageStatusPresentation.resolve(
            state: .interrupted(
                Self.snapshot(verified: 47_600_000_000),
                reason: .volumeDisappeared("K3NVME")),
            remains: .driveNotConnected(volume: "K3NVME"),
            installations: [], mounted: [],
            transferDestinationName: "K3NVME",
            copyLocationName: nil)

        XCTAssertEqual(status.tone, .attention)
        XCTAssertEqual(status.sentence, "Interrupted — K3NVME disconnected.")
    }

    // MARK: - What is on the drive

    func testAFullyVerifiedCopyIsGreenAndNamesTheDriveItIsOn() {
        let artifact = Self.artifact(verification: .fullyVerified)
        let status = ModelPageStatusPresentation.resolve(
            state: .notStarted, remains: .noDestination,
            installations: [artifact], mounted: [artifact],
            transferDestinationName: nil,
            copyLocationName: "K3NVME")

        XCTAssertEqual(status.tone, .ready)
        XCTAssertEqual(
            status.sentence, "Ready on K3NVME · every file matches its published digest.")
    }

    func testACopyOnAnUnpluggedDriveAsksForTheDriveRatherThanADownload() {
        let artifact = Self.artifact(verification: .fullyVerified)
        let status = ModelPageStatusPresentation.resolve(
            state: .notStarted, remains: .noDestination,
            installations: [artifact], mounted: [],
            transferDestinationName: nil,
            copyLocationName: "K3NVME")

        XCTAssertEqual(status.tone, .attention)
        XCTAssertEqual(
            status.sentence,
            "K3NVME is not connected. Nothing needs to be downloaded again.")
    }

    func testASpotCheckedCopyStillAsksForFullVerification() {
        let artifact = Self.artifact(verification: .spotChecked)
        let status = ModelPageStatusPresentation.resolve(
            state: .notStarted, remains: .noDestination,
            installations: [artifact], mounted: [artifact],
            transferDestinationName: nil,
            copyLocationName: "K3NVME")

        XCTAssertEqual(status.tone, .attention)
        XCTAssertTrue(status.sentence.contains("a sample matched"))
        XCTAssertTrue(status.sentence.contains("Verify all files"))
    }

    func testNothingAnywhereIsGreyAndOffersBothWaysIn() {
        let status = ModelPageStatusPresentation.resolve(
            state: .notStarted, remains: .noDestination,
            installations: [], mounted: [],
            transferDestinationName: nil, copyLocationName: nil)

        XCTAssertEqual(status.tone, .idle)
        XCTAssertEqual(
            status.sentence,
            "Not on this device. Download a copy, or add the folder that already contains it.")
    }

    /// The rule the whole app is built on, at the top of this screen: a
    /// transfer record that says `ready` cannot claim a copy the scan has seen
    /// and found unverified.
    func testTheDiskOutranksAFinishedTransferRecord() {
        let artifact = Self.artifact(verification: .unverified)
        let status = ModelPageStatusPresentation.resolve(
            state: .ready(measuredBytes: 517_269_118_806, fileCount: 624),
            remains: .checking,
            installations: [artifact], mounted: [artifact],
            transferDestinationName: "K3NVME",
            copyLocationName: "K3NVME")

        XCTAssertEqual(status.tone, .attention)
        XCTAssertEqual(
            status.sentence,
            "On K3NVME · verify all files before using this copy in a chat.")
    }

    /// A finished transfer the scan has not caught up with yet still says so,
    /// rather than falling through to "not on this device".
    func testAFinishedTransferSpeaksUntilTheScanHasSeenIt() {
        let status = ModelPageStatusPresentation.resolve(
            state: .ready(measuredBytes: 517_269_118_806, fileCount: 624),
            remains: .checking,
            installations: [], mounted: [],
            transferDestinationName: "K3NVME", copyLocationName: nil)

        XCTAssertEqual(status.tone, .ready)
        XCTAssertTrue(status.sentence.hasPrefix("Downloaded and checked."))
    }

    // MARK: - The dot itself

    func testThePanelsFiveChipTonesCollapseOntoThePagesFour() {
        XCTAssertEqual(MRPageStatusTone.from(.neutral), .idle)
        XCTAssertEqual(MRPageStatusTone.from(.ok), .ready)
        XCTAssertEqual(MRPageStatusTone.from(.caution), .attention)
        // A sample that has to be finished and a device that cannot run the
        // model are the same message to a reader scanning the page: this one
        // is not going to work until something changes.
        XCTAssertEqual(MRPageStatusTone.from(.verify), .attention)
        XCTAssertEqual(MRPageStatusTone.from(.refuse), .attention)
    }

    /// Colour is not the only carrier: VoiceOver reads the state in words.
    func testEveryToneSpeaksItsStateAndOnlyTheLiveOnesGlow() {
        XCTAssertEqual(MRPageStatusTone.moving.spokenState, "in progress")
        XCTAssertEqual(MRPageStatusTone.ready.spokenState, "ready")
        XCTAssertEqual(MRPageStatusTone.idle.spokenState, "not available")
        XCTAssertEqual(MRPageStatusTone.attention.spokenState, "needs attention")
        XCTAssertNil(MRPageStatusTone.idle.halo, "a dot meaning nothing here must not glow")
        for tone in MRPageStatusTone.allCases where tone != .idle {
            XCTAssertNotNil(tone.halo, "\(tone)")
        }
    }

    func testACopysOwnDotFollowsCompletenessBeforeVerification() {
        XCTAssertEqual(
            ModelPageStatusPresentation.tone(
                for: Self.artifact(verification: .fullyVerified)), .ready)
        XCTAssertEqual(
            ModelPageStatusPresentation.tone(
                for: Self.artifact(verification: .spotChecked)), .attention)
        XCTAssertEqual(
            ModelPageStatusPresentation.tone(
                for: Self.artifact(verification: .fullyVerified, files: 3)), .attention,
            "a verified sample of an incomplete directory is still incomplete")
    }

    // MARK: - Fixtures

    static func snapshot(verified: UInt64) -> ProgressSnapshot {
        ProgressSnapshot(
            totalBytes: 517_269_118_806, verifiedBytes: verified,
            fetchedUnverifiedBytes: 0, inFlightBytes: 0,
            filesTotal: 624, filesDone: 490,
            currentFilePath: "units/unit41/unit41-w3.fp8tile",
            currentFileBytes: 8_192_016_384, currentFileOffset: 1_073_741_824,
            bytesPerSecond: 35_000_000, estimatedTimeRemaining: 3.7 * 3600,
            networkBytes: verified, wastedBytes: 0)
    }

    static func artifact(
        verification: ArtifactVerification, files: Int = 624
    ) -> DiscoveredArtifact {
        DiscoveredArtifact(
            rootPath: "/Volumes/K3NVME/DeepSeek-V4.1-Flash-minirun",
            locationPath: "/Volumes/K3NVME",
            index: .unreadable,
            model: .deepseekV41Flash,
            displayName: "DeepSeek V4.1 Flash",
            bytesOnDisk: 517_269_118_806,
            fileCount: files,
            expectedFileCount: 624,
            expectedBytes: 517_269_118_806,
            verification: verification,
            verifiedAt: verification == .unverified
                ? nil : Date(timeIntervalSince1970: 1_786_000_000),
            scannedAt: Date(timeIntervalSince1970: 1_786_000_000))
    }
}

/// Readiness is two facts, and they are two rows of the About list.
///
/// They used to be a column of their own: a heading and two wrapped sentences,
/// holding a third of the page's width for a claim that fits in three words —
/// and pushing *Copies on this device*, which is the part of the page an
/// operator can act on, below the fold on the day a 517 GB transfer finished.
final class ModelPageReadinessTests: XCTestCase {

    private static func fitness(
        _ verdict: PlatformFitness.Verdict, reason: String
    ) -> PlatformFitness {
        PlatformFitness(
            verdict: verdict, reason: reason, requiredFreeBytes: 1, headroomBytes: 1)
    }

    func testEachReadinessFactIsOneShortPhraseThatStatesItsState() {
        let supported = ModelCompatibilityPresentation.resolve(
            modelName: "Kimi K3",
            fitness: Self.fitness(.runnable, reason: "fits"),
            availability: .available)
        XCTAssertEqual(supported.line, "supported here")

        let noRunner = ModelCompatibilityPresentation.resolve(
            modelName: "MiniMax H3",
            fitness: Self.fitness(.noRunner, reason: "no runner"),
            availability: .runtimeUnavailable)
        XCTAssertEqual(noRunner.line, "not in this version")

        let refused = ModelCompatibilityPresentation.resolve(
            modelName: "Kimi K3",
            fitness: Self.fitness(.refused, reason: "this device offers 6.44 GB"),
            availability: .available)
        XCTAssertEqual(refused.line, "not on this device")
        XCTAssertEqual(
            refused.reason, "this device offers 6.44 GB",
            "the sentence that says what to do is still there for the notes to use")
    }

    func testTheFilesFactSaysWhatWasEstablishedAboutTheBytes() {
        func integrity(_ verification: ArtifactVerification, mounted: Bool = true) -> String {
            let artifact = DiscoveredArtifact(
                rootPath: "/Volumes/K3NVME/k3", locationPath: "/Volumes/K3NVME",
                index: .unreadable, model: .kimiK3, displayName: "Kimi K3",
                bytesOnDisk: 10, fileCount: 2, expectedFileCount: 2, expectedBytes: 10,
                verification: verification, verifiedAt: nil, scannedAt: Date())
            return ModelIntegrityPresentation.resolve(
                installations: [artifact], mounted: mounted ? [artifact] : []
            ).line
        }
        XCTAssertEqual(integrity(.fullyVerified), "fully verified")
        XCTAssertEqual(integrity(.spotChecked), "spot-check only")
        XCTAssertEqual(integrity(.unverified), "not verified")
        XCTAssertEqual(integrity(.fullyVerified, mounted: false), "drive not connected")
        XCTAssertEqual(
            ModelIntegrityPresentation.resolve(installations: [], mounted: []).line,
            "not on this device")
    }

    /// Where the two facts sit, and in what order the page's sections come.
    ///
    /// Read off the source because it is a layout claim and a layout claim is
    /// exactly what a green value suite lets through: the readiness column was
    /// removed, its two facts are rows of the About list, and the copies come
    /// before About on both platforms.
    func testTheCopiesComeBeforeAboutAndReadinessHasNoColumnOfItsOwn() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("Sources/Screens/ModelDetailView.swift"),
            encoding: .utf8)
        let copies = try XCTUnwrap(source.range(of: "copiesSection(entry)"))
        let about = try XCTUnwrap(source.range(of: "aboutSection(entry)"))
        XCTAssertLessThan(
            copies.lowerBound, about.lowerBound,
            "the copies on the drive are the actionable part of this page")

        let factList = try XCTUnwrap(source.range(of: "MRFactList {"))
        let readinessRow = try XCTUnwrap(
            source.range(of: "readinessFacts(entry)", range: factList.upperBound..<source.endIndex))
        let sizeRow = try XCTUnwrap(
            source.range(of: "sizeFact(entry)", range: factList.upperBound..<source.endIndex))
        XCTAssertLessThan(
            readinessRow.lowerBound, sizeRow.lowerBound,
            "readiness is the first thing the About list says")
        XCTAssertNil(
            source.range(of: "Text(\"Readiness\")"),
            "the readiness column and its heading are gone")
        XCTAssertNil(
            source.range(of: "MRColumns {"),
            "and nothing on this page spends a third of its width on two phrases")
    }
}

/// The page's other composed strings: the line under the name, the pace line
/// beside the big number, and the link that leaves the app.
final class ModelPageCopyTests: XCTestCase {

    func testTheLineUnderTheNameIsPurposeFamilyAndLicenceAndInventsNoParameterCount() {
        XCTAssertEqual(
            ModelPurposePresentation.line(for: CatalogFixtures.deepseekV41Flash.descriptor),
            "Text chat · Mixture of experts · MIT License")
        XCTAssertEqual(
            ModelPurposePresentation.line(for: CatalogFixtures.minimaxH3.descriptor),
            "Video generation · Diffusion transformer · MiniMax Model License")
    }

    /// The catalog's own licence tag for K3 is a whole sentence. A header line
    /// takes its identifier; the full string still appears in the facts below.
    func testALicenceSentenceIsShortenedToItsIdentifierInTheHeader() {
        XCTAssertEqual(
            ModelLicensePresentation.shortName(
                "other (repository tag license:other; see the repository LICENSE)"),
            "other")
        XCTAssertEqual(ModelLicensePresentation.shortName("MIT License"), "MIT License")
        XCTAssertNil(ModelLicensePresentation.shortName(""))
    }

    /// Precision is read off the declared layout — the repository's own file
    /// shape — and is silent where the layout names no numeric width.
    func testPrecisionIsReadOffTheLayoutAndIsSilentWhereItCannotBe() {
        XCTAssertEqual(
            ArtifactPrecisionPresentation.sentence(for: .k3FlagshipLayerStreams),
            "FP4 experts · BF16 dense layers")
        XCTAssertEqual(
            ArtifactPrecisionPresentation.sentence(for: .v41FlashUnitBundle),
            "FP4 experts · FP8 matrices · FP8 memory tables")
        XCTAssertNil(ArtifactPrecisionPresentation.sentence(for: .h3UnitBundle))
        XCTAssertNil(ArtifactPrecisionPresentation.sentence(for: .unrecognized))
    }

    func testThePaceLineCarriesRateFileAndEstimateWhileItMoves() {
        let line = TransferPacePresentation.line(
            ModelPageStatusTests.snapshot(verified: 47_600_000_000), isMoving: true)
        XCTAssertEqual(line, "35 MB/s · file 491 of 624")
    }

    /// A stopped transfer has no speed and no estimate. The same snapshot,
    /// asked as a stopped one, keeps only the position — and even that is
    /// dropped elsewhere when the reconciliation disagrees with it.
    func testThePaceLineDropsRateAndEstimateWhenNothingIsMoving() {
        let line = TransferPacePresentation.line(
            ModelPageStatusTests.snapshot(verified: 47_600_000_000), isMoving: false)
        XCTAssertEqual(line, "file 491 of 624")
    }

    func testTheSourceLinkGoesToTheRepositoryAtTheExactRevision() {
        let repo = HuggingFaceRepoRef(
            repoID: "nanguoyu/DeepSeek-V4.1-Flash-minirun",
            revision: "fbf8d74eae864a622d2773d62085b4a0bc99344e")
        XCTAssertEqual(
            ModelSourceLink.url(for: repo).absoluteString,
            "https://huggingface.co/nanguoyu/DeepSeek-V4.1-Flash-minirun/tree/"
                + "fbf8d74eae864a622d2773d62085b4a0bc99344e")
    }
}

/// One row of the models list, as a value.
///
/// The row's sentence is the page's sentence with the room taken away, and the
/// same rule decides both: a transfer in motion speaks first, and then the disk
/// speaks last. A list that resolved its own state separately from the page it
/// opens would be a list that can disagree with the page it opens.
final class ModelListStatusTests: XCTestCase {

    private static let k3nvme = ["/Volumes/K3NVME": ModelRowLocation(
        name: "K3NVME", isMounted: true)]
    private static let archive = ["/Volumes/K3NVME": ModelRowLocation(
        name: "K3NVME", isMounted: false)]

    func testAVerifiedCopyOnAConnectedDriveIsGreenAndNamesTheDrive() {
        let status = ModelListStatusPresentation.resolve(
            state: .notStarted,
            installations: [ModelPageStatusTests.artifact(verification: .fullyVerified)],
            locations: Self.k3nvme, hasStorageLocation: true)

        XCTAssertEqual(status.tone, .ready)
        XCTAssertEqual(status.sentence, "Ready on K3NVME")
    }

    func testARunningTransferOutranksTheScanAndStatesThePosition() {
        let status = ModelListStatusPresentation.resolve(
            state: .active(ModelPageStatusTests.snapshot(verified: 47_600_000_000)),
            installations: [ModelPageStatusTests.artifact(verification: .unverified)],
            locations: Self.k3nvme, hasStorageLocation: true)

        XCTAssertEqual(status.tone, .moving)
        XCTAssertEqual(status.sentence, "Downloading · 47.6 GB of 517 GB")
    }

    func testAnUnpluggedDriveSaysSoRatherThanReadingAsEmpty() {
        let status = ModelListStatusPresentation.resolve(
            state: .notStarted,
            installations: [ModelPageStatusTests.artifact(verification: .fullyVerified)],
            locations: Self.archive, hasStorageLocation: true)

        XCTAssertEqual(status.tone, .attention)
        XCTAssertEqual(status.sentence, "K3NVME is not connected")
    }

    func testNothingAnywhereIsGreyAndSaysOnlyThat() {
        let status = ModelListStatusPresentation.resolve(
            state: .notStarted, installations: [], locations: [:],
            hasStorageLocation: true)

        XCTAssertEqual(status.tone, .idle)
        XCTAssertEqual(status.sentence, "Not on this device")
    }

    /// With no registered folder the app has not read a byte of anybody's
    /// disk. "Not on this device" there is a guess wearing a state's clothes.
    func testWithNoFolderRegisteredTheRowSaysWhyItCannotKnow() {
        let status = ModelListStatusPresentation.resolve(
            state: .notStarted, installations: [], locations: [:],
            hasStorageLocation: false)

        XCTAssertEqual(status.tone, .idle)
        XCTAssertEqual(status.sentence, "No folder added to look in")
    }

    func testASpotCheckedCopyIsAmberAndSaysASampleMatched() {
        let status = ModelListStatusPresentation.resolve(
            state: .notStarted,
            installations: [ModelPageStatusTests.artifact(verification: .spotChecked)],
            locations: Self.k3nvme, hasStorageLocation: true)

        XCTAssertEqual(status.tone, .attention)
        XCTAssertEqual(status.sentence, "On K3NVME · a sample matched")
    }

    func testAnIncompleteCopyNamesTheMissingFilesBeforeItsVerification() {
        let status = ModelListStatusPresentation.resolve(
            state: .notStarted,
            installations: [
                ModelPageStatusTests.artifact(verification: .fullyVerified, files: 3)
            ],
            locations: Self.k3nvme, hasStorageLocation: true)

        XCTAssertEqual(status.tone, .attention)
        XCTAssertEqual(status.sentence, "On K3NVME · files are missing")
    }

    /// Two copies on two drives is a fact worth seeing. The row states the
    /// better one and says how many there are; the page holds the rest.
    func testASecondCopyIsCountedInTheSentence() {
        let onArchive = DiscoveredArtifact(
            rootPath: "/Volumes/ARCHIVE/DeepSeek-V4.1-Flash-minirun",
            locationPath: "/Volumes/ARCHIVE",
            index: .unreadable, model: .deepseekV41Flash,
            displayName: "DeepSeek V4.1 Flash",
            bytesOnDisk: 517_269_118_806, fileCount: 624,
            expectedFileCount: 624, expectedBytes: 517_269_118_806,
            verification: .unverified, verifiedAt: nil,
            scannedAt: Date(timeIntervalSince1970: 1_786_000_000))
        let status = ModelListStatusPresentation.resolve(
            state: .notStarted,
            installations: [
                onArchive, ModelPageStatusTests.artifact(verification: .fullyVerified),
            ],
            locations: Self.k3nvme.merging(
                ["/Volumes/ARCHIVE": ModelRowLocation(name: "ARCHIVE", isMounted: true)],
                uniquingKeysWith: { first, _ in first }),
            hasStorageLocation: true)

        XCTAssertEqual(status.tone, .ready)
        XCTAssertEqual(status.sentence, "Ready on K3NVME · 2 copies")
    }

    /// A finished or cancelled transfer speaks only while the scan has found
    /// nothing. After that the drive outranks the record, exactly as it does
    /// on the model page.
    func testTheDiskOutranksAFinishedTransferRecord() {
        let record = DownloadState.ready(measuredBytes: 517_269_118_806, fileCount: 624)
        XCTAssertEqual(
            ModelListStatusPresentation.resolve(
                state: record, installations: [], locations: [:], hasStorageLocation: true
            ).sentence,
            "Downloaded and checked")
        XCTAssertEqual(
            ModelListStatusPresentation.resolve(
                state: record,
                installations: [ModelPageStatusTests.artifact(verification: .unverified)],
                locations: Self.k3nvme, hasStorageLocation: true
            ).sentence,
            "On K3NVME · not verified")
    }
}

/// One row of the Downloads list, as a value.
final class DownloadsRowPresentationTests: XCTestCase {

    /// The Downloads list is a list of transfers, and its sentence is the model
    /// page's own sentence for the same job — one resolver, so the two screens
    /// cannot disagree about what a transfer is doing.
    func testARunningTransferNamesTheDriveAndTheTimeLeft() {
        let status = DownloadsRowPresentation.status(
            state: .active(ModelPageStatusTests.snapshot(verified: 47_600_000_000)),
            remains: .checking, destinationName: "K3NVME")

        XCTAssertEqual(status.tone, .moving)
        XCTAssertEqual(status.sentence, "Downloading to K3NVME · about 3.7 h left")
    }

    /// A job with no destination yet is the one state a transfer record cannot
    /// describe, and this screen answers it rather than falling through to a
    /// sentence about the drive it has not looked at.
    func testAJobThatHasNotMovedSaysExactlyThat() {
        let status = DownloadsRowPresentation.status(
            state: .notStarted, remains: .noDestination, destinationName: nil)

        XCTAssertEqual(status.tone, .idle)
        XCTAssertEqual(status.sentence, "Nothing has been transferred yet.")
    }

    /// Nothing kept gets no big number: `0 GB` under an empty rail is a claim
    /// about a drive that has nothing on it.
    func testNothingKeptDrawsNoNumberAtAll() {
        XCTAssertNil(DownloadsRowPresentation.keptHeadline(.nothingKept))
        XCTAssertNil(DownloadsRowPresentation.keptHeadline(.checking))
        XCTAssertEqual(
            DownloadsRowPresentation.keptHeadline(
                .kept(completeFiles: 120, partialFiles: 1, bytes: 47_600_000_000)),
            "47.6 GB")
    }

    /// A cancelled transfer's status sentence already ends with the
    /// reconciliation. The block under it must not print the same sentence
    /// again; an interrupted one, whose sentence carries the reason instead,
    /// must.
    func testTheKeptSentenceIsPrintedOnceAndOnlyWhereItIsMissing() {
        let remains = TransferRemains.kept(
            completeFiles: 120, partialFiles: 1, bytes: 47_600_000_000)
        XCTAssertNil(
            DownloadsRowPresentation.keptSentence(
                state: .cancelled(ModelPageStatusTests.snapshot(verified: 0)),
                remains: remains))
        XCTAssertEqual(
            DownloadsRowPresentation.keptSentence(
                state: .interrupted(
                    ModelPageStatusTests.snapshot(verified: 47_600_000_000),
                    reason: .volumeDisappeared("K3NVME")),
                remains: remains),
            TransferRemainsPresentation.sentence(remains))
        XCTAssertNil(
            DownloadsRowPresentation.keptSentence(
                state: .active(ModelPageStatusTests.snapshot(verified: 1)),
                remains: remains))
    }
}

/// The page is drawn, not merely computed.
///
/// This project has shipped a layout bug through a green suite before — the
/// launch layout that hid the entire sidebar — so a redesign whose sections
/// silently stop drawing has to fail something. Every state the page has is
/// rendered offscreen in both appearances and written where a reviewer can look
/// at it, and each render has to be taller than the one with less in it.
#if os(macOS)
    import AppKit

    @MainActor
    final class ModelPageRenderTests: XCTestCase {

        private static var outputDirectory: URL {
            URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("minirun-model-page", isDirectory: true)
        }

        func testEveryStateOfThePageDrawsInBothAppearances() throws {
            for (name, make) in Self.states {
                for scheme in [ColorScheme.light, ColorScheme.dark] {
                    let model = make()
                    let image = try render(
                        model: model, modelID: Self.subject(of: name), width: 760,
                        scheme: scheme)
                    try write(image, named: "\(name)-\(scheme)")
                    XCTAssertGreaterThan(image.size.width, 0, "\(name) \(scheme)")
                    XCTAssertGreaterThan(
                        image.size.height, 200,
                        "\(name) \(scheme): the page drew almost nothing")
                }
            }
        }

        /// The phone layout, at the width of the narrowest device the app
        /// ships on. This runs in a macOS test host, so the `#if os(iOS)`
        /// branches of the screen are not compiled into it; what is checked
        /// here is that the page's own stacking survives 390 points.
        func testThePageStacksAtPhoneWidth() throws {
            for scheme in [ColorScheme.light, ColorScheme.dark] {
                let image = try render(
                    model: Self.downloadingModel(), modelID: .deepseekV41Flash,
                    width: 390, scheme: scheme)
                try write(image, named: "downloading-390-\(scheme)")
                XCTAssertGreaterThan(image.size.width, 0)
            }
        }

        /// Two copies of the same model draw a taller page than one, because
        /// the second copy is a row with its own path and its own actions.
        /// Cheap, and it is exactly the failure a silently absent section
        /// produces.
        func testASecondCopyDrawsASecondRow() throws {
            let one = try render(
                model: Self.readyModel(), modelID: .deepseekV4Flash, width: 760,
                scheme: .light)
            let two = try render(
                model: Self.twoCopiesModel(), modelID: .deepseekV4Flash, width: 760,
                scheme: .light)
            XCTAssertGreaterThan(two.size.height, one.size.height + 40)
        }

        /// One directory covered by two registered locations is one row, and
        /// the page it draws is the page one location draws.
        ///
        /// This is the drawing of the bug the owner saw: two identical rows
        /// under *Copies on this device*, both *624 files · Not verified*,
        /// each with its own *Verify all files*.
        func testATreeCoveredByTwoLocationsDrawsOneCopy() throws {
            let one = try render(
                model: Self.readyModel(), modelID: .deepseekV4Flash, width: 760,
                scheme: .light)
            let twice = try render(
                model: Self.doublyRegisteredModel(), modelID: .deepseekV4Flash, width: 760,
                scheme: .light)
            XCTAssertEqual(
                try XCTUnwrap(one.tiffRepresentation),
                try XCTUnwrap(twice.tiffRepresentation),
                "a second registered root over the same tree is not a second copy")
        }

        /// Four states, four different drawings. A page that answered every
        /// state with the same pixels would be a page that had stopped reading
        /// its model, which is the bug a height assertion alone can miss.
        func testTheFourStatesDoNotDrawTheSamePage() throws {
            var seen: [String: Data] = [:]
            for (name, make) in Self.states {
                let image = try render(
                    model: make(), modelID: Self.subject(of: name), width: 760,
                    scheme: .light)
                let data = try XCTUnwrap(image.tiffRepresentation)
                for (other, otherData) in seen {
                    XCTAssertNotEqual(data, otherData, "\(name) drew the same page as \(other)")
                }
                seen[name] = data
            }
        }

        // MARK: Rendering

        /// The page, not the screen: `ImageRenderer` draws nothing for a macOS
        /// `ScrollView`, which is the reason `ModelDetailPage` is a view of its
        /// own. What is rendered here is exactly what the screen puts inside
        /// its scroll view.
        private func render(
            model: AppModel, modelID: ModelID, width: CGFloat, scheme: ColorScheme
        ) throws -> NSImage {
            let renderer = ImageRenderer(
                content: ModelDetailPage(
                    modelID: modelID,
                    showingDestination: .constant(false),
                    pendingFullVerification: .constant(nil),
                    pendingDownloadVerification: .constant(false),
                    pendingRemoval: .constant(nil),
                    pendingForget: .constant(nil),
                    forgetWarning: .constant(nil)
                )
                .environment(model)
                .frame(width: width)
                .background(MRColor.panel)
                .environment(\.colorScheme, scheme))
            renderer.scale = 2
            return try XCTUnwrap(renderer.nsImage)
        }

        private func write(_ image: NSImage, named name: String) throws {
            let directory = Self.outputDirectory
            try? FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let tiff = try XCTUnwrap(image.tiffRepresentation)
            let png = try XCTUnwrap(
                NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]))
            try png.write(to: directory.appendingPathComponent("\(name).png"))
        }

        // MARK: Fixtures

        /// The four states the page has, and the model each one is about.
        static var states: [(String, () -> AppModel)] {
            [
                ("downloading", { ModelPageRenderTests.downloadingModel() }),
                ("cancelled", { ModelPageRenderTests.cancelledModel() }),
                ("ready", { ModelPageRenderTests.readyModel() }),
                ("drive-not-connected", { ModelPageRenderTests.unmountedModel() }),
            ]
        }

        /// The two states that are about a copy on a drive are V4 Flash, which
        /// is the model this build can actually chat with; the two that are
        /// about a transfer are V4.1 Flash, the 517 GB one being downloaded.
        private static func subject(of caseName: String) -> ModelID {
            switch caseName {
            case "ready", "drive-not-connected": return .deepseekV4Flash
            default: return .deepseekV41Flash
            }
        }

        /// 47.6 GB of 517 GB, 624 files, onto a drive called K3NVME.
        static func downloadingModel() -> AppModel {
            let model = makeModel(report: .empty)
            model.prepareDownload(for: .deepseekV41Flash)?
                .markStateForPreview(
                    .active(ModelPageStatusTests.snapshot(verified: 47_600_000_000)),
                    destinationPath: "/Volumes/K3NVME/DeepSeek-V4.1-Flash-minirun",
                    volumeName: "K3NVME")
            return model
        }

        /// Cancelled, and a look at the drive found nothing left.
        static func cancelledModel() -> AppModel {
            let model = makeModel(report: .empty)
            model.prepareDownload(for: .deepseekV41Flash)?
                .markStateForPreview(
                    .cancelled(ModelPageStatusTests.snapshot(verified: 0)),
                    destinationPath: "/Volumes/K3NVME/DeepSeek-V4.1-Flash-minirun",
                    volumeName: "K3NVME",
                    keptFiles: KeptFiles(
                        destination: .missing, plannedFileCount: 624,
                        plannedBytes: 517_269_118_806, completeFileCount: 0,
                        completeBytes: 0, partialFileCount: 0, partialBytes: 0))
            return model
        }

        /// V4 Flash, every file matched, on a drive that is plugged in.
        static func readyModel() -> AppModel {
            makeModel(report: report(mounted: true, verification: .fullyVerified))
        }

        /// The same, found on a second drive as well.
        static func twoCopiesModel() -> AppModel {
            makeModel(
                report: report(mounted: true, verification: .fullyVerified, drives: 2))
        }

        /// One copy, reached through two registered locations: the drive, and
        /// the artifact folder inside it — which is what a finished transfer
        /// used to register for itself.
        static func doublyRegisteredModel() -> AppModel {
            let base = report(mounted: true, verification: .fullyVerified)
            let drive = base.locations[0]
            let artifact = drive.artifacts[0]
            let nested = LocationScan(
                rootPath: artifact.rootPath, displayName: "DeepSeek-V4-Flash-0731-minirun",
                isMounted: true, storageKey: nil,
                artifacts: [
                    DiscoveredArtifact(
                        rootPath: artifact.rootPath, locationPath: artifact.rootPath,
                        index: artifact.index, model: artifact.model,
                        displayName: artifact.displayName, bytesOnDisk: artifact.bytesOnDisk,
                        fileCount: artifact.fileCount,
                        expectedFileCount: artifact.expectedFileCount,
                        expectedBytes: artifact.expectedBytes,
                        verification: artifact.verification, verifiedAt: artifact.verifiedAt,
                        scannedAt: artifact.scannedAt)
                ],
                unreadable: [:], scannedAt: drive.scannedAt)
            return makeModel(
                report: DiscoveryReport(
                    locations: [drive, nested], scannedAt: base.scannedAt))
        }

        /// The same copy, on a drive in a drawer.
        static func unmountedModel() -> AppModel {
            makeModel(report: report(mounted: false, verification: .fullyVerified))
        }

        /// Nothing anywhere: no copy, no transfer.
        static func emptyModel() -> AppModel { makeModel(report: .empty) }

        private static func report(
            mounted: Bool, verification: ArtifactVerification, drives: Int = 1
        ) -> DiscoveryReport {
            let now = Date(timeIntervalSince1970: 1_786_000_000)
            let descriptor = CatalogFixtures.deepseekV4Flash.descriptor
            let names = ["K3NVME", "ARCHIVE"]
            let locations = (0..<drives).map { index -> LocationScan in
                let volume = "/Volumes/\(names[index % names.count])"
                let artifact = DiscoveredArtifact(
                    rootPath: "\(volume)/DeepSeek-V4-Flash-0731-minirun",
                    locationPath: volume,
                    index: .unreadable,
                    model: .deepseekV4Flash,
                    displayName: descriptor.displayName,
                    bytesOnDisk: descriptor.totalBytes,
                    fileCount: descriptor.totalFileCount,
                    expectedFileCount: descriptor.totalFileCount,
                    expectedBytes: descriptor.totalBytes,
                    verification: verification,
                    verifiedAt: now,
                    scannedAt: now)
                return LocationScan(
                    rootPath: volume, displayName: names[index % names.count],
                    isMounted: mounted, storageKey: nil, artifacts: [artifact],
                    unreadable: [:], scannedAt: now)
            }
            return DiscoveryReport(locations: locations, scannedAt: now)
        }

        /// Not private: the list screens are drawn from the same app model,
        /// and two builders would be two chances for the screens under review
        /// to be reviewed against different fixtures.
        static func makeModel(report: DiscoveryReport) -> AppModel {
            let entries = CatalogFixtures.all
            let snapshot = CatalogFixtures.snapshot
            let storage = StorageManager(bookmarkLedger: InMemoryBookmarkLedger())
            let installed = InstalledModels(
                storage: storage, catalog: snapshot,
                ledger: InMemoryVerificationLedger(),
                initialReport: report,
                scanOperation: { _, _, _ in report })
            return AppModel(
                entries: entries,
                catalogSnapshot: snapshot,
                catalogService: ModelCatalog(bundled: snapshot, cache: nil),
                allowsLiveCatalogRefresh: false,
                runtimes: .preview(entries: entries),
                store: .ephemeral(),
                userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
                persistsDefaults: false,
                storage: storage,
                downloadServices: .preview(entries: entries),
                installed: installed,
                assessments: VolumeAssessments(
                    storage: storage, catalog: snapshot,
                    ledger: InMemoryAssessmentLedger()),
                seedRecordedRuns: false,
                startDiscovery: false)
        }
    }

    /// The three list screens are drawn too.
    ///
    /// Two things render as placeholders offscreen rather than as themselves,
    /// and both are AppKit-backed for the same reason `Link` is: the file
    /// filter's `TextField` and the payload `Toggle`. They draw correctly in
    /// the app; in a PNG they are the yellow rectangle `ImageRenderer` uses
    /// for a view it cannot rasterise.
    ///
    /// Same reason as the model page: this project has shipped a layout bug
    /// through a green suite before. A row whose status column silently stops
    /// drawing, or a section that disappears when a transfer stops, has to fail
    /// something — so every one of them is rendered offscreen in both
    /// appearances and at the narrowest width the product ships on, and written
    /// where a reviewer can look at it.
    @MainActor
    final class ListScreenRenderTests: XCTestCase {

        private static var outputDirectory: URL {
            URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("minirun-list-screens", isDirectory: true)
        }

        // MARK: The models list

        func testTheModelsListDrawsEveryStateInBothAppearances() throws {
            for scheme in [ColorScheme.light, ColorScheme.dark] {
                let image = try render(
                    ModelCatalogList(), model: ListScreenFixtures.catalog(), width: 900,
                    scheme: scheme)
                try write(image, named: "models-list-\(scheme)")
                XCTAssertGreaterThan(
                    image.size.height, 200, "the models list drew almost nothing")
            }
        }

        /// A list of four models is taller than a list of one. Cheap, and it is
        /// exactly the failure a fixed row height would hide.
        func testEveryModelFoundGetsARow() throws {
            let all = try render(
                ModelCatalogList(), model: ListScreenFixtures.catalog(), width: 900,
                scheme: .light)
            let one = try render(
                ModelCatalogList(), model: ListScreenFixtures.catalog(drives: 1),
                width: 900, scheme: .light)
            XCTAssertGreaterThan(all.size.height, one.size.height + 40)
        }

        // MARK: Downloads

        func testDownloadsDrawsARunningAndAStoppedTransfer() throws {
            for scheme in [ColorScheme.light, ColorScheme.dark] {
                let image = try render(
                    downloadsPage(), model: ListScreenFixtures.transfers(), width: 900,
                    scheme: scheme)
                try write(image, named: "downloads-\(scheme)")
                XCTAssertGreaterThan(
                    image.size.height, 300, "the transfer list drew almost nothing")
            }
        }

        /// The running transfer and the cancelled one do not draw the same
        /// block. A screen that answered both with the same pixels would be a
        /// screen that had stopped reading its transfers.
        func testARunningTransferAndAStoppedOneDrawDifferently() throws {
            let both = try render(
                downloadsPage(), model: ListScreenFixtures.transfers(), width: 900,
                scheme: .light)
            let running = try render(
                downloadsPage(), model: ListScreenFixtures.transfers(includesStopped: false),
                width: 900, scheme: .light)
            XCTAssertGreaterThan(both.size.height, running.size.height + 40)
            XCTAssertNotEqual(
                try XCTUnwrap(both.tiffRepresentation),
                try XCTUnwrap(running.tiffRepresentation))
        }

        // MARK: Download details

        func testDownloadDetailsDrawsItsSections() throws {
            for scheme in [ColorScheme.light, ColorScheme.dark] {
                let image = try render(
                    downloadDetailPage(), model: ListScreenFixtures.transfers(), width: 900,
                    scheme: scheme)
                try write(image, named: "download-details-\(scheme)")
                XCTAssertGreaterThan(
                    image.size.height, 400, "download details drew almost nothing")
            }
        }

        // MARK: The phone's width

        /// The narrowest device the app ships on. This runs in a macOS test
        /// host, so the `#if os(iOS)` branches are not compiled into it; what
        /// is checked here is that each screen's own stacking survives 390
        /// points — and, for a list, that the trailing column moves under the
        /// name instead of squeezing it into a vertical alphabet.
        func testTheThreeScreensStackAtPhoneWidth() throws {
            for scheme in [ColorScheme.light, ColorScheme.dark] {
                let models = try render(
                    ModelCatalogList(), model: ListScreenFixtures.catalog(), width: 390,
                    scheme: scheme)
                try write(models, named: "models-list-390-\(scheme)")
                let downloads = try render(
                    downloadsPage(), model: ListScreenFixtures.transfers(), width: 390,
                    scheme: scheme)
                try write(downloads, named: "downloads-390-\(scheme)")
                let details = try render(
                    downloadDetailPage(), model: ListScreenFixtures.transfers(), width: 390,
                    scheme: scheme)
                try write(details, named: "download-details-390-\(scheme)")
                XCTAssertGreaterThan(models.size.height, 200)
                XCTAssertGreaterThan(downloads.size.height, 300)
                XCTAssertGreaterThan(details.size.height, 400)
            }
        }

        /// A 900-point window gets the two-column row; a 390-point one gets the
        /// stacked one, which is taller. `ViewThatFits` used to answer this
        /// question wrongly — a column of wrapping sentences reports the width
        /// of its longest line as its ideal — so the row asks the container.
        func testAPhoneWidthRowIsTallerThanADesktopOne() throws {
            let wide = try render(
                ModelCatalogList(), model: ListScreenFixtures.catalog(), width: 900,
                scheme: .light)
            let narrow = try render(
                ModelCatalogList(), model: ListScreenFixtures.catalog(), width: 390,
                scheme: .light)
            XCTAssertGreaterThan(narrow.size.height, wide.size.height)
        }

        // MARK: Rendering

        private func downloadsPage() -> DownloadsPage {
            DownloadsPage(
                destinationRequest: .constant(nil),
                pendingForget: .constant(nil),
                forgetWarning: .constant(nil))
        }

        /// The file list is drawn with the filter set to one unit.
        ///
        /// Unfiltered, the screen draws its full 300-row cap: at 2x that is a
        /// thirty-thousand-pixel-tall bitmap, which no TIFF encoder produces
        /// and no reviewer reads. One unit shows the same row, the same
        /// monospaced path and the same tabular size, plus the "more not
        /// shown" line the cap is there for.
        private func downloadDetailPage() -> DownloadDetailPage {
            DownloadDetailPage(
                modelID: .deepseekV41Flash,
                filter: .constant("layers03"),
                showsPayloadOnly: .constant(true),
                pendingForget: .constant(nil),
                forgetWarning: .constant(nil))
        }

        private func render<Content: View>(
            _ content: Content, model: AppModel, width: CGFloat, scheme: ColorScheme
        ) throws -> NSImage {
            let renderer = ImageRenderer(
                content: content
                    .environment(model)
                    .frame(width: width)
                    .background(MRColor.panel)
                    .environment(\.colorScheme, scheme))
            renderer.scale = 2
            return try XCTUnwrap(renderer.nsImage)
        }

        private func write(_ image: NSImage, named name: String) throws {
            let directory = Self.outputDirectory
            try? FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let tiff = try XCTUnwrap(image.tiffRepresentation)
            let png = try XCTUnwrap(
                NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]))
            try png.write(to: directory.appendingPathComponent("\(name).png"))
        }
    }

    /// The device these three screens are reviewed on: two drives, four models
    /// in four different states, one transfer running and one cancelled.
    @MainActor
    enum ListScreenFixtures {

        /// K3NVME is plugged in and holds a verified V4 Flash, a spot-checked
        /// H3 and a V4.1 Flash that is still arriving; ARCHIVE is in a drawer
        /// with a verified K3 on it.
        static func catalog(drives: Int = 2) -> AppModel {
            let model = ModelPageRenderTests.makeModel(report: report(drives: drives))
            grantAFolder(to: model)
            if drives > 1 { startRunningTransfer(in: model) }
            return model
        }

        /// The same device, with the Downloads list's two jobs attached.
        static func transfers(includesStopped: Bool = true) -> AppModel {
            let model = ModelPageRenderTests.makeModel(report: report(drives: 2))
            grantAFolder(to: model)
            startRunningTransfer(in: model)
            if includesStopped { startCancelledTransfer(in: model) }
            return model
        }

        /// A registered folder, so the list is a list of what a scan found
        /// rather than the "nobody has told me where to look" state. The rows
        /// below are entitled to claim what is on a drive only once one has
        /// been granted, and the screenshot has to show the same thing.
        private static func grantAFolder(to model: AppModel) {
            let directory = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("minirun-list-fixture-\(UUID().uuidString)")
            try? FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            model.installed.addLocation(directory)
        }

        // MARK: Transfers

        private static let runningJob = DownloadJobID(
            rawValue: UUID(uuidString: "00000000-0000-0000-0000-0000000000A1")!)
        private static let cancelledJob = DownloadJobID(
            rawValue: UUID(uuidString: "00000000-0000-0000-0000-0000000000B2")!)

        /// 47.6 GB of 517 GB, 624 files, onto a drive called K3NVME.
        private static func startRunningTransfer(in model: AppModel) {
            guard let controller = model.prepareDownload(for: .deepseekV41Flash) else { return }
            controller.markStateForPreview(
                .active(ModelPageStatusTests.snapshot(verified: 47_600_000_000)),
                destinationPath: "/Volumes/K3NVME/DeepSeek-V4.1-Flash-minirun",
                volumeName: "K3NVME",
                job: runningJob)
            model.registerDownloadJobForPreview(runningJob, controller: controller)
        }

        /// Cancelled, and a look at the drive found 120 whole files and one
        /// part-file still there. Its numbers are H3's own — a fixture that
        /// borrowed the 517 GB transfer's totals would draw a bar that is
        /// three quarters empty for a model three quarters downloaded.
        private static func startCancelledTransfer(in model: AppModel) {
            guard let entry = CatalogFixtures.all.first(where: { $0.id == .minimaxH3 }),
                let controller = model.prepareDownload(for: .minimaxH3)
            else { return }
            let total = entry.descriptor.totalBytes
            controller.markStateForPreview(
                .cancelled(
                    ProgressSnapshot.empty(
                        totalBytes: total, filesTotal: entry.descriptor.totalFileCount)),
                destinationPath: "/Volumes/K3NVME/MiniMax-H3-minirun",
                volumeName: "K3NVME",
                keptFiles: KeptFiles(
                    destination: .present,
                    plannedFileCount: entry.descriptor.totalFileCount,
                    plannedBytes: total, completeFileCount: 120,
                    completeBytes: total / 2, partialFileCount: 1,
                    partialBytes: 500_000_000),
                job: cancelledJob)
            model.registerDownloadJobForPreview(cancelledJob, controller: controller)
        }

        // MARK: The drives

        private static func report(drives: Int) -> DiscoveryReport {
            let now = Date(timeIntervalSince1970: 1_786_000_000)
            var locations: [LocationScan] = [
                LocationScan(
                    rootPath: "/Volumes/K3NVME", displayName: "K3NVME",
                    isMounted: true, storageKey: nil,
                    artifacts: [
                        artifact(
                            .deepseekV4Flash, on: "/Volumes/K3NVME",
                            verification: .fullyVerified),
                        artifact(
                            .minimaxH3, on: "/Volumes/K3NVME", verification: .spotChecked),
                        artifact(
                            .deepseekV41Flash, on: "/Volumes/K3NVME",
                            verification: .unverified, missingFiles: 134),
                    ],
                    unreadable: [:], scannedAt: now)
            ]
            if drives > 1 {
                locations.append(
                    LocationScan(
                        rootPath: "/Volumes/ARCHIVE", displayName: "ARCHIVE",
                        isMounted: false, storageKey: nil,
                        artifacts: [
                            artifact(
                                .kimiK3, on: "/Volumes/ARCHIVE",
                                verification: .fullyVerified)
                        ],
                        unreadable: [:], scannedAt: now))
            }
            return DiscoveryReport(locations: locations, scannedAt: now)
        }

        private static func artifact(
            _ id: ModelID, on volume: String, verification: ArtifactVerification,
            missingFiles: Int = 0
        ) -> DiscoveredArtifact {
            let now = Date(timeIntervalSince1970: 1_786_000_000)
            let descriptor = CatalogFixtures.all.first { $0.id == id }!.descriptor
            return DiscoveredArtifact(
                rootPath: "\(volume)/\(id.rawValue)-minirun",
                locationPath: volume,
                index: .unreadable,
                model: id,
                displayName: descriptor.displayName,
                bytesOnDisk: missingFiles > 0
                    ? descriptor.totalBytes / UInt64(max(2, missingFiles)) : descriptor.totalBytes,
                fileCount: descriptor.totalFileCount - missingFiles,
                expectedFileCount: descriptor.totalFileCount,
                expectedBytes: descriptor.totalBytes,
                verification: verification,
                verifiedAt: verification == .unverified ? nil : now,
                scannedAt: now)
        }
    }

#endif

// =============================================================================
// MARK: - Phase 2: Settings, Storage and About
// =============================================================================
//
// The same language, on the screens DESIGN I.36 listed fourth in line. The
// values those pages compose are checked here for the reason the model page's
// are: a status sentence assembled inside a `body` is a claim nobody can test,
// and every one of these is assembled from two independent facts.

/// The sentences and glyphs the Storage page composes.
final class StoragePageCopyTests: XCTestCase {

    private let now = Date(timeIntervalSince1970: 1_786_000_000)

    // MARK: - A folder's one line

    func testAMountedFolderNamesWhenItsDriveWasLastMeasured() {
        let status = StorageFolderStatusPresentation.resolve(
            isMounted: true, hasRecordedPath: true,
            assessedAt: now.addingTimeInterval(-3 * 86400), now: now)

        XCTAssertEqual(status.tone, .ready)
        XCTAssertEqual(status.sentence, "Mounted · assessed 3 days ago")
    }

    /// A drive that is plugged in is usable whether or not anybody has timed
    /// it, so the dot stays green and the sentence says which half is missing.
    /// An amber dot here would ask the operator to fix something that is not
    /// broken.
    func testAFolderNobodyHasMeasuredSaysSoRatherThanClaimingARate() {
        let status = StorageFolderStatusPresentation.resolve(
            isMounted: true, hasRecordedPath: true, assessedAt: nil, now: now)

        XCTAssertEqual(status.tone, .ready)
        XCTAssertEqual(status.sentence, "Mounted · read rate not measured")
        XCTAssertFalse(status.sentence.contains("assessed"))
    }

    /// The drive outranks the measurement: a report taken an hour ago says
    /// nothing about a folder whose drive is in a drawer, so the line is about
    /// the drawer.
    func testAnUnpluggedDriveAsksForTheDriveAndNotForTheMeasurement() {
        let status = StorageFolderStatusPresentation.resolve(
            isMounted: false, hasRecordedPath: true,
            assessedAt: now.addingTimeInterval(-3600), now: now)

        XCTAssertEqual(status.tone, .attention)
        XCTAssertEqual(
            status.sentence, "Not connected. Plug the drive in, or remove the folder.")
        XCTAssertFalse(status.sentence.contains("assessed"))
    }

    func testAGrantWithNoRecordedPathNamesTheOneActionThatFixesIt() {
        let status = StorageFolderStatusPresentation.resolve(
            isMounted: true, hasRecordedPath: false, assessedAt: nil, now: now)

        XCTAssertEqual(status.tone, .attention)
        XCTAssertTrue(status.sentence.contains("add the folder again"))
    }

    // MARK: - The glyph

    /// Colour and shape are never the only carrier — the row's sentence says
    /// "not connected" in words — but the glyph must not contradict it either.
    func testTheDriveGlyphFollowsTheDriveAndTheSentenceAgreesWithIt() {
        XCTAssertEqual(
            StorageFolderPresentation.systemImage(isInternal: true, isMounted: true),
            "internaldrive")
        XCTAssertEqual(
            StorageFolderPresentation.systemImage(isInternal: false, isMounted: true),
            "externaldrive")
        XCTAssertEqual(
            StorageFolderPresentation.systemImage(isInternal: nil, isMounted: true),
            "externaldrive",
            "a folder no mounted volume claims is not the machine's own disk")
        XCTAssertEqual(
            StorageFolderPresentation.systemImage(isInternal: true, isMounted: false),
            "externaldrive.badge.xmark")
    }

    // MARK: - The measurement

    /// A quantity is split at its unit so a column of rates lines up; a
    /// sentence is not a quantity and is printed whole. "free space unknown"
    /// reaches the same slot and must not become "free space" · "unknown".
    func testAQuantityIsSplitAtItsUnitAndASentenceIsNot() {
        let rate = MRUnitValue.split("1.85 GB/s")
        XCTAssertEqual(rate?.number, "1.85")
        XCTAssertEqual(rate?.unit, "GB/s")

        let bytes = MRUnitValue.split("517 GB")
        XCTAssertEqual(bytes?.number, "517")
        XCTAssertEqual(bytes?.unit, "GB")

        XCTAssertEqual(MRUnitValue.split("<1 MB")?.number, "<1")
        XCTAssertNil(MRUnitValue.split("free space unknown"))
        XCTAssertNil(MRUnitValue.split("not measured"))
        XCTAssertNil(MRUnitValue.split("—"))
    }

    /// The one control that reads a volume says how much it will read before
    /// it is pressed, and says that it writes nothing.
    func testTheAssessControlStatesItsCostAndThatItWritesNothing() {
        let help = StorageSettingsPage.assessmentHelp(plannedBytes: 4_000_000_000)
        XCTAssertTrue(help.contains("4.00 GB"))
        XCTAssertTrue(help.contains("Nothing is written."))
    }

    /// First-run guidance is not a warning, and the page says so in one place:
    /// the sentence under "Folders Minirun can access" explains the grant
    /// rather than warning about it.
    func testTheFolderSectionExplainsTheGrantInsteadOfWarningAboutIt() {
        let explanation = StorageLocationsPresentation.explanation
        XCTAssertTrue(explanation.contains("only the folders you add"))
        XCTAssertTrue(explanation.contains("relaunch"))
        XCTAssertFalse(explanation.localizedCaseInsensitiveContains("warning"))
    }
}

/// The Settings pages are drawn, not merely computed.
///
/// Same contract as `ModelPageRenderTests`, and for the same reason: this
/// project has shipped a layout bug through a green suite before. Every page
/// this revision rewrote is rendered offscreen in both appearances, at the Mac
/// width and at 390 points, and written where a reviewer can look at it.
#if os(macOS)
    @MainActor
    final class SettingsPageRenderTests: XCTestCase {

        private static var outputDirectory: URL {
            URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("minirun-settings-pages", isDirectory: true)
        }

        // MARK: The pages

        func testGeneralStorageAndAboutDrawInBothAppearances() throws {
            for (name, width) in [("general", 900.0), ("storage", 820.0), ("about", 760.0)] {
                for scheme in [ColorScheme.light, ColorScheme.dark] {
                    let image = try render(page: name, width: width, scheme: scheme)
                    try write(image, named: "\(name)-\(scheme)")
                    XCTAssertGreaterThan(image.size.width, 0, "\(name) \(scheme)")
                    XCTAssertGreaterThan(
                        image.size.height, 200,
                        "\(name) \(scheme): the page drew almost nothing")
                }
            }
        }

        /// The narrowest device the app ships on. This runs in a macOS test
        /// host, so the `#if os(iOS)` branches are not compiled into it; what
        /// is checked here is that each page's own stacking survives 390
        /// points.
        func testEveryPageStacksAtPhoneWidth() throws {
            for name in ["general", "storage", "about"] {
                for scheme in [ColorScheme.light, ColorScheme.dark] {
                    let image = try render(page: name, width: 390, scheme: scheme)
                    try write(image, named: "\(name)-390-\(scheme)")
                    XCTAssertGreaterThan(image.size.height, 200, "\(name) \(scheme)")
                }
            }
        }

        /// Two folders and no folders are not the same page. A render that
        /// answered every state with the same pixels would be a page that had
        /// stopped reading its model, which is the bug a height assertion alone
        /// can miss.
        func testTheStoragePageDrawsItsStatesDifferently() throws {
            let two = try render(page: "storage", width: 820, scheme: .light)
            let none = try render(page: "storage-empty", width: 820, scheme: .light)
            try write(none, named: "storage-empty-light")

            XCTAssertNotEqual(
                try XCTUnwrap(two.tiffRepresentation),
                try XCTUnwrap(none.tiffRepresentation))
            XCTAssertGreaterThan(two.size.height, none.size.height + 40)
        }

        // MARK: The rail

        /// The Settings rail is a native sidebar `List`, which `ImageRenderer`
        /// cannot draw — so what is rendered here is the thing this revision
        /// actually changed: the row, in both of its states, in the order the
        /// rail shows them.
        func testTheRailRowDrawsItsSelectedStateDifferentlyFromItsRestingOne() throws {
            for scheme in [ColorScheme.light, ColorScheme.dark] {
                let image = try renderView(
                    Self.rail(selected: .general), width: 220, scheme: scheme)
                try write(image, named: "settings-rail-\(scheme)")
                XCTAssertGreaterThan(image.size.height, 100, "\(scheme)")
            }
            let general = try renderView(
                Self.rail(selected: .general), width: 220, scheme: .light)
            let storage = try renderView(
                Self.rail(selected: .storage), width: 220, scheme: .light)
            XCTAssertNotEqual(
                try XCTUnwrap(general.tiffRepresentation),
                try XCTUnwrap(storage.tiffRepresentation),
                "the selected row must not draw the same as an unselected one")
        }

        private static func rail(selected: AppModel.SettingsSection) -> some View {
            VStack(spacing: 2) {
                ForEach(AppModel.SettingsSection.allCases) { section in
                    SettingsSidebarRow(section: section, isSelected: section == selected)
                }
            }
            .padding(MRSpace.s2)
        }

        // MARK: Rendering

        @ViewBuilder private func page(_ name: String) -> some View {
            switch name {
            case "general":
                GeneralSettingsPage().environment(Self.settingsModel())
            case "storage":
                StorageSettingsPage(pendingLocationRemoval: .constant(nil))
                    .environment(Self.settingsModel())
            case "storage-empty":
                StorageSettingsPage(pendingLocationRemoval: .constant(nil))
                    .environment(Self.emptyStorageModel())
            default:
                AboutPage()
            }
        }

        private func render(page name: String, width: CGFloat, scheme: ColorScheme) throws
            -> NSImage
        {
            try renderView(page(name), width: width, scheme: scheme)
        }

        private func renderView<Content: View>(
            _ content: Content, width: CGFloat, scheme: ColorScheme
        ) throws -> NSImage {
            let renderer = ImageRenderer(
                content: content
                    .frame(width: width)
                    .background(MRColor.panel)
                    .environment(\.colorScheme, scheme))
            renderer.scale = 2
            return try XCTUnwrap(renderer.nsImage)
        }

        private func write(_ image: NSImage, named name: String) throws {
            let directory = Self.outputDirectory
            try? FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let tiff = try XCTUnwrap(image.tiffRepresentation)
            let png = try XCTUnwrap(
                NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]))
            try png.write(to: directory.appendingPathComponent("\(name).png"))
        }

        // MARK: Fixtures

        static let scannedAt = Date(timeIntervalSince1970: 1_786_000_000)
        /// Three days back from whenever the suite runs, so the shot a
        /// reviewer looks at shows the state the page is designed around — a
        /// fresh measurement — rather than the stale-measurement note a fixed
        /// date drifts into.
        static var assessedAt: Date { Date().addingTimeInterval(-3 * 86400) }

        /// Two folders: one on a plugged-in drive measured three days ago, one
        /// on a drive in a drawer.
        static func settingsModel() -> AppModel {
            let storage = SettingsRenderStorage(
                locations: [
                    StorageKey("artifact-location.archive"): "/Volumes/ARCHIVE",
                    StorageKey("artifact-location.k3nvme"): "/Volumes/K3NVME",
                ],
                volumes: [
                    volume(name: "Macintosh HD", path: "/", isInternal: true),
                    volume(name: "K3NVME", path: "/Volumes/K3NVME", isInternal: false),
                ])
            let ledger = InMemoryAssessmentLedger()
            ledger.record(assessment(locationPath: "/Volumes/K3NVME"))
            let model = makeModel(
                storage: storage,
                report: DiscoveryReport(
                    locations: [
                        LocationScan(
                            rootPath: "/Volumes/ARCHIVE", displayName: "ARCHIVE",
                            isMounted: false,
                            storageKey: StorageKey("artifact-location.archive"),
                            artifacts: [], unreadable: [:], scannedAt: scannedAt),
                        LocationScan(
                            rootPath: "/Volumes/K3NVME", displayName: "K3NVME",
                            isMounted: true,
                            storageKey: StorageKey("artifact-location.k3nvme"),
                            artifacts: [], unreadable: [:], scannedAt: scannedAt),
                    ],
                    scannedAt: scannedAt),
                assessmentLedger: ledger)
            model.refreshVolumes()
            // The screen loads persisted reports in `onAppear`, which does not
            // run offscreen. Doing it here is what the screen does, not a
            // fixture shortcut around it.
            model.assessments.load(locationPaths: ["/Volumes/ARCHIVE", "/Volumes/K3NVME"])
            return model
        }

        /// Nothing registered: the first-run guidance IS the page.
        static func emptyStorageModel() -> AppModel {
            makeModel(
                storage: SettingsRenderStorage(locations: [:], volumes: []),
                report: .empty, assessmentLedger: InMemoryAssessmentLedger())
        }

        private static func volume(
            name: String, path: String, isInternal: Bool
        ) -> VolumeDescriptor {
            VolumeDescriptor(
                mountPath: path, name: name, filesystemType: "apfs",
                space: FreeSpace(
                    optimisticBytes: 1_400_000_000_000,
                    dependableBytes: 1_200_000_000_000,
                    totalBytes: 2_000_000_000_000,
                    volumeName: name, isReadOnly: false, isRemovable: !isInternal),
                isInternal: isInternal, isEjectable: !isInternal, writeRefusal: nil)
        }

        /// A USB4-class drive, three days old, with both halves measured.
        private static func assessment(locationPath: String) -> VolumeAssessmentReport {
            let spot = ReadSpotResult(
                locationPath: locationPath, targets: [],
                sequentialBytesRead: 12_000_000_000, sequentialSeconds: 6.5,
                scatteredBytesRead: 512_000_000, scatteredSeconds: 1.8,
                scatteredReadCount: 4096, scatteredReadBytes: 131_072,
                shortReadCount: 0, cacheBypassed: true,
                startedAt: assessedAt,
                durationSeconds: 8.3)
            return VolumeAssessmentReport(
                locationPath: locationPath, volumeName: "K3NVME",
                volumeMountPath: locationPath,
                space: FreeSpace(
                    optimisticBytes: 1_400_000_000_000,
                    dependableBytes: 1_200_000_000_000,
                    totalBytes: 2_000_000_000_000,
                    volumeName: "K3NVME", isReadOnly: false, isRemovable: true),
                assessedAt: assessedAt,
                spot: spot, caution: nil,
                classification: LinkClassification(
                    linkClass: .usb4OrThunderbolt,
                    measuredBytesPerSecond: 12_000_000_000 / 6.5,
                    provenance: .measurement, hardware: nil,
                    fasterTierMultiple: nil, fasterTierName: nil),
                verdicts: [])
        }

        private static func makeModel(
            storage: SettingsRenderStorage, report: DiscoveryReport,
            assessmentLedger: any AssessmentLedger
        ) -> AppModel {
            let entries = CatalogFixtures.all
            let snapshot = CatalogFixtures.snapshot
            let installed = InstalledModels(
                storage: storage, catalog: snapshot,
                ledger: InMemoryVerificationLedger(),
                initialReport: report,
                scanOperation: { _, _, _ in report })
            return AppModel(
                entries: entries,
                catalogSnapshot: snapshot,
                catalogService: ModelCatalog(bundled: snapshot, cache: nil),
                allowsLiveCatalogRefresh: false,
                runtimes: .preview(entries: entries),
                store: .ephemeral(),
                userDefaults: UserDefaults(suiteName: "minirun.tests.\(UUID().uuidString)")!,
                persistsDefaults: false,
                storage: storage,
                downloadServices: .preview(entries: entries),
                installed: installed,
                assessments: VolumeAssessments(
                    storage: storage, catalog: snapshot, ledger: assessmentLedger),
                seedRecordedRuns: false,
                startDiscovery: false)
        }
    }

    /// A storage layer that owns exactly what the Settings pages read: which
    /// folders are registered, where each one was recorded, and which volumes
    /// are mounted. Nothing here touches the filesystem.
    private final class SettingsRenderStorage: StorageManaging, @unchecked Sendable {
        private let registered: [StorageKey: String]
        private let mounted: [VolumeDescriptor]

        init(locations: [StorageKey: String], volumes: [VolumeDescriptor]) {
            self.registered = locations
            self.mounted = volumes
        }

        func volumes() throws -> [VolumeDescriptor] { mounted }
        func describe(_ url: URL) -> StorageInfo { StorageInfo.describing(path: url.path) }
        func freeSpace(at url: URL) -> FreeSpace { FreeSpace(describe(url)) }
        func canHold(bytes: UInt64, at url: URL, headroomBytes: UInt64) -> SpaceVerdict {
            .fits(spareBytes: 0)
        }
        func scope(for url: URL) throws -> StorageScope { StorageScope(url: url) }

        @discardableResult
        func remember(_ url: URL, as key: StorageKey) throws -> StorageBookmark {
            StorageBookmark(key: key, createdAt: Date(), recordedPath: url.path, byteCount: 0)
        }

        func resolve(_ key: StorageKey) throws -> StorageScope {
            guard let path = registered[key] else { throw StorageError.noBookmark(key) }
            return StorageScope(url: URL(fileURLWithPath: path))
        }

        func bookmark(_ key: StorageKey) -> StorageBookmark? {
            registered[key].map {
                StorageBookmark(key: key, createdAt: Date(), recordedPath: $0, byteCount: 0)
            }
        }

        func forget(_ key: StorageKey) {}

        func knownLocations() -> [StorageKey] {
            registered.keys.sorted { $0.rawValue < $1.rawValue }
        }

        func bookmarkData(for url: URL) throws -> Data { Data(url.path.utf8) }
    }
#endif

/// A resumed transfer must not read as starting from zero: the files it
/// carried in are on the drive and count as done, and the big number is what
/// is on the drive, not what this session fetched.
final class ResumedTransferPresentationTests: XCTestCase {
    @MainActor func testCarriedFilesCountAsDoneAndBytesOnDiskIncludeThem() {
        let files = (0..<4).map { index in
            RepoFile(
                path: "layer0\(index)/w.bin", sizeBytes: 1_000,
                digest: .sha256(hex: String(repeating: "a", count: 64)), isPayload: true)
        }
        let plan = DownloadPlan(
            model: .kimiK3,
            repo: HuggingFaceRepoRef(
                repoID: "nanguoyu/Kimi-K3-minirun",
                revision: "159987d3ac437e0aceaff0763d43ddeb549b1842"),
            files: files, index: nil,
            reconciliation: IndexReconciliation.between(files: files, claim: nil))
        // Two files carried in unverified, one verified this session, one in flight.
        let progress = DownloadProgress(
            job: DownloadJobID(), planTotalBytes: 4_000,
            verifiedBytes: 1_000, fetchedUnverifiedBytes: 2_000, inFlightBytes: 300,
            remainingBytes: 700,
            filesTotal: 4, filesVerified: 1, filesFetchedUnverified: 2, filesFailed: 0,
            networkBytes: 1_300, wastedBytes: 0,
            instantaneousBytesPerSecond: 10, smoothedBytesPerSecond: 10,
            estimatedTimeRemaining: 70, startedAt: Date(), elapsed: 130)
        let snapshot = DownloadController.snapshot(from: progress, plan: plan)
        XCTAssertEqual(snapshot.filesDone, 3)
        XCTAssertEqual(snapshot.bytesOnDisk, 3_300)
        XCTAssertEqual(
            TransferPacePresentation.line(snapshot, isMoving: true),
            "0 MB/s · file 4 of 4")
    }
}

/// The closing pass says how far it is, in the same shape as a transfer.
final class VerificationPhasePresentationTests: XCTestCase {
    func testACheckingPassReportsFilesBytesAndAFraction() {
        let phase = VerificationPhase.checking(
            filesChecked: 311, filesTotal: 624, bytesChecked: 258_000_000_000,
            bytesTotal: 516_000_000_000)
        XCTAssertEqual(
            phase.sentence,
            "Checking every file against the published digests · 311 of 624")
        XCTAssertEqual(phase.filesLine, "file 312 of 624")
        XCTAssertEqual(phase.bytesChecked, 258_000_000_000)
        XCTAssertEqual(phase.fraction.map { ($0 * 100).rounded() }, 50)
    }

    func testAPassWithNoReportYetIsIndeterminate() {
        XCTAssertNil(VerificationPhase.fileDigests.fraction)
        XCTAssertNil(VerificationPhase.fileDigests.filesLine)
        XCTAssertNil(VerificationPhase.fileDigests.bytesChecked)
    }
}
