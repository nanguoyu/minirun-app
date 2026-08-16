import Foundation
import MinirunKit
import XCTest

@testable import MinirunApp

/// The report card's arithmetic and its wording.
///
/// The card is the screen an operator reads before deciding whether to spend a
/// day copying 1.56 TB onto a drive, so every string it composes is checked
/// here rather than eyeballed in a preview.
final class AssessmentCardTests: XCTestCase {

    private let k3Bytes: UInt64 = 1_559_976_181_760

    // MARK: - The fit value

    func testAFitShowsWhatItNeedsAndWhatIsLeft() {
        let fit = FitCheck.check(
            model: .kimiK3, displayName: "Kimi K3", catalogBytes: k3Bytes, bytesOnDisk: nil,
            availableBytes: 2_000_000_000_000)
        // `MRFormat`'s precision rule: three significant-ish digits, so a
        // 431 GB remainder loses its decimals and a 1.56 TB total keeps two.
        XCTAssertEqual(AssessmentFormat.fitValue(fit), "1.56 TB · 431 GB spare")
    }

    func testAVolumeTooSmallShowsHowFarShort() {
        let fit = FitCheck.check(
            model: .kimiK3, displayName: "Kimi K3", catalogBytes: k3Bytes, bytesOnDisk: nil,
            availableBytes: 500_000_000_000)
        XCTAssertEqual(AssessmentFormat.fitValue(fit), "1.07 TB short")
        XCTAssertFalse(fit.fits)
    }

    /// An artifact that is here reports what the directory holds, and the card
    /// sets it in the measured face rather than the declared one.
    func testAnArtifactAlreadyHereReadsAsMeasured() {
        let fit = FitCheck.check(
            model: .kimiK3, displayName: "Kimi K3", catalogBytes: k3Bytes,
            bytesOnDisk: 1_559_000_000_000, availableBytes: 100_000_000_000)
        XCTAssertEqual(AssessmentFormat.fitValue(fit), "1.56 TB here")
        XCTAssertTrue(fit.requiredIsMeasured)
    }

    func testAnUnprobeableVolumeSaysSoRatherThanShowingAZero() {
        let fit = FitCheck.check(
            model: .kimiK3, displayName: "Kimi K3", catalogBytes: k3Bytes, bytesOnDisk: nil,
            availableBytes: nil)
        XCTAssertEqual(AssessmentFormat.fitValue(fit), "free space unknown")
    }

    // MARK: - The projection

    /// A range, in `m:ss`, with no `≈` of its own — the prefix belongs to
    /// `ValueText`'s projected provenance and must appear exactly once.
    func testTheProjectionRendersAsARangeAndCarriesNoApproximatelySign() throws {
        let projection = try XCTUnwrap(
            TokenTimeProjector.project(
                workload: workload(), bytesPerSecond: 0.95e9))
        let text = AssessmentFormat.range(projection)
        XCTAssertTrue(text.hasSuffix("/ token"))
        XCTAssertFalse(text.contains("≈"))
        XCTAssertEqual(
            text,
            "\(MRFormat.clock(projection.secondsPerTokenLow)) – "
                + "\(MRFormat.clock(projection.secondsPerTokenHigh)) / token")
    }

    func testTheOnVolumeLineNamesTheVolumeAndTheDate() throws {
        let when = Date(timeIntervalSince1970: 1_770_000_000)
        let projection = try XCTUnwrap(
            TokenTimeProjector.project(workload: workload(), bytesPerSecond: 0.95e9))
        let line = AssessmentFormat.onVolumeLine(
            volumeName: "K3NVME", projection: projection, measuredAt: when)
        XCTAssertTrue(line.hasPrefix("on K3NVME: ≈ "))
        XCTAssertTrue(line.contains("/ token"))
        XCTAssertTrue(line.contains(MRFormat.timestamp(when)))
    }

    func testTheOnVolumeLineDoesNotInventAVolumeName() throws {
        let projection = try XCTUnwrap(
            TokenTimeProjector.project(workload: workload(), bytesPerSecond: 0.95e9))
        XCTAssertTrue(
            AssessmentFormat.onVolumeLine(
                volumeName: nil, projection: projection, measuredAt: Date()
            ).hasPrefix("on this volume:"))
    }

    // MARK: - The two projectors are one projector

    /// The kit's projector and the dial's are the same arithmetic, and this is
    /// what keeps them that way. A drift here would mean the storage screen and
    /// the memory dial quote different seconds for the same run, which is the
    /// kind of disagreement nobody notices until it is in a record.
    func testTheKitProjectorAgreesWithTheDialsProjector() throws {
        let entry = CatalogFixtures.kimiK3
        let link = VolumeCalibration(
            volumeMountPath: "/Volumes/K3NVME", volumeName: "K3NVME", bytesPerSecond: 0.95e9,
            busDescription: "USB3 10 Gb/s", measuredAt: Date(), sampleSeconds: 4)

        for budget in [6_000_000_000, 8_000_000_000, 12_000_000_000, 24_000_000_000] as [UInt64] {
            let plan = BudgetPlan(
                model: .kimiK3, modelName: entry.descriptor.displayName, profile: entry.memory,
                budgetBytes: budget, maximumNewTokens: 64,
                deviceCeilingBytes: 64_000_000_000, readAheadDepth: 1)
            guard plan.isRunnable else { continue }
            let dial = try XCTUnwrap(
                SpeedProjector.project(plan: plan, terms: entry.terms, link: link))
            let kit = try XCTUnwrap(
                TokenTimeProjector.project(
                    workload: TokenWorkload(
                        deterministicBytesPerToken: entry.terms.deterministicBytesPerToken,
                        expertBytesPerToken: entry.terms.expertBytesPerToken,
                        computeSecondsPerToken: entry.terms.computeSecondsPerToken,
                        expertCacheFraction: 0,
                        deterministicOverlapsCompute: plan.stagedTierIsFunded),
                    bytesPerSecond: link.bytesPerSecond))

            XCTAssertEqual(kit.mid, dial.mid, accuracy: 1e-9, "at \(budget) B")
            XCTAssertEqual(
                kit.secondsPerTokenLow, dial.secondsPerTokenLow, accuracy: 1e-9, "at \(budget) B")
            XCTAssertEqual(
                kit.secondsPerTokenHigh, dial.secondsPerTokenHigh, accuracy: 1e-9,
                "at \(budget) B")
            XCTAssertEqual(kit.deterministicHidden, dial.deterministicHidden)
        }
        XCTAssertEqual(TokenTimeProjector.relativeUncertainty, SpeedProjector.relativeUncertainty)
    }

    /// The app hands the assessment the budget's own answers, so the card and
    /// the dial are describing the same run.
    @MainActor
    func testTheAppsWorkloadsCarryTheBudgetsCacheFractionAndStagedFlag() throws {
        let model = AppModel.freshForTests()
        let workloads = model.assessmentWorkloads
        let k3 = try XCTUnwrap(workloads[.kimiK3])
        let plan = try XCTUnwrap(model.defaultBudgetPlan(for: .kimiK3))

        XCTAssertEqual(
            k3.deterministicBytesPerToken, CatalogFixtures.kimiK3.terms.deterministicBytesPerToken)
        XCTAssertEqual(k3.expertBytesPerToken, CatalogFixtures.kimiK3.terms.expertBytesPerToken)
        // Zero, and not a function of the budget any more. The derived ladder
        // reaches an expert only past full deterministic residency, so no budget
        // this device can state pins one and there is no cache fraction to pass.
        XCTAssertEqual(k3.expertCacheFraction, 0, accuracy: 1e-12)
        XCTAssertEqual(plan.hotSetBytes, 0)
        XCTAssertEqual(k3.deterministicOverlapsCompute, plan.stagedTierIsFunded)

        // A model this build cannot run gets no terms, so it gets no projected
        // token time anywhere.
        XCTAssertNil(workloads[.minimaxH3])
    }

    // MARK: - What the card shows when there is nothing to show

    /// A report with no spot test carries a caution, keeps its fit checks, and
    /// projects nothing at all.
    func testAReportWithoutAMeasurementProjectsNothing() {
        let report = VolumeAssessmentReport(
            locationPath: "/Volumes/EMPTY", volumeName: "EMPTY", volumeMountPath: "/Volumes/EMPTY",
            space: FreeSpace(
                optimisticBytes: nil, dependableBytes: 400_000_000_000, totalBytes: nil,
                volumeName: "EMPTY", isReadOnly: false, isRemovable: true),
            assessedAt: Date(), spot: nil,
            caution: "nothing under '/Volumes/EMPTY' is large enough to measure a link with",
            classification: LinkClassifier.classify(bytesPerSecond: nil),
            verdicts: [
                ModelVerdict(
                    fit: FitCheck.check(
                        model: .kimiK3, displayName: "Kimi K3", catalogBytes: k3Bytes,
                        bytesOnDisk: nil, availableBytes: 400_000_000_000),
                    projection: nil)
            ])
        XCTAssertNil(report.measuredBytesPerSecond)
        XCTAssertNil(report.degradedLinkSentence)
        XCTAssertEqual(report.headline, report.caution)
        XCTAssertEqual(AssessmentFormat.fitValue(report.verdicts[0].fit), "1.17 TB short")
    }

    /// A degraded link says how much faster the alternative measured, and says
    /// it with the numbers rather than an adjective.
    func testADegradedLinkQuotesTheMultiple() throws {
        let report = makeMeasuredReport(bytesPerSecond: 0.95e9)
        let sentence = try XCTUnwrap(report.degradedLinkSentence)
        XCTAssertTrue(sentence.contains("3.0×"))
        XCTAssertTrue(report.headline.contains("MB/s"))
        XCTAssertNil(makeMeasuredReport(bytesPerSecond: 2.84e9).degradedLinkSentence)
    }

    /// Storage reports storage. Its comparison may name a link and a rate, but
    /// must not bring model verdicts or token projections back onto that page.
    func testStorageLinkComparisonContainsNoModelProjection() throws {
        let report = makeMeasuredReport(bytesPerSecond: 0.95e9)
        let sentence = try XCTUnwrap(
            StorageAssessmentPresentation.fasterLinkSentence(report))
        XCTAssertTrue(sentence.contains("3.0×"))
        XCTAssertTrue(sentence.contains("GB/s"))
        XCTAssertFalse(sentence.localizedCaseInsensitiveContains("token"))
        XCTAssertFalse(sentence.localizedCaseInsensitiveContains("model"))
    }

    // MARK: - Helpers

    private func workload() -> TokenWorkload {
        TokenWorkload(
            deterministicBytesPerToken: CatalogFixtures.kimiK3.terms.deterministicBytesPerToken,
            expertBytesPerToken: CatalogFixtures.kimiK3.terms.expertBytesPerToken,
            computeSecondsPerToken: CatalogFixtures.kimiK3.terms.computeSecondsPerToken)
    }

    private func makeMeasuredReport(bytesPerSecond: Double) -> VolumeAssessmentReport {
        let spot = ReadSpotResult(
            locationPath: "/Volumes/K3NVME",
            targets: [SpotTestTarget(path: "/Volumes/K3NVME/k3/layer00.tile", sizeBytes: 1 << 30)],
            sequentialBytesRead: 2_000_000_000,
            sequentialSeconds: 2_000_000_000 / bytesPerSecond,
            scatteredBytesRead: 1_073_741_824, scatteredSeconds: 1.5,
            scatteredReadCount: 256, scatteredReadBytes: 4 << 20, shortReadCount: 0,
            cacheBypassed: true, startedAt: Date(), durationSeconds: 4)
        return VolumeAssessmentReport(
            locationPath: "/Volumes/K3NVME", volumeName: "K3NVME",
            volumeMountPath: "/Volumes/K3NVME",
            space: FreeSpace(
                optimisticBytes: nil, dependableBytes: 2_000_000_000_000, totalBytes: nil,
                volumeName: "K3NVME", isReadOnly: true, isRemovable: true),
            assessedAt: Date(), spot: spot, caution: nil,
            classification: LinkClassifier.classify(bytesPerSecond: bytesPerSecond),
            verdicts: [])
    }
}
